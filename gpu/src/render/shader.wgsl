// Cherenkov GPU slice: one instanced quad pipeline for every primitive family.
//
// Every instance is a quad. The vertex shader places it; the fragment shader
// computes analytic coverage (an SDF, a Gaussian-blurred rounded box, or an
// atlas texel), multiplies by the instance's clip coverage and opacity, and
// evaluates the paint at the pixel centre in the instance's local space. The
// output is premultiplied linear Display P3 into an rgba16float target with
// fixed-function src-over blending.
//
// Layouts here mirror `render::instance` on the CPU side exactly.

const KIND_FILL: u32 = 0u;
const KIND_STROKE_OFFSET: u32 = 1u; // coverage(outer shape) - coverage(inner shape)
const KIND_STROKE_DIST: u32 = 2u;   // coverage(d - hw) - coverage(d + hw)
const KIND_SHADOW: u32 = 3u;        // Gaussian-blurred rounded box
const KIND_GLYPH: u32 = 4u;         // coverage from the glyph atlas

const PAINT_SOLID: u32 = 0u;
const PAINT_LINEAR: u32 = 1u;
const PAINT_RADIAL: u32 = 2u;
const PAINT_TEXTURE: u32 = 3u;      // composite: sample the bound texture at the device pixel

const EXTEND_PAD: u32 = 0u;
const EXTEND_REPEAT: u32 = 1u;
const EXTEND_REFLECT: u32 = 2u;

const INTERP_WORKING: u32 = 0u;
const INTERP_SRGB: u32 = 1u;

const FLAG_HAS_CLIP: u32 = 1u;
const FLAG_HAS_INNER: u32 = 2u;

// A rounded box centred at the origin. `radii` are the corner radii along x
// in the order top-left, top-right, bottom-right, bottom-left; the radius
// along y is `radius * aspect`. `exponent` is the Lamé exponent of the corner
// curve (2 = circular / elliptical).
struct Shape {
    half: vec2<f32>,
    aspect: f32,
    exponent: f32,
    radii: vec4<f32>,
}

struct Instance {
    // local -> device, kurbo coefficient order [a, b, c, d, e, f]:
    // x' = a x + c y + e ; y' = b x + d y + f.
    affine: array<vec4<f32>, 2>,
    // Quad rectangle (x0, y0, x1, y1). Local space, except KIND_GLYPH where it
    // is the device-space atlas cell rectangle.
    bounds: vec4<f32>,
    shape: Shape,
    inner: Shape,
    // device -> clip-local affine, same coefficient order.
    clip_inv: array<vec4<f32>, 2>,
    clip: Shape,
    // Straight-alpha working-space colour (solid paint, glyph, shadow).
    color: vec4<f32>,
    // Linear: start.xy, end.xy. Radial: start centre.xy, end centre.xy.
    grad: vec4<f32>,
    // Radial: start radius, end radius.
    grad2: vec4<f32>,
    // Glyph: atlas cell origin (x, y) in texels.
    uv: vec4<f32>,
    // x: stroke half width (STROKE_DIST) or shadow sigma. y: opacity.
    params: vec4<f32>,
    // x: kind, y: paint, z: first stop index, w: stops | interp << 16 | extend << 20 | flags << 24
    meta_: vec4<u32>,
}

struct Stop {
    color: vec4<f32>, // straight alpha, in the interpolation space
    offset: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Globals {
    size: vec2<f32>,
    pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read> stops: array<Stop>;
@group(0) @binding(3) var atlas: texture_2d<f32>;
@group(1) @binding(0) var source: texture_2d<f32>;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) device: vec2<f32>,
    @location(2) @interpolate(flat) instance: u32,
}

fn apply(m: array<vec4<f32>, 2>, p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        m[0].x * p.x + m[0].z * p.y + m[1].x,
        m[0].y * p.x + m[0].w * p.y + m[1].y,
    );
}

fn apply_inverse(m: array<vec4<f32>, 2>, p: vec2<f32>) -> vec2<f32> {
    let a = m[0].x;
    let b = m[0].y;
    let c = m[0].z;
    let d = m[0].w;
    let det = a * d - b * c;
    let inv_det = select(1.0 / det, 0.0, abs(det) < 1e-12);
    let q = p - m[1].xy;
    return vec2<f32>((d * q.x - c * q.y) * inv_det, (-b * q.x + a * q.y) * inv_det);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let inst = instances[ii];
    // Two triangles: 0 1 2, 2 1 3 over the corners (x0,y0) (x1,y0) (x0,y1) (x1,y1).
    let corner = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u)[vi];
    let sx = f32(corner & 1u);
    let sy = f32(corner >> 1u);
    let p = vec2<f32>(mix(inst.bounds.x, inst.bounds.z, sx), mix(inst.bounds.y, inst.bounds.w, sy));
    var out: VsOut;
    if inst.meta_.x == KIND_GLYPH {
        out.device = p;
        out.local = apply_inverse(inst.affine, p);
    } else {
        out.local = p;
        out.device = apply(inst.affine, p);
    }
    let ndc = out.device / globals.size * 2.0 - 1.0;
    out.position = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    out.instance = ii;
    return out;
}

// Signed distance from `p` to the rounded box `s`. Exact for straight edges
// and circular corners; first-order (Newton) for elliptical and Lamé corners.
fn sdf(s: Shape, p: vec2<f32>) -> f32 {
    let right = p.x > 0.0;
    let bottom = p.y > 0.0;
    let r = select(
        select(s.radii.x, s.radii.w, bottom),
        select(s.radii.y, s.radii.z, bottom),
        right,
    );
    let rx = max(r, 0.0);
    let ry = rx * s.aspect;
    let a = abs(p) - s.half;
    if rx <= 0.0 || ry <= 0.0 {
        return length(max(a, vec2<f32>(0.0))) + min(max(a.x, a.y), 0.0);
    }
    let q = a + vec2<f32>(rx, ry);
    if q.x > 0.0 && q.y > 0.0 {
        let n = s.exponent;
        let u = q / vec2<f32>(rx, ry);
        if abs(n - 2.0) < 1e-4 {
            let g = length(u);
            let grad = length(u / vec2<f32>(rx, ry)) / max(g, 1e-6);
            return (g - 1.0) / max(grad, 1e-6);
        }
        let f = pow(u.x, n) + pow(u.y, n);
        let g = pow(f, 1.0 / n);
        // d/dq of f^(1/n) = f^(1/n - 1) * (u^(n-1) / r)
        let scale = pow(f, 1.0 / n - 1.0);
        let grad = scale * length(vec2<f32>(pow(u.x, n - 1.0) / rx, pow(u.y, n - 1.0) / ry));
        return (g - 1.0) / max(grad, 1e-6);
    }
    return max(a.x, a.y);
}

// Length in device pixels of the local-space gradient `g` of a signed
// distance, for the local -> device affine `m` (J^-T g).
fn device_grad(m: array<vec4<f32>, 2>, g: vec2<f32>) -> f32 {
    let a = m[0].x;
    let b = m[0].y;
    let c = m[0].z;
    let d = m[0].w;
    let det = a * d - b * c;
    let inv_det = select(1.0 / det, 0.0, abs(det) < 1e-12);
    return max(length(vec2<f32>(d * g.x - b * g.y, -c * g.x + a * g.y)) * inv_det, 1e-6);
}

// Local-space gradient of the signed distance to `s` at `p`, by central
// differences. Derivative builtins are not used: they are unreliable in the
// helper lanes along the quad's triangle seam.
fn sdf_grad(s: Shape, p: vec2<f32>) -> vec2<f32> {
    const E: f32 = 0.05;
    return vec2<f32>(
        sdf(s, p + vec2<f32>(E, 0.0)) - sdf(s, p - vec2<f32>(E, 0.0)),
        sdf(s, p + vec2<f32>(0.0, E)) - sdf(s, p - vec2<f32>(0.0, E)),
    ) / (2.0 * E);
}

// Area coverage of the half-plane `d <= 0` where `d` changes by `g` per
// device pixel; exact for straight edges under any affine.
fn coverage(d: f32, g: f32) -> f32 {
    return clamp(0.5 - d / g, 0.0, 1.0);
}

// Coverage of the shape `s` at local point `p`, `m` mapping local to device.
fn shape_coverage(s: Shape, p: vec2<f32>, m: array<vec4<f32>, 2>) -> f32 {
    return coverage(sdf(s, p), device_grad(m, sdf_grad(s, p)));
}

// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7.
fn erf(x: f32) -> f32 {
    let s = sign(x);
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}

fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma)) / (2.5066282746 * sigma);
}

// Horizontal inset of a circular corner of radius `r` at distance `dy` past
// the start of the corner (dy <= 0 means the straight part of the edge).
fn corner_inset(r: f32, dy: f32) -> f32 {
    if dy <= 0.0 || r <= 0.0 {
        return 0.0;
    }
    let dd = min(dy, r);
    return r - sqrt(max(r * r - dd * dd, 0.0));
}

// Indicator of the rounded box `s` (circular corners), convolved with an
// isotropic Gaussian of standard deviation `sigma`, evaluated at `p`. The x
// integral of each row is analytic (erf); the y integral is a midpoint rule
// over ±3σ.
fn shadow(s: Shape, p: vec2<f32>, sigma: f32) -> f32 {
    const N: i32 = 16;
    let inv_sqrt2_sigma = 1.0 / (sigma * 1.4142135624);
    let step = 6.0 * sigma / f32(N);
    var acc = 0.0;
    for (var i = 0; i < N; i++) {
        let dy = (f32(i) + 0.5) * step - 3.0 * sigma;
        let y = p.y + dy;
        if abs(y) > s.half.y {
            continue;
        }
        let top = y < 0.0;
        let rl = select(s.radii.w, s.radii.x, top);
        let rr = select(s.radii.z, s.radii.y, top);
        let ay = abs(y);
        let xl = -s.half.x + corner_inset(rl, ay - (s.half.y - rl));
        let xr = s.half.x - corner_inset(rr, ay - (s.half.y - rr));
        if xr <= xl {
            continue;
        }
        let row = 0.5 * (erf((xr - p.x) * inv_sqrt2_sigma) - erf((xl - p.x) * inv_sqrt2_sigma));
        acc += row * gaussian(dy, sigma) * step;
    }
    return clamp(acc, 0.0, 1.0);
}

fn srgb_encode(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn srgb_decode(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((max(c, vec3<f32>(0.0)) + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

// Linear sRGB -> linear Display P3 (column-major constructor: columns).
const SRGB_TO_P3 = mat3x3<f32>(
    vec3<f32>(0.8224621, 0.0331941, 0.0170827),
    vec3<f32>(0.1775380, 0.9668058, 0.0723974),
    vec3<f32>(0.0, 0.0, 0.9105199),
);

fn extend_t(t: f32, mode: u32) -> f32 {
    switch mode {
        case EXTEND_REPEAT: {
            return t - floor(t);
        }
        case EXTEND_REFLECT: {
            let m = t - 2.0 * floor(t * 0.5);
            return 1.0 - abs(m - 1.0);
        }
        default: {
            return clamp(t, 0.0, 1.0);
        }
    }
}

// Premultiplied working-space colour of the gradient at parameter `t`.
fn eval_stops(first: u32, count: u32, interp: u32, t: f32) -> vec4<f32> {
    var c: vec4<f32>;
    if count == 0u {
        return vec4<f32>(0.0);
    }
    if t <= stops[first].offset || count == 1u {
        c = stops[first].color;
    } else if t >= stops[first + count - 1u].offset {
        c = stops[first + count - 1u].color;
    } else {
        var i = first;
        loop {
            if i + 1u >= first + count - 1u || t < stops[i + 1u].offset {
                break;
            }
            i += 1u;
        }
        let s0 = stops[i];
        let s1 = stops[i + 1u];
        let span = s1.offset - s0.offset;
        let f = select((t - s0.offset) / span, 0.0, span <= 0.0);
        c = mix(s0.color, s1.color, f);
    }
    var rgb = c.rgb;
    if interp == INTERP_SRGB {
        rgb = SRGB_TO_P3 * srgb_decode(rgb);
    }
    return vec4<f32>(rgb * c.a, c.a);
}

fn linear_t(inst: Instance, p: vec2<f32>) -> f32 {
    let d = inst.grad.zw - inst.grad.xy;
    let dd = dot(d, d);
    if dd <= 0.0 {
        return 0.0;
    }
    return dot(p - inst.grad.xy, d) / dd;
}

// Two-point conical gradient parameter, a literal port of the oracle's
// radial_t: the larger real root of |p - (c0 + t·dc)| = r0 + t·dr.
// Degenerate coincident circles use the relative distance from the centre.
// NaN (no solution) yields a transparent pixel.
fn radial_t(inst: Instance, p: vec2<f32>) -> f32 {
    let c0 = inst.grad.xy;
    let c1 = inst.grad.zw;
    let r0 = inst.grad2.x;
    let r1 = inst.grad2.y;
    let dc = c1 - c0;
    let dr = r1 - r0;
    let pd = p - c0;
    let a = dot(dc, dc) - dr * dr;
    // b = -2·((p - c0)·dc + r0·dr)
    let b = -2.0 * (dot(pd, dc) + r0 * dr);
    let c = dot(pd, pd) - r0 * r0;
    if abs(a) < 1e-12 {
        if abs(b) < 1e-12 {
            // Coincident circles: distance relative to r0.
            if abs(r0) < 1e-12 {
                return 0.0;
            }
            return (length(pd) - r0) / abs(r0);
        }
        return -c / b;
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return bitcast<f32>(0x7fc00000u);
    }
    let sq = sqrt(disc);
    // The cone answer is the larger root; when `a` is negative that is the
    // smaller numerator, so compare the roots themselves.
    return max((-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a));
}

fn paint(inst: Instance, local: vec2<f32>, device: vec2<f32>) -> vec4<f32> {
    switch inst.meta_.y {
        case PAINT_SOLID: {
            return vec4<f32>(inst.color.rgb * inst.color.a, inst.color.a);
        }
        case PAINT_TEXTURE: {
            return textureLoad(source, vec2<i32>(floor(device)), 0);
        }
        default: {
            var t: f32;
            if inst.meta_.y == PAINT_LINEAR {
                t = linear_t(inst, local);
            } else {
                t = radial_t(inst, local);
            }
            // NaN (exponent all-ones, nonzero mantissa) → transparent.
            // `t != t` is not reliable under every driver.
            let tbits = bitcast<u32>(t);
            if (tbits & 0x7f800000u) == 0x7f800000u && (tbits & 0x007fffffu) != 0u {
                return vec4<f32>(0.0);
            }
            let extend = (inst.meta_.w >> 20u) & 0xfu;
            let interp = (inst.meta_.w >> 16u) & 0xfu;
            let count = inst.meta_.w & 0xffffu;
            return eval_stops(inst.meta_.z, count, interp, extend_t(t, extend));
        }
    }
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let inst = instances[in.instance];
    let flags = (inst.meta_.w >> 24u) & 0xffu;
    var cov: f32;
    switch inst.meta_.x {
        case KIND_STROKE_OFFSET: {
            cov = shape_coverage(inst.shape, in.local, inst.affine);
            if (flags & FLAG_HAS_INNER) != 0u {
                cov -= shape_coverage(inst.inner, in.local, inst.affine);
            }
        }
        case KIND_STROKE_DIST: {
            let d = sdf(inst.shape, in.local);
            let g = device_grad(inst.affine, sdf_grad(inst.shape, in.local));
            let hw = inst.params.x;
            cov = coverage(d - hw, g) - coverage(d + hw, g);
        }
        case KIND_SHADOW: {
            let sigma = inst.params.x;
            if sigma < 0.25 {
                cov = shape_coverage(inst.shape, in.local, inst.affine);
            } else {
                cov = shadow(inst.shape, in.local, sigma);
            }
        }
        case KIND_GLYPH: {
            let texel = vec2<i32>(floor(in.device - inst.bounds.xy)) + vec2<i32>(inst.uv.xy);
            cov = textureLoad(atlas, texel, 0).r;
        }
        default: {
            cov = shape_coverage(inst.shape, in.local, inst.affine);
        }
    }
    if (flags & FLAG_HAS_CLIP) != 0u {
        // `clip_inv` maps device to clip-local: J^-T is its transpose.
        let pc = apply(inst.clip_inv, in.device);
        let g = sdf_grad(inst.clip, pc);
        let ci = inst.clip_inv;
        let dg = vec2<f32>(ci[0].x * g.x + ci[0].y * g.y, ci[0].z * g.x + ci[0].w * g.y);
        cov *= coverage(sdf(inst.clip, pc), max(length(dg), 1e-6));
    }
    cov = clamp(cov, 0.0, 1.0) * inst.params.y;
    return paint(inst, in.local, in.device) * cov;
}
