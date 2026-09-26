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
const KIND_SPAN: u32 = 5u;          // a full-coverage device-space run

const PAINT_SOLID: u32 = 0u;
const PAINT_LINEAR: u32 = 1u;
const PAINT_RADIAL: u32 = 2u;
const PAINT_TEXTURE: u32 = 3u;      // composite: sample the bound texture at the device pixel
const PAINT_SWEEP: u32 = 4u;
const PAINT_IMAGE: u32 = 5u;

const EXTEND_PAD: u32 = 0u;
const EXTEND_REPEAT: u32 = 1u;
const EXTEND_REFLECT: u32 = 2u;
const EXTEND_NONE: u32 = 3u;

const INTERP_WORKING: u32 = 0u;
const INTERP_SRGB: u32 = 1u;

const FLAG_HAS_CLIP: u32 = 1u;
const FLAG_HAS_INNER: u32 = 2u;
const FLAG_HAS_MASK: u32 = 4u;      // clip coverage x atlas mask cell

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
    // Quad rectangle (x0, y0, x1, y1). Local space, except KIND_GLYPH and
    // KIND_SPAN where it is the device-space atlas cell rectangle.
    bounds: vec4<f32>,
    shape: Shape,
    inner: Shape,
    // device -> clip-local affine, same coefficient order.
    clip_inv: array<vec4<f32>, 2>,
    // The clip shape; for a masked clip (always a sharp rect) `aspect` and
    // `exponent` — unread by its SDF — carry the mask cell size.
    clip: Shape,
    // Straight-alpha working-space colour (solid paint, glyph, shadow).
    color: vec4<f32>,
    // Linear: start.xy, end.xy. Radial: start centre.xy, end centre.xy.
    grad: vec4<f32>,
    // Radial: start radius, end radius.
    grad2: vec4<f32>,
    // Glyph/cell: atlas cell origin (x, y) in texels. zw: mask atlas origin.
    uv: vec4<f32>,
    // x: stroke half width (STROKE_DIST) or shadow sigma. y: opacity.
    // zw: mask device origin.
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
    // Device-space origin of this pass's target region.
    origin: vec2<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read> stops: array<Stop>;
@group(0) @binding(3) var atlas: texture_2d<f32>;
@group(1) @binding(0) var source: texture_2d<f32>;
// The blend backdrop: a copy of the target's region, sampled like `source`.
@group(1) @binding(1) var backdrop: texture_2d<f32>;
@group(1) @binding(2) var image_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) pixel: vec2<f32>,
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
    if inst.meta_.x == KIND_GLYPH || inst.meta_.x == KIND_SPAN {
        out.pixel = p;
        out.local = apply_inverse(inst.affine, p);
    } else {
        out.local = p;
        out.pixel = apply(inst.affine, p);
    }
    let ndc = (out.pixel - globals.origin) / globals.size * 2.0 - 1.0;
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

// Returns false when EXTEND_NONE leaves t outside [0,1] — the caller
// returns transparent. NaN fails the range test and is also rejected.
fn extend_ok(t: f32, mode: u32) -> bool {
    return mode != EXTEND_NONE || (t >= 0.0 && t <= 1.0);
}

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

fn linear_t(i: u32, p: vec2<f32>) -> f32 {
    let d = instances[i].grad.zw - instances[i].grad.xy;
    let dd = dot(d, d);
    if dd <= 0.0 {
        return 0.0;
    }
    return dot(p - instances[i].grad.xy, d) / dd;
}

// Two-point conical gradient parameter, a literal port of the oracle's
// radial_t: the larger real root of |p - (c0 + t·dc)| = r0 + t·dr.
// Degenerate coincident circles use the relative distance from the centre.
// NaN (no solution) yields a transparent pixel.
fn radial_t(i: u32, p: vec2<f32>) -> f32 {
    let c0 = instances[i].grad.xy;
    let c1 = instances[i].grad.zw;
    let r0 = instances[i].grad2.x;
    let r1 = instances[i].grad2.y;
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

// Sweep (conic) parameter: the wrapped angle of p - center mapped into
// [start_angle, end_angle). A literal port of the oracle's sweep_t.
fn sweep_t(i: u32, p: vec2<f32>) -> f32 {
    let start = instances[i].grad2.x;
    var end = instances[i].grad2.y;
    let tau = 6.283185307179586;
    while end <= start {
        end += tau;
    }
    let span = end - start;
    let c = instances[i].grad.xy;
    var theta = atan2(p.y - c.y, p.x - c.x);
    while theta < start {
        theta += tau;
    }
    while theta >= start + tau {
        theta -= tau;
    }
    return (theta - start) / span;
}

// Samples `tex` like the oracle's sample_image: texel centres at n + 0.5,
// coordinate already in image-pixel space after the per-axis extend.
fn sample_image_tex(tex: texture_2d<f32>, u: f32, v: f32, w: f32, h: f32, bilinear: bool) -> vec4<f32> {
    let dims = vec2<f32>(w, h);
    if bilinear {
        // Clamp the sample coordinate into texel-centre space before the
        // fraction: taps outside the border texels collapse onto the edge.
        let f = clamp(vec2<f32>(u, v) - 0.5, vec2<f32>(0.0), dims - 1.0);
        let lo = vec2<i32>(floor(f));
        let hi = min(lo + 1, vec2<i32>(dims) - 1);
        let t = f - floor(f);
        let c00 = textureLoad(tex, vec2<i32>(lo.x, lo.y), 0);
        let c10 = textureLoad(tex, vec2<i32>(hi.x, lo.y), 0);
        let c01 = textureLoad(tex, vec2<i32>(lo.x, hi.y), 0);
        let c11 = textureLoad(tex, vec2<i32>(hi.x, hi.y), 0);
        return mix(mix(c00, c10, t.x), mix(c01, c11, t.x), t.y);
    }
    let xy = clamp(round(vec2<f32>(u, v) - 0.5), vec2<f32>(0.0), dims - 1.0);
    return textureLoad(tex, vec2<i32>(xy), 0);
}

// Image paint: `grad`/`grad2` carry the local→image affine [a b c d e f]
// and the image size [w, h]; meta.w packs extend_x | extend_y<<4 |
// sampling<<8. The extends run in image-pixel space, like the oracle's
// eval_image_paint.
fn paint_image(i: u32, local: vec2<f32>) -> vec4<f32> {
    let g = instances[i].grad;
    let g2 = instances[i].grad2;
    let q = vec2<f32>(
        g.x * local.x + g.z * local.y + g2.x,
        g.y * local.x + g.w * local.y + g2.y,
    );
    let meta_w = instances[i].meta_.w;
    let ex = meta_w & 0xfu;
    let ey = (meta_w >> 4u) & 0xfu;
    let tu = q.x / g2.z;
    let tv = q.y / g2.w;
    if !extend_ok(tu, ex) || !extend_ok(tv, ey) {
        return vec4<f32>(0.0);
    }
    let u = extend_t(tu, ex) * g2.z;
    let v = extend_t(tv, ey) * g2.w;
    return sample_image_tex(image_tex, u, v, g2.z, g2.w, ((meta_w >> 8u) & 1u) != 0u);
}

fn paint(i: u32, local: vec2<f32>, pixel: vec2<f32>) -> vec4<f32> {
    switch instances[i].meta_.y {
        case PAINT_SOLID: {
            let color = instances[i].color;
            return vec4<f32>(color.rgb * color.a, color.a);
        }
        case PAINT_TEXTURE: {
            // `grad.xy` carries the source region's device-space origin.
            return textureLoad(source, vec2<i32>(floor(pixel - instances[i].grad.xy)), 0);
        }
        case PAINT_IMAGE: {
            return paint_image(i, local);
        }
        default: {
            var t: f32;
            if instances[i].meta_.y == PAINT_LINEAR {
                t = linear_t(i, local);
            } else if instances[i].meta_.y == PAINT_SWEEP {
                t = sweep_t(i, local);
            } else {
                t = radial_t(i, local);
            }
            // NaN (exponent all-ones, nonzero mantissa) → transparent.
            // `t != t` is not reliable under every driver.
            let tbits = bitcast<u32>(t);
            if (tbits & 0x7f800000u) == 0x7f800000u && (tbits & 0x007fffffu) != 0u {
                return vec4<f32>(0.0);
            }
            let meta_w = instances[i].meta_.w;
            let extend = (meta_w >> 20u) & 0xfu;
            let interp = (meta_w >> 16u) & 0xfu;
            let count = meta_w & 0xffffu;
            if !extend_ok(t, extend) {
                return vec4<f32>(0.0);
            }
            return eval_stops(instances[i].meta_.z, count, interp, extend_t(t, extend));
        }
    }
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Access instance fields through the storage array: copying the whole
    // 272-byte struct per fragment is expensive on tilers.
    let i = in.instance;
    let flags = (instances[i].meta_.w >> 24u) & 0xffu;
    var cov: f32;
    switch instances[i].meta_.x {
        case KIND_STROKE_OFFSET: {
            cov = shape_coverage(instances[i].shape, in.local, instances[i].affine);
            if (flags & FLAG_HAS_INNER) != 0u {
                cov -= shape_coverage(instances[i].inner, in.local, instances[i].affine);
            }
        }
        case KIND_STROKE_DIST: {
            let d = sdf(instances[i].shape, in.local);
            let g = device_grad(instances[i].affine, sdf_grad(instances[i].shape, in.local));
            let hw = instances[i].params.x;
            cov = coverage(d - hw, g) - coverage(d + hw, g);
        }
        case KIND_SHADOW: {
            let sigma = instances[i].params.x;
            if sigma < 0.25 {
                cov = shape_coverage(instances[i].shape, in.local, instances[i].affine);
            } else {
                cov = shadow(instances[i].shape, in.local, sigma);
            }
        }
        case KIND_GLYPH: {
            let texel = vec2<i32>(floor(in.pixel - instances[i].bounds.xy)) + vec2<i32>(instances[i].uv.xy);
            cov = textureLoad(atlas, texel, 0).r;
        }
        case KIND_SPAN: {
            cov = 1.0;
        }
        default: {
            cov = shape_coverage(instances[i].shape, in.local, instances[i].affine);
        }
    }
    if (flags & FLAG_HAS_CLIP) != 0u {
        // `clip_inv` maps device to clip-local: J^-T is its transpose.
        let pc = apply(instances[i].clip_inv, in.pixel);
        let g = sdf_grad(instances[i].clip, pc);
        let ci = instances[i].clip_inv;
        let dg = vec2<f32>(ci[0].x * g.x + ci[0].y * g.y, ci[0].z * g.x + ci[0].w * g.y);
        cov *= coverage(sdf(instances[i].clip, pc), max(length(dg), 1e-6));
    }
    if (flags & FLAG_HAS_MASK) != 0u {
        // Mask texel for this device pixel; texels outside the cell
        // contribute zero coverage.
        let mp = floor(in.pixel) - instances[i].params.zw;
        let msize = vec2<f32>(instances[i].clip.aspect, instances[i].clip.exponent);
        let inside = all(mp >= vec2<f32>(0.0)) && all(mp < msize);
        cov *= select(0.0, textureLoad(atlas, vec2<i32>(mp) + vec2<i32>(instances[i].uv.zw), 0).r, inside);
    }
    cov = clamp(cov, 0.0, 1.0) * instances[i].params.y;
    // A blended composite carries its mode in meta.w bits 16-23: sample the
    // source and backdrop, blend, and write the composited result verbatim
    // (the pass runs the Replace pipeline).
    if instances[i].meta_.y == PAINT_TEXTURE {
        let mode = (instances[i].meta_.w >> 16u) & 0xffu;
        if mode != 0u {
            let coord = vec2<i32>(floor(in.pixel - instances[i].grad.xy));
            let cs = textureLoad(source, coord, 0) * cov;
            let cb = textureLoad(backdrop, coord, 0);
            return blend_color(mode, cb, cs);
        }
    }
    return paint(i, in.local, in.pixel) * cov;
}

// W3C Compositing and Blending Level 1, a literal port of
// oracle/src/blend.rs. Premultiplied inputs and output.

fn lum(c: vec3<f32>) -> f32 {
    return 0.3 * c.x + 0.59 * c.y + 0.11 * c.z;
}

fn sat(c: vec3<f32>) -> f32 {
    return max(c.x, max(c.y, c.z)) - min(c.x, min(c.y, c.z));
}

fn clip_color(c_in: vec3<f32>) -> vec3<f32> {
    var c = c_in;
    let l = lum(c);
    let n = min(c.x, min(c.y, c.z));
    let x = max(c.x, max(c.y, c.z));
    if n < 0.0 {
        c = l + (c - l) * l / (l - n);
    }
    if x > 1.0 {
        c = l + (c - l) * (1.0 - l) / (x - l);
    }
    return c;
}

fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    let d = l - lum(c);
    return clip_color(c + d);
}

fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    var mn = 0;
    var mx = 0;
    for (var i = 1; i < 3; i += 1) {
        if c[i] < c[mn] {
            mn = i;
        }
        if c[i] > c[mx] {
            mx = i;
        }
    }
    let imid = 3 - mn - mx;
    var out = vec3<f32>(0.0);
    if c[mx] > c[mn] {
        out[imid] = (c[imid] - c[mn]) * s / (c[mx] - c[mn]);
        out[mx] = s;
    }
    return out;
}

// B(Cb, Cs) for one channel pair, separable modes; non-separable modes and
// Porter-Duff operators are handled in blend_color, not here.
fn blend_channel(mode: u32, cb: f32, cs: f32) -> f32 {
    switch mode {
        case 1u: { return cb * cs; }                                 // Multiply
        case 2u: { return cb + cs - cb * cs; }                       // Screen
        case 3u: {                                                   // Overlay
            if cb <= 0.5 { return 2.0 * cb * cs; }
            return 1.0 - 2.0 * (1.0 - cb) * (1.0 - cs);
        }
        case 4u: { return min(cb, cs); }                             // Darken
        case 5u: { return max(cb, cs); }                             // Lighten
        case 6u: {                                                   // ColorDodge
            if cs >= 1.0 { return 1.0; }
            return min(cb / (1.0 - cs), 1.0);
        }
        case 7u: {                                                   // ColorBurn
            if cs <= 0.0 { return 0.0; }
            return 1.0 - min((1.0 - cb) / cs, 1.0);
        }
        case 8u: {                                                   // HardLight
            if cs <= 0.5 { return 2.0 * cb * cs; }
            return 1.0 - 2.0 * (1.0 - cb) * (1.0 - cs);
        }
        case 9u: {                                                   // SoftLight
            if cs <= 0.5 {
                return cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb);
            }
            var d: f32;
            if cb <= 0.25 {
                d = ((16.0 * cb - 12.0) * cb + 4.0) * cb;
            } else {
                d = sqrt(cb);
            }
            return cb + (2.0 * cs - 1.0) * (d - cb);
        }
        case 10u: { return abs(cb - cs); }                           // Difference
        case 11u: { return cb + cs - 2.0 * cb * cs; }                // Exclusion
        default: { return cs; }
    }
}

// Porter-Duff: co = αs·Fa·Cs + αb·Fb·Cb in premultiplied form.
fn porter_duff(fa: f32, fb: f32, cb: vec4<f32>, cs: vec4<f32>) -> vec4<f32> {
    return fa * cs + fb * cb;
}

// blend(mode, cb, cs): premultiplied backdrop and source, composited
// premultiplied output — oracle blend().
fn blend_color(mode: u32, cb: vec4<f32>, cs: vec4<f32>) -> vec4<f32> {
    let ab = cb.a;
    let as_ = cs.a;
    switch mode {
        case 16u: { return vec4<f32>(0.0); }                              // Clear
        case 17u: { return cs; }                                          // Src
        case 18u: { return cb; }                                          // Dst
        case 19u: { return porter_duff(1.0 - ab, 1.0, cb, cs); }           // DestOver
        case 20u: { return porter_duff(ab, 0.0, cb, cs); }                 // SrcIn
        case 21u: { return porter_duff(0.0, as_, cb, cs); }                // DestIn
        case 22u: { return porter_duff(1.0 - ab, 0.0, cb, cs); }           // SrcOut
        case 23u: { return porter_duff(0.0, 1.0 - as_, cb, cs); }          // DestOut
        case 24u: { return porter_duff(ab, 1.0 - as_, cb, cs); }           // SrcAtop
        case 25u: { return porter_duff(1.0 - ab, as_, cb, cs); }           // DestAtop
        case 26u: { return porter_duff(1.0 - ab, 1.0 - as_, cb, cs); }     // Xor
        case 27u: { return porter_duff(1.0, 1.0, cb, cs); }                // PlusLighter
        default: {}
    }
    if as_ == 0.0 {
        return cb;
    }
    var ub = vec3<f32>(0.0);
    if ab > 0.0 {
        ub = cb.rgb / ab;
    }
    let us = cs.rgb / as_;
    var b: vec3<f32>;
    switch mode {
        case 12u: { b = set_lum(set_sat(us, sat(ub)), lum(ub)); }          // Hue
        case 13u: { b = set_lum(set_sat(ub, sat(us)), lum(ub)); }          // Saturation
        case 14u: { b = set_lum(us, lum(ub)); }                            // Color
        case 15u: { b = set_lum(ub, lum(us)); }                            // Luminosity
        default: {
            b = vec3<f32>(
                blend_channel(mode, ub.x, us.x),
                blend_channel(mode, ub.y, us.y),
                blend_channel(mode, ub.z, us.z),
            );
        }
    }
    // Cr = (1-αb)·Cs + αb·B(Cb,Cs); premultiplied: αs·Cr + (1-αs)·Cb.
    var out: vec4<f32>;
    for (var i = 0; i < 3; i += 1) {
        let cr = (1.0 - ab) * us[i] + ab * b[i];
        out[i] = as_ * cr + (1.0 - as_) * cb[i];
    }
    out.a = as_ + ab * (1.0 - as_);
    return out;
}
