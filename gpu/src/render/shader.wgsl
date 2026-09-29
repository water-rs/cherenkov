// Cherenkov GPU slice: one instanced quad pipeline for every primitive family.
//
// Every instance is a quad. The vertex shader places it; the fragment shader
// computes analytic coverage (an SDF, a Gaussian-blurred rounded box, or an
// atlas texel), multiplies by the instance's clip coverage and opacity, and
// evaluates the paint at the pixel centre in the instance's local space. The
// output is premultiplied linear Display P3 into an rgba16float target with
// fixed-function src-over blending.
//
// `shared.wgsl` precedes this file in every build: the consts, instance
// layout, vertex stage and coverage machinery live there.

// The Rust side prepends `const VARIANT: u32 = <n>u;` when building each
// module; the file stays compilable standalone.
const VARIANT_SIMPLE: u32 = 0u;
const VARIANT_SHADOW: u32 = 1u;
const VARIANT_FULL: u32 = 2u;

@group(1) @binding(0) var source: texture_2d<f32>;
// The blend backdrop: a copy of the target's region, sampled like `source`.
@group(1) @binding(1) var backdrop: texture_2d<f32>;
@group(1) @binding(2) var image_tex: texture_2d<f32>;

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

// One corner-band row of the shadow integrand: the analytic x integral of
// row `y` (sides inset by the corners of radii `rl`, `rr`) times the
// Gaussian weight of its distance to `p.y`.
fn corner_row(s: Shape, p: vec2<f32>, sigma: f32, k: f32, rl: f32, rr: f32, y: f32) -> f32 {
    let ay = abs(y);
    let xl = -s.half.x + corner_inset(rl, ay - (s.half.y - rl));
    let xr = s.half.x - corner_inset(rr, ay - (s.half.y - rr));
    if xr <= xl {
        return 0.0;
    }
    let row = 0.5 * (erf((xr - p.x) * k) - erf((xl - p.x) * k));
    return row * gaussian(y - p.y, sigma);
}

// The symmetric pair of Gauss–Legendre nodes `mid ± hw * x` with weight `w`.
fn corner_pair(
    s: Shape, p: vec2<f32>, sigma: f32, k: f32, rl: f32, rr: f32,
    mid: f32, hw: f32, x: f32, w: f32,
) -> f32 {
    return (corner_row(s, p, sigma, k, rl, rr, mid - hw * x)
        + corner_row(s, p, sigma, k, rl, rr, mid + hw * x)) * w * hw;
}

// Indicator of the rounded box `s` (circular corners), convolved with an
// isotropic Gaussian of standard deviation `sigma`, evaluated at `p`. The x
// integral of each row is analytic (erf); the y integral over the corner
// bands within ±3σ is an 8-point Gauss–Legendre rule.
fn shadow(s: Shape, p: vec2<f32>, sigma: f32) -> f32 {
    let k = 1.0 / (sigma * 1.4142135624);
    // Rows within ±3σ of p that intersect the box.
    let lo = max(p.y - 3.0 * sigma, -s.half.y);
    let hi = min(p.y + 3.0 * sigma, s.half.y);
    if hi <= lo {
        return 0.0;
    }
    let rmax = max(max(max(s.radii.x, s.radii.y), max(s.radii.z, s.radii.w)), 0.0);
    // Rows whose corner insets are beyond 4σ of p integrate like straight rows.
    let straight = abs(p.x) <= s.half.x - rmax - 4.0 * sigma;
    // Rows |y| < band have straight sides: the integral is separable.
    let band = select(max(s.half.y - rmax, 0.0), s.half.y, straight);
    var acc = 0.0;
    let ya = max(lo, -band);
    let yb = min(hi, band);
    if yb > ya {
        let row = 0.5 * (erf((s.half.x - p.x) * k) - erf((-s.half.x - p.x) * k));
        acc += row * 0.5 * (erf((yb - p.y) * k) - erf((ya - p.y) * k));
    }
    // Corner bands: Gauss–Legendre over the rows still inside ±3σ.
    for (var side = 0; side < 2; side++) {
        let top = side == 0;
        let a = max(select(band, lo, top), lo);
        let b = min(select(hi, -band, top), hi);
        if b <= a {
            continue;
        }
        let rl = select(s.radii.w, s.radii.x, top);
        let rr = select(s.radii.z, s.radii.y, top);
        let mid = 0.5 * (a + b);
        let hw = 0.5 * (b - a);
        acc += corner_pair(s, p, sigma, k, rl, rr, mid, hw, 0.1834346425, 0.3626837834);
        acc += corner_pair(s, p, sigma, k, rl, rr, mid, hw, 0.5255324099, 0.3137066459);
        acc += corner_pair(s, p, sigma, k, rl, rr, mid, hw, 0.7966664774, 0.2223810345);
        acc += corner_pair(s, p, sigma, k, rl, rr, mid, hw, 0.9602898565, 0.1012285363);
    }
    return clamp(acc, 0.0, 1.0);
}

fn srgb_encode(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
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
// Explicit validity avoids a non-finite constant, which WGSL rejects.
struct RadialParameter {
    value: f32,
    valid: bool,
}
fn radial_t(i: u32, p: vec2<f32>) -> RadialParameter {
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
                return RadialParameter(0.0, true);
            }
            return RadialParameter((length(pd) - r0) / abs(r0), true);
        }
        return RadialParameter(-c / b, true);
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return RadialParameter(0.0, false);
    }
    let sq = sqrt(disc);
    // The cone answer is the larger root; when `a` is negative that is the
    // smaller numerator, so compare the roots themselves.
    return RadialParameter(max((-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)), true);
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
// and the image size [w, h]; meta_.w packs extend_x | extend_y<<4 |
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

// Bilinear inverse in f32, solving for v. The f64 oracle solves for u.
fn mesh_cross(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

fn mesh_uv(point: vec2<f32>, top: vec4<f32>, bottom: vec4<f32>) -> vec3<f32> {
    let horizontal = top.zw - top.xy;
    let vertical = bottom.xy - top.xy;
    let bend = bottom.zw - bottom.xy - horizontal;
    let delta = point - top.xy;
    let qa = -mesh_cross(vertical, bend);
    let qb = mesh_cross(delta, bend) - mesh_cross(vertical, horizontal);
    let qc = mesh_cross(delta, horizontal);
    var roots = vec2<f32>(-1.0);
    if qa == 0.0 {
        if qb == 0.0 { return vec3<f32>(0.0); }
        roots = vec2<f32>(-qc / qb);
    } else {
        let discriminant = qb * qb - 4.0 * qa * qc;
        if discriminant < 0.0 { return vec3<f32>(0.0); }
        let signed_root = select(-sqrt(discriminant), sqrt(discriminant), qb >= 0.0);
        let numerator = -0.5 * (qb + signed_root);
        roots = vec2<f32>(numerator / qa);
        if numerator != 0.0 { roots.y = qc / numerator; }
    }
    var answer = vec3<f32>(0.0);
    for (var index = 0u; index < 2u; index += 1u) {
        let v = roots[index];
        if !(v >= 0.0 && v <= 1.0) { continue; }
        let direction = horizontal + v * bend;
        let remainder = delta - v * vertical;
        var u: f32;
        if abs(direction.x) >= abs(direction.y) {
            if direction.x == 0.0 { continue; }
            u = remainder.x / direction.x;
        } else {
            u = remainder.y / direction.y;
        }
        if !(u >= 0.0 && u <= 1.0) { continue; }
        if mesh_cross(direction, vertical + u * bend) == 0.0 { continue; }
        if answer.z == 0.0 || v > answer.y || (v == answer.y && u > answer.x) {
            answer = vec3<f32>(u, v, 1.0);
        }
    }
    return answer;
}

fn paint_mesh(first: u32, count: u32, point: vec2<f32>, smooth_color: bool) -> vec4<f32> {
    for (var remaining = count; remaining > 0u; remaining -= 1u) {
        let base = first + (remaining - 1u) * 6u;
        let uv = mesh_uv(point, stops[base].color, stops[base + 1u].color);
        if uv.z != 0.0 {
            var weight = uv.xy;
            if smooth_color { weight = weight * weight * (3.0 - 2.0 * weight); }
            let top = mix(stops[base + 2u].color, stops[base + 3u].color, weight.x);
            let bottom = mix(stops[base + 4u].color, stops[base + 5u].color, weight.x);
            return mix(top, bottom, weight.y);
        }
    }
    return vec4<f32>(0.0);
}

fn paint(i: u32, meta_: vec4<u32>, color: vec4<f32>, local: vec2<f32>, pixel: vec2<f32>) -> vec4<f32> {
    let kind = meta_.y & 0xffffu;
    var point = local;
    if (meta_.y & 0x10000u) != 0u {
        let linear = stops[meta_.z - 2u].color;
        let offset = stops[meta_.z - 1u].color.xy;
        point = vec2<f32>(linear.x * local.x + linear.z * local.y,
                          linear.y * local.x + linear.w * local.y) + offset;
    }
    switch kind {
        case PAINT_SOLID: {
            return vec4<f32>(color.rgb * color.a, color.a);
        }
        case PAINT_TEXTURE: {
            // `grad.xy` carries the source region's device-space origin.
            return textureLoad(source, vec2<i32>(floor(pixel - instances[i].grad.xy)), 0);
        }
        case PAINT_MESH: {
            return paint_mesh(meta_.z, meta_.w & 0x00ffffffu, point, (meta_.y & 0x20000u) != 0u);
        }
        case PAINT_IMAGE: {
            return paint_image(i, point);
        }
        default: {
            var t: f32;
            if kind == PAINT_LINEAR {
                t = linear_t(i, point);
            } else if kind == PAINT_SWEEP {
                t = sweep_t(i, point);
            } else {
                let radial = radial_t(i, point);
                if !radial.valid {
                    return vec4<f32>(0.0);
                }
                t = radial.value;
            }
            // NaN (exponent all-ones, nonzero mantissa) → transparent.
            // `t != t` is not reliable under every driver.
            let tbits = bitcast<u32>(t);
            if (tbits & 0x7f800000u) == 0x7f800000u && (tbits & 0x007fffffu) != 0u {
                return vec4<f32>(0.0);
            }
            let meta_w = meta_.w;
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
    if VARIANT == VARIANT_SIMPLE {
        return fs_simple(in);
    }
    if VARIANT == VARIANT_SHADOW {
        return fs_shadow(in);
    }
    return fs_full(in);
}

// Solid fill/span/glyph coverage: no clip, mask, inner, or paint()
// evaluation, and no `instances` reads at all.
fn fs_simple(in: VsOut) -> vec4<f32> {
    let s = Shape(in.shape_a.xy, in.shape_a.z, in.shape_a.w, in.shape_radii);
    let m = array<vec4<f32>, 2>(in.affine0, in.affine1);
    var cov: f32;
    switch in.meta_.x {
        case KIND_GLYPH: {
            let texel = vec2<i32>(floor(in.pixel - in.cell.xy)) + vec2<i32>(in.cell.zw);
            cov = textureLoad(atlas, texel, 0).r;
        }
        case KIND_SPAN: {
            cov = 1.0;
        }
        default: {
            cov = shape_coverage(s, in.local, m);
        }
    }
    cov = clamp(cov, 0.0, 1.0) * in.params.y;
    return vec4<f32>(in.color.rgb * in.color.a, in.color.a) * cov;
}

// The shadow kernel plus the same opacity/solid-colour tail.
fn fs_shadow(in: VsOut) -> vec4<f32> {
    let s = Shape(in.shape_a.xy, in.shape_a.z, in.shape_a.w, in.shape_radii);
    let m = array<vec4<f32>, 2>(in.affine0, in.affine1);
    let sigma = in.params.x;
    var cov: f32;
    if sigma < 0.25 {
        cov = shape_coverage(s, in.local, m);
    } else {
        cov = shadow(s, in.local, sigma);
    }
    cov = clamp(cov, 0.0, 1.0) * in.params.y;
    return vec4<f32>(in.color.rgb * in.color.a, in.color.a) * cov;
}

fn fs_full(in: VsOut) -> vec4<f32> {
    // The constants every fragment needs arrive as flat varyings; the
    // storage array is read only for kind-specific fields (inner, clip,
    // mask, gradient data).
    let i = in.instance;
    let s = Shape(in.shape_a.xy, in.shape_a.z, in.shape_a.w, in.shape_radii);
    let m = array<vec4<f32>, 2>(in.affine0, in.affine1);
    let flags = (in.meta_.w >> 24u) & 0xffu;
    var cov: f32;
    switch in.meta_.x {
        case KIND_STROKE_OFFSET: {
            cov = shape_coverage(s, in.local, m);
            if (flags & FLAG_HAS_INNER) != 0u {
                cov -= shape_coverage(instances[i].inner, in.local, m);
            }
        }
        case KIND_STROKE_DIST: {
            let d = sdf(s, in.local);
            let g = sdf_grad(s, in.local);
            let v = device_grad_vec(m, g.xy);
            let scale = device_grad_scale(m);
            let ramp = g.w > 0.0;
            let hw = in.params.x;
            cov = coverage_dir(d - hw, v, scale, ramp, select(g.z + hw, 0.0, g.z <= 0.0))
                - coverage_dir(d + hw, v, scale, ramp, select(max(g.z - hw, 0.0), 0.0, g.z <= 0.0));
        }
        case KIND_SHADOW: {
            let sigma = in.params.x;
            if sigma < 0.25 {
                cov = shape_coverage(s, in.local, m);
            } else {
                cov = shadow(s, in.local, sigma);
            }
        }
        case KIND_GLYPH: {
            let texel = vec2<i32>(floor(in.pixel - in.cell.xy)) + vec2<i32>(in.cell.zw);
            cov = textureLoad(atlas, texel, 0).r;
        }
        case KIND_SPAN: {
            cov = 1.0;
        }
        default: {
            cov = shape_coverage(s, in.local, m);
        }
    }
    cov *= clip_mask_coverage(in);
    // Coverage before the opacity multiply is the composite's clip coverage:
    // the destructive Porter-Duff branch antialiases the clip edge between
    // the backdrop and the blended result.
    let inside_cov = clamp(cov, 0.0, 1.0);
    cov = inside_cov * in.params.y;
    // A blended composite carries its mode in meta_.w bits 16-23: sample the
    // source and backdrop, blend, and write the composited result verbatim
    // (the pass runs the Replace pipeline).
    if in.meta_.y == PAINT_TEXTURE {
        let mode = (in.meta_.w >> 16u) & 0xffu;
        if mode != 0u {
            let coord = vec2<i32>(floor(in.pixel - instances[i].grad.xy));
            if blend_is_destructive(mode) {
                // Destructive operators composite over the whole region: a
                // transparent source still writes over the backdrop.
                let cs = textureLoad(source, coord, 0) * in.params.y;
                let cb = textureLoad(backdrop, coord, 0);
                return mix(cb, blend_color(mode, cb, cs), inside_cov);
            }
            let cs = textureLoad(source, coord, 0) * cov;
            let cb = textureLoad(backdrop, coord, 0);
            return blend_color(mode, cb, cs);
        }
    }
    return paint(i, in.meta_, in.color, in.local, in.pixel) * cov;
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

// The Porter-Duff operators whose transparent source replaces the
// destination rather than leaving it unchanged (codes per blend_code).
fn blend_is_destructive(mode: u32) -> bool {
    return mode == 16u || mode == 17u || mode == 20u || mode == 21u || mode == 22u || mode == 25u;
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
        case 27u: { return vec4<f32>(cs.rgb + cb.rgb, min(as_ + ab, 1.0)); } // PlusLighter
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
