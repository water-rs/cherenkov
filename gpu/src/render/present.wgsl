// Presents retained premultiplied linear Display P3 on a host attachment.
// sRGB output premultiplies after the transfer function, including when
// hardware applies that transfer. Linear P3 output preserves extended values.
//
// Out-of-gamut linear sRGB components are gamut-mapped — preserving hue
// instead of clipping each channel (#96). The map is Ottosson's analytic
// OKLab clip, chosen over the CSS Color 4 binary search for its fixed
// per-pixel cost (the search measured ~2.7x slower on lavapipe while the
// analytic clip stayed under one ΔE_OK JND of it on the corpus sweep).

struct Present {
    // 1: shader sRGB transfer, 0: hardware sRGB transfer, 2: linear P3.
    encode: u32,
    // 0: opaque (alpha forced to 1), 1: premultiplied in the output space,
    // 2: straight alpha.
    alpha: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> present: Present;

struct Vertex {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> Vertex {
    // One triangle covering the clip-space square.
    let x = f32(i32(index & 1u) * 4 - 1);
    let y = f32(i32(index >> 1u) * 4 - 1);
    var out: Vertex;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

fn srgb_encode(c: f32) -> f32 {
    if c <= 0.0031308 {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

fn srgb_decode(c: f32) -> f32 {
    if c <= 0.04045 { return c / 12.92; }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn present_color(rgb: vec3<f32>, alpha: f32) -> vec4<f32> {
    if present.alpha == 1u {
        return vec4<f32>(rgb * alpha, alpha);
    }
    return vec4<f32>(rgb, alpha);
}

// ---- Gamut mapping (linear sRGB in [0,1] bounds) -----------------------

// Signed cube root — the OKLab LMS nonlinearity for possibly-negative
// components an out-of-gamut colour produces.
fn gamut_cbrt(x: f32) -> f32 {
    return sign(x) * pow(abs(x), 1.0 / 3.0);
}

fn srgb_to_oklab(rgb: vec3<f32>) -> vec3<f32> {
    let l_ = gamut_cbrt(0.4122214708 * rgb.r + 0.5363325363 * rgb.g + 0.0514459929 * rgb.b);
    let m_ = gamut_cbrt(0.2119034982 * rgb.r + 0.6806995451 * rgb.g + 0.1073969566 * rgb.b);
    let s_ = gamut_cbrt(0.0883024619 * rgb.r + 0.2817188376 * rgb.g + 0.6299787005 * rgb.b);
    return vec3<f32>(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    );
}

fn oklab_to_srgb(lab: vec3<f32>) -> vec3<f32> {
    let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    return vec3<f32>(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    );
}

fn in_gamut(rgb: vec3<f32>) -> bool {
    return all(rgb >= vec3<f32>(0.0)) && all(rgb <= vec3<f32>(1.0));
}

// The analytic OKLab clip — Ottosson's cusp-triangle boundary
// with one Halley refinement, projecting towards the lightness axis.

// The cubic and its two derivatives of an sRGB channel along the L=1
// saturation ray, evaluated at s.
fn gamut_sat_eval(s: f32, w: vec3<f32>, kl: vec3<f32>) -> vec3<f32> {
    let l_ = 1.0 + s * kl.x;
    let m_ = 1.0 + s * kl.y;
    let s_ = 1.0 + s * kl.z;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let sc = s_ * s_ * s_;
    let l_ds = 3.0 * kl.x * l_ * l_;
    let m_ds = 3.0 * kl.y * m_ * m_;
    let s_ds = 3.0 * kl.z * s_ * s_;
    let l_ds2 = 6.0 * kl.x * kl.x * l_;
    let m_ds2 = 6.0 * kl.y * kl.y * m_;
    let s_ds2 = 6.0 * kl.z * kl.z * s_;
    return vec3<f32>(
        w.x * l + w.y * m + w.z * sc,
        w.x * l_ds + w.y * m_ds + w.z * s_ds,
        w.x * l_ds2 + w.y * m_ds2 + w.z * s_ds2,
    );
}

// One channel of ok_color.h's compute_max_saturation: the polynomial
// estimate plus one Halley step, returning (root, |residual at root|).
fn gamut_sat_candidate(k: array<f32, 5>, w: vec3<f32>, kl: vec3<f32>, a: f32, b: f32) -> vec2<f32> {
    let est = k[0] + k[1] * a + k[2] * b + k[3] * a * a + k[4] * a * b;
    let e = gamut_sat_eval(est, w, kl);
    let den = e.y * e.y - 0.5 * e.x * e.z;
    if den == 0.0 {
        return vec2<f32>(-1.0, 1.0);
    }
    let root = est - e.x * e.y / den;
    return vec2<f32>(root, abs(gamut_sat_eval(root, w, kl).x));
}

// Max saturation S = C/L for the normalized hue direction (a, b).
// ok_color.h picks one channel's surface by a fitted hue partition; that
// partition is a ~1e-6-sensitive coin flip at the sRGB vertex hues across
// f32/f64, so all three channel surfaces are solved instead and the
// smallest converged root wins — one Halley step each, deterministic.
fn gamut_max_saturation(a: f32, b: f32) -> f32 {
    let kl = vec3<f32>(
        0.3963377774 * a + 0.2158037573 * b,
        -0.1055613458 * a - 0.0638541728 * b,
        -0.0894841775 * a - 1.2914855480 * b,
    );
    var cands: array<vec2<f32>, 3>;
    cands[0] = gamut_sat_candidate(
        array<f32, 5>(1.19086277, 1.76576728, 0.59662641, 0.75515197, 0.56771245),
        vec3<f32>(4.0767416621, -3.3077115913, 0.2309699292), kl, a, b);
    cands[1] = gamut_sat_candidate(
        array<f32, 5>(0.73956515, -0.45954404, 0.08285427, 0.12541070, 0.14503204),
        vec3<f32>(-1.2684380046, 2.6097574011, -0.3413193965), kl, a, b);
    cands[2] = gamut_sat_candidate(
        array<f32, 5>(1.35733652, -0.00915799, -1.15130210, -0.50559606, 0.00692167),
        vec3<f32>(-0.0041960863, -0.7034186147, 1.7076147010), kl, a, b);
    var s = 3.4e38;
    for (var i = 0; i < 3; i++) {
        let cand = cands[i];
        if cand.x > 0.0 && cand.y < 0.05 {
            s = min(s, cand.x);
        }
    }
    if s < 1.0e38 {
        return s;
    }
    // No step converged: fall back to the smallest positive estimate.
    var est_s = 3.4e38;
    // Recompute the three polynomial estimates for the fallback.
    let a2 = a * a;
    let ab = a * b;
    est_s = min(est_s, 1.19086277 + 1.76576728 * a + 0.59662641 * b + 0.75515197 * a2 + 0.56771245 * ab);
    est_s = min(est_s, 0.73956515 - 0.45954404 * a + 0.08285427 * b + 0.12541070 * a2 + 0.14503204 * ab);
    est_s = min(est_s, 1.35733652 - 0.00915799 * a - 1.15130210 * b - 0.50559606 * a2 + 0.00692167 * ab);
    return max(est_s, 0.0);
}

// The sRGB cusp (L, C) of the hue slice — ok_color.h's find_cusp.
fn gamut_cusp(a: f32, b: f32) -> vec2<f32> {
    let s = gamut_max_saturation(a, b);
    let rgb = oklab_to_srgb(vec3<f32>(1.0, s * a, s * b));
    let l_cusp = gamut_cbrt(1.0 / max(rgb.r, max(rgb.g, rgb.b)));
    return vec2<f32>(l_cusp, l_cusp * s);
}

// Intersects the segment (l0,0) -> (l1,c1) with the gamut boundary in the
// hue slice — ok_color.h's find_gamut_intersection.
fn gamut_intersection(a: f32, b: f32, l1: f32, c1: f32, l0: f32, cusp: vec2<f32>) -> f32 {
    var t: f32;
    if (l1 - l0) * cusp.y - (cusp.x - l0) * c1 <= 0.0 {
        t = cusp.y * l0 / (c1 * cusp.x + cusp.y * (l0 - l1));
    } else {
        t = cusp.y * (l0 - 1.0) / (c1 * (cusp.x - 1.0) + cusp.y * (l0 - l1));
        let dl = l1 - l0;
        let dc = c1;
        let k_l = 0.3963377774 * a + 0.2158037573 * b;
        let k_m = -0.1055613458 * a - 0.0638541728 * b;
        let k_s = -0.0894841775 * a - 1.2914855480 * b;
        let l_dt = dl + dc * k_l;
        let m_dt = dl + dc * k_m;
        let s_dt = dl + dc * k_s;
        let l_at = l0 * (1.0 - t) + t * l1;
        let c_at = t * c1;
        let l_ = l_at + c_at * k_l;
        let m_ = l_at + c_at * k_m;
        let s_ = l_at + c_at * k_s;
        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let s = s_ * s_ * s_;
        let ldt = 3.0 * l_dt * l_ * l_;
        let mdt = 3.0 * m_dt * m_ * m_;
        let sdt = 3.0 * s_dt * s_ * s_;
        let ldt2 = 6.0 * l_dt * l_dt * l_;
        let mdt2 = 6.0 * m_dt * m_dt * m_;
        let sdt2 = 6.0 * s_dt * s_dt * s_;
        let r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s - 1.0;
        let r1 = 4.0767416621 * ldt - 3.3077115913 * mdt + 0.2309699292 * sdt;
        let r2 = 4.0767416621 * ldt2 - 3.3077115913 * mdt2 + 0.2309699292 * sdt2;
        let g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s - 1.0;
        let g1 = -1.2684380046 * ldt + 2.6097574011 * mdt - 0.3413193965 * sdt;
        let g2 = -1.2684380046 * ldt2 + 2.6097574011 * mdt2 - 0.3413193965 * sdt2;
        let bch = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s - 1.0;
        let b1 = -0.0041960863 * ldt - 0.7034186147 * mdt + 1.7076147010 * sdt;
        let b2 = -0.0041960863 * ldt2 - 0.7034186147 * mdt2 + 1.7076147010 * sdt2;
        let u_r = r1 / (r1 * r1 - 0.5 * r * r2);
        let u_g = g1 / (g1 * g1 - 0.5 * g * g2);
        let u_b = b1 / (b1 * b1 - 0.5 * bch * b2);
        var t_r = select(3.4e38, -r * u_r, u_r >= 0.0);
        var t_g = select(3.4e38, -g * u_g, u_g >= 0.0);
        var t_b = select(3.4e38, -bch * u_b, u_b >= 0.0);
        t += min(t_r, min(t_g, t_b));
    }
    return t;
}

fn gamut_map(rgb: vec3<f32>) -> vec3<f32> {
    // In-gamut colours keep the pre-#96 bits: map and clamp agree exactly.
    if in_gamut(rgb) {
        return rgb;
    }
    let lab = srgb_to_oklab(rgb);
    // Local-MINDE (ΔE_OK JND = 0.02): when the plain channel-clip is
    // already within a JND of the colour, keep the clip's bytes — the
    // pre-#96 output. A colour a few ULPs out of gamut (an in-sRGB value
    // round-tripped through the P3 matrices) then never enters the
    // projection where the cusp fit is weakest.
    let clipped = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    if distance(srgb_to_oklab(clipped), lab) < 0.02 {
        return clipped;
    }
    let l = lab.x;
    let chroma = length(lab.yz);
    if chroma <= 0.00001 {
        return clamp(oklab_to_srgb(vec3<f32>(clamp(l, 0.0, 1.0), 0.0, 0.0)),
                     vec3<f32>(0.0), vec3<f32>(1.0));
    }
    let c = max(0.00001, chroma);
    let a_ = lab.y / c;
    let b_ = lab.z / c;
    let cusp = gamut_cusp(a_, b_);
    // gamut_clip_adaptive_L0_L_cusp (alpha = 0.05): the anchor blends from
    // the colour's lightness towards the cusp's as the colour recedes.
    let ld = l - cusp.x;
    let k = 2.0 * select(cusp.x, 1.0 - cusp.x, ld > 0.0);
    let e1 = 0.5 * k + abs(ld) + 0.05 * c / k;
    let l0 = cusp.x + 0.5 * sign(ld) * (e1 - sqrt(e1 * e1 - 2.0 * k * abs(ld)));
    let t = gamut_intersection(a_, b_, l, c, l0, cusp);
    let l_out = l0 * (1.0 - t) + t * l;
    let c_out = t * c;
    return clamp(oklab_to_srgb(vec3<f32>(l_out, c_out * a_, c_out * b_)),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: Vertex) -> @location(0) vec4<f32> {
    var p3 = textureSample(source, source_sampler, in.uv);
    var alpha = 1.0;
    if present.alpha != 0u {
        alpha = p3.a;
        if p3.a > 0.0 {
            p3 = vec4<f32>(p3.rgb / p3.a, p3.a);
        }
    }
    if present.encode == 2u {
        return present_color(p3.rgb, alpha);
    }
    // Linear Display P3 → linear sRGB (Bradford-adapted, D65).
    let rgb = vec3<f32>(
        1.2249402 * p3.r - 0.2249402 * p3.g,
        -0.04205695 * p3.r + 1.0420569 * p3.g,
        -0.01963755 * p3.r - 0.07863605 * p3.g + 1.0982736 * p3.b,
    );
    let clamped = gamut_map(rgb);
    if present.encode == 1u {
        return present_color(vec3<f32>(
            srgb_encode(clamped.r),
            srgb_encode(clamped.g),
            srgb_encode(clamped.b),
        ), alpha);
    }
    // Hardware applies the transfer after this shader. Undo the transfer of
    // the encoded-domain premultiplied result so stored bytes match Unorm.
    let encoded = vec3<f32>(srgb_encode(clamped.r), srgb_encode(clamped.g), srgb_encode(clamped.b));
    let result = present_color(encoded, alpha);
    return vec4<f32>(srgb_decode(result.r), srgb_decode(result.g), srgb_decode(result.b), result.a);
}
