// Blends the input with an auxiliary image (`aux0`, sampled at the same uv)
// through one of sixteen blend operators, then mixes by `amount`. The
// operators work on the texel values as sampled.
//
// `mode` selects the operator: 0 normal, 1 multiply, 2 screen, 3 overlay,
// 4 darken, 5 lighten, 6 soft light, 7 hard light, 8 difference,
// 9 exclusion, 10 colour dodge, 11 colour burn, 12 hue, 13 saturation,
// 14 colour, 15 luminosity.

struct Params {
    amount: f32,
    mode: f32,
}

fn texel_at(image: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(image));
    let coord = clamp(vec2<i32>(floor(uv * vec2<f32>(size))), vec2<i32>(0), size - vec2<i32>(1));
    return textureLoad(image, coord, 0);
}

fn blend_overlay(base: vec3<f32>, top: vec3<f32>) -> vec3<f32> {
    let low = 2.0 * base * top;
    let high = 1.0 - 2.0 * (1.0 - base) * (1.0 - top);
    let mask = step(vec3<f32>(0.5), base);
    return mix(low, high, mask);
}

// The HSL composite modes (hue, saturation, colour, luminosity) follow the
// algorithm sketched in the SVG/PDF compositing spec, in HSL rather than HSY.

fn rgb_to_hsl(rgb: vec3<f32>) -> vec3<f32> {
    let max_c = max(max(rgb.r, rgb.g), rgb.b);
    let min_c = min(min(rgb.r, rgb.g), rgb.b);
    let l = (max_c + min_c) * 0.5;
    if max_c == min_c {
        return vec3<f32>(0.0, 0.0, l);
    }
    let d = max_c - min_c;
    let s = select(d / (2.0 - max_c - min_c), d / (max_c + min_c), l > 0.5);
    var h: f32;
    if max_c == rgb.r {
        h = (rgb.g - rgb.b) / d + select(0.0, 6.0, rgb.g < rgb.b);
    } else if max_c == rgb.g {
        h = (rgb.b - rgb.r) / d + 2.0;
    } else {
        h = (rgb.r - rgb.g) / d + 4.0;
    }
    return vec3<f32>(h / 6.0, s, l);
}

fn hue_to_rgb(p: f32, q: f32, t_in: f32) -> f32 {
    var t = t_in;
    if t < 0.0 { t = t + 1.0; }
    if t > 1.0 { t = t - 1.0; }
    if t < 1.0 / 6.0 { return p + (q - p) * 6.0 * t; }
    if t < 0.5 { return q; }
    if t < 2.0 / 3.0 { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    return p;
}

fn hsl_to_rgb(hsl: vec3<f32>) -> vec3<f32> {
    if hsl.y == 0.0 {
        return vec3<f32>(hsl.z);
    }
    let q = select(hsl.z + hsl.y - hsl.z * hsl.y, hsl.z * (1.0 + hsl.y), hsl.z < 0.5);
    let p = 2.0 * hsl.z - q;
    return vec3<f32>(
        hue_to_rgb(p, q, hsl.x + 1.0 / 3.0),
        hue_to_rgb(p, q, hsl.x),
        hue_to_rgb(p, q, hsl.x - 1.0 / 3.0),
    );
}

fn blend_hsl(base: vec3<f32>, top: vec3<f32>, take_h_top: bool, take_s_top: bool, take_l_top: bool) -> vec3<f32> {
    let base_hsl = rgb_to_hsl(base);
    let top_hsl = rgb_to_hsl(top);
    let h = select(base_hsl.x, top_hsl.x, take_h_top);
    let s = select(base_hsl.y, top_hsl.y, take_s_top);
    let l = select(base_hsl.z, top_hsl.z, take_l_top);
    return hsl_to_rgb(vec3<f32>(h, s, l));
}

fn blend_soft_light(base: vec3<f32>, top: vec3<f32>) -> vec3<f32> {
    let low = base - (1.0 - 2.0 * top) * base * (1.0 - base);
    let high = base + (2.0 * top - 1.0) * (sqrt(max(base, vec3<f32>(0.0))) - base);
    let mask = step(vec3<f32>(0.5), top);
    return mix(low, high, mask);
}

fn blend_color(base: vec3<f32>, top: vec3<f32>, mode: u32) -> vec3<f32> {
    switch mode {
        case 1u: {
            return base * top;
        }
        case 2u: {
            return 1.0 - (1.0 - base) * (1.0 - top);
        }
        case 3u: {
            return blend_overlay(base, top);
        }
        case 4u: {
            return min(base, top);
        }
        case 5u: {
            return max(base, top);
        }
        case 6u: {
            return blend_soft_light(base, top);
        }
        case 7u: {
            return blend_overlay(top, base);
        }
        case 8u: {
            return abs(base - top);
        }
        case 9u: {
            return base + top - 2.0 * base * top;
        }
        case 10u: {
            return base / max(vec3<f32>(1.0) - top, vec3<f32>(0.0001));
        }
        case 11u: {
            return 1.0 - (1.0 - base) / max(top, vec3<f32>(0.0001));
        }
        case 12u: {
            return blend_hsl(base, top, true, false, false);
        }
        case 13u: {
            return blend_hsl(base, top, false, true, false);
        }
        case 14u: {
            return blend_hsl(base, top, true, true, false);
        }
        case 15u: {
            return blend_hsl(base, top, false, false, true);
        }
        default: {
            return top;
        }
    }
}

fn apply(input: texture_2d<f32>, input_point_sampler: sampler, uv: vec2<f32>, params: Params, aux0: texture_2d<f32>) -> vec4<f32> {
    let base = textureSampleLevel(input, input_point_sampler, uv, 0.0);
    let top = texel_at(aux0, uv);
    let blended = blend_color(base.rgb, top.rgb, u32(params.mode + 0.5));
    return vec4<f32>(mix(base.rgb, blended, clamp(params.amount, 0.0, 1.0)), base.a);
}
