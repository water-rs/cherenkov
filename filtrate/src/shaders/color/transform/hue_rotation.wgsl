// Hue rotation: rotates the straight-alpha colour's HSL hue by `angle`
// degrees. The HSL round trip is piecewise, so the filter is not linear.

struct Params {
    angle: f32,
}
fn apply(color: vec4<f32>, params: Params) -> vec4<f32> {
    var hsl = rgb_to_hsl(color.rgb / max(color.a, 1e-6));
    hsl.x = fract(hsl.x + params.angle / 360.0);
    return vec4<f32>(hsl_to_rgb(hsl) * color.a, color.a);
}
