// 3x3 morphological gradient: the per-channel `max - min` of the
// neighbourhood's colour. Both accumulators are seeded from the centre
// texel, so extended values produce correct gradients. Alpha keeps the
// centre's coverage.

fn load(input: texture_2d<f32>, input_point_sampler: sampler, size: vec2<f32>, pixel: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(input, input_point_sampler, (pixel + 0.5) / size, 0.0);
}

fn apply(input: texture_2d<f32>, input_point_sampler: sampler, uv: vec2<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(input));
    let pixel = floor(uv * size);

    let centre = load(input, input_point_sampler, size, pixel);
    var lo = centre.rgb;
    var hi = centre.rgb;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let texel = load(input, input_point_sampler, size, pixel + vec2<f32>(f32(dx), f32(dy))).rgb;
            lo = min(lo, texel);
            hi = max(hi, texel);
        }
    }
    return vec4<f32>(hi - lo, centre.a);
}
