// One pass of a separable box blur: averages the `2 * radius + 1` texels
// along `axis` ((1, 0) or (0, 1)), clamped at the edges.

struct Params {
    radius: f32,
    axis: vec2<f32>,
}

fn load(input: texture_2d<f32>, input_point_sampler: sampler, size: vec2<f32>, pixel: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(input, input_point_sampler, (pixel + 0.5) / size, 0.0);
}

fn apply(input: texture_2d<f32>, input_point_sampler: sampler, uv: vec2<f32>, params: Params) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(input));
    let pixel = floor(uv * size);
    let radius = max(i32(round(params.radius)), 0);
    var sum = load(input, input_point_sampler, size, pixel);
    for (var offset = 1; offset <= radius; offset++) {
        let delta = params.axis * f32(offset);
        sum += load(input, input_point_sampler, size, pixel - delta)
            + load(input, input_point_sampler, size, pixel + delta);
    }
    return sum / f32(2 * radius + 1);
}
