struct Params {
    amount: f32,
}

// `textureDimensions` observes `input`'s extent, which the materialized
// prefix may not share: the stage is not foldable.
fn apply(input: texture_2d<f32>, input_point_sampler: sampler, uv: vec2<f32>, params: Params) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(input));
    let texel = textureSampleLevel(input, input_point_sampler, uv, 0.0);
    return texel * params.amount / (size.x * size.y);
}
