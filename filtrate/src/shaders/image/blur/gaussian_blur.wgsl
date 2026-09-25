// One pass of a separable gaussian blur along `axis` ((1, 0) or (0, 1)).
// The kernel radius is ceil(3 * sigma). Weights use the incremental gaussian
// recurrence (GPU Gems 3, ch. 40), so the loop costs two multiplies per tap
// instead of an `exp`.

struct Params {
    sigma: f32,
    axis: vec2<f32>,
}
fn apply(input: texture_2d<f32>, input_point_sampler: sampler, uv: vec2<f32>, size: vec2<f32>, params: Params) -> vec4<f32> {
    let pixel = floor(uv * size);
    let sigma = max(params.sigma, 0.001);
    let radius = max(i32(ceil(sigma * 3.0)), 0);
    if radius == 0 {
        return load(input, input_point_sampler, size, pixel);
    }

    // w(o) = exp(-o^2 / (2 sigma^2)); w(o + 1) = w(o) * ratio(o), and
    // ratio(o) = exp(-(2o + 1) / (2 sigma^2)) advances by a constant factor.
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);
    let ratio_step = exp(-2.0 * inv_two_sigma_sq);
    var side_weight = 1.0;
    var side_ratio = exp(-inv_two_sigma_sq);

    var sum = load(input, input_point_sampler, size, pixel);
    var weight_total = 1.0;
    for (var offset = 1; offset <= radius; offset++) {
        side_weight *= side_ratio;
        side_ratio *= ratio_step;
        let delta = params.axis * f32(offset);
        sum += (load(input, input_point_sampler, size, pixel - delta)
            + load(input, input_point_sampler, size, pixel + delta))
            * side_weight;
        weight_total += 2.0 * side_weight;
    }
    return sum / weight_total;
}
