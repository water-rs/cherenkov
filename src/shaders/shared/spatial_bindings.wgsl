// Shared bindings for spatial filters — identical between the compute and
// fragment executions of the same stage. Only the *output* declaration
// differs: compute writes a storage texture (spatial_output_compute.wgsl),
// fragment writes its render target through `store_output`
// (spatial_output_fragment.wgsl).
//
// Token contract (substituted by the runtime, never valid WGSL as-is):
// - `PARAM_VEC4S`: number of `vec4<f32>` rows in the parameter array,
//   derived from `filtrate_core::MAX_FILTER_PARAM_VEC4S`.
//
// Alpha contract: every texture is premultiplied alpha. Linear operations
// (blurs, convolutions, resampling) are correct on premultiplied data as-is
// and must process all four channels together.

struct Uniforms {
    output_dimensions: vec2<f32>,
    input_dimensions: vec2<f32>,
    // Dimensions of `original_texture` (only meaningful for shaders recorded
    // via `spatial_shader_with_original`).
    original_dimensions: vec2<f32>,
    _padding: vec2<f32>,
    params: array<vec4<f32>, PARAM_VEC4S>,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
// Bound only for `spatial_shader_with_original` passes: the texture that fed
// this filter's first stage. Shaders that do not reference it compile without
// the binding being present in the layout.
@group(0) @binding(3) var original_texture: texture_2d<f32>;
