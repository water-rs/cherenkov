// Fragment execution: the render target cannot be addressed as a storage
// texture, so the compiler strips every `textureStore` call on the output
// binding and rewrites it as `store_output(coord, value)`. Every spatial
// body writes exactly one texel at its own invocation coordinate, so the
// translation is exact — the fragment's own coordinate IS the store
// coordinate.
//
// The sentinel keeps that contract honest: a body that somehow retains a
// literal `textureStore` after rewriting fails type-checking (the builtin
// requires a storage texture, not an f32) rather than silently doing nothing.

var<private> output_texture: f32;

var<private> stored_output_color: vec4<f32>;

fn store_output(coord: vec2<i32>, value: vec4<f32>) {
    stored_output_color = value;
}
