// Fragment entry points — appended after the stage body. The vertex stage is
// self-contained (no inputs), which keeps every spatial-fragment pipeline on
// the same vertex layout regardless of the body's bindings. `fs_main`
// reconstructs the compute `gid` from the fragment position, runs the shared
// body, and emits the texel the body stored via `store_output`.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Full-screen quad using 6 vertices (2 triangles) — same layout as the
    // color and blit passes.
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let gid = vec3<u32>(u32(position.x), u32(position.y), 0u);
    main(gid);
    return stored_output_color;
}
