// Copy-and-convert tail (#168): `external-cost --path c` draws the
// producer frame's copied planes through the engine's own YUV decode —
// this file is concatenated after `shared.wgsl` and `external.wgsl`, so
// `ext_frame_yuv`, `ExtParams` and the `ext_y`/`ext_uv`/`params` bindings
// are the same source the external-frame path composites with. The draw
// is the GpuContent's one fullscreen triangle; the output is the
// working-space premultiplied pixel the producer attachment expects.
//
// Mirrors water-rs/video-gpu's `render_surface` draw
// (src/runtime_player.rs): plane textures in, converted colour out.

struct ConvertVertex {
    @builtin(position) pos: vec4<f32>,
}

// Fullscreen triangle, no vertex buffer: positions (-1,1), (3,1),
// (-1,-3) cover the attachment.
@vertex
fn vs_convert(@builtin(vertex_index) i: u32) -> ConvertVertex {
    var out: ConvertVertex;
    let x = f32(i & 1u) * 4.0 - 1.0;
    let y = 1.0 - f32(i >> 1u) * 4.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_convert(in: ConvertVertex) -> @location(0) vec4<f32> {
    // `in.pos.xy` lands on pixel centres, the convention
    // `ext_frame_yuv` expects.
    return ext_frame_yuv(in.pos.xy);
}
