// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Presents a surface's premultiplied linear Display P3 target on a window:
// converts to linear sRGB and, when the swapchain format is not an sRGB
// format, applies the sRGB transfer function itself.

struct Present {
    // 1 when the shader encodes sRGB, 0 when the swapchain format does.
    encode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> present: Present;

struct Vertex {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> Vertex {
    // One triangle covering the clip-space square.
    let x = f32(i32(index & 1u) * 4 - 1);
    let y = f32(i32(index >> 1u) * 4 - 1);
    var out: Vertex;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

fn srgb_encode(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

@fragment
fn fs_main(in: Vertex) -> @location(0) vec4<f32> {
    let p3 = textureSample(source, source_sampler, in.uv);
    // Linear Display P3 → linear sRGB (Bradford-adapted, D65).
    let rgb = vec3<f32>(
        1.224_940_2 * p3.r - 0.224_940_2 * p3.g,
        -0.042_056_95 * p3.r + 1.042_056_9 * p3.g,
        -0.019_637_55 * p3.r - 0.078_636_05 * p3.g + 1.098_273_6 * p3.b,
    );
    let clamped = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    if present.encode == 1u {
        return vec4<f32>(
            srgb_encode(clamped.r),
            srgb_encode(clamped.g),
            srgb_encode(clamped.b),
            1.0,
        );
    }
    return vec4<f32>(clamped, 1.0);
}
