// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    if uv.x < 0.5 {
        return cherenkov_srgb(vec4<f32>(1.0, 0.5, 0.25, 0.5));
    }
    return cherenkov_premultiplied_srgb(vec4<f32>(0.5, 0.25, 0.125, 0.5));
}
