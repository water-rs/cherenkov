// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return vec4(params[0].rgb + vec3(uniforms.time, uv.x, uv.y) * params[1].xyz, params[0].a);
}
