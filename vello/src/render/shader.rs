// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shader paints: one wgpu pipeline per registered shader, and a retained
//! texture + uniform buffer per use inside a display list.

use std::borrow::Cow;
use std::collections::HashMap;

use vello::peniko;

use cherenkov::ResourceError;

/// The maximum number of `f32` uniform values a [`Paint::Shader`] use may
/// carry: `params` is `array<vec4<f32>, 16>`.
///
/// [`Paint::Shader`]: cherenkov::Paint::Shader
pub const MAX_SHADER_PARAMS: usize = 64;

/// The WGSL prelude prepended to every `ShaderSource`
/// fragment: `uniforms` (time and resolution), `params` (zero-padded use
/// uniforms) and a fullscreen-triangle vertex shader whose `uv` spans
/// `[0,1]` with y down.
const PRELUDE: &str = r"
struct Uniforms {
    time: f32,
    resolution: vec2<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> params: array<vec4<f32>, 16>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Full-screen triangle: uv in [0,1], y down.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    let pos = positions[vertex_index];
    var output: VertexOutput;
    output.position = vec4<f32>(pos, 0.0, 1.0);
    output.uv = vec2<f32>((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
    return output;
}
";

/// Bytes of the `uniforms` block (`time`, `resolution`, padding).
const UNIFORMS_LEN: usize = 16;
/// Bytes of the `params` block (16 `vec4<f32>`).
const PARAMS_LEN: usize = 4 * MAX_SHADER_PARAMS;
/// `UNIFORMS_LEN` as `u64`.
const UNIFORMS_SIZE: u64 = UNIFORMS_LEN as u64;
/// `PARAMS_LEN` as `u64`.
const PARAMS_SIZE: u64 = PARAMS_LEN as u64;

/// A registered shader's pipeline and layout.
pub struct ShaderEntry {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    /// Whether the shader re-renders every frame.
    pub animated: bool,
}

/// The shader registry, keyed by `ShaderId::raw`.
#[derive(Default)]
pub struct ShaderRegistry {
    entries: HashMap<u64, ShaderEntry>,
}

/// One shader paint use's retained GPU state (per layer, per command).
///
/// The `image` identity is created at lowering time; the texture, buffer
/// and bind group are created lazily the first frame the use is drawn.
pub struct ShaderUse {
    /// The shader id.
    pub shader: u64,
    /// The use's uniforms (front-end `ShaderPaint::uniforms`, ≤ 64 floats).
    pub uniforms: Vec<f32>,
    /// The texture size: the shape's device-space bounding box, ceiled.
    pub size: (u32, u32),
    /// The vello-side image identity (`override_image` key).
    pub image: peniko::ImageData,
    /// Retained texture, view, uniform buffers and bind group.
    pub texture: Option<wgpu::Texture>,
    /// Its view.
    pub view: Option<wgpu::TextureView>,
    /// Its `uniforms` buffer (binding 0).
    pub uniforms_buffer: Option<wgpu::Buffer>,
    /// Its `params` buffer (binding 1); a separate buffer keeps both
    /// uniform binding offsets at 0 and within the required alignment.
    pub params_buffer: Option<wgpu::Buffer>,
    /// Its bind group.
    pub bind_group: Option<wgpu::BindGroup>,
    /// Whether the texture has ever been rendered.
    pub rendered: bool,
}

/// A fresh `peniko::ImageData` identity for a `w`×`h` GPU texture, keyed by
/// a unique blob id; the pixels themselves are supplied through
/// `override_image`, so the blob is empty.
pub fn texture_image(width: u32, height: u32, alpha: peniko::ImageAlphaType) -> peniko::ImageData {
    let empty: std::sync::Arc<[u8]> = std::sync::Arc::from(&[][..]);
    peniko::ImageData {
        data: peniko::Blob::new(std::sync::Arc::new(super::SharedBytes(empty))),
        format: peniko::ImageFormat::Rgba8,
        alpha_type: alpha,
        width,
        height,
    }
}

impl ShaderRegistry {
    /// Whether `id` is registered.
    pub fn contains(&self, id: u64) -> bool {
        self.entries.contains_key(&id)
    }

    /// Whether `id` is an animated shader.
    pub fn animated(&self, id: u64) -> bool {
        self.entries.get(&id).is_some_and(|e| e.animated)
    }

    /// Builds the pipeline for `source` and registers it.
    ///
    /// Validation is wrapped in a wgpu error scope so a bad WGSL reports a
    /// [`ResourceError::Shader`] instead of panicking.
    pub fn add(
        &mut self,
        device: &wgpu::Device,
        id: u64,
        spec: &super::ShaderSpec,
    ) -> Result<(), ResourceError> {
        let source = format!("{PRELUDE}\n{}", spec.source);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader paint"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shader paint uniforms"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(UNIFORMS_SIZE),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(PARAMS_SIZE),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shader paint"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shader paint"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let error = pollster::block_on(scope.pop());
        if let Some(error) = error {
            return Err(ResourceError::Shader(error.to_string()));
        }
        self.entries.insert(
            id,
            ShaderEntry {
                pipeline,
                layout,
                animated: spec.animated,
            },
        );
        Ok(())
    }

    /// Drops the shader's pipeline.
    pub fn remove(&mut self, id: u64) {
        self.entries.remove(&id);
    }

    /// Ensures `use_`'s texture, buffer and bind group exist.
    fn materialize(&self, device: &wgpu::Device, use_: &mut ShaderUse) -> Result<(), RenderShader> {
        let Some(entry) = self.entries.get(&use_.shader) else {
            return Err(RenderShader::Unregistered(use_.shader));
        };
        let (w, h) = use_.size;
        if use_.texture.is_none() {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("shader paint texture"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            use_.view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
            use_.texture = Some(texture);
        }
        if use_.uniforms_buffer.is_none() {
            use_.uniforms_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shader paint uniforms"),
                size: UNIFORMS_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if use_.params_buffer.is_none() {
            use_.params_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shader paint params"),
                size: PARAMS_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if use_.bind_group.is_none() {
            let uniforms = use_.uniforms_buffer.as_ref().expect("just created");
            let params = use_.params_buffer.as_ref().expect("just created");
            use_.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shader paint uniforms"),
                layout: &entry.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniforms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                ],
            }));
        }
        Ok(())
    }

    /// Renders `use_`'s texture for this frame: writes its uniforms
    /// (`time` is seconds since engine start) and draws the fullscreen
    /// triangle.
    pub fn evaluate(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        use_: &mut ShaderUse,
        time: f32,
    ) -> Result<(), RenderShader> {
        self.materialize(device, use_)?;
        let entry = self
            .entries
            .get(&use_.shader)
            .expect("checked in materialize");
        let mut uniforms = [0u8; UNIFORMS_LEN];
        let (w, h) = use_.size;
        uniforms[0..4].copy_from_slice(&time.to_le_bytes());
        uniforms[4..8].copy_from_slice(&f64_to_f32(w).to_le_bytes());
        uniforms[8..12].copy_from_slice(&f64_to_f32(h).to_le_bytes());
        queue.write_buffer(
            use_.uniforms_buffer.as_ref().expect("materialized"),
            0,
            &uniforms,
        );
        let mut params = [0u8; PARAMS_LEN];
        for (i, v) in use_.uniforms.iter().take(MAX_SHADER_PARAMS).enumerate() {
            params[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        queue.write_buffer(
            use_.params_buffer.as_ref().expect("materialized"),
            0,
            &params,
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shader paint"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shader paint"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: use_.view.as_ref().expect("materialized"),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&entry.pipeline);
            pass.set_bind_group(0, use_.bind_group.as_ref().expect("materialized"), &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
        use_.rendered = true;
        Ok(())
    }
}

/// `u32` → `f32` for uniform texel dimensions (exact within 2^24).
#[expect(
    clippy::cast_precision_loss,
    reason = "texture sizes stay well below 2^24"
)]
const fn f64_to_f32(v: u32) -> f32 {
    v as f32
}

/// A shader-use failure: the paint's shader is not registered.
#[derive(Debug)]
pub enum RenderShader {
    /// The shader id is not registered.
    Unregistered(u64),
}
