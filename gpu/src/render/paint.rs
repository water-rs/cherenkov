//! Shader registration and retained shader-paint textures.

use std::borrow::Cow;
use std::collections::HashMap;

use askama::Template;
use cherenkov::{RenderError, ResourceError, ShaderSource};

#[derive(Template)]
#[template(path = "paint.wgsl", escape = "none")]
struct Source<'a> {
    fragment: &'a str,
}

/// One shader use; uniforms are hashed by their exact IEEE representation.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Key {
    pub shader: u64,
    pub uniforms: Vec<u32>,
    pub size: (u32, u32),
}

struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    animated: bool,
}

pub struct Texture {
    pub image: super::GpuImage,
    globals: wgpu::Buffer,
    parameters: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    rendered: bool,
}

#[derive(Default)]
pub struct Registry(HashMap<u64, Pipeline>);

impl Registry {
    pub fn add(
        &mut self,
        device: &wgpu::Device,
        id: u64,
        source: &ShaderSource,
    ) -> Result<(), ResourceError> {
        let source_text = Source {
            fragment: &source.source,
        }
        .render()
        .map_err(|e| ResourceError::Shader(e.to_string()))?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader paint"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source_text)),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shader paint"),
            entries: &[uniform_binding(0, 16), uniform_binding(1, 256)],
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
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: super::TARGET_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(ResourceError::Shader(error.to_string()));
        }
        self.0.insert(
            id,
            Pipeline {
                pipeline,
                layout,
                animated: source.animated,
            },
        );
        Ok(())
    }

    pub fn remove(&mut self, id: u64) {
        self.0.remove(&id);
    }

    pub fn animated(&self, key: &Key) -> bool {
        self.0
            .get(&key.shader)
            .is_some_and(|pipeline| pipeline.animated)
    }

    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: &Key,
        textures: &mut HashMap<Key, Texture>,
        time: f32,
    ) -> Result<(), RenderError> {
        let pipeline = self
            .0
            .get(&key.shader)
            .ok_or_else(|| RenderError::Render(format!("unregistered shader {}", key.shader)))?;
        if key.uniforms.len() > 64 {
            return Err(RenderError::Render(
                "shader paint accepts at most 64 uniform floats".into(),
            ));
        }
        let texture = textures.entry(key.clone()).or_insert_with(|| {
            let (texture, view) = super::create_target(
                device,
                "shader paint",
                key.size,
                super::TARGET_USAGES,
                super::TARGET_FORMAT,
            );
            let globals = uniform_buffer(device, 16);
            let parameters = uniform_buffer(device, 256);
            let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shader paint"),
                layout: &pipeline.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: globals.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: parameters.as_entire_binding(),
                    },
                ],
            });
            Texture {
                image: super::GpuImage {
                    texture,
                    view,
                    width: key.size.0,
                    height: key.size.1,
                },
                globals,
                parameters,
                bindings,
                rendered: false,
            }
        });
        if texture.rendered && !pipeline.animated {
            return Ok(());
        }
        #[expect(clippy::cast_precision_loss, reason = "GPU texture dimensions fit f32")]
        let globals = [time, 0.0, key.size.0 as f32, key.size.1 as f32];
        queue.write_buffer(&texture.globals, 0, bytemuck::cast_slice(&globals));
        let mut parameters = [0u32; 64];
        parameters[..key.uniforms.len()].copy_from_slice(&key.uniforms);
        queue.write_buffer(&texture.parameters, 0, bytemuck::cast_slice(&parameters));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shader paint"),
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture.image.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &texture.bindings, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        queue.submit([encoder.finish()]);
        texture.rendered = true;
        Ok(())
    }
}

fn uniform_binding(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

fn uniform_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shader paint uniforms"),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
