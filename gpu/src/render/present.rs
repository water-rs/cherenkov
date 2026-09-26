// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Window presentation: blits a surface's f16 target onto its swapchain.

use std::collections::HashMap;

use cherenkov::{RenderError, SurfaceError};

/// A window's swapchain and its configuration.
pub struct WindowSurface {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

impl WindowSurface {
    /// Creates and configures the swapchain for `handle` at `size`.
    ///
    /// # Errors
    /// [`SurfaceError::UnsupportedTarget`] when wgpu cannot create or the adapter
    /// cannot present to the window.
    pub fn new(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        handle: Box<dyn wgpu::WindowHandle>,
        size: (u32, u32),
        transparent: bool,
    ) -> Result<Self, SurfaceError> {
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Window(handle))
            .map_err(|e| SurfaceError::UnsupportedTarget(format!("window surface: {e}")))?;
        let caps = surface.get_capabilities(adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .or_else(|| caps.formats.first().copied())
            .ok_or_else(|| {
                SurfaceError::UnsupportedTarget("adapter cannot present to the window".into())
            })?;
        let alpha_mode = if transparent {
            [
                wgpu::CompositeAlphaMode::PreMultiplied,
                wgpu::CompositeAlphaMode::PostMultiplied,
                wgpu::CompositeAlphaMode::Inherit,
            ]
            .into_iter()
            .find(|mode| caps.alpha_modes.contains(mode))
            .ok_or_else(|| {
                SurfaceError::UnsupportedTarget(format!(
                    "a transparent window needs a transparency-capable composite alpha mode, the adapter offers {:?}",
                    caps.alpha_modes
                ))
            })?
        } else if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto)
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.0.max(1),
            height: size.1.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: Vec::new(),
        };
        surface.configure(device, &config);
        Ok(Self { surface, config })
    }

    /// Reconfigures the swapchain to `size`.
    pub fn resize(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        self.config.width = size.0.max(1);
        self.config.height = size.1.max(1);
        self.surface.configure(device, &self.config);
    }

    /// Acquires the next swapchain image, reconfiguring once when the
    /// swapchain is outdated or lost. `None` skips the frame: the window is
    /// occluded or the acquire timed out.
    fn acquire(&self, device: &wgpu::Device) -> Result<Option<wgpu::SurfaceTexture>, RenderError> {
        use wgpu::CurrentSurfaceTexture as Current;
        let acquired = match self.surface.get_current_texture() {
            Current::Outdated | Current::Lost => {
                self.surface.configure(device, &self.config);
                self.surface.get_current_texture()
            }
            other => other,
        };
        match acquired {
            Current::Success(texture) | Current::Suboptimal(texture) => Ok(Some(texture)),
            Current::Timeout | Current::Occluded => Ok(None),
            Current::Outdated | Current::Lost => Err(RenderError::Render(
                "swapchain acquire failed after reconfiguration".into(),
            )),
            Current::Validation => Err(RenderError::Render("swapchain acquire: validation".into())),
        }
    }
}

/// The present pipelines, one per swapchain format seen.
/// Presents retained engine textures into host-owned textures.
pub struct Presenter {
    module: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
    pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    /// Indexed by `encode * 3 + alpha`: see `Present` in `present.wgsl`.
    uniforms: [wgpu::Buffer; 9],
}

impl Presenter {
    /// Creates the shared present state.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("present"),
            source: wgpu::ShaderSource::Wgsl(include_str!("present.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("present"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("present"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("present"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..wgpu::SamplerDescriptor::default()
        });
        let uniform = |encode: u32, alpha: u32| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("present uniform"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            buffer.slice(..).get_mapped_range_mut().copy_from_slice(
                &[encode.to_ne_bytes(), alpha.to_ne_bytes(), [0; 4], [0; 4]].concat(),
            );
            buffer.unmap();
            buffer
        };
        Self {
            module,
            layout,
            pipeline_layout,
            sampler,
            pipelines: HashMap::new(),
            uniforms: [
                uniform(0, 0),
                uniform(0, 1),
                uniform(0, 2),
                uniform(1, 0),
                uniform(1, 1),
                uniform(1, 2),
                uniform(2, 0),
                uniform(2, 1),
                uniform(2, 2),
            ],
        }
    }

    fn pipeline(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> &wgpu::RenderPipeline {
        self.pipelines.entry(format).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("present"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.module,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        })
    }

    /// Blits `source` onto the window's next swapchain image and presents it.
    ///
    /// # Errors
    /// [`RenderError`] when the swapchain image cannot be acquired.
    pub fn present(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        window: &WindowSurface,
        source: &wgpu::TextureView,
    ) -> Result<(), RenderError> {
        let Some(frame) = window.acquire(device)? else {
            return Ok(());
        };
        let alpha = match window.config.alpha_mode {
            wgpu::CompositeAlphaMode::PreMultiplied | wgpu::CompositeAlphaMode::Inherit => {
                OutputAlpha::Premultiplied
            }
            wgpu::CompositeAlphaMode::PostMultiplied => OutputAlpha::Straight,
            wgpu::CompositeAlphaMode::Auto | wgpu::CompositeAlphaMode::Opaque => {
                OutputAlpha::Opaque
            }
        };
        self.texture(
            device,
            queue,
            source,
            TextureOutput {
                texture: &frame.texture,
                color: OutputColor::Srgb,
                alpha,
            },
        );
        frame.present();
        Ok(())
    }

    /// Composites an engine texture into a native texture on the same device.
    /// `source` contains premultiplied linear Display P3. `color` describes
    /// the destination's color space; sRGB texture formats encode in hardware.
    pub fn texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::TextureView,
        output: TextureOutput<'_>,
    ) {
        let TextureOutput {
            texture: target,
            color,
            alpha,
        } = output;
        let format = target.format();
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let encode = match color {
            OutputColor::Srgb => usize::from(!format.is_srgb()),
            OutputColor::LinearDisplayP3 => 2,
        };
        let alpha = match alpha {
            OutputAlpha::Opaque => 0,
            OutputAlpha::Premultiplied => 1,
            OutputAlpha::Straight => 2,
        };
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("present"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniforms[encode * 3 + alpha].as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("present"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(self.pipeline(device, format));
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
    }
}

/// Color space of a host-owned presentation texture.
#[derive(Clone, Copy, Debug)]
pub enum OutputColor {
    /// sRGB primaries and transfer; sRGB texture formats encode in hardware.
    Srgb,
    /// Extended linear Display P3, preserving HDR values in float targets.
    LinearDisplayP3,
}

/// Alpha convention of a host-owned presentation texture.
#[derive(Clone, Copy, Debug)]
pub enum OutputAlpha {
    /// The destination is opaque.
    Opaque,
    /// Channels are multiplied by alpha.
    Premultiplied,
    /// Channels are independent of alpha.
    Straight,
}

/// A host-owned texture and its presentation conventions.
#[derive(Clone, Copy, Debug)]
pub struct TextureOutput<'a> {
    /// Attachment on the same device as the source.
    pub texture: &'a wgpu::Texture,
    /// Destination color encoding.
    pub color: OutputColor,
    /// Destination alpha convention.
    pub alpha: OutputAlpha,
}
