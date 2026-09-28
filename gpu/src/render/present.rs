// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Window presentation: blits a surface's f16 target onto its swapchain.

use std::collections::HashMap;

use cherenkov::{RenderError, SurfaceError};

use super::{bindings, layout_entries, shaders::ShaderDelivery};

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
            *caps
                .alpha_modes
                .first()
                .expect("a configurable surface reports at least one alpha mode")
        };
        let config = wgpu::SurfaceConfiguration {
            color_space: wgpu::SurfaceColorSpace::Auto,
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
    /// `{ encode, alpha, headroom, pad }`: see `Present` in
    /// `present.wgsl`. Written per call — the headroom follows the
    /// display every frame (#97).
    uniform: wgpu::Buffer,
}

impl Presenter {
    /// Creates the shared present state.
    #[must_use]
    pub fn new(device: &wgpu::Device, delivery: ShaderDelivery) -> Self {
        let module = delivery.present_module(device);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("present"),
            entries: &layout_entries(bindings::PRESENT_GROUP0),
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
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("present uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            module,
            layout,
            pipeline_layout,
            sampler,
            pipelines: HashMap::new(),
            uniform,
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

    /// Blits `source` onto the window's next swapchain image and presents
    /// it. `headroom` is the display's declared HDR headroom for this
    /// frame.
    ///
    /// # Errors
    /// [`RenderError`] when the swapchain image cannot be acquired.
    pub fn present(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        window: &WindowSurface,
        source: &wgpu::TextureView,
        headroom: f32,
    ) -> Result<bool, RenderError> {
        let Some(frame) = window.acquire(device)? else {
            return Ok(false);
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
                headroom,
            },
        );
        queue.present(frame);
        Ok(true)
    }

    /// Composites an engine texture into a native texture on the same device.
    /// `source` contains premultiplied linear Display P3. `color` describes
    /// the destination's color space; sRGB texture formats encode in hardware.
    /// The destination must be a single-sampled, renderable 2D texture on this
    /// device. The blit replaces its contents, scaling the source to fit.
    ///
    /// # Panics
    /// If linear Display P3 output is requested for an sRGB texture format,
    /// or the textures violate wgpu's attachment and sampling requirements.
    pub fn texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::TextureView,
        output: TextureOutput<'_>,
    ) {
        self.texture_timed(device, queue, source, output, None);
    }

    /// [`Self::texture`], writing GPU timestamps around the render pass —
    /// the presentation cost probe the cross-engine bench's `present-cost`
    /// mode uses (#96). The device must have `Features::TIMESTAMP_QUERY`.
    ///
    /// # Panics
    /// As [`Self::texture`].
    pub fn texture_timed(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::TextureView,
        output: TextureOutput<'_>,
        timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let TextureOutput {
            texture: target,
            color,
            alpha,
            headroom,
        } = output;
        let format = target.format();
        assert!(
            !matches!(color, OutputColor::LinearDisplayP3) || !format.is_srgb(),
            "linear Display P3 output requires a non-sRGB texture format"
        );
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let encode: u32 = match color {
            OutputColor::Srgb => u32::from(!format.is_srgb()),
            OutputColor::LinearDisplayP3 => 2,
        };
        let alpha: u32 = match alpha {
            OutputAlpha::Opaque => 0,
            OutputAlpha::Premultiplied => 1,
            OutputAlpha::Straight => 2,
        };
        let headroom = headroom.max(0.0);
        queue.write_buffer(
            &self.uniform,
            0,
            &[
                encode.to_ne_bytes(),
                alpha.to_ne_bytes(),
                headroom.to_bits().to_ne_bytes(),
                [0; 4],
            ]
            .concat(),
        );
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
                    resource: self.uniform.as_entire_binding(),
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
                timestamp_writes: timestamps,
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
    /// Channels are multiplied by alpha in the destination encoding. For
    /// sRGB this multiplication follows transfer encoding, including when the
    /// texture format applies the transfer in hardware.
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
    /// The display's HDR headroom the destination reaches: SDR content is
    /// `1.0`. Values above `1` roll off smoothly towards it (#97); a headroom
    /// change applies to the next call, not to the content.
    pub headroom: f32,
}
