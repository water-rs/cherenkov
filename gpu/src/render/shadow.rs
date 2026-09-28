//! Engine-owned silhouette morphology and Gaussian convolution on the GPU.
use cherenkov::RenderError;
use kurbo::Affine;
use wgpu::util::DeviceExt;

/// The local-to-device covariance and spread of a captured silhouette.
#[derive(Clone, Copy)]
pub struct Parameters {
    pub transform: Affine,
    pub sigma: f64,
    pub spread: f64,
}

struct Kernel {
    key: Vec<u64>,
    buffer: wgpu::Buffer,
}

/// Fixed pipeline, one reusable intermediate, and at most three last kernels.
/// Bind groups retain replaced buffers until their queued commands complete.
pub struct Blur {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    kernels: [Option<Kernel>; 3],
    temporary: Option<super::ScratchTarget>,
}

impl Blur {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("silhouette blur"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("silhouette convolution"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadow.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("silhouette blur"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("silhouette blur"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            layout,
            kernels: std::array::from_fn(|_| None),
            temporary: None,
        }
    }

    pub fn gpu_bytes(&self) -> u64 {
        self.kernels
            .iter()
            .flatten()
            .map(|kernel| kernel.buffer.size())
            .sum::<u64>()
            + self.temporary.as_ref().map_or(0, |target| {
                u64::from(target.width)
                    * u64::from(target.height)
                    * if target.texture.format() == wgpu::TextureFormat::Rgba16Float {
                        8
                    } else {
                        4
                    }
            })
    }

    pub fn apply(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &super::ScratchTarget,
        size: (u32, u32),
        parameters: Parameters,
    ) -> Result<(), RenderError> {
        if self
            .temporary
            .as_ref()
            .is_none_or(|t| (t.width, t.height) != size)
        {
            let (texture, view) = super::create_target(
                device,
                "silhouette intermediate",
                size,
                super::TARGET_USAGES,
                input.texture.format(),
            );
            self.temporary = Some(super::ScratchTarget {
                texture,
                view,
                width: size.0,
                height: size.1,
            });
        }
        let [a, b, c, d, _, _] = parameters.transform.as_coeffs();
        for (slot, mode, axis) in [
            (0, if parameters.spread >= 0. { 1 } else { 2 }, [0., 0.]),
            (1, 0, [a, b]),
            (2, 0, [c, d]),
        ] {
            if (slot == 0 && parameters.spread == 0.) || (slot != 0 && parameters.sigma <= 0.) {
                continue;
            }
            let key = vec![
                u64::from(size.0),
                u64::from(size.1),
                parameters.sigma.to_bits(),
                parameters.spread.to_bits(),
                a.to_bits(),
                b.to_bits(),
                c.to_bits(),
                d.to_bits(),
            ];
            if self.kernels[slot]
                .as_ref()
                .is_none_or(|kernel| kernel.key != key)
            {
                let taps = if slot == 0 {
                    spread_taps(parameters, device.limits().max_storage_buffer_binding_size)?
                } else {
                    gaussian_taps(
                        axis,
                        parameters.sigma,
                        device.limits().max_storage_buffer_binding_size,
                    )?
                };
                let count = u32::try_from(taps.len()).map_err(|_| {
                    RenderError::Render("shadow kernel exceeds addressable storage".into())
                })?;
                let mut bytes = bytemuck::cast_slice(&[size.0, size.1, count, mode]).to_vec();
                bytes.extend_from_slice(bytemuck::cast_slice(&taps));
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("silhouette kernel"),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                });
                self.kernels[slot] = Some(Kernel { key, buffer });
            }
            self.dispatch(device, encoder, input, size, slot);
        }
        Ok(())
    }
    fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &super::ScratchTarget,
        size: (u32, u32),
        slot: usize,
    ) {
        let kernel = self.kernels[slot].as_ref().expect("kernel prepared");
        let temporary = self.temporary.as_ref().expect("intermediate prepared");
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("silhouette blur"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kernel.buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("silhouette convolution"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &temporary.view,
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        encoder.copy_texture_to_texture(
            temporary.texture.as_image_copy(),
            input.texture.as_image_copy(),
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );
    }
}

#[expect(clippy::cast_precision_loss, reason = "device byte limits fit f64")]
fn bounded_count(count: f64, limit: u64) -> Result<usize, RenderError> {
    if !count.is_finite() || count < 0. || count.mul_add(16., 16.) > limit as f64 {
        return Err(RenderError::Render(
            "shadow kernel exceeds device storage limit".into(),
        ));
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "validated against device byte limit"
    )]
    Ok(count as usize)
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded kernel indices and f32 GPU coefficients"
)]
fn gaussian_taps(axis: [f64; 2], sigma: f64, limit: u64) -> Result<Vec<[f32; 4]>, RenderError> {
    let length = axis[0].hypot(axis[1]);
    let sigma = sigma * length;
    if sigma <= 1e-9 {
        return Ok(vec![[0., 0., 1., 0.]]);
    }
    let radius = (6. * sigma).ceil();
    let count = bounded_count(2.0f64.mul_add(radius, 1.), limit)?;
    let inv = 1. / (sigma * std::f64::consts::SQRT_2);
    let mut taps = Vec::with_capacity(count);
    let mut total = 0.;
    for i in 0..count {
        let offset = i as f64 - radius;
        let weight = 0.5 * (libm::erf((offset + 0.5) * inv) - libm::erf((offset - 0.5) * inv));
        total += weight;
        taps.push([
            (axis[0] / length * offset) as f32,
            (axis[1] / length * offset) as f32,
            weight as f32,
            0.,
        ]);
    }
    for tap in &mut taps {
        tap[2] /= total as f32;
    }
    Ok(taps)
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bounded morphology footprint and f32 GPU coordinates"
)]
fn spread_taps(parameters: Parameters, limit: u64) -> Result<Vec<[f32; 4]>, RenderError> {
    let [a, b, c, d, _, _] = parameters.transform.as_coeffs();
    let scale = a.hypot(b).max(c.hypot(d)).max(1.);
    let radius = parameters.spread.abs();
    let steps = (radius * scale * 2.).ceil().max(1.);
    let count = bounded_count(2.0f64.mul_add(steps, 1.).powi(2), limit)?;
    let mut taps = Vec::with_capacity(count);
    let side = 2.0f64.mul_add(steps, 1.) as usize;
    for y in 0..side {
        for x in 0..side {
            let px = (x as f64 - steps) / steps * radius;
            let py = (y as f64 - steps) / steps * radius;
            if px.hypot(py) <= radius {
                taps.push([
                    a.mul_add(px, c * py) as f32,
                    b.mul_add(px, d * py) as f32,
                    1.,
                    0.,
                ]);
            }
        }
    }
    Ok(taps)
}
