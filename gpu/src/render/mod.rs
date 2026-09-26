// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The render thread: sole owner of GPU state.

mod glyph;
mod instance;
mod lower;
mod raster;

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender};

use cherenkov::ContentChange;

use crate::config::{GpuConfig, GpuInfo, MemoryUsage, Pressure};
use crate::error::{EngineError, RenderError, SurfaceError};
use crate::message::{ChangeSet, LayerId, LayerOp, Message, SurfaceId};
use crate::surface::{FrameStats, Next, Readback};
use glyph::{Atlas, FontData};
use lower::{ContentData, Frame, GlyphContext, LayerNode, Lowering, Target};

/// The surface target format: premultiplied linear Display P3.
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

const TARGET_USAGES: wgpu::TextureUsages = wgpu::TextureUsages::from_bits_retain(
    wgpu::TextureUsages::RENDER_ATTACHMENT.bits()
        | wgpu::TextureUsages::COPY_SRC.bits()
        | wgpu::TextureUsages::TEXTURE_BINDING.bits(),
);

/// The render thread's reply to [`crate::Engine::new`].
pub struct Init {
    /// Adapter info.
    pub info: GpuInfo,
}

/// One surface's GPU-side state.
struct SurfaceState {
    size: (u32, u32),
    target: wgpu::Texture,
    view: wgpu::TextureView,
    /// Scratch textures, one per isolation depth, each with its cached
    /// group-1 bind group.
    scratch: Vec<(wgpu::Texture, wgpu::TextureView, wgpu::BindGroup)>,
    layers: HashMap<LayerId, LayerNode>,
    clear: cherenkov::WorkingColor,
    dirty: bool,
    frame: Frame,
}

impl SurfaceState {
    /// Bytes held by this surface's textures.
    fn gpu_bytes(&self) -> u64 {
        let texel = u64::from(self.size.0) * u64::from(self.size.1) * 8;
        texel * (1 + self.scratch.len() as u64)
    }
}

/// All render-thread state.
struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    layout0: wgpu::BindGroupLayout,
    layout1: wgpu::BindGroupLayout,
    globals: wgpu::Buffer,
    instances: wgpu::Buffer,
    stops: wgpu::Buffer,
    bind0: wgpu::BindGroup,
    /// The atlas generation `bind0` was built against.
    bound_atlas: u64,
    /// The buffer sizes `bind0` was built against.
    bound_instance_size: u64,
    /// The stop buffer size `bind0` was built against.
    bound_stop_size: u64,
    atlas: Atlas,
    dummy_bind1: wgpu::BindGroup,
    surfaces: HashMap<SurfaceId, SurfaceState>,
    fonts: HashMap<u64, FontData>,
    timestamps: bool,
    query_set: Option<wgpu::QuerySet>,
    query_buffer: Option<wgpu::Buffer>,
    query_staging: wgpu::Buffer,
    timestamps_inside: bool,
    max_texture: u32,
}

/// A new default layer node.
const fn node() -> LayerNode {
    LayerNode {
        transform: kurbo::Affine::IDENTITY,
        opacity: 1.0,
        clip: None,
        content: None,
        children: Vec::new(),
    }
}

/// Creates an adapter plus device. Fails when no adapter allows the target
/// format's required usages.
fn create_device(
    config: &GpuConfig,
) -> Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue), EngineError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: config.backends,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters = pollster::block_on(instance.enumerate_adapters(config.backends));
    let adapter = adapters
        .into_iter()
        .find(|a| {
            a.get_texture_format_features(TARGET_FORMAT)
                .allowed_usages
                .contains(TARGET_USAGES)
        })
        .ok_or(EngineError::NoAdapter)?;
    let supported = adapter.features();
    let mut required = wgpu::Features::empty();
    if config.timestamps && supported.contains(wgpu::Features::TIMESTAMP_QUERY) {
        required |= wgpu::Features::TIMESTAMP_QUERY;
    }
    if supported.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS) {
        required |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    }
    if config.pipeline_cache.is_some() && supported.contains(wgpu::Features::PIPELINE_CACHE) {
        required |= wgpu::Features::PIPELINE_CACHE;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cherenkov-gpu"),
        required_features: required,
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| EngineError::RequestDevice(format!("{e}")))?;
    Ok((adapter, device, queue))
}

const fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, fragment: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: if fragment {
            wgpu::ShaderStages::FRAGMENT
        } else {
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT
        },
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// The bind group layouts: group 0 is engine data, group 1 the texture a
/// composite samples.
fn create_layouts(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    let layout0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("engine data"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage_entry(1, false),
            storage_entry(2, true),
            texture_entry(3),
        ],
    });
    let layout1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("source texture"),
        entries: &[texture_entry(0)],
    });
    (layout0, layout1)
}

/// Builds the group-0 bind group over the current buffers and atlas.
fn make_bind0(
    device: &wgpu::Device,
    layout0: &wgpu::BindGroupLayout,
    globals: &wgpu::Buffer,
    instances: &wgpu::Buffer,
    stops: &wgpu::Buffer,
    atlas: &Atlas,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("engine data"),
        layout: layout0,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: instances.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: stops.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(atlas.view()),
            },
        ],
    })
}

/// The closed pipeline set: one instanced-quad pipeline from `shader.wgsl`.
/// When a pipeline cache path is configured and supported, the cache is
/// loaded beforehand and persisted afterwards, best effort.
fn create_pipeline(
    device: &wgpu::Device,
    config: &GpuConfig,
    layout0: &wgpu::BindGroupLayout,
    layout1: &wgpu::BindGroupLayout,
) -> Result<wgpu::RenderPipeline, EngineError> {
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cherenkov"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cherenkov"),
        bind_group_layouts: &[Some(layout0), Some(layout1)],
        immediate_size: 0,
    });
    let cache = config.pipeline_cache.as_ref().and_then(|path| {
        if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            return None;
        }
        let data = std::fs::read(path).ok();
        // SAFETY: `data` is either a blob previously produced by wgpu or
        // absent; `fallback: true` keeps us off the unsafe fallback path.
        Some(unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("cherenkov"),
                data: data.as_deref(),
                fallback: true,
            })
        })
    });
    let component = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    };
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cherenkov"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: TARGET_FORMAT,
                blend: Some(wgpu::BlendState {
                    color: component,
                    alpha: component,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: cache.as_ref(),
    });
    if let Some(error) = pollster::block_on(error_scope.pop()) {
        return Err(EngineError::Shader(format!("{error}")));
    }
    if let (Some(cache), Some(path)) = (&cache, &config.pipeline_cache)
        && let Some(data) = cache.get_data()
    {
        let _ = std::fs::write(path, data);
    }
    Ok(pipeline)
}

/// A `w` × `h` texture in the target format.
fn create_target(
    device: &wgpu::Device,
    label: &'static str,
    size: (u32, u32),
    usages: wgpu::TextureUsages,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: usages,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// The render-thread entry point: initializes, replies, then loops over
/// messages until [`Message::Shutdown`].
#[expect(
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    reason = "moved into the render thread"
)]
pub fn run(config: GpuConfig, rx: Receiver<Message>, init_tx: Sender<Result<Init, EngineError>>) {
    let init = create_device(&config).and_then(|(adapter, device, queue)| {
        let info = adapter.get_info();
        let (layout0, layout1) = create_layouts(&device);
        let pipeline = create_pipeline(&device, &config, &layout0, &layout1)?;
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: 272 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stops = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stops"),
            size: 32 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let atlas = Atlas::new(&device, config.budget.gpu.0);
        let bind0 = make_bind0(&device, &layout0, &globals, &instances, &stops, &atlas);
        let (_, dummy_view) = create_target(
            &device,
            "dummy source",
            (1, 1),
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let dummy_bind1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dummy source"),
            layout: &layout1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&dummy_view),
            }],
        });
        let timestamps = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let timestamps_inside = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let (query_set, query_buffer) = if timestamps {
            (
                Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("frame timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 2,
                })),
                Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("timestamp resolve"),
                    size: 16,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })),
            )
        } else {
            (None, None)
        };
        let query_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp staging"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let renderer = Renderer {
            max_texture: device.limits().max_texture_dimension_2d,
            device,
            queue,
            pipeline,
            layout0,
            layout1,
            globals,
            instances,
            stops,
            bind0,
            bound_atlas: 0,
            bound_instance_size: 272 * 16,
            bound_stop_size: 32 * 16,
            atlas,
            dummy_bind1,
            surfaces: HashMap::new(),
            fonts: HashMap::new(),
            timestamps,
            query_set,
            query_buffer,
            query_staging,
            timestamps_inside,
        };
        Ok((renderer, info))
    });
    let (mut renderer, info) = match init {
        Ok(pair) => pair,
        Err(e) => {
            let _ = init_tx.send(Err(e));
            return;
        }
    };
    let _ = init_tx.send(Ok(Init {
        info: GpuInfo {
            name: info.name,
            backend: format!("{:?}", info.backend),
            vendor: info.vendor,
            device: info.device,
            device_type: format!("{:?}", info.device_type),
            driver: info.driver,
            driver_info: info.driver_info,
        },
    }));
    while let Ok(message) = rx.recv() {
        match message {
            Message::CreateSurface { id, size, reply } => {
                let _ = reply.send(renderer.create_surface(id, size));
            }
            Message::DestroySurface { id } => {
                renderer.surfaces.remove(&id);
            }
            Message::AddFont { id, data, index } => {
                renderer.fonts.insert(id, FontData { data, index });
            }
            Message::Commit { surface, changes } => {
                renderer.commit(surface, changes);
            }
            Message::Render { time, reply } => {
                // The frame time exists for future scheduling; this slice
                // renders immediately.
                let _ = time;
                let _ = reply.send(renderer.render_frame());
            }
            Message::Readback { surface, reply } => {
                let _ = reply.send(renderer.readback(surface));
            }
            Message::Memory { reply } => {
                let _ = reply.send(renderer.memory());
            }
            Message::Trim(pressure) => {
                if pressure == Pressure::Critical {
                    renderer.atlas.clear();
                }
            }
            Message::Shutdown => break,
        }
    }
}

impl Renderer {
    /// Creates a surface's target texture and layer tree.
    fn create_surface(&mut self, id: SurfaceId, size: (u32, u32)) -> Result<(), SurfaceError> {
        if size.0 == 0 || size.1 == 0 {
            return Err(SurfaceError::ZeroSize);
        }
        if size.0 > self.max_texture || size.1 > self.max_texture {
            return Err(SurfaceError::TooLarge {
                width: size.0,
                height: size.1,
                max: self.max_texture,
            });
        }
        let (target, view) = create_target(&self.device, "surface target", size, TARGET_USAGES);
        let mut layers = HashMap::new();
        layers.insert(0, node());
        self.surfaces.insert(
            id,
            SurfaceState {
                size,
                target,
                view,
                scratch: Vec::new(),
                layers,
                clear: cherenkov::WorkingColor::TRANSPARENT,
                dirty: true,
                frame: Frame::default(),
            },
        );
        Ok(())
    }

    /// Applies one surface's change set.
    fn commit(&mut self, surface: SurfaceId, changes: ChangeSet) {
        let Some(state) = self.surfaces.get_mut(&surface) else {
            return;
        };
        if let Some(clear) = changes.clear {
            state.clear = clear;
            state.dirty = true;
        }
        for op in changes.ops {
            state.dirty = true;
            match op {
                LayerOp::Create(id) => {
                    state.layers.entry(id).or_insert_with(node);
                }
                LayerOp::Remove(id) => {
                    Self::remove_node(&mut state.layers, id);
                }
                LayerOp::Transform(id, t) => {
                    if let Some(node) = state.layers.get_mut(&id) {
                        node.transform = t;
                    }
                }
                LayerOp::Opacity(id, o) => {
                    if let Some(node) = state.layers.get_mut(&id) {
                        node.opacity = o;
                    }
                }
                LayerOp::Clip(id, clip) => {
                    if let Some(node) = state.layers.get_mut(&id) {
                        node.clip = clip;
                    }
                }
                LayerOp::Content(id, picture) => {
                    if let Some(node) = state.layers.get_mut(&id) {
                        node.content = picture.map(ContentData::Picture);
                    }
                }
                LayerOp::ContentChange(id, change) => {
                    if let Some(node) = state.layers.get_mut(&id) {
                        match change {
                            ContentChange::Replace(list) => {
                                node.content = Some(ContentData::List(list));
                            }
                            ContentChange::Update(updates) => {
                                if let Some(ContentData::List(list)) = &mut node.content {
                                    let _ = list.apply(updates);
                                }
                            }
                        }
                    }
                }
                LayerOp::Push { parent, child } => {
                    Self::detach(&mut state.layers, child);
                    if let Some(node) = state.layers.get_mut(&parent) {
                        node.children.push(child);
                    }
                }
                LayerOp::Insert {
                    parent,
                    index,
                    child,
                } => {
                    Self::detach(&mut state.layers, child);
                    if let Some(node) = state.layers.get_mut(&parent) {
                        node.children.insert(index.min(node.children.len()), child);
                    }
                }
                LayerOp::Detach { parent, child } => {
                    if let Some(node) = state.layers.get_mut(&parent) {
                        node.children.retain(|c| *c != child);
                    }
                }
            }
        }
    }

    /// Removes `child` from every child list holding it.
    fn detach(layers: &mut HashMap<LayerId, LayerNode>, child: LayerId) {
        for node in layers.values_mut() {
            node.children.retain(|c| *c != child);
        }
    }

    /// Removes a node and its descendants.
    fn remove_node(layers: &mut HashMap<LayerId, LayerNode>, id: LayerId) {
        Self::detach(layers, id);
        if let Some(node) = layers.remove(&id) {
            for child in node.children {
                Self::remove_node(layers, child);
            }
        }
    }

    /// Memory usage across buffers, textures and the atlas.
    fn memory(&self) -> MemoryUsage {
        let gpu = self.instances.size()
            + self.stops.size()
            + self.globals.size()
            + self.atlas.gpu_bytes()
            + self
                .surfaces
                .values()
                .map(SurfaceState::gpu_bytes)
                .sum::<u64>();
        MemoryUsage {
            gpu: crate::Bytes(gpu),
            cpu: crate::Bytes(self.atlas.cpu_bytes()),
        }
    }

    /// Lowers and submits every dirty surface, bracketed by drained
    /// timestamp queries when enabled.
    fn render_frame(&mut self) -> Result<(Next, FrameStats), RenderError> {
        let mut stats = FrameStats::default();
        let mut dirty: Vec<SurfaceId> = self
            .surfaces
            .iter()
            .filter(|(_, s)| s.dirty)
            .map(|(id, _)| *id)
            .collect();
        dirty.sort_unstable();
        if dirty.is_empty() {
            return Ok((Next::Idle, stats));
        }
        self.drain_and_stamp(0)?;
        let mut result = Ok(());
        for id in dirty {
            result = self.render_surface(id, &mut stats);
            if result.is_err() {
                break;
            }
        }
        if self.timestamps {
            self.drain_and_stamp(1)?;
            stats.gpu_seconds = self.resolve_timestamps()?;
        }
        self.wait()?;
        result?;
        Ok((Next::Idle, stats))
    }

    /// Lowers and submits one surface.
    #[expect(clippy::too_many_lines)]
    fn render_surface(&mut self, id: SurfaceId, stats: &mut FrameStats) -> Result<(), RenderError> {
        // Lowering needs `surf.layers` and `surf.frame` plus `atlas`,
        // `fonts`, `device` and `queue`; take the layer map out of the
        // surface so the borrows stay disjoint.
        let lowered = {
            let Some(surf) = self.surfaces.get_mut(&id) else {
                return Ok(());
            };
            surf.frame.reset();
            let layers = std::mem::take(&mut surf.layers);
            // A full atlas is a recoverable signal: grow while the budget
            // allows, then clear once; a second failure after the clear
            // means the frame's live set exceeds the maximum atlas.
            let mut cleared = false;
            let result = loop {
                let result = if let Some(root) = layers.get(&0) {
                    let mut glyphs = GlyphContext {
                        atlas: &mut self.atlas,
                        queue: &self.queue,
                        fonts: &self.fonts,
                    };
                    let mut lowering = Lowering::new(&mut surf.frame, surf.size);
                    let result = lowering.run(root, &layers, surf.clear, &mut glyphs);
                    stats.glyphs_rasterized += lowering.glyphs_rasterized();
                    result
                } else {
                    Ok(())
                };
                match result {
                    Err(RenderError::AtlasFull) if !cleared => {
                        surf.frame.reset();
                        if self.atlas.size() < self.atlas.cap() {
                            self.atlas.grow(&self.device);
                        } else {
                            self.atlas.clear();
                            cleared = true;
                        }
                    }
                    Err(RenderError::AtlasFull) => break Err(RenderError::AtlasExhausted),
                    other => break other,
                }
            };
            surf.layers = layers;
            result
        };
        lowered?;
        // Scratch textures for the frame's deepest isolation level.
        let Some(surf) = self.surfaces.get_mut(&id) else {
            return Ok(());
        };
        let max_scratch = surf
            .frame
            .passes
            .iter()
            .filter_map(|p| match p.target {
                Target::Scratch(i) => Some(i + 1),
                Target::Surface => None,
            })
            .max()
            .unwrap_or(0);
        while surf.scratch.len() < max_scratch {
            let (texture, view) =
                create_target(&self.device, "isolation scratch", surf.size, TARGET_USAGES);
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scratch source"),
                layout: &self.layout1,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                }],
            });
            surf.scratch.push((texture, view, bind));
        }
        // Write the globals, grow and write the instance/stop buffers. The
        // group-0 bind group is rebuilt only when a buffer or the atlas
        // texture was recreated.
        self.queue.write_buffer(
            &self.globals,
            0,
            bytemuck::bytes_of(&lower::globals(surf.size)),
        );
        let inst_bytes = bytemuck::cast_slice::<instance::Instance, u8>(&surf.frame.instances);
        if !inst_bytes.is_empty() && inst_bytes.len() as u64 > self.instances.size() {
            let size = (inst_bytes.len() as u64).next_power_of_two();
            self.instances = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instances"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !inst_bytes.is_empty() {
            self.queue.write_buffer(&self.instances, 0, inst_bytes);
        }
        let stop_bytes = bytemuck::cast_slice::<instance::Stop, u8>(&surf.frame.stops);
        if !stop_bytes.is_empty() && stop_bytes.len() as u64 > self.stops.size() {
            let size = (stop_bytes.len() as u64).next_power_of_two();
            self.stops = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stops"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !stop_bytes.is_empty() {
            self.queue.write_buffer(&self.stops, 0, stop_bytes);
        }
        if self.atlas.generation() != self.bound_atlas {
            self.bind0 = make_bind0(
                &self.device,
                &self.layout0,
                &self.globals,
                &self.instances,
                &self.stops,
                &self.atlas,
            );
            self.bound_atlas = self.atlas.generation();
        }
        // Buffers grown above leave `bind0` stale; rebuild when capacity
        // changed since the bind group was built.
        if self.instances.size() > self.bound_instance_size
            || self.stops.size() > self.bound_stop_size
        {
            self.bind0 = make_bind0(
                &self.device,
                &self.layout0,
                &self.globals,
                &self.instances,
                &self.stops,
                &self.atlas,
            );
            self.bound_atlas = self.atlas.generation();
            self.bound_instance_size = self.instances.size();
            self.bound_stop_size = self.stops.size();
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        for pass in &surf.frame.passes {
            let view = match pass.target {
                Target::Surface => &surf.view,
                Target::Scratch(i) => &surf.scratch[i].1,
            };
            let load = match pass.clear {
                Some([r, g, b, a]) => wgpu::LoadOp::Clear(wgpu::Color {
                    r: f64::from(r),
                    g: f64::from(g),
                    b: f64::from(b),
                    a: f64::from(a),
                }),
                None => wgpu::LoadOp::Load,
            };
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.bind0, &[]);
            for range in &pass.ranges {
                stats.draws += 1;
                let bind = match range.source {
                    Some(i) => &surf.scratch[i].2,
                    None => &self.dummy_bind1,
                };
                render_pass.set_bind_group(1, bind, &[]);
                render_pass.draw(0..6, range.instances.clone());
            }
        }
        self.queue.submit([encoder.finish()]);
        stats.passes += u32::try_from(surf.frame.passes.len()).unwrap_or(u32::MAX);
        stats.instances += u32::try_from(surf.frame.instances.len()).unwrap_or(u32::MAX);
        surf.dirty = false;
        Ok(())
    }

    /// Blocks until the queue is drained.
    fn wait(&self) -> Result<(), RenderError> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| RenderError::DeviceLost)?;
        Ok(())
    }

    /// Writes a timestamp query in its own drained submission, mirroring
    /// `bench/src/wgpu_ctx.rs`.
    fn drain_and_stamp(&self, index: u32) -> Result<(), RenderError> {
        let Some(qs) = &self.query_set else {
            return Ok(());
        };
        self.wait()?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("timestamp"),
            });
        if self.timestamps_inside {
            encoder.write_timestamp(qs, index);
        } else {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("timestamp"),
                timestamp_writes: None,
            });
            pass.write_timestamp(qs, index);
        }
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Resolves the timestamp pair into seconds.
    fn resolve_timestamps(&self) -> Result<Option<f64>, RenderError> {
        let (Some(qs), Some(buf)) = (&self.query_set, &self.query_buffer) else {
            return Ok(None);
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("timestamp resolve"),
            });
        encoder.resolve_query_set(qs, 0..2, buf, 0);
        encoder.copy_buffer_to_buffer(buf, 0, &self.query_staging, 0, 16);
        self.queue.submit([encoder.finish()]);
        let slice = self.query_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.wait()?;
        let data = slice.get_mapped_range();
        let ticks: &[u64] = bytemuck::cast_slice(&data);
        #[expect(clippy::cast_precision_loss)]
        let seconds = if ticks.len() >= 2 && ticks[1] > ticks[0] {
            Some(f64::from(self.queue.get_timestamp_period()) * (ticks[1] - ticks[0]) as f64 * 1e-9)
        } else {
            None
        };
        drop(data);
        self.query_staging.unmap();
        Ok(seconds)
    }

    /// Copies a surface's target into `Readback` pixels, decoding f16 → f32.
    fn readback(&self, surface: SurfaceId) -> Result<Readback, RenderError> {
        let Some(state) = self.surfaces.get(&surface) else {
            return Err(RenderError::Readback("unknown surface".into()));
        };
        let (w, h) = state.size;
        let bytes_per_row = (w * 8).div_ceil(256) * 256;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(bytes_per_row) * u64::from(h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &state.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| RenderError::Readback(format!("poll: {e}")))?;
        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for row in 0..h {
            let start = (row * bytes_per_row) as usize;
            for px in data[start..start + (w * 8) as usize].as_chunks::<8>().0 {
                let bits: [u16; 4] = bytemuck::cast(*px);
                pixels.push([
                    half::f16::from_bits(bits[0]).to_f32(),
                    half::f16::from_bits(bits[1]).to_f32(),
                    half::f16::from_bits(bits[2]).to_f32(),
                    half::f16::from_bits(bits[3]).to_f32(),
                ]);
            }
        }
        drop(data);
        buf.unmap();
        Ok(Readback {
            width: w,
            height: h,
            pixels,
        })
    }
}
