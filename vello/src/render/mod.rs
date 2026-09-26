// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The render thread: sole owner of GPU state.

mod convert;
mod lower;

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::mpsc::{Receiver, Sender};

use cherenkov::ContentChange;
use vello::peniko;
use vello::{AaConfig, AaSupport, RendererOptions};

use crate::error::{EngineError, RenderError, SurfaceError, Unsupported};
use crate::message::{ChangeSet, LayerId, LayerOp, Message, SurfaceId, TargetSpec};
use crate::surface::{FrameStats, Next, Readback};
use crate::{Bytes, interop};
use crate::{GpuInfo, MemoryUsage, Pressure, VelloConfig};

/// The surface target format: premultiplied sRGB-encoded sRGB.
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

const TARGET_USAGES: wgpu::TextureUsages = wgpu::TextureUsages::from_bits_retain(
    wgpu::TextureUsages::RENDER_ATTACHMENT.bits()
        | wgpu::TextureUsages::STORAGE_BINDING.bits()
        | wgpu::TextureUsages::COPY_SRC.bits()
        | wgpu::TextureUsages::TEXTURE_BINDING.bits(),
);

/// How the engine gets its device: created by the render thread itself, or
/// an existing one shared by the embedder.
pub enum DeviceRequest {
    /// Create the instance, adapter and device.
    Create,
    /// Drive the embedder's device.
    Existing(interop::wgpu::DeviceSource),
}

/// The render thread's reply to `Engine::new`/`Engine::with_device`.
pub struct Init {
    /// Adapter info.
    pub info: GpuInfo,
}

/// What a layer draws, on the render thread.
enum ContentData {
    /// A live display list, patched by `ContentChange::Update`s.
    List(cherenkov::DisplayList),
    /// A shared immutable picture.
    Picture(cherenkov::Picture),
}

/// A retained layer node.
struct LayerNode {
    transform: cherenkov::kurbo::Affine,
    opacity: f32,
    clip: Option<cherenkov::ShapeData>,
    blend: cherenkov::BlendMode,
    filter: Option<cherenkov::FilterId>,
    content: Option<ContentData>,
    /// The content lowered into a reusable scene fragment; rebuilt whenever
    /// `content` changes or the surface is resized.
    fragment: Option<vello::Scene>,
    children: Vec<LayerId>,
}

/// A new default layer node.
const fn node() -> LayerNode {
    LayerNode {
        transform: cherenkov::kurbo::Affine::IDENTITY,
        opacity: 1.0,
        clip: None,
        blend: cherenkov::BlendMode::Normal,
        filter: None,
        content: None,
        fragment: None,
        children: Vec::new(),
    }
}

/// One surface's GPU-side state.
struct SurfaceState {
    size: (u32, u32),
    target: wgpu::Texture,
    view: wgpu::TextureView,
    readable: bool,
    layers: HashMap<LayerId, LayerNode>,
    clear: cherenkov::WorkingColor,
    dirty: bool,
}

impl SurfaceState {
    /// Bytes held by this surface's textures.
    fn gpu_bytes(&self) -> u64 {
        u64::from(self.size.0) * u64::from(self.size.1) * 4
    }
}

/// All render-thread state.
struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    vello: vello::Renderer,
    surfaces: HashMap<SurfaceId, SurfaceState>,
    fonts: HashMap<u64, peniko::FontData>,
    images: HashMap<u64, peniko::ImageData>,
    timestamps: bool,
    query_set: Option<wgpu::QuerySet>,
    query_buffer: Option<wgpu::Buffer>,
    query_staging: wgpu::Buffer,
    timestamps_inside: bool,
    max_texture: u32,
}

/// Creates an instance, adapter and device.
fn create_device(
    config: &VelloConfig,
) -> Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue), EngineError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::from_env().unwrap_or_default(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    // Honour `WGPU_ADAPTER_NAME` when set; otherwise request one with the
    // configured power preference.
    let adapter = pollster::block_on(async {
        match wgpu::util::initialize_adapter_from_env(&instance, None).await {
            Ok(adapter) => Some(adapter),
            Err(_) => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: match config.power {
                        crate::PowerPreference::Low => wgpu::PowerPreference::LowPower,
                        crate::PowerPreference::High => wgpu::PowerPreference::HighPerformance,
                    },
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .ok(),
        }
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
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cherenkov-vello"),
        required_features: required,
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| EngineError::RequestDevice(format!("{e}")))?;
    Ok((adapter, device, queue))
}

fn gpu_info(info: &wgpu::AdapterInfo) -> GpuInfo {
    GpuInfo {
        name: info.name.clone(),
        backend: format!("{:?}", info.backend),
        vendor: info.vendor,
        device: info.device,
        device_type: format!("{:?}", info.device_type),
        driver: info.driver.clone(),
        driver_info: info.driver_info.clone(),
    }
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
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    reason = "moved into the render thread"
)]
pub fn run(
    config: VelloConfig,
    request: DeviceRequest,
    rx: Receiver<Message>,
    init_tx: Sender<Result<Init, EngineError>>,
) {
    let init = (|| {
        let (adapter, device, queue) = match request {
            DeviceRequest::Create => {
                let (adapter, device, queue) = create_device(&config)?;
                (adapter, device, queue)
            }
            DeviceRequest::Existing(source) => {
                let interop::wgpu::DeviceSource {
                    adapter,
                    device,
                    queue,
                } = source;
                (adapter, device, queue)
            }
        };
        let vello = vello::Renderer::new(
            &device,
            RendererOptions {
                use_cpu: false,
                antialiasing_support: AaSupport::area_only(),
                num_init_threads: NonZeroUsize::new(1),
                pipeline_cache: None,
            },
        )
        .map_err(|e| EngineError::Renderer(format!("{e}")))?;
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
            vello,
            surfaces: HashMap::new(),
            fonts: HashMap::new(),
            images: HashMap::new(),
            timestamps,
            query_set,
            query_buffer,
            query_staging,
            timestamps_inside,
        };
        Ok((renderer, adapter.get_info()))
    })();
    let (mut renderer, info) = match init {
        Ok(pair) => pair,
        Err(e) => {
            let _ = init_tx.send(Err(e));
            return;
        }
    };
    let _ = init_tx.send(Ok(Init {
        info: gpu_info(&info),
    }));
    while let Ok(message) = rx.recv() {
        match message {
            Message::CreateSurface { id, target, reply } => {
                let _ = reply.send(renderer.create_surface(id, target));
            }
            Message::ResizeSurface { id, size } => {
                renderer.resize_surface(id, size);
            }
            Message::DestroySurface { id } => {
                renderer.surfaces.remove(&id);
            }
            Message::AddFont { id, data, index } => {
                renderer.fonts.insert(
                    id,
                    peniko::FontData::new(
                        peniko::Blob::new(std::sync::Arc::new(crate::message::SharedBytes(data))),
                        index,
                    ),
                );
            }
            Message::RemoveFont { id } => {
                renderer.fonts.remove(&id);
            }
            Message::AddImage { id, image } => {
                renderer.images.insert(id, image);
            }
            Message::RemoveImage { id } => {
                renderer.images.remove(&id);
            }
            Message::Commit { surface, changes } => {
                renderer.commit(surface, changes);
            }
            Message::Render { time, reply } => {
                let _ = reply.send(renderer.render_frame(time));
            }
            Message::Readback { surface, reply } => {
                let _ = reply.send(renderer.readback(surface));
            }
            Message::Memory { reply } => {
                let _ = reply.send(renderer.memory());
            }
            Message::Trim(pressure) => {
                if pressure == Pressure::Critical {
                    for surface in renderer.surfaces.values_mut() {
                        for node in surface.layers.values_mut() {
                            node.fragment = None;
                        }
                        surface.dirty = true;
                    }
                }
            }
            Message::Shutdown => break,
        }
    }
}

impl Renderer {
    /// Creates a surface's target texture and layer tree.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the phase-2 window target is consumed"
    )]
    fn create_surface(&mut self, id: SurfaceId, target: TargetSpec) -> Result<(), SurfaceError> {
        let (size, readable) = match target {
            TargetSpec::Offscreen { size } => (size, true),
            TargetSpec::Window(_) => {
                return Err(Unsupported::WindowSurface.into());
            }
        };
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
                readable,
                layers,
                clear: cherenkov::WorkingColor::TRANSPARENT,
                dirty: true,
            },
        );
        Ok(())
    }

    /// Resizes a surface, recreating its target.
    fn resize_surface(&mut self, id: SurfaceId, size: (u32, u32)) {
        let Some(state) = self.surfaces.get_mut(&id) else {
            return;
        };
        if size == state.size || size.0 > self.max_texture || size.1 > self.max_texture {
            return;
        }
        let (target, view) = create_target(&self.device, "surface target", size, TARGET_USAGES);
        state.size = size;
        state.target = target;
        state.view = view;
        // Fragments cached against the old size are still valid (they are
        // recorded in user space), but clip-less group layers used the old
        // surface rect; rebuild everything to keep it simple.
        for node in state.layers.values_mut() {
            node.fragment = None;
        }
        state.dirty = true;
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
                LayerOp::Blend(id, blend) => {
                    if let Some(node) = state.layers.get_mut(&id) {
                        node.blend = blend;
                    }
                }
                LayerOp::Filter(id, filter) => {
                    if let Some(node) = state.layers.get_mut(&id) {
                        node.filter = filter;
                    }
                }
                LayerOp::Content(id, content) => {
                    if let Some(node) = state.layers.get_mut(&id) {
                        node.content = content.map(|c| match c {
                            crate::message::LayerContentMsg::Picture(p) => ContentData::Picture(p),
                        });
                        node.fragment = None;
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
                        node.fragment = None;
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

    /// Memory usage across the retained textures.
    fn memory(&self) -> MemoryUsage {
        MemoryUsage {
            gpu: Bytes(
                self.surfaces
                    .values()
                    .map(SurfaceState::gpu_bytes)
                    .sum::<u64>(),
            ),
            cpu: Bytes(0),
        }
    }

    /// Composes `layer`'s fragment, building it when absent.
    fn layer_fragment(
        node: &mut LayerNode,
        fonts: &HashMap<u64, peniko::FontData>,
        images: &HashMap<u64, peniko::ImageData>,
        target_size: (u32, u32),
    ) -> Result<(), RenderError> {
        if node.fragment.is_some() {
            return Ok(());
        }
        let Some(content) = &node.content else {
            return Ok(());
        };
        let mut scene = vello::Scene::new();
        let resources = lower::Resources {
            fonts,
            images,
            target_size,
        };
        match content {
            ContentData::List(list) => lower::lower(
                list,
                &mut scene,
                cherenkov::kurbo::Affine::IDENTITY,
                &resources,
            ),
            ContentData::Picture(picture) => lower::lower(
                picture.display_list(),
                &mut scene,
                cherenkov::kurbo::Affine::IDENTITY,
                &resources,
            ),
        }?;
        node.fragment = Some(scene);
        Ok(())
    }

    /// Composes one layer and its children into `scene` under `parent_xf`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the registries and the layer map are disjoint borrows"
    )]
    fn compose(
        layers: &mut HashMap<LayerId, LayerNode>,
        fonts: &HashMap<u64, peniko::FontData>,
        images: &HashMap<u64, peniko::ImageData>,
        id: LayerId,
        parent_xf: cherenkov::kurbo::Affine,
        target_size: (u32, u32),
        scene: &mut vello::Scene,
        stats: &mut FrameStats,
    ) -> Result<(), RenderError> {
        let Some(node) = layers.get_mut(&id) else {
            return Ok(());
        };
        let world = parent_xf * node.transform;
        if node.filter.is_some() {
            return Err(Unsupported::Filter.into());
        }
        Self::layer_fragment(node, fonts, images, target_size)?;
        let needs_layer =
            node.opacity < 1.0 || node.blend != cherenkov::BlendMode::Normal || node.clip.is_some();
        if needs_layer {
            let clip = node.clip.as_ref().map_or_else(
                || convert::opaque_clip(target_size.0, target_size.1),
                convert::shape_path,
            );
            scene.push_layer(
                peniko::Fill::NonZero,
                convert::blend(node.blend),
                node.opacity,
                world,
                &clip,
            );
        }
        if let Some(fragment) = &node.fragment {
            stats.draws += 1;
            scene.append(fragment, Some(world));
        }
        let children = std::mem::take(&mut node.children);
        let mut result = Ok(());
        for child in &children {
            result = Self::compose(
                layers,
                fonts,
                images,
                *child,
                world,
                target_size,
                scene,
                stats,
            );
            if result.is_err() {
                break;
            }
        }
        let node = layers.get_mut(&id).expect("the node exists");
        node.children = children;
        result?;
        if needs_layer {
            scene.pop_layer();
        }
        Ok(())
    }

    /// Lowers and submits every dirty surface, bracketed by drained
    /// timestamp queries when enabled.
    fn render_frame(&mut self, _time: crate::FrameTime) -> Result<(Next, FrameStats), RenderError> {
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

    /// Composes and renders one surface's scene into its target.
    fn render_surface(&mut self, id: SurfaceId, stats: &mut FrameStats) -> Result<(), RenderError> {
        let Some(surf) = self.surfaces.get_mut(&id) else {
            return Ok(());
        };
        let mut scene = vello::Scene::new();
        let mut layers = std::mem::take(&mut surf.layers);
        let size = surf.size;
        let result = Self::compose(
            &mut layers,
            &self.fonts,
            &self.images,
            0,
            cherenkov::kurbo::Affine::IDENTITY,
            size,
            &mut scene,
            stats,
        );
        surf.layers = layers;
        result?;
        self.vello
            .render_to_texture(
                &self.device,
                &self.queue,
                &scene,
                &surf.view,
                &vello::RenderParams {
                    base_color: convert::color(&surf.clear),
                    width: size.0,
                    height: size.1,
                    antialiasing_method: AaConfig::Area,
                },
            )
            .map_err(|e| RenderError::Readback(format!("vello render: {e}")))?;
        stats.passes += 1;
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

    /// Writes a timestamp query in its own drained submission.
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
        #[expect(
            clippy::cast_precision_loss,
            reason = "tick deltas are small relative to f64's mantissa"
        )]
        let seconds = if ticks.len() >= 2 && ticks[1] > ticks[0] {
            Some(f64::from(self.queue.get_timestamp_period()) * (ticks[1] - ticks[0]) as f64 * 1e-9)
        } else {
            None
        };
        drop(data);
        self.query_staging.unmap();
        Ok(seconds)
    }

    /// Copies a surface's target into `Readback` pixels: each stored
    /// sRGB-encoded premultiplied `rgba8` is decoded per channel to linear
    /// sRGB and mapped into linear Display P3.
    fn readback(&self, surface: SurfaceId) -> Result<Readback, RenderError> {
        let Some(state) = self.surfaces.get(&surface) else {
            return Err(RenderError::Readback("unknown surface".into()));
        };
        if !state.readable {
            return Err(RenderError::NotReadable);
        }
        let (w, h) = state.size;
        let bytes_per_row = (w * 4).div_ceil(256) * 256;
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
            for px in data[start..start + (w * 4) as usize].as_chunks::<4>().0 {
                pixels.push(rgba8_to_working(*px));
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

/// sRGB transfer-function decode.
fn srgb_decode(e: f32) -> f32 {
    if e <= 0.04045 {
        e / 12.92
    } else {
        ((e + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB → linear Display P3 (the inverse of the front end's
/// `LINEAR_DISPLAY_P3_TO_LINEAR_SRGB`).
const LINEAR_SRGB_TO_LINEAR_P3: [[f32; 3]; 3] = [
    [0.822_461_96, 0.177_538_04, 0.0],
    [0.033_194_2, 0.966_805_8, 0.0],
    [0.017_082_632, 0.072_397_44, 0.910_519_96],
];

/// One stored `rgba8` pixel (sRGB-encoded, premultiplied) decoded to
/// premultiplied linear Display P3 — the same math as the bench's
/// `rgba8_to_working`.
fn rgba8_to_working(px: [u8; 4]) -> [f32; 4] {
    let lin = [
        srgb_decode(f32::from(px[0]) / 255.0),
        srgb_decode(f32::from(px[1]) / 255.0),
        srgb_decode(f32::from(px[2]) / 255.0),
    ];
    let m = &LINEAR_SRGB_TO_LINEAR_P3;
    let dot = |row: &[f32; 3]| row[2].mul_add(lin[2], row[1].mul_add(lin[1], row[0] * lin[0]));
    [dot(&m[0]), dot(&m[1]), dot(&m[2]), f32::from(px[3]) / 255.0]
}
