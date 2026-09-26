// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The render thread: sole owner of GPU state.

mod convert;
pub mod filter;
mod gpu_content;
mod lower;
mod shader;

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use cherenkov::ContentChange;
use cherenkov::kurbo::{self, Shape as _};
use vello::peniko;
use vello::{AaConfig, AaSupport, RendererOptions};

use crate::error::{EngineError, RenderError, SurfaceError};
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
    /// GPU-produced content.
    Gpu(Box<gpu_content::GpuSlot>),
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
    /// Shader paint uses inside `fragment`, in command order.
    shader_uses: Vec<shader::ShaderUse>,
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
        shader_uses: Vec::new(),
        children: Vec::new(),
    }
}

/// What a surface renders into, on the render thread.
enum TargetState {
    /// An offscreen texture.
    Offscreen {
        /// The target texture.
        texture: wgpu::Texture,
        /// Its view.
        view: wgpu::TextureView,
    },
    /// A window surface plus its intermediate render texture.
    Window {
        /// The wgpu surface.
        surface: wgpu::Surface<'static>,
        /// The applied configuration.
        config: wgpu::SurfaceConfiguration,
        /// The intermediate texture vello renders into.
        texture: wgpu::Texture,
        /// Its view.
        view: wgpu::TextureView,
    },
}

impl TargetState {
    /// The view vello renders into.
    const fn view(&self) -> &wgpu::TextureView {
        match self {
            Self::Offscreen { view, .. } | Self::Window { view, .. } => view,
        }
    }

    /// The target texture (readback).
    const fn texture(&self) -> &wgpu::Texture {
        match self {
            Self::Offscreen { texture, .. } | Self::Window { texture, .. } => texture,
        }
    }
}

/// One surface's GPU-side state.
struct SurfaceState {
    size: (u32, u32),
    target: TargetState,
    readable: bool,
    layers: HashMap<LayerId, LayerNode>,
    clear: cherenkov::WorkingColor,
    dirty: bool,
    /// Whether the last composed frame asked for another one (animated
    /// shaders, `GpuContent` redraws or `redraw_hint` filters). The redraw
    /// scan re-dirties the surface so the animation survives to the next
    /// frame; a compose that finds nothing animating clears it.
    wants_next: bool,
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
    shaders: shader::ShaderRegistry,
    filters: filter::FilterRegistry,
    /// Presentation blit pipelines, one per swapchain format seen.
    blitters: HashMap<wgpu::TextureFormat, wgpu::util::TextureBlitter>,
    /// Engine start instant; `uniforms.time` and content `elapsed` are
    /// measured from it.
    start: Instant,
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
    // Zero extents (a minimized window, an empty offscreen target) are
    // legal surface sizes but illegal texture sizes; the texture backs a
    // scratch layer, so clamp it — rendering is skipped while a
    // dimension is zero.
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0.max(1),
            height: size.1.max(1),
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
            shaders: shader::ShaderRegistry::default(),
            filters: filter::FilterRegistry::default(),
            blitters: HashMap::new(),
            start: Instant::now(),
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
                renderer.remove_font(id);
            }
            Message::AddImage { id, image } => {
                renderer.images.insert(id, image);
            }
            Message::RemoveImage { id } => {
                renderer.remove_image(id);
            }
            Message::AddShader { id, source, reply } => {
                let _ = reply.send(renderer.shaders.add(&renderer.device, id, &source));
            }
            Message::RemoveShader { id } => {
                renderer.remove_shader(id);
            }
            Message::AddFilter { id, source } => {
                renderer.filters.add(id, source);
            }
            Message::RemoveFilter { id } => {
                renderer.remove_filter(id);
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
                    renderer.flush_fragments();
                }
            }
            Message::Shutdown => break,
        }
    }
}

impl Renderer {
    /// Creates a surface's target texture and layer tree.
    fn create_surface(&mut self, id: SurfaceId, target: TargetSpec) -> Result<(), SurfaceError> {
        // Validate before any wgpu call: `create_texture`/`configure`
        // panic on out-of-limit dimensions, which would kill the render
        // thread and turn one bad size into `SurfaceError::Lost` for the
        // whole engine.
        let size = match &target {
            TargetSpec::Offscreen { size } => *size,
            TargetSpec::Window(window) => (window.config.width, window.config.height),
        };
        if size.0 > self.max_texture || size.1 > self.max_texture {
            return Err(SurfaceError::TooLarge {
                width: size.0,
                height: size.1,
                max: self.max_texture,
            });
        }
        let (target, readable) = match target {
            TargetSpec::Offscreen { size } => {
                let (texture, view) =
                    create_target(&self.device, "surface target", size, TARGET_USAGES);
                (TargetState::Offscreen { texture, view }, true)
            }
            TargetSpec::Window(window) => {
                let interop::wgpu::Window { surface, config } = *window;
                surface.configure(&self.device, &config);
                self.blitters.entry(config.format).or_insert_with(|| {
                    wgpu::util::TextureBlitter::new(&self.device, config.format)
                });
                let (texture, view) =
                    create_target(&self.device, "surface target", size, TARGET_USAGES);
                (
                    TargetState::Window {
                        surface,
                        config,
                        texture,
                        view,
                    },
                    false,
                )
            }
        };
        let mut layers = HashMap::new();
        layers.insert(0, node());
        self.surfaces.insert(
            id,
            SurfaceState {
                size,
                target,
                readable,
                layers,
                clear: cherenkov::WorkingColor::TRANSPARENT,
                dirty: true,
                wants_next: false,
            },
        );
        Ok(())
    }

    /// Resizes a surface, recreating its target (reconfiguring a window).
    fn resize_surface(&mut self, id: SurfaceId, size: (u32, u32)) {
        let Some(state) = self.surfaces.get_mut(&id) else {
            return;
        };
        if size == state.size || size.0 > self.max_texture || size.1 > self.max_texture {
            return;
        }
        let (texture, view) = create_target(&self.device, "surface target", size, TARGET_USAGES);
        state.size = size;
        if let TargetState::Window {
            surface,
            config,
            texture: t,
            view: v,
        } = &mut state.target
        {
            config.width = size.0.max(1);
            config.height = size.1.max(1);
            // `configure` panics on a zero extent — e.g. a minimized
            // window. Keep the clamped config; the surface is skipped at
            // render time until a non-zero resize reconfigures it.
            if size.0 != 0 && size.1 != 0 {
                surface.configure(&self.device, config);
            }
            *t = texture;
            *v = view;
        } else {
            state.target = TargetState::Offscreen { texture, view };
        }
        // Fragments cached against the old size are still valid (they are
        // recorded in user space), but clip-less group layers used the old
        // surface rect; rebuild everything to keep it simple.
        for node in state.layers.values_mut() {
            node.fragment = None;
        }
        state.dirty = true;
    }

    /// Releases a node's current content: the replaced `GpuContent`'s
    /// texture binding and the cached fragment's shader uses must be
    /// unbound, not leaked in vello's override map.
    fn release_content(vello: &mut vello::Renderer, node: &mut LayerNode) {
        if let Some(old) = node.content.take() {
            Self::unregister_content(vello, old);
        }
        for use_ in node.shader_uses.drain(..) {
            vello.override_image(&use_.image, None);
        }
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
                    Self::remove_node(&mut self.vello, &mut state.layers, id);
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
                        Self::release_content(&mut self.vello, node);
                        node.content = content.map(|c| match c {
                            crate::message::LayerContentMsg::Picture(p) => ContentData::Picture(p),
                            crate::message::LayerContentMsg::Gpu(m) => {
                                ContentData::Gpu(Box::new(gpu_content::GpuSlot::new(m)))
                            }
                        });
                        node.fragment = None;
                    }
                }
                LayerOp::ContentChange(id, change) => {
                    if let Some(node) = state.layers.get_mut(&id) {
                        match change {
                            ContentChange::Replace(list) => {
                                Self::release_content(&mut self.vello, node);
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

    /// Removes a font and flushes every cached fragment: fragments embed
    /// resources by value, so without a flush the removed font's retained
    /// pixels keep drawing instead of erroring.
    fn remove_font(&mut self, id: u64) {
        self.fonts.remove(&id);
        self.flush_fragments();
    }

    /// Removes an image and flushes every cached fragment (same contract
    /// as [`Self::remove_font`]).
    fn remove_image(&mut self, id: u64) {
        self.images.remove(&id);
        self.flush_fragments();
    }

    /// Removes a shader and flushes every cached fragment, unbinding the
    /// shader-use overrides that referenced it.
    fn remove_shader(&mut self, id: u64) {
        self.shaders.remove(id);
        self.flush_fragments();
    }

    /// Removes a filter, unbinding its output image's vello override, and
    /// marks all surfaces dirty so a layer still referencing it errors on
    /// the next compose.
    fn remove_filter(&mut self, id: u64) {
        if let Some(image) = self.filters.remove(id) {
            self.vello.override_image(&image, None);
        }
        for surface in self.surfaces.values_mut() {
            surface.dirty = true;
        }
    }

    /// Drops every cached fragment and unbinds its shader-use overrides,
    /// marking all surfaces dirty: the next compose re-lowers and reports
    /// a removed resource instead of drawing its retained pixels.
    fn flush_fragments(&mut self) {
        for surface in self.surfaces.values_mut() {
            for node in surface.layers.values_mut() {
                for use_ in node.shader_uses.drain(..) {
                    self.vello.override_image(&use_.image, None);
                }
                node.fragment = None;
            }
            surface.dirty = true;
        }
    }

    /// Removes `child` from every child list holding it.
    fn detach(layers: &mut HashMap<LayerId, LayerNode>, child: LayerId) {
        for node in layers.values_mut() {
            node.children.retain(|c| *c != child);
        }
    }

    /// Removes a node and its descendants, unbinding any vello image
    /// overrides their textures registered.
    fn remove_node(
        vello: &mut vello::Renderer,
        layers: &mut HashMap<LayerId, LayerNode>,
        id: LayerId,
    ) {
        Self::detach(layers, id);
        if let Some(mut node) = layers.remove(&id) {
            if let Some(content) = node.content.take() {
                Self::unregister_content(vello, content);
            }
            for use_ in node.shader_uses.drain(..) {
                vello.override_image(&use_.image, None);
            }
            for child in node.children {
                Self::remove_node(vello, layers, child);
            }
        }
    }

    /// Unbinds a dropped content's vello image overrides.
    fn unregister_content(vello: &mut vello::Renderer, content: ContentData) {
        if let ContentData::Gpu(slot) = content {
            slot.unregister(vello);
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

    /// Composes `layer`'s fragment, building it when absent. Shader uses
    /// discovered while lowering replace the node's previous set.
    fn layer_fragment(
        &mut self,
        node: &mut LayerNode,
        target_size: (u32, u32),
    ) -> Result<(), RenderError> {
        if node.fragment.is_some() {
            return Ok(());
        }
        let Some(content) = &node.content else {
            return Ok(());
        };
        if !matches!(content, ContentData::List(_) | ContentData::Picture(_)) {
            return Ok(());
        }
        for use_ in node.shader_uses.drain(..) {
            self.vello.override_image(&use_.image, None);
        }
        let mut scene = vello::Scene::new();
        let mut uses = Vec::new();
        {
            let mut resources = lower::Resources {
                fonts: &self.fonts,
                images: &self.images,
                shaders: &self.shaders,
                target_size,
                max_texture: self.max_texture,
                shader_uses: &mut uses,
            };
            match content {
                ContentData::List(list) => lower::lower(
                    list,
                    &mut scene,
                    cherenkov::kurbo::Affine::IDENTITY,
                    &mut resources,
                ),
                ContentData::Picture(picture) => lower::lower(
                    picture.display_list(),
                    &mut scene,
                    cherenkov::kurbo::Affine::IDENTITY,
                    &mut resources,
                ),
                ContentData::Gpu(_) => unreachable!("early return above"),
            }?;
        }
        node.shader_uses = uses;
        node.fragment = Some(scene);
        Ok(())
    }

    /// Renders the shader paints `fragment` references that need
    /// re-evaluation this frame (new, animated, or resized), and binds
    /// their textures into vello's atlas.
    fn evaluate_shader_uses(
        &mut self,
        node: &mut LayerNode,
        now: Instant,
        wants_next: &mut bool,
    ) -> Result<(), RenderError> {
        let time = now.saturating_duration_since(self.start).as_secs_f32();
        for use_ in std::mem::take(&mut node.shader_uses) {
            let mut use_ = use_;
            if !use_.rendered || self.shaders.animated(use_.shader) {
                self.shaders
                    .evaluate(&self.device, &self.queue, &mut use_, time)
                    .map_err(|e| match e {
                        shader::RenderShader::Unregistered(id) => {
                            RenderError::Shader(format!("unregistered shader {id}"))
                        }
                    })?;
                self.vello.override_image(
                    &use_.image,
                    Some(wgpu::TexelCopyTextureInfoBase {
                        texture: use_.texture.clone().expect("evaluated"),
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    }),
                );
                self.vello.mark_override_image_dirty(&use_.image);
            }
            *wants_next |= self.shaders.animated(use_.shader);
            node.shader_uses.push(use_);
        }
        Ok(())
    }

    /// Composes one layer and its children into `scene` under `parent_xf`.
    #[expect(
        clippy::too_many_arguments,
        reason = "compose threads the scene, stats and refresh flag through recursion"
    )]
    fn compose(
        &mut self,
        layers: &mut HashMap<LayerId, LayerNode>,
        id: LayerId,
        parent_xf: cherenkov::kurbo::Affine,
        target_size: (u32, u32),
        scene: &mut vello::Scene,
        stats: &mut FrameStats,
        wants_next: &mut bool,
        now: Instant,
    ) -> Result<(), RenderError> {
        let Some(node) = layers.get_mut(&id) else {
            return Ok(());
        };
        let world = parent_xf * node.transform;
        self.layer_fragment(node, target_size)?;
        self.evaluate_shader_uses(node, now, wants_next)?;
        if node.filter.is_some() {
            return self.compose_filtered(
                layers,
                id,
                parent_xf,
                world,
                target_size,
                scene,
                stats,
                wants_next,
                now,
            );
        }
        let needs_layer =
            node.opacity < 1.0 || node.blend != cherenkov::BlendMode::Normal || node.clip.is_some();
        if needs_layer {
            let clip = node.clip.as_ref().map_or_else(
                || convert::opaque_clip(target_size.0, target_size.1),
                convert::shape_path,
            );
            let rule = node.clip.as_ref().map_or(peniko::Fill::NonZero, |clip| {
                convert::fill(convert::shape_rule(clip))
            });
            scene.push_layer(
                rule,
                convert::blend(node.blend),
                node.opacity,
                world,
                &clip,
            );
        }
        self.compose_contents(
            layers,
            id,
            world,
            target_size,
            scene,
            stats,
            wants_next,
            now,
        )?;
        if needs_layer {
            scene.pop_layer();
        }
        Ok(())
    }

    /// Appends `id`'s content fragment (or gpu content image) and children
    /// at `world` — the part of `compose` the filter capture path reuses.
    #[expect(
        clippy::too_many_arguments,
        reason = "shared inner step of compose and compose_filtered"
    )]
    fn compose_contents(
        &mut self,
        layers: &mut HashMap<LayerId, LayerNode>,
        id: LayerId,
        world: cherenkov::kurbo::Affine,
        target_size: (u32, u32),
        scene: &mut vello::Scene,
        stats: &mut FrameStats,
        wants_next: &mut bool,
        now: Instant,
    ) -> Result<(), RenderError> {
        let Some(node) = layers.get_mut(&id) else {
            return Ok(());
        };
        if let Some(fragment) = &node.fragment {
            stats.draws += 1;
            scene.append(fragment, Some(world));
        }
        let gpu_image = if let Some(ContentData::Gpu(slot)) = &mut node.content {
            *wants_next |=
                slot.evaluate(&self.device, &self.queue, &mut self.vello, self.start, now);
            let ready = slot.ready.as_ref().expect("evaluate ensured it");
            Some((
                ready.image.clone(),
                f64::from(slot.size.0) / f64::from(ready.image.width),
                f64::from(slot.size.1) / f64::from(ready.image.height),
            ))
        } else {
            None
        };
        let children = std::mem::take(&mut node.children);
        if let Some((image, sx, sy)) = gpu_image {
            scene.draw_image(
                &peniko::ImageBrush {
                    image,
                    sampler: peniko::ImageSampler {
                        x_extend: peniko::Extend::Pad,
                        y_extend: peniko::Extend::Pad,
                        quality: peniko::ImageQuality::Medium,
                        alpha: 1.0,
                    },
                },
                world * cherenkov::kurbo::Affine::scale_non_uniform(sx, sy),
            );
            stats.draws += 1;
        }
        let mut result = Ok(());
        for child in &children {
            result = self.compose(
                layers,
                *child,
                world,
                target_size,
                scene,
                stats,
                wants_next,
                now,
            );
            if result.is_err() {
                break;
            }
        }
        let node = layers.get_mut(&id).expect("the node exists");
        node.children = children;
        result
    }

    /// The filtered path: renders the layer's subtree into a capture
    /// texture, runs the effect, and draws the output as an image at the
    /// capture bounds. The filtered output honours `opacity`/`blend`/`clip`
    /// through a pushed composite layer, like the unfiltered path.
    #[expect(
        clippy::too_many_arguments,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "mirrors compose's parameters; device-pixel sizes are small positive values"
    )]
    fn compose_filtered(
        &mut self,
        layers: &mut HashMap<LayerId, LayerNode>,
        id: LayerId,
        parent_xf: cherenkov::kurbo::Affine,
        world: cherenkov::kurbo::Affine,
        target_size: (u32, u32),
        scene: &mut vello::Scene,
        stats: &mut FrameStats,
        wants_next: &mut bool,
        now: Instant,
    ) -> Result<(), RenderError> {
        let node = layers.get(&id).expect("the node exists");
        let filter_id = node.filter.expect("checked by compose");
        // Capture bounds in target space: the clip's device-space bounds,
        // intersected with the target; the whole target when unclipped.
        let clip_bounds = node.clip.as_ref().map(|clip| {
            let rect = (world * convert::shape_path(clip)).bounding_box();
            rect.intersect(kurbo::Rect::new(
                0.0,
                0.0,
                f64::from(target_size.0),
                f64::from(target_size.1),
            ))
        });
        let bounds = clip_bounds.unwrap_or_else(|| {
            kurbo::Rect::new(0.0, 0.0, f64::from(target_size.0), f64::from(target_size.1))
        });
        let size = (
            (bounds.width().ceil() as u32).clamp(1, self.max_texture),
            (bounds.height().ceil() as u32).clamp(1, self.max_texture),
        );
        // Compose the subtree into a capture scene; its parent transform
        // shifts the capture bounds' origin to the texture's.
        let capture_parent =
            cherenkov::kurbo::Affine::translate((-bounds.x0, -bounds.y0)) * parent_xf;
        let mut sub = vello::Scene::new();
        let capture_world = capture_parent * node.transform;
        self.compose_contents(
            layers,
            id,
            capture_world,
            target_size,
            &mut sub,
            stats,
            wants_next,
            now,
        )?;
        let capture_view = self
            .filters
            .ensure_capture(&self.device, filter_id.raw(), size)?;
        self.vello
            .render_to_texture(
                &self.device,
                &self.queue,
                &sub,
                &capture_view,
                &vello::RenderParams {
                    base_color: peniko::Color::TRANSPARENT,
                    width: size.0,
                    height: size.1,
                    antialiasing_method: AaConfig::Area,
                },
            )
            .map_err(|e| RenderError::Render(format!("filter capture: {e}")))?;
        stats.passes += 1;
        let (_out_view, image, again) = self.filters.evaluate(
            &self.device,
            &self.queue,
            &mut self.vello,
            filter_id.raw(),
            size,
        )?;
        let out_texture = self
            .filters
            .output_texture(filter_id.raw())
            .expect("evaluate produced it");
        self.vello.override_image(
            &image,
            Some(wgpu::TexelCopyTextureInfoBase {
                texture: out_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            }),
        );
        self.vello.mark_override_image_dirty(&image);
        Self::filtered_output(layers, id, world, bounds, target_size, scene, stats, image);
        *wants_next |= again || self.filters.redraw_hint(filter_id.raw());
        Ok(())
    }

    /// Draws a filter's output image at `bounds` in `scene`, wrapped in a
    /// pushed layer when `id` declares opacity, a non-normal blend or a clip.
    #[expect(
        clippy::too_many_arguments,
        reason = "shares compose_filtered's parameters"
    )]
    fn filtered_output(
        layers: &HashMap<LayerId, LayerNode>,
        id: LayerId,
        world: cherenkov::kurbo::Affine,
        bounds: kurbo::Rect,
        target_size: (u32, u32),
        scene: &mut vello::Scene,
        stats: &mut FrameStats,
        image: peniko::ImageData,
    ) {
        let node = layers.get(&id).expect("the node exists");
        let needs_layer =
            node.opacity < 1.0 || node.blend != cherenkov::BlendMode::Normal || node.clip.is_some();
        if needs_layer {
            let clip = node.clip.as_ref().map_or_else(
                || convert::opaque_clip(target_size.0, target_size.1),
                convert::shape_path,
            );
            let rule = node.clip.as_ref().map_or(peniko::Fill::NonZero, |clip| {
                convert::fill(convert::shape_rule(clip))
            });
            scene.push_layer(
                rule,
                convert::blend(node.blend),
                node.opacity,
                world,
                &clip,
            );
        }
        scene.draw_image(
            &peniko::ImageBrush {
                image,
                sampler: peniko::ImageSampler {
                    x_extend: peniko::Extend::Pad,
                    y_extend: peniko::Extend::Pad,
                    quality: peniko::ImageQuality::Medium,
                    alpha: 1.0,
                },
            },
            cherenkov::kurbo::Affine::translate((bounds.x0, bounds.y0)),
        );
        if needs_layer {
            scene.pop_layer();
        }
        stats.draws += 1;
    }

    /// Marks surfaces dirty for pending `GpuContent`/`Filter` redraw
    /// requests and for surfaces whose last frame asked for another
    /// (animated shaders, redraw-hinting filters, looping `GpuContent`).
    /// Returns whether a redraw is pending.
    fn scan_redraw_requests(&mut self) -> bool {
        let mut pending = false;
        for surface in self.surfaces.values_mut() {
            if surface.wants_next {
                surface.dirty = true;
            }
            for node in surface.layers.values() {
                if let Some(ContentData::Gpu(slot)) = &node.content
                    && slot.dirty.load(std::sync::atomic::Ordering::Relaxed)
                {
                    surface.dirty = true;
                    pending = true;
                }
            }
        }
        if self.filters.take_redraw_requests() {
            for surface in self.surfaces.values_mut() {
                surface.dirty = true;
            }
            pending = true;
        }
        pending
    }

    /// Lowers and submits every dirty surface, bracketed by drained
    /// timestamp queries when enabled.
    fn render_frame(&mut self, time: crate::FrameTime) -> Result<(Next, FrameStats), RenderError> {
        let mut stats = FrameStats::default();
        let mut wants_next = self.scan_redraw_requests();
        let mut dirty: Vec<SurfaceId> = self
            .surfaces
            .iter()
            .filter(|(_, s)| s.dirty)
            .map(|(id, _)| *id)
            .collect();
        dirty.sort_unstable();
        if dirty.is_empty() {
            return Ok((
                if wants_next {
                    Self::next_frame(time)
                } else {
                    Next::Idle
                },
                stats,
            ));
        }
        let now = Instant::now();
        self.drain_and_stamp(0)?;
        let mut result = Ok(());
        for id in dirty {
            result = self.render_surface(id, &mut stats, &mut wants_next, now);
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
        Ok((
            if wants_next {
                Self::next_frame(time)
            } else {
                Next::Idle
            },
            stats,
        ))
    }

    /// The refresh request for an animating frame: one tick at 60 Hz.
    fn next_frame(time: crate::FrameTime) -> Next {
        Next::At {
            time: time.0 + Duration::from_secs_f64(1.0 / 60.0),
            rate: 60..=60,
        }
    }

    /// Composes and renders one surface's scene into its target, then
    /// presents a window surface.
    fn render_surface(
        &mut self,
        id: SurfaceId,
        stats: &mut FrameStats,
        wants_next: &mut bool,
        now: Instant,
    ) -> Result<(), RenderError> {
        let Some(mut surf) = self.surfaces.remove(&id) else {
            return Ok(());
        };
        let result = self.render_surface_inner(&mut surf, stats, wants_next, now);
        self.surfaces.insert(id, surf);
        result
    }

    /// The body of `render_surface`, with the surface taken out of the map
    /// so `self` is free for `compose`.
    fn render_surface_inner(
        &mut self,
        surf: &mut SurfaceState,
        stats: &mut FrameStats,
        wants_next: &mut bool,
        now: Instant,
    ) -> Result<(), RenderError> {
        let mut scene = vello::Scene::new();
        let size = surf.size;
        if size.0 == 0 || size.1 == 0 {
            // Nothing drawable — e.g. a minimized window. Clear the flags;
            // a later resize marks the surface dirty again and animation
            // state is re-derived by the next real compose.
            surf.dirty = false;
            surf.wants_next = false;
            return Ok(());
        }
        // `wants_next` is per-surface until composed: a frame that asked
        // for a follow-up marks only the surfaces still animating, so one
        // idle surface cannot keep another surface's animation dirty.
        let mut surface_next = false;
        self.compose(
            &mut surf.layers,
            0,
            cherenkov::kurbo::Affine::IDENTITY,
            size,
            &mut scene,
            stats,
            &mut surface_next,
            now,
        )?;
        self.vello
            .render_to_texture(
                &self.device,
                &self.queue,
                &scene,
                surf.target.view(),
                &vello::RenderParams {
                    base_color: convert::color(&surf.clear),
                    width: size.0,
                    height: size.1,
                    antialiasing_method: AaConfig::Area,
                },
            )
            .map_err(|e| RenderError::Render(format!("vello render: {e}")))?;
        stats.passes += 1;
        surf.wants_next = surface_next;
        *wants_next |= surface_next;
        if let TargetState::Window {
            surface, config, ..
        } = &surf.target
        {
            use wgpu::CurrentSurfaceTexture as Current;
            let frame = match surface.get_current_texture() {
                Current::Success(frame) | Current::Suboptimal(frame) => frame,
                Current::Lost | Current::Outdated => {
                    // Reconfigure and try again next render.
                    surface.configure(&self.device, config);
                    return Ok(());
                }
                Current::Timeout | Current::Occluded => {
                    // Skip presenting this frame; the surface stays dirty.
                    return Ok(());
                }
                Current::Validation => {
                    return Err(RenderError::Render(
                        "surface acquisition hit a validation error".into(),
                    ));
                }
            };
            let format = config.format;
            let blitter = self
                .blitters
                .entry(format)
                .or_insert_with(|| wgpu::util::TextureBlitter::new(&self.device, format));
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("window blit"),
                });
            let frame_view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            blitter.copy(&self.device, &mut encoder, surf.target.view(), &frame_view);
            self.queue.submit([encoder.finish()]);
            frame.present();
        }
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
                texture: state.target.texture(),
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use cherenkov::Draw as _;

    use crate::message::{GpuContentMsg, LayerContentMsg};

    use super::*;

    /// A `Renderer` on the test adapter, or `None` without an adapter.
    fn renderer() -> Option<Renderer> {
        let (_adapter, device, queue) = create_device(&VelloConfig::default()).ok()?;
        let vello = vello::Renderer::new(
            &device,
            RendererOptions {
                use_cpu: false,
                antialiasing_support: AaSupport::area_only(),
                num_init_threads: NonZeroUsize::new(1),
                pipeline_cache: None,
            },
        )
        .expect("vello renderer");
        let query_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp staging"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Some(Renderer {
            max_texture: device.limits().max_texture_dimension_2d,
            device,
            queue,
            vello,
            surfaces: HashMap::new(),
            fonts: HashMap::new(),
            images: HashMap::new(),
            shaders: shader::ShaderRegistry::default(),
            filters: filter::FilterRegistry::default(),
            blitters: HashMap::new(),
            start: Instant::now(),
            timestamps: false,
            query_set: None,
            query_buffer: None,
            query_staging,
            timestamps_inside: false,
        })
    }

    /// `GpuContent` that renders nothing.
    struct NoopContent;

    impl crate::gpu_content::GpuContent for NoopContent {
        async fn setup(&mut self, _gpu: &interop::wgpu::Context<'_>) {}

        fn render(&mut self, _frame: &mut interop::wgpu::Frame<'_>) {}
    }

    /// Whether `image` has a vello override bound — checked
    /// non-destructively: a found binding is restored.
    fn override_bound(renderer: &mut Renderer, image: &peniko::ImageData) -> bool {
        let prev = renderer.vello.override_image(image, None);
        let bound = prev.is_some();
        renderer.vello.override_image(image, prev);
        bound
    }

    /// The image identity `layer`'s Gpu slot bound on `surface`.
    fn gpu_image(renderer: &Renderer, surface: SurfaceId, layer: LayerId) -> peniko::ImageData {
        let node = &renderer.surfaces[&surface].layers[&layer];
        let Some(ContentData::Gpu(slot)) = &node.content else {
            panic!("expected gpu content");
        };
        slot.ready.as_ref().expect("rendered").image.clone()
    }

    /// Commits a surface with one layer holding a `GpuContent`, renders
    /// once, and returns the image identity the slot bound.
    fn bound_gpu_image(renderer: &mut Renderer) -> peniko::ImageData {
        renderer
            .create_surface(1, TargetSpec::Offscreen { size: (32, 32) })
            .expect("surface");
        renderer.commit(
            1,
            ChangeSet {
                clear: None,
                ops: vec![
                    LayerOp::Create(1),
                    LayerOp::Push { parent: 0, child: 1 },
                    LayerOp::Content(
                        1,
                        Some(LayerContentMsg::Gpu(GpuContentMsg {
                            id: 1,
                            size: (8, 8),
                            dirty: Arc::new(AtomicBool::new(false)),
                            content: Box::new(NoopContent),
                        })),
                    ),
                ],
            },
        );
        renderer
            .render_frame(crate::FrameTime::now())
            .expect("render");
        gpu_image(renderer, 1, 1)
    }

    /// Removing a shader must invalidate the fragments that embedded it:
    /// the next render re-lowers and reports the dangling shader instead
    /// of drawing its retained texture.
    #[test]
    fn removing_a_shader_invalidates_cached_fragments() {
        let Some(mut renderer) = renderer() else { return };
        renderer
            .create_surface(1, TargetSpec::Offscreen { size: (32, 32) })
            .expect("surface");
        renderer
            .shaders
            .add(
                &renderer.device,
                7,
                &crate::message::ShaderSpec {
                    source: std::borrow::Cow::Owned(
                        "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { return vec4<f32>(uv, 0.0, 1.0); }"
                            .into(),
                    ),
                    animated: false,
                },
            )
            .expect("shader");
        renderer.commit(
            1,
            ChangeSet {
                clear: None,
                ops: vec![
                    LayerOp::Create(1),
                    LayerOp::Push { parent: 0, child: 1 },
                    LayerOp::ContentChange(
                        1,
                        cherenkov::Content::record(|c| {
                            c.fill(
                                cherenkov::kurbo::Rect::new(0., 0., 8., 8.),
                                cherenkov::ShaderPaint {
                                    shader: cherenkov::ShaderId::new(7),
                                    uniforms: vec![],
                                },
                            );
                        })
                        .take_change()
                        .expect("first change is Replace"),
                    ),
                ],
            },
        );
        renderer
            .render_frame(crate::FrameTime::now())
            .expect("render");
        let image = {
            let node = &renderer.surfaces[&1].layers[&1];
            assert!(node.fragment.is_some(), "setup: fragment cached");
            assert!(!node.shader_uses.is_empty(), "setup: shader use");
            node.shader_uses[0].image.clone()
        };
        assert!(override_bound(&mut renderer, &image), "setup: bound");

        renderer.remove_shader(7);

        {
            let node = &renderer.surfaces[&1].layers[&1];
            assert!(node.fragment.is_none(), "fragment flushed");
            assert!(node.shader_uses.is_empty(), "shader uses drained");
        }
        assert!(
            !override_bound(&mut renderer, &image),
            "use's override unbound"
        );
        let result = renderer.render_frame(crate::FrameTime::now());
        assert!(
            matches!(result, Err(RenderError::Shader(_))),
            "re-lower must report the missing shader: {result:?}"
        );
    }

    /// The same invalidation for an embedded image: no override, but the
    /// cached fragment must be flushed so the re-lower reports the
    /// dangling id instead of drawing retained pixels.
    #[test]
    fn removing_an_image_invalidates_cached_fragments() {
        let Some(mut renderer) = renderer() else { return };
        renderer
            .create_surface(1, TargetSpec::Offscreen { size: (32, 32) })
            .expect("surface");
        let bytes: Arc<[u8]> = Arc::from(&[255u8, 255, 255, 255][..]);
        renderer.images.insert(
            9,
            peniko::ImageData {
                data: peniko::Blob::new(Arc::new(crate::message::SharedBytes(bytes))),
                format: peniko::ImageFormat::Rgba8,
                alpha_type: peniko::ImageAlphaType::AlphaPremultiplied,
                width: 1,
                height: 1,
            },
        );
        renderer.commit(
            1,
            ChangeSet {
                clear: None,
                ops: vec![
                    LayerOp::Create(1),
                    LayerOp::Push { parent: 0, child: 1 },
                    LayerOp::ContentChange(
                        1,
                        cherenkov::Content::record(|c| {
                            c.image(
                                cherenkov::ImageId::new(9),
                                cherenkov::kurbo::Rect::new(0., 0., 8., 8.),
                                cherenkov::Sampling::Nearest,
                            );
                        })
                        .take_change()
                        .expect("first change is Replace"),
                    ),
                ],
            },
        );
        renderer
            .render_frame(crate::FrameTime::now())
            .expect("render");
        assert!(renderer.surfaces[&1].layers[&1].fragment.is_some());

        renderer.remove_image(9);

        let result = renderer.render_frame(crate::FrameTime::now());
        assert!(
            matches!(result, Err(RenderError::Image(_))),
            "re-lower must report the missing image: {result:?}"
        );
    }

    /// Replacing a layer's live `Content` must run the same release as
    /// `LayerOp::Content`: the old `GpuContent`'s override binding must be
    /// removed, not leaked in vello's override map.
    #[test]
    fn replace_releases_the_old_contents_binding() {
        let Some(mut renderer) = renderer() else { return };
        let image = bound_gpu_image(&mut renderer);
        assert!(override_bound(&mut renderer, &image), "setup: bound");
        let replace = cherenkov::Content::record(|_| {})
            .take_change()
            .expect("first change is Replace");
        renderer.commit(
            1,
            ChangeSet {
                clear: None,
                ops: vec![LayerOp::ContentChange(1, replace)],
            },
        );
        assert!(
            !override_bound(&mut renderer, &image),
            "replaced content's override must be unbound"
        );
    }
}
