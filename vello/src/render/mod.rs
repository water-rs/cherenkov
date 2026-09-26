// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The render side of the [`Vello`](crate::Vello) backend: sole owner of
//! GPU state, driven by the shared front end's render loop.

mod convert;
pub mod filter;
mod gpu_content;
mod lower;
mod shader;

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;

use cherenkov::FrameStats;
use cherenkov::kurbo::{self, Shape as _};
use cherenkov::{
    ContentOp, EngineError, FontData, FontId, ImageId, ImageUpload, LayerId, LayerNode,
    MemoryUsage, OffscreenFormat, Pressure, Readback, Redraw, RenderError, Renderer, ResourceError,
    SurfaceError, SurfaceFrame, SurfaceId, SurfaceInfo, SurfaceTree,
};
use vello::peniko;
use vello::{AaConfig, AaSupport, RendererOptions};

use crate::interop;
use crate::{PowerPreference, VelloConfig, VelloInfo, VelloTarget};

/// The surface target format: premultiplied sRGB-encoded sRGB.
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

const TARGET_USAGES: wgpu::TextureUsages = wgpu::TextureUsages::from_bits_retain(
    wgpu::TextureUsages::RENDER_ATTACHMENT.bits()
        | wgpu::TextureUsages::STORAGE_BINDING.bits()
        | wgpu::TextureUsages::COPY_SRC.bits()
        | wgpu::TextureUsages::TEXTURE_BINDING.bits(),
);

/// `Arc<[u8]>` wrapped so it coerces into `peniko::Blob::new`'s
/// `Arc<dyn AsRef<[u8]> + Send + Sync>` (an `Arc<[u8]>` cannot unsize to a
/// trait object directly).
pub struct SharedBytes(pub Arc<[u8]>);

impl AsRef<[u8]> for SharedBytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// A shader's WGSL fragment source, crossing to the render thread.
#[derive(Debug)]
pub struct ShaderSpec {
    /// The fragment source (without the prelude).
    pub source: std::borrow::Cow<'static, str>,
    /// Whether the shader is re-rendered every frame (`time` uniform).
    pub animated: bool,
}

/// What a layer draws, on the render thread.
enum ContentData {
    /// A live display list, patched by `ContentOp::Update`s.
    List(cherenkov::DisplayList),
    /// A shared immutable picture.
    Picture(cherenkov::Picture),
    /// GPU-produced content.
    Gpu(Box<gpu_content::GpuSlot>),
}

/// A layer's retained render-side caches, keyed by layer id inside each
/// surface. The sampled layer state lives in the front end's
/// [`SurfaceTree`]; this holds only what lowering produced.
#[derive(Default)]
struct LayerCache {
    /// The layer's current content.
    content: Option<ContentData>,
    /// The content lowered into a reusable scene fragment; rebuilt whenever
    /// `content` changes or the surface is resized.
    fragment: Option<vello::Scene>,
    /// Shader paint uses inside `fragment`, in command order.
    shader_uses: Vec<shader::ShaderUse>,
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
    /// Per-layer render caches, created on first content.
    layers: HashMap<LayerId, LayerCache>,
}

impl SurfaceState {
    /// Bytes held by this surface's textures.
    fn gpu_bytes(&self) -> u64 {
        u64::from(self.size.0) * u64::from(self.size.1) * 4
    }
}

/// All render-thread state: the [`Vello`](crate::Vello) backend's
/// [`Renderer`] implementation.
pub struct VelloRenderer {
    /// The wgpu device.
    pub device: wgpu::Device,
    /// The submission queue.
    pub queue: wgpu::Queue,
    /// The vello renderer.
    pub vello: vello::Renderer,
    surfaces: HashMap<SurfaceId, SurfaceState>,
    /// Registered fonts, by `FontId::raw`.
    fonts: HashMap<u64, peniko::FontData>,
    /// Registered images, by `ImageId::raw`.
    images: HashMap<u64, peniko::ImageData>,
    /// Registered shaders, by `ShaderId::raw`.
    pub shaders: shader::ShaderRegistry,
    /// Registered filters, by `FilterId::raw`.
    pub filters: filter::FilterRegistry,
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

impl std::fmt::Debug for VelloRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VelloRenderer").finish_non_exhaustive()
    }
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
                        PowerPreference::Low => wgpu::PowerPreference::LowPower,
                        PowerPreference::High => wgpu::PowerPreference::HighPerformance,
                    },
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .ok(),
        }
    })
    .ok_or_else(|| EngineError::Backend("no suitable wgpu adapter".into()))?;
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
    .map_err(|e| EngineError::Backend(format!("device request failed: {e}")))?;
    Ok((adapter, device, queue))
}

fn gpu_info(info: &wgpu::AdapterInfo) -> VelloInfo {
    VelloInfo {
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

/// The `Backend::init` implementation: creates or adopts the device and
/// builds the renderer on the render thread.
///
/// # Errors
/// [`EngineError::Backend`] when no adapter exists, device creation fails,
/// or the vello renderer fails to initialize.
#[expect(
    clippy::needless_pass_by_value,
    reason = "the Backend contract moves the config onto the render thread"
)]
pub fn init(config: VelloConfig) -> Result<(VelloRenderer, VelloInfo), EngineError> {
    let VelloConfig { device, .. } = config.clone();
    let (adapter, device, queue) = match device {
        Some(source) => {
            let interop::wgpu::DeviceSource {
                adapter,
                device,
                queue,
            } = source;
            (adapter, device, queue)
        }
        None => create_device(&config)?,
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
    .map_err(|e| EngineError::Backend(format!("renderer: {e}")))?;
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
    let renderer = VelloRenderer {
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
    Ok((renderer, gpu_info(&adapter.get_info())))
}

impl VelloRenderer {
    /// Attaches a [`GpuContent`](crate::interop::GpuContent) box to a layer:
    /// the [`GpuContent`](cherenkov::GpuContent) capability hook.
    pub fn set_gpu_content(
        &mut self,
        surface: SurfaceId,
        layer: LayerId,
        size: (u32, u32),
        content: crate::interop::GpuContentBox,
    ) {
        let Some(state) = self.surfaces.get_mut(&surface) else {
            return;
        };
        let slot = gpu_content::GpuSlot::new(size, content.dirty, content.content);
        let cache = state.layers.entry(layer).or_default();
        Self::clear_content(&mut self.vello, cache);
        cache.content = Some(ContentData::Gpu(Box::new(slot)));
    }

    /// Drops a layer cache's content, unbinding vello overrides.
    fn clear_content(vello: &mut vello::Renderer, cache: &mut LayerCache) {
        if let Some(old) = cache.content.take() {
            Self::unregister_content(vello, old);
        }
        for use_ in cache.shader_uses.drain(..) {
            vello.override_image(&use_.image, None);
        }
        cache.fragment = None;
    }

    /// Unbinds a dropped content's vello image overrides.
    fn unregister_content(vello: &mut vello::Renderer, content: ContentData) {
        if let ContentData::Gpu(slot) = content {
            slot.unregister(vello);
        }
    }

    /// Memory usage across the retained textures.
    fn memory_usage(&self) -> MemoryUsage {
        MemoryUsage {
            gpu: cherenkov::Bytes(
                self.surfaces
                    .values()
                    .map(SurfaceState::gpu_bytes)
                    .sum::<u64>()
                    + self
                        .images
                        .values()
                        .map(|image| image.data.data().len() as u64)
                        .sum::<u64>(),
            ),
            cpu: cherenkov::Bytes(0),
        }
    }

    /// Composes `layer`'s fragment, building it when absent. Shader uses
    /// discovered while lowering replace the cache's previous set.
    fn layer_fragment(
        &mut self,
        cache: &mut LayerCache,
        target_size: (u32, u32),
    ) -> Result<(), RenderError> {
        if cache.fragment.is_some() {
            return Ok(());
        }
        let Some(content) = &cache.content else {
            return Ok(());
        };
        if !matches!(content, ContentData::List(_) | ContentData::Picture(_)) {
            return Ok(());
        }
        for use_ in cache.shader_uses.drain(..) {
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
        cache.shader_uses = uses;
        cache.fragment = Some(scene);
        Ok(())
    }

    /// Renders the shader paints `cache`'s fragment references that need
    /// re-evaluation this frame (new, animated, or resized), and binds
    /// their textures into vello's atlas.
    fn evaluate_shader_uses(
        &mut self,
        cache: &mut LayerCache,
        now: Instant,
        wants_next: &mut bool,
    ) -> Result<(), RenderError> {
        let time = now.saturating_duration_since(self.start).as_secs_f32();
        for use_ in std::mem::take(&mut cache.shader_uses) {
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
            cache.shader_uses.push(use_);
        }
        Ok(())
    }

    /// Composes layer `id` and its children into `scene` under `parent_xf`.
    /// `id`'s sampled state comes from `tree`; `layers` holds the render
    /// caches keyed by the same id.
    #[expect(
        clippy::too_many_arguments,
        reason = "compose threads the scene, stats and refresh flag through recursion"
    )]
    fn compose(
        &mut self,
        layers: &mut HashMap<LayerId, LayerCache>,
        id: LayerId,
        tree: &SurfaceTree,
        parent_xf: cherenkov::kurbo::Affine,
        target_size: (u32, u32),
        scene: &mut vello::Scene,
        stats: &mut FrameStats,
        wants_next: &mut bool,
        now: Instant,
    ) -> Result<(), RenderError> {
        let node = tree.layer(id);
        let world = parent_xf * node.transform;
        let content_world = parent_xf * node.content_transform();
        let mut cache = layers.remove(&id).unwrap_or_default();
        let result = (|| {
            self.layer_fragment(&mut cache, target_size)?;
            self.evaluate_shader_uses(&mut cache, now, wants_next)?;
            if node.filter.is_some() {
                return self.compose_filtered(
                    layers,
                    node,
                    tree,
                    &mut cache,
                    parent_xf,
                    world,
                    target_size,
                    scene,
                    stats,
                    wants_next,
                    now,
                );
            }
            let needs_layer = node.opacity < 1.0
                || node.blend != cherenkov::BlendMode::Normal
                || node.clip.is_some();
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
            self.compose_contents(
                layers,
                node,
                tree,
                &mut cache,
                content_world,
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
        })();
        layers.insert(id, cache);
        result
    }

    /// Appends `node`'s content fragment (or gpu content image) and children
    /// at `content_world` — the part of `compose` the filter capture path
    /// reuses.
    #[expect(
        clippy::too_many_arguments,
        reason = "shared inner step of compose and compose_filtered"
    )]
    fn compose_contents(
        &mut self,
        layers: &mut HashMap<LayerId, LayerCache>,
        node: &LayerNode,
        tree: &SurfaceTree,
        cache: &mut LayerCache,
        content_world: cherenkov::kurbo::Affine,
        target_size: (u32, u32),
        scene: &mut vello::Scene,
        stats: &mut FrameStats,
        wants_next: &mut bool,
        now: Instant,
    ) -> Result<(), RenderError> {
        if let Some(fragment) = &cache.fragment {
            stats.draws += 1;
            scene.append(fragment, Some(content_world));
        }
        let gpu_image = if let Some(ContentData::Gpu(slot)) = &mut cache.content {
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
                content_world * cherenkov::kurbo::Affine::scale_non_uniform(sx, sy),
            );
            stats.draws += 1;
        }
        for child in &node.children {
            self.compose(
                layers,
                *child,
                tree,
                content_world,
                target_size,
                scene,
                stats,
                wants_next,
                now,
            )?;
        }
        Ok(())
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
        layers: &mut HashMap<LayerId, LayerCache>,
        node: &LayerNode,
        tree: &SurfaceTree,
        cache: &mut LayerCache,
        parent_xf: cherenkov::kurbo::Affine,
        world: cherenkov::kurbo::Affine,
        target_size: (u32, u32),
        scene: &mut vello::Scene,
        stats: &mut FrameStats,
        wants_next: &mut bool,
        now: Instant,
    ) -> Result<(), RenderError> {
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
        let capture_world = capture_parent * node.content_transform();
        self.compose_contents(
            layers,
            node,
            tree,
            cache,
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
        let (_out_view, image, again) =
            self.filters
                .evaluate(&self.device, &self.queue, filter_id.raw(), size)?;
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
        Self::filtered_output(node, world, bounds, target_size, scene, stats, image);
        *wants_next |= again || self.filters.redraw_hint(filter_id.raw());
        Ok(())
    }

    /// Draws a filter's output image at `bounds` in `scene`, wrapped in a
    /// pushed layer when `node` declares opacity, a non-normal blend or a
    /// clip.
    fn filtered_output(
        node: &LayerNode,
        world: cherenkov::kurbo::Affine,
        bounds: kurbo::Rect,
        target_size: (u32, u32),
        scene: &mut vello::Scene,
        stats: &mut FrameStats,
        image: peniko::ImageData,
    ) {
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

    /// Whether any of the surface's GPU contents asked for a re-render or
    /// any animated shader paint is in use — the backend-side dirty scan.
    /// Does not consume flags; [`gpu_content::GpuSlot::evaluate`] takes
    /// them when it renders. First-rendered content needs no flag: setting
    /// the content already marked the surface `changed`.
    fn surface_needs_redraw(&self, state: &SurfaceState) -> bool {
        state.layers.values().any(|cache| {
            if let Some(ContentData::Gpu(slot)) = &cache.content
                && slot.wants_redraw()
            {
                return true;
            }
            cache
                .shader_uses
                .iter()
                .any(|use_| self.shaders.animated(use_.shader))
        })
    }

    /// Composes and renders one surface's scene into its target, then
    /// presents a window surface.
    fn render_surface(
        &mut self,
        frame: &SurfaceFrame<'_>,
        stats: &mut FrameStats,
        wants_next: &mut bool,
        now: Instant,
    ) -> Result<(), RenderError> {
        let Some(mut surf) = self.surfaces.remove(&frame.id) else {
            return Ok(());
        };
        let result = self.render_surface_inner(&mut surf, frame, stats, wants_next, now);
        self.surfaces.insert(frame.id, surf);
        result
    }

    /// The body of `render_surface`, with the surface taken out of the map
    /// so `self` is free for `compose`.
    fn render_surface_inner(
        &mut self,
        surf: &mut SurfaceState,
        frame: &SurfaceFrame<'_>,
        stats: &mut FrameStats,
        wants_next: &mut bool,
        now: Instant,
    ) -> Result<(), RenderError> {
        let mut scene = vello::Scene::new();
        let size = surf.size;
        self.compose(
            &mut surf.layers,
            frame.tree.root(),
            frame.tree,
            cherenkov::kurbo::Affine::IDENTITY,
            size,
            &mut scene,
            stats,
            wants_next,
            now,
        )?;
        self.vello
            .render_to_texture(
                &self.device,
                &self.queue,
                &scene,
                surf.target.view(),
                &vello::RenderParams {
                    base_color: convert::color(&frame.clear),
                    width: size.0,
                    height: size.1,
                    antialiasing_method: AaConfig::Area,
                },
            )
            .map_err(|e| RenderError::Render(format!("vello render: {e}")))?;
        stats.passes += 1;
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

/// Lowers an [`ImageUpload`] into a `peniko` image. Only `Rgba8` is
/// drawable (the [`Uploads`](cherenkov::Uploads) capability gates the API;
/// defensive for direct calls).
fn peniko_image(image: &ImageUpload) -> Result<peniko::ImageData, ResourceError> {
    if image.format != cherenkov::ImageFormat::Rgba8 {
        return Err(ResourceError::Image(format!(
            "unsupported image format {:?}",
            image.format
        )));
    }
    Ok(peniko::ImageData {
        data: peniko::Blob::new(Arc::new(SharedBytes(image.data.clone()))),
        format: peniko::ImageFormat::Rgba8,
        alpha_type: if image.premultiplied {
            peniko::ImageAlphaType::AlphaPremultiplied
        } else {
            peniko::ImageAlphaType::Alpha
        },
        width: image.width,
        height: image.height,
    })
}

/// Validates font data with `skrifa`, rejecting unparseable data and
/// out-of-range face indices.
///
/// Colour fonts (`COLR`, `CBDT`/`CBLC` or `sbix` outlines) are accepted —
/// vello rasterizes colour glyphs through `skrifa`.
fn validate_font(data: &[u8], index: u32) -> Result<(), ResourceError> {
    skrifa::FontRef::from_index(data, index).map_err(|e| ResourceError::Font(format!("{e}")))?;
    Ok(())
}

impl Renderer for VelloRenderer {
    type Target = VelloTarget;

    fn create_surface(
        &mut self,
        id: SurfaceId,
        target: Self::Target,
    ) -> Result<SurfaceInfo, SurfaceError> {
        let (target, size, readable) = match target {
            VelloTarget::Offscreen(offscreen) => {
                // The vello target is `Rgba8Unorm`; `LinearF16` is the
                // offscreen contract's sRGB-equivalent choice here.
                if offscreen.format != OffscreenFormat::LinearF16 {
                    return Err(SurfaceError::UnsupportedFormat(offscreen.format));
                }
                let (texture, view) = create_target(
                    &self.device,
                    "surface target",
                    offscreen.size,
                    TARGET_USAGES,
                );
                (
                    TargetState::Offscreen { texture, view },
                    offscreen.size,
                    true,
                )
            }
            VelloTarget::Window(window) => {
                let interop::wgpu::Window { surface, config } = window;
                let size = (config.width, config.height);
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
                    size,
                    false,
                )
            }
        };
        if size.0 > self.max_texture || size.1 > self.max_texture {
            return Err(SurfaceError::TooLarge {
                width: size.0,
                height: size.1,
                max: self.max_texture,
            });
        }
        self.surfaces.insert(
            id,
            SurfaceState {
                size,
                target,
                readable,
                layers: HashMap::new(),
            },
        );
        Ok(SurfaceInfo { size, readable })
    }

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
            config.width = size.0;
            config.height = size.1;
            surface.configure(&self.device, config);
            *t = texture;
            *v = view;
        } else {
            state.target = TargetState::Offscreen { texture, view };
        }
        // Fragments cached against the old size are still valid (they are
        // recorded in user space), but clip-less group layers used the old
        // surface rect; rebuild everything to keep it simple.
        for cache in state.layers.values_mut() {
            cache.fragment = None;
        }
    }

    fn destroy_surface(&mut self, id: SurfaceId) {
        if let Some(mut state) = self.surfaces.remove(&id) {
            for (_, mut cache) in state.layers.drain() {
                Self::clear_content(&mut self.vello, &mut cache);
            }
        }
    }

    fn add_font(&mut self, id: FontId, font: FontData) -> Result<(), ResourceError> {
        validate_font(&font.data, font.index)?;
        self.fonts.insert(
            id.raw(),
            peniko::FontData::new(
                peniko::Blob::new(Arc::new(SharedBytes(font.data))),
                font.index,
            ),
        );
        Ok(())
    }

    fn remove_font(&mut self, id: FontId) {
        self.fonts.remove(&id.raw());
    }

    fn add_image(&mut self, id: ImageId, image: ImageUpload) -> Result<(), ResourceError> {
        self.images.insert(id.raw(), peniko_image(&image)?);
        Ok(())
    }

    fn remove_image(&mut self, id: ImageId) {
        self.images.remove(&id.raw());
    }

    fn set_content(&mut self, surface: SurfaceId, layer: LayerId, content: Option<ContentOp>) {
        let Some(state) = self.surfaces.get_mut(&surface) else {
            return;
        };
        let cache = state.layers.entry(layer).or_default();
        match content {
            Some(ContentOp::Replace(list)) => {
                Self::clear_content(&mut self.vello, cache);
                cache.content = Some(ContentData::List(list));
            }
            Some(ContentOp::Update(updates)) => {
                if let Some(ContentData::List(list)) = &mut cache.content {
                    let _ = list.apply(updates);
                    cache.fragment = None;
                }
            }
            Some(ContentOp::Picture(picture)) => {
                Self::clear_content(&mut self.vello, cache);
                cache.content = Some(ContentData::Picture(picture));
            }
            None => {
                Self::clear_content(&mut self.vello, cache);
            }
        }
    }

    fn remove_layer(&mut self, surface: SurfaceId, layer: LayerId) {
        let Some(state) = self.surfaces.get_mut(&surface) else {
            return;
        };
        if let Some(mut cache) = state.layers.remove(&layer) {
            Self::clear_content(&mut self.vello, &mut cache);
        }
    }

    /// Lowers and submits every dirty surface, bracketed by drained
    /// timestamp queries when enabled.
    fn render(
        &mut self,
        frame: &cherenkov::Frame<'_>,
        stats: &mut FrameStats,
    ) -> Result<Redraw, RenderError> {
        let mut wants_next = false;
        // A surface is dirty when the front end says its tree or content
        // changed, or a backend-side source (GPU content redraw flag,
        // animated shader, filter redraw callback) asks for a frame.
        let filter_redraw = self.filters.take_redraw_requests();
        if filter_redraw {
            wants_next = true;
        }
        let dirty: Vec<SurfaceId> = frame
            .surfaces
            .iter()
            .filter(|sf| {
                let Some(surface) = self.surfaces.get(&sf.id) else {
                    return false;
                };
                let backend_dirty = self.surface_needs_redraw(surface);
                wants_next |= backend_dirty;
                filter_redraw || sf.changed || backend_dirty
            })
            .map(|sf| sf.id)
            .collect();
        let now = Instant::now();
        let mut result = Ok(());
        if !dirty.is_empty() {
            self.drain_and_stamp(0)?;
            for sf in frame.surfaces {
                if !dirty.contains(&sf.id) {
                    continue;
                }
                result = self.render_surface(sf, stats, &mut wants_next, now);
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
        }
        Ok(if wants_next {
            Redraw::Wanted
        } else {
            Redraw::None
        })
    }

    /// Copies a surface's target into `Readback` pixels: each stored
    /// sRGB-encoded premultiplied `rgba8` is decoded per channel to linear
    /// sRGB and mapped into linear Display P3.
    fn readback(&mut self, surface: SurfaceId) -> Result<Readback, RenderError> {
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

    fn memory(&self) -> MemoryUsage {
        self.memory_usage()
    }

    fn trim(&mut self, pressure: Pressure) {
        if pressure == Pressure::Critical {
            for surface in self.surfaces.values_mut() {
                for cache in surface.layers.values_mut() {
                    cache.fragment = None;
                }
            }
        }
    }
}
