//! The render thread: sole owner of GPU state.

mod bindings;
mod bitmap;
mod colr;
pub mod diag;
mod external;
pub mod filter;
mod glyph;
mod gpu_content;
mod instance;
mod lower;
mod paint;
mod path;
mod prepared;
pub mod present;
mod raster;
pub mod shaders;
mod shadow;
mod upload;

use cherenkov::Instant;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{GpuConfig, GpuInfo, GpuTarget, ScratchFormat, TimestampSupport, names};
use bitmap::BitmapKey;
use cherenkov::{
    ContentOp, EngineError, FontData as EngineFontData, FontId, Frame, FrameId, FrameStats,
    FrameTiming, ImageId, ImageUpload, LayerId, MemoryUsage, PassTiming, Pressure, Readback,
    Redraw, RenderError, Renderer, ResourceError, SurfaceError, SurfaceFrame, SurfaceId,
    SurfaceInfo,
};
use glyph::{Atlas, FontData, PendingRaster};
use lower::{
    BackdropGroupInfo, ContentData, Frame as LoweredFrame, GlyphContext, Lowered, Lowering,
    PipelineKind, ShaderVariant, Source, Target,
};
use shaders::backdrop_effect_text;

/// The pipeline bound for a pass range: engine pipelines and the external
/// frame pipeline are mutually exclusive, so an engine range always rebinds
/// after an external one and vice versa.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    Engine(PipelineKind, ShaderVariant),
    External,
}

/// The surface target format: premultiplied linear Display P3.
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Amortize Metal's limited number of counter sample buffers across frames.
/// Each range remains exclusive until its frame's readback completes.
const TIMESTAMP_FRAMES_PER_SET: u32 = 64;

const TARGET_USAGES: wgpu::TextureUsages = wgpu::TextureUsages::from_bits_retain(
    wgpu::TextureUsages::RENDER_ATTACHMENT.bits()
        | wgpu::TextureUsages::COPY_SRC.bits()
        | wgpu::TextureUsages::COPY_DST.bits()
        | wgpu::TextureUsages::TEXTURE_BINDING.bits(),
);

/// Linear sRGB (BT.709 primaries, D65) to CIE XYZ — the oracle's
/// `SRGB_TO_XYZ`.
const SRGB_TO_XYZ: [[f64; 3]; 3] = [
    [
        0.412_390_799_265_959_4,
        0.357_584_339_383_878,
        0.180_480_788_401_834_3,
    ],
    [
        0.212_639_005_871_510_4,
        0.715_168_678_767_756,
        0.072_192_315_360_733_7,
    ],
    [
        0.019_330_818_715_591_8,
        0.119_194_779_410_625_9,
        0.950_532_152_249_660_5,
    ],
];

/// CIE XYZ to linear Display P3 (`P3_TO_XYZ` inverted), precomputed.
const XYZ_TO_P3: [[f64; 3]; 3] = [
    [
        2.493_496_911_941_425,
        -0.931_383_617_919_123_9,
        -0.402_710_784_450_716_2,
    ],
    [
        -0.829_488_969_561_574_7,
        1.762_664_060_318_226_3,
        0.023_624_685_848_943_6,
    ],
    [
        0.035_845_830_243_784_5,
        -0.076_172_389_268_041_4,
        0.956_884_524_007_687_1,
    ],
];

/// The `wgpu` format for a [`ScratchFormat`].
const fn scratch_wgpu(format: ScratchFormat) -> wgpu::TextureFormat {
    match format {
        ScratchFormat::LinearF16 => wgpu::TextureFormat::Rgba16Float,
        ScratchFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
    }
}

/// The pass-report spelling of a texture format.
const fn format_name(format: wgpu::TextureFormat) -> &'static str {
    match format {
        wgpu::TextureFormat::Rgba16Float => "rgba16float",
        wgpu::TextureFormat::Rgba8Unorm => "rgba8unorm",
        _ => "unknown",
    }
}

/// One isolation scratch or backdrop texture.
struct ScratchTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

/// A GPU-resident image registered with the engine.
pub struct GpuImage {
    /// The texture holding premultiplied linear-P3 f16 texels.
    pub texture: wgpu::Texture,
    /// Its view for bind group 1.
    pub view: wgpu::TextureView,
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
}

pub struct GpuBitmap {
    pub(super) image: GpuImage,
    pub(super) em: kurbo::Rect,
}

/// One surface's GPU-side state.
struct SurfaceState {
    window: Option<present::WindowSurface>,
    textures: Option<std::sync::mpsc::Sender<wgpu::Texture>>,
    refresh: cherenkov::RefreshRange,
    present_pending: bool,
    size: (u32, u32),
    /// The scratch texture format (set at creation).
    scratch_format: wgpu::TextureFormat,
    target: wgpu::Texture,
    view: wgpu::TextureView,
    /// Scratch textures, one per isolation depth, sized to the largest
    /// region seen so far.
    scratch: Vec<ScratchTarget>,
    /// Backdrop copies for blend composites: index 0 matches the surface
    /// format, index 1 the scratch format.
    backdrop: [Option<ScratchTarget>; 2],
    /// Backdrop groups registered on this surface by raw id.
    backdrop_groups: FxHashMap<u64, BackdropGroupState>,
    layers: FxHashMap<LayerId, ContentData>,
    content: FxHashMap<LayerId, gpu_content::Slot>,
    /// Retained external frames by layer (`cherenkov::ExternalFrames`).
    external: FxHashMap<LayerId, external::Slot>,
    shader_textures: FxHashMap<std::sync::Arc<paint::Key>, paint::Texture>,
    frame: LoweredFrame,
    /// This frame's offsets into the shared buffers: instances and globals
    /// (256-byte slots) are laid out surface by surface so one upload covers
    /// every dirty surface.
    inst_base: u32,
    globals_base: u32,
    /// Bumped whenever a scratch or backdrop texture is (re)created — a
    /// cached group-1 bind group referencing the old view must rebuild.
    bind_gen: u64,
    /// Group-1 bind groups keyed by `(source, backdrop, image,
    /// mask texture)`, reused across frames while `binds1_stamp` is
    /// current.
    binds1: FxHashMap<Bind1Key, wgpu::BindGroup>,
    /// The `(bind_gen, images_gen, mask_texture_gen)` triple `binds1` was
    /// built under.
    binds1_stamp: (u64, u64, u64),
}

/// A registered backdrop group: its optional capture filter and the
/// capture textures, one per region (#117 sparse capture), each exactly
/// sized to this frame's region.
struct BackdropGroupState {
    /// The group's filter chain key, when registered with a filter.
    filter: Option<filter::FilterKey>,
    /// The capture textures indexed by region, empty until a frame
    /// samples the group.
    captures: Vec<ScratchTarget>,
}

impl SurfaceState {
    /// The `BackdropGroupInfo` map lowering needs for this surface.
    fn backdrop_info(&self, filters: &mut filter::Registry) -> FxHashMap<u64, BackdropGroupInfo> {
        self.backdrop_groups
            .iter()
            .map(|(g, state)| {
                (
                    *g,
                    BackdropGroupInfo {
                        filter: state.filter,
                        footprint: state
                            .filter
                            .map_or(Some(filtrate_core::Footprint::ZERO), |key| {
                                filters.footprint_bound(key)
                            }),
                    },
                )
            })
            .collect()
    }

    /// Bytes held by this surface's backdrop captures, summed over all
    /// regions of all groups.
    fn backdrop_bytes(&self) -> u64 {
        self.backdrop_groups
            .values()
            .flat_map(|g| &g.captures)
            .map(|c| {
                u64::from(c.width)
                    * u64::from(c.height)
                    * if c.texture.format() == wgpu::TextureFormat::Rgba16Float {
                        8
                    } else {
                        4
                    }
            })
            .sum()
    }

    fn content_wants_redraw(&self) -> bool {
        !self.content.is_empty()
            && self.frame.content.iter().any(|id| {
                self.content
                    .get(id)
                    .is_some_and(gpu_content::Slot::wants_redraw)
            })
    }

    /// Bytes held by this surface's textures.
    fn gpu_bytes(&self) -> u64 {
        let surface_bytes = u64::from(self.size.0) * u64::from(self.size.1) * 8;
        let scratch_texel = if self.scratch_format == wgpu::TextureFormat::Rgba8Unorm {
            4
        } else {
            8
        };
        let scratch_bytes: u64 = self
            .scratch
            .iter()
            .map(|s| u64::from(s.width) * u64::from(s.height) * scratch_texel)
            .sum();
        let backdrop_bytes: u64 = self
            .backdrop
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                // Index 0 mirrors the surface format (f16), index 1 the
                // scratch format.
                let texel = if i == 0 { 8 } else { scratch_texel };
                b.as_ref()
                    .map(|b| u64::from(b.width) * u64::from(b.height) * texel)
            })
            .sum();
        let shader_bytes = self
            .shader_textures
            .values()
            .map(|texture| {
                u64::from(texture.image.width) * u64::from(texture.image.height) * 8 + 272
            })
            .sum::<u64>();
        surface_bytes
            + scratch_bytes
            + backdrop_bytes
            + shader_bytes
            + self
                .content
                .values()
                .filter_map(|slot| slot.image.as_ref())
                .map(|image| u64::from(image.width) * u64::from(image.height) * 8)
                .sum::<u64>()
            + self
                .external
                .values()
                .map(|_| external::Slot::GPU_BYTES)
                .sum::<u64>()
            + self.backdrop_bytes()
    }
}

/// The group-1 bind group key: `(source, backdrop-needed,
/// image, mask texture)`.
type Bind1Key = (
    Option<lower::Source>,
    bool,
    Option<lower::ImageSource>,
    Option<u64>,
);

/// Drops the group-1 bind groups `reject` marks obsolete — unsubmitted
/// groups whose views keep a replaced texture's predecessor alive
/// (#169 A4) — and refreshes `binds1_stamp` so the survivors persist
/// past the next encode. Bind groups already consumed by a submitted
/// encoder are wgpu's to retain until that submission completes.
fn retire_binds1(
    surf: &mut SurfaceState,
    device: &wgpu::Device,
    images_gen: u64,
    mask_gen: u64,
    reason: &'static str,
    reject: impl Fn(&Bind1Key) -> bool,
) {
    let before = surf.binds1.len();
    surf.binds1.retain(|key, _| !reject(key));
    let dropped = before - surf.binds1.len();
    if dropped > 0 {
        diag::bind_groups_dropped(device, dropped as u64, reason);
    }
    surf.binds1_stamp = (surf.bind_gen, images_gen, mask_gen);
}

/// All render-thread state.
pub struct GpuRenderer {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    presenter: Option<present::Presenter>,
    /// How the fixed modules reach this device (`shaders.rs`): SPIR-V,
    /// metallib, or WGSL — decided once at init by the adapter backend.
    shader_delivery: shaders::ShaderDelivery,
    shaders: paint::Registry,
    filters: filter::Registry,
    shadow_blur: shadow::Blur,
    last_frame: Option<Instant>,
    origin: Option<Instant>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// `[format index][pipeline kind][shader variant]`:
    /// format 0 = surface, 1 = scratch; kind 0 = source-over, 1 = replace;
    /// variant 0/1/2 = simple/shadow/full fragment shader.
    pipelines: [[[wgpu::RenderPipeline; 3]; 2]; 2],
    /// Registered backdrop effect shaders, by raw id, one Full-variant
    /// `SrcOver` pipeline per target format (`[surface, scratch]`),
    /// compiled at registration.
    backdrop_shaders: FxHashMap<u64, [wgpu::RenderPipeline; 2]>,
    /// The configured isolation texture format.
    scratch_format: wgpu::TextureFormat,
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
    /// The globals buffer size `bind0` was built against.
    bound_globals_size: u64,
    atlas: Atlas,
    /// Per-commit cell writes, rebuilt in place each frame.
    commit_writes: Vec<glyph::CellWrite>,
    /// Per-surface pending origins, rebuilt in place each apply.
    pending_origins: Vec<PendingOrigin>,
    /// A dummy 1×1 view for unused group-1 slots.
    dummy_view: wgpu::TextureView,
    /// A dummy 1×1 `u32` view for unused external-frame plane slots.
    dummy_uint_view: wgpu::TextureView,
    /// The external-frame group-1 layout, `None` until a surface first
    /// draws an external frame.
    ext_layout: Option<wgpu::BindGroupLayout>,
    /// `[format index]` external pipelines: 0 = surface, 1 = scratch —
    /// `None` until the first external draw prepares them.
    external_pipes: [Option<wgpu::RenderPipeline>; 2],
    surfaces: FxHashMap<SurfaceId, SurfaceState>,
    fonts: FxHashMap<u64, FontData>,
    /// Registered images.
    images: FxHashMap<u64, GpuImage>,
    bitmaps: FxHashMap<BitmapKey, GpuBitmap>,
    /// Bumped on every `images` insert/remove — every cached group-1
    /// bind group samples an image view, so an image change rebuilds them.
    images_gen: u64,
    timestamps: bool,
    query_set: Option<wgpu::QuerySet>,
    /// First query of the active frame's independent range.
    query_base: u32,
    /// Free frame ranges: (set, first query, capacity). Sharing a set keeps
    /// many frames in flight without exhausting Metal's sample-buffer limit.
    query_pool: Vec<(wgpu::QuerySet, u32, u32)>,
    /// Frames still writing their pass-boundary samples on the GPU.
    pending_queries: VecDeque<PendingQueries>,
    /// Last draw submission of the current frame.
    frame_submission: Option<wgpu::SubmissionIndex>,
    /// Staging ring for the frame-wide instance, stop and globals uploads.
    uploads: upload::Uploads,
    query_buffer: Option<wgpu::Buffer>,
    /// Readback buffers recycled between frames; a frame's resolve owns
    /// one until its samples are read.
    query_staging: Vec<wgpu::Buffer>,
    /// Capacity of one frame's query range; pass `i` writes `base + 2i`
    /// at its start and `base + 2i + 1` at its end. GPU time runs from the
    /// first pass's start to the last pass's end: pass boundaries are the
    /// one timestamp position every backend with `TIMESTAMP_QUERY`
    /// supports (Metal on Apple GPUs samples only at stage boundaries).
    query_capacity: u32,
    /// Frames whose timestamp resolve was submitted but whose staging
    /// buffer is not mapped yet — read on a later call, in order.
    pending_timestamps: VecDeque<PendingTimestamps>,
    /// Resolved timings retained until `finish_timings`.
    timings: Vec<FrameTiming>,
    /// Passes encoded this frame, for the per-pass report.
    frame_pass_count: u32,
    /// `(name, width, height, format)` of each encoded pass this frame.
    pass_meta: Vec<PassMeta>,
    /// Bound on every GPU wait; see [`GpuConfig::wait_timeout`].
    wait_timeout: Duration,
    max_texture: u32,
    /// The allocation-event diagnostic sink (issue #169); `None` in
    /// timed runs.
    diag: Option<diag::Sink>,
    /// Kept so registered-effect pipelines use the same pipeline cache.
    config: GpuConfig,
}

/// Atlas origins produced by one deferred raster.
/// The batch commit's outcome for `lower_all`'s retry loop (#169 A3).
enum Commit {
    /// Every surface's pending rasters committed; a surface that could
    /// not be placed got `AtlasExhausted` in its result.
    Done,
    /// Grow the atlas once to this edge and re-lower.
    Grow(u32),
}

enum PendingOrigin {
    /// Cell origins — one for a glyph, one per cell for a path
    /// emission — and the shelves the admission's cells live on, so
    /// patched emissions can reference the bands (#119).
    Cells(Vec<(u32, u32)>, Vec<u32>),
    /// A clip mask's cell origin.
    Mask([f32; 2]),
    /// A COLR cache insert; no instance patch.
    None,
}

/// A frame owns its query range until its samples have been resolved and read.
/// Resolving is encoded only after the draw completion callback fires: on
/// newer Apple GPUs an earlier Metal blit resolve can see incomplete end
/// samples even with an explicit GPU fence or event.
struct PendingQueries {
    frame: FrameId,
    submission: wgpu::SubmissionIndex,
    query_set: wgpu::QuerySet,
    base: u32,
    capacity: u32,
    count: u32,
    meta: Vec<PassMeta>,
    complete: Arc<AtomicU8>,
}

/// One submitted frame's timestamp queries awaiting GPU completion.
///
/// The resolve and copy are submitted after the frame's draws complete.
/// The map and read happen on a later non-blocking poll, so rendering never
/// stalls on GPU idle. The query range cannot be reused until this readback
/// completes.
struct PendingTimestamps {
    /// Samples remain owned by this frame until the resolve copy completes.
    query_set: wgpu::QuerySet,
    query_base: u32,
    query_capacity: u32,
    /// The frame these queries measure.
    frame: FrameId,
    /// The submission carrying the resolve and the copy into `staging`.
    submission: wgpu::SubmissionIndex,
    staging: wgpu::Buffer,
    /// Queries resolved: `2 * passes`.
    count: u32,
    /// Pass metadata for the per-pass report.
    meta: Vec<PassMeta>,
    /// Whether `map_async` was requested for `staging`.
    map_requested: bool,
    /// Set by the map callback: 1 once the copy is readable, 2 on a
    /// failed map.
    ready: Arc<AtomicU8>,
}

impl PendingTimestamps {
    /// Requests the map of `staging`; the callback records the outcome
    /// in `ready`.
    fn request_map(&mut self) {
        if self.map_requested {
            return;
        }
        let flag = Arc::clone(&self.ready);
        self.staging.slice(..u64::from(self.count) * 8).map_async(
            wgpu::MapMode::Read,
            move |result| {
                flag.store(u8::from(result.is_err()) + 1, Ordering::Relaxed);
            },
        );
        self.map_requested = true;
    }
}

/// One encoded pass's report metadata.
struct PassMeta {
    name: String,
    width: u32,
    height: u32,
    format: &'static str,
}

/// Recreates a frame-wide buffer at `size` bytes. Its contents need not
/// survive: every dirty surface's slice is copied in from the upload ring
/// at the start of the frame's first submission, after all growth.
fn grow_buffer(
    device: &wgpu::Device,
    label: &'static str,
    old: &wgpu::Buffer,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    tracing::debug!(label, from = old.size(), to = size, "buffer grown");
    let new = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    });
    diag::grow(
        device,
        label,
        diag::Class::for_label(label),
        old.size(),
        size,
        0,
        false,
    );
    new
}

/// Creates an adapter plus device. Fails when no adapter allows the target
/// format's required usages.
#[cfg(not(target_arch = "wasm32"))]
fn create_device(
    config: &GpuConfig,
) -> Result<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue), EngineError> {
    if let Some(shared) = &config.device {
        return Ok((
            shared.instance.clone(),
            shared.adapter.clone(),
            shared.device.clone(),
            shared.queue.clone(),
        ));
    }
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
        .ok_or_else(|| EngineError::Backend("no adapter".into()))?;
    let supported = adapter.features();
    let info = adapter.get_info();
    tracing::info!(
        name = %info.name,
        backend = ?info.backend,
        device_type = ?info.device_type,
        driver = %info.driver,
        driver_info = %info.driver_info,
        timestamp_query = supported.contains(wgpu::Features::TIMESTAMP_QUERY),
        timestamps_inside_encoders =
            supported.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
        timestamps_inside_passes = supported.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
        "adapter"
    );
    tracing::debug!(limits = ?adapter.limits(), "adapter limits");
    let mut required = wgpu::Features::empty();
    // The fixed engine shaders are precompiled (#57): Vulkan loads SPIR-V
    // and Metal a metallib through the passthrough API; wgpu-hal advertises
    // the feature unconditionally on both backends.
    if matches!(info.backend, wgpu::Backend::Vulkan | wgpu::Backend::Metal)
        && supported.contains(wgpu::Features::PASSTHROUGH_SHADERS)
    {
        required |= wgpu::Features::PASSTHROUGH_SHADERS;
    }
    // Only pass-boundary timestamps are requested: Metal on Apple GPUs
    // advertises `TIMESTAMP_QUERY_INSIDE_ENCODERS` but samples only at
    // stage boundaries, so an encoder-level `write_timestamp` goes through
    // a dummy blit encoder that wgpu itself documents as unreliable.
    if config.timestamps && supported.contains(wgpu::Features::TIMESTAMP_QUERY) {
        required |= wgpu::Features::TIMESTAMP_QUERY;
    }
    if config.pipeline_cache.is_some() && supported.contains(wgpu::Features::PIPELINE_CACHE) {
        required |= wgpu::Features::PIPELINE_CACHE;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cherenkov-gpu"),
        required_features: required,
        // Clamp the portable defaults to what the adapter reports:
        // iOS Metal offers 15 inter-stage varyings (60 components)
        // where `Limits::default` asks for 16.
        required_limits: wgpu::Limits::default().or_worse_values_from(&adapter.limits()),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| EngineError::Backend(format!("{e}")))?;
    tracing::info!(features = ?device.features(), "device");
    Ok((instance, adapter, device, queue))
}

#[cfg(not(target_arch = "wasm32"))]
impl crate::interop::SharedDevice {
    /// Creates the adapter and device the GPU engine would use for `config`.
    ///
    /// Pass the result back through [`GpuConfig::device`] to drive the engine
    /// with this exact device.
    ///
    /// # Errors
    /// [`EngineError`] when no adapter allows the target format or device
    /// creation fails.
    pub fn create(config: &GpuConfig) -> Result<Self, EngineError> {
        let (instance, adapter, device, queue) = create_device(config)?;
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}

#[cfg(target_arch = "wasm32")]
#[expect(
    clippy::future_not_send,
    reason = "the browser engine is single-threaded and its futures run on the page's event loop"
)]
async fn create_device(
    config: &GpuConfig,
) -> Result<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue), EngineError> {
    if let Some(shared) = &config.device {
        return Ok((
            shared.instance.clone(),
            shared.adapter.clone(),
            shared.device.clone(),
            shared.queue.clone(),
        ));
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: config.backends,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: config.power_preference,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
            compatible_surface: None,
        })
        .await
        .map_err(|error| EngineError::Backend(error.to_string()))?;
    if !adapter
        .get_texture_format_features(TARGET_FORMAT)
        .allowed_usages
        .contains(TARGET_USAGES)
    {
        return Err(EngineError::Backend(
            "adapter cannot render the working-space format".into(),
        ));
    }
    let supported = adapter.features();
    let info = adapter.get_info();
    tracing::info!(
        name = %info.name,
        backend = ?info.backend,
        device_type = ?info.device_type,
        driver = %info.driver,
        driver_info = %info.driver_info,
        timestamp_query = supported.contains(wgpu::Features::TIMESTAMP_QUERY),
        timestamps_inside_encoders =
            supported.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
        timestamps_inside_passes = supported.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
        "adapter"
    );
    tracing::debug!(limits = ?adapter.limits(), "adapter limits");
    let mut required = wgpu::Features::empty();
    // Only pass-boundary timestamps are requested: Metal on Apple GPUs
    // advertises `TIMESTAMP_QUERY_INSIDE_ENCODERS` but samples only at
    // stage boundaries, so an encoder-level `write_timestamp` goes through
    // a dummy blit encoder that wgpu itself documents as unreliable.
    if config.timestamps && supported.contains(wgpu::Features::TIMESTAMP_QUERY) {
        required |= wgpu::Features::TIMESTAMP_QUERY;
    }
    if config.pipeline_cache.is_some() && supported.contains(wgpu::Features::PIPELINE_CACHE) {
        required |= wgpu::Features::PIPELINE_CACHE;
    }
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("cherenkov-gpu"),
            required_features: required,
            // Clamp the portable defaults to what the adapter reports:
            // iOS Metal offers 15 inter-stage varyings (60 components)
            // where `Limits::default` asks for 16.
            required_limits: wgpu::Limits::default().or_worse_values_from(&adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| EngineError::Backend(format!("{e}")))?;
    tracing::info!(features = ?device.features(), "device");
    Ok((instance, adapter, device, queue))
}

/// Maps a layout-table entry to the wgpu descriptor. The same table feeds
/// `build.rs`'s slot assignment, so a precompiled shader cannot drift from
/// the layout built here.
pub fn layout_entries(entries: &[bindings::Entry]) -> Vec<wgpu::BindGroupLayoutEntry> {
    entries
        .iter()
        .map(|entry| wgpu::BindGroupLayoutEntry {
            binding: entry.binding,
            visibility: wgpu::ShaderStages::from_bits_retain(u32::from(entry.stages)),
            ty: match entry.kind {
                bindings::Kind::Uniform => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: entry.dynamic_offset,
                    min_binding_size: wgpu::BufferSize::new(entry.min_size),
                },
                bindings::Kind::StorageRead => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: entry.dynamic_offset,
                    min_binding_size: wgpu::BufferSize::new(entry.min_size),
                },
                bindings::Kind::Texture => wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                bindings::Kind::TextureUint => wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                bindings::Kind::Sampler => {
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
                }
            },
            count: None,
        })
        .collect()
}

/// The bind group layouts: group 0 is engine data, group 1 the texture a
/// composite samples.
fn create_layouts(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    let layout0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("engine data"),
        entries: &layout_entries(bindings::ENGINE_GROUP0),
    });
    let layout1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("source texture"),
        // 0: composite source, 1: blend backdrop, 2: image paint,
        // 3: clip mask texture.
        entries: &layout_entries(bindings::ENGINE_GROUP1),
    });
    (layout0, layout1)
}

/// Builds a group-1 bind group; `None` binds the dummy view.
fn make_bind1(
    device: &wgpu::Device,
    layout1: &wgpu::BindGroupLayout,
    dummy: &wgpu::TextureView,
    source: Option<&wgpu::TextureView>,
    backdrop: Option<&wgpu::TextureView>,
    image: Option<&wgpu::TextureView>,
    mask: Option<&wgpu::TextureView>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("source texture"),
        layout: layout1,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source.unwrap_or(dummy)),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(backdrop.unwrap_or(dummy)),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(image.unwrap_or(dummy)),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(mask.unwrap_or(dummy)),
            },
        ],
    })
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
                // One 24-byte Globals window; the dynamic offset selects
                // the pass's slot inside the buffer.
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: globals,
                    offset: 0,
                    size: wgpu::BufferSize::new(24),
                }),
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

/// The closed pipeline set: instanced-quad pipelines from `shader.wgsl`,
/// specialised per fragment variant.
const fn variant_index(variant: ShaderVariant) -> usize {
    match variant {
        ShaderVariant::Simple => 0,
        ShaderVariant::Shadow => 1,
        ShaderVariant::Full => 2,
    }
}

/// The closed pipeline set: one instanced-quad pipeline from `shader.wgsl`.
/// When a pipeline cache path is configured and supported, the cache is
/// loaded beforehand and persisted afterwards, best effort.
#[cfg(not(target_arch = "wasm32"))]
fn create_pipeline(
    device: &wgpu::Device,
    config: &GpuConfig,
    layout0: &wgpu::BindGroupLayout,
    layout1: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
    replace: bool,
) -> Result<wgpu::RenderPipeline, EngineError> {
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
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
    // Source-over premultiplied compositing, or `Replace` writing the
    // shader's already-composited result verbatim.
    let component = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: if replace {
            wgpu::BlendFactor::Zero
        } else {
            wgpu::BlendFactor::OneMinusSrcAlpha
        },
        operation: wgpu::BlendOperation::Add,
    };
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cherenkov"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
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
        return Err(EngineError::Backend(format!("{error}")));
    }
    if let (Some(cache), Some(path)) = (&cache, &config.pipeline_cache)
        && let Some(data) = cache.get_data()
    {
        let _ = std::fs::write(path, data);
    }
    Ok(pipeline)
}

#[cfg(target_arch = "wasm32")]
#[expect(
    clippy::future_not_send,
    reason = "the browser engine is single-threaded and its futures run on the page's event loop"
)]
async fn create_pipeline(
    device: &wgpu::Device,
    config: &GpuConfig,
    layout0: &wgpu::BindGroupLayout,
    layout1: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
    replace: bool,
) -> Result<wgpu::RenderPipeline, EngineError> {
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
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
    // Source-over premultiplied compositing, or `Replace` writing the
    // shader's already-composited result verbatim.
    let component = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: if replace {
            wgpu::BlendFactor::Zero
        } else {
            wgpu::BlendFactor::OneMinusSrcAlpha
        },
        operation: wgpu::BlendOperation::Add,
    };
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cherenkov"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
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
    if let Some(error) = error_scope.pop().await {
        return Err(EngineError::Backend(format!("{error}")));
    }
    if let (Some(cache), Some(path)) = (&cache, &config.pipeline_cache)
        && let Some(data) = cache.get_data()
    {
        let _ = std::fs::write(path, data);
    }
    Ok(pipeline)
}

/// One instanced-quad pipeline from `external.wgsl` for `format`:
/// source-over only — an external frame composites like any image.
fn create_external_pipeline(
    device: &wgpu::Device,
    layout0: &wgpu::BindGroupLayout,
    layout1: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cherenkov external"),
        bind_group_layouts: &[Some(layout0), Some(layout1)],
        immediate_size: 0,
    });
    let component = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cherenkov external"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs_external"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
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
        cache: None,
    })
}

/// A `w` × `h` texture in `format`.
fn create_target(
    device: &wgpu::Device,
    label: &'static str,
    size: (u32, u32),
    usages: wgpu::TextureUsages,
    format: wgpu::TextureFormat,
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
        format,
        usage: usages,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Byte size of a `w` × `h` uncompressed texture in `format` — what the
/// diagnostic tracks for target allocations.
pub fn texel_bytes(format: wgpu::TextureFormat) -> u64 {
    u64::from(format.block_copy_size(None).unwrap_or(0))
}

/// Creates GPU state on the shared engine render thread.
///
/// # Errors
/// Returns initialization and pipeline validation errors from the backend.
#[expect(clippy::too_many_lines, reason = "moved into the render thread")]
#[cfg(not(target_arch = "wasm32"))]
pub fn init(config: GpuConfig) -> Result<(GpuRenderer, GpuInfo), EngineError> {
    let _diag_guard = diag::Guard::scope(config.alloc_diag.as_ref());
    create_device(&config).and_then(|(instance, adapter, device, queue)| {
        let info = adapter.get_info();
        let supported = adapter.features();
        let timestamp_support = if supported.contains(wgpu::Features::TIMESTAMP_QUERY) {
            TimestampSupport::PassBoundaries
        } else {
            TimestampSupport::Unsupported
        };
        let (layout0, layout1) = create_layouts(&device);
        let scratch_format = scratch_wgpu(config.scratch_format);
        // Three specialised fragment shaders from one source file, compiled
        // at build time: the prepended `VARIANT` constant makes fs_main a
        // constant-folded dispatch to fs_simple/fs_shadow/fs_full. On Vulkan
        // and Metal these are the embedded passthrough binaries (#57).
        let shader_delivery = shaders::delivery(info.backend, &device)?;
        let modules = [0usize, 1, 2].map(|v| shader_delivery.engine_module(&device, v));
        let pipelines = |format: wgpu::TextureFormat, replace: bool| {
            Ok::<_, EngineError>([
                create_pipeline(
                    &device,
                    &config,
                    &layout0,
                    &layout1,
                    &modules[0],
                    format,
                    replace,
                )?,
                create_pipeline(
                    &device,
                    &config,
                    &layout0,
                    &layout1,
                    &modules[1],
                    format,
                    replace,
                )?,
                create_pipeline(
                    &device,
                    &config,
                    &layout0,
                    &layout1,
                    &modules[2],
                    format,
                    replace,
                )?,
            ])
        };
        let pipelines = [
            [
                pipelines(TARGET_FORMAT, false)?,
                pipelines(TARGET_FORMAT, true)?,
            ],
            [
                pipelines(scratch_format, false)?,
                pipelines(scratch_format, true)?,
            ],
        ];
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            // One 256-byte stride slot: a single pass's Globals entry.
            size: 256,
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
        diag::create(&device, "globals", globals.size());
        diag::create(&device, "instances", instances.size());
        diag::create(&device, "stops", stops.size());
        let atlas = Atlas::new(&device, config.budget.gpu.0);
        let bind0 = make_bind0(&device, &layout0, &globals, &instances, &stops, &atlas);
        let (_, dummy_view) = create_target(
            &device,
            "dummy source",
            (1, 1),
            wgpu::TextureUsages::TEXTURE_BINDING,
            TARGET_FORMAT,
        );
        diag::create(&device, "dummy source", 8);
        let (_, dummy_uint_view) = create_target(
            &device,
            "dummy uint source",
            (1, 1),
            wgpu::TextureUsages::TEXTURE_BINDING,
            wgpu::TextureFormat::R8Uint,
        );
        diag::create(&device, "dummy uint source", 1);
        let timestamps =
            config.timestamps && device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let (query_set, query_buffer) = if timestamps {
            (
                Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("frame timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 2 * TIMESTAMP_FRAMES_PER_SET,
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
        if let Some(buffer) = &query_buffer {
            diag::create(&device, "frame timestamps", 0);
            diag::create(&device, "timestamp resolve", buffer.size());
        }
        // Two queries per frame, reserving 64 independent frame ranges.
        let query_capacity = if query_set.is_some() { 2 } else { 0 };
        let query_pool = query_set.as_ref().map_or_else(Vec::new, |set| {
            (1..TIMESTAMP_FRAMES_PER_SET)
                .map(|slot| (set.clone(), slot * 2, 2))
                .collect()
        });
        let renderer = GpuRenderer {
            instance,
            adapter,
            presenter: None,
            shader_delivery,
            shaders: paint::Registry::default(),
            backdrop_shaders: FxHashMap::default(),
            filters: filter::Registry::new(config.redraw.clone()),
            shadow_blur: shadow::Blur::new(&device, scratch_format),
            last_frame: None,
            origin: None,
            max_texture: device.limits().max_texture_dimension_2d,
            device,
            queue,
            pipelines,
            scratch_format,
            layout0,
            layout1,
            globals,
            instances,
            stops,
            bind0,
            dummy_view,
            dummy_uint_view,
            ext_layout: None,
            external_pipes: [None, None],
            bound_atlas: 0,
            bound_instance_size: 272 * 16,
            bound_stop_size: 32 * 16,
            bound_globals_size: 256,
            atlas,
            commit_writes: Vec::new(),
            pending_origins: Vec::new(),
            surfaces: FxHashMap::default(),
            fonts: FxHashMap::default(),
            images: FxHashMap::default(),
            bitmaps: FxHashMap::default(),
            images_gen: 0,
            timestamps,
            query_set,
            query_base: 0,
            query_pool,
            pending_queries: VecDeque::new(),
            frame_submission: None,
            uploads: upload::Uploads::default(),
            query_buffer,
            query_staging: Vec::new(),
            query_capacity,
            pending_timestamps: VecDeque::new(),
            timings: Vec::new(),
            frame_pass_count: 0,
            pass_meta: Vec::new(),
            wait_timeout: config.wait_timeout,
            diag: config.alloc_diag.clone(),
            config,
        };
        Ok((
            renderer,
            GpuInfo {
                name: info.name,
                backend: format!("{:?}", info.backend),
                vendor: info.vendor,
                device: info.device,
                device_type: format!("{:?}", info.device_type),
                driver: info.driver,
                driver_info: info.driver_info,
                timestamps: timestamp_support,
            },
        ))
    })
}

#[cfg(target_arch = "wasm32")]
#[expect(
    clippy::future_not_send,
    reason = "the browser engine is single-threaded and its futures run on the page's event loop"
)]
#[expect(
    clippy::too_many_lines,
    reason = "adapter and device requests plus pipeline setup form one linear sequence"
)]
pub async fn init(config: GpuConfig) -> Result<(GpuRenderer, GpuInfo), EngineError> {
    let _diag_guard = diag::Guard::scope(config.alloc_diag.as_ref());
    let (instance, adapter, device, queue) = create_device(&config).await?;
    let info = adapter.get_info();
    let supported = adapter.features();
    let timestamp_support = if supported.contains(wgpu::Features::TIMESTAMP_QUERY) {
        TimestampSupport::PassBoundaries
    } else {
        TimestampSupport::Unsupported
    };
    let (layout0, layout1) = create_layouts(&device);
    let scratch_format = scratch_wgpu(config.scratch_format);
    // Three specialised fragment shaders from one source file: the
    // prepended `VARIANT` constant makes fs_main a constant-folded
    // dispatch to fs_simple/fs_shadow/fs_full. WebGPU keeps WGSL (#57).
    let shader_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader_delivery = shaders::delivery(info.backend, &device)?;
    let modules = [0usize, 1, 2].map(|v| shader_delivery.engine_module(&device, v));
    if let Some(error) = shader_scope.pop().await {
        return Err(EngineError::Backend(error.to_string()));
    }
    let pipelines = async |format: wgpu::TextureFormat, replace: bool| {
        Ok::<_, EngineError>([
            create_pipeline(
                &device,
                &config,
                &layout0,
                &layout1,
                &modules[0],
                format,
                replace,
            )
            .await?,
            create_pipeline(
                &device,
                &config,
                &layout0,
                &layout1,
                &modules[1],
                format,
                replace,
            )
            .await?,
            create_pipeline(
                &device,
                &config,
                &layout0,
                &layout1,
                &modules[2],
                format,
                replace,
            )
            .await?,
        ])
    };
    let pipelines = [
        [
            pipelines(TARGET_FORMAT, false).await?,
            pipelines(TARGET_FORMAT, true).await?,
        ],
        [
            pipelines(scratch_format, false).await?,
            pipelines(scratch_format, true).await?,
        ],
    ];
    let globals = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("globals"),
        // One 256-byte stride slot: a single pass's Globals entry.
        size: 256,
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
    diag::create(&device, "globals", globals.size());
    diag::create(&device, "instances", instances.size());
    diag::create(&device, "stops", stops.size());
    let atlas = Atlas::new(&device, config.budget.gpu.0);
    let bind0 = make_bind0(&device, &layout0, &globals, &instances, &stops, &atlas);
    let (_, dummy_view) = create_target(
        &device,
        "dummy source",
        (1, 1),
        wgpu::TextureUsages::TEXTURE_BINDING,
        TARGET_FORMAT,
    );
    diag::create(&device, "dummy source", 8);
    let (_, dummy_uint_view) = create_target(
        &device,
        "dummy uint source",
        (1, 1),
        wgpu::TextureUsages::TEXTURE_BINDING,
        wgpu::TextureFormat::R8Uint,
    );
    diag::create(&device, "dummy uint source", 1);
    let timestamps =
        config.timestamps && device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let (query_set, query_buffer) = if timestamps {
        (
            Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("frame timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2 * TIMESTAMP_FRAMES_PER_SET,
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
    if let Some(buffer) = &query_buffer {
        diag::create(&device, "frame timestamps", 0);
        diag::create(&device, "timestamp resolve", buffer.size());
    }
    // Two queries per frame, reserving 64 independent frame ranges.
    let query_capacity = if query_set.is_some() { 2 } else { 0 };
    let query_pool = query_set.as_ref().map_or_else(Vec::new, |set| {
        (1..TIMESTAMP_FRAMES_PER_SET)
            .map(|slot| (set.clone(), slot * 2, 2))
            .collect()
    });
    let renderer = GpuRenderer {
        instance,
        adapter,
        presenter: None,
        shader_delivery,
        shaders: paint::Registry::default(),
        backdrop_shaders: FxHashMap::default(),
        filters: filter::Registry::new(config.redraw.clone()),
        shadow_blur: shadow::Blur::new(&device, scratch_format),
        last_frame: None,
        origin: None,
        max_texture: device.limits().max_texture_dimension_2d,
        device,
        queue,
        pipelines,
        scratch_format,
        layout0,
        layout1,
        globals,
        instances,
        stops,
        bind0,
        dummy_view,
        dummy_uint_view,
        ext_layout: None,
        external_pipes: [None, None],
        bound_atlas: 0,
        bound_instance_size: 272 * 16,
        bound_stop_size: 32 * 16,
        bound_globals_size: 256,
        atlas,
        commit_writes: Vec::new(),
        pending_origins: Vec::new(),
        surfaces: FxHashMap::default(),
        fonts: FxHashMap::default(),
        images: FxHashMap::default(),
        bitmaps: FxHashMap::default(),
        images_gen: 0,
        timestamps,
        query_set,
        query_base: 0,
        query_pool,
        pending_queries: VecDeque::new(),
        frame_submission: None,
        uploads: upload::Uploads::default(),
        query_buffer,
        query_staging: Vec::new(),
        query_capacity,
        pending_timestamps: VecDeque::new(),
        timings: Vec::new(),
        frame_pass_count: 0,
        pass_meta: Vec::new(),
        wait_timeout: config.wait_timeout,
        diag: config.alloc_diag.clone(),
        config,
    };
    Ok((
        renderer,
        GpuInfo {
            name: info.name,
            backend: format!("{:?}", info.backend),
            vendor: info.vendor,
            device: info.device,
            device_type: format!("{:?}", info.device_type),
            driver: info.driver,
            driver_info: info.driver_info,
            timestamps: timestamp_support,
        },
    ))
}

/// Validates font data and detects native colour-glyph formats.
///
/// `COLR` fonts render through the colour-glyph lowering; supported sbix and
/// CBDT/CBLC glyphs are decoded as images at realization.
fn validate_font(
    data: &[u8],
    index: u32,
) -> Result<(bool, Option<Arc<bitmap::BitmapFont>>), ResourceError> {
    use skrifa::raw::TableProvider as _;
    let font = skrifa::FontRef::from_index(data, index)
        .map_err(|e| ResourceError::Font(format!("{e}")))?;
    if font.data_for_tag(skrifa::Tag::new(b"SVG ")).is_some() {
        return Err(ResourceError::Unsupported(names::COLOR_FONT));
    }
    let has_colr = font.colr().is_ok();
    let bitmap = bitmap::BitmapFont::detect(data, index)?.map(Arc::new);
    Ok((has_colr, bitmap))
}

impl Renderer for GpuRenderer {
    type Target = GpuTarget;
    fn create_surface(
        &mut self,
        id: SurfaceId,
        target: GpuTarget,
    ) -> Result<SurfaceInfo, SurfaceError> {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        diag::set_surface(Some(id.raw()));
        let size = match &target {
            GpuTarget::Offscreen(offscreen) => offscreen.size,
            GpuTarget::Window(window) => window.size,
            GpuTarget::Texture(texture) => texture.size,
        };
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
        let (window, textures, refresh) = match target {
            GpuTarget::Offscreen(offscreen) => (None, None, offscreen.refresh),
            GpuTarget::Texture(texture) => (None, Some(texture.textures), texture.refresh),
            GpuTarget::Window(window) => {
                let surface = present::WindowSurface::new(
                    &self.instance,
                    &self.adapter,
                    &self.device,
                    window.handle,
                    size,
                    window.transparent,
                )?;
                self.presenter.get_or_insert_with(|| {
                    present::Presenter::new(&self.device, self.shader_delivery)
                });
                (Some(surface), None, window.refresh)
            }
        };
        let (target, view) = create_target(
            &self.device,
            "surface target",
            size,
            TARGET_USAGES,
            TARGET_FORMAT,
        );
        if let Some(sender) = &textures {
            let _ = sender.send(target.clone());
        }
        diag::create(
            &self.device,
            "surface target",
            u64::from(size.0) * u64::from(size.1) * texel_bytes(TARGET_FORMAT),
        );
        self.surfaces.insert(
            id,
            SurfaceState {
                window,
                textures,
                refresh,
                present_pending: false,
                size,
                scratch_format: self.scratch_format,
                target,
                view,
                scratch: Vec::new(),
                backdrop: [None, None],
                backdrop_groups: FxHashMap::default(),
                layers: FxHashMap::default(),
                content: FxHashMap::default(),
                external: FxHashMap::default(),
                shader_textures: FxHashMap::default(),
                frame: LoweredFrame::default(),
                inst_base: 0,
                globals_base: 0,
                bind_gen: 0,
                binds1: FxHashMap::default(),
                binds1_stamp: (u64::MAX, u64::MAX, u64::MAX),
            },
        );
        diag::set_surface(None);
        Ok(SurfaceInfo {
            max_dimension: self.max_texture,
            size,
            readable: true,
        })
    }

    fn resize_surface(&mut self, id: SurfaceId, size: (u32, u32)) {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        diag::set_surface(Some(id.raw()));
        let Some(state) = self.surfaces.get_mut(&id) else {
            diag::set_surface(None);
            return;
        };
        let (target, view) = create_target(
            &self.device,
            "surface target",
            size,
            TARGET_USAGES,
            TARGET_FORMAT,
        );
        if let Some(window) = &mut state.window {
            window.resize(&self.device, size);
        }
        if let Some(sender) = &state.textures {
            let _ = sender.send(target.clone());
        }
        let dropped = state.binds1.len() as u64;
        if dropped > 0 {
            diag::bind_groups_dropped(&self.device, dropped, "resize");
        }
        let scratch_bytes: u64 = state
            .scratch
            .iter()
            .map(|s| u64::from(s.width) * u64::from(s.height) * texel_bytes(s.texture.format()))
            .sum();
        let backdrop_bytes: u64 = state
            .backdrop
            .iter()
            .flatten()
            .map(|s| u64::from(s.width) * u64::from(s.height) * texel_bytes(s.texture.format()))
            .sum();
        let old_target =
            u64::from(state.size.0) * u64::from(state.size.1) * texel_bytes(state.target.format());
        if scratch_bytes > 0 {
            diag::retire(
                &self.device,
                diag::RetireArgs {
                    label: "isolation scratch",
                    class: diag::Class::Target,
                    bytes: scratch_bytes,
                    used_in_latest_submit: true,
                    reason: "resize",
                },
            );
        }
        if backdrop_bytes > 0 {
            diag::retire(
                &self.device,
                diag::RetireArgs {
                    label: "blend backdrop",
                    class: diag::Class::Target,
                    bytes: backdrop_bytes,
                    used_in_latest_submit: true,
                    reason: "resize",
                },
            );
        }
        state.size = size;
        state.target = target;
        state.view = view;
        state.scratch.clear();
        state.backdrop = [None, None];
        state.binds1.clear();
        state.bind_gen += 1;
        diag::grow(
            &self.device,
            "surface target",
            diag::Class::Target,
            old_target,
            u64::from(size.0) * u64::from(size.1) * texel_bytes(TARGET_FORMAT),
            0,
            true,
        );
        diag::set_surface(None);
    }

    fn destroy_surface(&mut self, id: SurfaceId) {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        if let Some(state) = self.surfaces.get(&id) {
            diag::set_surface(Some(id.raw()));
            let target_bytes = u64::from(state.size.0)
                * u64::from(state.size.1)
                * texel_bytes(state.target.format());
            let scratch_bytes: u64 = state
                .scratch
                .iter()
                .map(|s| u64::from(s.width) * u64::from(s.height) * texel_bytes(s.texture.format()))
                .sum();
            let backdrop_bytes: u64 = state
                .backdrop
                .iter()
                .flatten()
                .map(|s| u64::from(s.width) * u64::from(s.height) * texel_bytes(s.texture.format()))
                .sum();
            let capture_bytes: u64 = state
                .backdrop_groups
                .values()
                .flat_map(|g| &g.captures)
                .map(|s| u64::from(s.width) * u64::from(s.height) * texel_bytes(s.texture.format()))
                .sum();
            let dropped = state.binds1.len() as u64;
            if dropped > 0 {
                diag::bind_groups_dropped(&self.device, dropped, "destroy");
            }
            for (label, bytes) in [
                ("surface target", target_bytes),
                ("isolation scratch", scratch_bytes),
                ("blend backdrop", backdrop_bytes),
                ("backdrop capture", capture_bytes),
            ] {
                if bytes > 0 {
                    diag::retire(
                        &self.device,
                        diag::RetireArgs {
                            label,
                            class: diag::Class::Target,
                            bytes,
                            used_in_latest_submit: true,
                            reason: "destroy",
                        },
                    );
                }
            }
        }
        diag::set_surface(None);
        self.surfaces.remove(&id);
        self.update_filter_activity();
    }

    fn add_font(&mut self, id: FontId, font: EngineFontData) -> Result<(), ResourceError> {
        let (has_colr, bitmap) = validate_font(&font.data, font.index)?;
        self.fonts.insert(
            id.raw(),
            FontData {
                data: font.data,
                index: font.index,
                has_colr,
                has_bitmap: bitmap.is_some(),
                bitmap,
                colr: std::cell::RefCell::new(FxHashMap::default()),
            },
        );
        Ok(())
    }

    fn remove_font(&mut self, id: FontId) {
        self.fonts.remove(&id.raw());
        self.bitmaps.retain(|key, _| key.font != id.raw());
        self.images_gen += 1;
        for surface in self.surfaces.values_mut() {
            for content in surface.layers.values_mut() {
                content.invalidate();
            }
        }
        self.atlas.remove_font(id.raw());
    }

    fn set_content(
        &mut self,
        surface: SurfaceId,
        layer: LayerId,
        content: Option<ContentOp>,
    ) -> Option<cherenkov::Picture> {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        let state = self.surfaces.get_mut(&surface)?;
        if state.content.remove(&layer).is_some() {
            diag::bind_groups_dropped(&self.device, state.binds1.len() as u64, "content change");
            state.binds1.clear();
        }
        state.external.remove(&layer);
        match content {
            Some(ContentOp::Replace(list)) => {
                if let Some(content) = state.layers.get_mut(&layer) {
                    Some(content.replace(list))
                } else {
                    state.layers.insert(layer, ContentData::new(list));
                    None
                }
            }
            Some(ContentOp::Update(updates)) => {
                state
                    .layers
                    .get_mut(&layer)
                    .expect("slot update targets a layer without content")
                    .update(updates);
                None
            }
            Some(ContentOp::Picture(picture)) => state
                .layers
                .insert(layer, ContentData::picture(picture))
                .map(ContentData::into_picture),
            None => state.layers.remove(&layer).map(ContentData::into_picture),
        }
    }

    fn remove_layer(&mut self, surface: SurfaceId, layer: LayerId) {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        if let Some(state) = self.surfaces.get_mut(&surface) {
            state.layers.remove(&layer);
            state.external.remove(&layer);
            if state.content.remove(&layer).is_some() {
                diag::bind_groups_dropped(
                    &self.device,
                    state.binds1.len() as u64,
                    "content change",
                );
                state.binds1.clear();
            }
        }
    }

    fn add_image(&mut self, id: ImageId, image: ImageUpload) -> Result<(), ResourceError> {
        let data = image_texels_f16(&image)?;
        let image = upload_image(
            &self.device,
            &self.queue,
            (image.width, image.height),
            &data,
        );
        self.images.insert(id.raw(), image);
        self.images_gen += 1;
        Ok(())
    }

    fn replace_image(&mut self, id: ImageId, image: ImageUpload) -> Result<(), ResourceError> {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        let data = image_texels_f16(&image)?;
        let size = (image.width, image.height);
        let current = self
            .images
            .get(&id.raw())
            .expect("replace targets a registered image");
        if (current.width, current.height) == size {
            // The copy is queued ahead of the next submission, after every
            // frame already submitted, so no frame samples a partly
            // written texture. The view is unchanged: bind groups and
            // lowered paints stay valid.
            write_image(&self.device, &self.queue, &current.texture, size, &data);
            return Ok(());
        }
        let replacement = upload_image(&self.device, &self.queue, size, &data);
        let old = self
            .images
            .insert(id.raw(), replacement)
            .expect("replace targets a registered image");
        self.images_gen += 1;
        diag::retire(
            &self.device,
            diag::RetireArgs {
                label: "image",
                class: diag::Class::Image,
                bytes: u64::from(old.width) * u64::from(old.height) * 8,
                used_in_latest_submit: true,
                reason: "replace_image",
            },
        );
        // #169 A4: bind groups created against the replaced view are
        // stale — `Registered(id)` now binds the new texture; submitted
        // encoders keep the old one until completion. Lowering resolved
        // the old dimensions into the image's paints, so content sampling
        // it is lowered again.
        let images_gen = self.images_gen;
        let mask_gen = self.atlas.mask_texture_generation();
        for surf in self.surfaces.values_mut() {
            retire_binds1(
                surf,
                &self.device,
                images_gen,
                mask_gen,
                "image replaced",
                |key| key.2 == Some(lower::ImageSource::Registered(id.raw())),
            );
            for content in surf.layers.values_mut() {
                content.invalidate_image(id);
            }
        }
        Ok(())
    }

    fn samples_image(&self, surface: SurfaceId, id: ImageId) -> bool {
        self.surfaces.get(&surface).is_some_and(|state| {
            state
                .layers
                .values()
                .any(|content| content.references_image(id))
        })
    }

    fn remove_image(&mut self, id: ImageId) {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        if let Some(image) = self.images.get(&id.raw()) {
            diag::retire(
                &self.device,
                diag::RetireArgs {
                    label: "image",
                    class: diag::Class::Image,
                    bytes: u64::from(image.width) * u64::from(image.height) * 8,
                    used_in_latest_submit: true,
                    reason: "remove_image",
                },
            );
        }
        self.images.remove(&id.raw());
        self.images_gen += 1;
        // #169 A4: drop unsubmitted bind groups holding the removed
        // image's view; submitted encoders keep it until completion.
        let images_gen = self.images_gen;
        let mask_gen = self.atlas.mask_texture_generation();
        for surf in self.surfaces.values_mut() {
            retire_binds1(
                surf,
                &self.device,
                images_gen,
                mask_gen,
                "image removed",
                |key| key.2 == Some(lower::ImageSource::Registered(id.raw())),
            );
            for content in surf.layers.values_mut() {
                content.invalidate();
            }
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "rebuilds every surface's scratch, bindings and buffers in pressure order"
    )]
    fn trim(&mut self, pressure: Pressure) {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        diag::set_phase("trim");
        self.shadow_blur.trim();
        for surf in self.surfaces.values_mut() {
            let scratch_bytes: u64 = surf
                .scratch
                .iter()
                .map(|s| u64::from(s.width) * u64::from(s.height) * texel_bytes(s.texture.format()))
                .sum();
            let backdrop_bytes: u64 = surf
                .backdrop
                .iter()
                .flatten()
                .map(|s| u64::from(s.width) * u64::from(s.height) * texel_bytes(s.texture.format()))
                .sum();
            let capture_bytes: u64 = surf
                .backdrop_groups
                .values()
                .flat_map(|state| state.captures.iter())
                .map(|s| u64::from(s.width) * u64::from(s.height) * texel_bytes(s.texture.format()))
                .sum();
            let dropped = surf.binds1.len() as u64;
            if dropped > 0 {
                diag::bind_groups_dropped(&self.device, dropped, "trim");
            }
            for (label, bytes) in [
                ("isolation scratch", scratch_bytes),
                ("blend backdrop", backdrop_bytes),
                ("backdrop capture", capture_bytes),
            ] {
                if bytes > 0 {
                    diag::retire(
                        &self.device,
                        diag::RetireArgs {
                            label,
                            class: diag::Class::Target,
                            bytes,
                            used_in_latest_submit: true,
                            reason: "trim",
                        },
                    );
                }
            }
            surf.scratch.clear();
            surf.backdrop = [None, None];
            for state in surf.backdrop_groups.values_mut() {
                state.captures.clear();
            }
            // The bind groups' views died with the textures.
            surf.binds1.clear();
            surf.bind_gen += 1;
        }
        let filter_bytes = self.filters.trim();
        if filter_bytes > 0 {
            diag::retire(
                &self.device,
                diag::RetireArgs {
                    label: "filter targets",
                    class: diag::Class::Target,
                    bytes: filter_bytes,
                    used_in_latest_submit: true,
                    reason: "trim",
                },
            );
        }
        if pressure != Pressure::Critical {
            return;
        }
        self.atlas.clear();
        self.bitmaps.clear();
        self.images_gen += 1;
        for font in self.fonts.values() {
            font.colr.borrow_mut().clear();
        }
        for surf in self.surfaces.values_mut() {
            for content in surf.layers.values_mut() {
                content.trim();
            }
            surf.frame.instances.shrink_to_fit();
            surf.frame.stops.shrink_to_fit();
            surf.frame.passes.shrink_to_fit();
        }
        self.uploads = upload::Uploads::default();
        diag::grow(
            &self.device,
            "instances",
            diag::Class::Buffer,
            self.instances.size(),
            272 * 16,
            0,
            true,
        );
        diag::grow(
            &self.device,
            "stops",
            diag::Class::Buffer,
            self.stops.size(),
            32 * 16,
            0,
            true,
        );
        diag::grow(
            &self.device,
            "globals",
            diag::Class::Buffer,
            self.globals.size(),
            16,
            0,
            true,
        );
        self.instances = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: 272 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.stops = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stops"),
            size: 32 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.globals = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            // One 256-byte stride slot: a single pass's Globals entry.
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.bound_instance_size = self.instances.size();
        self.bound_stop_size = self.stops.size();
        self.bound_globals_size = self.globals.size();
        let atlas = &self.atlas;
        diag::bind_groups_dropped(&self.device, 1, "trim");
        self.bind0 = make_bind0(
            &self.device,
            &self.layout0,
            &self.globals,
            &self.instances,
            &self.stops,
            atlas,
        );
        self.bound_atlas = atlas.generation();
        diag::set_phase("render");
    }

    fn memory(&self) -> MemoryUsage {
        let gpu = self.instances.size()
            + self.uploads.gpu_bytes()
            + self.stops.size()
            + self.globals.size()
            + self.atlas.gpu_bytes()
            + self.atlas.mask_texture_bytes()
            + self
                .surfaces
                .values()
                .map(SurfaceState::gpu_bytes)
                .sum::<u64>()
            + self
                .images
                .values()
                .map(|i| u64::from(i.width) * u64::from(i.height) * 8)
                .sum::<u64>()
            + self
                .bitmaps
                .values()
                .map(|bitmap| u64::from(bitmap.image.width) * u64::from(bitmap.image.height) * 8)
                .sum::<u64>();
        let captures = self
            .surfaces
            .values()
            .map(SurfaceState::backdrop_bytes)
            .sum();
        let capture_format = self
            .surfaces
            .values()
            .flat_map(|surf| surf.backdrop_groups.values())
            .flat_map(|g| &g.captures)
            .map(|c| format_name(c.texture.format()))
            .next();
        MemoryUsage {
            gpu: cherenkov::Bytes(gpu + self.filters.gpu_bytes() + self.shadow_blur.gpu_bytes()),
            cpu: cherenkov::Bytes(self.atlas.cpu_bytes()),
            backdrop_captures: cherenkov::Bytes(captures),
            backdrop_capture_format: capture_format,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn render(&mut self, frame: &Frame<'_>, stats: &mut FrameStats) -> Result<Redraw, RenderError> {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        diag::set_phase("render");
        diag::frame_boundary(&self.device, frame.id.get(), true, false);
        let outcome = self.render_inner(frame, stats);
        diag::frame_boundary(&self.device, frame.id.get(), false, outcome.is_ok());
        outcome
    }

    #[cfg(target_arch = "wasm32")]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn render(
        &mut self,
        frame: &Frame<'_>,
        stats: &mut FrameStats,
    ) -> Result<Redraw, RenderError> {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        diag::set_phase("render");
        diag::frame_boundary(&self.device, frame.id.get(), true, false);
        let outcome = self.render_inner(frame, stats).await;
        diag::frame_boundary(&self.device, frame.id.get(), false, outcome.is_ok());
        outcome
    }

    /// Waits for the GPU to finish every pending frame and returns their
    /// timings, oldest first.
    #[cfg(not(target_arch = "wasm32"))]
    fn finish_timings(&mut self) -> Result<Vec<FrameTiming>, RenderError> {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        // Tooling may wait; the frame path only polls. First complete draws
        // so their resolves can be encoded, then complete the resolve copies.
        if let Some(last) = self.pending_queries.back().map(|p| p.submission.clone()) {
            self.wait(last, "timestamp draws")?;
        }
        self.drain_timestamps();
        if !self.pending_queries.is_empty() {
            return Err(RenderError::Readback(
                "timestamp draws: the completion callback did not run after the wait".into(),
            ));
        }
        if let Some(last) = self.pending_timestamps.back().map(|p| p.submission.clone()) {
            for pending in &mut self.pending_timestamps {
                pending.request_map();
            }
            self.wait(last, "timestamp resolve")?;
            self.drain_timestamps();
        }
        if self.pending_timestamps.is_empty() {
            Ok(std::mem::take(&mut self.timings))
        } else {
            Err(RenderError::Readback(
                "timestamp resolve: the map callback did not run after the wait".into(),
            ))
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn finish_timings(&mut self) -> Result<Vec<FrameTiming>, RenderError> {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        // Tooling may wait; the frame path only polls. First complete draws
        // so their resolves can be encoded, then complete the resolve copies.
        if let Some(last) = self.pending_queries.back().map(|p| p.submission.clone()) {
            self.wait(last, "timestamp draws").await?;
        }
        self.drain_timestamps();
        if !self.pending_queries.is_empty() {
            return Err(RenderError::Readback(
                "timestamp draws: the completion callback did not run after the wait".into(),
            ));
        }
        if let Some(last) = self.pending_timestamps.back().map(|p| p.submission.clone()) {
            for pending in &mut self.pending_timestamps {
                pending.request_map();
            }
            self.wait(last, "timestamp resolve").await?;
            self.drain_timestamps();
        }
        if self.pending_timestamps.is_empty() {
            Ok(std::mem::take(&mut self.timings))
        } else {
            Err(RenderError::Readback(
                "timestamp resolve: the map callback did not run after the wait".into(),
            ))
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn readback(&mut self, surface: SurfaceId) -> Result<Readback, RenderError> {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        let Some(state) = self.surfaces.get(&surface) else {
            return Err(RenderError::Readback("unknown surface".into()));
        };
        let (w, h) = state.size;
        diag::set_surface(Some(surface.raw()));
        diag::set_phase("readback");
        let bytes_per_row = (w * 8).div_ceil(256) * 256;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(bytes_per_row) * u64::from(h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        diag::create(&self.device, "readback", buf.size());
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
        let submission = self.queue.submit([encoder.finish()]);
        diag::submit(&self.device, &self.queue, "readback");
        tracing::trace!(?surface, ?submission, "readback submitted");
        let slice = buf.slice(..);
        self.map_read(slice, submission, "the pixel readback")?;
        let data = slice
            .get_mapped_range()
            .expect("buffer range is mapped and not overlapping");
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
        diag::retire(
            &self.device,
            diag::RetireArgs {
                label: "readback",
                class: diag::Class::MapBuffer,
                bytes: buf.size(),
                used_in_latest_submit: true,
                reason: "readback done",
            },
        );
        diag::set_surface(None);
        diag::set_phase("render");
        Ok(Readback {
            width: w,
            height: h,
            pixels,
        })
    }

    #[cfg(target_arch = "wasm32")]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn readback(&mut self, surface: SurfaceId) -> Result<Readback, RenderError> {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        let Some(state) = self.surfaces.get(&surface) else {
            return Err(RenderError::Readback("unknown surface".into()));
        };
        let (w, h) = state.size;
        diag::set_surface(Some(surface.raw()));
        diag::set_phase("readback");
        let bytes_per_row = (w * 8).div_ceil(256) * 256;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(bytes_per_row) * u64::from(h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        diag::create(&self.device, "readback", buf.size());
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
        let submission = self.queue.submit([encoder.finish()]);
        diag::submit(&self.device, &self.queue, "readback");
        tracing::trace!(?surface, ?submission, "readback submitted");
        let slice = buf.slice(..);
        self.map_read(slice, submission, "the pixel readback")
            .await?;
        let data = slice
            .get_mapped_range()
            .expect("buffer range is mapped and not overlapping");
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
        diag::retire(
            &self.device,
            diag::RetireArgs {
                label: "readback",
                class: diag::Class::MapBuffer,
                bytes: buf.size(),
                used_in_latest_submit: true,
                reason: "readback done",
            },
        );
        diag::set_surface(None);
        diag::set_phase("render");
        Ok(Readback {
            width: w,
            height: h,
            pixels,
        })
    }
}

impl GpuRenderer {
    #[cfg(target_arch = "wasm32")]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "the frame pipeline: lowers, uploads, encodes and presents in one pass"
    )]
    async fn render_inner(
        &mut self,
        frame: &Frame<'_>,
        stats: &mut FrameStats,
    ) -> Result<Redraw, RenderError> {
        let origin = *self.origin.get_or_insert(frame.time.0);
        self.drain_timestamps();
        let dirty: Vec<_> = frame
            .surfaces
            .iter()
            .filter(|sf| {
                sf.changed
                    || self.surfaces[&sf.id]
                        .frame
                        .filters
                        .iter()
                        .any(|(_, id)| self.filters.wants_redraw(*id))
                    || self.surfaces[&sf.id].content_wants_redraw()
                    || self.surfaces[&sf.id]
                        .shader_textures
                        .keys()
                        .any(|key| self.shaders.animated(key))
            })
            .collect();
        if dirty.is_empty() {
            diag::set_phase("present");
            return self.present_windows(frame);
        }
        let timing = self.filter_timing(frame, origin);
        self.frame_pass_count = 0;
        self.frame_submission = None;
        self.pass_meta.clear();
        // Lower every dirty surface first: the GPU timestamp bracket must
        // start after CPU lowering (rasters, uploads) so it measures GPU
        // work only. Instances, stops and globals are appended frame-wide
        // at per-surface bases so a later surface's upload can't clobber
        // an earlier one before it is encoded.
        // Take the states out so the lowering workers own them.
        let t_lower = Instant::now();
        let mut pending: Vec<SurfaceState> = dirty
            .iter()
            .map(|id| {
                self.surfaces
                    .remove(&id.id)
                    .expect("dirty surface must exist")
            })
            .collect();
        diag::set_phase("lower");
        let results = self.lower_all(&mut pending, &dirty);
        for (id, surf) in dirty.iter().zip(pending) {
            self.surfaces.insert(id.id, surf);
        }
        let mut inst_base = 0u32;
        let mut stop_base = 0u32;
        let mut globals_base = 0u32;
        let mut result = Ok(());
        for (sf, lowered) in dirty.iter().zip(results) {
            let id = sf.id;
            result = self.lower_surface(id, stats, inst_base, stop_base, globals_base, lowered);
            if result.is_err() {
                break;
            }
            if let Some(surf) = self.surfaces.get(&id) {
                inst_base += u32::try_from(surf.frame.instances.len()).unwrap_or(u32::MAX);
                stop_base += u32::try_from(surf.frame.stops.len()).unwrap_or(u32::MAX);
                globals_base += u32::try_from(surf.frame.passes.len()).unwrap_or(u32::MAX);
            }
        }
        let wait = stats.phases.wait_seconds;
        if result.is_ok() {
            result = self.upload_frame(&dirty, stats).await;
        }
        stats.phases.lower_seconds =
            t_lower.elapsed().as_secs_f64() - (stats.phases.wait_seconds - wait);
        tracing::debug!(
            surfaces = dirty.len(),
            lower_ms = stats.phases.lower_seconds * 1e3,
            ok = result.is_ok(),
            "frame lowered"
        );
        if result.is_ok() {
            self.update_filter_activity();
            for sf in &dirty {
                self.render_shaders(
                    sf.id,
                    frame.time.0.saturating_duration_since(origin).as_secs_f32(),
                )?;
                self.render_producers(sf, frame.time.0).await?;
                self.prepare_filters(sf.id).await?;
            }
            diag::set_phase("encode");
            let t = Instant::now();
            for sf in &dirty {
                let count = self.frame_pass_count;
                let meta = self.pass_meta.len();
                if let Err(error) = self.encode_surface(sf.id, timing, stats) {
                    self.frame_pass_count = count;
                    self.pass_meta.truncate(meta);
                    result = Err(error);
                    break;
                }
                let surface = self.surfaces.get_mut(&sf.id).expect("rendered surface");
                surface.present_pending = surface.window.is_some();
            }
            stats.phases.encode_seconds = t.elapsed().as_secs_f64();
            stats.frame = Some(frame.id);
            if self.timestamps && self.frame_pass_count > 0 {
                diag::set_phase("timestamps");
                let t = Instant::now();
                self.queue_timestamps(2 * self.frame_pass_count, frame.id);
                stats.phases.stamp_seconds += t.elapsed().as_secs_f64();
            }
            self.evict_dead_mask_textures();
        }
        result?;
        diag::set_phase("present");
        let present = self.present_windows(frame)?;
        Ok(self.requested_redraw(present))
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[expect(
        clippy::too_many_lines,
        reason = "the frame pipeline: lowers, uploads, encodes and presents in one pass"
    )]
    fn render_inner(
        &mut self,
        frame: &Frame<'_>,
        stats: &mut FrameStats,
    ) -> Result<Redraw, RenderError> {
        let origin = *self.origin.get_or_insert(frame.time.0);
        self.drain_timestamps();
        let dirty: Vec<_> = frame
            .surfaces
            .iter()
            .filter(|sf| {
                sf.changed
                    || self.surfaces[&sf.id]
                        .frame
                        .filters
                        .iter()
                        .any(|(_, id)| self.filters.wants_redraw(*id))
                    || self.surfaces[&sf.id].content_wants_redraw()
                    || self.surfaces[&sf.id]
                        .shader_textures
                        .keys()
                        .any(|key| self.shaders.animated(key))
            })
            .collect();
        if dirty.is_empty() {
            diag::set_phase("present");
            return self.present_windows(frame);
        }
        let timing = self.filter_timing(frame, origin);
        self.frame_pass_count = 0;
        self.frame_submission = None;
        self.pass_meta.clear();
        // Lower every dirty surface first: the GPU timestamp bracket must
        // start after CPU lowering (rasters, uploads) so it measures GPU
        // work only. Instances, stops and globals are appended frame-wide
        // at per-surface bases so a later surface's upload can't clobber
        // an earlier one before it is encoded.
        // Take the states out so the lowering workers own them.
        let t_lower = Instant::now();
        let mut pending: Vec<SurfaceState> = dirty
            .iter()
            .map(|id| {
                self.surfaces
                    .remove(&id.id)
                    .expect("dirty surface must exist")
            })
            .collect();
        diag::set_phase("lower");
        let results = self.lower_all(&mut pending, &dirty);
        for (id, surf) in dirty.iter().zip(pending) {
            self.surfaces.insert(id.id, surf);
        }
        let mut inst_base = 0u32;
        let mut stop_base = 0u32;
        let mut globals_base = 0u32;
        let mut result = Ok(());
        for (sf, lowered) in dirty.iter().zip(results) {
            let id = sf.id;
            result = self.lower_surface(id, stats, inst_base, stop_base, globals_base, lowered);
            if result.is_err() {
                break;
            }
            if let Some(surf) = self.surfaces.get(&id) {
                inst_base += u32::try_from(surf.frame.instances.len()).unwrap_or(u32::MAX);
                stop_base += u32::try_from(surf.frame.stops.len()).unwrap_or(u32::MAX);
                globals_base += u32::try_from(surf.frame.passes.len()).unwrap_or(u32::MAX);
            }
        }
        let wait = stats.phases.wait_seconds;
        result = result.and_then(|()| self.upload_frame(&dirty, stats));
        stats.phases.lower_seconds =
            t_lower.elapsed().as_secs_f64() - (stats.phases.wait_seconds - wait);
        tracing::debug!(
            surfaces = dirty.len(),
            lower_ms = stats.phases.lower_seconds * 1e3,
            ok = result.is_ok(),
            "frame lowered"
        );
        if result.is_ok() {
            self.update_filter_activity();
            for sf in &dirty {
                self.render_shaders(
                    sf.id,
                    frame.time.0.saturating_duration_since(origin).as_secs_f32(),
                )?;
                self.render_producers(sf, frame.time.0)?;
            }
            diag::set_phase("encode");
            let t = Instant::now();
            for sf in &dirty {
                let count = self.frame_pass_count;
                let meta = self.pass_meta.len();
                if let Err(error) = self.encode_surface(sf.id, timing, stats) {
                    self.frame_pass_count = count;
                    self.pass_meta.truncate(meta);
                    result = Err(error);
                    break;
                }
                let surface = self.surfaces.get_mut(&sf.id).expect("rendered surface");
                surface.present_pending = surface.window.is_some();
            }
            stats.phases.encode_seconds = t.elapsed().as_secs_f64();
            stats.frame = Some(frame.id);
            if self.timestamps && self.frame_pass_count > 0 {
                diag::set_phase("timestamps");
                let t = Instant::now();
                self.queue_timestamps(2 * self.frame_pass_count, frame.id);
                stats.phases.stamp_seconds += t.elapsed().as_secs_f64();
            }
            self.evict_dead_mask_textures();
        }
        result?;
        diag::set_phase("present");
        let present = self.present_windows(frame)?;
        Ok(self.requested_redraw(present))
    }

    fn filter_timing(&mut self, frame: &Frame<'_>, origin: Instant) -> filtrate::EffectFrameTiming {
        let timing = filtrate::EffectFrameTiming::new(
            frame.time.0.saturating_duration_since(origin),
            self.last_frame.map_or(std::time::Duration::ZERO, |last| {
                frame.time.0.saturating_duration_since(last)
            }),
            frame.id.get(),
        );
        self.last_frame = Some(frame.time.0);
        timing
    }

    fn update_filter_activity(&self) {
        let active = self
            .surfaces
            .values()
            .flat_map(|surface| surface.frame.filters.iter().map(|(_, id)| *id))
            .collect();
        self.filters.set_active(&active);
    }
    pub(crate) fn add_filter(&mut self, id: cherenkov::FilterId, source: Box<dyn filter::Source>) {
        self.filters.add(filter::FilterKey::Layer(id.raw()), source);
    }
    pub(crate) fn remove_filter(&mut self, id: cherenkov::FilterId) {
        self.filters.remove(filter::FilterKey::Layer(id.raw()));
    }
    pub(crate) fn add_backdrop_group(
        &mut self,
        surface: SurfaceId,
        id: cherenkov::BackdropId,
        source: Option<Box<dyn filter::Source>>,
    ) {
        let Some(surf) = self.surfaces.get_mut(&surface) else {
            return;
        };
        let filter = source.map(|source| {
            let key = filter::FilterKey::Backdrop {
                surface: surface.raw(),
                group: id.raw(),
            };
            self.filters.add(key, source);
            key
        });
        surf.backdrop_groups.insert(
            id.raw(),
            BackdropGroupState {
                filter,
                captures: Vec::new(),
            },
        );
    }
    pub(crate) fn remove_backdrop_group(&mut self, surface: SurfaceId, id: cherenkov::BackdropId) {
        let Some(surf) = self.surfaces.get_mut(&surface) else {
            return;
        };
        if let Some(state) = surf.backdrop_groups.remove(&id.raw()) {
            if let Some(key) = state.filter {
                let bytes = self.filters.remove(key);
                if bytes > 0 {
                    diag::retire(
                        &self.device,
                        diag::RetireArgs {
                            label: "filter targets",
                            class: diag::Class::Target,
                            bytes,
                            used_in_latest_submit: true,
                            reason: "backdrop group removed",
                        },
                    );
                }
            }
            if !state.captures.is_empty() {
                surf.bind_gen += 1;
                retire_binds1(
                    surf,
                    &self.device,
                    self.images_gen,
                    self.atlas.mask_texture_generation(),
                    "backdrop group removed",
                    |key| matches!(key.0, Some(Source::Backdrop { group, .. }) if group == id.raw()),
                );
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn add_shader(
        &mut self,
        id: cherenkov::ShaderId,
        source: &cherenkov::ShaderSource,
    ) -> Result<(), ResourceError> {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        self.shaders.add(&self.device, id.raw(), source)
    }

    #[cfg(target_arch = "wasm32")]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    pub(crate) async fn add_shader(
        &mut self,
        id: cherenkov::ShaderId,
        source: &cherenkov::ShaderSource,
    ) -> Result<(), ResourceError> {
        self.shaders.add(&self.device, id.raw(), source).await
    }

    /// Compiles `source` into a backdrop effect pipeline per target
    /// format. The module text is the stock shader with the stub
    /// `backdrop_effect` removed and the user source appended — the
    /// stub sits between two `// backdrop-effect-stub` marker lines, so
    /// removal is a plain string split.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn add_backdrop_shader(
        &mut self,
        id: cherenkov::BackdropShaderId,
        source: &cherenkov::BackdropShaderSource,
    ) -> Result<(), ResourceError> {
        // The scope covers module creation: invalid WGSL reports at
        // module use, and `create_pipeline` scopes only itself.
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("backdrop effect"),
                source: wgpu::ShaderSource::Wgsl(backdrop_effect_text(&source.source)),
            });
        let make = |format| -> Result<wgpu::RenderPipeline, ResourceError> {
            create_pipeline(
                &self.device,
                &self.config,
                &self.layout0,
                &self.layout1,
                &module,
                format,
                false,
            )
            .map_err(|e| ResourceError::Shader(e.to_string()))
        };
        let surface = make(TARGET_FORMAT);
        let scratch = make(self.scratch_format);
        let scope_error = pollster::block_on(scope.pop());
        // An invalid module reports through the scope and fails the
        // pipelines only as a consequence: surface it first.
        if let Some(error) = scope_error {
            return Err(ResourceError::Shader(format!("{error}")));
        }
        let pipelines = [surface?, scratch?];
        self.backdrop_shaders.insert(id.raw(), pipelines);
        Ok(())
    }

    /// The wasm variant of [`GpuRenderer::add_backdrop_shader`].
    #[cfg(target_arch = "wasm32")]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    pub(crate) async fn add_backdrop_shader(
        &mut self,
        id: cherenkov::BackdropShaderId,
        source: &cherenkov::BackdropShaderSource,
    ) -> Result<(), ResourceError> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("backdrop effect"),
                source: wgpu::ShaderSource::Wgsl(backdrop_effect_text(&source.source)),
            });
        let surface = create_pipeline(
            &self.device,
            &self.config,
            &self.layout0,
            &self.layout1,
            &module,
            TARGET_FORMAT,
            false,
        )
        .await
        .map_err(|e| ResourceError::Shader(e.to_string()));
        let scratch = create_pipeline(
            &self.device,
            &self.config,
            &self.layout0,
            &self.layout1,
            &module,
            self.scratch_format,
            false,
        )
        .await
        .map_err(|e| ResourceError::Shader(e.to_string()));
        // The scope is popped before `?` propagates: an early return must
        // not leak an unbalanced error scope. An invalid module reports
        // through the scope and fails the pipelines only as a
        // consequence: surface it first.
        let scope_error = scope.pop().await;
        if let Some(error) = scope_error {
            return Err(ResourceError::Shader(format!("{error}")));
        }
        let pipelines = [surface?, scratch?];
        self.backdrop_shaders.insert(id.raw(), pipelines);
        Ok(())
    }

    /// Frees a backdrop effect shader's pipelines. A member that still
    /// samples it fails at encode with a render error.
    pub(crate) fn remove_backdrop_shader(&mut self, id: cherenkov::BackdropShaderId) {
        self.backdrop_shaders.remove(&id.raw());
    }

    pub(crate) fn remove_shader(&mut self, id: cherenkov::ShaderId) {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        self.shaders.remove(id.raw());
        for surface in self.surfaces.values_mut() {
            surface
                .shader_textures
                .retain(|key, _| key.shader != id.raw());
            surface.binds1.clear();
        }
    }
    fn render_shaders(&mut self, id: SurfaceId, elapsed: f32) -> Result<(), RenderError> {
        let surface = self.surfaces.get_mut(&id).expect("registered surface");
        if self.shaders.has_registrations() {
            let keys: FxHashSet<_> = surface
                .frame
                .passes
                .iter()
                .flat_map(|pass| &pass.ranges)
                .filter_map(|range| match &range.image {
                    Some(lower::ImageSource::Shader(key)) => Some(std::sync::Arc::clone(key)),
                    _ => None,
                })
                .collect();
            let count = surface.shader_textures.len();
            surface.shader_textures.retain(|key, _| keys.contains(key));
            if count != surface.shader_textures.len() {
                surface.binds1.clear();
            }
            for key in keys {
                self.shaders.render(
                    &self.device,
                    &self.queue,
                    &key,
                    &mut surface.shader_textures,
                    elapsed,
                )?;
            }
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn render_producers(
        &mut self,
        sf: &SurfaceFrame<'_>,
        time: Instant,
    ) -> Result<(), RenderError> {
        let surface = self.surfaces.get_mut(&sf.id).expect("registered surface");
        for (id, slot) in &mut surface.content {
            slot.set_active(surface.frame.content.contains(id));
        }
        for id in &surface.frame.content {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "validated display scale fits f32"
            )]
            surface
                .content
                .get_mut(id)
                .expect("composed GPU content")
                .render(
                    &self.adapter,
                    &self.device,
                    &self.queue,
                    time,
                    sf.display.scale as f32,
                )?;
        }
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn render_producers(
        &mut self,
        sf: &SurfaceFrame<'_>,
        time: Instant,
    ) -> Result<(), RenderError> {
        let surface = self.surfaces.get_mut(&sf.id).expect("registered surface");
        for (id, slot) in &mut surface.content {
            slot.set_active(surface.frame.content.contains(id));
        }
        for id in &surface.frame.content {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "validated display scale fits f32"
            )]
            surface
                .content
                .get_mut(id)
                .expect("composed GPU content")
                .render(
                    &self.adapter,
                    &self.device,
                    &self.queue,
                    time,
                    sf.display.scale as f32,
                )
                .await?;
        }
        Ok(())
    }

    fn requested_redraw(&self, present: Redraw) -> Redraw {
        let mut rate = match present {
            Redraw::None => None,
            Redraw::Wanted { rate } => Some(rate),
        };
        for surface in self.surfaces.values() {
            if surface
                .frame
                .filters
                .iter()
                .any(|(_, id)| self.filters.wants_redraw(*id))
                || surface.content_wants_redraw()
                || surface
                    .shader_textures
                    .keys()
                    .any(|key| self.shaders.animated(key))
            {
                rate = Some(rate.map_or_else(
                    || surface.refresh.clone(),
                    |rate| {
                        (*rate.start()).min(*surface.refresh.start())
                            ..=(*rate.end()).max(*surface.refresh.end())
                    },
                ));
            }
        }
        rate.map_or(Redraw::None, |rate| Redraw::Wanted { rate })
    }

    pub(crate) fn set_gpu_content(
        &mut self,
        surface: SurfaceId,
        layer: LayerId,
        size: (u32, u32),
        content: crate::interop::GpuContentBox,
    ) {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        let state = self.surfaces.get_mut(&surface).expect("GPU surface exists");
        state.layers.remove(&layer);
        state.external.remove(&layer);
        state
            .content
            .insert(layer, gpu_content::Slot::new(content, size));
        diag::bind_groups_dropped(&self.device, state.binds1.len() as u64, "content change");
        state.binds1.clear();
    }

    /// Installs a retained external frame on `layer` (`cherenkov::ExternalFrames`).
    ///
    /// The planes are sampled where the frame lands; nothing is copied or
    /// rasterized. Any recorded or GPU content on `layer` is dropped — a
    /// layer has one content kind at a time.
    pub(crate) fn set_external_frame(
        &mut self,
        surface: SurfaceId,
        layer: LayerId,
        frame: crate::interop::ExternalFrame,
    ) {
        let state = self.surfaces.get_mut(&surface).expect("GPU surface exists");
        state.layers.remove(&layer);
        state.content.remove(&layer);
        state
            .external
            .insert(layer, external::Slot::new(&self.device, &self.queue, frame));
    }

    pub(crate) fn resize_gpu_content(
        &mut self,
        surface: SurfaceId,
        layer: LayerId,
        size: (u32, u32),
    ) {
        let _diag_guard = diag::Guard::scope(self.diag.as_ref());
        let state = self.surfaces.get_mut(&surface).expect("GPU surface exists");
        state
            .content
            .get_mut(&layer)
            .expect("layer has GPU content")
            .resize(size);
        // #169 A4: only entries referencing this layer's content image
        // went stale — drop them, keep the rest.
        let images_gen = self.images_gen;
        let mask_gen = self.atlas.mask_texture_generation();
        retire_binds1(
            state,
            &self.device,
            images_gen,
            mask_gen,
            "content resize",
            |key| key.2 == Some(lower::ImageSource::Content(layer)),
        );
    }

    /// Builds the external-frame group-1 layout and both format pipelines
    /// on the first surface draw that samples an external slot.
    #[cfg_attr(
        target_arch = "wasm32",
        expect(
            clippy::unnecessary_wraps,
            reason = "browser WebGPU reports pipeline errors asynchronously, so only the native error scope can fail here"
        )
    )]
    fn ensure_external(&mut self) -> Result<(), RenderError> {
        if self.ext_layout.is_some() {
            return Ok(());
        }
        let ext_layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cherenkov external 1"),
                entries: &layout_entries(bindings::EXTERNAL_GROUP1),
            });
        let module = self.shader_delivery.external_module(&self.device);
        // Error scopes resolve asynchronously; only the native path pops
        // synchronously. A failure here is an engine bug, so the wasm path
        // reports through the uncaptured-error handler instead.
        #[cfg(not(target_arch = "wasm32"))]
        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let formats = [TARGET_FORMAT, self.scratch_format];
        for (pipe, format) in self.external_pipes.iter_mut().zip(formats) {
            *pipe = Some(create_external_pipeline(
                &self.device,
                &self.layout0,
                &ext_layout,
                &module,
                format,
            ));
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            return Err(RenderError::Render(format!("external pipeline: {error}")));
        }
        self.ext_layout = Some(ext_layout);
        Ok(())
    }

    fn present_windows(&mut self, frame: &Frame<'_>) -> Result<Redraw, RenderError> {
        let Some(presenter) = &mut self.presenter else {
            return Ok(Redraw::None);
        };
        let mut redraw = None::<cherenkov::RefreshRange>;
        for sf in frame.surfaces {
            let surface = self.surfaces.get_mut(&sf.id).expect("registered surface");
            if surface.present_pending {
                let window = surface.window.as_ref().expect("pending window");
                surface.present_pending = !presenter.present(
                    &self.device,
                    &self.queue,
                    window,
                    &surface.view,
                    sf.display.headroom,
                )?;
                if surface.present_pending {
                    redraw = Some(redraw.map_or_else(
                        || surface.refresh.clone(),
                        |rate| {
                            (*rate.start()).min(*surface.refresh.start())
                                ..=(*rate.end()).max(*surface.refresh.end())
                        },
                    ));
                }
            }
        }
        Ok(redraw.map_or(Redraw::None, |rate| Redraw::Wanted { rate }))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn lower_all(
        &mut self,
        pending: &mut [SurfaceState],
        frames: &[&SurfaceFrame<'_>],
    ) -> Vec<Result<Lowered, RenderError>> {
        let mut grew = false;
        loop {
            let group_maps: Vec<FxHashMap<u64, BackdropGroupInfo>> = pending
                .iter()
                .map(|surf| surf.backdrop_info(&mut self.filters))
                .collect();
            let mut results: Vec<Result<Lowered, RenderError>> = if pending.len() > 1 {
                let (atlas, images, bitmaps) = (&self.atlas, &self.images, &self.bitmaps);
                // `FontData`'s COLR cache is a `RefCell` — !Sync — so
                // each worker moves in its own snapshot built here.
                let snapshots: Vec<FxHashMap<u64, FontData>> = pending
                    .iter()
                    .map(|_| {
                        self.fonts
                            .iter()
                            .map(|(id, f)| (*id, f.snapshot()))
                            .collect()
                    })
                    .collect();
                std::thread::scope(|s| {
                    pending
                        .iter_mut()
                        .zip(snapshots)
                        .zip(frames)
                        .zip(&group_maps)
                        .map(|(((surf, fonts), frame), groups)| {
                            s.spawn(move || {
                                Self::lower_content(
                                    surf, frame, atlas, &fonts, images, bitmaps, groups,
                                )
                            })
                        })
                        .collect::<Vec<_>>()
                        .into_iter()
                        .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                        .collect()
                })
            } else {
                pending
                    .iter_mut()
                    .zip(frames)
                    .zip(&group_maps)
                    .map(|((surf, frame), groups)| {
                        Self::lower_content(
                            surf,
                            frame,
                            &self.atlas,
                            &self.fonts,
                            &self.images,
                            &self.bitmaps,
                            groups,
                        )
                    })
                    .collect()
            };
            // The batch's pending rasters commit transactionally
            // (#169 A3): a dry run decides fit / grow once / recycle
            // before any placement or upload happens, so a failed
            // placement never enqueues uploads into an atlas the same
            // preparation abandons.
            match self.commit_rasters(pending, &mut results, grew) {
                Commit::Done => break results,
                Commit::Grow(size) => {
                    self.atlas.grow_to(&self.device, size);
                    grew = true;
                    tracing::debug!(
                        size = self.atlas.size(),
                        generation = self.atlas.generation(),
                        "atlas grown"
                    );
                }
            }
            // Growing emptied the atlas: every hit any lowering took is
            // now a miss, so lower the whole batch again.
            diag::event(
                &self.device,
                diag::EventKind::Phase {
                    name: "atlas retry",
                },
            );
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn lower_all(
        &mut self,
        pending: &mut [SurfaceState],
        frames: &[&SurfaceFrame<'_>],
    ) -> Vec<Result<Lowered, RenderError>> {
        let mut grew = false;
        loop {
            let group_maps: Vec<FxHashMap<u64, BackdropGroupInfo>> = pending
                .iter()
                .map(|surf| surf.backdrop_info(&mut self.filters))
                .collect();
            let mut results: Vec<Result<Lowered, RenderError>> = pending
                .iter_mut()
                .zip(frames)
                .zip(&group_maps)
                .map(|((surf, frame), groups)| {
                    Self::lower_content(
                        surf,
                        frame,
                        &self.atlas,
                        &self.fonts,
                        &self.images,
                        &self.bitmaps,
                        groups,
                    )
                })
                .collect();
            // The batch's pending rasters commit transactionally
            // (#169 A3): a dry run decides fit / grow once / recycle
            // before any placement or upload happens, so a failed
            // placement never enqueues uploads into an atlas the same
            // preparation abandons.
            match self.commit_rasters(pending, &mut results, grew) {
                Commit::Done => break results,
                Commit::Grow(size) => {
                    self.atlas.grow_to(&self.device, size);
                    grew = true;
                    tracing::debug!(
                        size = self.atlas.size(),
                        generation = self.atlas.generation(),
                        "atlas grown"
                    );
                }
            }
            // Growing emptied the atlas: every hit any lowering took is
            // now a miss, so lower the whole batch again.
            diag::event(
                &self.device,
                diag::EventKind::Phase {
                    name: "atlas retry",
                },
            );
        }
    }

    fn lower_content(
        surf: &mut SurfaceState,
        frame: &SurfaceFrame<'_>,
        atlas: &Atlas,
        fonts: &FxHashMap<u64, FontData>,
        images: &FxHashMap<u64, GpuImage>,
        bitmaps: &FxHashMap<BitmapKey, GpuBitmap>,
        groups: &FxHashMap<u64, BackdropGroupInfo>,
    ) -> Result<Lowered, RenderError> {
        surf.frame.reset();
        // Lowering borrows `layers` immutably while mutating `frame`;
        // taking the map out keeps the two borrows disjoint.
        let mut layers = std::mem::take(&mut surf.layers);
        let mut lowered = Lowered::default();
        let result = {
            let glyphs = GlyphContext {
                atlas,
                live_stamp: atlas.live_stamp(),
                fonts,
                images,
                bitmaps,
                content: &surf.content,
                external: &surf.external,
            };
            let mut lowering = Lowering::new(&mut surf.frame, surf.size);
            let result = lowering.run(frame.tree, &mut layers, frame.clear, &glyphs, groups);
            lowered.commands = lowering.commands_lowered;
            lowered.layers = lowering.layers_composed;
            lowered.glyphs = lowering.glyphs_rasterized();
            lowered.paths = lowering.paths_rasterized();
            lowered.cell_patches = std::mem::take(&mut lowering.cell_patches);
            lowered.mask_patches = std::mem::take(&mut lowering.mask_patches);
            lowered.pending = std::mem::take(&mut lowering.pending);
            lowered.touches = std::mem::take(&mut lowering.touches);
            result
        };
        surf.layers = layers;
        surf.frame.content.sort_unstable_by_key(|id| id.raw());
        surf.frame.content.dedup();
        surf.frame.external.sort_unstable_by_key(|id| id.raw());
        surf.frame.external.dedup();
        result.map(|()| lowered)
    }

    /// Over budget, drops every mask texture no retained frame references:
    /// bind groups and frames holding one keep it alive by key.
    fn evict_dead_mask_textures(&mut self) {
        if !self.atlas.mask_textures_over_budget() {
            return;
        }
        let live: FxHashSet<u64> = self
            .surfaces
            .values()
            .flat_map(|surf| surf.frame.passes.iter())
            .flat_map(|pass| pass.ranges.iter())
            .filter_map(|range| range.mask)
            .collect();
        let gen_before = self.atlas.mask_texture_generation();
        self.atlas
            .evict_mask_textures(&self.device, |key| live.contains(&key));
        if self.atlas.mask_texture_generation() == gen_before {
            return;
        }
        // #169 A4: unsubmitted bind groups holding an evicted view are
        // obsolete — drop them now rather than at the next encode.
        let mask_gen = self.atlas.mask_texture_generation();
        let images_gen = self.images_gen;
        for surf in self.surfaces.values_mut() {
            retire_binds1(
                surf,
                &self.device,
                images_gen,
                mask_gen,
                "mask texture evict",
                |key| key.3.is_some_and(|mask| !live.contains(&mask)),
            );
        }
    }

    /// Commits every surface's pending rasters transactionally
    /// (#169 A3). The shelves this lowering's hits live on are marked
    /// first so no placement can evict them; [`Atlas::plan`] then
    /// dry-runs all placements against shelf metadata alone; only a
    /// `Fits` verdict — or the last attempt after a grow — commits for
    /// real. When even an emptied atlas cannot hold the batch the
    /// commit runs evicting instead of clearing: cold shelves are
    /// reclaimed in place, so surviving entries — and the retained
    /// emissions referencing them — stay valid (#119). The upload
    /// batches the committed cells into one `write_texture` per newly
    /// allocated shelf region.
    fn commit_rasters(
        &mut self,
        pending: &mut [SurfaceState],
        results: &mut [Result<Lowered, RenderError>],
        grew: bool,
    ) -> Commit {
        let mut touches: Vec<u32> = results
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .flat_map(|l| l.touches.iter().copied())
            .collect();
        // An emission whose stamp matches the atlas verified live this
        // frame — either on the fast path or by re-walking its refs —
        // so its bands are the replay pins the commit must not evict
        // (#119).
        let stamp = self.atlas.live_stamp();
        for surf in pending.iter_mut() {
            for content in surf.layers.values_mut() {
                let (_, emissions) = content.retained.prepared();
                for e in emissions.iter().filter_map(|e| e.data.as_ref()) {
                    if e.live_stamp == stamp {
                        touches.extend(
                            content.storage.refs[e.refs.clone()]
                                .iter()
                                .map(|&(slot, _)| slot),
                        );
                    }
                }
            }
        }
        self.atlas.begin_commit(&touches);
        let rasters: Vec<&glyph::PendingRaster> = results
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .flat_map(|l| l.pending.iter())
            .collect();
        let n_cells: usize = rasters.iter().map(|r| r.cell_count()).sum();
        let plan_dbg = match self.atlas.plan(&rasters) {
            glyph::AtlasPlan::Fits => "fits",
            glyph::AtlasPlan::Grow(size) if !grew => return Commit::Grow(size),
            glyph::AtlasPlan::FitsEviction => {
                self.atlas.enable_evicting();
                "fits-eviction"
            }
            glyph::AtlasPlan::Grow(_) | glyph::AtlasPlan::Recycle => {
                // Bounded in-place eviction makes room instead of a
                // wholesale clear: the commit below reclaims shelves
                // nothing touched until the batch places or nothing
                // untouchable remains, then exhausts the first surface
                // whose raster still does not fit (#119).
                self.atlas.enable_evicting();
                "exhaust-candidate"
            }
        };
        let mut writes = std::mem::take(&mut self.commit_writes);
        writes.clear();
        for (surf, result) in pending.iter_mut().zip(results.iter_mut()) {
            let Ok(lowered) = result else {
                continue;
            };
            match self.apply_pending(surf, lowered, &mut writes) {
                Ok(()) => {}
                Err(RenderError::AtlasFull) => {
                    *result = Err(RenderError::AtlasExhausted);
                    break;
                }
                Err(e) => {
                    *result = Err(e);
                    break;
                }
            }
        }
        let evicted = self.atlas.take_evicted();
        tracing::debug!(
            evictions = evicted.len(),
            evicted_bytes = evicted.iter().map(|e| e.0).sum::<u64>(),
            plan = ?plan_dbg,
            pending_cells = n_cells,
            touches = touches.len(),
            occupancy = ?self.atlas.occupancy(),
            "atlas commit"
        );
        for (bytes, used_in_latest_submit) in evicted {
            diag::retire(
                &self.device,
                diag::RetireArgs {
                    label: "glyph atlas",
                    class: diag::Class::Atlas,
                    bytes,
                    used_in_latest_submit,
                    reason: "atlas evict",
                },
            );
        }
        self.atlas
            .upload_committed(&self.device, &self.queue, &writes);
        self.commit_writes = writes;
        Commit::Done
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "atlas coordinates fit exactly in f32"
    )]
    fn apply_pending(
        &mut self,
        surf: &mut SurfaceState,
        lowered: &mut Lowered,
        writes: &mut Vec<glyph::CellWrite>,
    ) -> Result<(), RenderError> {
        let pending = std::mem::take(&mut lowered.pending);
        let mut origins = std::mem::take(&mut self.pending_origins);
        origins.clear();
        for raster in pending {
            origins.push(self.apply_raster(raster, writes)?);
        }
        let cell_origin = |p: u32, c: u32| {
            let PendingOrigin::Cells(cells, _) = &origins[p as usize] else {
                unreachable!("cell patch must reference cell raster");
            };
            let (x, y) = cells[c as usize];
            [x as f32, y as f32]
        };
        for (inst, p, c) in lowered.cell_patches.drain(..) {
            let [x, y] = cell_origin(p, c);
            surf.frame.instances[inst as usize].uv[..2].copy_from_slice(&[x, y]);
        }
        for content in surf.layers.values_mut() {
            let (_, emissions) = content.retained.prepared();
            for emission in emissions.iter_mut().filter_map(|e| e.data.as_mut()) {
                // The emission's atlas references are the shelves it
                // touched while lowering plus the bands its deferred
                // rasters resolved to — recorded as `(slot, band epoch)`
                // pairs, deduplicated and contiguous at the storage
                // tail (#119).
                let mut slots: rustc_hash::FxHashSet<u32> = rustc_hash::FxHashSet::default();
                for (inst, p, c) in emission.pending_cells.drain(..) {
                    content.storage.instances[emission.instances.start + inst as usize].uv[..2]
                        .copy_from_slice(&cell_origin(p, c));
                    if let PendingOrigin::Cells(_, bands) = &origins[p as usize] {
                        slots.extend(bands.iter().copied());
                    }
                }
                if emission.live_stamp == lower::LIVE_PENDING {
                    // `refs` still addresses the frame's touches: fold
                    // those slots in, then swap the range for resolved
                    // `(slot, band epoch)` pairs — always, even empty,
                    // or it keeps addressing `touches` (#119).
                    slots.extend(lowered.touches[emission.refs.clone()].iter().copied());
                    let first = content.storage.refs.len();
                    for slot in slots {
                        content
                            .storage
                            .refs
                            .push((slot, self.atlas.shelf_epoch(slot)));
                    }
                    emission.refs = first..content.storage.refs.len();
                }
                // Restamp only when every reference survived this
                // commit's evictions; a stale emission must keep an
                // older stamp so its next hit check walks the refs
                // and re-lowers (#119).
                if content.storage.refs[emission.refs.clone()]
                    .iter()
                    .all(|&(s, ep)| self.atlas.shelf_epoch(s) == ep)
                {
                    emission.live_stamp = self.atlas.live_stamp();
                }
            }
        }
        for (inst, p) in lowered.mask_patches.drain(..) {
            let PendingOrigin::Mask(origin) = &origins[p as usize] else {
                unreachable!("mask patch must reference mask raster");
            };
            surf.frame.instances[inst as usize].uv[2..].copy_from_slice(origin);
        }
        self.pending_origins = origins;
        Ok(())
    }

    fn apply_raster(
        &mut self,
        raster: PendingRaster,
        writes: &mut Vec<glyph::CellWrite>,
    ) -> Result<PendingOrigin, RenderError> {
        match raster {
            PendingRaster::Glyph {
                key,
                left,
                top,
                w,
                h,
                texels,
            } => {
                let hit = self.atlas.get(&key).is_some();
                let out = self
                    .atlas
                    .place_glyph(key, left, top, w, h, texels, writes)
                    .map(|(x, y)| {
                        PendingOrigin::Cells(
                            vec![(x, y)],
                            vec![self.atlas.get(&key).expect("just stored").slot],
                        )
                    })
                    .ok_or(RenderError::AtlasFull)?;
                if !hit && w == 0 {
                    diag::atlas_cell(&self.device, (0, 0, 0, 0));
                }
                Ok(out)
            }
            PendingRaster::Path { key, emit, cells } => {
                self.atlas
                    .place_path(key, emit, cells, writes)
                    .ok_or(RenderError::AtlasFull)?;
                Ok(PendingOrigin::Cells(
                    self.atlas.path_origins(key).expect("just stored"),
                    self.atlas.path(key).expect("just stored").slots.to_vec(),
                ))
            }
            PendingRaster::Mask {
                key,
                mask,
                w,
                h,
                texels,
            } => {
                self.atlas
                    .place_mask(key, mask, w, h, texels, writes)
                    .ok_or(RenderError::AtlasFull)?;
                Ok(PendingOrigin::Mask(
                    self.atlas.mask_origin(key).expect("just stored"),
                ))
            }
            PendingRaster::MaskTexture {
                key,
                mask,
                w,
                h,
                texels,
            } => {
                self.atlas
                    .store_mask_texture(&self.device, &self.queue, key, mask, w, h, &texels);
                Ok(PendingOrigin::None)
            }
            PendingRaster::Colr { font, key, picture } => {
                if let Some(font) = self.fonts.get_mut(&font) {
                    font.colr.borrow_mut().entry(key).or_insert(picture);
                }
                Ok(PendingOrigin::None)
            }
            PendingRaster::Bitmap {
                key,
                em,
                width,
                height,
                texels,
            } => {
                if !self.bitmaps.contains_key(&key) {
                    let image = create_gpu_image(
                        &self.device,
                        &self.queue,
                        "bitmap glyph",
                        width,
                        height,
                        &texels,
                    );
                    self.bitmaps.insert(key, GpuBitmap { image, em });
                }
                Ok(PendingOrigin::None)
            }
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one surface's frame application and buffer growth"
    )]
    fn lower_surface(
        &mut self,
        id: SurfaceId,
        stats: &mut FrameStats,
        inst_base: u32,
        stop_base: u32,
        globals_base: u32,
        lowered: Result<Lowered, RenderError>,
    ) -> Result<(), RenderError> {
        diag::set_surface(Some(id.raw()));
        let Lowered {
            glyphs,
            paths,
            commands,
            layers,
            ..
        } = lowered?;
        stats.commands_lowered += commands;
        stats.layers_composed += layers;
        stats.glyphs_rasterized += glyphs;
        stats.paths_rasterized += paths;
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
                Target::Surface | Target::Backdrop { .. } => None,
            })
            .max()
            .unwrap_or(0);
        // The largest region each isolation depth must hold this frame.
        let mut region_max = vec![(0u32, 0u32); max_scratch];
        for pass in &surf.frame.passes {
            if let Target::Scratch(i) = pass.target {
                region_max[i].0 = region_max[i].0.max(pass.region[2]);
                region_max[i].1 = region_max[i].1.max(pass.region[3]);
            }
        }
        // Grow each scratch to its needed size; never shrink.
        for (i, &(w, h)) in region_max.iter().enumerate() {
            let (nw, nh) = (
                w.max(surf.scratch.get(i).map_or(0, |s| s.width)),
                h.max(surf.scratch.get(i).map_or(0, |s| s.height)),
            );
            if surf
                .scratch
                .get(i)
                .is_some_and(|s| s.width >= w && s.height >= h)
            {
                continue;
            }
            if nw > self.device.limits().max_texture_dimension_2d
                || nh > self.device.limits().max_texture_dimension_2d
            {
                return Err(RenderError::Render(
                    "isolation capture exceeds device texture extent".into(),
                ));
            }
            let old_scratch = surf.scratch.get(i).map_or(0, |s| {
                u64::from(s.width) * u64::from(s.height) * texel_bytes(s.texture.format())
            });
            let (texture, view) = create_target(
                &self.device,
                "isolation scratch",
                (nw, nh),
                TARGET_USAGES,
                self.scratch_format,
            );
            let target = ScratchTarget {
                texture,
                view,
                width: nw,
                height: nh,
            };
            diag::grow(
                &self.device,
                "isolation scratch",
                diag::Class::Target,
                old_scratch,
                u64::from(nw) * u64::from(nh) * texel_bytes(self.scratch_format),
                0,
                true,
            );
            if i < surf.scratch.len() {
                surf.scratch[i] = target;
            } else {
                surf.scratch.push(target);
            }
            surf.bind_gen += 1;
            // #169 A4: unsubmitted group-1 bind groups referencing the
            // replaced view keep its predecessor alive — drop them now.
            retire_binds1(
                surf,
                &self.device,
                self.images_gen,
                self.atlas.mask_texture_generation(),
                "scratch regen",
                |key| key.0 == Some(Source::Scratch(i)),
            );
        }
        // Backdrop-group captures are exactly their pass's region, in the
        // format of the target the capture copies from, and sampled by
        // later passes.
        for pass in &surf.frame.passes {
            let Some(capture) = pass.capture else {
                continue;
            };
            let (w, h) = (pass.region[2], pass.region[3]);
            let format = match capture.copy_from {
                Target::Surface => TARGET_FORMAT,
                Target::Scratch(_) => self.scratch_format,
                Target::Backdrop { .. } => {
                    return Err(RenderError::Render(format!(
                        "backdrop group {} copies from a capture",
                        capture.group
                    )));
                }
            };
            let Some(group_state) = surf.backdrop_groups.get_mut(&capture.group) else {
                return Err(RenderError::Render(format!(
                    "backdrop group {} was not registered",
                    capture.group
                )));
            };
            let r = capture.region as usize;
            // #169 A4: like scratch, a capture is never shrunk or
            // regrown around an animated region size — it grows only
            // when the frame needs more, and `trim` releases it outside
            // the hot path.
            let (nw, nh) = (
                w.max(group_state.captures.get(r).map_or(0, |c| c.width)),
                h.max(group_state.captures.get(r).map_or(0, |c| c.height)),
            );
            if group_state
                .captures
                .get(r)
                .is_none_or(|c| c.width < w || c.height < h || c.texture.format() != format)
            {
                let old_capture = group_state.captures.get(r).map_or(0, |c| {
                    u64::from(c.width) * u64::from(c.height) * texel_bytes(c.texture.format())
                });
                let (texture, view) = create_target(
                    &self.device,
                    "backdrop capture",
                    (nw, nh),
                    TARGET_USAGES | wgpu::TextureUsages::COPY_DST,
                    format,
                );
                let target = ScratchTarget {
                    texture,
                    view,
                    width: nw,
                    height: nh,
                };
                if r < group_state.captures.len() {
                    group_state.captures[r] = target;
                } else {
                    group_state.captures.resize_with(r, || ScratchTarget {
                        texture: surf.target.clone(),
                        view: surf.view.clone(),
                        width: 0,
                        height: 0,
                    });
                    group_state.captures.push(target);
                }
                diag::grow(
                    &self.device,
                    "backdrop capture",
                    diag::Class::Target,
                    old_capture,
                    u64::from(nw) * u64::from(nh) * texel_bytes(format),
                    0,
                    true,
                );
                surf.bind_gen += 1;
                // #169 A4: unsubmitted group-1 bind groups referencing
                // the replaced view keep its predecessor alive — drop
                // them now.
                let before = surf.binds1.len();
                surf.binds1
                    .retain(|key, _| {
                        !matches!(key.0, Some(Source::Backdrop { group, .. }) if group == capture.group)
                    });
                let dropped = before - surf.binds1.len();
                if dropped > 0 {
                    diag::bind_groups_dropped(&self.device, dropped as u64, "capture regen");
                }
                surf.binds1_stamp = (
                    surf.bind_gen,
                    self.images_gen,
                    self.atlas.mask_texture_generation(),
                );
            }
        }
        // Regions dropped between frames drop their textures too.
        let needed: FxHashMap<u64, u32> = surf
            .frame
            .passes
            .iter()
            .filter_map(|p| p.capture.map(|c| (c.group, c.region + 1)))
            .fold(FxHashMap::default(), |mut m, (g, n)| {
                m.entry(g).and_modify(|e| *e = (*e).max(n)).or_insert(n);
                m
            });
        for (gid, n) in needed {
            if let Some(state) = surf.backdrop_groups.get_mut(&gid)
                && state.captures.len() > n as usize
            {
                state.captures.truncate(n as usize);
                surf.bind_gen += 1;
                let before = surf.binds1.len();
                surf.binds1.retain(
                    |key, _| !matches!(key.0, Some(Source::Backdrop { group, .. }) if group == gid),
                );
                let dropped = before - surf.binds1.len();
                if dropped > 0 {
                    diag::bind_groups_dropped(&self.device, dropped as u64, "capture trim");
                }
                surf.binds1_stamp = (
                    surf.bind_gen,
                    self.images_gen,
                    self.atlas.mask_texture_generation(),
                );
            }
        }
        // Backdrop textures for blend composites, sized like the scratch
        // pool to the largest region copied this frame.
        let mut backdrop_max = [(0u32, 0u32); 2];
        for pass in &surf.frame.passes {
            if let Some(r) = pass.backdrop_copy {
                let slot = match pass.target {
                    Target::Surface => 0,
                    Target::Scratch(_) => 1,
                    Target::Backdrop { .. } => {
                        return Err(RenderError::Render(
                            "a blend backdrop copy on a capture pass".into(),
                        ));
                    }
                };
                backdrop_max[slot].0 = backdrop_max[slot].0.max(r[2]);
                backdrop_max[slot].1 = backdrop_max[slot].1.max(r[3]);
            }
        }
        for (slot, &(w, h)) in backdrop_max.iter().enumerate() {
            if w == 0 || h == 0 {
                continue;
            }
            if surf.backdrop[slot]
                .as_ref()
                .is_some_and(|b| b.width >= w && b.height >= h)
            {
                continue;
            }
            let (nw, nh) = (
                w.max(surf.backdrop[slot].as_ref().map_or(0, |b| b.width)),
                h.max(surf.backdrop[slot].as_ref().map_or(0, |b| b.height)),
            );
            let format = if slot == 0 {
                TARGET_FORMAT
            } else {
                self.scratch_format
            };
            let old_backdrop = surf.backdrop[slot].as_ref().map_or(0, |b| {
                u64::from(b.width) * u64::from(b.height) * texel_bytes(b.texture.format())
            });
            let (texture, view) = create_target(
                &self.device,
                "blend backdrop",
                (nw, nh),
                TARGET_USAGES | wgpu::TextureUsages::COPY_DST,
                format,
            );
            surf.backdrop[slot] = Some(ScratchTarget {
                texture,
                view,
                width: nw,
                height: nh,
            });
            diag::grow(
                &self.device,
                "blend backdrop",
                diag::Class::Target,
                old_backdrop,
                u64::from(nw) * u64::from(nh) * texel_bytes(format),
                0,
                true,
            );
            surf.bind_gen += 1;
            retire_binds1(
                surf,
                &self.device,
                self.images_gen,
                self.atlas.mask_texture_generation(),
                "backdrop regen",
                |key| key.1,
            );
        }
        // Gradient instances index stops absolutely; shift each instance's
        // first-stop index by this surface's stop base. Only gradient
        // paints read `meta.z`, so bumping it unconditionally is safe.
        if stop_base != 0 {
            for inst in &mut surf.frame.instances {
                inst.meta[2] += stop_base;
            }
        }
        surf.inst_base = inst_base;
        surf.globals_base = globals_base;
        {
            let atlas = &self.atlas;
            if atlas.generation() != self.bound_atlas {
                diag::bind_groups_dropped(&self.device, 1, "atlas generation");
                self.bind0 = make_bind0(
                    &self.device,
                    &self.layout0,
                    &self.globals,
                    &self.instances,
                    &self.stops,
                    atlas,
                );
                self.bound_atlas = atlas.generation();
            }
        }
        // Buffers grown above leave `bind0` stale; rebuild when capacity
        // changed since the bind group was built.
        if self.instances.size() > self.bound_instance_size
            || self.stops.size() > self.bound_stop_size
            || self.globals.size() > self.bound_globals_size
        {
            diag::bind_groups_dropped(&self.device, 1, "buffer growth");
            let atlas = &self.atlas;
            self.bind0 = make_bind0(
                &self.device,
                &self.layout0,
                &self.globals,
                &self.instances,
                &self.stops,
                atlas,
            );
            self.bound_atlas = atlas.generation();
            self.bound_instance_size = self.instances.size();
            self.bound_stop_size = self.stops.size();
            self.bound_globals_size = self.globals.size();
        }

        Ok(())
    }

    /// Grows each frame-wide buffer once for the frame's whole upload
    /// range, before the frame's first submission — never per surface
    /// mid-frame (#169 A2 on the staging ring).
    fn grow_frame_buffers(&mut self, copies: &[upload::Copy]) {
        let mut passes = 0u64;
        for copy in copies {
            let (label, usage, buffer) = match copy.dest {
                upload::Dest::Instances => (
                    "instances",
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    &mut self.instances,
                ),
                upload::Dest::Stops => (
                    "stops",
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    &mut self.stops,
                ),
                upload::Dest::Globals => (
                    "globals",
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    &mut self.globals,
                ),
            };
            let needed = copy.dst + copy.size;
            if needed > buffer.size() {
                *buffer = grow_buffer(
                    &self.device,
                    label,
                    buffer,
                    needed.next_power_of_two(),
                    usage,
                );
            }
            if matches!(copy.dest, upload::Dest::Globals) {
                passes = copy.size / 256;
            }
        }
        // The timestamp query set and resolve buffer grow once for the
        // frame's whole pass count; never mid-encoder.
        if self.timestamps && passes > 0 {
            self.ensure_query_capacity(2 * u32::try_from(passes).unwrap_or(u32::MAX));
        }
    }

    /// The frame's upload size and copies: every dirty surface's instances,
    /// stops and 256-byte globals entries, laid out in the staging slot in
    /// the order they occupy the frame-wide buffers from offset 0.
    fn upload_layout(&self, dirty: &[&SurfaceFrame<'_>]) -> (u64, Vec<upload::Copy>) {
        let (mut instances, mut stops, mut passes) = (0u64, 0u64, 0u64);
        for surf in dirty.iter().filter_map(|sf| self.surfaces.get(&sf.id)) {
            instances += surf.frame.instances.len() as u64;
            stops += surf.frame.stops.len() as u64;
            passes += surf.frame.passes.len() as u64;
        }
        let sizes = [
            (
                upload::Dest::Instances,
                instances * std::mem::size_of::<instance::Instance>() as u64,
            ),
            (
                upload::Dest::Stops,
                stops * std::mem::size_of::<instance::Stop>() as u64,
            ),
            (upload::Dest::Globals, passes * 256),
        ];
        let mut src = 0;
        let mut copies = Vec::new();
        for (dest, size) in sizes {
            if size > 0 {
                copies.push(upload::Copy {
                    dest,
                    src,
                    dst: 0,
                    size,
                });
                src += size;
            }
        }
        (src, copies)
    }

    /// Fills the acquired staging slot with this frame's uploads.
    #[expect(
        clippy::cast_precision_loss,
        reason = "pixel sizes are well within f32"
    )]
    fn write_uploads(&mut self, dirty: &[&SurfaceFrame<'_>], size: u64, copies: &[upload::Copy]) {
        diag::upload(&self.device, "frame staging", size, None);
        let surfaces: Vec<&SurfaceState> = dirty
            .iter()
            .filter_map(|sf| self.surfaces.get(&sf.id))
            .collect();
        self.uploads.write(size, copies, |bytes| {
            let mut at = 0;
            let mut put = |data: &[u8]| {
                bytes.slice(at..at + data.len()).copy_from_slice(data);
                at += data.len();
            };
            for surf in &surfaces {
                put(bytemuck::cast_slice(&surf.frame.instances));
            }
            for surf in &surfaces {
                put(bytemuck::cast_slice(&surf.frame.stops));
            }
            let mut entry = [0u8; 256];
            for surf in &surfaces {
                for pass in &surf.frame.passes {
                    let g = lower::globals(
                        [pass.region[2] as f32, pass.region[3] as f32],
                        [pass.region[0] as f32, pass.region[1] as f32],
                        pass.space,
                    );
                    let g = bytemuck::bytes_of(&g);
                    entry[..g.len()].copy_from_slice(g);
                    put(&entry);
                }
            }
        });
    }

    /// Stages the frame's uploads; time spent waiting for a staging slot
    /// the GPU has not finished copying out of goes to `wait_seconds`.
    #[cfg(not(target_arch = "wasm32"))]
    fn upload_frame(
        &mut self,
        dirty: &[&SurfaceFrame<'_>],
        stats: &mut FrameStats,
    ) -> Result<(), RenderError> {
        let (size, copies) = self.upload_layout(dirty);
        if size == 0 {
            return Ok(());
        }
        self.grow_frame_buffers(&copies);
        if let upload::Acquire::Wait(submission) = self.uploads.acquire(&self.device, size)? {
            let start = Instant::now();
            self.wait(submission, "upload staging")?;
            stats.phases.wait_seconds += start.elapsed().as_secs_f64();
            self.uploads.check_mapped()?;
        }
        self.write_uploads(dirty, size, &copies);
        Ok(())
    }

    /// Stages the frame's uploads; time spent waiting for a staging slot
    /// the GPU has not finished copying out of goes to `wait_seconds`.
    #[cfg(target_arch = "wasm32")]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn upload_frame(
        &mut self,
        dirty: &[&SurfaceFrame<'_>],
        stats: &mut FrameStats,
    ) -> Result<(), RenderError> {
        let (size, copies) = self.upload_layout(dirty);
        if size == 0 {
            return Ok(());
        }
        self.grow_frame_buffers(&copies);
        if let upload::Acquire::Wait(submission) = self.uploads.acquire(&self.device, size)? {
            tracing::trace!(?submission, "awaiting the upload slot's map");
            let start = Instant::now();
            if let Some(mapped) = self.uploads.take_mapped() {
                browser_wait(mapped, self.wait_timeout, "upload staging").await?;
            }
            stats.phases.wait_seconds += start.elapsed().as_secs_f64();
            self.uploads.check_mapped()?;
        }
        self.write_uploads(dirty, size, &copies);
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        reason = "pixel sizes are well within f32"
    )]
    fn encode_surface(
        &mut self,
        id: SurfaceId,
        timing: filtrate::EffectFrameTiming,
        stats: &mut FrameStats,
    ) -> Result<(), RenderError> {
        // External pipelines stay lazy until a surface first samples an
        // external frame; the scan stays on the (unchanged) surface state.
        let needs_external = self.surfaces.get(&id).is_some_and(|surf| {
            surf.frame
                .passes
                .iter()
                .flat_map(|pass| &pass.ranges)
                .any(|range| matches!(range.image, Some(lower::ImageSource::External(_))))
        });
        if needs_external {
            self.ensure_external()?;
        }
        let Some(surf) = self.surfaces.get_mut(&id) else {
            return Ok(());
        };
        diag::set_surface(Some(id.raw()));
        // Buffers grown during lowering leave `bind0` stale; rebuild when
        // capacity changed since the bind group was built.
        if self.instances.size() > self.bound_instance_size
            || self.stops.size() > self.bound_stop_size
            || self.globals.size() > self.bound_globals_size
        {
            let atlas = &self.atlas;
            diag::bind_groups_dropped(&self.device, 1, "buffer growth");
            self.bind0 = make_bind0(
                &self.device,
                &self.layout0,
                &self.globals,
                &self.instances,
                &self.stops,
                atlas,
            );
            self.bound_atlas = atlas.generation();
            self.bound_instance_size = self.instances.size();
            self.bound_stop_size = self.stops.size();
            self.bound_globals_size = self.globals.size();
        }
        let inst_base = surf.inst_base;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        // The frame's first submission carries every dirty surface's
        // uploads ahead of its passes.
        let uploads = self.uploads.take_copies();
        if let Some((staging, copies)) = &uploads {
            for copy in copies {
                let dest = match copy.dest {
                    upload::Dest::Instances => &self.instances,
                    upload::Dest::Stops => &self.stops,
                    upload::Dest::Globals => &self.globals,
                };
                encoder.copy_buffer_to_buffer(staging, copy.src, dest, copy.dst, copy.size);
            }
        }
        // Group-1 bind groups persist across frames, keyed by
        // (source scratch, backdrop-needed, image, mask texture); the
        // stamp rebuilds them when a scratch/backdrop texture, the image
        // set, or the mask textures changed.
        let stamp = (
            surf.bind_gen,
            self.images_gen,
            self.atlas.mask_texture_generation(),
        );
        if surf.binds1_stamp != stamp {
            let dropped = surf.binds1.len() as u64;
            if dropped > 0 {
                diag::bind_groups_dropped(&self.device, dropped, "stamp change");
            }
            surf.binds1.clear();
            surf.binds1_stamp = stamp;
        }
        for (i, pass) in surf.frame.passes.iter().enumerate() {
            let (view, texture) = match pass.target {
                Target::Surface => (&surf.view, &surf.target),
                Target::Scratch(i) => (&surf.scratch[i].view, &surf.scratch[i].texture),
                Target::Backdrop { group, region } => {
                    let capture = &surf.backdrop_groups[&group].captures[region as usize];
                    (&capture.view, &capture.texture)
                }
            };
            // A backdrop-group capture first copies `pass.region` out of
            // its `copy_from` target into the group's capture texture.
            if let Some(capture) = pass.capture {
                let (src, sx, sy) = match capture.copy_from {
                    Target::Surface => (&surf.target, 0, 0),
                    Target::Scratch(k) => (&surf.scratch[k].texture, 0, 0),
                    Target::Backdrop { group, region } => {
                        let capture = &surf.backdrop_groups[&group].captures[region as usize];
                        (&capture.texture, 0, 0)
                    }
                };
                debug_assert!(pass.region[0] + pass.region[2] <= src.width());
                debug_assert!(pass.region[1] + pass.region[3] <= src.height());
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: src,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: pass.region[0].saturating_sub(sx),
                            y: pass.region[1].saturating_sub(sy),
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: pass.region[2],
                        height: pass.region[3],
                        depth_or_array_layers: 1,
                    },
                );
            }
            // A blend pass reads the target's prior contents from a copy;
            // the copy must complete before the pass starts.
            if let Some([bx, by, bw, bh]) = pass.backdrop_copy {
                let slot = match pass.target {
                    Target::Surface => 0,
                    Target::Scratch(_) => 1,
                    Target::Backdrop { .. } => {
                        return Err(RenderError::Render(
                            "a blend backdrop copy on a capture pass".into(),
                        ));
                    }
                };
                let backdrop = surf.backdrop[slot].as_ref().expect("grown above");
                // The copy region is recorded in device space; a scratch
                // target stores its contents offset by its pass region.
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: bx.saturating_sub(pass.region[0]),
                            y: by.saturating_sub(pass.region[1]),
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &backdrop.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: bw,
                        height: bh,
                        depth_or_array_layers: 1,
                    },
                );
            }
            let load = match pass.clear {
                Some([r, g, b, a]) => wgpu::LoadOp::Clear(wgpu::Color {
                    r: f64::from(r),
                    g: f64::from(g),
                    b: f64::from(b),
                    a: f64::from(a),
                }),
                None => wgpu::LoadOp::Load,
            };
            let pass_index = self.frame_pass_count;
            self.frame_pass_count += 1;
            let timestamp_writes =
                self.query_set
                    .as_ref()
                    .map(|qs| wgpu::RenderPassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(self.query_base + 2 * pass_index),
                        end_of_pass_write_index: Some(self.query_base + 2 * pass_index + 1),
                    });
            // Only the timestamp path reads `pass_meta`; skip the
            // allocation when timing is off.
            if self.timestamps {
                self.pass_meta.push(PassMeta {
                    name: match pass.target {
                        Target::Surface => "surface".to_string(),
                        Target::Scratch(i) => format!("scratch{i}"),
                        Target::Backdrop { group, region } => {
                            format!("backdrop{group}.{region}")
                        }
                    },
                    width: pass.region[2],
                    height: pass.region[3],
                    format: format_name(texture.format()),
                });
            }
            let scratch_backdrop = pass.backdrop_copy.is_some();
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
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let format_i = usize::from(texture.format() != TARGET_FORMAT);
            render_pass.set_pipeline(&self.pipelines[format_i][0][0]);
            // Region-targeted passes cover only their region; the surface
            // pass the whole target. `in.device` stays in true device
            // space via the per-pass Globals origin.
            if !matches!(pass.target, Target::Surface) {
                render_pass.set_viewport(
                    0.0,
                    0.0,
                    pass.region[2] as f32,
                    pass.region[3] as f32,
                    0.0,
                    1.0,
                );
                render_pass.set_scissor_rect(0, 0, pass.region[2], pass.region[3]);
            }
            // The uniform slot written for this pass above (256-byte
            // stride), which matches `surf.frame.passes` ordering.
            let offset = pass_index * 256;
            render_pass.set_bind_group(0, &self.bind0, &[offset]);
            // External ranges bind a slot's group-1 over the external
            // pipeline; an engine range after one must rebind its pipeline.
            let mut pipeline = Bound::Engine(PipelineKind::SrcOver, ShaderVariant::Simple);
            for range in &pass.ranges {
                stats.draws += 1;
                if let Some(lower::ImageSource::External(layer)) = &range.image {
                    if pipeline != Bound::External {
                        pipeline = Bound::External;
                        stats.pipeline_switches += 1;
                        let Some(pipe) = &self.external_pipes[format_i] else {
                            return Err(RenderError::Render("external pipeline unbuilt".into()));
                        };
                        render_pass.set_pipeline(pipe);
                    }
                    let Some(ext_layout) = &self.ext_layout else {
                        return Err(RenderError::Render("external layout unbuilt".into()));
                    };
                    let Some(slot) = surf.external.get_mut(layer) else {
                        return Err(RenderError::Render(format!(
                            "no external frame on layer {layer:?}"
                        )));
                    };
                    let bind = slot.bind(
                        &self.device,
                        ext_layout,
                        external::MaskBinding {
                            key: range.mask,
                            view: range.mask.map(|key| {
                                self.atlas
                                    .mask_texture_view(key)
                                    .expect("mask texture stored before encode")
                            }),
                            generation: self.atlas.mask_texture_generation(),
                        },
                        &self.dummy_view,
                        &self.dummy_uint_view,
                    );
                    render_pass.set_bind_group(1, bind, &[]);
                    render_pass.draw(
                        0..6,
                        (inst_base + range.instances.start)..(inst_base + range.instances.end),
                    );
                    continue;
                }
                let want = Bound::Engine(range.pipeline, range.variant);
                if want != pipeline {
                    pipeline = want;
                    stats.pipeline_switches += 1;
                    let Bound::Engine(kind, variant) = pipeline else {
                        unreachable!("engine want")
                    };
                    let pipe = match kind {
                        PipelineKind::Effect(id) => self
                            .backdrop_shaders
                            .get(&id)
                            .map(|p| &p[format_i])
                            .ok_or_else(|| {
                                RenderError::Render(format!(
                                    "backdrop shader {id} is not registered"
                                ))
                            })?,
                        kind => {
                            &self.pipelines[format_i][usize::from(kind == PipelineKind::Replace)]
                                [variant_index(variant)]
                        }
                    };
                    render_pass.set_pipeline(pipe);
                }
                let key = (
                    range.source,
                    scratch_backdrop,
                    range.image.clone(),
                    range.mask,
                );
                let bind = match surf.binds1.entry(key) {
                    std::collections::hash_map::Entry::Occupied(e) => &*e.into_mut(),
                    std::collections::hash_map::Entry::Vacant(e) => {
                        stats.bind_groups_created += 1;
                        let backdrop = if scratch_backdrop {
                            let slot = match pass.target {
                                Target::Surface => 0,
                                Target::Scratch(_) | Target::Backdrop { .. } => 1,
                            };
                            surf.backdrop[slot].as_ref().map(|b| &b.view)
                        } else {
                            None
                        };
                        &*e.insert(make_bind1(
                            &self.device,
                            &self.layout1,
                            &self.dummy_view,
                            range.source.map(|s| match s {
                                Source::Scratch(i) => &surf.scratch[i].view,
                                Source::Backdrop { group, region } => {
                                    &surf.backdrop_groups[&group].captures[region as usize].view
                                }
                            }),
                            backdrop,
                            range.image.as_ref().and_then(|source| match source {
                                lower::ImageSource::Registered(id) => {
                                    self.images.get(id).map(|image| &image.view)
                                }
                                lower::ImageSource::Bitmap(key) => {
                                    self.bitmaps.get(key).map(|bitmap| &bitmap.image.view)
                                }
                                lower::ImageSource::Shader(key) => {
                                    Some(&surf.shader_textures[key].image.view)
                                }
                                lower::ImageSource::Content(layer) => Some(
                                    &surf.content[layer]
                                        .image
                                        .as_ref()
                                        .expect("rendered content")
                                        .view,
                                ),
                                lower::ImageSource::External(_) => {
                                    unreachable!("external ranges draw with the external pipeline")
                                }
                            }),
                            range.mask.map(|k| {
                                self.atlas
                                    .mask_texture_view(k)
                                    .expect("mask texture stored before encode")
                            }),
                        ))
                    }
                };
                render_pass.set_bind_group(1, bind, &[]);
                render_pass.draw(
                    0..6,
                    (inst_base + range.instances.start)..(inst_base + range.instances.end),
                );
            }
            drop(render_pass);
            if let Some((_, parameters)) = surf.frame.shadows.iter().find(|(pass, _)| *pass == i) {
                let Target::Scratch(depth) = pass.target else {
                    unreachable!("shadow captures scratch")
                };
                self.shadow_blur.apply(
                    &self.device,
                    &mut encoder,
                    &surf.scratch[depth],
                    (pass.region[2], pass.region[3]),
                    *parameters,
                )?;
                stats.passes +=
                    u32::from(parameters.spread != 0.0) + 2 * u32::from(parameters.sigma > 0.0);
            }
            if let Some((_, filter)) = surf.frame.filters.iter().find(|(pass, _)| *pass == i) {
                let capture = match pass.target {
                    Target::Scratch(depth) => &surf.scratch[depth],
                    Target::Backdrop { group, region } => {
                        &surf.backdrop_groups[&group].captures[region as usize]
                    }
                    Target::Surface => {
                        return Err(RenderError::Render(format!(
                            "filter {filter:?} registered on a surface pass"
                        )));
                    }
                };
                self.filters.apply(
                    *filter,
                    &filtrate::EffectContext {
                        device: &self.device,
                        queue: &self.queue,
                        input_format: TARGET_FORMAT,
                        output_format: TARGET_FORMAT,
                    },
                    capture,
                    (pass.region[2], pass.region[3]),
                    timing,
                    &mut encoder,
                )?;
            }
        }
        // Producer sync: each external frame's `wait` event becomes a
        // raw Metal command buffer committed ahead of the frame's, so the
        // GPU blocks in-queue — no CPU wait and no copy. Commit order on
        // the queue orders the wait before wgpu's own command buffer.
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _};
            for layer in &surf.frame.external {
                let Some(crate::interop::FrameSync::Metal { event, value }) = surf
                    .external
                    .get(layer)
                    .and_then(|slot| slot.frame.wait.as_ref())
                else {
                    continue;
                };
                let hal_queue = unsafe { self.queue.as_hal::<wgpu::hal::metal::Api>() }
                    .expect("the engine queue is Metal");
                let buffer = hal_queue
                    .as_raw()
                    .commandBuffer()
                    .expect("Metal command buffer");
                buffer.encodeWaitForEvent_value(event.as_ref(), *value);
                buffer.commit();
            }
        }
        let submission = self.queue.submit([encoder.finish()]);
        if uploads.is_some() {
            self.uploads.submitted(submission.clone());
        }
        diag::submit(&self.device, &self.queue, "frame");
        self.frame_submission = Some(submission.clone());
        tracing::trace!(
            surface = ?id,
            passes = surf.frame.passes.len(),
            instances = surf.frame.instances.len(),
            ?submission,
            "surface submitted"
        );
        stats.passes += u32::try_from(surf.frame.passes.len()).unwrap_or(u32::MAX);
        stats.instances += u32::try_from(surf.frame.instances.len()).unwrap_or(u32::MAX);
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(
        &self,
        submission: wgpu::SubmissionIndex,
        what: &'static str,
    ) -> Result<(), RenderError> {
        let start = Instant::now();
        let status = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(self.wait_timeout),
        });
        diag::poll(&self.device, status.is_ok());
        let elapsed = start.elapsed();
        match status {
            Ok(status) => {
                tracing::trace!(
                    what,
                    ?status,
                    wait_ms = elapsed.as_secs_f64() * 1e3,
                    "waited"
                );
                Ok(())
            }
            Err(wgpu::PollError::Timeout) => {
                tracing::error!(what, ?elapsed, "GPU wait timed out");
                Err(RenderError::Timeout {
                    what,
                    timeout: self.wait_timeout,
                })
            }
            Err(e) => {
                tracing::error!(what, %e, "GPU wait failed");
                Err(RenderError::DeviceLost)
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn wait(
        &self,
        _submission: wgpu::SubmissionIndex,
        what: &'static str,
    ) -> Result<(), RenderError> {
        let (tx, rx) = futures_channel::oneshot::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = tx.send(());
        });
        browser_wait(rx, self.wait_timeout, what).await
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn map_read(
        &self,
        slice: wgpu::BufferSlice<'_>,
        submission: wgpu::SubmissionIndex,
        what: &'static str,
    ) -> Result<(), RenderError> {
        let (tx, rx) = std::sync::mpsc::channel();
        tracing::trace!(what, "map requested");
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        diag::map(&self.device, what, 0);
        self.wait(submission, what)?;
        match rx.try_recv() {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(RenderError::Readback(format!("{what}: map failed: {e}"))),
            Err(_) => Err(RenderError::Readback(format!(
                "{what}: the map callback did not run after the wait"
            ))),
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn map_read(
        &self,
        slice: wgpu::BufferSlice<'_>,
        _submission: wgpu::SubmissionIndex,
        what: &'static str,
    ) -> Result<(), RenderError> {
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        browser_wait(rx, self.wait_timeout, what)
            .await?
            .map_err(|error| RenderError::Readback(format!("{what}: {error}")))
    }

    /// Hands this frame's samples to the completion queue without waiting.
    fn queue_timestamps(&mut self, count: u32, frame: FrameId) {
        let query_set = self.query_set.take().expect("a timed frame owns queries");
        let submission = self.frame_submission.clone().expect("a timed frame drew");
        let complete = Arc::new(AtomicU8::new(0));
        let flag = Arc::clone(&complete);
        self.queue.on_submitted_work_done(move || {
            flag.store(1, Ordering::Release);
        });
        self.pending_queries.push_back(PendingQueries {
            frame,
            submission,
            query_set,
            base: self.query_base,
            capacity: self.query_capacity,
            count,
            meta: std::mem::take(&mut self.pass_meta),
            complete,
        });
    }

    /// Encodes the resolve only once the frame's samples are complete.
    fn resolve_timestamps(&mut self, pending: PendingQueries) {
        let staging = self.timestamp_staging();
        let buf = self
            .query_buffer
            .as_ref()
            .expect("timing has a resolve buffer");
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("timestamp resolve"),
            });
        encoder.resolve_query_set(
            &pending.query_set,
            pending.base..pending.base + pending.count,
            buf,
            0,
        );
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, u64::from(pending.count) * 8);
        let submission = self.queue.submit([encoder.finish()]);
        diag::submit(&self.device, &self.queue, "timestamp resolve");
        tracing::trace!(
            frame = pending.frame.get(),
            count = pending.count,
            ?submission,
            "timestamps resolved after draw completion"
        );
        self.pending_timestamps.push_back(PendingTimestamps {
            query_set: pending.query_set,
            query_base: pending.base,
            query_capacity: pending.capacity,
            frame: pending.frame,
            submission,
            staging,
            count: pending.count,
            meta: pending.meta,
            map_requested: false,
            ready: Arc::new(AtomicU8::new(0)),
        });
    }

    fn timestamp_staging(&mut self) -> wgpu::Buffer {
        let size = u64::from(self.query_capacity) * 8;
        self.query_staging.retain(|buffer| buffer.size() >= size);
        self.query_staging.pop().unwrap_or_else(|| {
            diag::create(&self.device, "timestamp staging", size);
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp staging"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        })
    }

    /// Reads back every pending frame's resolved timestamps whose copy
    /// has landed, oldest first — submissions complete in order, so the
    /// first unfinished one ends the drain. Never blocks.
    fn drain_timestamps(&mut self) {
        if !self.pending_queries.is_empty() {
            // Poll dispatches completion callbacks; it never waits for GPU idle.
            let _ = self.device.poll(wgpu::PollType::Poll);
            diag::poll(&self.device, false);
            while self
                .pending_queries
                .front()
                .is_some_and(|pending| pending.complete.load(Ordering::Acquire) != 0)
            {
                let pending = self.pending_queries.pop_front().expect("checked above");
                self.resolve_timestamps(pending);
            }
        }
        let period = f64::from(self.queue.get_timestamp_period());
        while let Some(pending) = self.pending_timestamps.front_mut() {
            pending.request_map();
            diag::map(
                &self.device,
                "timestamp staging",
                u64::from(pending.count) * 8,
            );
            if self.device.poll(wgpu::PollType::Poll).is_err() {
                break;
            }
            diag::poll(&self.device, false);
            match pending.ready.load(Ordering::Relaxed) {
                // A failed map drops the frame's timing instead of
                // blocking every later drain.
                2 => {
                    tracing::error!(
                        frame = pending.frame.get(),
                        "the timestamp readback map failed"
                    );
                    if let Some(pending) = self.pending_timestamps.pop_front() {
                        pending.staging.unmap();
                    }
                    continue;
                }
                1 => {}
                _ => break,
            }
            let Some(pending) = self.pending_timestamps.pop_front() else {
                break;
            };
            let timing = {
                let data = pending
                    .staging
                    .slice(..u64::from(pending.count) * 8)
                    .get_mapped_range()
                    .expect("buffer range is mapped and not overlapping");
                let ticks: &[u64] = bytemuck::cast_slice(&data);
                tracing::trace!(
                    frame = pending.frame.get(),
                    period,
                    ?ticks,
                    "timestamp ticks"
                );
                #[expect(clippy::cast_precision_loss)]
                let delta = |from: usize, to: usize| {
                    ticks
                        .get(to)
                        .zip(ticks.get(from))
                        .filter(|(end, start)| end > start)
                        .map(|(end, start)| period * (end - start) as f64 * 1e-9)
                };
                FrameTiming {
                    frame: pending.frame,
                    // The frame's GPU time runs from the first pass's
                    // start to the last pass's end.
                    gpu_seconds: delta(0, pending.count as usize - 1),
                    passes: pending
                        .meta
                        .into_iter()
                        .enumerate()
                        .map(|(i, meta)| PassTiming {
                            name: meta.name,
                            width: meta.width,
                            height: meta.height,
                            format: meta.format,
                            gpu_seconds: delta(2 * i, 2 * i + 1),
                        })
                        .collect(),
                }
            };
            tracing::debug!(
                frame = timing.frame.get(),
                passes = timing.passes.len(),
                gpu_ms = timing.gpu_seconds.map(|s| s * 1e3),
                "frame timed"
            );
            self.timings.push(timing);
            pending.staging.unmap();
            if pending.query_capacity >= self.query_capacity {
                self.query_pool.push((
                    pending.query_set,
                    pending.query_base,
                    pending.query_capacity,
                ));
            }
            if pending.staging.size() >= u64::from(self.query_capacity) * 8 {
                self.query_staging.push(pending.staging);
            }
        }
    }

    /// Acquires a frame's queries and grows the resolve buffer if needed.
    /// All surfaces are lowered before encoding, so replacing an undersized
    /// active set here cannot discard samples already written this frame.
    fn ensure_query_capacity(&mut self, queries: u32) {
        if queries <= self.query_capacity && self.query_set.is_some() {
            return;
        }
        let capacity = queries.next_power_of_two().max(2).max(self.query_capacity);
        if let Some(index) = self
            .query_pool
            .iter()
            .position(|(_, _, size)| *size == capacity)
        {
            let (set, base, _) = self.query_pool.swap_remove(index);
            self.query_set = Some(set);
            self.query_base = base;
        } else {
            let slots = (wgpu::QUERY_SET_MAX_QUERIES / capacity).clamp(1, TIMESTAMP_FRAMES_PER_SET);
            let set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("frame timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: capacity * slots,
            });
            diag::create(&self.device, "frame timestamps", 0);
            self.query_pool
                .extend((1..slots).map(|slot| (set.clone(), slot * capacity, capacity)));
            self.query_set = Some(set);
            self.query_base = 0;
            tracing::debug!(capacity, slots, "timestamp query ranges allocated");
        }
        if capacity > self.query_capacity {
            let old = self.query_buffer.as_ref().map_or(0, wgpu::Buffer::size);
            self.query_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp resolve"),
                size: u64::from(capacity) * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            diag::grow(
                &self.device,
                "timestamp resolve",
                diag::Class::Query,
                old,
                u64::from(capacity) * 8,
                0,
                true,
            );
            self.query_capacity = capacity;
            self.query_pool.retain(|(_, _, size)| *size >= capacity);
        }
    }
}

/// A registered image's texture of `size`, holding `data` (f16 texels
/// from [`image_texels_f16`]).
fn upload_image(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    (width, height): (u32, u32),
    data: &[u8],
) -> GpuImage {
    let (texture, view) = create_target(
        device,
        "image",
        (width, height),
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        TARGET_FORMAT,
    );
    diag::create(
        device,
        "image",
        u64::from(width) * u64::from(height) * texel_bytes(TARGET_FORMAT),
    );
    write_image(device, queue, &texture, (width, height), data);
    GpuImage {
        texture,
        view,
        width,
        height,
    }
}

/// Writes `data` (f16 texels from [`image_texels_f16`]) over the whole of
/// an image texture of `size`.
fn write_image(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    (width, height): (u32, u32),
    data: &[u8],
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 8),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    diag::upload(
        device,
        "image",
        data.len() as u64,
        Some((0, 0, width, height)),
    );
}

/// `Rgba8` or `Rgba16F` upload bytes -> premultiplied linear-P3 f16 texels,
/// the working texel format of [`GpuImage`].
///
/// Straight-alpha input decodes each channel; premultiplied input is
/// un-premultiplied in the encoded domain first — bounded at 1.0 for the
/// quantized `Rgba8` encoding, unbounded for `Rgba16F`, whose texels keep
/// extended (HDR and wide-gamut) values.
fn image_texels_f16(image: &ImageUpload) -> Result<Vec<u8>, ResourceError> {
    let convert = |enc: [f64; 4], unpremul_max: f64, data: &mut Vec<u8>| {
        let a = enc[3];
        // Straight-alpha input decodes each channel; premultiplied input
        // is un-premultiplied in the encoded domain first.
        let decode = |v: f64| {
            if image.premultiplied && a > 0.0 {
                (v / a).min(unpremul_max)
            } else {
                v
            }
        };
        let lin = match image.color_space {
            cherenkov::ImageColorSpace::LinearSrgb | cherenkov::ImageColorSpace::LinearP3 => {
                [decode(enc[0]), decode(enc[1]), decode(enc[2])]
            }
            _ => [
                srgb_decode_u8_f64(decode(enc[0])),
                srgb_decode_u8_f64(decode(enc[1])),
                srgb_decode_u8_f64(decode(enc[2])),
            ],
        };
        // sRGB-primaries input additionally needs the primaries' matrix;
        // Display P3 uses sRGB's transfer function, so the decode above
        // covers both encoded spaces. `LinearP3` is already the working
        // space: no transfer, no matrix.
        let lin_p3 = match image.color_space {
            cherenkov::ImageColorSpace::Srgb | cherenkov::ImageColorSpace::LinearSrgb => {
                let [x, y, z] = [
                    SRGB_TO_XYZ[0][2].mul_add(
                        lin[2],
                        SRGB_TO_XYZ[0][1].mul_add(lin[1], SRGB_TO_XYZ[0][0] * lin[0]),
                    ),
                    SRGB_TO_XYZ[1][2].mul_add(
                        lin[2],
                        SRGB_TO_XYZ[1][1].mul_add(lin[1], SRGB_TO_XYZ[1][0] * lin[0]),
                    ),
                    SRGB_TO_XYZ[2][2].mul_add(
                        lin[2],
                        SRGB_TO_XYZ[2][1].mul_add(lin[1], SRGB_TO_XYZ[2][0] * lin[0]),
                    ),
                ];
                [
                    XYZ_TO_P3[0][2].mul_add(z, XYZ_TO_P3[0][1].mul_add(y, XYZ_TO_P3[0][0] * x)),
                    XYZ_TO_P3[1][2].mul_add(z, XYZ_TO_P3[1][1].mul_add(y, XYZ_TO_P3[1][0] * x)),
                    XYZ_TO_P3[2][2].mul_add(z, XYZ_TO_P3[2][1].mul_add(y, XYZ_TO_P3[2][0] * x)),
                ]
            }
            cherenkov::ImageColorSpace::DisplayP3 | cherenkov::ImageColorSpace::LinearP3 => lin,
        };
        for v in [a * lin_p3[0], a * lin_p3[1], a * lin_p3[2], a] {
            data.extend_from_slice(&half::f16::from_f64(v).to_le_bytes());
        }
    };
    let mut data = Vec::with_capacity(image.width as usize * image.height as usize * 8);
    let pixels: &[u8] = &image.data;
    match image.format {
        cherenkov::ImageFormat::Rgba8 => {
            for px in pixels.as_chunks::<4>().0 {
                convert(px.map(|v| f64::from(v) / 255.0), 1.0, &mut data);
            }
        }
        cherenkov::ImageFormat::Rgba16F => {
            for px in pixels.as_chunks::<8>().0 {
                let enc = std::array::from_fn(|i| {
                    f64::from(half::f16::from_le_bytes([px[2 * i], px[2 * i + 1]]))
                });
                convert(enc, f64::INFINITY, &mut data);
            }
        }
        format => {
            return Err(ResourceError::Image(format!(
                "unsupported image format {format:?}"
            )));
        }
    }
    Ok(data)
}

fn image_texels(
    pixels: &[u8],
    color_space: cherenkov::ImageColorSpace,
    premultiplied: bool,
) -> Vec<u8> {
    let mut data = Vec::with_capacity(pixels.len() * 2);
    for px in pixels.as_chunks::<4>().0 {
        let a = f64::from(px[3]) / 255.0;
        let decode = |v: u8| {
            if premultiplied && a > 0.0 {
                ((f64::from(v) / 255.0) / a).min(1.0)
            } else {
                f64::from(v) / 255.0
            }
        };
        let lin = match color_space {
            cherenkov::ImageColorSpace::LinearSrgb | cherenkov::ImageColorSpace::LinearP3 => {
                [decode(px[0]), decode(px[1]), decode(px[2])]
            }
            _ => [
                srgb_decode_u8_f64(decode(px[0])),
                srgb_decode_u8_f64(decode(px[1])),
                srgb_decode_u8_f64(decode(px[2])),
            ],
        };
        let lin_p3 = match color_space {
            cherenkov::ImageColorSpace::Srgb | cherenkov::ImageColorSpace::LinearSrgb => {
                let [x, y, z] = [
                    SRGB_TO_XYZ[0][2].mul_add(
                        lin[2],
                        SRGB_TO_XYZ[0][1].mul_add(lin[1], SRGB_TO_XYZ[0][0] * lin[0]),
                    ),
                    SRGB_TO_XYZ[1][2].mul_add(
                        lin[2],
                        SRGB_TO_XYZ[1][1].mul_add(lin[1], SRGB_TO_XYZ[1][0] * lin[0]),
                    ),
                    SRGB_TO_XYZ[2][2].mul_add(
                        lin[2],
                        SRGB_TO_XYZ[2][1].mul_add(lin[1], SRGB_TO_XYZ[2][0] * lin[0]),
                    ),
                ];
                [
                    XYZ_TO_P3[0][2].mul_add(z, XYZ_TO_P3[0][1].mul_add(y, XYZ_TO_P3[0][0] * x)),
                    XYZ_TO_P3[1][2].mul_add(z, XYZ_TO_P3[1][1].mul_add(y, XYZ_TO_P3[1][0] * x)),
                    XYZ_TO_P3[2][2].mul_add(z, XYZ_TO_P3[2][1].mul_add(y, XYZ_TO_P3[2][0] * x)),
                ]
            }
            cherenkov::ImageColorSpace::DisplayP3 | cherenkov::ImageColorSpace::LinearP3 => lin,
        };
        for v in [a * lin_p3[0], a * lin_p3[1], a * lin_p3[2], a] {
            data.extend_from_slice(&half::f16::from_f64(v).to_le_bytes());
        }
    }
    data
}

fn create_gpu_image(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    width: u32,
    height: u32,
    texels: &[u8],
) -> GpuImage {
    let (texture, view) = create_target(
        device,
        label,
        (width, height),
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        TARGET_FORMAT,
    );
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 8),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    GpuImage {
        texture,
        view,
        width,
        height,
    }
}

fn srgb_decode_u8_f64(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// A browser completion with the same configured timeout as native waits.
#[cfg(target_arch = "wasm32")]
#[expect(
    clippy::future_not_send,
    reason = "the browser engine is single-threaded and its futures run on the page's event loop"
)]
async fn browser_wait<T>(
    rx: futures_channel::oneshot::Receiver<T>,
    timeout: std::time::Duration,
    what: &'static str,
) -> Result<T, RenderError> {
    use futures_util::future::{Either, select};
    let millis = u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX);
    match select(
        Box::pin(rx),
        Box::pin(gloo_timers::future::TimeoutFuture::new(millis)),
    )
    .await
    {
        Either::Left((Ok(value), _)) => Ok(value),
        Either::Left((Err(_), _)) => Err(RenderError::DeviceLost),
        Either::Right(_) => Err(RenderError::Timeout { what, timeout }),
    }
}

#[cfg(target_arch = "wasm32")]
impl GpuRenderer {
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn prepare_filters(&mut self, surface: SurfaceId) -> Result<(), RenderError> {
        let surface = &self.surfaces[&surface];
        let uses: Vec<_> = surface
            .frame
            .filters
            .iter()
            .map(|(pass, id)| {
                let format = match surface.frame.passes[*pass].target {
                    Target::Scratch(depth) => surface.scratch[depth].texture.format(),
                    Target::Backdrop { group, region } => surface.backdrop_groups[&group].captures
                        [region as usize]
                        .texture
                        .format(),
                    Target::Surface => TARGET_FORMAT,
                };
                (*id, format)
            })
            .collect();
        for (id, format) in uses {
            self.filters
                .prepare(
                    id,
                    &filtrate::EffectContext {
                        device: &self.device,
                        queue: &self.queue,
                        input_format: format,
                        output_format: format,
                    },
                )
                .await?;
        }
        Ok(())
    }
}
