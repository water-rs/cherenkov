// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The render thread: sole owner of GPU state.

mod colr;
mod glyph;
mod instance;
mod lower;
mod path;
mod raster;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

use crate::{GpuConfig, GpuInfo, GpuTarget, ScratchFormat, TimestampSupport, names};
use cherenkov::{
    ContentOp, EngineError, FontData as EngineFontData, FontId, Frame, FrameStats, ImageId,
    ImageUpload, LayerId, MemoryUsage, PassTiming, Pressure, Readback, Redraw, RenderError,
    Renderer, ResourceError, SurfaceError, SurfaceFrame, SurfaceId, SurfaceInfo,
};
use glyph::{Atlas, FontData, PendingRaster};
use lower::{
    ContentData, Frame as LoweredFrame, GlyphContext, Lowered, Lowering, PipelineKind,
    ShaderVariant, Target,
};

/// The surface target format: premultiplied linear Display P3.
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

const TARGET_USAGES: wgpu::TextureUsages = wgpu::TextureUsages::from_bits_retain(
    wgpu::TextureUsages::RENDER_ATTACHMENT.bits()
        | wgpu::TextureUsages::COPY_SRC.bits()
        | wgpu::TextureUsages::TEXTURE_BINDING.bits(),
);

/// Decodes one sRGB-encoded byte channel to linear, `u8 → f64`.
fn srgb_decode_u8(c: u8) -> f64 {
    let v = f64::from(c) / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

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
    #[expect(dead_code, reason = "the texture keeps the view alive")]
    pub texture: wgpu::Texture,
    /// Its view for bind group 1.
    pub view: wgpu::TextureView,
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
}

/// One surface's GPU-side state.
struct SurfaceState {
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
    layers: HashMap<LayerId, ContentData>,
    frame: LoweredFrame,
    /// This frame's offsets into the shared buffers: instances and globals
    /// (256-byte slots) are laid out surface by surface so one upload covers
    /// every dirty surface.
    inst_base: u32,
    globals_base: u32,
    /// Bumped whenever a scratch or backdrop texture is (re)created — a
    /// cached group-1 bind group referencing the old view must rebuild.
    bind_gen: u64,
    /// Group-1 bind groups keyed by `(source scratch, backdrop, image)`,
    /// reused across frames while `binds1_stamp` is current.
    binds1: HashMap<(Option<usize>, bool, Option<u64>), wgpu::BindGroup>,
    /// The `(bind_gen, images_gen)` pair `binds1` was built under.
    binds1_stamp: (u64, u64),
}

impl SurfaceState {
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
        surface_bytes + scratch_bytes + backdrop_bytes
    }
}

/// All render-thread state.
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// `[format index][pipeline kind][shader variant]`:
    /// format 0 = surface, 1 = scratch; kind 0 = source-over, 1 = replace;
    /// variant 0/1/2 = simple/shadow/full fragment shader.
    pipelines: [[[wgpu::RenderPipeline; 3]; 2]; 2],
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
    /// A dummy 1×1 view for unused group-1 slots.
    dummy_view: wgpu::TextureView,
    surfaces: HashMap<SurfaceId, SurfaceState>,
    fonts: HashMap<u64, FontData>,
    /// Registered images.
    images: HashMap<u64, GpuImage>,
    /// Bumped on every `images` insert/remove — every cached group-1
    /// bind group samples an image view, so an image change rebuilds them.
    images_gen: u64,
    timestamps: bool,
    query_set: Option<wgpu::QuerySet>,
    query_buffer: Option<wgpu::Buffer>,
    /// A spare staging buffer recycled between frames; `None` while a
    /// frame's resolve owns one.
    query_staging: Option<wgpu::Buffer>,
    /// The query set's capacity in queries; pass `i` writes `2i` at its
    /// start and `2i + 1` at its end. The frame's GPU time runs from the
    /// first pass's start to the last pass's end: pass boundaries are the
    /// one timestamp position every backend with `TIMESTAMP_QUERY`
    /// supports (Metal on Apple GPUs samples only at stage boundaries).
    query_capacity: u32,
    /// Frames whose timestamp resolve was submitted but whose staging
    /// buffer is not mapped yet — read on a later call, in order.
    pending_timestamps: VecDeque<PendingTimestamps>,
    /// Passes encoded this frame, for the per-pass report.
    frame_pass_count: u32,
    /// `(name, width, height, format)` of each encoded pass this frame.
    pass_meta: Vec<PassMeta>,
    /// Bound on every GPU wait; see [`GpuConfig::wait_timeout`].
    wait_timeout: Duration,
    max_texture: u32,
}

/// Slot updates addressed to a layer with no live list — without the
enum PendingOrigin {
    /// Cell origins: one for a glyph, one per cell for a path emission.
    Cells(Vec<(u32, u32)>),
    /// A clip mask's cell origin.
    Mask([f32; 2]),
    /// A COLR cache insert; no instance patch.
    None,
}

/// One submitted frame's timestamp queries awaiting GPU completion.
///
/// The resolve and the copy into `staging` are submitted with the frame;
/// the map and read happen on a later [`Renderer::render_frame`] once a
/// non-blocking poll reports the copy done, so rendering never stalls on
/// GPU idle.
struct PendingTimestamps {
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

/// One encoded pass's report metadata.
struct PassMeta {
    name: String,
    width: u32,
    height: u32,
    format: &'static str,
}

/// Recreates `old` at `size` bytes, preserving the first `preserve` bytes
/// of its contents. Earlier dirty surfaces write their slices before a
/// later surface grows a shared buffer; dropping the old buffer would lose
/// those uploads, so the used prefix is copied over in a separate
/// submission — queue order guarantees the earlier `write_buffer`s landed
/// in `old` first. `usage` must include `COPY_SRC` and `COPY_DST`.
fn grow_preserving(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    old: &wgpu::Buffer,
    size: u64,
    usage: wgpu::BufferUsages,
    preserve: u64,
) -> wgpu::Buffer {
    let new = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    });
    if preserve > 0 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("buffer grow"),
        });
        encoder.copy_buffer_to_buffer(old, 0, &new, 0, preserve);
        let submission = queue.submit([encoder.finish()]);
        tracing::debug!(
            label,
            from = old.size(),
            to = size,
            preserve,
            ?submission,
            "buffer grown"
        );
    } else {
        tracing::debug!(label, from = old.size(), to = size, "buffer grown");
    }
    new
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
        .ok_or(EngineError::Backend("no adapter".into()))?;
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
                    // Each pass binds its own 256-byte-aligned Globals.
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(16),
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
        // 0: composite source, 1: blend backdrop, 2: image paint.
        entries: &[texture_entry(0), texture_entry(1), texture_entry(2)],
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
                // One 16-byte Globals window; the dynamic offset selects
                // the pass's slot inside the buffer.
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: globals,
                    offset: 0,
                    size: wgpu::BufferSize::new(16),
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

/// The render-thread entry point: initializes, replies, then loops over
/// messages until [`Message::Shutdown`].
#[expect(
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    reason = "moved into the render thread"
)]
pub fn init(config: GpuConfig) -> Result<(GpuRenderer, GpuInfo), EngineError> {
    create_device(&config).and_then(|(adapter, device, queue)| {
        let info = adapter.get_info();
        let supported = adapter.features();
        let timestamp_support =
            if supported.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS) {
                TimestampSupport::Encoders
            } else if supported.contains(wgpu::Features::TIMESTAMP_QUERY) {
                TimestampSupport::PassBoundaries
            } else {
                TimestampSupport::Unsupported
            };
        let (layout0, layout1) = create_layouts(&device);
        let scratch_format = scratch_wgpu(config.scratch_format);
        // Three specialised fragment shaders from one source file: the
        // prepended `VARIANT` constant makes fs_main a constant-folded
        // dispatch to fs_simple/fs_shadow/fs_full.
        let modules = [0u32, 1, 2].map(|v| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cherenkov"),
                source: wgpu::ShaderSource::Wgsl(
                    format!(
                        "const VARIANT: u32 = {v}u;\n{}",
                        include_str!("shader.wgsl")
                    )
                    .into(),
                ),
            })
        });
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
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: 272 * 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stops = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stops"),
            size: 32 * 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let atlas = Atlas::new(&device, config.budget.gpu.0);
        let bind0 = make_bind0(&device, &layout0, &globals, &instances, &stops, &atlas);
        let (_, dummy_view) = create_target(
            &device,
            "dummy source",
            (1, 1),
            wgpu::TextureUsages::TEXTURE_BINDING,
            TARGET_FORMAT,
        );
        let timestamps = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
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
        // The initial query set holds one pass's two queries.
        let query_capacity = if query_set.is_some() { 2 } else { 0 };
        let renderer = GpuRenderer {
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
            bound_atlas: 0,
            bound_instance_size: 272 * 16,
            bound_stop_size: 32 * 16,
            bound_globals_size: 16,
            atlas,
            surfaces: HashMap::new(),
            fonts: HashMap::new(),
            images: HashMap::new(),
            images_gen: 0,
            timestamps,
            query_set,
            query_buffer,
            query_staging: None,
            query_capacity,
            pending_timestamps: VecDeque::new(),
            frame_pass_count: 0,
            pass_meta: Vec::new(),
            wait_timeout: config.wait_timeout,
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

/// Validates font data with `skrifa`, rejecting bitmap-only colour fonts.
///
/// `COLR` fonts render through the colour-glyph lowering; fonts carrying
/// `CBDT`/`CBLC` or `sbix` bitmaps without outline glyphs cannot
/// rasterize.
fn validate_font(data: &[u8], index: u32) -> Result<(), ResourceError> {
    use skrifa::MetadataProvider as _;
    use skrifa::raw::TableProvider as _;
    let font = skrifa::FontRef::from_index(data, index)
        .map_err(|e| ResourceError::Font(format!("{e}")))?;
    if font.outline_glyphs().iter().next().is_none()
        && [skrifa::Tag::new(b"CBDT"), skrifa::Tag::new(b"sbix")]
            .iter()
            .any(|tag| font.data_for_tag(*tag).is_some())
    {
        return Err(ResourceError::Unsupported(names::COLOR_FONT));
    }
    Ok(())
}

impl Renderer for GpuRenderer {
    type Target = GpuTarget;
    fn create_surface(
        &mut self,
        id: SurfaceId,
        target: GpuTarget,
    ) -> Result<SurfaceInfo, SurfaceError> {
        let GpuTarget::Offscreen(offscreen) = target;
        let size = offscreen.size;
        // The target is always Rgba16Float; both offscreen formats are
        // accepted and readback decodes f16.
        let _ = offscreen.format;
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
        let (target, view) = create_target(
            &self.device,
            "surface target",
            size,
            TARGET_USAGES,
            TARGET_FORMAT,
        );
        self.surfaces.insert(
            id,
            SurfaceState {
                size,
                scratch_format: self.scratch_format,
                target,
                view,
                scratch: Vec::new(),
                backdrop: [None, None],
                layers: HashMap::new(),
                frame: LoweredFrame::default(),
                inst_base: 0,
                globals_base: 0,
                bind_gen: 0,
                binds1: HashMap::new(),
                binds1_stamp: (u64::MAX, u64::MAX),
            },
        );
        Ok(SurfaceInfo {
            size,
            readable: true,
        })
    }

    fn resize_surface(&mut self, id: SurfaceId, size: (u32, u32)) {
        let Some(state) = self.surfaces.get_mut(&id) else {
            return;
        };
        let (target, view) = create_target(
            &self.device,
            "surface target",
            size,
            TARGET_USAGES,
            TARGET_FORMAT,
        );
        state.size = size;
        state.target = target;
        state.view = view;
        state.scratch.clear();
        state.backdrop = [None, None];
        state.binds1.clear();
        state.bind_gen += 1;
    }

    fn destroy_surface(&mut self, id: SurfaceId) {
        self.surfaces.remove(&id);
    }

    fn add_font(&mut self, id: FontId, font: EngineFontData) -> Result<(), ResourceError> {
        validate_font(&font.data, font.index)?;
        self.fonts.insert(
            id.raw(),
            FontData {
                data: font.data,
                index: font.index,
                colr: std::cell::RefCell::new(HashMap::new()),
            },
        );
        Ok(())
    }

    fn remove_font(&mut self, id: FontId) {
        self.fonts.remove(&id.raw());
        self.atlas.remove_font(id.raw());
    }

    fn set_content(&mut self, surface: SurfaceId, layer: LayerId, content: Option<ContentOp>) {
        let Some(state) = self.surfaces.get_mut(&surface) else {
            return;
        };
        match content {
            Some(ContentOp::Replace(list)) => {
                state.layers.insert(layer, ContentData::List(list));
            }
            Some(ContentOp::Update(updates)) => {
                if let Some(ContentData::List(list)) = state.layers.get_mut(&layer) {
                    let _ = list.apply(updates);
                }
            }
            Some(ContentOp::Picture(picture)) => {
                state.layers.insert(layer, ContentData::Picture(picture));
            }
            None => {
                state.layers.remove(&layer);
            }
        }
    }

    fn remove_layer(&mut self, surface: SurfaceId, layer: LayerId) {
        if let Some(state) = self.surfaces.get_mut(&surface) {
            state.layers.remove(&layer);
        }
    }

    fn add_image(&mut self, id: ImageId, image: ImageUpload) -> Result<(), ResourceError> {
        if image.format != cherenkov::ImageFormat::Rgba8 {
            return Err(ResourceError::Image(format!(
                "unsupported image format {:?}",
                image.format
            )));
        }
        let (width, height) = (image.width, image.height);
        let pixels: &[u8] = &image.data;
        let (texture, view) = create_target(
            &self.device,
            "image",
            (width, height),
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            TARGET_FORMAT,
        );
        let mut data = Vec::with_capacity(pixels.len() * 2);
        for px in pixels.as_chunks::<4>().0 {
            let a = f64::from(px[3]) / 255.0;
            // Straight-alpha input decodes each channel; premultiplied
            // input is un-premultiplied in the encoded domain first.
            let decode = |v: u8| {
                if image.premultiplied && a > 0.0 {
                    ((f64::from(v) / 255.0) / a).min(1.0)
                } else {
                    f64::from(v) / 255.0
                }
            };
            let lin = match image.color_space {
                cherenkov::ImageColorSpace::LinearSrgb => {
                    [decode(px[0]), decode(px[1]), decode(px[2])]
                }
                _ => [
                    srgb_decode_u8_f64(decode(px[0])),
                    srgb_decode_u8_f64(decode(px[1])),
                    srgb_decode_u8_f64(decode(px[2])),
                ],
            };
            // sRGB-primaries input additionally needs the primaries'
            // matrix; Display P3 uses sRGB's transfer function, so the
            // decode above covers both encoded spaces.
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
                cherenkov::ImageColorSpace::DisplayP3 => lin,
            };
            for v in [a * lin_p3[0], a * lin_p3[1], a * lin_p3[2], a] {
                data.extend_from_slice(&half::f16::from_f64(v).to_le_bytes());
            }
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
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
        self.images.insert(
            id.raw(),
            GpuImage {
                texture,
                view,
                width,
                height,
            },
        );
        self.images_gen += 1;
        Ok(())
    }

    fn remove_image(&mut self, id: ImageId) {
        self.images.remove(&id.raw());
        self.images_gen += 1;
    }

    fn trim(&mut self, pressure: Pressure) {
        for surf in self.surfaces.values_mut() {
            surf.scratch.clear();
            surf.backdrop = [None, None];
            // The bind groups' views died with the textures.
            surf.binds1.clear();
            surf.bind_gen += 1;
        }
        if pressure != Pressure::Critical {
            return;
        }
        self.atlas.clear();
        for font in self.fonts.values() {
            font.colr.borrow_mut().clear();
        }
        for surf in self.surfaces.values_mut() {
            surf.frame.instances.shrink_to_fit();
            surf.frame.stops.shrink_to_fit();
            surf.frame.passes.shrink_to_fit();
        }
        self.instances = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: 272 * 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.stops = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stops"),
            size: 32 * 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.globals = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.bound_instance_size = self.instances.size();
        self.bound_stop_size = self.stops.size();
        self.bound_globals_size = self.globals.size();
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
    }

    fn memory(&self) -> MemoryUsage {
        let gpu = self.instances.size()
            + self.stops.size()
            + self.globals.size()
            + self.atlas.gpu_bytes()
            + self
                .surfaces
                .values()
                .map(SurfaceState::gpu_bytes)
                .sum::<u64>()
            + self
                .images
                .values()
                .map(|i| u64::from(i.width) * u64::from(i.height) * 8)
                .sum::<u64>();
        MemoryUsage {
            gpu: cherenkov::Bytes(gpu),
            cpu: cherenkov::Bytes(self.atlas.cpu_bytes()),
        }
    }

    fn render(&mut self, frame: &Frame<'_>, stats: &mut FrameStats) -> Result<Redraw, RenderError> {
        self.drain_timestamps(stats);
        let dirty: Vec<_> = frame.surfaces.iter().filter(|sf| sf.changed).collect();
        if dirty.is_empty() {
            return Ok(Redraw::None);
        }
        self.frame_pass_count = 0;
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
        stats.phases.lower_seconds = t_lower.elapsed().as_secs_f64();
        tracing::debug!(
            surfaces = dirty.len(),
            lower_ms = stats.phases.lower_seconds * 1e3,
            ok = result.is_ok(),
            "frame lowered"
        );
        if result.is_ok() {
            let t = Instant::now();
            for sf in &dirty {
                self.encode_surface(sf.id, stats);
            }
            stats.phases.encode_seconds = t.elapsed().as_secs_f64();
            if self.timestamps && self.frame_pass_count > 0 {
                let t = Instant::now();
                self.resolve_timestamps(2 * self.frame_pass_count);
                stats.phases.stamp_seconds += t.elapsed().as_secs_f64();
            }
        }
        result?;
        Ok(Redraw::None)
    }

    fn readback(&mut self, surface: SurfaceId) -> Result<Readback, RenderError> {
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
        let submission = self.queue.submit([encoder.finish()]);
        tracing::trace!(?surface, ?submission, "readback submitted");
        let slice = buf.slice(..);
        self.map_read(slice, submission, "the pixel readback")?;
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

impl GpuRenderer {
    fn lower_all(
        &mut self,
        pending: &mut [SurfaceState],
        frames: &[&SurfaceFrame<'_>],
    ) -> Vec<Result<Lowered, RenderError>> {
        let mut cleared = false;
        'batch: loop {
            let mut results: Vec<Result<Lowered, RenderError>> = if pending.len() > 1 {
                let (atlas, images) = (&self.atlas, &self.images);
                // `FontData`'s COLR cache is a `RefCell` — !Sync — so
                // each worker moves in its own snapshot built here.
                let snapshots: Vec<HashMap<u64, FontData>> = pending
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
                        .map(|((surf, fonts), frame)| {
                            s.spawn(move || Self::lower_content(surf, frame, atlas, &fonts, images))
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
                    .map(|(surf, frame)| {
                        Self::lower_content(surf, frame, &self.atlas, &self.fonts, &self.images)
                    })
                    .collect()
            };
            // Commit every surface's pending rasters serially, in dirty
            // order.
            for (surf, result) in pending.iter_mut().zip(results.iter_mut()) {
                let Ok(lowered) = result else {
                    continue;
                };
                match self.apply_pending(surf, lowered) {
                    Ok(()) => {}
                    Err(RenderError::AtlasFull) => {
                        if self.atlas.size() < self.atlas.cap() {
                            self.atlas.grow(&self.device);
                            tracing::debug!(
                                size = self.atlas.size(),
                                generation = self.atlas.generation(),
                                "atlas grown"
                            );
                        } else if cleared {
                            *result = Err(RenderError::AtlasExhausted);
                            break 'batch results;
                        } else {
                            self.atlas.clear();
                            cleared = true;
                            tracing::debug!(size = self.atlas.size(), "atlas cleared");
                        }
                        // Growing or clearing emptied the atlas: every
                        // hit any lowering took is now a miss, so lower
                        // the whole batch again.
                        continue 'batch;
                    }
                    Err(e) => {
                        *result = Err(e);
                        break 'batch results;
                    }
                }
            }
            break results;
        }
    }

    fn lower_content(
        surf: &mut SurfaceState,
        frame: &SurfaceFrame<'_>,
        atlas: &Atlas,
        fonts: &HashMap<u64, FontData>,
        images: &HashMap<u64, GpuImage>,
    ) -> Result<Lowered, RenderError> {
        surf.frame.reset();
        // Lowering borrows `layers` immutably while mutating `frame`;
        // taking the map out keeps the two borrows disjoint.
        let layers = std::mem::take(&mut surf.layers);
        let mut lowered = Lowered::default();
        let result = {
            let glyphs = GlyphContext {
                atlas,
                fonts,
                images,
            };
            let mut lowering = Lowering::new(&mut surf.frame, surf.size);
            let result = lowering.run(frame.tree, &layers, frame.clear, &glyphs);
            lowered.glyphs = lowering.glyphs_rasterized();
            lowered.paths = lowering.paths_rasterized();
            lowered.cell_patches = std::mem::take(&mut lowering.cell_patches);
            lowered.mask_patches = std::mem::take(&mut lowering.mask_patches);
            lowered.pending = std::mem::take(&mut lowering.pending);
            result
        };
        surf.layers = layers;
        result.map(|()| lowered)
    }

    fn apply_pending(
        &mut self,
        surf: &mut SurfaceState,
        lowered: &mut Lowered,
    ) -> Result<(), RenderError> {
        let pending = std::mem::take(&mut lowered.pending);
        let mut origins = Vec::with_capacity(pending.len());
        for raster in pending {
            origins.push(self.apply_raster(raster)?);
        }
        for (inst, p, c) in lowered.cell_patches.drain(..) {
            let Some(PendingOrigin::Cells(cells)) = origins.get(p as usize) else {
                continue;
            };
            let Some(&(x, y)) = cells.get(c as usize) else {
                continue;
            };
            surf.frame.instances[inst as usize].uv[0] = x as f32;
            surf.frame.instances[inst as usize].uv[1] = y as f32;
        }
        for (inst, p) in lowered.mask_patches.drain(..) {
            let Some(PendingOrigin::Mask([x, y])) = origins.get(p as usize) else {
                continue;
            };
            surf.frame.instances[inst as usize].uv[2] = *x;
            surf.frame.instances[inst as usize].uv[3] = *y;
        }
        Ok(())
    }

    fn apply_raster(&mut self, raster: PendingRaster) -> Result<PendingOrigin, RenderError> {
        match raster {
            PendingRaster::Glyph {
                key,
                left,
                top,
                w,
                h,
                texels,
            } => self
                .atlas
                .store_glyph(&self.queue, key, left, top, w, h, &texels)
                .map(|(x, y)| PendingOrigin::Cells(vec![(x, y)]))
                .ok_or(RenderError::AtlasFull),
            PendingRaster::Path { key, emit, cells } => {
                self.atlas
                    .store_path(&self.queue, key, emit, &cells)
                    .ok_or(RenderError::AtlasFull)?;
                Ok(PendingOrigin::Cells(
                    self.atlas.path_origins(key).expect("just stored"),
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
                    .store_mask(&self.queue, key, mask, w, h, &texels)
                    .ok_or(RenderError::AtlasFull)?;
                Ok(PendingOrigin::Mask(
                    self.atlas.mask_origin(key).expect("just stored"),
                ))
            }
            PendingRaster::Colr { font, key, picture } => {
                if let Some(font) = self.fonts.get_mut(&font) {
                    font.colr.borrow_mut().entry(key).or_insert(picture);
                }
                Ok(PendingOrigin::None)
            }
        }
    }

    fn lower_surface(
        &mut self,
        id: SurfaceId,
        stats: &mut FrameStats,
        inst_base: u32,
        stop_base: u32,
        globals_base: u32,
        lowered: Result<Lowered, RenderError>,
    ) -> Result<(), RenderError> {
        let Lowered { glyphs, paths, .. } = lowered?;
        stats.glyphs_rasterized += glyphs;
        stats.paths_rasterized += paths;
        // Grow the query set lazily when this frame's passes exceed its
        // capacity; never mid-encoder.
        if self.timestamps {
            let passes = self.surfaces.get(&id).map_or(0, |s| {
                u32::try_from(s.frame.passes.len()).unwrap_or(u32::MAX)
            });
            self.ensure_query_capacity(2 * (globals_base + passes));
        }
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
            if i < surf.scratch.len() {
                surf.scratch[i] = target;
            } else {
                surf.scratch.push(target);
            }
            surf.bind_gen += 1;
        }
        // Backdrop textures for blend composites, sized like the scratch
        // pool to the largest region copied this frame.
        let mut backdrop_max = [(0u32, 0u32); 2];
        for pass in &surf.frame.passes {
            if let Some(r) = pass.backdrop_copy {
                let slot = match pass.target {
                    Target::Surface => 0,
                    Target::Scratch(_) => 1,
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
            surf.bind_gen += 1;
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
        let inst_bytes = bytemuck::cast_slice::<instance::Instance, u8>(&surf.frame.instances);
        let inst_offset = u64::from(inst_base) * std::mem::size_of::<instance::Instance>() as u64;
        if !inst_bytes.is_empty() && inst_offset + inst_bytes.len() as u64 > self.instances.size() {
            let size = (inst_offset + inst_bytes.len() as u64).next_power_of_two();
            self.instances = grow_preserving(
                &self.device,
                &self.queue,
                "instances",
                &self.instances,
                size,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                inst_offset,
            );
        }
        if !inst_bytes.is_empty() {
            self.queue
                .write_buffer(&self.instances, inst_offset, inst_bytes);
        }
        let stop_bytes = bytemuck::cast_slice::<instance::Stop, u8>(&surf.frame.stops);
        let stop_offset = u64::from(stop_base) * std::mem::size_of::<instance::Stop>() as u64;
        if !stop_bytes.is_empty() && stop_offset + stop_bytes.len() as u64 > self.stops.size() {
            let size = (stop_offset + stop_bytes.len() as u64).next_power_of_two();
            self.stops = grow_preserving(
                &self.device,
                &self.queue,
                "stops",
                &self.stops,
                size,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                stop_offset,
            );
        }
        if !stop_bytes.is_empty() {
            self.queue
                .write_buffer(&self.stops, stop_offset, stop_bytes);
        }
        {
            let atlas = &self.atlas;
            if atlas.generation() != self.bound_atlas {
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
        // One Globals entry per pass at a 256-byte stride, continuing the
        // frame-wide slot sequence across dirty surfaces.
        let needed = (u64::from(globals_base) + surf.frame.passes.len().max(1) as u64) * 256;
        if needed > self.globals.size() {
            let size = needed.next_power_of_two();
            self.globals = grow_preserving(
                &self.device,
                &self.queue,
                "globals",
                &self.globals,
                size,
                wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                u64::from(globals_base) * 256,
            );
        }
        for (i, pass) in surf.frame.passes.iter().enumerate() {
            let g = lower::globals(
                [pass.region[2] as f32, pass.region[3] as f32],
                [pass.region[0] as f32, pass.region[1] as f32],
            );
            self.queue.write_buffer(
                &self.globals,
                (u64::from(globals_base) + i as u64) * 256,
                bytemuck::bytes_of(&g),
            );
        }
        // Buffers grown above leave `bind0` stale; rebuild when capacity
        // changed since the bind group was built.
        if self.instances.size() > self.bound_instance_size
            || self.stops.size() > self.bound_stop_size
            || self.globals.size() > self.bound_globals_size
        {
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

    fn encode_surface(&mut self, id: SurfaceId, stats: &mut FrameStats) {
        let Some(surf) = self.surfaces.get_mut(&id) else {
            return;
        };
        // Buffers grown during lowering leave `bind0` stale; rebuild when
        // capacity changed since the bind group was built.
        if self.instances.size() > self.bound_instance_size
            || self.stops.size() > self.bound_stop_size
            || self.globals.size() > self.bound_globals_size
        {
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
        let inst_base = surf.inst_base;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        // Group-1 bind groups persist across frames, keyed by
        // (source scratch, backdrop-needed, image); the stamp rebuilds
        // them when a scratch/backdrop texture or the image set changed.
        let stamp = (surf.bind_gen, self.images_gen);
        if surf.binds1_stamp != stamp {
            surf.binds1.clear();
            surf.binds1_stamp = stamp;
        }
        for pass in &surf.frame.passes {
            let (view, texture) = match pass.target {
                Target::Surface => (&surf.view, &surf.target),
                Target::Scratch(i) => (&surf.scratch[i].view, &surf.scratch[i].texture),
            };
            // A blend pass reads the target's prior contents from a copy;
            // the copy must complete before the pass starts.
            if let Some([bx, by, bw, bh]) = pass.backdrop_copy {
                let slot = match pass.target {
                    Target::Surface => 0,
                    Target::Scratch(_) => 1,
                };
                let backdrop = surf.backdrop[slot].as_ref().expect("grown above");
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: bx, y: by, z: 0 },
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
                        beginning_of_pass_write_index: Some(2 * pass_index),
                        end_of_pass_write_index: Some(2 * pass_index + 1),
                    });
            // Only the timestamp path reads `pass_meta`; skip the
            // allocation when timing is off.
            if self.timestamps {
                self.pass_meta.push(PassMeta {
                    name: match pass.target {
                        Target::Surface => "surface".to_string(),
                        Target::Scratch(i) => format!("scratch{i}"),
                    },
                    width: pass.region[2],
                    height: pass.region[3],
                    format: format_name(match pass.target {
                        Target::Surface => TARGET_FORMAT,
                        Target::Scratch(_) => self.scratch_format,
                    }),
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
            let format_i = match pass.target {
                Target::Surface => 0,
                Target::Scratch(_) => 1,
            };
            render_pass.set_pipeline(&self.pipelines[format_i][0][0]);
            // Scratch passes cover only their region; the surface pass the
            // whole target. `in.device` stays in true device space via the
            // per-pass Globals origin.
            if let Target::Scratch(_) = pass.target {
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
            let mut pipeline = (PipelineKind::SrcOver, ShaderVariant::Simple);
            for range in &pass.ranges {
                stats.draws += 1;
                let want = (range.pipeline, range.variant);
                if want != pipeline {
                    pipeline = want;
                    stats.pipeline_switches += 1;
                    render_pass.set_pipeline(
                        &self.pipelines[format_i][usize::from(pipeline.0 == PipelineKind::Replace)]
                            [variant_index(pipeline.1)],
                    );
                }
                let key = (range.source, scratch_backdrop, range.image);
                let bind = match surf.binds1.entry(key) {
                    std::collections::hash_map::Entry::Occupied(e) => &*e.into_mut(),
                    std::collections::hash_map::Entry::Vacant(e) => {
                        stats.bind_groups_created += 1;
                        let backdrop = if scratch_backdrop {
                            let slot = match pass.target {
                                Target::Surface => 0,
                                Target::Scratch(_) => 1,
                            };
                            surf.backdrop[slot].as_ref().map(|b| &b.view)
                        } else {
                            None
                        };
                        &*e.insert(make_bind1(
                            &self.device,
                            &self.layout1,
                            &self.dummy_view,
                            range.source.map(|i| &surf.scratch[i].view),
                            backdrop,
                            range
                                .image
                                .and_then(|id| self.images.get(&id).map(|i| &i.view)),
                        ))
                    }
                };
                render_pass.set_bind_group(1, bind, &[]);
                render_pass.draw(
                    0..6,
                    (inst_base + range.instances.start)..(inst_base + range.instances.end),
                );
            }
        }
        let submission = self.queue.submit([encoder.finish()]);
        tracing::trace!(
            surface = ?id,
            passes = surf.frame.passes.len(),
            instances = surf.frame.instances.len(),
            ?submission,
            "surface submitted"
        );
        stats.passes += u32::try_from(surf.frame.passes.len()).unwrap_or(u32::MAX);
        stats.instances += u32::try_from(surf.frame.instances.len()).unwrap_or(u32::MAX);
    }

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
        self.wait(submission, what)?;
        match rx.try_recv() {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(RenderError::Readback(format!("{what}: map failed: {e}"))),
            Err(_) => Err(RenderError::Readback(format!(
                "{what}: the map callback did not run after the wait"
            ))),
        }
    }

    fn resolve_timestamps(&mut self, count: u32) {
        if self.query_set.is_none() || self.query_buffer.is_none() {
            self.pass_meta.clear();
            return;
        }
        let staging = self.timestamp_staging();
        let (qs, buf) = (
            self.query_set.as_ref().expect("checked above"),
            self.query_buffer.as_ref().expect("checked above"),
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("timestamp resolve"),
            });
        encoder.resolve_query_set(qs, 0..count, buf, 0);
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, u64::from(count) * 8);
        let submission = self.queue.submit([encoder.finish()]);
        tracing::trace!(count, ?submission, "timestamps resolved");
        self.pending_timestamps.push_back(PendingTimestamps {
            staging,
            count,
            meta: std::mem::take(&mut self.pass_meta),
            map_requested: false,
            ready: Arc::new(AtomicU8::new(0)),
        });
    }

    fn timestamp_staging(&mut self) -> wgpu::Buffer {
        let size = u64::from(self.query_capacity) * 8;
        match self.query_staging.take() {
            Some(buffer) if buffer.size() >= size => buffer,
            _ => self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp staging"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        }
    }

    fn drain_timestamps(&mut self, stats: &mut FrameStats) {
        let period = f64::from(self.queue.get_timestamp_period());
        while let Some(pending) = self.pending_timestamps.front_mut() {
            if !pending.map_requested {
                let flag = Arc::clone(&pending.ready);
                pending
                    .staging
                    .slice(..u64::from(pending.count) * 8)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        flag.store(u8::from(result.is_err()) + 1, Ordering::Relaxed);
                    });
                pending.map_requested = true;
            }
            if self.device.poll(wgpu::PollType::Poll).is_err() {
                break;
            }
            match pending.ready.load(Ordering::Relaxed) {
                // A failed map drops the frame's timing instead of
                // blocking every later drain.
                2 => {
                    tracing::error!("the timestamp readback map failed");
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
            {
                let data = pending
                    .staging
                    .slice(..u64::from(pending.count) * 8)
                    .get_mapped_range();
                let ticks: &[u64] = bytemuck::cast_slice(&data);
                #[expect(clippy::cast_precision_loss)]
                let delta = |from: usize, to: usize| {
                    ticks
                        .get(to)
                        .zip(ticks.get(from))
                        .filter(|(end, start)| end > start)
                        .map(|(end, start)| period * (end - start) as f64 * 1e-9)
                };
                // The frame's GPU time runs from the first pass's start
                // to the last pass's end.
                stats.gpu_seconds = delta(0, pending.count as usize - 1);
                stats.passes_timed = pending
                    .meta
                    .into_iter()
                    .enumerate()
                    .map(|(i, meta)| PassTiming {
                        name: meta.name,
                        width: meta.width,
                        height: meta.height,
                        format: meta.format,
                        gpu_seconds: delta(2 * i, 2 * i + 1).unwrap_or(0.0),
                    })
                    .collect();
            }
            tracing::debug!(
                passes = stats.passes_timed.len(),
                gpu_ms = stats.gpu_seconds.map(|s| s * 1e3),
                "frame timed"
            );
            pending.staging.unmap();
            if pending.staging.size() >= u64::from(self.query_capacity) * 8 {
                self.query_staging = Some(pending.staging);
            }
        }
    }

    fn ensure_query_capacity(&mut self, queries: u32) {
        if queries <= self.query_capacity {
            return;
        }
        let capacity = queries.next_power_of_two().max(2);
        self.query_set = Some(self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frame timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: capacity,
        }));
        self.query_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp resolve"),
            size: u64::from(capacity) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        self.query_capacity = capacity;
    }
}
fn srgb_decode_u8_f64(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}
