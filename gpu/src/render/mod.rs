// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The render thread: sole owner of GPU state.

mod colr;
mod glyph;
mod instance;
mod lower;
mod path;
mod present;
mod raster;

use std::collections::HashMap;

use cherenkov::{
    ContentOp, EngineError, FontData as EngineFontData, FontId, Frame, FrameStats, ImageId,
    ImageUpload, LayerId, MemoryUsage, PassTiming, Pressure, Readback, Redraw, RenderError,
    Renderer, ResourceError, SurfaceError, SurfaceFrame, SurfaceId, SurfaceInfo,
};
use glyph::{Atlas, FontData};
use lower::{ContentData, Frame as LoweredFrame, GlyphContext, Lowering, PipelineKind, Target};

use crate::{GpuConfig, GpuInfo, GpuTarget, ScratchFormat, names};

/// The surface target format: premultiplied linear Display P3.
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

const TARGET_USAGES: wgpu::TextureUsages = wgpu::TextureUsages::from_bits_retain(
    wgpu::TextureUsages::RENDER_ATTACHMENT.bits()
        | wgpu::TextureUsages::COPY_SRC.bits()
        | wgpu::TextureUsages::TEXTURE_BINDING.bits(),
);

/// Decodes one sRGB-encoded channel to linear, `0..=1 → f64`.
fn srgb_decode_u8_f64(v: f64) -> f64 {
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

/// A lowering error: keeps the recoverable atlas-full signal typed so
/// the retry loop matches it without string sentinels. Converted to
/// [`RenderError`] only at the [`Renderer`] boundary.
pub enum Encode {
    /// The glyph/path atlas had no room for a cell; the caller may grow
    /// or clear the atlas and retry the lowering.
    AtlasFull,
    /// Any other lowering failure.
    Other(RenderError),
}

impl From<RenderError> for Encode {
    fn from(error: RenderError) -> Self {
        Self::Other(error)
    }
}

impl From<Encode> for RenderError {
    fn from(error: Encode) -> Self {
        match error {
            Encode::AtlasFull => Self::Render("glyph atlas exhausted".into()),
            Encode::Other(e) => e,
        }
    }
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
    /// Per-layer content caches; the sampled layer state lives in the
    /// front end's [`cherenkov::SurfaceTree`].
    layers: HashMap<LayerId, ContentData>,
    /// The reused lowering output (instances, stops, passes).
    frame: LoweredFrame,
    /// The swapchain the target is presented on, for window surfaces.
    window: Option<present::WindowSurface>,
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

/// All render-thread state: the [`Gpu`](crate::Gpu) backend's
/// [`Renderer`] implementation.
pub struct GpuRenderer {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    presenter: present::Presenter,
    /// `[format index][pipeline kind]`: 0 = surface format, 1 = scratch
    /// format; kind 0 = source-over, 1 = replace.
    pipelines: [[wgpu::RenderPipeline; 2]; 2],
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
    timestamps: bool,
    query_set: Option<wgpu::QuerySet>,
    query_buffer: Option<wgpu::Buffer>,
    query_staging: wgpu::Buffer,
    /// The query set's capacity in queries; indices 0/1 bracket the frame,
    /// `2 + 2i` each pass.
    query_capacity: u32,
    /// Passes encoded this frame, for the per-pass report.
    frame_pass_count: u32,
    /// `(name, width, height, format)` of each encoded pass this frame.
    pass_meta: Vec<PassMeta>,
    timestamps_inside: bool,
    max_texture: u32,
}

/// One encoded pass's report metadata.
struct PassMeta {
    name: String,
    width: u32,
    height: u32,
    format: &'static str,
}

/// Creates an adapter plus device. Fails when no adapter allows the target
/// format's required usages.
fn create_device(
    config: &GpuConfig,
) -> Result<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue), EngineError> {
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
        .ok_or_else(|| EngineError::Backend("no suitable wgpu adapter".into()))?;
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
    .map_err(|e| EngineError::Backend(format!("device request failed: {e}")))?;
    Ok((instance, adapter, device, queue))
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

/// The closed pipeline set: one instanced-quad pipeline from `shader.wgsl`.
/// When a pipeline cache path is configured and supported, the cache is
/// loaded beforehand and persisted afterwards, best effort.
fn create_pipeline(
    device: &wgpu::Device,
    config: &GpuConfig,
    layout0: &wgpu::BindGroupLayout,
    layout1: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    replace: bool,
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
        return Err(EngineError::Backend(format!("shader: {error}")));
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

/// Runs on the render thread once: creates the device and precompiles
/// the closed pipeline set, returning the backend's [`Renderer`].
///
/// # Errors
/// [`EngineError::Backend`] when no suitable adapter exists, device
/// creation fails or a pipeline fails validation.
#[expect(
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    reason = "the contract moves the config onto the render thread"
)]
pub fn init(config: GpuConfig) -> Result<(GpuRenderer, GpuInfo), EngineError> {
    create_device(&config).and_then(|(instance, adapter, device, queue)| {
        let info = adapter.get_info();
        let presenter = present::Presenter::new(&device);
        let (layout0, layout1) = create_layouts(&device);
        let scratch_format = scratch_wgpu(config.scratch_format);
        let pipelines = |format| {
            Ok::<_, EngineError>([
                create_pipeline(&device, &config, &layout0, &layout1, format, false)?,
                create_pipeline(&device, &config, &layout0, &layout1, format, true)?,
            ])
        };
        let pipelines = [pipelines(TARGET_FORMAT)?, pipelines(scratch_format)?];
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
            TARGET_FORMAT,
        );
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
        // The initial query set holds the two frame-bracketing queries.
        let query_capacity = if query_set.is_some() { 2 } else { 0 };
        let renderer = GpuRenderer {
            max_texture: device.limits().max_texture_dimension_2d,
            instance,
            adapter,
            device,
            queue,
            presenter,
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
            timestamps,
            query_set,
            query_buffer,
            query_staging,
            query_capacity,
            frame_pass_count: 0,
            pass_meta: Vec::new(),
            timestamps_inside,
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
        let (size, window) = match target {
            GpuTarget::Offscreen(offscreen) => {
                // The target is always Rgba16Float and readback decodes f16,
                // so only the f16 readback format is honest.
                if offscreen.format != cherenkov::OffscreenFormat::LinearF16 {
                    return Err(SurfaceError::UnsupportedFormat(offscreen.format));
                }
                (offscreen.size, None)
            }
            GpuTarget::Window(window) => {
                let (handle, size) = window.into_parts();
                if size.0 == 0 || size.1 == 0 {
                    return Err(SurfaceError::ZeroSize);
                }
                let window = present::WindowSurface::new(
                    &self.instance,
                    &self.adapter,
                    &self.device,
                    handle,
                    size,
                )?;
                (size, Some(window))
            }
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
                window,
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
        if let Some(window) = &mut state.window {
            window.resize(&self.device, size);
        }
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

    /// Uploads a registered image, converting the upload's RGBA8 into
    /// premultiplied linear Display P3 f16 — the oracle's
    /// `Resources::image` conversion.
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
        Ok(())
    }

    fn remove_image(&mut self, id: ImageId) {
        self.images.remove(&id.raw());
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

    fn trim(&mut self, pressure: Pressure) {
        if pressure == Pressure::Critical {
            self.atlas.clear();
        }
    }

    /// Lowers and submits every changed surface, bracketed by drained
    /// timestamp queries when enabled.
    fn render(&mut self, frame: &Frame<'_>, stats: &mut FrameStats) -> Result<Redraw, RenderError> {
        let dirty: Vec<&SurfaceFrame<'_>> = frame.surfaces.iter().filter(|sf| sf.changed).collect();
        if dirty.is_empty() {
            return Ok(Redraw::None);
        }
        self.frame_pass_count = 0;
        self.pass_meta.clear();
        self.drain_and_stamp(0)?;
        let mut result = Ok(());
        for sf in dirty {
            result = self.render_surface(sf, stats);
            if result.is_err() {
                break;
            }
        }
        if self.timestamps {
            self.drain_and_stamp(1)?;
            let ticks = self.resolve_timestamps(2 + 2 * self.frame_pass_count)?;
            let period = f64::from(self.queue.get_timestamp_period());
            #[expect(clippy::cast_precision_loss)]
            let delta = |from: usize, to: usize| {
                ticks
                    .get(to)
                    .zip(ticks.get(from))
                    .filter(|(end, start)| end > start)
                    .map(|(end, start)| period * (end - start) as f64 * 1e-9)
            };
            stats.gpu_seconds = delta(0, 1);
            for (i, meta) in self.pass_meta.drain(..).enumerate() {
                stats.passes_timed.push(PassTiming {
                    name: meta.name,
                    width: meta.width,
                    height: meta.height,
                    format: meta.format,
                    gpu_seconds: delta(2 + 2 * i, 3 + 2 * i).unwrap_or(0.0),
                });
            }
        }
        self.wait()?;
        result?;
        // No backend-side redraw sources in this slice.
        Ok(Redraw::None)
    }

    /// Copies a surface's target into `Readback` pixels, decoding f16 → f32.
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

impl GpuRenderer {
    /// Lowers and submits one surface.
    #[expect(
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        reason = "pixel sizes are well within f32"
    )]
    fn render_surface(
        &mut self,
        sf: &SurfaceFrame<'_>,
        stats: &mut FrameStats,
    ) -> Result<(), RenderError> {
        let id = sf.id;
        // Lowering needs `surf.layers` and `surf.frame` plus `atlas`,
        // `fonts`, `device` and `queue`; take the layer map out of the
        // surface so the borrows stay disjoint.
        let lowered = {
            let Some(surf) = self.surfaces.get_mut(&id) else {
                return Ok(());
            };
            surf.frame.reset();
            let caches = std::mem::take(&mut surf.layers);
            // A full atlas is a recoverable signal: grow while the budget
            // allows, then clear once; a second failure after the clear
            // means the frame's live set exceeds the maximum atlas.
            let mut cleared = false;
            let result = loop {
                let result = {
                    let mut glyphs = GlyphContext {
                        atlas: &mut self.atlas,
                        queue: &self.queue,
                        fonts: &self.fonts,
                        images: &self.images,
                    };
                    let mut lowering = Lowering::new(&mut surf.frame, surf.size);
                    let result = lowering.run(sf.tree, &caches, sf.clear, &mut glyphs);
                    stats.glyphs_rasterized += lowering.glyphs_rasterized();
                    stats.paths_rasterized += lowering.paths_rasterized();
                    result
                };
                match result {
                    Err(Encode::AtlasFull) if !cleared => {
                        surf.frame.reset();
                        if self.atlas.size() < self.atlas.cap() {
                            self.atlas.grow(&self.device);
                        } else {
                            self.atlas.clear();
                            cleared = true;
                        }
                    }
                    Err(Encode::AtlasFull) => {
                        break Err(Encode::AtlasFull);
                    }
                    other => break other,
                }
            };
            surf.layers = caches;
            result
        };
        lowered.map_err(RenderError::from)?;
        // Grow the query set lazily when this frame's passes exceed its
        // capacity; never mid-encoder.
        if self.timestamps {
            let passes = self.surfaces.get(&id).map_or(0, |s| {
                u32::try_from(s.frame.passes.len()).unwrap_or(u32::MAX)
            });
            self.ensure_query_capacity(2 + 2 * (self.frame_pass_count + passes));
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
        }
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
        // One Globals entry per pass at a 256-byte stride; grow the
        // uniform buffer lazily and write each pass's target frame.
        let needed = (surf.frame.passes.len().max(1) as u64) * 256;
        if needed > self.globals.size() {
            let size = needed.next_power_of_two();
            self.globals = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("globals"),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        for (i, pass) in surf.frame.passes.iter().enumerate() {
            let g = lower::globals(
                [pass.region[2] as f32, pass.region[3] as f32],
                [pass.region[0] as f32, pass.region[1] as f32],
            );
            self.queue
                .write_buffer(&self.globals, (i as u64) * 256, bytemuck::bytes_of(&g));
        }
        // Buffers grown above leave `bind0` stale; rebuild when capacity
        // changed since the bind group was built.
        if self.instances.size() > self.bound_instance_size
            || self.stops.size() > self.bound_stop_size
            || self.globals.size() > self.bound_globals_size
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
            self.bound_globals_size = self.globals.size();
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        // Lazily-built group-1 bind groups for this frame, keyed by
        // (source, backdrop-needed, image). Created up front so scratch
        // borrows stay immutable inside the encoder loop.
        let mut range_binds: HashMap<(Option<usize>, bool, Option<u64>), wgpu::BindGroup> =
            HashMap::new();
        for (i, pass) in surf.frame.passes.iter().enumerate() {
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
                        beginning_of_pass_write_index: Some(2 + 2 * pass_index),
                        end_of_pass_write_index: Some(3 + 2 * pass_index),
                    });
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
            render_pass.set_pipeline(&self.pipelines[format_i][0]);
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
            #[expect(clippy::cast_possible_truncation)]
            let offset = (i * 256) as u32;
            render_pass.set_bind_group(0, &self.bind0, &[offset]);
            let mut pipeline = PipelineKind::SrcOver;
            for range in &pass.ranges {
                stats.draws += 1;
                if range.pipeline != pipeline {
                    pipeline = range.pipeline;
                    render_pass.set_pipeline(
                        &self.pipelines[format_i][usize::from(pipeline == PipelineKind::Replace)],
                    );
                }
                let key = (range.source, scratch_backdrop, range.image);
                let bind = match range_binds.entry(key) {
                    std::collections::hash_map::Entry::Occupied(e) => &*e.into_mut(),
                    std::collections::hash_map::Entry::Vacant(e) => {
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
                render_pass.draw(0..6, range.instances.clone());
            }
        }
        self.queue.submit([encoder.finish()]);
        stats.passes += u32::try_from(surf.frame.passes.len()).unwrap_or(u32::MAX);
        stats.instances += u32::try_from(surf.frame.instances.len()).unwrap_or(u32::MAX);
        if let Some(window) = &surf.window {
            self.presenter
                .present(&self.device, &self.queue, window, &surf.view)?;
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

    /// Grows the query set and its resolve buffers to hold `queries`,
    /// between frames — never mid-encoder.
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
        self.query_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp staging"),
            size: u64::from(capacity) * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.query_capacity = capacity;
    }

    /// Resolves the first `count` timestamp queries into raw ticks.
    fn resolve_timestamps(&self, count: u32) -> Result<Vec<u64>, RenderError> {
        let (Some(qs), Some(buf)) = (&self.query_set, &self.query_buffer) else {
            return Ok(Vec::new());
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("timestamp resolve"),
            });
        encoder.resolve_query_set(qs, 0..count, buf, 0);
        encoder.copy_buffer_to_buffer(buf, 0, &self.query_staging, 0, u64::from(count) * 8);
        self.queue.submit([encoder.finish()]);
        let slice = self.query_staging.slice(..u64::from(count) * 8);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.wait()?;
        let data = slice.get_mapped_range();
        let ticks: Vec<u64> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.query_staging.unmap();
        Ok(ticks)
    }
}
