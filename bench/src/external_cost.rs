//! `external-cost`: issue #168's hand-off A/B — the engine's retained
//! external-frame path (`--path e`) against copy-and-convert
//! (`--path c`), over identical produced frames.
//!
//! Both paths share one producer: a ring of platform video buffers —
//! `CVPixelBuffer` on Apple, `AHardwareBuffer` on Android — filled on the
//! CPU with a drifting gradient once per frame. Path `e` wraps the
//! buffer's planes and installs them as a retained [`ExternalFrame`] the
//! engine samples in place. Path `c` copies the planes into engine-side
//! textures with `Queue::write_texture` and converts them in a
//! [`GpuContent`] draw — the engine's own YUV decode, reached by
//! concatenating `gpu/src/render/external.wgsl` into the bench's shader —
//! mirroring water-rs/video-gpu's copy-then-shader composite
//! (`src/runtime_player.rs` `upload_frame_texture` + `render_surface`).
//!
//! Timing: the engine's own GPU timestamps cover the composite
//! submission; path `c` additionally brackets its upload+convert window
//! with pass-boundary stamps (the `wgpu_ctx::stamp` scheme).

use std::sync::Arc;
use std::time::{Duration, Instant};

use cherenkov::{Engine, FrameTime, Offscreen, OffscreenFormat};
use cherenkov_gpu::interop::{
    ChromaOffset, ExternalFrame, FrameColor, GpuContent, GpuContentBox, Primaries, SharedDevice,
    Transfer, YuvMatrix, YuvRange, wgpu,
};
use cherenkov_gpu::{Gpu, GpuConfig};

use crate::cli::{ExternalCostArgs, ExternalPath, ExternalSize, ExternalTransfer};
use crate::memory::{
    AdapterMemory, EngineBytes, MemoryReport, MemorySnapshot, Reading, SampleDetail,
    wgpu_allocator, wgpu_vk_memory_budget,
};
use crate::motion::Clock;
use crate::report::{Conditions, EnergyReport, Pacing, percentiles};
use crate::timing::Timings;
use crate::{BenchError, DeviceInfo, PhaseSample, affinity, conditions, energy};

/// Producer ring depth — a decode queue cycles a few buffers ahead.
const RING: usize = 3;

/// Codes of ramp drift in the generated pattern: the window the
/// frame/row offset slides over.
const SHIFT: usize = 256;

/// What one run measures.
#[derive(Clone, Copy)]
struct Spec {
    width: u32,
    height: u32,
    /// 8 for NV12 (SDR), 10 for P010 (PQ).
    bits: u32,
    color: FrameColor,
}

impl Spec {
    const fn new(size: ExternalSize, transfer: ExternalTransfer) -> Self {
        let (width, height) = match size {
            ExternalSize::P1080 => (1920, 1080),
            ExternalSize::P4k => (3840, 2160),
        };
        let (bits, color) = match transfer {
            ExternalTransfer::Sdr => (8, FrameColor::BT709_VIDEO),
            ExternalTransfer::Pq => (10, FrameColor::BT2020_PQ),
        };
        Self {
            width,
            height,
            bits,
            color,
        }
    }

    /// Bytes per stored code: 1 for NV12, 2 for P010's 16-bit words.
    const fn code_bytes(self) -> usize {
        if self.bits == 8 { 1 } else { 2 }
    }

    /// `wgpu` formats of the luma and interleaved-chroma planes — the
    /// [`ExternalFrame::yuv`] contract (path `e`) and the copy targets
    /// (path `c`).
    const fn formats(self) -> (wgpu::TextureFormat, wgpu::TextureFormat) {
        if self.bits == 8 {
            (wgpu::TextureFormat::R8Uint, wgpu::TextureFormat::Rg8Uint)
        } else {
            (wgpu::TextureFormat::R16Uint, wgpu::TextureFormat::Rg16Uint)
        }
    }

    const fn layout(self) -> &'static str {
        if self.bits == 8 { "nv12" } else { "p010" }
    }

    const fn transfer_name(self) -> &'static str {
        match self.color.transfer {
            Transfer::Bt709 => "bt709-sdr",
            _ => "bt2020-pq",
        }
    }
}

/// The shader-visible decode arguments — the same 192-byte layout as
/// `cherenkov-gpu`'s `render::external::Params`, which `external.wgsl`
/// declares as `ExtParams`. Baked here with the same math; the engine's
/// copy lives in `gpu/src/render/external/mod.rs::params`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    info: [u32; 4],
    dims: [f32; 4],
    norm: [f32; 4],
    site: [f32; 4],
    yuv: [[f32; 4]; 4],
    prim: [[f32; 4]; 3],
    luma: [f32; 4],
}

const KIND_NV12: u32 = 1;
const KIND_P010: u32 = 2;
const FLAG_SHIFT6: u32 = 2;

/// The `s` in "chroma texel `i` centres on luma position `2i + s`".
const fn siting(offset: ChromaOffset) -> f32 {
    match offset {
        ChromaOffset::Cosited => 0.0,
        ChromaOffset::Centered => 0.5,
    }
}

/// `Kr, Kb` of a `Y'CbCr` matrix — mirrors `external/mod.rs::matrix_coeffs`.
const fn matrix_coeffs(matrix: YuvMatrix) -> (f32, f32) {
    match matrix {
        YuvMatrix::Bt601 => (0.299, 0.114),
        YuvMatrix::Bt709 => (0.2126, 0.0722),
        YuvMatrix::Bt2020 => (0.2627, 0.0593),
    }
}

/// xy chromaticities `(red, green, blue)` of a primaries set; all D65 —
/// mirrors `external/mod.rs::primaries_xy`.
const fn primaries_xy(primaries: Primaries) -> [[f32; 2]; 3] {
    match primaries {
        Primaries::Bt709 => [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]],
        Primaries::DisplayP3 => [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
        Primaries::Bt2020 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
    }
}

/// xy → XYZ at unit luminance.
fn to_xyz(xy: [f32; 2]) -> [f32; 3] {
    let [x, y] = xy;
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// `M · v` for a row-major 3×3.
fn mat_mul(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][2].mul_add(v[2], m[0][1].mul_add(v[1], m[0][0] * v[0])),
        m[1][2].mul_add(v[2], m[1][1].mul_add(v[1], m[1][0] * v[0])),
        m[2][2].mul_add(v[2], m[2][1].mul_add(v[1], m[2][0] * v[0])),
    ]
}

fn mat_invert(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let det = m[0][2].mul_add(
        m[1][0].mul_add(m[2][1], -(m[1][1] * m[2][0])),
        (-m[0][1]).mul_add(
            m[1][0].mul_add(m[2][2], -(m[1][2] * m[2][0])),
            m[0][0] * m[1][1].mul_add(m[2][2], -(m[1][2] * m[2][1])),
        ),
    );
    let inv = 1.0 / det;
    [
        [
            m[1][1].mul_add(m[2][2], -(m[1][2] * m[2][1])) * inv,
            m[0][2].mul_add(m[2][1], -(m[0][1] * m[2][2])) * inv,
            m[0][1].mul_add(m[1][2], -(m[0][2] * m[1][1])) * inv,
        ],
        [
            m[1][2].mul_add(m[2][0], -(m[1][0] * m[2][2])) * inv,
            m[0][0].mul_add(m[2][2], -(m[0][2] * m[2][0])) * inv,
            m[0][2].mul_add(m[1][0], -(m[0][0] * m[1][2])) * inv,
        ],
        [
            m[1][0].mul_add(m[2][1], -(m[1][1] * m[2][0])) * inv,
            m[0][1].mul_add(m[2][0], -(m[0][0] * m[2][1])) * inv,
            m[0][0].mul_add(m[1][1], -(m[0][1] * m[1][0])) * inv,
        ],
    ]
}

fn mat_product(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (r, row) in a.iter().enumerate() {
        for (c, out_c) in out[r].iter_mut().enumerate() {
            *out_c = row[2].mul_add(b[2][c], row[1].mul_add(b[1][c], row[0] * b[0][c]));
        }
    }
    out
}

/// RGB → XYZ of a primaries set, normalized to D65 white.
fn rgb_to_xyz(primaries: Primaries) -> [[f32; 3]; 3] {
    let [red, green, blue] = primaries_xy(primaries);
    let cols = [to_xyz(red), to_xyz(green), to_xyz(blue)];
    let unscaled = [
        [cols[0][0], cols[1][0], cols[2][0]],
        [cols[0][1], cols[1][1], cols[2][1]],
        [cols[0][2], cols[1][2], cols[2][2]],
    ];
    let d65 = to_xyz([0.3127, 0.3290]);
    let scales = mat_mul(mat_invert(unscaled), d65);
    let mut out = [[0.0; 3]; 3];
    for (c, col) in cols.iter().enumerate() {
        for (r, out_r) in out.iter_mut().enumerate() {
            out_r[c] = col[r] * scales[c];
        }
    }
    out
}

/// The absolute-to-white-relative scale of a transfer; mirrors
/// `external/mod.rs::value_scale`.
fn value_scale(color: &FrameColor) -> f32 {
    match color.transfer {
        Transfer::Pq => 10_000.0 / color.reference_white,
        Transfer::Hlg => color.hlg_peak / color.reference_white,
        _ => 1.0,
    }
}

/// Code normalization `code * scale + offset` — mirrors
/// `external/mod.rs::code_norm`.
const fn code_norm(range: YuvRange, bits: u32) -> [f32; 4] {
    match (range, bits) {
        (YuvRange::Video, 8) => [1.0 / 219.0, -16.0 / 219.0, 1.0 / 224.0, -128.0 / 224.0],
        (YuvRange::Full, 8) => [1.0 / 255.0, 0.0, 1.0 / 255.0, -128.0 / 255.0],
        (YuvRange::Video, _) => [1.0 / 876.0, -64.0 / 876.0, 1.0 / 896.0, -512.0 / 896.0],
        (YuvRange::Full, _) => [1.0 / 1023.0, 0.0, 1.0 / 1023.0, -512.0 / 1023.0],
    }
}

/// `R'G'B'` decode columns — mirrors `external/mod.rs::yuv_columns`.
fn yuv_columns(matrix: YuvMatrix) -> [[f32; 4]; 4] {
    let (kr, kb) = matrix_coeffs(matrix);
    let kg = 1.0 - kr - kb;
    [
        [1.0, 1.0, 1.0, 0.0],
        [0.0, -2.0 * kb * (1.0 - kb) / kg, 2.0 * (1.0 - kb), 0.0],
        [2.0 * (1.0 - kr), -2.0 * kr * (1.0 - kr) / kg, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
}

/// Bakes a spec's decode into the `ExtParams` uniform — the same bake
/// `external/mod.rs::params` performs for an installed frame.
#[expect(clippy::cast_precision_loss, reason = "texture dimensions fit f32")]
fn params(spec: &Spec) -> Params {
    let color = &spec.color;
    let (kind, flags) = match spec.bits {
        8 => (KIND_NV12, 0),
        _ => (KIND_P010, FLAG_SHIFT6),
    };
    let scale = value_scale(color);
    let to_xyz = rgb_to_xyz(color.primaries);
    let to_p3 = mat_product(mat_invert(rgb_to_xyz(Primaries::DisplayP3)), to_xyz);
    let prim = [
        [
            to_p3[0][0] * scale,
            to_p3[1][0] * scale,
            to_p3[2][0] * scale,
            0.0,
        ],
        [
            to_p3[0][1] * scale,
            to_p3[1][1] * scale,
            to_p3[2][1] * scale,
            0.0,
        ],
        [
            to_p3[0][2] * scale,
            to_p3[1][2] * scale,
            to_p3[2][2] * scale,
            0.0,
        ],
    ];
    let luma = [to_xyz[1][0], to_xyz[1][1], to_xyz[1][2], 0.0];
    Params {
        info: [
            kind,
            color.transfer as u32,
            0, /* RgbAlpha::Opaque */
            flags,
        ],
        dims: [
            spec.width as f32,
            spec.height as f32,
            spec.width.div_ceil(2) as f32,
            spec.height.div_ceil(2) as f32,
        ],
        norm: code_norm(color.range, spec.bits),
        site: [
            siting(color.chroma_siting.x),
            siting(color.chroma_siting.y),
            if color.transfer == Transfer::Hlg {
                0.42f32.mul_add((color.hlg_peak / 1000.0).log10(), 1.2)
            } else {
                0.0
            },
            0.0,
        ],
        yuv: yuv_columns(color.matrix),
        prim,
        luma,
    }
}

/// The synthetic frame pattern: a luma ramp and an interleaved chroma
/// ramp that drift by `frame` and `row` — valid video-range codes,
/// non-constant, identical for both paths.
struct Ramps {
    /// `(w + SHIFT)` luma codes of `bytes` each.
    luma: Vec<u8>,
    /// `(w / 2 + SHIFT)` interleaved `(cb, cr)` pairs of `bytes` each.
    chroma: Vec<u8>,
    /// Bytes per stored code (1 NV12, 2 P010).
    bytes: usize,
    /// Row width in luma codes.
    width: usize,
}

impl Ramps {
    /// Video-range code ramps: luma sweeps `16..=235` (8-bit) or
    /// `64..=940` (10-bit `<<6`) left to right; the chroma pair steps
    /// through its range at a different rate so the picture varies in
    /// both planes.
    fn new(spec: &Spec) -> Self {
        let width = spec.width as usize;
        let bytes = spec.code_bytes();
        let mut luma = Vec::with_capacity((width + SHIFT) * bytes);
        let mut chroma = Vec::with_capacity((width / 2 + SHIFT) * 2 * bytes);
        for i in 0..(width + SHIFT) {
            let code = if spec.bits == 8 {
                16 + u32::try_from(i * 219 / (width + SHIFT)).expect("luma code")
            } else {
                (64 + u32::try_from(i * 876 / (width + SHIFT)).expect("luma code")) << 6
            };
            luma.extend_from_slice(&code.to_le_bytes()[..bytes]);
        }
        for i in 0..(width / 2 + SHIFT) {
            for component in 0..2 {
                let code = if spec.bits == 8 {
                    16 + u32::try_from((i * 7 + component * 113) % 224).expect("chroma code")
                } else {
                    (64 + u32::try_from((i * 7 + component * 449) % 896).expect("chroma code")) << 6
                };
                chroma.extend_from_slice(&code.to_le_bytes()[..bytes]);
            }
        }
        Self {
            luma,
            chroma,
            bytes,
            width,
        }
    }

    /// Writes the `row`'th luma row of `frame` into `dst`
    /// (`width * bytes` long).
    fn luma_row(&self, frame: u32, row: u32, dst: &mut [u8]) {
        let row_bytes = self.width * self.bytes;
        let off = (row.wrapping_mul(3).wrapping_add(frame.wrapping_mul(11)) as usize % SHIFT)
            * self.bytes;
        dst[..row_bytes].copy_from_slice(&self.luma[off..off + row_bytes]);
    }

    /// Writes the `row`'th chroma row of `frame` into `dst`
    /// (`width * bytes` long — `width / 2` interleaved pairs).
    fn chroma_row(&self, frame: u32, row: u32, dst: &mut [u8]) {
        let row_bytes = self.width * self.bytes;
        let pair = self.bytes * 2;
        let off =
            (row.wrapping_mul(5).wrapping_add(frame.wrapping_mul(13)) as usize % SHIFT) * pair;
        dst[..row_bytes].copy_from_slice(&self.chroma[off..off + row_bytes]);
    }
}

/// `external.wgsl` verbatim — the `shared.wgsl` prelude it needs
/// included — followed by the bench's fullscreen convert pass
/// (`external_convert.wgsl`), which calls `ext_frame_yuv`. Path `c` runs
/// the identical decode the engine's external-frame path does, over the
/// copied planes.
const CONVERT_WGSL: &str = concat!(
    include_str!("../../gpu/src/render/shared.wgsl"),
    include_str!("../../gpu/src/render/external.wgsl"),
    include_str!("external_convert.wgsl")
);

/// The `GpuContent` producer of path `c`: one fullscreen triangle
/// sampling the copied planes through the engine's decode into the
/// layer's working-space attachment.
struct Convert {
    /// Copied-plane views (`R8Uint`/`Rg8Uint` or `R16Uint`/`Rg16Uint`).
    y: wgpu::TextureView,
    uv: wgpu::TextureView,
    /// The baked `ExtParams` uniform — `external.wgsl`'s group-1
    /// binding 4.
    params: wgpu::Buffer,
    /// Pass-boundary stamps: `3f + 1`/`3f + 2` around the convert pass;
    /// the host stamps `3f` ahead of the plane uploads.
    queries: Arc<wgpu::QuerySet>,
    /// Producer-side frame counter (render calls).
    frame: u32,
    /// Built on the render thread in `setup`.
    live: Option<Live>,
}

/// What `setup` builds: the convert pipeline and its bound planes.
struct Live {
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
}

impl GpuContent for Convert {
    fn setup(&mut self, ctx: &wgpu::Context<'_>) -> impl Future<Output = ()> {
        let u32_tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("external-cost convert planes"),
                entries: &[
                    u32_tex(0),
                    u32_tex(1),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<Params>() as u64
                            ),
                        },
                        count: None,
                    },
                ],
            });
        let empty = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("external-cost empty group"),
                entries: &[],
            });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("external-cost convert layout"),
                bind_group_layouts: &[Some(&empty), Some(&layout)],
                immediate_size: 0,
            });
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("external-cost convert"),
                source: wgpu::ShaderSource::Wgsl(CONVERT_WGSL.into()),
            });
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("external-cost convert"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_convert"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_convert"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("external-cost convert planes"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.y),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.uv),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params.as_entire_binding(),
                },
            ],
        });
        self.live = Some(Live { pipeline, bind });
        core::future::ready(())
    }

    fn render(&mut self, frame: &mut wgpu::Frame<'_>) {
        let Live { pipeline, bind } = self.live.as_ref().expect("setup ran before render");
        let f = self.frame;
        self.frame += 1;
        let mut encoder = frame
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("external-cost convert"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("external-cost convert"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
                    query_set: &self.queries,
                    beginning_of_pass_write_index: Some(3 * f + 1),
                    end_of_pass_write_index: Some(3 * f + 2),
                }),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, bind, &[]);
            pass.draw(0..3, 0..1);
        }
        frame.queue.submit([encoder.finish()]);
        // The producer draws every frame: a new frame's planes arrive
        // each interval, so the content is never settled.
        frame.request_redraw();
    }
}

/// Path `c`'s host-side state: the copy targets, the bench's stamp
/// resources and the resolution buffers.
struct CopyPath {
    /// Luma copy target.
    y: wgpu::Texture,
    /// Interleaved-chroma copy target.
    uv: wgpu::Texture,
    /// `3f` opens the upload window (end-stamp of a marker pass
    /// submitted just before the `write_texture` calls), `3f + 1`/`3f +
    /// 2` bracket the convert pass.
    queries: Arc<wgpu::QuerySet>,
    /// The 1×1 renderable the marker passes clear.
    marker: wgpu::TextureView,
    /// Query resolve target and its map staging.
    resolve: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl CopyPath {
    fn new(shared: &SharedDevice, spec: &Spec, query_count: u32) -> Self {
        let device = &shared.device;
        let (y_format, uv_format) = spec.formats();
        let plane_texture = |name, width, height, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(name),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let y = plane_texture(
            "external-cost copied luma",
            spec.width,
            spec.height,
            y_format,
        );
        let uv = plane_texture(
            "external-cost copied chroma",
            spec.width.div_ceil(2),
            spec.height.div_ceil(2),
            uv_format,
        );
        let marker = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("external-cost stamp marker"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let queries = Arc::new(device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("external-cost handoff stamps"),
            ty: wgpu::QueryType::Timestamp,
            count: query_count,
        }));
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("external-cost stamp resolve"),
            size: u64::from(query_count) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("external-cost stamp staging"),
            size: u64::from(query_count) * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            y,
            uv,
            queries,
            marker,
            resolve,
            staging,
        }
    }

    /// Stamps `3 * frame`, then copies the produced buffer's planes
    /// into the engine-side textures. The serial queue puts the
    /// `write_texture` flush after the stamp and before the convert
    /// pass's begin timestamp.
    fn upload(&self, producer: &platform::Producer, shared: &SharedDevice, frame: u32) {
        let mut encoder = shared
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("external-cost stamp"),
            });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("external-cost stamp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.marker,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
                    query_set: &self.queries,
                    beginning_of_pass_write_index: None,
                    end_of_pass_write_index: Some(3 * frame),
                }),
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        shared.queue.submit([encoder.finish()]);
        producer.upload(frame, &shared.queue, &self.y, &self.uv);
    }

    /// Resolves the stamp set and returns per-frame `(handoff, convert)`
    /// seconds for the measured window (`warmup..`).
    #[expect(
        clippy::cast_precision_loss,
        reason = "GPU timestamp nanoseconds fit f64 precision"
    )]
    fn read_stamps(
        &self,
        shared: &SharedDevice,
        warmup: u32,
        frames: u32,
    ) -> Result<Vec<StampPair>, BenchError> {
        let count = 3 * (warmup + frames);
        let mut encoder = shared
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("external-cost stamp resolve"),
            });
        encoder.resolve_query_set(&self.queries, 0..count, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.staging, 0, u64::from(count) * 8);
        let submission = shared.queue.submit([encoder.finish()]);
        let (send, recv) = std::sync::mpsc::channel();
        self.staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = send.send(result);
            });
        shared
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(30)),
            })
            .map_err(|e| BenchError::Gpu(format!("external-cost stamp wait: {e}")))?;
        recv.recv()
            .map_err(|e| BenchError::Gpu(format!("external-cost stamp readback: {e}")))?
            .map_err(|e| BenchError::Gpu(format!("external-cost stamp map: {e}")))?;
        let ticks: Vec<u64> = {
            let data = self
                .staging
                .slice(..)
                .get_mapped_range()
                .map_err(|e| BenchError::Gpu(format!("external-cost stamp range: {e}")))?;
            data.as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_le_bytes(*b))
                .collect()
        };
        self.staging.unmap();
        Ok((0..frames)
            .map(|i| {
                let f = (warmup + i) as usize;
                let (open, begin, end) = (ticks[3 * f], ticks[3 * f + 1], ticks[3 * f + 2]);
                let ns = |a: u64, b: u64| (b > a).then(|| (b - a) as f64 / 1.0e9);
                (ns(open, end), ns(begin, end))
            })
            .collect())
    }
}

/// One frame's resolved path-`c` stamps: `(handoff, convert)` seconds.
type StampPair = (Option<f64>, Option<f64>);

/// Per-frame sample of the report.
#[derive(Clone, serde::Serialize)]
struct CostSample {
    /// Host-side seconds: the producer fill plus path `e`'s
    /// import+install or path `c`'s stamp+plane upload.
    encode_seconds: f64,
    /// `Engine::render` wall seconds.
    submit_seconds: f64,
    /// Engine composite GPU seconds (the engine's timestamp queries
    /// resolved after completion); `null` where the adapter wrote none.
    gpu_seconds: Option<f64>,
    /// Path `c` only: GPU seconds from the pre-upload stamp to the
    /// convert pass's end — the plane copies, the conversion, and the
    /// scheduling gap between those submissions.
    handoff_seconds: Option<f64>,
    /// Path `c` only: GPU seconds of the convert pass alone.
    convert_seconds: Option<f64>,
    /// Render-thread CPU phases of the engine frame.
    phases: Vec<PhaseSample>,
}

/// One `external-cost` report.
#[derive(serde::Serialize)]
struct CostReport {
    adapter: String,
    backend: String,
    driver: String,
    driver_info: String,
    /// `external` (retained planes sampled in place) or `copy-convert`.
    path: &'static str,
    /// `nv12` or `p010`.
    layout: &'static str,
    /// `bt709-sdr` or `bt2020-pq`.
    transfer: &'static str,
    width: u32,
    height: u32,
    warmup_frames: u32,
    measured_frames: u32,
    /// One sample per measured frame.
    samples: Vec<CostSample>,
    /// `gpu_seconds` percentiles `[p50, p90, p99]` — the engine's own
    /// submission: the whole GPU cost of path `e`, path `c`'s composite
    /// of the converted image.
    composite_seconds: Option<[f64; 3]>,
    /// Path `c`'s `handoff_seconds` percentiles.
    handoff_seconds: Option<[f64; 3]>,
    /// Path `c`'s `convert_seconds` percentiles.
    convert_seconds: Option<[f64; 3]>,
    /// The whole per-frame GPU cost: `gpu_seconds` for path `e`,
    /// `composite + handoff` for path `c`.
    total_seconds: Option<[f64; 3]>,
    /// Host-side per-frame work percentiles.
    encode_seconds: [f64; 3],
    submit_seconds: [f64; 3],
    pacing: Option<Pacing>,
    energy: Option<EnergyReport>,
    conditions: Conditions,
    memory: MemoryReport,
    device: DeviceInfo,
    note: &'static str,
}

/// Maps a render-time error into a `BenchError`.
#[expect(
    clippy::needless_pass_by_value,
    reason = "Timings::render_frame takes a fn(RenderError) pointer"
)]
fn render_error(e: cherenkov::RenderError) -> BenchError {
    BenchError::Engine(format!("cherenkov render: {e}"))
}

/// The engine + shared device every external-cost run builds on.
fn engine_and_device() -> Result<(Engine<Gpu>, SharedDevice), BenchError> {
    let config = GpuConfig {
        timestamps: true,
        ..GpuConfig::default()
    };
    let shared = SharedDevice::create(&config)
        .map_err(|e| BenchError::Gpu(format!("external-cost device: {e}")))?;
    let engine = Engine::<Gpu>::new(GpuConfig {
        device: Some(shared.clone()),
        timestamps: true,
        ..GpuConfig::default()
    })
    .map_err(|e| BenchError::Gpu(format!("external-cost engine: {e}")))?;
    Ok((engine, shared))
}

/// Snapshots engine counters + the shared device's allocator.
fn memory_snapshot(
    engine: &Engine<Gpu>,
    shared: &SharedDevice,
    detail: SampleDetail,
) -> MemorySnapshot {
    let usage = engine.memory();
    let info = shared.adapter.get_info();
    MemorySnapshot::capture(
        AdapterMemory {
            engine: Reading::Measured(EngineBytes {
                cpu_bytes: usage.cpu.0,
                gpu_bytes: usage.gpu.0,
                backdrop_capture_bytes: usage.backdrop_captures.0,
            }),
            wgpu_allocator: wgpu_allocator(&shared.device, info.backend),
            skia_budgeted: Reading::unavailable("not a Skia adapter"),
            vk_memory_budget: wgpu_vk_memory_budget(&shared.device, info.backend, &info.name),
        },
        detail,
    )
}

/// How far past a pacing deadline a frame may start before it counts as
/// missed — `measure`'s tolerance.
const PACING_TOLERANCE: Duration = Duration::from_millis(1);

/// Runs the measurement and writes the JSON report.
///
/// # Errors
/// Platform producer allocation, GPU setup, metering and I/O failures.
#[expect(
    clippy::too_many_lines,
    reason = "one linear setup-measure-report sequence"
)]
pub(crate) fn run(args: &ExternalCostArgs) -> Result<(), BenchError> {
    let ExternalCostArgs {
        path,
        size,
        transfer,
        frames,
        warmup,
        rate,
        energy: measure_energy,
        cpu,
        out,
    } = args;
    let (frames, warmup, rate, measure_energy, out) =
        (*frames, *warmup, *rate, *measure_energy, out.as_path());
    let spec = Spec::new(*size, *transfer);
    let path = *path;
    if measure_energy {
        energy::Meter::probe()?;
    }
    if let Some(cpus) = cpu {
        affinity::pin_current_thread(cpus)?;
    }
    let (engine, shared) = engine_and_device()?;
    let timestamps = shared
        .device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY);
    let adapter = shared.adapter.get_info();
    let idle = memory_snapshot(&engine, &shared, SampleDetail::Full);
    let producer = platform::Producer::new(&spec, &shared)?;
    let surface = engine
        .surface(Offscreen::new(
            (spec.width, spec.height),
            OffscreenFormat::LinearF16,
        ))
        .map_err(|e| BenchError::Gpu(format!("external-cost surface: {e}")))?;
    let layer = surface.layer();
    let total = warmup + frames;

    // Path setup: `e` installs a fresh external frame per produced
    // buffer; `c` installs the convert GpuContent once and uploads the
    // planes per frame.
    let copy = match path {
        ExternalPath::External => {
            surface.update(|tx| {
                tx[surface.root()].push(&layer);
            });
            None
        }
        ExternalPath::Copy => {
            if !timestamps {
                return Err(BenchError::Gpu(
                    "external-cost --path c needs TIMESTAMP_QUERY for the handoff stamps".into(),
                ));
            }
            let copy = CopyPath::new(&shared, &spec, 3 * total);
            let params_buffer = shared.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("external-cost convert params"),
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            shared
                .queue
                .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params(&spec)));
            let content = Convert {
                y: copy.y.create_view(&wgpu::TextureViewDescriptor::default()),
                uv: copy.uv.create_view(&wgpu::TextureViewDescriptor::default()),
                params: params_buffer,
                queries: Arc::clone(&copy.queries),
                frame: 0,
                live: None,
            };
            let handle = engine.gpu_content(
                (spec.width, spec.height),
                GpuContentBox::new(content, || {}),
            );
            surface.update(|tx| {
                tx[surface.root()].push(&layer);
                tx[&layer].content(handle);
            });
            Some(copy)
        }
    };

    let preparation = memory_snapshot(&engine, &shared, SampleDetail::Full);
    let mut warmup_snapshots = Vec::with_capacity(warmup as usize);
    let mut samples: Vec<CostSample> = Vec::with_capacity(frames as usize);
    let mut timings = Timings::default();
    let mut clock = Clock::new();
    let period = rate.map(|hz| Duration::from_secs_f64(1.0 / hz));
    let window_hint = period.map_or(Duration::from_secs(1), |p| p * total);
    let mut meter = None;
    let mut start = Instant::now();
    let mut missed_deadlines = 0u32;

    for frame in 0..total {
        if frame == warmup {
            if measure_energy {
                meter = Some(energy::Meter::begin(window_hint)?);
            }
            start = Instant::now();
        }
        if frame >= warmup
            && let Some(period) = period
            && let Some(deadline) = period
                .checked_mul(frame - warmup)
                .and_then(|d| start.checked_add(d))
        {
            let now = Instant::now();
            if now < deadline {
                std::thread::sleep(deadline - now);
                if Instant::now().saturating_duration_since(deadline) > PACING_TOLERANCE {
                    missed_deadlines += 1;
                }
            } else if frame > warmup {
                missed_deadlines += 1;
            }
        }
        let t0 = Instant::now();
        producer.fill(frame);
        match path {
            ExternalPath::External => {
                let external = producer.external(frame, spec.color)?;
                let handle = engine.external_frame(external);
                surface.update(|tx| {
                    tx[&layer].content(handle);
                });
            }
            ExternalPath::Copy => {
                copy.as_ref()
                    .expect("path c has copy state")
                    .upload(&producer, &shared, frame);
            }
        }
        let t1 = Instant::now();
        timings.render_frame(&engine, &mut clock, u64::from(frame), false, render_error)?;
        clock.advance();
        let stats = engine.stats();
        let t2 = Instant::now();
        if frame < warmup {
            warmup_snapshots.push(memory_snapshot(&engine, &shared, SampleDetail::Frame));
        } else {
            let phases = stats.phases;
            samples.push(CostSample {
                encode_seconds: t1.duration_since(t0).as_secs_f64(),
                submit_seconds: t2.duration_since(t1).as_secs_f64(),
                gpu_seconds: None,
                handoff_seconds: None,
                convert_seconds: None,
                phases: [
                    ("lower", phases.lower_seconds),
                    ("encode", phases.encode_seconds),
                    ("stamp", phases.stamp_seconds),
                    ("wait", phases.wait_seconds),
                ]
                .into_iter()
                .map(|(name, seconds)| PhaseSample {
                    name: name.to_string(),
                    seconds,
                })
                .collect(),
            });
        }
    }

    let end = Instant::now();
    let energy_outcome = match meter {
        Some(meter) => Some(meter.finish(start, end, frames)?),
        None => None,
    };
    let steady = memory_snapshot(&engine, &shared, SampleDetail::Full);
    engine.trim(cherenkov::Pressure::Critical);
    let post_retire = memory_snapshot(&engine, &shared, SampleDetail::Full);

    // Engine-side GPU timings resolve after completion: each rendered
    // frame's composite submission lands in `gpu_seconds`.
    for timing in timings.samples(engine.finish_timings().map_err(render_error)?) {
        let Some(index) = timing.frame.checked_sub(u64::from(warmup)) else {
            continue;
        };
        samples[usize::try_from(index).expect("a frame index fits usize")].gpu_seconds =
            timing.gpu_seconds;
    }

    // Resolve the path-c stamps: per global frame f, `3f` is the
    // pre-upload stamp and `3f+1`/`3f+2` bracket the convert pass.
    if let Some(copy) = &copy {
        for (sample, (handoff, convert)) in samples
            .iter_mut()
            .zip(copy.read_stamps(&shared, warmup, frames)?)
        {
            sample.handoff_seconds = handoff;
            sample.convert_seconds = convert;
        }
    }

    let composited: Vec<f64> = samples.iter().filter_map(|s| s.gpu_seconds).collect();
    let handoff: Vec<f64> = samples.iter().filter_map(|s| s.handoff_seconds).collect();
    let convert: Vec<f64> = samples.iter().filter_map(|s| s.convert_seconds).collect();
    let totals: Vec<f64> = samples
        .iter()
        .map(|s| s.gpu_seconds.unwrap_or(0.0) + s.handoff_seconds.unwrap_or(0.0))
        .collect();
    let encode: Vec<f64> = samples.iter().map(|s| s.encode_seconds).collect();
    let submit: Vec<f64> = samples.iter().map(|s| s.submit_seconds).collect();
    let window_seconds = end.duration_since(start).as_secs_f64();
    let pacing = period.map(|period| Pacing {
        requested_hz: 1.0 / period.as_secs_f64(),
        achieved_hz: f64::from(frames) / window_seconds,
        missed_deadlines,
        window_seconds,
    });

    let report = CostReport {
        adapter: adapter.name.clone(),
        backend: format!("{:?}", adapter.backend),
        driver: adapter.driver.clone(),
        driver_info: adapter.driver_info.clone(),
        path: match path {
            ExternalPath::External => "external",
            ExternalPath::Copy => "copy-convert",
        },
        layout: spec.layout(),
        transfer: spec.transfer_name(),
        width: spec.width,
        height: spec.height,
        warmup_frames: warmup,
        measured_frames: frames,
        samples,
        composite_seconds: percentiles(&composited),
        handoff_seconds: percentiles(&handoff),
        convert_seconds: percentiles(&convert),
        total_seconds: percentiles(&totals),
        encode_seconds: percentiles(&encode).unwrap_or([0.0; 3]),
        submit_seconds: percentiles(&submit).unwrap_or([0.0; 3]),
        pacing,
        conditions: conditions::collect(
            energy_outcome
                .as_ref()
                .and_then(|o| o.thermal_pressure.clone()),
        ),
        energy: energy_outcome.map(|o| o.report),
        memory: MemoryReport::new(
            idle,
            &{
                let mut all = vec![preparation];
                all.extend(warmup_snapshots);
                all.push(steady);
                all
            },
            Some(post_retire),
        ),
        device: DeviceInfo {
            adapter: Some(adapter.name),
            backend: Some(format!("{:?}", adapter.backend)),
            driver: Some(adapter.driver),
            driver_info: Some(adapter.driver_info),
            vendor: Some(adapter.vendor),
            device: Some(adapter.device),
            target_format: Some("Rgba16Float".to_string()),
            cpu: crate::cpu_model(),
            thermal_celsius: crate::thermal_celsius(),
        },
        note: "gpu_seconds = the engine's composite submission; path c adds a \
               marker-stamp window over the plane upload and the convert pass",
    };
    let file = std::fs::File::create(out)
        .map_err(|e| BenchError::Gpu(format!("write {}: {e}", out.display())))?;
    serde_json::to_writer_pretty(file, &report)
        .map_err(|e| BenchError::Gpu(format!("report json: {e}")))?;
    tracing::info!(out = %out.display(), "external-cost");
    Ok(())
}

/// Renders one synthetic frame through `path` on a fresh engine and
/// returns the composited working-space pixels — the E-vs-C pair the
/// correctness test (`tests/external_cost.rs`) compares.
///
/// # Errors
/// Device, producer, engine and readback failures.
pub fn composite_frame(
    path: ExternalPath,
    size: ExternalSize,
    transfer: ExternalTransfer,
    frame: u32,
) -> Result<Vec<[f32; 4]>, BenchError> {
    let spec = Spec::new(size, transfer);
    let (engine, shared) = engine_and_device()?;
    let producer = platform::Producer::new(&spec, &shared)?;
    let surface = engine
        .surface(Offscreen::new(
            (spec.width, spec.height),
            OffscreenFormat::LinearF16,
        ))
        .map_err(|e| BenchError::Gpu(format!("external-cost surface: {e}")))?;
    let layer = surface.layer();

    producer.fill(frame);
    match path {
        ExternalPath::External => {
            let external = producer.external(frame, spec.color)?;
            let handle = engine.external_frame(external);
            surface.update(|tx| {
                tx[surface.root()].push(&layer);
                tx[&layer].content(handle);
            });
        }
        ExternalPath::Copy => {
            let copy = CopyPath::new(&shared, &spec, 3);
            let params_buffer = shared.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("external-cost convert params"),
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            shared
                .queue
                .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params(&spec)));
            let content = Convert {
                y: copy.y.create_view(&wgpu::TextureViewDescriptor::default()),
                uv: copy.uv.create_view(&wgpu::TextureViewDescriptor::default()),
                params: params_buffer,
                queries: copy.queries.clone(),
                frame: 0,
                live: None,
            };
            let handle = engine.gpu_content(
                (spec.width, spec.height),
                GpuContentBox::new(content, || {}),
            );
            surface.update(|tx| {
                tx[surface.root()].push(&layer);
                tx[&layer].content(handle);
            });
            copy.upload(&producer, &shared, frame);
        }
    }
    engine.render(FrameTime::now()).map_err(render_error)?;
    let rb = surface.readback().map_err(render_error)?;
    Ok(rb.pixels)
}

/// `CVPixelBuffer` producer — the Apple video-decode output model.
#[cfg(target_vendor = "apple")]
#[path = "external_cost/apple.rs"]
mod platform;

/// `AHardwareBuffer` producer — the Android video-decode output model.
#[cfg(target_os = "android")]
#[path = "external_cost/android.rs"]
mod platform;

/// No platform buffer API: `external-cost` fails fast.
#[cfg(not(any(target_vendor = "apple", target_os = "android")))]
mod platform {
    use super::{BenchError, ExternalFrame, FrameColor, SharedDevice, Spec, wgpu};

    /// The stub producer.
    pub struct Producer;

    impl Producer {
        /// Always fails: no platform video-buffer API exists here.
        pub fn new(_spec: &Spec, _shared: &SharedDevice) -> Result<Self, BenchError> {
            Err(BenchError::Gpu(
                "external-cost has no platform frame producer on this OS".into(),
            ))
        }

        /// Unreachable — [`Producer::new`] always fails.
        pub fn fill(&self, _frame: u32) {
            unreachable!("Producer::new failed")
        }

        /// Unreachable — [`Producer::new`] always fails.
        pub fn upload(
            &self,
            _frame: u32,
            _queue: &wgpu::Queue,
            _y: &wgpu::Texture,
            _uv: &wgpu::Texture,
        ) {
            unreachable!("Producer::new failed")
        }

        /// Unreachable — [`Producer::new`] always fails.
        pub fn external(
            &self,
            _frame: u32,
            _color: FrameColor,
        ) -> Result<ExternalFrame, BenchError> {
            unreachable!("Producer::new failed")
        }
    }
}
