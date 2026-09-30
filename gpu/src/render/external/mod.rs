//! Retained external frames (`cherenkov::ExternalFrames`).
//!
//! A slot owns the views over the producer's planes, the uniform that bakes
//! the frame's decode (range, matrix, primaries, transfer, siting) into
//! shader arguments, and the group-1 bind groups cached per mask texture.
//! The planes themselves are producer-owned `wgpu::Texture`s on the shared
//! device: dropping the slot — replacement, detach, surface or engine
//! teardown — retires the lease, nothing copies them.

use rustc_hash::FxHashMap;

use crate::interop::{
    ChromaOffset, ExternalFrame, FramePlanes, Primaries, RgbAlpha, Transfer, YuvMatrix, YuvRange,
};

/// Native external-frame import on Vulkan (issue #166).
#[cfg(all(unix, not(target_vendor = "apple")))]
pub mod vulkan;

/// External frame kinds, mirrored by `external.wgsl`.
const KIND_RGB: u32 = 0;
/// Params kind byte: 8-bit two-plane 4:2:0 (NV12 semantics).
pub const KIND_NV12: u32 = 1;
/// Params kind byte: 16-bit padded two-plane 4:2:0 (P010 semantics).
pub const KIND_P010: u32 = 2;

/// `Params::info.w` bit: swap the R and B samples of an RGB plane
/// (`Bgra8Unorm`).
const FLAG_BGR: u32 = 1 << 0;
/// `Params::info.w` bit: strip the low six padding bits of a P010 code
/// before normalization.
const FLAG_SHIFT6: u32 = 1 << 1;

/// The shader arguments for one retained frame.
///
/// All colour math that does not depend on the pixel is baked here on the
/// CPU: the range normalization, the `Y'CbCr` to `R'G'B'` matrix, the
/// primaries-to-working-space matrix (with the absolute-level scale folded
/// in) and the HLG OOTF parameters. The fragment stage applies them
/// verbatim; `external.wgsl` declares the same layout.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    /// x: plane kind (`KIND_*`), y: [`Transfer`] discriminant, z:
    /// [`RgbAlpha`] discriminant (RGB only), w: `FLAG_*`.
    info: [u32; 4],
    /// Sampled dimensions: luma or RGB plane `w, h`, then chroma `w, h`
    /// (zero for RGB planes).
    dims: [f32; 4],
    /// Code normalization `code * scale + offset`: luma then chroma. The
    /// chroma offsets include the −0.5 centring.
    norm: [f32; 4],
    /// x, y: the luma-space position offset of chroma texels on each axis
    /// (`0` cosited, `0.5` centered). z: the HLG OOTF gamma; w: unused.
    site: [f32; 4],
    /// `R'G'B' = y' * yuv[0] + cb' * yuv[1] + cr' * yuv[2] + yuv[3]` — column
    /// vectors plus a tail offset.
    yuv: [[f32; 4]; 4],
    /// Source primaries to extended linear Display P3, column-major, scaled
    /// so the decoded signal lands white-relative (absolute transfers divide
    /// their nits by `reference_white` here).
    prim: [[f32; 4]; 3],
    /// The Y row of the source primaries' RGB→XYZ matrix: the luma
    /// coefficients the HLG OOTF needs. Zero otherwise.
    luma: [f32; 4],
}

const _: () = assert!(
    std::mem::size_of::<Params>() as u64 == crate::render::bindings::EXTERNAL_PARAMS_SIZE,
    "external params must match the uniform's min_binding_size"
);

/// The `s` in "chroma texel `i` centres on luma position `2i + s`".
const fn siting(offset: ChromaOffset) -> f32 {
    match offset {
        ChromaOffset::Cosited => 0.0,
        ChromaOffset::Centered => 0.5,
    }
}

/// `Kr, Kb` of a `Y'CbCr` matrix.
const fn matrix_coeffs(matrix: YuvMatrix) -> (f32, f32) {
    match matrix {
        YuvMatrix::Bt601 => (0.299, 0.114),
        YuvMatrix::Bt709 => (0.2126, 0.0722),
        YuvMatrix::Bt2020 => (0.2627, 0.0593),
    }
}

/// xy chromaticities `(red, green, blue)` of a primaries set; all three use
/// D65.
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
    let inv =
        |a: usize, b: usize, c: usize, d: usize| m[a][d].mul_add(-m[c][b], m[a][b] * m[c][d]) / det;
    [
        [inv(1, 1, 2, 2), inv(0, 2, 2, 1), inv(0, 1, 1, 2)],
        [inv(1, 2, 2, 0), inv(0, 0, 2, 2), inv(0, 2, 1, 0)],
        [inv(1, 0, 2, 1), inv(0, 1, 2, 0), inv(0, 0, 1, 1)],
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
    // The matrix whose columns are the primaries maps S·(1,1,1) to white:
    // solve for the per-channel scales first.
    let unscaled = [
        [cols[0][0], cols[1][0], cols[2][0]],
        [cols[0][1], cols[1][1], cols[2][1]],
        [cols[0][2], cols[1][2], cols[2][2]],
    ];
    let d65 = to_xyz([0.3127, 0.3290]);
    let scales = mat_mul(mat_invert(unscaled), d65);
    let mut out = [[0.0; 3]; 3];
    for c in 0..3 {
        for r in 0..3 {
            out[r][c] = cols[c][r] * scales[c];
        }
    }
    out
}

/// The absolute-to-white-relative scale of a transfer: PQ decodes to nits,
/// HLG's OOTF yields scene-linear scaled to `hlg_peak` nits; relative
/// transfers already reach 1.0 at reference white.
fn value_scale(color: &crate::interop::FrameColor) -> f32 {
    match color.transfer {
        Transfer::Pq => 1.0 / color.reference_white,
        Transfer::Hlg => color.hlg_peak / color.reference_white,
        _ => 1.0,
    }
}

/// BT.2100 §3: the OOTF display gamma for an HLG peak luminance in nits.
fn hlg_gamma(peak_nits: f32) -> f32 {
    0.42f32.mul_add((peak_nits / 1000.0).log10(), 1.2)
}

/// Plane kind, alpha mode, flags and sampled sizes for one frame.
fn plane_contract(frame: &ExternalFrame) -> (u32, u32, u32, wgpu::Extent3d, wgpu::Extent3d) {
    match &frame.planes {
        FramePlanes::Yuv { y, uv } => {
            let (kind, shift) = match y.format() {
                wgpu::TextureFormat::R8Uint => (KIND_NV12, 0),
                _ => (KIND_P010, FLAG_SHIFT6),
            };
            (kind, 0, shift, y.size(), uv.size())
        }
        FramePlanes::Rgb { plane, alpha } => {
            let bgr = if plane.format() == wgpu::TextureFormat::Bgra8Unorm {
                FLAG_BGR
            } else {
                0
            };
            let alpha = match alpha {
                RgbAlpha::Opaque => 0,
                RgbAlpha::Straight => 1,
                RgbAlpha::Premultiplied => 2,
            };
            (KIND_RGB, alpha, bgr, plane.size(), plane.size())
        }
        #[cfg(all(unix, not(target_vendor = "apple")))]
        FramePlanes::Native(frame) => {
            let size = wgpu::Extent3d {
                width: frame.size().0,
                height: frame.size().1,
                depth_or_array_layers: 1,
            };
            let chroma = wgpu::Extent3d {
                width: size.width.div_ceil(2),
                height: size.height.div_ceil(2),
                depth_or_array_layers: 1,
            };
            match frame.repr() {
                vulkan::Repr::Rgb { format } => {
                    let bgr = if format == wgpu::TextureFormat::Bgra8Unorm {
                        FLAG_BGR
                    } else {
                        0
                    };
                    let alpha = match frame.generation.alpha {
                        RgbAlpha::Opaque => 0,
                        RgbAlpha::Straight => 1,
                        RgbAlpha::Premultiplied => 2,
                    };
                    (KIND_RGB, alpha, bgr, size, size)
                }
                vulkan::Repr::Planes { kind } => {
                    let shift = if kind == KIND_P010 { FLAG_SHIFT6 } else { 0 };
                    (kind, 0, shift, size, chroma)
                }
                // The sampler conversion yields encoded `R'G'B'`; the
                // shader decodes it like an opaque RGB plane.
                vulkan::Repr::ExternalFormat { .. } => (KIND_RGB, 0, 0, size, size),
            }
        }
    }
}

/// Normalization `code * scale + offset` landing on `[0, 1]` of the code's
/// nominal range; the chroma offsets fold in the −0.5 centring. P010 codes
/// normalize after `FLAG_SHIFT6` strips their padding bits.
const fn code_norm(range: YuvRange, kind: u32) -> [f32; 4] {
    match (range, kind) {
        (YuvRange::Video, KIND_NV12) => [1.0 / 219.0, -16.0 / 219.0, 1.0 / 224.0, -128.0 / 224.0],
        (YuvRange::Full, KIND_NV12) => [1.0 / 255.0, 0.0, 1.0 / 255.0, -128.0 / 255.0],
        (YuvRange::Video, _) => [1.0 / 876.0, -64.0 / 876.0, 1.0 / 896.0, -512.0 / 896.0],
        (YuvRange::Full, _) => [1.0 / 1023.0, 0.0, 1.0 / 1023.0, -512.0 / 1023.0],
    }
}

/// `R'G'B'` decode columns for a `Y'CbCr` matrix: the standard
/// non-constant-luminance reconstruction
/// `G = Y − 2Kb(1−Kb)/Kg·Cb − 2Kr(1−Kr)/Kg·Cr`.
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

/// Source-primaries → extended linear Display P3 columns and the source
/// luma row, with the absolute-level scale folded into the columns.
fn primaries_params(color: &crate::interop::FrameColor) -> ([[f32; 4]; 3], [f32; 4]) {
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
    (prim, [to_xyz[1][0], to_xyz[1][1], to_xyz[1][2], 0.0])
}

/// Bakes an `ExternalFrame`'s decode into its shader arguments.
#[expect(clippy::cast_precision_loss, reason = "texture dimensions fit f32")]
pub fn params(frame: &ExternalFrame) -> Params {
    let color = &frame.color;
    let (kind, alpha, flags, plane_size, chroma_size) = plane_contract(frame);
    let (prim, luma) = primaries_params(color);
    Params {
        info: [kind, color.transfer as u32, alpha, flags],
        dims: [
            plane_size.width as f32,
            plane_size.height as f32,
            chroma_size.width as f32,
            chroma_size.height as f32,
        ],
        norm: code_norm(color.range, kind),
        site: [
            siting(color.chroma_siting.x),
            siting(color.chroma_siting.y),
            if color.transfer == Transfer::Hlg {
                hlg_gamma(color.hlg_peak)
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

/// The clip-mask part of an external draw's bind.
#[derive(Clone, Copy)]
pub struct MaskBinding<'a> {
    /// The pass's mask texture key (`None` = no mask).
    pub key: Option<u64>,
    /// Its view (`None` binds the f32 dummy).
    pub view: Option<&'a wgpu::TextureView>,
    /// The atlas's mask-texture generation; a change clears the bind cache.
    pub generation: u64,
}

/// One retained frame bound to a layer.
///
/// Dropping the slot — a replacement frame, `LayerEdit::content`, layer
/// removal, or surface teardown — releases the views, params buffer and
/// binds, which is the lease the producer's planes retire on.
pub struct Slot {
    /// The frame as installed; its `wait` is consumed by the Apple submit
    /// loop, and holding it is the lease that keeps the planes resident.
    #[cfg_attr(
        not(target_vendor = "apple"),
        expect(
            dead_code,
            reason = "frame.wait is read only by the Apple Metal submit loop"
        )
    )]
    pub frame: ExternalFrame,
    /// The emitted quad's size in layer-local space: the luma or RGB plane
    /// dimensions.
    pub size: (u32, u32),
    /// The native generation on Vulkan — `Some` for `FramePlanes::Native`
    /// slots. A `Repr::Rgb` generation still draws on the ordinary external
    /// pipeline through `rgb`; multiplanar and external-format generations
    /// draw in the Vulkan native operation.
    #[cfg(all(unix, not(target_vendor = "apple")))]
    native: Option<vulkan::Frame>,
    /// The luma plane view for YUV frames.
    y: Option<wgpu::TextureView>,
    /// The interleaved chroma plane view for YUV frames.
    uv: Option<wgpu::TextureView>,
    /// The plane view for RGB frames.
    rgb: Option<wgpu::TextureView>,
    /// The baked [`Params`], written once at install.
    params: wgpu::Buffer,
    /// Group-1 binds cached per mask texture key (`u64::MAX` = no mask).
    binds: FxHashMap<u64, wgpu::BindGroup>,
    /// The mask-texture generation `binds` was built under; a change clears
    /// the cache so a re-created mask texture rebinds.
    binds_gen: u64,
}

impl Slot {
    /// GPU bytes one slot owns: its params buffer. The planes are producer
    /// memory.
    pub const GPU_BYTES: u64 = std::mem::size_of::<Params>() as u64;

    /// Creates the views and the params buffer for a frame.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, frame: ExternalFrame) -> Self {
        #[cfg(all(unix, not(target_vendor = "apple")))]
        let mut native = None;
        let (y, uv, rgb, size) = match &frame.planes {
            FramePlanes::Yuv { y, uv } => (
                Some(y.create_view(&wgpu::TextureViewDescriptor::default())),
                Some(uv.create_view(&wgpu::TextureViewDescriptor::default())),
                None,
                (y.size().width, y.size().height),
            ),
            FramePlanes::Rgb { plane, .. } => (
                None,
                None,
                Some(plane.create_view(&wgpu::TextureViewDescriptor::default())),
                (plane.size().width, plane.size().height),
            ),
            #[cfg(all(unix, not(target_vendor = "apple")))]
            FramePlanes::Native(frame) => {
                // An RGB-repr generation samples its wgpu-wrapped plane
                // through the ordinary path; the others bind natively.
                let rgb = match frame.repr() {
                    vulkan::Repr::Rgb { .. } => frame
                        .generation
                        .rgb_wrap
                        .as_ref()
                        .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default())),
                    _ => None,
                };
                // The engine-side lease: `vulkan_frame` clones held by the
                // producer don't count, so dropping this slot's clone is
                // the retirement the release submission waits on.
                frame.lease();
                native = Some(frame.clone());
                (None, None, rgb, frame.size())
            }
        };
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("external frame params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&params(&frame)));
        Self {
            frame,
            size,
            #[cfg(all(unix, not(target_vendor = "apple")))]
            native,
            y,
            uv,
            rgb,
            params: buffer,
            binds: FxHashMap::default(),
            binds_gen: 0,
        }
    }

    /// The native generation when the slot draws in the Vulkan native
    /// operation (`Planes` and `ExternalFormat` representations). An
    /// RGB-repr native frame binds its wrapped plane on the ordinary
    /// pipeline and returns `None`.
    #[cfg(all(unix, not(target_vendor = "apple")))]
    pub fn native_frame(&self) -> Option<&vulkan::Frame> {
        self.native
            .as_ref()
            .filter(|&frame| !matches!(frame.repr(), vulkan::Repr::Rgb { .. }))
    }

    /// The native generation on ANY representation — including `Rgb`,
    /// which still needs the acquire barrier and producer wait staged
    /// before its first wgpu use.
    #[cfg(all(unix, not(target_vendor = "apple")))]
    pub const fn vulkan_frame(&self) -> Option<&vulkan::Frame> {
        self.native.as_ref()
    }

    /// The params uniform the native operation binds per draw.
    #[cfg(all(unix, not(target_vendor = "apple")))]
    pub const fn params_buffer(&self) -> &wgpu::Buffer {
        &self.params
    }

    /// The group-1 bind for a draw under `mask`.
    ///
    /// Unused plane bindings carry the matching-type dummy so validation
    /// holds on every frame shape. `mask.generation` is the atlas's
    /// mask-texture generation: a rebuild clears the cache so a re-created
    /// mask texture rebinds.
    pub fn bind(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        mask: MaskBinding<'_>,
        dummy_f32: &wgpu::TextureView,
        dummy_uint: &wgpu::TextureView,
    ) -> &wgpu::BindGroup {
        if self.binds_gen != mask.generation {
            self.binds.clear();
            self.binds_gen = mask.generation;
        }
        let key = mask.key.unwrap_or(u64::MAX);
        self.binds.entry(key).or_insert_with(|| {
            let (y, uv, rgb) = if self.y.is_some() {
                (self.y.as_ref(), self.uv.as_ref(), None)
            } else {
                (None, None, self.rgb.as_ref())
            };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("external frame"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(y.unwrap_or(dummy_uint)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(uv.unwrap_or(dummy_uint)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(rgb.unwrap_or(dummy_f32)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            mask.view.unwrap_or(dummy_f32),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.params.as_entire_binding(),
                    },
                ],
            })
        })
    }
}

/// Dropping the slot ends the engine-side lease on the native frame; the
/// last such lease schedules the producer release (#166).
#[cfg(all(unix, not(target_vendor = "apple")))]
impl Drop for Slot {
    fn drop(&mut self) {
        if let Some(frame) = &self.native {
            frame.unlease();
        }
    }
}
