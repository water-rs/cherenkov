//! Device sharing and presentation of the engine's retained working-space output.

pub use crate::render::filter::EffectBox;
pub use crate::render::present::{OutputAlpha, OutputColor, Presenter, TextureOutput};
pub use crate::render::shaders::{ShaderDelivery, delivery as shader_delivery};
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// The device types and drawing contexts exposed to custom GPU producers.
pub mod wgpu {
    pub use ::wgpu::*;
    use std::time::Duration;

    /// Persistent device resources supplied once before rendering content.
    pub struct Context<'a> {
        /// Adapter that owns the engine's device.
        pub adapter: &'a Adapter,
        /// Engine-owned device.
        pub device: &'a Device,
        /// Engine-owned submission queue.
        pub queue: &'a Queue,
        /// Output texture format, containing premultiplied linear Display P3.
        pub format: TextureFormat,
        /// Requests a new frame after asynchronous producer work completes.
        pub redraw: super::RedrawHandle,
    }

    /// An engine-allocated output texture for one custom content frame.
    pub struct Frame<'a> {
        /// Engine-owned device.
        pub device: &'a Device,
        /// Engine-owned queue.
        pub queue: &'a Queue,
        /// Output texture in premultiplied linear Display P3.
        pub texture: &'a Texture,
        /// Output attachment view.
        pub view: &'a TextureView,
        /// Output format.
        pub format: TextureFormat,
        /// Texture width in physical pixels.
        pub width: u32,
        /// Texture height in physical pixels.
        pub height: u32,
        /// Display scale.
        pub scale: f32,
        /// Presentation time relative to this producer's first frame.
        pub elapsed: Duration,
        /// Time since the previous presentation of this content.
        pub delta: Duration,
        pub(crate) redraw: bool,
    }

    impl Frame<'_> {
        /// Keeps the engine refreshing for the next frame.
        pub const fn request_redraw(&mut self) {
            self.redraw = true;
        }
    }
}

/// A producer moved to the engine's render thread for its entire lifetime.
/// UI-thread-bound producers send owned frame data over a channel to this object.
pub trait GpuContent: cherenkov::RenderTransfer + 'static {
    /// Creates persistent resources once before the first frame.
    fn setup(&mut self, context: &wgpu::Context<'_>) -> impl Future<Output = ()>;
    /// Draws into the provided engine-owned attachment.
    fn render(&mut self, frame: &mut wgpu::Frame<'_>);
}

/// A producer boxed for `Engine::gpu_content`.
pub struct GpuContentBox {
    pub(crate) content: Box<dyn Content>,
    pub(crate) redraw: RedrawHandle,
}

impl std::fmt::Debug for GpuContentBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContentBox").finish_non_exhaustive()
    }
}

impl GpuContentBox {
    /// Creates a producer with the host's event-loop wake callback.
    /// The callback must be safe to invoke from a producer thread, including
    /// while the engine is idle (for example, a window event-loop proxy).
    #[must_use]
    pub fn new(content: impl GpuContent, wake: impl Fn() + Send + Sync + 'static) -> Self {
        Self {
            content: Box::new(content),
            redraw: RedrawHandle {
                dirty: Arc::new(AtomicBool::new(true)),
                active: Arc::new(AtomicBool::new(false)),
                wake: Arc::new(wake),
            },
        }
    }

    /// Obtains a redraw requester before the producer moves to the engine.
    #[must_use]
    pub fn redraw_handle(&self) -> RedrawHandle {
        self.redraw.clone()
    }
}

/// A thread-safe request to redraw this content on the next engine frame.
#[derive(Clone)]
pub struct RedrawHandle {
    pub(crate) dirty: Arc<AtomicBool>,
    pub(crate) active: Arc<AtomicBool>,
    wake: Arc<dyn Fn() + Send + Sync>,
}

impl std::fmt::Debug for RedrawHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedrawHandle")
            .field("dirty", &self.is_dirty())
            .finish_non_exhaustive()
    }
}

impl RedrawHandle {
    /// Marks the producer's output stale. Requests coalesce until consumed.
    /// Detached or removed content retains the request without waking the host;
    /// reattachment draws its latest state.
    pub fn request_redraw(&self) {
        if !self.dirty.swap(true, Ordering::AcqRel) && self.active.load(Ordering::Acquire) {
            (self.wake)();
        }
    }

    /// Whether the producer has an unconsumed redraw request.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }
}

pub(crate) trait Content: cherenkov::RenderTransfer {
    #[cfg(not(target_arch = "wasm32"))]
    fn setup(&mut self, context: &wgpu::Context<'_>);
    #[cfg(target_arch = "wasm32")]
    fn setup<'a>(
        &'a mut self,
        context: &'a wgpu::Context<'a>,
    ) -> core::pin::Pin<Box<dyn Future<Output = ()> + 'a>>;
    fn render(&mut self, frame: &mut wgpu::Frame<'_>);
}

impl<C: GpuContent> Content for C {
    #[cfg(not(target_arch = "wasm32"))]
    fn setup(&mut self, context: &wgpu::Context<'_>) {
        pollster::block_on(GpuContent::setup(self, context));
    }

    #[cfg(target_arch = "wasm32")]
    fn setup<'a>(
        &'a mut self,
        context: &'a wgpu::Context<'a>,
    ) -> core::pin::Pin<Box<dyn Future<Output = ()> + 'a>> {
        Box::pin(GpuContent::setup(self, context))
    }

    fn render(&mut self, frame: &mut wgpu::Frame<'_>) {
        GpuContent::render(self, frame);
    }
}

/// A host event-loop callback callable from the engine or producer threads.
#[derive(Clone)]
pub struct RedrawCallback(Arc<dyn Fn() + Send + Sync>);

impl RedrawCallback {
    /// Wraps the host's display-link or event-loop wake operation.
    pub fn new(wake: impl Fn() + Send + Sync + 'static) -> Self {
        Self(Arc::new(wake))
    }

    /// Asks the host to schedule an engine frame.
    pub fn wake(&self) {
        (self.0)();
    }
}

impl std::fmt::Debug for RedrawCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedrawCallback").finish_non_exhaustive()
    }
}

/// An existing device shared with a native presentation host.
/// All four handles must belong to the same device creation chain.
///
/// On Vulkan and Metal adapters the device must be created with
/// `wgpu::Features::PASSTHROUGH_SHADERS`: the engine's fixed shaders are
/// precompiled binaries loaded through the passthrough API (issue #57), so
/// `Engine::new` fails explicitly on a device created without the feature.
#[derive(Clone, Debug)]
pub struct SharedDevice {
    /// Instance used to create the adapter and native surfaces.
    pub instance: wgpu::Instance,
    /// Adapter used to create the device.
    pub adapter: wgpu::Adapter,
    /// Device used for both engine composition and native presentation.
    /// Request `Features::PASSTHROUGH_SHADERS` on Vulkan and Metal.
    pub device: wgpu::Device,
    /// This device's submission queue.
    pub queue: wgpu::Queue,
}

/// A surface whose engine-owned linear P3 texture is handed to a native host.
///
/// The host receives a new texture only on creation and resize, then samples
/// the retained texture after `Engine::render` completes. Presentation must
/// use the same device supplied through `GpuConfig::device`. Dropping the
/// receiver stops notifications; it does not destroy the surface. A host must
/// consume resize notifications before sampling output after a resize.
#[derive(Debug)]
pub struct TextureTarget {
    pub(crate) size: (u32, u32),
    pub(crate) refresh: cherenkov::RefreshRange,
    pub(crate) textures: std::sync::mpsc::Sender<wgpu::Texture>,
}

impl TextureTarget {
    /// Sets the refresh range for animated content.
    ///
    /// # Panics
    /// When the range is empty or includes zero.
    #[must_use]
    pub fn rate(mut self, rate: cherenkov::RefreshRange) -> Self {
        assert!(
            *rate.start() > 0 && !rate.is_empty(),
            "refresh range must be positive and ordered"
        );
        self.refresh = rate;
        self
    }

    /// Creates a target and its resize notification channel.
    /// Textures contain premultiplied linear Display P3, in `Rgba16Float`.
    #[must_use]
    pub fn new(size: (u32, u32)) -> (Self, std::sync::mpsc::Receiver<wgpu::Texture>) {
        let (textures, receiver) = std::sync::mpsc::channel();
        (
            Self {
                size,
                textures,
                refresh: cherenkov::DEFAULT_REFRESH,
            },
            receiver,
        )
    }
}

/// Compiles producer WGSL with color conversion helpers.
///
/// `cherenkov_srgb`
/// accepts straight-alpha encoded sRGB; `cherenkov_premultiplied_srgb` accepts
/// encoded-domain premultiplied sRGB. Both return premultiplied linear Display P3.
/// Extended signed components are preserved.
///
/// # Panics
/// Invalid WGSL is reported through wgpu's configured error handler.
#[must_use]
pub fn shader_module(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(
            format!("{}\n{source}", include_str!("render/color.wgsl")).into(),
        ),
    })
}

/// A retained external frame the engine composites in place.
///
/// The frame references textures that live on the engine's shared device;
/// the engine samples them where the owning layer composes, with no plane
/// copy and no raster fallback. Installing a frame is
/// `engine.external_frame(frame)` followed by `tx[&layer].content(handle)`:
/// the planes stay alive until the frame is replaced, the layer's content is
/// detached or replaced, or the surface or engine is torn down.
///
/// A new frame wakes the surface through the same coalesced waker as
/// recorded content; the engine never polls the producer.
#[derive(Debug)]
pub struct ExternalFrame {
    /// The textures the fragment stage samples.
    pub planes: FramePlanes,
    /// How the planes decode into the working space.
    pub color: FrameColor,
    /// An optional GPU-side wait the sampled planes are ordered behind.
    pub wait: Option<FrameSync>,
}

/// The planes an [`ExternalFrame`] samples.
#[derive(Debug)]
pub enum FramePlanes {
    /// Two-plane 4:2:0 YUV: one luma plane and one interleaved chroma plane.
    ///
    /// NV12 is `R8Uint` + `Rg8Uint`; P010 is `R16Uint` + `Rg16Uint`. For P010
    /// the low six padding bits of every code are stripped before decode.
    /// Chroma dimensions must be `ceil(luma / 2)` on each axis.
    Yuv {
        /// The luma plane: `R8Uint` for NV12, `R16Uint` for P010.
        y: wgpu::Texture,
        /// The interleaved chroma plane: `Rg8Uint` for NV12, `Rg16Uint` for
        /// P010.
        uv: wgpu::Texture,
    },
    /// A single interleaved RGB(A) plane.
    ///
    /// `Rgba8Unorm`, `Bgra8Unorm` or `Rgba16Float`; gamma-encoded formats are
    /// rejected because the decoder applies `color.transfer` itself.
    Rgb {
        /// The pixel plane.
        plane: wgpu::Texture,
        /// How the plane's alpha channel composes.
        alpha: RgbAlpha,
    },
}

/// How a [`FramePlanes::Rgb`] plane's alpha composes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RgbAlpha {
    /// Every pixel is fully opaque.
    Opaque,
    /// Straight (unpremultiplied) alpha.
    Straight,
    /// Premultiplied alpha in the encoded domain.
    Premultiplied,
}

/// The `Y'CbCr` matrix of a [`FramePlanes::Yuv`] frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YuvMatrix {
    /// BT.601 (`Kr = 0.299`, `Kb = 0.114`).
    Bt601,
    /// BT.709 (`Kr = 0.2126`, `Kb = 0.0722`).
    Bt709,
    /// BT.2020 (`Kr = 0.2627`, `Kb = 0.0593`).
    Bt2020,
}

/// The code range of a [`FramePlanes::Yuv`] frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YuvRange {
    /// Studio range: luma `16..=235`, chroma `16..=240` at 8 bits.
    Video,
    /// Full code range.
    Full,
}

/// The position of chroma samples on one axis, relative to luma samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaOffset {
    /// Co-sited with the even luma sample.
    Cosited,
    /// Centred between the two luma samples the chroma covers.
    Centered,
}

/// The two-axis chroma sample location of a [`FramePlanes::Yuv`] frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChromaSiting {
    /// Chroma position on the horizontal axis.
    pub x: ChromaOffset,
    /// Chroma position on the vertical axis.
    pub y: ChromaOffset,
}

impl ChromaSiting {
    /// Centered on both axes (JPEG convention).
    pub const CENTERED: Self = Self {
        x: ChromaOffset::Centered,
        y: ChromaOffset::Centered,
    };
    /// Co-sited horizontally, centered vertically (MPEG-2 convention).
    pub const LEFT: Self = Self {
        x: ChromaOffset::Cosited,
        y: ChromaOffset::Centered,
    };
    /// Co-sited on both axes.
    pub const TOP_LEFT: Self = Self {
        x: ChromaOffset::Cosited,
        y: ChromaOffset::Cosited,
    };
}

/// The additive primaries a frame's decoded signal is expressed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primaries {
    /// BT.709 / sRGB primaries.
    Bt709,
    /// Display P3 primaries.
    DisplayP3,
    /// BT.2020 primaries.
    Bt2020,
}

/// The opto-electronic transfer of a frame's encoded signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transfer {
    /// Already linear light.
    Linear,
    /// The sRGB piecewise curve.
    Srgb,
    /// The BT.709 OETF piecewise curve.
    Bt709,
    /// SMPTE ST 2084 perceptual quantizer; decodes to absolute nits.
    Pq,
    /// The BT.2100 hybrid log-gamma scene transfer; the signal's own OOTF
    /// applies at decode.
    Hlg,
}

/// How a frame's planes decode into the working space.
///
/// For YUV planes the matrix, range and chroma siting decode `Y'CbCr` codes
/// into `R'G'B'`; the primaries, transfer and reference level then map that
/// signal into extended linear Display P3, the engine's working space, where
/// 1.0 is reference white. For RGB planes only the primaries, transfer and
/// reference level apply; the YUV fields are ignored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameColor {
    /// The `Y'CbCr` matrix. Ignored for RGB planes.
    pub matrix: YuvMatrix,
    /// The code range. Ignored for RGB planes.
    pub range: YuvRange,
    /// The two-axis chroma sample location. Ignored for RGB planes.
    pub chroma_siting: ChromaSiting,
    /// The primaries the decoded signal is expressed on.
    pub primaries: Primaries,
    /// The transfer the signal is encoded with.
    pub transfer: Transfer,
    /// The nits reference white occupies for this signal.
    ///
    /// Absolute transfers ([`Transfer::Pq`], [`Transfer::Hlg`]) decode to
    /// nits and are divided by this to reach the engine's white-relative
    /// working space; for relative transfers it is documentation only — the
    /// decoded signal already reaches 1.0 at reference white.
    pub reference_white: f32,
    /// The nits the HLG OOTF was produced for. Only read for
    /// [`Transfer::Hlg`].
    pub hlg_peak: f32,
}

impl FrameColor {
    /// BT.709 studio-range video with BT.709 transfer and primaries.
    pub const BT709_VIDEO: Self = Self {
        matrix: YuvMatrix::Bt709,
        range: YuvRange::Video,
        chroma_siting: ChromaSiting::LEFT,
        primaries: Primaries::Bt709,
        transfer: Transfer::Bt709,
        reference_white: 203.0,
        hlg_peak: 0.0,
    };
    /// BT.2020 full-range video with PQ transfer on BT.2020 primaries.
    pub const BT2020_PQ: Self = Self {
        matrix: YuvMatrix::Bt2020,
        range: YuvRange::Video,
        chroma_siting: ChromaSiting::LEFT,
        primaries: Primaries::Bt2020,
        transfer: Transfer::Pq,
        reference_white: 203.0,
        hlg_peak: 0.0,
    };
    /// BT.2020 studio-range video with HLG transfer on BT.2020 primaries.
    ///
    /// `peak` is the display peak the HLG OOTF was produced for, in nits.
    #[must_use]
    pub const fn bt2020_hlg(peak: f32) -> Self {
        Self {
            matrix: YuvMatrix::Bt2020,
            range: YuvRange::Video,
            chroma_siting: ChromaSiting::LEFT,
            primaries: Primaries::Bt2020,
            transfer: Transfer::Hlg,
            reference_white: 203.0,
            hlg_peak: peak,
        }
    }
    /// sRGB on BT.709 primaries, for 8-bit RGB planes.
    pub const SRGB: Self = Self {
        matrix: YuvMatrix::Bt709,
        range: YuvRange::Full,
        chroma_siting: ChromaSiting::CENTERED,
        primaries: Primaries::Bt709,
        transfer: Transfer::Srgb,
        reference_white: 203.0,
        hlg_peak: 0.0,
    };
    /// Linear-light Display P3, for `Rgba16Float` RGB planes.
    pub const LINEAR_P3: Self = Self {
        matrix: YuvMatrix::Bt709,
        range: YuvRange::Full,
        chroma_siting: ChromaSiting::CENTERED,
        primaries: Primaries::DisplayP3,
        transfer: Transfer::Linear,
        reference_white: 203.0,
        hlg_peak: 0.0,
    };
}

/// A GPU-side wait an [`ExternalFrame`] is ordered behind.
///
/// The engine encodes the wait on the queue inside the submission that
/// samples the frame's planes; it never waits on the CPU. A producer
/// signalling later work simply installs the next frame with its own sync.
#[non_exhaustive]
#[derive(Debug)]
pub enum FrameSync {
    /// A Metal shared event and the value it must reach.
    ///
    /// The producer signals `value` on its own submission; the engine's wait
    /// runs on the GPU before the frame's planes are read.
    #[cfg(target_vendor = "apple")]
    Metal {
        /// The shared event to wait on.
        event: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent>>,
        /// The value the event must reach.
        value: u64,
    },
}

/// Why an [`ExternalFrame`] was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum InvalidFrame {
    /// A plane's pixel format does not match its role.
    #[error(
        "a YUV frame needs R8Uint/Rg8Uint (NV12) or R16Uint/Rg16Uint (P010) planes; \
         an RGB frame needs Rgba8Unorm, Bgra8Unorm or Rgba16Float"
    )]
    PlaneFormat,
    /// The chroma plane is not `ceil(luma / 2)` on each axis, or a plane is
    /// empty.
    #[error("the chroma plane must be ceil(luma / 2) on each axis and no plane may be empty")]
    PlaneDimensions,
    /// A plane is not a single 2D mip level.
    #[error("a frame plane must be a single-layer 2D texture with one mip level")]
    PlaneGeometry,
    /// A plane was created without `TEXTURE_BINDING`.
    #[error("a frame plane needs TextureUsages::TEXTURE_BINDING")]
    PlaneUsage,
    /// `reference_white` or `hlg_peak` is not positive and finite.
    #[error("reference_white and hlg_peak must be positive and finite")]
    ColorLevel,
}

fn check_plane(texture: &wgpu::Texture) -> Result<wgpu::Extent3d, InvalidFrame> {
    let size = texture.size();
    if texture.dimension() != wgpu::TextureDimension::D2
        || size.depth_or_array_layers != 1
        || texture.mip_level_count() != 1
    {
        return Err(InvalidFrame::PlaneGeometry);
    }
    if size.width == 0 || size.height == 0 {
        return Err(InvalidFrame::PlaneDimensions);
    }
    if !texture
        .usage()
        .contains(wgpu::TextureUsages::TEXTURE_BINDING)
    {
        return Err(InvalidFrame::PlaneUsage);
    }
    Ok(size)
}

fn check_color(color: &FrameColor) -> Result<(), InvalidFrame> {
    if color.reference_white <= 0.0
        || !color.reference_white.is_finite()
        || (color.transfer == Transfer::Hlg
            && (color.hlg_peak <= 0.0 || !color.hlg_peak.is_finite()))
    {
        return Err(InvalidFrame::ColorLevel);
    }
    Ok(())
}

impl ExternalFrame {
    /// A two-plane 4:2:0 YUV frame: `y` is the luma plane and `uv` the
    /// interleaved chroma plane.
    ///
    /// # Errors
    /// [`InvalidFrame`] when a plane's format, geometry or usage does not
    /// meet the contract, or `color`'s levels are invalid.
    pub fn yuv(
        y: wgpu::Texture,
        uv: wgpu::Texture,
        color: FrameColor,
    ) -> Result<Self, InvalidFrame> {
        let y_size = check_plane(&y)?;
        let uv_size = check_plane(&uv)?;
        let yuv = matches!(
            (y.format(), uv.format()),
            (wgpu::TextureFormat::R8Uint, wgpu::TextureFormat::Rg8Uint)
                | (wgpu::TextureFormat::R16Uint, wgpu::TextureFormat::Rg16Uint)
        );
        if !yuv {
            return Err(InvalidFrame::PlaneFormat);
        }
        if uv_size.width != y_size.width.div_ceil(2) || uv_size.height != y_size.height.div_ceil(2)
        {
            return Err(InvalidFrame::PlaneDimensions);
        }
        check_color(&color)?;
        Ok(Self {
            planes: FramePlanes::Yuv { y, uv },
            color,
            wait: None,
        })
    }

    /// A single-plane RGB frame.
    ///
    /// # Errors
    /// [`InvalidFrame`] when the plane's format, geometry or usage does not
    /// meet the contract, or `color`'s levels are invalid.
    pub fn rgb(
        plane: wgpu::Texture,
        alpha: RgbAlpha,
        color: FrameColor,
    ) -> Result<Self, InvalidFrame> {
        check_plane(&plane)?;
        if !matches!(
            plane.format(),
            wgpu::TextureFormat::Rgba8Unorm
                | wgpu::TextureFormat::Bgra8Unorm
                | wgpu::TextureFormat::Rgba16Float
        ) {
            return Err(InvalidFrame::PlaneFormat);
        }
        check_color(&color)?;
        Ok(Self {
            planes: FramePlanes::Rgb { plane, alpha },
            color,
            wait: None,
        })
    }

    /// Orders the sampled planes behind a GPU-side sync.
    ///
    /// The wait runs on the GPU inside the submission that reads the planes.
    /// Only Apple targets have a sync primitive, so only they offer this.
    #[cfg(target_vendor = "apple")]
    #[must_use]
    pub fn sync(mut self, sync: FrameSync) -> Self {
        self.wait = Some(sync);
        self
    }
}

/// Apple interop: importing Metal resources onto the shared device.
#[cfg(target_vendor = "apple")]
pub mod metal {
    /// Wraps an `MTLTexture` as a `wgpu::Texture` on the engine's device, for
    /// use as an [`ExternalFrame`](super::ExternalFrame) plane.
    ///
    /// The texture is imported in place through wgpu-hal on the shared
    /// device — there is no copy and no conversion texture. The `MTLTexture`
    /// itself stays owned by the caller (typically an `IOSurface` plane made
    /// with `-[MTLDevice newTextureWithDescriptor:iosurface:plane:]` or a
    /// `CVMetalTextureCache` texture); the returned `wgpu::Texture` retains
    /// it until dropped, which the engine's frame leases tie to the
    /// retained frame's lifetime.
    ///
    /// `format` must describe the texels the `MTLTexture` holds exactly — a
    /// format the frame contract accepts, in the plane's role.
    ///
    /// # Safety
    /// `raw` must be a live `MTLTexture` created on the same `MTLDevice` the
    /// `wgpu::Device` wraps, or on another device in its peer group, and
    /// `format` must be byte-compatible with its pixel format. The texture
    /// must stay alive and unwritten-except-by-the-producer for as long as a
    /// frame referencing it can be in flight.
    ///
    /// # Panics
    /// When the `MTLTexture`'s dimensions do not fit `u32`.
    #[must_use]
    pub unsafe fn import_texture(
        device: &wgpu::Device,
        raw: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>,
        format: wgpu::TextureFormat,
    ) -> wgpu::Texture {
        use objc2_metal::MTLTexture;
        let extent = wgpu::Extent3d {
            width: u32::try_from(raw.width()).expect("plane width fits u32"),
            height: u32::try_from(raw.height()).expect("plane height fits u32"),
            depth_or_array_layers: 1,
        };
        let hal_texture = unsafe {
            wgpu::hal::metal::Device::texture_from_raw(
                raw,
                format,
                objc2_metal::MTLTextureType::Type2D,
                1,
                1,
                extent.into(),
                None,
            )
        };
        unsafe {
            device.create_texture_from_hal::<wgpu::hal::metal::Api>(
                hal_texture,
                &wgpu::TextureDescriptor {
                    label: Some("external frame plane"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::wgt::TextureUses::RESOURCE,
            )
        }
    }
}
