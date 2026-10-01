//! The Apple realization of system-compositor planes: a Core Animation
//! layer tree the engine owns under the host view's backing layer.
//!
//! ```text
//! host layer (the view's)
//! └─ root                       the surface, in points
//!    ├─ part 0: CAMetalLayer    engine content below the first plane
//!    ├─ plane 0                 pixel space: scale(1 / display scale)
//!    │  └─ level … level        one node per tree layer on the path:
//!    │     └─ display layer       transform → clip → scroll
//!    ├─ part 1: CAMetalLayer    engine content above plane 0
//!    └─ …
//! ```
//!
//! A promoted external frame is shown by an `AVSampleBufferDisplayLayer`
//! fed a `CVPixelBuffer` that wraps the frame's own `IOSurface`, not by an
//! IOSurface-backed `CALayer`. The display layer is the documented path for
//! uncompressed video frames: it reads the `CVImageBuffer` colour
//! attachments (matrix, primaries, transfer, chroma siting), so the system
//! applies its EDR tone mapping to PQ and HLG content, while a `CALayer`'s
//! `IOSurface` contents have no documented HDR metadata path (`CAEDRMetadata`
//! exists only on `CAMetalLayer`). It is also the only layer `FairPlay`
//! decrypts into, so protected frames (#212) reuse this realization.
//!
//! Engine parts present through `CAMetalLayer`s with
//! `presentsWithTransaction`, and every geometry change, part presentation
//! and frame hand-off of one frame commits in one `CATransaction`, so parts
//! and planes change on screen together.

use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use kurbo::{Affine, Rect, Vec2};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_av_foundation::{
    AVLayerVideoGravityResize, AVQueuedSampleBufferRendering, AVQueuedSampleBufferRenderingStatus,
    AVSampleBufferDisplayLayer, AVSampleBufferVideoRenderer,
};
use objc2_core_foundation::{
    CFBoolean, CFMutableDictionary, CFRetained, CFString, CGAffineTransform, CGPoint, CGRect,
    CGSize,
};
use objc2_core_media::{
    CMSampleBuffer, CMSampleTimingInfo, CMVideoFormatDescription,
    CMVideoFormatDescriptionCreateForImageBuffer, kCMSampleAttachmentKey_DisplayImmediately,
    kCMTimeInvalid,
};
use objc2_core_video::{
    CVAttachmentMode, CVPixelBuffer, CVPixelBufferCreateWithIOSurface,
    kCVImageBufferChromaLocation_Center, kCVImageBufferChromaLocation_Left,
    kCVImageBufferChromaLocation_Top, kCVImageBufferChromaLocation_TopLeft,
    kCVImageBufferChromaLocationBottomFieldKey, kCVImageBufferChromaLocationTopFieldKey,
    kCVImageBufferColorPrimaries_ITU_R_709_2, kCVImageBufferColorPrimaries_ITU_R_2020,
    kCVImageBufferColorPrimaries_P3_D65, kCVImageBufferColorPrimariesKey,
    kCVImageBufferTransferFunction_ITU_R_709_2, kCVImageBufferTransferFunction_ITU_R_2100_HLG,
    kCVImageBufferTransferFunction_Linear, kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ,
    kCVImageBufferTransferFunction_sRGB, kCVImageBufferTransferFunctionKey,
    kCVImageBufferYCbCrMatrix_ITU_R_601_4, kCVImageBufferYCbCrMatrix_ITU_R_709_2,
    kCVImageBufferYCbCrMatrix_ITU_R_2020, kCVImageBufferYCbCrMatrixKey, kCVPixelFormatType_32BGRA,
    kCVPixelFormatType_64RGBAHalf, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
    kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
    kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
    kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, kCVReturnSuccess,
};
use objc2_foundation::NSArray;
use objc2_io_surface::IOSurfaceRef;
use objc2_metal::{MTLSharedEvent, MTLSharedEventListener, MTLTexture};
use objc2_quartz_core::{
    CACornerMask, CALayer, CAMetalLayer, CATransaction, kCACornerCurveCircular,
    kCACornerCurveContinuous,
};
use rustc_hash::{FxHashMap, FxHashSet};

use cherenkov::{ContinuousRect, LayerId, RenderError, ShapeData, SurfaceError};

use super::{Composition, Compositor, Level, Placement, Plane, PlaneContent, SystemPlanes};
use crate::interop::{
    ChromaOffset, ExternalFrame, FramePlanes, FrameSync, Primaries, RgbAlpha, Transfer, YuvMatrix,
    YuvRange,
};
use crate::render::present::WindowSurface;

/// The backing layer of the host window's view: the system-compositor
/// parent the engine builds its planes under.
pub struct Parent {
    layer: Retained<CALayer>,
    /// The host window, kept alive for the surface's lifetime.
    window: Box<dyn wgpu::WindowHandle>,
}

// SAFETY: Core Animation layers may be modified from any thread inside an
// explicit `CATransaction`. The render thread is the only thread that
// touches the parent after capture, and only inside explicit transactions.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "the layer is used on the render thread only, inside explicit transactions"
)]
unsafe impl Send for Parent {}

impl Parent {
    /// Captures the backing layer of the view behind `handle`, making an
    /// `AppKit` view layer-backed first.
    ///
    /// # Panics
    /// Off the main thread (a view is main-thread state), when the handle is
    /// unavailable, or when it is not an `AppKit` or `UIKit` view.
    pub fn capture(handle: Box<dyn wgpu::WindowHandle>) -> Self {
        use wgpu::rwh::{HasWindowHandle as _, RawWindowHandle};
        let _main = objc2::MainThreadMarker::new().expect(
            "WindowTarget::new must run on the main thread on Apple platforms: the view's layer is main-thread state",
        );
        let raw = handle
            .window_handle()
            .expect("the window handle is available")
            .as_raw();
        let layer = match raw {
            #[cfg(target_os = "macos")]
            RawWindowHandle::AppKit(view) => {
                // SAFETY: an AppKit window handle names a live `NSView`, and
                // this is the main thread.
                let view: &objc2_app_kit::NSView = unsafe { view.ns_view.cast().as_ref() };
                view.setWantsLayer(true);
                view.layer().expect("a layer-backed view has a layer")
            }
            #[cfg(not(target_os = "macos"))]
            RawWindowHandle::UiKit(view) => {
                // SAFETY: a UIKit window handle names a live `UIView`, and
                // this is the main thread.
                let view: &objc2_ui_kit::UIView = unsafe { view.ui_view.cast().as_ref() };
                view.layer()
            }
            other => panic!("an Apple window handle is an AppKit or UIKit view, not {other:?}"),
        };
        Self {
            layer,
            window: handle,
        }
    }
}

/// Commits a `CATransaction` with implicit animations disabled when
/// dropped, on every exit path.
struct Transaction;

impl Transaction {
    fn begin() -> Self {
        CATransaction::begin();
        CATransaction::setDisableActions(true);
        Self
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        CATransaction::commit();
    }
}

/// One engine part: the metal layer and the swapchain the presenter draws
/// into.
struct PartLayer {
    layer: Retained<CAMetalLayer>,
    surface: WindowSurface,
}

/// One promoted plane's layers.
struct PlaneLayers {
    /// The promoted tree layer.
    layer: LayerId,
    /// The tree layers on the path and whether each has a clip: the shape
    /// the nested layers were built for.
    shape: Vec<(LayerId, bool)>,
    /// The pixel-space root of the plane.
    top: Retained<CALayer>,
    levels: Vec<LevelLayers>,
    /// The display layer and renderer once the main queue has created
    /// them, and what they owe the plane until then.
    link: Arc<Mutex<DisplayLink>>,
    /// The generation of the frame last handed to the display layer.
    generation: Option<u64>,
    /// The size of the frame last handed to the display layer.
    shown: Option<(u32, u32)>,
}

/// The layers mirroring one tree layer: its transform, its optional clip,
/// and its scroll offset, outermost first.
struct LevelLayers {
    node: Retained<CALayer>,
    clip: Option<Retained<CALayer>>,
    scroll: Retained<CALayer>,
}

impl LevelLayers {
    /// The layer the next level (or the display layer) nests in.
    fn inner(&self) -> &CALayer {
        &self.scroll
    }
}

/// A surface's planes on Core Animation.
pub struct LayerPlanes {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    /// Keeps the host window alive for the surface's lifetime.
    _window: Box<dyn wgpu::WindowHandle>,
    root: Retained<CALayer>,
    transparent: bool,
    /// The swapchain's required colour space (#98): every part negotiates
    /// under it.
    required: Option<wgpu::SurfaceColorSpace>,
    /// The host's display-probe channel: the first part's swapchain
    /// delivers it; every part is on the same window's display.
    probe: Option<std::sync::mpsc::Sender<crate::render::present::DisplayProbe>>,
    size: (u32, u32),
    scale: f64,
    parts: Vec<PartLayer>,
    planes: Vec<PlaneLayers>,
    /// Candidates whose display layer is being born on the main queue:
    /// a layer promotes only once its link's `display` exists, so until
    /// the queued attach runs it keeps compositing in-engine. Entries
    /// are consumed by `compose`'s `plane_layers` at promotion.
    pending: FxHashMap<LayerId, Arc<Mutex<DisplayLink>>>,
    /// Detached layer the pending displays attach to while they are
    /// born: never added to the tree, so a candidate's not-yet-promoted
    /// layer composites nothing.
    staging: Retained<CALayer>,
    /// Candidates the last `prepare` reported ready: a candidate newly
    /// ready since then sets `woke`, which `wants_plan` answers once so
    /// the surface re-plans and promotes it. A candidate the plan then
    /// rejects stays in `offered` — it does not re-wake every frame.
    offered: FxHashSet<LayerId>,
    woke: bool,
    /// The host wake-up an attach completion fires to pull the frame
    /// that promotes the born candidate — the same `waker.wake` an
    /// external frame or an invalidation uses to wake the render loop.
    waker: Option<cherenkov::MainWaker>,
    /// Set when parts or planes were added or removed, so the root's
    /// sublayer order is rebuilt.
    restack: bool,
}

/// An origin-anchored layer: its position is its superlayer point for its
/// bounds origin, so a nested chain composes plain affine maps.
fn anchored() -> Retained<CALayer> {
    let layer = CALayer::new();
    layer.setAnchorPoint(CGPoint::new(0.0, 0.0));
    layer.setPosition(CGPoint::new(0.0, 0.0));
    layer.setBounds(CGRect::new(CGPoint::new(0.0, 0.0), CGSize::new(0.0, 0.0)));
    layer
}

/// What the display layer's deferred creation owes its plane.
///
/// An `AVSampleBufferDisplayLayer` binds its video queue on the thread
/// it is created on; created off the main thread that binding can
/// complete only after the first `enqueueSampleBuffer`, which the queue
/// then drops without an error. Plane construction runs on the render
/// thread, so the layer is born on the main queue, where the binding is
/// up before the first enqueue — asynchronously, since the render
/// thread may not wait on the UI thread's run loop. Until the link is
/// up the layer's geometry and newest frame are held here and applied
/// at attach; the renderer afterwards serves `show()` directly. `dead`
/// retires a queued attach when the layer is demoted or the surface
/// torn down before the block ran.
#[derive(Default)]
struct DisplayLink {
    display: Option<Retained<AVSampleBufferDisplayLayer>>,
    renderer: Option<Retained<AVSampleBufferVideoRenderer>>,
    bounds: Option<CGRect>,
    opacity: Option<f32>,
    sample: Option<CFRetained<CMSampleBuffer>>,
    dead: bool,
}

// SAFETY: the objects are touched only under the mutex: the render
// thread fills the geometry and sample slots, the main queue fills the
// layer and renderer and drains the sample.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "every field is touched only while the mutex is held"
)]
unsafe impl Send for DisplayLink {}

/// Creates a plane's display layer on the main queue and attaches it,
/// draining the geometry and frame the render thread left in `link`.
struct DisplayAttach {
    parent: Retained<CALayer>,
    link: Arc<Mutex<DisplayLink>>,
    waker: Option<cherenkov::MainWaker>,
}

// SAFETY: the parent is touched only on the main queue, inside an
// explicit transaction, where layer mutation is safe; the link is
// mutex-guarded.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "the layer is touched only on the main queue"
)]
unsafe impl Send for DisplayAttach {}

impl DisplayAttach {
    /// Runs `attach` on the main queue. A display layer created off the
    /// main thread silently drops its first enqueue — measured on
    /// paravirtual Metal, 1/30 frames landed when the layer was created
    /// on a background thread vs 30/30 when created on main — so the
    /// render thread queues the create+attach rather than waiting on
    /// the UI thread's run loop, which it must never block on.
    fn run(self) {
        if objc2::MainThreadMarker::new().is_some() {
            self.attach();
        } else {
            dispatch2::DispatchQueue::main().exec_async(move || self.attach());
        }
    }

    fn attach(self) {
        let mut link = self.link.lock().expect("link poisoned");
        if link.dead {
            // The layer was demoted or the surface torn down while this
            // block was queued: change nothing.
            return;
        }
        // SAFETY: a new display layer, owned by the plane's tree.
        let display = unsafe { AVSampleBufferDisplayLayer::new() };
        display.setAnchorPoint(CGPoint::new(0.0, 0.0));
        display.setPosition(CGPoint::new(0.0, 0.0));
        unsafe {
            display.setVideoGravity(AVLayerVideoGravityResize.expect("AVLayerVideoGravityResize"));
            // Display sleep is the player's policy, not the compositor's.
            display.setPreventsDisplaySleepDuringVideoPlayback(false);
        }
        {
            let _tx = Transaction::begin();
            if let Some(bounds) = link.bounds {
                display.setBounds(bounds);
            }
            if let Some(opacity) = link.opacity {
                display.setOpacity(opacity);
            }
            self.parent.addSublayer(&display);
        }
        // The commit above attached the layer; a display layer lays out
        // its video sublayer correctly only on the main thread (the
        // `MainLayout` race), and this block is already there.
        display.setNeedsLayout();
        display.layoutIfNeeded();
        // SAFETY: the display layer's own renderer.
        let renderer = unsafe { display.sampleBufferRenderer() };
        if let Some(sample) = link.sample.take() {
            // SAFETY: a display layer's renderer is fed from an
            // arbitrary queue (its `requestMediaDataWhenReady`
            // contract).
            unsafe { renderer.enqueueSampleBuffer(&sample) };
        }
        link.display = Some(display);
        link.renderer = Some(renderer);
        drop(link);
        // The attach is the event that makes this candidate promotable:
        // an external frame or an invalidation wakes the host for the
        // frame that shows it, and this does the same — the next render
        // promotes the now-born candidate instead of compositing it
        // in-engine forever on a static scene.
        if let Some(waker) = &self.waker {
            waker.wake();
        }
    }
}

const fn cg_affine(t: Affine) -> CGAffineTransform {
    let [xx, yx, xy, yy, x0, y0] = t.as_coeffs();
    CGAffineTransform {
        a: xx,
        b: yx,
        c: xy,
        d: yy,
        tx: x0,
        ty: y0,
    }
}

const fn cg_rect(r: Rect) -> CGRect {
    CGRect::new(CGPoint::new(r.x0, r.y0), CGSize::new(r.width(), r.height()))
}

/// A clip a `CALayer` expresses: its rect, one corner radius on the masked
/// corners, and the corner curve.
#[derive(Debug, PartialEq)]
pub struct LayerClip {
    /// The clip rectangle.
    pub rect: Rect,
    /// The radius of every rounded corner.
    pub radius: f64,
    /// The rounded corners.
    pub corners: CACornerMask,
    /// Whether the corners are continuous rather than circular.
    pub continuous: bool,
}

impl LayerClip {
    /// The layer clip equal to `clip`, when one exists: a rectangle, or a
    /// rectangle whose corners are each square or share one radius no larger
    /// than half its shorter side, with circular corners or continuous
    /// corners at the system smoothing
    /// ([`ContinuousRect::DEFAULT_SMOOTHING`]).
    #[must_use]
    #[expect(
        clippy::float_cmp,
        reason = "a layer carries one radius, so corners share it exactly or are exactly square"
    )]
    pub fn of(clip: &ShapeData) -> Option<Self> {
        let rounded = |rect: Rect, radii: [f64; 4], continuous: bool| {
            let radius = radii.into_iter().fold(0.0_f64, f64::max);
            if radii.iter().any(|&r| r != 0.0 && r != radius)
                || radius > rect.width().min(rect.height()) / 2.0
            {
                return None;
            }
            // In the engine's y-down space the minimum y edge is the top.
            let corners = [
                CACornerMask::LayerMinXMinYCorner,
                CACornerMask::LayerMaxXMinYCorner,
                CACornerMask::LayerMaxXMaxYCorner,
                CACornerMask::LayerMinXMaxYCorner,
            ]
            .into_iter()
            .zip(radii)
            .filter(|(_, r)| *r != 0.0)
            .fold(CACornerMask::empty(), |mask, (corner, _)| mask | corner);
            Some(Self {
                rect,
                radius,
                corners,
                continuous,
            })
        };
        match clip {
            ShapeData::Rect(rect) => Some(Self {
                rect: *rect,
                radius: 0.0,
                corners: CACornerMask::empty(),
                continuous: false,
            }),
            ShapeData::RoundedRect(r) => {
                let radii = r.radii();
                rounded(
                    r.rect(),
                    [
                        radii.top_left,
                        radii.top_right,
                        radii.bottom_right,
                        radii.bottom_left,
                    ],
                    false,
                )
            }
            ShapeData::Continuous(ContinuousRect {
                rect,
                radii,
                smoothing,
            }) => {
                let continuous = if *smoothing == 0.0 {
                    false
                } else if *smoothing == ContinuousRect::DEFAULT_SMOOTHING {
                    true
                } else {
                    return None;
                };
                rounded(
                    *rect,
                    [
                        radii.top_left,
                        radii.top_right,
                        radii.bottom_right,
                        radii.bottom_left,
                    ],
                    continuous,
                )
            }
            ShapeData::Circle(c) => {
                let rect = Rect::from_center_size(c.center, (2.0 * c.radius, 2.0 * c.radius));
                rounded(rect, [c.radius; 4], false)
            }
            ShapeData::Ellipse(e) => {
                // A round ellipse is a circle whatever its rotation; its radii
                // come out of a decomposition, equal up to rounding.
                let radii = e.radii();
                ((radii.x - radii.y).abs() <= 1e-9 * radii.x.max(radii.y)).then_some(())?;
                let radius = f64::midpoint(radii.x, radii.y);
                let rect = Rect::from_center_size(e.center(), (2.0 * radius, 2.0 * radius));
                rounded(rect, [radius; 4], false)
            }
            ShapeData::Line(_) | ShapeData::Path { .. } => None,
        }
    }

    fn apply(&self, layer: &CALayer) {
        layer.setBounds(cg_rect(self.rect));
        layer.setPosition(CGPoint::new(self.rect.x0, self.rect.y0));
        layer.setMasksToBounds(true);
        layer.setCornerRadius(self.radius);
        layer.setMaskedCorners(self.corners);
        // SAFETY: the corner-curve constants are immutable statics.
        let curve = unsafe {
            if self.continuous {
                kCACornerCurveContinuous
            } else {
                kCACornerCurveCircular
            }
        };
        layer.setCornerCurve(curve);
    }
}

/// An external frame's `IOSurface` and pixel format, when its planes are
/// the planes of one `IOSurface` in the layout the frame declares.
fn frame_surface(frame: &ExternalFrame) -> Option<Retained<IOSurfaceRef>> {
    /// The `MTLTexture` behind a plane and the `IOSurface` plane it maps.
    fn plane(texture: &wgpu::Texture) -> Option<(Retained<IOSurfaceRef>, usize)> {
        // SAFETY: the texture lives on a Metal device (planes are imported
        // with `interop::metal::import_texture`); the guard is dropped before
        // the texture.
        let hal = unsafe { texture.as_hal::<wgpu::hal::metal::Api>() }?;
        let raw = hal.raw_handle();
        Some((raw.iosurface()?, raw.iosurfacePlane()))
    }
    let (surface, format) = match &frame.planes {
        FramePlanes::Yuv { y, uv } => {
            let (surface, 0) = plane(y)? else {
                return None;
            };
            let (chroma, 1) = plane(uv)? else {
                return None;
            };
            if Retained::as_ptr(&surface) != Retained::as_ptr(&chroma) {
                return None;
            }
            let ten = y.format() == wgpu::TextureFormat::R16Uint;
            let format = match (ten, frame.color.range) {
                (false, YuvRange::Video) => kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                (false, YuvRange::Full) => kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                (true, YuvRange::Video) => kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
                (true, YuvRange::Full) => kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
            };
            (surface, format)
        }
        FramePlanes::Rgb {
            plane: texture,
            alpha,
        } => {
            // A display layer shows opaque video; translucent frames stay in
            // the engine, which composites their alpha.
            if *alpha != RgbAlpha::Opaque {
                return None;
            }
            let (surface, 0) = plane(texture)? else {
                return None;
            };
            let format = match texture.format() {
                wgpu::TextureFormat::Bgra8Unorm => kCVPixelFormatType_32BGRA,
                wgpu::TextureFormat::Rgba16Float => kCVPixelFormatType_64RGBAHalf,
                _ => return None,
            };
            (surface, format)
        }
    };
    (surface.pixel_format() == format).then_some(surface)
}

/// The `CVImageBuffer` colour attachments equal to a frame's declared
/// colour, as `(key, value)` pairs.
fn color_attachments(frame: &ExternalFrame) -> Vec<(&'static CFString, &'static CFString)> {
    let color = &frame.color;
    // SAFETY: CoreVideo's attachment keys and values are immutable statics.
    unsafe {
        let primaries = match color.primaries {
            Primaries::Bt709 => kCVImageBufferColorPrimaries_ITU_R_709_2,
            Primaries::DisplayP3 => kCVImageBufferColorPrimaries_P3_D65,
            Primaries::Bt2020 => kCVImageBufferColorPrimaries_ITU_R_2020,
        };
        let transfer = match color.transfer {
            Transfer::Linear => kCVImageBufferTransferFunction_Linear,
            Transfer::Srgb => kCVImageBufferTransferFunction_sRGB,
            Transfer::Bt709 => kCVImageBufferTransferFunction_ITU_R_709_2,
            Transfer::Pq => kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ,
            Transfer::Hlg => kCVImageBufferTransferFunction_ITU_R_2100_HLG,
        };
        let mut pairs = vec![
            (kCVImageBufferColorPrimariesKey, primaries),
            (kCVImageBufferTransferFunctionKey, transfer),
        ];
        if matches!(frame.planes, FramePlanes::Yuv { .. }) {
            let matrix = match color.matrix {
                YuvMatrix::Bt601 => kCVImageBufferYCbCrMatrix_ITU_R_601_4,
                YuvMatrix::Bt709 => kCVImageBufferYCbCrMatrix_ITU_R_709_2,
                YuvMatrix::Bt2020 => kCVImageBufferYCbCrMatrix_ITU_R_2020,
            };
            let siting = match (color.chroma_siting.x, color.chroma_siting.y) {
                (ChromaOffset::Cosited, ChromaOffset::Centered) => {
                    kCVImageBufferChromaLocation_Left
                }
                (ChromaOffset::Centered, ChromaOffset::Centered) => {
                    kCVImageBufferChromaLocation_Center
                }
                (ChromaOffset::Cosited, ChromaOffset::Cosited) => {
                    kCVImageBufferChromaLocation_TopLeft
                }
                (ChromaOffset::Centered, ChromaOffset::Cosited) => kCVImageBufferChromaLocation_Top,
            };
            pairs.extend([
                (kCVImageBufferYCbCrMatrixKey, matrix),
                (kCVImageBufferChromaLocationTopFieldKey, siting),
                (kCVImageBufferChromaLocationBottomFieldKey, siting),
            ]);
        }
        pairs
    }
}

/// A pixel buffer over the frame's `IOSurface`, carrying the frame's
/// colour as attachments. No pixel is copied.
///
/// # Errors
/// When the frame's planes are not an `IOSurface` ([`LayerPlanes::shows`]
/// admits only frames whose planes are) or `CoreVideo` refuses the wrap.
fn pixel_buffer(frame: &ExternalFrame) -> Result<CFRetained<CVPixelBuffer>, RenderError> {
    let surface = frame_surface(frame).ok_or_else(|| {
        RenderError::Render("a promoted external frame's planes are not one IOSurface".into())
    })?;
    let mut out = std::ptr::null_mut();
    let status =
        unsafe { CVPixelBufferCreateWithIOSurface(None, &surface, None, NonNull::from(&mut out)) };
    let buffer = NonNull::new(out)
        .filter(|_| status == kCVReturnSuccess)
        .ok_or_else(|| {
            RenderError::Render(format!(
                "CoreVideo refused to wrap a promoted frame's IOSurface (CVReturn {status})"
            ))
        })?;
    // SAFETY: the create call returned a +1 pixel buffer.
    let buffer = unsafe { CFRetained::from_raw(buffer) };
    for (key, value) in color_attachments(frame) {
        // SAFETY: every colour attachment's value is a CFString.
        unsafe { buffer.set_attachment(key, value, CVAttachmentMode::ShouldPropagate) };
    }
    Ok(buffer)
}

/// A sample buffer the display layer shows as soon as it is enqueued.
fn sample_buffer(buffer: &CVPixelBuffer) -> Result<CFRetained<CMSampleBuffer>, RenderError> {
    let mut format = std::ptr::null();
    let status = unsafe {
        CMVideoFormatDescriptionCreateForImageBuffer(None, buffer, NonNull::from(&mut format))
    };
    let format = NonNull::new(format.cast_mut())
        .filter(|_| status == 0)
        .ok_or_else(|| {
            RenderError::Render(format!(
                "CoreMedia refused a promoted frame's format description (OSStatus {status})"
            ))
        })?;
    // SAFETY: the create call returned a +1 format description.
    let format: CFRetained<CMVideoFormatDescription> = unsafe { CFRetained::from_raw(format) };
    // SAFETY: `kCMTimeInvalid` is an immutable static.
    let invalid = unsafe { kCMTimeInvalid };
    let mut timing = CMSampleTimingInfo {
        duration: invalid,
        presentationTimeStamp: invalid,
        decodeTimeStamp: invalid,
    };
    let mut out = std::ptr::null_mut();
    let status = unsafe {
        CMSampleBuffer::create_ready_with_image_buffer(
            None,
            buffer,
            &format,
            NonNull::from(&mut timing),
            NonNull::from(&mut out),
        )
    };
    let sample = NonNull::new(out).filter(|_| status == 0).ok_or_else(|| {
        RenderError::Render(format!(
            "CoreMedia refused a promoted frame's sample buffer (OSStatus {status})"
        ))
    })?;
    // SAFETY: the create call returned a +1 sample buffer.
    let sample = unsafe { CFRetained::from_raw(sample) };
    let attachments = unsafe { sample.sample_attachments_array(true) }.ok_or_else(|| {
        RenderError::Render("a promoted frame's sample buffer has no attachments".into())
    })?;
    // SAFETY: a one-sample buffer's attachment array holds one mutable
    // dictionary; the key and value are CoreMedia and CoreFoundation statics.
    unsafe {
        let dictionary: &CFMutableDictionary =
            &*attachments.value_at_index(0).cast::<CFMutableDictionary>();
        CFMutableDictionary::set_value(
            Some(dictionary),
            std::ptr::from_ref(kCMSampleAttachmentKey_DisplayImmediately).cast(),
            std::ptr::from_ref::<CFBoolean>(
                objc2_core_foundation::kCFBooleanTrue.expect("kCFBooleanTrue"),
            )
            .cast(),
        );
    }
    Ok(sample)
}

impl LayerPlanes {
    /// Builds the surface's root layer under `parent` and its first part.
    ///
    /// # Errors
    /// [`SurfaceError::UnsupportedTarget`] when the adapter cannot present
    /// to a metal layer.
    #[expect(
        clippy::too_many_arguments,
        reason = "the window's full negotiation input: device triple, parent,                   size, transparency, required space and the probe channel"
    )]
    pub fn new(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        parent: Parent,
        size: (u32, u32),
        transparent: bool,
        required: Option<wgpu::SurfaceColorSpace>,
        probe: Option<std::sync::mpsc::Sender<crate::render::present::DisplayProbe>>,
        waker: Option<cherenkov::MainWaker>,
    ) -> Result<Self, SurfaceError> {
        let _tx = Transaction::begin();
        let Parent {
            layer: host,
            window,
        } = parent;
        let root = anchored();
        // SAFETY: the name is an immutable literal.
        root.setName(Some(&objc2_foundation::NSString::from_str("cherenkov")));
        // Engine geometry is y-down. AppKit layers are y-up unless the view
        // is flipped; UIKit layers are y-down.
        #[cfg(target_os = "macos")]
        root.setGeometryFlipped(!host.contentsAreFlipped());
        host.addSublayer(&root);
        let scale = host.contentsScale();
        let mut planes = Self {
            instance: instance.clone(),
            adapter: adapter.clone(),
            device: device.clone(),
            _window: window,
            root,
            transparent,
            required,
            probe,
            size,
            scale,
            parts: Vec::new(),
            planes: Vec::new(),
            pending: FxHashMap::default(),
            staging: CALayer::new(),
            offered: FxHashSet::default(),
            woke: false,
            waker,
            restack: true,
        };
        planes.geometry();
        planes.push_part()?;
        planes.stack();
        Ok(planes)
    }

    /// Sizes the root to the surface in points.
    fn geometry(&self) {
        let points = CGRect::new(
            CGPoint::new(0.0, 0.0),
            CGSize::new(
                f64::from(self.size.0) / self.scale,
                f64::from(self.size.1) / self.scale,
            ),
        );
        self.root.setBounds(points);
        for part in &self.parts {
            part.layer.setFrame(points);
            part.layer.setContentsScale(self.scale);
        }
        for plane in &self.planes {
            plane
                .top
                .setAffineTransform(cg_affine(Affine::scale(1.0 / self.scale)));
        }
    }

    fn push_part(&mut self) -> Result<(), SurfaceError> {
        let layer = CAMetalLayer::new();
        layer.setPresentsWithTransaction(true);
        layer.setAnchorPoint(CGPoint::new(0.0, 0.0));
        layer.setFrame(self.root.bounds());
        layer.setContentsScale(self.scale);
        // Parts above the first always show what lies beneath them.
        let transparent = self.transparent || !self.parts.is_empty();
        let surface = WindowSurface::from_layer(
            &self.instance,
            &self.adapter,
            &self.device,
            &layer,
            self.size,
            transparent,
            self.required,
            self.probe.take(),
        )?;
        self.parts.push(PartLayer { layer, surface });
        self.restack = true;
        Ok(())
    }

    /// Orders the root's sublayers: part 0, plane 0, part 1, …
    fn stack(&mut self) {
        if !self.restack {
            return;
        }
        let mut order: Vec<&CALayer> = Vec::new();
        for (i, part) in self.parts.iter().enumerate() {
            order.push(&part.layer);
            if let Some(plane) = self.planes.get(i) {
                order.push(&plane.top);
            }
        }
        for plane in self.planes.iter().skip(self.parts.len()) {
            order.push(&plane.top);
        }
        let array = NSArray::from_slice(&order);
        // SAFETY: every element is a `CALayer` this tree owns.
        unsafe { self.root.setSublayers(Some(&array)) };
        self.restack = false;
    }

    /// Builds the layers for `placement`.
    /// `layer` can show a frame now: a committed plane already shows it,
    /// or its pending link's display exists — the queued main-queue
    /// attach has run.
    fn born(&self, layer: LayerId) -> bool {
        self.planes.iter().any(|plane| plane.layer == layer)
            || self
                .pending
                .get(&layer)
                .is_some_and(|link| link.lock().expect("link poisoned").display.is_some())
    }

    /// Builds the plane's level chain for `placement` and reparents the
    /// display layer `link` carries — born and staged under `root` by
    /// [`SystemPlanes::prepare`], or taken over from the plane it
    /// replaces — under the chain's inner level inside this transaction,
    /// which is safe from any thread.
    fn plane_layers(&self, placement: &Placement, link: Arc<Mutex<DisplayLink>>) -> PlaneLayers {
        let top = anchored();
        top.setAffineTransform(cg_affine(Affine::scale(1.0 / self.scale)));
        let mut outer: Retained<CALayer> = top.clone();
        let mut levels = Vec::with_capacity(placement.path.len());
        for level in &placement.path {
            let node = anchored();
            outer.addSublayer(&node);
            let clip = level.clip.as_ref().map(|_| {
                let clip = anchored();
                node.addSublayer(&clip);
                clip
            });
            let scroll = anchored();
            clip.as_deref().unwrap_or(&node).addSublayer(&scroll);
            outer = scroll.clone();
            levels.push(LevelLayers { node, clip, scroll });
        }
        if let Some(display) = &link.lock().expect("link poisoned").display {
            levels
                .last()
                .map_or(&*top, LevelLayers::inner)
                .addSublayer(display);
        }
        PlaneLayers {
            layer: placement.layer,
            shape: shape(placement),
            top,
            levels,
            link,
            generation: None,
            shown: None,
        }
    }

    /// Hands `frame` to `plane`'s display layer: at once, or when the
    /// producer's GPU signals the frame's event, never waiting on the CPU.
    /// `resized` when its size differs from the frame shown before, the first
    /// frame included: the display layer then needs its main-thread layout
    /// (`MainLayout`).
    ///
    /// Called after the composition's transaction has committed: each enqueue
    /// commits its own transaction, and the layout is queued only after it.
    fn show(plane: &PlaneLayers, frame: &ExternalFrame, resized: bool) -> Result<(), RenderError> {
        let buffer = pixel_buffer(frame)?;
        let sample = sample_buffer(&buffer)?;
        let mut link = plane.link.lock().expect("link poisoned");
        let Some(renderer) = link.renderer.clone() else {
            // The layer is still being born on the main queue; the
            // attach hands it the latest frame.
            link.sample = Some(sample);
            return Ok(());
        };
        drop(link);
        let layout = if resized {
            plane
                .link
                .lock()
                .expect("link poisoned")
                .display
                .clone()
                .map(MainLayout)
        } else {
            None
        };
        let enqueue_renderer = renderer.clone();
        let enqueue = move || {
            {
                let _tx = Transaction::begin();
                // SAFETY: a display layer's renderer is fed from an arbitrary
                // queue (its `requestMediaDataWhenReady` contract); the
                // sample buffer stays retained here.
                unsafe { enqueue_renderer.enqueueSampleBuffer(&sample) };
            }
            if let Some(layout) = layout {
                layout.queue();
            }
        };
        match &frame.wait {
            None => enqueue(),
            Some(FrameSync::Metal { event, value }) => {
                let enqueue = std::cell::Cell::new(Some(enqueue));
                let block = block2::RcBlock::new(
                    move |_: NonNull<ProtocolObject<dyn MTLSharedEvent>>, _: u64| {
                        if let Some(enqueue) = enqueue.take() {
                            enqueue();
                        }
                    },
                );
                unsafe {
                    event.notifyListener_atValue_block(
                        &MTLSharedEventListener::sharedListener(),
                        *value,
                        block2::RcBlock::as_ptr(&block),
                    );
                }
            }
        }
        // SAFETY: reading the renderer's status.
        if unsafe { renderer.status() } == AVQueuedSampleBufferRenderingStatus::Failed {
            let reason = unsafe { renderer.error() }.map_or_else(
                || "no error reported".to_owned(),
                |e| e.localizedDescription().to_string(),
            );
            return Err(RenderError::Render(format!(
                "the system compositor rejected layer {:?}'s plane: {reason}",
                plane.layer
            )));
        }
        Ok(())
    }
}

/// A display layer whose layout runs on the main queue.
///
/// A display layer derives the transform that fits its video into its bounds
/// from the size of the frames it is fed, in a layout pass that is correct
/// only on the main thread. Fed from the render thread, it queues a layout
/// of its own on the main queue when the transaction holding the change
/// commits, and that one keeps the transform computed before the first
/// frame, offsetting the video by half its size. A layout queued after that
/// commit runs after it and recomputes the transform; until it runs (one main
/// run-loop turn) the frame shows at the stale transform.
struct MainLayout(Retained<AVSampleBufferDisplayLayer>);

// SAFETY: the layer is only retained and released off the main thread; it is
// laid out on the main queue, where layer layout belongs.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "the display layer is touched only on the main queue"
)]
unsafe impl Send for MainLayout {}

impl MainLayout {
    fn queue(self) {
        // A method call moves the whole wrapper into the closure; a field
        // pattern would capture only the layer.
        dispatch2::DispatchQueue::main().exec_async(move || self.lay_out());
    }

    fn lay_out(self) {
        self.0.setNeedsLayout();
        self.0.layoutIfNeeded();
    }
}

/// The nested-layer shape a placement needs.
fn shape(placement: &Placement) -> Vec<(LayerId, bool)> {
    placement
        .path
        .iter()
        .map(|level| (level.layer, level.clip.is_some()))
        .collect()
}

/// Applies a level's sampled properties to its layers.
fn place(level: &Level, layers: &LevelLayers) {
    layers.node.setAffineTransform(cg_affine(level.transform));
    if let (Some(clip), Some(layer)) = (&level.clip, &layers.clip) {
        LayerClip::of(clip)
            .expect("eligibility admits only clips a layer expresses")
            .apply(layer);
    }
    let Vec2 { x, y } = level.scroll;
    layers
        .scroll
        .setBounds(CGRect::new(CGPoint::new(x, y), CGSize::new(0.0, 0.0)));
}

impl Compositor for LayerPlanes {
    /// Two planes: a main video and one picture-in-picture. Each plane adds
    /// a full-surface engine part and swapchain above it, and planes beyond
    /// what the display pipes scan out are composited by the system on the
    /// GPU anyway, which removes the energy win the promotion is for.
    const BUDGET: usize = 2;

    fn expresses_transform(transform: Affine) -> bool {
        // A layer carries any affine matrix; only a finite one is valid.
        transform.as_coeffs().iter().all(|c| c.is_finite())
    }

    fn expresses_clip(clip: &ShapeData) -> bool {
        LayerClip::of(clip).is_some()
    }

    fn shows(frame: &ExternalFrame) -> bool {
        // The platform must decode the frame the way the engine would:
        // an `ITU_R_709_2` frame is decoded with the inverse OETF while
        // the engine applies BT.1886 γ2.4, and no colour tag reproduces
        // γ2.4 (a `GammaLevel` tag is ignored by the decode path —
        // measured levels 1.0/2.2/2.4 producing identical output), so a
        // promoted BT.709 plane would land darker than engine output.
        if frame.color.transfer == Transfer::Bt709 {
            return false;
        }
        frame_surface(frame).is_some()
    }
}

impl SystemPlanes for LayerPlanes {
    #[expect(
        clippy::float_cmp,
        reason = "any change of the display scale re-lays out the tree"
    )]
    fn compose(&mut self, c: Composition<'_>) -> Result<bool, RenderError> {
        // Frames are shown after this commits (`show`).
        let tx = Transaction::begin();
        if c.display.scale != self.scale || c.size != self.size {
            self.scale = c.display.scale;
            self.size = c.size;
            self.geometry();
        }
        while self.parts.len() < c.parts.len() {
            self.push_part()
                .map_err(|e| RenderError::Render(format!("an engine part's metal layer: {e}")))?;
        }
        if self.parts.len() > c.parts.len() {
            for part in self.parts.drain(c.parts.len()..) {
                part.layer.removeFromSuperlayer();
            }
            self.restack = true;
        }
        for (i, plane) in c.planes.iter().enumerate() {
            let fits = self.planes.get(i).is_some_and(|built| {
                built.layer == plane.placement.layer && built.shape == shape(plane.placement)
            });
            if !fits {
                // A re-placed plane keeps its born display layer; a new
                // one takes over the link `prepare` staged for its
                // candidate — its display is attached by definition,
                // since `prepare` admits only born layers.
                let link = self
                    .planes
                    .get(i)
                    .filter(|old| old.layer == plane.placement.layer)
                    .map(|old| Arc::clone(&old.link))
                    .or_else(|| self.pending.remove(&plane.placement.layer))
                    .expect("a promoted candidate's display layer is born");
                let built = self.plane_layers(plane.placement, link);
                if i < self.planes.len() {
                    let old = std::mem::replace(&mut self.planes[i], built);
                    old.top.removeFromSuperlayer();
                } else {
                    self.planes.push(built);
                }
                self.restack = true;
            }
        }
        if self.planes.len() > c.planes.len() {
            for plane in self.planes.drain(c.planes.len()..) {
                plane.top.removeFromSuperlayer();
            }
            self.restack = true;
        }
        self.stack();
        for (plane, built) in c.planes.iter().zip(&mut self.planes) {
            for (level, layers) in plane.placement.path.iter().zip(&built.levels) {
                place(level, layers);
            }
            let (w, h) = plane.placement.size;
            let bounds = CGRect::new(
                CGPoint::new(0.0, 0.0),
                CGSize::new(f64::from(w), f64::from(h)),
            );
            let mut link = built.link.lock().expect("link poisoned");
            link.bounds = Some(bounds);
            link.opacity = Some(plane.placement.opacity);
            if let Some(display) = &link.display {
                display.setBounds(bounds);
                display.setOpacity(plane.placement.opacity);
            }
            drop(link);
        }
        let mut presented = true;
        for (part, target) in c.parts.iter().zip(&self.parts) {
            presented &= c.presenter.present(
                c.device,
                c.queue,
                &target.surface,
                part.view,
                c.display.headroom,
            )?;
        }
        drop(tx);
        for (plane, built) in c.planes.iter().zip(&mut self.planes) {
            match plane.content {
                PlaneContent::Frame { frame, generation } => {
                    if built.generation != Some(generation) {
                        let resized = built.shown != Some(plane.placement.size);
                        Self::show(built, frame, resized)?;
                        built.generation = Some(generation);
                        built.shown = Some(plane.placement.size);
                    }
                }
            }
        }
        Ok(presented)
    }

    fn groom(&mut self, candidates: &FxHashMap<LayerId, (u32, u32)>) {
        // A candidate that left the eligible set before its attach ran is
        // dead: the queued block sees `dead` and changes nothing.
        self.pending.retain(|layer, link| {
            let keep = candidates.contains_key(layer);
            if !keep {
                link.lock().expect("link poisoned").dead = true;
            }
            keep
        });
        for &layer in candidates.keys() {
            if self.born(layer) || self.pending.contains_key(&layer) {
                continue;
            }
            let link = Arc::new(Mutex::new(DisplayLink::default()));
            DisplayAttach {
                parent: self.staging.clone(),
                link: Arc::clone(&link),
                waker: self.waker.clone(),
            }
            .run();
            self.pending.insert(layer, link);
        }
        self.offered.retain(|layer| candidates.contains_key(layer));
        self.woke = false;
        for &layer in candidates.keys() {
            if self.born(layer) {
                if self.offered.insert(layer) {
                    self.woke = true;
                }
            } else {
                self.offered.remove(&layer);
            }
        }
    }

    fn wants_plan(&self) -> bool {
        self.woke
    }

    fn prepare(
        &mut self,
        candidates: &FxHashMap<LayerId, (u32, u32)>,
        ready: &mut FxHashSet<LayerId>,
    ) {
        self.groom(candidates);
        ready.clear();
        ready.extend(candidates.keys().copied().filter(|layer| self.born(*layer)));
    }

    /// Hands each promoted layer's new frame to its display layer: the
    /// layer tree and geometry are the ones `compose` built for the
    /// committed plan, and the parts' layers keep their shown buffers —
    /// nothing here presents or reconfigures them (#90).
    ///
    /// # Errors
    /// A [`RenderError`] naming the cause when the system rejects a plane
    /// or a refresh names a layer no plane shows.
    fn refresh<'a>(&mut self, frames: impl Iterator<Item = Plane<'a>>) -> Result<(), RenderError> {
        for update in frames {
            match update.content {
                PlaneContent::Frame { frame, generation } => {
                    let Some(built) = self
                        .planes
                        .iter_mut()
                        .find(|plane| plane.layer == update.placement.layer)
                    else {
                        return Err(RenderError::Render(format!(
                            "layer {:?}'s frame changed while no plane shows it",
                            update.placement.layer
                        )));
                    };
                    if built.generation != Some(generation) {
                        let resized = built.shown != Some(update.placement.size);
                        Self::show(built, frame, resized)?;
                        built.generation = Some(generation);
                        built.shown = Some(update.placement.size);
                    }
                }
            }
        }
        Ok(())
    }

    fn resize(&mut self, size: (u32, u32)) {
        // Reconfiguring a part changes its layer: the render thread has no
        // run loop to commit an implicit transaction.
        let _tx = Transaction::begin();
        for part in &mut self.parts {
            part.surface.resize(&self.device, size);
        }
    }

    fn reselect(&mut self, adapter: &wgpu::Adapter, device: &wgpu::Device) {
        let _tx = Transaction::begin();
        for part in &mut self.parts {
            part.surface.reselect(adapter, device);
        }
    }
}

impl Drop for LayerPlanes {
    fn drop(&mut self) {
        // A queued attach that has not run sees `dead` and changes
        // nothing beyond releasing its link.
        for (_, link) in self.pending.drain() {
            link.lock().expect("link poisoned").dead = true;
        }
        let _tx = Transaction::begin();
        self.root.removeFromSuperlayer();
    }
}

#[cfg(test)]
mod tests;
