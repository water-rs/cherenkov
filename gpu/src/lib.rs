// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cherenkov-gpu`: the wgpu backend for the Cherenkov 2D rendering
//! engine.
//!
//! The shared front end lives in the [`cherenkov`] crate: [`Engine`],
//! [`Surface`], [`Layer`], the layer tree and the render thread's loop are
//! all generic over [`Backend`]. This crate supplies the render side only —
//! [`Gpu`]'s [`Backend`] implementation drives the wgpu device on the
//! render thread.
//!
//! ```no_run
//! use cherenkov::{Draw, Engine, Offscreen, OffscreenFormat, WorkingColor};
//! use cherenkov::kurbo::Rect;
//! use cherenkov_gpu::{Gpu, GpuConfig};
//!
//! let engine = Engine::<Gpu>::new(GpuConfig::default())?;
//! let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
//! surface.update(|tx| {
//!     tx[surface.root()].content(
//!         surface.record(|c| c.fill(Rect::new(0., 0., 64., 64.), WorkingColor::WHITE)),
//!     );
//! });
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod interop;
mod names;
mod render;

/// The allocation-event diagnostic sink (issue #169).
pub use render::diag;
/// The registered-effect module text the renderer compiles; exposed for
/// `tests/shader.rs`, which translates it through every naga backend.
#[doc(hidden)]
pub use render::shaders::backdrop_effect_text;

use std::path::PathBuf;

use cherenkov::{Backend, EngineError, Offscreen, Rgba8, Rgba16F, Uploads};

/// Adapter information for provenance.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    /// Adapter name.
    pub name: String,
    /// Backend (e.g. `Vulkan`).
    pub backend: String,
    /// PCI vendor id.
    pub vendor: u32,
    /// PCI device id.
    pub device: u32,
    /// Device class (e.g. `IntegratedGpu`).
    pub device_type: String,
    /// Driver name.
    pub driver: String,
    /// Driver version detail.
    pub driver_info: String,
    /// Supported GPU timestamp positions.
    pub timestamps: TimestampSupport,
}

/// Where the adapter lets the renderer sample GPU timestamps.
///
/// The renderer samples only at pass boundaries, the one position every
/// adapter with timestamp queries honours. Encoder-level sampling is not
/// a level here: Metal on Apple GPUs advertises it but samples only at
/// stage boundaries, through a dummy blit encoder wgpu documents as
/// unreliable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimestampSupport {
    /// No timestamp queries; `finish_timings` returns no GPU timings.
    Unsupported,
    /// At render and compute pass boundaries.
    PassBoundaries,
}

/// The texture format used for intermediate (isolation) render targets.
///
/// The surface itself stays `Rgba16Float` regardless; this only picks the
/// precision of offscreen layers the compositor reads back in the same
/// frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScratchFormat {
    /// `Rgba16Float`: linear, no banding, 8 bytes per texel.
    #[default]
    LinearF16,
    /// `Rgba8Unorm`: half the bandwidth, 8-bit precision; intermediate
    /// results are clamped to `0..=1`.
    Rgba8Unorm,
}

/// Configuration for the GPU engine.
#[derive(Clone, Debug)]
pub struct GpuConfig {
    /// Wakes an idle host for asynchronous filter parameter changes. The
    /// callback can run on producer threads; offscreen callers may omit it.
    pub redraw: Option<interop::RedrawCallback>,
    /// Uses an existing host device. Handles must share one creation chain.
    /// Device limits and enabled features govern engine capabilities.
    pub device: Option<interop::SharedDevice>,
    /// Which wgpu backends may be used. Defaults to all.
    pub backends: wgpu::Backends,
    /// Adapter power preference. Defaults to high performance.
    pub power_preference: wgpu::PowerPreference,
    /// When true and the adapter supports it,
    /// [`Engine::render`](cherenkov::Engine::render) measures GPU time
    /// with timestamp queries resolved after completion and reported on a
    /// later render, never stalling the frame on GPU idle.
    pub timestamps: bool,
    /// Maximum duration of a GPU wait.
    pub wait_timeout: std::time::Duration,
    /// Memory budgets.
    pub budget: cherenkov::Budget,
    /// When set and the adapter supports pipeline caches, the closed
    /// pipeline set is persisted at this path, best effort.
    pub pipeline_cache: Option<PathBuf>,
    /// The isolation (scratch) texture format. Defaults to
    /// [`ScratchFormat::LinearF16`].
    pub scratch_format: ScratchFormat,
    /// When set, the renderer records an allocation-event trace into
    /// this sink. Diagnostics only; the per-event allocator snapshot is
    /// deliberately expensive, so keep it out of timed runs.
    pub alloc_diag: Option<diag::Sink>,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device: None,
            redraw: None,
            backends: wgpu::Backends::all(),
            power_preference: wgpu::PowerPreference::HighPerformance,
            timestamps: false,
            wait_timeout: std::time::Duration::from_secs(30),
            budget: cherenkov::Budget::default(),
            pipeline_cache: None,
            scratch_format: ScratchFormat::default(),
            alloc_diag: None,
        }
    }
}

/// The surface targets [`Gpu`] draws into, retaining linear Display P3 output.
#[derive(Debug)]
pub enum GpuTarget {
    /// An offscreen texture.
    Offscreen(Offscreen),
    /// A native window; retains readable working-space pixels before presentation.
    Window(WindowTarget),
    /// Engine-owned working-space texture shared with a native host.
    Texture(interop::TextureTarget),
}

/// A window the engine presents on: a raw window handle and the drawable
/// size in pixels. The engine renders into its own linear f16 target and
/// blits it onto the swapchain, so the surface stays readable.
pub struct WindowTarget {
    handle: Box<dyn wgpu::WindowHandle>,
    size: (u32, u32),
    transparent: bool,
    refresh: cherenkov::RefreshRange,
}

impl WindowTarget {
    /// Wraps `handle` (any `raw-window-handle` window, e.g. an
    /// `Arc<winit::window::Window>`) at `size` device pixels.
    pub fn new(handle: impl wgpu::WindowHandle + 'static, size: (u32, u32)) -> Self {
        Self {
            handle: Box::new(handle),
            size,
            transparent: false,
            refresh: cherenkov::DEFAULT_REFRESH,
        }
    }

    /// Presents with a composite alpha mode the compositor sees through
    /// (premultiplied, else postmultiplied). Surface creation
    /// fails when the adapter offers none: an opaque composite would present
    /// every pixel with no alpha.
    #[must_use]
    pub const fn transparent(mut self, transparent: bool) -> Self {
        self.transparent = transparent;
        self
    }

    /// Sets the refresh range for backend animation and presentation retries.
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

    /// The drawable size the swapchain is configured to.
    #[must_use]
    pub const fn size(&self) -> (u32, u32) {
        self.size
    }
}

impl core::fmt::Debug for WindowTarget {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WindowTarget")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl From<WindowTarget> for GpuTarget {
    fn from(window: WindowTarget) -> Self {
        Self::Window(window)
    }
}

impl From<interop::TextureTarget> for GpuTarget {
    fn from(target: interop::TextureTarget) -> Self {
        Self::Texture(target)
    }
}

impl From<Offscreen> for GpuTarget {
    fn from(offscreen: Offscreen) -> Self {
        Self::Offscreen(offscreen)
    }
}

/// The wgpu backend: renders the shared front end's layer trees through a
/// closed instanced-quad pipeline.
#[derive(Clone, Copy, Debug, Default)]
pub struct Gpu;

impl Backend for Gpu {
    type Config = GpuConfig;
    type Info = GpuInfo;
    type Target = GpuTarget;
    type Renderer = render::GpuRenderer;

    #[cfg(not(target_arch = "wasm32"))]
    fn init(config: GpuConfig) -> Result<(Self::Renderer, Self::Info), EngineError> {
        render::init(config)
    }

    #[cfg(target_arch = "wasm32")]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn init(config: GpuConfig) -> Result<(Self::Renderer, Self::Info), EngineError> {
        render::init(config).await
    }
}

impl Uploads<Rgba8> for Gpu {}
impl Uploads<Rgba16F> for Gpu {}

// HDR output (#97): `LinearDisplayP3` texture output and window
// presentation tone-map to the display's `Display::headroom` in the
// present shader — see `render::present`.
impl cherenkov::HdrOutput for Gpu {}

impl cherenkov::GpuContent for Gpu {
    type Content = interop::GpuContentBox;
    fn set_gpu_content(
        r: &mut Self::Renderer,
        surface: cherenkov::SurfaceId,
        layer: cherenkov::LayerId,
        size: (u32, u32),
        content: Self::Content,
    ) {
        r.set_gpu_content(surface, layer, size, content);
    }
    fn resize_gpu_content(
        r: &mut Self::Renderer,
        surface: cherenkov::SurfaceId,
        layer: cherenkov::LayerId,
        size: (u32, u32),
    ) {
        r.resize_gpu_content(surface, layer, size);
    }
}

// External frames (#165): retained producer planes sampled in place — no
// copy, no raster path — decoded and converted into the working space in
// the external fragment pipeline; see `render::external` and
// `render::external.wgsl`.
impl cherenkov::ExternalFrames for Gpu {
    type Frame = interop::ExternalFrame;
    fn set_external_frame(
        r: &mut Self::Renderer,
        surface: cherenkov::SurfaceId,
        layer: cherenkov::LayerId,
        frame: Self::Frame,
    ) {
        r.set_external_frame(surface, layer, frame);
    }
}

impl cherenkov::ShaderPaintCapability for Gpu {
    #[cfg(not(target_arch = "wasm32"))]
    fn add_shader(
        r: &mut Self::Renderer,
        id: cherenkov::ShaderId,
        source: cherenkov::ShaderSource,
    ) -> Result<(), cherenkov::ResourceError> {
        r.add_shader(id, &source)
    }

    #[cfg(target_arch = "wasm32")]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn add_shader(
        r: &mut Self::Renderer,
        id: cherenkov::ShaderId,
        source: cherenkov::ShaderSource,
    ) -> Result<(), cherenkov::ResourceError> {
        r.add_shader(id, &source).await
    }
    fn remove_shader(r: &mut Self::Renderer, id: cherenkov::ShaderId) {
        r.remove_shader(id);
    }
}

impl cherenkov::Filters for Gpu {
    fn remove_filter(r: &mut Self::Renderer, id: cherenkov::FilterId) {
        r.remove_filter(id);
    }
}
impl<F: filtrate_core::Filter + cherenkov::RenderTransfer> cherenkov::Runs<F> for Gpu {
    fn add_filter(r: &mut Self::Renderer, id: cherenkov::FilterId, filter: F) {
        r.add_filter(id, Box::new(render::filter::FromFilter(filter)));
    }
}
impl cherenkov::Effects for Gpu {
    type Effect = interop::EffectBox;
    fn add_effect(r: &mut Self::Renderer, id: cherenkov::FilterId, effect: Self::Effect) {
        r.add_filter(id, effect.0);
    }
}
impl cherenkov::Backdrop for Gpu {
    fn add_backdrop_group(
        r: &mut Self::Renderer,
        surface: cherenkov::SurfaceId,
        id: cherenkov::BackdropId,
    ) {
        r.add_backdrop_group(surface, id, None);
    }
    fn remove_backdrop_group(
        r: &mut Self::Renderer,
        surface: cherenkov::SurfaceId,
        id: cherenkov::BackdropId,
    ) {
        r.remove_backdrop_group(surface, id);
    }
}
impl<K, F> cherenkov::BackdropRuns<K, F> for Gpu
where
    K: filtrate_core::kind::Kind,
    F: cherenkov::BackdropChain<K> + cherenkov::RenderTransfer,
{
    fn add_filtered_backdrop_group(
        r: &mut Self::Renderer,
        surface: cherenkov::SurfaceId,
        id: cherenkov::BackdropId,
        filter: F,
    ) {
        r.add_backdrop_group(
            surface,
            id,
            Some(Box::new(render::filter::FromBackdropChain::<K, F>(
                filter,
                std::marker::PhantomData,
            ))),
        );
    }
}
impl cherenkov::BackdropShaders for Gpu {
    #[cfg(not(target_arch = "wasm32"))]
    fn add_backdrop_shader(
        r: &mut Self::Renderer,
        id: cherenkov::BackdropShaderId,
        source: cherenkov::BackdropShaderSource,
    ) -> Result<(), cherenkov::ResourceError> {
        r.add_backdrop_shader(id, &source)
    }
    #[cfg(target_arch = "wasm32")]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn add_backdrop_shader(
        r: &mut Self::Renderer,
        id: cherenkov::BackdropShaderId,
        source: cherenkov::BackdropShaderSource,
    ) -> Result<(), cherenkov::ResourceError> {
        r.add_backdrop_shader(id, &source).await
    }
    fn remove_backdrop_shader(r: &mut Self::Renderer, id: cherenkov::BackdropShaderId) {
        r.remove_backdrop_shader(id);
    }
}
