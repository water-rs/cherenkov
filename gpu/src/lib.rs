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

use std::path::PathBuf;

use cherenkov::{Backend, EngineError, Offscreen, Rgba8, Uploads};

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
    /// Which wgpu backends may be used. Defaults to all.
    pub backends: wgpu::Backends,
    /// Adapter power preference. Defaults to high performance.
    pub power_preference: wgpu::PowerPreference,
    /// When true and the adapter supports it,
    /// [`Engine::render`](cherenkov::Engine::render) measures GPU time
    /// with drained timestamp queries.
    pub timestamps: bool,
    /// Memory budgets.
    pub budget: cherenkov::Budget,
    /// When set and the adapter supports pipeline caches, the closed
    /// pipeline set is persisted at this path, best effort.
    pub pipeline_cache: Option<PathBuf>,
    /// The isolation (scratch) texture format. Defaults to
    /// [`ScratchFormat::LinearF16`].
    pub scratch_format: ScratchFormat,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backends: wgpu::Backends::all(),
            power_preference: wgpu::PowerPreference::HighPerformance,
            timestamps: false,
            budget: cherenkov::Budget::default(),
            pipeline_cache: None,
            scratch_format: ScratchFormat::default(),
        }
    }
}

/// The surface targets [`Gpu`] draws into: an [`Offscreen`] texture or a
/// [`WindowTarget`] presented through a wgpu swapchain.
#[derive(Debug)]
pub enum GpuTarget {
    /// An offscreen texture.
    Offscreen(Offscreen),
    /// A window.
    Window(WindowTarget),
}

/// A window the engine presents on: a raw window handle and the drawable
/// size in pixels. The engine renders into its own linear f16 target and
/// blits it onto the swapchain, so the surface stays readable.
pub struct WindowTarget {
    handle: Box<dyn wgpu::WindowHandle>,
    size: (u32, u32),
    transparent: bool,
}

impl WindowTarget {
    /// Wraps `handle` (any `raw-window-handle` window, e.g. an
    /// `Arc<winit::window::Window>`) at `size` device pixels.
    pub fn new(handle: impl wgpu::WindowHandle + 'static, size: (u32, u32)) -> Self {
        Self {
            handle: Box::new(handle),
            size,
            transparent: false,
        }
    }

    /// Presents with a composite alpha mode the compositor sees through
    /// (premultiplied, else postmultiplied, else inherited). Surface creation
    /// fails when the adapter offers none: an opaque composite would present
    /// every pixel with no alpha.
    #[must_use]
    pub const fn transparent(mut self, transparent: bool) -> Self {
        self.transparent = transparent;
        self
    }

    /// The drawable size the swapchain is configured to.
    #[must_use]
    pub const fn size(&self) -> (u32, u32) {
        self.size
    }

    pub(crate) fn into_parts(self) -> (Box<dyn wgpu::WindowHandle>, (u32, u32), bool) {
        (self.handle, self.size, self.transparent)
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

    fn init(config: GpuConfig) -> Result<(Self::Renderer, Self::Info), EngineError> {
        render::init(config)
    }
}

impl Uploads<Rgba8> for Gpu {}

impl cherenkov::Filters for Gpu {
    fn remove_filter(r: &mut Self::Renderer, id: cherenkov::FilterId) {
        r.filters.remove(id.raw());
    }
}

impl<F: filtrate_core::Filter + Send> cherenkov::Runs<F> for Gpu {
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
