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
    /// No timestamp queries; [`FrameStats::timings`] stays empty.
    ///
    /// [`FrameStats::timings`]: cherenkov::FrameStats::timings
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
    /// Which wgpu backends may be used. Defaults to all.
    pub backends: wgpu::Backends,
    /// Adapter power preference. Defaults to high performance.
    pub power_preference: wgpu::PowerPreference,
    /// When true and the adapter supports it,
    /// [`Engine::render`](cherenkov::Engine::render) measures GPU time
    /// with deferred timestamp queries.
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
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backends: wgpu::Backends::all(),
            power_preference: wgpu::PowerPreference::HighPerformance,
            timestamps: false,
            wait_timeout: std::time::Duration::from_secs(30),
            budget: cherenkov::Budget::default(),
            pipeline_cache: None,
            scratch_format: ScratchFormat::default(),
        }
    }
}

/// The surface targets [`Gpu`] draws into: only an [`Offscreen`] texture
/// in this slice.
#[derive(Debug)]
pub enum GpuTarget {
    /// An offscreen texture.
    Offscreen(Offscreen),
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
