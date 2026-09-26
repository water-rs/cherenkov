// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Engine configuration and reporting types.

use std::path::PathBuf;
use std::time::Duration;

/// A byte count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bytes(pub u64);

impl Bytes {
    /// A count of mebibytes.
    #[must_use]
    pub const fn mib(n: u64) -> Self {
        Self(n * 1024 * 1024)
    }
}

/// The engine's memory budgets.
///
/// This slice only records the budgets; the GPU budget caps the glyph atlas
/// size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// Device-side memory (buffers, textures, the glyph atlas).
    pub gpu: Bytes,
    /// CPU-side memory (the rasterized glyph cache).
    pub cpu: Bytes,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            gpu: Bytes::mib(512),
            cpu: Bytes::mib(96),
        }
    }
}

/// System memory pressure reported to the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pressure {
    /// Reduce caches where cheap.
    Moderate,
    /// Drop every cache.
    Critical,
}

/// The engine's current memory usage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryUsage {
    /// Buffers, textures and the glyph atlas, in bytes.
    pub gpu: Bytes,
    /// CPU-side atlas cache bytes.
    pub cpu: Bytes,
}

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
    /// Where the adapter can sample GPU timestamps.
    pub timestamps: TimestampSupport,
}

/// Where an adapter can sample GPU timestamps. The renderer only ever
/// samples at pass boundaries, which every supporting level offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimestampSupport {
    /// No timestamp queries; [`FrameStats::timings`] stays empty.
    ///
    /// [`FrameStats::timings`]: crate::FrameStats::timings
    Unsupported,
    /// Only at render and compute pass boundaries (Apple GPUs on Metal
    /// sample at stage boundaries).
    PassBoundaries,
    /// Anywhere inside a command encoder as well as at pass boundaries.
    Encoders,
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
    /// When true and the adapter supports it, [`Engine::render`] measures
    /// GPU time with timestamp queries resolved a frame late — never
    /// stalling the frame on GPU idle.
    ///
    /// [`Engine::render`]: crate::Engine::render
    pub timestamps: bool,
    /// Memory budgets.
    pub budget: Budget,
    /// When set and the adapter supports pipeline caches, the closed pipeline
    /// set is persisted at this path, best effort.
    pub pipeline_cache: Option<PathBuf>,
    /// The isolation (scratch) texture format. Defaults to
    /// [`ScratchFormat::LinearF16`].
    pub scratch_format: ScratchFormat,
    /// The longest the render thread blocks on the GPU for one submission
    /// (frame passes, timestamp or pixel readback) before it fails with
    /// [`RenderError::Timeout`] instead of spinning forever. Defaults to
    /// 30 seconds.
    ///
    /// [`RenderError::Timeout`]: crate::RenderError::Timeout
    pub wait_timeout: Duration,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backends: wgpu::Backends::all(),
            power_preference: wgpu::PowerPreference::HighPerformance,
            timestamps: false,
            budget: Budget::default(),
            pipeline_cache: None,
            scratch_format: ScratchFormat::default(),
            wait_timeout: Duration::from_secs(30),
        }
    }
}
