// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Frame driving: [`Engine::render`](crate::Engine::render) input, the
//! [`Next`] scheduling answer, per-frame statistics and readback types.
//!
//! [`Engine::render`]: crate::Engine::render

use std::ops::RangeInclusive;
use std::time::Instant;

/// The presentation timestamp handed to [`Engine::render`](crate::Engine::render).
#[derive(Clone, Copy, Debug)]
pub struct FrameTime(pub Instant);

impl FrameTime {
    /// The frame time at `t`.
    #[must_use]
    pub const fn at(t: Instant) -> Self {
        Self(t)
    }

    /// The frame time now.
    #[must_use]
    pub fn now() -> Self {
        Self(Instant::now())
    }
}

/// An inclusive refresh-rate range in hertz.
pub type RefreshRange = RangeInclusive<u32>;

/// What the engine needs next, returned by
/// [`Engine::render`](crate::Engine::render).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Next {
    /// No animation is running; the display link may sleep.
    Idle,
    /// The next frame is needed at `time`, at a refresh rate in `rate`.
    At {
        /// When the next frame is due.
        time: Instant,
        /// The acceptable refresh rates.
        rate: RefreshRange,
    },
}

/// One timed render pass of the last [`Engine::render`](crate::Engine::render).
#[derive(Clone, Debug)]
pub struct PassTiming {
    /// The pass's deterministic name: `"surface"` or `"scratch{n}"` by
    /// isolation depth.
    pub name: String,
    /// Target width in pixels.
    pub width: u32,
    /// Target height in pixels.
    pub height: u32,
    /// Target texture format (`"rgba16float"`, `"rgba8unorm"`, ...).
    pub format: &'static str,
    /// GPU seconds the pass took, between its pass-boundary timestamp
    /// writes.
    pub gpu_seconds: f64,
}

/// Measurements of the last [`Engine::render`](crate::Engine::render).
#[derive(Clone, Debug, Default)]
pub struct FrameStats {
    /// GPU seconds the frame took, when timestamp queries are enabled and
    /// supported.
    pub gpu_seconds: Option<f64>,
    /// Per-pass GPU seconds, in submission order; empty when timestamp
    /// queries are disabled or unsupported.
    pub passes_timed: Vec<PassTiming>,
    /// Render passes recorded (`render_to_texture` and effect passes).
    pub passes: u32,
    /// Scene commands encoded this frame.
    pub draws: u32,
    /// Layer content instances drawn.
    pub instances: u32,
    /// Glyphs rasterized this frame.
    pub glyphs_rasterized: u32,
    /// Paths rasterized this frame.
    pub paths_rasterized: u32,
    /// Display-list commands lowered this frame (a full lowering or the
    /// dirty commands of a slot update).
    pub commands_lowered: u32,
    /// Layers whose device-space content run was rebuilt rather than reused.
    pub layers_composed: u32,
}

/// Decoded pixels of a surface readback: premultiplied linear Display P3,
/// row-major.
#[derive(Clone, Debug)]
pub struct Readback {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width * height` premultiplied RGBA pixels.
    pub pixels: Vec<[f32; 4]>,
}

/// The storage format of an [`Offscreen`] target.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OffscreenFormat {
    /// Linear `Rgba16Float`.
    #[default]
    LinearF16,
    /// Linear `Rgba32Float`.
    LinearF32,
}

/// An offscreen render target description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Offscreen {
    /// The target size in pixels.
    pub size: (u32, u32),
    /// The target storage format.
    pub format: OffscreenFormat,
}

impl Offscreen {
    /// An offscreen target of `size` pixels in `format`.
    #[must_use]
    pub const fn new(size: (u32, u32), format: OffscreenFormat) -> Self {
        Self { size, format }
    }
}
