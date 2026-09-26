// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cherenkov-cpu`: the CPU raster backend for the Cherenkov 2D rendering
//! engine.
//!
//! The shared front end lives in the [`cherenkov`] crate: [`Engine`],
//! [`Surface`], [`Layer`], the layer tree and the render thread's loop are
//! all generic over [`Backend`]. This crate supplies the render side only —
//! [`Raster`]'s [`Backend`] implementation drives a rayon worker pool on
//! the render thread.
//!
//! Framebuffers are premultiplied linear Display P3, one f32 per channel,
//! rasterized in horizontal bands of 16 rows by an exact signed-area
//! coverage accumulator (font-rs / vello-cpu style): every flattened edge
//! deposits trapezoid areas into a row accumulator and a prefix sum turns
//! it into winding-weighted coverage. Unlike the oracle's per-pixel
//! geometric area, the accumulator is exact only for polygons that do not
//! self-overlap inside a single pixel.
//!
//! Measured on the render corpus, materializing readbacks as f16 costs
//! +0.0013 mean FLIP versus keeping f32 (0.00406 vs 0.00278) — the
//! framebuffer itself is always f32.
//!
//! ```no_run
//! use cherenkov::{Draw, Engine, Offscreen, OffscreenFormat, WorkingColor};
//! use cherenkov::kurbo::Rect;
//! use cherenkov_cpu::{Raster, RasterConfig};
//!
//! let engine = Engine::<Raster>::new(RasterConfig::default())?;
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

use cherenkov::{Backend, EngineError, Offscreen};

/// CPU worker information for provenance.
#[derive(Clone, Debug)]
pub struct RasterInfo {
    /// Number of rayon worker threads.
    pub threads: usize,
    /// The composite kernel in use (`"scalar"` in this slice).
    pub simd: &'static str,
    /// Host CPU model name, best effort.
    pub cpu: Option<String>,
}

/// Configuration for the CPU raster engine.
#[derive(Clone, Debug, Default)]
pub struct RasterConfig {
    /// Worker thread count for the banded rasterizer. `None` uses the
    /// rayon default (one thread per logical core).
    pub threads: Option<usize>,
    /// Memory budgets; only `budget.cpu` is used (the glyph mask cache —
    /// this slice has no device-side memory).
    pub budget: cherenkov::Budget,
}

/// The surface targets [`Raster`] draws into: only an [`Offscreen`]
/// framebuffer in this slice.
#[derive(Debug)]
pub enum RasterTarget {
    /// An offscreen framebuffer.
    Offscreen(Offscreen),
}

impl From<Offscreen> for RasterTarget {
    fn from(offscreen: Offscreen) -> Self {
        Self::Offscreen(offscreen)
    }
}

/// The CPU raster backend: renders the shared front end's layer trees
/// into f32 framebuffers on a rayon pool.
#[derive(Clone, Copy, Debug, Default)]
pub struct Raster;

impl Backend for Raster {
    type Config = RasterConfig;
    type Info = RasterInfo;
    type Target = RasterTarget;
    type Renderer = render::RasterRenderer;

    fn init(config: RasterConfig) -> Result<(Self::Renderer, Self::Info), EngineError> {
        render::init(config)
    }
}
