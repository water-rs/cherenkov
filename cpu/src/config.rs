// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Engine configuration and reporting types.

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
/// The CPU budget is split equally between glyph masks and prepared coverage.
/// Framebuffers and in-flight frame data are accounted separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// Retained CPU caches (glyph masks and prepared geometric coverage).
    pub cpu: Bytes,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
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
    /// Surface framebuffer bytes (f32 RGBA per pixel, plus isolation
    /// scratch retained for reuse).
    pub framebuffers: Bytes,
    /// Rasterized glyph masks and retained outline cache bytes.
    pub glyph_cache: Bytes,
    /// Prepared shape, clip and shadow coverage cache bytes.
    pub coverage_cache: Bytes,
    /// Registered image pixel stores (premultiplied f32 RGBA).
    pub images: Bytes,
}

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
    /// Memory budgets.
    pub budget: Budget,
}
