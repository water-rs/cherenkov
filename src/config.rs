// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Engine configuration and reporting types shared by every backend.

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
    /// CPU-side cache bytes.
    pub cpu: Bytes,
}
