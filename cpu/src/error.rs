// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Error types for the CPU raster backend.

/// Engine initialization or engine-wide failure.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// The render thread failed.
    #[error("render thread: {0}")]
    Thread(String),
}

/// Surface creation or surface-level failure.
#[derive(Debug, thiserror::Error)]
pub enum SurfaceError {
    /// The engine failed while creating the surface.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// The requested size exceeds the framebuffer limit.
    #[error("surface {width}x{height} exceeds the maximum surface size {max}")]
    TooLarge {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
        /// Engine maximum.
        max: u32,
    },
    /// The render thread is gone.
    #[error("the render thread is gone")]
    Lost,
}

/// Resource registration failure.
#[derive(Debug, thiserror::Error)]
pub enum ResourceError {
    /// The font data could not be parsed.
    #[error("font: {0}")]
    Font(String),
    /// The image data failed validation.
    #[error("image: {0}")]
    Image(String),
    /// The resource needs a feature this slice does not implement.
    #[error(transparent)]
    Unsupported(#[from] Unsupported),
    /// Reading the resource failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Rendering or readback failure.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// The frame needs a feature this slice does not implement.
    #[error(transparent)]
    Unsupported(#[from] Unsupported),
    /// The render thread failed or stopped.
    #[error("render thread stopped")]
    Thread,
    /// A paint or draw references an unregistered image.
    #[error("unregistered image {0}")]
    Image(u64),
    /// Pixel readback failed.
    #[error("readback: {0}")]
    Readback(String),
    /// A glyph run references a font that is not registered, or glyph
    /// rasterization is not yet implemented.
    #[error("font: {0}")]
    Font(String),
    /// The frame's glyph masks exceed the glyph cache budget.
    #[error("glyph cache exhausted")]
    GlyphCacheExhausted,
}

/// A feature the engine vocabulary has but this backend slice does not draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum Unsupported {
    /// A user shader paint.
    Shader,
    /// A filter on a group.
    Filter,
    /// A colour font (COLR, CBDT or sbix).
    ColorFont,
}

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Shader => "shader-paint",
            Self::Filter => "filter",
            Self::ColorFont => "color-font",
        })
    }
}
