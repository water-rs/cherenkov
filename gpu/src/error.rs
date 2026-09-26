// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Error types for the GPU backend.

/// Engine initialization or engine-wide failure.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// No suitable GPU adapter was found.
    #[error("no suitable wgpu adapter")]
    NoAdapter,
    /// The device request failed.
    #[error("device request failed: {0}")]
    RequestDevice(String),
    /// The render thread failed.
    #[error("render thread: {0}")]
    Thread(String),
    /// Shader module or pipeline creation failed.
    #[error("shader: {0}")]
    Shader(String),
}

/// Surface creation or surface-level failure.
#[derive(Debug, thiserror::Error)]
pub enum SurfaceError {
    /// The engine failed while creating the surface.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// The requested size exceeds the device limit.
    #[error("surface {width}x{height} exceeds the maximum texture size {max}")]
    TooLarge {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
        /// Device maximum.
        max: u32,
    },
    /// The GPU device or surface is lost.
    #[error("the device was lost")]
    Lost,
    /// A zero-size surface cannot hold a target.
    #[error("surface size must be non-zero")]
    ZeroSize,
}

/// Resource registration failure.
#[derive(Debug, thiserror::Error)]
pub enum ResourceError {
    /// The font data could not be parsed.
    #[error("font: {0}")]
    Font(String),
    /// The image data is malformed.
    #[error("image: {0}")]
    Image(String),
    /// The resource needs a feature this slice does not implement.
    #[error(transparent)]
    Unsupported(#[from] Unsupported),
    /// Reading the resource failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// The render thread is gone.
    #[error("the render thread is gone")]
    Lost,
}

/// Rendering or readback failure.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// The frame needs a feature this slice does not implement.
    #[error(transparent)]
    Unsupported(#[from] Unsupported),
    /// The GPU device was lost.
    #[error("the device was lost")]
    DeviceLost,
    /// The render thread failed or stopped.
    #[error("render thread stopped")]
    Thread,
    /// Pixel readback failed.
    #[error("readback: {0}")]
    Readback(String),
    /// A glyph run references a font that is not registered.
    #[error("font: {0}")]
    Font(String),
    /// A draw references an image that is not registered.
    #[error("image: {0}")]
    Image(u64),
    /// The glyph atlas is full; the caller may grow or clear it and retry.
    #[error("glyph atlas full")]
    AtlasFull,
    /// The frame's live atlas set exceeds the maximum atlas size.
    #[error("glyph atlas exhausted")]
    AtlasExhausted,
}

/// A feature the engine vocabulary has but this backend slice does not draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, thiserror::Error)]
pub enum Unsupported {
    /// A general path.
    Path,
    /// A sweep (conic) gradient.
    Sweep,
    /// A mesh gradient.
    Mesh,
    /// An image draw or image paint.
    Image,
    /// A user shader paint.
    Shader,
    /// A blend mode other than normal.
    Blend(cherenkov::BlendMode),
    /// A filter on a group.
    Filter,
    /// A dashed stroke.
    StrokeDash,
    /// A stroke join or cap combination with no analytic form.
    StrokeJoin,
    /// A stroked glyph run.
    GlyphStroke,
    /// A per-glyph transform.
    GlyphTransform,
    /// A colour font (COLR, CBDT or sbix).
    ColorFont,
    /// A blend space other than linear.
    BlendSpace,
    /// A shadow from a shape without a rounded-box form.
    Shadow,
    /// A path clip whose rasterized mask does not fit the atlas.
    PathClipTooLarge,
}

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Path => "path",
            Self::Sweep => "sweep-gradient",
            Self::Mesh => "mesh-gradient",
            Self::Image => "image",
            Self::Shader => "shader-paint",
            Self::Blend(_) => "blend-mode",
            Self::Filter => "filter",
            Self::StrokeDash => "stroke-dash",
            Self::StrokeJoin => "stroke-join",
            Self::GlyphStroke => "glyph-stroke",
            Self::GlyphTransform => "glyph-transform",
            Self::ColorFont => "color-font",
            Self::BlendSpace => "blend-space",
            Self::Shadow => "shadow",
            Self::PathClipTooLarge => "path-clip-too-large",
        })
    }
}
