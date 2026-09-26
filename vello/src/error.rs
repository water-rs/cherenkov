// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Error types for the Vello backend.

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
    /// The vello renderer or a shader failed to initialize.
    #[error("renderer: {0}")]
    Renderer(String),
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
    /// The surface needs a feature this slice does not implement.
    #[error(transparent)]
    Unsupported(#[from] Unsupported),
    /// The GPU device or surface is lost.
    #[error("the device was lost")]
    Lost,
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
    /// The surface's pixels cannot be read back (window surfaces).
    #[error("the surface is not readable")]
    NotReadable,
    /// Pixel readback failed.
    #[error("readback: {0}")]
    Readback(String),
    /// A glyph run references a font that is not registered.
    #[error("font: {0}")]
    Font(String),
    /// A draw references an image that is not registered.
    #[error("image: {0}")]
    Image(String),
}

/// A feature the engine vocabulary has but this backend slice does not draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum Unsupported {
    /// A mesh gradient.
    MeshGradient,
    /// A gradient interpolation space other than the working space or
    /// sRGB-encoded.
    Interpolation,
    /// A user shader paint.
    Shader,
    /// A shader paint applied to a glyph run.
    ShaderGlyphs,
    /// A blend space vello cannot honour (it blends in the encoded 8-bit
    /// target).
    BlendSpace,
    /// A filter on a group. Filters are a layer property in this backend.
    GroupFilter,
    /// A filter on a layer.
    Filter,
    /// A window surface.
    WindowSurface,
    /// An image draw or image paint.
    Image,
    /// A shadow from a shape the blurred rounded-rect primitive cannot
    /// express.
    Shadow,
    /// A per-glyph transform.
    GlyphTransform,
    /// A colour font (COLR, CBDT or sbix).
    ColorFont,
}

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::MeshGradient => "mesh-gradient",
            Self::Interpolation => "gradient-interpolation",
            Self::Shader => "shader-paint",
            Self::ShaderGlyphs => "shader-glyphs",
            Self::BlendSpace => "blend-space",
            Self::GroupFilter => "group-filter",
            Self::Filter => "filter",
            Self::WindowSurface => "window-surface",
            Self::Image => "image",
            Self::Shadow => "shadow",
            Self::GlyphTransform => "glyph-transform",
            Self::ColorFont => "color-font",
        })
    }
}
