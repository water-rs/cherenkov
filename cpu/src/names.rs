// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Feature-name strings carried by
//! [`RenderError::Unsupported`](cherenkov::RenderError) and
//! [`ResourceError::Unsupported`](cherenkov::ResourceError). These are the
//! names the deleted `Unsupported` enum displayed; the benchmark harness
//! maps them back to scene features.

/// A user shader paint.
pub const SHADER: &str = "shader-paint";

/// A filter requiring an auxiliary GPU image with no CPU texels.
pub const FILTER_GPU_IMAGE: &str = "filter-gpu-image";
/// A backdrop member without a clip shape.
pub const BACKDROP_UNCLIPPED: &str = "backdrop-unclipped";
/// A backdrop group whose filter footprint cannot be bounded.
pub const BACKDROP_FOOTPRINT: &str = "backdrop-footprint";
/// A stroked glyph run.
pub const GLYPH_STROKE: &str = "glyph-stroke";
/// A colour font construct this backend cannot render: a bitmap-only
/// font (CBDT/sbix without outlines) or an unmapped COLR paint.
pub const COLOR_FONT: &str = "color-font";

/// A shadow from a shape without a rounded-box form (in this slice,
/// every shadow).
pub const SHADOW: &str = "shadow";
