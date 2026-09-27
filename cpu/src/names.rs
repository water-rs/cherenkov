// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Feature-name strings carried by
//! [`RenderError::Unsupported`](cherenkov::RenderError) and
//! [`ResourceError::Unsupported`](cherenkov::ResourceError). These are the
//! names the deleted `Unsupported` enum displayed; the benchmark harness
//! maps them back to scene features.

/// A sweep (conic) gradient.
pub const SWEEP: &str = "sweep-gradient";
/// A mesh gradient.
pub const MESH: &str = "mesh-gradient";
/// An image draw or image paint.
pub const IMAGE: &str = "image";
/// A user shader paint.
pub const SHADER: &str = "shader-paint";
/// A blend mode other than normal.
pub const BLEND: &str = "blend-mode";
/// A filter on a group or layer.
pub const FILTER: &str = "filter";
/// A backdrop group.
pub const BACKDROP: &str = "backdrop";
/// A stroked glyph run.
pub const GLYPH_STROKE: &str = "glyph-stroke";
/// A per-glyph transform.
pub const GLYPH_TRANSFORM: &str = "glyph-transform";
/// A colour font (COLR, CBDT or sbix).
pub const COLOR_FONT: &str = "color-font";
/// A blend space other than linear.
pub const BLEND_SPACE: &str = "blend-space";
/// A shadow from a shape without a rounded-box form (in this slice,
/// every shadow).
pub const SHADOW: &str = "shadow";
