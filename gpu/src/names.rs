// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Feature-name strings carried by
//! [`RenderError::Unsupported`](cherenkov::RenderError) and
//! [`ResourceError::Unsupported`](cherenkov::ResourceError). These are the
//! names the deleted `Unsupported` enum displayed; the benchmark harness
//! maps them back to scene features.

/// A general path.
pub const PATH: &str = "path";
/// A mesh gradient.
pub const MESH: &str = "mesh-gradient";
/// An image draw or image paint.
#[expect(
    dead_code,
    reason = "vocabulary reserved for lower paths landing later"
)]
pub const IMAGE: &str = "image";
/// A blend mode other than normal.
#[expect(
    dead_code,
    reason = "blend modes are supported; the name is kept for parity"
)]
pub const BLEND: &str = "blend-mode";
/// A backdrop group.
pub const BACKDROP: &str = "backdrop";
/// A stroke join or cap combination with no analytic form.
pub const STROKE_JOIN: &str = "stroke-join";
/// A stroked glyph run.
pub const GLYPH_STROKE: &str = "glyph-stroke";
/// A per-glyph transform.
pub const GLYPH_TRANSFORM: &str = "glyph-transform";
/// A colour font (COLR, CBDT or sbix).
pub const COLOR_FONT: &str = "color-font";
/// A blend space other than linear.
pub const BLEND_SPACE: &str = "blend-space";
/// A shadow from a shape without a rounded-box form.
pub const SHADOW: &str = "shadow";
/// A path clip whose rasterized mask does not fit the atlas.
pub const PATH_CLIP_TOO_LARGE: &str = "path-clip-too-large";
