// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Feature-name strings carried by
//! [`RenderError::Unsupported`](cherenkov::RenderError) and
//! [`ResourceError::Unsupported`](cherenkov::ResourceError). These are the
//! names the deleted `Unsupported` enum displayed; the benchmark harness
//! maps them back to scene features.

/// A mesh gradient.
pub const MESH_GRADIENT: &str = "mesh-gradient";
/// A gradient interpolation space other than the working space or
/// sRGB-encoded.
pub const INTERPOLATION: &str = "gradient-interpolation";
/// A user shader paint.
#[expect(
    dead_code,
    reason = "vocabulary reserved for lower paths landing later"
)]
pub const SHADER: &str = "shader-paint";
/// A shader paint declaring more than 64 uniform floats.
pub const SHADER_PARAMS: &str = "shader-params";
/// A shader paint applied to a glyph run.
pub const SHADER_GLYPHS: &str = "shader-glyphs";
/// A blend space vello cannot honour (it blends in the encoded 8-bit
/// target).
pub const BLEND_SPACE: &str = "blend-space";
/// A filter on a group. Filters are a layer property in this backend.
pub const GROUP_FILTER: &str = "group-filter";
/// A filter on a layer.
#[expect(
    dead_code,
    reason = "vocabulary reserved for lower paths landing later"
)]
pub const FILTER: &str = "filter";
/// A window surface.
#[expect(
    dead_code,
    reason = "vocabulary reserved for lower paths landing later"
)]
pub const WINDOW_SURFACE: &str = "window-surface";
/// An image draw or image paint.
#[expect(
    dead_code,
    reason = "vocabulary reserved for lower paths landing later"
)]
pub const IMAGE: &str = "image";
/// A shadow from a shape the blurred rounded-rect primitive cannot
/// express.
pub const SHADOW: &str = "shadow";
/// A per-glyph transform.
pub const GLYPH_TRANSFORM: &str = "glyph-transform";
/// A colour font (COLR, CBDT or sbix).
#[expect(
    dead_code,
    reason = "vocabulary reserved for lower paths landing later"
)]
pub const COLOR_FONT: &str = "color-font";
/// A transparent gradient or pattern extend (`Extend::None`).
pub const EXTEND: &str = "extend-none";
