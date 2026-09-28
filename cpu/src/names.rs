// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Feature-name strings carried by
//! [`RenderError::Unsupported`](cherenkov::RenderError) and
//! [`ResourceError::Unsupported`](cherenkov::ResourceError). These are the
//! names the deleted `Unsupported` enum displayed; the benchmark harness
//! maps them back to scene features.

/// A user shader paint.
pub const SHADER: &str = "shader-paint";

/// A filter on a group or layer.
pub const FILTER: &str = "filter";
/// A backdrop group.
pub const BACKDROP: &str = "backdrop";
/// A stroked glyph run.
pub const GLYPH_STROKE: &str = "glyph-stroke";
/// A colour font (COLR, CBDT or sbix).
pub const COLOR_FONT: &str = "color-font";

/// A shadow from a shape without a rounded-box form (in this slice,
/// every shadow).
pub const SHADOW: &str = "shadow";
