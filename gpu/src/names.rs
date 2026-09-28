// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Feature-name strings carried by
//! [`RenderError::Unsupported`](cherenkov::RenderError) and
//! [`ResourceError::Unsupported`](cherenkov::ResourceError). These are the
//! names the deleted `Unsupported` enum displayed; the benchmark harness
//! maps them back to scene features.

/// A general path.
pub const PATH: &str = "path";
/// A user shader paint.
pub const SHADER: &str = "shader-paint";
/// A backdrop group member without a clip.
pub const BACKDROP_UNCLIPPED: &str = "backdrop-unclipped";
/// A backdrop filter footprint too large to bound.
pub const BACKDROP_FOOTPRINT: &str = "backdrop-footprint";
/// A colour font (COLR, CBDT or sbix).
pub const COLOR_FONT: &str = "color-font";
/// A blend space other than linear.
pub const BLEND_SPACE: &str = "blend-space";
/// A path clip whose rasterized mask does not fit the atlas.
pub const PATH_CLIP_TOO_LARGE: &str = "path-clip-too-large";
