//! Vortex distortion filter implementation.

use crate::Filter;

/// Applies a vortex-style spiral distortion.
///
/// Parameters: center x and y (uv), radius (1.0 = the shorter edge) and
/// angle (degrees). The displacement is a fraction of the image size, so the
/// footprint is unbounded in pixels.
#[derive(Debug, Clone, Filter)]
#[filter(
    spatial,
    shader = "distortion/vortex_distortion.wgsl",
    footprint = f32::INFINITY
)]
pub struct VortexDistortion<T>(pub [T; 4]);
