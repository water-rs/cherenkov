//! Twirl distortion filter implementation.

use crate::Filter;

/// Applies a twirl distortion around a center point.
///
/// Parameters: center x and y (uv), radius (1.0 = the shorter edge) and
/// angle (degrees). The displacement is a fraction of the image size, so the
/// footprint is unbounded in pixels.
#[derive(Debug, Clone, Filter)]
#[filter(
    spatial,
    shader = "distortion/twirl_distortion.wgsl",
    footprint = f32::INFINITY
)]
pub struct TwirlDistortion<T>(pub [T; 4]);
