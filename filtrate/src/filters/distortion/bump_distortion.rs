//! Bump distortion filter implementation.

use crate::Filter;

/// Applies convex/concave bump distortion around a center.
///
/// Parameters: center x and y (uv), radius (1.0 = the shorter edge) and
/// scale. The displacement is a fraction of the image size, so the footprint
/// is unbounded in pixels.
#[derive(Debug, Clone, Filter)]
#[filter(
    spatial,
    shader = "distortion/bump_distortion.wgsl",
    footprint = f32::INFINITY
)]
pub struct BumpDistortion<T>(pub [T; 4]);
