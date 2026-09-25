//! Perspective transform filter implementation.

use crate::Filter;

/// Maps the image onto a quadrilateral; pixels outside it are transparent.
///
/// Parameters: the quad's top-left, top-right, bottom-right and bottom-left
/// corners, x then y, in uv. Any pixel can read any other, so the footprint
/// is unbounded.
#[derive(Debug, Clone, Filter)]
#[filter(
    spatial,
    shader = "distortion/perspective.wgsl",
    footprint = f32::INFINITY,
    constants = [1.0]
)]
pub struct PerspectiveTransform<T>(pub [T; 8]);
