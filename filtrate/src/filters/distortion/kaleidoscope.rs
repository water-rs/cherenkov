//! Kaleidoscope filter implementation.

use crate::Filter;

/// Reflects content around repeated angular wedges.
///
/// Parameters: segments, rotation (degrees) and center x and y (uv). Any
/// pixel can read any other, so the footprint is unbounded.
#[derive(Debug, Clone, Filter)]
#[filter(
    spatial,
    shader = "distortion/kaleidoscope.wgsl",
    footprint = f32::INFINITY
)]
pub struct Kaleidoscope<T>(pub [T; 4]);
