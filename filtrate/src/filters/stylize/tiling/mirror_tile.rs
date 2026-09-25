//! Mirror tile filter implementation.

use crate::Filter;

/// Repeats the image through mirrored tiling.
///
/// Parameters: the repeat counts along x and y. Any pixel can read any
/// other, so the footprint is unbounded.
#[derive(Debug, Clone, Filter)]
#[filter(
    spatial,
    shader = "stylize/tiling/mirror_tile.wgsl",
    footprint = f32::INFINITY
)]
pub struct MirrorTile<T>(pub [T; 2]);
