//! Hue rotation filter implementation.

use crate::Filter;

/// Rotates the hue of all colors around the color wheel.
///
/// Converts straight-alpha colour to HSL, rotates hue, and converts back to
/// RGB. The HSL round trip is piecewise, so the filter is not
/// [`LINEAR`](crate::ColorFilter::LINEAR).
///
/// # Parameters
///
/// - `angle`: Rotation angle in degrees (0-360)
///
/// # Example
///
/// ```rust
/// # use filtrate::Filter;
/// use filtrate::filters::HueRotation;
///
/// // Rotate 180 degrees (complementary colors)
/// let complement = HueRotation(180.0_f32);
/// # assert_eq!(complement.params(), [180.0]);
/// ```
#[derive(Debug, Clone, Copy, Filter)]
#[filter(color, shader = "color/transform/hue_rotation.wgsl", linear = false)]
pub struct HueRotation<T>(pub T);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Filter;

    #[test]
    fn test_hue_rotation_params() {
        let filter = HueRotation(90.0f32);
        assert_eq!(filter.params(), [90.0]);
    }
}
