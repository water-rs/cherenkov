//! Gaussian blur filter implementation.

use crate::{
    Filter, FilterParam, Footprint, OperatingSpace, ParamSource, Placed, SignalVisitor,
    SpatialFilter, SpatialStage, StageCollector, kind,
};

/// The separable gaussian blur's stage: one axis, specialized per pass.
const GAUSSIAN_BLUR: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/shaders/image/blur/gaussian_blur.wgsl"
));

const HORIZONTAL: SpatialStage = SpatialStage {
    name: "gaussian_blur_horizontal",
    source: GAUSSIAN_BLUR,
    params: &[ParamSource::Param(0), ParamSource::Constant(&[1.0, 0.0])],
    space: OperatingSpace::Working,
    shape: None,
    aux: &[],
};

const VERTICAL: SpatialStage = SpatialStage {
    name: "gaussian_blur_vertical",
    source: GAUSSIAN_BLUR,
    params: &[ParamSource::Param(0), ParamSource::Constant(&[0.0, 1.0])],
    space: OperatingSpace::Working,
    shape: None,
    aux: &[],
};

/// Applies a separable gaussian blur: a horizontal then a vertical pass.
///
/// # Parameters
///
/// - `sigma`: Gaussian standard deviation in pixels; the kernel radius, and
///   the footprint, is `ceil(3 * sigma)`.
#[derive(Debug, Clone, Copy)]
pub struct GaussianBlur<T>(pub T);

impl<T: FilterParam> Filter for GaussianBlur<T> {
    type Kind = kind::Spatial;
    type Params = [f32; 1];

    #[inline]
    fn params(&self) -> [f32; 1] {
        [self.0.snapshot()]
    }

    fn collect_stages<C: StageCollector>(&self, c: &mut C) {
        c.spatial(Placed::new(&HORIZONTAL));
        c.spatial(Placed::new(&VERTICAL));
    }

    fn visit_signals<V: SignalVisitor>(&self, v: &mut V) {
        v.visit(0, &self.0);
    }
}

impl<T: FilterParam> SpatialFilter for GaussianBlur<T> {
    fn footprint_of(params: &[f32; 1]) -> Footprint {
        Footprint::pixels((params[0].max(0.001) * 3.0).ceil())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_footprint_is_three_sigma() {
        assert_eq!(GaussianBlur(2.0f32).footprint(), Footprint::pixels(6.0));
        assert_eq!(GaussianBlur(0.4f32).footprint(), Footprint::pixels(2.0));
    }
}
