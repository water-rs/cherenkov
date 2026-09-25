//! The [`Filter`] trait, its kinds, and [`Chain`].

use crate::{
    ImageVisitor, ParamArray, SignalVisitor, StageCollector,
    kind::{self, ChainFootprint, Kind},
    visitor::{OffsetCollector, OffsetImages, OffsetVisitor},
};

/// A filter: pure data describing the stages that apply it and the
/// parameters that feed them.
///
/// Filters hold no GPU state. [`Filter::collect_stages`] reports each stage
/// in order; an executor composes the stages' shader functions and runs them.
/// A filter also implements exactly one of [`ColorFilter`] and
/// [`SpatialFilter`], matching its [`Filter::Kind`].
pub trait Filter: 'static {
    /// [`kind::Color`] or [`kind::Spatial`].
    type Kind: Kind;

    /// The flattened parameter values, in the order the stages index them.
    type Params: ParamArray;

    /// The number of auxiliary images the filter provides through
    /// [`Filter::visit_images`].
    const IMAGES: usize = 0;

    /// Snapshots the current parameter values.
    fn params(&self) -> Self::Params;

    /// Reports every stage, in application order.
    ///
    /// Chains forward to their halves, shifting the second half's parameter
    /// and image indices past the first half's.
    fn collect_stages<C: StageCollector>(&self, collector: &mut C);

    /// Visits every reactive parameter at its index in [`Filter::Params`].
    /// Parameters that are not visited keep their snapshot value.
    fn visit_signals<V: SignalVisitor>(&self, _visitor: &mut V) {}

    /// Visits the auxiliary images, indexed `0..Self::IMAGES`.
    fn visit_images<V: ImageVisitor>(&self, _visitor: &mut V) {}
}

/// A filter that maps each pixel's colour to a colour, independently of its
/// neighbours and of its position.
pub trait ColorFilter: Filter<Kind = kind::Color> {
    /// Whether the filter is a linear map on premultiplied RGBA with an
    /// identity alpha row and zero offset.
    ///
    /// Only such a map commutes with source-over compositing,
    /// `M(a + (1 - αa) b) = M a + (1 - αa) M b`, which is necessary (not
    /// sufficient) for an executor to push the filter down into the shading
    /// of each primitive instead of applying it to the composited group. An
    /// operation that touches alpha, adds an offset that does not scale with
    /// alpha, clamps, or unpremultiplies is not linear.
    const LINEAR: bool;
}

/// A filter that samples its input around each pixel.
pub trait SpatialFilter: Filter<Kind = kind::Spatial> {
    /// The largest distance, in pixels along either axis, between an output
    /// pixel and any input texel it reads, for the given parameters.
    /// `f32::INFINITY` when the reach is unbounded or scales with the image
    /// size (warps expressed in normalized coordinates, for example).
    ///
    /// Executors bound an animating filter by evaluating this at every
    /// parameter's largest magnitude over its animation track (see
    /// [`AnimationTrack::magnitude_bound`](crate::AnimationTrack::magnitude_bound)).
    /// That bound is sound because implementations satisfy, for every
    /// parameter `p`: the footprint at `p` is at most the footprint at `|p|`,
    /// and the footprint does not decrease as a non-negative parameter grows.
    fn footprint_of(params: &Self::Params) -> f32;

    /// The footprint for the current parameters.
    fn footprint(&self) -> f32 {
        Self::footprint_of(&self.params())
    }
}

/// Two filters applied in sequence.
///
/// A chain is a [`ColorFilter`] exactly when both halves are, and a
/// [`SpatialFilter`] otherwise; its parameters are the pair of its halves'
/// parameters.
#[derive(Debug, Clone, Copy)]
pub struct Chain<A: Filter, B: Filter> {
    /// The filter applied first.
    pub first: A,
    /// The filter applied second.
    pub second: B,
}

impl<A: Filter, B: Filter> Filter for Chain<A, B> {
    type Kind = <A::Kind as Kind>::Then<B::Kind>;
    type Params = (A::Params, B::Params);
    const IMAGES: usize = A::IMAGES + B::IMAGES;

    #[inline]
    fn params(&self) -> Self::Params {
        (self.first.params(), self.second.params())
    }

    fn collect_stages<C: StageCollector>(&self, collector: &mut C) {
        self.first.collect_stages(collector);
        self.second.collect_stages(&mut OffsetCollector::new(
            collector,
            <A::Params as ParamArray>::LEN,
            A::IMAGES,
        ));
    }

    fn visit_signals<V: SignalVisitor>(&self, visitor: &mut V) {
        self.first.visit_signals(visitor);
        self.second.visit_signals(&mut OffsetVisitor::new(
            visitor,
            <A::Params as ParamArray>::LEN,
        ));
    }

    fn visit_images<V: ImageVisitor>(&self, visitor: &mut V) {
        self.first.visit_images(visitor);
        self.second
            .visit_images(&mut OffsetImages::new(visitor, A::IMAGES));
    }
}

impl<A: ColorFilter, B: ColorFilter> ColorFilter for Chain<A, B> {
    const LINEAR: bool = A::LINEAR && B::LINEAR;
}

impl<A: Filter, B: Filter> SpatialFilter for Chain<A, B>
where
    A::Kind: Kind<Then<B::Kind> = kind::Spatial>,
    (A::Kind, B::Kind): ChainFootprint<A, B>,
{
    fn footprint_of(params: &Self::Params) -> f32 {
        <(A::Kind, B::Kind) as ChainFootprint<A, B>>::footprint(&params.0, &params.1)
    }
}

/// Extension trait for chaining filters.
pub trait FilterExt: Filter + Sized {
    /// Applies `filter` after this one.
    fn then<F: Filter>(self, filter: F) -> Chain<Self, F> {
        Chain {
            first: self,
            second: filter,
        }
    }
}

impl<T: Filter> FilterExt for T {}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec::Vec;

    use super::*;
    use crate::{ColorStage, OperatingSpace, ParamSource, Placed, SpatialStage};

    const COLOR: ColorStage = ColorStage {
        name: "color",
        source: "",
        params: &[ParamSource::Param(0)],
        space: OperatingSpace::Working,
    };

    const SPATIAL: SpatialStage = SpatialStage {
        name: "spatial",
        source: "",
        params: &[ParamSource::Param(0), ParamSource::Param(1)],
        space: OperatingSpace::Working,
        shape: None,
        aux: &[],
    };

    struct Tint;
    impl Filter for Tint {
        type Kind = kind::Color;
        type Params = [f32; 1];

        fn params(&self) -> [f32; 1] {
            [1.0]
        }
        fn collect_stages<C: StageCollector>(&self, c: &mut C) {
            c.color(Placed::new(&COLOR));
        }
    }
    impl ColorFilter for Tint {
        const LINEAR: bool = true;
    }

    struct Clamp;
    impl Filter for Clamp {
        type Kind = kind::Color;
        type Params = [f32; 0];

        fn params(&self) -> [f32; 0] {
            []
        }
        fn collect_stages<C: StageCollector>(&self, _: &mut C) {}
    }
    impl ColorFilter for Clamp {
        const LINEAR: bool = false;
    }

    struct Spread;
    impl Filter for Spread {
        type Kind = kind::Spatial;
        type Params = [f32; 2];

        fn params(&self) -> [f32; 2] {
            [2.0, 3.0]
        }
        fn collect_stages<C: StageCollector>(&self, c: &mut C) {
            c.spatial(Placed::new(&SPATIAL));
        }
    }
    impl SpatialFilter for Spread {
        fn footprint_of(params: &[f32; 2]) -> f32 {
            params[0]
        }
    }

    /// Accepts only colour filters, to prove a chain's kind statically.
    const fn linear<F: ColorFilter>() -> bool {
        F::LINEAR
    }

    #[test]
    fn a_chain_of_colour_filters_is_a_colour_filter() {
        const { assert!(linear::<Chain<Tint, Tint>>()) };
        const { assert!(!linear::<Chain<Tint, Chain<Tint, Clamp>>>()) };
    }

    #[test]
    fn a_chain_with_a_spatial_half_is_spatial_and_adds_footprints() {
        assert_eq!(Tint.then(Spread).footprint(), 2.0);
        assert_eq!(Spread.then(Tint).footprint(), 2.0);
        assert_eq!(Spread.then(Tint).then(Spread).footprint(), 4.0);
    }

    #[test]
    fn chain_params_are_concatenated_tuple() {
        let (a, b) = Tint.then(Spread).params();
        assert_eq!(a, [1.0]);
        assert_eq!(b, [2.0, 3.0]);
    }

    #[test]
    fn collect_stages_walks_the_chain_and_shifts_indices() {
        struct Recording(Vec<(&'static str, usize, usize)>);
        impl StageCollector for Recording {
            fn color(&mut self, stage: Placed<ColorStage>) {
                self.0
                    .push((stage.stage.name, stage.param_base, stage.image_base));
            }
            fn spatial(&mut self, stage: Placed<SpatialStage>) {
                self.0
                    .push((stage.stage.name, stage.param_base, stage.image_base));
            }
        }

        let mut recording = Recording(Vec::new());
        Tint.then(Spread).then(Tint).collect_stages(&mut recording);
        assert_eq!(
            recording.0,
            alloc::vec![("color", 0, 0), ("spatial", 1, 0), ("color", 3, 0)]
        );
    }
}
