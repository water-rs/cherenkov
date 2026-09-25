//! Optional CPU implementations of colour filters.

use crate::{Chain, ColorFilter, WorkingSpace};

/// A CPU implementation of a colour filter, which CPU backends run instead of
/// its shader.
///
/// A kernel must compute what the filter's stages compute: executors and the
/// correctness oracle cross-check the two.
pub trait CpuKernel: ColorFilter {
    /// Applies the filter to `pixels` in place. Pixels are premultiplied
    /// RGBA in the filter's operating space.
    fn apply_cpu(params: &Self::Params, space: &WorkingSpace, pixels: &mut [[f32; 4]]);

    /// Applies the filter with its current parameters.
    fn apply_cpu_now(&self, space: &WorkingSpace, pixels: &mut [[f32; 4]]) {
        Self::apply_cpu(&self.params(), space, pixels);
    }
}

impl<A: CpuKernel, B: CpuKernel> CpuKernel for Chain<A, B> {
    fn apply_cpu(params: &Self::Params, space: &WorkingSpace, pixels: &mut [[f32; 4]]) {
        A::apply_cpu(&params.0, space, pixels);
        B::apply_cpu(&params.1, space, pixels);
    }
}
