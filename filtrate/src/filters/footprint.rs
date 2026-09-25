//! Footprints of the built-in spatial filters, in pixels.
//!
//! Each is non-decreasing in its parameters' magnitudes, as
//! [`SpatialFilter::footprint_of`](crate::SpatialFilter::footprint_of)
//! requires, so an executor can bound an animation by evaluating it at the
//! parameters' largest magnitudes.

/// A radius the stage rounds to whole pixels and clamps at zero.
pub fn rounded(radius: f32) -> f32 {
    radius.round().max(0.0)
}

/// A radius the stage rounds to whole pixels and clamps at one.
pub fn rounded_at_least_one(radius: f32) -> f32 {
    radius.round().max(1.0)
}

/// A pixel offset read through a nearest sampler: the texel reached is at
/// most the offset rounded up.
pub fn offset(pixels: f32) -> f32 {
    pixels.abs().ceil()
}

/// `f32::INFINITY` when `amount` displaces samples by a fraction of the image
/// size, which has no bound in pixels; zero when the stage reads only its
/// own texel.
pub fn relative(amount: f32) -> f32 {
    if amount > 0.0 { f32::INFINITY } else { 0.0 }
}

/// [`Pixellate`](super::Pixellate): the cell centre is at most half a cell
/// away.
pub fn pixellate(params: &[f32; 1]) -> f32 {
    (params[0].max(1.0) * 0.5).ceil()
}

/// [`Crystallize`](super::Crystallize): the chosen seed lies in one of the
/// 3x3 neighbouring cells, jittered by at most 0.4 cells, so at most 1.9
/// cells away on either axis.
pub fn crystallize(params: &[f32; 1]) -> f32 {
    (params[0].max(1.0) * 1.9).ceil()
}

/// [`MotionBlur`](super::MotionBlur): `radius` taps each way, each filtered
/// bilinearly, so one texel further.
pub fn motion_blur(params: &[f32; 2]) -> f32 {
    let radius = rounded(params[0]);
    if radius > 0.0 { radius + 1.0 } else { 0.0 }
}

/// [`ZoomBlur`](super::ZoomBlur): taps toward the centre, a fraction of the
/// image size away.
pub fn zoom_blur(params: &[f32; 3]) -> f32 {
    // The stage reads only its own texel up to this amount.
    relative(params[0] - 0.0001)
}

/// [`EdgeWork`](super::EdgeWork): the gradient taps at `radius`, at least one.
pub fn edge_work(params: &[f32; 2]) -> f32 {
    rounded_at_least_one(params[0])
}
