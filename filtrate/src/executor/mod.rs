//! The reference executor: runs a [`Filter`] on a wgpu texture.
//!
//! It asks the shared composer, `cherenkov-shader`, to compose the filter's
//! stages and takes the reference alternative of the composition: every
//! piece of the composition is one full-screen fragment pass, every spatial
//! stage is materialized, and no colour prefix is folded into a spatial
//! stage's samples. Intermediates are `Rgba16Float`. Stages that operate in
//! sRGB are wrapped in conversions from and back to the working space.
//!
//! It is deliberately simple: it is the behaviour the Cherenkov engine's
//! fused execution is verified against, and it serves consumers that are not
//! user interfaces, such as video processing.

extern crate alloc;

mod animation;
mod entry;
mod gpu;
mod plan;

#[cfg(test)]
mod tests;

use core::fmt;

use filtrate_core::{Chain, Filter, ParamArray, SpatialFilter};

use crate::effect::{
    Effect, EffectContext, EffectInput, EffectOutput, EffectRedrawCallback, EffectRenderError,
    EffectRenderResult, EffectSetupError, EffectSetupResult,
};
use animation::ParamAnimator;
use gpu::Gpu;

/// Runs a [`Filter`] on wgpu textures: texture in, texture of the same size
/// out.
///
/// Parameters that are reactive ([`FilterParam`](crate::FilterParam)
/// signals) are watched; a change carrying an interpolator animates, and
/// [`Effect::encode_render`] reports whether another frame is needed.
pub struct Executor<F: Filter> {
    filter: F,
    animator: ParamAnimator,
    gpu: Option<Gpu>,
    /// Sticky setup error: once set, rendering fails fast.
    setup_error: Option<EffectSetupError>,
}

impl<F: Filter> fmt::Debug for Executor<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Executor")
            .field("animator", &self.animator)
            .field("set_up", &self.gpu.is_some())
            .field("setup_error", &self.setup_error)
            .finish_non_exhaustive()
    }
}

impl<F: Filter> Executor<F> {
    /// An executor for `filter`. Pipelines are built by [`Effect::setup`].
    #[must_use]
    pub fn new(filter: F) -> Self {
        let mut targets = alloc::vec![0.0; <F::Params as ParamArray>::LEN];
        filter.params().write_to(&mut targets);
        let animator = ParamAnimator::new(targets, |installer| filter.visit_signals(installer));
        Self {
            filter,
            animator,
            gpu: None,
            setup_error: None,
        }
    }

    /// The filter this executor runs.
    #[must_use]
    pub const fn filter(&self) -> &F {
        &self.filter
    }

    /// An executor for this executor's filter followed by `filter`, keeping
    /// the installed redraw callback. The new executor needs its own setup.
    #[must_use]
    pub fn then<G: Filter>(self, filter: G) -> Executor<Chain<F, G>> {
        let redraw_callback = self.animator.redraw_callback();
        let next = Executor::new(Chain {
            first: self.filter,
            second: filter,
        });
        if let Some(callback) = redraw_callback {
            next.animator.install_redraw_callback(callback);
        }
        next
    }

    /// The largest distance, in pixels, between an output pixel and any
    /// input texel it reads, for every value the parameters take until
    /// their running animations complete.
    ///
    /// It evaluates [`SpatialFilter::footprint_of`] at every parameter's
    /// largest magnitude over its animation track (see
    /// [`AnimationTrack::magnitude_bound`](crate::AnimationTrack::magnitude_bound)),
    /// after applying the parameter changes received so far.
    pub fn footprint(&mut self) -> f32
    where
        F: SpatialFilter,
    {
        F::footprint_of(&F::Params::read_from(&self.animator.magnitude_bounds()))
    }

    /// The per-frame sampled parameter values, for test observation.
    #[cfg(test)]
    pub(crate) fn animated_values(&self) -> &[f32] {
        self.animator.current_values()
    }

    /// `setup` with every input format treated as unfilterable — tests
    /// exercise the manual-bilinear path on a device that could filter.
    #[cfg(test)]
    #[expect(
        clippy::future_not_send,
        reason = "the executor owns device-bound pipelines and is set up on the GPU host thread"
    )]
    pub(crate) async fn setup_unfilterable(
        &mut self,
        ctx: &EffectContext<'_>,
    ) -> EffectSetupResult {
        self.attach(Gpu::with_filterability(&self.filter, ctx, false, false).await)
    }

    /// Runs the built pipelines, or sticks the setup error.
    fn attach(&mut self, result: Result<Gpu, EffectSetupError>) -> EffectSetupResult {
        match result {
            Ok(gpu) => {
                self.gpu = Some(gpu);
                self.setup_error = None;
                self.animator.ensure_redraw_callback();
                self.animator.apply_targets_to_current();
                Ok(())
            }
            Err(error) => {
                tracing::error!("[filtrate] executor setup failed: {error}");
                self.gpu = None;
                self.setup_error = Some(error.clone());
                Err(error)
            }
        }
    }
}

impl<F: Filter> Effect for Executor<F> {
    fn set_redraw_callback(&mut self, callback: EffectRedrawCallback) {
        self.animator.install_redraw_callback(callback);
    }

    #[expect(
        clippy::future_not_send,
        reason = "the executor owns device-bound pipelines and is set up on the GPU host thread"
    )]
    async fn setup(&mut self, ctx: &EffectContext<'_>) -> EffectSetupResult {
        self.attach(Gpu::new(&self.filter, ctx).await)
    }

    fn encode_render(
        &mut self,
        input: &EffectInput,
        output: &EffectOutput,
        encoder: &mut wgpu::CommandEncoder,
    ) -> EffectRenderResult {
        if let Some(error) = &self.setup_error {
            return Err(EffectRenderError::SetupFailed(error.clone()));
        }
        let gpu = self.gpu.as_mut().ok_or(EffectRenderError::NotSetUp)?;
        let needs_redraw = self.animator.update(input.timing.delta());
        gpu.encode(input, output, encoder, self.animator.current_values())?;
        self.animator.mark_rendered();
        Ok(needs_redraw)
    }

    fn redraw_hint(&self) -> bool {
        self.animator.redraw_hint()
    }
}
