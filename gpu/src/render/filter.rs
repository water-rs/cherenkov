// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Engine execution of composed filtrate filters and custom effects.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use std::collections::{HashMap, HashSet};

use cherenkov::RenderError;
use filtrate::{
    Effect, EffectContext, EffectFrameTiming, EffectInput, EffectOutput, ShapeTextures,
};

/// An effect moved to the render thread before its device resources exist.
pub struct EffectBox(pub(crate) Box<dyn Source>);

impl std::fmt::Debug for EffectBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EffectBox").finish_non_exhaustive()
    }
}

impl<E: Effect + cherenkov::RenderTransfer> From<E> for EffectBox {
    fn from(effect: E) -> Self {
        Self(Box::new(effect))
    }
}

pub trait Source: cherenkov::RenderTransfer {
    fn build(self: Box<Self>) -> Box<dyn Runnable>;
}

impl<E: Effect + cherenkov::RenderTransfer> Source for E {
    fn build(self: Box<Self>) -> Box<dyn Runnable> {
        self
    }
}

pub struct FromFilter<F>(pub F);

impl<F: filtrate_core::Filter + cherenkov::RenderTransfer> Source for FromFilter<F> {
    fn build(self: Box<Self>) -> Box<dyn Runnable> {
        Box::new(filtrate::Executor::new(self.0))
    }
}

/// A filter chain registered for a backdrop group capture.
pub struct FromBackdropChain<K, F>(pub F, pub std::marker::PhantomData<fn() -> K>);

impl<K, F> Source for FromBackdropChain<K, F>
where
    K: filtrate_core::kind::Kind,
    F: cherenkov::BackdropChain<K> + cherenkov::RenderTransfer,
{
    fn build(self: Box<Self>) -> Box<dyn Runnable> {
        Box::new(BackdropRunnable::<K, F>(
            filtrate::Executor::new(self.0),
            std::marker::PhantomData,
        ))
    }
}

/// An `Executor` over a backdrop chain, reporting its footprint bound.
struct BackdropRunnable<K, F: filtrate_core::Filter>(
    filtrate::Executor<F>,
    std::marker::PhantomData<fn() -> K>,
);

impl<K, F> Runnable for BackdropRunnable<K, F>
where
    K: filtrate_core::kind::Kind,
    F: cherenkov::BackdropChain<K> + cherenkov::RenderTransfer,
{
    #[cfg(not(target_arch = "wasm32"))]
    fn setup(&mut self, ctx: &EffectContext<'_>) -> Result<(), filtrate::EffectSetupError> {
        Runnable::setup(&mut self.0, ctx)
    }

    #[cfg(target_arch = "wasm32")]
    fn setup<'a>(
        &'a mut self,
        ctx: &'a EffectContext<'a>,
    ) -> core::pin::Pin<
        Box<dyn core::future::Future<Output = Result<(), filtrate::EffectSetupError>> + 'a>,
    > {
        Runnable::setup(&mut self.0, ctx)
    }

    fn encode(
        &mut self,
        input: &EffectInput<'_>,
        output: &EffectOutput<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<bool, filtrate::EffectRenderError> {
        Runnable::encode(&mut self.0, input, output, encoder)
    }

    fn set_redraw(&mut self, callback: filtrate::EffectRedrawCallback) {
        Runnable::set_redraw(&mut self.0, callback);
    }

    fn footprint_bound(&mut self) -> Option<filtrate_core::Footprint> {
        Some(<F as cherenkov::BackdropChain<K>>::footprint_bound(
            &self.0.param_bounds(),
        ))
    }
}

/// A registered filter's identity: a layer filter or a backdrop group's
/// capture chain on a surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FilterKey {
    /// A layer's filter (`FilterId`'s raw value).
    Layer(u64),
    /// A backdrop group's capture chain.
    Backdrop {
        /// The surface's raw id.
        surface: u64,
        /// The group's raw id.
        group: u64,
    },
}

pub trait Runnable {
    /// The filter's spatial footprint bound; `None` for effects without
    /// one (a colour filter contributes no reach either way).
    fn footprint_bound(&mut self) -> Option<filtrate_core::Footprint> {
        None
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn setup(&mut self, ctx: &EffectContext<'_>) -> Result<(), filtrate::EffectSetupError>;
    #[cfg(target_arch = "wasm32")]
    fn setup<'a>(
        &'a mut self,
        ctx: &'a EffectContext<'a>,
    ) -> core::pin::Pin<
        Box<dyn core::future::Future<Output = Result<(), filtrate::EffectSetupError>> + 'a>,
    >;
    fn encode(
        &mut self,
        input: &EffectInput<'_>,
        output: &EffectOutput<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<bool, filtrate::EffectRenderError>;
    fn set_redraw(&mut self, callback: filtrate::EffectRedrawCallback);
}

impl<E: Effect> Runnable for E {
    #[cfg(not(target_arch = "wasm32"))]
    fn setup(&mut self, ctx: &EffectContext<'_>) -> Result<(), filtrate::EffectSetupError> {
        pollster::block_on(Effect::setup(self, ctx))
    }

    #[cfg(target_arch = "wasm32")]
    fn setup<'a>(
        &'a mut self,
        ctx: &'a EffectContext<'a>,
    ) -> core::pin::Pin<
        Box<dyn core::future::Future<Output = Result<(), filtrate::EffectSetupError>> + 'a>,
    > {
        Box::pin(Effect::setup(self, ctx))
    }
    fn encode(
        &mut self,
        input: &EffectInput<'_>,
        output: &EffectOutput<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<bool, filtrate::EffectRenderError> {
        self.encode_render(input, output, encoder)
    }
    fn set_redraw(&mut self, callback: filtrate::EffectRedrawCallback) {
        self.set_redraw_callback(callback);
    }
}

struct Entry {
    effect: Box<dyn Runnable>,
    dirty: Arc<AtomicBool>,
    active: Arc<AtomicBool>,
    again: bool,
    sequence: Option<u64>,
    setup: Option<Result<(), String>>,
    #[cfg(target_arch = "wasm32")]
    setup_pending_frame: bool,
    input: Option<(wgpu::Texture, wgpu::TextureView)>,
    output: Option<(wgpu::Texture, wgpu::TextureView)>,
}

impl Entry {
    fn check_setup(
        &mut self,
        id: FilterKey,
        context: &EffectContext<'_>,
        format: wgpu::TextureFormat,
    ) -> Result<(), RenderError> {
        #[cfg(not(target_arch = "wasm32"))]
        let setup = self.setup.get_or_insert_with(|| {
            self.effect
                .setup(&EffectContext {
                    input_format: format,
                    output_format: format,
                    ..*context
                })
                .map_err(|error| error.to_string())
        });
        #[cfg(target_arch = "wasm32")]
        let setup = {
            let _ = (context, format);
            self.setup
                .as_ref()
                .expect("browser prepares active filters before encoding")
        };
        setup
            .as_ref()
            .copied()
            .map_err(|error| RenderError::Render(format!("filter {id:?} setup: {error}")))
    }
}

pub struct Registry {
    entries: HashMap<FilterKey, Entry>,
    host: Option<crate::interop::RedrawCallback>,
}

impl Registry {
    pub fn new(host: Option<crate::interop::RedrawCallback>) -> Self {
        Self {
            entries: HashMap::new(),
            host,
        }
    }

    pub fn add(&mut self, id: FilterKey, source: Box<dyn Source>) {
        let mut effect = source.build();
        let dirty = Arc::new(AtomicBool::new(false));
        let active = Arc::new(AtomicBool::new(false));
        let (request, attached, host) =
            (Arc::clone(&dirty), Arc::clone(&active), self.host.clone());
        effect.set_redraw(Arc::new(move || {
            if !request.swap(true, Ordering::AcqRel)
                && attached.load(Ordering::Acquire)
                && let Some(host) = &host
            {
                host.wake();
            }
        }));
        self.entries.insert(
            id,
            Entry {
                effect,
                dirty,
                active,
                again: false,
                sequence: None,
                setup: None,
                #[cfg(target_arch = "wasm32")]
                setup_pending_frame: false,
                input: None,
                output: None,
            },
        );
    }

    pub fn remove(&mut self, id: FilterKey) {
        if let Some(entry) = self.entries.remove(&id) {
            entry.active.store(false, Ordering::Release);
        }
    }

    pub fn set_active(&self, uses: &HashSet<FilterKey>) {
        for (id, entry) in &self.entries {
            entry.active.store(uses.contains(id), Ordering::Release);
        }
    }

    pub fn wants_redraw(&self, id: FilterKey) -> bool {
        self.entries
            .get(&id)
            .is_some_and(|entry| entry.again || entry.dirty.load(Ordering::Acquire))
    }

    pub fn gpu_bytes(&self) -> u64 {
        self.entries
            .values()
            .flat_map(|entry| [entry.input.as_ref(), entry.output.as_ref()])
            .flatten()
            .map(|(texture, _)| {
                u64::from(texture.width())
                    * u64::from(texture.height())
                    * if texture.format() == wgpu::TextureFormat::Rgba16Float {
                        8
                    } else {
                        4
                    }
            })
            .sum()
    }

    /// The registered filter's footprint bound; `None` when `key` is not
    /// registered (an error for callers).
    pub fn footprint_bound(&mut self, key: FilterKey) -> Option<filtrate_core::Footprint> {
        self.entries
            .get_mut(&key)
            .and_then(|entry| entry.effect.footprint_bound())
    }

    pub(super) fn apply(
        &mut self,
        id: FilterKey,
        context: &EffectContext<'_>,
        scratch: &super::ScratchTarget,
        size: (u32, u32),
        timing: EffectFrameTiming,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), RenderError> {
        let (device, queue) = (context.device, context.queue);
        let entry = self
            .entries
            .get_mut(&id)
            .ok_or_else(|| RenderError::Render(format!("unregistered filter {id:?}")))?;
        let repeated = entry.sequence == Some(timing.sequence());
        let timing = if repeated {
            EffectFrameTiming::new(
                timing.presentation_time(),
                std::time::Duration::ZERO,
                timing.sequence(),
            )
            .with_discontinuity(timing.is_discontinuity())
        } else {
            timing
        };
        if !repeated {
            #[cfg(not(target_arch = "wasm32"))]
            {
                entry.dirty.swap(false, Ordering::AcqRel);
                entry.again = false;
            }
            #[cfg(target_arch = "wasm32")]
            {
                // Browser setup consumes the initial request before awaiting.
                // A request arriving during setup still needs a second frame.
                let requested = entry.dirty.swap(false, Ordering::AcqRel);
                entry.again = std::mem::take(&mut entry.setup_pending_frame) && requested;
            }
            entry.sequence = Some(timing.sequence());
        }
        let format = scratch.texture.format();
        entry.check_setup(id, context, format)?;
        if entry
            .input
            .as_ref()
            .is_none_or(|(texture, _)| (texture.width(), texture.height()) != size)
        {
            entry.input = Some(super::create_target(
                device,
                "filter input",
                size,
                super::TARGET_USAGES,
                format,
            ));
            entry.output = Some(super::create_target(
                device,
                "filter output",
                size,
                super::TARGET_USAGES,
                format,
            ));
        }
        let (input_texture, input_view) = entry.input.as_ref().expect("input allocated");
        let (output_texture, output_view) = entry.output.as_ref().expect("output allocated");
        let extent = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        };
        encoder.copy_texture_to_texture(
            scratch.texture.as_image_copy(),
            input_texture.as_image_copy(),
            extent,
        );
        let input = EffectInput {
            device,
            queue,
            texture: input_texture,
            view: input_view.clone(),
            format,
            width: size.0,
            height: size.1,
            timing,
            shape: ShapeTextures::default(),
        };
        let output = EffectOutput {
            device,
            queue,
            texture: output_texture,
            view: output_view.clone(),
            format,
            width: size.0,
            height: size.1,
        };
        entry.again |= entry
            .effect
            .encode(&input, &output, encoder)
            .map_err(|error| RenderError::Render(format!("filter {id:?}: {error}")))?;
        encoder.copy_texture_to_texture(
            output_texture.as_image_copy(),
            scratch.texture.as_image_copy(),
            extent,
        );
        Ok(())
    }
}

impl Drop for Registry {
    fn drop(&mut self) {
        for entry in self.entries.values() {
            entry.active.store(false, Ordering::Release);
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl Registry {
    pub(super) async fn prepare(
        &mut self,
        id: FilterKey,
        context: &EffectContext<'_>,
    ) -> Result<(), RenderError> {
        let entry = self
            .entries
            .get_mut(&id)
            .ok_or_else(|| RenderError::Render(format!("unregistered filter {id:?}")))?;
        if entry.setup.is_none() {
            entry.dirty.swap(false, Ordering::AcqRel);
            entry.setup_pending_frame = true;
            entry.setup = Some(
                entry
                    .effect
                    .setup(context)
                    .await
                    .map_err(|error| error.to_string()),
            );
        }
        entry
            .setup
            .as_ref()
            .expect("setup completed")
            .as_ref()
            .copied()
            .map_err(|error| RenderError::Render(format!("filter {id:?} setup: {error}")))
    }
}
