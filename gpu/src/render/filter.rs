// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Engine execution of composed filtrate filters and custom effects.

use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

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

impl<E: Effect + Send> From<E> for EffectBox {
    fn from(effect: E) -> Self {
        Self(Box::new(effect))
    }
}

pub trait Source: Send {
    fn build(self: Box<Self>) -> Box<dyn Runnable>;
}

impl<E: Effect + Send> Source for E {
    fn build(self: Box<Self>) -> Box<dyn Runnable> {
        self
    }
}

pub struct FromFilter<F>(pub F);

impl<F: filtrate_core::Filter + Send> Source for FromFilter<F> {
    fn build(self: Box<Self>) -> Box<dyn Runnable> {
        Box::new(filtrate::Executor::new(self.0))
    }
}

pub trait Runnable {
    fn setup(&mut self, ctx: &EffectContext<'_>) -> Result<(), filtrate::EffectSetupError>;
    fn encode(
        &mut self,
        input: &EffectInput<'_>,
        output: &EffectOutput<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<bool, filtrate::EffectRenderError>;
    fn set_redraw(&mut self, callback: filtrate::EffectRedrawCallback);
}

impl<E: Effect> Runnable for E {
    fn setup(&mut self, ctx: &EffectContext<'_>) -> Result<(), filtrate::EffectSetupError> {
        pollster::block_on(Effect::setup(self, ctx))
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
    input: Option<(wgpu::Texture, wgpu::TextureView)>,
    output: Option<(wgpu::Texture, wgpu::TextureView)>,
}

pub struct Registry {
    entries: HashMap<u64, Entry>,
    host: Option<crate::interop::RedrawCallback>,
}

impl Registry {
    pub fn new(host: Option<crate::interop::RedrawCallback>) -> Self {
        Self {
            entries: HashMap::new(),
            host,
        }
    }

    pub fn add(&mut self, id: u64, source: Box<dyn Source>) {
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
                input: None,
                output: None,
            },
        );
    }

    pub fn remove(&mut self, id: u64) {
        if let Some(entry) = self.entries.remove(&id) {
            entry.active.store(false, Ordering::Release);
        }
    }

    pub fn set_active(&self, uses: &std::collections::HashSet<u64>) {
        for (id, entry) in &self.entries {
            entry.active.store(uses.contains(id), Ordering::Release);
        }
    }

    pub fn wants_redraw(&self, id: u64) -> bool {
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

    pub(super) fn apply(
        &mut self,
        id: u64,
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
            .ok_or_else(|| RenderError::Render(format!("unregistered filter {id}")))?;
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
            entry.dirty.swap(false, Ordering::AcqRel);
            entry.again = false;
            entry.sequence = Some(timing.sequence());
        }
        let format = scratch.texture.format();
        let setup = entry.setup.get_or_insert_with(|| {
            entry
                .effect
                .setup(&EffectContext {
                    device,
                    queue,
                    input_format: format,
                    output_format: format,
                })
                .map_err(|error| error.to_string())
        });
        if let Err(error) = setup {
            return Err(RenderError::Render(format!("filter {id} setup: {error}")));
        }
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
            .map_err(|error| RenderError::Render(format!("filter {id}: {error}")))?;
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
