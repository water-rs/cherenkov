//! Engine execution of composed filtrate filters and custom effects.

use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use cherenkov::RenderError;
use filtrate::{Effect, EffectContext, EffectFrameClock, EffectInput, EffectOutput, ShapeTextures};

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

pub(crate) trait Source: Send {
    fn build(self: Box<Self>) -> Box<dyn Runnable>;
}

impl<E: Effect + Send> Source for E {
    fn build(self: Box<Self>) -> Box<dyn Runnable> {
        self
    }
}

pub(crate) struct FromFilter<F>(pub F);

impl<F: filtrate_core::Filter + Send> Source for FromFilter<F> {
    fn build(self: Box<Self>) -> Box<dyn Runnable> {
        Box::new(filtrate::Executor::new(self.0))
    }
}

pub(crate) trait Runnable {
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
    clock: EffectFrameClock,
    dirty: Arc<AtomicBool>,
    again: bool,
    output: Option<(wgpu::Texture, wgpu::TextureView)>,
}

#[derive(Default)]
pub(crate) struct Registry(HashMap<u64, Entry>);

impl Registry {
    pub fn add(
        &mut self,
        id: u64,
        source: Box<dyn Source>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
    ) {
        let mut effect = source.build();
        effect
            .setup(&EffectContext {
                device,
                queue,
                input_format: format,
                output_format: format,
            })
            .unwrap_or_else(|e| panic!("filter {id} registration failed: {e}"));
        let dirty = Arc::new(AtomicBool::new(false));
        let wake = dirty.clone();
        effect.set_redraw(Arc::new(move || {
            wake.store(true, Ordering::Release);
        }));
        self.0.insert(
            id,
            Entry {
                effect,
                clock: EffectFrameClock::new(),
                dirty,
                again: false,
                output: None,
            },
        );
    }

    pub fn remove(&mut self, id: u64) {
        self.0.remove(&id);
    }

    pub fn wants_redraw(&self) -> bool {
        self.0
            .values()
            .any(|entry| entry.again || entry.dirty.load(Ordering::Acquire))
    }

    /// Runs after the capture pass and before its parent samples the scratch.
    pub fn apply(
        &mut self,
        id: u64,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scratch: &super::ScratchTarget,
        size: (u32, u32),
    ) -> Result<(), RenderError> {
        let entry = self
            .0
            .get_mut(&id)
            .ok_or_else(|| RenderError::Render(format!("unregistered filter {id}")))?;
        let format = scratch.texture.format();
        if entry
            .output
            .as_ref()
            .is_none_or(|(tex, _)| (tex.width(), tex.height()) != size)
        {
            entry.output = Some(super::create_target(
                device,
                "filter output",
                size,
                super::TARGET_USAGES,
                format,
            ));
        }
        let (texture, view) = entry.output.as_ref().expect("output allocated");
        entry.dirty.store(false, Ordering::Release);
        let input = EffectInput {
            device,
            queue,
            texture: &scratch.texture,
            view: scratch.view.clone(),
            format,
            width: size.0,
            height: size.1,
            timing: entry.clock.tick(),
            shape: ShapeTextures::default(),
        };
        let output = EffectOutput {
            device,
            queue,
            texture,
            view: view.clone(),
            format,
            width: size.0,
            height: size.1,
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine filter"),
        });
        entry.again = entry
            .effect
            .encode(&input, &output, &mut encoder)
            .map_err(|e| RenderError::Render(format!("filter {id}: {e}")))?;
        encoder.copy_texture_to_texture(
            texture.as_image_copy(),
            scratch.texture.as_image_copy(),
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        Ok(())
    }
}
