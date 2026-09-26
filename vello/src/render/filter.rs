// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Layer filters: erased `filtrate` effects run over a captured texture of
//! the layer's subtree.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use filtrate::{
    Effect, EffectContext, EffectFrameClock, EffectInput, EffectOutput, Filter, ShapeTextures,
};
use vello::peniko;

use cherenkov::RenderError;

/// A filter or effect crossing to the render thread. `filtrate::Executor`
/// is `!Send` (`WatchGuard` boxes `dyn Any`), so the message carries the
/// raw `Filter`/`Effect` and the executor is built render-side.
pub trait FilterSource: Send {
    /// Builds the runnable effect on the render thread.
    fn build(self: Box<Self>) -> Box<dyn ErasedEffect>;
}

/// A `filtrate::Filter`, built into an [`filtrate::Executor`].
pub struct FromFilter<F: Filter + Send>(pub F);

impl<F: Filter + Send> FilterSource for FromFilter<F> {
    fn build(self: Box<Self>) -> Box<dyn ErasedEffect> {
        Box::new(filtrate::Executor::new(self.0))
    }
}

/// A custom `filtrate::Effect`, used directly.
pub struct FromEffect<E: Effect + Send>(pub E);

impl<E: Effect + Send> FilterSource for FromEffect<E> {
    fn build(self: Box<Self>) -> Box<dyn ErasedEffect> {
        Box::new(self.0)
    }
}

/// Object-safe adapter over [`filtrate::Effect`]: `Effect::setup` returns
/// an `impl Future`, so the boxed form blocks on it with `pollster` on the
/// render thread. Errors surface as strings into [`RenderError::Render`].
///
/// Lives on the render thread only; it needs no `Send` bound because
/// `filtrate::Executor` is `!Send`.
pub trait ErasedEffect {
    /// Runs `setup` to completion (once, on the render thread).
    fn setup(&mut self, ctx: &EffectContext<'_>) -> Result<(), String>;
    /// Encodes one frame of effect work.
    fn encode_render(
        &mut self,
        input: &EffectInput<'_>,
        output: &EffectOutput<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<bool, String>;
    /// See [`Effect::redraw_hint`].
    fn redraw_hint(&self) -> bool;
    /// See [`Effect::set_redraw_callback`].
    fn set_redraw_callback(&mut self, callback: filtrate::EffectRedrawCallback);
}

impl<E: Effect> ErasedEffect for E {
    fn setup(&mut self, ctx: &EffectContext<'_>) -> Result<(), String> {
        pollster::block_on(Effect::setup(self, ctx)).map_err(|e| e.to_string())
    }

    fn encode_render(
        &mut self,
        input: &EffectInput<'_>,
        output: &EffectOutput<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<bool, String> {
        Effect::encode_render(self, input, output, encoder).map_err(|e| e.to_string())
    }

    fn redraw_hint(&self) -> bool {
        Effect::redraw_hint(self)
    }

    fn set_redraw_callback(&mut self, callback: filtrate::EffectRedrawCallback) {
        Effect::set_redraw_callback(self, callback);
    }
}

/// A registered filter's render-thread state.
pub struct FilterEntry {
    effect: Box<dyn ErasedEffect>,
    clock: EffectFrameClock,
    /// Set when the effect has been set up on this device.
    set_up: bool,
    /// Sticky setup failure.
    setup_error: Option<String>,
    /// Set by the effect's redraw callback.
    pub redraw: Arc<AtomicBool>,
    /// Capture texture (the layer subtree) at its current size.
    capture: Option<(wgpu::Texture, wgpu::TextureView, (u32, u32))>,
    /// Output texture and its vello image identity.
    output: Option<(wgpu::Texture, wgpu::TextureView, peniko::ImageData)>,
}

/// The filter registry, keyed by `FilterId::raw`.
#[derive(Default)]
pub struct FilterRegistry {
    entries: HashMap<u64, FilterEntry>,
}

/// Capture/output texture usages: the capture is rendered into and then
/// sampled; the output is a render target vello copies into its atlas.
const FILTER_USAGES: wgpu::TextureUsages = wgpu::TextureUsages::from_bits_retain(
    wgpu::TextureUsages::RENDER_ATTACHMENT.bits()
        | wgpu::TextureUsages::TEXTURE_BINDING.bits()
        | wgpu::TextureUsages::STORAGE_BINDING.bits()
        | wgpu::TextureUsages::COPY_SRC.bits()
        | wgpu::TextureUsages::COPY_DST.bits(),
);

fn filter_texture(
    device: &wgpu::Device,
    label: &'static str,
    size: (u32, u32),
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0.max(1),
            height: size.1.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: FILTER_USAGES,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

impl FilterRegistry {
    /// Registers `source` under `id`, building the effect and installing
    /// a redraw callback that flags every surface dirty on the next frame.
    pub fn add(&mut self, id: u64, source: Box<dyn FilterSource>) {
        let mut effect = source.build();
        let redraw = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&redraw);
        effect.set_redraw_callback(Arc::new(move || {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }));
        self.entries.insert(
            id,
            FilterEntry {
                effect,
                clock: EffectFrameClock::new(),
                set_up: false,
                setup_error: None,
                redraw,
                capture: None,
                output: None,
            },
        );
    }

    /// Drops the filter, returning the output image's identity when one
    /// was produced (the caller unbinds its vello override).
    pub fn remove(&mut self, id: u64) -> Option<peniko::ImageData> {
        self.entries
            .remove(&id)
            .and_then(|entry| entry.output.map(|(.., image)| image))
    }

    /// Whether any filter's redraw callback fired since the last check,
    /// clearing the flags.
    pub fn take_redraw_requests(&self) -> bool {
        self.entries
            .values()
            .any(|e| e.redraw.swap(false, std::sync::atomic::Ordering::Relaxed))
    }

    /// Whether `id` wants another frame (`redraw_hint`).
    pub fn redraw_hint(&self, id: u64) -> bool {
        self.entries
            .get(&id)
            .is_some_and(|e| e.effect.redraw_hint())
    }

    /// Ensures `id`'s capture texture exists at `size`, returning its view
    /// (the texture the layer subtree renders into).
    pub fn ensure_capture(
        &mut self,
        device: &wgpu::Device,
        id: u64,
        size: (u32, u32),
    ) -> Result<wgpu::TextureView, RenderError> {
        let Some(entry) = self.entries.get_mut(&id) else {
            return Err(RenderError::Render(format!("unregistered filter {id}")));
        };
        let stale = entry.capture.as_ref().is_none_or(|(.., s)| *s != size);
        if stale {
            let (texture, view) = filter_texture(device, "filter capture", size);
            entry.capture = Some((texture, view, size));
        }
        Ok(entry.capture.as_ref().expect("just ensured").1.clone())
    }

    /// The output texture vello binds as an image, when produced.
    pub fn output_texture(&self, id: u64) -> Option<wgpu::Texture> {
        self.entries
            .get(&id)
            .and_then(|e| e.output.as_ref().map(|(t, ..)| t.clone()))
    }

    /// The image identity of `id`'s output texture, when produced — for
    /// unbinding the override when a referencing layer is torn down.
    pub fn output_image(&self, id: u64) -> Option<peniko::ImageData> {
        self.entries
            .get(&id)
            .and_then(|e| e.output.as_ref().map(|(.., i)| i.clone()))
    }

    /// Runs `id`'s effect over its capture texture, producing the output
    /// texture's view and its vello image identity, plus whether the
    /// effect asked for another frame. When the capture resizes, the old
    /// output image's override is unbound from `vello`.
    ///
    /// The effect's setup runs once (blocking) with
    /// `EffectContext { Rgba8Unorm, Rgba8Unorm }`.
    pub fn evaluate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vello: &mut vello::Renderer,
        id: u64,
        size: (u32, u32),
    ) -> Result<(wgpu::TextureView, peniko::ImageData, bool), RenderError> {
        let Some(entry) = self.entries.get_mut(&id) else {
            return Err(RenderError::Render(format!("unregistered filter {id}")));
        };
        if let Some(error) = &entry.setup_error {
            return Err(RenderError::Render(format!("filter: {error}")));
        }
        if !entry.set_up {
            let ctx = EffectContext {
                device,
                queue,
                input_format: wgpu::TextureFormat::Rgba8Unorm,
                output_format: wgpu::TextureFormat::Rgba8Unorm,
            };
            if let Err(error) = entry.effect.setup(&ctx) {
                entry.setup_error = Some(error.clone());
                return Err(RenderError::Render(format!("filter: {error}")));
            }
            entry.set_up = true;
        }
        let resize = entry
            .output
            .as_ref()
            .is_none_or(|(tex, ..)| (tex.width(), tex.height()) != size);
        if resize {
            let (texture, view) = filter_texture(device, "filter output", size);
            let image = super::shader::texture_image(
                size.0.max(1),
                size.1.max(1),
                peniko::ImageAlphaType::AlphaPremultiplied,
            );
            if let Some((.., old_image)) = entry.output.replace((texture, view, image)) {
                vello.override_image(&old_image, None);
            }
        }
        let (cap_texture, cap_view, _) = entry.capture.as_ref().expect("ensure_capture first");
        let cap_texture = cap_texture.clone();
        let cap_view = cap_view.clone();
        let (out_texture, out_view, out_image) = entry.output.as_mut().expect("just ensured");
        let input = EffectInput {
            device,
            queue,
            texture: &cap_texture,
            view: cap_view,
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: size.0.max(1),
            height: size.1.max(1),
            timing: entry.clock.tick(),
            shape: ShapeTextures::default(),
        };
        let output = EffectOutput {
            device,
            queue,
            texture: out_texture,
            view: out_view.clone(),
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: size.0.max(1),
            height: size.1.max(1),
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("filter effect"),
        });
        let again = entry
            .effect
            .encode_render(&input, &output, &mut encoder)
            .map_err(|e| RenderError::Render(format!("filter: {e}")))?;
        queue.submit([encoder.finish()]);
        Ok((out_view.clone(), out_image.clone(), again))
    }
}
