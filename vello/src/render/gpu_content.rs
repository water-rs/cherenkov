// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Render-side `GpuContent`: each content object owns a retained
//! `Rgba8Unorm` texture vello composites as an image.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use vello::peniko;

use crate::interop::AnyGpuContent;
use crate::interop::wgpu::{Context, Frame};

/// One `GpuContent` attachment's render-thread state.
pub struct GpuSlot {
    /// Content size in pixels.
    pub size: (u32, u32),
    /// The shared redraw flag (`RedrawHandle`).
    pub dirty: Arc<AtomicBool>,
    /// The content object.
    pub content: Box<dyn AnyGpuContent>,
    /// Created lazily on the first drawn frame.
    pub ready: Option<GpuReady>,
    initialized: bool,
}

/// The lazily-created GPU state of a [`GpuSlot`].
pub struct GpuReady {
    /// The content texture.
    pub texture: wgpu::Texture,
    /// Its view.
    pub view: wgpu::TextureView,
    /// The vello image identity bound to the texture.
    pub image: peniko::ImageData,
    /// The instant the last frame was rendered (for `delta`).
    pub last_frame: Option<Instant>,
    /// Whether the last `render` requested another frame.
    pub wants_redraw: bool,
}

/// Content texture usages.
const CONTENT_USAGES: wgpu::TextureUsages = wgpu::TextureUsages::from_bits_retain(
    wgpu::TextureUsages::RENDER_ATTACHMENT.bits()
        | wgpu::TextureUsages::TEXTURE_BINDING.bits()
        | wgpu::TextureUsages::COPY_SRC.bits(),
);

impl GpuSlot {
    /// Wraps a [`GpuContentBox`](crate::interop::GpuContentBox)'s payload.
    pub fn new(size: (u32, u32), dirty: Arc<AtomicBool>, content: Box<dyn AnyGpuContent>) -> Self {
        Self {
            size,
            dirty,
            content,
            ready: None,
            initialized: false,
        }
    }

    /// Whether the content asked for another frame — a `RedrawHandle`
    /// request or the previous frame's `request_redraw`. Unlike
    /// [`GpuSlot::needs_render`], the not-yet-rendered state does not
    /// count: that state is already covered by the content change.
    pub fn wants_redraw(&self) -> bool {
        self.dirty.load(std::sync::atomic::Ordering::Relaxed)
            || self.ready.as_ref().is_some_and(|r| r.wants_redraw)
    }

    /// Whether the content needs re-rendering this frame: first frame, a
    /// `RedrawHandle` request, or the previous frame asked for one.
    pub fn needs_render(&self) -> bool {
        self.dirty.load(std::sync::atomic::Ordering::Relaxed)
            || self.ready.as_ref().is_none_or(|r| r.wants_redraw)
    }

    /// Consumes the pending `RedrawHandle` request flag.
    pub fn take_dirty(&self) -> bool {
        self.dirty.swap(false, std::sync::atomic::Ordering::Relaxed)
    }

    /// Ensures the texture exists and the content is set up, then re-renders
    /// it when dirty and returns the vello image to draw plus whether the
    /// content asked for another frame. `origin` is the engine start time.
    pub fn evaluate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vello: &mut vello::Renderer,
        origin: Instant,
        now: Instant,
    ) -> bool {
        let (w, h) = self.size;
        let size_changed = self
            .ready
            .as_ref()
            .is_some_and(|r| (r.texture.width(), r.texture.height()) != (w, h));
        if size_changed {
            if let Some(ready) = &self.ready {
                vello.override_image(&ready.image, None);
            }
            self.ready = None;
        }
        if !self.initialized {
            let ctx = Context {
                device,
                queue,
                format: wgpu::TextureFormat::Rgba8Unorm,
            };
            self.content.setup(&ctx);
            self.initialized = true;
        }
        if self.ready.is_none() {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gpu content"),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: CONTENT_USAGES,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let image = super::shader::texture_image(
                w.max(1),
                h.max(1),
                peniko::ImageAlphaType::AlphaPremultiplied,
            );
            self.ready = Some(GpuReady {
                texture,
                view,
                image,
                last_frame: None,
                wants_redraw: false,
            });
        }
        let wants =
            self.needs_render() || self.ready.as_ref().is_none_or(|r| r.last_frame.is_none());
        if wants {
            self.take_dirty();
            let ready = self.ready.as_mut().expect("just ensured");
            let elapsed = now.saturating_duration_since(origin);
            let delta = ready.last_frame.map_or_else(
                || Duration::from_secs_f32(1.0 / 60.0),
                |last| {
                    now.saturating_duration_since(last)
                        .min(Duration::from_millis(100))
                },
            );
            let mut frame = Frame {
                device,
                queue,
                texture: &ready.texture,
                view: &ready.view,
                format: wgpu::TextureFormat::Rgba8Unorm,
                width: w.max(1),
                height: h.max(1),
                scale: 1.0,
                elapsed,
                delta,
                redraw: false,
            };
            self.content.render(&mut frame);
            ready.wants_redraw = frame.redraw_requested();
            ready.last_frame = Some(now);
            // (Re)bind the texture and mark it fresh for the next vello
            // render so the atlas re-copies it.
            vello.override_image(
                &ready.image,
                Some(wgpu::TexelCopyTextureInfoBase {
                    texture: ready.texture.clone(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                }),
            );
            vello.mark_override_image_dirty(&ready.image);
        }
        wants && self.ready.as_ref().is_some_and(|r| r.wants_redraw)
    }

    /// Unbinds the texture from vello's atlas (layer removal/drop).
    pub fn unregister(&self, vello: &mut vello::Renderer) {
        if let Some(ready) = &self.ready {
            vello.override_image(&ready.image, None);
        }
    }
}
