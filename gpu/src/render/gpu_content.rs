//! Retained producer attachments. Allocation and setup happen on first use.

use crate::interop::{
    GpuContentBox,
    wgpu::{Context, Frame},
};
use cherenkov::{Instant, RenderError, SurfaceVisibility};
use std::sync::atomic::Ordering;
use std::time::Duration;

pub struct Slot {
    content: GpuContentBox,
    /// The visibility of the surface the content is installed on: its
    /// producer's wakes are gated on it.
    surface: SurfaceVisibility,
    pub size: (u32, u32),
    pub image: Option<super::GpuImage>,
    initialized: bool,
    origin: Option<Instant>,
    last_frame: Option<Instant>,
    last_scale: Option<f32>,
    again: bool,
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.set_active(false);
    }
}

impl Slot {
    pub const fn new(content: GpuContentBox, size: (u32, u32), surface: SurfaceVisibility) -> Self {
        Self {
            content,
            surface,
            size,
            image: None,
            initialized: false,
            origin: None,
            last_frame: None,
            last_scale: None,
            again: false,
        }
    }

    /// Whether the content is composed on its surface: only then may its
    /// producer's requests wake the host, and only while the surface is
    /// visible.
    pub fn set_active(&self, active: bool) {
        if active {
            self.content
                .redraw
                .gate
                .set(std::slice::from_ref(&self.surface));
        } else {
            self.content.redraw.gate.close();
        }
    }

    pub fn resize(&mut self, size: (u32, u32)) {
        if self.size != size {
            self.size = size;
            self.image = None;
        }
    }

    pub fn wants_redraw(&self) -> bool {
        self.image.is_none() || self.again || self.content.redraw.is_dirty()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn render(
        &mut self,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        time: Instant,
        scale: f32,
    ) -> Result<(), RenderError> {
        let maximum = device.limits().max_texture_dimension_2d;
        if self.size.0 == 0 || self.size.1 == 0 || self.size.0 > maximum || self.size.1 > maximum {
            return Err(RenderError::Render(format!(
                "GPU content size {:?} must be nonzero and at most {maximum}",
                self.size
            )));
        }
        if !self.wants_redraw() && self.last_scale.map(f32::to_bits) == Some(scale.to_bits()) {
            return Ok(());
        }
        // Consume before setup/render, so an asynchronous request during either
        // remains pending and schedules another frame.
        self.content.redraw.dirty.swap(false, Ordering::AcqRel);
        if !self.initialized {
            self.content.content.setup(&Context {
                adapter,
                device,
                queue,
                format: super::TARGET_FORMAT,
                redraw: self.content.redraw.clone(),
            });
            self.initialized = true;
        }
        let image = self.image.get_or_insert_with(|| {
            let (texture, view) = super::create_target(
                device,
                "GPU content",
                self.size,
                super::TARGET_USAGES,
                super::TARGET_FORMAT,
            );
            crate::diag::create(
                device,
                "GPU content",
                u64::from(self.size.0)
                    * u64::from(self.size.1)
                    * super::texel_bytes(super::TARGET_FORMAT),
            );
            super::GpuImage {
                texture,
                view,
                width: self.size.0,
                height: self.size.1,
            }
        });
        let mut frame = Frame {
            device,
            queue,
            texture: &image.texture,
            view: &image.view,
            format: super::TARGET_FORMAT,
            width: self.size.0,
            height: self.size.1,
            scale,
            elapsed: time.saturating_duration_since(*self.origin.get_or_insert(time)),
            delta: self
                .last_frame
                .map_or(Duration::ZERO, |last| time.saturating_duration_since(last)),
            redraw: false,
        };
        self.content.content.render(&mut frame);
        self.again = frame.redraw;
        self.last_frame = Some(time);
        self.last_scale = Some(scale);
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    pub async fn render(
        &mut self,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        time: Instant,
        scale: f32,
    ) -> Result<(), RenderError> {
        let maximum = device.limits().max_texture_dimension_2d;
        if self.size.0 == 0 || self.size.1 == 0 || self.size.0 > maximum || self.size.1 > maximum {
            return Err(RenderError::Render(format!(
                "GPU content size {:?} must be nonzero and at most {maximum}",
                self.size
            )));
        }
        if !self.wants_redraw() && self.last_scale.map(f32::to_bits) == Some(scale.to_bits()) {
            return Ok(());
        }
        // Consume before setup/render, so an asynchronous request during either
        // remains pending and schedules another frame.
        self.content.redraw.dirty.swap(false, Ordering::AcqRel);
        if !self.initialized {
            self.content
                .content
                .setup(&Context {
                    adapter,
                    device,
                    queue,
                    format: super::TARGET_FORMAT,
                    redraw: self.content.redraw.clone(),
                })
                .await;
            self.initialized = true;
        }
        let image = self.image.get_or_insert_with(|| {
            let (texture, view) = super::create_target(
                device,
                "GPU content",
                self.size,
                super::TARGET_USAGES,
                super::TARGET_FORMAT,
            );
            crate::diag::create(
                device,
                "GPU content",
                u64::from(self.size.0)
                    * u64::from(self.size.1)
                    * super::texel_bytes(super::TARGET_FORMAT),
            );
            super::GpuImage {
                texture,
                view,
                width: self.size.0,
                height: self.size.1,
            }
        });
        let mut frame = Frame {
            device,
            queue,
            texture: &image.texture,
            view: &image.view,
            format: super::TARGET_FORMAT,
            width: self.size.0,
            height: self.size.1,
            scale,
            elapsed: time.saturating_duration_since(*self.origin.get_or_insert(time)),
            delta: self
                .last_frame
                .map_or(Duration::ZERO, |last| time.saturating_duration_since(last)),
            redraw: false,
        };
        self.content.content.render(&mut frame);
        self.again = frame.redraw;
        self.last_frame = Some(time);
        self.last_scale = Some(scale);
        Ok(())
    }
}
