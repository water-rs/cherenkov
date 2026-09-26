//! Retained textures produced by render-thread GPU content.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::interop::{
    GpuContentBox,
    wgpu::{Context, Frame},
};

pub struct Slot {
    content: GpuContentBox,
    pub image: super::GpuImage,
    last_frame: Option<Instant>,
    again: bool,
}

impl Slot {
    pub fn new(
        mut content: GpuContentBox,
        size: (u32, u32),
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        assert!(size.0 > 0 && size.1 > 0, "GPU content size must be nonzero");
        content.content.setup(&Context {
            adapter,
            device,
            queue,
            format: super::TARGET_FORMAT,
            redraw: content.redraw.clone(),
        });
        let (texture, view) = super::create_target(
            device,
            "GPU content",
            size,
            super::TARGET_USAGES,
            super::TARGET_FORMAT,
        );
        Self {
            content,
            image: super::GpuImage {
                texture,
                view,
                width: size.0,
                height: size.1,
            },
            last_frame: None,
            again: false,
        }
    }

    pub fn wants_redraw(&self) -> bool {
        self.again || self.content.redraw.is_dirty()
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        origin: Instant,
        time: Instant,
        scale: f32,
    ) {
        if self.last_frame.is_some() && !self.wants_redraw() {
            return;
        }
        self.content.redraw.dirty.swap(false, Ordering::AcqRel);
        let mut frame = Frame {
            device,
            queue,
            texture: &self.image.texture,
            view: &self.image.view,
            format: super::TARGET_FORMAT,
            width: self.image.width,
            height: self.image.height,
            scale,
            elapsed: time.saturating_duration_since(origin),
            delta: self
                .last_frame
                .map_or(Duration::ZERO, |last| time.saturating_duration_since(last)),
            redraw: false,
        };
        self.content.content.render(&mut frame);
        self.again = frame.redraw;
        self.last_frame = Some(time);
    }
}
