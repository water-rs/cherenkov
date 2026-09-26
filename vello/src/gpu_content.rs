// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU-produced layer content.
//!
//! A [`GpuContent`] implementation renders into a retained `Rgba8Unorm`
//! texture on the render thread; the engine composites that texture as an
//! image wherever the layer is drawn. Content is set up once on the first
//! frame it is drawn, re-rendered on its first frame, when
//! [`RedrawHandle::request_redraw`] was called, or when the previous frame's
//! [`Frame`](crate::interop::wgpu::Frame) requested a redraw; otherwise the
//! retained texture is reused.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::interop;

/// Content a layer draws by rendering into a GPU texture.
///
/// Implementations run on the render thread and may hold any wgpu objects
/// they create in [`GpuContent::setup`].
pub trait GpuContent: 'static {
    /// Creates GPU resources. Awaited once, on the render thread, before the
    /// first [`GpuContent::render`].
    fn setup(&mut self, gpu: &interop::wgpu::Context<'_>) -> impl Future<Output = ()>;

    /// Renders one frame into `frame`'s texture.
    ///
    /// Call [`Frame::request_redraw`](crate::interop::wgpu::Frame::request_redraw)
    /// to be re-rendered on the next frame.
    fn render(&mut self, frame: &mut interop::wgpu::Frame<'_>);

    /// Whether the produced texture is fully opaque. Informational only:
    /// the texture is always composited through vello.
    fn is_opaque(&self) -> bool {
        false
    }
}

/// A handle to GPU content created by
/// [`Engine::gpu_content`](crate::Engine::gpu_content). Attach it to a layer
/// with [`LayerEdit::content`](crate::LayerEdit::content).
pub struct GpuContentHandle {
    pub(crate) id: u64,
    /// Content size in pixels.
    pub(crate) size: (u32, u32),
    /// The redraw flag shared with the render thread.
    pub(crate) dirty: Arc<AtomicBool>,
    /// The content object, until it moves to the render thread.
    pub(crate) content: Option<Box<dyn AnyGpuContent>>,
}

impl std::fmt::Debug for GpuContentHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContentHandle")
            .field("id", &self.id)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl GpuContentHandle {
    /// A handle another thread can use to request a re-render of this
    /// content's texture.
    #[must_use]
    pub fn redraw_handle(&self) -> RedrawHandle {
        RedrawHandle(Arc::clone(&self.dirty))
    }
}

/// A shareable redraw requester for a [`GpuContentHandle`].
#[derive(Clone, Debug)]
pub struct RedrawHandle(Arc<AtomicBool>);

impl RedrawHandle {
    /// Marks the content dirty: the engine re-renders it at the next
    /// [`Engine::render`](crate::Engine::render).
    pub fn request_redraw(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether a redraw is pending.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Object-safe adapter over [`GpuContent`]: `GpuContent::setup` returns an
/// `impl Future`, which is not object safe, so the boxed form polls it with
/// `pollster` on the render thread.
pub trait AnyGpuContent: Send {
    /// Runs `setup` to completion.
    fn setup(&mut self, gpu: &interop::wgpu::Context<'_>);
    /// Renders one frame.
    fn render(&mut self, frame: &mut interop::wgpu::Frame<'_>);
    /// See [`GpuContent::is_opaque`].
    fn is_opaque(&self) -> bool;
}

impl<T: GpuContent + Send> AnyGpuContent for T {
    fn setup(&mut self, gpu: &interop::wgpu::Context<'_>) {
        pollster::block_on(GpuContent::setup(self, gpu));
    }

    fn render(&mut self, frame: &mut interop::wgpu::Frame<'_>) {
        GpuContent::render(self, frame);
    }

    fn is_opaque(&self) -> bool {
        GpuContent::is_opaque(self)
    }
}
