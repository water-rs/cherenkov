// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The engine: owns the render thread and every resource's identity.
//! `!Send`, lives on the UI thread.

use super::{Waker, thread};

use crate::local::Sender;
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;

use crate::ShaderId;
use crate::backend::{Backend, Renderer};
use crate::capability::{
    Effects, ExternalFrames, Filters, GpuContent, Runs, ShaderPaint, ShaderSource, Uploads,
};
use crate::config::{MemoryUsage, Pressure};
use crate::error::{EngineError, RenderError, ResourceError, SurfaceError};
use crate::frame::{FrameStats, FrameTime, FrameTiming, Next};
use crate::glyph::FontId;
use crate::image::{Format, ImageData};
use crate::message::{ChangeSet, FontData, Message, ResOp, SurfaceId};
use crate::paint::ImageId;
use crate::resource::{Filter, Font, FontSource, Image, Shader};
use crate::style::FilterId;
use crate::surface::{ExternalFrameHandle, GpuContentHandle, Shared, Surface};

/// Owns the device and a serial executor on the creating JS thread.
///
/// `Engine`, its resources and futures are deliberately `!Send`. Operations
/// awaiting WebGPU yield without moving browser objects to another thread.
/// Native builds expose the synchronous render-thread API instead.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<cherenkov::Engine<cherenkov::testing::Null>>();
/// ```
pub struct Engine<B: Backend> {
    tx: Sender<Message<B>>,
    info: B::Info,
    stats: RefCell<FrameStats>,
    /// The live surfaces' shared queues, drained into one `Render` message
    /// per frame.
    surfaces: RefCell<Vec<std::rc::Weak<RefCell<Shared<B>>>>>,
    next_surface: Cell<u64>,
    next_font: Cell<u64>,
    next_image: Cell<u64>,
    next_shader: Cell<u64>,
    next_filter: Cell<u64>,
    /// The type-erased `Message::Resource` sender resource drops use.
    release: Rc<dyn Fn(ResOp<B>)>,
    waker: Rc<Waker>,
    // `!Send`: the engine lives on the UI thread.
    _not_send: PhantomData<Rc<()>>,
}

impl<B: Backend> std::fmt::Debug for Engine<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine").finish_non_exhaustive()
    }
}

impl<B: Backend> Engine<B> {
    /// Initializes the backend and its serial executor on the current JS thread.
    /// Await this future from the browser event loop; do not block on it.
    ///
    /// # Errors
    /// [`EngineError`] when the backend fails to initialize or the render
    /// thread cannot start.
    pub async fn new(config: B::Config) -> Result<Self, EngineError> {
        let (tx, info) = thread::local::<B>(config).await?;
        let release_tx = tx.clone();
        Ok(Self {
            tx,
            info,
            stats: RefCell::new(FrameStats::default()),
            surfaces: RefCell::new(Vec::new()),
            next_surface: Cell::new(0),
            next_font: Cell::new(1),
            next_image: Cell::new(1),
            next_shader: Cell::new(1),
            next_filter: Cell::new(1),
            release: Rc::new(move |op: ResOp<B>| {
                let _ = release_tx.send(Message::Resource(op));
            }),
            waker: Rc::new(Waker::new()),
            _not_send: PhantomData,
        })
    }

    /// The backend's provenance (`B::Info`).
    #[must_use]
    pub const fn info(&self) -> &B::Info {
        &self.info
    }

    /// Statistics of the last [`Engine::render`]. A backend whose GPU
    /// timestamps resolve after its submit reports earlier frames'
    /// timings here, once the GPU has caught up.
    #[must_use]
    pub fn stats(&self) -> FrameStats {
        self.stats.borrow().clone()
    }

    /// Waits for every submitted frame whose timing no render has reported
    /// yet, and returns those timings, oldest first — the end of a
    /// measured window, where the last frames are still on the GPU. Awaits
    /// completion without blocking JavaScript; intended for tooling. Empty when the backend reports no
    /// GPU timing or has nothing outstanding.
    ///
    /// # Errors
    /// [`RenderError::Timeout`] when the GPU does not finish in time,
    /// [`RenderError::Readback`] when a timing buffer cannot be read, and
    /// [`RenderError::Thread`] when the render thread is gone.
    pub async fn finish_timings(&self) -> Result<Vec<FrameTiming>, RenderError> {
        let (reply, rx) = crate::local::channel();
        self.tx
            .send(Message::FinishTimings { reply })
            .map_err(|_| RenderError::Thread)?;
        rx.recv().await.map_err(|_| RenderError::Thread)?
    }

    /// The engine's current memory usage.
    ///
    /// # Panics
    /// Panics if the reply channel drops without answering, which cannot
    /// happen while the render thread is alive.
    #[must_use]
    pub async fn memory(&self) -> MemoryUsage {
        let (reply, rx) = crate::local::channel();
        if self.tx.send(Message::Memory { reply }).is_err() {
            return MemoryUsage::default();
        }
        rx.recv().await.unwrap_or_default()
    }

    /// Reports system memory pressure. `Critical` drops every cache.
    pub fn trim(&self, pressure: Pressure) {
        let _ = self.tx.send(Message::Trim(pressure));
    }

    /// Registers the host wake-up callback.
    ///
    /// Changes made outside a frame (a `surface.update`, a layer drop, a
    /// bound signal firing) are queued, not sent. When the display link is
    /// paused after `Next::Idle`, the host must learn that a frame is
    /// needed: the engine calls `f` at most once between two
    /// [`Engine::render`]s, the first time something is queued.
    pub fn set_waker(&self, f: impl Fn() + 'static) {
        *self.waker.callback.borrow_mut() = Some(Box::new(f));
    }

    fn alloc(cell: &Cell<u64>) -> u64 {
        let id = cell.get();
        cell.set(id + 1);
        id
    }

    /// Registers a font.
    ///
    /// The data is checked for emptiness on the caller thread; parsing is
    /// the backend's `add_font`.
    ///
    /// # Errors
    /// [`ResourceError::Font`] for empty data or a backend-side parse
    /// failure, [`ResourceError::Lost`] when the render thread is gone.
    pub async fn font(&self, source: FontSource) -> Result<Font, ResourceError> {
        if source.data.is_empty() {
            return Err(ResourceError::Font("empty font data".into()));
        }
        let id = FontId::new(Self::alloc(&self.next_font));
        let (reply, rx) = crate::local::channel();
        let data = FontData {
            data: source.data,
            index: source.index,
        };
        (self.release)(Box::new(move |r: &mut B::Renderer| {
            let _ = reply.send(r.add_font(id, data));
        }));
        rx.recv().await.map_err(|_| ResourceError::Lost)??;
        let release = Rc::clone(&self.release);
        Ok(Font::new(id, move || {
            release(Box::new(move |r: &mut B::Renderer| r.remove_font(id)));
        }))
    }

    /// Registers an image.
    ///
    /// `image` is validated by [`ImageData::new`] before it is passed here;
    /// the backend may still reject it (format conversion failure), so the
    /// registration is reply-carrying.
    ///
    /// # Errors
    /// [`ResourceError::Image`] when the backend rejects the upload,
    /// [`ResourceError::Lost`] when the render thread is gone.
    pub async fn image<F: Format>(&self, image: ImageData<F>) -> Result<Image<F>, ResourceError>
    where
        B: Uploads<F>,
    {
        let id = ImageId::new(Self::alloc(&self.next_image));
        let (reply, rx) = crate::local::channel();
        let upload = image.into_upload();
        (self.release)(Box::new(move |r: &mut B::Renderer| {
            let _ = reply.send(r.add_image(id, upload));
        }));
        rx.recv().await.map_err(|_| ResourceError::Lost)??;
        let release = Rc::clone(&self.release);
        Ok(Image::new(id, move || {
            release(Box::new(move |r: &mut B::Renderer| r.remove_image(id)));
        }))
    }

    /// Creates a surface over `target`: an [`Offscreen`](crate::Offscreen)
    /// texture or an interop window target.
    ///
    /// # Errors
    /// [`SurfaceError`] when the backend cannot draw the target, or
    /// [`SurfaceError::Lost`] when the render thread is gone.
    pub async fn surface(&self, target: impl Into<B::Target>) -> Result<Surface<B>, SurfaceError> {
        let id = SurfaceId::new(Self::alloc(&self.next_surface));
        let (reply, rx) = crate::local::channel();
        self.tx
            .send(Message::CreateSurface {
                id,
                target: target.into(),
                reply,
            })
            .map_err(|_| SurfaceError::Lost)?;
        let info = rx.recv().await.map_err(|_| SurfaceError::Lost)??;
        let surface = Surface::new(id, info, self.tx.clone(), Rc::clone(&self.waker));
        self.surfaces
            .borrow_mut()
            .push(Rc::downgrade(&surface.shared));
        Ok(surface)
    }

    /// The number of surfaces still alive, for leak testing.
    #[doc(hidden)]
    pub fn live_surfaces(&self) -> usize {
        let mut surfaces = self.surfaces.borrow_mut();
        surfaces.retain(|weak| weak.strong_count() > 0);
        surfaces.len()
    }

    /// Applies queued commits, samples animations and renders on this JS thread.
    /// Browser work yields to the event loop. Mutations while awaiting this call
    /// belong to the next frame and still wake the host.
    ///
    /// # Errors
    /// [`RenderError`] fails this call; a surface that failed to render is
    /// left in its previous state.
    pub async fn render(&self, time: FrameTime) -> Result<Next, RenderError> {
        let mut commits: Vec<(SurfaceId, ChangeSet<B>)> = Vec::new();
        self.surfaces.borrow_mut().retain(|weak| {
            let Some(shared) = weak.upgrade() else {
                return false;
            };
            let mut shared_mut = shared.borrow_mut();
            if let Some(changes) = shared_mut.take_changes() {
                commits.push((shared_mut.id, changes));
            }
            true
        });
        // Re-arm before yielding: a signal fired during browser work must
        // request the next frame, even if the current frame returns Idle.
        self.waker.arm();
        let (reply, rx) = crate::local::channel();
        self.tx
            .send(Message::Render {
                time,
                commits,
                reply,
            })
            .map_err(|_| RenderError::Thread)?;
        let (next, stats) = rx.recv().await.map_err(|_| RenderError::Thread)??;
        *self.stats.borrow_mut() = stats;
        Ok(next)
    }

    fn on_drop(
        &self,
        op: impl FnOnce(&mut B::Renderer) + crate::RenderTransfer + 'static,
    ) -> impl FnOnce() + 'static {
        let release = Rc::clone(&self.release);
        move || release(Box::new(op))
    }
}

impl<B: ShaderPaint> Engine<B> {
    /// Registers a WGSL shader, awaiting browser compilation and validation.
    ///
    /// # Errors
    /// [`ResourceError::Shader`] when the source fails validation or
    /// pipeline creation, [`ResourceError::Lost`] when the render thread is
    /// gone.
    pub async fn shader(&self, source: ShaderSource) -> Result<Shader, ResourceError> {
        let id = ShaderId::new(Self::alloc(&self.next_shader));
        let (reply, rx) = crate::local::channel();
        self.tx
            .send(Message::AsyncResource(Box::new(move |r| {
                Box::pin(async move {
                    let _ = reply.send(B::add_shader(r, id, source).await);
                })
            })))
            .map_err(|_| ResourceError::Lost)?;
        rx.recv().await.map_err(|_| ResourceError::Lost)??;
        Ok(Shader::new(
            id,
            self.on_drop(move |r| B::remove_shader(r, id)),
        ))
    }
}

impl<B: Filters> Engine<B> {
    /// Registers a [`filtrate_core::Filter`] run on the render thread.
    #[must_use]
    pub fn filter<F: filtrate_core::Filter + crate::RenderTransfer>(&self, filter: F) -> Filter
    where
        B: Runs<F>,
    {
        let id = FilterId::new(Self::alloc(&self.next_filter));
        (self.release)(Box::new(move |r: &mut B::Renderer| {
            B::add_filter(r, id, filter);
        }));
        Filter::new(id, self.on_drop(move |r| B::remove_filter(r, id)))
    }

    /// Registers a custom effect run on the render thread.
    #[must_use]
    pub fn effect(&self, effect: impl Into<B::Effect>) -> Filter
    where
        B: Effects,
    {
        let id = FilterId::new(Self::alloc(&self.next_filter));
        let effect = effect.into();
        (self.release)(Box::new(move |r: &mut B::Renderer| {
            B::add_effect(r, id, effect);
        }));
        Filter::new(id, self.on_drop(move |r| B::remove_filter(r, id)))
    }
}

impl<B: GpuContent> Engine<B> {
    /// Creates GPU content of `size` pixels, attachable to a layer with
    /// [`LayerEdit::content`](crate::LayerEdit::content).
    #[must_use]
    pub fn gpu_content(
        &self,
        size: (u32, u32),
        content: impl Into<B::Content>,
    ) -> GpuContentHandle<B> {
        GpuContentHandle {
            size,
            content: content.into(),
        }
    }
}

impl<B: ExternalFrames> Engine<B> {
    /// Wraps an externally produced frame, attachable to a layer with
    /// [`LayerEdit::content`](crate::LayerEdit::content).
    #[must_use]
    pub fn external_frame(&self, frame: impl Into<B::Frame>) -> ExternalFrameHandle<B> {
        ExternalFrameHandle {
            frame: frame.into(),
        }
    }
}

impl<B: Backend> Drop for Engine<B> {
    fn drop(&mut self) {
        let _ = self.tx.send(Message::Shutdown);
    }
}
