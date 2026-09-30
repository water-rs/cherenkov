// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The engine: owns the render thread and every resource's identity.
//! `!Send`, lives on the UI thread.

use super::{Waker, thread};

use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, SyncSender};

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
use crate::message::{ChangeSet, FontData, MemoryReply, Message, RenderReply, ResOp, SurfaceId};
use crate::paint::ImageId;
use crate::resource::{Filter, Font, FontSource, Image, ReplaceImage, Shader};
use crate::style::FilterId;
use crate::surface::{ExternalFrameHandle, GpuContentHandle, Shared, Surface};

/// The engine: owns the device and the render thread. `!Send`, lives on
/// the UI thread.
///
/// The render thread owns every backend object. `Engine` is deliberately
/// `!Send` — everything it hands out (`Surface`, `Layer`, resource
/// handles) may only live on the UI thread that created the engine.
///
/// The native UI-to-render message queue is bounded to 64 messages. If the
/// UI thread gets more than 64 messages ahead, it waits for the render thread.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<cherenkov::Engine<cherenkov::testing::Null>>();
/// ```
pub struct Engine<B: Backend> {
    tx: SyncSender<Message<B>>,
    info: B::Info,
    stats: RefCell<FrameStats>,
    render_reply: RefCell<Option<SyncSender<RenderReply<B>>>>,
    render_reply_rx: Receiver<RenderReply<B>>,
    memory_reply: RefCell<Option<SyncSender<MemoryReply>>>,
    memory_reply_rx: Receiver<MemoryReply>,
    commits: RefCell<Vec<(SurfaceId, ChangeSet<B>)>>,
    /// The live surfaces' shared queues, drained into one `Render` message
    /// per frame.
    surfaces: RefCell<Vec<std::rc::Weak<RefCell<Shared<B>>>>>,
    next_surface: Cell<u64>,
    next_font: Cell<u64>,
    next_image: Cell<u64>,
    next_shader: Cell<u64>,
    next_filter: Cell<u64>,
    next_backdrop_shader: Cell<u64>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// The type-erased `Message::Resource` sender resource drops use.
    release: Rc<dyn Fn(ResOp<B>)>,
    /// The `Message::ReplaceImage` round trip every image handle shares.
    replace_image: ReplaceImage,
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
    /// Spawns the render thread and runs `B::init` on it, blocking until it
    /// reports success or failure.
    ///
    /// # Errors
    /// [`EngineError`] when the backend fails to initialize or the render
    /// thread cannot start.
    pub fn new(config: B::Config) -> Result<Self, EngineError> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Message<B>>(64);
        let (init_tx, init_rx) = std::sync::mpsc::channel();
        let (render_reply, render_reply_rx) = std::sync::mpsc::sync_channel(1);
        let (memory_reply, memory_reply_rx) = std::sync::mpsc::sync_channel(1);
        let render_thread = std::thread::Builder::new()
            .name("cherenkov-render".into())
            .spawn(move || thread::run::<B>(config, &rx, &init_tx))
            .map_err(|e| EngineError::Thread(format!("spawn failed: {e}")))?;
        let info = init_rx
            .recv()
            .map_err(|_| EngineError::Thread("render thread died during init".into()))??;
        let release_tx = tx.clone();
        let waker = Rc::new(Waker::new());
        let replace_image = {
            let tx = tx.clone();
            let waker = Rc::clone(&waker);
            Rc::new(move |id, image| {
                let (reply, rx) = std::sync::mpsc::channel();
                tx.send(Message::ReplaceImage { id, image, reply })
                    .map_err(|_| ResourceError::Lost)?;
                // A surface drawing the image was marked changed: a host
                // paused after `Next::Idle` needs a frame to show it.
                if rx.recv().map_err(|_| ResourceError::Lost)?? {
                    waker.wake();
                }
                Ok(())
            }) as ReplaceImage
        };
        Ok(Self {
            tx,
            info,
            stats: RefCell::new(FrameStats::default()),
            render_reply: RefCell::new(Some(render_reply)),
            render_reply_rx,
            memory_reply: RefCell::new(Some(memory_reply)),
            memory_reply_rx,
            commits: RefCell::new(Vec::new()),
            surfaces: RefCell::new(Vec::new()),
            next_surface: Cell::new(0),
            next_font: Cell::new(1),
            next_image: Cell::new(1),
            next_shader: Cell::new(1),
            next_filter: Cell::new(1),
            next_backdrop_shader: Cell::new(1),
            thread: Some(render_thread),
            release: Rc::new(move |op: ResOp<B>| {
                let _ = release_tx.send(Message::Resource(op));
            }),
            replace_image,
            waker,
            _not_send: PhantomData,
        })
    }

    /// The backend's provenance (`B::Info`).
    #[must_use]
    pub const fn info(&self) -> &B::Info {
        &self.info
    }

    /// Statistics of the last [`Engine::render`]. GPU timings are kept by
    /// the render thread and returned by [`Engine::finish_timings`] instead.
    #[must_use]
    pub fn stats(&self) -> FrameStats {
        self.stats.borrow().clone()
    }

    /// Returns all GPU timings accumulated since the previous call,
    /// oldest first. Blocks the render thread until the GPU finishes frames
    /// still in flight, so this marks the end of a measured window and
    /// belongs to tooling, never to a frame path. Empty when the backend
    /// reports no GPU timing or has nothing outstanding.
    ///
    /// # Errors
    /// [`RenderError::Timeout`] when the GPU does not finish in time,
    /// [`RenderError::Readback`] when a timing buffer cannot be read, and
    /// [`RenderError::Thread`] when the render thread is gone.
    pub fn finish_timings(&self) -> Result<Vec<FrameTiming>, RenderError> {
        let (reply, rx) = std::sync::mpsc::channel();
        self.tx
            .send(Message::FinishTimings { reply })
            .map_err(|_| RenderError::Thread)?;
        rx.recv().map_err(|_| RenderError::Thread)?
    }

    /// The engine's current memory usage.
    ///
    #[must_use]
    pub fn memory(&self) -> MemoryUsage {
        let Some(reply_sender) = self.memory_reply.borrow_mut().take() else {
            return MemoryUsage::default();
        };
        if let Err(error) = self.tx.send(Message::Memory {
            reply: reply_sender,
        }) {
            if let Message::Memory { reply } = error.0 {
                *self.memory_reply.borrow_mut() = Some(reply);
            }
            return MemoryUsage::default();
        }
        let Ok(reply) = self.memory_reply_rx.recv() else {
            return MemoryUsage::default();
        };
        *self.memory_reply.borrow_mut() = Some(reply.sender);
        reply.usage
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
    pub fn font(&self, source: FontSource) -> Result<Font, ResourceError> {
        if source.data.is_empty() {
            return Err(ResourceError::Font("empty font data".into()));
        }
        let id = FontId::new(Self::alloc(&self.next_font));
        let (reply, rx) = std::sync::mpsc::channel();
        let data = FontData {
            data: source.data,
            index: source.index,
        };
        (self.release)(Box::new(move |r: &mut B::Renderer| {
            let _ = reply.send(r.add_font(id, data));
        }));
        rx.recv().map_err(|_| ResourceError::Lost)??;
        let release = Rc::clone(&self.release);
        Ok(Font::new(id, move || {
            release(Box::new(move |r: &mut B::Renderer| r.remove_font(id)));
        }))
    }

    /// Registers an image. [`Image::replace`] later swaps its pixels
    /// behind the same id.
    ///
    /// `image` is validated by [`ImageData::new`] before it is passed here;
    /// the backend may still reject it (format conversion failure), so the
    /// registration is reply-carrying.
    ///
    /// # Errors
    /// [`ResourceError::Image`] when the backend rejects the upload,
    /// [`ResourceError::Lost`] when the render thread is gone.
    pub fn image<F: Format>(&self, image: ImageData<F>) -> Result<Image<F>, ResourceError>
    where
        B: Uploads<F>,
    {
        let id = ImageId::new(Self::alloc(&self.next_image));
        let (reply, rx) = std::sync::mpsc::channel();
        let upload = image.into_upload();
        (self.release)(Box::new(move |r: &mut B::Renderer| {
            let _ = reply.send(r.add_image(id, upload));
        }));
        rx.recv().map_err(|_| ResourceError::Lost)??;
        let release = Rc::clone(&self.release);
        Ok(Image::new(id, Rc::clone(&self.replace_image), move || {
            release(Box::new(move |r: &mut B::Renderer| r.remove_image(id)));
        }))
    }

    /// Creates a surface over `target`: an [`Offscreen`](crate::Offscreen)
    /// texture or an interop window target.
    ///
    /// # Errors
    /// [`SurfaceError`] when the backend cannot draw the target, or
    /// [`SurfaceError::Lost`] when the render thread is gone.
    pub fn surface(&self, target: impl Into<B::Target>) -> Result<Surface<B>, SurfaceError> {
        let id = SurfaceId::new(Self::alloc(&self.next_surface));
        let (reply, rx) = std::sync::mpsc::channel();
        self.tx
            .send(Message::CreateSurface {
                id,
                target: target.into(),
                reply,
            })
            .map_err(|_| SurfaceError::Lost)?;
        let info = rx.recv().map_err(|_| SurfaceError::Lost)??;
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

    /// Renders every dirty surface for the frame at `time`, blocking until
    /// the render thread has applied the queued commits, sampled the
    /// animations and rendered.
    ///
    /// # Errors
    /// [`RenderError`] fails this call; a surface that failed to render is
    /// left in its previous state.
    pub fn render(&self, time: FrameTime) -> Result<Next, RenderError> {
        let Some(reply_sender) = self.render_reply.borrow_mut().take() else {
            return Err(RenderError::Thread);
        };
        let mut commits = std::mem::take(&mut *self.commits.borrow_mut());
        commits.clear();
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
        if let Err(error) = self.tx.send(Message::Render {
            time,
            commits,
            reply: reply_sender,
        }) {
            if let Message::Render { commits, reply, .. } = error.0 {
                *self.commits.borrow_mut() = commits;
                *self.render_reply.borrow_mut() = Some(reply);
            }
            return Err(RenderError::Thread);
        }
        let mut reply = self
            .render_reply_rx
            .recv()
            .map_err(|_| RenderError::Thread)?;
        *self.render_reply.borrow_mut() = Some(reply.sender);
        self.recycle_commits(&mut reply.commits);
        reply.commits.clear();
        *self.commits.borrow_mut() = reply.commits;
        let (next, stats) = reply.result?;
        *self.stats.borrow_mut() = stats;
        self.waker.arm();
        Ok(next)
    }

    fn recycle_commits(&self, commits: &mut [(SurfaceId, ChangeSet<B>)]) {
        let surfaces = self.surfaces.borrow();
        for (id, changes) in commits {
            let Some(shared) = surfaces
                .iter()
                .filter_map(std::rc::Weak::upgrade)
                .find(|shared| shared.borrow().id == *id)
            else {
                continue;
            };
            shared
                .borrow_mut()
                .recycle(std::mem::take(&mut changes.ops), &mut changes.recycled);
        }
    }

    fn on_drop(
        &self,
        op: impl FnOnce(&mut B::Renderer) + Send + 'static,
    ) -> impl FnOnce() + 'static {
        let release = Rc::clone(&self.release);
        move || release(Box::new(op))
    }
}

impl<B: ShaderPaint> Engine<B> {
    /// Registers a WGSL shader, blocking until the render thread has
    /// compiled and validated it.
    ///
    /// # Errors
    /// [`ResourceError::Shader`] when the source fails validation or
    /// pipeline creation, [`ResourceError::Lost`] when the render thread is
    /// gone.
    pub fn shader(&self, source: ShaderSource) -> Result<Shader, ResourceError> {
        let id = ShaderId::new(Self::alloc(&self.next_shader));
        let (reply, rx) = std::sync::mpsc::channel();
        (self.release)(Box::new(move |r: &mut B::Renderer| {
            let _ = reply.send(B::add_shader(r, id, source));
        }));
        rx.recv().map_err(|_| ResourceError::Lost)??;
        Ok(Shader::new(
            id,
            self.on_drop(move |r| B::remove_shader(r, id)),
        ))
    }
}

impl<B: crate::BackdropShaders> Engine<B> {
    /// Registers a backdrop effect shader, blocking until the render
    /// thread has compiled and validated it.
    ///
    /// # Errors
    /// [`ResourceError::Shader`] when the source fails validation, its
    /// `reach` is not a finite non-negative number, or pipeline creation
    /// fails; [`ResourceError::Lost`] when the render thread is gone.
    pub fn backdrop_shader(
        &self,
        source: crate::BackdropShaderSource,
    ) -> Result<crate::BackdropShader, ResourceError> {
        if !(source.reach.is_finite() && source.reach >= 0.0) {
            return Err(ResourceError::Shader(
                "backdrop shader reach must be a finite non-negative number".into(),
            ));
        }
        let id = crate::BackdropShaderId::new(Self::alloc(&self.next_backdrop_shader));
        let reach = source.reach;
        let (reply, rx) = std::sync::mpsc::channel();
        (self.release)(Box::new(move |r: &mut B::Renderer| {
            let _ = reply.send(B::add_backdrop_shader(r, id, source));
        }));
        rx.recv().map_err(|_| ResourceError::Lost)??;
        Ok(crate::BackdropShader::new(
            id,
            reach,
            self.on_drop(move |r| B::remove_backdrop_shader(r, id)),
        ))
    }
}

impl<B: Filters> Engine<B> {
    /// Registers a [`filtrate_core::Filter`] run on the render thread.
    #[must_use]
    pub fn filter<F: filtrate_core::Filter + Send>(&self, filter: F) -> Filter
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
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
