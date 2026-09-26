// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cherenkov-gpu`: the wgpu backend for the Cherenkov 2D rendering engine.
//!
//! The engine owns a render thread which holds every GPU object; the UI
//! thread owns the [`Engine`], [`Surface`]s and [`Layer`]s and talks to it
//! over owned messages. There are no locks: `Engine`, `Surface` and `Layer`
//! are `!Send` and everything crossing the channel is `Send`.
//!
//! ```no_run
//! use cherenkov::{Draw, WorkingColor};
//! use cherenkov::kurbo::Rect;
//! use cherenkov_gpu::{Engine, Gpu, GpuConfig, Offscreen, OffscreenFormat};
//!
//! let engine = Engine::<Gpu>::new(GpuConfig::default())?;
//! let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
//! surface.update(|tx| {
//!     tx[surface.root()].content(
//!         surface.record(|c| c.fill(Rect::new(0., 0., 64., 64.), WorkingColor::WHITE)),
//!     );
//! });
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod config;
mod error;
mod font;
mod message;
mod render;
mod surface;

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::{Rc, Weak};
use std::sync::mpsc::Sender;

pub use crate::config::{Budget, Bytes, GpuConfig, GpuInfo, MemoryUsage, Pressure, ScratchFormat};
pub use crate::error::{EngineError, RenderError, ResourceError, SurfaceError, Unsupported};
pub use crate::font::{Font, FontSource};
pub use crate::surface::{
    FrameStats, FrameTime, Layer, LayerContent, LayerEdit, Next, Offscreen, OffscreenFormat,
    PassTiming, Readback, RefreshRange, Surface, Transaction,
};

use crate::message::{Message, SurfaceId};
use crate::surface::SurfaceShared;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Gpu {}
}

/// A rendering backend. Sealed: only [`Gpu`] exists in this slice.
pub trait Backend: sealed::Sealed + 'static {
    /// The backend's configuration type.
    type Config;
}

/// The wgpu backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct Gpu;

impl Backend for Gpu {
    type Config = GpuConfig;
}

/// The engine: owns the device and the render thread. `!Send`, lives on the
/// UI thread.
#[derive(Debug)]
pub struct Engine<B: Backend> {
    tx: Sender<Message>,
    info: GpuInfo,
    stats: RefCell<FrameStats>,
    /// Weak handles to live surfaces, purged of dead entries on every use.
    surfaces: RefCell<HashMap<SurfaceId, Weak<RefCell<SurfaceShared>>>>,
    next_surface: Cell<SurfaceId>,
    next_font: Cell<u64>,
    thread: Option<std::thread::JoinHandle<()>>,
    _backend: PhantomData<B>,
    // `!Send`: the engine lives on the UI thread.
    _not_send: PhantomData<Rc<()>>,
}

impl Engine<Gpu> {
    /// Creates the device and precompiles the closed pipeline set on the
    /// render thread, blocking until it reports success or failure.
    ///
    /// # Errors
    /// [`EngineError::NoAdapter`] when no suitable adapter exists,
    /// [`EngineError::RequestDevice`] when device creation fails,
    /// [`EngineError::Shader`] when the pipeline fails validation and
    /// [`EngineError::Thread`] when the render thread dies.
    pub fn new(config: GpuConfig) -> Result<Self, EngineError> {
        let (tx, rx) = std::sync::mpsc::channel::<Message>();
        let (init_tx, init_rx) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("cherenkov-render".into())
            .spawn(move || render::run(config, rx, init_tx))
            .map_err(|e| EngineError::Thread(format!("spawn failed: {e}")))?;
        let init = init_rx
            .recv()
            .map_err(|_| EngineError::Thread("render thread died during init".into()))??;
        Ok(Self {
            tx,
            info: init.info,
            stats: RefCell::new(FrameStats::default()),
            surfaces: RefCell::new(HashMap::new()),
            next_surface: Cell::new(0),
            next_font: Cell::new(1),
            thread: Some(thread),
            _backend: PhantomData,
            _not_send: PhantomData,
        })
    }

    /// Adapter information for provenance.
    #[must_use]
    pub const fn info(&self) -> &GpuInfo {
        &self.info
    }

    /// Reports system memory pressure. `Critical` clears the glyph atlas;
    /// `Moderate` currently does nothing.
    ///
    /// # Errors
    /// [`EngineError::Thread`] when the render thread is gone.
    pub fn trim(&self, pressure: Pressure) -> Result<(), EngineError> {
        self.tx
            .send(Message::Trim(pressure))
            .map_err(|_| EngineError::Thread("render thread gone".into()))
    }

    /// The engine's current memory usage.
    ///
    /// # Panics
    /// Panics if the render thread has stopped, which cannot happen while an
    /// `Engine` is alive.
    #[must_use]
    pub fn memory(&self) -> MemoryUsage {
        let (reply, rx) = std::sync::mpsc::channel();
        self.tx
            .send(Message::Memory { reply })
            .expect("render thread alive");
        rx.recv().expect("render thread alive")
    }

    /// Registers a font.
    ///
    /// The data is parsed on the caller thread to reject invalid data and
    /// colour fonts (`COLR`, `CBDT` or `sbix` tables), then handed to the
    /// render thread.
    ///
    /// # Errors
    /// [`ResourceError::Font`] for unparseable data and
    /// `ResourceError::Unsupported(Unsupported::ColorFont)` for colour
    /// fonts.
    pub fn font(&self, source: FontSource) -> Result<Font, ResourceError> {
        font::validate_font(&source.data, source.index)?;
        let id = self.next_font.get();
        self.next_font.set(id + 1);
        self.tx
            .send(Message::AddFont {
                id,
                data: source.data,
                index: source.index,
            })
            .map_err(|_| ResourceError::Lost)?;
        Ok(Font::new(cherenkov::FontId::new(id)))
    }

    /// Creates an offscreen surface.
    ///
    /// # Errors
    /// [`SurfaceError::ZeroSize`] when a dimension is zero,
    /// [`SurfaceError::TooLarge`] when the size exceeds the device limit and
    /// [`SurfaceError::Lost`] when the render thread is gone.
    pub fn surface(&self, target: Offscreen) -> Result<Surface, SurfaceError> {
        let id = self.next_surface.get();
        self.next_surface.set(id + 1);
        let surface = Surface::new(id, target.size, self.tx.clone())?;
        let mut surfaces = self.surfaces.borrow_mut();
        surfaces.retain(|_, s| s.strong_count() > 0);
        surfaces.insert(id, Rc::downgrade(&surface.shared));
        Ok(surface)
    }

    /// The number of surfaces still alive, for leak testing.
    #[doc(hidden)]
    pub fn live_surfaces(&self) -> usize {
        let mut surfaces = self.surfaces.borrow_mut();
        surfaces.retain(|_, s| s.strong_count() > 0);
        surfaces.len()
    }

    /// Renders every dirty surface for the frame at `time`, blocking until
    /// the render thread has submitted and — when timestamps are enabled —
    /// drained the GPU.
    ///
    /// # Errors
    /// [`RenderError::Unsupported`] when a surface's content needs a feature
    /// this slice does not draw, [`RenderError::DeviceLost`] when the device
    /// is lost and [`RenderError::Thread`] when the render thread is gone.
    pub fn render(&self, time: FrameTime) -> Result<Next, RenderError> {
        {
            let mut surfaces = self.surfaces.borrow_mut();
            surfaces.retain(|_, s| s.strong_count() > 0);
            for (id, shared) in surfaces.iter() {
                if let Some(shared) = shared.upgrade()
                    && let Some(changes) = shared.borrow_mut().take_changes()
                {
                    let _ = self.tx.send(Message::Commit {
                        surface: *id,
                        changes,
                    });
                }
            }
        }
        let (reply, rx) = std::sync::mpsc::channel();
        self.tx
            .send(Message::Render { time, reply })
            .map_err(|_| RenderError::Thread)?;
        let (next, stats) = rx.recv().map_err(|_| RenderError::Thread)??;
        self.stats.replace(stats);
        Ok(next)
    }

    /// Statistics of the last [`Engine::render`].
    #[must_use]
    pub fn stats(&self) -> FrameStats {
        self.stats.borrow().clone()
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
