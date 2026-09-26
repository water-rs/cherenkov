// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cherenkov-cpu`: the CPU raster backend for the Cherenkov 2D rendering
//! engine.
//!
//! The engine owns a render thread which holds every framebuffer; the UI
//! thread owns the [`Engine`], [`Surface`]s and [`Layer`]s and talks to it
//! over owned messages. There are no locks: `Engine`, `Surface` and `Layer`
//! are `!Send` and everything crossing the channel is `Send`.
//!
//! Framebuffers are premultiplied linear Display P3, one f32 per channel,
//! rasterized in horizontal bands of 16 rows by an exact signed-area
//! coverage accumulator (font-rs / vello-cpu style): every flattened edge
//! deposits trapezoid areas into a row accumulator and a prefix sum turns
//! it into winding-weighted coverage. Unlike the oracle's per-pixel
//! geometric area, the accumulator is exact only for polygons that do not
//! self-overlap inside a single pixel.
//!
//! Measured on the render corpus, materializing readbacks as f16 costs
//! +0.0013 mean FLIP versus keeping f32 (0.00406 vs 0.00278) — the
//! framebuffer itself is always f32.
//!
//! ```no_run
//! use cherenkov::{Draw, WorkingColor};
//! use cherenkov::kurbo::Rect;
//! use cherenkov_cpu::{Engine, Offscreen, OffscreenFormat, Raster, RasterConfig};
//!
//! let engine = Engine::<Raster>::new(RasterConfig::default())?;
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
use std::rc::Rc;
use std::sync::mpsc::Sender;

pub use crate::config::{Budget, Bytes, MemoryUsage, Pressure, RasterConfig, RasterInfo};
pub use crate::error::{EngineError, RenderError, ResourceError, SurfaceError, Unsupported};
pub use crate::font::{Font, FontSource};
pub use crate::surface::{
    FrameStats, FrameTime, Layer, LayerContent, LayerEdit, Next, Offscreen, OffscreenFormat,
    Readback, RefreshRange, Surface, Transaction,
};

use crate::message::{Message, SurfaceId};
use crate::surface::SurfaceShared;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Raster {}
}

/// A rendering backend. Sealed: only [`Raster`] exists in this slice.
pub trait Backend: sealed::Sealed + 'static {
    /// The backend's configuration type.
    type Config;
}

/// The CPU raster backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct Raster;

impl Backend for Raster {
    type Config = RasterConfig;
}

/// The engine: owns the worker pool and the render thread. `!Send`, lives
/// on the UI thread.
#[derive(Debug)]
pub struct Engine<B: Backend> {
    tx: Sender<Message>,
    info: RasterInfo,
    stats: Cell<FrameStats>,
    surfaces: RefCell<HashMap<SurfaceId, Rc<RefCell<SurfaceShared>>>>,
    next_surface: Cell<SurfaceId>,
    next_font: Cell<u64>,
    thread: Option<std::thread::JoinHandle<()>>,
    _backend: PhantomData<B>,
    // `!Send`: the engine lives on the UI thread.
    _not_send: PhantomData<*const ()>,
}

impl Engine<Raster> {
    /// Creates the worker pool on the render thread, blocking until it
    /// reports success or failure.
    ///
    /// # Errors
    /// [`EngineError::Thread`] when the render thread fails to start or
    /// the worker pool cannot be built.
    pub fn new(config: RasterConfig) -> Result<Self, EngineError> {
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
            stats: Cell::new(FrameStats::default()),
            surfaces: RefCell::new(HashMap::new()),
            next_surface: Cell::new(0),
            next_font: Cell::new(1),
            thread: Some(thread),
            _backend: PhantomData,
            _not_send: PhantomData,
        })
    }

    /// Worker pool information for provenance.
    #[must_use]
    pub const fn info(&self) -> &RasterInfo {
        &self.info
    }

    /// Reports system memory pressure. `Critical` clears the glyph mask
    /// cache; `Moderate` currently does nothing.
    pub fn trim(&self, pressure: Pressure) {
        let _ = self.tx.send(Message::Trim(pressure));
    }

    /// The engine's current memory usage.
    #[must_use]
    pub fn memory(&self) -> MemoryUsage {
        let (reply, rx) = std::sync::mpsc::channel();
        if self.tx.send(Message::Memory { reply }).is_err() {
            return MemoryUsage::default();
        }
        rx.recv().unwrap_or_default()
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
        let _ = self.tx.send(Message::AddFont {
            id,
            data: source.data,
            index: source.index,
        });
        Ok(Font::new(cherenkov::FontId::new(id)))
    }

    /// Creates an offscreen surface.
    ///
    /// # Errors
    /// [`SurfaceError::TooLarge`] when the size exceeds the framebuffer
    /// limit and [`SurfaceError::Lost`] when the render thread is gone.
    pub fn surface(&self, target: Offscreen) -> Result<Surface, SurfaceError> {
        let id = self.next_surface.get();
        self.next_surface.set(id + 1);
        let surface = Surface::new(id, target.size, target.format(), self.tx.clone())?;
        self.surfaces
            .borrow_mut()
            .insert(id, Rc::clone(&surface.shared));
        Ok(surface)
    }

    /// Renders every dirty surface for the frame at `time`, blocking until
    /// the render thread has rasterized them.
    ///
    /// # Errors
    /// [`RenderError::Unsupported`] when a surface's content needs a feature
    /// this slice does not draw and [`RenderError::Thread`] when the render
    /// thread is gone.
    pub fn render(&self, time: FrameTime) -> Result<Next, RenderError> {
        for (id, shared) in self.surfaces.borrow().iter() {
            if let Some(changes) = shared.borrow_mut().take_changes() {
                let _ = self.tx.send(Message::Commit {
                    surface: *id,
                    changes,
                });
            }
        }
        let (reply, rx) = std::sync::mpsc::channel();
        self.tx
            .send(Message::Render { time, reply })
            .map_err(|_| RenderError::Thread)?;
        let (next, stats) = rx.recv().map_err(|_| RenderError::Thread)??;
        self.stats.set(stats);
        Ok(next)
    }

    /// Statistics of the last [`Engine::render`].
    #[must_use]
    pub const fn stats(&self) -> FrameStats {
        self.stats.get()
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
