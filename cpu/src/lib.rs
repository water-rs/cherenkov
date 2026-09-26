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
//! Framebuffers are premultiplied linear Display P3, one f32 per channel.
//! Geometry is compiled into sparse exact-area coverage runs before
//! independent horizontal bands shade and composite pixels. Fill rules,
//! self-intersections, and nested clips are resolved geometrically before
//! integrating area. Exactness is relative to the flattened device-space
//! boundaries; readback can round the f32 framebuffer to f16.
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
//! })?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![doc = include_str!("../DESIGN-coverage.md")]
#![doc = include_str!("../DESIGN-effects.md")]

mod config;
mod error;
mod font;
mod image;
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
pub use crate::image::{Image, ImageColorSpace, ImageSource};
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

    /// Reports system memory pressure. `Critical` clears glyph and prepared
    /// coverage caches and retained band scratch; `Moderate` does nothing.
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
    /// # Errors
    /// [`EngineError::Thread`] when the render thread is gone.
    pub fn memory(&self) -> Result<MemoryUsage, EngineError> {
        let (reply, rx) = std::sync::mpsc::channel();
        self.tx
            .send(Message::Memory { reply })
            .map_err(|_| EngineError::Thread("render thread gone".into()))?;
        rx.recv()
            .map_err(|_| EngineError::Thread("render thread gone".into()))
    }

    /// Registers a font.
    ///
    /// The data is parsed on the caller thread to reject invalid data,
    /// bitmap-only colour fonts (`CBDT`/`sbix` without outlines) and
    /// SVG-in-OpenType fonts, then handed to the render thread. `COLR`
    /// colour fonts render through the colour-glyph lowering.
    ///
    /// # Errors
    /// [`ResourceError::Font`] for unparseable data,
    /// `ResourceError::Unsupported(Unsupported::ColorFont)` for colour
    /// fonts and [`ResourceError::Io`] when the render thread is gone.
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
            .map_err(|_| {
                ResourceError::Io(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "render thread gone",
                ))
            })?;
        Ok(Font::new(cherenkov::FontId::new(id)))
    }

    /// Registers an image. The caller-thread [`ImageSource`] is converted
    /// to premultiplied linear Display P3 on the render thread.
    ///
    /// # Errors
    /// [`ResourceError::Image`] for a zero dimension or a pixel-length
    /// mismatch.
    pub fn image(&self, source: ImageSource) -> Result<Image, ResourceError> {
        source.validate()?;
        let id = self.next_font.get();
        self.next_font.set(id + 1);
        self.tx
            .send(Message::AddImage {
                id,
                width: source.width,
                height: source.height,
                pixels: source.pixels.into(),
                color_space: source.color_space,
            })
            .map_err(|_| {
                ResourceError::Io(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "render thread gone",
                ))
            })?;
        Ok(Image::new(cherenkov::ImageId::new(id), self.tx.clone()))
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
                self.tx
                    .send(Message::Commit {
                        surface: *id,
                        changes,
                    })
                    .map_err(|_| RenderError::Thread)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::LayerOp;

    /// An `Engine` whose render-thread receiver is already dropped.
    fn dead_engine() -> Engine<Raster> {
        let (tx, rx) = std::sync::mpsc::channel::<Message>();
        drop(rx);
        Engine {
            tx,
            info: RasterInfo {
                threads: 1,
                simd: "scalar",
                cpu: None,
            },
            stats: Cell::new(FrameStats::default()),
            surfaces: RefCell::new(HashMap::new()),
            next_surface: Cell::new(0),
            next_font: Cell::new(1),
            thread: None,
            _backend: PhantomData,
            _not_send: PhantomData,
        }
    }

    #[test]
    fn a_dead_render_thread_errors_instead_of_dropping_silently() {
        let engine = dead_engine();
        assert!(matches!(
            engine.trim(Pressure::Critical),
            Err(EngineError::Thread(_))
        ));
        assert!(matches!(engine.memory(), Err(EngineError::Thread(_))));
        let data = std::fs::read("../scenes/fonts/NotoSans.ttf").expect("test font");
        assert!(engine.font(FontSource::bytes(data)).is_err());
        assert!(matches!(
            engine.render(FrameTime::now()),
            Err(RenderError::Thread)
        ));
        // A pending surface change set fails its commit as well.
        engine.surfaces.borrow_mut().insert(
            7,
            Rc::new(RefCell::new(SurfaceShared {
                pending: vec![LayerOp::Create(0)],
                ..SurfaceShared::default()
            })),
        );
        assert!(matches!(
            engine.render(FrameTime::now()),
            Err(RenderError::Thread)
        ));
    }
}
