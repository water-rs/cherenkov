// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cherenkov-vello`: the transitional [vello](https://vello.dev) backend
//! for the Cherenkov 2D rendering engine.
//!
//! The engine owns a render thread which holds every GPU object — the wgpu
//! device and queue, the `vello::Renderer`, the retained layer trees and the
//! resource registries. The UI thread owns the [`Engine`], [`Surface`]s and
//! [`Layer`]s and talks to it over owned messages. There are no locks:
//! `Engine`, `Surface` and `Layer` are `!Send` and everything crossing the
//! channel is `Send`.
//!
//! ```no_run
//! use cherenkov::{Draw, WorkingColor};
//! use cherenkov::kurbo::Rect;
//! use cherenkov_vello::{Engine, Offscreen, Vello, VelloConfig};
//!
//! let engine = Engine::<Vello>::new(VelloConfig::default())?;
//! let surface = engine.surface(Offscreen::new((64, 64)))?;
//! surface.update(|tx| {
//!     tx[surface.root()].content(
//!         surface.record(|c| c.fill(Rect::new(0., 0., 64., 64.), WorkingColor::WHITE)),
//!     );
//! });
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod capability;
mod error;
mod gpu_content;
pub mod interop;
mod message;
mod render;
mod resource;
mod surface;

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::mpsc::Sender;

pub use crate::error::{EngineError, RenderError, ResourceError, SurfaceError, Unsupported};
pub use crate::gpu_content::{GpuContent, GpuContentHandle, RedrawHandle};
pub use crate::resource::{Filter, Font, FontSource, Image, ImageSource, Shader, ShaderSource};
pub use crate::surface::{
    FrameStats, FrameTime, Layer, LayerContent, LayerEdit, Next, Offscreen, Readback, RefreshRange,
    Surface, Target, Transaction,
};

use crate::message::{Message, SurfaceId};
use crate::surface::SurfaceShared;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Vello {}
}

/// A rendering backend. Sealed: only [`Vello`] exists in this slice.
pub trait Backend: sealed::Sealed + 'static {
    /// The backend's configuration type.
    type Config;
}

/// The vello backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct Vello;

impl Backend for Vello {
    type Config = VelloConfig;
}

/// A byte count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bytes(pub u64);

impl Bytes {
    /// A count of mebibytes.
    #[must_use]
    pub const fn mib(n: u64) -> Self {
        Self(n * 1024 * 1024)
    }
}

/// The engine's memory budgets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// Device-side memory (buffers and textures).
    pub gpu: Bytes,
    /// CPU-side memory.
    pub cpu: Bytes,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            gpu: Bytes::mib(512),
            cpu: Bytes::mib(96),
        }
    }
}

/// System memory pressure reported to the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pressure {
    /// Reduce caches where cheap.
    Moderate,
    /// Drop every cache.
    Critical,
}

/// The engine's current memory usage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryUsage {
    /// Buffers and textures, in bytes.
    pub gpu: Bytes,
    /// CPU-side cache bytes.
    pub cpu: Bytes,
}

/// Which adapter class the engine prefers when creating its own device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PowerPreference {
    /// Battery-saving adapters.
    Low,
    /// Performance adapters.
    #[default]
    High,
}

/// Adapter information for provenance.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    /// Adapter name.
    pub name: String,
    /// Backend (e.g. `Vulkan`).
    pub backend: String,
    /// PCI vendor id.
    pub vendor: u32,
    /// PCI device id.
    pub device: u32,
    /// Device class (e.g. `IntegratedGpu`).
    pub device_type: String,
    /// Driver name.
    pub driver: String,
    /// Driver version detail.
    pub driver_info: String,
}

/// Configuration for the vello engine.
#[derive(Clone, Debug)]
pub struct VelloConfig {
    /// Memory budgets.
    pub budget: Budget,
    /// When true and the adapter supports it, [`Engine::render`] measures
    /// GPU time with drained timestamp queries.
    ///
    /// [`Engine::render`]: Engine::render
    pub timestamps: bool,
    /// Adapter power preference for [`Engine::new`].
    ///
    /// [`Engine::new`]: Engine::new
    pub power: PowerPreference,
}

impl Default for VelloConfig {
    fn default() -> Self {
        Self {
            budget: Budget::default(),
            timestamps: false,
            power: PowerPreference::High,
        }
    }
}

/// The engine: owns the device and the render thread. `!Send`, lives on the
/// UI thread.
///
/// The render thread owns every GPU object. `Engine` is deliberately `!Send`
/// — everything it hands out (`Surface`, `Layer`, resource handles) may only
/// live on the UI thread that created the engine.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<cherenkov_vello::Engine<cherenkov_vello::Vello>>();
/// ```
#[derive(Debug)]
pub struct Engine<B: Backend> {
    tx: Sender<Message>,
    info: GpuInfo,
    stats: Cell<FrameStats>,
    surfaces: RefCell<HashMap<SurfaceId, Rc<RefCell<SurfaceShared>>>>,
    next_surface: Cell<SurfaceId>,
    next_font: Cell<u64>,
    next_image: Cell<u64>,
    next_shader: Cell<u64>,
    next_filter: Cell<u64>,
    next_content: Cell<u64>,
    thread: Option<std::thread::JoinHandle<()>>,
    _backend: PhantomData<B>,
    // `!Send`: the engine lives on the UI thread.
    _not_send: PhantomData<Rc<()>>,
}

impl Engine<Vello> {
    /// Creates the instance, adapter, device and vello renderer on the
    /// render thread, blocking until it reports success or failure.
    ///
    /// The adapter honours the `WGPU_ADAPTER_NAME`/`WGPU_BACKEND`
    /// environment variables, falling back to [`VelloConfig::power`].
    ///
    /// # Errors
    /// [`EngineError::NoAdapter`] when no suitable adapter exists,
    /// [`EngineError::RequestDevice`] when device creation fails,
    /// [`EngineError::Renderer`] when the vello renderer fails to
    /// initialize, and [`EngineError::Thread`] when the render thread dies.
    pub fn new(config: VelloConfig) -> Result<Self, EngineError> {
        Self::spawn(config, render::DeviceRequest::Create)
    }

    /// Creates an engine driving an existing wgpu device (the embedder's
    /// shared GPU context), building the vello renderer on the render
    /// thread.
    ///
    /// # Errors
    /// [`EngineError::Renderer`] when the vello renderer fails to
    /// initialize, and [`EngineError::Thread`] when the render thread dies.
    pub fn with_device(
        config: VelloConfig,
        gpu: interop::wgpu::DeviceSource,
    ) -> Result<Self, EngineError> {
        Self::spawn(config, render::DeviceRequest::Existing(gpu))
    }

    fn spawn(config: VelloConfig, request: render::DeviceRequest) -> Result<Self, EngineError> {
        let (tx, rx) = std::sync::mpsc::channel::<Message>();
        let (init_tx, init_rx) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("cherenkov-vello-render".into())
            .spawn(move || render::run(config, request, rx, init_tx))
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
            next_image: Cell::new(1),
            next_shader: Cell::new(1),
            next_filter: Cell::new(1),
            next_content: Cell::new(1),
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

    /// Reports system memory pressure. `Critical` drops fragment caches.
    pub fn trim(&self, pressure: Pressure) {
        let _ = self.tx.send(Message::Trim(pressure));
    }

    /// The engine's current memory usage.
    ///
    /// # Panics
    /// Panics if the render thread has stopped, which cannot happen while an
    /// `Engine` is alive.
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
        resource::validate_font(&source.data, source.index)?;
        let id = self.next_font.get();
        self.next_font.set(id + 1);
        let _ = self.tx.send(Message::AddFont {
            id,
            data: source.data,
            index: source.index,
        });
        Ok(Font::new(cherenkov::FontId::new(id), self.tx.clone()))
    }

    /// Registers an image.
    ///
    /// # Errors
    /// This slice accepts every well-formed [`ImageSource`]; it never fails.
    pub fn image(&self, source: ImageSource) -> Result<Image, ResourceError> {
        let id = self.next_image.get();
        self.next_image.set(id + 1);
        let _ = self.tx.send(Message::AddImage {
            id,
            image: source.into_image_data(),
        });
        Ok(Image::new(cherenkov::ImageId::new(id), self.tx.clone()))
    }

    /// Creates a surface over `target`: an [`Offscreen`] texture or an
    /// [`interop::wgpu::Window`].
    ///
    /// # Errors
    /// [`SurfaceError::TooLarge`] when the size exceeds the device limit,
    /// [`SurfaceError::Unsupported`] when the target is not drawable yet,
    /// and [`SurfaceError::Lost`] when the render thread is gone.
    pub fn surface(&self, target: impl Into<Target>) -> Result<Surface, SurfaceError> {
        let id = self.next_surface.get();
        self.next_surface.set(id + 1);
        let surface = Surface::new(id, target.into(), self.tx.clone())?;
        self.surfaces
            .borrow_mut()
            .insert(id, Rc::clone(&surface.shared));
        Ok(surface)
    }

    /// Renders every dirty surface for the frame at `time`, blocking until
    /// the render thread has submitted and — when timestamps are enabled —
    /// drained the GPU.
    ///
    /// # Errors
    /// [`RenderError::Unsupported`] when a surface's content needs a feature
    /// this slice does not draw, [`RenderError::DeviceLost`] when the device
    /// is lost and [`RenderError::Thread`] when the render thread is gone.
    /// A surface that failed to render is left in its previous state.
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

impl<B: Backend + capability::ShaderPaint> Engine<B> {
    /// Registers a WGSL shader, blocking until the render thread has
    /// compiled and validated its pipeline.
    ///
    /// `source` supplies the fragment body (without the prelude); see
    /// [`ShaderSource`].
    ///
    /// # Errors
    /// [`ResourceError::Shader`] when the WGSL fails validation or the
    /// pipeline fails to build, and [`ResourceError::Io`] is never returned
    /// here.
    pub fn shader(&self, source: ShaderSource) -> Result<Shader, ResourceError> {
        let id = self.next_shader.get();
        self.next_shader.set(id + 1);
        let (reply, rx) = std::sync::mpsc::channel();
        self.tx
            .send(Message::AddShader {
                id,
                source: crate::message::ShaderSpec {
                    source: source.source,
                    animated: source.animated,
                },
                reply,
            })
            .map_err(|_| ResourceError::Shader("the render thread stopped".into()))?;
        rx.recv()
            .map_err(|_| ResourceError::Shader("the render thread stopped".into()))??;
        Ok(Shader::new(cherenkov::ShaderId::new(id), self.tx.clone()))
    }
}

impl<B: Backend + capability::GpuContent> Engine<B> {
    /// Creates GPU content of `size` pixels, attachable to a layer with
    /// [`LayerEdit::content`](crate::LayerEdit::content).
    #[must_use]
    pub fn gpu_content(
        &self,
        size: (u32, u32),
        content: impl gpu_content::GpuContent + Send,
    ) -> GpuContentHandle {
        let id = self.next_content.get();
        self.next_content.set(id + 1);
        GpuContentHandle {
            id,
            size,
            dirty: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            content: Some(Box::new(content)),
        }
    }
}

impl<B: Backend> Engine<B> {
    /// Registers a [`filtrate::Filter`] run by the reference
    /// [`filtrate::Executor`] on the render thread.
    pub fn filter<F: filtrate::Filter + Send>(&self, filter: F) -> Filter
    where
        B: capability::Runs<F>,
    {
        self.register_effect(render::filter::FromFilter(filter))
    }

    /// Registers a custom [`filtrate::Effect`] (e.g. a `WaterUI` `ViewEffect`
    /// renderer) run on the render thread.
    pub fn effect<E: filtrate::Effect + Send>(&self, effect: E) -> Filter {
        self.register_effect(render::filter::FromEffect(effect))
    }

    fn register_effect(&self, source: impl render::filter::FilterSource + 'static) -> Filter {
        let id = self.next_filter.get();
        self.next_filter.set(id + 1);
        let _ = self.tx.send(Message::AddFilter {
            id,
            source: Box::new(source),
        });
        Filter::new(cherenkov::FilterId::new(id), self.tx.clone())
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
