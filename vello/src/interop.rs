// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Interop with the wrapped GPU stack.
//!
//! [`interop::wgpu`] is the only place `wgpu` (and, through it, `vello`)
//! types appear in this crate's public API: sharing a device or a surface
//! with the embedder needs the real types. [`GpuContent`], [`RedrawHandle`]
//! and [`GpuContentBox`] let user GPU work reach that same device, and
//! [`EffectBox`] boxes a custom `filtrate` effect for
//! [`Engine::effect`](cherenkov::Engine::effect) — `filtrate::Effect` is
//! not object safe, so the erased form lives behind
//! [`crate::render::filter::FilterSource`].

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::render;

/// The `wgpu` surface of the engine.
pub mod wgpu {
    use std::time::Duration;

    pub use ::wgpu::*;

    /// An existing device and queue the engine drives instead of creating
    /// its own (the embedder's shared GPU context), configured with
    /// [`VelloConfig::device`](crate::VelloConfig::device).
    ///
    /// The engine uses it to create its vello renderer; shader pipelines
    /// and resources it builds are freed with the engine.
    #[derive(Clone)]
    pub struct DeviceSource {
        /// The adapter the device was requested from (for
        /// [`VelloInfo`](crate::VelloInfo)).
        pub adapter: Adapter,
        /// The device.
        pub device: Device,
        /// The queue.
        pub queue: Queue,
    }

    impl std::fmt::Debug for DeviceSource {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("DeviceSource").finish_non_exhaustive()
        }
    }

    impl DeviceSource {
        /// A source over an existing adapter, device and queue.
        #[must_use]
        pub const fn new(adapter: Adapter, device: Device, queue: Queue) -> Self {
            Self {
                adapter,
                device,
                queue,
            }
        }
    }

    /// A window target: the embedder's configured surface, passed to
    /// [`Engine::surface`](cherenkov::Engine::surface).
    ///
    /// The engine configures `surface` with `config` on the render thread
    /// and reconfigures it on [`Surface::resize`](cherenkov::Surface::resize).
    pub struct Window {
        /// The wgpu surface.
        pub surface: Surface<'static>,
        /// The configuration the engine applies.
        pub config: SurfaceConfiguration,
        /// The refresh-rate range of the window's display, in hertz.
        ///
        /// wgpu exposes no display refresh rate, so the embedder fills
        /// this in from the platform (e.g. the monitor's refresh rate
        /// from the windowing toolkit) for [`cherenkov::Next::At`] to report
        /// the display's real cadence.
        pub rate: cherenkov::RefreshRange,
    }

    impl std::fmt::Debug for Window {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Window").finish_non_exhaustive()
        }
    }

    /// The GPU context handed to GPU content implementations at setup time.
    pub struct Context<'a> {
        /// The device.
        pub device: &'a Device,
        /// The queue.
        pub queue: &'a Queue,
        /// The texture format content renders into.
        pub format: TextureFormat,
    }

    /// One frame's drawing context handed to GPU content implementations.
    ///
    /// The content draws into `texture`/`view`, a `w` × `h` `Rgba8Unorm`
    /// texture whose values are premultiplied sRGB-encoded.
    pub struct Frame<'a> {
        /// The device.
        pub device: &'a Device,
        /// The queue.
        pub queue: &'a Queue,
        /// The texture being drawn into.
        pub texture: &'a Texture,
        /// Its view.
        pub view: &'a TextureView,
        /// The texture format.
        pub format: TextureFormat,
        /// Width in pixels.
        pub width: u32,
        /// Height in pixels.
        pub height: u32,
        /// Scale factor of the target.
        pub scale: f32,
        /// Time since the engine started.
        pub elapsed: Duration,
        /// Time since the previous frame, capped at 100 ms (defaulting to
        /// 1/60 s for the first frame).
        pub delta: Duration,
        /// Set by `request_redraw`.
        pub(crate) redraw: bool,
    }

    impl Frame<'_> {
        /// Requests another frame after this one.
        pub const fn request_redraw(&mut self) {
            self.redraw = true;
        }

        /// Whether a redraw was requested this frame.
        #[must_use]
        pub const fn redraw_requested(&self) -> bool {
            self.redraw
        }
    }
}

/// Content a layer draws by rendering into a GPU texture.
///
/// Implementations run on the render thread and may hold any wgpu objects
/// they create in [`GpuContent::setup`]. A [`GpuContentBox`] wraps one for
/// [`Engine::gpu_content`](cherenkov::Engine::gpu_content); the engine
/// composites its retained `Rgba8Unorm` texture as an image wherever the
/// layer is drawn.
///
/// This is the vello backend's content contract, not the shared
/// [`GpuContent`](cherenkov::GpuContent) capability trait.
pub trait GpuContent: 'static {
    /// Creates GPU resources. Awaited once, on the render thread, before the
    /// first [`GpuContent::render`].
    fn setup(&mut self, gpu: &wgpu::Context<'_>) -> impl Future<Output = ()>;

    /// Renders one frame into `frame`'s texture.
    ///
    /// Call [`wgpu::Frame::request_redraw`] to be re-rendered on the next
    /// frame.
    fn render(&mut self, frame: &mut wgpu::Frame<'_>);

    /// Whether the produced texture is fully opaque. Informational only:
    /// the texture is always composited through vello.
    fn is_opaque(&self) -> bool {
        false
    }
}

/// A `filtrate` effect boxed for [`Engine::effect`](cherenkov::Engine::effect).
///
/// `filtrate::Effect::setup` returns `impl Future`, so `Effect` is not
/// object safe and `Box<dyn Effect>` cannot exist; `EffectBox` erases it
/// behind the object-safe [`FilterSource`](crate::render::filter::FilterSource)
/// instead.
pub struct EffectBox {
    /// The erased source the render thread builds into a runnable effect.
    pub(crate) inner: Box<dyn render::filter::FilterSource>,
}

impl std::fmt::Debug for EffectBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EffectBox").finish_non_exhaustive()
    }
}

impl<E: filtrate::Effect + Send> From<E> for EffectBox {
    fn from(effect: E) -> Self {
        Self {
            inner: Box::new(render::filter::FromEffect(effect)),
        }
    }
}

/// GPU content boxed for [`Engine::gpu_content`](cherenkov::Engine::gpu_content):
/// the content object plus the redraw flag its [`RedrawHandle`] shares with
/// the render thread.
///
/// Obtained by `Into` conversion from any [`GpuContent`]:
/// `engine.gpu_content(size, my_content)` boxes `my_content` and yields a
/// [`GpuContentHandle`](cherenkov::GpuContentHandle) whose `.content` is
/// this box — [`GpuContentBox::redraw_handle`] re-renders the texture later.
pub struct GpuContentBox {
    /// The shared redraw flag the slot polls each frame.
    pub(crate) dirty: Arc<AtomicBool>,
    /// The content object, until it moves to the render thread.
    pub(crate) content: Box<dyn AnyGpuContent>,
}

impl std::fmt::Debug for GpuContentBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContentBox").finish_non_exhaustive()
    }
}

impl<C: GpuContent + Send> From<C> for GpuContentBox {
    fn from(content: C) -> Self {
        Self {
            dirty: Arc::new(AtomicBool::new(false)),
            content: Box::new(content),
        }
    }
}

impl GpuContentBox {
    /// A handle another thread can use to request a re-render of this
    /// content's texture.
    #[must_use]
    pub fn redraw_handle(&self) -> RedrawHandle {
        RedrawHandle(Arc::clone(&self.dirty))
    }

    /// Whether a redraw is pending.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Relaxed)
    }
}

/// A shareable redraw requester for a [`GpuContentBox`].
#[derive(Clone, Debug)]
pub struct RedrawHandle(Arc<AtomicBool>);

impl RedrawHandle {
    /// Marks the content dirty: the engine re-renders it at the next
    /// [`Engine::render`](cherenkov::Engine::render).
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
    fn setup(&mut self, gpu: &wgpu::Context<'_>);
    /// Renders one frame.
    fn render(&mut self, frame: &mut wgpu::Frame<'_>);
}

impl<T: GpuContent + Send> AnyGpuContent for T {
    fn setup(&mut self, gpu: &wgpu::Context<'_>) {
        pollster::block_on(GpuContent::setup(self, gpu));
    }

    fn render(&mut self, frame: &mut wgpu::Frame<'_>) {
        GpuContent::render(self, frame);
    }
}
