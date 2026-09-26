//! GPU integration with the engine.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub use crate::render::filter::EffectBox;
pub use crate::render::present::{OutputAlpha, OutputColor, Presenter, TextureOutput};

/// The device types and drawing contexts exposed to custom GPU producers.
pub mod wgpu {
    pub use ::wgpu::*;
    use std::time::Duration;

    /// Persistent device resources supplied once before rendering content.
    pub struct Context<'a> {
        /// Adapter that owns the engine's device.
        pub adapter: &'a Adapter,
        /// Engine-owned device.
        pub device: &'a Device,
        /// Engine-owned submission queue.
        pub queue: &'a Queue,
        /// Output texture format, containing premultiplied linear Display P3.
        pub format: TextureFormat,
        /// Requests a new frame after asynchronous producer work completes.
        pub redraw: super::RedrawHandle,
    }

    /// An engine-allocated output texture for one custom content frame.
    pub struct Frame<'a> {
        /// Engine-owned device.
        pub device: &'a Device,
        /// Engine-owned queue.
        pub queue: &'a Queue,
        /// Output texture in premultiplied linear Display P3.
        pub texture: &'a Texture,
        /// Output attachment view.
        pub view: &'a TextureView,
        /// Output format.
        pub format: TextureFormat,
        /// Texture width in physical pixels.
        pub width: u32,
        /// Texture height in physical pixels.
        pub height: u32,
        /// Display scale.
        pub scale: f32,
        /// Presentation time relative to the first frame.
        pub elapsed: Duration,
        /// Time since the previous presentation of this content.
        pub delta: Duration,
        pub(crate) redraw: bool,
    }

    impl Frame<'_> {
        /// Keeps the engine refreshing for the next frame.
        pub const fn request_redraw(&mut self) {
            self.redraw = true;
        }
    }
}

/// A producer moved to the engine's render thread for its entire lifetime.
/// UI-thread-bound producers send owned frame data over a channel to this object.
pub trait GpuContent: Send + 'static {
    /// Creates persistent resources once before the first frame.
    fn setup(&mut self, context: &wgpu::Context<'_>) -> impl Future<Output = ()>;
    /// Draws into the provided engine-owned attachment.
    fn render(&mut self, frame: &mut wgpu::Frame<'_>);
}

/// A producer boxed for `Engine::gpu_content`.
pub struct GpuContentBox {
    pub(crate) content: Box<dyn Content>,
    pub(crate) redraw: RedrawHandle,
}

impl std::fmt::Debug for GpuContentBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContentBox").finish_non_exhaustive()
    }
}

impl GpuContentBox {
    /// Creates a producer with the host's event-loop wake callback.
    /// The callback must be safe to invoke from a producer thread, including
    /// while the engine is idle (for example, a window event-loop proxy).
    #[must_use]
    pub fn new(content: impl GpuContent, wake: impl Fn() + Send + Sync + 'static) -> Self {
        Self {
            content: Box::new(content),
            redraw: RedrawHandle {
                dirty: Arc::new(AtomicBool::new(true)),
                wake: Arc::new(wake),
            },
        }
    }

    /// Obtains a redraw requester before the producer moves to the engine.
    #[must_use]
    pub fn redraw_handle(&self) -> RedrawHandle {
        self.redraw.clone()
    }
}

/// A thread-safe request to redraw this content on the next engine frame.
#[derive(Clone)]
pub struct RedrawHandle {
    pub(crate) dirty: Arc<AtomicBool>,
    wake: Arc<dyn Fn() + Send + Sync>,
}

impl std::fmt::Debug for RedrawHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedrawHandle")
            .field("dirty", &self.is_dirty())
            .finish_non_exhaustive()
    }
}

impl RedrawHandle {
    /// Marks the producer's output stale.
    pub fn request_redraw(&self) {
        if !self.dirty.swap(true, Ordering::AcqRel) {
            (self.wake)();
        }
    }

    /// Whether the producer has an unconsumed redraw request.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }
}

pub(crate) trait Content: Send {
    fn setup(&mut self, context: &wgpu::Context<'_>);
    fn render(&mut self, frame: &mut wgpu::Frame<'_>);
}

impl<C: GpuContent> Content for C {
    fn setup(&mut self, context: &wgpu::Context<'_>) {
        pollster::block_on(GpuContent::setup(self, context));
    }

    fn render(&mut self, frame: &mut wgpu::Frame<'_>) {
        GpuContent::render(self, frame);
    }
}

/// A host event-loop callback callable from the engine or producer threads.
#[derive(Clone)]
pub struct RedrawCallback(Arc<dyn Fn() + Send + Sync>);

impl RedrawCallback {
    /// Wraps the host's display-link or event-loop wake operation.
    pub fn new(wake: impl Fn() + Send + Sync + 'static) -> Self {
        Self(Arc::new(wake))
    }

    /// Asks the host to schedule an engine frame.
    pub fn wake(&self) {
        (self.0)();
    }
}

impl std::fmt::Debug for RedrawCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedrawCallback").finish_non_exhaustive()
    }
}

/// An existing device shared with a native presentation host.
/// All four handles must belong to the same device creation chain.
#[derive(Clone, Debug)]
pub struct SharedDevice {
    /// Instance used to create the adapter and native surfaces.
    pub instance: wgpu::Instance,
    /// Adapter used to create the device.
    pub adapter: wgpu::Adapter,
    /// Device used for both engine composition and native presentation.
    pub device: wgpu::Device,
    /// This device's submission queue.
    pub queue: wgpu::Queue,
}

/// A surface whose engine-owned linear P3 texture is handed to a native host.
/// The host receives a new texture only on creation and resize, then samples
/// the retained texture after `Engine::render` completes. Presentation must
/// use the same device supplied through `GpuConfig::device`.
#[derive(Debug)]
pub struct TextureTarget {
    pub(crate) size: (u32, u32),
    pub(crate) textures: std::sync::mpsc::Sender<wgpu::Texture>,
}

impl TextureTarget {
    /// Creates a target and its resize notification channel.
    /// Textures contain premultiplied linear Display P3, in `Rgba16Float`.
    #[must_use]
    pub fn new(size: (u32, u32)) -> (Self, std::sync::mpsc::Receiver<wgpu::Texture>) {
        let (textures, receiver) = std::sync::mpsc::channel();
        (Self { size, textures }, receiver)
    }
}

/// Compiles a producer's WGSL with the engine's color conversion helpers.
/// `cherenkov_srgb` converts straight-alpha sRGB and
/// `cherenkov_premultiplied_srgb` converts encoded premultiplied sRGB into
/// premultiplied linear Display P3, the GPU content attachment convention.
///
/// # Panics
/// When shader creation fails or the compiled template cannot be rendered.
#[must_use]
pub fn shader_module(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    use askama::Template as _;
    #[derive(askama::Template)]
    #[template(path = "content.wgsl", escape = "none")]
    struct Source<'a> {
        source: &'a str,
    }
    let source = Source { source }
        .render()
        .expect("GPU shader template rendering failed");
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}
