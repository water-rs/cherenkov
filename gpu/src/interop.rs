//! Device sharing and presentation of the engine's retained working-space output.

pub use crate::render::present::{OutputAlpha, OutputColor, Presenter, TextureOutput};
/// Device types used by GPU integrations.
pub mod wgpu {
    pub use ::wgpu::*;
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
///
/// The host receives a new texture only on creation and resize, then samples
/// the retained texture after `Engine::render` completes. Presentation must
/// use the same device supplied through `GpuConfig::device`. Dropping the
/// receiver stops notifications; it does not destroy the surface. A host must
/// consume resize notifications before sampling output after a resize.
#[derive(Debug)]
pub struct TextureTarget {
    pub(crate) size: (u32, u32),
    pub(crate) refresh: cherenkov::RefreshRange,
    pub(crate) textures: std::sync::mpsc::Sender<wgpu::Texture>,
}

impl TextureTarget {
    /// Sets the refresh range for animated content.
    ///
    /// # Panics
    /// When the range is empty or includes zero.
    #[must_use]
    pub fn rate(mut self, rate: cherenkov::RefreshRange) -> Self {
        assert!(
            *rate.start() > 0 && !rate.is_empty(),
            "refresh range must be positive and ordered"
        );
        self.refresh = rate;
        self
    }

    /// Creates a target and its resize notification channel.
    /// Textures contain premultiplied linear Display P3, in `Rgba16Float`.
    #[must_use]
    pub fn new(size: (u32, u32)) -> (Self, std::sync::mpsc::Receiver<wgpu::Texture>) {
        let (textures, receiver) = std::sync::mpsc::channel();
        (
            Self {
                size,
                textures,
                refresh: 60..=60,
            },
            receiver,
        )
    }
}
