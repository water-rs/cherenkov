// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cherenkov-vello`: the transitional [vello](https://vello.dev) backend
//! for the Cherenkov 2D rendering engine.
//!
//! The shared front end lives in the [`cherenkov`] crate: [`Engine`],
//! [`Surface`], [`Layer`], the layer tree and the render thread's loop are
//! all generic over [`Backend`]. This crate supplies the render side only —
//! [`Vello`]'s [`Backend`] implementation drives the wgpu device and the
//! `vello` renderer on the render thread.
//!
//! ```no_run
//! use cherenkov::{Draw, Engine, Offscreen, OffscreenFormat, WorkingColor};
//! use cherenkov::kurbo::Rect;
//! use cherenkov_vello::{Vello, VelloConfig};
//!
//! let engine = Engine::<Vello>::new(VelloConfig::default())?;
//! let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
//! surface.update(|tx| {
//!     tx[surface.root()].content(
//!         surface.record(|c| c.fill(Rect::new(0., 0., 64., 64.), WorkingColor::WHITE)),
//!     );
//! });
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod interop;
mod names;
mod render;

use cherenkov::Rgba8;
use cherenkov::{
    Backend, EngineError, FilterId, GpuContent as GpuContentCapability, LayerId, Offscreen,
    ResourceError, ShaderId, ShaderSource, SurfaceId, Uploads,
};
use cherenkov::{Effects, Filters, Runs, ShaderPaintCapability};

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
pub struct VelloInfo {
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
    pub budget: cherenkov::Budget,
    /// When true and the adapter supports it,
    /// [`Engine::render`](cherenkov::Engine::render) measures GPU time with
    /// drained timestamp queries.
    pub timestamps: bool,
    /// Adapter power preference for [`Engine::new`](cherenkov::Engine::new).
    ///
    /// Honoured only when `device` is `None`: an embedder-provided device
    /// keeps its own adapter.
    pub power: PowerPreference,
    /// An existing device the engine drives instead of creating its own
    /// (the embedder's shared GPU context).
    pub device: Option<interop::wgpu::DeviceSource>,
}

impl Default for VelloConfig {
    fn default() -> Self {
        Self {
            budget: cherenkov::Budget::default(),
            timestamps: false,
            power: PowerPreference::High,
            device: None,
        }
    }
}

impl VelloConfig {
    /// A configuration driving an existing wgpu device.
    #[must_use]
    pub fn with_device(device: interop::wgpu::DeviceSource) -> Self {
        Self {
            device: Some(device),
            ..Self::default()
        }
    }
}

/// The surface targets [`Vello`] draws into: an [`Offscreen`] texture or an
/// [`interop::wgpu::Window`].
#[derive(Debug)]
pub enum VelloTarget {
    /// An offscreen texture.
    Offscreen(Offscreen),
    /// The embedder's window surface.
    Window(interop::wgpu::Window),
}

impl From<Offscreen> for VelloTarget {
    fn from(offscreen: Offscreen) -> Self {
        Self::Offscreen(offscreen)
    }
}

impl From<interop::wgpu::Window> for VelloTarget {
    fn from(window: interop::wgpu::Window) -> Self {
        Self::Window(window)
    }
}

/// The vello backend: renders the shared front end's layer trees through
/// `vello` on wgpu.
#[derive(Clone, Copy, Debug, Default)]
pub struct Vello;

impl Backend for Vello {
    type Config = VelloConfig;
    type Info = VelloInfo;
    type Target = VelloTarget;
    type Renderer = render::VelloRenderer;

    fn init(config: VelloConfig) -> Result<(Self::Renderer, Self::Info), EngineError> {
        render::init(config)
    }
}

impl ShaderPaintCapability for Vello {
    fn add_shader(
        r: &mut Self::Renderer,
        id: ShaderId,
        source: ShaderSource,
    ) -> Result<(), ResourceError> {
        r.shaders.add(
            &r.device,
            id.raw(),
            &render::ShaderSpec {
                source: source.source,
                animated: source.animated,
            },
        )
    }

    fn remove_shader(r: &mut Self::Renderer, id: ShaderId) {
        r.remove_shader(id);
    }
}

impl Filters for Vello {
    fn remove_filter(r: &mut Self::Renderer, id: FilterId) {
        r.remove_filter(id);
    }
}

impl<F: filtrate_core::Filter + Send> Runs<F> for Vello {
    fn add_filter(r: &mut <Self as Backend>::Renderer, id: FilterId, filter: F) {
        r.filters
            .add(id.raw(), Box::new(render::filter::FromFilter(filter)));
    }
}

impl Effects for Vello {
    type Effect = interop::EffectBox;

    fn add_effect(r: &mut <Self as Backend>::Renderer, id: FilterId, effect: Self::Effect) {
        r.filters.add(id.raw(), effect.inner);
    }
}

impl GpuContentCapability for Vello {
    type Content = interop::GpuContentBox;

    fn set_gpu_content(
        r: &mut Self::Renderer,
        surface: SurfaceId,
        layer: LayerId,
        size: (u32, u32),
        content: Self::Content,
    ) {
        r.set_gpu_content(surface, layer, size, content);
    }
}

impl Uploads<Rgba8> for Vello {}
