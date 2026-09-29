// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Capability traits on the backend type.
//!
//! A capability is not a marker: it declares the render-side hook the
//! render loop calls, so a backend without the capability has no code path
//! to reach, and no default or stub exists. `Engine`, `Surface` and
//! `Transaction` methods bounded by these traits wrap the hook into an
//! owned `FnOnce(&mut B::Renderer) + Send` op that travels in the commit
//! with the layer ops, in order.

use std::borrow::Cow;

use crate::ShaderId;
use crate::backend::Backend;
use crate::error::ResourceError;
use crate::image::Format;
use crate::message::{BackdropId, BackdropShaderId, LayerId, SurfaceId};
use crate::style::FilterId;

/// The backend draws user WGSL shader paints.
pub trait ShaderPaint: Backend {
    /// Registers a shader on the render thread.
    ///
    /// # Errors
    /// [`ResourceError::Shader`] when the source fails validation or
    /// pipeline creation.
    #[cfg(not(target_arch = "wasm32"))]
    fn add_shader(
        r: &mut Self::Renderer,
        id: ShaderId,
        source: ShaderSource,
    ) -> Result<(), ResourceError>;
    /// Validates shader registration without blocking the JS event loop.
    ///
    /// # Errors
    /// Returns shader validation errors.
    #[cfg(target_arch = "wasm32")]
    fn add_shader(
        r: &mut Self::Renderer,
        id: ShaderId,
        source: ShaderSource,
    ) -> impl core::future::Future<Output = Result<(), ResourceError>>;
    /// Unregisters a shader.
    fn remove_shader(r: &mut Self::Renderer, id: ShaderId);
}

/// The backend runs filtrate filters.
pub trait Filters: Backend {
    /// Unregisters a filter or effect.
    fn remove_filter(r: &mut Self::Renderer, id: FilterId);
}

/// The backend can run the filtrate filter `F`.
pub trait Runs<F: filtrate_core::Filter + crate::RenderTransfer>: Filters {
    /// Registers a filter on the render thread.
    fn add_filter(r: &mut Self::Renderer, id: FilterId, filter: F);
}

/// The backend runs custom effects (`Box<dyn filtrate::Effect + Send>` on
/// GPU backends).
pub trait Effects: Filters {
    /// The effect payload type.
    type Effect: crate::RenderTransfer + 'static;
    /// Registers an effect on the render thread.
    fn add_effect(r: &mut Self::Renderer, id: FilterId, effect: Self::Effect);
}

/// The backend composites user GPU-rendered content as layer content.
pub trait GpuContent: Backend {
    /// The content payload type.
    type Content: crate::RenderTransfer + 'static;
    /// Resizes an installed producer's attachment without repeating setup.
    /// Called in transaction order; the layer must contain GPU content.
    fn resize_gpu_content(
        r: &mut Self::Renderer,
        surface: SurfaceId,
        layer: LayerId,
        size: (u32, u32),
    );
    /// Attaches GPU content to a layer.
    fn set_gpu_content(
        r: &mut Self::Renderer,
        surface: SurfaceId,
        layer: LayerId,
        size: (u32, u32),
        content: Self::Content,
    );
}

/// The backend consumes externally produced frames (video, web views).
pub trait ExternalFrames: Backend {
    /// The frame payload type.
    type Frame: crate::RenderTransfer + 'static;
    /// Attaches an external frame to a layer.
    fn set_external_frame(
        r: &mut Self::Renderer,
        surface: SurfaceId,
        layer: LayerId,
        frame: Self::Frame,
    );
}

/// Which image storage formats `add_image` accepts: the backend uploads
/// [`ImageData<F>`](crate::ImageData).
pub trait Uploads<F: Format>: Backend {}

/// A filtrate chain a backdrop group can run, with its footprint bound.
///
/// Spatial chains report their own footprint; colour chains read no
/// neighbour texels and report [`Footprint::ZERO`](filtrate_core::Footprint::ZERO).
pub trait BackdropChain<K: filtrate_core::kind::Kind>: filtrate_core::Filter<Kind = K> {
    /// The chain's footprint for `params` (see
    /// [`SpatialFilter::footprint_of`](filtrate_core::SpatialFilter::footprint_of)).
    fn footprint_bound(params: &Self::Params) -> filtrate_core::Footprint;
}

impl<F: filtrate_core::SpatialFilter> BackdropChain<filtrate_core::kind::Spatial> for F {
    fn footprint_bound(params: &Self::Params) -> filtrate_core::Footprint {
        F::footprint_of(params)
    }
}

impl<F: filtrate_core::Filter<Kind = filtrate_core::kind::Color>>
    BackdropChain<filtrate_core::kind::Color> for F
{
    fn footprint_bound(_params: &Self::Params) -> filtrate_core::Footprint {
        filtrate_core::Footprint::ZERO
    }
}

/// The backend captures and samples backdrops
/// (`Surface::backdrop_group_unfiltered`, `LayerEdit::backdrop`).
pub trait Backdrop: Filters {
    /// Registers backdrop group `id` on `surface` with no filter chain.
    fn add_backdrop_group(r: &mut Self::Renderer, surface: SurfaceId, id: BackdropId);

    /// Unregisters a backdrop group; frames that still sample it fail.
    fn remove_backdrop_group(r: &mut Self::Renderer, surface: SurfaceId, id: BackdropId);
}

/// The backend can run the backdrop chain `F` of kind `K`
/// (`Surface::backdrop_group`).
pub trait BackdropRuns<K: filtrate_core::kind::Kind, F: BackdropChain<K> + crate::RenderTransfer>:
    Backdrop
{
    /// Registers backdrop group `id` on `surface` whose capture runs
    /// through `filter`.
    fn add_filtered_backdrop_group(
        r: &mut Self::Renderer,
        surface: SurfaceId,
        id: BackdropId,
        filter: F,
    );
}

/// The backend compiles per-member backdrop effect shaders
/// (`Engine::backdrop_shader`).
pub trait BackdropShaders: Backdrop {
    /// Registers a backdrop effect shader on the render thread, compiled
    /// for the composite contract; the pipeline is built here, never at
    /// draw time.
    ///
    /// # Errors
    /// [`ResourceError::Shader`] when the source fails validation or
    /// pipeline creation.
    #[cfg(not(target_arch = "wasm32"))]
    fn add_backdrop_shader(
        r: &mut Self::Renderer,
        id: BackdropShaderId,
        source: crate::BackdropShaderSource,
    ) -> Result<(), ResourceError>;

    /// Validates backdrop effect shader registration without blocking the
    /// JS event loop.
    ///
    /// # Errors
    /// Returns shader validation errors.
    #[cfg(target_arch = "wasm32")]
    fn add_backdrop_shader(
        r: &mut Self::Renderer,
        id: BackdropShaderId,
        source: crate::BackdropShaderSource,
    ) -> impl core::future::Future<Output = Result<(), ResourceError>>;

    /// Unregisters a backdrop effect shader; frames that still sample it
    /// fail.
    fn remove_backdrop_shader(r: &mut Self::Renderer, id: BackdropShaderId);
}

/// The backend produces HDR output.
pub trait HdrOutput: Backend {}

/// The backend presents on multiple hardware planes.
pub trait Planes: Backend {}

/// A user shader's WGSL fragment source.
#[derive(Clone, Debug)]
pub struct ShaderSource {
    /// The fragment source, without the backend's prelude.
    pub source: Cow<'static, str>,
    /// Whether the shader animates: when true, it is re-rendered every
    /// frame so its time uniform advances and the engine keeps refreshing.
    pub animated: bool,
}

impl ShaderSource {
    /// A static shader from a WGSL fragment body.
    pub fn wgsl(fragment: impl Into<Cow<'static, str>>) -> Self {
        Self {
            source: fragment.into(),
            animated: false,
        }
    }

    /// Marks the shader as animated (re-rendered each frame).
    #[must_use]
    pub fn animated(self) -> Self {
        Self {
            source: self.source,
            animated: true,
        }
    }
}
