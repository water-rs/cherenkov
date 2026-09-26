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
use crate::message::{LayerId, SurfaceId};
use crate::style::FilterId;

/// The backend draws user WGSL shader paints.
pub trait ShaderPaint: Backend {
    /// Registers a shader on the render thread.
    ///
    /// # Errors
    /// [`ResourceError::Shader`] when the source fails validation or
    /// pipeline creation.
    fn add_shader(
        r: &mut Self::Renderer,
        id: ShaderId,
        source: ShaderSource,
    ) -> Result<(), ResourceError>;
    /// Unregisters a shader.
    fn remove_shader(r: &mut Self::Renderer, id: ShaderId);
}

/// The backend runs filtrate filters.
pub trait Filters: Backend {
    /// Unregisters a filter or effect.
    fn remove_filter(r: &mut Self::Renderer, id: FilterId);
}

/// The backend can run the filtrate filter `F`.
pub trait Runs<F: filtrate_core::Filter + Send>: Filters {
    /// Registers a filter on the render thread.
    fn add_filter(r: &mut Self::Renderer, id: FilterId, filter: F);
}

/// The backend runs custom effects (`Box<dyn filtrate::Effect + Send>` on
/// GPU backends).
pub trait Effects: Filters {
    /// The effect payload type.
    type Effect: Send + 'static;
    /// Registers an effect on the render thread.
    fn add_effect(r: &mut Self::Renderer, id: FilterId, effect: Self::Effect);
}

/// The backend composites user GPU-rendered content as layer content.
pub trait GpuContent: Backend {
    /// The content payload type.
    type Content: Send + 'static;
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
    type Frame: Send + 'static;
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

/// The backend samples the backdrop behind a layer.
///
/// The `surface.backdrop_group` / `tx[&l].backdrop` API lands with the
/// first backend implementing this trait; [`BackdropId`](crate::BackdropId)
/// and the [`LayerNode::backdrop`](crate::LayerNode::backdrop) field exist
/// already.
pub trait Backdrop: Backend {}

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
