// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The render-thread contract every backend implements.
//!
//! The `cherenkov` crate owns the whole front end and the render thread's
//! loop. A backend crate supplies the render side only: a config type, a
//! [`Backend`] implementation, its capability implementations and an
//! `interop` module.

use crate::WorkingColor;
use crate::config::{MemoryUsage, Pressure};
use crate::error::{EngineError, RenderError, ResourceError, SurfaceError};
use crate::frame::{FrameStats, FrameTime, Readback};
use crate::glyph::FontId;
use crate::image::ImageUpload;
use crate::message::{ContentOp, FontData, LayerId, SurfaceId};
use crate::paint::ImageId;
use crate::tree::SurfaceTree;

/// The render-thread contract. Implemented by a zero-sized marker type
/// (`Gpu`, `Vello`, `Raster`).
pub trait Backend: Sized + 'static {
    /// The backend's configuration type.
    type Config: Send + 'static;
    /// Provenance for reports.
    type Info: Clone + Send + 'static;
    /// A surface target: [`Offscreen`](crate::Offscreen) or an interop
    /// window target.
    type Target: From<crate::Offscreen> + Send + 'static;
    /// The render-thread state; never leaves that thread.
    type Renderer: Renderer<Target = Self::Target>;

    /// Runs on the render thread, once. Creates the device or worker pool.
    ///
    /// # Errors
    /// [`EngineError`] when the device or pool cannot be created.
    fn init(config: Self::Config) -> Result<(Self::Renderer, Self::Info), EngineError>;
}

/// Everything the render loop asks of a backend. Every method runs on the
/// render thread.
pub trait Renderer: 'static {
    /// A surface target.
    type Target;

    /// Creates the render-side state for surface `id`.
    ///
    /// # Errors
    /// [`SurfaceError`] when the target cannot be drawn.
    fn create_surface(
        &mut self,
        id: SurfaceId,
        target: Self::Target,
    ) -> Result<SurfaceInfo, SurfaceError>;

    /// Resizes a surface's target.
    fn resize_surface(&mut self, id: SurfaceId, size: (u32, u32));

    /// Destroys a surface's render-side state.
    fn destroy_surface(&mut self, id: SurfaceId);

    /// Registers a font.
    ///
    /// # Errors
    /// [`ResourceError`] when the data cannot be used.
    fn add_font(&mut self, id: FontId, font: FontData) -> Result<(), ResourceError>;

    /// Unregisters a font.
    fn remove_font(&mut self, id: FontId);

    /// Registers an image.
    ///
    /// # Errors
    /// [`ResourceError`] when the upload cannot be used.
    fn add_image(&mut self, id: ImageId, image: ImageUpload) -> Result<(), ResourceError>;

    /// Unregisters an image.
    fn remove_image(&mut self, id: ImageId);

    /// Replaces or updates a layer's recorded content, or clears it.
    fn set_content(&mut self, surface: SurfaceId, layer: LayerId, content: Option<ContentOp>);

    /// The layer is gone: drop every cache keyed on it.
    fn remove_layer(&mut self, surface: SurfaceId, layer: LayerId);

    /// Renders every surface in `frame` whose tree or content changed;
    /// returns whether a backend-side source (custom GPU content, an
    /// animated shader) wants another frame.
    ///
    /// # Errors
    /// [`RenderError`] fails the whole `render` call.
    fn render(&mut self, frame: &Frame<'_>, stats: &mut FrameStats) -> Result<Redraw, RenderError>;

    /// Reads back a surface's pixels.
    ///
    /// # Errors
    /// [`RenderError::NotReadable`] for non-readable surfaces,
    /// [`RenderError::Readback`] on failure.
    fn readback(&mut self, surface: SurfaceId) -> Result<Readback, RenderError>;

    /// The backend's current memory usage.
    fn memory(&self) -> MemoryUsage;

    /// Releases memory under system `pressure`.
    fn trim(&mut self, pressure: Pressure);
}

/// The render-side facts of a surface, answered by
/// [`Renderer::create_surface`].
#[derive(Clone, Copy, Debug)]
pub struct SurfaceInfo {
    /// The drawable size in pixels.
    pub size: (u32, u32),
    /// Whether [`Renderer::readback`] works on the surface.
    pub readable: bool,
}

/// Whether a backend wants another frame after the current one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Redraw {
    /// Nothing backend-side is animated.
    None,
    /// A backend source (custom GPU content, an animated shader paint)
    /// wants the next frame.
    Wanted,
}

/// One frame's render input: every live surface with its sampled tree.
pub struct Frame<'a> {
    /// The frame's presentation time.
    pub time: FrameTime,
    /// The surfaces to consider; render those with `changed` set.
    pub surfaces: &'a [SurfaceFrame<'a>],
}

/// One surface's input to [`Renderer::render`].
pub struct SurfaceFrame<'a> {
    /// The surface id.
    pub id: SurfaceId,
    /// The drawable size in pixels.
    pub size: (u32, u32),
    /// The display properties.
    pub display: Display,
    /// The clear colour.
    pub clear: WorkingColor,
    /// Whether a property op, a content op or an animation step touched the
    /// surface since the last render.
    pub changed: bool,
    /// The sampled layer tree.
    pub tree: &'a SurfaceTree,
}

/// The properties of the display a surface presents on. Headroom and scale
/// belong to the display, so the host sets them when they change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Display {
    /// Device pixels per logical pixel.
    pub scale: f64,
    /// HDR headroom: the ratio of peak white to SDR white.
    pub headroom: f32,
}

impl Default for Display {
    fn default() -> Self {
        Self {
            scale: 1.0,
            headroom: 1.0,
        }
    }
}
