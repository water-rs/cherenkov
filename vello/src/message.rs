// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Messages the UI thread sends to the render thread. Everything crossing the
//! channel is owned and `Send`; there are no locks anywhere in the engine.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;

use cherenkov::kurbo::Affine;
use cherenkov::{BlendMode, ContentChange, FilterId, Picture, ShapeData, WorkingColor};
use vello::peniko;

use crate::error::{RenderError, ResourceError, SurfaceError};
use crate::surface::{FrameStats, FrameTime, Next, Offscreen, Readback};
use crate::{MemoryUsage, Pressure};

/// `Arc<[u8]>` wrapped so it coerces into `peniko::Blob::new`'s
/// `Arc<dyn AsRef<[u8]> + Send + Sync>` (an `Arc<[u8]>` cannot unsize to a
/// trait object directly).
pub struct SharedBytes(pub Arc<[u8]>);

impl AsRef<[u8]> for SharedBytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Identifier of a surface.
pub type SurfaceId = u64;
/// Identifier of a layer within a surface.
pub type LayerId = u64;

/// What a [`Surface`](crate::Surface) renders into.
#[derive(Debug)]
pub enum TargetSpec {
    /// An offscreen texture.
    Offscreen {
        /// Size in pixels.
        size: (u32, u32),
    },
    /// A window surface.
    Window(Box<crate::interop::wgpu::Window>),
}

impl From<Offscreen> for TargetSpec {
    fn from(offscreen: Offscreen) -> Self {
        Self::Offscreen {
            size: offscreen.size,
        }
    }
}

impl From<crate::interop::wgpu::Window> for TargetSpec {
    fn from(window: crate::interop::wgpu::Window) -> Self {
        Self::Window(Box::new(window))
    }
}

/// What a layer draws, crossing the channel.
pub enum LayerContentMsg {
    /// A shared immutable picture.
    Picture(Picture),
    /// GPU-rendered content retained on the render thread.
    Gpu(GpuContentMsg),
}

impl std::fmt::Debug for LayerContentMsg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Picture(_) => f.write_str("Picture(..)"),
            Self::Gpu(g) => f
                .debug_struct("Gpu")
                .field("id", &g.id)
                .field("size", &g.size)
                .finish(),
        }
    }
}

/// A [`GpuContentHandle`](crate::GpuContentHandle) crossing to the render
/// thread.
pub struct GpuContentMsg {
    /// The content's engine-wide id.
    pub id: u64,
    /// Content size in pixels.
    pub size: (u32, u32),
    /// The shared redraw flag.
    pub dirty: Arc<AtomicBool>,
    /// The content object.
    pub content: Box<dyn crate::gpu_content::AnyGpuContent>,
}

/// A shader's WGSL fragment source, crossing to the render thread.
#[derive(Debug)]
pub struct ShaderSpec {
    /// The fragment source (without the prelude).
    pub source: std::borrow::Cow<'static, str>,
    /// Whether the shader is re-rendered every frame (`time` uniform).
    pub animated: bool,
}

/// One layer mutation in a committed change set.
#[derive(Debug)]
pub enum LayerOp {
    /// Create a detached layer node.
    Create(LayerId),
    /// Remove a layer node and its descendants.
    Remove(LayerId),
    /// Set the local transform.
    Transform(LayerId, Affine),
    /// Set the opacity.
    Opacity(LayerId, f32),
    /// Set or clear the clip shape.
    Clip(LayerId, Option<ShapeData>),
    /// Set the blend mode.
    Blend(LayerId, BlendMode),
    /// Set or clear the filter.
    Filter(LayerId, Option<FilterId>),
    /// Set the layer content, or clear it.
    Content(LayerId, Option<LayerContentMsg>),
    /// A live content's change: the initial `Replace` or later slot updates.
    ContentChange(LayerId, ContentChange),
    /// Append a child.
    Push {
        /// The parent.
        parent: LayerId,
        /// The child.
        child: LayerId,
    },
    /// Insert a child at an index.
    Insert {
        /// The parent.
        parent: LayerId,
        /// Child index.
        index: usize,
        /// The child.
        child: LayerId,
    },
    /// Remove a child from a parent's child list.
    Detach {
        /// The parent.
        parent: LayerId,
        /// The child.
        child: LayerId,
    },
}

/// The committed change set for one surface.
#[derive(Debug)]
pub struct ChangeSet {
    /// New clear colour, when set this commit.
    pub clear: Option<WorkingColor>,
    /// The layer mutations, in order.
    pub ops: Vec<LayerOp>,
}

impl std::fmt::Debug for dyn crate::render::filter::FilterSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FilterSource(..)")
    }
}

/// A message to the render thread.
#[derive(Debug)]
pub enum Message {
    /// Create a surface.
    CreateSurface {
        /// The new surface id.
        id: SurfaceId,
        /// What it renders into.
        target: TargetSpec,
        /// Result of the creation.
        reply: Sender<Result<(), SurfaceError>>,
    },
    /// Resize a surface.
    ResizeSurface {
        /// The surface id.
        id: SurfaceId,
        /// New size in pixels.
        size: (u32, u32),
    },
    /// Destroy a surface and its layer tree.
    DestroySurface {
        /// The surface id.
        id: SurfaceId,
    },
    /// Register a font.
    AddFont {
        /// The font id (`FontId::raw`).
        id: u64,
        /// The font file data.
        data: Arc<[u8]>,
        /// Font index inside a collection.
        index: u32,
    },
    /// Unregister a font.
    RemoveFont {
        /// The font id.
        id: u64,
    },
    /// Register an image.
    AddImage {
        /// The image id (`ImageId::raw`).
        id: u64,
        /// The decoded image.
        image: peniko::ImageData,
    },
    /// Unregister an image.
    RemoveImage {
        /// The image id.
        id: u64,
    },
    /// Register a shader and build its pipeline.
    AddShader {
        /// The shader id (`ShaderId::raw`).
        id: u64,
        /// The fragment source.
        source: ShaderSpec,
        /// Validation result.
        reply: Sender<Result<(), ResourceError>>,
    },
    /// Unregister a shader.
    RemoveShader {
        /// The shader id.
        id: u64,
    },
    /// Register a filter effect (built into an executor/effect on the
    /// render thread: `filtrate::Executor` is not `Send`).
    AddFilter {
        /// The filter id (`FilterId::raw`).
        id: u64,
        /// The erased filter or effect.
        source: Box<dyn crate::render::filter::FilterSource>,
    },
    /// Unregister a filter.
    RemoveFilter {
        /// The filter id.
        id: u64,
    },
    /// Commit a surface's change set.
    Commit {
        /// The surface.
        surface: SurfaceId,
        /// The changes.
        changes: ChangeSet,
    },
    /// Render every dirty surface for the frame at `time`.
    Render {
        /// The frame time.
        time: FrameTime,
        /// What the next frame needs and this frame's stats.
        reply: Sender<Result<(Next, FrameStats), RenderError>>,
    },
    /// Read back a surface's pixels.
    Readback {
        /// The surface.
        surface: SurfaceId,
        /// The decoded pixels.
        reply: Sender<Result<Readback, RenderError>>,
    },
    /// Report memory usage.
    Memory {
        /// The usage.
        reply: Sender<MemoryUsage>,
    },
    /// System memory pressure.
    Trim(Pressure),
    /// Stop the render thread.
    Shutdown,
}
