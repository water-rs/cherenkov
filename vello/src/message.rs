// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Messages the UI thread sends to the render thread. Everything crossing the
//! channel is owned and `Send`; there are no locks anywhere in the engine.

use std::sync::Arc;
use std::sync::mpsc::Sender;

use cherenkov::kurbo::Affine;
use cherenkov::{BlendMode, ContentChange, FilterId, Picture, ShapeData, WorkingColor};
use vello::peniko;

use crate::error::{RenderError, SurfaceError};
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
    /// A window surface (phase 2; not yet drawable).
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
#[derive(Debug)]
pub enum LayerContentMsg {
    /// A shared immutable picture.
    Picture(Picture),
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
