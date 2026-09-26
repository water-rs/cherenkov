// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Messages the UI thread sends to the render thread, and the change-set
//! types they carry. Everything crossing the channel is owned and `Send`;
//! there are no locks anywhere in the engine.

use std::sync::Arc;
use std::sync::mpsc::Sender;

use kurbo::{Affine, Vec2};

use crate::WorkingColor;
use crate::animation::Animation;
use crate::backend::{Backend, Display, SurfaceInfo};
use crate::config::{MemoryUsage, Pressure};
use crate::display_list::{Picture, SlotUpdate};
use crate::error::{RenderError, SurfaceError};
use crate::frame::{FrameStats, FrameTime, Next, Readback};
use crate::shape::ShapeData;
use crate::style::{BlendMode, FilterId};

/// A render-thread operation a capability method or a resource drop queues.
pub type ResOp<B> = Box<dyn FnOnce(&mut <B as Backend>::Renderer) + Send>;

/// Identifier of a surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceId(u64);

impl SurfaceId {
    /// Creates an identifier from a raw value.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw value.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Identifier of a layer within a surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayerId(u64);

impl LayerId {
    /// Creates an identifier from a raw value.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw value.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Identifier of a backdrop group. No backend implements
/// [`Backdrop`](crate::Backdrop) yet; the group API lands with the first
/// backend that does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BackdropId(u64);

impl BackdropId {
    /// Creates an identifier from a raw value.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw value.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// A font crossing to the render thread.
#[derive(Clone)]
pub struct FontData {
    /// The raw font data.
    pub data: Arc<[u8]>,
    /// The font index inside a collection.
    pub index: u32,
}

impl std::fmt::Debug for FontData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FontData")
            .field("len", &self.data.len())
            .field("index", &self.index)
            .finish()
    }
}

/// A property target plus the animation that reaches it.
#[derive(Clone, Debug)]
pub struct Prop<T> {
    /// The value the property moves to.
    pub target: T,
    /// The animation applied, if any; `None` snaps.
    pub animation: Option<Animation>,
}

/// What a layer draws, crossing the channel.
#[derive(Clone, Debug)]
pub enum ContentOp {
    /// The whole display list of a live content, sent on first commit.
    Replace(Picture),
    /// New values for a live content's bound slots.
    Update(Vec<SlotUpdate>),
    /// A shared immutable picture.
    Picture(Picture),
}

/// One layer mutation in a committed change set.
#[derive(Clone, Debug)]
pub enum LayerOp {
    /// Create a detached layer node.
    Create(LayerId),
    /// Remove a layer node and its descendants.
    Remove(LayerId),
    /// Set the local transform.
    Transform(LayerId, Prop<Affine>),
    /// Set the opacity.
    Opacity(LayerId, Prop<f32>),
    /// Set the scroll offset.
    ScrollOffset(LayerId, Prop<Vec2>),
    /// Set or clear the clip shape.
    Clip(LayerId, Option<ShapeData>),
    /// Set the blend mode.
    Blend(LayerId, BlendMode),
    /// Set or clear the filter.
    Filter(LayerId, Option<FilterId>),
    /// Set or clear the backdrop group. Part of the wire format; the
    /// front-end setter lands with the first [`Backdrop`](crate::Backdrop)
    /// backend.
    Backdrop(LayerId, Option<BackdropId>),
    /// Set the layer content, or clear it.
    Content(LayerId, Option<ContentOp>),
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

/// One committed op: a layer mutation, or an opaque render-side install a
/// capability method wrapped (GPU content, external frames) travelling in
/// order with the layer ops.
pub enum Op<B: Backend> {
    /// A layer-tree mutation.
    Layer(LayerOp),
    /// An opaque render-side operation, applied in order.
    Install(ResOp<B>),
}

/// The committed change set for one surface.
pub struct ChangeSet<B: Backend> {
    /// New clear colour, when set this commit.
    pub clear: Option<WorkingColor>,
    /// The ops, in order.
    pub ops: Vec<Op<B>>,
}

/// A message to the render thread.
pub enum Message<B: Backend> {
    /// Create a surface.
    CreateSurface {
        /// The new surface id.
        id: SurfaceId,
        /// What it renders into.
        target: B::Target,
        /// Result of the creation.
        reply: Sender<Result<SurfaceInfo, SurfaceError>>,
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
    /// Update a surface's display properties.
    Display {
        /// The surface id.
        id: SurfaceId,
        /// The new display properties.
        display: Display,
    },
    /// An opaque render-thread operation: resource registration and
    /// removal, capability hooks. Reply-carrying operations capture their
    /// `Sender` in the closure.
    Resource(ResOp<B>),
    /// Render every dirty surface for the frame at `time`, applying every
    /// surface's queued change set first.
    Render {
        /// The frame time.
        time: FrameTime,
        /// The surfaces' queued change sets, one entry per dirty surface.
        commits: Vec<(SurfaceId, ChangeSet<B>)>,
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
