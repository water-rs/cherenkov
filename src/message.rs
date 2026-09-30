//! Messages the UI thread sends to the render thread, and the change-set
//! types they carry. Everything crossing the channel is owned and `Send`;
//! there are no locks anywhere in the engine.

#[cfg(target_arch = "wasm32")]
use crate::local::ReplySender as Sender;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{Sender, SyncSender};

#[cfg(target_arch = "wasm32")]
pub type FrameReplySender<T> = Sender<T>;
#[cfg(not(target_arch = "wasm32"))]
pub type FrameReplySender<T> = SyncSender<T>;

use kurbo::{Affine, Vec2};

use crate::WorkingColor;
use crate::animation::Animation;
use crate::backend::{Backend, Display, SurfaceInfo};
use crate::config::{MemoryUsage, Pressure};
use crate::display_list::{Picture, SlotUpdate};
use crate::error::{RenderError, ResourceError, SurfaceError};
use crate::frame::{FrameStats, FrameTime, FrameTiming, Next, Readback};
use crate::image::ImageUpload;
use crate::paint::ImageId;
use crate::shape::ShapeData;
use crate::style::{BlendMode, FilterId};

/// A render-thread operation a capability method or a resource drop queues.
#[cfg(not(target_arch = "wasm32"))]
pub type ResOp<B> = Box<dyn FnOnce(&mut <B as Backend>::Renderer) + Send>;
/// A resource operation that stays on the creating JS thread.
#[cfg(target_arch = "wasm32")]
pub type ResOp<B> = Box<dyn FnOnce(&mut <B as Backend>::Renderer)>;
#[cfg(target_arch = "wasm32")]
/// An asynchronous resource operation on the owning JS thread.
pub type AsyncResOp<B> = Box<
    dyn for<'a> FnOnce(
        &'a mut <B as Backend>::Renderer,
    ) -> core::pin::Pin<Box<dyn core::future::Future<Output = ()> + 'a>>,
>;

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

/// Identifier of a backdrop group, allocated per surface by
/// [`Surface::backdrop_group`](crate::Surface::backdrop_group).
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

/// Identifier of a backdrop effect shader, allocated by
/// [`Engine::backdrop_shader`](crate::Engine::backdrop_shader).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BackdropShaderId(u64);

impl BackdropShaderId {
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
    /// Sets the translation in local coordinates; initially zero.
    Translation(LayerId, Prop<Vec2>),
    /// Sets the unwrapped rotation angle in radians; initially zero.
    Rotation(LayerId, Prop<f64>),
    /// Sets the x/y scale factors; initially (1, 1).
    Scale(LayerId, Prop<Vec2>),
    /// Sets x/y skew angles in radians; initially zero.
    Skew(LayerId, Prop<Vec2>),
    /// Sets the local pivot for rotation, skew and scale; initially zero.
    Pivot(LayerId, Prop<Vec2>),
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
    /// Set or clear the backdrop sample (group and optional per-member
    /// effect).
    Backdrop(LayerId, Option<crate::BackdropSample>),
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
    /// Replaced pictures, cleared on the render thread, whose storage returns to the UI thread.
    pub recycled: Vec<(LayerId, Picture)>,
}

/// The render result and drained buffers returned to the UI thread.
pub struct RenderReply<B: Backend> {
    /// The result of rendering the frame.
    pub result: Result<(Next, FrameStats), RenderError>,
    /// The drained commits, including their reusable empty op vectors.
    pub commits: Vec<(SurfaceId, ChangeSet<B>)>,
    /// The persistent reply sender, returned so a disconnected render thread
    /// releases the receiver.
    #[cfg(not(target_arch = "wasm32"))]
    pub sender: FrameReplySender<Self>,
}

/// Memory usage and its persistent reply sender.
pub struct MemoryReply {
    /// The engine's current usage.
    pub usage: MemoryUsage,
    /// The persistent reply sender, returned so a disconnected render thread
    /// releases the receiver.
    #[cfg(not(target_arch = "wasm32"))]
    pub sender: FrameReplySender<Self>,
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
    /// Replace a registered image's pixels behind the same id, then mark
    /// changed every surface whose content samples the image.
    ReplaceImage {
        /// The image.
        id: ImageId,
        /// The new pixels.
        image: ImageUpload,
        /// Result of the replacement: whether any surface was marked
        /// changed, which is when the host needs a frame to show it.
        reply: Sender<Result<bool, ResourceError>>,
    },
    /// Browser operation awaiting local device work.
    #[cfg(target_arch = "wasm32")]
    AsyncResource(AsyncResOp<B>),
    /// Render every dirty surface for the frame at `time`, applying every
    /// surface's queued change set first.
    Render {
        /// The frame time.
        time: FrameTime,
        /// The surfaces' queued change sets, one entry per dirty surface.
        commits: Vec<(SurfaceId, ChangeSet<B>)>,
        /// The render result and the buffers returned to the UI thread.
        reply: FrameReplySender<RenderReply<B>>,
    },
    /// Wait for every outstanding frame timing and return it.
    FinishTimings {
        /// The timings, oldest first.
        reply: Sender<Result<Vec<FrameTiming>, RenderError>>,
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
        reply: FrameReplySender<MemoryReply>,
    },
    /// System memory pressure.
    Trim(Pressure),
    /// Stop the render thread.
    Shutdown,
}
