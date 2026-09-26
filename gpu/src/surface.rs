// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Surfaces, layers and transactions: the UI-thread state machine.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, RangeInclusive};
use std::rc::Rc;
use std::sync::mpsc::Sender;
use std::time::Instant;

use cherenkov::kurbo::Affine;
use cherenkov::{Content, Picture, ShapeData, WorkingColor};

use crate::error::{RenderError, SurfaceError};
use crate::message::{ChangeSet, LayerId, LayerOp, Message, SurfaceId};

/// State shared between a [`Surface`], its [`Layer`] handles and the engine.
///
/// A layer's drop queues a [`LayerOp::Remove`] here; the next
/// [`Surface::update`] or [`crate::Engine::render`] flushes the pending ops
/// into a [`Message::Commit`]. Live [`Content`]s are kept per layer so their
/// signal-driven [`cherenkov::ContentChange`]s ship with every flush without
/// needing a transaction.
#[derive(Debug, Default)]
pub struct SurfaceShared {
    /// Ops queued outside transactions (layer creates and drops).
    pub pending: Vec<LayerOp>,
    /// Live contents per layer.
    pub contents: HashMap<LayerId, Content>,
    /// Pending clear colour.
    pub clear: Option<WorkingColor>,
    /// Layer id allocator (0 is the root).
    pub next_layer: Cell<LayerId>,
}

impl SurfaceShared {
    /// Drains pending ops and content changes into a change set. Returns
    /// `None` when nothing changed.
    pub fn take_changes(&mut self) -> Option<ChangeSet> {
        let mut ops = std::mem::take(&mut self.pending);
        for (id, content) in &mut self.contents {
            if let Some(change) = content.take_change() {
                ops.push(LayerOp::ContentChange(*id, change));
            }
        }
        let clear = self.clear.take();
        (clear.is_some() || !ops.is_empty()).then_some(ChangeSet { clear, ops })
    }
}

/// What a layer draws.
#[derive(Debug)]
pub enum LayerContent {
    /// Live recorded content.
    Content(Content),
    /// A shared immutable picture.
    Picture(Picture),
    /// Nothing.
    None,
}

impl From<Content> for LayerContent {
    fn from(content: Content) -> Self {
        Self::Content(content)
    }
}

impl From<Picture> for LayerContent {
    fn from(picture: Picture) -> Self {
        Self::Picture(picture)
    }
}

/// An offscreen render target description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Offscreen {
    /// The target size in pixels.
    pub size: (u32, u32),
    format: OffscreenFormat,
}

impl Offscreen {
    /// An offscreen target of `size` pixels in `format`.
    #[must_use]
    pub const fn new(size: (u32, u32), format: OffscreenFormat) -> Self {
        Self { size, format }
    }
}

/// The storage format of an offscreen target.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OffscreenFormat {
    /// Premultiplied linear Display P3, 16-bit float per channel.
    #[default]
    LinearF16,
}

/// The presentation timestamp handed to [`crate::Engine::render`].
#[derive(Clone, Copy, Debug)]
pub struct FrameTime(pub Instant);

impl FrameTime {
    /// The frame time at `t`.
    #[must_use]
    pub const fn at(t: Instant) -> Self {
        Self(t)
    }

    /// The frame time now.
    #[must_use]
    pub fn now() -> Self {
        Self(Instant::now())
    }
}

/// An inclusive refresh-rate range in hertz.
pub type RefreshRange = RangeInclusive<u32>;

/// What the engine needs next, returned by [`crate::Engine::render`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Next {
    /// No animation is running; the display link may sleep.
    Idle,
    /// The next frame is needed at `time`, at a refresh rate in `rate`.
    At {
        /// When the next frame is due.
        time: Instant,
        /// The acceptable refresh rates.
        rate: RefreshRange,
    },
}

/// Measurements of the last [`crate::Engine::render`].
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    /// GPU seconds the frame took, when timestamp queries are enabled and
    /// supported.
    pub gpu_seconds: Option<f64>,
    /// Render passes recorded.
    pub passes: u32,
    /// Draw calls issued.
    pub draws: u32,
    /// Quads drawn.
    pub instances: u32,
    /// Glyphs rasterized into the atlas this frame.
    pub glyphs_rasterized: u32,
}

/// Decoded pixels of a surface readback: premultiplied linear Display P3,
/// row-major.
#[derive(Clone, Debug)]
pub struct Readback {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width * height` premultiplied RGBA pixels.
    pub pixels: Vec<[f32; 4]>,
}

/// A layer handle. Layers are `!Send` and live on the UI thread; dropping one
/// removes it from its surface at the next commit.
#[derive(Debug)]
pub struct Layer {
    id: LayerId,
    shared: Rc<RefCell<SurfaceShared>>,
    remove_on_drop: bool,
}

impl Layer {
    fn new_detached(shared: &Rc<RefCell<SurfaceShared>>) -> Self {
        let shared_ref = shared.borrow();
        let id = shared_ref.next_layer.get();
        shared_ref.next_layer.set(id + 1);
        drop(shared_ref);
        shared.borrow_mut().pending.push(LayerOp::Create(id));
        Self {
            id,
            shared: Rc::clone(shared),
            remove_on_drop: true,
        }
    }
}

impl Drop for Layer {
    fn drop(&mut self) {
        if self.remove_on_drop {
            self.shared
                .borrow_mut()
                .pending
                .push(LayerOp::Remove(self.id));
        }
    }
}

/// One layer's pending edits, collected inside a [`Transaction`]. Each method
/// records an op and returns `&mut Self` for chaining.
#[derive(Debug, Default)]
pub struct LayerEdit {
    /// The recorded edit ops, applied in order.
    pub ops: Vec<EditOp>,
}

/// A recorded layer edit.
#[derive(Debug)]
pub enum EditOp {
    Transform(Affine),
    Opacity(f32),
    Clip(Option<ShapeData>),
    Content(LayerContent),
    Push(LayerId),
    Insert(usize, LayerId),
    Detach(LayerId),
}

impl LayerEdit {
    /// Sets the local transform.
    pub fn transform(&mut self, t: Affine) -> &mut Self {
        self.ops.push(EditOp::Transform(t));
        self
    }

    /// Sets the opacity.
    pub fn opacity(&mut self, o: f32) -> &mut Self {
        self.ops.push(EditOp::Opacity(o));
        self
    }

    /// Sets the clip shape.
    #[expect(clippy::needless_pass_by_value, reason = "shapes are Copy")]
    pub fn clip<S: cherenkov::Shape>(&mut self, shape: S) -> &mut Self {
        self.ops.push(EditOp::Clip(Some(ShapeData::of(&shape))));
        self
    }

    /// Clears the clip.
    pub fn clear_clip(&mut self) -> &mut Self {
        self.ops.push(EditOp::Clip(None));
        self
    }

    /// Sets the content.
    pub fn content(&mut self, content: impl Into<LayerContent>) -> &mut Self {
        self.ops.push(EditOp::Content(content.into()));
        self
    }

    /// Appends a child layer.
    pub fn push(&mut self, child: &Layer) -> &mut Self {
        self.ops.push(EditOp::Push(child.id));
        self
    }

    /// Inserts a child layer at `index`.
    pub fn insert(&mut self, index: usize, child: &Layer) -> &mut Self {
        self.ops.push(EditOp::Insert(index, child.id));
        self
    }

    /// Removes a child layer.
    pub fn remove(&mut self, child: &Layer) -> &mut Self {
        self.ops.push(EditOp::Detach(child.id));
        self
    }
}

/// A transaction's edits to a surface's layer tree. `tx[&layer]` returns the
/// [`LayerEdit`] accumulating that layer's changes.
#[derive(Debug, Default)]
pub struct Transaction<'a> {
    edits: Vec<(LayerId, LayerEdit)>,
    _surface: PhantomData<&'a Surface>,
}

impl Transaction<'_> {
    fn edit(&mut self, layer: &Layer) -> &mut LayerEdit {
        let id = layer.id;
        let index = self.edits.iter().position(|(l, _)| *l == id);
        if let Some(index) = index {
            &mut self.edits[index].1
        } else {
            self.edits.push((id, LayerEdit::default()));
            &mut self.edits.last_mut().expect("just pushed").1
        }
    }
}

impl Index<&Layer> for Transaction<'_> {
    type Output = LayerEdit;

    fn index(&self, layer: &Layer) -> &Self::Output {
        self.edits
            .iter()
            .find(|(id, _)| *id == layer.id)
            .map(|(_, edit)| edit)
            .expect("the layer has no edits in this transaction yet")
    }
}

impl IndexMut<&Layer> for Transaction<'_> {
    fn index_mut(&mut self, layer: &Layer) -> &mut Self::Output {
        self.edit(layer)
    }
}

/// An offscreen surface: a render target plus its layer tree. `!Send`;
/// dropping sends [`Message::DestroySurface`].
#[derive(Debug)]
pub struct Surface {
    /// The surface's identifier on the render thread.
    pub id: SurfaceId,
    size: (u32, u32),
    root: Layer,
    /// The shared pending-changes state.
    pub shared: Rc<RefCell<SurfaceShared>>,
    tx: Sender<Message>,
}

impl Surface {
    /// Registers a surface of `size` pixels with the render thread.
    ///
    /// # Errors
    /// [`SurfaceError::TooLarge`] when a dimension exceeds the device limit
    /// and [`SurfaceError::Lost`] when the render thread is gone.
    pub fn new(id: SurfaceId, size: (u32, u32), tx: Sender<Message>) -> Result<Self, SurfaceError> {
        let shared = Rc::new(RefCell::new(SurfaceShared {
            next_layer: Cell::new(1),
            ..SurfaceShared::default()
        }));
        let root = Layer {
            id: 0,
            shared: Rc::clone(&shared),
            remove_on_drop: false,
        };
        let (reply, rx) = std::sync::mpsc::channel();
        tx.send(Message::CreateSurface { id, size, reply })
            .map_err(|_| SurfaceError::Lost)?;
        rx.recv().map_err(|_| SurfaceError::Lost)??;
        Ok(Self {
            id,
            size,
            root,
            shared,
            tx,
        })
    }

    /// The root layer.
    #[must_use]
    pub const fn root(&self) -> &Layer {
        &self.root
    }

    /// A new detached layer.
    #[must_use]
    pub fn layer(&self) -> Layer {
        Layer::new_detached(&self.shared)
    }

    /// The surface size in pixels.
    #[must_use]
    pub const fn size(&self) -> (u32, u32) {
        self.size
    }

    /// The clear colour, queued into the pending change set. Defaults to
    /// transparent.
    pub fn clear_color(&self, color: WorkingColor) {
        self.shared.borrow_mut().clear = Some(color);
    }

    /// Records live content for this surface.
    #[must_use]
    pub fn record(&self, body: impl FnOnce(&mut cherenkov::Recorder)) -> Content {
        Content::record(body)
    }

    /// Commits a transaction: builds one change set and sends it to the
    /// render thread in a single message.
    ///
    /// # Panics
    /// Panics if `body` panics; the transaction is then dropped unapplied.
    pub fn update(&self, body: impl FnOnce(&mut Transaction<'_>)) {
        let mut tx = Transaction {
            edits: Vec::new(),
            _surface: PhantomData,
        };
        body(&mut tx);
        let mut ops = self.shared.borrow_mut().pending.split_off(0);
        {
            let mut shared = self.shared.borrow_mut();
            for (id, edit) in tx.edits {
                for op in edit.ops {
                    match op {
                        EditOp::Transform(t) => ops.push(LayerOp::Transform(id, t)),
                        EditOp::Opacity(o) => ops.push(LayerOp::Opacity(id, o)),
                        EditOp::Clip(shape) => ops.push(LayerOp::Clip(id, shape)),
                        EditOp::Content(LayerContent::Content(content)) => {
                            shared.contents.insert(id, content);
                            let stored = shared.contents.get_mut(&id).expect("just inserted");
                            if let Some(change) = stored.take_change() {
                                ops.push(LayerOp::ContentChange(id, change));
                            }
                        }
                        EditOp::Content(LayerContent::Picture(picture)) => {
                            shared.contents.remove(&id);
                            ops.push(LayerOp::Content(id, Some(picture)));
                        }
                        EditOp::Content(LayerContent::None) => {
                            shared.contents.remove(&id);
                            ops.push(LayerOp::Content(id, None));
                        }
                        EditOp::Push(child) => ops.push(LayerOp::Push { parent: id, child }),
                        EditOp::Insert(index, child) => ops.push(LayerOp::Insert {
                            parent: id,
                            index,
                            child,
                        }),
                        EditOp::Detach(child) => ops.push(LayerOp::Detach { parent: id, child }),
                    }
                }
            }
            for (id, content) in &mut shared.contents {
                if let Some(change) = content.take_change() {
                    ops.push(LayerOp::ContentChange(*id, change));
                }
            }
        }
        let clear = self.shared.borrow_mut().clear.take();
        let _ = self.tx.send(Message::Commit {
            surface: self.id,
            changes: ChangeSet { clear, ops },
        });
    }

    /// The pixels of the surface after the last [`crate::Engine::render`].
    ///
    /// # Errors
    /// [`RenderError::Readback`] when the buffer map fails, or
    /// [`RenderError::Thread`] when the render thread is gone.
    pub fn readback(&self) -> Result<Readback, RenderError> {
        let (reply, rx) = std::sync::mpsc::channel();
        self.tx
            .send(Message::Readback {
                surface: self.id,
                reply,
            })
            .map_err(|_| RenderError::Thread)?;
        rx.recv().map_err(|_| RenderError::Thread)?
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        let _ = self.tx.send(Message::DestroySurface { id: self.id });
    }
}
