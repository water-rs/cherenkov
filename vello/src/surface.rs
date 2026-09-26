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
use cherenkov::{BlendMode, Content, Picture, ShapeData, WorkingColor};

use crate::error::{RenderError, SurfaceError};
use crate::message::{
    ChangeSet, LayerContentMsg, LayerId, LayerOp, Message, SurfaceId, TargetSpec,
};

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
    pub(crate) pending: Vec<LayerOp>,
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
    pub(crate) fn take_changes(&mut self) -> Option<ChangeSet> {
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
    /// GPU-rendered content.
    Gpu(crate::GpuContentHandle),
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

impl From<crate::GpuContentHandle> for LayerContent {
    fn from(handle: crate::GpuContentHandle) -> Self {
        Self::Gpu(handle)
    }
}

/// An offscreen render target description.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Offscreen {
    /// The target size in pixels.
    pub size: (u32, u32),
    /// The refresh-rate range [`Next::At`] reports for this surface, in
    /// hertz. Defaults to `60..=60`; set [`Self::rate`].
    pub rate: RefreshRange,
}

impl Offscreen {
    /// An offscreen target of `size` pixels.
    ///
    /// The target is `Rgba8Unorm`: premultiplied sRGB-encoded sRGB.
    #[must_use]
    pub const fn new(size: (u32, u32)) -> Self {
        Self {
            size,
            rate: 60..=60,
        }
    }

    /// Sets the refresh-rate range this surface's [`Next::At`] reports.
    ///
    /// An offscreen target has no display to read a rate from, so the
    /// caller configures the range the host's frame scheduling supports.
    ///
    /// # Panics
    /// Panics on an empty range or a maximum of 0 Hz.
    #[must_use]
    pub fn rate(mut self, rate: RefreshRange) -> Self {
        assert!(
            !rate.is_empty() && *rate.end() > 0,
            "an offscreen refresh range must be non-empty and positive"
        );
        self.rate = rate;
        self
    }
}

/// The target of a [`Surface`]: an [`Offscreen`] texture or, in a later
/// slice, an [`interop::wgpu::Window`](crate::interop::wgpu::Window).
#[derive(Debug)]
pub struct Target(pub(crate) TargetSpec);

impl From<Offscreen> for Target {
    fn from(offscreen: Offscreen) -> Self {
        Self(TargetSpec::Offscreen {
            size: offscreen.size,
            rate: offscreen.rate,
        })
    }
}

impl From<crate::interop::wgpu::Window> for Target {
    fn from(window: crate::interop::wgpu::Window) -> Self {
        Self(TargetSpec::Window(Box::new(window)))
    }
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
    /// Render passes recorded (`render_to_texture` and effect passes).
    pub passes: u32,
    /// Scene commands encoded this frame.
    pub draws: u32,
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
        let mut shared = self.shared.borrow_mut();
        // A live `Content` lives in `contents`, not in the op stream:
        // without this the entry (and its signal subscriptions) would
        // survive the layer for the surface's lifetime.
        shared.contents.remove(&self.id);
        if self.remove_on_drop {
            shared.pending.push(LayerOp::Remove(self.id));
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
    /// Set the local transform.
    Transform(Affine),
    /// Set the opacity.
    Opacity(f32),
    /// Set or clear the clip shape.
    Clip(Option<ShapeData>),
    /// Set the blend mode.
    Blend(BlendMode),
    /// Set or clear the filter.
    Filter(Option<cherenkov::FilterId>),
    /// Set or clear the content.
    Content(LayerContent),
    /// Append a child.
    Push(LayerId),
    /// Insert a child at an index.
    Insert(usize, LayerId),
    /// Detach a child.
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

    /// Sets the blend mode the layer composites onto its parent with.
    pub fn blend(&mut self, blend: BlendMode) -> &mut Self {
        self.ops.push(EditOp::Blend(blend));
        self
    }

    /// Sets the filter applied to this layer's subtree.
    pub fn filter(&mut self, filter: &crate::Filter) -> &mut Self {
        self.ops.push(EditOp::Filter(Some(filter.id())));
        self
    }

    /// Clears the layer's filter.
    pub fn clear_filter(&mut self) -> &mut Self {
        self.ops.push(EditOp::Filter(None));
        self
    }

    /// Sets the content.
    pub fn content(&mut self, content: impl Into<LayerContent>) -> &mut Self {
        self.ops.push(EditOp::Content(content.into()));
        self
    }

    /// Clears the content.
    pub fn clear_content(&mut self) -> &mut Self {
        self.ops.push(EditOp::Content(LayerContent::None));
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

/// A surface: a render target plus its layer tree. `!Send`; dropping sends
/// [`Message::DestroySurface`].
#[derive(Debug)]
pub struct Surface {
    /// The surface's identifier on the render thread.
    pub id: SurfaceId,
    size: (u32, u32),
    readable: bool,
    root: Layer,
    /// The shared pending-changes state.
    pub shared: Rc<RefCell<SurfaceShared>>,
    tx: Sender<Message>,
}

impl Surface {
    /// Registers a surface on the render thread.
    ///
    /// # Errors
    /// [`SurfaceError::TooLarge`] when a dimension exceeds the device limit,
    /// [`SurfaceError::Unsupported`] when the target is not drawable yet,
    /// and [`SurfaceError::Lost`] when the render thread is gone.
    pub(crate) fn new(
        id: SurfaceId,
        target: Target,
        tx: Sender<Message>,
    ) -> Result<Self, SurfaceError> {
        let shared = Rc::new(RefCell::new(SurfaceShared {
            next_layer: Cell::new(1),
            ..SurfaceShared::default()
        }));
        let root = Layer {
            id: 0,
            shared: Rc::clone(&shared),
            remove_on_drop: false,
        };
        let (size, readable) = match &target.0 {
            TargetSpec::Offscreen { size, .. } => (*size, true),
            TargetSpec::Window(window) => ((window.config.width, window.config.height), false),
        };
        let (reply, rx) = std::sync::mpsc::channel();
        tx.send(Message::CreateSurface {
            id,
            target: target.0,
            reply,
        })
        .map_err(|_| SurfaceError::Lost)?;
        rx.recv().map_err(|_| SurfaceError::Lost)??;
        Ok(Self {
            id,
            size,
            readable,
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

    /// Resizes the surface. The render thread validates the size on receipt;
    /// an oversized size fails the next [`crate::Engine::render`].
    ///
    /// # Errors
    /// [`SurfaceError::Lost`] when the render thread is gone.
    pub fn resize(&mut self, size: (u32, u32)) -> Result<(), SurfaceError> {
        self.tx
            .send(Message::ResizeSurface { id: self.id, size })
            .map_err(|_| SurfaceError::Lost)?;
        self.size = size;
        Ok(())
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
                        EditOp::Blend(blend) => ops.push(LayerOp::Blend(id, blend)),
                        EditOp::Filter(filter) => ops.push(LayerOp::Filter(id, filter)),
                        EditOp::Content(LayerContent::Content(content)) => {
                            shared.contents.insert(id, content);
                            let stored = shared.contents.get_mut(&id).expect("just inserted");
                            if let Some(change) = stored.take_change() {
                                ops.push(LayerOp::ContentChange(id, change));
                            }
                        }
                        EditOp::Content(LayerContent::Picture(picture)) => {
                            shared.contents.remove(&id);
                            ops.push(LayerOp::Content(
                                id,
                                Some(LayerContentMsg::Picture(picture)),
                            ));
                        }
                        EditOp::Content(LayerContent::Gpu(mut handle)) => {
                            shared.contents.remove(&id);
                            let msg = crate::message::GpuContentMsg {
                                id: handle.id,
                                size: handle.size,
                                dirty: std::sync::Arc::clone(&handle.dirty),
                                content: handle
                                    .content
                                    .take()
                                    .expect("a GpuContentHandle is consumed once"),
                            };
                            ops.push(LayerOp::Content(id, Some(LayerContentMsg::Gpu(msg))));
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
    /// Only offscreen surfaces are readable.
    ///
    /// # Errors
    /// [`RenderError::NotReadable`] for a window surface,
    /// [`RenderError::Readback`] when the buffer map fails, or
    /// [`RenderError::Thread`] when the render thread is gone.
    pub fn readback(&self) -> Result<Readback, RenderError> {
        if !self.readable {
            return Err(RenderError::NotReadable);
        }
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

#[cfg(test)]
mod tests {
    use cherenkov::Draw as _;
    use cherenkov::kurbo::Rect;

    use super::*;

    /// A `Surface` backed by a stub thread that only answers
    /// `CreateSurface`; later messages accumulate on the channel.
    fn stub_surface() -> Surface {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            while let Ok(msg) = rx.recv() {
                if let Message::CreateSurface { reply, .. } = msg {
                    let _ = reply.send(Ok(()));
                }
            }
        });
        Surface::new(1, Offscreen::new((8, 8)).into(), tx).expect("surface")
    }

    /// Dropping a layer must drop its `contents` entry too: the recorded
    /// `Content` and its signal subscriptions must not outlive the layer.
    #[test]
    fn layer_drop_releases_its_content() {
        let surface = stub_surface();
        let layer = surface.layer();
        surface.update(|tx| {
            tx[&layer].content(surface.record(|c| {
                c.fill(Rect::new(0., 0., 4., 4.), WorkingColor::WHITE);
            }));
        });
        let id = layer.id;
        assert!(surface.shared.borrow().contents.contains_key(&id));
        drop(layer);
        assert!(
            !surface.shared.borrow().contents.contains_key(&id),
            "dropped layer's content must be released"
        );
    }
}
