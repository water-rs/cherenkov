// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Surfaces, layers and transactions: the UI-thread state machine.
//!
//! `Surface::update`, layer drops and bound-signal changes only queue
//! owned ops; [`Engine::render`](crate::Engine::render) drains every
//! surface's queue into one [`Message::Render`], so the render thread
//! wakes once per frame.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::rc::Rc;
use std::sync::mpsc::Sender;

use kurbo::{Affine, Vec2};
use nami_core::watcher::Context;

use crate::animation::Animation;
use crate::backend::{Backend, Display, SurfaceInfo};
use crate::capability::{ExternalFrames, GpuContent};
use crate::engine::Waker;
use crate::error::{RenderError, SurfaceError};
use crate::frame::Readback;
use crate::message::{ChangeSet, ContentOp, LayerId, LayerOp, Message, Op, Prop, SurfaceId};
use crate::record::{Content, Live};
use crate::shape::{Shape, ShapeData};
use crate::style::{BlendMode, FilterId};
use crate::{ContentChange, Picture, WorkingColor};

/// Which bound property a subscription updates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PropKind {
    Transform,
    Opacity,
    ScrollOffset,
    Clip,
}

/// State shared between a [`Surface`], its [`Layer`] handles and the
/// engine.
///
/// Everything here is queued, not sent: [`Engine::render`](crate::Engine::render)
/// drains `pending` into the single per-frame [`Message::Render`].
pub struct Shared<B: Backend> {
    /// The surface's identifier on the render thread.
    pub id: SurfaceId,
    /// Ops queued outside transactions (layer creates and drops, bound
    /// signal changes) plus queued transaction ops.
    pending: Vec<Op<B>>,
    /// Live contents per layer.
    contents: HashMap<LayerId, Content>,
    /// Pending clear colour.
    clear: Option<WorkingColor>,
    /// Layer id allocator (0 is the root).
    next_layer: Cell<u64>,
    /// Live property subscriptions, keyed by layer and property. Binding a
    /// property replaces its previous subscription; dropping a layer drops
    /// them all.
    bindings: HashMap<(u64, PropKind), Box<dyn Any>>,
    /// The engine wake-up, fired when an op is queued outside a frame.
    waker: Rc<Waker>,
    /// The last display properties announced to the render thread.
    display: Cell<Display>,
}

impl<B: Backend> std::fmt::Debug for Shared<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shared")
            .field("id", &self.id)
            .field("pending", &self.pending.len())
            .finish_non_exhaustive()
    }
}

impl<B: Backend> Shared<B> {
    fn new(id: SurfaceId, waker: Rc<Waker>) -> Self {
        Self {
            id,
            pending: Vec::new(),
            contents: HashMap::new(),
            clear: None,
            next_layer: Cell::new(1),
            bindings: HashMap::new(),
            waker,
            display: Cell::new(Display::default()),
        }
    }

    /// Queues an op and pokes the waker.
    fn push(&mut self, op: Op<B>) {
        self.pending.push(op);
        self.waker.wake();
    }

    /// Drains pending ops and live content changes into a change set.
    /// Returns `None` when nothing changed.
    pub fn take_changes(&mut self) -> Option<ChangeSet<B>> {
        let mut ops = std::mem::take(&mut self.pending);
        for (id, content) in &mut self.contents {
            if let Some(change) = content.take_change() {
                ops.push(Op::Layer(LayerOp::Content(
                    *id,
                    Some(match change {
                        ContentChange::Replace(list) => ContentOp::Replace(list),
                        ContentChange::Update(updates) => ContentOp::Update(updates),
                    }),
                )));
            }
        }
        let clear = self.clear.take();
        (clear.is_some() || !ops.is_empty()).then_some(ChangeSet { clear, ops })
    }

    /// Binds `subscribe` so changes queue `op(layer, value, animation)` and
    /// fire the waker. Replaces the property's previous binding.
    fn bind<T, F>(
        shared: &Rc<RefCell<Self>>,
        layer: LayerId,
        kind: PropKind,
        subscribe: crate::record::Subscribe<T>,
        op: F,
    ) where
        T: 'static,
        F: Fn(LayerId, T, Option<Animation>) -> LayerOp + 'static,
    {
        let weak = Rc::downgrade(shared);
        let waker = Rc::clone(&shared.borrow().waker);
        let guard = subscribe(Box::new(move |context: Context<T>| {
            let animation = context.metadata().try_get::<Animation>();
            let target = context.into_value();
            if let Some(shared) = weak.upgrade() {
                shared
                    .borrow_mut()
                    .pending
                    .push(Op::Layer(op(layer, target, animation)));
                waker.wake();
            }
        }));
        let mut shared_mut = shared.borrow_mut();
        if let Some(guard) = guard {
            shared_mut.bindings.insert((layer.raw(), kind), guard);
        } else {
            // Nothing to keep alive, but a previous binding is still
            // replaced by this subscription.
            shared_mut.bindings.remove(&(layer.raw(), kind));
        }
    }
}

/// A surface's bookkeeping shared with its [`Layer`] handles.
pub trait LayerOwner {
    /// Allocates a layer id and queues its `Create`.
    fn allocate(&self) -> LayerId;
    /// Queues a `Remove` and drops the layer's bindings and contents.
    fn remove(&self, id: LayerId);
}

impl<B: Backend> LayerOwner for RefCell<Shared<B>> {
    fn allocate(&self) -> LayerId {
        let mut shared = self.borrow_mut();
        let id = LayerId::new(shared.next_layer.get());
        shared.next_layer.set(id.raw() + 1);
        shared.push(Op::Layer(LayerOp::Create(id)));
        id
    }

    fn remove(&self, id: LayerId) {
        let mut shared = self.borrow_mut();
        shared.bindings.retain(|(layer, _), _| *layer != id.raw());
        shared.contents.remove(&id);
        shared.push(Op::Layer(LayerOp::Remove(id)));
    }
}

/// A layer handle. Layers are `!Send` and not `Clone`; dropping one removes
/// it from its surface at the next commit.
pub struct Layer {
    id: LayerId,
    owner: Rc<dyn LayerOwner>,
    remove_on_drop: bool,
}

impl std::fmt::Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layer")
            .field("id", &self.id)
            .field("remove_on_drop", &self.remove_on_drop)
            .finish_non_exhaustive()
    }
}

impl Layer {
    /// The layer's id.
    #[must_use]
    pub const fn id(&self) -> LayerId {
        self.id
    }
}

impl Drop for Layer {
    fn drop(&mut self) {
        if self.remove_on_drop {
            self.owner.remove(self.id);
        }
    }
}

/// What a layer draws.
pub enum LayerContent<B: Backend> {
    /// Live recorded content; its bound signals keep updating it with no
    /// further transactions.
    Content(Content),
    /// A shared immutable picture.
    Picture(Picture),
    /// An opaque render-side install (GPU content, external frames). The
    /// closure learns the surface and layer it is installed on when the
    /// edit is applied in `update`.
    Install(InstallOp<B>),
    /// Nothing.
    None,
}

impl<B: Backend> From<Content> for LayerContent<B> {
    fn from(content: Content) -> Self {
        Self::Content(content)
    }
}

impl<B: Backend> From<Picture> for LayerContent<B> {
    fn from(picture: Picture) -> Self {
        Self::Picture(picture)
    }
}

/// Custom GPU content of a fixed size, attachable to a layer. Created by
/// [`Engine::gpu_content`](crate::Engine::gpu_content). Not `Clone`.
pub struct GpuContentHandle<B: GpuContent> {
    /// The content size in pixels.
    pub size: (u32, u32),
    /// The content object.
    pub(crate) content: B::Content,
}

impl<B: GpuContent> From<GpuContentHandle<B>> for LayerContent<B> {
    fn from(handle: GpuContentHandle<B>) -> Self {
        let GpuContentHandle { size, content } = handle;
        Self::Install(Box::new(move |r, surface, layer| {
            B::set_gpu_content(r, surface, layer, size, content);
        }))
    }
}

/// An externally produced frame (video, web views), attachable to a
/// layer. Created by
/// [`Engine::external_frame`](crate::Engine::external_frame). Not `Clone`.
pub struct ExternalFrameHandle<B: ExternalFrames> {
    /// The frame object.
    pub(crate) frame: B::Frame,
}

impl<B: ExternalFrames> From<ExternalFrameHandle<B>> for LayerContent<B> {
    fn from(handle: ExternalFrameHandle<B>) -> Self {
        let frame = handle.frame;
        Self::Install(Box::new(move |r, surface, layer| {
            B::set_external_frame(r, surface, layer, frame);
        }))
    }
}

/// An opaque render-side install a [`GpuContent`] or [`ExternalFrames`]
/// capability wraps; the closure learns its surface and layer at apply
/// time.
type InstallOp<B> = Box<dyn FnOnce(&mut <B as Backend>::Renderer, SurfaceId, LayerId) + Send>;

/// A recorded layer edit inside a [`Transaction`].
enum EditOp<B: Backend> {
    Transform(Prop<Affine>),
    Opacity(Prop<f32>),
    ScrollOffset(Prop<Vec2>),
    Clip(Option<ShapeData>),
    Blend(BlendMode),
    Filter(Option<FilterId>),
    Content(LayerContent<B>),
    Push(LayerId),
    Insert(usize, LayerId),
    Detach(LayerId),
}

/// One layer's pending edits, collected inside a [`Transaction`]. Each
/// method records an op and returns `&mut Self` for chaining.
///
/// `transform`, `opacity`, `scroll_offset` and `clip` accept a constant or
/// a nami signal (`impl Into<Live<T>>`): a bound signal keeps updating the
/// layer with no further transactions, and a change whose nami `Context`
/// metadata carries an [`Animation`] interpolates on the render thread.
pub struct LayerEdit<B: Backend> {
    ops: Vec<EditOp<B>>,
    layer: LayerId,
    shared: Rc<RefCell<Shared<B>>>,
    /// The transaction-wide animation, filled for animatable ops that lack
    /// one.
    default_animation: Option<Animation>,
}

impl<B: Backend> LayerEdit<B> {
    /// Sets the local transform.
    pub fn transform(&mut self, transform: impl Into<Live<Affine>>) -> &mut Self {
        let live = transform.into();
        self.ops.push(EditOp::Transform(Prop {
            target: live.value,
            animation: self.default_animation,
        }));
        Shared::bind(
            &self.shared,
            self.layer,
            PropKind::Transform,
            live.subscribe,
            |layer, target, animation| LayerOp::Transform(layer, Prop { target, animation }),
        );
        self
    }

    /// Sets the opacity.
    pub fn opacity(&mut self, opacity: impl Into<Live<f32>>) -> &mut Self {
        let live = opacity.into();
        self.ops.push(EditOp::Opacity(Prop {
            target: live.value,
            animation: self.default_animation,
        }));
        Shared::bind(
            &self.shared,
            self.layer,
            PropKind::Opacity,
            live.subscribe,
            |layer, target, animation| LayerOp::Opacity(layer, Prop { target, animation }),
        );
        self
    }

    /// Sets the scroll offset.
    pub fn scroll_offset(&mut self, offset: impl Into<Live<Vec2>>) -> &mut Self {
        let live = offset.into();
        self.ops.push(EditOp::ScrollOffset(Prop {
            target: live.value,
            animation: self.default_animation,
        }));
        Shared::bind(
            &self.shared,
            self.layer,
            PropKind::ScrollOffset,
            live.subscribe,
            |layer, target, animation| LayerOp::ScrollOffset(layer, Prop { target, animation }),
        );
        self
    }

    /// Sets the clip shape, applied in the layer's own space.
    pub fn clip<S: Shape + 'static>(&mut self, shape: impl Into<Live<S>>) -> &mut Self {
        let live = shape.into();
        self.ops
            .push(EditOp::Clip(Some(ShapeData::of(&live.value))));
        Shared::bind(
            &self.shared,
            self.layer,
            PropKind::Clip,
            live.subscribe,
            |layer, shape: S, _| LayerOp::Clip(layer, Some(ShapeData::of(&shape))),
        );
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
    pub fn content(&mut self, content: impl Into<LayerContent<B>>) -> &mut Self {
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

    /// Overrides the animation of the last recorded property op.
    ///
    /// # Panics
    /// Panics unless the last op was `transform`, `opacity` or
    /// `scroll_offset` — `.animation(...)` on any other property is an
    /// invariant violation — and panics when `animation` is a
    /// [`Decay`](crate::Decay) on anything but `scroll_offset`.
    pub fn animation(&mut self, animation: impl Into<Animation>) -> &mut Self {
        let animation = animation.into();
        assert!(
            !matches!(animation, Animation::Decay(_))
                || matches!(self.ops.last(), Some(EditOp::ScrollOffset(_))),
            "Decay is only legal on scroll_offset"
        );
        match self.ops.last_mut() {
            Some(EditOp::Transform(prop)) => prop.animation = Some(animation),
            Some(EditOp::Opacity(prop)) => prop.animation = Some(animation),
            Some(EditOp::ScrollOffset(prop)) => prop.animation = Some(animation),
            _ => panic!("animation() must follow transform, opacity or scroll_offset"),
        }
        self
    }
}

/// A transaction's edits to a surface's layer tree. `tx[&layer]` returns
/// the [`LayerEdit`] accumulating that layer's changes.
pub struct Transaction<'a, B: Backend> {
    edits: Vec<(LayerId, LayerEdit<B>)>,
    shared: &'a Rc<RefCell<Shared<B>>>,
    /// The transaction-wide animation (`Surface::update_animated`).
    animation: Option<Animation>,
    _surface: PhantomData<&'a Surface<B>>,
}

impl<B: Backend> Transaction<'_, B> {
    fn edit(&mut self, layer: &Layer) -> &mut LayerEdit<B> {
        let id = layer.id;
        let index = self.edits.iter().position(|(l, _)| *l == id);
        if let Some(index) = index {
            &mut self.edits[index].1
        } else {
            self.edits.push((
                id,
                LayerEdit {
                    ops: Vec::new(),
                    layer: id,
                    shared: Rc::clone(self.shared),
                    default_animation: self.animation,
                },
            ));
            &mut self.edits.last_mut().expect("just pushed").1
        }
    }
}

impl<B: Backend> Index<&Layer> for Transaction<'_, B> {
    type Output = LayerEdit<B>;

    fn index(&self, layer: &Layer) -> &Self::Output {
        self.edits
            .iter()
            .find(|(id, _)| *id == layer.id)
            .map(|(_, edit)| edit)
            .expect("the layer has no edits in this transaction yet")
    }
}

impl<B: Backend> IndexMut<&Layer> for Transaction<'_, B> {
    fn index_mut(&mut self, layer: &Layer) -> &mut Self::Output {
        self.edit(layer)
    }
}

/// A surface: a render target plus its layer tree. `!Send`; dropping sends
/// [`Message::DestroySurface`].
pub struct Surface<B: Backend> {
    /// The shared pending-changes state, also registered with the engine
    /// for the per-frame drain.
    pub shared: Rc<RefCell<Shared<B>>>,
    id: SurfaceId,
    size: Cell<(u32, u32)>,
    readable: bool,
    root: Layer,
    tx: Sender<Message<B>>,
}

impl<B: Backend> std::fmt::Debug for Surface<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Surface")
            .field("id", &self.id)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl<B: Backend> Surface<B> {
    /// Builds the UI-thread handle once `CreateSurface` succeeded.
    pub fn new(id: SurfaceId, info: SurfaceInfo, tx: Sender<Message<B>>, waker: Rc<Waker>) -> Self {
        let shared = Rc::new(RefCell::new(Shared::new(id, waker)));
        let owner: Rc<dyn LayerOwner> = Rc::clone(&shared) as Rc<dyn LayerOwner>;
        Self {
            shared,
            id,
            size: Cell::new(info.size),
            readable: info.readable,
            root: Layer {
                id: LayerId::new(0),
                owner,
                remove_on_drop: false,
            },
            tx,
        }
    }

    /// The surface's identifier.
    #[must_use]
    pub const fn id(&self) -> SurfaceId {
        self.id
    }

    /// The root layer.
    #[must_use]
    pub const fn root(&self) -> &Layer {
        &self.root
    }

    /// A new detached layer.
    #[must_use]
    pub fn layer(&self) -> Layer {
        let owner: Rc<dyn LayerOwner> = Rc::clone(&self.shared) as Rc<dyn LayerOwner>;
        let id = owner.allocate();
        Layer {
            id,
            owner,
            remove_on_drop: true,
        }
    }

    /// The surface size in pixels.
    #[must_use]
    pub const fn size(&self) -> (u32, u32) {
        self.size.get()
    }

    /// Resizes the surface.
    ///
    /// # Errors
    /// [`SurfaceError::Lost`] when the render thread is gone.
    pub fn resize(&self, size: (u32, u32)) -> Result<(), SurfaceError> {
        self.tx
            .send(Message::ResizeSurface { id: self.id, size })
            .map_err(|_| SurfaceError::Lost)?;
        self.size.set(size);
        Ok(())
    }

    /// Announces the display's properties (scale and HDR headroom) to the
    /// surface.
    ///
    /// # Errors
    /// [`SurfaceError::Lost`] when the render thread is gone.
    pub fn display(&self, display: Display) -> Result<(), SurfaceError> {
        self.tx
            .send(Message::Display {
                id: self.id,
                display,
            })
            .map_err(|_| SurfaceError::Lost)?;
        self.shared.borrow().display.set(display);
        Ok(())
    }

    /// The clear colour, queued into the pending change set. Defaults to
    /// transparent.
    pub fn clear_color(&self, color: WorkingColor) {
        let mut shared = self.shared.borrow_mut();
        shared.clear = Some(color);
        shared.waker.wake();
    }

    /// Records live content for this surface.
    #[must_use]
    pub fn record(&self, body: impl FnOnce(&mut crate::Recorder)) -> Content {
        Content::record(body)
    }

    /// Queues a transaction's edits into the surface's change set. Nothing
    /// is sent; [`Engine::render`](crate::Engine::render) drains the queue.
    ///
    /// # Panics
    /// Panics if `body` panics; the transaction is then dropped unapplied.
    pub fn update(&self, body: impl FnOnce(&mut Transaction<'_, B>)) {
        self.run_transaction(None, body);
    }

    /// Like [`Surface::update`], filling `animation` for every animatable
    /// op that lacks one.
    ///
    /// # Panics
    /// Panics if `body` panics.
    pub fn update_animated(
        &self,
        animation: impl Into<Animation>,
        body: impl FnOnce(&mut Transaction<'_, B>),
    ) {
        self.run_transaction(Some(animation.into()), body);
    }

    fn run_transaction(
        &self,
        animation: Option<Animation>,
        body: impl FnOnce(&mut Transaction<'_, B>),
    ) {
        let mut tx = Transaction {
            edits: Vec::new(),
            shared: &self.shared,
            animation,
            _surface: PhantomData,
        };
        body(&mut tx);
        let mut shared = self.shared.borrow_mut();
        // Ops queued between updates (layer creates, drops, bound-signal
        // changes) come first.
        let pending = std::mem::take(&mut shared.pending);
        let mut ops = pending;
        for (id, edit) in tx.edits {
            for op in edit.ops {
                match op {
                    EditOp::Transform(prop) => ops.push(Op::Layer(LayerOp::Transform(id, prop))),
                    EditOp::Opacity(prop) => ops.push(Op::Layer(LayerOp::Opacity(id, prop))),
                    EditOp::ScrollOffset(prop) => {
                        ops.push(Op::Layer(LayerOp::ScrollOffset(id, prop)));
                    }
                    EditOp::Clip(clip) => ops.push(Op::Layer(LayerOp::Clip(id, clip))),
                    EditOp::Blend(blend) => ops.push(Op::Layer(LayerOp::Blend(id, blend))),
                    EditOp::Filter(filter) => ops.push(Op::Layer(LayerOp::Filter(id, filter))),
                    EditOp::Content(LayerContent::Content(content)) => {
                        let stored = shared.contents.entry(id).or_insert(content);
                        if let Some(change) = stored.take_change() {
                            let content_op = match change {
                                ContentChange::Replace(list) => ContentOp::Replace(list),
                                ContentChange::Update(updates) => ContentOp::Update(updates),
                            };
                            ops.push(Op::Layer(LayerOp::Content(id, Some(content_op))));
                        }
                    }
                    EditOp::Content(LayerContent::Picture(picture)) => {
                        shared.contents.remove(&id);
                        ops.push(Op::Layer(LayerOp::Content(
                            id,
                            Some(ContentOp::Picture(picture)),
                        )));
                    }
                    EditOp::Content(LayerContent::Install(install)) => {
                        shared.contents.remove(&id);
                        let surface = self.id;
                        ops.push(Op::Install(Box::new(move |r| {
                            install(r, surface, id);
                        })));
                    }
                    EditOp::Content(LayerContent::None) => {
                        shared.contents.remove(&id);
                        ops.push(Op::Layer(LayerOp::Content(id, None)));
                    }
                    EditOp::Push(child) => ops.push(Op::Layer(LayerOp::Push { parent: id, child })),
                    EditOp::Insert(index, child) => ops.push(Op::Layer(LayerOp::Insert {
                        parent: id,
                        index,
                        child,
                    })),
                    EditOp::Detach(child) => {
                        ops.push(Op::Layer(LayerOp::Detach { parent: id, child }))
                    }
                }
            }
        }
        shared.pending = ops;
        shared.waker.wake();
    }

    /// The pixels of the surface after the last
    /// [`Engine::render`](crate::Engine::render). Only readable surfaces
    /// (offscreen targets) answer.
    ///
    /// # Errors
    /// [`RenderError::NotReadable`] for a non-readable surface,
    /// [`RenderError::Readback`] when the readback fails, or
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

impl<B: Backend> Drop for Surface<B> {
    fn drop(&mut self) {
        let _ = self.tx.send(Message::DestroySurface { id: self.id });
    }
}
