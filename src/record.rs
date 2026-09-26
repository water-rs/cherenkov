//! Recording: the drawing verbs, the two recorders, and the change set a
//! commit sends to the render thread.

use std::any::Any;
use std::cell::RefCell;
use std::mem::{needs_drop, size_of};
use std::rc::{Rc, Weak};

use kurbo::{Affine, Rect, Stroke};
use nami_core::Signal;
use serde::{Deserialize, Serialize};

use crate::display_list::{Command, DisplayList, Operand, Picture, SlotUpdate};
use crate::glyph::GlyphRun;
use crate::paint::{ImageId, Paint, Sampling};
use crate::shape::{Shape, ShapeData};
use crate::style::{Group, Shadow};

/// The drawing verbs, shared by [`StaticRecorder`] and [`Recorder`].
///
/// [`Draw::Value`] decides what a parameter accepts: a plain value for
/// [`StaticRecorder`], any nami [`Signal`] for [`Recorder`]. Constants are
/// signals, so a plain value works with both.
///
/// State changes are closure scopes only: [`clip`](Draw::clip),
/// [`transform`](Draw::transform) and [`group`](Draw::group) take the body
/// that runs inside them, so a scope cannot be left open.
pub trait Draw {
    /// What a parameter of type `T` accepts.
    type Value<T: 'static>;

    /// Fills a shape.
    fn fill<S: Shape, P: Into<Paint> + 'static>(
        &mut self,
        shape: impl Into<Self::Value<S>>,
        paint: impl Into<Self::Value<P>>,
    );

    /// Strokes a shape.
    fn stroke<S: Shape, P: Into<Paint> + 'static>(
        &mut self,
        shape: impl Into<Self::Value<S>>,
        stroke: impl Into<Self::Value<Stroke>>,
        paint: impl Into<Self::Value<P>>,
    );

    /// Casts a shadow from a shape.
    fn shadow<S: Shape>(
        &mut self,
        shape: impl Into<Self::Value<S>>,
        shadow: impl Into<Self::Value<Shadow>>,
    );

    /// Draws a glyph run.
    fn glyphs<P: Into<Paint> + 'static>(
        &mut self,
        run: impl Into<Self::Value<GlyphRun>>,
        paint: impl Into<Self::Value<P>>,
    );

    /// Draws an image into a rectangle.
    fn image(&mut self, image: ImageId, dst: impl Into<Self::Value<Rect>>, sampling: Sampling);

    /// Draws a shared picture.
    fn picture(&mut self, picture: &Picture, transform: impl Into<Self::Value<Affine>>);

    /// Runs `body` clipped to a shape.
    fn clip<S: Shape>(&mut self, shape: impl Into<Self::Value<S>>, body: impl FnOnce(&mut Self));

    /// Runs `body` under a transform.
    fn transform(
        &mut self,
        transform: impl Into<Self::Value<Affine>>,
        body: impl FnOnce(&mut Self),
    );

    /// Runs `body` isolated as a group.
    fn group(&mut self, group: impl Into<Self::Value<Group>>, body: impl FnOnce(&mut Self));
}

/// A plain value accepted by [`StaticRecorder`].
#[derive(Clone, Copy, Debug)]
pub struct Fixed<T>(pub T);

impl<T> From<T> for Fixed<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

/// Records constants on any thread into a [`Picture`].
#[derive(Debug, Default)]
pub struct StaticRecorder {
    list: DisplayList,
}

impl Picture {
    /// Records a picture. Pictures hold no signals, so they can be recorded on
    /// any thread and shared by any number of layers.
    #[must_use]
    pub fn record(body: impl FnOnce(&mut StaticRecorder)) -> Self {
        let mut recorder = StaticRecorder::default();
        body(&mut recorder);
        Self::new(recorder.list)
    }
}

impl Draw for StaticRecorder {
    type Value<T: 'static> = Fixed<T>;

    fn fill<S: Shape, P: Into<Paint> + 'static>(
        &mut self,
        shape: impl Into<Fixed<S>>,
        paint: impl Into<Fixed<P>>,
    ) {
        self.list.push(Command::Fill {
            shape: ShapeData::of(&shape.into().0),
            paint: paint.into().0.into(),
        });
    }

    fn stroke<S: Shape, P: Into<Paint> + 'static>(
        &mut self,
        shape: impl Into<Fixed<S>>,
        stroke: impl Into<Fixed<Stroke>>,
        paint: impl Into<Fixed<P>>,
    ) {
        self.list.push(Command::Stroke {
            shape: ShapeData::of(&shape.into().0),
            stroke: stroke.into().0,
            paint: paint.into().0.into(),
        });
    }

    fn shadow<S: Shape>(&mut self, shape: impl Into<Fixed<S>>, shadow: impl Into<Fixed<Shadow>>) {
        self.list.push(Command::Shadow {
            shape: ShapeData::of(&shape.into().0),
            shadow: shadow.into().0,
        });
    }

    fn glyphs<P: Into<Paint> + 'static>(
        &mut self,
        run: impl Into<Fixed<GlyphRun>>,
        paint: impl Into<Fixed<P>>,
    ) {
        self.list.push(Command::Glyphs {
            run: run.into().0,
            paint: paint.into().0.into(),
        });
    }

    fn image(&mut self, image: ImageId, dst: impl Into<Fixed<Rect>>, sampling: Sampling) {
        self.list.push(Command::Image {
            image,
            dst: dst.into().0,
            sampling,
        });
    }

    fn picture(&mut self, picture: &Picture, transform: impl Into<Fixed<Affine>>) {
        self.list.push(Command::Picture {
            picture: picture.clone(),
            transform: transform.into().0,
        });
    }

    fn clip<S: Shape>(&mut self, shape: impl Into<Fixed<S>>, body: impl FnOnce(&mut Self)) {
        let begin = self.list.push(Command::BeginClip {
            shape: ShapeData::of(&shape.into().0),
            end: 0,
        });
        body(self);
        self.list.end(begin);
    }

    fn transform(&mut self, transform: impl Into<Fixed<Affine>>, body: impl FnOnce(&mut Self)) {
        let begin = self.list.push(Command::BeginTransform {
            transform: transform.into().0,
            end: 0,
        });
        body(self);
        self.list.end(begin);
    }

    fn group(&mut self, group: impl Into<Fixed<Group>>, body: impl FnOnce(&mut Self)) {
        let begin = self.list.push(Command::BeginGroup {
            group: group.into().0,
            end: 0,
        });
        body(self);
        self.list.end(begin);
    }
}

/// A signal's change subscription: `subscribe(callback)` registers
/// `callback` for later changes, receiving the full nami
/// [`Context`](nami_core::watcher::Context) so metadata (an `Animation`)
/// reaches the consumer, and returns the guard to keep alive, if any.
pub type Subscribe<T> =
    Box<dyn FnOnce(Box<dyn Fn(nami_core::watcher::Context<T>)>) -> Option<Box<dyn Any>>>;

/// A value accepted by [`Recorder`]: the current value of a nami signal, and
/// the subscription that reports its later changes.
pub struct Live<T> {
    #[doc(hidden)]
    pub value: T,
    #[doc(hidden)]
    pub subscribe: Subscribe<T>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for Live<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Live")
            .field("value", &self.value)
            .finish_non_exhaustive()
    }
}

impl<T: 'static, S: Signal<Output = T>> From<S> for Live<T> {
    fn from(signal: S) -> Self {
        let value = signal.snapshot();
        Self {
            value,
            subscribe: Box::new(
                move |on_change: Box<dyn Fn(nami_core::watcher::Context<T>)>| {
                    let guard = signal.watch(on_change);
                    // A guard with no size and no drop glue unsubscribes nothing, so
                    // there is nothing to keep alive.
                    if size_of::<S::Guard>() == 0 && !needs_drop::<S::Guard>() {
                        None
                    } else {
                        Some(Box::new(guard) as Box<dyn Any>)
                    }
                },
            ),
        }
    }
}

/// State shared between a [`Content`] and the watchers of its signals.
#[derive(Default)]
struct LiveState {
    pending: RefCell<Vec<SlotUpdate>>,
    guards: RefCell<Vec<Box<dyn Any>>>,
}

impl LiveState {
    fn push(&self, update: SlotUpdate) {
        let mut pending = self.pending.borrow_mut();
        let slot = update.slot();
        match pending.iter_mut().find(|queued| queued.slot() == slot) {
            Some(queued) => *queued = update,
            None => pending.push(update),
        }
    }
}

/// Records on the UI thread, accepting nami signals anywhere a value is
/// accepted.
pub struct Recorder {
    list: DisplayList,
    live: Rc<LiveState>,
}

impl std::fmt::Debug for Recorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recorder")
            .field("list", &self.list)
            .finish_non_exhaustive()
    }
}

impl Recorder {
    /// Subscribes to a value's later changes, which update operand `convert`
    /// produces on command `command`.
    fn subscribe<T: 'static>(
        &self,
        subscribe: Subscribe<T>,
        command: u32,
        convert: impl Fn(T) -> Operand + 'static,
    ) {
        let state: Weak<LiveState> = Rc::downgrade(&self.live);
        let guard = subscribe(Box::new(move |context: nami_core::watcher::Context<T>| {
            if let Some(state) = state.upgrade() {
                state.push(SlotUpdate {
                    command,
                    value: convert(context.into_value()),
                });
            }
        }));
        if let Some(guard) = guard {
            self.live.guards.borrow_mut().push(guard);
        }
    }
}

fn paint_operand<P: Into<Paint>>(paint: P) -> Operand {
    Operand::Paint(paint.into())
}

impl Draw for Recorder {
    type Value<T: 'static> = Live<T>;

    fn fill<S: Shape, P: Into<Paint> + 'static>(
        &mut self,
        shape: impl Into<Live<S>>,
        paint: impl Into<Live<P>>,
    ) {
        let (shape, paint) = (shape.into(), paint.into());
        let command = self.list.push(Command::Fill {
            shape: ShapeData::of(&shape.value),
            paint: paint.value.into(),
        });
        self.subscribe(shape.subscribe, command, |shape: S| {
            Operand::Shape(ShapeData::of(&shape))
        });
        self.subscribe(paint.subscribe, command, paint_operand::<P>);
    }

    fn stroke<S: Shape, P: Into<Paint> + 'static>(
        &mut self,
        shape: impl Into<Live<S>>,
        stroke: impl Into<Live<Stroke>>,
        paint: impl Into<Live<P>>,
    ) {
        let (shape, stroke, paint) = (shape.into(), stroke.into(), paint.into());
        let command = self.list.push(Command::Stroke {
            shape: ShapeData::of(&shape.value),
            stroke: stroke.value,
            paint: paint.value.into(),
        });
        self.subscribe(shape.subscribe, command, |shape: S| {
            Operand::Shape(ShapeData::of(&shape))
        });
        self.subscribe(stroke.subscribe, command, Operand::Stroke);
        self.subscribe(paint.subscribe, command, paint_operand::<P>);
    }

    fn shadow<S: Shape>(&mut self, shape: impl Into<Live<S>>, shadow: impl Into<Live<Shadow>>) {
        let (shape, shadow) = (shape.into(), shadow.into());
        let command = self.list.push(Command::Shadow {
            shape: ShapeData::of(&shape.value),
            shadow: shadow.value,
        });
        self.subscribe(shape.subscribe, command, |shape: S| {
            Operand::Shape(ShapeData::of(&shape))
        });
        self.subscribe(shadow.subscribe, command, Operand::Shadow);
    }

    fn glyphs<P: Into<Paint> + 'static>(
        &mut self,
        run: impl Into<Live<GlyphRun>>,
        paint: impl Into<Live<P>>,
    ) {
        let run = run.into();
        let paint = paint.into();
        let command = self.list.push(Command::Glyphs {
            run: run.value,
            paint: paint.value.into(),
        });
        self.subscribe(run.subscribe, command, Operand::Run);
        self.subscribe(paint.subscribe, command, paint_operand::<P>);
    }

    fn image(&mut self, image: ImageId, dst: impl Into<Live<Rect>>, sampling: Sampling) {
        let dst = dst.into();
        let command = self.list.push(Command::Image {
            image,
            dst: dst.value,
            sampling,
        });
        self.subscribe(dst.subscribe, command, Operand::Rect);
    }

    fn picture(&mut self, picture: &Picture, transform: impl Into<Live<Affine>>) {
        let transform = transform.into();
        let command = self.list.push(Command::Picture {
            picture: picture.clone(),
            transform: transform.value,
        });
        self.subscribe(transform.subscribe, command, Operand::Transform);
    }

    fn clip<S: Shape>(&mut self, shape: impl Into<Live<S>>, body: impl FnOnce(&mut Self)) {
        let shape = shape.into();
        let begin = self.list.push(Command::BeginClip {
            shape: ShapeData::of(&shape.value),
            end: 0,
        });
        self.subscribe(shape.subscribe, begin, |shape: S| {
            Operand::Shape(ShapeData::of(&shape))
        });
        body(self);
        self.list.end(begin);
    }

    fn transform(&mut self, transform: impl Into<Live<Affine>>, body: impl FnOnce(&mut Self)) {
        let transform = transform.into();
        let begin = self.list.push(Command::BeginTransform {
            transform: transform.value,
            end: 0,
        });
        self.subscribe(transform.subscribe, begin, Operand::Transform);
        body(self);
        self.list.end(begin);
    }

    fn group(&mut self, group: impl Into<Live<Group>>, body: impl FnOnce(&mut Self)) {
        let group = group.into();
        let begin = self.list.push(Command::BeginGroup {
            group: group.value,
            end: 0,
        });
        self.subscribe(group.subscribe, begin, Operand::Group);
        body(self);
        self.list.end(begin);
    }
}

/// Content recorded on the UI thread. It owns the subscriptions of the signals
/// it was recorded with, and turns their changes into [`ContentChange`]s.
///
/// `Content` is not `Send`: its signals live on the UI thread. What crosses to
/// the render thread is the owned [`ContentChange`].
pub struct Content {
    list: DisplayList,
    live: Rc<LiveState>,
    sent: bool,
}

impl std::fmt::Debug for Content {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Content")
            .field("list", &self.list)
            .field("sent", &self.sent)
            .finish_non_exhaustive()
    }
}

impl Content {
    /// Records content.
    #[must_use]
    pub fn record(body: impl FnOnce(&mut Recorder)) -> Self {
        let mut recorder = Recorder {
            list: DisplayList::default(),
            live: Rc::default(),
        };
        body(&mut recorder);
        Self {
            list: recorder.list,
            live: recorder.live,
            sent: false,
        }
    }

    /// The change to send at the next commit, if any. The first call sends the
    /// whole display list; later calls send only the slots whose signals
    /// changed.
    pub fn take_change(&mut self) -> Option<ContentChange> {
        let updates = self.live.pending.take();
        if !updates.is_empty() {
            let _ = self.list.apply(updates.iter().cloned());
        }
        if !self.sent {
            self.sent = true;
            return Some(ContentChange::Replace(self.list.clone()));
        }
        (!updates.is_empty()).then_some(ContentChange::Update(updates))
    }

    /// The display list with every signal change received so far applied, for
    /// capturing a frame.
    pub fn snapshot(&mut self) -> &DisplayList {
        let updates = self.live.pending.take();
        if !updates.is_empty() {
            let _ = self.list.apply(updates.iter().cloned());
            if self.sent {
                // Keep the pending updates for the render thread.
                *self.live.pending.borrow_mut() = updates;
            }
        }
        &self.list
    }
}

/// What a commit sends to the render thread for one content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ContentChange {
    /// The whole display list, sent when the content is first committed.
    Replace(DisplayList),
    /// New values for slots, sent when signals change afterwards.
    Update(Vec<SlotUpdate>),
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use kurbo::{Circle, Rect};
    use nami::{SignalExt, binding};

    use super::*;
    use crate::color::{Color, Srgb};

    fn red() -> Color<Srgb> {
        Color::new([1., 0., 0., 1.])
    }

    #[test]
    fn a_signal_change_regenerates_only_the_commands_that_reference_it() {
        let radius = binding::<f64>(8.);
        let mut content = Content::record(|c| {
            c.fill(Rect::new(0., 0., 10., 10.), red());
            c.fill(radius.map(|r| Circle::new((50., 50.), r)), red());
            c.fill(Rect::new(20., 20., 30., 30.), red());
        });

        let Some(ContentChange::Replace(mut remote)) = content.take_change() else {
            panic!("the first commit sends the whole list");
        };
        assert_eq!(remote.len(), 3);
        assert_eq!(content.take_change(), None, "nothing changed yet");

        radius.set(16.);
        let Some(ContentChange::Update(updates)) = content.take_change() else {
            panic!("a changed signal sends an update");
        };
        assert_eq!(updates.len(), 1);

        let dirty = remote.apply(updates);
        assert_eq!(dirty.ranges(), [Range { start: 1, end: 2 }]);
        let Command::Fill { shape, .. } = &remote.commands()[1] else {
            panic!("command 1 is the circle fill");
        };
        assert_eq!(*shape, ShapeData::Circle(Circle::new((50., 50.), 16.)));
    }

    #[test]
    fn a_scope_change_regenerates_the_whole_scope() {
        let offset = binding::<f64>(0.);
        let mut content = Content::record(|c| {
            c.fill(Rect::new(0., 0., 1., 1.), red());
            c.transform(offset.map(|x| Affine::translate((x, 0.))), |c| {
                c.fill(Rect::new(0., 0., 1., 1.), red());
                c.fill(Rect::new(1., 1., 2., 2.), red());
            });
            c.fill(Rect::new(5., 5., 6., 6.), red());
        });
        let Some(ContentChange::Replace(mut remote)) = content.take_change() else {
            panic!("the first commit sends the whole list");
        };

        offset.set(4.);
        let Some(ContentChange::Update(updates)) = content.take_change() else {
            panic!("a changed signal sends an update");
        };
        let dirty = remote.apply(updates);
        // BeginTransform (1), two fills (2, 3) and End (4).
        assert_eq!(dirty.ranges(), [Range { start: 1, end: 5 }]);
        assert!(!dirty.contains(0) && !dirty.contains(5));
    }

    #[test]
    fn repeated_changes_before_a_commit_send_one_update() {
        let radius = binding::<f64>(1.);
        let mut content = Content::record(|c| {
            c.fill(radius.map(|r| Circle::new((0., 0.), r)), red());
        });
        let _ = content.take_change();
        radius.set(2.);
        radius.set(3.);
        let Some(ContentChange::Update(updates)) = content.take_change() else {
            panic!("a changed signal sends an update");
        };
        assert_eq!(updates.len(), 1);
        assert_eq!(
            updates[0].value,
            Operand::Shape(ShapeData::Circle(Circle::new((0., 0.), 3.)))
        );
    }

    #[test]
    fn dropping_content_releases_its_subscriptions() {
        let radius = binding::<f64>(1.);
        let content = Content::record(|c| {
            c.fill(radius.map(|r| Circle::new((0., 0.), r)), red());
        });
        drop(content);
        // A watcher that outlived its content would upgrade a dead state;
        // setting must simply do nothing.
        radius.set(2.);
    }

    #[test]
    fn pictures_are_send_sync_and_shared() {
        fn assert_send_sync<T: Send + Sync + Clone>() {}
        assert_send_sync::<Picture>();
        assert_send_sync::<ContentChange>();

        let picture = Picture::record(|c| {
            c.fill(Rect::new(0., 0., 4., 4.), red());
            c.clip(Rect::new(0., 0., 2., 2.), |c| {
                c.fill(Circle::new((1., 1.), 1.), red());
            });
        });
        let shared = picture.clone();
        let from_thread = std::thread::spawn(move || shared.display_list().len())
            .join()
            .expect("the thread does not panic");
        assert_eq!(from_thread, 4);
        assert!(matches!(
            picture.display_list().commands()[1],
            Command::BeginClip { end: 3, .. }
        ));
    }

    #[test]
    fn change_sets_round_trip_through_serde() {
        let mut content = Content::record(|c| {
            c.stroke(
                Rect::new(0., 0., 1., 1.),
                Stroke::new(2.),
                crate::paint::LinearGradient::new((0., 0.), (1., 0.))
                    .stop(0., red())
                    .stop(1., Color::<Srgb>::new([0., 0., 1., 1.])),
            );
        });
        let change = content
            .take_change()
            .expect("the first commit sends the list");
        let json = serde_json::to_string(&change).expect("serializes");
        let back: ContentChange = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back, change);
    }
}
