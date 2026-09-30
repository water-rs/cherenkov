//! Hidden surfaces (#204) on the [`Null`] backend, on the native render
//! thread and in the browser executor: a hidden surface wakes no host and
//! no frame draws it, its state changes keep applying, and showing it asks
//! for exactly one frame, which draws its current state.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use kurbo::{Affine, Rect};

use super::{Event, FrameRecord, Null};
use crate::image::ImageData;
use crate::{
    Curve, Draw as _, Engine, Image, Layer, ResourceId, Rgba8, Sampling, SurfaceId, Visibility,
    WorkingColor,
};

/// The opacity curve's duration; the show frame samples it halfway.
const RUN: Duration = Duration::from_secs(10);
/// One frame interval.
const TICK: Duration = Duration::from_millis(16);
/// The transform a bound signal sets while the surface is hidden.
const MOVED: Affine = Affine::new([1.0, 0.0, 0.0, 1.0, 3.0, 4.0]);

/// What a surface keeps bound while it animates.
struct Scene {
    layer: Layer,
    transform: nami::Binding<Affine>,
    color: nami::Binding<WorkingColor>,
}

/// Installs on `surface` a layer running a 10 s linear opacity curve from
/// 0 to 1, with a bound transform and live content whose colour operand
/// is bound.
fn animate(surface: &crate::Surface<Null>) -> Scene {
    let layer = surface.layer();
    let transform = nami::binding(Affine::IDENTITY);
    let color = nami::binding(WorkingColor::WHITE);
    let content = surface.record(|c| c.fill(Rect::new(0.0, 0.0, 8.0, 8.0), color.clone()));
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
        tx[&layer]
            .transform(transform.clone())
            .opacity(0.0_f32)
            .content(content);
    });
    surface.update_animated(Curve::linear(RUN), |tx| {
        tx[&layer].opacity(1.0_f32);
    });
    Scene {
        layer,
        transform,
        color,
    }
}

/// Changes everything a hidden surface accepts: a bound property, a live
/// operand, a transaction, a new layer and the clear colour. Returns the
/// new layer.
fn change(surface: &crate::Surface<Null>, scene: &Scene) -> Layer {
    scene.transform.set(MOVED);
    scene.color.set(WorkingColor::BLACK);
    let child = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].push(&child);
    });
    surface.clear_color(WorkingColor::BLACK);
    child
}

/// Counts the host wakes from now on.
fn count_wakes(engine: &Engine<Null>) -> Rc<Cell<u32>> {
    let wakes = Rc::new(Cell::new(0));
    engine.set_waker({
        let wakes = Rc::clone(&wakes);
        move || wakes.set(wakes.get() + 1)
    });
    wakes
}

/// Every event reported so far.
fn drain(rx: &Receiver<Event>) -> Vec<Event> {
    std::iter::from_fn(|| rx.try_recv().ok()).collect()
}

/// The frame record of `surface` in `events`.
fn frame(events: &[Event], surface: SurfaceId) -> Option<&FrameRecord> {
    events.iter().find_map(|event| match event {
        Event::Frame(record) if record.surface == surface => Some(record),
        _ => None,
    })
}

/// The first frame after the surface is shown draws what changed while it
/// was hidden, with the animation sampled at the show frame's time.
fn assert_current(events: &[Event], surface: SurfaceId, scene: &Scene, child: &Layer) {
    let content = events.iter().position(
        |event| matches!(event, Event::SetContent(s, l) if *s == surface && *l == scene.layer.id()),
    );
    let drawn = events
        .iter()
        .position(|event| matches!(event, Event::Frame(record) if record.surface == surface));
    assert!(
        matches!((content, drawn), (Some(content), Some(drawn)) if content < drawn),
        "the operand written while hidden lands before the show frame: {events:?}"
    );
    assert_sampled(events, surface, scene, child);
}

/// The frame drew `surface` with what changed while it was hidden, and
/// with the animation sampled at the frame's time.
fn assert_sampled(events: &[Event], surface: SurfaceId, scene: &Scene, child: &Layer) {
    let record = frame(events, surface).expect("the show frame draws the surface");
    assert!(record.changed, "{record:?}");
    let sample = |id| {
        record
            .layers
            .iter()
            .find(|layer| layer.id == id)
            .expect("layer sampled")
    };
    let layer = sample(scene.layer.id());
    assert_eq!(
        layer.transform, MOVED,
        "the bound transform set while hidden"
    );
    assert!(
        (layer.opacity - 0.5).abs() < 1e-3,
        "the curve jumps to its value at the show frame, no catch-up: {}",
        layer.opacity
    );
    assert!(
        sample(crate::LayerId::new(0))
            .children
            .contains(&child.id()),
        "the layer pushed while hidden"
    );
}

/// One transparent pixel.
fn pixel() -> ImageData<Rgba8> {
    ImageData::<Rgba8>::new(1, 1, vec![0u8; 4]).expect("image data")
}

/// Registers a one-pixel image.
fn image(engine: &Engine<Null>) -> Image<Rgba8> {
    engine.image(pixel()).expect("image")
}

/// Records a draw of `image` into a new layer of `surface`.
fn draw(surface: &crate::Surface<Null>, image: &Image<Rgba8>) -> Layer {
    let layer = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
        tx[&layer]
            .record(|c| c.image(image.id(), Rect::new(0.0, 0.0, 8.0, 8.0), Sampling::Nearest));
    });
    layer
}

/// While every surface is hidden, the render loop only hid `surface` and
/// replaced the image: no commit was applied and no frame drawn.
fn assert_untouched(events: &[Event], surface: SurfaceId) {
    assert!(
        matches!(
            events,
            [Event::Visibility(id, Visibility::Hidden), Event::ReplaceImage(..)] if *id == surface
        ),
        "the render loop did nothing but hide the surface and take the new pixels: {events:?}"
    );
}

/// A render with `hidden` hidden drew `visible` only, and applied the
/// commit `hidden` made to `layer` while hidden.
fn assert_left_out(events: &[Event], visible: SurfaceId, hidden: SurfaceId, layer: &Layer) {
    assert!(frame(events, visible).is_some(), "{events:?}");
    assert!(
        frame(events, hidden).is_none(),
        "no frame draws a hidden surface: {events:?}"
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            Event::SetContent(s, l) if *s == hidden && *l == layer.id()
        )),
        "the hidden surface's commit applied: {events:?}"
    );
}

/// The frame drew `surface` although nothing on it changed but its
/// visibility.
fn assert_redrawn(events: &[Event], surface: SurfaceId) {
    let record = frame(events, surface).expect("the show frame draws the surface");
    assert!(
        record.changed,
        "shown with nothing new queued, it is still drawn: {record:?}"
    );
}

/// The render failed on the rejected `image`.
fn assert_rejected<T: std::fmt::Debug>(
    result: &Result<T, crate::RenderError>,
    image: &Image<Rgba8>,
) {
    assert!(
        matches!(
            result,
            Err(crate::RenderError::Rejected { resource, .. })
                if *resource == ResourceId::Image(image.id())
        ),
        "{result:?}"
    );
}

/// Installs on `surface` live content whose colour operand animates along
/// a 10 s linear curve on every change.
fn animated_operand(surface: &crate::Surface<Null>) -> (Layer, nami::Binding<WorkingColor>) {
    use nami::SignalExt as _;
    let layer = surface.layer();
    let color = nami::binding(WorkingColor::WHITE);
    let content = surface.record(|c| {
        c.fill(
            Rect::new(0.0, 0.0, 8.0, 8.0),
            color
                .clone()
                .with(crate::Animation::from(Curve::linear(RUN))),
        );
    });
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
        tx[&layer].content(content);
    });
    (layer, color)
}

/// How many content updates reached `layer` of `surface`.
fn content_updates(events: &[Event], surface: SurfaceId, layer: &Layer) -> usize {
    events
        .iter()
        .filter(
            |event| matches!(event, Event::SetContent(s, l) if *s == surface && *l == layer.id()),
        )
        .count()
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::collections::HashSet;
    use std::sync::mpsc::Receiver;

    use super::{
        RUN, TICK, animate, animated_operand, assert_current, assert_left_out, assert_redrawn,
        assert_rejected, assert_sampled, assert_untouched, change, content_updates, count_wakes,
        drain, draw, image, pixel,
    };
    use crate::testing::{Event, Null, NullConfig, NullReject};
    use crate::{
        Engine, FrameTime, Instant, Next, Offscreen, OffscreenFormat, RenderError, Surface,
        Visibility,
    };

    fn engine(reject: HashSet<NullReject>) -> (Engine<Null>, Receiver<Event>) {
        let (events, rx) = std::sync::mpsc::channel();
        let engine = Engine::<Null>::new(NullConfig { events, reject }).expect("init");
        (engine, rx)
    }

    fn surface(engine: &Engine<Null>) -> Surface<Null> {
        engine
            .surface(Offscreen::new((8, 8), OffscreenFormat::LinearF16))
            .expect("surface")
    }

    #[test]
    fn a_hidden_surface_wakes_nothing_and_refuses_render() {
        let (engine, rx) = engine(HashSet::new());
        let surface = surface(&engine);
        // A dropped surface no longer counts as visible.
        drop(self::surface(&engine));
        let scene = animate(&surface);
        let image = image(&engine);
        let t0 = Instant::now();
        let next = engine.render(FrameTime::at(t0)).expect("render");
        assert!(matches!(next, Next::At { .. }), "the curve runs: {next:?}");
        let _ = drain(&rx);
        let wakes = count_wakes(&engine);

        surface.visibility(Visibility::Hidden).expect("hide");
        let _child = change(&surface, &scene);
        image.replace(pixel()).expect("replace");
        assert_eq!(wakes.get(), 0, "a hidden surface wakes no host");
        assert!(
            matches!(
                engine.render(FrameTime::at(t0 + TICK)),
                Err(RenderError::Hidden)
            ),
            "a render with every surface hidden is an error"
        );
        // The replacement is fire-and-forget; a request/reply behind it
        // has run once it answers.
        let _ = engine.memory();
        assert_untouched(&drain(&rx), surface.id());
        assert_eq!(wakes.get(), 0, "a refused render wakes no host");
    }

    #[test]
    fn showing_a_surface_wakes_once_and_draws_its_current_state() {
        let (engine, rx) = engine(HashSet::new());
        let surface = surface(&engine);
        let scene = animate(&surface);
        let t0 = Instant::now();
        engine.render(FrameTime::at(t0)).expect("render");
        let wakes = count_wakes(&engine);
        // A wake the host never answered with a render: showing must
        // still reach it.
        surface.clear_color(crate::WorkingColor::WHITE);
        assert_eq!(wakes.get(), 1);

        surface.visibility(Visibility::Hidden).expect("hide");
        let child = change(&surface, &scene);
        assert_eq!(wakes.get(), 1, "a hidden surface wakes no host");
        surface.visibility(Visibility::Visible).expect("show");
        assert_eq!(wakes.get(), 2, "showing the surface wakes the host");
        surface.visibility(Visibility::Visible).expect("show again");
        surface.clear_color(crate::WorkingColor::BLACK);
        assert_eq!(wakes.get(), 2, "one wake until the frame it asked for");
        let _ = drain(&rx);

        let next = engine.render(FrameTime::at(t0 + RUN / 2)).expect("render");
        assert!(
            matches!(next, Next::At { .. }),
            "the curve still runs: {next:?}"
        );
        assert_current(&drain(&rx), surface.id(), &scene, &child);
    }

    #[test]
    fn a_hidden_surface_is_left_out_of_frames_and_its_commits_apply() {
        let (engine, rx) = engine(HashSet::new());
        let visible = surface(&engine);
        let hidden = surface(&engine);
        let scene = animate(&hidden);
        let t0 = Instant::now();
        engine.render(FrameTime::at(t0)).expect("render");
        let wakes = count_wakes(&engine);
        hidden.visibility(Visibility::Hidden).expect("hide");
        let child = change(&hidden, &scene);
        assert_eq!(
            wakes.get(),
            0,
            "a hidden surface wakes no host while another is visible"
        );
        let _ = drain(&rx);

        let next = engine.render(FrameTime::at(t0 + TICK)).expect("render");
        assert_eq!(
            next,
            Next::Idle,
            "a hidden surface's animation asks for no frame"
        );
        assert_left_out(&drain(&rx), visible.id(), hidden.id(), &scene.layer);

        hidden.visibility(Visibility::Visible).expect("show");
        engine.render(FrameTime::at(t0 + RUN / 2)).expect("render");
        assert_sampled(&drain(&rx), hidden.id(), &scene, &child);

        visible.visibility(Visibility::Hidden).expect("hide");
        engine.render(FrameTime::at(t0 + RUN)).expect("render");
        visible.visibility(Visibility::Visible).expect("show");
        let _ = drain(&rx);
        engine
            .render(FrameTime::at(t0 + RUN + TICK))
            .expect("render");
        assert_redrawn(&drain(&rx), visible.id());
    }

    #[test]
    fn a_hidden_surface_s_rejection_fails_only_its_show_frame() {
        let (engine, _rx) = engine(HashSet::from([NullReject::Image]));
        let visible = surface(&engine);
        let hidden = surface(&engine);
        hidden.visibility(Visibility::Hidden).expect("hide");
        let image = image(&engine);
        let _layer = draw(&hidden, &image);
        engine
            .render(FrameTime::now())
            .expect("the visible surface renders");
        engine
            .render(FrameTime::now())
            .expect("the visible surface renders again");
        hidden.visibility(Visibility::Visible).expect("show");
        assert_rejected(&engine.render(FrameTime::now()), &image);
        drop(visible);
    }
    #[test]
    fn a_hidden_surface_s_operand_animation_is_not_sampled() {
        let (engine, rx) = engine(HashSet::new());
        let _visible = surface(&engine);
        let hidden = surface(&engine);
        let (layer, color) = animated_operand(&hidden);
        let t0 = Instant::now();
        engine.render(FrameTime::at(t0)).expect("render");
        color.set(crate::WorkingColor::BLACK);
        let next = engine.render(FrameTime::at(t0 + TICK)).expect("render");
        assert!(
            matches!(next, Next::At { .. }),
            "the operand runs: {next:?}"
        );
        hidden.visibility(Visibility::Hidden).expect("hide");
        let _ = drain(&rx);

        let next = engine.render(FrameTime::at(t0 + 2 * TICK)).expect("render");
        assert_eq!(next, Next::Idle, "a hidden operand asks for no frame");
        engine.render(FrameTime::at(t0 + 3 * TICK)).expect("render");
        assert_eq!(
            content_updates(&drain(&rx), hidden.id(), &layer),
            0,
            "a hidden surface's operand is not sampled"
        );
        hidden.visibility(Visibility::Visible).expect("show");
        engine.render(FrameTime::at(t0 + RUN / 2)).expect("render");
        assert_eq!(
            content_updates(&drain(&rx), hidden.id(), &layer),
            1,
            "the show frame samples the operand once"
        );
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::collections::HashSet;
    use std::sync::mpsc::Receiver;

    use wasm_bindgen_test::wasm_bindgen_test;

    use super::{
        RUN, TICK, animate, animated_operand, assert_current, assert_left_out, assert_redrawn,
        assert_rejected, assert_sampled, assert_untouched, change, content_updates, count_wakes,
        drain, draw, image, pixel,
    };
    use crate::testing::{Event, Null, NullConfig, NullReject};
    use crate::{
        Engine, FrameTime, Instant, Next, Offscreen, OffscreenFormat, RenderError, Surface,
        Visibility,
    };

    async fn engine(reject: HashSet<NullReject>) -> (Engine<Null>, Receiver<Event>) {
        let (events, rx) = std::sync::mpsc::channel();
        let engine = Engine::<Null>::new(NullConfig { events, reject })
            .await
            .expect("init");
        (engine, rx)
    }

    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn surface(engine: &Engine<Null>) -> Surface<Null> {
        engine
            .surface(Offscreen::new((8, 8), OffscreenFormat::LinearF16))
            .await
            .expect("surface")
    }

    #[wasm_bindgen_test]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn a_hidden_surface_wakes_nothing_and_refuses_render() {
        let (engine, rx) = engine(HashSet::new()).await;
        let surface = surface(&engine).await;
        // A dropped surface no longer counts as visible.
        drop(self::surface(&engine).await);
        let scene = animate(&surface);
        let image = image(&engine);
        let t0 = Instant::now();
        let next = engine.render(FrameTime::at(t0)).await.expect("render");
        assert!(matches!(next, Next::At { .. }), "the curve runs: {next:?}");
        let _ = drain(&rx);
        let wakes = count_wakes(&engine);

        surface.visibility(Visibility::Hidden).await.expect("hide");
        let _child = change(&surface, &scene);
        image.replace(pixel()).expect("replace");
        assert_eq!(wakes.get(), 0, "a hidden surface wakes no host");
        assert!(
            matches!(
                engine.render(FrameTime::at(t0 + TICK)).await,
                Err(RenderError::Hidden)
            ),
            "a render with every surface hidden is an error"
        );
        // The replacement is applied by the serial executor after the
        // hide; a request/reply behind it has run once it answers.
        let _ = engine.memory().await;
        assert_untouched(&drain(&rx), surface.id());
        assert_eq!(wakes.get(), 0, "a refused render wakes no host");
    }

    #[wasm_bindgen_test]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn showing_a_surface_wakes_once_and_draws_its_current_state() {
        let (engine, rx) = engine(HashSet::new()).await;
        let surface = surface(&engine).await;
        let scene = animate(&surface);
        let t0 = Instant::now();
        engine.render(FrameTime::at(t0)).await.expect("render");
        let wakes = count_wakes(&engine);
        // A wake the host never answered with a render: showing must
        // still reach it.
        surface.clear_color(crate::WorkingColor::WHITE);
        assert_eq!(wakes.get(), 1);

        surface.visibility(Visibility::Hidden).await.expect("hide");
        let child = change(&surface, &scene);
        assert_eq!(wakes.get(), 1, "a hidden surface wakes no host");
        surface.visibility(Visibility::Visible).await.expect("show");
        assert_eq!(wakes.get(), 2, "showing the surface wakes the host");
        surface
            .visibility(Visibility::Visible)
            .await
            .expect("show again");
        surface.clear_color(crate::WorkingColor::BLACK);
        assert_eq!(wakes.get(), 2, "one wake until the frame it asked for");
        let _ = drain(&rx);

        let next = engine
            .render(FrameTime::at(t0 + RUN / 2))
            .await
            .expect("render");
        assert!(
            matches!(next, Next::At { .. }),
            "the curve still runs: {next:?}"
        );
        assert_current(&drain(&rx), surface.id(), &scene, &child);
    }

    #[wasm_bindgen_test]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn a_hidden_surface_is_left_out_of_frames_and_its_commits_apply() {
        let (engine, rx) = engine(HashSet::new()).await;
        let visible = surface(&engine).await;
        let hidden = surface(&engine).await;
        let scene = animate(&hidden);
        let t0 = Instant::now();
        engine.render(FrameTime::at(t0)).await.expect("render");
        let wakes = count_wakes(&engine);
        hidden.visibility(Visibility::Hidden).await.expect("hide");
        let child = change(&hidden, &scene);
        assert_eq!(
            wakes.get(),
            0,
            "a hidden surface wakes no host while another is visible"
        );
        let _ = drain(&rx);

        let next = engine
            .render(FrameTime::at(t0 + TICK))
            .await
            .expect("render");
        assert_eq!(
            next,
            Next::Idle,
            "a hidden surface's animation asks for no frame"
        );
        assert_left_out(&drain(&rx), visible.id(), hidden.id(), &scene.layer);

        hidden.visibility(Visibility::Visible).await.expect("show");
        engine
            .render(FrameTime::at(t0 + RUN / 2))
            .await
            .expect("render");
        assert_sampled(&drain(&rx), hidden.id(), &scene, &child);

        visible.visibility(Visibility::Hidden).await.expect("hide");
        engine
            .render(FrameTime::at(t0 + RUN))
            .await
            .expect("render");
        visible.visibility(Visibility::Visible).await.expect("show");
        let _ = drain(&rx);
        engine
            .render(FrameTime::at(t0 + RUN + TICK))
            .await
            .expect("render");
        assert_redrawn(&drain(&rx), visible.id());
    }

    #[wasm_bindgen_test]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn a_hidden_surface_s_rejection_fails_only_its_show_frame() {
        let (engine, _rx) = engine(HashSet::from([NullReject::Image])).await;
        let visible = surface(&engine).await;
        let hidden = surface(&engine).await;
        hidden.visibility(Visibility::Hidden).await.expect("hide");
        let image = image(&engine);
        let _layer = draw(&hidden, &image);
        engine
            .render(FrameTime::now())
            .await
            .expect("the visible surface renders");
        engine
            .render(FrameTime::now())
            .await
            .expect("the visible surface renders again");
        hidden.visibility(Visibility::Visible).await.expect("show");
        assert_rejected(&engine.render(FrameTime::now()).await, &image);
        drop(visible);
    }
    #[wasm_bindgen_test]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn a_hidden_surface_s_operand_animation_is_not_sampled() {
        let (engine, rx) = engine(HashSet::new()).await;
        let _visible = surface(&engine).await;
        let hidden = surface(&engine).await;
        let (layer, color) = animated_operand(&hidden);
        let t0 = Instant::now();
        engine.render(FrameTime::at(t0)).await.expect("render");
        color.set(crate::WorkingColor::BLACK);
        let next = engine
            .render(FrameTime::at(t0 + TICK))
            .await
            .expect("render");
        assert!(
            matches!(next, Next::At { .. }),
            "the operand runs: {next:?}"
        );
        hidden.visibility(Visibility::Hidden).await.expect("hide");
        let _ = drain(&rx);

        let next = engine
            .render(FrameTime::at(t0 + 2 * TICK))
            .await
            .expect("render");
        assert_eq!(next, Next::Idle, "a hidden operand asks for no frame");
        engine
            .render(FrameTime::at(t0 + 3 * TICK))
            .await
            .expect("render");
        assert_eq!(
            content_updates(&drain(&rx), hidden.id(), &layer),
            0,
            "a hidden surface's operand is not sampled"
        );
        hidden.visibility(Visibility::Visible).await.expect("show");
        engine
            .render(FrameTime::at(t0 + RUN / 2))
            .await
            .expect("render");
        assert_eq!(
            content_updates(&drain(&rx), hidden.id(), &layer),
            1,
            "the show frame samples the operand once"
        );
    }
}
