//! A hidden surface's custom GPU content and filters on a real device
//! (#204): they are not pulled, wake no host and ask for no frame while
//! the surface is hidden, and showing the surface asks for one frame that
//! draws it current.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};
use std::time::{Duration, Instant};

use cherenkov::kurbo::Rect;
use cherenkov::{
    Draw as _, Engine, FrameTime, Next, Offscreen, OffscreenFormat, RenderError, Visibility,
    WorkingColor,
};
use cherenkov_gpu::{
    Gpu, GpuConfig,
    interop::{GpuContent, GpuContentBox, RedrawCallback, wgpu},
};
use filtrate::{AnimatedCallback, AnimatedTarget, FilterParam, Interpolator, WatchGuard, filters};

/// Clears its output to green and asks for the next frame every time it
/// draws, like a running particle system; reports each frame's `delta`.
struct Animated {
    deltas: mpsc::Sender<Duration>,
}

impl GpuContent for Animated {
    async fn setup(&mut self, _: &wgpu::Context<'_>) {}

    fn render(&mut self, frame: &mut wgpu::Frame<'_>) {
        self.deltas.send(frame.delta).expect("delta receiver");
        let mut encoder = frame
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        drop(pass);
        frame.queue.submit([encoder.finish()]);
        frame.request_redraw();
    }
}

#[test]
fn a_hidden_surface_pulls_no_gpu_content_and_wakes_nothing()
-> Result<(), Box<dyn std::error::Error>> {
    const TICK: Duration = Duration::from_millis(16);
    let engine = Engine::<Gpu>::new(GpuConfig::default())?;
    let wakes = Rc::new(Cell::new(0u32));
    engine.set_waker({
        let wakes = Rc::clone(&wakes);
        move || wakes.set(wakes.get() + 1)
    });
    let surface = engine.surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))?;
    let producer_wakes = Arc::new(AtomicUsize::new(0));
    let (deltas, drawn) = mpsc::channel();
    let content = GpuContentBox::new(Animated { deltas }, {
        let producer_wakes = Arc::clone(&producer_wakes);
        move || {
            producer_wakes.fetch_add(1, Ordering::Relaxed);
        }
    });
    let redraw = content.redraw_handle();
    let layer = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
        tx[&layer].content(engine.gpu_content((8, 8), content));
    });
    let t0 = Instant::now();
    assert!(
        matches!(engine.render(FrameTime::at(t0))?, Next::At { .. }),
        "the producer asks for the next frame"
    );
    assert_eq!(drawn.try_iter().count(), 1);
    engine.render(FrameTime::at(t0 + TICK))?;
    assert_eq!(drawn.try_iter().count(), 1);
    let wakes_before = wakes.get();

    surface.visibility(Visibility::Hidden)?;
    redraw.request_redraw();
    assert_eq!(
        producer_wakes.load(Ordering::Relaxed),
        0,
        "a hidden surface's producer wakes no host"
    );
    assert!(
        matches!(
            engine.render(FrameTime::at(t0 + 2 * TICK)),
            Err(RenderError::Hidden)
        ),
        "a render with every surface hidden is an error"
    );

    // Another surface keeps rendering: the hidden producer is neither
    // pulled nor keeps the frames coming.
    let other = engine.surface(Offscreen::new((8, 8), OffscreenFormat::LinearF16))?;
    assert_eq!(
        engine.render(FrameTime::at(t0 + 3 * TICK))?,
        Next::Idle,
        "a hidden producer asks for no frame"
    );
    assert_eq!(
        drawn.try_iter().count(),
        0,
        "a hidden surface's producer is not pulled"
    );
    assert_eq!(wakes.get(), wakes_before, "nothing woke the host");

    surface.visibility(Visibility::Visible)?;
    assert_eq!(wakes.get(), wakes_before + 1, "showing wakes the host once");
    let shown = t0 + Duration::from_secs(2);
    assert!(
        matches!(engine.render(FrameTime::at(shown))?, Next::At { .. }),
        "the shown producer runs again"
    );
    let deltas: Vec<_> = drawn.try_iter().collect();
    assert_eq!(
        deltas,
        [shown - (t0 + TICK)],
        "one producer frame at the show time, with no catch-up"
    );
    let pixels = surface.readback()?.pixels;
    assert!(
        (pixels[2 * 16 + 2][1] - 1.0).abs() < 1e-3,
        "the show frame draws the producer: {:?}",
        pixels[2 * 16 + 2]
    );
    assert_eq!(
        producer_wakes.load(Ordering::Relaxed),
        0,
        "the request made while hidden was drawn without waking the host"
    );
    drop(other);
    Ok(())
}

/// A filter parameter whose animated changes the test fires by hand.
struct ScriptedParam(mpsc::Sender<AnimatedCallback>);

impl FilterParam for ScriptedParam {
    fn snapshot(&self) -> f32 {
        0.0
    }

    fn watch_animated(&self, callback: AnimatedCallback) -> WatchGuard {
        self.0
            .send(callback)
            .expect("parameter callback channel open");
        WatchGuard::new(())
    }
}

struct LinearRamp(Duration);

impl Interpolator for LinearRamp {
    fn duration(&self) -> Duration {
        self.0
    }

    fn interpolate(&self, from: f32, to: f32, elapsed: Duration) -> f32 {
        let progress = (elapsed.as_secs_f32() / self.0.as_secs_f32()).min(1.0);
        (to - from).mul_add(progress, from)
    }
}

#[test]
fn a_hidden_surface_s_animated_filter_wakes_nothing() -> Result<(), Box<dyn std::error::Error>> {
    let wakes = Arc::new(AtomicUsize::new(0));
    let engine = Engine::<Gpu>::new(GpuConfig {
        redraw: Some(RedrawCallback::new({
            let wakes = Arc::clone(&wakes);
            move || {
                wakes.fetch_add(1, Ordering::Relaxed);
            }
        })),
        ..GpuConfig::default()
    })?;
    let surface = engine.surface(Offscreen::new((4, 4), OffscreenFormat::LinearF32))?;
    let (callbacks, installed) = mpsc::channel();
    let filter = engine.filter(filters::Brightness(ScriptedParam(callbacks)));
    surface.update(|tx| {
        tx[surface.root()]
            .filter(&filter)
            .content(surface.record(|r| {
                r.fill(
                    Rect::new(0.0, 0.0, 4.0, 4.0),
                    WorkingColor::new([0.1, 0.1, 0.1, 1.0]),
                );
            }));
    });
    let start = Instant::now();
    engine.render(FrameTime::at(start))?;
    let fire = |value, interpolator: Option<Box<dyn Interpolator>>| {
        let callback = installed
            .try_iter()
            .last()
            .expect("filter watcher installed");
        callback(AnimatedTarget {
            value,
            interpolator,
        });
        callback
    };
    // The same parameter wakes the host while the surface is visible.
    let callback = fire(0.2, None);
    assert_eq!(wakes.load(Ordering::Relaxed), 1);
    engine.render(FrameTime::at(start + Duration::from_millis(10)))?;

    surface.visibility(Visibility::Hidden)?;
    callback(AnimatedTarget {
        value: 0.8,
        interpolator: Some(Box::new(LinearRamp(Duration::from_millis(100)))),
    });
    assert_eq!(
        wakes.load(Ordering::Relaxed),
        1,
        "a hidden surface's filter wakes no host"
    );
    assert!(
        matches!(
            engine.render(FrameTime::at(start + Duration::from_millis(20))),
            Err(RenderError::Hidden)
        ),
        "a render with every surface hidden is an error"
    );
    let other = engine.surface(Offscreen::new((4, 4), OffscreenFormat::LinearF32))?;
    assert_eq!(
        engine.render(FrameTime::at(start + Duration::from_millis(30)))?,
        Next::Idle,
        "a hidden surface's filter animation asks for no frame"
    );

    surface.visibility(Visibility::Visible)?;
    engine.render(FrameTime::at(start + Duration::from_millis(500)))?;
    let shown = surface.readback()?.pixels[2 * 4 + 2];
    assert!((shown[0] - 0.9).abs() < 1.0e-3, "shown pixel: {shown:?}");
    drop(other);
    Ok(())
}
