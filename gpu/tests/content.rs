//! Custom content lifetime, clipping, and external redraw behavior.

use cherenkov::kurbo::{Affine, Rect};
use cherenkov::{__engine_test as split_test, __engine_wait as wait};
use cherenkov::{
    Draw as _, Engine, FrameTime, Next, Offscreen, OffscreenFormat, Visibility, WorkingColor,
};
use cherenkov_gpu::{
    Gpu, GpuConfig,
    interop::{GpuContent, GpuContentBox, wgpu},
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};

struct Producer {
    colors: mpsc::Receiver<wgpu::Color>,
    setups: Arc<AtomicUsize>,
    frames: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl Drop for Producer {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

impl GpuContent for Producer {
    #[cfg_attr(
        target_arch = "wasm32",
        expect(
            clippy::future_not_send,
            reason = "the wasm32 harness runs on the single-threaded page event loop"
        )
    )]
    async fn setup(&mut self, _: &wgpu::Context<'_>) {
        self.setups.fetch_add(1, Ordering::Relaxed);
    }

    fn render(&mut self, frame: &mut wgpu::Frame<'_>) {
        self.frames.fetch_add(1, Ordering::Relaxed);
        let color = self.colors.try_recv().expect("producer frame available");
        let mut encoder = frame
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        drop(pass);
        frame.queue.submit([encoder.finish()]);
    }
}

split_test! {
fn content_is_retained_clipped_and_wakes_an_idle_host() -> Result<(), Box<dyn std::error::Error>> {
    let engine = wait!(Engine::<Gpu>::new(GpuConfig::default()))?;
    let surface = wait!(engine.surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16)))?;
    let layer = surface.layer();
    let setups = Arc::new(AtomicUsize::new(0));
    let frames = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let wakes = Arc::new(AtomicUsize::new(0));
    let (send, colors) = mpsc::channel();
    let wake = wakes.clone();
    let content = GpuContentBox::new(
        Producer {
            colors,
            setups: setups.clone(),
            frames: frames.clone(),
            drops: drops.clone(),
        },
        move || {
            wake.fetch_add(1, Ordering::Relaxed);
        },
    );
    let redraw = content.redraw_handle();
    send.send(wgpu::Color::RED)?;
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
        tx[&layer]
            .transform(Affine::translate((4.0, 4.0)))
            .clip(Rect::new(0.0, 0.0, 4.0, 4.0))
            .opacity(0.5_f32)
            .content(engine.gpu_content((8, 8), content));
    });
    assert_eq!(wait!(engine.render(FrameTime::now()))?, Next::Idle);
    assert_eq!(setups.load(Ordering::Relaxed), 1);
    assert_eq!(frames.load(Ordering::Relaxed), 1);
    let pixels = wait!(surface.readback())?.pixels;
    assert!((pixels[5 * 16 + 5][0] - 0.5).abs() < 0.001);
    assert!(
        pixels[5 * 16 + 9][3].abs() < 0.001,
        "layer clip applies to GPU content"
    );
    wait!(engine.render(FrameTime::now()))?;
    assert_eq!(
        frames.load(Ordering::Relaxed),
        1,
        "idle render retains producer texture"
    );
    send.send(wgpu::Color::GREEN)?;
    redraw.request_redraw();
    redraw.request_redraw();
    assert_eq!(
        wakes.load(Ordering::Relaxed),
        1,
        "requests coalesce until consumed"
    );
    assert_eq!(wait!(engine.render(FrameTime::now()))?, Next::Idle);
    assert_eq!(frames.load(Ordering::Relaxed), 2);
    assert_eq!(setups.load(Ordering::Relaxed), 1);
    let pixels = wait!(surface.readback())?.pixels;
    assert!((pixels[5 * 16 + 5][1] - 0.5).abs() < 0.001);
    send.send(wgpu::Color::BLUE)?;
    surface.update(|tx| {
        tx[&layer].gpu_content_size((4, 4));
    });
    wait!(engine.render(FrameTime::now()))?;
    assert_eq!(setups.load(Ordering::Relaxed), 1, "resize preserves setup");
    assert_eq!(frames.load(Ordering::Relaxed), 3);
    assert!((wait!(surface.readback())?.pixels[5 * 16 + 5][2] - 0.5).abs() < 0.001);
    surface.update(|tx| {
        tx[surface.root()].remove(&layer);
    });
    assert_eq!(wait!(engine.render(FrameTime::now()))?, Next::Idle);
    let before = wakes.load(Ordering::Relaxed);
    send.send(wgpu::Color::RED)?;
    redraw.request_redraw();
    assert_eq!(
        wakes.load(Ordering::Relaxed),
        before,
        "detached content does not wake host"
    );
    assert_eq!(wait!(engine.render(FrameTime::now()))?, Next::Idle);
    assert_eq!(
        frames.load(Ordering::Relaxed),
        3,
        "detached producer is not rendered"
    );
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
    });
    assert_eq!(wait!(engine.render(FrameTime::now()))?, Next::Idle);
    assert_eq!(
        frames.load(Ordering::Relaxed),
        4,
        "reattach consumes the pending update"
    );
    drop(layer);
    wait!(engine.render(FrameTime::now()))?;
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    Ok(())
}
}

split_test! {
/// A hidden surface pulls no GPU content and asks for no frame, while a
/// visible surface of the same engine keeps rendering: a producer's
/// request wakes no host and is not drawn, and a live operand changed
/// while hidden wakes nothing. Showing the surface wakes the host once,
/// and that frame draws the producer's latest output and the operand's
/// latest value; the producer's requests wake the host again afterwards.
fn hidden_surface_pulls_no_content_and_shows_current_state()
-> Result<(), Box<dyn std::error::Error>> {
    let engine = wait!(Engine::<Gpu>::new(GpuConfig::default()))?;
    let surface = wait!(engine.surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16)))?;
    let other = wait!(engine.surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16)))?;
    let producer_layer = surface.layer();
    let fill_layer = surface.layer();
    let frames = Arc::new(AtomicUsize::new(0));
    let producer_wakes = Arc::new(AtomicUsize::new(0));
    let (send, colors) = mpsc::channel();
    let content = GpuContentBox::new(
        Producer {
            colors,
            setups: Arc::new(AtomicUsize::new(0)),
            frames: frames.clone(),
            drops: Arc::new(AtomicUsize::new(0)),
        },
        {
            let wakes = producer_wakes.clone();
            move || {
                wakes.fetch_add(1, Ordering::Relaxed);
            }
        },
    );
    let redraw = content.redraw_handle();
    let fill = nami::binding(WorkingColor::WHITE);
    let recorded =
        surface.record(|c| c.fill(Rect::new(8.0, 8.0, 16.0, 16.0), fill.clone()));
    send.send(wgpu::Color::RED)?;
    surface.update(|tx| {
        tx[surface.root()].push(&producer_layer).push(&fill_layer);
        tx[&producer_layer].content(engine.gpu_content((8, 8), content));
        tx[&fill_layer].content(recorded);
    });
    assert_eq!(wait!(engine.render(FrameTime::now()))?, Next::Idle);
    assert_eq!(frames.load(Ordering::Relaxed), 1);

    let wakes = Arc::new(AtomicUsize::new(0));
    engine.set_waker({
        let wakes = wakes.clone();
        move || {
            wakes.fetch_add(1, Ordering::Relaxed);
        }
    });
    surface.visibility(Visibility::Hidden)?;
    // The reply lands only after the render thread applied the hide.
    let _ = wait!(engine.memory());
    send.send(wgpu::Color::GREEN)?;
    redraw.request_redraw();
    fill.set(WorkingColor::BLACK);
    assert_eq!(
        producer_wakes.load(Ordering::Relaxed),
        0,
        "a hidden surface's producer wakes no host"
    );
    assert_eq!(
        wakes.load(Ordering::Relaxed),
        0,
        "a hidden surface's operand wakes no host"
    );
    other.clear_color(WorkingColor::WHITE);
    assert_eq!(wakes.load(Ordering::Relaxed), 1, "the visible surface wakes");
    assert_eq!(
        wait!(engine.render(FrameTime::now()))?,
        Next::Idle,
        "the hidden producer's request asks for no frame"
    );
    assert_eq!(
        frames.load(Ordering::Relaxed),
        1,
        "a hidden surface's producer is not pulled"
    );
    assert!(
        (wait!(other.readback())?.pixels[0][0] - 1.0).abs() < 0.001,
        "the visible surface renders"
    );

    surface.visibility(Visibility::Visible)?;
    assert_eq!(
        wakes.load(Ordering::Relaxed),
        2,
        "showing the surface asks for one frame"
    );
    assert_eq!(wait!(engine.render(FrameTime::now()))?, Next::Idle);
    assert_eq!(
        frames.load(Ordering::Relaxed),
        2,
        "the shown frame draws the pending producer output"
    );
    let pixels = wait!(surface.readback())?.pixels;
    assert!(
        (pixels[2 * 16 + 2][1] - 1.0).abs() < 0.001 && pixels[2 * 16 + 2][0].abs() < 0.001,
        "the producer's latest output: {:?}",
        pixels[2 * 16 + 2]
    );
    assert!(
        pixels[12 * 16 + 12][..3].iter().all(|c| c.abs() < 0.001)
            && (pixels[12 * 16 + 12][3] - 1.0).abs() < 0.001,
        "the operand's latest value: {:?}",
        pixels[12 * 16 + 12]
    );
    send.send(wgpu::Color::BLUE)?;
    redraw.request_redraw();
    assert_eq!(
        producer_wakes.load(Ordering::Relaxed),
        1,
        "a visible surface's producer wakes the host again"
    );
    Ok(())
}
}

struct TimeSample {
    elapsed: std::time::Duration,
    delta: std::time::Duration,
    size: (u32, u32),
    scale: f32,
}

struct TimedProducer {
    adapter: mpsc::Sender<String>,
    samples: mpsc::Sender<TimeSample>,
    first: bool,
}

impl GpuContent for TimedProducer {
    #[cfg_attr(
        target_arch = "wasm32",
        expect(
            clippy::future_not_send,
            reason = "the wasm32 harness runs on the single-threaded page event loop"
        )
    )]
    async fn setup(&mut self, context: &wgpu::Context<'_>) {
        self.adapter
            .send(context.adapter.get_info().name)
            .expect("adapter receiver");
    }

    fn render(&mut self, frame: &mut wgpu::Frame<'_>) {
        self.samples
            .send(TimeSample {
                elapsed: frame.elapsed,
                delta: frame.delta,
                size: (frame.width, frame.height),
                scale: frame.scale,
            })
            .expect("sample receiver");
        if self.first {
            frame.request_redraw();
            self.first = false;
        }
    }
}

split_test! {
fn producer_samples_engine_time_and_keeps_setup_across_display_changes()
-> Result<(), Box<dyn std::error::Error>> {
    use std::time::Duration;
use cherenkov::Instant;
    let engine = wait!(Engine::<Gpu>::new(GpuConfig::default()))?;
    let surface =
        wait!(engine.surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16).rate(30..=120)))?;
    let (adapter, adapters) = mpsc::channel();
    let (samples, times) = mpsc::channel();
    surface.update(|tx| {
        tx[surface.root()].content(engine.gpu_content(
            (8, 8),
            GpuContentBox::new(
                TimedProducer {
                    adapter,
                    samples,
                    first: true,
                },
                || {},
            ),
        ));
    });
    let start = Instant::now();
    let Next::At { rate, .. } = wait!(engine.render(FrameTime::at(start)))? else {
        panic!("producer requests another frame");
    };
    assert_eq!(rate, 30..=120);
    assert_eq!(adapters.try_recv()?, engine.info().name);
    let first = times.try_recv()?;
    assert_eq!(first.elapsed, Duration::ZERO);
    assert_eq!(first.delta, Duration::ZERO);
    assert_eq!(first.size, (8, 8));
    assert_eq!(
        wait!(engine.render(FrameTime::at(start + Duration::from_millis(250))))?,
        Next::Idle
    );
    let second = times.try_recv()?;
    assert_eq!(second.elapsed, Duration::from_millis(250));
    assert_eq!(second.delta, Duration::from_millis(250));
    surface.display(cherenkov::Display {
        scale: 2.0,
        ..cherenkov::Display::default()
    })?;
    wait!(engine.render(FrameTime::at(start + Duration::from_millis(500))))?;
    let third = times.try_recv()?;
    assert_eq!(third.elapsed, Duration::from_millis(500));
    assert_eq!(third.delta, Duration::from_millis(250));
    assert_eq!(third.scale.to_bits(), 2.0_f32.to_bits());
    surface.update(|tx| {
        tx[surface.root()].gpu_content_size((4, 4));
    });
    wait!(engine.render(FrameTime::at(start + Duration::from_millis(750))))?;
    let resized = times.try_recv()?;
    assert_eq!(resized.size, (4, 4));
    assert_eq!(resized.elapsed, Duration::from_millis(750));
    assert_eq!(resized.delta, Duration::from_millis(250));
    assert!(adapters.try_recv().is_err(), "setup is not repeated");
    assert_eq!(
        wait!(engine.render(FrameTime::at(start + Duration::from_secs(1))))?,
        Next::Idle
    );
    assert!(times.try_recv().is_err(), "idle retains producer output");
    Ok(())
}
}

/// An effect whose redraw-callback installation parks the render thread:
/// it meets the test at `parked`, then waits at `release`, so a message
/// queued in between is not applied until the test releases it.
#[cfg(not(target_arch = "wasm32"))]
struct ParkingEffect {
    parked: Arc<std::sync::Barrier>,
    release: Arc<std::sync::Barrier>,
}

#[cfg(not(target_arch = "wasm32"))]
impl filtrate::Effect for ParkingEffect {
    fn set_redraw_callback(&mut self, _callback: filtrate::EffectRedrawCallback) {
        self.parked.wait();
        self.release.wait();
    }

    fn setup(
        &mut self,
        _: &filtrate::EffectContext<'_>,
    ) -> impl std::future::Future<Output = filtrate::EffectSetupResult> {
        std::future::ready(Ok(()))
    }

    fn encode_render(
        &mut self,
        _: &filtrate::EffectInput<'_>,
        _: &filtrate::EffectOutput<'_>,
        _: &mut wgpu::CommandEncoder,
    ) -> filtrate::EffectRenderResult {
        unreachable!("the parking effect is never attached")
    }
}

/// An effect that hands its redraw callback to the test and passes its
/// input through.
#[cfg(not(target_arch = "wasm32"))]
struct CallbackEffect(mpsc::Sender<filtrate::EffectRedrawCallback>);

#[cfg(not(target_arch = "wasm32"))]
impl filtrate::Effect for CallbackEffect {
    fn set_redraw_callback(&mut self, callback: filtrate::EffectRedrawCallback) {
        self.0.send(callback).expect("callback receiver");
    }

    fn setup(
        &mut self,
        _: &filtrate::EffectContext<'_>,
    ) -> impl std::future::Future<Output = filtrate::EffectSetupResult> {
        std::future::ready(Ok(()))
    }

    fn encode_render(
        &mut self,
        input: &filtrate::EffectInput<'_>,
        output: &filtrate::EffectOutput<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> filtrate::EffectRenderResult {
        encoder.copy_texture_to_texture(
            input.texture.as_image_copy(),
            output.texture.as_image_copy(),
            wgpu::Extent3d {
                width: input.width,
                height: input.height,
                depth_or_array_layers: 1,
            },
        );
        Ok(false)
    }
}

/// A hidden surface's producer and filter stop waking the host the moment
/// the host hides the surface, before the render thread has applied the
/// change: the render thread is parked across the hide and the requests.
/// Showing the surface draws both, and both wake the host again (#204).
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn hidden_wakes_stop_before_the_render_thread_applies_the_hide()
-> Result<(), Box<dyn std::error::Error>> {
    use cherenkov_gpu::interop::{EffectBox, RedrawCallback};

    let filter_wakes = Arc::new(AtomicUsize::new(0));
    let engine = Engine::<Gpu>::new(GpuConfig {
        redraw: Some(RedrawCallback::new({
            let wakes = filter_wakes.clone();
            move || {
                wakes.fetch_add(1, Ordering::Relaxed);
            }
        })),
        ..GpuConfig::default()
    })?;
    let surface = engine.surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))?;
    let producer_layer = surface.layer();
    let filtered_layer = surface.layer();
    let frames = Arc::new(AtomicUsize::new(0));
    let producer_wakes = Arc::new(AtomicUsize::new(0));
    let (send, colors) = mpsc::channel();
    let content = GpuContentBox::new(
        Producer {
            colors,
            setups: Arc::new(AtomicUsize::new(0)),
            frames: frames.clone(),
            drops: Arc::new(AtomicUsize::new(0)),
        },
        {
            let wakes = producer_wakes.clone();
            move || {
                wakes.fetch_add(1, Ordering::Relaxed);
            }
        },
    );
    let redraw = content.redraw_handle();
    let (callbacks, installed) = mpsc::channel();
    let effect = engine.effect(EffectBox::from(CallbackEffect(callbacks)));
    send.send(wgpu::Color::RED)?;
    surface.update(|tx| {
        tx[surface.root()]
            .push(&producer_layer)
            .push(&filtered_layer);
        tx[&producer_layer].content(engine.gpu_content((8, 8), content));
        tx[&filtered_layer].filter(&effect).content(
            surface.record(|c| c.fill(Rect::new(8.0, 8.0, 16.0, 16.0), WorkingColor::WHITE)),
        );
    });
    assert_eq!(engine.render(FrameTime::now())?, Next::Idle);
    let callback = installed.try_recv()?;

    let parked = Arc::new(std::sync::Barrier::new(2));
    let release = Arc::new(std::sync::Barrier::new(2));
    let _parking = engine.effect(EffectBox::from(ParkingEffect {
        parked: parked.clone(),
        release: release.clone(),
    }));
    parked.wait();
    let hidden = surface.visibility(Visibility::Hidden);
    redraw.request_redraw();
    callback();
    let woke = (
        producer_wakes.load(Ordering::Relaxed),
        filter_wakes.load(Ordering::Relaxed),
    );
    // Released before asserting, so a failure does not leave the render
    // thread parked under the engine's drop.
    release.wait();
    hidden?;
    assert_eq!(
        woke.0, 0,
        "a hidden surface's producer woke the host before the render thread applied the hide"
    );
    assert_eq!(
        woke.1, 0,
        "a hidden surface's filter woke the host before the render thread applied the hide"
    );

    send.send(wgpu::Color::GREEN)?;
    surface.visibility(Visibility::Visible)?;
    assert_eq!(engine.render(FrameTime::now())?, Next::Idle);
    assert_eq!(
        frames.load(Ordering::Relaxed),
        2,
        "the shown frame draws the producer's pending request"
    );
    redraw.request_redraw();
    callback();
    assert_eq!(
        producer_wakes.load(Ordering::Relaxed),
        1,
        "a shown surface's producer wakes the host again"
    );
    assert_eq!(
        filter_wakes.load(Ordering::Relaxed),
        1,
        "a shown surface's filter wakes the host again"
    );
    Ok(())
}
