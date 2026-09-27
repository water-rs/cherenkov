//! Custom content lifetime, clipping, and external redraw behavior.

use cherenkov::kurbo::{Affine, Rect};
use cherenkov::{Engine, FrameTime, Next, Offscreen, OffscreenFormat};
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

#[test]
fn content_is_retained_clipped_and_wakes_an_idle_host() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::<Gpu>::new(GpuConfig::default())?;
    let surface = engine.surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))?;
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
    assert_eq!(engine.render(FrameTime::now())?, Next::Idle);
    assert_eq!(setups.load(Ordering::Relaxed), 1);
    assert_eq!(frames.load(Ordering::Relaxed), 1);
    let pixels = surface.readback()?.pixels;
    assert!((pixels[5 * 16 + 5][0] - 0.5).abs() < 0.001);
    assert!(
        pixels[5 * 16 + 9][3].abs() < 0.001,
        "layer clip applies to GPU content"
    );
    engine.render(FrameTime::now())?;
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
    assert_eq!(engine.render(FrameTime::now())?, Next::Idle);
    assert_eq!(frames.load(Ordering::Relaxed), 2);
    assert_eq!(setups.load(Ordering::Relaxed), 1);
    let pixels = surface.readback()?.pixels;
    assert!((pixels[5 * 16 + 5][1] - 0.5).abs() < 0.001);
    send.send(wgpu::Color::BLUE)?;
    surface.update(|tx| {
        tx[&layer].gpu_content_size((4, 4));
    });
    engine.render(FrameTime::now())?;
    assert_eq!(setups.load(Ordering::Relaxed), 1, "resize preserves setup");
    assert_eq!(frames.load(Ordering::Relaxed), 3);
    assert!((surface.readback()?.pixels[5 * 16 + 5][2] - 0.5).abs() < 0.001);
    surface.update(|tx| {
        tx[surface.root()].remove(&layer);
    });
    assert_eq!(engine.render(FrameTime::now())?, Next::Idle);
    let before = wakes.load(Ordering::Relaxed);
    send.send(wgpu::Color::RED)?;
    redraw.request_redraw();
    assert_eq!(
        wakes.load(Ordering::Relaxed),
        before,
        "detached content does not wake host"
    );
    assert_eq!(engine.render(FrameTime::now())?, Next::Idle);
    assert_eq!(
        frames.load(Ordering::Relaxed),
        3,
        "detached producer is not rendered"
    );
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
    });
    assert_eq!(engine.render(FrameTime::now())?, Next::Idle);
    assert_eq!(
        frames.load(Ordering::Relaxed),
        4,
        "reattach consumes the pending update"
    );
    drop(layer);
    engine.render(FrameTime::now())?;
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    Ok(())
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

#[test]
fn producer_samples_engine_time_and_keeps_setup_across_display_changes()
-> Result<(), Box<dyn std::error::Error>> {
    use std::time::{Duration, Instant};
    let engine = Engine::<Gpu>::new(GpuConfig::default())?;
    let surface =
        engine.surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16).rate(30..=120))?;
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
    let Next::At { rate, .. } = engine.render(FrameTime::at(start))? else {
        panic!("producer requests another frame");
    };
    assert_eq!(rate, 30..=120);
    assert_eq!(adapters.try_recv()?, engine.info().name);
    let first = times.try_recv()?;
    assert_eq!(first.elapsed, Duration::ZERO);
    assert_eq!(first.delta, Duration::ZERO);
    assert_eq!(first.size, (8, 8));
    assert_eq!(
        engine.render(FrameTime::at(start + Duration::from_millis(250)))?,
        Next::Idle
    );
    let second = times.try_recv()?;
    assert_eq!(second.elapsed, Duration::from_millis(250));
    assert_eq!(second.delta, Duration::from_millis(250));
    surface.display(cherenkov::Display {
        scale: 2.0,
        ..cherenkov::Display::default()
    })?;
    engine.render(FrameTime::at(start + Duration::from_millis(500)))?;
    let third = times.try_recv()?;
    assert_eq!(third.elapsed, Duration::from_millis(500));
    assert_eq!(third.delta, Duration::from_millis(250));
    assert_eq!(third.scale.to_bits(), 2.0_f32.to_bits());
    surface.update(|tx| {
        tx[surface.root()].gpu_content_size((4, 4));
    });
    engine.render(FrameTime::at(start + Duration::from_millis(750)))?;
    let resized = times.try_recv()?;
    assert_eq!(resized.size, (4, 4));
    assert_eq!(resized.elapsed, Duration::from_millis(750));
    assert_eq!(resized.delta, Duration::from_millis(250));
    assert!(adapters.try_recv().is_err(), "setup is not repeated");
    assert_eq!(
        engine.render(FrameTime::at(start + Duration::from_secs(1)))?,
        Next::Idle
    );
    assert!(times.try_recv().is_err(), "idle retains producer output");
    Ok(())
}
