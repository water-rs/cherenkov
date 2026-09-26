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
    drop(layer);
    engine.render(FrameTime::now())?;
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    Ok(())
}
