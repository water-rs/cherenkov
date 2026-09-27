// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Real browser WebGPU execution, including non-Send producers and device reuse.
#![cfg(target_arch = "wasm32")]

use cherenkov::kurbo::Rect;
use cherenkov::{Draw, Engine, FrameTime, Next, Offscreen, OffscreenFormat, WorkingColor};
use cherenkov_gpu::{
    Gpu, GpuConfig,
    interop::{GpuContent, GpuContentBox, SharedDevice, wgpu},
};
use std::cell::Cell;
use std::rc::Rc;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

struct LocalProducer {
    setups: Rc<Cell<u32>>,
    frames: Rc<Cell<u32>>,
    drops: Rc<Cell<u32>>,
    during_setup: Option<Box<dyn FnOnce()>>,
    device: wgpu::Device,
}

impl Drop for LocalProducer {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}

impl GpuContent for LocalProducer {
    async fn setup(&mut self, context: &wgpu::Context<'_>) {
        assert_eq!(context.device, &self.device, "supplied device identity");
        assert_eq!(context.device.features(), self.device.features());
        gloo_timers::future::TimeoutFuture::new(0).await;
        self.during_setup.take().expect("one setup")();
        self.setups.set(self.setups.get() + 1);
    }
    fn render(&mut self, frame: &mut wgpu::Frame<'_>) {
        self.frames.set(self.frames.get() + 1);
        let mut encoder = frame
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        }));
        frame.queue.submit([encoder.finish()]);
    }
}

#[wasm_bindgen_test(async)]
async fn local_producers_share_device_and_preserve_wakes_during_await() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("WebGPU adapter required");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .expect("WebGPU device");
    let engine = Engine::<Gpu>::new(GpuConfig {
        device: Some(SharedDevice {
            instance,
            adapter,
            device: device.clone(),
            queue,
        }),
        ..Default::default()
    })
    .await
    .expect("engine");
    let surface = engine
        .surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))
        .await
        .expect("surface");
    let color = nami::binding(WorkingColor::WHITE);
    surface.update(|tx| {
        tx[surface.root()]
            .content(surface.record(|r| r.fill(Rect::new(0., 0., 8., 16.), color.clone())));
    });
    engine.render(FrameTime::now()).await.expect("first frame");
    let wakes = Rc::new(Cell::new(0));
    let counter = wakes.clone();
    engine.set_waker(move || counter.set(counter.get() + 1));
    let setups = Rc::new(Cell::new(0));
    let frames = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let producer = GpuContentBox::new(
        LocalProducer {
            setups: setups.clone(),
            frames: frames.clone(),
            drops: drops.clone(),
            device,
            during_setup: Some(Box::new(move || {
                color.set(WorkingColor::new([0., 1., 0., 1.]))
            })),
        },
        || {},
    );
    let layer = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
        tx[&layer]
            .transform(cherenkov::kurbo::Affine::translate((8., 0.)))
            .content(engine.gpu_content((8, 16), producer));
    });
    let before = wakes.get();
    engine
        .render(FrameTime::now())
        .await
        .expect("producer setup");
    assert_eq!(
        wakes.get(),
        before + 1,
        "signal during setup requests next frame"
    );
    assert_eq!(
        engine.render(FrameTime::now()).await.expect("live update"),
        Next::Idle
    );
    assert_eq!(engine.stats().commands_lowered, 1);
    let pixels = surface.readback().await.expect("browser mapping").pixels;
    assert!((pixels[4 * 16 + 4][1] - 1.).abs() < 0.001);
    assert!((pixels[4 * 16 + 12][0] - 1.).abs() < 0.001);
    wasm_bindgen_test::console_log!("BROWSER_PIXELS {:?}", pixels);
    assert_eq!(setups.get(), 1);
    assert_eq!(frames.get(), 1);
    engine.render(FrameTime::now()).await.expect("idle");
    assert_eq!(engine.stats().commands_lowered, 0);
    assert_eq!(frames.get(), 1);
    drop(engine);
    // Readback queues after shutdown and observes disconnection, including
    // when surface/layer handles outlive the engine.
    assert!(surface.readback().await.is_err());
    assert_eq!(drops.get(), 1);
}

#[wasm_bindgen_test(async)]
async fn shader_validation_yields_and_returns_errors() {
    let engine = Engine::<Gpu>::new(GpuConfig::default())
        .await
        .expect("engine");
    assert!(matches!(
        engine
            .shader(cherenkov::ShaderSource::wgsl("invalid shader"))
            .await,
        Err(cherenkov::ResourceError::Shader(_))
    ));
    let shader = engine.shader(cherenkov::ShaderSource::wgsl("@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 0.0, 1.0, 1.0); }")).await.expect("shader");
    let surface = engine
        .surface(Offscreen::new((8, 8), OffscreenFormat::LinearF16))
        .await
        .expect("surface");
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|r| {
            r.fill(
                Rect::new(0., 0., 8., 8.),
                cherenkov::ShaderPaint {
                    shader: shader.id(),
                    uniforms: vec![],
                },
            )
        }));
    });
    assert_eq!(
        engine
            .render(FrameTime::now())
            .await
            .expect("shader render"),
        Next::Idle
    );
    let pixels = surface.readback().await.expect("pixels").pixels;
    assert!((pixels[4 * 8 + 4][2] - 1.).abs() < 0.001);
}

struct YieldingEffect {
    callback: Option<filtrate::EffectRedrawCallback>,
    frames: Rc<Cell<u32>>,
}

impl filtrate::Effect for YieldingEffect {
    fn set_redraw_callback(&mut self, callback: filtrate::EffectRedrawCallback) {
        self.callback = Some(callback);
    }
    async fn setup(&mut self, _: &filtrate::EffectContext<'_>) -> filtrate::EffectSetupResult {
        gloo_timers::future::TimeoutFuture::new(0).await;
        self.callback.as_ref().expect("redraw installed")();
        Ok(())
    }
    fn encode_render(
        &mut self,
        input: &filtrate::EffectInput<'_>,
        output: &filtrate::EffectOutput<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> filtrate::EffectRenderResult {
        self.frames.set(self.frames.get() + 1);
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

#[wasm_bindgen_test(async)]
async fn filters_keep_redraw_requests_made_during_async_setup() {
    let engine = Engine::<Gpu>::new(GpuConfig::default())
        .await
        .expect("engine");
    let frames = Rc::new(Cell::new(0));
    let filter = engine.effect(cherenkov_gpu::interop::EffectBox::from(YieldingEffect {
        callback: None,
        frames: frames.clone(),
    }));
    let surface = engine
        .surface(Offscreen::new((8, 8), OffscreenFormat::LinearF16))
        .await
        .expect("surface");
    surface.update(|tx| {
        tx[surface.root()]
            .filter(&filter)
            .content(surface.record(|r| r.fill(Rect::new(0., 0., 8., 8.), WorkingColor::WHITE)));
    });
    assert!(matches!(
        engine.render(FrameTime::now()).await.expect("setup frame"),
        Next::At { .. }
    ));
    assert_eq!(
        engine
            .render(FrameTime::now())
            .await
            .expect("requested frame"),
        Next::Idle
    );
    assert_eq!(frames.get(), 2);
    engine.render(FrameTime::now()).await.expect("idle");
    assert_eq!(frames.get(), 2);
    let pixels = surface.readback().await.expect("filtered pixels").pixels;
    assert!((pixels[4 * 8 + 4][0] - 1.).abs() < 0.001);
}

#[wasm_bindgen_test(async)]
async fn browser_incremental_matches_full_lowering() {
    use cherenkov::Backend;
    let (mut renderer, _) = Gpu::init(GpuConfig::default())
        .await
        .expect("WebGPU adapter required");
    cherenkov::testing::incremental::equivalence(&mut renderer).await;
}
