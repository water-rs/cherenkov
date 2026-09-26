//! Filter composition, capture sizes and engine-provided effect timing.

use cherenkov::kurbo::Rect;
use cherenkov::{Draw, Engine, FrameTime, Next, Offscreen, OffscreenFormat, WorkingColor};
use cherenkov_gpu::{Gpu, GpuConfig, interop::EffectBox};
use filtrate::{
    Effect, EffectContext, EffectFrameTiming, EffectInput, EffectOutput, EffectRenderResult,
    EffectSetupResult,
};
use std::sync::mpsc;
use std::time::{Duration, Instant};

struct CopyEffect(mpsc::Sender<EffectFrameTiming>);

impl Effect for CopyEffect {
    async fn setup(&mut self, _: &EffectContext<'_>) -> EffectSetupResult {
        Ok(())
    }

    fn encode_render(
        &mut self,
        input: &EffectInput<'_>,
        output: &EffectOutput<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> EffectRenderResult {
        assert_eq!(input.texture.width(), input.width);
        assert_eq!(input.texture.height(), input.height);
        self.0.send(input.timing).expect("timing receiver alive");
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

#[test]
fn engine_executes_composed_filters_and_effects_after_resize()
-> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::<Gpu>::new(GpuConfig::default())?;
    let surface = engine.surface(Offscreen::new((32, 32), OffscreenFormat::LinearF16))?;
    let layer = surface.layer();
    let (send, receive) = mpsc::channel();
    let effect = engine.effect(EffectBox::from(CopyEffect(send)));
    let invert = engine.filter(filtrate::filters::Invert);
    surface.update(|tx| {
        tx[surface.root()].push(&layer).filter(&effect);
        tx[&layer].filter(&invert).content(surface.record(|r| {
            r.fill(
                Rect::new(0.0, 0.0, 32.0, 32.0),
                WorkingColor::new([0.25, 0.5, 0.75, 1.0]),
            );
        }));
    });
    let start = Instant::now();
    assert_eq!(engine.render(FrameTime::at(start))?, Next::Idle);
    assert_eq!(receive.try_recv()?.presentation_time(), Duration::ZERO);
    let readback = surface.readback()?;
    for (actual, expected) in readback.pixels[16 * 32 + 16]
        .into_iter()
        .zip([0.75, 0.5, 0.25, 1.0])
    {
        assert!(
            (actual - expected).abs() < 0.002,
            "filter output {actual}, expected {expected}"
        );
    }
    surface.resize((8, 8));
    engine.render(FrameTime::at(start + Duration::from_millis(250)))?;
    let timing = receive.try_recv()?;
    assert_eq!(timing.presentation_time(), Duration::from_millis(250));
    assert_eq!(timing.delta(), Duration::from_millis(250));
    assert_eq!(timing.sequence(), 1);
    Ok(())
}
