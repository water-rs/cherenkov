//! Shader paints inherit geometry, clipping and engine frame timing.

use cherenkov::kurbo::{Circle, Rect};
use cherenkov::{
    Draw, Engine, FrameTime, Next, Offscreen, OffscreenFormat, ResourceError, ShaderPaint,
    ShaderSource,
};
use cherenkov_gpu::{Gpu, GpuConfig};
use std::time::{Duration, Instant};

#[test]
fn shader_registration_rejects_invalid_source() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::<Gpu>::new(GpuConfig::default())?;
    assert!(matches!(
        engine.shader(ShaderSource::wgsl("invalid shader")),
        Err(ResourceError::Shader(_))
    ));
    Ok(())
}

#[test]
fn shader_paint_uses_shape_coverage_and_presentation_time() -> Result<(), Box<dyn std::error::Error>>
{
    let engine = Engine::<Gpu>::new(GpuConfig::default())?;
    let shader =
        engine.shader(ShaderSource::wgsl(include_str!("shaders/paint.wgsl")).animated())?;
    let surface = engine.surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))?;
    let layer = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
        tx[&layer]
            .clip(Rect::new(0.0, 0.0, 8.0, 16.0))
            .content(surface.record(|r| {
                r.fill(
                    Circle::new((8.0, 8.0), 6.0),
                    ShaderPaint {
                        shader: shader.id(),
                        uniforms: vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    },
                );
            }));
    });
    let start = Instant::now();
    assert!(matches!(
        engine.render(FrameTime::at(start))?,
        Next::At { .. }
    ));
    engine.render(FrameTime::at(start + Duration::from_millis(500)))?;
    let pixels = surface.readback()?.pixels;
    assert!(
        (pixels[8 * 16 + 5][0] - 0.5).abs() < 0.001,
        "time advances without rerecording"
    );
    assert!(
        pixels[2 * 16 + 2][3].abs() < 0.001,
        "shape excludes bounding-box corner"
    );
    assert!(
        pixels[8 * 16 + 10][3].abs() < 0.001,
        "clip applies to shader paint"
    );
    drop(layer);
    assert_eq!(
        engine.render(FrameTime::at(start + Duration::from_secs(1)))?,
        Next::Idle
    );
    Ok(())
}

#[test]
fn producer_color_helpers_match_working_space_and_alpha() -> Result<(), Box<dyn std::error::Error>>
{
    let engine = Engine::<Gpu>::new(GpuConfig::default())?;
    let shader = engine.shader(ShaderSource::wgsl(include_str!("shaders/srgb.wgsl")))?;
    let surface = engine.surface(Offscreen::new((8, 4), OffscreenFormat::LinearF16))?;
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|r| {
            r.fill(
                Rect::new(0.0, 0.0, 8.0, 4.0),
                ShaderPaint {
                    shader: shader.id(),
                    uniforms: vec![],
                },
            );
        }));
    });
    engine.render(FrameTime::now())?;
    let expected = cherenkov::WorkingColor::from(cherenkov::Color::<cherenkov::Srgb>::new([
        1.0, 0.5, 0.25, 0.5,
    ]));
    let pixels = surface.readback()?.pixels;
    for index in [9, 14] {
        let [r, g, b, a] = expected.components;
        for (actual, expected) in pixels[index].into_iter().zip([r * a, g * a, b * a, a]) {
            assert!(
                (actual - expected).abs() < 0.001,
                "color conversion: {actual} != {expected}"
            );
        }
    }
    Ok(())
}
