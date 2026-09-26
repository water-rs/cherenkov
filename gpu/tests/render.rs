// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Integration tests for the GPU backend. They skip when no GPU adapter is
//! available, which CI may not have.

use cherenkov::kurbo::{BezPath, Rect};
use cherenkov::{Draw, WorkingColor};
use cherenkov_gpu::{
    Engine, EngineError, Gpu, GpuConfig, Next, Offscreen, OffscreenFormat, RenderError, Unsupported,
};

/// An engine, or `None` when no adapter exists.
fn engine() -> Option<Engine<Gpu>> {
    match Engine::new(GpuConfig::default()) {
        Ok(engine) => Some(engine),
        Err(EngineError::NoAdapter) => None,
        Err(e) => panic!("engine init failed: {e}"),
    }
}

#[test]
#[expect(clippy::float_cmp, reason = "the clear colour is exact")]
fn a_red_rect_renders_and_reads_back() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(
                Rect::new(8., 8., 56., 56.),
                WorkingColor::new([1., 0., 0., 1.]),
            );
        }));
    });
    let next = engine.render(cherenkov_gpu::FrameTime::now())?;
    assert_eq!(next, Next::Idle);
    let readback = surface.readback()?;
    let px = |x: u32, y: u32| readback.pixels[(y * readback.width + x) as usize];
    let [r, g, b, a] = px(32, 32);
    assert!(
        (r - 1.0).abs() < 1e-2 && g.abs() < 1e-2 && b.abs() < 1e-2 && (a - 1.0).abs() < 1e-2,
        "center pixel: {r} {g} {b} {a}"
    );
    assert_eq!(px(2, 2), [0.0; 4], "corner pixel must be the clear colour");
    Ok(())
}

#[test]
fn a_path_fill_renders() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    let mut path = BezPath::new();
    path.move_to((4., 4.));
    path.curve_to((20., 60.), (44., 60.), (60., 4.));
    path.close_path();
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(path, WorkingColor::new([1., 0., 0., 1.]));
        }));
    });
    engine.render(cherenkov_gpu::FrameTime::now())?;
    let readback = surface.readback()?;
    let [r, ..] = readback.pixels[(30 * readback.width + 30) as usize];
    assert!(r > 0.5, "interior pixel: {r}");
    Ok(())
}

#[test]
fn a_path_shadow_reports_unsupported() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    let mut path = BezPath::new();
    path.move_to((4., 4.));
    path.curve_to((20., 60.), (44., 60.), (60., 4.));
    path.close_path();
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.shadow(
                path,
                cherenkov::Shadow::new(4.0, WorkingColor::new([0., 0., 0., 1.])),
            );
        }));
    });
    let result = engine.render(cherenkov_gpu::FrameTime::now());
    assert!(
        matches!(result, Err(RenderError::Unsupported(Unsupported::Path))),
        "expected Unsupported(Path), got {result:?}"
    );
    Ok(())
}
