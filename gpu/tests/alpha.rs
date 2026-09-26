// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A translucent fill in a child layer composites exactly once over the
//! clear colour.

use cherenkov::kurbo::{Affine, Rect};
use cherenkov::{Draw, WorkingColor};
use cherenkov_gpu::{Engine, EngineError, Gpu, GpuConfig, Offscreen, OffscreenFormat};

#[test]
fn a_child_layer_draws_once() -> Result<(), Box<dyn std::error::Error>> {
    let engine = match Engine::<Gpu>::new(GpuConfig::default()) {
        Ok(engine) => engine,
        Err(EngineError::NoAdapter) => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    surface.clear_color(WorkingColor::new([0.8, 0.8, 0.8, 1.0]));
    let layer = surface.layer();
    surface.update(|tx| {
        tx[&layer].transform(Affine::IDENTITY).opacity(1.0);
        tx[surface.root()].push(&layer);
    });
    surface.update(|tx| {
        tx[&layer].content(surface.record(|c| {
            c.fill(
                Rect::new(8., 8., 56., 56.),
                WorkingColor::new([0.0, 0.0, 1.0, 0.5]),
            );
        }));
    });
    engine.render(cherenkov_gpu::FrameTime::now())?;
    let readback = surface.readback()?;
    let [r, g, b, a] = readback.pixels[(32 * readback.width + 32) as usize];
    assert!(
        (r - 0.4).abs() < 2e-3
            && (g - 0.4).abs() < 2e-3
            && (b - 0.9).abs() < 2e-3
            && (a - 1.0).abs() < 1e-6,
        "centre: {r} {g} {b} {a}"
    );
    Ok(())
}
