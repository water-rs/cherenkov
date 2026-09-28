// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Pixel-area coverage on oblique edges (lavapipe).

use cherenkov::kurbo::Rect;
use cherenkov::{Draw, Engine, EngineError, Offscreen, OffscreenFormat, WorkingColor};
use cherenkov_gpu::{Gpu, GpuConfig};

/// An engine, or `None` when no adapter exists.
fn engine() -> Option<Engine<Gpu>> {
    match Engine::<Gpu>::new(GpuConfig::default()) {
        Ok(engine) => Some(engine),
        Err(EngineError::Backend(_)) => None,
        Err(e) => panic!("engine init failed: {e}"),
    }
}

/// Supersampled area of `f(x, y) <= 0` over the unit pixel at (x, y).
fn supersample(x: u32, y: u32, f: impl Fn(f64, f64) -> f64) -> f64 {
    const N: u32 = 64;
    let mut inside = 0usize;
    for sy in 0..N {
        for sx in 0..N {
            let px = f64::from(x) + (f64::from(sx) + 0.5) / f64::from(N);
            let py = f64::from(y) + (f64::from(sy) + 0.5) / f64::from(N);
            if f(px, py) <= 0.0 {
                inside += 1;
            }
        }
    }
    f64::from(u32::try_from(inside).unwrap()) / f64::from(N * N)
}

/// A circle's oblique rim pixels get their exact pixel area, not the
/// linear ramp (which saturates at |d| = 0.5 instead of the true
//  0.707 at 45°); axis-aligned edges keep the ramp.
#[test]
fn oblique_rim_pixels_get_their_exact_area() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let surface = engine.surface(Offscreen::new((32, 32), OffscreenFormat::LinearF16))?;
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(
                cherenkov::kurbo::Circle::new((16.0, 16.0), 14.0),
                WorkingColor::WHITE,
            );
            c.fill(Rect::new(4.25, 4.5, 31.75, 8.5), WorkingColor::WHITE);
        }));
    });
    engine.render(cherenkov::FrameTime::now())?;
    let pixels = surface.readback()?.pixels;
    let alpha = |x: u32, y: u32| f64::from(pixels[(y * 32 + x) as usize][3]);
    // Every pixel in the rim band compares against the 64×64
    // supersampled exact circle area (the oracle's model).
    let mut max_err = 0.0_f64;
    for y in 0..32u32 {
        for x in 0..32u32 {
            let dx = f64::from(x) + 0.5 - 16.0;
            let dy = f64::from(y) + 0.5 - 16.0;
            let r = dx.hypot(dy);
            if !(12.0..=16.0).contains(&r) {
                continue;
            }
            // The white rect overpaints the circle inside it; skip it.
            if x >= 4 && (4..=8).contains(&y) {
                continue;
            }
            let exact = supersample(x, y, |px, py| {
                let ex = px - 16.0;
                let ey = py - 16.0;
                ex.hypot(ey) - 14.0
            });
            max_err = max_err.max((alpha(x, y) - exact).abs());
        }
    }
    assert!(max_err < 0.02, "max |Δalpha| on the rim: {max_err}");
    // Axis-aligned rect coverage stays the linear ramp — the
    // quarter-covered edge columns read 0.75, the interior 1.
    assert!((alpha(4, 5) - 0.75).abs() < 1e-3, "x=4: {}", alpha(4, 5));
    assert!((alpha(31, 5) - 0.75).abs() < 1e-3, "x=31: {}", alpha(31, 5));
    assert!(
        (alpha(16, 5) - 1.0).abs() < 1e-3,
        "interior: {}",
        alpha(16, 5)
    );
    Ok(())
}
