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

/// Two dirty surfaces sharing one frame: the second surface lowers far
/// more instances than the initial instance buffer holds, forcing a grow
/// that must preserve the first surface's upload.
#[test]
fn an_earlier_surfaces_uploads_survive_a_shared_buffer_grow()
-> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let small = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    let big = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    small.update(|tx| {
        tx[small.root()].content(small.record(|c| {
            c.fill(
                Rect::new(8., 8., 56., 56.),
                WorkingColor::new([1., 0., 0., 1.]),
            );
        }));
    });
    big.update(|tx| {
        tx[big.root()].content(big.record(|c| {
            // 3000 4×4 green rects in a grid — far past the 16-instance
            // initial buffer.
            for i in 0..3000u32 {
                let x = f64::from(i % 55) * 1.0;
                let y = f64::from(i / 55) * 1.0;
                if y > 60.0 {
                    break;
                }
                c.fill(
                    Rect::new(x, y, x + 0.5, y + 0.5),
                    WorkingColor::new([0., 1., 0., 1.]),
                );
            }
        }));
    });
    let next = engine.render(cherenkov_gpu::FrameTime::now())?;
    assert_eq!(next, Next::Idle);
    let small_rb = small.readback()?;
    let [r, g, b, a] = small_rb.pixels[(32 * small_rb.width + 32) as usize];
    assert!(
        (r - 1.0).abs() < 1e-2 && g.abs() < 1e-2 && b.abs() < 1e-2 && (a - 1.0).abs() < 1e-2,
        "first surface's pixel must still be red: {r} {g} {b} {a}"
    );
    let big_rb = big.readback()?;
    let [r, g, b, a] = big_rb.pixels[(4 * big_rb.width + 4) as usize];
    // 0.5-wide rects cover the pixel partially; green is what matters.
    assert!(
        g > 0.1 && r < 0.1,
        "second surface must render green: {r} {g} {b} {a}"
    );
    Ok(())
}

/// A `Shadow` immediately followed by an opaque solid fill of the same
/// shape lowers to up-to-four border quads: the covered interior is
/// skipped. An opaque card's result must be pixel-identical outside the
/// card and within fill-alpha error inside, vs the same card whose fill
/// is alpha 0.999 (which disables the split).
#[test]
fn a_shadow_under_an_opaque_fill_loses_only_its_interior() -> Result<(), Box<dyn std::error::Error>>
{
    let Some(engine) = engine() else {
        return Ok(());
    };
    let render_card = |alpha: f32| -> Result<cherenkov_gpu::Readback, Box<dyn std::error::Error>> {
        let surface = engine.surface(Offscreen::new((128, 128), OffscreenFormat::LinearF16))?;
        surface.update(|tx| {
            tx[surface.root()].content(surface.record(|c| {
                let card = Rect::new(24., 24., 104., 104.);
                c.shadow(
                    card,
                    cherenkov::Shadow::new(6.0, WorkingColor::new([0., 0., 0., 1.]))
                        .offset((2., 3.)),
                );
                c.fill(card, WorkingColor::new([0.9, 0.3, 0.1, alpha]));
            }));
        });
        engine.render(cherenkov_gpu::FrameTime::now())?;
        Ok(surface.readback()?)
    };
    let split = render_card(1.0)?;
    let whole = render_card(0.999)?;
    for y in 0..whole.height {
        for x in 0..whole.width {
            let i = (y * whole.width + x) as usize;
            let inside = (24.0..=104.0).contains(&(f64::from(x) + 0.5))
                && (24.0..=104.0).contains(&(f64::from(y) + 0.5));
            // Strips vs one quad interpolate `local` over different
            // corners, so coverage can round one f16 ulp (~1e-3 at 1.0)
            // either way; 2e-3 bounds that while still catching a
            // missed or doubled rasterization region.
            let tol = if inside { 1e-2 } else { 2e-3 };
            for c in 0..4 {
                let d = (split.pixels[i][c] - whole.pixels[i][c]).abs();
                assert!(
                    d <= tol,
                    "pixel ({x},{y}) ch {c}: split {} vs whole {}",
                    split.pixels[i][c],
                    whole.pixels[i][c]
                );
            }
        }
    }
    Ok(())
}

/// A large axis-aligned box fill lowers to one `KIND_SPAN` interior plus
/// border quads: sampled pixels must equal the analytic gradient, and an
/// edge pixel must still show partial coverage.
#[test]
fn a_large_fill_spans_its_interior() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let surface = engine.surface(Offscreen::new((320, 320), OffscreenFormat::LinearF16))?;
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(
                Rect::new(10.5, 10.5, 309.5, 309.5),
                cherenkov::LinearGradient::new((10., 0.), (310., 0.))
                    .stop(0.0, WorkingColor::new([1., 0., 0., 1.]))
                    .stop(1.0, WorkingColor::new([0., 0., 1., 1.])),
            );
        }));
    });
    engine.render(cherenkov_gpu::FrameTime::now())?;
    let readback = surface.readback()?;
    let px = |x: u32, y: u32| readback.pixels[(y * readback.width + x) as usize];
    for (px_x, px_y) in [
        (40u32, 160u32),
        (160, 160),
        (280, 160),
        (160, 40),
        (160, 280),
    ] {
        let [r, g, b, a] = px(px_x, px_y);
        let t = (f64::from(px_x) - 10.0 + 0.5) / 300.0;
        assert!(
            (f64::from(r) - (1.0 - t)).abs() < 1.0 / 255.0
                && f64::from(g) < 1.0 / 255.0
                && (f64::from(b) - t).abs() < 1.0 / 255.0
                && (f64::from(a) - 1.0).abs() < 1.0 / 255.0,
            "pixel ({px_x},{px_y}): {r} {g} {b} {a}, expected t {t}"
        );
    }
    // Pixel x == 10 has its centre on the rect's left edge (10.5): it is
    // half-covered by antialiasing.
    let [.., edge_a] = px(10, 160);
    assert!(
        edge_a > 0.05 && edge_a < 0.95,
        "edge pixel must be partially covered: {edge_a}"
    );
    Ok(())
}
