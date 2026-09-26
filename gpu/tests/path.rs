// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! General paths: an even-odd star leaves its centre uncovered while the
//! same star filled non-zero covers it, and a dashed circle's gaps stay
//! clear.

use cherenkov::kurbo::{BezPath, Circle, Point};
use cherenkov::{Draw, EvenOdd, WorkingColor};
use cherenkov_gpu::{Engine, EngineError, Gpu, GpuConfig, Offscreen, OffscreenFormat, Surface};

const CLEAR: WorkingColor = WorkingColor::new([0.0, 0.0, 0.0, 1.0]);
const RED: WorkingColor = WorkingColor::new([1.0, 0.0, 0.0, 1.0]);

/// A self-intersecting 5-point star over a 64×64 surface, centred.
fn star() -> BezPath {
    let mut path = BezPath::new();
    let c = Point::new(32.0, 32.0);
    for i in 0..5 {
        let angle = f64::from(i).mul_add(144.0, -90.0).to_radians();
        let p = c + cherenkov::kurbo::Vec2::new(angle.cos() * 24.0, angle.sin() * 24.0);
        if i == 0 {
            path.move_to(p);
        } else {
            path.line_to(p);
        }
    }
    path.close_path();
    path
}

fn render(
    draw: impl FnOnce(&mut cherenkov::Recorder),
) -> Result<Option<cherenkov_gpu::Readback>, Box<dyn std::error::Error>> {
    let engine = match Engine::<Gpu>::new(GpuConfig::default()) {
        Ok(engine) => engine,
        Err(EngineError::NoAdapter) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let surface: Surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    surface.clear_color(CLEAR);
    let layer = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
    });
    surface.update(|tx| {
        tx[&layer].content(surface.record(draw));
    });
    engine.render(cherenkov_gpu::FrameTime::now())?;
    Ok(Some(surface.readback()?))
}

fn px(readback: &cherenkov_gpu::Readback, x: u32, y: u32) -> [f32; 4] {
    readback.pixels[(y * readback.width + x) as usize]
}

#[test]
fn an_even_odd_star_leaves_its_centre_uncovered() -> Result<(), Box<dyn std::error::Error>> {
    let Some(readback) = render(|c| c.fill(EvenOdd(star()), RED))? else {
        return Ok(());
    };
    // The pentagonal hole at the centre reads back as the clear colour.
    let [r, g, b, a] = px(&readback, 32, 30);
    assert!(
        r < 0.05 && g < 0.05 && b < 0.05 && a > 0.99,
        "centre: {r} {g} {b} {a}"
    );
    // Deep inside the top arm the star is singly wound, hence covered.
    let [r, ..] = px(&readback, 32, 17);
    assert!(r > 0.9, "arm: {r}");
    Ok(())
}

#[test]
fn a_nonzero_star_covers_its_centre() -> Result<(), Box<dyn std::error::Error>> {
    let Some(readback) = render(|c| c.fill(star(), RED))? else {
        return Ok(());
    };
    let [r, ..] = px(&readback, 32, 30);
    assert!(r > 0.9, "centre: {r}");
    Ok(())
}

#[test]
fn a_dashed_circle_stroke_has_gaps() -> Result<(), Box<dyn std::error::Error>> {
    let mut stroke = kurbo::Stroke::new(4.0);
    stroke.dash_pattern.extend([8.0, 8.0]);
    let Some(readback) = render(move |c| {
        c.stroke(Circle::new((32.0, 32.0), 20.0), stroke, RED);
    })?
    else {
        return Ok(());
    };
    // Scan the top of the ring (y = 32 - 20): some columns are covered,
    // some fall in dash gaps.
    let mut covered = 0u32;
    let mut gap = 0u32;
    for x in 12..52 {
        let [r, ..] = px(&readback, x, 12);
        if r > 0.5 {
            covered += 1;
        } else {
            gap += 1;
        }
    }
    assert!(covered > 0 && gap > 0, "covered {covered} gap {gap}");
    // Well inside the ring nothing is stroked.
    let [r, ..] = px(&readback, 32, 32);
    assert!(r < 0.05, "interior: {r}");
    Ok(())
}

/// A large square hanging off the surface's top-left corner.
fn overhang() -> BezPath {
    let mut path = BezPath::new();
    path.move_to((-100.0, -100.0));
    path.line_to((40.0, -100.0));
    path.line_to((40.0, 40.0));
    path.line_to((-100.0, 40.0));
    path.close_path();
    path
}

/// Coverage clipped by the surface is cached per integer offset: the same
/// path drawn first half off-screen, then shifted, must not replay the
/// clipped coverage.
#[test]
fn a_clipped_path_is_cached_per_offset() -> Result<(), Box<dyn std::error::Error>> {
    let engine = match Engine::<Gpu>::new(GpuConfig::default()) {
        Ok(engine) => engine,
        Err(EngineError::NoAdapter) => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    surface.clear_color(CLEAR);
    let layer = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
    });
    // Frame 1: the path hangs off the top-left, so its coverage is clipped.
    surface.update(|tx| {
        tx[&layer].content(surface.record(|c| c.fill(overhang(), RED)));
    });
    engine.render(cherenkov_gpu::FrameTime::now())?;
    // Frame 2: the same path at a different integer offset.
    surface.update(|tx| {
        tx[&layer].content(surface.record(|c| {
            c.transform(cherenkov::kurbo::Affine::translate((64.0, 64.0)), |c| {
                c.fill(overhang(), RED);
            });
        }));
    });
    engine.render(cherenkov_gpu::FrameTime::now())?;
    let shifted = surface.readback()?;
    // A fresh engine renders the same shifted draw for comparison.
    let Some(fresh) = render(|c| {
        c.transform(cherenkov::kurbo::Affine::translate((64.0, 64.0)), |c| {
            c.fill(overhang(), RED);
        });
    })?
    else {
        return Ok(());
    };
    for (i, (a, b)) in shifted.pixels.iter().zip(&fresh.pixels).enumerate() {
        for c in 0..4 {
            assert!(
                (a[c] - b[c]).abs() < 1e-3,
                "pixel {i} channel {c}: {} vs {}",
                a[c],
                b[c]
            );
        }
    }
    Ok(())
}

/// A path clip multiplies content by the rasterized star's coverage.
#[test]
fn a_path_clip_masks_content() -> Result<(), Box<dyn std::error::Error>> {
    let Some(readback) = render(|c| {
        c.clip(star(), |c| {
            c.fill(cherenkov::kurbo::Rect::new(0.0, 0.0, 64.0, 64.0), RED);
        });
    })?
    else {
        return Ok(());
    };
    // Deep inside the top arm the star covers.
    let [r, ..] = px(&readback, 32, 17);
    assert!(r > 0.9, "arm: {r}");
    // The concave notch between the top and right arm tips is outside.
    let [r, ..] = px(&readback, 46, 14);
    assert!(r < 0.05, "notch: {r}");
    // Somewhere along an edge a pixel is partially covered.
    let partial = readback.pixels.iter().any(|p| p[0] > 0.1 && p[0] < 0.9);
    assert!(partial, "no partially covered edge pixel");
    Ok(())
}

/// A rect drawn as a path hanging off the surface's left edge: the
/// out-of-window edge's winding deposit must still reach column 0, or the
/// interior stays clear.
#[test]
fn a_fill_overhanging_the_left_edge_paints_its_interior() -> Result<(), Box<dyn std::error::Error>>
{
    let mut left_edge_rect = BezPath::new();
    left_edge_rect.move_to((-40.0, 10.0));
    left_edge_rect.line_to((30.0, 10.0));
    left_edge_rect.line_to((30.0, 50.0));
    left_edge_rect.line_to((-40.0, 50.0));
    left_edge_rect.close_path();
    let Some(readback) = render(|c| c.fill(left_edge_rect, RED))? else {
        return Ok(());
    };
    for x in [5, 15, 28] {
        let [r, g, b, a] = px(&readback, x, 20);
        assert!(
            r > 0.9 && g < 0.1 && b < 0.1 && a > 0.9,
            "interior ({x},20): {r} {g} {b} {a}"
        );
    }
    // Right of the fill stays clear.
    let [r, ..] = px(&readback, 40, 20);
    assert!(r < 0.05, "outside: {r}");
    Ok(())
}

/// A clip path whose left edge sits outside the window still masks its
/// interior: a path clip rasterizes through the same deposit pipeline.
#[test]
fn a_clip_overhanging_the_left_edge_masks_its_interior() -> Result<(), Box<dyn std::error::Error>> {
    let mut clip = BezPath::new();
    clip.move_to((-40.0, -10.0));
    clip.line_to((32.0, -10.0));
    clip.line_to((32.0, 74.0));
    clip.line_to((-40.0, 74.0));
    clip.close_path();
    let Some(readback) = render(|c| {
        c.clip(clip, |c| {
            c.fill(cherenkov::kurbo::Rect::new(0.0, 0.0, 64.0, 64.0), RED);
        });
    })?
    else {
        return Ok(());
    };
    // Inside the clip's right half.
    let [r, ..] = px(&readback, 20, 32);
    assert!(r > 0.9, "clipped interior: {r}");
    // Right of the clip the fill is masked out.
    let [r, ..] = px(&readback, 40, 32);
    assert!(r < 0.05, "outside clip: {r}");
    Ok(())
}

/// A rect clip merged with a path clip: the mask still applies inside the
/// intersected rectangle.
#[test]
fn a_rect_clip_merges_with_a_path_clip() -> Result<(), Box<dyn std::error::Error>> {
    let Some(readback) = render(|c| {
        c.clip(cherenkov::kurbo::Rect::new(8.0, 8.0, 56.0, 56.0), |c| {
            c.clip(EvenOdd(star()), |c| {
                c.fill(cherenkov::kurbo::Rect::new(0.0, 0.0, 64.0, 64.0), RED);
            });
        });
    })?
    else {
        return Ok(());
    };
    // Inside the rect and the star arm.
    let [r, ..] = px(&readback, 32, 17);
    assert!(r > 0.9, "arm: {r}");
    // Inside the rect but in the even-odd hole.
    let [r, ..] = px(&readback, 32, 30);
    assert!(r < 0.05, "hole: {r}");
    // Outside the rect even where the star would cover.
    let [r, ..] = px(&readback, 4, 32);
    assert!(r < 0.05, "outside rect: {r}");
    Ok(())
}
