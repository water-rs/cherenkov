//! Blend-mode composites against the W3C formulas on lavapipe.

#![expect(clippy::float_cmp, reason = "clear pixels are exact")]

use cherenkov::kurbo::{Point, Rect};
use cherenkov::{BlendMode, ColorStop, Draw, Extend, Group, Interpolation, Paint, WorkingColor};
use cherenkov::{Engine, EngineError, Offscreen, OffscreenFormat};
use cherenkov_gpu::{Gpu, GpuConfig};

const RED: WorkingColor = WorkingColor::new([1.0, 0.0, 0.0, 1.0]);
const BLUE: WorkingColor = WorkingColor::new([0.0, 0.0, 1.0, 1.0]);
const GREEN: WorkingColor = WorkingColor::new([0.0, 1.0, 0.0, 1.0]);
/// HDR primaries so a `PlusLighter` sum exceeds SDR white.
const HDR_RED: WorkingColor = WorkingColor::new([1.5, 0.0, 0.0, 1.0]);
const HDR_BLUE: WorkingColor = WorkingColor::new([0.0, 0.0, 1.25, 1.0]);

/// An engine, or `None` when no adapter exists.
fn engine() -> Option<Engine<Gpu>> {
    match Engine::<Gpu>::new(GpuConfig::default()) {
        Ok(engine) => Some(engine),
        Err(EngineError::Backend(_)) => None,
        Err(e) => panic!("engine init failed: {e}"),
    }
}

/// Renders a red backdrop rect and a blue blend-layer rect; returns the
/// overlap pixel at (24,32) — inside the backdrop only for x < 32 — plus
/// the pass count.
fn render_blend(
    engine: &Engine<Gpu>,
    mode: BlendMode,
) -> Result<([f32; 4], u32), Box<dyn std::error::Error>> {
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(Rect::new(0., 0., 32., 64.), RED);
        }));
    });
    let layer = surface.layer();
    surface.update(|tx| {
        tx[&layer].blend(mode);
        tx[surface.root()].push(&layer);
    });
    surface.update(|tx| {
        tx[&layer].content(surface.record(|c| {
            c.fill(Rect::new(16., 0., 64., 64.), BLUE);
        }));
    });
    engine.render(cherenkov::FrameTime::now())?;
    let rb = surface.readback()?;
    Ok((
        rb.pixels[(32 * rb.width + 24) as usize],
        engine.stats().passes,
    ))
}

#[test]
fn multiply_blends_the_overlap() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let (px, passes) = render_blend(&engine, BlendMode::Multiply)?;
    // Opaque red × opaque blue = opaque black.
    for c in &px[..3] {
        assert!(c.abs() < 1e-2, "multiply overlap: {px:?}");
    }
    assert!((px[3] - 1.0).abs() < 1e-2, "multiply overlap: {px:?}");
    assert!(passes >= 2, "a blend layer must isolate: passes {passes}");
    Ok(())
}

#[test]
fn screen_blends_the_overlap() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let (px, _) = render_blend(&engine, BlendMode::Screen)?;
    // screen(red, blue) = 1-(1-r)(1-b) per channel → (1, 0, 1) magenta.
    let [r, g, b, a] = px;
    assert!(
        (r - 1.0).abs() < 1e-2 && g.abs() < 1e-2 && (b - 1.0).abs() < 1e-2 && a > 0.99,
        "screen overlap: {px:?}"
    );
    Ok(())
}

/// `DestOut` keeps the backdrop where the source is absent and knocks the
/// overlap out to clear.
#[test]
fn dest_out_knocks_out_the_overlap() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let (overlap, _) = render_blend(&engine, BlendMode::DestOut)?;
    for c in overlap {
        assert!(
            c.abs() < 1e-2,
            "dest-out overlap should be clear: {overlap:?}"
        );
    }
    Ok(())
}

/// Hue sets the backdrop's luminance on the source's hue and saturation:
/// `SetLum(blue, Lum(red))` = (0.19, 0.19, 1) — the non-separable formula,
/// not a channel-wise blend.
#[test]
fn hue_is_non_separable() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let (px, _) = render_blend(&engine, BlendMode::Hue)?;
    let [r, g, b, _a] = px;
    assert!(
        (r - 0.19).abs() < 5e-2 && (g - 0.19).abs() < 5e-2 && (b - 1.0).abs() < 5e-2,
        "hue overlap should be SetLum(blue, 0.3): {px:?}"
    );
    Ok(())
}

/// `Extend::None` gradients are transparent outside the range.
#[test]
fn extend_none_is_transparent_outside_the_range() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(
                Rect::new(0., 0., 64., 64.),
                Paint::Linear(cherenkov::LinearGradient {
                    start: Point::new(16., 0.),
                    end: Point::new(48., 0.),
                    stops: vec![
                        ColorStop {
                            offset: 0.0,
                            color: RED,
                        },
                        ColorStop {
                            offset: 1.0,
                            color: BLUE,
                        },
                    ],
                    extend: Extend::None,
                    interpolation: Interpolation::Working,
                }),
            );
        }));
    });
    engine.render(cherenkov::FrameTime::now())?;
    let rb = surface.readback()?;
    let px = |x: u32, y: u32| rb.pixels[(y * rb.width + x) as usize];
    assert_eq!(px(4, 32), [0.0; 4], "left of range must be clear");
    assert_eq!(px(60, 32), [0.0; 4], "right of range must be clear");
    assert!(px(32, 32)[3] > 0.99, "mid-range must be opaque");
    Ok(())
}

/// A full-turn sweep gradient: angle 0 (+x) is the first stop.
#[test]
fn a_sweep_gradient_resolves_angles() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(
                Rect::new(0., 0., 64., 64.),
                Paint::Sweep(cherenkov::SweepGradient {
                    center: Point::new(32., 32.),
                    start_angle: 0.0,
                    end_angle: std::f64::consts::TAU,
                    stops: vec![
                        ColorStop {
                            offset: 0.0,
                            color: RED,
                        },
                        ColorStop {
                            offset: 1.0,
                            color: BLUE,
                        },
                    ],
                    extend: Extend::Pad,
                    interpolation: Interpolation::Working,
                }),
            );
        }));
    });
    engine.render(cherenkov::FrameTime::now())?;
    let rb = surface.readback()?;
    let px = |x: u32, y: u32| rb.pixels[(y * rb.width + x) as usize];
    let right = px(56, 32);
    let top = px(32, 8);
    assert!(
        right[0] > 0.9 && right[2] < 0.1,
        "angle 0 is red: {right:?}"
    );
    // atan2 of (0,−1) ≈ 3π/2 → t ≈ 0.75, mostly blue.
    assert!(top[2] > 0.6 && top[0] < 0.4, "top is blue-ish: {top:?}");
    Ok(())
}

/// A `Normal` group flattens unless a descendant group blends; then it
/// must isolate, or the descendant's composite would reach the scene.
#[test]
fn blended_descendant_isolates_its_normal_group() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let surface = engine.surface(Offscreen::new((8, 8), OffscreenFormat::LinearF16))?;
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(Rect::new(0., 0., 8., 8.), RED);
            c.group(Group::new(), |c| {
                c.fill(Rect::new(0., 0., 4., 8.), BLUE);
                c.group(Group::new().blend(BlendMode::Clear), |c| {
                    c.fill(Rect::new(2., 0., 6., 8.), WorkingColor::WHITE);
                });
            });
        }));
    });
    engine.render(cherenkov::FrameTime::now())?;
    let rb = surface.readback()?;
    // `Clear` zeroes the inner group's whole raster — but only inside the
    // outer group's offscreen, which then composites `Normal` over the red
    // background. Without outer isolation the `Clear` reached the scene
    // framebuffer and every pixel came out transparent.
    for (i, px) in rb.pixels.iter().enumerate() {
        assert_eq!(*px, [1.0, 0.0, 0.0, 1.0], "pixel {i}");
    }
    Ok(())
}

#[test]
fn tree_layer_isolates_blended_child_layer() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let surface = engine.surface(Offscreen::new((8, 8), OffscreenFormat::LinearF16))?;
    let background = surface.layer();
    let pass = surface.layer();
    let cutout = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].push(&background).push(&pass);
        tx[&background].content(surface.record(|c| {
            c.fill(Rect::new(0.0, 0.0, 8.0, 8.0), RED);
        }));
        tx[&pass].content(surface.record(|c| {
            c.fill(Rect::new(0.0, 0.0, 4.0, 8.0), BLUE);
        }));
        tx[&pass].push(&cutout);
        tx[&cutout]
            .blend(BlendMode::DestOut)
            .content(surface.record(|c| {
                c.fill(Rect::new(2.0, 0.0, 6.0, 8.0), WorkingColor::WHITE);
            }));
    });
    engine.render(cherenkov::FrameTime::now())?;
    let pixels = surface.readback()?.pixels;
    let pixel = |x: usize, y: usize| pixels[y * 8 + x];
    assert_eq!(
        pixel(1, 3).map(f32::to_bits),
        [0.0_f32, 0.0, 1.0, 1.0].map(f32::to_bits)
    );
    assert_eq!(
        pixel(3, 3).map(f32::to_bits),
        [1.0_f32, 0.0, 0.0, 1.0].map(f32::to_bits)
    );
    Ok(())
}

#[test]
fn tree_layer_isolates_blended_content_group() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let surface = engine.surface(Offscreen::new((8, 8), OffscreenFormat::LinearF16))?;
    let background = surface.layer();
    let pass = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].push(&background).push(&pass);
        tx[&background].content(surface.record(|c| {
            c.fill(Rect::new(0.0, 0.0, 8.0, 8.0), RED);
        }));
        tx[&pass].content(surface.record(|c| {
            c.fill(Rect::new(0.0, 0.0, 4.0, 8.0), BLUE);
            c.group(Group::new().blend(BlendMode::DestOut), |c| {
                c.fill(Rect::new(2.0, 0.0, 6.0, 8.0), WorkingColor::WHITE);
            });
        }));
    });
    engine.render(cherenkov::FrameTime::now())?;
    let pixels = surface.readback()?.pixels;
    let pixel = |x: usize, y: usize| pixels[y * 8 + x];
    assert_eq!(
        pixel(1, 3).map(f32::to_bits),
        [0.0_f32, 0.0, 1.0, 1.0].map(f32::to_bits)
    );
    assert_eq!(
        pixel(3, 3).map(f32::to_bits),
        [1.0_f32, 0.0, 0.0, 1.0].map(f32::to_bits)
    );
    Ok(())
}

#[test]
fn nested_tree_layers_isolate_at_the_blending_parent() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let surface = engine.surface(Offscreen::new((8, 8), OffscreenFormat::LinearF16))?;
    let background = surface.layer();
    let outer = surface.layer();
    let inner = surface.layer();
    let cutout = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].push(&background).push(&outer);
        tx[&outer].push(&inner);
        tx[&inner].push(&cutout);
        tx[&background].content(surface.record(|c| {
            c.fill(Rect::new(0.0, 0.0, 8.0, 8.0), RED);
        }));
        tx[&outer].content(surface.record(|c| {
            c.fill(Rect::new(0.0, 0.0, 8.0, 8.0), BLUE);
        }));
        tx[&inner].content(surface.record(|c| {
            c.fill(Rect::new(0.0, 0.0, 4.0, 8.0), GREEN);
        }));
        tx[&cutout]
            .blend(BlendMode::DestOut)
            .content(surface.record(|c| {
                c.fill(Rect::new(2.0, 0.0, 6.0, 8.0), WorkingColor::WHITE);
            }));
    });
    engine.render(cherenkov::FrameTime::now())?;
    let pixels = surface.readback()?.pixels;
    let pixel = |x: usize, y: usize| pixels[y * 8 + x];
    assert_eq!(
        pixel(1, 3).map(f32::to_bits),
        [0.0_f32, 1.0, 0.0, 1.0].map(f32::to_bits)
    );
    assert_eq!(
        pixel(3, 3).map(f32::to_bits),
        [0.0_f32, 0.0, 1.0, 1.0].map(f32::to_bits)
    );
    Ok(())
}

/// Two overlapping opaque `PlusLighter` layers: coverage saturates at 1,
/// summed light exceeds 1 and survives in the extended working space (#126).
#[test]
fn plus_lighter_saturates_alpha_not_colour() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(Rect::new(0., 0., 32., 64.), HDR_RED);
        }));
    });
    let layer = surface.layer();
    surface.update(|tx| {
        tx[&layer].blend(BlendMode::PlusLighter);
        tx[surface.root()].push(&layer);
    });
    surface.update(|tx| {
        tx[&layer].content(surface.record(|c| {
            c.fill(Rect::new(16., 0., 64., 64.), HDR_BLUE);
        }));
    });
    engine.render(cherenkov::FrameTime::now())?;
    let px = surface.readback()?.pixels[(32 * 64 + 24) as usize];
    assert!(
        px[3] <= 1.0 + 1e-3,
        "plus-lighter alpha must saturate: {px:?}"
    );
    assert!(
        (px[0] - 1.5).abs() < 1e-2 && (px[2] - 1.25).abs() < 1e-2,
        "plus-lighter colour adds unclamped: {px:?}"
    );
    Ok(())
}
