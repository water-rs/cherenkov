// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for the slice-1 review fixes: atlas growth as a result, the
//! clear-only dirty flag, dropped-surface cleanup, the oracle-exact radial
//! gradient parameter and the zero-size surface error.

use cherenkov::kurbo::{Point, Rect};
use cherenkov::{Draw, GlyphRun, WorkingColor};
use cherenkov_gpu::{
    Budget, Bytes, Engine, EngineError, Gpu, GpuConfig, Offscreen, OffscreenFormat, RenderError,
    SurfaceError,
};

/// An engine under `config`, or `None` when no adapter exists.
fn engine(config: GpuConfig) -> Option<Engine<Gpu>> {
    match Engine::<Gpu>::new(config) {
        Ok(engine) => Some(engine),
        Err(EngineError::NoAdapter) => None,
        Err(e) => panic!("engine init failed: {e}"),
    }
}

/// The committed corpus subset of Noto Sans (never a host system font).
const FONT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../scenes/fonts/NotoSans.ttf");

/// Glyph ids the subset has outlines for.
const FONT_GLYPHS: u32 = 200;

fn font() -> cherenkov_gpu::FontSource {
    cherenkov_gpu::FontSource::bytes(std::fs::read(FONT_PATH).expect("scenes/fonts/NotoSans.ttf"))
}

/// `count` distinct glyph-cache entries at `size` px, tiled on a grid: the
/// glyph ids cycle through the subset, and each cycle steps the size so no
/// two entries share an atlas cell.
#[expect(clippy::cast_precision_loss)]
fn text_runs(font: cherenkov::FontId, count: u32, size: f32) -> Vec<GlyphRun> {
    (0..count.div_ceil(FONT_GLYPHS))
        .map(|cycle| {
            let size = (cycle as f32).mul_add(2.0, size);
            let glyphs = (cycle * FONT_GLYPHS..(cycle * FONT_GLYPHS + FONT_GLYPHS).min(count))
                .map(|i| cherenkov::Glyph {
                    id: 1 + i % FONT_GLYPHS,
                    x: (i % 32) as f32 * (size * 0.8),
                    y: (1 + i / 32) as f32 * size,
                    transform: None,
                })
                .collect();
            GlyphRun {
                font,
                size,
                coords: Vec::new(),
                glyphs,
                style: cherenkov::GlyphStyle::Fill,
            }
        })
        .collect()
}

fn render_text(
    config: GpuConfig,
    count: u32,
    size: f32,
) -> Option<Result<Vec<[f32; 4]>, RenderError>> {
    let engine = engine(config)?;
    let font = engine.font(font()).expect("font");
    let surface = engine
        .surface(Offscreen::new((512, 512), OffscreenFormat::LinearF16))
        .expect("surface");
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            for run in text_runs(font.id(), count, size) {
                c.glyphs(&run, WorkingColor::WHITE);
            }
        }));
    });
    Some(
        engine
            .render(cherenkov_gpu::FrameTime::now())
            .and_then(|_| surface.readback())
            .map(|r| r.pixels),
    )
}

/// A 128-glyph run renders identically whether the atlas can grow or is
/// capped at its start size; a 2000-glyph live set overflows the cap.
#[test]
fn a_full_atlas_grows_then_reports_exhaustion() -> Result<(), Box<dyn std::error::Error>> {
    let tiny = GpuConfig {
        budget: Budget {
            gpu: Bytes::mib(1),
            ..Budget::default()
        },
        ..GpuConfig::default()
    };
    let (Some(big), Some(small)) = (
        render_text(GpuConfig::default(), 128, 48.0),
        render_text(tiny.clone(), 128, 48.0),
    ) else {
        return Ok(());
    };
    let big = big?;
    let small = small?;
    assert!(big.iter().any(|p| p[3] > 0.0), "default render is empty");
    assert!(small.iter().any(|p| p[3] > 0.0), "capped render is empty");
    let eps = 1.0 / 255.0 + 1e-4;
    for (i, (a, b)) in big.iter().zip(&small).enumerate() {
        for c in 0..4 {
            assert!(
                (a[c] - b[c]).abs() <= eps,
                "pixel {i} channel {c}: {} vs {}",
                a[c],
                b[c]
            );
        }
    }
    // A live set that does not fit the capped atlas exhausts it.
    let Some(exhausted) = render_text(tiny, 2000, 48.0) else {
        return Ok(());
    };
    assert!(
        matches!(exhausted, Err(RenderError::AtlasExhausted)),
        "expected AtlasExhausted, got {exhausted:?}"
    );
    Ok(())
}

/// A clear-colour-only commit marks the surface dirty and re-renders.
#[test]
fn a_clear_only_commit_renders() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine(GpuConfig::default()) else {
        return Ok(());
    };
    let surface = engine.surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))?;
    surface.clear_color(WorkingColor::new([1.0, 0.0, 0.0, 1.0]));
    engine.render(cherenkov_gpu::FrameTime::now())?;
    assert!(surface.readback()?.pixels[0][0] > 0.9, "red clear");
    surface.clear_color(WorkingColor::new([0.0, 0.0, 1.0, 1.0]));
    engine.render(cherenkov_gpu::FrameTime::now())?;
    let [r, g, b, a] = surface.readback()?.pixels[0];
    assert!(
        b > 0.9 && r < 0.1 && g < 0.1 && a > 0.9,
        "blue clear: {r} {g} {b} {a}"
    );
    Ok(())
}

/// Dropping a surface releases its engine entry and GPU textures.
#[test]
fn dropped_surfaces_do_not_leak() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine(GpuConfig::default()) else {
        return Ok(());
    };
    let before = engine.memory().gpu;
    for _ in 0..200 {
        drop(engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?);
    }
    engine.render(cherenkov_gpu::FrameTime::now())?;
    assert_eq!(engine.live_surfaces(), 0, "surfaces still live");
    assert_eq!(engine.memory().gpu, before, "gpu memory grew");
    Ok(())
}

/// The oracle's `radial_t`, ported for the cone test.
fn oracle_t(p: (f64, f64), c0: (f64, f64), r0: f64, c1: (f64, f64), r1: f64) -> f64 {
    let (px, py) = (p.0 - c0.0, p.1 - c0.1);
    let (dcx, dcy) = (c1.0 - c0.0, c1.1 - c0.1);
    let dr = r1 - r0;
    let a = dr.mul_add(-dr, dcy.mul_add(dcy, dcx * dcx));
    let b = -2.0 * r0.mul_add(dr, dcy.mul_add(py, dcx * px));
    let c = r0.mul_add(-r0, py.mul_add(py, px * px));
    if a.abs() < 1e-12 {
        if b.abs() < 1e-12 {
            return if r0.abs() < 1e-12 {
                0.0
            } else {
                (px.hypot(py) - r0) / r0.abs()
            };
        }
        return -c / b;
    }
    let disc = (4.0 * a).mul_add(-c, b * b);
    if disc < 0.0 {
        return f64::NAN;
    }
    let sq = disc.sqrt();
    ((-b + sq) / (2.0 * a)).max((-b - sq) / (2.0 * a))
}

fn radial_fill(
    center0: (f64, f64),
    r0: f64,
    center1: (f64, f64),
    r1: f64,
) -> cherenkov::RadialGradient {
    cherenkov::RadialGradient {
        start_center: Point::new(center0.0, center0.1),
        start_radius: r0,
        end_center: Point::new(center1.0, center1.1),
        end_radius: r1,
        stops: vec![
            cherenkov::ColorStop {
                offset: 0.0,
                color: WorkingColor::new([0.0, 0.0, 0.0, 1.0]),
            },
            cherenkov::ColorStop {
                offset: 1.0,
                color: WorkingColor::new([1.0, 1.0, 1.0, 1.0]),
            },
        ],
        extend: cherenkov::Extend::Pad,
        interpolation: cherenkov::Interpolation::Working,
    }
}

fn render_radial(gradient: cherenkov::RadialGradient) -> Option<Vec<[f32; 4]>> {
    let engine = engine(GpuConfig::default())?;
    let surface = engine
        .surface(Offscreen::new((128, 128), OffscreenFormat::LinearF16))
        .expect("surface");
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(
                Rect::new(0.0, 0.0, 128.0, 128.0),
                cherenkov::Paint::Radial(gradient),
            );
        }));
    });
    engine
        .render(cherenkov_gpu::FrameTime::now())
        .expect("render");
    Some(surface.readback().expect("readback").pixels)
}

/// Coincident circles (c0 == c1, r0 == r1) interpolate on distance / r0.
#[test]
fn coincident_circles_interpolate_by_distance() {
    let Some(pixels) = render_radial(radial_fill((64., 64.), 20., (64., 64.), 20.)) else {
        return;
    };
    let at = |x: usize, y: usize| pixels[y * 128 + x];
    // Pixel centres land at half-integer coordinates: distance 30.5 gives
    // t = (30.5 - 20) / 20 = 0.525.
    let px = at(94, 64);
    assert!((px[0] - 0.525).abs() < 5e-3, "t=0.525 pixel: {px:?}");
    // Distance 10 is inside r0: t < 0 pads to the first stop.
    let inside = at(74, 64);
    assert!(inside[0] < 1e-3, "inside pixel: {inside:?}");
}

/// A cone where the quadratic's negative-radius branch matters: the shader
/// follows the oracle, including NaN outside the cone.
#[test]
#[expect(clippy::cast_precision_loss)]
#[expect(clippy::cast_possible_truncation)]
fn a_cone_gradient_matches_the_oracle() {
    let c0 = (32., 64.);
    let c1 = (72., 64.);
    let Some(pixels) = render_radial(radial_fill(c0, 0., c1, 30.)) else {
        return;
    };
    let at = |x: usize, y: usize| pixels[y * 128 + x];
    // Pixels where the oracle's discriminant is negative get NaN →
    // transparent.
    for (x, y) in [(32, 20), (10, 20), (32, 44)] {
        let t = oracle_t((x as f64 + 0.5, y as f64 + 0.5), c0, 0., c1, 30.);
        assert!(t.is_nan(), "expected NaN at ({x}, {y}), got {t}");
        assert!(
            at(x, y)[3] < 1e-6,
            "outside pixel ({x}, {y}): {:?}",
            at(x, y)
        );
    }
    // Inside the cone the shader's t equals the oracle's (clamped to the
    // stop range by the pad extension).
    for (x, y) in [(33, 64), (35, 64), (80, 44)] {
        let t = oracle_t((x as f64 + 0.5, y as f64 + 0.5), c0, 0., c1, 30.);
        let want = t.clamp(0.0, 1.0) as f32;
        assert!(
            (at(x, y)[0] - want).abs() < 5e-2,
            "cone pixel ({x}, {y}) t={t}: {:?}",
            at(x, y)
        );
    }
}

/// Dropping the last `Font` clone frees the renderer's font state: the
/// atlas's glyph cells are purged and a later frame referencing the id
/// fails fast instead of silently keeping the font alive.
#[test]
fn dropping_a_font_frees_its_renderer_state() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine(GpuConfig::default()) else {
        return Ok(());
    };
    let font = engine.font(font())?;
    let font_id = font.id();
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    let runs = text_runs(font_id, 8, 24.0);
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.glyphs(&runs[0], WorkingColor::WHITE);
        }));
    });
    engine.render(cherenkov_gpu::FrameTime::now())?;
    assert!(engine.memory().cpu > Bytes(0), "glyph cells cached");
    drop(font);
    // Dirty the surface so the next frame re-lowers and consults the font.
    surface.clear_color(WorkingColor::new([0.0, 0.0, 0.0, 1.0]));
    assert!(
        matches!(
            engine.render(cherenkov_gpu::FrameTime::now()),
            Err(RenderError::Font(_))
        ),
        "render after the last Font clone dropped"
    );
    assert_eq!(engine.memory().cpu, Bytes(0), "font cells released");
    Ok(())
}

/// Timestamp queries resolve on a later render — the submitting frame
/// never waits for GPU idle: the first render reports no GPU timing yet,
/// and a later render carries that frame's bracket and per-pass timings.
#[test]
fn timestamps_resolve_a_frame_late() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine(GpuConfig {
        timestamps: true,
        ..GpuConfig::default()
    }) else {
        return Ok(());
    };
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(
                Rect::new(0.0, 0.0, 64.0, 64.0),
                WorkingColor::new([1.0, 0.0, 0.0, 1.0]),
            );
        }));
    });
    engine.render(cherenkov_gpu::FrameTime::now())?;
    assert!(
        engine.stats().gpu_seconds.is_none(),
        "a submitting frame returns before its queries resolve"
    );
    // An idle render still drains the pending resolve.
    engine.render(cherenkov_gpu::FrameTime::now())?;
    assert!(
        engine.stats().gpu_seconds.is_some(),
        "the previous frame's timing arrives a render late"
    );
    assert!(
        engine
            .stats()
            .passes_timed
            .iter()
            .any(|p| p.name == "surface"),
        "per-pass timing survives the deferred resolve"
    );
    Ok(())
}

/// With timestamps disabled a render reports no timing — and never
/// allocates the per-pass metadata only the timestamp path consumes.
#[test]
fn timestamps_off_reports_no_passes() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine(GpuConfig::default()) else {
        return Ok(());
    };
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(
                Rect::new(0.0, 0.0, 64.0, 64.0),
                WorkingColor::new([1.0, 0.0, 0.0, 1.0]),
            );
        }));
    });
    engine.render(cherenkov_gpu::FrameTime::now())?;
    let stats = engine.stats();
    assert!(stats.gpu_seconds.is_none());
    assert!(stats.passes_timed.is_empty());
    Ok(())
}

/// Re-rendering an unchanged scene creates no group-1 bind groups: the
/// second frame binds the views cached under (scratch generation, image
/// generation) rather than rebuilding them per encode.
#[test]
fn bind_groups_are_reused_across_frames() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine(GpuConfig::default()) else {
        return Ok(());
    };
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    let record = |c: &mut cherenkov::Recorder| {
        c.fill(
            Rect::new(0.0, 0.0, 64.0, 64.0),
            WorkingColor::new([1.0, 0.0, 0.0, 1.0]),
        );
        // An isolated group with overlapping contents forces a scratch
        // pass → a source-texture bind group.
        c.group(cherenkov::Group::new().opacity(0.5), |c| {
            c.fill(
                Rect::new(8.0, 8.0, 32.0, 32.0),
                WorkingColor::new([0.0, 0.0, 1.0, 1.0]),
            );
            c.fill(
                Rect::new(16.0, 16.0, 48.0, 48.0),
                WorkingColor::new([0.0, 1.0, 0.0, 1.0]),
            );
        });
    };
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| record(c)));
    });
    engine.render(cherenkov_gpu::FrameTime::now())?;
    assert!(
        engine.stats().bind_groups_created > 0,
        "the first frame builds the bind groups"
    );
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| record(c)));
    });
    engine.render(cherenkov_gpu::FrameTime::now())?;
    assert_eq!(
        engine.stats().bind_groups_created,
        0,
        "an identical frame reuses the cached bind groups"
    );
    Ok(())
}

/// A zero-size surface is rejected synchronously.
#[test]
fn a_zero_size_surface_is_an_error() {
    let Some(engine) = engine(GpuConfig::default()) else {
        return;
    };
    for size in [(0, 64), (64, 0), (0, 0)] {
        let result = engine.surface(Offscreen::new(size, OffscreenFormat::LinearF16));
        assert!(
            matches!(result, Err(SurfaceError::ZeroSize)),
            "{size:?}: {result:?}"
        );
    }
}
