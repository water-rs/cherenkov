// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `COLRv1` glyphs render through the colour-glyph lowering: palette fills,
//! clips and blend groups, not the foreground paint alone.

use cherenkov::{Draw, Glyph, GlyphRun, WorkingColor};
use cherenkov_gpu::{Engine, EngineError, FontSource, Gpu, GpuConfig, Offscreen, OffscreenFormat};

const FONT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../scenes/corpus/text-colr/resources/",
    "04dd974b8bf7e440fd8927b6af23764194287cee28d15acda268c5e0a51091f5"
);

/// The scene's glyphs: ids 1, 25 and 61 at 64 px.
fn run(font: cherenkov::FontId) -> GlyphRun {
    GlyphRun {
        font,
        size: 64.0,
        coords: Vec::new(),
        glyphs: [
            Glyph {
                id: 1,
                x: 16.0,
                y: 96.0,
                transform: None,
            },
            Glyph {
                id: 25,
                x: 88.0,
                y: 96.0,
                transform: None,
            },
            Glyph {
                id: 61,
                x: 160.0,
                y: 96.0,
                transform: None,
            },
        ]
        .into(),
        style: cherenkov::GlyphStyle::Fill,
    }
}

#[test]
fn colr_glyphs_render_their_paint_graph() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(bytes) = std::fs::read(FONT_PATH) else {
        eprintln!("text-colr font not checked out; skipping");
        return Ok(());
    };
    let engine = match Engine::<Gpu>::new(GpuConfig::default()) {
        Ok(engine) => engine,
        Err(EngineError::NoAdapter) => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    let font = engine.font(FontSource::bytes(bytes))?;
    let surface = engine.surface(Offscreen::new((320, 160), OffscreenFormat::LinearF16))?;
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.glyphs(&run(font.id()), WorkingColor::new([1.0, 0.0, 0.0, 1.0]));
        }));
    });
    engine.render(cherenkov_gpu::FrameTime::now())?;
    let rb = surface.readback()?;
    // Count distinct hues: quantize each non-clear pixel's chroma angle.
    let mut hues = std::collections::HashSet::new();
    let mut coloured = 0usize;
    for [r, g, b, a] in &rb.pixels {
        if *a < 0.05 {
            continue;
        }
        let (r, g, b) = (*r, *g, *b);
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        if max - min < 0.05 {
            continue;
        }
        coloured += 1;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "hue sectors fit u8 comfortably"
        )]
        let sector = ((b.atan2(r - g)) * 16.0) as u8;
        hues.insert(sector);
    }
    assert!(coloured > 500, "expected coloured glyph pixels: {coloured}");
    assert!(
        hues.len() >= 2,
        "COLR glyph should use palette colours, not the run paint: hues {hues:?}"
    );
    Ok(())
}
