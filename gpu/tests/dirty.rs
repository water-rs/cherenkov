// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Requires a working adapter; CI uses Vulkan lavapipe.

use cherenkov::Backend;

#[test]
fn randomized_incremental_matches_full_lowering() {
    let (mut renderer, _) = cherenkov_gpu::Gpu::init(cherenkov_gpu::GpuConfig::default())
        .expect("GPU adapter required");
    cherenkov::testing::incremental::equivalence(&mut renderer);
}

/// A glyph's atlas coordinates must not overwrite a path clip's atlas origin.
#[test]
fn a_glyph_keeps_its_path_clip_mask_when_composed() {
    use cherenkov::kurbo::{Rect, Shape as _};
    use cherenkov::{
        Draw, Engine, FontSource, FrameTime, Glyph, GlyphRun, GlyphStyle, Offscreen,
        OffscreenFormat, WorkingColor,
    };
    let engine = Engine::<cherenkov_gpu::Gpu>::new(cherenkov_gpu::GpuConfig::default())
        .expect("GPU adapter required");
    let font = engine
        .font(FontSource::bytes(
            std::fs::read("../scenes/fonts/NotoSans.ttf").expect("font"),
        ))
        .expect("register font");
    let run = GlyphRun {
        font: font.id(),
        size: 32.,
        coords: Vec::new(),
        glyphs: vec![Glyph {
            id: 36,
            x: 8.,
            y: 40.,
            transform: None,
        }],
        style: GlyphStyle::Fill,
    };
    let surface = engine
        .surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))
        .expect("surface");
    let clip = Rect::new(14., 0., 24., 64.);
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| c.glyphs(run.clone(), WorkingColor::WHITE)));
    });
    engine.render(FrameTime::now()).expect("unclipped render");
    let unclipped = surface.readback().expect("readback");
    assert!(
        unclipped
            .pixels
            .iter()
            .enumerate()
            .any(|(i, p)| !(14..24).contains(&(i % 64)) && p[3] > 0.1)
    );
    // The first render populated glyph cells; the clip now occupies a different
    // atlas origin, so losing its UVs cannot accidentally sample the right mask.
    surface.update(|tx| {
        tx[surface.root()].content(
            surface.record(|c| c.clip(clip.to_path(0.01), |c| c.glyphs(run, WorkingColor::WHITE))),
        );
    });
    engine.render(FrameTime::now()).expect("path clip render");
    let masked = surface.readback().expect("readback");
    assert!(masked.pixels.iter().any(|p| p[3] > 0.5), "glyph is visible");
    // An integer-edged path mask has exact 0/1 coverage. Crop the unmasked
    // glyph to obtain its reference without involving the analytic SDF's AA.
    for (index, (a, b)) in unclipped.pixels.iter().zip(masked.pixels).enumerate() {
        let expected = if (14..24).contains(&(index % 64)) {
            *a
        } else {
            [0.0; 4]
        };
        assert_eq!(
            expected.map(f32::to_bits),
            b.map(f32::to_bits),
            "pixel {index}"
        );
    }
}
