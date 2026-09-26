// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Frontend glyph contracts use real, unhinted outlines as their reference.

use cherenkov::{FontId, Glyph, GlyphRun, GlyphStyle};
use cherenkov_oracle::glyphs::styled_outline;
use kurbo::{Affine, Shape as _, Stroke};

fn run() -> GlyphRun {
    GlyphRun {
        font: FontId::new(1),
        size: 40.0,
        coords: Vec::new(),
        glyphs: vec![Glyph {
            id: 36,
            x: 13.25,
            y: 48.5,
            transform: None,
        }],
        style: GlyphStyle::Fill,
    }
}

#[test]
fn glyph_transform_is_about_the_origin_in_run_units() {
    let bytes = include_bytes!("../../scenes/fonts/NotoSans.ttf");
    let mut run = run();
    let initial =
        styled_outline(bytes, 0, &run, &run.glyphs[0], Affine::IDENTITY).expect("outline");
    let local = Affine::new([0.0, 1.5, -0.75, 0.0, 2.25, -1.125]);
    run.glyphs[0].transform = Some(local);
    let actual = styled_outline(bytes, 0, &run, &run.glyphs[0], Affine::IDENTITY)
        .expect("transformed outline");
    let origin = Affine::translate((13.25, 48.5));
    let expected = origin * local * origin.inverse() * initial;
    let actual_bounds = actual.bounding_box();
    let expected_bounds = expected.bounding_box();
    for (value, expected) in [
        actual_bounds.x0,
        actual_bounds.y0,
        actual_bounds.x1,
        actual_bounds.y1,
    ]
    .into_iter()
    .zip([
        expected_bounds.x0,
        expected_bounds.y0,
        expected_bounds.x1,
        expected_bounds.y1,
    ]) {
        assert!((value - expected).abs() < 1e-10);
    }
}

#[test]
fn glyph_stroke_width_is_in_run_units_before_placement() {
    let bytes = include_bytes!("../../scenes/fonts/NotoSans.ttf");
    let mut run = run();
    run.glyphs[0].x = 0.0;
    run.glyphs[0].y = 0.0;
    let fill = styled_outline(bytes, 0, &run, &run.glyphs[0], Affine::IDENTITY).expect("fill");
    run.style = GlyphStyle::Stroke(Stroke::new(4.0).with_join(kurbo::Join::Round));
    let stroke = styled_outline(bytes, 0, &run, &run.glyphs[0], Affine::IDENTITY).expect("stroke");
    let fill = fill.bounding_box();
    let stroke = stroke.bounding_box();
    assert!((fill.x0 - stroke.x0 - 2.0).abs() < 1e-3);
    assert!((stroke.x1 - fill.x1 - 2.0).abs() < 1e-3);
    assert!((fill.y0 - stroke.y0 - 2.0).abs() < 1e-3);
    assert!((stroke.y1 - fill.y1 - 2.0).abs() < 1e-3);
}
