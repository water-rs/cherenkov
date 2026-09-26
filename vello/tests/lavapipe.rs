// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Pixel tests against a real adapter (lavapipe in CI/dev machines).
//!
//! Every expected pixel is computed from first principles: linear Display
//! P3 → linear sRGB matrix → sRGB encode → quantize to u8 → decode back →
//! linear P3, so the tolerance is 2/255 per channel.

use cherenkov::kurbo::{Affine, Rect};
use cherenkov::{Draw, WorkingColor};
use cherenkov_vello::{Engine, Next, Offscreen, Vello, VelloConfig};

/// sRGB transfer-function encode.
fn srgb_encode(c: f64) -> f64 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055f64.mul_add(c.powf(1.0 / 2.4), -0.055)
    }
}

/// sRGB transfer-function decode.
fn srgb_decode(e: f64) -> f64 {
    if e <= 0.04045 {
        e / 12.92
    } else {
        ((e + 0.055) / 1.055).powf(2.4)
    }
}

/// linear sRGB → linear Display P3 (inverse of the front end's matrix).
const LINEAR_SRGB_TO_LINEAR_P3: [[f64; 3]; 3] = [
    [0.822_461_96, 0.177_538_04, 0.0],
    [0.033_194_2, 0.966_805_8, 0.0],
    [0.017_082_632, 0.072_397_44, 0.910_519_96],
];

const LINEAR_P3_TO_LINEAR_SRGB: [[f64; 3]; 3] = [
    [1.224_940_2, -0.224_940_18, 0.0],
    [-0.042_056_955, 1.042_056_9, 0.0],
    [-0.019_637_555, -0.078_636_04, 1.098_273_6],
];

fn mat_vec(m: &[[f64; 3]; 3], [x, y, z]: [f64; 3]) -> [f64; 3] {
    let dot = |row: &[f64; 3]| row[2].mul_add(z, row[1].mul_add(y, row[0] * x));
    [dot(&m[0]), dot(&m[1]), dot(&m[2])]
}

/// The expected stored-then-decoded pixel of straight working-space colour
/// `src` composited src-over onto `dst` (also straight working space).
///
/// Vello renders into the sRGB-encoded 8-bit target and blends in that
/// encoded space, so the model is: straight linear P3 → linear sRGB → sRGB
/// encode → premultiply → src-over → quantize to u8 → decode → linear P3.
fn expected_pixel(src: [f64; 4], dst: [f64; 4]) -> [f64; 4] {
    let to_premul_encoded = |[r, g, b, a]: [f64; 4]| {
        let [r, g, b] = mat_vec(&LINEAR_P3_TO_LINEAR_SRGB, [r, g, b]);
        [
            srgb_encode(r.clamp(0.0, 1.0)) * a,
            srgb_encode(g.clamp(0.0, 1.0)) * a,
            srgb_encode(b.clamp(0.0, 1.0)) * a,
            a,
        ]
    };
    let s = to_premul_encoded(src);
    let d = to_premul_encoded(dst);
    let s3 = s[3];
    let over = |s: f64, d: f64| (1.0 - s3).mul_add(d, s);
    let out = [
        over(s[0], d[0]),
        over(s[1], d[1]),
        over(s[2], d[2]),
        over(s3, d[3]),
    ];
    // Quantize each encoded channel to u8, decode, convert to linear P3.
    let enc: Vec<f64> = out
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() / 255.0)
        .collect();
    let lin = [
        srgb_decode(enc[0]),
        srgb_decode(enc[1]),
        srgb_decode(enc[2]),
    ];
    let p3 = mat_vec(&LINEAR_SRGB_TO_LINEAR_P3, lin);
    [p3[0], p3[1], p3[2], enc[3]]
}

fn engine() -> Option<Engine<Vello>> {
    match Engine::<Vello>::new(VelloConfig::default()) {
        Ok(engine) => Some(engine),
        Err(e) => {
            eprintln!("no adapter, skipping ({e})");
            None
        }
    }
}

fn assert_pixel(actual: [f32; 4], expected: [f64; 4], what: &str) {
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (f64::from(*a) - e).abs() <= 2.0 / 255.0 + 1e-4,
            "{what}: {actual:?} != {expected:?}"
        );
    }
}

fn px(readback: &cherenkov_vello::Readback, x: u32, y: u32) -> [f32; 4] {
    readback.pixels[(y * readback.width + x) as usize]
}

#[test]
fn solid_fill_matches_expected_pixels() {
    let Some(engine) = engine() else { return };
    let surface = engine.surface(Offscreen::new((64, 64))).expect("surface");
    surface.clear_color(WorkingColor::TRANSPARENT);
    // A P3 colour: mostly red.
    let fill = WorkingColor::new([0.9, 0.1, 0.2, 1.0]);
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| c.fill(Rect::new(8., 8., 40., 40.), fill)));
    });
    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("render");
    assert_eq!(next, Next::Idle);
    let readback = surface.readback().expect("readback");
    // Centre pixel: the fill colour through the full conversion pipeline.
    assert_pixel(
        px(&readback, 24, 24),
        expected_pixel(fill.components.map(f64::from), [0., 0., 0., 0.]),
        "centre",
    );
    // Corner pixel: untouched clear colour.
    assert_pixel(
        px(&readback, 0, 0),
        expected_pixel([0., 0., 0., 0.], [0., 0., 0., 0.]),
        "corner",
    );
    // Just outside the fill.
    assert_pixel(
        px(&readback, 41, 24),
        expected_pixel([0., 0., 0., 0.], [0., 0., 0., 0.]),
        "outside",
    );
}

#[test]
fn layer_tree_composes_transform_opacity_clip() {
    let Some(engine) = engine() else { return };
    let surface = engine.surface(Offscreen::new((64, 64))).expect("surface");
    surface.clear_color(WorkingColor::BLACK);

    let moved = surface.layer();
    let clipped = surface.layer();

    let white = surface.record(|c| {
        c.fill(Rect::new(0., 0., 20., 20.), WorkingColor::WHITE);
    });
    let wide = surface.record(|c| {
        c.fill(Rect::new(0., 0., 64., 64.), WorkingColor::WHITE);
    });

    surface.update(|tx| {
        // `moved`: translate (10,10), opacity 0.5, white square.
        tx[&moved]
            .transform(Affine::translate((10., 10.)))
            .opacity(0.5)
            .content(white);
        // `clipped`: clip to Rect(0,0,20,20), full-width white fill.
        tx[&clipped].clip(Rect::new(0., 0., 20., 20.)).content(wide);
        tx[surface.root()].push(&moved).push(&clipped);
    });

    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("render");
    assert_eq!(next, Next::Idle);
    let readback = surface.readback().expect("readback");

    let half = expected_pixel(
        [1., 1., 1., 0.5],
        WorkingColor::BLACK.components.map(f64::from),
    );
    let full = expected_pixel(
        WorkingColor::WHITE.components.map(f64::from),
        WorkingColor::BLACK.components.map(f64::from),
    );
    let clear = expected_pixel(
        [0., 0., 0., 0.],
        WorkingColor::BLACK.components.map(f64::from),
    );
    // `moved` covers (10..30, 10..30); `clipped` covers (0..20, 0..20).
    // Inside both (15,15): clipped white under 0.5-opacity white = white.
    assert_pixel(px(&readback, 15, 15), full, "inside both layers");
    // Inside `clipped` only: full white.
    assert_pixel(px(&readback, 5, 5), full, "clipped white pixel");
    // Inside `moved` but outside `clipped` (25,25): 0.5 white over black.
    assert_pixel(px(&readback, 25, 25), half, "moved-only pixel");
    // Outside both: untouched black clear.
    assert_pixel(px(&readback, 50, 50), clear, "cleared pixel");
}
