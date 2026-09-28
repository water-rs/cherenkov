// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The CPU backend's sRGB presentation against the `f64` oracle (#96):
//! `present_srgb8` must produce the oracle's presented bytes for
//! in-gamut pixels exactly (the pre-#96 clamp contract) and stay within
//! one unorm-8 step elsewhere — same algorithm, `f32` against `f64`.

use cherenkov_cpu::present_srgb8;
use cherenkov_oracle::Image;
use cherenkov_oracle::present::{present_srgb, quantize_unorm8};

/// Oracle-presented bytes for one opaque premultiplied pixel.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "quantized pixels are in [0,1]; the byte store rounds"
)]
fn oracle_bytes(p: [f32; 4]) -> [u8; 4] {
    let image = Image {
        width: 1,
        height: 1,
        pixels: vec![p.map(f64::from)],
    };
    let presented = quantize_unorm8(&present_srgb(1.0, &image));
    presented.pixels[0].map(|c| (c * 255.0).round() as u8)
}

#[test]
fn present_srgb8_matches_oracle() {
    let pixels: [[f32; 4]; 14] = [
        [0.5, 0.5, 0.5, 1.0],   // in-gamut grey
        [0.25, 0.5, 0.75, 1.0], // in-gamut colour
        [1.0, 0.0, 0.0, 1.0],   // P3 red
        [0.0, 1.0, 0.0, 1.0],   // P3 green
        [0.0, 0.0, 1.0, 1.0],   // P3 blue
        [0.0, 1.0, 1.0, 1.0],   // P3 cyan
        [1.0, 0.0, 1.0, 1.0],   // P3 magenta
        [1.0, 1.0, 0.0, 1.0],   // P3 yellow
        [0.95, 0.3, 0.15, 1.0], // crossing the boundary
        [4.0, 4.0, 4.0, 1.0],   // HDR white
        [-0.1, 0.5, 0.5, 1.0],  // negative channel
        [1.2, 0.9, 0.4, 1.0],   // bright orange, partly out
        // An in-sRGB colour round-tripped through the P3 matrices lands a
        // few ULPs out of gamut; the map must keep the pre-#96 bytes.
        [-7e-18, -3e-17, 1.0, 1.0],
        // Half-alpha premultiplied P3 red.
        [0.5, 0.0, 0.0, 0.5],
    ];
    let got = present_srgb8(&pixels);
    for (i, p) in pixels.iter().enumerate() {
        let want = oracle_bytes(*p);
        for (g, w) in got[i * 4..i * 4 + 4].iter().zip(want) {
            assert!(g.abs_diff(w) <= 1, "pixel {i} {p:?}: cpu {g} vs oracle {w}");
        }
    }
}
