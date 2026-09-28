// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Presentation of the working-space image into an output encoding — the
//! oracle twin of `cherenkov-gpu`'s `render/present.wgsl`, evaluated in
//! `f64` on the oracle's [`Image`].
//!
//! The bench's `render --present` compares an engine image that went
//! through the real presentation pass against the same reference image run
//! through these functions.

use crate::Image;
use crate::color::{linear_srgb_to_linear_p3, mat3_mul, srgb_decode, srgb_encode};

/// Linear Display P3 → linear sRGB — the Bradford-adapted D65 matrix of
/// `present.wgsl`, the same constants in `f64`.
const P3_TO_LINEAR_SRGB: [[f64; 3]; 3] = [
    [1.224_940_2, -0.224_940_2, 0.0],
    [-0.042_056_95, 1.042_056_9, 0.0],
    [-0.019_637_55, -0.078_636_05, 1.098_273_6],
];

/// Presents `image` to an sRGB output at display `headroom`.
///
/// The `present.wgsl` shader-transfer path per pixel under premultiplied
/// output alpha: unpremultiply, the P3 → sRGB matrix, clamp to `[0, 1]`,
/// the sRGB transfer, re-premultiply. Out-of-gamut P3 colours and values
/// above `1.0` clip, matching the pass's current contract.
///
/// Returns premultiplied *encoded* sRGB in `[0, 1]` with linear alpha —
/// the values an sRGB presentation texture stores.
///
/// `headroom` is the scene's declared presentation headroom; presentation
/// does not tone-map yet (#97), so it is accepted and unused.
#[must_use]
pub fn present_srgb(headroom: f64, image: &Image) -> Image {
    let _ = headroom;
    Image {
        width: image.width,
        height: image.height,
        pixels: image
            .pixels
            .iter()
            .map(|&p| present_srgb_pixel(p))
            .collect(),
    }
}

/// Presents `image` to an extended linear Display P3 host texture
/// (`OutputColor::LinearDisplayP3`): the identity — the extended range is
/// handed to the host unchanged. `headroom` is carried for #97.
#[must_use]
pub fn present_linear_p3(headroom: f64, image: &Image) -> Image {
    let _ = headroom;
    image.clone()
}

/// Lifts one premultiplied encoded-sRGB presented pixel back into the
/// working space.
///
/// The sRGB transfer is decoded, then sRGB → P3 primaries applied on the
/// premultiplied channels (a linear matrix commutes with the alpha scale).
/// A presented sRGB output and a presented reference compare in
/// [`crate::metrics`] once both are lifted like this.
#[must_use]
pub fn presented_srgb_to_working([r, g, b, a]: [f64; 4]) -> [f64; 4] {
    let p3 = linear_srgb_to_linear_p3([srgb_decode(r), srgb_decode(g), srgb_decode(b)]);
    [p3[0], p3[1], p3[2], a]
}

/// `present.wgsl`'s sRGB path for one premultiplied working-space pixel
/// under `OutputAlpha::Premultiplied`: straight alpha is recovered for the
/// gamut conversion, and the encoded colour is re-premultiplied.
fn present_srgb_pixel([r, g, b, a]: [f64; 4]) -> [f64; 4] {
    let straight = if a > 0.0 {
        [r / a, g / a, b / a]
    } else {
        [0.0; 3]
    };
    let srgb = mat3_mul(&P3_TO_LINEAR_SRGB, straight).map(|c| srgb_encode(c.clamp(0.0, 1.0)));
    [srgb[0] * a, srgb[1] * a, srgb[2] * a, a]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(p: [f64; 4]) -> Image {
        Image::filled(1, 1, p)
    }

    fn px(image: &Image) -> [f64; 4] {
        image.pixels[0]
    }

    fn assert_close(actual: [f64; 4], expected: [f64; 4]) {
        for (a, e) in actual.into_iter().zip(expected) {
            assert!((a - e).abs() < 1e-6, "{a} != {e}");
        }
    }

    #[test]
    fn srgb_mid_gray_in_gamut() {
        // 0.5 in P3 maps to ~0.5 in sRGB; sRGB-encodes to ~0.7354.
        assert_close(
            px(&present_srgb(1.0, &img([0.5, 0.5, 0.5, 1.0]))),
            [0.735357, 0.735357, 0.735357, 1.0],
        );
    }

    #[test]
    fn srgb_p3_red_clips_to_srgb_red() {
        // P3's red primary is outside sRGB: the [0,1] clamp lands on
        // encoded sRGB red.
        assert_close(
            px(&present_srgb(1.0, &img([1.0, 0.0, 0.0, 1.0]))),
            [1.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn srgb_out_of_gamut_channel_clips() {
        // P3 (0, 1, 0.5): r and g clip, b survives at 0.47050075 linear
        // → 0.7156 encoded.
        assert_close(
            px(&present_srgb(1.0, &img([0.0, 1.0, 0.5, 1.0]))),
            [0.0, 1.0, 0.715583, 1.0],
        );
    }

    #[test]
    fn srgb_hdr_white_clips_to_sdr_white() {
        // 4× SDR white clamps at the [0,1] contract — no tone mapping yet.
        assert_close(
            px(&present_srgb(4.0, &img([4.0, 4.0, 4.0, 1.0]))),
            [1.0, 1.0, 1.0, 1.0],
        );
    }

    #[test]
    fn srgb_in_gamut_colour_encodes() {
        // P3 (0.25, 0.5, 0.75): in-gamut after the matrix, then encoded.
        assert_close(
            px(&present_srgb(1.0, &img([0.25, 0.5, 0.75, 1.0]))),
            [0.477456, 0.742240, 0.895978, 1.0],
        );
    }

    #[test]
    fn srgb_premultiplied_output() {
        // Premultiplied input is unpremultiplied for the conversion and
        // the encoded result is re-premultiplied.
        assert_close(
            px(&present_srgb(1.0, &img([0.1, 0.2, 0.05, 0.5]))),
            [0.215091, 0.335728, 0.151213, 0.5],
        );
        // Transparent pixels present as transparent black.
        assert_close(px(&present_srgb(1.0, &img([0.0, 0.0, 0.0, 0.0]))), [0.0; 4]);
    }

    #[test]
    fn linear_p3_is_identity() {
        // Extended range — HDR and out-of-sRGB values pass through.
        let pixel = [4.0, -0.25, 1.5, 0.75];
        assert_eq!(px(&present_linear_p3(4.0, &img(pixel))), pixel);
    }

    #[test]
    fn presented_srgb_round_trips_to_working() {
        // An in-gamut colour presented and lifted back lands on the
        // original working-space value (the simplified WGSL constants are
        // not the exact matrices, so the round trip is approximate).
        let pixel = [0.25, 0.5, 0.75, 1.0];
        let presented = px(&present_srgb(1.0, &img(pixel)));
        let lifted = presented_srgb_to_working(presented);
        assert_close(lifted, pixel);
    }
}
