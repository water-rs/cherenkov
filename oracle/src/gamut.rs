// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gamut mapping into sRGB — the `f64` reference for the presentation
//! pass's out-of-gamut handling.
//!
//! The destination is linear sRGB `r, g, b ∈ [0, 1]`: the gamut's
//! component bounds. Everything here is linear-light; the sRGB transfer
//! applies after mapping, in the caller.
//!
//! The selected algorithm is an analytic `OKLab` clip — Ottosson's
//! cusp-triangle gamut boundary with one Halley refinement, projecting
//! towards the lightness axis at adaptively-chosen lightness
//! (`ok_color.h`'s `gamut_clip_adaptive_L0_L_cusp`). It preserves hue
//! exactly and runs at fixed per-pixel cost, which is why the present
//! pass ships it rather than the CSS Color 4 binary search (#96: the
//! search's data-dependent loop was ~2.7x the analytic pass's GPU time
//! on lavapipe, and its `ΔE_OK` to the spec map stayed under one JND on
//! the sweep). `bench gamut-sweep` carries the spec algorithm as its
//! measurement reference.
//!
//! In-gamut colours return bit-for-bit: the caller's in-gamut fast path,
//! and the function's own bounds check, return the input untouched, so
//! an in-sRGB colour's output equals plain conversion plus clamp.
//!
use crate::color::mat3_mul;

/// Linear sRGB → `LMS` (the `OKLab` cone-response matrix, before the
/// nonlinearity).
const SRGB_TO_LMS: [[f64; 3]; 3] = [
    [0.412_221_470_8, 0.536_332_536_3, 0.051_445_992_9],
    [0.211_903_498_2, 0.680_699_545_1, 0.107_396_956_6],
    [0.088_302_461_9, 0.281_718_837_6, 0.629_978_700_5],
];

/// `cbrt(LMS)` → `OKLab`.
const LMS_TO_OKLAB: [[f64; 3]; 3] = [
    [0.210_454_255_3, 0.793_617_785_0, -0.004_072_046_8],
    [1.977_998_495_1, -2.428_592_205_0, 0.450_593_709_9],
    [0.025_904_037_1, 0.782_771_766_2, -0.808_675_766_0],
];

/// `OKLab` → `LMS′` (the pre-cube intermediates; the matrix's first column
/// is all `1` — `L` enters each channel whole).
const OKLAB_TO_LMS: [[f64; 3]; 3] = [
    [1.0, 0.396_337_777_4, 0.215_803_757_3],
    [1.0, -0.105_561_345_8, -0.063_854_172_8],
    [1.0, -0.089_484_177_5, -1.291_485_548_0],
];

/// LMS (cubed) → linear sRGB.
const LMS_TO_SRGB: [[f64; 3]; 3] = [
    [4.076_741_662_1, -3.307_711_591_3, 0.230_969_929_2],
    [-1.268_438_004_6, 2.609_757_401_1, -0.341_319_396_5],
    [-0.004_196_086_3, -0.703_418_614_7, 1.707_614_701_0],
];

/// Signed cube root — the LMS nonlinearity, safe for the negative
/// components an out-of-gamut colour produces.
fn cbrt(x: f64) -> f64 {
    x.cbrt()
}

/// Linear sRGB (possibly outside `[0, 1]`) → `OKLab` `[L, a, b]`.
#[must_use]
pub fn linear_srgb_to_oklab(rgb: [f64; 3]) -> [f64; 3] {
    let lms = mat3_mul(&SRGB_TO_LMS, rgb);
    mat3_mul(&LMS_TO_OKLAB, lms.map(cbrt))
}

/// `OKLab` → linear sRGB (not clamped).
#[must_use]
pub fn oklab_to_linear_srgb(lab: [f64; 3]) -> [f64; 3] {
    let lms_ = mat3_mul(&OKLAB_TO_LMS, lab);
    mat3_mul(&LMS_TO_SRGB, lms_.map(|v| v * v * v))
}

/// `ΔE_OK` — Euclidean distance in `OKLab` (CSS Color 4 §20.3).
#[must_use]
pub fn delta_e_ok(one: [f64; 3], two: [f64; 3]) -> f64 {
    let [dl, da, db] = [one[0] - two[0], one[1] - two[1], one[2] - two[2]];
    db.mul_add(db, da.mul_add(da, dl * dl)).sqrt()
}

/// Whether a linear-sRGB colour is inside the gamut's `[0, 1]` bounds.
fn in_gamut(rgb: [f64; 3]) -> bool {
    rgb.iter().all(|&c| (0.0..=1.0).contains(&c))
}

/// Clamps each component to `[0, 1]` — the spec's `clip`.
fn clip(rgb: [f64; 3]) -> [f64; 3] {
    rgb.map(|c| c.clamp(0.0, 1.0))
}

/// The maximum saturation `S = C/L` reachable along the `OKLab` hue
/// direction `(a, b)` (normalized) inside the sRGB gamut — `ok_color.h`'s
/// `compute_max_saturation` structure (per-channel polynomial estimates
/// refined by one Halley step each).
fn compute_max_saturation(a: f64, b: f64) -> f64 {
    // Max saturation is where one sRGB channel first goes to zero. The
    // reference implementation picks the channel by a fitted partition of
    // hue space; instead we solve all three channel surfaces and take the
    // smallest root that one Halley step actually lands on (residual near
    // zero). That is cheaper to make deterministic across `f32` and `f64`:
    // the partition conditions sit within ~1e-6 of 1.0 at the sRGB vertex
    // hues, so the two precisions can pick different branches there.
    const SETS: [[f64; 8]; 3] = [
        // Red, green, blue: five polynomial coefficients then the LMS→sRGB
        // channel weights.
        [
            1.190_862_77,
            1.765_767_28,
            0.596_626_41,
            0.755_151_97,
            0.567_712_45,
            4.076_741_662_1,
            -3.307_711_591_3,
            0.230_969_929_2,
        ],
        [
            0.739_565_15,
            -0.459_544_04,
            0.082_854_27,
            0.125_410_70,
            0.145_032_04,
            -1.268_438_004_6,
            2.609_757_401_1,
            -0.341_319_396_5,
        ],
        [
            1.357_336_52,
            -0.009_157_99,
            -1.151_302_10,
            -0.505_596_06,
            0.006_921_67,
            -0.004_196_086_3,
            -0.703_418_614_7,
            1.707_614_701_0,
        ],
    ];
    let k_l = 0.396_337_777_4f64.mul_add(a, 0.215_803_757_3 * b);
    let k_m = (-0.105_561_345_8f64).mul_add(a, -0.063_854_172_8 * b);
    let k_s = (-0.089_484_177_5f64).mul_add(a, -1.291_485_548_0 * b);

    // One Halley step from each channel's polynomial estimate; a step that
    // diverged (no real root in reach — the channel that only exceeds the
    // gamut, never zeros) leaves a large residual and is discarded.
    let eval = |s: f64, w: [f64; 3]| {
        let l_ = s.mul_add(k_l, 1.0);
        let m_ = s.mul_add(k_m, 1.0);
        let s_ = s.mul_add(k_s, 1.0);
        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let sc = s_ * s_ * s_;
        let l_ds = 3.0 * k_l * l_ * l_;
        let m_ds = 3.0 * k_m * m_ * m_;
        let s_ds = 3.0 * k_s * s_ * s_;
        let l_ds2 = 6.0 * k_l * k_l * l_;
        let m_ds2 = 6.0 * k_m * k_m * m_;
        let s_ds2 = 6.0 * k_s * k_s * s_;
        (
            ws_eval(w, l, m, sc),
            ws_eval(w, l_ds, m_ds, s_ds),
            ws_eval(w, l_ds2, m_ds2, s_ds2),
        )
    };
    let mut s = f64::INFINITY;
    let mut fallback = f64::INFINITY;
    for [k0, k1, k2, k3, k4, wl, wm, ws] in SETS {
        // Polynomial estimate of this channel's root.
        let est = k4.mul_add(a * b, k3.mul_add(a * a, b.mul_add(k2, a.mul_add(k1, k0))));
        if est > 0.0 {
            fallback = fallback.min(est);
        }
        let (f, f1, f2) = eval(est, [wl, wm, ws]);
        let den = f1.mul_add(f1, -0.5 * f * f2);
        if den == 0.0 {
            continue;
        }
        let root = est - f * f1 / den;
        // Keep the root only if the step converged onto the surface — a
        // diverged step leaves a large residual at its landing point.
        if root.is_finite() && root > 0.0 && eval(root, [wl, wm, ws]).0.abs() < 0.05 {
            s = s.min(root);
        }
    }
    if s.is_finite() { s } else { fallback.max(0.0) }
}

fn ws_eval(w: [f64; 3], l: f64, m: f64, s: f64) -> f64 {
    w[2].mul_add(s, w[0].mul_add(l, w[1] * m))
}

/// The cusp of the sRGB gamut slice at hue `(a, b)` — the `(L, C)` point
/// of maximum chroma on that hue's boundary (`ok_color.h`'s
/// `find_cusp`).
fn find_cusp(a: f64, b: f64) -> (f64, f64) {
    let s_cusp = compute_max_saturation(a, b);
    // At max saturation one channel sits at zero; scale so the largest
    // channel reaches 1 — that point is the cusp.
    let rgb = oklab_to_linear_srgb([1.0, s_cusp * a, s_cusp * b]);
    let l_cusp = cbrt(1.0 / rgb[0].max(rgb[1]).max(rgb[2]));
    (l_cusp, l_cusp * s_cusp)
}

/// Intersects the segment `(L0, 0) → (L1, C1)` in the `OKLCh` slice of hue
/// `(a, b)` with the sRGB gamut's boundary, returning the parameter `t`
/// (`t·C1` is the boundary chroma). The lower half of the boundary is
/// the cusp triangle's linear edge; the upper half takes one Halley step
/// against the true boundary surface (`ok_color.h`'s
/// `find_gamut_intersection`).
#[allow(clippy::similar_names, clippy::many_single_char_names)] // names mirror ok_color.h
fn find_gamut_intersection(a: f64, b: f64, l1: f64, c1: f64, l0: f64, cusp: (f64, f64)) -> f64 {
    let (lc, cc) = cusp;
    if (l1 - l0).mul_add(cc, -(lc - l0) * c1) <= 0.0 {
        // Lower half: the triangle's linear edge is the approximation.
        return cc * l0 / c1.mul_add(lc, cc * (l0 - l1));
    }
    // Upper half: intersect the triangle edge, then take one Halley step
    // against the true (cubic) boundary — the upper surface is curved.
    let t = cc * (l0 - 1.0) / (c1.mul_add(lc - 1.0, cc * (l0 - l1)));
    let dl = l1 - l0;
    let dc = c1;
    let k_l = 0.396_337_777_4f64.mul_add(a, 0.215_803_757_3 * b);
    let k_m = (-0.105_561_345_8f64).mul_add(a, -0.063_854_172_8 * b);
    let k_s = (-0.089_484_177_5f64).mul_add(a, -1.291_485_548_0 * b);
    let l_dt = dc.mul_add(k_l, dl);
    let m_dt = dc.mul_add(k_m, dl);
    let s_dt = dc.mul_add(k_s, dl);
    let l_at = l0.mul_add(1.0 - t, t * l1);
    let c_at = t * c1;
    let l_ = c_at.mul_add(k_l, l_at);
    let m_ = c_at.mul_add(k_m, l_at);
    let s_ = c_at.mul_add(k_s, l_at);
    let l3 = l_ * l_ * l_;
    let m3 = m_ * m_ * m_;
    let s3 = s_ * s_ * s_;
    let ldt = 3.0 * l_dt * l_ * l_;
    let mdt = 3.0 * m_dt * m_ * m_;
    let sdt = 3.0 * s_dt * s_ * s_;
    let ldt2 = 6.0 * l_dt * l_dt * l_;
    let mdt2 = 6.0 * m_dt * m_dt * m_;
    let sdt2 = 6.0 * s_dt * s_dt * s_;
    // Halley on each channel's `= 1` face; the smallest forward step wins.
    let mut step = f64::MAX;
    let weights: [[f64; 3]; 3] = [
        [4.076_741_662_1, -3.307_711_591_3, 0.230_969_929_2],
        [-1.268_438_004_6, 2.609_757_401_1, -0.341_319_396_5],
        [-0.004_196_086_3, -0.703_418_614_7, 1.707_614_701_0],
    ];
    for w in weights {
        let r = w[0].mul_add(l3, w[1].mul_add(m3, w[2] * s3)) - 1.0;
        let r1 = w[0].mul_add(ldt, w[1].mul_add(mdt, w[2] * sdt));
        let r2 = w[0].mul_add(ldt2, w[1].mul_add(mdt2, w[2] * sdt2));
        let u = r1 / r1.mul_add(r1, -0.5 * r * r2);
        if u >= 0.0 {
            step = step.min(-r * u);
        }
    }
    t + step
}

/// The analytic `OKLab` gamut clip: `ok_color.h`'s
/// `gamut_clip_adaptive_L0_L_cusp` with `alpha = 0.05`.
///
/// Projects the out-of-gamut `rgb` towards the lightness axis in its
/// hue's `OKLCh` slice — the segment from `(L0, 0)` to the colour — and
/// returns the boundary point. `L0` blends adaptively between the
/// colour's own lightness (near the gamut) and the cusp's lightness
/// (far out), so saturated brights keep their chroma and lightness moves
/// only as far as the projection travels. Hue is preserved exactly.
/// Fixed per-pixel cost: one cusp solve, one segment intersection, one
/// Halley step on the upper half.
///
/// A final `[0, 1]` clamp absorbs the triangle approximation's residual
/// — under a thousandth of the range — so the output is always storable.
#[must_use]
pub fn gamut_map_srgb_analytic(rgb: [f64; 3]) -> [f64; 3] {
    gamut_map_srgb_project(rgb, Anchor::Adaptive)
}

/// Which point on the lightness axis the projection anchors at —
/// `ok_color.h`'s `L0` choices.
#[derive(Clone, Copy)]
pub enum Anchor {
    /// `clamp(L, 0, 1)` — the colour's own lightness
    /// (`gamut_clip_preserve_chroma`).
    Preserve,
    /// The hue slice's cusp lightness (`gamut_clip_project_to_L_cusp`).
    Cusp,
    /// Soft interpolation between own lightness and cusp lightness
    /// (`gamut_clip_adaptive_L0_L_cusp`, `alpha = 0.05`).
    Adaptive,
}

/// The shared projection core under [`Anchor`].
#[must_use]
#[allow(clippy::many_single_char_names)] // l/a/b/c mirror OKLab notation
pub fn gamut_map_srgb_project(rgb: [f64; 3], anchor: Anchor) -> [f64; 3] {
    if in_gamut(rgb) {
        return rgb;
    }
    let lab = linear_srgb_to_oklab(rgb);
    // The spec's local-MINDE rule: when the plain channel-clip is already
    // within a ΔE_OK JND of the colour, keep the clip — the pre-#96 bytes.
    // Besides the perceptual headroom (the projection could only move the
    // colour a sub-just-noticeable distance), this keeps a colour that is
    // a few ULPs out of gamut — an in-sRGB value round-tripped through the
    // P3 matrices — from entering the projection where the cusp fit is
    // weakest.
    let clipped = clip(rgb);
    if delta_e_ok(linear_srgb_to_oklab(clipped), lab) < 0.02 {
        return clipped;
    }
    let [l, a, b] = lab;
    let c = (1e-5_f64).max(a.hypot(b));
    let (a_, b_) = (a / c, b / c);
    if a.hypot(b) <= 1e-5 {
        // Achromatic: no hue direction to clip along. The only way to be
        // out of gamut at zero chroma is lightness past an end — the
        // anchor is the axis and the clamp lands exactly on it.
        return clip(oklab_to_linear_srgb([l.clamp(0.0, 1.0), 0.0, 0.0]));
    }
    let cusp = find_cusp(a_, b_);
    let l0 = match anchor {
        Anchor::Preserve => l.clamp(0.0, 1.0),
        Anchor::Cusp => cusp.0,
        Anchor::Adaptive => {
            // Smoothly blend the anchor towards the cusp as the colour
            // recedes — `gamut_clip_adaptive_L0_L_cusp`, alpha = 0.05.
            let ld = l - cusp.0;
            let k = 2.0 * if ld > 0.0 { 1.0 - cusp.0 } else { cusp.0 };
            let e1 = 0.05f64.mul_add(c / k, 0.5f64.mul_add(k, ld.abs()));
            let root = e1.mul_add(e1, -2.0 * k * ld.abs()).sqrt();
            (0.5 * ld.signum()).mul_add(e1 - root, cusp.0)
        }
    };
    let t = find_gamut_intersection(a_, b_, l, c, l0, cusp);
    let l_out = l0.mul_add(1.0 - t, t * l);
    let c_out = t * c;
    clip(oklab_to_linear_srgb([l_out, c_out * a_, c_out * b_]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `OKLab` → `OKLCh` `[L, C, h]` with `h` in radians.
    fn oklab_to_oklch([l, a, b]: [f64; 3]) -> [f64; 3] {
        [l, a.hypot(b), b.atan2(a)]
    }

    fn assert_close3(actual: [f64; 3], expected: [f64; 3], tol: f64) {
        for (a, e) in actual.into_iter().zip(expected) {
            assert!((a - e).abs() <= tol, "{a} != {e}");
        }
    }

    #[test]
    fn oklab_round_trip() {
        for rgb in [
            [0.0; 3],
            [1.0; 3],
            [0.25, 0.5, 0.75],
            [1.0, 0.0, 0.0],
            [0.02, 0.9, 0.3],
        ] {
            // The published OKLab matrices carry f32 precision — the
            // round trip's residual is ~1e-7, far under one JND (0.02).
            assert_close3(oklab_to_linear_srgb(linear_srgb_to_oklab(rgb)), rgb, 1e-6);
        }
    }

    #[test]
    fn oklab_white_is_one() {
        let [l, a, b] = linear_srgb_to_oklab([1.0, 1.0, 1.0]);
        assert!((l - 1.0).abs() < 1e-6);
        assert!(a.abs() < 1e-6 && b.abs() < 1e-6);
    }

    #[test]
    fn in_gamut_is_identity() {
        // The map returns an in-gamut colour bit-for-bit.
        let colors = [
            [0.0; 3],
            [1.0; 3],
            [0.25, 0.5, 0.75],
            [0.001, 0.999, 0.5],
            [0.5, 0.5, 0.5],
        ];
        for rgb in colors {
            assert_eq!(
                gamut_map_srgb_analytic(rgb).map(f64::to_bits),
                rgb.map(f64::to_bits)
            );
        }
    }

    #[test]
    fn out_of_gamut_lands_in_gamut() {
        // P3 red in linear sRGB has a negative green and blue and a red
        // above 1; the map lands inside [0, 1] at preserved hue.
        let out = [1.224_940_2, -0.224_940_2, 0.0];
        let mapped = gamut_map_srgb_analytic(out);
        assert!(in_gamut(mapped), "{mapped:?}");
        // Hue is preserved exactly by construction; the boundary
        let origin_h = oklab_to_oklch(linear_srgb_to_oklab(out))[2];
        let mapped_h = oklab_to_oklch(linear_srgb_to_oklab(mapped))[2];
        let dh = (mapped_h - origin_h)
            .abs()
            .rem_euclid(std::f64::consts::TAU);
        let dh = dh.min(std::f64::consts::TAU - dh);
        assert!(dh.to_degrees() < 15.0, "hue moved {} deg", dh.to_degrees());
    }

    #[test]
    fn hdr_lightness_maps_to_white_or_black() {
        // Lightness past the gamut's ends lands on the axis: 4x white
        // maps to white, a negative colour to black.
        assert_close3(gamut_map_srgb_analytic([4.0; 3]), [1.0; 3], 1e-3);
        assert_close3(gamut_map_srgb_analytic([-2.0; 3]), [0.0; 3], 1e-3);
    }

    #[test]
    fn p3_primaries_move_reasonably() {
        // P3 primaries (linear P3 coords -> linear sRGB, out of gamut)
        // should map to strongly saturated sRGB colours, not muddy ones.
        for p3 in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let srgb = crate::color::linear_p3_to_linear_srgb(p3);
            let mapped = gamut_map_srgb_analytic(srgb);
            assert!(in_gamut(mapped));
            let c = oklab_to_oklch(linear_srgb_to_oklab(mapped))[1];
            assert!(c > 0.1, "chroma collapsed: {mapped:?}");
        }
    }
}
