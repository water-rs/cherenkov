// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! W3C Compositing and Blending Level 1 formulas, ported from the
//! oracle's `blend.rs` into `f32`.
//!
//! Blending is done on un-premultiplied channels per the spec:
//! `Cr = (1 - αb)·Cs + αb·B(Cb, Cs)` where `B` is the blend function, then
//! the blended colour `Cr` is composited source-over with source alpha
//! `αs` — in premultiplied terms `Co = αs·Cr + (1 - αs)·Cb'`.

use cherenkov::BlendMode;

fn lum(c: [f32; 3]) -> f32 {
    0.11f32.mul_add(c[2], 0.59f32.mul_add(c[1], 0.3 * c[0]))
}

fn sat(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

fn clip_color(c: [f32; 3]) -> [f32; 3] {
    let l = lum(c);
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    let mut c = c;
    if n < 0.0 {
        for ch in &mut c {
            *ch = l + (*ch - l) * l / (l - n);
        }
    }
    if x > 1.0 {
        for ch in &mut c {
            *ch = l + (*ch - l) * (1.0 - l) / (x - l);
        }
    }
    c
}

fn set_lum(c: [f32; 3], l: f32) -> [f32; 3] {
    let d = l - lum(c);
    clip_color([c[0] + d, c[1] + d, c[2] + d])
}

#[expect(clippy::similar_names, reason = "names mirror the W3C spec")]
fn set_sat(c: [f32; 3], s: f32) -> [f32; 3] {
    let (imin, imax) = {
        let mut mn = 0;
        let mut mx = 0;
        for i in 1..3 {
            if c[i] < c[mn] {
                mn = i;
            }
            if c[i] > c[mx] {
                mx = i;
            }
        }
        (mn, mx)
    };
    let imid = 3 - imin - imax;
    let mut out = [0.0; 3];
    if c[imax] > c[imin] {
        out[imid] = (c[imid] - c[imin]) * s / (c[imax] - c[imin]);
        out[imax] = s;
    }
    out
}

/// The blend function `B(Cb, Cs)` for one channel pair, separable modes.
fn blend_channel(mode: BlendMode, cb: f32, cs: f32) -> f32 {
    match mode {
        BlendMode::Multiply => cb * cs,
        BlendMode::Screen => cb.mul_add(-cs, cb + cs),
        BlendMode::Overlay => {
            if cb <= 0.5 {
                2.0 * cb * cs
            } else {
                (2.0 * (1.0 - cb)).mul_add(-(1.0 - cs), 1.0)
            }
        }
        BlendMode::Darken => cb.min(cs),
        BlendMode::Lighten => cb.max(cs),
        BlendMode::ColorDodge => {
            if cs >= 1.0 {
                1.0
            } else {
                (cb / (1.0 - cs)).min(1.0)
            }
        }
        BlendMode::ColorBurn => {
            if cs <= 0.0 {
                0.0
            } else {
                1.0 - ((1.0 - cb) / cs).min(1.0)
            }
        }
        BlendMode::HardLight => {
            if cs <= 0.5 {
                2.0 * cb * cs
            } else {
                (2.0 * (1.0 - cb)).mul_add(-(1.0 - cs), 1.0)
            }
        }
        BlendMode::SoftLight => {
            if cs <= 0.5 {
                (2.0f32.mul_add(-cs, 1.0) * cb).mul_add(-(1.0 - cb), cb)
            } else {
                let d = if cb <= 0.25 {
                    16.0f32.mul_add(cb, -12.0).mul_add(cb, 4.0) * cb
                } else {
                    cb.sqrt()
                };
                2.0f32.mul_add(cs, -1.0).mul_add(d - cb, cb)
            }
        }
        BlendMode::Difference => (cb - cs).abs(),
        BlendMode::Exclusion => (2.0 * cb).mul_add(-cs, cb + cs),
        // `Normal` and the non-separable modes (evaluated per-pixel in
        // `blend`, not here) pass the source channel through.
        _ => cs,
    }
}

/// Blend premultiplied source `cs` onto premultiplied backdrop `cb`,
/// returning the composited premultiplied `co` directly.
///
/// `Cs' = (1-αb)·Cs + αb·B(Cb, Cs)` and `co = αs·Cs' + αb·Cb·(1-αs)` —
/// so a fully transparent source leaves the backdrop unchanged, and
/// [`BlendMode::Normal`] reduces to [`src_over`].
#[must_use]
pub fn blend(mode: BlendMode, cb: [f32; 4], cs: [f32; 4]) -> [f32; 4] {
    let (ab, as_) = (cb[3], cs[3]);
    // Porter-Duff compositing operators (COLRv1 `PaintComposite`): no
    // colour blending, `co = αs·Fa·Cs + αb·Fb·Cb` in premultiplied form.
    let porter_duff = |fa: f32, fb: f32| -> [f32; 4] {
        [
            fa.mul_add(cs[0], fb * cb[0]),
            fa.mul_add(cs[1], fb * cb[1]),
            fa.mul_add(cs[2], fb * cb[2]),
            fa.mul_add(as_, fb * ab),
        ]
    };
    match mode {
        BlendMode::Clear => return [0.0; 4],
        BlendMode::Src => return cs,
        BlendMode::Dst => return cb,
        BlendMode::DestOver => return porter_duff(1.0 - ab, 1.0),
        BlendMode::SrcIn => return porter_duff(ab, 0.0),
        BlendMode::DestIn => return porter_duff(0.0, as_),
        BlendMode::SrcOut => return porter_duff(1.0 - ab, 0.0),
        BlendMode::DestOut => return porter_duff(0.0, 1.0 - as_),
        BlendMode::SrcAtop => return porter_duff(ab, 1.0 - as_),
        BlendMode::DestAtop => return porter_duff(1.0 - ab, as_),
        BlendMode::Xor => return porter_duff(1.0 - ab, 1.0 - as_),
        BlendMode::PlusLighter => return porter_duff(1.0, 1.0),
        _ => {}
    }
    if as_ == 0.0 {
        return cb;
    }
    // Un-premultiply (clamped to [0,1]; components are finite by construction).
    let ub = if ab > 0.0 {
        [cb[0] / ab, cb[1] / ab, cb[2] / ab]
    } else {
        [0.0; 3]
    };
    let us = [cs[0] / as_, cs[1] / as_, cs[2] / as_];
    let b: [f32; 3] = match mode {
        BlendMode::Hue => set_lum(set_sat(us, sat(ub)), lum(ub)),
        BlendMode::Saturation => set_lum(set_sat(ub, sat(us)), lum(ub)),
        BlendMode::Color => set_lum(us, lum(ub)),
        BlendMode::Luminosity => set_lum(ub, lum(us)),
        _ => [
            blend_channel(mode, ub[0], us[0]),
            blend_channel(mode, ub[1], us[1]),
            blend_channel(mode, ub[2], us[2]),
        ],
    };
    // Cr = (1-αb)·Cs + αb·B(Cb,Cs); premultiplied output: αs·Cr + (1-αs)·Cb.
    let mut out = [0.0; 4];
    for i in 0..3 {
        let cr = ab.mul_add(b[i], (1.0 - ab) * us[i]);
        out[i] = (1.0 - as_).mul_add(cb[i], as_ * cr);
    }
    out[3] = ab.mul_add(1.0 - as_, as_);
    out
}

/// Composite premultiplied `src` over premultiplied `dst` (source-over).
/// `src` is expected already blended for non-normal blends.
#[must_use]
pub fn src_over(dst: [f32; 4], src: [f32; 4]) -> [f32; 4] {
    [
        dst[0].mul_add(1.0 - src[3], src[0]),
        dst[1].mul_add(1.0 - src[3], src[1]),
        dst[2].mul_add(1.0 - src[3], src[2]),
        dst[3].mul_add(1.0 - src[3], src[3]),
    ]
}

#[cfg(test)]
mod tests {
    use cherenkov_scene::BlendMode as SceneBlend;

    use super::*;

    /// The scene blend mode matching a front-end mode by name.
    const fn scene_mode(m: BlendMode) -> SceneBlend {
        match m {
            BlendMode::Normal => SceneBlend::Normal,
            BlendMode::Multiply => SceneBlend::Multiply,
            BlendMode::Screen => SceneBlend::Screen,
            BlendMode::Overlay => SceneBlend::Overlay,
            BlendMode::Darken => SceneBlend::Darken,
            BlendMode::Lighten => SceneBlend::Lighten,
            BlendMode::ColorDodge => SceneBlend::ColorDodge,
            BlendMode::ColorBurn => SceneBlend::ColorBurn,
            BlendMode::HardLight => SceneBlend::HardLight,
            BlendMode::SoftLight => SceneBlend::SoftLight,
            BlendMode::Difference => SceneBlend::Difference,
            BlendMode::Exclusion => SceneBlend::Exclusion,
            BlendMode::Hue => SceneBlend::Hue,
            BlendMode::Saturation => SceneBlend::Saturation,
            BlendMode::Color => SceneBlend::Color,
            BlendMode::Luminosity => SceneBlend::Luminosity,
            BlendMode::Clear => SceneBlend::Clear,
            BlendMode::Src => SceneBlend::Src,
            BlendMode::Dst => SceneBlend::Dst,
            BlendMode::DestOver => SceneBlend::DestOver,
            BlendMode::SrcIn => SceneBlend::SrcIn,
            BlendMode::DestIn => SceneBlend::DestIn,
            BlendMode::SrcOut => SceneBlend::SrcOut,
            BlendMode::DestOut => SceneBlend::DestOut,
            BlendMode::SrcAtop => SceneBlend::SrcAtop,
            BlendMode::DestAtop => SceneBlend::DestAtop,
            BlendMode::Xor => SceneBlend::Xor,
            BlendMode::PlusLighter => SceneBlend::PlusLighter,
        }
    }

    #[test]
    fn blend_matches_the_oracle() {
        let modes = [
            BlendMode::Normal,
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Overlay,
            BlendMode::Darken,
            BlendMode::Lighten,
            BlendMode::ColorDodge,
            BlendMode::ColorBurn,
            BlendMode::HardLight,
            BlendMode::SoftLight,
            BlendMode::Difference,
            BlendMode::Exclusion,
            BlendMode::Hue,
            BlendMode::Saturation,
            BlendMode::Color,
            BlendMode::Luminosity,
            BlendMode::Clear,
            BlendMode::Src,
            BlendMode::Dst,
            BlendMode::DestOver,
            BlendMode::SrcIn,
            BlendMode::DestIn,
            BlendMode::SrcOut,
            BlendMode::DestOut,
            BlendMode::SrcAtop,
            BlendMode::DestAtop,
            BlendMode::Xor,
            BlendMode::PlusLighter,
        ];
        // Pseudo-random premultiplied pairs: alpha from {0, 0.5, 1} and
        // channels from a small LCG, components kept <= alpha.
        let mut seed = 0x9e3779b9u32;
        let mut next = move || {
            seed = seed.wrapping_mul(747796405).wrapping_add(2891336453);
            (seed >> 24) as f32 / 255.0
        };
        let mut cases = Vec::new();
        for i in 0..50u32 {
            let ab = [0.0, 0.5, 1.0][i as usize % 3];
            let as_ = [0.0, 0.3, 1.0][(i as usize / 3) % 3];
            let cb = [next() * ab, next() * ab, next() * ab, ab];
            let cs = [next() * as_, next() * as_, next() * as_, as_];
            cases.push((cb, cs));
        }
        for mode in modes {
            for (cb, cs) in &cases {
                let got = blend(mode, *cb, *cs);
                let want = cherenkov_oracle::blend::blend(
                    scene_mode(mode),
                    cb.map(f64::from),
                    cs.map(f64::from),
                );
                for c in 0..4 {
                    assert!(
                        (f64::from(got[c]) - want[c]).abs() < 1e-5,
                        "{mode:?} {cb:?} {cs:?}: got {got:?} want {want:?}"
                    );
                }
            }
        }
    }
}
