// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Colour conversion into the working space, linear Display P3.
//!
//! Transfer curves: sRGB and Display P3 share the sRGB companding. Rec. 2020
//! scene colours are treated as linear-light (the suite's HDR authoring
//! convention; the Rec. 2020 primaries still map into P3 chromatically).

use cherenkov_scene::{Color, ColorSpace};

/// Multiply a 3×3 row-major matrix by a column vector.
pub(crate) fn mat3_mul(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][2].mul_add(v[2], m[0][1].mul_add(v[1], m[0][0] * v[0])),
        m[1][2].mul_add(v[2], m[1][1].mul_add(v[1], m[1][0] * v[0])),
        m[2][2].mul_add(v[2], m[2][1].mul_add(v[1], m[2][0] * v[0])),
    ]
}

/// Linear sRGB (BT.709 primaries, D65) to CIE XYZ.
pub(crate) const SRGB_TO_XYZ: [[f64; 3]; 3] = [
    [
        0.412_390_799_265_959_4,
        0.357_584_339_383_878,
        0.180_480_788_401_834_3,
    ],
    [
        0.212_639_005_871_510_4,
        0.715_168_678_767_756,
        0.072_192_315_360_733_7,
    ],
    [
        0.019_330_818_715_591_8,
        0.119_194_779_794_626,
        0.950_532_152_249_660_7,
    ],
];

/// Linear Display P3 (D65) to CIE XYZ.
pub(crate) const P3_TO_XYZ: [[f64; 3]; 3] = [
    [
        0.486_570_948_648_216_2,
        0.265_667_693_169_093_1,
        0.198_217_285_234_362_5,
    ],
    [
        0.228_974_564_069_748_8,
        0.691_738_521_836_506_2,
        0.079_286_914_093_745_0,
    ],
    [0.0, 0.045_113_381_858_902_6, 1.043_944_368_900_976],
];

/// Linear Rec. 2020 (D65) to CIE XYZ.
pub(crate) const REC2020_TO_XYZ: [[f64; 3]; 3] = [
    [
        0.636_958_048_301_291_4,
        0.144_616_903_586_208_3,
        0.168_880_975_164_172,
    ],
    [
        0.262_700_212_011_267_1,
        0.677_998_071_518_870_8,
        0.059_301_716_469_861_9,
    ],
    [0.0, 0.028_072_693_049_087_4, 1.060_985_057_710_791],
];

/// Invert a 3×3 matrix.
pub(crate) fn mat3_inv(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][2].mul_add(
        m[1][1].mul_add(-m[2][0], m[1][0] * m[2][1]),
        m[0][1].mul_add(
            -m[1][2].mul_add(-m[2][0], m[1][0] * m[2][2]),
            m[0][0] * m[1][2].mul_add(-m[2][1], m[1][1] * m[2][2]),
        ),
    );
    let inv_det = 1.0 / det;
    let mut inv = [[0.0; 3]; 3];
    inv[0][0] = m[1][2].mul_add(-m[2][1], m[1][1] * m[2][2]) * inv_det;
    inv[0][1] = m[0][1].mul_add(-m[2][2], m[0][2] * m[2][1]) * inv_det;
    inv[0][2] = m[0][2].mul_add(-m[1][1], m[0][1] * m[1][2]) * inv_det;
    inv[1][0] = m[1][0].mul_add(-m[2][2], m[1][2] * m[2][0]) * inv_det;
    inv[1][1] = m[0][2].mul_add(-m[2][0], m[0][0] * m[2][2]) * inv_det;
    inv[1][2] = m[0][0].mul_add(-m[1][2], m[0][2] * m[1][0]) * inv_det;
    inv[2][0] = m[1][1].mul_add(-m[2][0], m[1][0] * m[2][1]) * inv_det;
    inv[2][1] = m[0][0].mul_add(-m[2][1], m[0][1] * m[2][0]) * inv_det;
    inv[2][2] = m[0][1].mul_add(-m[1][0], m[0][0] * m[1][1]) * inv_det;
    inv
}

type Mat3 = [[f64; 3]; 3];

/// Precomputed `linear sRGB -> linear P3` and `Rec. 2020 -> linear P3`
/// matrices.
fn build_matrices() -> (Mat3, Mat3, Mat3) {
    let p3_inv = mat3_inv(&P3_TO_XYZ);
    let mm = |a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]| -> [[f64; 3]; 3] {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = a[i][2].mul_add(b[2][j], a[i][1].mul_add(b[1][j], a[i][0] * b[0][j]));
            }
        }
        out
    };
    (
        mm(&p3_inv, &SRGB_TO_XYZ),
        mm(&p3_inv, &REC2020_TO_XYZ),
        mat3_inv(&SRGB_TO_XYZ),
    )
}

/// sRGB companding: encoded -> linear.
pub(crate) fn srgb_decode(c: f64) -> f64 {
    if c.abs() <= 0.04045 {
        c / 12.92
    } else {
        ((c.abs() + 0.055) / 1.055).powf(2.4).copysign(c)
    }
}

/// sRGB companding: linear -> encoded.
#[must_use]
pub fn srgb_encode(c: f64) -> f64 {
    if c.abs() <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055f64
            .mul_add(c.abs().powf(1.0 / 2.4), -0.055)
            .copysign(c)
    }
}

/// Convert one channel of a scene colour to linear light for its declared
/// colour space's transfer function.
fn to_linear_channel(space: ColorSpace, c: f64) -> f64 {
    match space {
        ColorSpace::Srgb | ColorSpace::DisplayP3 => srgb_decode(c),
        ColorSpace::LinearSrgb | ColorSpace::LinearP3 | ColorSpace::Rec2020 => c,
    }
}

/// Convert a scene colour to premultiplied linear Display P3 `[r, g, b, a]`.
///
/// The three colour channels are converted to linear light in their declared
/// primaries, mapped into P3 primaries, then premultiplied by alpha.
#[must_use]
pub fn to_working(color: &Color) -> [f64; 4] {
    let [r, g, b, a] = color.components;
    let a = f64::from(a);
    let linear = [
        to_linear_channel(color.space, f64::from(r)),
        to_linear_channel(color.space, f64::from(g)),
        to_linear_channel(color.space, f64::from(b)),
    ];
    let (srgb_to_p3, rec2020_to_p3, _) = build_matrices();
    let p3 = match color.space {
        ColorSpace::LinearP3 | ColorSpace::DisplayP3 => linear,
        ColorSpace::Srgb | ColorSpace::LinearSrgb => mat3_mul(&srgb_to_p3, linear),
        ColorSpace::Rec2020 => mat3_mul(&rec2020_to_p3, linear),
    };
    [p3[0] * a, p3[1] * a, p3[2] * a, a]
}

/// Convert a linear Display P3 colour to linear sRGB (for sRGB-encoded
/// output such as PNG). Values are not clamped.
#[must_use]
pub fn linear_p3_to_linear_srgb(p3: [f64; 3]) -> [f64; 3] {
    let (_, _, srgb_inv) = build_matrices();
    let xyz = mat3_mul(&P3_TO_XYZ, p3);
    mat3_mul(&srgb_inv, xyz)
}

/// Convert a linear sRGB colour to linear Display P3 (for engines whose
/// output is decoded to sRGB primaries). Values are not clamped.
#[must_use]
pub fn linear_srgb_to_linear_p3(srgb: [f64; 3]) -> [f64; 3] {
    let (srgb_to_p3, _, _) = build_matrices();
    mat3_mul(&srgb_to_p3, srgb)
}
