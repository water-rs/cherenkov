// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Paint evaluation, matching `oracle/src/paint.rs` in f32: the paint is
//! sampled at the pixel centre mapped by the item's inverse transform into
//! content space, then premultiplied.

use cherenkov::kurbo::Affine;
use cherenkov::{ColorStop, Extend, Interpolation, Paint};

use crate::error::Unsupported;

/// Linear Display P3 to linear sRGB, for `SrgbEncoded` gradient stops.
const P3_TO_SRGB: [[f32; 3]; 3] = [
    [1.224_940_1, -0.224_940_4, 0.0],
    [-0.042_056_9, 1.042_057_1, 0.0],
    [-0.019_637_6, -0.078_636_1, 1.098_273_5],
];

/// Linear sRGB to linear Display P3.
const SRGB_TO_P3: [[f32; 3]; 3] = [
    [0.822_461_96, 0.177_538_04, 0.0],
    [0.033_194_2, 0.966_805_8, 0.0],
    [0.017_082_632, 0.072_397_44, 0.910_519_96],
];

fn mat3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0].mul_add(v[0], m[0][1].mul_add(v[1], m[0][2] * v[2])),
        m[1][0].mul_add(v[0], m[1][1].mul_add(v[1], m[1][2] * v[2])),
        m[2][0].mul_add(v[0], m[2][1].mul_add(v[1], m[2][2] * v[2])),
    ]
}

/// sRGB-encodes one channel, preserving sign.
fn srgb_encode(x: f32) -> f32 {
    let e = if x.abs() <= 0.003_130_8 {
        x.abs() * 12.92
    } else {
        1.055f32.mul_add(x.abs().powf(1.0 / 2.4), -0.055)
    };
    e.copysign(x)
}

/// sRGB-decodes one channel, preserving sign.
fn srgb_decode(x: f32) -> f32 {
    let d = if x.abs() <= 0.04045 {
        x.abs() / 12.92
    } else {
        ((x.abs() + 0.055) / 1.055).powf(2.4)
    };
    d.copysign(x)
}

/// A gradient stop in interpolation-space straight-alpha components.
#[derive(Clone, Copy, Debug)]
pub struct Stop {
    /// The stop offset.
    pub offset: f32,
    /// Straight-alpha components in the interpolation space.
    pub color: [f32; 4],
}

/// A resolved paint: the inverse device-to-content transform plus the
/// parameters the evaluator needs.
#[derive(Clone, Debug)]
pub enum PaintData {
    /// Premultiplied solid colour.
    Solid([f32; 4]),
    /// A linear gradient.
    Linear {
        /// Device-to-content transform.
        inv: [f32; 6],
        /// Start and end points in content space.
        end_points: [f32; 4],
        /// Sorted stops.
        stops: Box<[Stop]>,
        /// The continuation mode.
        extend: Extend,
        /// The interpolation space.
        interpolation: Interpolation,
    },
    /// A two-point radial gradient.
    Radial {
        /// Device-to-content transform.
        inv: [f32; 6],
        /// Centres: `x0, y0, x1, y1` in content space.
        centres: [f32; 4],
        /// Radii: `r0, r1`.
        radii: [f32; 2],
        /// Sorted stops.
        stops: Box<[Stop]>,
        /// The continuation mode.
        extend: Extend,
        /// The interpolation space.
        interpolation: Interpolation,
    },
    /// An image pattern.
    Image {
        /// Device-to-image-pixel-space transform.
        inv: [f32; 6],
        /// The registered image.
        image: std::sync::Arc<crate::render::CpuImage>,
        /// Horizontal continuation.
        extend_x: Extend,
        /// Vertical continuation.
        extend_y: Extend,
        /// Sampling.
        sampling: cherenkov::Sampling,
    },
    /// A sweep (conic) gradient.
    Sweep {
        /// Device-to-content transform.
        inv: [f32; 6],
        /// The centre in content space.
        center: [f32; 2],
        /// Start angle in radians.
        start: f32,
        /// Angular span in radians, adjusted to `> 0` like the oracle.
        span: f32,
        /// Sorted stops.
        stops: Box<[Stop]>,
        /// The continuation mode.
        extend: Extend,
        /// The interpolation space.
        interpolation: Interpolation,
    },
}

/// An affine as six f32 coefficients `[a, b, c, d, e, f]`.
#[expect(clippy::cast_possible_truncation, reason = "geometry is f32")]
pub const fn affine_f32(t: Affine) -> [f32; 6] {
    let c = t.as_coeffs();
    [
        c[0] as f32,
        c[1] as f32,
        c[2] as f32,
        c[3] as f32,
        c[4] as f32,
        c[5] as f32,
    ]
}

/// Applies `[a, b, c, d, e, f]` to `(x, y)`.
const fn apply(m: [f32; 6], x: f32, y: f32) -> (f32, f32) {
    (
        m[0].mul_add(x, m[2].mul_add(y, m[4])),
        m[1].mul_add(x, m[3].mul_add(y, m[5])),
    )
}

/// f64 to f32.
#[expect(clippy::cast_possible_truncation)]
const fn f32_f64(v: f64) -> f32 {
    v as f32
}

/// Converts stops into interpolation-space straight-alpha components,
/// sorted by offset.
fn stops(stops: &[ColorStop], interpolation: Interpolation) -> Box<[Stop]> {
    let mut sorted = stops.to_vec();
    sorted.sort_by(|a, b| a.offset.total_cmp(&b.offset));
    sorted
        .iter()
        .map(|s| {
            let [r, g, b, a] = s.color.components;
            let color = if interpolation == Interpolation::SrgbEncoded {
                let lin = mat3(&P3_TO_SRGB, [r, g, b]);
                [
                    srgb_encode(lin[0]),
                    srgb_encode(lin[1]),
                    srgb_encode(lin[2]),
                    a,
                ]
            } else {
                [r, g, b, a]
            };
            Stop {
                offset: s.offset,
                color,
            }
        })
        .collect()
}

/// Samples an image at pixel-space `(u, v)` — the oracle's
/// `sample_image` in f32 (centre-based: texel centres at integer + 0.5).
#[expect(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "u/v/w/h/x/y are the natural names; texel coordinates are clamped before indexing"
)]
fn sample_image(
    img: &crate::render::CpuImage,
    u: f32,
    v: f32,
    sampling: cherenkov::Sampling,
) -> [f32; 4] {
    let (w, h) = (img.width as usize, img.height as usize);
    let at = |x: usize, y: usize| img.pixels[y.min(h - 1) * w + x.min(w - 1)];
    match sampling {
        cherenkov::Sampling::Nearest => {
            let x = (u - 0.5).round().clamp(0.0, w as f32 - 1.0) as usize;
            let y = (v - 0.5).round().clamp(0.0, h as f32 - 1.0) as usize;
            at(x, y)
        }
        cherenkov::Sampling::Linear => {
            // Clamp the *sample coordinate* into texel-centre space before
            // taking the fraction: outside the border texels every tap must
            // collapse onto the edge texel, not blend inward with a flipped
            // weight.
            let fx = (u - 0.5).clamp(0.0, w as f32 - 1.0);
            let fy = (v - 0.5).clamp(0.0, h as f32 - 1.0);
            let x0 = fx.floor() as usize;
            let y0 = fy.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            let tx = fx - x0 as f32;
            let ty = fy - y0 as f32;
            let (c00, c10, c01, c11) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
            let mut out = [0.0; 4];
            for i in 0..4 {
                let top = c00[i] + tx * (c10[i] - c00[i]);
                let bot = c01[i] + tx * (c11[i] - c01[i]);
                out[i] = top + ty * (bot - top);
            }
            out
        }
    }
}

/// Lowers a `Paint` into [`PaintData`]. `inv` maps device space into the
/// content space the gradient parameters live in.
#[expect(
    clippy::many_single_char_names,
    clippy::while_float,
    reason = "r/g/b/a are the channel names; the sweep span wrap loop is the clearest form"
)]
pub fn paint_data(
    paint: &Paint,
    inv: Affine,
    images: &std::collections::HashMap<u64, std::sync::Arc<crate::render::CpuImage>>,
) -> Result<PaintData, crate::error::RenderError> {
    Ok(match paint {
        Paint::Solid(c) => {
            let [r, g, b, a] = c.components;
            PaintData::Solid([r * a, g * a, b * a, a])
        }
        Paint::Linear(g) => PaintData::Linear {
            inv: affine_f32(inv),
            end_points: [
                f32_f64(g.start.x),
                f32_f64(g.start.y),
                f32_f64(g.end.x),
                f32_f64(g.end.y),
            ],
            stops: stops(&g.stops, g.interpolation),
            extend: g.extend,
            interpolation: g.interpolation,
        },
        Paint::Radial(g) => PaintData::Radial {
            inv: affine_f32(inv),
            centres: [
                f32_f64(g.start_center.x),
                f32_f64(g.start_center.y),
                f32_f64(g.end_center.x),
                f32_f64(g.end_center.y),
            ],
            radii: [f32_f64(g.start_radius), f32_f64(g.end_radius)],
            stops: stops(&g.stops, g.interpolation),
            extend: g.extend,
            interpolation: g.interpolation,
        },
        Paint::Sweep(g) => {
            let mut span = f32_f64(g.end_angle) - f32_f64(g.start_angle);
            while span <= 0.0 {
                span += std::f32::consts::TAU;
            }
            PaintData::Sweep {
                inv: affine_f32(inv),
                center: [f32_f64(g.center.x), f32_f64(g.center.y)],
                start: f32_f64(g.start_angle),
                span,
                stops: stops(&g.stops, g.interpolation),
                extend: g.extend,
                interpolation: g.interpolation,
            }
        }
        Paint::Mesh(_) => return Err(Unsupported::Mesh.into()),
        Paint::Image(p) => {
            let Some(image) = images.get(&p.image.raw()) else {
                return Err(crate::error::RenderError::Image(p.image.raw()));
            };
            PaintData::Image {
                inv: affine_f32(p.transform.inverse() * inv),
                image: std::sync::Arc::clone(image),
                extend_x: p.extend_x,
                extend_y: p.extend_y,
                sampling: p.sampling,
            }
        }
        Paint::Shader(_) => return Err(Unsupported::Shader.into()),
    })
}

/// Applies `extend` to `t`.
fn extend_t(t: f32, extend: Extend) -> Option<f32> {
    match extend {
        Extend::Pad => Some(t.clamp(0.0, 1.0)),
        Extend::Repeat => Some(t - t.floor()),
        Extend::Reflect => {
            let m = (t * 0.5).floor().mul_add(-2.0, t);
            Some(if m > 1.0 { 2.0 - m } else { m })
        }
        Extend::None => (0.0..=1.0).contains(&t).then_some(t),
    }
}

/// Premultiplies interpolation-space straight components into linear P3.
#[expect(
    clippy::many_single_char_names,
    reason = "r/g/b/a are the channel names"
)]
fn to_premul(c: [f32; 4], interpolation: Interpolation) -> [f32; 4] {
    let [r, g, b, a] = c;
    let [pr, pg, pb] = if interpolation == Interpolation::SrgbEncoded {
        mat3(
            &SRGB_TO_P3,
            [srgb_decode(r), srgb_decode(g), srgb_decode(b)],
        )
    } else {
        [r, g, b]
    };
    [pr * a, pg * a, pb * a, a]
}

/// Evaluates `stops` at `t`, returning premultiplied linear P3.
fn eval_stops(stops: &[Stop], t: f32, interpolation: Interpolation) -> [f32; 4] {
    let Some(first) = stops.first() else {
        return [0.0; 4];
    };
    if stops.len() == 1 || t <= first.offset {
        return to_premul(first.color, interpolation);
    }
    let last = &stops[stops.len() - 1];
    if t >= last.offset {
        return to_premul(last.color, interpolation);
    }
    for w in stops.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if t >= a.offset && t <= b.offset {
            let span = b.offset - a.offset;
            let f = if span > 0.0 {
                (t - a.offset) / span
            } else {
                0.0
            };
            let (ca, cb) = (a.color, b.color);
            let mixed = [
                f.mul_add(cb[0] - ca[0], ca[0]),
                f.mul_add(cb[1] - ca[1], ca[1]),
                f.mul_add(cb[2] - ca[2], ca[2]),
                f.mul_add(cb[3] - ca[3], ca[3]),
            ];
            return to_premul(mixed, interpolation);
        }
    }
    to_premul(last.color, interpolation)
}

/// `t` along the two-point radial gradient for point `(px, py)` in content
/// space; the larger real root, `NaN` when there is none.
#[allow(clippy::many_single_char_names)] // quadratic notation mirrors the spec
fn radial_t(px: f32, py: f32, centres: [f32; 4], radii: [f32; 2]) -> f32 {
    let (px, py) = (px - centres[0], py - centres[1]);
    let (dcx, dcy) = (centres[2] - centres[0], centres[3] - centres[1]);
    let (r0, dr) = (radii[0], radii[1] - radii[0]);
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
        return f32::NAN;
    }
    let sq = disc.sqrt();
    let (r1, r2) = ((-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a));
    r1.max(r2)
}

impl PaintData {
    /// Evaluates the paint at device-space pixel centre `(dx, dy)`,
    /// returning premultiplied linear Display P3.
    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions are far below 2^24"
    )]
    pub fn eval(&self, dx: f32, dy: f32) -> [f32; 4] {
        match self {
            Self::Solid(c) => *c,
            Self::Image {
                inv,
                image,
                extend_x,
                extend_y,
                sampling,
            } => {
                let (qx, qy) = apply(*inv, dx, dy);
                let (iw, ih) = (image.width as f32, image.height as f32);
                let Some(u) = extend_t(qx / iw, *extend_x).map(|t| t * iw) else {
                    return [0.0; 4];
                };
                let Some(v) = extend_t(qy / ih, *extend_y).map(|t| t * ih) else {
                    return [0.0; 4];
                };
                sample_image(image, u, v, *sampling)
            }
            Self::Linear {
                inv,
                end_points,
                stops,
                extend,
                interpolation,
            } => {
                let (px, py) = apply(*inv, dx, dy);
                let (sx, sy, ex, ey) = (end_points[0], end_points[1], end_points[2], end_points[3]);
                let (ddx, ddy) = (ex - sx, ey - sy);
                let len2 = ddy.mul_add(ddy, ddx * ddx);
                let t = if len2 == 0.0 {
                    0.0
                } else {
                    (py - sy).mul_add(ddy, (px - sx) * ddx) / len2
                };
                extend_t(t, *extend).map_or([0.0; 4], |t| eval_stops(stops, t, *interpolation))
            }
            Self::Radial {
                inv,
                centres,
                radii,
                stops,
                extend,
                interpolation,
            } => {
                let (px, py) = apply(*inv, dx, dy);
                let t = radial_t(px, py, *centres, *radii);
                if t.is_finite() {
                    extend_t(t, *extend).map_or([0.0; 4], |t| eval_stops(stops, t, *interpolation))
                } else {
                    [0.0; 4]
                }
            }
            Self::Sweep {
                inv,
                center,
                start,
                span,
                stops,
                extend,
                interpolation,
            } => {
                let (px, py) = apply(*inv, dx, dy);
                let raw = (py - center[1]).atan2(px - center[0]) - *start;
                let theta = *start + raw.rem_euclid(std::f32::consts::TAU);
                let t = (theta - *start) / *span;
                extend_t(t, *extend).map_or([0.0; 4], |t| eval_stops(stops, t, *interpolation))
            }
        }
    }
}
