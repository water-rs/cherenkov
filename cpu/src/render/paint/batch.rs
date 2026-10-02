//! Structure-of-arrays paint kernels. Irregular lookups gather four lanes;
//! affine mapping, interpolation and premultiplication operate on vectors.

use wide::f32x4;

use super::{ImagePaintData, PaintData, SRGB_TO_P3, Stop, affine_f32, extend_t, srgb_decode};
use cherenkov::{Extend, Interpolation, Sampling};

type Channels = [f32x4; 4];

fn gather(pixels: [[f32; 4]; 4]) -> Channels {
    f32x4::transpose(pixels.map(f32x4::from))
}

fn apply(matrix: [f32; 6], x: f32x4, y: f32x4) -> (f32x4, f32x4) {
    let matrix = matrix.map(f32x4::splat);
    (
        matrix[0].mul_add(x, matrix[2].mul_add(y, matrix[4])),
        matrix[1].mul_add(x, matrix[3].mul_add(y, matrix[5])),
    )
}

fn premultiply(mut color: Channels, interpolation: Interpolation) -> Channels {
    if interpolation == Interpolation::SrgbEncoded {
        let decoded: [f32x4; 3] =
            std::array::from_fn(|channel| f32x4::from(color[channel].to_array().map(srgb_decode)));
        for channel in 0..3 {
            let matrix = SRGB_TO_P3[channel].map(f32x4::splat);
            color[channel] = matrix[0].mul_add(
                decoded[0],
                matrix[1].mul_add(decoded[1], matrix[2] * decoded[2]),
            );
        }
    }
    [
        color[0] * color[3],
        color[1] * color[3],
        color[2] * color[3],
        color[3],
    ]
}

fn gradient(
    stops: &[Stop],
    values: f32x4,
    extend: Extend,
    interpolation: Interpolation,
) -> Channels {
    let mut first = [[0.0; 4]; 4];
    let mut last = first;
    let mut fractions = [0.0; 4];
    for (lane, value) in values.to_array().into_iter().enumerate() {
        let Some(value) = extend_t(value, extend) else {
            continue;
        };
        let Some(stop) = stops.first() else { continue };
        if stops.len() == 1 || value <= stop.offset {
            first[lane] = stop.color;
            last[lane] = stop.color;
            continue;
        }
        let end = stops.last().expect("nonempty gradient");
        // The scalar stop evaluator selects the final stop when a repeating
        // or reflecting parameter overflows and its reduction produces NaN.
        if value >= end.offset || value.is_nan() {
            first[lane] = end.color;
            last[lane] = end.color;
            continue;
        }
        let pair = stops
            .windows(2)
            .find(|pair| value >= pair[0].offset && value <= pair[1].offset)
            .expect("sorted stops enclose interior parameter");
        first[lane] = pair[0].color;
        last[lane] = pair[1].color;
        let width = pair[1].offset - pair[0].offset;
        fractions[lane] = if width > 0.0 {
            (value - pair[0].offset) / width
        } else {
            0.0
        };
    }
    let first = gather(first);
    let last = gather(last);
    let fraction = f32x4::from(fractions);
    premultiply(
        std::array::from_fn(|channel| {
            fraction.mul_add(last[channel] - first[channel], first[channel])
        }),
        interpolation,
    )
}

#[expect(
    clippy::many_single_char_names,
    reason = "quadratic coefficients mirror scalar radial evaluation"
)]
fn radial(x: f32x4, y: f32x4, centres: [f32; 4], radii: [f32; 2]) -> f32x4 {
    let px = x - centres[0];
    let py = y - centres[1];
    let dcx = centres[2] - centres[0];
    let dcy = centres[3] - centres[1];
    let r0 = radii[0];
    let dr = radii[1] - radii[0];
    let a = dr.mul_add(-dr, dcy.mul_add(dcy, dcx * dcx));
    let b = f32x4::splat(-2.0)
        * f32x4::splat(r0).mul_add(f32x4::splat(dr), f32x4::splat(dcy).mul_add(py, px * dcx));
    let c = f32x4::splat(r0).mul_add(f32x4::splat(-r0), py.mul_add(py, px * px));
    if a.abs() < 1e-12 {
        // hypot has explicit scaling and rounding semantics; only degenerate
        // conics need that operation, so evaluate it lane-wise.
        let origin = if r0.abs() < 1e-12 {
            f32x4::ZERO
        } else {
            let px = px.to_array();
            let py = py.to_array();
            f32x4::from(std::array::from_fn(|lane| {
                (px[lane].hypot(py[lane]) - r0) / r0.abs()
            }))
        };
        return b.abs().simd_lt(f32x4::splat(1e-12)).select(origin, -c / b);
    }
    let discriminant = f32x4::splat(4.0 * a).mul_add(-c, b * b);
    let root = discriminant.sqrt();
    let upper = (-b + root) / (2.0 * a);
    let lower = (-b - root) / (2.0 * a);
    discriminant
        .simd_lt(f32x4::ZERO)
        .select(f32x4::splat(f32::NAN), upper.max(lower))
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "image dimensions are memory bounded; indices are clamped to texel bounds"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "texel-centre subtraction follows scalar sampling's separately rounded coordinate scaling"
)]
fn image(data: &ImagePaintData, x: f32x4, y: f32x4) -> Channels {
    let (x, y) = apply(data.inv, x, y);
    let (width, height) = (data.image.width as usize, data.image.height as usize);
    let xs = (x / width as f32).to_array();
    let ys = (y / height as f32).to_array();
    let mut taps = [[[0.0; 4]; 4]; 4];
    let mut tx = [0.0; 4];
    let mut ty = [0.0; 4];
    for lane in 0..4 {
        let (Some(u), Some(v)) = (
            extend_t(xs[lane], data.extend_x),
            extend_t(ys[lane], data.extend_y),
        ) else {
            continue;
        };
        let x = u * width as f32 - 0.5;
        let y = v * height as f32 - 0.5;
        if data.sampling == Sampling::Nearest {
            let x = x.round().clamp(0.0, width as f32 - 1.0) as usize;
            let y = y.round().clamp(0.0, height as f32 - 1.0) as usize;
            taps[0][lane] = data.image.pixels[y * width + x];
        } else {
            let x = x.clamp(0.0, width as f32 - 1.0);
            let y = y.clamp(0.0, height as f32 - 1.0);
            let x0 = x.floor() as usize;
            let y0 = y.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);
            tx[lane] = x - x0 as f32;
            ty[lane] = y - y0 as f32;
            for (tap, index) in [
                y0 * width + x0,
                y0 * width + x1,
                y1 * width + x0,
                y1 * width + x1,
            ]
            .into_iter()
            .enumerate()
            {
                taps[tap][lane] = data.image.pixels[index];
            }
        }
    }
    if data.sampling == Sampling::Nearest {
        return gather(taps[0]);
    }
    let [c00, c10, c01, c11] = taps.map(gather);
    let tx = f32x4::from(tx);
    let ty = f32x4::from(ty);
    std::array::from_fn(|channel| {
        let top = tx.mul_add(c10[channel] - c00[channel], c00[channel]);
        let bottom = tx.mul_add(c11[channel] - c01[channel], c01[channel]);
        ty.mul_add(bottom - top, top)
    })
}

impl PaintData {
    /// Four device pixel centres, returned as premultiplied P3 channels.
    pub fn eval4(&self, x: f32x4, y: f32x4) -> Channels {
        match self {
            Self::Solid(color) => color.map(f32x4::splat),
            Self::Transformed(inner, inverse) => {
                let (x, y) = apply(affine_f32(*inverse), x, y);
                inner.eval4(x, y)
            }
            Self::Mesh(mesh) => {
                let x = x.to_array();
                let y = y.to_array();
                gather(std::array::from_fn(|lane| mesh.eval(x[lane], y[lane])))
            }
            Self::Image(data) => image(data, x, y),
            Self::Linear {
                inv,
                end_points,
                stops,
                extend,
                interpolation,
            } => {
                let (x, y) = apply(*inv, x, y);
                let [sx, sy, ex, ey] = *end_points;
                let (dx, dy) = (ex - sx, ey - sy);
                let length = dy.mul_add(dy, dx * dx);
                let parameter = if length.to_bits() == 0 {
                    f32x4::ZERO
                } else {
                    (y - sy).mul_add(f32x4::splat(dy), (x - sx) * dx) / length
                };
                gradient(stops, parameter, *extend, *interpolation)
            }
            Self::Radial {
                inv,
                centres,
                radii,
                stops,
                extend,
                interpolation,
            } => {
                let (x, y) = apply(*inv, x, y);
                let parameter = radial(x, y, *centres, *radii);
                gradient(stops, parameter, *extend, *interpolation)
                    .map(|channel| parameter.is_finite().select(channel, f32x4::ZERO))
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
                let (x, y) = apply(*inv, x, y);
                let x = (x - center[0]).to_array();
                let y = (y - center[1]).to_array();
                let parameter = f32x4::from(std::array::from_fn(|lane| {
                    (y[lane].atan2(x[lane]) - start).rem_euclid(std::f32::consts::TAU) / span
                }));
                gradient(stops, parameter, *extend, *interpolation)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cherenkov::kurbo::Affine;
    use std::sync::Arc;

    fn compare(paint: &PaintData) {
        for xs in [
            [-3.0, -0.5, 0.5, 1.0],
            [1.5, 2.25, 4.5, 8.0],
            [20.0, 21.25, 32.5, 127.5],
        ] {
            for ys in [[0.5; 4], [2.0, 3.0, 7.25, -1.0]] {
                let actual = f32x4::transpose(paint.eval4(f32x4::from(xs), f32x4::from(ys)))
                    .map(f32x4::to_array);
                for lane in 0..4 {
                    let expected = paint.eval(xs[lane], ys[lane]);
                    for channel in 0..4 {
                        assert_eq!(
                            actual[lane][channel].to_bits(),
                            expected[channel].to_bits(),
                            "paint={paint:?}, position=({}, {}), channel={channel}",
                            xs[lane],
                            ys[lane]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn vector_gradient_extensions_match_scalar_after_parameter_overflow() {
        let stops = Arc::from([
            Stop {
                offset: 0.0,
                color: [1.0, 0.0, 0.0, 1.0],
            },
            Stop {
                offset: 1.0,
                color: [0.0, 1.0, 0.0, 1.0],
            },
        ]);
        for extend in [Extend::Pad, Extend::None, Extend::Repeat, Extend::Reflect] {
            compare(&PaintData::Linear {
                inv: [f32::MAX, 0.0, 0.0, 1.0, 0.0, 0.0],
                end_points: [0.0, 0.0, 1.0, 0.0],
                stops: Arc::clone(&stops),
                extend,
                interpolation: Interpolation::Working,
            });
        }
    }

    #[test]
    fn vector_gradients_match_scalar_including_extensions_and_hdr() {
        let stops: Arc<[Stop]> = Arc::from([
            Stop {
                offset: 0.0,
                color: [-0.125, 2.0, 0.0, 0.5],
            },
            Stop {
                offset: 0.4,
                color: [0.75, 0.5, 1.25, 1.0],
            },
            Stop {
                offset: 0.4,
                color: [0.5, 0.75, 1.5, 0.25],
            },
            Stop {
                offset: 1.0,
                color: [4.0, -0.25, 2.0, 1.0],
            },
        ]);
        let inv = [1.0, 0.125, -0.25, 1.0, -2.5, 0.25];
        for extend in [Extend::None, Extend::Pad, Extend::Repeat, Extend::Reflect] {
            for interpolation in [Interpolation::Working, Interpolation::SrgbEncoded] {
                compare(&PaintData::Linear {
                    inv,
                    end_points: [0.0, 0.0, 16.0, 7.0],
                    stops: stops.clone(),
                    extend,
                    interpolation,
                });
                compare(&PaintData::Sweep {
                    inv,
                    center: [2.0, 3.0],
                    start: 0.25,
                    span: 3.5,
                    stops: stops.clone(),
                    extend,
                    interpolation,
                });
                for (centres, radii) in [
                    ([2.0, 3.0, 2.0, 3.0], [0.0, 8.0]),
                    ([0.0, 0.0, 8.0, 0.0], [0.0, 8.0]),
                    ([1.0, 1.0, 3.0, 4.0], [8.0, 2.0]),
                ] {
                    compare(&PaintData::Radial {
                        inv,
                        centres,
                        radii,
                        stops: stops.clone(),
                        extend,
                        interpolation,
                    });
                }
            }
        }
    }

    #[test]
    fn vector_image_taps_match_scalar_at_borders_and_rounding_ties() {
        let image = Arc::new(super::super::super::image::CpuImage {
            width: 2,
            height: 2,
            pixels: Box::new([
                [0.25, 0.5, 0.75, 0.5],
                [2.0, -0.25, 0.5, 1.0],
                [0.0; 4],
                [1.0, 2.0, 4.0, 1.0],
            ]),
        });
        for extend in [Extend::None, Extend::Pad, Extend::Repeat, Extend::Reflect] {
            for sampling in [Sampling::Nearest, Sampling::Linear] {
                compare(&PaintData::Image(Box::new(ImagePaintData {
                    mapping: Affine::IDENTITY,
                    inv: affine_f32(Affine::IDENTITY),
                    image: image.clone(),
                    extend_x: extend,
                    extend_y: extend,
                    sampling,
                })));
            }
        }
    }
}
