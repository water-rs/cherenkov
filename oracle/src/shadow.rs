// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shadows convolve exact, offset and clip-intersected coverage with a Gaussian.
//!
//! Sigma and spread are in shape units. For the linear part `A` of the drawing
//! transform the device Gaussian has covariance `sigma² A Aᵀ`. Translation
//! affects the caster, not the kernel. Each tap is the probability mass in its
//! device-pixel square, not a sample of the density. Retain offsets through
//! `ceil(6 * marginal_sigma)` on each device axis, then normalize the retained
//! mass. This reduces to the original six-sigma integrated, separable taps for
//! an identity transform. Sampling outside the surface clamps to its edge.
//!
//! Correlated tap integrals use conditional normal probabilities and adaptive
//! quadrature in `f64`. Rank-one transforms integrate the resulting line measure;
//! zero variance is a point mass. There is no positive-sigma sharpness threshold.

/// Close each authored contour, including an implicit final closing segment.
fn closed_contours(path: &kurbo::BezPath) -> kurbo::BezPath {
    let mut closed = kurbo::BezPath::new();
    let mut open = false;
    for &element in path.elements() {
        match element {
            kurbo::PathEl::MoveTo(_) => {
                if open {
                    closed.close_path();
                }
                open = true;
            }
            kurbo::PathEl::ClosePath => open = false,
            _ => {}
        }
        closed.push(element);
    }
    if open {
        closed.close_path();
    }
    closed
}

/// Coverage of a spread caster before convolution.
///
/// Rectangles and rounded
/// rectangles grow their half-extents by `spread`; a positive corner radius
/// becomes `max(0, radius + spread)`, while a sharp corner stays sharp. A
/// nonpositive resulting extent is empty. Other shapes offset their closed
/// authored contours with miter joins, miter limit 4 (the SVG default), and
/// bevels beyond the limit.
/// Positive spread unions the offset band with the fill; negative spread
/// subtracts it. All contours, including internal contours, participate.
/// The transform includes the local shadow offset; clips remain in device
/// space. All Boolean operations precede pixel-area integration.
#[must_use]
pub fn spread_coverage(
    shape: &cherenkov_scene::Shape,
    rule: cherenkov_scene::FillRule,
    transform: kurbo::Affine,
    spread: f64,
    clips: &[Vec<crate::clip::Segment>],
    size: (usize, usize),
) -> Vec<f64> {
    use crate::{
        clip::intersect_edges,
        coverage::Coverage,
        path::{SEGMENT_TOLERANCE, edges, flatten_at},
    };
    use cherenkov_scene::FillRule;
    let field = |mut segments: Vec<crate::clip::Segment>, mut rule| {
        for clip in clips {
            segments = intersect_edges(&segments, rule, clip);
            rule = FillRule::NonZero;
        }
        let mut coverage = Coverage::new(size.0, size.1);
        for (x0, y0, x1, y1) in segments {
            coverage.add_line(x0, y0, x1, y1);
        }
        coverage.finish(rule)
    };
    let tolerance = SEGMENT_TOLERANCE / crate::path::sigma_max(transform).max(1e-12);
    let rounded = match shape {
        cherenkov_scene::Shape::Rect(rect) => Some(kurbo::RoundedRect::from_rect(*rect, 0.0)),
        cherenkov_scene::Shape::RoundedRect(rect) => Some(*rect),
        cherenkov_scene::Shape::Line(_) => return vec![0.0; size.0 * size.1],
        _ => None,
    };
    let path = if let Some(rounded) = rounded {
        use kurbo::Shape as _;
        let rect = rounded.rect().inflate(spread, spread);
        if rect.width() <= 0.0 || rect.height() <= 0.0 {
            return vec![0.0; size.0 * size.1];
        }
        let radii = rounded.radii();
        let radius = |r: f64| if r > 0.0 { (r + spread).max(0.0) } else { 0.0 };
        kurbo::RoundedRect::from_rect(
            rect,
            kurbo::RoundedRectRadii::new(
                radius(radii.top_left),
                radius(radii.top_right),
                radius(radii.bottom_right),
                radius(radii.bottom_left),
            ),
        )
        .to_path(tolerance)
    } else {
        shape.to_path_at(tolerance)
    };
    let caster = edges(&flatten_at(&(transform * path.clone()), SEGMENT_TOLERANCE));
    let original = field(caster.clone(), rule);
    if spread == 0.0 || rounded.is_some() {
        return original;
    }
    let outline = kurbo::stroke(
        closed_contours(&path),
        &kurbo::Stroke::new(2.0 * spread.abs())
            .with_join(kurbo::Join::Miter)
            .with_miter_limit(4.0),
        &kurbo::StrokeOpts::default(),
        SEGMENT_TOLERANCE / crate::path::sigma_max(transform).max(1e-12),
    );
    let band = edges(&flatten_at(&(transform * outline), SEGMENT_TOLERANCE));
    let intersection = field(intersect_edges(&caster, rule, &band), FillRule::NonZero);
    let band = field(band, FillRule::NonZero);
    original
        .into_iter()
        .zip(intersection)
        .zip(band)
        .map(|((shape, overlap), band)| {
            if spread > 0.0 {
                (shape + band - overlap).clamp(0.0, 1.0)
            } else {
                (shape - overlap).clamp(0.0, 1.0)
            }
        })
        .collect()
}

/// Standard normal probability of an interval; `erfc` preserves tail precision.
fn normal_interval(low: f64, high: f64) -> f64 {
    if low >= high {
        return 0.0;
    }
    let scale = std::f64::consts::FRAC_1_SQRT_2;
    if low >= 0.0 {
        0.5 * (libm::erfc(low * scale) - libm::erfc(high * scale))
    } else if high <= 0.0 {
        normal_interval(-high, -low)
    } else {
        0.5 * (libm::erf(high * scale) - libm::erf(low * scale))
    }
}

/// A conditional representation: `X = sx Z`, `Y = slope Z + residual W`,
/// for independent standard normal variables `Z`, `W`.
struct Gaussian {
    sx: f64,
    sy: f64,
    slope: f64,
    residual: f64,
}

impl Gaussian {
    #[expect(
        clippy::suboptimal_flops,
        reason = "separate products preserve exact zero covariance and determinant for orthogonal or singular matrices"
    )]
    fn new(sigma: f64, transform: kurbo::Affine) -> Self {
        let [a, b, c, d, _, _] = transform.as_coeffs();
        let horizontal = a.hypot(c);
        let sigma = sigma.max(0.0);
        let slope = if horizontal > 0.0 {
            sigma * ((a * b + c * d) / horizontal)
        } else {
            0.0
        };
        // Use the determinant, not sy² - slope²: subtraction of almost equal
        // variances loses the narrow conditional Gaussian near rank one.
        let residual = if horizontal > 0.0 {
            sigma * ((a * d - c * b) / horizontal).abs()
        } else {
            sigma * b.hypot(d)
        };
        Self {
            sx: sigma * horizontal,
            sy: sigma * b.hypot(d),
            slope,
            residual,
        }
    }

    fn tap(&self, x: f64, y: f64) -> f64 {
        if self.sx == 0.0 {
            return if x == 0.0 { axis_tap(y, self.sy) } else { 0.0 };
        }
        if self.slope == 0.0 {
            return axis_tap(x, self.sx) * axis_tap(y, self.sy);
        }
        let low = (x - 0.5) / self.sx;
        let high = (x + 0.5) / self.sx;
        if self.residual == 0.0 {
            let first = (y - 0.5) / self.slope;
            let last = (y + 0.5) / self.slope;
            return normal_interval(low.max(first.min(last)), high.min(first.max(last)));
        }
        // Beyond 12 standard deviations the omitted mass is < 4e-33, far
        // below f64 quadrature precision. Split at integer z and at every
        // conditional transition, so a near-singular covariance cannot hide
        // a narrow interval between quadrature nodes.
        let low = low.max(-12.0);
        let high = high.min(12.0);
        if low >= high {
            return 0.0;
        }
        let mut cuts = vec![low, high];
        cuts.extend((-11..12).map(f64::from).filter(|&z| z > low && z < high));
        for boundary in [y - 0.5, y + 0.5] {
            for offset in [-12.0_f64, -4.0, -1.0, 0.0, 1.0, 4.0, 12.0] {
                let z = offset.mul_add(self.residual, boundary) / self.slope;
                if z > low && z < high {
                    cuts.push(z);
                }
            }
        }
        cuts.sort_unstable_by(f64::total_cmp);
        cuts.dedup();
        let density = |z: f64| {
            let conditional = normal_interval(
                (-self.slope).mul_add(z, y - 0.5) / self.residual,
                (-self.slope).mul_add(z, y + 0.5) / self.residual,
            );
            (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt() * conditional
        };
        cuts.windows(2)
            .map(|interval| integrate(&density, interval[0], interval[1], 2e-15, 24))
            .sum()
    }
}

fn axis_tap(distance: f64, sigma: f64) -> f64 {
    if sigma == 0.0 {
        return if distance == 0.0 { 1.0 } else { 0.0 };
    }
    normal_interval((distance - 0.5) / sigma, (distance + 0.5) / sigma)
}

/// Adaptive Simpson integration; the cuts above isolate every rapid transition.
fn integrate(f: &impl Fn(f64) -> f64, low: f64, high: f64, tolerance: f64, depth: u32) -> f64 {
    let mid = low.midpoint(high);
    let ends = f(low) + f(high);
    let centre = f(mid);
    let whole = (high - low) * 4.0_f64.mul_add(centre, ends) / 6.0;
    let quarters = f(low.midpoint(mid)) + f(mid.midpoint(high));
    let halves = (high - low) * 4.0_f64.mul_add(quarters, 2.0_f64.mul_add(centre, ends)) / 12.0;
    if (halves - whole).abs() <= 15.0 * tolerance {
        return (halves - whole).mul_add(1.0 / 15.0, halves).max(0.0);
    }
    assert!(depth > 0, "Gaussian tap quadrature failed to converge");
    integrate(f, low, mid, tolerance * 0.5, depth - 1)
        + integrate(f, mid, high, tolerance * 0.5, depth - 1)
}

/// Convolve a device coverage field with a shape-space Gaussian pushed forward
/// by `transform`.
///
/// The six-sigma integrated tap definition is given in the
/// module documentation. Nonpositive sigma is the identity. Singular transforms
/// yield line or point distributions; no inverse transform is required.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "finite kernel radii and surface indices fit usize and i64"
)]
pub fn gaussian_blur(
    src: &[f64],
    width: usize,
    height: usize,
    sigma: f64,
    transform: kurbo::Affine,
) -> Vec<f64> {
    if src.is_empty() {
        return src.to_vec();
    }
    let gaussian = Gaussian::new(sigma, transform);
    let rx = (6.0 * gaussian.sx).ceil() as i64;
    let ry = (6.0 * gaussian.sy).ceil() as i64;
    if gaussian.slope == 0.0 {
        let mut result = src.to_vec();
        for (radius, sigma, stride) in [(rx, gaussian.sx, (1, 0)), (ry, gaussian.sy, (0, 1))] {
            let weights: Vec<_> = (-radius..=radius)
                .map(|i| axis_tap(i as f64, sigma))
                .collect();
            let mass: f64 = weights.iter().sum();
            let mut output = vec![0.0; src.len()];
            for y in 0..height {
                for x in 0..width {
                    let mut value = 0.0;
                    for (i, &weight) in weights.iter().enumerate() {
                        let distance = i as i64 - radius;
                        let column =
                            (x as i64 + distance * stride.0).clamp(0, width as i64 - 1) as usize;
                        let row =
                            (y as i64 + distance * stride.1).clamp(0, height as i64 - 1) as usize;
                        value = (weight / mass).mul_add(result[row * width + column], value);
                    }
                    output[y * width + x] = value;
                }
            }
            result = output;
        }
        return result;
    }
    let mut taps = Vec::new();
    for y in -ry..=ry {
        for x in -rx..=rx {
            let weight = gaussian.tap(x as f64, y as f64);
            if weight > 0.0 {
                taps.push((x, y, weight));
            }
        }
    }
    let mass: f64 = taps.iter().map(|&(_, _, weight)| weight).sum();
    let mut result = vec![0.0; src.len()];
    for y in 0..height {
        for x in 0..width {
            let mut value = 0.0;
            for &(dx, dy, weight) in &taps {
                let column = (x as i64 + dx).clamp(0, width as i64 - 1) as usize;
                let row = (y as i64 + dy).clamp(0, height as i64 - 1) as usize;
                value = (weight / mass).mul_add(src[row * width + column], value);
            }
            result[y * width + x] = value;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use cherenkov_scene::{FillRule, Shape};
    use kurbo::{Affine, Rect, RoundedRect, RoundedRectRadii, Shape as _};

    #[test]
    fn signed_miter_spread_has_the_expected_rect_area() {
        let rect = Rect::new(5.0, 5.0, 15.0, 15.0);
        for shape in [
            Shape::Rect(rect),
            Shape::Path {
                path: rect.to_path(1e-4),
            },
        ] {
            for (spread, area) in [(-5.0, 0.0), (-1.5, 49.0), (0.0, 100.0), (1.5, 169.0)] {
                let field = spread_coverage(
                    &shape,
                    FillRule::NonZero,
                    Affine::IDENTITY,
                    spread,
                    &[],
                    (24, 24),
                );
                assert!((field.iter().sum::<f64>() - area).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn rounded_spread_keeps_sharp_corners_and_clamps_shrinking_radii() {
        let shape = Shape::RoundedRect(RoundedRect::from_rect(
            Rect::new(6.0, 6.0, 18.0, 18.0),
            RoundedRectRadii::new(0.0, 1.0, 3.0, 4.0),
        ));
        for (spread, side, squared_radii) in [(-2.0, 8.0_f64, 5.0), (2.0, 16.0, 70.0)] {
            let field = spread_coverage(
                &shape,
                FillRule::NonZero,
                Affine::IDENTITY,
                spread,
                &[],
                (24, 24),
            );
            let area = side.mul_add(side, -(1.0 - std::f64::consts::FRAC_PI_4) * squared_radii);
            assert!((field.iter().sum::<f64>() - area).abs() < 0.001);
            // The authored top-left corner stays square after positive spread.
            if spread > 0.0 {
                assert!((field[4 * 24 + 4] - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn beyond_limit_miters_are_beveled() {
        let mut path = kurbo::BezPath::new();
        path.move_to((12.0, 10.0));
        path.line_to((13.0, 20.0));
        path.line_to((11.0, 20.0));
        path.close_path();
        let field = spread_coverage(
            &Shape::Path { path },
            FillRule::NonZero,
            Affine::IDENTITY,
            2.0,
            &[],
            (24, 24),
        );
        // A miter at the acute tip would reach y < 0. The bevel lies just
        // above y = 10, whereas a round join would reach y = 8.
        assert!(field[8 * 24 + 11].abs() < 1e-12);
        assert!(field[10 * 24 + 11] > 0.99);
    }

    #[test]
    fn integrated_taps_follow_scale_rotation_reflection_and_translation() {
        let source = [0.0, 0.0, 1.0, 0.0, 0.0];
        let expected = gaussian_blur(&source, 5, 1, 1.5, Affine::IDENTITY);
        for transform in [
            Affine::scale(3.0),
            Affine::rotate(0.7) * Affine::scale(3.0),
            Affine::new([-3.0, 0.0, 0.0, 3.0, 123.0, -50.0]),
        ] {
            let actual = gaussian_blur(&source, 5, 1, 0.5, transform);
            for (&a, &b) in actual.iter().zip(&expected) {
                assert!((a - b).abs() < 1e-13);
            }
        }
        let anisotropic = Gaussian::new(0.5, Affine::scale_non_uniform(6.0, 1.0));
        let expected = normal_interval(0.5 / 3.0, 1.5 / 3.0) * normal_interval(3.0, 5.0);
        assert!((anisotropic.tap(1.0, 2.0) - expected).abs() < 1e-15);
    }

    #[test]
    fn correlated_integrals_match_the_analytic_quadrant_probability() {
        for correlation in [-0.999_999_999_999_f64, -0.6, 0.6, 0.999_999_999_999] {
            let residual = (-correlation).mul_add(correlation, 1.0).sqrt();
            let gaussian = Gaussian::new(
                0.025,
                Affine::new([1.0, correlation, 0.0, residual, 0.0, 0.0]),
            );
            // The upper boundaries are 40 sigma away, so this unit square is
            // the positive quadrant to much better than quadrature precision.
            let expected = 0.25 + correlation.asin() / (2.0 * std::f64::consts::PI);
            assert!((gaussian.tap(0.5, 0.5) - expected).abs() < 2e-12);
        }
    }

    #[test]
    fn tiny_local_sigma_is_not_discarded_before_magnification() {
        let source = [0.0, 1.0, 0.0];
        let actual = gaussian_blur(&source, 3, 1, 1e-10, Affine::scale(1e10));
        let expected = gaussian_blur(&source, 3, 1, 1.0, Affine::IDENTITY);
        for (&a, &b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-14);
        }
        assert!(actual[0] > 0.2);
    }

    #[test]
    fn singular_gaussians_are_line_and_point_measures() {
        let diagonal = Gaussian::new(1.0, Affine::new([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]));
        assert!(diagonal.tap(0.0, 1.0).abs() < 1e-15);
        assert!((diagonal.tap(1.0, 1.0) - normal_interval(0.5, 1.5)).abs() < 1e-15);
        let vertical = Gaussian::new(2.0, Affine::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]));
        assert!((vertical.tap(0.0, 1.0) - normal_interval(0.25, 0.75)).abs() < 1e-15);
        let source = [0.25, 1.0, 0.75];
        let actual = gaussian_blur(&source, 3, 1, 3.0, Affine::scale(0.0));
        assert_eq!(actual, source);
    }

    #[test]
    fn sheared_impulse_has_the_transformed_covariance() {
        let mut source = vec![0.0; 33 * 33];
        source[16 * 33 + 16] = 1.0;
        let field = gaussian_blur(
            &source,
            33,
            33,
            2.0,
            Affine::new([1.0, 0.6, 0.0, 0.8, 0.0, 0.0]),
        );
        let mut variance = 0.0;
        let mut covariance = 0.0;
        for y in 0..33_u32 {
            for x in 0..33_u32 {
                let weight = field[(y * 33 + x) as usize];
                let dx = f64::from(x) - 16.0;
                variance = (dx * dx).mul_add(weight, variance);
                covariance = (dx * (f64::from(y) - 16.0)).mul_add(weight, covariance);
            }
        }
        assert!((field.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((variance - (4.0 + 1.0 / 12.0)).abs() < 2e-6);
        assert!((covariance - 2.4).abs() < 2e-6);
    }
}
