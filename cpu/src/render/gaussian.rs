// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Integrated device taps of the affine push-forward of a local Gaussian.

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
            .map(|interval| integrate(&density, interval[0], interval[1], 2e-14, 20))
            .sum()
    }
}

fn axis_tap(distance: f64, sigma: f64) -> f64 {
    if sigma == 0.0 {
        return if distance == 0.0 { 1.0 } else { 0.0 };
    }
    normal_interval((distance - 0.5) / sigma, (distance + 0.5) / sigma)
}

/// Eight-point Gauss-Legendre quadrature, refined by comparing two half panels.
fn panel(f: &impl Fn(f64) -> f64, low: f64, high: f64) -> f64 {
    let mid = low.midpoint(high);
    let half = (high - low) * 0.5;
    let mut value = 0.0;
    for (node, weight) in [
        (0.183_434_642_495_649_8_f64, 0.362_683_783_378_362_f64),
        (0.525_532_409_916_329, 0.313_706_645_877_887_3),
        (0.796_666_477_413_626_7, 0.222_381_034_453_374_5),
        (0.960_289_856_497_536_3, 0.101_228_536_290_376_3),
    ] {
        value = weight.mul_add(
            f(node.mul_add(half, mid)) + f((-node).mul_add(half, mid)),
            value,
        );
    }
    value * half
}

fn integrate(f: &impl Fn(f64) -> f64, low: f64, high: f64, tolerance: f64, depth: u32) -> f64 {
    let mid = low.midpoint(high);
    let whole = panel(f, low, high);
    let halves = panel(f, low, mid) + panel(f, mid, high);
    if (halves - whole).abs() <= tolerance {
        return halves;
    }
    assert!(depth > 0, "Gaussian tap quadrature failed to converge");
    integrate(f, low, mid, tolerance * 0.5, depth - 1)
        + integrate(f, mid, high, tolerance * 0.5, depth - 1)
}

/// Independent axes use two linear passes; correlated taps use full rows.
pub enum Kernel {
    /// Product of the two marginal distributions.
    Separable {
        /// Horizontal integrated weights, centred at the middle element.
        horizontal: Vec<f64>,
        /// Vertical integrated weights, centred at the middle element.
        vertical: Vec<f64>,
    },
    /// Correlated two-dimensional distribution.
    Correlated {
        /// Horizontal half-width of every kernel row.
        radius_x: usize,
        /// Integrated weights, centred at the middle row and column.
        rows: Vec<Vec<f64>>,
    },
}

/// Prepare the finite, normalized six-sigma kernel once per coverage cache miss.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "finite kernel radii and tap offsets are bounded device coordinates"
)]
pub fn kernel(sigma: f64, transform: kurbo::Affine) -> Kernel {
    let gaussian = Gaussian::new(sigma, transform);
    let radius_x = (6.0 * gaussian.sx).ceil() as usize;
    let radius_y = (6.0 * gaussian.sy).ceil() as usize;
    if gaussian.slope == 0.0 {
        let axis = |radius: usize, sigma| {
            let mut weights: Vec<_> = (0..=2 * radius)
                .map(|i| axis_tap(i as f64 - radius as f64, sigma))
                .collect();
            let mass: f64 = weights.iter().sum();
            for weight in &mut weights {
                *weight /= mass;
            }
            weights
        };
        return Kernel::Separable {
            horizontal: axis(radius_x, gaussian.sx),
            vertical: axis(radius_y, gaussian.sy),
        };
    }
    let mut rows: Vec<Vec<_>> = (0..=2 * radius_y)
        .map(|y| {
            (0..=2 * radius_x)
                .map(|x| gaussian.tap(x as f64 - radius_x as f64, y as f64 - radius_y as f64))
                .collect()
        })
        .collect();
    let mass: f64 = rows.iter().flatten().sum();
    for weight in rows.iter_mut().flatten() {
        *weight /= mass;
    }
    Kernel::Correlated { radius_x, rows }
}
