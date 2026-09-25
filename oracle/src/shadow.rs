// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gaussian blur of a coverage field — the oracle shadow model: the exact
//! coverage of the (clip-intersected) shape convolved with a true Gaussian
//! kernel in `f64`, separable, edge-clamped.
//!
//! Kernel weights are the Gaussian *integrated* over each pixel-wide tap
//! — `w_d = ½ (erf((d+½)/(σ√2)) − erf((d−½)/(σ√2)))` — rather than
//! point-sampled, and the tail is truncated at `⌈6σ⌉` where the outside
//! mass is below 10⁻⁹, so the convolution error stays under 10⁻⁴.

/// Separable Gaussian convolution of `src` (`width`×`height`).
/// `sigma <= 0` returns a copy.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "kernel radius and pixel indices are small non-negative values"
)]
pub fn gaussian_blur(src: &[f64], width: usize, height: usize, sigma: f64) -> Vec<f64> {
    if sigma <= 1e-9 || src.is_empty() {
        return src.to_vec();
    }
    let radius = (6.0 * sigma).ceil() as usize;
    let width_k = 2 * radius + 1;
    let inv = 1.0 / (sigma * std::f64::consts::SQRT_2);
    let mut kernel = Vec::with_capacity(width_k);
    let mut sum = 0.0;
    for i in 0..width_k {
        let d = i as f64 - radius as f64;
        // The Gaussian integrated over the tap's one-pixel footprint.
        let w = 0.5 * (libm::erf((d + 0.5) * inv) - libm::erf((d - 0.5) * inv));
        kernel.push(w);
        sum += w;
    }
    for w in &mut kernel {
        *w /= sum;
    }

    let mut tmp = vec![0.0; src.len()];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0;
            for (i, &w) in kernel.iter().enumerate() {
                let dx = i as i64 - radius as i64;
                let xx = (x as i64 + dx).clamp(0, width as i64 - 1) as usize;
                acc = w.mul_add(src[y * width + xx], acc);
            }
            tmp[y * width + x] = acc;
        }
    }
    let mut out = vec![0.0; src.len()];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0;
            for (i, &w) in kernel.iter().enumerate() {
                let dy = i as i64 - radius as i64;
                let yy = (y as i64 + dy).clamp(0, height as i64 - 1) as usize;
                acc = w.mul_add(tmp[yy * width + x], acc);
            }
            out[y * width + x] = acc;
        }
    }
    out
}
