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

/// Close each authored contour, including an implicit final closing segment.
fn closed_contours(path: &kurbo::BezPath) -> kurbo::BezPath {
    let mut closed = kurbo::BezPath::new();
    let mut open = false;
    for &element in path.elements() {
        match element {
            kurbo::PathEl::MoveTo(_) => {
                if open { closed.close_path(); }
                open = true;
            }
            kurbo::PathEl::ClosePath => open = false,
            _ => {}
        }
        closed.push(element);
    }
    if open { closed.close_path(); }
    closed
}

/// Coverage of a spread caster, before convolution. Spread is a round-joined
/// contour band of width `2 * abs(spread)` in shape units: positive spread
/// unions that band with the fill, negative spread subtracts it. All authored
/// contours participate, including internal contours. The transform includes
/// the shadow offset; clips remain in device space. Boolean area identities
/// are applied to exact intersections, never products of coverage values.
#[must_use]
pub fn spread_coverage(
    path: &kurbo::BezPath,
    rule: cherenkov_scene::FillRule,
    transform: kurbo::Affine,
    spread: f64,
    clips: &[Vec<crate::clip::Segment>],
    size: (usize, usize),
) -> Vec<f64> {
    use crate::{clip::intersect_edges, coverage::Coverage, path::{edges, flatten_at, SEGMENT_TOLERANCE}};
    use cherenkov_scene::FillRule;
    let field = |mut segments: Vec<crate::clip::Segment>, mut rule| {
        for clip in clips {
            segments = intersect_edges(&segments, rule, clip);
            rule = FillRule::NonZero;
        }
        let mut coverage = Coverage::new(size.0, size.1);
        for (x0, y0, x1, y1) in segments { coverage.add_line(x0, y0, x1, y1); }
        coverage.finish(rule)
    };
    let caster = edges(&flatten_at(&(transform * path.clone()), SEGMENT_TOLERANCE));
    let original = field(caster.clone(), rule);
    if spread == 0.0 { return original; }
    let outline = kurbo::stroke(closed_contours(path), &kurbo::Stroke::new(2.0 * spread.abs())
        .with_join(kurbo::Join::Round).with_caps(kurbo::Cap::Round),
        &kurbo::StrokeOpts::default(), SEGMENT_TOLERANCE / crate::path::sigma_max(transform).max(1e-12));
    let band = edges(&flatten_at(&(transform * outline), SEGMENT_TOLERANCE));
    let intersection = field(intersect_edges(&caster, rule, &band), FillRule::NonZero);
    let band = field(band, FillRule::NonZero);
    original.into_iter().zip(intersection).zip(band)
        .map(|((shape, overlap), band)| {
            if spread > 0.0 { (shape + band - overlap).clamp(0.0, 1.0) }
            else { (shape - overlap).clamp(0.0, 1.0) }
        }).collect()
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use kurbo::Shape as _;

    #[test]
    fn signed_contour_spread_has_the_expected_rect_area() {
        let path = kurbo::Rect::new(5.0, 5.0, 15.0, 15.0).to_path(1e-4);
        for (spread, area) in [(-1.5, 49.0), (0.0, 100.0),
            (1.5, 160.0 + std::f64::consts::PI * 2.25)] {
            let field = spread_coverage(&path, cherenkov_scene::FillRule::NonZero,
                kurbo::Affine::IDENTITY, spread, &[], (24, 24));
            assert!((field.iter().sum::<f64>() - area).abs() < 0.002);
        }
    }
}
