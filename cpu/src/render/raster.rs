// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sparse coverage compositing over independently owned framebuffer bands.
//! See `coverage` for the geometric intersection compiler.

use rayon::prelude::*;

use crate::render::blend::{blend, src_over};
use crate::render::coverage::Coverage;
use crate::render::lower::{IRect, Item};
use crate::render::paint::PaintData;

/// Rows per rasterization band.
pub const BAND_H: usize = 16;

/// Sample count of the shadow y-quadrature (the GPU uses the same).
const SHADOW_N: usize = 16;

/// A directed edge in device space.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    /// Start point.
    pub x0: f32,
    /// Start point.
    pub y0: f32,
    /// End point.
    pub x1: f32,
    /// End point.
    pub y1: f32,
}

/// Abramowitz & Stegun 7.1.26 — the same approximation the GPU WGSL
/// uses, |error| < 1.5e-7.
#[expect(
    clippy::many_single_char_names,
    clippy::excessive_precision,
    clippy::suboptimal_flops,
    reason = "the A&S formula and its coefficients are cited verbatim"
)]
fn erf(x: f32) -> f32 {
    let s = x.signum();
    let a = x.abs();
    let t = (0.327_591_1_f32 * a + 1.0).recip();
    let polynomial = ((((1.061_405_429_f32 * t - 1.453_152_027) * t + 1.421_413_741) * t
        - 0.284_496_736) * t + 0.254_829_592) * t;
    let y = 1.0 - polynomial * (-a * a).exp();
    s * y
}

/// `exp(-x²/2σ²) / (σ√2π)`.
#[expect(clippy::excessive_precision, reason = "sqrt(2π) to f32 accuracy")]
fn gaussian(x: f32, sigma: f32) -> f32 {
    (-(x * x) / (2.0 * sigma * sigma)).exp() / (2.506_628_274_6 * sigma)
}

/// Horizontal inset of a circular corner of radius `r` at distance `dy`
/// past the start of the corner (`dy <= 0` is the straight edge).
#[expect(clippy::suboptimal_flops, reason = "the reference formula verbatim")]
fn corner_inset(r: f32, dy: f32) -> f32 {
    if dy <= 0.0 || r <= 0.0 {
        return 0.0;
    }
    let dd = dy.min(r);
    r - (r * r - dd * dd).max(0.0).sqrt()
}

/// The buffer at the top of the isolation stack, or the band's
/// framebuffer slice.
fn top<'a>(fb: &'a mut [[f32; 4]], stack: &'a mut [Vec<[f32; 4]>]) -> &'a mut [[f32; 4]] {
    stack.last_mut().map_or(fb, Vec::as_mut_slice)
}

/// Retained scheduling and isolation storage for one framebuffer band.
#[derive(Default)]
pub struct BandScratch {
    items: Vec<usize>,
    stack: Vec<Vec<[f32; 4]>>,
    spare: Vec<Vec<[f32; 4]>>,
}

impl BandScratch {
    /// Retained allocation bytes.
    pub fn bytes(&self) -> usize {
        self.items.capacity() * size_of::<usize>()
            + self
                .stack
                .iter()
                .chain(&self.spare)
                .map(|buffer| buffer.capacity() * size_of::<[f32; 4]>())
                .sum::<usize>()
    }
}

/// Bins items once, then shades disjoint bands in painter order. Isolation
/// markers reach every band because non-normal blends can affect empty source.
pub fn render_bands(
    items: &[Item],
    clear: [f32; 4],
    fb: &mut [[f32; 4]],
    w: usize,
    h: usize,
    scratch: &mut Vec<BandScratch>,
) -> (u32, u32) {
    if w == 0 || h == 0 {
        return (0, 0);
    }
    scratch.resize_with(h.div_ceil(BAND_H), BandScratch::default);
    for band in &mut *scratch {
        band.items.clear();
    }
    let (mut draws, mut edges) = (0_u32, 0_u32);
    for (i, item) in items.iter().enumerate() {
        let (top, bottom) = match item {
            Item::Draw {
                coverage,
                edge_count,
                ..
            } => {
                draws += 1;
                edges += u32::try_from(*edge_count).unwrap_or(u32::MAX);
                (coverage.top.min(h), coverage.bottom().min(h))
            }
            Item::Glyph { slot, y, .. } => {
                let mask = slot.get().expect("glyphs resolved before compositing");
                let top = i64::from(*y) + i64::from(mask.top);
                let bottom = top + i64::from(mask.h);
                (
                    usize::try_from(top).unwrap_or(0).min(h),
                    usize::try_from(bottom).unwrap_or(0).min(h),
                )
            }
            Item::PushIsolate | Item::PopIsolate { .. } => (0, h),
        };
        if top < bottom {
            for band in &mut scratch[top / BAND_H..bottom.div_ceil(BAND_H)] {
                band.items.push(i);
            }
        }
    }
    fb.par_chunks_mut(BAND_H * w)
        .zip(scratch.par_iter_mut())
        .enumerate()
        .for_each(|(index, (slice, scratch))| {
            slice.fill(clear);
            let len = slice.len();
            let mut band = Band {
                fb: slice,
                w,
                y0: index * BAND_H,
            };
            for &i in &scratch.items {
                match &items[i] {
                    Item::Draw { coverage, paint, .. } => {
                        band.draw(&mut scratch.stack, coverage, paint);
                    }
                    Item::PushIsolate => {
                        let mut buffer = scratch.spare.pop().unwrap_or_default();
                        buffer.resize(len, [0.0; 4]);
                        buffer.fill([0.0; 4]);
                        scratch.stack.push(buffer);
                    }
                    Item::PopIsolate { opacity, blend } => {
                        let buffer = scratch.stack.pop().expect("balanced isolation items");
                        band.composite_isolate(&buffer, *opacity, *blend, &mut scratch.stack);
                        scratch.spare.push(buffer);
                    }
                    Item::Glyph {
                        slot, x, y, paint, ..
                    } => {
                        band.glyph(&mut scratch.stack, slot, *x, *y, paint);
                    }
                }
            }
            assert!(scratch.stack.is_empty(), "balanced isolation items");
        });
    (draws, edges)
}

/// Evaluates the analytic unoccluded rounded-box shadow once per geometry key.
pub fn shadow_coverage(
    rbox: &[f32; 4],
    radii: &[f32; 4],
    sigma: f32,
    bbox: IRect,
    w: usize,
) -> Coverage {
    let top = usize::try_from(bbox.y0).expect("clamped shadow bounds");
    let bottom = usize::try_from(bbox.y1).expect("clamped shadow bounds");
    let mut pixels = vec![[0.0; 4]; w * (bottom - top)];
    let mut band = Band {
        fb: &mut pixels,
        w,
        y0: top,
    };
    band.shadow(rbox, radii, sigma, &[1.0; 4], bbox);
    Coverage::from_rows(
        top,
        pixels
            .chunks_exact(w)
            .map(|row| row.iter().map(|pixel| pixel[3]).collect()),
    )
}

/// The oracle's integrated Gaussian taps, applied after exact caster clipping.
/// The convolution is restricted to the caster's support plus the kernel halo.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::suboptimal_flops,
    reason = "bounded surface/tap indices; f64 integration rounds once to coverage precision"
)]
pub fn blur_coverage(source: &Coverage, width: usize, height: usize, sigma: f64) -> Coverage {
    if width == 0 || height == 0 || source.top == source.bottom() {
        return Coverage::default();
    }
    let radius = if sigma <= 1e-9 {
        0
    } else {
        (6.0 * sigma).ceil() as usize
    };
    let top = source.top.saturating_sub(radius);
    let bottom = source.bottom().saturating_add(radius).min(height);
    let mut kernel = Vec::with_capacity(2 * radius + 1);
    if sigma <= 1e-9 {
        kernel.push(1.0);
    } else {
        let inv = 1.0 / (sigma * std::f64::consts::SQRT_2);
        for tap in 0..=2 * radius {
            let distance = tap as f64 - radius as f64;
            kernel.push(
                0.5 * (libm::erf((distance + 0.5) * inv) - libm::erf((distance - 0.5) * inv)),
            );
        }
        let sum: f64 = kernel.iter().sum();
        for weight in &mut kernel {
            *weight /= sum;
        }
    }
    let mut horizontal = vec![0.0_f64; width * (source.bottom() - source.top)];
    let mut row = vec![0.0_f64; width];
    for y in source.top..source.bottom() {
        row.fill(0.0);
        let spans = source.row(y);
        for span in spans {
            for x in span.columns.clone() {
                row[x] = f64::from(span.at(x));
            }
        }
        if let (Some(first), Some(last)) = (spans.first(), spans.last()) {
            let left = first.columns.start.saturating_sub(radius);
            let right = last.columns.end.saturating_add(radius).min(width);
            for x in left..right {
                let mut value = 0.0;
                for (tap, &weight) in kernel.iter().enumerate() {
                    let column =
                        (x as i64 + tap as i64 - radius as i64).clamp(0, width as i64 - 1) as usize;
                    value += weight * row[column];
                }
                horizontal[(y - source.top) * width + x] = value;
            }
        }
    }
    Coverage::from_rows(top, (top..bottom).map(|y| {
        let mut row = vec![0.0_f32; width];
        for (x, value) in row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (tap, &weight) in kernel.iter().enumerate() {
                let sample_y =
                    (y as i64 + tap as i64 - radius as i64).clamp(0, height as i64 - 1) as usize;
                if (source.top..source.bottom()).contains(&sample_y) {
                    sum += weight * horizontal[(sample_y - source.top) * width + x];
                }
            }
            *value = sum as f32;
        }
        row
    }))
}

/// One band's rasterization state.
struct Band<'a> {
    /// The band's framebuffer rows.
    fb: &'a mut [[f32; 4]],
    /// Surface width.
    w: usize,
    /// Device y of the band's first row.
    y0: usize,
}

impl Band<'_> {
    /// Shades only nonempty runs. Opaque solid spans are direct stores.
    #[expect(clippy::cast_precision_loss, reason = "surface coordinates fit f32")]
    fn draw(&mut self, stack: &mut [Vec<[f32; 4]>], coverage: &Coverage, paint: &PaintData) {
        let bottom = (self.y0 + self.fb.len() / self.w).min(coverage.bottom());
        let dst = top(&mut *self.fb, stack);
        for y in self.y0.max(coverage.top)..bottom {
            let row = (y - self.y0) * self.w;
            for span in coverage.row(y) {
                if let PaintData::Solid(color) = paint {
                    let pixels = &mut dst[row + span.columns.start..row + span.columns.end];
                    if span.samples.is_empty() {
                        let src = color.map(|value| value * span.alpha);
                        if src[3].to_bits() == 1.0_f32.to_bits() {
                            pixels.fill(src);
                        } else {
                            for pixel in pixels {
                                *pixel = src_over(*pixel, src);
                            }
                        }
                    } else {
                        for (pixel, &alpha) in pixels.iter_mut().zip(&span.samples) {
                            *pixel = src_over(*pixel, color.map(|value| value * alpha));
                        }
                    }
                } else {
                    for x in span.columns.clone() {
                        let src = paint
                            .eval(x as f32 + 0.5, y as f32 + 0.5)
                            .map(|value| value * span.at(x));
                        dst[row + x] = src_over(dst[row + x], src);
                    }
                }
            }
        }
    }

    /// Rasterizes a blurred rounded box: analytic coverage per pixel in
    /// `bbox` ∩ band.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::float_cmp,
        clippy::suboptimal_flops,
        clippy::too_many_lines,
        reason = "pixel indices and band offsets are far below 2^24; the \
                  flat-row test is intentionally exact and the quadrature \
                  weights mirror the reference formula"
    )]
    fn shadow(
        &mut self,
        rbox: &[f32; 4],
        radii: &[f32; 4],
        sigma_eff: f32,
        color: &[f32; 4],
        bbox: IRect,
    ) {
        let bh = self.fb.len() / self.w;
        let (y_lo, y_hi) = (
            usize::try_from(bbox.y0)
                .unwrap_or(0)
                .saturating_sub(self.y0)
                .min(bh),
            usize::try_from(bbox.y1)
                .unwrap_or(0)
                .saturating_sub(self.y0)
                .min(bh),
        );
        let (x_lo, x_hi) = (
            usize::try_from(bbox.x0).unwrap_or(0).min(self.w),
            usize::try_from(bbox.x1).unwrap_or(0).min(self.w),
        );
        if y_lo >= y_hi || x_lo >= x_hi {
            return;
        }
        let (cx, cy) = (rbox[0], rbox[1]);
        let half = [rbox[2], rbox[3]];
        let sigma = sigma_eff;
        let inv_sqrt2_sigma = 1.0 / (sigma * std::f32::consts::SQRT_2);
        let (cx_lo, cx_hi) = (x_lo, x_hi);
        // The 16 y-quadrature samples and their `gaussian*step` weights
        // are per item, not per pixel.
        let step = 6.0 * sigma / SHADOW_N as f32;
        let mut dy = [0.0_f32; SHADOW_N];
        let mut gw = [0.0_f32; SHADOW_N];
        for (i, (d, w)) in dy.iter_mut().zip(gw.iter_mut()).enumerate() {
            *d = (i as f32 + 0.5) * step - 3.0 * sigma;
            *w = gaussian(*d, sigma) * step;
        }
        // Per row the quadrature's x-integral saturates except near the
        // left/right edges: `erf` reaches ±1 within `MARGIN` of an edge,
        // so interior columns all evaluate to the same `S` (the sum of
        // in-box sample weights) and only the two edge bands pay the
        // 32-`erf` sum. For `sigma`-wide margins the tail error is
        // `erf(5)-1 < 1e-11`.
        let margin = 5.0 * sigma * std::f32::consts::SQRT_2;
        // Flat rows — every sample has zero corner inset — share the
        // straight-wall x-integral, one `erf` pair per column computed
        // once for the whole item.
        let mut x_term: Option<Vec<f32>> = None;
        let mut xl = [0.0_f32; SHADOW_N];
        let mut xr = [0.0_f32; SHADOW_N];
        let mut wg = [0.0_f32; SHADOW_N];
        for y in y_lo..y_hi {
            let py_i = self.y0 + y;
            let py = py_i as f32 + 0.5 - cy;
            // The row's sample table: left/right x-edges per in-box
            // sample, their weights, the interior-coverage sum `s`, and
            // the most-inset edges bounding the interior zone.
            let mut n = 0_usize;
            let mut s = 0.0_f32;
            let mut flat = true;
            let (mut xlm, mut xrm) = (f32::MIN, f32::MAX);
            for (d, w) in dy.iter().zip(gw.iter()) {
                let yi = py + d;
                if yi.abs() > half[1] {
                    continue;
                }
                let (rl, rr) = if yi < 0.0 {
                    (radii[0], radii[1])
                } else {
                    (radii[3], radii[2])
                };
                let ay = yi.abs();
                xl[n] = -half[0] + corner_inset(rl, ay - (half[1] - rl));
                xr[n] = half[0] - corner_inset(rr, ay - (half[1] - rr));
                wg[n] = *w;
                flat &= xl[n] == -half[0] && xr[n] == half[0];
                xlm = xlm.max(xl[n]);
                xrm = xrm.min(xr[n]);
                s += w;
                n += 1;
            }
            if n == 0 || s <= 0.0 {
                continue;
            }
            let dst = &mut *self.fb;
            if flat {
                // Separable: `cov = x_term[px] * s` across the row.
                let t = x_term.get_or_insert_with(|| {
                    (cx_lo..cx_hi)
                        .map(|x| {
                            let px = x as f32 + 0.5 - cx;
                            0.5 * (erf((half[0] - px) * inv_sqrt2_sigma)
                                - erf((-half[0] - px) * inv_sqrt2_sigma))
                        })
                        .collect()
                });
                for x in cx_lo..cx_hi {
                    let cov = t[x - cx_lo] * s;
                    if cov <= 0.0 {
                        continue;
                    }
                    let src = color.map(|v| v * cov.clamp(0.0, 1.0));
                    dst[y * self.w + x] = src_over(dst[y * self.w + x], src);
                }
                continue;
            }
            // Interior columns: `px >= xlm + margin` saturates every
            // `erf((xl-px)*is)` at -1 and `px <= xrm - margin` saturates
            // every `erf((xr-px)*is)` at +1, so `cov = s` for all of them.
            let x_in_lo = usize::try_from((cx + xlm + margin - 0.5).ceil() as i32)
                .unwrap_or(0)
                .clamp(cx_lo, cx_hi);
            let x_in_hi = usize::try_from((cx + xrm - margin + 0.5).floor() as i32 + 1)
                .unwrap_or(0)
                .clamp(x_in_lo, cx_hi);
            let dst = &mut *self.fb;
            let edge = |dst: &mut [[f32; 4]], f: usize, t: usize| {
                for x in f..t {
                    let px = x as f32 + 0.5 - cx;
                    let mut cov = 0.0_f32;
                    for k in 0..n {
                        cov += 0.5
                            * (erf((xr[k] - px) * inv_sqrt2_sigma)
                                - erf((xl[k] - px) * inv_sqrt2_sigma))
                            * wg[k];
                    }
                    if cov <= 0.0 {
                        continue;
                    }
                    let src = color.map(|v| v * cov.clamp(0.0, 1.0));
                    dst[y * self.w + x] = src_over(dst[y * self.w + x], src);
                }
            };
            edge(&mut *dst, cx_lo, x_in_lo);
            for x in x_in_lo..x_in_hi {
                let src = color.map(|v| v * s.clamp(0.0, 1.0));
                dst[y * self.w + x] = src_over(dst[y * self.w + x], src);
            }
            edge(&mut *dst, x_in_hi, cx_hi);
        }
    }

    /// Composites one unclipped glyph mask; clipped outlines are prepared draws.
    #[expect(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "mask coordinates and pixel indices are small"
    )]
    fn glyph(
        &mut self,
        stack: &mut [Vec<[f32; 4]>],
        slot: &std::sync::OnceLock<std::sync::Arc<crate::render::glyph::GlyphMask>>,
        ox: i32,
        oy: i32,
        paint: &PaintData,
    ) {
        let Some(mask) = slot.get() else { return };
        let bh = self.fb.len() / self.w;
        let (mx0, my0) = (ox + mask.left, oy + mask.top);
        let (x_lo, x_hi) = (
            usize::try_from(mx0).unwrap_or(0).min(self.w),
            usize::try_from(mx0 + mask.w as i32)
                .unwrap_or(0)
                .min(self.w),
        );
        let (y_lo, y_hi) = (
            usize::try_from(my0)
                .unwrap_or(0)
                .saturating_sub(self.y0)
                .min(bh),
            usize::try_from(my0 + mask.h as i32)
                .unwrap_or(0)
                .saturating_sub(self.y0)
                .min(bh),
        );
        if y_lo >= y_hi || x_lo >= x_hi {
            return;
        }
        for y in y_lo..y_hi {
            let py = self.y0 + y;
            let row = usize::try_from(py as i32 - my0).unwrap_or(0) * mask.w as usize;
            for x in x_lo..x_hi {
                let cov = mask.cov[row + usize::try_from(x as i32 - mx0).unwrap_or(0)];
                if cov <= 0.0 {
                    continue;
                }
                let src = paint
                    .eval(x as f32 + 0.5, py as f32 + 0.5)
                    .map(|v| v * cov);
                let dst = top(&mut *self.fb, stack);
                dst[y * self.w + x] = src_over(dst[y * self.w + x], src);
            }
        }
    }

    /// Composites the popped isolation buffer onto the buffer below.
    fn composite_isolate(
        &mut self,
        scratch: &[[f32; 4]],
        opacity: f32,
        mode: cherenkov::BlendMode,
        stack: &mut [Vec<[f32; 4]>],
    ) {
        let dst = top(&mut *self.fb, stack);
        for (i, &src) in scratch.iter().enumerate() {
            let s = src.map(|v| v * opacity);
            dst[i] = if mode == cherenkov::BlendMode::Normal {
                if s[3] == 0.0 {
                    continue;
                }
                src_over(dst[i], s)
            } else {
                blend(mode, dst[i], s)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    //! The sparse coverage compiler checked against the independent oracle.

    use super::*;
    use cherenkov::FillRule;
    use crate::render::coverage::{Operand, rasterize};

    fn coverage_mask(edges: &[Edge], rule: FillRule, w: usize, h: usize) -> Vec<f32> {
        let coverage = rasterize(&[Operand { edges: edges.into(), rule }], w, h);
        (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| coverage.at(x, y)).collect()
    }

    fn scene_rule(rule: FillRule) -> cherenkov_scene::FillRule {
        match rule {
            FillRule::NonZero => cherenkov_scene::FillRule::NonZero,
            FillRule::EvenOdd => cherenkov_scene::FillRule::EvenOdd,
        }
    }

    fn oracle_mask(edges: &[Edge], rule: FillRule, w: usize, h: usize) -> Vec<f64> {
        let mut cov = cherenkov_oracle::coverage::Coverage::new(w, h);
        for e in edges {
            cov.add_line(
                f64::from(e.x0),
                f64::from(e.y0),
                f64::from(e.x1),
                f64::from(e.y1),
            );
        }
        cov.finish(scene_rule(rule))
    }

    fn assert_matches_oracle(edges: &[Edge], name: &str) {
        for rule in [FillRule::NonZero, FillRule::EvenOdd] {
            let got = coverage_mask(edges, rule, 24, 24);
            let want = oracle_mask(edges, rule, 24, 24);
            let max_diff = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (f64::from(*g) - w).abs())
                .fold(0.0, f64::max);
            assert!(
                max_diff < 1e-4,
                "{name} {rule:?}: max |diff| {max_diff} vs oracle"
            );
        }
    }

    /// A closed polygon's edges.
    fn poly(points: &[(f32, f32)]) -> Vec<Edge> {
        points
            .iter()
            .enumerate()
            .map(|(i, &(x0, y0))| {
                let (x1, y1) = points[(i + 1) % points.len()];
                Edge { x0, y0, x1, y1 }
            })
            .collect()
    }

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::suboptimal_flops,
        reason = "test geometry is far below 2^53"
    )]
    fn exact_coverage_matches_the_oracle() {
        // (a) self-intersecting 5-point star.
        let star: Vec<(f32, f32)> = (0..5i32)
            .map(|i| {
                let a = f64::from(i) * std::f64::consts::TAU / 5.0 - std::f64::consts::FRAC_PI_2;
                (
                    (10.0 * a.cos() + 12.0) as f32,
                    (10.0 * a.sin() + 12.0) as f32,
                )
            })
            .collect();
        // Star polygon order (every other vertex) self-intersects.
        let star_order = [0, 2, 4, 1, 3].map(|i| star[i]);
        assert_matches_oracle(&poly(&star_order), "star");

        // (b) two nested same-orientation squares offset by 0.3px.
        let mut nested = poly(&[(2.3, 2.3), (21.3, 2.3), (21.3, 21.3), (2.3, 21.3)]);
        nested.extend(poly(&[(6.6, 6.6), (17.6, 6.6), (17.6, 17.6), (6.6, 17.6)]));
        assert_matches_oracle(&nested, "nested squares");

        // (c) a figure-eight.
        let eight = poly(&[
            (4.0, 4.0),
            (20.0, 12.0),
            (4.0, 20.0),
            (12.0, 12.0),
            (20.0, 4.0),
            (12.0, 12.0),
            (4.0, 12.0),
        ]);
        assert_matches_oracle(&eight, "figure-eight");

        // (d) bowtie: an interior crossing, not a shared endpoint.
        let bowtie = poly(&[
            (4.25, 4.125), (20.75, 20.875),
            (4.25, 20.875), (20.75, 4.125),
        ]);
        assert_matches_oracle(&bowtie, "bowtie");

        // (e) 20 random polygons of 6-12 vertices.
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u32 << 24) as f32 * 22.0 + 1.0
        };
        for n in 0..20 {
            let m = 6 + n % 7;
            let points: Vec<(f32, f32)> = (0..m).map(|_| (rng(), rng())).collect();
            assert_matches_oracle(&poly(&points), &format!("random {n}"));
        }
    }

    #[test]
    fn a_plain_square_keeps_its_exact_area() {
        // Regression for the common path: a 10.5x10.5 axis-aligned
        // square must still accumulate 110.25 of coverage.
        let edges = poly(&[(4.0, 4.0), (14.5, 4.0), (14.5, 14.5), (4.0, 14.5)]);
        let mask = coverage_mask(&edges, FillRule::NonZero, 24, 24);
        let total: f32 = mask.iter().sum();
        assert!((total - 110.25).abs() < 1e-3, "total coverage {total}");
    }
    #[test]
    fn sharp_stroke_joins_and_caps_match_the_oracle() {
        let mut path = kurbo::BezPath::new();
        path.move_to((2.125, 20.25));
        path.line_to((10.625, 3.125));
        path.line_to((11.125, 19.875));
        path.line_to((20.75, 3.625));
        path.line_to((2.125, 10.375));
        for join in [kurbo::Join::Miter, kurbo::Join::Bevel, kurbo::Join::Round] {
            for cap in [kurbo::Cap::Butt, kurbo::Cap::Square, kurbo::Cap::Round] {
                let stroke = kurbo::Stroke::new(3.75).with_join(join).with_caps(cap).with_miter_limit(12.0);
                let outline = kurbo::stroke(&path, &stroke, &kurbo::StrokeOpts::default(), 0.02);
                let edges = crate::render::lower::flatten_edges(outline, 0.02);
                assert_matches_oracle(&edges, "sharp stroke");
            }
        }
    }

    #[test]
    fn clipped_caster_is_blurred_after_geometric_intersection() {
        let caster = poly(&[(1.25, 1.25), (6.75, 1.25), (6.75, 6.75), (1.25, 6.75)]);
        let clip = poly(&[(0.25, 0.25), (7.75, 1.75), (2.25, 7.75)]);
        let operands = [
            Operand { edges: caster.clone().into(), rule: FillRule::NonZero },
            Operand { edges: clip.clone().into(), rule: FillRule::NonZero },
        ];
        let segments = |edges: &[Edge]| edges.iter().map(|e| (
            f64::from(e.x0), f64::from(e.y0), f64::from(e.x1), f64::from(e.y1),
        )).collect::<Vec<_>>();
        let intersection = cherenkov_oracle::clip::intersect_edges(
            &segments(&caster), cherenkov_scene::FillRule::NonZero, &segments(&clip),
        );
        let mut oracle = cherenkov_oracle::coverage::Coverage::new(8, 8);
        for (x0, y0, x1, y1) in intersection { oracle.add_line(x0, y0, x1, y1); }
        let exact = oracle.finish(cherenkov_scene::FillRule::NonZero);
        let source = rasterize(&operands, 8, 8);
        for sigma in [0.0, 1.25] {
            let expected = cherenkov_oracle::shadow::gaussian_blur(&exact, 8, 8, sigma);
            let actual = blur_coverage(&source, 8, 8, sigma);
            for (i, value) in expected.iter().enumerate() {
                assert!((f64::from(actual.at(i % 8, i / 8)) - value).abs() < 2e-6);
            }
        }
    }

    #[test]
    fn a_convex_contour_matches_the_oracle() {
        use cherenkov::kurbo::Shape as _;
        let path = cherenkov::kurbo::RoundedRect::new(2.3, 2.3, 21.7, 19.1, 4.2).to_path(0.1);
        let edges = crate::render::lower::flatten_edges(path, 0.05);
        assert_matches_oracle(&edges, "rounded rectangle");
    }
}
