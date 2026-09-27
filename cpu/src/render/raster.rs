// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The banded exact-area coverage rasterizer.
//!
//! A port of the accumulation rasterizer from font-rs (`raster.rs`), also
//! used by `cherenkov-gpu` for glyph masks: every flattened directed edge
//! deposits a signed area into a `(width + 2) * band_h` accumulator whose
//! column 0 guards everything left of the canvas and whose last column
//! guards everything right of it, then each row is prefix-summed and the
//! fill rule turns the winding-weighted area into coverage.
//!
//! The accumulator is exact for polygons that do not self-overlap inside a
//! pixel — the oracle's `pixel_area` is exact even then; this is the known
//! difference this backend documents.

use rayon::prelude::*;

use cherenkov::FillRule;

use crate::render::lower::{ClipMask, ClipRef, IRect, Item};
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

/// A signed-area accumulation buffer over a band of `h` rows.
///
/// Columns are indexed `x + 1`, so column 0 collects every deposit at
/// `x <= -1` and column `w + 1` every deposit at `x >= w`. Deposits are
/// clamped into that range: their row sum is what the prefix sum consumes,
/// and a deposit's exact column below 0 or above `w` never changes the
/// coverage of an on-canvas pixel.
#[derive(Debug)]
pub struct Accum {
    w: usize,
    h: usize,
    /// `(w + 2) * h` cells.
    a: Vec<f32>,
    /// Guard-column window of the current draw: deposits clamp into
    /// `[cmin, cmax]` (inclusive); clearing and the prefix sum touch
    /// only that range.
    cmin: usize,
    cmax: usize,
}

impl Accum {
    /// A zeroed accumulator of `w` × `h` cells.
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            a: vec![0.0; (w + 2) * h],
            cmin: 0,
            cmax: w + 1,
        }
    }

    /// Restricts the draw window to pixels `x0..x1` (already clamped to
    /// `0..w`): deposits clamp into guard columns `x0..x1 + 1`, so a
    /// draw only ever pays its own bounding box's width.
    pub fn set_window(&mut self, x0: usize, x1: usize) {
        self.cmin = x0;
        self.cmax = (x1 + 1).min(self.w + 1);
    }

    /// Zeros the window's columns for band rows `y_lo..y_hi`.
    pub fn clear_range(&mut self, y_lo: usize, y_hi: usize) {
        for y in y_lo..y_hi.min(self.h) {
            let row = y * (self.w + 2);
            self.a[row + self.cmin..=row + self.cmax].fill(0.0);
        }
    }

    /// The accumulation cell for edge column `x` of band row `y`.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "canvas width fits i32; the column is clamped non-negative"
    )]
    fn cell(&mut self, x: i32, y: usize) -> &mut f32 {
        let col = (x + 1).clamp(self.cmin as i32, self.cmax as i32) as usize;
        &mut self.a[y * (self.w + 2) + col]
    }

    /// Accumulates the signed area of the segment `(x0,y0)-(x1,y1)`, where
    /// `y` is band-local (`0..h`).
    #[expect(
        clippy::similar_names,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::suboptimal_flops,
        reason = "a direct port of font-rs's scanline area accounting"
    )]
    pub fn draw_line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        if (y0 - y1).abs() <= f32::EPSILON {
            return;
        }
        let (dir, x0, y0, x1, y1) = if y0 < y1 {
            (1.0, x0, y0, x1, y1)
        } else {
            (-1.0, x1, y1, x0, y0)
        };
        let dxdy = (x1 - x0) / (y1 - y0);
        let mut x = x0;
        if y0 < 0.0 {
            x -= y0 * dxdy;
        }
        let y_start = y0.max(0.0) as usize;
        let y_end = self.h.min((y1.ceil() as usize).min(self.h));
        for y in y_start..y_end {
            let dy = ((y + 1) as f32).min(y1) - (y as f32).max(y0);
            let xnext = x + dxdy * dy;
            let d = dy * dir;
            let (xa, xb) = if x < xnext { (x, xnext) } else { (xnext, x) };
            let xa_floor = xa.floor();
            let xa_i = xa_floor as i32;
            let xb_ceil = xb.ceil();
            let xb_i = xb_ceil as i32;
            if xb_i <= xa_i + 1 {
                // The piece stays within one cell column.
                let xmf = x.midpoint(xnext) - xa_floor;
                *self.cell(xa_i, y) += d - d * xmf;
                *self.cell(xa_i + 1, y) += d * xmf;
            } else {
                let s = (xb - xa).recip();
                let xa_f = xa - xa_floor;
                let a0 = 0.5 * s * (1.0 - xa_f) * (1.0 - xa_f);
                let xb_f = xb - xb_ceil + 1.0;
                let am = 0.5 * s * xb_f * xb_f;
                *self.cell(xa_i, y) += d * a0;
                if xb_i == xa_i + 2 {
                    *self.cell(xa_i + 1, y) += d * (1.0 - a0 - am);
                } else {
                    let a1 = s * (1.5 - xa_f);
                    *self.cell(xa_i + 1, y) += d * (a1 - a0);
                    for xi in xa_i + 2..xb_i - 1 {
                        *self.cell(xi, y) += d * s;
                    }
                    let a2 = a1 + (xb_i - xa_i - 3) as f32 * s;
                    *self.cell(xb_i - 1, y) += d * (1.0 - a2 - am);
                    *self.cell(xb_i, y) += d * am;
                    x = xnext;
                    continue;
                }
                *self.cell(xb_i, y) += d * am;
            }
            x = xnext;
        }
    }

    /// Folds band-local row `y` into per-pixel coverage under `rule`,
    /// calling `f(x, coverage)` for each pixel `x0..x1`.
    ///
    /// The prefix sum starts at guard column `x0` and the coverage of
    /// pixel `x` is the sum through column `x + 1`.
    pub fn coverage_row(
        &self,
        y: usize,
        rule: FillRule,
        x0: usize,
        x1: usize,
        mut f: impl FnMut(usize, f32),
    ) {
        let row = y * (self.w + 2);
        let mut acc = self.a[row + x0];
        for x in x0..x1.min(self.w) {
            acc += self.a[row + x + 1];
            let cov = match rule {
                FillRule::NonZero => acc.abs().min(1.0),
                FillRule::EvenOdd => {
                    let m = acc.rem_euclid(2.0);
                    if m > 1.0 { 2.0 - m } else { m }
                }
            };
            if cov > 0.0 {
                f(x, cov);
            }
        }
    }
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
    let t = (0.327_591_1_f32.mul_add(a, 1.0)).recip();
    let y = 1.0
        - (1.061_405_429_f32
            .mul_add(t, -1.453_152_027)
            .mul_add(t, 1.421_413_741)
            .mul_add(t, -0.284_496_736)
            .mul_add(t, 0.254_829_592))
            * t
            * (-a * a).exp();
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

/// The clip coverage of `(px, py)`: 1/0 for the rect fast path, the mask
/// sample otherwise.
#[expect(
    clippy::cast_sign_loss,
    reason = "clip rect edges are clamped non-negative before indexing"
)]
fn clip_cov(clip: Option<&ClipRef>, w: usize, px: usize, py: usize) -> f32 {
    match clip.map(std::convert::AsRef::as_ref) {
        None => 1.0,
        Some(ClipMask::Rect(r)) => f32::from(
            px >= (r.x0.max(0) as usize)
                && px < (r.x1.max(0) as usize)
                && py >= (r.y0.max(0) as usize)
                && py < (r.y1.max(0) as usize),
        ),
        Some(ClipMask::Cover(mask)) => mask[py * w + px],
    }
}

/// The buffer at the top of the isolation stack, or the band's
/// framebuffer slice.
fn top<'a>(fb: &'a mut [[f32; 4]], stack: &'a mut [Vec<[f32; 4]>]) -> &'a mut [[f32; 4]] {
    stack.last_mut().map_or(fb, Vec::as_mut_slice)
}

/// `src_over` composite of premultiplied `src` onto `dst`.
fn src_over(dst: [f32; 4], src: [f32; 4]) -> [f32; 4] {
    let inv = 1.0 - src[3];
    [
        src[0].mul_add(1.0, dst[0] * inv),
        src[1].mul_add(1.0, dst[1] * inv),
        src[2].mul_add(1.0, dst[2] * inv),
        src[3].mul_add(1.0, dst[3] * inv),
    ]
}

/// Rasterizes the whole surface's items into `fb` (length `w*h`,
/// premultiplied linear P3), parallel over [`BAND_H`]-row bands.
///
/// Each band owns a coverage accumulator and a stack of scratch colour
/// buffers for [`Item::PushIsolate`]/[`Item::PopIsolate`], and walks the
/// item list sequentially.
pub fn render_bands(
    items: &[Item],
    _clear: [f32; 4],
    fb: &mut [[f32; 4]],
    w: usize,
    _h: usize,
) -> (u32, u32) {
    let (mut draws, mut edges) = (0_u32, 0_u32);
    fb.par_chunks_mut(BAND_H * w)
        .enumerate()
        .for_each(|(band, slice)| {
            let y0 = band * BAND_H;
            let bh = slice.len() / w;
            let mut acc = Accum::new(w, bh);
            // The isolation stack: `slice` is the bottom. Each entry is a
            // scratch colour buffer of the band.
            let mut stack: Vec<Vec<[f32; 4]>> = Vec::new();
            let mut band = Band { fb: slice, w, y0 };
            for item in items {
                match item {
                    Item::Draw {
                        edges,
                        bbox,
                        rule,
                        paint,
                        clip,
                    } => {
                        band.draw(
                            &mut acc,
                            &mut stack,
                            edges,
                            *bbox,
                            *rule,
                            paint,
                            clip.as_ref(),
                        );
                    }
                    Item::PushIsolate => {
                        stack.push(vec![[0.0; 4]; slice_len(band.w, bh)]);
                    }
                    Item::PopIsolate { opacity, clip } => {
                        let Some(scratch) = stack.pop() else {
                            continue;
                        };
                        band.composite_isolate(&scratch, *opacity, clip.as_ref(), &mut stack);
                    }
                    Item::Shadow {
                        rbox,
                        radii,
                        sigma_eff,
                        color,
                        bbox,
                        clip,
                    } => {
                        band.shadow(
                            &mut stack,
                            rbox,
                            radii,
                            *sigma_eff,
                            color,
                            *bbox,
                            clip.as_ref(),
                        );
                    }
                    Item::Glyph {
                        slot,
                        x,
                        y,
                        paint,
                        clip,
                    } => {
                        band.glyph(&mut stack, slot, *x, *y, paint, clip.as_ref());
                    }
                }
            }
        });
    // Counted before the parallel pass.
    for item in items {
        if let Item::Draw { edges: e, .. } = item {
            draws += 1;
            edges += u32::try_from(e.len()).unwrap_or(u32::MAX);
        }
    }
    (draws, edges)
}

const fn slice_len(w: usize, bh: usize) -> usize {
    w * bh
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
    /// Rasterizes one draw item into the top isolation buffer.
    #[expect(
        clippy::cast_precision_loss,
        clippy::too_many_arguments,
        reason = "pixel indices and band offsets are far below 2^24"
    )]
    fn draw(
        &mut self,
        acc: &mut Accum,
        stack: &mut Vec<Vec<[f32; 4]>>,
        edges: &[Edge],
        bbox: crate::render::lower::IRect,
        rule: FillRule,
        paint: &PaintData,
        clip: Option<&ClipRef>,
    ) {
        let bh = self.fb.len() / self.w;
        // Band-intersect the device-space bounding box.
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
        if y_lo >= y_hi {
            return;
        }
        let (x_lo, x_hi) = (
            usize::try_from(bbox.x0).unwrap_or(0).min(self.w),
            usize::try_from(bbox.x1).unwrap_or(0).min(self.w),
        );
        // Only the item's bbox columns participate: clear and deposit
        // inside the guard window `x_lo .. x_hi + 1`.
        acc.set_window(x_lo, x_hi);
        acc.clear_range(y_lo, y_hi);
        for e in edges {
            // Only edges crossing the band deposit anything.
            let ey0 = e.y0 - self.y0 as f32;
            let ey1 = e.y1 - self.y0 as f32;
            if ey0.max(ey1) < 0.0 || ey0.min(ey1) >= bh as f32 {
                continue;
            }
            acc.draw_line(e.x0, ey0, e.x1, ey1);
        }
        for y in y_lo..y_hi {
            let py = self.y0 + y;
            acc.coverage_row(y, rule, x_lo, x_hi, |x, cov| {
                let cc = clip_cov(clip, self.w, x, py);
                if cc <= 0.0 {
                    return;
                }
                let src = paint
                    .eval(x as f32 + 0.5, py as f32 + 0.5)
                    .map(|v| v * cov * cc);
                let dst = top(&mut *self.fb, stack.as_mut_slice());
                dst[y * self.w + x] = src_over(dst[y * self.w + x], src);
            });
        }
    }

    /// Rasterizes a blurred rounded box: analytic coverage per pixel in
    /// `bbox` ∩ band.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::float_cmp,
        clippy::suboptimal_flops,
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "pixel indices and band offsets are far below 2^24; the \
                  flat-row test is intentionally exact and the quadrature \
                  weights mirror the reference formula"
    )]
    fn shadow(
        &mut self,
        stack: &mut Vec<Vec<[f32; 4]>>,
        rbox: &[f32; 4],
        radii: &[f32; 4],
        sigma_eff: f32,
        color: &[f32; 4],
        bbox: IRect,
        clip: Option<&ClipRef>,
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
        // Rect-clip rows touch only `cx_lo..cx_hi`; a coverage clip stays
        // a per-pixel sample.
        let (cx_lo, cx_hi, clip_mask) = match clip.map(AsRef::as_ref) {
            Some(ClipMask::Rect(r)) => (
                usize::try_from(r.x0).unwrap_or(0).clamp(x_lo, x_hi),
                usize::try_from(r.x1).unwrap_or(0).clamp(x_lo, x_hi),
                None,
            ),
            Some(ClipMask::Cover(mask)) => (x_lo, x_hi, Some(mask.as_slice())),
            None => (x_lo, x_hi, None),
        };
        if cx_lo >= cx_hi && clip_mask.is_none() {
            return;
        }
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
            let dst = top(&mut *self.fb, stack.as_mut_slice());
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
                    let cc = clip_mask.map_or(1.0, |m| m[py_i * self.w + x]);
                    if cc <= 0.0 {
                        continue;
                    }
                    let src = color.map(|v| v * cov.clamp(0.0, 1.0) * cc);
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
            let dst = top(&mut *self.fb, stack.as_mut_slice());
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
                    let cc = clip_mask.map_or(1.0, |m| m[py_i * self.w + x]);
                    if cc <= 0.0 {
                        continue;
                    }
                    let src = color.map(|v| v * cov.clamp(0.0, 1.0) * cc);
                    dst[y * self.w + x] = src_over(dst[y * self.w + x], src);
                }
            };
            edge(&mut *dst, cx_lo, x_in_lo);
            for x in x_in_lo..x_in_hi {
                let cc = clip_mask.map_or(1.0, |m| m[py_i * self.w + x]);
                if cc <= 0.0 {
                    continue;
                }
                let src = color.map(|v| v * s.clamp(0.0, 1.0) * cc);
                dst[y * self.w + x] = src_over(dst[y * self.w + x], src);
            }
            edge(&mut *dst, x_in_hi, cx_hi);
        }
    }

    /// Rasterizes one glyph mask instance: `mask` rows intersecting the
    /// band composite `paint * mask * clipcov`.
    #[expect(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "mask coordinates and pixel indices are small"
    )]
    fn glyph(
        &mut self,
        stack: &mut Vec<Vec<[f32; 4]>>,
        slot: &std::sync::OnceLock<std::sync::Arc<crate::render::glyph::GlyphMask>>,
        ox: i32,
        oy: i32,
        paint: &PaintData,
        clip: Option<&ClipRef>,
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
                let cc = clip_cov(clip, self.w, x, py);
                if cc <= 0.0 {
                    continue;
                }
                let cov = mask.cov[row + usize::try_from(x as i32 - mx0).unwrap_or(0)];
                if cov <= 0.0 {
                    continue;
                }
                let src = paint
                    .eval(x as f32 + 0.5, py as f32 + 0.5)
                    .map(|v| v * cov * cc);
                let dst = top(&mut *self.fb, stack.as_mut_slice());
                dst[y * self.w + x] = src_over(dst[y * self.w + x], src);
            }
        }
    }

    /// Composites the popped isolation buffer onto the buffer below.
    fn composite_isolate(
        &mut self,
        scratch: &[[f32; 4]],
        opacity: f32,
        clip: Option<&ClipRef>,
        stack: &mut Vec<Vec<[f32; 4]>>,
    ) {
        let dst = top(&mut *self.fb, stack.as_mut_slice());
        for (i, &src) in scratch.iter().enumerate() {
            let px = i % self.w;
            let py = self.y0 + i / self.w;
            let cc = clip_cov(clip, self.w, px, py);
            let s = src.map(|v| v * opacity * cc);
            dst[i] = src_over(dst[i], s);
        }
    }
}

/// Rasterizes a full-surface coverage mask of `edges` under `rule`,
/// parallel over bands.
#[expect(
    clippy::cast_precision_loss,
    reason = "band origins are small integers"
)]
pub fn coverage_mask(edges: &[Edge], rule: FillRule, w: usize, h: usize) -> Vec<f32> {
    let mut mask = vec![0.0; w * h];
    mask.par_chunks_mut(BAND_H * w)
        .enumerate()
        .for_each(|(band, slice)| {
            let y0 = band * BAND_H;
            let bh = slice.len() / w;
            let mut acc = Accum::new(w, bh);
            for e in edges {
                let ey0 = e.y0 - y0 as f32;
                let ey1 = e.y1 - y0 as f32;
                if ey0.max(ey1) < 0.0 || ey0.min(ey1) >= bh as f32 {
                    continue;
                }
                acc.draw_line(e.x0, ey0, e.x1, ey1);
            }
            for y in 0..bh {
                acc.coverage_row(y, rule, 0, w, |x, cov| {
                    slice[y * w + x] = cov;
                });
            }
        });
    mask
}
