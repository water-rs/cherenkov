// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sparse coverage compositing over independently owned framebuffer bands.
//! See `coverage` for the geometric intersection compiler.

use rayon::prelude::*;

use crate::render::blend::{in_space, src_over};
use crate::render::coverage::Coverage;
use crate::render::lower::Item;
use crate::render::paint::PaintData;

/// Rows per rasterization band.
pub const BAND_H: usize = 16;

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

/// The concrete target selected by the renderer-owned dispatch value.
pub fn simd_name(architecture: pulp::Arch) -> &'static str {
    struct Name;
    impl pulp::WithSimd for Name {
        type Output = &'static str;
        fn with_simd<S: pulp::Simd>(self, _: S) -> Self::Output {
            std::any::type_name::<S>()
        }
    }
    architecture.dispatch(Name)
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
    architecture: pulp::Arch,
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
            architecture.dispatch(ShadeBand {
                clear,
                items,
                slice,
                scratch,
                w,
                y0: index * BAND_H,
            });
        });
    (draws, edges)
}

/// One owned band and the immutable painter-order input for its SIMD dispatch.
struct ShadeBand<'a> {
    clear: [f32; 4],
    items: &'a [Item],
    slice: &'a mut [[f32; 4]],
    scratch: &'a mut BandScratch,
    w: usize,
    y0: usize,
}

impl pulp::WithSimd for ShadeBand<'_> {
    type Output = ();

    #[expect(
        clippy::inline_always,
        reason = "pulp requires the SIMD loop body inside its target-feature dispatch"
    )]
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) {
        let Self {
            clear,
            items,
            slice,
            scratch,
            w,
            y0,
        } = self;
        super::composite::fill(simd, slice, clear);
        let len = slice.len();
        let mut band = Band {
            fb: slice,
            w,
            y0,
            simd,
        };
        for &i in &scratch.items {
            match &items[i] {
                Item::Draw {
                    coverage, paint, ..
                } => {
                    band.draw(&mut scratch.stack, coverage, paint);
                }
                Item::PushIsolate => {
                    let mut buffer = scratch.spare.pop().unwrap_or_default();
                    buffer.resize(len, [0.0; 4]);
                    buffer.fill([0.0; 4]);
                    scratch.stack.push(buffer);
                }
                Item::PopIsolate {
                    opacity,
                    blend,
                    space,
                } => {
                    let buffer = scratch.stack.pop().expect("balanced isolation items");
                    band.composite_isolate(&buffer, *opacity, *blend, *space, &mut scratch.stack);
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
    }
}

/// Convolve the clipped caster using the Gaussian in the shape's coordinate system.
pub fn blur_coverage(
    source: &Coverage,
    width: usize,
    height: usize,
    sigma: f64,
    transform: kurbo::Affine,
) -> Coverage {
    use super::gaussian::{Kernel, kernel};
    if width == 0 || height == 0 || source.is_empty() {
        return Coverage::default();
    }
    match kernel(sigma, transform) {
        Kernel::Separable {
            horizontal,
            vertical,
        } => blur_separable(source, width, height, &horizontal, &vertical),
        Kernel::Correlated { radius_x, rows } => {
            blur_correlated(source, width, height, radius_x, &rows)
        }
    }
}

/// Independent covariance axes require only two one-dimensional passes.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::suboptimal_flops,
    reason = "bounded surface/tap indices; f64 integration rounds once to coverage precision"
)]
fn blur_separable(
    source: &Coverage,
    width: usize,
    height: usize,
    kernel_x: &[f64],
    kernel_y: &[f64],
) -> Coverage {
    let radius_x = kernel_x.len() / 2;
    let radius_y = kernel_y.len() / 2;
    let top = source.top.saturating_sub(radius_y);
    let bottom = source.bottom().saturating_add(radius_y).min(height);
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
            let left = first.columns.start.saturating_sub(radius_x);
            let right = last.columns.end.saturating_add(radius_x).min(width);
            for x in left..right {
                let mut value = 0.0;
                for (tap, &weight) in kernel_x.iter().enumerate() {
                    let column = (x as i64 + tap as i64 - radius_x as i64)
                        .clamp(0, width as i64 - 1) as usize;
                    value += weight * row[column];
                }
                horizontal[(y - source.top) * width + x] = value;
            }
        }
    }
    Coverage::from_rows(
        top,
        (top..bottom).map(|y| {
            let mut row = vec![0.0_f32; width];
            for (x, value) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for (tap, &weight) in kernel_y.iter().enumerate() {
                    let sample_y = (y as i64 + tap as i64 - radius_y as i64)
                        .clamp(0, height as i64 - 1) as usize;
                    if (source.top..source.bottom()).contains(&sample_y) {
                        sum += weight * horizontal[(sample_y - source.top) * width + x];
                    }
                }
                *value = sum as f32;
            }
            row
        }),
    )
}

/// General covariance uses integrated two-dimensional taps. The dense source
/// only stores occupied rows; coordinates still clamp to the surface boundary.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "bounded surface/tap indices and final f64-to-f32 coverage rounding"
)]
fn blur_correlated(
    source: &Coverage,
    width: usize,
    height: usize,
    radius_x: usize,
    kernel: &[Vec<f64>],
) -> Coverage {
    let radius_y = kernel.len() / 2;
    let top = source.top.saturating_sub(radius_y);
    let bottom = source.bottom().saturating_add(radius_y).min(height);
    let mut dense = vec![0.0; width * (source.bottom() - source.top)];
    for y in source.top..source.bottom() {
        for span in source.row(y) {
            for x in span.columns.clone() {
                dense[(y - source.top) * width + x] = f64::from(span.at(x));
            }
        }
    }
    Coverage::from_rows(
        top,
        (top..bottom).map(|y| {
            let mut result = vec![0.0; width];
            for (x, pixel) in result.iter_mut().enumerate() {
                let mut value = 0.0;
                for (tap_y, weights) in kernel.iter().enumerate() {
                    let sample_y = (y as i64 + tap_y as i64 - radius_y as i64)
                        .clamp(0, height as i64 - 1) as usize;
                    if !(source.top..source.bottom()).contains(&sample_y) {
                        continue;
                    }
                    let row = (sample_y - source.top) * width;
                    for (tap_x, &weight) in weights.iter().enumerate() {
                        let column = (x as i64 + tap_x as i64 - radius_x as i64)
                            .clamp(0, width as i64 - 1)
                            as usize;
                        value = weight.mul_add(dense[row + column], value);
                    }
                }
                *pixel = value as f32;
            }
            result
        }),
    )
}

/// One band's rasterization state.
struct Band<'a, S: pulp::Simd> {
    simd: S,
    /// The band's framebuffer rows.
    fb: &'a mut [[f32; 4]],
    /// Surface width.
    w: usize,
    /// Device y of the band's first row.
    y0: usize,
}

impl<S: pulp::Simd> Band<'_, S> {
    /// Shades only nonempty runs. Opaque solid spans are direct stores.
    #[expect(clippy::cast_precision_loss, reason = "surface coordinates fit f32")]
    #[expect(
        clippy::inline_always,
        reason = "the span loop must inherit the selected SIMD target features"
    )]
    #[inline(always)]
    fn draw(&mut self, stack: &mut [Vec<[f32; 4]>], coverage: &Coverage, paint: &PaintData) {
        let bottom = (self.y0 + self.fb.len() / self.w).min(coverage.bottom());
        let dst = top(&mut *self.fb, stack);
        for y in self.y0.max(coverage.top)..bottom {
            let row = (y - self.y0) * self.w;
            for span in coverage.row(y) {
                if let PaintData::Solid(color) = paint {
                    let pixels = &mut dst[row + span.columns.start..row + span.columns.end];
                    if span.samples().as_slice().is_empty() {
                        let src = color.map(|value| value * span.alpha);
                        super::composite::constant(self.simd, pixels, src);
                    } else {
                        super::composite::solid_span(self.simd, pixels, span.samples(), *color);
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

    /// Composites one unclipped glyph mask; clipped outlines are prepared draws.
    #[expect(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "mask coordinates and pixel indices are small"
    )]
    #[expect(
        clippy::inline_always,
        reason = "the glyph loop must inherit the selected SIMD target features"
    )]
    #[inline(always)]
    fn glyph(
        &mut self,
        stack: &mut [Vec<[f32; 4]>],
        slot: &std::sync::OnceLock<std::sync::Arc<crate::render::glyph::GlyphMask>>,
        ox: i32,
        oy: i32,
        paint: &PaintData,
    ) {
        let mask = slot.get().expect("glyphs resolved before compositing");
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
        let dst = top(&mut *self.fb, stack);
        for y in y_lo..y_hi {
            let py = self.y0 + y;
            let row =
                usize::try_from(py as i32 - my0).expect("clipped glyph row") * mask.w as usize;
            let start = row + usize::try_from(x_lo as i32 - mx0).expect("clipped glyph column");
            let samples = &mask.cov[start..start + x_hi - x_lo];
            let pixels = &mut dst[y * self.w + x_lo..y * self.w + x_hi];
            if let PaintData::Solid(color) = paint {
                super::composite::glyph_span(self.simd, pixels, samples, *color);
            } else {
                for (index, (pixel, &coverage)) in pixels.iter_mut().zip(samples).enumerate() {
                    if coverage > 0.0 {
                        let source = paint
                            .eval((x_lo + index) as f32 + 0.5, py as f32 + 0.5)
                            .map(|value| value * coverage);
                        *pixel = src_over(*pixel, source);
                    }
                }
            }
        }
    }

    /// Composites the popped isolation buffer onto the buffer below.
    #[expect(
        clippy::inline_always,
        reason = "the layer loop must inherit the selected SIMD target features"
    )]
    #[inline(always)]
    fn composite_isolate(
        &mut self,
        scratch: &[[f32; 4]],
        opacity: f32,
        mode: cherenkov::BlendMode,
        space: cherenkov::BlendSpace,
        stack: &mut [Vec<[f32; 4]>],
    ) {
        let dst = top(&mut *self.fb, stack);
        if mode == cherenkov::BlendMode::Normal && space == cherenkov::BlendSpace::Linear {
            super::composite::isolate(self.simd, dst, scratch, opacity);
        } else {
            for (pixel, source) in dst.iter_mut().zip(scratch) {
                *pixel = in_space(mode, space, *pixel, source.map(|value| value * opacity));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! The sparse coverage compiler checked against the independent oracle.

    use super::*;
    use crate::render::coverage::{Operand, rasterize};
    use cherenkov::FillRule;

    #[test]
    fn native_and_scalar_composition_are_bit_identical() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("test pool");
        let (width, height) = (37, 35);
        let coverage = std::sync::Arc::new(Coverage::from_rows(
            0,
            (0..height).map(|y| {
                (0..width)
                    .map(|x| f32::from(u16::try_from((x * 17 + y * 13) % 101).unwrap()) / 100.0)
                    .collect()
            }),
        ));
        for space in [
            cherenkov::BlendSpace::Linear,
            cherenkov::BlendSpace::SrgbEncoded,
        ] {
            for blend in [
                cherenkov::BlendMode::Normal,
                cherenkov::BlendMode::Multiply,
                cherenkov::BlendMode::DestIn,
            ] {
                let items = vec![
                    Item::Draw {
                        coverage: std::sync::Arc::new(Coverage::from_rows(
                            0,
                            (0..height).map(|_| vec![1.0; width]),
                        )),
                        edge_count: 0,
                        paint: PaintData::Solid([-0.25, 0.3, 1.4, 1.0]),
                    },
                    Item::Draw {
                        coverage: std::sync::Arc::new(Coverage::from_rows(
                            0,
                            (0..height).map(|_| vec![0.375; width]),
                        )),
                        edge_count: 0,
                        paint: PaintData::Solid([0.4, -0.1, 0.3, 0.75]),
                    },
                    Item::Draw {
                        coverage: coverage.clone(),
                        edge_count: 0,
                        paint: PaintData::Solid([-0.1, 0.5, 1.1, 0.75]),
                    },
                    Item::PushIsolate,
                    Item::Draw {
                        coverage: coverage.clone(),
                        edge_count: 0,
                        paint: PaintData::Solid([0.35, 0.1, 0.45, 0.5]),
                    },
                    Item::PopIsolate {
                        opacity: 0.625,
                        blend,
                        space,
                    },
                ];
                let mut scalar = vec![[0.0; 4]; width * height];
                let mut native = scalar.clone();
                let mut scratch = Vec::new();
                pool.install(|| {
                    render_bands(
                        &items,
                        [0.25; 4],
                        &mut scalar,
                        width,
                        height,
                        &mut scratch,
                        pulp::Arch::Scalar,
                    );
                    render_bands(
                        &items,
                        [0.25; 4],
                        &mut native,
                        width,
                        height,
                        &mut scratch,
                        pulp::Arch::new(),
                    );
                });
                for (scalar, native) in scalar.into_iter().zip(native) {
                    assert_eq!(
                        scalar.map(f32::to_bits),
                        native.map(f32::to_bits),
                        "{space:?}, {blend:?}"
                    );
                }
            }
        }
    }

    fn coverage_mask(edges: &[Edge], rule: FillRule, w: usize, h: usize) -> Vec<f32> {
        let coverage = rasterize(
            &[Operand {
                edges: edges.into(),
                rule,
            }],
            w,
            h,
        );
        (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| coverage.at(x, y))
            .collect()
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
            (4.25, 4.125),
            (20.75, 20.875),
            (4.25, 20.875),
            (20.75, 4.125),
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
                let stroke = kurbo::Stroke::new(3.75)
                    .with_join(join)
                    .with_caps(cap)
                    .with_miter_limit(12.0);
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
            Operand {
                edges: caster.clone().into(),
                rule: FillRule::NonZero,
            },
            Operand {
                edges: clip.clone().into(),
                rule: FillRule::NonZero,
            },
        ];
        let segments = |edges: &[Edge]| {
            edges
                .iter()
                .map(|e| {
                    (
                        f64::from(e.x0),
                        f64::from(e.y0),
                        f64::from(e.x1),
                        f64::from(e.y1),
                    )
                })
                .collect::<Vec<_>>()
        };
        let intersection = cherenkov_oracle::clip::intersect_edges(
            &segments(&caster),
            cherenkov_scene::FillRule::NonZero,
            &segments(&clip),
        );
        let mut oracle = cherenkov_oracle::coverage::Coverage::new(8, 8);
        for (x0, y0, x1, y1) in intersection {
            oracle.add_line(x0, y0, x1, y1);
        }
        let exact = oracle.finish(cherenkov_scene::FillRule::NonZero);
        let source = rasterize(&operands, 8, 8);
        for sigma in [0.0, 1.25] {
            let expected = cherenkov_oracle::shadow::gaussian_blur(
                &exact,
                8,
                8,
                sigma,
                kurbo::Affine::IDENTITY,
            );
            let actual = blur_coverage(&source, 8, 8, sigma, kurbo::Affine::IDENTITY);
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
