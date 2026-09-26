// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sparse exact coverage. Geometry is resolved before any pixel is shaded.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use cherenkov::FillRule;

use super::raster::Edge;

/// Boolean operation on the first two operands; remaining operands are clips.
#[derive(Clone, Copy)]
pub enum Combine {
    /// Intersect every operand.
    Intersection,
    /// Union the caster and its contour spread band, then intersect clips.
    Union,
    /// Remove the contour spread band from the caster, then intersect clips.
    Difference,
}

impl Combine {
    fn inside(self, outside: usize, winding: &[i32], rules: &[FillRule]) -> bool {
        match self {
            Self::Intersection => outside == 0,
            Self::Union | Self::Difference => {
                let first = inside(winding[0], rules[0]);
                let second = inside(winding[1], rules[1]);
                let clips_inside = outside == usize::from(!first) + usize::from(!second);
                clips_inside
                    && if matches!(self, Self::Union) {
                        first || second
                    } else {
                        first && !second
                    }
            }
        }
    }
}

/// One operand of a geometric intersection.
#[derive(Clone, Debug)]
pub struct Operand {
    /// Closed directed device-space boundary, including horizontal edges.
    pub edges: Arc<[Edge]>,
    /// Interior predicate for this operand.
    pub rule: FillRule,
}

/// One horizontal run, constant or sampled. Empty runs are never stored.
#[derive(Clone, Debug)]
pub struct Span {
    /// Device-space columns.
    pub columns: Range<usize>,
    /// Area of the covered part of each pixel.
    pub alpha: f32,
    /// Varying coverage, or empty for a constant run.
    pub samples: Vec<f32>,
}

impl Span {
    /// Coverage in this run at a device-space column.
    pub fn at(&self, x: usize) -> f32 {
        if self.samples.is_empty() {
            self.alpha
        } else {
            self.samples[x - self.columns.start]
        }
    }
}

/// Compressed rows, with no storage or shading work for empty pixels.
#[derive(Debug, Default)]
pub struct Coverage {
    /// Source boundary edges resolved for this coverage.
    pub edge_count: usize,
    /// First stored device-space row.
    pub top: usize,
    rows: Vec<Range<usize>>,
    spans: Vec<Span>,
}

impl Coverage {
    /// Whether the field has any covered pixels.
    pub const fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Runs for a device-space row; rows outside the geometry are empty.
    pub fn row(&self, y: usize) -> &[Span] {
        y.checked_sub(self.top)
            .and_then(|i| self.rows.get(i))
            .map_or(&[], |range| &self.spans[range.clone()])
    }

    /// First row after the stored rows.
    pub const fn bottom(&self) -> usize {
        self.top + self.rows.len()
    }

    /// Coverage at one pixel for oracle comparisons.
    #[cfg(test)]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        let row = self.row(y);
        let i = row.partition_point(|span| span.columns.end <= x);
        row.get(i)
            .filter(|span| span.columns.contains(&x))
            .map_or(0.0, |span| span.at(x))
    }

    /// Allocation bytes retained by this coverage.
    pub fn bytes(&self) -> usize {
        self.rows.capacity() * size_of::<Range<usize>>()
            + self.spans.capacity() * size_of::<Span>()
            + self
                .spans
                .iter()
                .map(|span| span.samples.capacity() * size_of::<f32>())
                .sum::<usize>()
    }

    fn push(&mut self, x0: usize, x1: usize, alpha: f32, row_start: usize) {
        if x0 >= x1 || alpha <= 0.0 {
            return;
        }
        let in_row = self.spans.len() > row_start;
        if let Some(last) = self.spans.last_mut()
            && in_row
            && last.columns.end == x0
        {
            if last.samples.is_empty() && last.alpha.to_bits() == alpha.to_bits() {
                last.columns.end = x1;
                return;
            }
            // Adjacent antialiased pixels share a sample allocation. Long
            // constant interiors remain runs, never expanded into samples.
            if alpha < 1.0
                && x1 == x0 + 1
                && (!last.samples.is_empty() || (last.alpha < 1.0 && last.columns.len() == 1))
            {
                if last.samples.is_empty() {
                    last.samples.push(last.alpha);
                }
                last.samples.push(alpha);
                last.columns.end = x1;
                return;
            }
        }
        self.spans.push(Span {
            columns: x0..x1,
            alpha,
            samples: Vec::new(),
        });
    }

    /// Compresses already evaluated rows, without changing their values.
    pub fn from_rows(top: usize, rows: impl Iterator<Item = Vec<f32>>) -> Self {
        let mut result = Self {
            top,
            ..Self::default()
        };
        for row in rows {
            let start = result.spans.len();
            let mut x = 0;
            while x < row.len() {
                if row[x] <= 0.0 {
                    x += 1;
                    continue;
                }
                let first = x;
                while x < row.len() && row[x] > 0.0 {
                    x += 1;
                }
                let alpha = row[first];
                let samples = if row[first..x].iter().all(|v| v.to_bits() == alpha.to_bits()) {
                    Vec::new()
                } else {
                    row[first..x].to_vec()
                };
                result.spans.push(Span {
                    columns: first..x,
                    alpha,
                    samples,
                });
            }
            result.rows.push(start..result.spans.len());
        }
        result
    }
}

/// Exact-bit keys and equality-checked lookup: hash collisions cannot reuse
/// different geometry. The renderer owns this cache, including eviction.
pub struct CoverageCache {
    entries: HashMap<Vec<u32>, Arc<Coverage>>,
    bytes: usize,
    budget: usize,
}

impl CoverageCache {
    /// Cache bounded by retained key and coverage allocation bytes.
    pub fn new(budget: usize) -> Self {
        Self {
            entries: HashMap::new(),
            bytes: 0,
            budget,
        }
    }

    /// Retained payload bytes.
    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    /// Releases retained geometry coverage.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }

    /// Resolve or prepare a coverage value. Oversized values are used by the
    /// current frame but are not retained by the cache.
    pub fn get_or_insert(
        &mut self,
        key: Vec<u32>,
        make: impl FnOnce() -> Coverage,
    ) -> Arc<Coverage> {
        if let Some(value) = self.entries.get(&key) {
            return value.clone();
        }
        let value = Arc::new(make());
        let bytes = key.capacity() * size_of::<u32>() + value.bytes();
        if bytes <= self.budget {
            if self.bytes + bytes > self.budget {
                self.clear();
            }
            self.bytes += bytes;
            self.entries.insert(key, value.clone());
        }
        value
    }

    /// Coverage of the intersection of all operands, cropped to the surface.
    pub fn intersection(&mut self, operands: &[Operand], w: usize, h: usize) -> Arc<Coverage> {
        let key = geometry_key(operands, w, h);
        self.get_or_insert(key, || rasterize(operands, w, h))
    }
}

/// Appends unambiguous operand boundaries, rules and coordinate bits.
pub fn geometry_key(operands: &[Operand], w: usize, h: usize) -> Vec<u32> {
    let mut key = vec![
        0,
        u32::try_from(w).expect("surface width"),
        u32::try_from(h).expect("surface height"),
        u32::try_from(operands.len()).expect("operand count"),
    ];
    for operand in operands {
        key.push(u32::try_from(operand.edges.len()).expect("edge count"));
        key.push(u32::from(operand.rule == FillRule::EvenOdd));
        for edge in &*operand.edges {
            key.extend([
                edge.x0.to_bits(),
                edge.y0.to_bits(),
                edge.x1.to_bits(),
                edge.y1.to_bits(),
            ]);
        }
    }
    key
}

#[derive(Clone, Copy)]
struct Line {
    top: f64,
    bottom: f64,
    x: f64,
    end_x: f64,
    slope: f64,
    dir: i32,
    operand: usize,
    left: f64,
    right: f64,
}

impl Line {
    #[expect(
        clippy::suboptimal_flops,
        clippy::float_cmp,
        reason = "exact endpoint events preserve connectivity; avoid software FMA on baseline targets"
    )]
    fn at(self, y: f64) -> f64 {
        if y == self.bottom {
            self.end_x
        } else {
            self.x + (y - self.top) * self.slope
        }
    }
}

const fn inside(winding: i32, rule: FillRule) -> bool {
    match rule {
        FillRule::NonZero => winding != 0,
        FillRule::EvenOdd => winding % 2 != 0,
    }
}

/// One row's reused event storage. No allocation per strip or trapezoid.
#[derive(Default)]
struct RowScratch {
    bounds: Vec<f64>,
    order: Vec<usize>,
    winding: Vec<i32>,
    delta: Vec<(usize, f64)>,
}

impl RowScratch {
    /// Add the integral of the half-plane to the right of a directed edge.
    /// Only crossed columns receive deltas; the interior is a constant run.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "surface-clamped column indices; analytic trapezoid integration without software FMA"
    )]
    fn boundary(&mut self, xa: f64, xb: f64, height: f64, w: usize) {
        let left = xa.min(xb);
        let right = xa.max(xb);
        let start = left.floor().clamp(0.0, w as f64) as usize;
        let end = right.ceil().clamp(0.0, w as f64) as usize;
        let mut previous = 0.0;
        for x in start..end {
            let a = x as f64;
            let b = a + 1.0;
            // Integral over x of clamp((x-left)/(right-left), 0, 1).
            // Subtracting large squared coordinates would lose precision.
            let lo = a.max(left);
            let hi = b.min(right);
            let ramp = if hi > lo {
                (hi - lo) * ((lo - left) + (hi - left)) / (2.0 * (right - left))
            } else {
                0.0
            };
            let area = height * (ramp + (b - right.max(a)).max(0.0));
            self.delta.push((x, area - previous));
            previous = area;
        }
        self.delta.push((end, height - previous));
    }

    /// Resolve one x-connected group. The baseline winding is constant in y:
    /// no boundary intersects the vertical gap separating it from its neighbour.
    fn group(
        &mut self,
        lines: &[Line],
        baseline: &[i32],
        rules: &[FillRule],
        w: usize,
        combine: Combine,
    ) {
        self.bounds.clear();
        for line in lines {
            self.bounds.extend([line.top, line.bottom]);
        }
        // Broad phase: lines are sorted by their minimum x in this row.
        // Only overlapping x projections can cross. Collinear segments need
        // endpoint events only; their signed deltas cancel exactly.
        for (index, first) in lines.iter().enumerate() {
            let right = first.right;
            for second in &lines[index + 1..] {
                if second.left > right {
                    break;
                }
                let top = first.top.max(second.top);
                let bottom = first.bottom.min(second.bottom);
                let slope = first.slope - second.slope;
                if top < bottom && slope != 0.0 {
                    let y = top + (second.at(top) - first.at(top)) / slope;
                    if y > top && y < bottom {
                        self.bounds.push(y);
                    }
                }
            }
        }
        self.bounds.sort_unstable_by(f64::total_cmp);
        self.bounds.dedup();
        for index in 1..self.bounds.len() {
            let top = self.bounds[index - 1];
            let bottom = self.bounds[index];
            let middle = top.midpoint(bottom);
            self.order.clear();
            self.order.extend((0..lines.len()).filter(|&crossing| {
                lines[crossing].top <= middle && lines[crossing].bottom > middle
            }));
            self.order.sort_unstable_by(|&first, &second| {
                lines[first].at(middle).total_cmp(&lines[second].at(middle))
            });
            self.winding.clear();
            self.winding.extend_from_slice(baseline);
            let mut outside = self
                .winding
                .iter()
                .zip(rules)
                .filter(|&(wind, rule)| !inside(*wind, *rule))
                .count();
            for crossing in 0..self.order.len() {
                let line = lines[self.order[crossing]];
                let was_inside = combine.inside(outside, &self.winding, rules);
                let wind = &mut self.winding[line.operand];
                outside -= usize::from(!inside(*wind, rules[line.operand]));
                *wind += line.dir;
                outside += usize::from(!inside(*wind, rules[line.operand]));
                let is_inside = combine.inside(outside, &self.winding, rules);
                if was_inside != is_inside {
                    let sign = if is_inside { 1.0 } else { -1.0 };
                    self.boundary(line.at(top), line.at(bottom), sign * (bottom - top), w);
                }
            }
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "final coverage rounds once to framebuffer precision"
    )]
    fn finish(&mut self, result: &mut Coverage, w: usize) {
        self.delta.sort_unstable_by_key(|&(x, _)| x);
        let row_start = result.spans.len();
        let mut area = 0.0_f64;
        let mut previous = 0;
        let mut i = 0;
        while i < self.delta.len() {
            let x = self.delta[i].0;
            result.push(previous, x, area.clamp(0.0, 1.0) as f32, row_start);
            while i < self.delta.len() && self.delta[i].0 == x {
                area += self.delta[i].1;
                i += 1;
            }
            previous = x;
        }
        result.push(previous, w, area.clamp(0.0, 1.0) as f32, row_start);
        result.rows.push(row_start..result.spans.len());
        self.delta.clear();
    }
}

/// Split edges at integer rows while preserving exact shared endpoints.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "exact horizontal-edge classification; surface-clamped indices fit f64"
)]
fn row_lines(operands: &[Operand], top: usize, bottom: usize) -> Vec<Vec<Line>> {
    let mut rows = vec![Vec::new(); bottom - top];
    for (operand, shape) in operands.iter().enumerate() {
        for edge in &*shape.edges {
            let (x0, y0, x1, y1) = (
                f64::from(edge.x0),
                f64::from(edge.y0),
                f64::from(edge.x1),
                f64::from(edge.y1),
            );
            let lo = y0.min(y1);
            let hi = y0.max(y1);
            let line = Line {
                top: lo,
                bottom: hi,
                x: if y0 < y1 { x0 } else { x1 },
                end_x: if y0 < y1 { x1 } else { x0 },
                slope: if y0 == y1 { 0.0 } else { (x1 - x0) / (y1 - y0) },
                dir: if y0 < y1 { 1 } else { -1 },
                operand,
                left: x0.min(x1),
                right: x0.max(x1),
            };
            let first = (lo.floor() as usize).clamp(top, bottom);
            let last = (hi.ceil() as usize).clamp(top, bottom);
            for y in first..last {
                let start = lo.max(y as f64);
                let end = hi.min((y + 1) as f64);
                let (left, right) = if y0 == y1 {
                    (line.left, line.right)
                } else {
                    (
                        line.at(start).min(line.at(end)),
                        line.at(start).max(line.at(end)),
                    )
                };
                rows[y - top].push(Line {
                    top: start,
                    bottom: end,
                    x: line.at(start),
                    end_x: line.at(end),
                    left,
                    right,
                    ..line
                });
            }
        }
    }
    rows
}

/// Exact area of the intersection of every operand.
pub fn rasterize(operands: &[Operand], w: usize, h: usize) -> Coverage {
    rasterize_combined(operands, w, h, Combine::Intersection)
}

/// Exact area of the intersection, relative to the supplied flattened edges.
/// Endpoint and crossing events resolve winding *before* integrating area.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "surface-clamped row indices and f32 input coordinates fit exactly in f64"
)]
pub fn rasterize_combined(operands: &[Operand], w: usize, h: usize, combine: Combine) -> Coverage {
    if operands.is_empty() || w == 0 || h == 0 {
        return Coverage::default();
    }
    // Intersect operand y extents before allocating row buckets.
    let mut top = 0;
    let mut bottom = h;
    for operand in operands
        .iter()
        .filter(|_| matches!(combine, Combine::Intersection))
    {
        let lo = operand
            .edges
            .iter()
            .map(|e| e.y0.min(e.y1))
            .fold(f32::INFINITY, f32::min);
        let hi = operand
            .edges
            .iter()
            .map(|e| e.y0.max(e.y1))
            .fold(f32::NEG_INFINITY, f32::max);
        top = top.max((lo.floor() as usize).min(h));
        bottom = bottom.min((hi.ceil() as usize).min(h));
    }
    if top >= bottom {
        return Coverage::default();
    }
    let mut rows = row_lines(operands, top, bottom);
    let rules: Vec<_> = operands.iter().map(|operand| operand.rule).collect();
    let mut baseline = vec![0; operands.len()];
    let mut scratch = RowScratch::default();
    let mut result = Coverage {
        top,
        edge_count: operands.last().map_or(0, |operand| operand.edges.len()),
        ..Coverage::default()
    };
    for (row, lines) in rows.iter_mut().enumerate() {
        lines.sort_unstable_by(|a, b| a.left.total_cmp(&b.left));
        baseline.fill(0);
        let mut first = 0;
        while first < lines.len() {
            let mut end = first + 1;
            let mut right = lines[first].right;
            while end < lines.len() && lines[end].left <= right {
                right = right.max(lines[end].right);
                end += 1;
            }
            scratch.group(&lines[first..end], &baseline, &rules, w, combine);
            let middle = (top + row) as f64 + 0.5;
            for line in &lines[first..end] {
                if line.top <= middle && line.bottom > middle {
                    baseline[line.operand] += line.dir;
                }
            }
            first = end;
        }
        scratch.finish(&mut result, w);
    }
    if let Some(first) = result.rows.iter().position(|row| !row.is_empty()) {
        let last = result
            .rows
            .iter()
            .rposition(|row| !row.is_empty())
            .expect("nonempty field");
        result.rows.truncate(last + 1);
        drop(result.rows.drain(..first));
        result.top += first;
    } else {
        result.rows.clear();
        result.top = 0;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn polygon(points: &[(f32, f32)], rule: FillRule) -> Operand {
        let edges = points
            .iter()
            .enumerate()
            .map(|(i, &(x0, y0))| {
                let (x1, y1) = points[(i + 1) % points.len()];
                Edge { x0, y0, x1, y1 }
            })
            .collect();
        Operand { edges, rule }
    }

    fn segments(operand: &Operand) -> Vec<cherenkov_oracle::clip::Segment> {
        operand
            .edges
            .iter()
            .map(|edge| {
                (
                    f64::from(edge.x0),
                    f64::from(edge.y0),
                    f64::from(edge.x1),
                    f64::from(edge.y1),
                )
            })
            .collect()
    }

    fn oracle(operands: &[Operand], w: usize, h: usize) -> Vec<f64> {
        let mut edges = segments(&operands[0]);
        let rule = match operands[0].rule {
            FillRule::NonZero => cherenkov_scene::FillRule::NonZero,
            FillRule::EvenOdd => cherenkov_scene::FillRule::EvenOdd,
        };
        for operand in &operands[1..] {
            edges = cherenkov_oracle::clip::intersect_edges(&edges, rule, &segments(operand));
        }
        let mut coverage = cherenkov_oracle::coverage::Coverage::new(w, h);
        for (x0, y0, x1, y1) in edges {
            coverage.add_line(x0, y0, x1, y1);
        }
        coverage.finish(rule)
    }

    fn compare(operands: &[Operand], w: usize, h: usize) {
        let got = rasterize(operands, w, h);
        for (i, want) in oracle(operands, w, h).into_iter().enumerate() {
            let actual = f64::from(got.at(i % w, i / w));
            assert!(
                (actual - want).abs() < 2e-6,
                "pixel ({}, {}): {actual} vs {want}",
                i % w,
                i / w
            );
        }
    }

    #[test]
    fn spread_predicates_resolve_partial_pixels_before_integration() {
        let lower = polygon(&[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], FillRule::NonZero);
        let upper = polygon(&[(0.0, 1.0), (1.0, 0.0), (1.0, 1.0)], FillRule::NonZero);
        for (second, union, difference) in [(lower.clone(), 0.5, 0.0), (upper, 1.0, 0.5)] {
            let operands = [lower.clone(), second];
            let merged = rasterize_combined(&operands, 1, 1, Combine::Union);
            let cut = rasterize_combined(&operands, 1, 1, Combine::Difference);
            assert!((merged.at(0, 0) - union).abs() < 1e-7);
            assert!((cut.at(0, 0) - difference).abs() < 1e-7);
            let clipped = rasterize_combined(
                &[operands[0].clone(), operands[1].clone(), lower.clone()],
                1,
                1,
                Combine::Union,
            );
            assert!((clipped.at(0, 0) - 0.5).abs() < 1e-7);
        }
    }

    #[test]
    fn nested_non_rectangular_clips_intersect_inside_pixels() {
        let shape = polygon(
            &[(1.125, 1.125), (21.75, 2.375), (3.25, 21.875)],
            FillRule::NonZero,
        );
        let first = polygon(
            &[
                (2.625, 0.75),
                (20.875, 5.125),
                (16.625, 21.625),
                (0.875, 16.125),
            ],
            FillRule::NonZero,
        );
        let second = polygon(
            &[(0.25, 3.75), (22.25, 1.25), (20.75, 20.75)],
            FillRule::NonZero,
        );
        compare(&[shape, first, second], 24, 24);
    }

    #[test]
    fn touching_half_pixel_clips_have_zero_intersection() {
        let first = polygon(&[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], FillRule::NonZero);
        let second = polygon(&[(0.0, 1.0), (1.0, 0.0), (1.0, 1.0)], FillRule::NonZero);
        compare(&[first.clone(), second.clone()], 1, 1);
        let joint = rasterize(&[first.clone(), second.clone()], 1, 1).at(0, 0);
        let product = rasterize(&[first], 1, 1).at(0, 0) * rasterize(&[second], 1, 1).at(0, 0);
        assert!(joint.abs() < 1e-7);
        assert!((product - 0.25).abs() < 1e-7);
    }

    #[test]
    fn duplicate_clips_preserve_area_and_each_operand_keeps_its_rule() {
        let first = polygon(&[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], FillRule::NonZero);
        compare(&[first.clone(), first.clone(), first.clone()], 1, 1);
        assert!((rasterize(&[first.clone(), first.clone()], 1, 1).at(0, 0) - 0.5).abs() < 1e-7);
        let doubled: Arc<[Edge]> = first
            .edges
            .iter()
            .chain(first.edges.iter())
            .copied()
            .collect();
        let even = Operand {
            edges: doubled,
            rule: FillRule::EvenOdd,
        };
        assert!(rasterize(&[first, even], 1, 1).at(0, 0).abs() < 1e-7);
    }

    #[test]
    fn fractional_horizontal_edges_connect_winding_groups() {
        // Dropping horizontal edges would make the winding carried through
        // the gap depend on y and incorrectly fill the upper/lower rows.
        let shape = polygon(
            &[
                (-20.25, 0.25),
                (20.75, 0.25),
                (20.75, 17.75),
                (-20.25, 17.75),
            ],
            FillRule::NonZero,
        );
        compare(&[shape], 24, 24);
        let shallow = polygon(
            &[
                (-30.5, 4.125),
                (40.25, 4.875),
                (40.25, 5.125),
                (-30.5, 4.375),
            ],
            FillRule::NonZero,
        );
        compare(&[shallow], 24, 24);
    }

    #[test]
    fn winding_orientation_and_large_values_do_not_fold_signed_area() {
        let contour = polygon(
            &[(2.25, 1.125), (21.75, 3.875), (4.25, 20.625)],
            FillRule::NonZero,
        );
        let repeated: Arc<[Edge]> = (0..8).flat_map(|_| contour.edges.iter().copied()).collect();
        for rule in [FillRule::NonZero, FillRule::EvenOdd] {
            compare(
                &[Operand {
                    edges: repeated.clone(),
                    rule,
                }],
                24,
                24,
            );
        }
        let reversed = Operand {
            edges: contour
                .edges
                .iter()
                .map(|e| Edge {
                    x0: e.x1,
                    y0: e.y1,
                    x1: e.x0,
                    y1: e.y0,
                })
                .collect(),
            rule: FillRule::NonZero,
        };
        compare(&[reversed], 24, 24);
    }

    #[test]
    fn cache_hits_eviction_and_disabled_cache_are_bit_identical() {
        let shape = polygon(
            &[
                (0.125, 0.25),
                (10.75, 0.25),
                (10.75, 12.875),
                (0.125, 12.875),
            ],
            FillRule::NonZero,
        );
        let mut cache = CoverageCache::new(1024 * 1024);
        let first = cache.intersection(std::slice::from_ref(&shape), 24, 24);
        let second = cache.intersection(std::slice::from_ref(&shape), 24, 24);
        assert!(Arc::ptr_eq(&first, &second));
        cache.clear();
        let third = cache.intersection(std::slice::from_ref(&shape), 24, 24);
        let uncached = CoverageCache::new(0).intersection(&[shape], 24, 24);
        for y in 0..24 {
            for x in 0..24 {
                assert_eq!(first.at(x, y).to_bits(), third.at(x, y).to_bits());
                assert_eq!(first.at(x, y).to_bits(), uncached.at(x, y).to_bits());
            }
        }
    }
}
