//! Sparse pixel areas of geometric intersections. Each operand's winding is
//! resolved before integration; only the final area is rounded to f32.

use std::ops::Range;
use std::sync::Arc;

use cherenkov::FillRule;

use super::raster::Edge;

/// A closed device-space boundary, including horizontal connecting edges.
#[derive(Clone, Debug, PartialEq)]
pub struct Operand {
    /// Directed polygon edges.
    pub edges: Arc<[Edge<f64>]>,
    /// The operand's own interior predicate.
    pub rule: FillRule,
}

/// One nonempty run in a coverage row.
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    /// First device column.
    pub x: u32,
    /// Number of pixels.
    pub len: u32,
    /// Constant area or an offset into the coverage's shared sample array.
    pub kind: SpanKind,
}

/// Storage for a run; constant interiors never expand to samples.
#[derive(Clone, Debug, PartialEq)]
pub enum SpanKind {
    /// Identical area for every pixel.
    Constant(f32),
    /// Consecutive, strictly positive f32 samples.
    Samples(u32),
}

/// Flat row and sample storage, with no allocation per row or span.
#[derive(Debug, Default, PartialEq)]
pub struct Coverage {
    /// First stored device row.
    pub top: usize,
    /// Number of input edges, for frame statistics.
    pub edge_count: usize,
    row_offsets: Vec<u32>,
    spans: Vec<Span>,
    samples: Vec<f32>,
}

impl Coverage {
    /// Runs at a device row, empty outside the field.
    pub fn row(&self, y: usize) -> &[Span] {
        let Some(row) = y.checked_sub(self.top) else {
            return &[];
        };
        let Some(offsets) = self.row_offsets.get(row..row + 2) else {
            return &[];
        };
        &self.spans[offsets[0] as usize..offsets[1] as usize]
    }

    /// Exact stored area at a device pixel.
    pub fn at(&self, x: usize, y: usize) -> f32 {
        let spans = self.row(y);
        let index = spans.partition_point(|span| (span.x + span.len) as usize <= x);
        let Some(span) = spans.get(index).filter(|span| span.x as usize <= x) else {
            return 0.0;
        };
        match span.kind {
            SpanKind::Constant(alpha) => alpha,
            SpanKind::Samples(offset) => self.samples[offset as usize + x - span.x as usize],
        }
    }

    /// Sampled areas for this span; constants have no sample allocation.
    pub fn samples(&self, span: &Span) -> &[f32] {
        match span.kind {
            SpanKind::Constant(_) => &[],
            SpanKind::Samples(offset) => {
                &self.samples[offset as usize..(offset + span.len) as usize]
            }
        }
    }

    /// Heap bytes retained by the sparse field.
    pub const fn bytes(&self) -> usize {
        self.row_offsets.capacity() * size_of::<u32>()
            + self.spans.capacity() * size_of::<Span>()
            + self.samples.capacity() * size_of::<f32>()
    }

    fn push(&mut self, x0: usize, x1: usize, alpha: f32, row_start: usize) {
        if x0 >= x1 || alpha <= 0.0 {
            return;
        }
        let x = u32::try_from(x0).expect("surface column");
        let len = u32::try_from(x1 - x0).expect("surface span");
        let same_row = self.spans.len() > row_start;
        if let Some(last) = self.spans.last_mut()
            && same_row
            && last.x + last.len == x
        {
            match last.kind {
                SpanKind::Constant(previous) if previous.to_bits() == alpha.to_bits() => {
                    last.len += len;
                    return;
                }
                SpanKind::Constant(previous)
                    if previous < 1.0 && alpha < 1.0 && last.len == 1 && len == 1 =>
                {
                    last.kind = SpanKind::Samples(
                        u32::try_from(self.samples.len()).expect("sample offset"),
                    );
                    self.samples.extend([previous, alpha]);
                    last.len += 1;
                    return;
                }
                SpanKind::Samples(_) if alpha < 1.0 && len == 1 => {
                    self.samples.push(alpha);
                    last.len += 1;
                    return;
                }
                _ => {}
            }
        }
        self.spans.push(Span {
            x,
            len,
            kind: SpanKind::Constant(alpha),
        });
    }
}

#[derive(Clone, Copy, Default)]
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

/// Count and scatter row-clipped edges into one allocation. Horizontal
/// boundaries at fractional y connect x groups even though they add no area.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "surface-clamped row indices; horizontal edges require exact classification"
)]
fn row_lines(
    operands: &[Operand],
    top: usize,
    bottom: usize,
    offsets: &mut Vec<usize>,
    lines: &mut Vec<Line>,
    cursors: &mut Vec<usize>,
) {
    offsets.clear();
    offsets.resize(bottom - top + 1, 0);
    for shape in operands {
        for edge in &*shape.edges {
            let first = (edge.y0.min(edge.y1).floor() as usize).clamp(top, bottom);
            let last = (edge.y0.max(edge.y1).ceil() as usize).clamp(top, bottom);
            for count in &mut offsets[(first - top + 1)..=(last - top)] {
                *count += 1;
            }
        }
    }
    for row in 1..offsets.len() {
        offsets[row] += offsets[row - 1];
    }
    lines.clear();
    lines.resize(offsets[bottom - top], Line::default());
    cursors.clear();
    cursors.extend_from_slice(offsets);
    for (operand, shape) in operands.iter().enumerate() {
        for edge in &*shape.edges {
            let (x0, y0, x1, y1) = (edge.x0, edge.y0, edge.x1, edge.y1);
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
                lines[cursors[y - top]] = Line {
                    top: start,
                    bottom: end,
                    x: line.at(start),
                    end_x: line.at(end),
                    left,
                    right,
                    ..line
                };
                cursors[y - top] += 1;
            }
        }
    }
}

/// Single-owner compiler scratch, reused across draws and bands. Only the
/// requested rows are materialized, including when a target streams bands.
#[derive(Default)]
pub struct Compiler {
    offsets: Vec<usize>,
    cursors: Vec<usize>,
    lines: Vec<Line>,
    rules: Vec<FillRule>,
    baseline: Vec<i32>,
    scratch: RowScratch,
    result: Coverage,
}

impl Compiler {
    /// Compile the geometric intersection into this compiler's sparse storage.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "surface-clamped row coordinates fit exactly in f64"
    )]
    pub fn compile(&mut self, operands: &[Operand], w: usize, rows: Range<usize>) -> &Coverage {
        self.result.row_offsets.clear();
        self.result.row_offsets.push(0);
        self.result.spans.clear();
        self.result.samples.clear();
        self.result.edge_count = operands.first().map_or(0, |operand| operand.edges.len());
        if operands.is_empty() || w == 0 || rows.is_empty() {
            return &self.result;
        }
        let mut top = rows.start;
        let mut bottom = rows.end;
        for operand in operands {
            let lo = operand
                .edges
                .iter()
                .map(|e| e.y0.min(e.y1))
                .fold(f64::INFINITY, f64::min);
            let hi = operand
                .edges
                .iter()
                .map(|e| e.y0.max(e.y1))
                .fold(f64::NEG_INFINITY, f64::max);
            top = top.max(lo.floor() as usize);
            bottom = bottom.min(hi.ceil() as usize);
        }
        self.result.top = top;
        if top >= bottom {
            return &self.result;
        }
        row_lines(
            operands,
            top,
            bottom,
            &mut self.offsets,
            &mut self.lines,
            &mut self.cursors,
        );
        self.rules.clear();
        self.rules
            .extend(operands.iter().map(|operand| operand.rule));
        self.baseline.resize(operands.len(), 0);
        for row in 0..bottom - top {
            let lines = &mut self.lines[self.offsets[row]..self.offsets[row + 1]];
            lines.sort_unstable_by(|a, b| a.left.total_cmp(&b.left));
            self.baseline.fill(0);
            let mut first = 0;
            while first < lines.len() {
                let mut end = first + 1;
                let mut right = lines[first].right;
                // Groups share a pixel whenever their rounded x envelopes
                // touch. Between groups the entire pixel is an interior or
                // exterior, known from winding without numerical integration.
                while end < lines.len()
                    && lines[end].left.floor().clamp(0.0, w as f64)
                        <= right.ceil().clamp(0.0, w as f64)
                {
                    right = right.max(lines[end].right);
                    end += 1;
                }
                self.scratch
                    .group(&lines[first..end], &self.baseline, &self.rules, w);
                let middle = (top + row) as f64 + 0.5;
                for line in &lines[first..end] {
                    if line.top <= middle && line.bottom > middle {
                        self.baseline[line.operand] += line.dir;
                    }
                }
                self.scratch.interiors.push((
                    right.ceil().clamp(0.0, w as f64) as usize,
                    self.baseline
                        .iter()
                        .zip(&self.rules)
                        .all(|(&wind, &rule)| inside(wind, rule)),
                ));
                first = end;
            }
            self.scratch.finish(&mut self.result, w);
        }
        &self.result
    }
}

/// Compile a retained clip field; draw compilation requests only band rows.
pub fn rasterize(operands: &[Operand], w: usize, h: usize) -> Coverage {
    let mut compiler = Compiler::default();
    compiler.compile(operands, w, 0..h);
    compiler.result
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
    starts: Vec<usize>,
    order: Vec<usize>,
    winding: Vec<i32>,
    delta: Vec<(usize, f64)>,
    /// At each group's right pixel boundary, winding determines the exact
    /// full-row area. This prevents integration residue from filling gaps.
    interiors: Vec<(usize, bool)>,
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
    fn group(&mut self, lines: &[Line], baseline: &[i32], rules: &[FillRule], w: usize) {
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
        self.starts.clear();
        self.starts.extend(0..lines.len());
        self.starts
            .sort_unstable_by(|&a, &b| lines[a].top.total_cmp(&lines[b].top));
        let mut next = 0;
        self.order.clear();
        for index in 1..self.bounds.len() {
            let top = self.bounds[index - 1];
            let bottom = self.bounds[index];
            let middle = top.midpoint(bottom);
            self.order
                .retain(|&crossing| lines[crossing].bottom > middle);
            while next < self.starts.len() && lines[self.starts[next]].top <= middle {
                let crossing = self.starts[next];
                if lines[crossing].bottom > middle {
                    self.order.push(crossing);
                }
                next += 1;
            }
            // Preserve the original boundary order at coincident crossings.
            // The active set changes only at endpoints; scanning every edge
            // at every strip made a finely flattened curve quadratic.
            self.order.sort_unstable();
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
                let was_inside = outside == 0;
                let wind = &mut self.winding[line.operand];
                outside -= usize::from(!inside(*wind, rules[line.operand]));
                *wind += line.dir;
                outside += usize::from(!inside(*wind, rules[line.operand]));
                let is_inside = outside == 0;
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
        let mut interior = 0;
        while i < self.delta.len() || interior < self.interiors.len() {
            let x = self
                .delta
                .get(i)
                .map_or(w, |event| event.0)
                .min(self.interiors.get(interior).map_or(w, |event| event.0));
            result.push(previous, x, area.clamp(0.0, 1.0) as f32, row_start);
            while i < self.delta.len() && self.delta[i].0 == x {
                area += self.delta[i].1;
                i += 1;
            }
            while interior < self.interiors.len() && self.interiors[interior].0 == x {
                area = f64::from(self.interiors[interior].1);
                interior += 1;
            }
            previous = x;
        }
        result.push(previous, w, area.clamp(0.0, 1.0) as f32, row_start);
        result
            .row_offsets
            .push(u32::try_from(result.spans.len()).expect("span index"));
        self.delta.clear();
        self.interiors.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn polygon(points: &[(f64, f64)], rule: FillRule) -> Operand {
        Operand {
            edges: points
                .iter()
                .enumerate()
                .map(|(index, &(x0, y0))| {
                    let (x1, y1) = points[(index + 1) % points.len()];
                    Edge { x0, y0, x1, y1 }
                })
                .collect(),
            rule,
        }
    }

    #[test]
    fn intersection_is_not_the_product_of_pixel_areas() {
        let lower = polygon(&[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], FillRule::NonZero);
        let upper = polygon(&[(0.0, 1.0), (1.0, 0.0), (1.0, 1.0)], FillRule::NonZero);
        let first = rasterize(std::slice::from_ref(&lower), 1, 1);
        let second = rasterize(std::slice::from_ref(&upper), 1, 1);
        assert_eq!(first.at(0, 0), 0.5);
        assert_eq!(second.at(0, 0), 0.5);
        assert_eq!(rasterize(&[lower.clone(), upper], 1, 1).at(0, 0), 0.0);
        assert_eq!(rasterize(&[lower.clone(), lower], 1, 1).at(0, 0), 0.5);
    }

    #[test]
    fn each_operand_resolves_its_own_winding() {
        let shape = polygon(&[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], FillRule::NonZero);
        let repeated = shape
            .edges
            .iter()
            .chain(shape.edges.iter())
            .copied()
            .collect();
        let doubled = Operand {
            edges: repeated,
            rule: FillRule::EvenOdd,
        };
        assert_eq!(
            rasterize(&[shape.clone(), doubled.clone()], 1, 1).at(0, 0),
            0.0
        );
        let union = Operand {
            rule: FillRule::NonZero,
            ..doubled
        };
        assert_eq!(rasterize(&[shape, union], 1, 1).at(0, 0), 0.5);
    }

    #[test]
    fn fractional_horizontal_edges_preserve_connected_groups() {
        let shape = polygon(
            &[
                (-20.25, 0.25),
                (20.75, 0.25),
                (20.75, 17.75),
                (-20.25, 17.75),
            ],
            FillRule::NonZero,
        );
        let coverage = rasterize(&[shape], 24, 24);
        assert_eq!(coverage.at(0, 0), 0.75);
        assert_eq!(coverage.at(20, 0), 0.5625);
        assert_eq!(coverage.at(0, 17), 0.75);
        assert_eq!(coverage.at(20, 17), 0.5625);
        assert_eq!(coverage.at(21, 0), 0.0);
        assert_eq!(coverage.at(0, 18), 0.0);
        assert_eq!(coverage.at(0, 1), 1.0);
    }

    #[test]
    fn opposite_windings_cancel_and_crossings_split_strips() {
        let shape = polygon(
            &[(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)],
            FillRule::NonZero,
        );
        assert_eq!(rasterize(std::slice::from_ref(&shape), 1, 1).at(0, 0), 0.5);
        let edges = shape
            .edges
            .iter()
            .copied()
            .chain(shape.edges.iter().map(|edge| Edge {
                x0: edge.x1,
                y0: edge.y1,
                x1: edge.x0,
                y1: edge.y0,
            }))
            .collect();
        assert_eq!(
            rasterize(
                &[Operand {
                    edges,
                    rule: FillRule::NonZero
                }],
                1,
                1
            )
            .at(0, 0),
            0.0
        );
    }

    #[test]
    fn flat_samples_and_constants_round_trip() {
        let mut coverage = Coverage {
            row_offsets: vec![0],
            ..Coverage::default()
        };
        for (x, alpha) in [0.0, 0.25, 0.5, 0.125, 1.0, 1.0, 0.5, 0.75]
            .into_iter()
            .enumerate()
        {
            coverage.push(x, x + 1, alpha, 0);
        }
        coverage
            .row_offsets
            .push(u32::try_from(coverage.spans.len()).unwrap());
        assert_eq!(coverage.spans.len(), 3);
        for (x, alpha) in [0.0, 0.25, 0.5, 0.125, 1.0, 1.0, 0.5, 0.75]
            .into_iter()
            .enumerate()
        {
            assert_eq!(coverage.at(x, 0), alpha);
        }
        assert_eq!(coverage.samples(&coverage.spans[0]), [0.25, 0.5, 0.125]);
        assert_eq!(coverage.samples(&coverage.spans[2]), [0.5, 0.75]);
    }

    #[test]
    fn disjoint_fractional_contours_leave_exactly_empty_gaps() {
        let triangle = polygon(&[(1.2, 0.1), (7.1, 3.3), (1.8, 2.7)], FillRule::NonZero);
        let shifted = triangle.edges.iter().map(|edge| Edge {
            x0: edge.x0 + 16.0,
            x1: edge.x1 + 16.0,
            ..*edge
        });
        let shape = Operand {
            edges: triangle.edges.iter().copied().chain(shifted).collect(),
            rule: FillRule::NonZero,
        };
        let coverage = rasterize(&[shape], 32, 4);
        for y in 0..4 {
            for x in (8..17).chain(24..32) {
                assert_eq!(coverage.at(x, y).to_bits(), 0.0_f32.to_bits(), "({x}, {y})");
            }
        }
    }

    #[test]
    fn band_compilation_matches_full_field_and_reuses_bounded_storage() {
        let operands = [polygon(
            &[(-0.25, 0.25), (10.75, 65.75), (31.5, 64.75), (20.25, 0.125)],
            FillRule::EvenOdd,
        )];
        let full = rasterize(&operands, 32, 80);
        let mut compiler = Compiler::default();
        for start in (0..80).step_by(16) {
            let band = compiler.compile(&operands, 32, start..start + 16);
            for y in start..start + 16 {
                for x in 0..32 {
                    assert_eq!(band.at(x, y).to_bits(), full.at(x, y).to_bits());
                }
            }
            assert!(compiler.offsets.len() <= 17);
            assert!(compiler.lines.len() <= 16 * operands[0].edges.len());
        }
        let empty = compiler.compile(&operands, 32, 128..144);
        assert_eq!(empty.row(128), []);
    }
}
