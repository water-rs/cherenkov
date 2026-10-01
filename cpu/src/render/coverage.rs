//! Sparse pixel areas of geometric intersections. Each operand's winding is
//! resolved before integration; only the final area is rounded to f32.

use std::sync::Arc;

use cherenkov::FillRule;

use super::raster::Edge;

/// A closed device-space boundary, including horizontal connecting edges.
#[derive(Clone, Debug, PartialEq)]
pub struct Operand {
    /// Directed polygon edges.
    pub edges: Arc<[Edge]>,
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

    #[cfg(test)]
    fn samples(&self, span: &Span) -> &[f32] {
        match span.kind {
            SpanKind::Constant(_) => &[],
            SpanKind::Samples(offset) => {
                &self.samples[offset as usize..(offset + span.len) as usize]
            }
        }
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
fn row_lines(operands: &[Operand], top: usize, bottom: usize) -> (Vec<usize>, Vec<Line>) {
    let mut offsets = vec![0; bottom - top + 1];
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
    let mut lines = vec![Line::default(); offsets[bottom - top]];
    let mut cursors = offsets.clone();
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
    (offsets, lines)
}

/// Compile the exact area of the intersection of all operands.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "surface-clamped row coordinates fit exactly in f64"
)]
pub fn rasterize(operands: &[Operand], w: usize, h: usize) -> Coverage {
    if operands.is_empty() || w == 0 || h == 0 {
        return Coverage::default();
    }
    let mut top = 0;
    let mut bottom = h;
    for operand in operands {
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
    let (offsets, mut lines) = row_lines(operands, top, bottom);
    let rules: Vec<_> = operands.iter().map(|operand| operand.rule).collect();
    let mut baseline = vec![0; operands.len()];
    let mut scratch = RowScratch::default();
    let mut result = Coverage {
        top,
        edge_count: operands[0].edges.len(),
        row_offsets: vec![0],
        ..Coverage::default()
    };
    for row in 0..bottom - top {
        let lines = &mut lines[offsets[row]..offsets[row + 1]];
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
            scratch.group(&lines[first..end], &baseline, &rules, w);
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
    result
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
        result
            .row_offsets
            .push(u32::try_from(result.spans.len()).expect("span index"));
        self.delta.clear();
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn polygon(points: &[(f32, f32)], rule: FillRule) -> Operand {
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
}
