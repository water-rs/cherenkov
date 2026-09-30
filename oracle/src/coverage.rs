//! Exact per-pixel area coverage of a flattened polygon.
//!
//! Each pixel's covered area is computed geometrically, not by folding a
//! signed area accumulator: the polygon's segments are clipped to the
//! pixel, the pixel is split into horizontal strips at every interior
//! segment endpoint, and within each strip the covered width is the sum
//! of intervals between consecutive edge crossings of the strip's
//! midline for which the winding number satisfies the fill rule. The
//! winding number is seeded at the pixel's left edge from the whole
//! boundary, so regions overlapping inside a pixel keep their true
//! coverage — two 60% regions winding the same way cover 84% of the
//! pixel, not 100%.
//!
//! Because curves are flattened to line segments before rasterizing
//! (tolerance `FLATTEN_TOLERANCE`), coverage is exact relative to the
//! flattened path up to `f64` rounding.

use cherenkov_scene::FillRule;

use crate::clip::{Segment, inside, is_left};

/// Tolerance for flattening cubic/quadratic curves into line segments, in
/// pixels. The shape→polyline pipeline splits it evenly between the
/// curve-approximation and the flattening stages.
pub const FLATTEN_TOLERANCE: f64 = 1e-4;

/// Matching tolerance for geometric predicates, in pixels.
const EPS: f64 = 1e-9;

/// The part of segment `(sx0,sy0)-(sx1,sy1)` inside the rectangle
/// `[x0,x1]×[y0,y1]`, clipped parametrically (Liang–Barsky).
fn clip_to_rect(s: Segment, x0: f64, y0: f64, x1: f64, y1: f64) -> Option<Segment> {
    let (sx0, sy0, sx1, sy1) = s;
    let (dx, dy) = (sx1 - sx0, sy1 - sy0);
    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    // Clip the parameter interval against each edge of the rectangle.
    for &(p, q) in &[
        (-dx, sx0 - x0),
        (dx, x1 - sx0),
        (-dy, sy0 - y0),
        (dy, y1 - sy0),
    ] {
        if p.abs() <= f64::EPSILON {
            // Parallel to this edge: inside iff q >= 0.
            if q < 0.0 {
                return None;
            }
            continue;
        }
        let t = q / p;
        if p < 0.0 {
            t0 = t0.max(t);
        } else {
            t1 = t1.min(t);
        }
        if t0 >= t1 {
            return None;
        }
    }
    Some((
        dx.mul_add(t0, sx0),
        dy.mul_add(t0, sy0),
        dx.mul_add(t1, sx0),
        dy.mul_add(t1, sy0),
    ))
}

/// Exact covered area of one pixel: the area of `pixel ∩ {inside}` under
/// `rule`.
///
/// Strips are cut at every y where the set of clipped segments changes
/// (interior segment endpoints) and at the y of every interior
/// edge–edge crossing, where two boundary edges swap order. Within a
/// strip every surviving segment is a straight line and no two cross,
/// so each segment's crossing position at the strip's midline `ymid`
/// is linear and correctly ordered. The crossings sort into pairs; the
/// interval between each consecutive pair is covered iff the winding
/// number there satisfies `rule`. The winding is seeded by
/// [`seed_winding`] of the full boundary at the strip's left edge and
/// updated by each crossing's direction (`+1` upward, `−1` downward).
fn pixel_area(segs: &[Segment], cand: &[u32], x0: f64, y0: f64, rule: FillRule) -> f64 {
    let (x1, y1) = (x0 + 1.0, y0 + 1.0);
    let mut inside_segs = Vec::new();
    let mut crit_ys = vec![y0, y1];
    for &i in cand {
        let s = segs[i as usize];
        // Cheap reject before exact clipping.
        if s.0.max(s.2) < x0 - EPS
            || s.0.min(s.2) > x1 + EPS
            || s.1.max(s.3) < y0 - EPS
            || s.1.min(s.3) > y1 + EPS
        {
            continue;
        }
        if let Some(c) = clip_to_rect(s, x0, y0, x1, y1) {
            for y in [c.1, c.3] {
                if y > y0 + EPS && y < y1 - EPS {
                    crit_ys.push(y);
                }
            }
            inside_segs.push(c);
        }
    }
    // Split strips at every interior edge–edge crossing: without a cut
    // there the two edges swap x-order mid-strip and the midline
    // sampling sees a kinked boundary.
    for i in 0..inside_segs.len() {
        for j in i + 1..inside_segs.len() {
            let (ax0, ay0, ax1, ay1) = inside_segs[i];
            let (bx0, by0, bx1, by1) = inside_segs[j];
            let denom = (ax1 - ax0).mul_add(by1 - by0, -(ay1 - ay0) * (bx1 - bx0));
            if denom.abs() <= EPS {
                continue;
            }
            let t = ((bx0 - ax0).mul_add(by1 - by0, -(by0 - ay0) * (bx1 - bx0))) / denom;
            let u = ((bx0 - ax0).mul_add(ay1 - ay0, -(by0 - ay0) * (ax1 - ax0))) / denom;
            if t > EPS && t < 1.0 - EPS && u > EPS && u < 1.0 - EPS {
                let y = (ay1 - ay0).mul_add(t, ay0);
                if y > y0 + EPS && y < y1 - EPS {
                    crit_ys.push(y);
                }
            }
        }
    }
    if inside_segs.is_empty() {
        // Uniformly covered or empty: decide by the centre's winding.
        return f64::from(inside(winding_at(segs, cand, x0 + 0.5, y0 + 0.5), rule));
    }
    crit_ys.sort_by(f64::total_cmp);
    crit_ys.dedup_by(|a, b| (*a - *b).abs() < EPS);
    let mut area = 0.0;
    for w in crit_ys.windows(2) {
        let (ya, yb) = (w[0], w[1]);
        if yb - ya < EPS {
            continue;
        }
        let ymid = 0.5f64.mul_add(ya, 0.5 * yb);
        // Crossings of the strip midline by interior segments, strictly
        // inside the pixel horizontally. A segment contributes only where
        // `ymid` lies within its y extent.
        let mut xs: Vec<(f64, i32)> = Vec::new();
        for &(sx0, sy0, sx1, sy1) in &inside_segs {
            let dy = sy1 - sy0;
            if dy.abs() <= EPS {
                continue;
            }
            let (ylo, yhi) = (sy0.min(sy1), sy0.max(sy1));
            if ymid <= ylo + EPS || ymid >= yhi - EPS {
                continue;
            }
            let xi = (sx1 - sx0).mul_add((ymid - sy0) / dy, sx0);
            if xi > x0 + EPS && xi < x1 - EPS {
                xs.push((xi, i32::from(dy > 0.0) - i32::from(dy < 0.0)));
            }
        }
        xs.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Wind from the pixel's left edge; crossings strictly right of it
        // are already counted by `seed_winding`.
        let mut wind = seed_winding(segs, cand, x0, ymid);
        let mut xprev = x0;
        let mut i = 0;
        while i < xs.len() {
            let xi = xs[i].0;
            if inside(wind, rule) {
                area = (yb - ya).mul_add(xi - xprev, area);
            }
            // A group of crossings at ~the same x leaves the winding
            // region together.
            let mut s = 0;
            while i < xs.len() && (xs[i].0 - xi).abs() < EPS {
                s += xs[i].1;
                i += 1;
            }
            wind -= s;
            xprev = xi;
        }
        if inside(wind, rule) {
            area = (yb - ya).mul_add(x1 - xprev, area);
        }
    }
    area.clamp(0.0, 1.0)
}

/// [`crate::clip::winding`] restricted to the candidate segment indices,
/// in their original order. A segment whose y range does not intersect
/// the candidate row can only contribute zero to `winding`, so the row
/// bucket reproduces the full-list result exactly.
fn winding_at(edges: &[Segment], cand: &[u32], px: f64, py: f64) -> i32 {
    let mut w = 0;
    for &i in cand {
        let (x0, y0, x1, y1) = edges[i as usize];
        if y0 <= py {
            if y1 > py && is_left(x0, y0, x1, y1, px, py) > 0.0 {
                w += 1;
            }
        } else if y1 <= py && is_left(x0, y0, x1, y1, px, py) < 0.0 {
            w -= 1;
        }
    }
    w
}

/// Winding number at `(px, py)` from the edges whose crossing of the
/// horizontal line through `py` lies more than `EPS` right of `px` — the
/// same classification `pixel_area` applies to strip crossings, so an
/// edge hugging the pixel's left border is either seeded or crossed,
/// never seeded and then dropped.
fn seed_winding(edges: &[Segment], cand: &[u32], px: f64, py: f64) -> i32 {
    let mut w = 0;
    for &i in cand {
        let (x0, y0, x1, y1) = edges[i as usize];
        let dir = if y0 <= py && y1 > py {
            1
        } else if y1 <= py && y0 > py {
            -1
        } else {
            continue;
        };
        let xi = (x1 - x0).mul_add((py - y0) / (y1 - y0), x0);
        if xi > px + EPS {
            w += dir;
        }
    }
    w
}

/// A coverage buffer: all boundary segments of the shape, accumulated for
/// per-pixel exact-area evaluation in [`Coverage::finish`].
pub struct Coverage {
    width: usize,
    height: usize,
    /// Directed boundary segments `(x0, y0, x1, y1)`.
    segs: Vec<Segment>,
}

impl Coverage {
    /// A zeroed coverage buffer of `width`×`height` pixels.
    #[must_use]
    pub const fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            segs: Vec::new(),
        }
    }

    /// Accumulate the boundary segment `(x0,y0)-(x1,y1)`. Horizontal or
    /// degenerate segments carry no area.
    pub fn add_line(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) {
        if (x0 - x1).abs() <= EPS && (y0 - y1).abs() <= EPS {
            return;
        }
        self.segs.push((x0, y0, x1, y1));
    }

    /// Accumulate the edges of a polyline (a sequence of points closed by an
    /// implicit edge back to the first point when `closed`).
    pub fn add_polyline(&mut self, points: &[(f64, f64)], closed: bool) {
        for w in points.windows(2) {
            self.add_line(w[0].0, w[0].1, w[1].0, w[1].1);
        }
        if closed && points.len() > 1 {
            let p0 = points[points.len() - 1];
            let p1 = points[0];
            self.add_line(p0.0, p0.1, p1.0, p1.1);
        }
    }

    /// Compute the exact covered area of every pixel under `rule`.
    ///
    /// Every per-pixel computation only consults segments whose y range
    /// intersects the pixel's row: the cheap reject in `pixel_area` skips
    /// anything else, and `winding`/`seed_winding` contributions require
    /// the segment to cross `py`. Bucketing segments by covered row once
    /// keeps every scanned list to the segments that can matter — the
    /// per-segment checks are unchanged, so results are bit-identical.
    ///
    /// # Panics
    ///
    /// Panics if more than `u32::MAX` segments were accumulated.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "pixel indices are far below 2^53; row indices are clamped to bounds"
    )]
    pub fn finish(&self, rule: FillRule) -> Vec<f64> {
        // `rows[r]` lists, in original order, the segments whose y range
        // overlaps `[r - EPS, r + 1 + EPS]` — a superset of the segments
        // that can contribute to any computation in the row.
        let mut out = vec![0.0; self.width * self.height];
        if self.height == 0 {
            return out;
        }
        let mut rows: Vec<Vec<u32>> = (0..self.height).map(|_| Vec::new()).collect();
        for (i, &s) in self.segs.iter().enumerate() {
            let (lo, hi) = (s.1.min(s.3), s.1.max(s.3));
            if !lo.is_finite() || !hi.is_finite() {
                continue;
            }
            let r0 = ((lo - 1.0 - EPS).ceil().max(0.0) as usize).min(self.height - 1);
            let r1 = ((hi + EPS).floor() as usize).min(self.height - 1);
            if r0 > r1 {
                continue;
            }
            let i = u32::try_from(i).expect("under 2^32 segments");
            for row in &mut rows[r0..=r1] {
                row.push(i);
            }
        }
        for y in 0..self.height {
            let cand = &rows[y];
            for x in 0..self.width {
                out[y * self.width + x] = pixel_area(&self.segs, cand, x as f64, y as f64, rule);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the (16,16)-(48,48) rectangle, optionally with its left edge
    /// replaced by ~300 collinear vertices at x = 16 + δ cycling through
    /// tiny sub-EPS offsets, as the Lamé flattening emits near a
    /// tangent-integer junction. `reverse` traverses the rect the other
    /// way.
    fn rect_polyline(noisy_edge: bool, reverse: bool) -> Vec<(f64, f64)> {
        // Counter-clockwise: TL -> TR -> BR -> BL, then up the left edge.
        let mut pts = vec![(16.0, 16.0), (48.0, 16.0), (48.0, 48.0)];
        if noisy_edge {
            const N: usize = 300;
            const DS: [f64; 4] = [0.0, 3e-10, 7e-10, 1e-10];
            for k in 0..=N {
                let y = 48.0
                    - 32.0 * f64::from(u32::try_from(k).unwrap())
                        / f64::from(u32::try_from(N).unwrap());
                pts.push((16.0 + DS[k % DS.len()], y));
            }
        } else {
            pts.push((16.0, 48.0));
        }
        if reverse {
            pts.reverse();
        }
        pts
    }

    #[test]
    fn collinear_run_on_pixel_column_matches_merged_edge() {
        for reverse in [false, true] {
            let mut a = Coverage::new(64, 64);
            a.add_polyline(&rect_polyline(true, reverse), true);
            let mut b = Coverage::new(64, 64);
            b.add_polyline(&rect_polyline(false, reverse), true);
            let pa = a.finish(FillRule::NonZero);
            let pb = b.finish(FillRule::NonZero);
            for (i, (x, y)) in pa.iter().zip(pb.iter()).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-9,
                    "reverse={reverse} pixel {i}: noisy {x} vs merged {y}"
                );
            }
            // The left-edge column is covered: the noisy run must not
            // leak a fully-transparent strip through it.
            for y in 16..48usize {
                let v = pa[y * 64 + 16];
                assert!(v >= 1.0 - 1e-9, "reverse={reverse} (16,{y}) = {v}");
            }
        }
    }
}
