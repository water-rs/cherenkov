//! Winding resolution before signed-area accumulation.
//!
//! The area accumulator reads each pixel's net winding: overlapping
//! same-sign regions clamp (`0.6 + 0.6 -> 1.0`, losing the union's true
//! 0.84) and opposite-sign regions cancel — both wrong under `NonZero`,
//! and the raw winding can exceed any rule's range on self-overlapping
//! outlines (kurbo stroke output overlaps at joins and caps). `resolve`
//! sweeps the flattened device-space segments left to right inside
//! crossing-free horizontal bands and emits each band's inside/outside
//! boundary edges, so every covered region carries winding `1` and the
//! accumulator becomes exact under `NonZero`.
//!
//! `None` means the input had no overlap: the caller keeps the original
//! segments and rule untouched, leaving every non-overlapping scene
//! bit-identical.

use cherenkov::FillRule;

/// Comparison epsilon for sweep positions and boundary times.
const EPS: f64 = 1e-9;

/// One non-horizontal segment normalized to run top-to-bottom in `f64`.
struct Seg {
    /// X at the top endpoint.
    x0: f64,
    /// Top y.
    y0: f64,
    /// Bottom y.
    y1: f64,
    /// dx/dy.
    slope: f64,
    /// Signed winding deposit: +1 downward, -1 upward.
    dir: f64,
}

impl Seg {
    fn x_at(&self, y: f64) -> f64 {
        self.slope.mul_add(y - self.y0, self.x0)
    }
}

/// Rewrites `segments` into a boundary edge set with winding 0 or 1
/// everywhere under `rule`, or returns `None` when the input has no
/// overlap to resolve.
#[expect(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    reason = "one sweep per design; emitted coordinates fit f32"
)]
pub fn resolve(
    segments: &[(f32, f32, f32, f32)],
    rule: FillRule,
) -> Option<Vec<(f32, f32, f32, f32)>> {
    let mut segs: Vec<Seg> = Vec::with_capacity(segments.len());
    for &(x0, y0, x1, y1) in segments {
        if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
            continue;
        }
        let (x0, y0, x1, y1) = (f64::from(x0), f64::from(y0), f64::from(x1), f64::from(y1));
        #[expect(clippy::float_cmp, reason = "horizontal edges carry no area")]
        if y0 == y1 {
            continue;
        }
        if y0 < y1 {
            segs.push(Seg {
                x0,
                y0,
                y1,
                slope: (x1 - x0) / (y1 - y0),
                dir: 1.0,
            });
        } else {
            segs.push(Seg {
                x0: x1,
                y0: y1,
                y1: y0,
                slope: (x0 - x1) / (y0 - y1),
                dir: -1.0,
            });
        }
    }
    if segs.is_empty() {
        return None;
    }
    let mut ys: Vec<f64> = segs.iter().flat_map(|s| [s.y0, s.y1]).collect();
    ys.sort_by(f64::total_cmp);
    ys.dedup();

    let mut out: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut overlap = false;
    let mut band = 0usize;
    while band + 1 < ys.len() {
        let (ya, yb) = (ys[band], ys[band + 1]);
        let mut active: Vec<&Seg> = segs
            .iter()
            .filter(|s| s.y0 <= ya + EPS && s.y1 >= yb - EPS)
            .collect();
        if active.is_empty() {
            band += 1;
            continue;
        }
        active.sort_by(|a, b| {
            a.x_at(ya)
                .total_cmp(&b.x_at(ya))
                .then_with(|| a.x_at(yb).total_cmp(&b.x_at(yb)))
        });
        // Any order change inside the band shows up as an adjacent
        // inversion at the band's bottom; split at the earliest valid
        // crossing y. A pair whose crossing sits on the boundary (a
        // tie) or that is ~parallel yields no split and does not stop
        // the scan — a later pair may still cross inside.
        let mut split = None;
        for pair in active.windows(2) {
            let (p, q) = (pair[0], pair[1]);
            if p.x_at(yb) > q.x_at(yb) + EPS && (p.slope - q.slope).abs() > EPS {
                // p.x0 + p.slope*(y-p.y0) == q.x0 + q.slope*(y-q.y0)
                let yc = q.slope.mul_add(q.y0, p.slope.mul_add(-p.y0, p.x0) - q.x0)
                    / (q.slope - p.slope);
                if yc > ya + EPS && yc < yb - EPS && split.is_none_or(|s| yc < s) {
                    split = Some(yc);
                }
            }
        }
        if let Some(yc) = split {
            ys.insert(band + 1, yc);
            overlap = true;
            continue;
        }
        let mut w = 0.0f64;
        let mut inside = false;
        let mut wmin = 0.0f64;
        let mut wmax = 0.0f64;
        for s in &active {
            w += s.dir;
            wmin = wmin.min(w);
            wmax = wmax.max(w);
            let now = match rule {
                FillRule::NonZero => w != 0.0,
                FillRule::EvenOdd => w.rem_euclid(2.0) > 0.5,
            };
            if now != inside {
                let (xa, xb) = (s.x_at(ya), s.x_at(yb));
                if now {
                    out.push((xa, ya, xb, yb));
                } else {
                    out.push((xb, yb, xa, ya));
                }
                inside = now;
            }
        }
        // A winding magnitude above one, or both signs in one band,
        // means regions overlap — only then is rewriting needed.
        if wmax >= 2.0 || wmin <= -2.0 || (wmin < 0.0 && wmax > 0.0) {
            overlap = true;
        }
        band += 1;
    }
    if !overlap {
        return None;
    }
    Some(
        out.iter()
            .map(|&(x0, y0, x1, y1)| (x0 as f32, y0 as f32, x1 as f32, y1 as f32))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::raster::Raster;

    /// Raw accumulation coverage of `segs` over a `w`×`h` surface
    /// (bypasses `resolve`, so the pre-resolution behaviour is visible).
    fn cover(segs: &[(f32, f32, f32, f32)], rule: FillRule, w: usize, h: usize) -> Vec<f32> {
        let mut r = Raster::new(w, h);
        for &(x0, y0, x1, y1) in segs {
            r.draw_line(x0, y0, x1, y1);
        }
        r.coverage_rule(rule)
    }

    fn square(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<(f32, f32, f32, f32)> {
        vec![
            (x0, y0, x1, y0),
            (x1, y0, x1, y1),
            (x1, y1, x0, y1),
            (x0, y1, x0, y0),
        ]
    }

    #[test]
    fn a_single_square_needs_no_resolution() {
        let segs = square(0.0, 0.0, 9.0, 9.0);
        assert!(resolve(&segs, FillRule::NonZero).is_none());
        assert!(resolve(&segs, FillRule::EvenOdd).is_none());
    }

    #[test]
    fn overlapping_squares_rasterize_to_their_union() {
        // [0,2.5]² and [0.5,3]², same winding. Pixel (2,0): square 1
        // covers x∈[2,2.5]→0.5, square 2 covers x∈[2,3],y∈[0.5,1]→0.5,
        // overlap 0.5·0.5=0.25 → union 0.75; raw accumulation gives 1.0.
        let mut segs = square(0.0, 0.0, 2.5, 2.5);
        segs.extend(square(0.5, 0.5, 3.0, 3.0));
        let resolved = resolve(&segs, FillRule::NonZero).expect("overlap present");
        let cov = cover(&resolved, FillRule::NonZero, 6, 6);
        let raw = cover(&segs, FillRule::NonZero, 6, 6);
        assert!((cov[2] - 0.75).abs() < 1e-5, "pixel (2,0) = {}", cov[2]);
        assert!(
            (raw[2] - 1.0).abs() < 1e-5,
            "accumulator clamps: {}",
            raw[2]
        );
        assert!((cov[2 * 6 + 2] - 1.0).abs() < 1e-5, "pixel (2,2) interior");
    }

    #[test]
    fn a_bow_tie_keeps_both_lobes() {
        // Self-crossing quad: mixed-sign windings cancel in the raw
        // accumulator; resolved coverage is the two-triangle union.
        let segs = vec![
            (0.0, 0.0, 4.0, 4.0),
            (4.0, 4.0, 4.0, 0.0),
            (4.0, 0.0, 0.0, 4.0),
            (0.0, 4.0, 0.0, 0.0),
        ];
        let resolved = resolve(&segs, FillRule::NonZero).expect("mixed signs overlap");
        let cov = cover(&resolved, FillRule::NonZero, 4, 4);
        let total: f32 = cov.iter().sum();
        assert!(
            (f64::from(total) - 8.0).abs() < 1e-3,
            "total coverage {total} != 8.0"
        );
        // Opposite windings sharing a pixel cancel in the raw
        // accumulator; resolution restores the union coverage.
        let mut cancel = square(0.0, 0.0, 2.5, 2.5);
        // Same footprint as [0.5,3]², counter-wound.
        cancel.extend([
            (0.5, 0.5, 0.5, 3.0),
            (0.5, 3.0, 3.0, 3.0),
            (3.0, 3.0, 3.0, 0.5),
            (3.0, 0.5, 0.5, 0.5),
        ]);
        // Counter-wound squares: NonZero covers the symmetric
        // difference (0.5), while the raw accumulator cancels to 0.
        let resolved = resolve(&cancel, FillRule::NonZero).expect("cancel overlap");
        let cov = cover(&resolved, FillRule::NonZero, 6, 6);
        let raw = cover(&cancel, FillRule::NonZero, 6, 6);
        assert!((cov[2] - 0.5).abs() < 1e-5, "pixel (2,0) = {}", cov[2]);
        assert!(raw[2] < 0.25, "raw cancels: {}", raw[2]);
    }

    #[test]
    fn even_odd_nested_squares_open_the_hole() {
        let mut segs = square(0.0, 0.0, 4.0, 4.0);
        segs.extend(square(1.0, 1.0, 3.0, 3.0));
        let resolved = resolve(&segs, FillRule::EvenOdd).expect("winding reaches 2");
        let cov = cover(&resolved, FillRule::NonZero, 4, 4);
        assert!(
            (cov[4 + 1] - 0.0).abs() < 1e-5,
            "hole pixel (1,1) = {}",
            cov[4 + 1]
        );
        assert!((cov[0] - 1.0).abs() < 1e-5, "ring pixel (0,0) = {}", cov[0]);
    }

    #[test]
    fn a_later_crossing_still_splits_the_band() {
        // The first inverted pair (0,0)-(4,4) vs (1e-10,0)-(-4,4) crosses
        // at y≈5e-11 — on the band's top boundary, a tie, no valid split.
        // A scan that stops at that pair would walk (5,0)-(4,4) and
        // (6,0)-(3,4) while they cross inside at y=2.
        let segs = vec![
            (0.0f32, 0.0, 4.0, 4.0),
            (1e-10, 0.0, -4.0, 4.0),
            (5.0, 0.0, 4.0, 4.0),
            (6.0, 0.0, 3.0, 4.0),
        ];
        let resolved = resolve(&segs, FillRule::NonZero).expect("crossing present");
        assert!(
            resolved
                .iter()
                .any(|e| (e.1 - 2.0).abs() < 1e-6 || (e.3 - 2.0).abs() < 1e-6),
            "no edge boundary at the y=2 crossing: {resolved:?}"
        );
    }
}
