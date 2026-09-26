// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Geometric clipping: coverage inside a layer clip is the exact area of
//! `shape ∩ clip`, never a product of two independent coverages.
//!
//! Polygon Boolean operations use iOverlay's i64 engine, independently of the
//! CPU rasterizer. Resolving each operand's winding produces consistently
//! oriented boundaries even for self-intersections and opposite contour
//! directions. Per-pixel area integration remains in [`crate::coverage`].

use cherenkov_scene::FillRule;

/// An undirected-or-directed line segment.
pub type Segment = (f64, f64, f64, f64);

/// Winding number of `edges` at point `(px, py)` (standard crossing rule,
/// following the convention of `font-rs`/PostScript: upward edges crossing to
/// the right of the point count `+1`).
#[must_use]
pub fn winding(edges: &[Segment], px: f64, py: f64) -> i32 {
    let mut w = 0;
    for &(x0, y0, x1, y1) in edges {
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

/// Whether `winding` satisfies `rule`.
#[must_use]
pub const fn inside(winding: i32, rule: FillRule) -> bool {
    match rule {
        FillRule::NonZero => winding != 0,
        FillRule::EvenOdd => winding % 2 != 0,
    }
}

fn is_left(x0: f64, y0: f64, x1: f64, y1: f64, px: f64, py: f64) -> f64 {
    (px - x0).mul_add(-(y1 - y0), (x1 - x0) * (py - y0))
}

/// Represent a directed boundary as a signed triangle fan. Radial edges
/// cancel at every shared endpoint, preserving winding without assuming that
/// segments arrive in contour order. Degenerate triangles contribute no area.
fn triangles(segments: &[Segment]) -> Vec<Vec<[f64; 2]>> {
    let Some(&(x, y, _, _)) = segments.first() else {
        return Vec::new();
    };
    segments
        .iter()
        .map(|&(x0, y0, x1, y1)| vec![[x, y], [x0, y0], [x1, y1]])
        .collect()
}

/// Directed segments tracing the boundary of `shape ∩ clip`.
///
/// Resolve `shape_rule` before intersecting with the nonzero clip. The result
/// has consistently oriented exterior and hole contours and can be integrated
/// with either fill rule. The i64 overlay grid retains f64 coordinate precision
/// at the device scales used by the oracle.
#[must_use]
pub fn intersect_edges(shape: &[Segment], shape_rule: FillRule, clip: &[Segment]) -> Vec<Segment> {
    use i_overlay::{
        core::{fill_rule::FillRule as Rule, overlay_rule::OverlayRule},
        float::overlay::FloatOverlay,
    };
    let mut subject = triangles(shape);
    if shape_rule == FillRule::EvenOdd {
        subject = FloatOverlay::<[f64; 2], i64>::from_subj(&subject)
            .overlay(OverlayRule::Subject, Rule::EvenOdd)
            .into_iter()
            .flatten()
            .collect();
    }
    let result = FloatOverlay::<[f64; 2], i64>::from_subj_and_clip(&subject, &triangles(clip))
        .overlay(OverlayRule::Intersect, Rule::NonZero);
    result
        .into_iter()
        .flatten()
        .flat_map(|contour| {
            let mut edges = Vec::with_capacity(contour.len());
            for (start, end) in contour
                .iter()
                .zip(contour.iter().cycle().skip(1))
                .take(contour.len())
            {
                edges.push((start[0], start[1], end[0], end[1]));
            }
            edges
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage::Coverage;

    fn edges(points: &[(f64, f64)]) -> Vec<Segment> {
        points
            .iter()
            .zip(points.iter().cycle().skip(1))
            .take(points.len())
            .map(|(&(x0, y0), &(x1, y1))| (x0, y0, x1, y1))
            .collect()
    }

    fn field(edges: &[Segment], rule: FillRule) -> Vec<f64> {
        let mut coverage = Coverage::new(8, 8);
        for &(x0, y0, x1, y1) in edges {
            coverage.add_line(x0, y0, x1, y1);
        }
        coverage.finish(rule)
    }

    #[test]
    fn intersection_preserves_winding_across_self_crossings_and_reversal() {
        let clip = edges(&[(0.25, 0.25), (7.75, 0.25), (7.75, 7.75), (0.25, 7.75)]);
        let shape = edges(&[(1.25, 1.25), (6.75, 6.75), (1.25, 6.75), (6.75, 1.25)]);
        let reversed: Vec<_> = shape
            .iter()
            .rev()
            .map(|&(x0, y0, x1, y1)| (x1, y1, x0, y0))
            .collect();
        for shape in [&shape, &reversed] {
            for rule in [FillRule::NonZero, FillRule::EvenOdd] {
                let expected = field(shape, rule);
                let actual = field(&intersect_edges(shape, rule, &clip), FillRule::NonZero);
                for (actual, expected) in actual.into_iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn evenodd_subject_does_not_change_a_nonzero_clips_fill_rule() {
        let shape = edges(&[(1.25, 1.25), (6.75, 1.25), (6.75, 6.75), (1.25, 6.75)]);
        let mut doubled_clip = shape.clone();
        doubled_clip.extend_from_slice(&shape);
        let actual = field(
            &intersect_edges(&shape, FillRule::EvenOdd, &doubled_clip),
            FillRule::NonZero,
        );
        for (actual, expected) in actual.into_iter().zip(field(&shape, FillRule::EvenOdd)) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}
