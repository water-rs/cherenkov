// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Compare coverage compilation followed by convolution with the independent
//! oracle geometry, clipping and Gaussian implementations.

use cherenkov::FillRule;
use cherenkov_oracle::{clip, coverage, path, shadow};
use cherenkov_scene::{FillRule as OracleRule, Shape};
use kurbo::{Affine, BezPath, Rect, Shape as _};

use super::coverage::{Operand, rasterize};
use super::lower::{FLATTEN_TOL, flatten_edges};
use super::raster::blur_coverage;

fn polygon(points: &[(f64, f64)]) -> BezPath {
    let mut outline = BezPath::new();
    outline.move_to(points[0]);
    for &point in &points[1..] {
        outline.line_to(point);
    }
    outline.close_path();
    outline
}

fn assert_shadow(outline: &BezPath, transform: Affine, clips: &[BezPath], size: (usize, usize)) {
    let (width, height) = size;
    let shape = Shape::Path {
        path: outline.clone(),
    };
    let mut segments = path::edges(&path::shape_polylines(&shape, transform));
    let mut operands = vec![Operand {
        edges: flatten_edges(transform * outline.clone(), FLATTEN_TOL).into(),
        rule: FillRule::NonZero,
    }];
    for outline in clips {
        let shape = Shape::Path {
            path: outline.clone(),
        };
        let boundary = path::edges(&path::shape_polylines(&shape, Affine::IDENTITY));
        segments = clip::intersect_edges(&segments, OracleRule::NonZero, &boundary);
        operands.push(Operand {
            edges: flatten_edges(outline.clone(), FLATTEN_TOL).into(),
            rule: FillRule::NonZero,
        });
    }
    let mut oracle = coverage::Coverage::new(width, height);
    for (x0, y0, x1, y1) in segments {
        oracle.add_line(x0, y0, x1, y1);
    }
    let exact = oracle.finish(OracleRule::NonZero);
    let source = rasterize(&operands, width, height);
    // Include effectively zero blur, a subpixel kernel, and a halo wider
    // than the canvas. The latter exercises clamping at every surface edge.
    for sigma in [0.0, 1e-10, 0.125, 0.75, 2.5, 9.0] {
        let expected = shadow::gaussian_blur(&exact, width, height, sigma, transform);
        let actual = blur_coverage(&source, width, height, sigma, transform);
        for (index, &value) in expected.iter().enumerate() {
            let x = index % width;
            let y = index / width;
            let rendered = f64::from(actual.at(x, y));
            assert!(
                (rendered - value).abs() < 2e-6,
                "({x}, {y}), sigma {sigma}: {rendered} versus oracle {value}"
            );
        }
    }
}

#[test]
fn self_intersecting_shadow_preserves_both_winding_signs() {
    let outline = polygon(&[
        (1.125, 1.25),
        (14.75, 13.625),
        (2.25, 14.75),
        (13.5, 0.25),
    ]);
    assert_shadow(&outline, Affine::IDENTITY, &[], (17, 17));
}

#[test]
fn rotated_and_sheared_shadow_uses_transformed_caster_coverage() {
    let outline = Rect::new(-4.125, -3.25, 4.75, 3.875).to_path(FLATTEN_TOL);
    for transform in [
        Affine::translate((8.125, 8.375)) * Affine::rotate(0.375),
        Affine::translate((8.125, 8.375)) * Affine::scale_non_uniform(1.5, 0.75),
        Affine::translate((8.125, 8.375)) * Affine::rotate(0.375) * Affine::scale_non_uniform(1.5, 0.75),
        Affine::new([1.0, 0.25, -0.5, 1.0, 8.125, 7.875]),
        Affine::new([-1.0, 0.25, 0.5, 1.0, 8.125, 7.875]),
    ] {
        assert_shadow(&outline, transform, &[], (17, 17));
    }
}

#[test]
fn a_shadow_preserves_a_hole_in_its_caster() {
    let mut outline = polygon(&[
        (1.25, 1.25),
        (15.75, 1.25),
        (15.75, 15.75),
        (1.25, 15.75),
    ]);
    // Opposite orientation removes the centre under the nonzero rule.
    let hole = polygon(&[
        (5.125, 5.125),
        (5.125, 11.875),
        (11.875, 11.875),
        (11.875, 5.125),
    ]);
    for &element in hole.elements() {
        outline.push(element);
    }
    assert_shadow(&outline, Affine::IDENTITY, &[], (17, 17));
}

#[test]
fn nested_partial_clips_intersect_before_shadow_convolution() {
    let outline = polygon(&[
        (0.125, 0.125),
        (6.75, 0.125),
        (6.75, 7.625),
        (0.125, 7.625),
    ]);
    let clips = [
        polygon(&[(0.125, 0.25), (7.875, 1.625), (1.75, 7.875)]),
        polygon(&[(0.375, 7.75), (3.625, 0.375), (7.625, 7.25)]),
    ];
    assert_shadow(
        &outline,
        Affine::translate((0.375, -0.125)),
        &clips,
        (9, 9),
    );
}

#[test]
fn shadow_convolution_clamps_the_surface_not_the_caster_bounds() {
    let outline = polygon(&[(-2.5, -2.5), (4.75, -2.5), (4.75, 6.25), (-2.5, 6.25)]);
    for size in [(1, 1), (1, 9), (9, 1), (9, 9)] {
        assert_shadow(&outline, Affine::IDENTITY, &[], size);
        assert_shadow(&outline, Affine::translate((20.0, 20.0)), &[], size);
    }
}

#[test]
fn affine_gaussian_impulses_match_oracle_including_degenerate_covariances() {
    let size = 17;
    let mut exact = vec![0.0; size * size];
    exact[8 * size + 8] = 1.0;
    let source = super::coverage::Coverage::from_rows(8,
        std::iter::once((0..size).map(|x| if x == 8 { 1.0 } else { 0.0 }).collect()));
    for (sigma, transform) in [
        (0.125, Affine::scale_non_uniform(3.0, 0.5)),
        (1.25, Affine::rotate(0.4) * Affine::scale_non_uniform(2.0, 0.5)),
        (0.75, Affine::new([1.0, 0.6, 0.0, 0.8, 0.0, 0.0])),
        (0.75, Affine::new([-1.0, 0.6, 0.0, 0.8, 0.0, 0.0])),
        (0.75, Affine::new([1.0, 1.0, 0.0, 1e-8, 0.0, 0.0])),
        (0.75, Affine::new([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])),
        (0.75, Affine::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])),
        (0.75, Affine::scale(0.0)),
        (1e-10, Affine::scale(1e10)),
    ] {
        let expected = shadow::gaussian_blur(&exact, size, size, sigma, transform);
        let actual = blur_coverage(&source, size, size, sigma, transform);
        for (index, &value) in expected.iter().enumerate() {
            let rendered = f64::from(actual.at(index % size, index / size));
            assert!((rendered - value).abs() < 3e-8,
                "tap {index}, sigma {sigma}, transform {transform:?}: {rendered} versus {value}");
        }
    }
}
