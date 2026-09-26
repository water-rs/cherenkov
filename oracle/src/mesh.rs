// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Reference mesh paint. Each grid cell is a bilinear patch, in row-major
//! order, with corners 00, 10, 01, 11. Geometry and premultiplied linear-P3
//! colour use the same bilinear weights. Outside all patches is transparent.
//! Overlapping patches use the last patch; folded patches use the inverse
//! with greatest v, then greatest u. Collapsed, zero-Jacobian samples are
//! transparent. These ownership rules select one paint value, not a stack
//! of translucent draws. The enclosing shape supplies pixel coverage.

use cherenkov::MeshGradient;
use kurbo::Point;

/// Inverts a bilinear patch at `point`, selecting the last parameter-space
/// branch. Corner order is 00, 10, 01, 11.
#[must_use]
#[expect(
    clippy::many_single_char_names,
    reason = "bilinear polynomial coefficients and parameters"
)]
pub fn coordinates(corners: [Point; 4], point: Point) -> Option<(f64, f64)> {
    let [origin, right, bottom, opposite] = corners;
    let e = right - origin;
    let f = bottom - origin;
    let g = (opposite - bottom) - e;
    let q = point - origin;
    // cross(q - f*v, e + g*v) = 0.
    let a = -f.cross(g);
    let b = q.cross(g) - f.cross(e);
    let c = q.cross(e);
    let roots = if a == 0.0 {
        if b == 0.0 {
            return None;
        }
        [-c / b; 2]
    } else {
        let discriminant = b.mul_add(b, -4.0 * a * c);
        if discriminant < 0.0 {
            return None;
        }
        let t = -0.5 * (b + discriminant.sqrt().copysign(b));
        if t == 0.0 {
            [-b / (2.0 * a); 2]
        } else {
            [t / a, c / t]
        }
    };
    let mut result: Option<(f64, f64)> = None;
    for v in roots {
        if !(0.0..=1.0).contains(&v) {
            continue;
        }
        let direction = e + g * v;
        let residual = q - f * v;
        let u = if direction.x.abs() >= direction.y.abs() {
            residual.x / direction.x
        } else {
            residual.y / direction.y
        };
        if (0.0..=1.0).contains(&u)
            && direction.cross(f + g * u) != 0.0
            && result.is_none_or(|(old_u, old_v)| (v, u) > (old_v, old_u))
        {
            result = Some((u, v));
        }
    }
    result
}

/// Evaluates mesh paint at a point in paint space, returning premultiplied
/// linear Display P3 without gamut or HDR clamping.
#[must_use]
pub fn sample(mesh: &MeshGradient, point: Point) -> [f64; 4] {
    let stride = mesh.columns() as usize + 1;
    for row in (0..mesh.rows() as usize).rev() {
        for column in (0..mesh.columns() as usize).rev() {
            let index = row * stride + column;
            let indices = [index, index + 1, index + stride, index + stride + 1];
            let Some((u, v)) = coordinates(indices.map(|i| mesh.points()[i]), point) else {
                continue;
            };
            let weights = [(1.0 - u) * (1.0 - v), u * (1.0 - v), (1.0 - u) * v, u * v];
            let mut color = [0.0; 4];
            for (index, weight) in indices.into_iter().zip(weights) {
                let channels = mesh.colors()[index].components.map(f64::from);
                for channel in 0..3 {
                    color[channel] =
                        (channels[channel] * channels[3]).mul_add(weight, color[channel]);
                }
                color[3] = channels[3].mul_add(weight, color[3]);
            }
            return color;
        }
    }
    [0.0; 4]
}

#[cfg(test)]
mod tests {
    use super::*;
    use cherenkov::WorkingColor;

    #[test]
    fn overlapping_cells_select_the_last_patch_without_compositing() {
        let red = WorkingColor::new([1.0, 0.0, 0.0, 1.0]);
        let blue = WorkingColor::new([0.0, 0.0, 1.0, 1.0]);
        let mesh = MeshGradient::new(
            2,
            1,
            vec![
                (0.0, 0.0).into(),
                (1.0, 0.0).into(),
                (0.0, 0.0).into(),
                (0.0, 1.0).into(),
                (1.0, 1.0).into(),
                (0.0, 1.0).into(),
            ],
            vec![red, red, blue, red, red, blue],
        );
        let actual = sample(&mesh, Point::new(0.25, 0.5));
        for (actual, expected) in actual.into_iter().zip([0.25, 0.0, 0.75, 1.0]) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn bilinear_premultiplication_and_outside_domain() {
        let mesh = MeshGradient::new(
            1,
            1,
            vec![
                (0.0, 0.0).into(),
                (2.0, 0.0).into(),
                (0.0, 2.0).into(),
                (4.0, 2.0).into(),
            ],
            vec![
                WorkingColor::new([8.0, 0.0, 0.0, 0.0]),
                WorkingColor::new([2.0, 0.0, 0.0, 0.5]),
                WorkingColor::new([0.0, -1.0, 0.0, 1.0]),
                WorkingColor::new([0.0, 0.0, 1.0, 1.0]),
            ],
        );
        let actual = sample(&mesh, Point::new(1.5, 1.0));
        for (value, expected) in actual.into_iter().zip([0.25, -0.25, 0.25, 0.625]) {
            assert!((value - expected).abs() < 1e-12);
        }
        assert!(
            sample(&mesh, Point::new(-1.0, 1.0))
                .iter()
                .all(|v| v.abs() < 1e-12)
        );
    }

    #[test]
    fn inverse_handles_reflection_collapse_and_two_roots() {
        let corners = [
            Point::ORIGIN,
            Point::new(1.0, 0.0),
            Point::new(0.0, 1.0),
            Point::new(-1.0, -1.0),
        ];
        let uv = coordinates(corners, Point::new(0.08, 0.08)).expect("fold has two branches");
        assert!((uv.0 - 0.4).abs() < 1e-12 && (uv.1 - 0.4).abs() < 1e-12);
        assert!(coordinates([Point::ORIGIN; 4], Point::ORIGIN).is_none());
        let reflected = corners.map(|p| Point::new(-p.x, p.y));
        let reflected_uv = coordinates(reflected, Point::new(-0.08, 0.08)).expect("reflected fold");
        assert!((uv.0 - reflected_uv.0).abs() < 1e-12);
    }
}
