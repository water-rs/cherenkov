// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Analytic signed-distance support for per-member backdrop effects.
//!
//! A literal `f64` port of `sdf`, `sdf_grad` and the `J⁻ᵀ` device-gradient
//! math in `gpu/src/render/shader.wgsl`, with the scene `Shape` → box
//! `Shape` mapping of `gpu::render::prepared::box_shape` (radii clamped to
//! the half extents, corner order top-left, top-right, bottom-right,
//! bottom-left).

use cherenkov_scene::Shape;
use kurbo::Affine;

/// A rounded box centred at the origin, mirroring the WGSL `Shape`.
#[derive(Clone, Copy, Debug)]
pub struct BoxShape {
    /// Half extents of the box.
    pub half: [f64; 2],
    /// Corner radius aspect (y radius / x radius).
    pub aspect: f64,
    /// Lamé exponent of the corner curve; 2.0 for circular corners.
    pub exponent: f64,
    /// Corner radii along x: top-left, top-right, bottom-right, bottom-left.
    pub radii: [f64; 4],
}

/// The clip shape mapped to the GPU's centred box form, and the local →
/// box-local affine `extra` (`transform * extra` maps clip space onto the
/// centred box). `None` for shapes with no analytic box (path, line).
pub fn box_params(shape: &Shape) -> Option<(BoxShape, Affine)> {
    let (boxed, extra) = match shape {
        Shape::Rect(r) => {
            let half = [r.width() / 2.0, r.height() / 2.0];
            (
                BoxShape {
                    half,
                    aspect: 1.0,
                    exponent: 2.0,
                    radii: [0.0; 4],
                },
                Affine::translate(r.center().to_vec2()),
            )
        }
        Shape::RoundedRect(rr) => {
            let r = rr.rect();
            let half = [r.width() / 2.0, r.height() / 2.0];
            (
                BoxShape {
                    half,
                    aspect: 1.0,
                    exponent: 2.0,
                    radii: clamped_radii(rr.radii(), half),
                },
                Affine::translate(r.center().to_vec2()),
            )
        }
        Shape::Continuous(c) => {
            let r = c.rect;
            let half = [r.width() / 2.0, r.height() / 2.0];
            let limit = half[0].min(half[1]);
            (
                BoxShape {
                    half,
                    aspect: 1.0,
                    exponent: c.smoothing.clamp(0.0, 1.0).mul_add(2.0, 2.0),
                    radii: [c.corner_radius.clamp(0.0, limit); 4],
                },
                Affine::translate(r.center().to_vec2()),
            )
        }
        Shape::Circle(c) => {
            let r = c.radius;
            if r <= 0.0 {
                return None;
            }
            (
                BoxShape {
                    half: [r, r],
                    aspect: 1.0,
                    exponent: 2.0,
                    radii: [r; 4],
                },
                Affine::translate(c.center.to_vec2()),
            )
        }
        Shape::Ellipse(e) => {
            let radii = e.radii();
            let (a, b) = (radii.x, radii.y);
            if a <= 0.0 {
                return None;
            }
            (
                BoxShape {
                    half: [a, b],
                    aspect: b / a,
                    exponent: 2.0,
                    radii: [a; 4],
                },
                Affine::translate(e.center().to_vec2()) * Affine::rotate(e.rotation()),
            )
        }
        Shape::Line(_) | Shape::Path { .. } => return None,
    };
    Some((boxed, extra))
}

fn clamped_radii(radii: kurbo::RoundedRectRadii, half: [f64; 2]) -> [f64; 4] {
    let limit = half[0].min(half[1]);
    [
        radii.top_left.clamp(0.0, limit),
        radii.top_right.clamp(0.0, limit),
        radii.bottom_right.clamp(0.0, limit),
        radii.bottom_left.clamp(0.0, limit),
    ]
}

/// The corner radius index for the quadrant `p` lies in, as in the WGSL
/// `sdf`: top-left, top-right, bottom-right, bottom-left.
fn corner_radii(s: &BoxShape, p: [f64; 2]) -> (f64, [f64; 2]) {
    let r = if p[0] > 0.0 {
        if p[1] > 0.0 { s.radii[2] } else { s.radii[1] }
    } else if p[1] > 0.0 {
        s.radii[3]
    } else {
        s.radii[0]
    };
    let rx = r.max(0.0);
    (rx, [rx, rx * s.aspect])
}

/// Signed distance to `s` at box-local `p` (negative inside), a literal
/// port of the WGSL `sdf`.
pub fn sdf(s: &BoxShape, p: [f64; 2]) -> f64 {
    let (rx, [_, ry]) = corner_radii(s, p);
    let a = [p[0].abs() - s.half[0], p[1].abs() - s.half[1]];
    if rx <= 0.0 || ry <= 0.0 {
        let m = [a[0].max(0.0), a[1].max(0.0)];
        return m[0].hypot(m[1]) + a[0].max(a[1]).min(0.0);
    }
    let q = [a[0] + rx, a[1] + ry];
    if q[0] > 0.0 && q[1] > 0.0 {
        let n = s.exponent;
        let u = [q[0] / rx, q[1] / ry];
        if (n - 2.0).abs() < 1e-4 {
            let g = u[0].hypot(u[1]);
            let grad = (u[0] / rx).hypot(u[1] / ry) / g.max(1e-6);
            return (g - 1.0) / grad.max(1e-6);
        }
        let f = u[0].powf(n) + u[1].powf(n);
        let g = f.powf(1.0 / n);
        let scale = f.powf(1.0 / n - 1.0);
        let grad = scale * (u[0].powf(n - 1.0) / rx).hypot(u[1].powf(n - 1.0) / ry);
        return (g - 1.0) / grad.max(1e-6);
    }
    a[0].max(a[1])
}

/// Local-space gradient of the signed distance to `s` at `p`, a literal
/// port of the WGSL `sdf_grad`: closed form for sharp and
/// circular/elliptical corners, central differences (E = 0.05) otherwise.
pub fn sdf_grad(s: &BoxShape, p: [f64; 2]) -> [f64; 2] {
    let analytic = (s.exponent - 2.0).abs() < 1e-4 || s.radii.iter().all(|r| *r <= 0.0);
    if analytic {
        let sgn = [
            if p[0] >= 0.0 { 1.0 } else { -1.0 },
            if p[1] >= 0.0 { 1.0 } else { -1.0 },
        ];
        let (rx, [_, ry]) = corner_radii(s, p);
        let a = [p[0].abs() - s.half[0], p[1].abs() - s.half[1]];
        let g = if rx <= 0.0 || ry <= 0.0 {
            if a[0] > 0.0 && a[1] > 0.0 {
                let len = a[0].hypot(a[1]);
                [a[0] / len, a[1] / len]
            } else if a[0] > a[1] {
                [1.0, 0.0]
            } else {
                [0.0, 1.0]
            }
        } else {
            let q = [a[0] + rx, a[1] + ry];
            if q[0] > 0.0 && q[1] > 0.0 {
                let v = [q[0] / (rx * rx), q[1] / (ry * ry)];
                let len = v[0].hypot(v[1]).max(1e-12);
                [v[0] / len, v[1] / len]
            } else if a[0] > a[1] {
                [1.0, 0.0]
            } else {
                [0.0, 1.0]
            }
        };
        return [sgn[0] * g[0], sgn[1] * g[1]];
    }
    const E: f64 = 0.05;
    [
        (sdf(s, [p[0] + E, p[1]]) - sdf(s, [p[0] - E, p[1]])) / (2.0 * E),
        (sdf(s, [p[0], p[1] + E]) - sdf(s, [p[0], p[1] - E])) / (2.0 * E),
    ]
}

/// The member clip's signed distance and unit outward normal at device
/// point `p`, the same math the WGSL clip-coverage block uses: `clip_inv`
/// maps device to box-local, `dg = clip_invᵀ · g`, `d = sdf / |dg|`,
/// `n = dg / |dg|`. `clip_inv` is `inverse(transform * extra)`.
pub fn distance_and_normal(s: &BoxShape, clip_inv: &Affine, p: [f64; 2]) -> (f64, [f64; 2]) {
    let pc = *clip_inv * kurbo::Point::new(p[0], p[1]);
    let pc = [pc.x, pc.y];
    let g = sdf_grad(s, pc);
    let [a, b, c, d, _, _] = clip_inv.as_coeffs();
    let dg = [a * g[0] + b * g[1], c * g[0] + d * g[1]];
    let len = dg[0].hypot(dg[1]).max(1e-6);
    (sdf(s, pc) / len, [dg[0] / len, dg[1] / len])
}

/// Bilinear sample of a capture at device point `q`, a literal port of the
/// WGSL `backdrop_sample`: texel centres at `n + 0.5`, clamped to
/// `[origin, origin + size - 1]`, values unclamped. The oracle captures the
/// full canvas (origin `(0,0)`, size the canvas); the GPU captures the
/// group's bounded region — both regions contain every point a member's
/// reach can sample, so the clamp is a no-op on both sides.
pub fn bilinear(capture: &[[f64; 4]], width: usize, height: usize, q: [f64; 2]) -> [f64; 4] {
    let (w, h) = (width as f64, height as f64);
    let f = [
        (q[0] - 0.5).clamp(0.0, w - 1.0),
        (q[1] - 0.5).clamp(0.0, h - 1.0),
    ];
    let lo = [f[0].floor() as usize, f[1].floor() as usize];
    let hi = [(lo[0] + 1).min(width - 1), (lo[1] + 1).min(height - 1)];
    let t = [f[0] - f[0].floor(), f[1] - f[1].floor()];
    let c00 = capture[lo[1] * width + lo[0]];
    let c10 = capture[lo[1] * width + hi[0]];
    let c01 = capture[hi[1] * width + lo[0]];
    let c11 = capture[hi[1] * width + hi[0]];
    let mix = |c0: [f64; 4], c1: [f64; 4], t: f64| {
        [
            c0[0] + (c1[0] - c0[0]) * t,
            c0[1] + (c1[1] - c0[1]) * t,
            c0[2] + (c1[2] - c0[2]) * t,
            c0[3] + (c1[3] - c0[3]) * t,
        ]
    };
    mix(mix(c00, c10, t[0]), mix(c01, c11, t[0]), t[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect_shape(w: f64, h: f64) -> BoxShape {
        BoxShape {
            half: [w / 2.0, h / 2.0],
            aspect: 1.0,
            exponent: 2.0,
            radii: [0.0; 4],
        }
    }

    #[test]
    fn rect_distance_is_the_box_distance() {
        let s = rect_shape(20.0, 10.0);
        // Inside: -min distance to the edges.
        assert!((sdf(&s, [0.0, 0.0]) - -5.0).abs() < 1e-9);
        assert!((sdf(&s, [8.0, 0.0]) - -2.0).abs() < 1e-9);
        // On the edge and outside.
        assert!((sdf(&s, [10.0, 0.0]) - 0.0).abs() < 1e-9);
        assert!((sdf(&s, [13.0, 0.0]) - 3.0).abs() < 1e-9);
        // Corner point: Euclidean to the corner.
        assert!((sdf(&s, [13.0, 9.0]) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn circle_distance_is_minus_radius() {
        let s = BoxShape {
            half: [4.0, 4.0],
            aspect: 1.0,
            exponent: 2.0,
            radii: [4.0; 4],
        };
        // |p| - r in every direction.
        assert!((sdf(&s, [0.0, 0.0]) - -4.0).abs() < 1e-9);
        assert!((sdf(&s, [3.0, 4.0]) - 1.0).abs() < 1e-9);
        assert!((sdf(&s, [2.0, 0.0]) - -2.0).abs() < 1e-9);
    }

    #[test]
    fn rounded_rect_corner_is_the_offset_circle() {
        let s = BoxShape {
            half: [10.0, 5.0],
            aspect: 1.0,
            exponent: 2.0,
            radii: [2.0; 4],
        };
        // A point past the top-right corner, on the 45° ray: distance to the
        // corner centre (8,3) minus the radius.
        let centre = [8.0, 3.0];
        let k = 2.0f64;
        let p = [centre[0] + k / 2f64.sqrt(), centre[1] + k / 2f64.sqrt()];
        assert!((sdf(&s, p) - (k - 2.0)).abs() < 1e-9);
        // Straight edges stay the box distance.
        assert!((sdf(&s, [9.0, 0.0]) - -1.0).abs() < 1e-9);
    }

    #[test]
    fn gradient_points_outward() {
        let s = rect_shape(20.0, 10.0);
        assert_eq!(sdf_grad(&s, [11.0, 0.0]), [1.0, 0.0]);
        assert_eq!(sdf_grad(&s, [0.0, -6.0]), [0.0, -1.0]);
        // Interior gradient of a sharp box points along the nearer axis.
        assert_eq!(sdf_grad(&s, [8.0, 0.0]), [1.0, 0.0]);
        // The device-space normal points out of the nearer edge.
        let (_, n) = distance_and_normal(&s, &Affine::IDENTITY, [-9.5, 0.0]);
        assert!((n[0] + 1.0).abs() < 1e-9 && n[1].abs() < 1e-9);
    }

    #[test]
    fn bilinear_at_texel_centres_is_the_texel() {
        let px: Vec<[f64; 4]> = (0..16)
            .map(|i| [i as f64, (i * 3) as f64, 0.5, 1.0])
            .collect();
        for y in 0..4 {
            for x in 0..4 {
                let v = bilinear(&px, 4, 4, [x as f64 + 0.5, y as f64 + 0.5]);
                assert_eq!(v, px[y * 4 + x]);
            }
        }
        // Off-centre mixes the neighbours.
        let v = bilinear(&px, 4, 4, [1.0, 0.5]);
        assert_eq!(v[0], 0.5);
        // Clamped outside the capture.
        let v = bilinear(&px, 4, 4, [-3.0, 0.5]);
        assert_eq!(v, px[0]);
    }
}
