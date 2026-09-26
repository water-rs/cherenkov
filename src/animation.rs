// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Animation: the value a nami `Context` metadata may carry, and the
//! per-lane math the layer tree's tracks evaluate.
//!
//! Tracks animate [`transform`](crate::LayerEdit::transform),
//! [`opacity`](crate::LayerEdit::opacity) and
//! [`scroll_offset`](crate::LayerEdit::scroll_offset). Each animatable
//! property decomposes into lanes: an [`Affine`] has six, a [`Vec2`] two
//! and an `f32` one.

use std::time::Duration;

use kurbo::{Affine, Point, Rect, Vec2};

/// Per-lane storage for an [`Animatable`] value.
///
/// Implemented by `[f64; 1]`, `[f64; 2]` and `[f64; 6]`.
pub trait Lanes: Copy + Send + 'static {
    /// The number of lanes.
    const N: usize;
    /// Lane `i`.
    fn get(&self, i: usize) -> f64;
    /// Sets lane `i`.
    fn set(&mut self, i: usize, value: f64);

    /// All lanes zero.
    #[must_use]
    fn zero() -> Self {
        let mut lanes = Self::zeroed();
        for i in 0..Self::N {
            lanes.set(i, 0.0);
        }
        lanes
    }

    /// Allocates the storage; called once by [`Lanes::zero`].
    fn zeroed() -> Self;

    /// `self - other`, lane-wise.
    #[must_use]
    fn sub(&self, other: &Self) -> Self {
        let mut out = Self::zero();
        for i in 0..Self::N {
            out.set(i, self.get(i) - other.get(i));
        }
        out
    }

    /// `self + other`, lane-wise.
    #[must_use]
    fn add(&self, other: &Self) -> Self {
        let mut out = Self::zero();
        for i in 0..Self::N {
            out.set(i, self.get(i) + other.get(i));
        }
        out
    }

    /// `self * k`, lane-wise.
    #[must_use]
    fn scale(&self, k: f64) -> Self {
        let mut out = Self::zero();
        for i in 0..Self::N {
            out.set(i, self.get(i) * k);
        }
        out
    }

    /// `self + other * k`, lane-wise.
    #[must_use]
    fn add_scaled(&self, other: &Self, k: f64) -> Self {
        let mut out = Self::zero();
        for i in 0..Self::N {
            out.set(i, other.get(i).mul_add(k, self.get(i)));
        }
        out
    }

    /// The largest absolute lane value.
    #[must_use]
    fn max_abs(&self) -> f64 {
        let mut max = 0.0_f64;
        for i in 0..Self::N {
            max = max.max(self.get(i).abs());
        }
        max
    }

    /// Every lane within `tol` of `other`'s.
    #[must_use]
    fn near(&self, other: &Self, tol: f64) -> bool {
        self.sub(other).max_abs() <= tol
    }
}

impl Lanes for [f64; 1] {
    const N: usize = 1;
    fn get(&self, i: usize) -> f64 {
        self[i]
    }
    fn set(&mut self, i: usize, value: f64) {
        self[i] = value;
    }
    fn zeroed() -> Self {
        [0.0]
    }
}

impl Lanes for [f64; 2] {
    const N: usize = 2;
    fn get(&self, i: usize) -> f64 {
        self[i]
    }
    fn set(&mut self, i: usize, value: f64) {
        self[i] = value;
    }
    fn zeroed() -> Self {
        [0.0, 0.0]
    }
}

impl Lanes for [f64; 6] {
    const N: usize = 6;
    fn get(&self, i: usize) -> f64 {
        self[i]
    }
    fn set(&mut self, i: usize, value: f64) {
        self[i] = value;
    }
    fn zeroed() -> Self {
        [0.0; 6]
    }
}

/// A property value that can be interpolated lane-wise.
pub trait Animatable: Copy + Send + 'static {
    /// The per-lane representation (`f64` lanes).
    type Lanes: Lanes;
    /// Decomposes into lanes.
    fn into_lanes(self) -> Self::Lanes;
    /// Recomposes from lanes.
    fn from_lanes(lanes: Self::Lanes) -> Self;
}

impl Animatable for f32 {
    type Lanes = [f64; 1];
    fn into_lanes(self) -> Self::Lanes {
        [f64::from(self)]
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "f64 lanes quantize to f32 opacity"
    )]
    fn from_lanes(lanes: Self::Lanes) -> Self {
        lanes[0] as Self
    }
}

impl Animatable for Vec2 {
    type Lanes = [f64; 2];
    fn into_lanes(self) -> Self::Lanes {
        [self.x, self.y]
    }
    fn from_lanes(lanes: Self::Lanes) -> Self {
        Self::new(lanes[0], lanes[1])
    }
}

impl Animatable for Affine {
    type Lanes = [f64; 6];
    fn into_lanes(self) -> Self::Lanes {
        self.as_coeffs()
    }
    fn from_lanes(lanes: Self::Lanes) -> Self {
        Self::new(lanes)
    }
}

/// The animation a property change runs under: the value nami `Context`
/// metadata may carry, and what `.animation(...)` overrides apply.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Animation {
    /// A damped oscillator; continuous in position and velocity on retarget.
    Spring(Spring),
    /// A cubic Bézier timing curve over a fixed duration.
    Curve(Curve),
    /// An exponential deceleration; scroll `scroll_offset` only.
    Decay(Decay),
}

impl From<Spring> for Animation {
    fn from(spring: Spring) -> Self {
        Self::Spring(spring)
    }
}

impl From<Curve> for Animation {
    fn from(curve: Curve) -> Self {
        Self::Curve(curve)
    }
}

impl From<Decay> for Animation {
    fn from(decay: Decay) -> Self {
        Self::Decay(decay)
    }
}

/// A damped oscillator: `response` is the period in seconds, `damping` the
/// damping ratio (`1` is critically damped).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spring {
    /// The oscillation period in seconds.
    pub response: f64,
    /// The damping ratio; `1` is critically damped.
    pub damping: f64,
}

impl Spring {
    /// A smooth, critically damped spring.
    #[must_use]
    pub const fn smooth() -> Self {
        Self {
            response: 0.5,
            damping: 1.0,
        }
    }

    /// A spring with slight overshoot.
    #[must_use]
    pub const fn snappy() -> Self {
        Self {
            response: 0.5,
            damping: 0.85,
        }
    }

    /// A spring with visible bounce.
    #[must_use]
    pub const fn bouncy() -> Self {
        Self {
            response: 0.5,
            damping: 0.7,
        }
    }

    /// A spring from physical parameters (unit mass), as `WaterUI`'s `Spring`
    /// carries.
    #[must_use]
    pub fn from_physics(stiffness: f64, damping: f64) -> Self {
        // Unit mass: ω0 = sqrt(stiffness), ζ = damping / (2 ω0).
        let w0 = stiffness.max(0.0).sqrt();
        Self {
            response: if w0 > 0.0 {
                2.0 * std::f64::consts::PI / w0
            } else {
                f64::MAX
            },
            damping: if w0 > 0.0 { damping / (2.0 * w0) } else { 1.0 },
        }
    }
}

/// A cubic Bézier timing curve with a duration. `p1` and `p2` are the
/// control points of `x(t)`, `y(t)` on `[0, 1]`; the endpoints are `(0, 0)`
/// and `(1, 1)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Curve {
    /// The first control point.
    pub p1: Point,
    /// The second control point.
    pub p2: Point,
    /// The curve's duration.
    pub duration: Duration,
}

impl Curve {
    /// A curve from control-point coordinates, as `WaterUI`'s `Bezier`
    /// carries.
    #[must_use]
    pub const fn bezier(duration: Duration, x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        Self {
            p1: Point::new(x1, y1),
            p2: Point::new(x2, y2),
            duration,
        }
    }

    /// Linear interpolation.
    #[must_use]
    pub const fn linear(duration: Duration) -> Self {
        Self::bezier_const(duration, 0.0, 0.0, 1.0, 1.0)
    }

    /// Ease-in (`cubic-bezier(0.42, 0, 1, 1)`).
    #[must_use]
    pub const fn ease_in(duration: Duration) -> Self {
        Self::bezier_const(duration, 0.42, 0.0, 1.0, 1.0)
    }

    /// Ease-out (`cubic-bezier(0, 0, 0.58, 1)`).
    #[must_use]
    pub const fn ease_out(duration: Duration) -> Self {
        Self::bezier_const(duration, 0.0, 0.0, 0.58, 1.0)
    }

    /// Ease-in-out (`cubic-bezier(0.42, 0, 0.58, 1)`).
    #[must_use]
    pub const fn ease_in_out(duration: Duration) -> Self {
        Self::bezier_const(duration, 0.42, 0.0, 0.58, 1.0)
    }

    const fn bezier_const(duration: Duration, x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        Self {
            p1: Point::new(x1, y1),
            p2: Point::new(x2, y2),
            duration,
        }
    }
}

/// An exponential deceleration, starting at `velocity` and slowing at
/// `deceleration` per second. Scroll `scroll_offset` only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decay {
    /// The initial velocity in logical pixels per second.
    pub velocity: Vec2,
    /// The exponential deceleration constant, per second.
    pub deceleration: f64,
    /// The bounds a rubber band returns the offset to.
    pub rubber_band: Option<Rect>,
}

impl Decay {
    /// A decay starting at `velocity` with the default deceleration
    /// (`4.0` per second).
    #[must_use]
    pub const fn new(velocity: Vec2) -> Self {
        Self {
            velocity,
            deceleration: 4.0,
            rubber_band: None,
        }
    }

    /// Rubber-bands the decay back into `bounds`.
    #[must_use]
    pub const fn rubber_band(self, bounds: Rect) -> Self {
        Self {
            rubber_band: Some(bounds),
            ..self
        }
    }
}

/// The position and velocity of a damped oscillator `dt` seconds in.
///
/// `from` is the start position, `velocity` the start velocity, `target`
/// the spring's rest position. Returns `(position, velocity)` lanes.
#[must_use]
pub fn spring_step<L: Lanes>(from: L, velocity: L, target: L, spring: &Spring, dt: f64) -> (L, L) {
    let w0 = 2.0 * std::f64::consts::PI / spring.response;
    let zeta = spring.damping;
    let mut pos = L::zero();
    let mut vel = L::zero();
    for i in 0..L::N {
        let d0 = from.get(i) - target.get(i);
        let v0 = velocity.get(i);
        let (x, v) = spring_lane(d0, v0, w0, zeta, dt);
        pos.set(i, target.get(i) + x);
        vel.set(i, v);
    }
    (pos, vel)
}

/// One lane of the damped oscillator: initial `offset` from the target,
/// initial `velocity`, angular frequency `omega` and damping `ratio`,
/// evaluated `dt` seconds in. Returns `(position, velocity)`.
fn spring_lane(offset: f64, velocity: f64, omega: f64, ratio: f64, dt: f64) -> (f64, f64) {
    const EPS: f64 = 1e-9;
    if (ratio - 1.0).abs() < EPS {
        // Critically damped: x = e^-ωt (d0 + (v0 + ω d0) t).
        let pos_coeff = velocity;
        let slope = omega.mul_add(offset, pos_coeff);
        let decay = (-omega * dt).exp();
        let factor = slope.mul_add(dt, offset);
        (
            decay * factor,
            decay * (omega * factor).mul_add(-1.0, slope),
        )
    } else if ratio < 1.0 {
        // Underdamped.
        let wd = omega * ratio.mul_add(-ratio, 1.0).sqrt();
        let coeff_a = offset;
        let coeff_b = (ratio * omega).mul_add(offset, velocity) / wd;
        let decay = (-ratio * omega * dt).exp();
        let (sin, cos) = (wd * dt).sin_cos();
        (
            decay * coeff_a.mul_add(cos, coeff_b * sin),
            decay
                * ((ratio * omega).mul_add(coeff_b, coeff_a * wd).mul_add(
                    -sin,
                    (coeff_b * wd).mul_add(cos, -(ratio * omega * coeff_a) * cos),
                )),
        )
    } else {
        // Overdamped.
        let root = omega * ratio.mul_add(ratio, -1.0).sqrt();
        let rate_a = -(ratio * omega) + root;
        let rate_b = -(ratio * omega) - root;
        let coeff_b = (rate_a.mul_add(-offset, velocity)) / (rate_b - rate_a);
        let coeff_a = offset - coeff_b;
        (
            coeff_a.mul_add((rate_a * dt).exp(), coeff_b * (rate_b * dt).exp()),
            (coeff_a * rate_a).mul_add((rate_a * dt).exp(), coeff_b * rate_b * (rate_b * dt).exp()),
        )
    }
}

/// Whether a spring's lanes have settled at `target`: every lane within
/// `1e-3` of the target and slower than `1e-3` per second.
#[must_use]
pub fn settled<L: Lanes>(pos: L, velocity: L, target: L) -> bool {
    pos.near(&target, 1e-3) && velocity.max_abs() < 1e-3
}

/// Evaluates a cubic Bézier `y(x⁻¹(t))` on `[0, 1]`² at parameter `t`
/// (already normalized to `[0, 1]`).
#[must_use]
pub fn curve_value(curve: &Curve, t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    let (x1, y1) = (curve.p1.x, curve.p1.y);
    let (x2, y2) = (curve.p2.x, curve.p2.y);
    let bx = |u: f64| -> f64 {
        let v = 1.0 - u;
        (3.0 * v * v * u).mul_add(x1, (3.0 * v * u * u).mul_add(x2, u * u * u))
    };
    let dbx = |u: f64| -> f64 {
        let v = 1.0 - u;
        (3.0 * v * v).mul_add(
            x1,
            (6.0 * v * u).mul_add(x2 - x1, (3.0 * u * u).mul_add(1.0 - x2, 0.0)),
        )
    };
    // Invert x(u) = t for u: Newton iterations, falling back to bisection
    // on a bracket when the slope vanishes or Newton leaves it.
    let mut u = t;
    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..8 {
        let err = bx(u) - t;
        if err.abs() < 1e-6 {
            break;
        }
        if err > 0.0 {
            hi = u;
        } else {
            lo = u;
        }
        let slope = dbx(u);
        u = if slope.abs() > 1e-6 {
            (u - err / slope).clamp(lo, hi)
        } else {
            f64::midpoint(lo, hi)
        };
    }
    let v = 1.0 - u;
    (3.0 * v * v * u).mul_add(y1, (3.0 * v * u * u).mul_add(y2, u * u * u))
}

/// The position and velocity of an exponential decay `dt` seconds in:
/// `x(t) = x₀ + v·(1 − e^(−k·t)) / k`, `v(t) = v·e^(−k·t)`.
#[must_use]
pub fn decay_step<L: Lanes>(from: L, velocity: L, deceleration: f64, dt: f64) -> (L, L) {
    let k = deceleration.max(1e-9);
    let e = (-k * dt).exp();
    (from.add_scaled(&velocity, (1.0 - e) / k), velocity.scale(e))
}

/// The total distance a decay covers from `velocity`: `|v| / k`.
#[must_use]
#[cfg(test)]
pub fn decay_distance(velocity: Vec2, deceleration: f64) -> Vec2 {
    let k = deceleration.max(1e-9);
    Vec2::new(velocity.x / k, velocity.y / k)
}

/// The nearest point of `bounds` to `p`.
#[must_use]
pub const fn clamp_to_rect(p: Vec2, bounds: Rect) -> Vec2 {
    Vec2::new(
        p.x.clamp(bounds.min_x(), bounds.max_x()),
        p.y.clamp(bounds.min_y(), bounds.max_y()),
    )
}

/// The critically damped spring a rubber-banding decay hands off to.
#[must_use]
pub const fn rubber_band_spring() -> Spring {
    Spring {
        response: 0.4,
        damping: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 1.0 / 120.0;

    fn run_spring(
        from: [f64; 1],
        vel: [f64; 1],
        target: [f64; 1],
        spring: &Spring,
        steps: usize,
    ) -> ([f64; 1], [f64; 1]) {
        let (mut pos, mut v) = (from, vel);
        for _ in 0..steps {
            let r = spring_step(pos, v, target, spring, DT);
            pos = r.0;
            v = r.1;
        }
        (pos, v)
    }

    #[test]
    fn spring_settles_to_target() {
        let spring = Spring::snappy();
        let (pos, vel) = run_spring([0.0], [0.0], [10.0], &spring, 120 * 4);
        assert!(pos.near(&[10.0], 1e-3), "pos {pos:?}");
        assert!(vel.max_abs() < 1e-3);
    }

    #[test]
    fn critically_damped_spring_never_overshoots() {
        let spring = Spring::smooth();
        let (mut pos, mut v) = ([0.0], [0.0]);
        for _ in 0..120 * 4 {
            let r = spring_step(pos, v, [10.0], &spring, DT);
            pos = r.0;
            v = r.1;
            assert!(pos[0] <= 10.0 + 1e-9, "overshot to {}", pos[0]);
        }
    }

    #[test]
    fn curve_hits_exact_endpoints() {
        let curve = Curve::ease_in_out(Duration::from_millis(300));
        assert!(curve_value(&curve, 0.0).abs() < f64::EPSILON);
        assert!((curve_value(&curve, 1.0) - 1.0).abs() < f64::EPSILON);
        assert!((curve_value(&curve, 2.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn linear_curve_is_linear_at_midpoint() {
        let curve = Curve::linear(Duration::from_secs(1));
        let mid = curve_value(&curve, 0.5);
        assert!((mid - 0.5).abs() < 1e-6, "mid {mid}");
        let q = curve_value(&curve, 0.25);
        assert!((q - 0.25).abs() < 1e-6, "quarter {q}");
    }

    #[test]
    fn decay_covers_v_over_k() {
        let velocity = Vec2::new(800.0, -400.0);
        let decay = Decay::new(velocity);
        let (mut pos, mut vel) = (Vec2::ZERO.into_lanes(), velocity.into_lanes());
        for _ in 0..120 * 8 {
            let r = decay_step(pos, vel, decay.deceleration, DT);
            pos = r.0;
            vel = r.1;
        }
        let expected = decay_distance(velocity, decay.deceleration);
        let pos = Vec2::from_lanes(pos);
        assert!(
            (pos - Vec2::new(expected.x, expected.y)).hypot() < 1.0,
            "decay ended at {pos:?}, expected {expected:?}"
        );
        assert!(Vec2::from_lanes(vel).hypot() < 1e-3);
    }

    #[test]
    fn rubber_band_returns_to_bounds() {
        // Simulated at the track level in the tree tests; here check the
        // handoff spring reaches the bound from a position outside it.
        let bounds = Rect::new(0.0, 0.0, 100.0, 50.0);
        let target = clamp_to_rect(Vec2::new(140.0, 30.0), bounds);
        assert_eq!(target, Vec2::new(100.0, 30.0));
        let spring = rubber_band_spring();
        let (pos, vel) = run_spring([140.0], [-200.0], [target.x], &spring, 120 * 2);
        assert!(pos.near(&[100.0], 1e-2), "pos {pos:?}");
        assert!(vel.max_abs() < 1.0);
    }

    #[test]
    fn retarget_is_continuous_in_position_and_velocity() {
        let spring = Spring::snappy();
        // Sample a running spring at t = 0.1.
        let (pos_at, vel_at) = spring_step([0.0], [0.0], [10.0], &spring, 0.1);
        // Retarget: new track starts from that position and velocity.
        let new_target = [30.0];
        let eps = 1e-3;
        let (pos_next, _) = spring_step(pos_at, vel_at, new_target, &spring, eps);
        let expected = vel_at[0].mul_add(eps, pos_at[0]);
        assert!(
            (pos_next[0] - expected).abs() / expected.abs().max(1.0) < 0.01,
            "position {} vs expected {expected}",
            pos_next[0]
        );
    }

    #[test]
    fn affine_and_vec2_lanes_round_trip() {
        let a = Affine::translate((3.0, -2.0)) * Affine::rotate(0.3);
        assert_eq!(Affine::from_lanes(a.into_lanes()), a);
        let v = Vec2::new(1.5, -7.25);
        assert_eq!(Vec2::from_lanes(v.into_lanes()), v);
        let o: f32 = 0.625;
        assert!((f32::from_lanes(o.into_lanes()) - o).abs() < f32::EPSILON);
    }
}
