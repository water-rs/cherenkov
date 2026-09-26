// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! One-time layer motion for the cherenkov adapters: a scene `motion`
//! translated into the front-end's animation types and driven on a fixed
//! frame clock.
//!
//! A `motion` starts from its `from` state (a plain set) and animates or
//! decays to the layer's static property, so a `readback` run renders
//! frames until `Next::Idle` and compares the settled scene against the
//! oracle.

use std::time::{Duration, Instant};

use cherenkov::{Animation, Backend, Curve, Decay, FrameTime, Layer, Spring, Surface};
use cherenkov_scene::{Motion, MotionAnimation};
use kurbo::{Affine, Vec2};

/// A scene [`Motion`] translated into front-end animation types.
pub enum LayerMotion {
    /// `transform` animates `from` → `to` under `animation`.
    Transform {
        /// The start transform.
        from: Affine,
        /// The static (settled) transform.
        to: Affine,
        /// How it moves.
        animation: Animation,
    },
    /// `scroll_offset` decays from `from` under `decay`; a decay starts
    /// at the committed value and travels `velocity / deceleration`, so
    /// the generator arranges `from + v/k` to be the static offset.
    Scroll {
        /// The start offset.
        from: Vec2,
        /// The decay parameters.
        decay: Decay,
    },
}

impl LayerMotion {
    /// Translates a scene `motion`, given the layer's static `transform`.
    /// The static `scroll_offset` is unused: a decay commits its `from`
    /// state and comes to rest at `from + velocity / deceleration`.
    pub fn from_scene(motion: &Motion, transform: Affine) -> Self {
        match motion {
            Motion::Transform { from, animation } => Self::Transform {
                from: *from,
                to: transform,
                animation: motion_animation(*animation),
            },
            Motion::Scroll {
                from,
                velocity,
                deceleration,
                bounds,
            } => Self::Scroll {
                from: *from,
                decay: Decay {
                    velocity: *velocity,
                    deceleration: *deceleration,
                    rubber_band: *bounds,
                },
            },
        }
    }

    /// Commits the motion on `layer`: the plain `from` set followed by the
    /// animated commit to the static value.
    pub fn apply<B: Backend>(&self, surface: &Surface<B>, layer: &Layer) {
        match self {
            Self::Transform {
                from,
                to,
                animation,
            } => {
                surface.update(|tx| {
                    tx[layer].transform(*from);
                });
                surface.update_animated(*animation, |tx| {
                    tx[layer].transform(*to);
                });
            }
            Self::Scroll { from, decay } => {
                surface.update(|tx| {
                    tx[layer].scroll_offset(*from);
                });
                surface.update(|tx| {
                    tx[layer].scroll_offset(*from).animation(*decay);
                });
            }
        }
    }
}

/// The scene `MotionAnimation` → the front-end `Animation`.
fn motion_animation(animation: MotionAnimation) -> Animation {
    match animation {
        MotionAnimation::Spring { response, damping } => {
            Animation::from(Spring { response, damping })
        }
        MotionAnimation::Curve {
            duration_ms,
            x1,
            y1,
            x2,
            y2,
        } => Animation::from(Curve::bezier(
            Duration::from_millis(duration_ms),
            x1,
            y1,
            x2,
            y2,
        )),
    }
}

/// The adapter's frame clock: a fixed origin advanced a 1/120 s tick per
/// frame, so animation sampling is deterministic across engines.
pub struct Clock {
    origin: Instant,
    frame: u64,
}

/// The tick the frame clock advances per frame.
const TICK: Duration = Duration::from_nanos(1_000_000_000 / 120);

impl Clock {
    /// A clock at frame zero, origin now.
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
            frame: 0,
        }
    }

    /// The current frame time.
    pub fn time(&self) -> FrameTime {
        FrameTime::at(self.origin + TICK * u32::try_from(self.frame).unwrap_or(u32::MAX))
    }

    /// Advances the clock one tick.
    pub fn advance(&mut self) {
        self.frame += 1;
    }
}
