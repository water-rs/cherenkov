// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use serde::{Deserialize, Serialize};

use crate::{BlendMode, Draw, Shape};
use kurbo::{Affine, Rect, Vec2};

/// One item in a layer's ordered item list: a child layer or a draw command.
///
/// Serialized externally tagged (`{"draw": ..}` / `{"layer": ..}`): internally
/// and untagged serde representations cannot nest enums within the buffered
/// content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Item {
    /// A draw command.
    Draw(Draw),
    /// A child layer.
    Layer(Layer),
}

/// A layer of the scene tree.
///
/// Items are drawn in order into the layer's own buffer; the layer is then
/// composited onto its parent with `transform`, `clip`, `opacity` and
/// `blend` applied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    /// The layer's affine transform relative to its parent.
    #[serde(default)]
    pub transform: Affine,
    /// An optional clip shape (in the layer's own coordinate space) masking
    /// the whole layer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip: Option<Shape>,
    /// Group opacity, `0.0..=1.0`. `1.0` (the default) leaves alpha unchanged.
    #[serde(default = "Layer::default_opacity")]
    pub opacity: f64,
    /// The blend mode used when compositing onto the parent.
    #[serde(default)]
    pub blend: BlendMode,
    /// The layer's scroll offset: content and children are translated by
    /// `-scroll_offset` inside the layer's clip; `transform` is untouched.
    /// Zero (the default) draws them untranslated.
    #[serde(default, skip_serializing_if = "vec2_is_zero")]
    pub scroll_offset: Vec2,
    /// One-time motion for this layer: an animation or decay that runs
    /// from its `from` state and comes to rest at the layer's static
    /// properties. Engines that cannot animate report it unsupported and
    /// fall back to the settled scene.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub motion: Option<Motion>,
    /// Ordered items: child layers and draw commands.
    #[serde(default)]
    pub items: Vec<Item>,
}

/// A layer's one-time motion, applied once when it enters the scene.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "motion", content = "value")]
pub enum Motion {
    /// `transform` starts at `from` and animates to the layer's
    /// `transform`.
    Transform {
        /// The transform the layer starts at.
        from: Affine,
        /// How it moves to the static `transform`.
        animation: MotionAnimation,
    },
    /// `scroll_offset` starts at `from` and decays with `velocity`
    /// (deceleration per second), optionally rubber-banding to `bounds`.
    /// It must come to rest at the layer's static `scroll_offset` (the
    /// generator guarantees this).
    Scroll {
        /// The scroll offset the layer starts at.
        from: Vec2,
        /// The fling velocity in px/s.
        velocity: Vec2,
        /// Deceleration in px/s².
        deceleration: f64,
        /// Optional rubber-band bounds.
        bounds: Option<Rect>,
    },
}

/// How a [`Motion::Transform`] animates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "animation", content = "value")]
pub enum MotionAnimation {
    /// A physical spring (`response` seconds, `damping` ratio).
    Spring {
        /// The spring's response period.
        response: f64,
        /// The spring's damping ratio; below 1.0 overshoots.
        damping: f64,
    },
    /// A Bézier-timed curve.
    Curve {
        /// Duration in milliseconds.
        duration_ms: u64,
        /// First control point x.
        x1: f64,
        /// First control point y.
        y1: f64,
        /// Second control point x.
        x2: f64,
        /// Second control point y.
        y2: f64,
    },
}

/// Serde helper: a zero `scroll_offset` is left out of the JSON.
fn vec2_is_zero(v: &Vec2) -> bool {
    *v == Vec2::ZERO
}

impl Layer {
    const fn default_opacity() -> f64 {
        1.0
    }
}

impl Default for Layer {
    fn default() -> Self {
        Self {
            transform: Affine::IDENTITY,
            clip: None,
            opacity: 1.0,
            blend: BlendMode::Normal,
            scroll_offset: Vec2::ZERO,
            motion: None,
            items: Vec::new(),
        }
    }
}
