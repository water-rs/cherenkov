// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use serde::{Deserialize, Serialize};

use crate::{BlendMode, Draw, Shape};
use kurbo::Affine;

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
    /// Ordered items: child layers and draw commands.
    #[serde(default)]
    pub items: Vec<Item>,
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
            items: Vec::new(),
        }
    }
}
