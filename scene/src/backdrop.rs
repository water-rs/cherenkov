// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use serde::{Deserialize, Serialize};

/// A backdrop group: one backdrop capture shared by every layer that
/// references its `id` ([`crate::Layer::backdrop`]), plus the filter chain
/// applied to that capture once.
///
/// The group's capture is taken at the moment its first member (in paint
/// order) begins, over the content already drawn into the member's
/// compositing canvas; `filters` then run over the capture and every member
/// composites the filtered result as the bottom-most content inside its own
/// clip. A member layer must have a clip ([`crate::Scene::load`] validates
/// this) or the scene fails to load.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BackdropGroup {
    /// The id member layers reference.
    pub id: u32,
    /// The filters applied to the capture, in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub filters: Vec<BackdropFilter>,
}

/// One filter in a backdrop group's capture chain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "filter", content = "value", rename_all = "kebab-case")]
pub enum BackdropFilter {
    /// A separable Gaussian blur (filtrate's `GaussianBlur`): kernel radius
    /// `⌈3σ⌉`, clamp-to-edge sampling at the capture boundary.
    GaussianBlur {
        /// The blur's standard deviation in pixels.
        sigma: f64,
    },
    /// A 3×4 colour matrix (filtrate's `ColorMatrix<T>([T;12])`): three rows
    /// of four applied to the premultiplied `[r, g, b, a]` pixel —
    /// `out_i = dot(row_i, pixel)` for `i` in `0..3`, the fourth column a
    /// bias that scales with alpha. The output alpha is the input alpha.
    ColorMatrix {
        /// The 12 coefficients, row-major.
        matrix: [f64; 12],
    },
}
