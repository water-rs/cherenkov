//! Glyph runs. Shaping happens outside the engine; a run is positioned glyphs.

use kurbo::{Affine, Stroke};
use serde::{Deserialize, Serialize};

/// A font registered with the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FontId(u64);

impl FontId {
    /// Creates an identifier from a backend-assigned raw value.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw value.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// One positioned glyph.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Glyph {
    /// Glyph index in the font.
    pub id: u32,
    /// Horizontal position of the glyph origin.
    pub x: f32,
    /// Vertical position of the glyph origin.
    pub y: f32,
    /// Per-glyph transform about its origin, for example an upright glyph in
    /// vertical CJK text.
    pub transform: Option<Affine>,
}

/// How glyphs are drawn.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum GlyphStyle {
    /// Filled outlines.
    #[default]
    Fill,
    /// Stroked outlines.
    Stroke(Stroke),
}

/// A run of glyphs sharing a font, size and variation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GlyphRun {
    /// The font.
    pub font: FontId,
    /// Size in the drawing's units per em.
    pub size: f32,
    /// Normalized variation coordinates, in `F2Dot14`.
    pub coords: Vec<i16>,
    /// The glyphs.
    pub glyphs: Vec<Glyph>,
    /// Fill or stroke.
    pub style: GlyphStyle,
}
