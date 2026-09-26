// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use serde::{Deserialize, Serialize};

use crate::{Color, ColorSpace, ResourceHash, Shape};
use kurbo::{Affine, Point, Rect};

/// How a fill decides coverage: the winding rule or parity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FillRule {
    /// The default non-zero winding rule.
    #[default]
    NonZero,
    /// The even-odd parity rule.
    EvenOdd,
}

/// How source pixels combine with the destination.
///
/// The first 16 variants are the separable and non-separable blend modes of
/// W3C Compositing and Blending Level 1 (all composed source-over). The
/// remaining variants are the other Porter-Duff compositing operators and
/// plus-lighter, needed to express every `COLRv1` `PaintComposite` mode.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum BlendMode {
    /// Plain source-over compositing.
    #[default]
    Normal,
    /// Separable modes.
    Multiply,
    /// Screen.
    Screen,
    /// Overlay.
    Overlay,
    /// Darken.
    Darken,
    /// Lighten.
    Lighten,
    /// Colour dodge.
    ColorDodge,
    /// Colour burn.
    ColorBurn,
    /// Hard light.
    HardLight,
    /// Soft light.
    SoftLight,
    /// Difference.
    Difference,
    /// Exclusion.
    Exclusion,
    /// Non-separable (colour-component) modes.
    Hue,
    /// Saturation.
    Saturation,
    /// Colour.
    Color,
    /// Luminosity.
    Luminosity,
    /// Porter-Duff compositing operators and plus-lighter (`COLRv1`
    /// `PaintComposite` modes).
    /// Both source and destination are cleared.
    Clear,
    /// The source replaces the destination.
    Src,
    /// The destination replaces the source (source discarded).
    Dst,
    /// The destination is placed over the source.
    DestOver,
    /// The parts of the source that overlap the destination.
    SrcIn,
    /// The parts of the destination that overlap the source.
    DestIn,
    /// The parts of the source outside the destination.
    SrcOut,
    /// The parts of the destination outside the source.
    DestOut,
    /// The parts of the source overlapping the destination replace it.
    SrcAtop,
    /// The parts of the destination overlapping the source replace it.
    DestAtop,
    /// The non-overlapping regions of source and destination.
    Xor,
    /// Source and destination are summed without clamping (`COLRv1`'s `Plus`
    /// mode — unlike the CSS `plus-darker`, the sum is not clamped).
    PlusLighter,
}

impl BlendMode {
    /// Every blend/composite mode, for adapters claiming them all.
    pub const ALL: [Self; 28] = [
        Self::Normal,
        Self::Multiply,
        Self::Screen,
        Self::Overlay,
        Self::Darken,
        Self::Lighten,
        Self::ColorDodge,
        Self::ColorBurn,
        Self::HardLight,
        Self::SoftLight,
        Self::Difference,
        Self::Exclusion,
        Self::Hue,
        Self::Saturation,
        Self::Color,
        Self::Luminosity,
        Self::Clear,
        Self::Src,
        Self::Dst,
        Self::DestOver,
        Self::SrcIn,
        Self::DestIn,
        Self::SrcOut,
        Self::DestOut,
        Self::SrcAtop,
        Self::DestAtop,
        Self::Xor,
        Self::PlusLighter,
    ];
}

/// Edge behaviour of a gradient or image pattern outside its domain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Extend {
    /// Clamp to the edge colour (pad).
    #[default]
    Pad,
    /// Repeat the gradient period.
    Repeat,
    /// Mirror the gradient period.
    Reflect,
    /// Transparent outside the domain.
    None,
}

/// Texture sampling quality.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Sampling {
    /// Nearest-neighbour sampling.
    Nearest,
    /// Bilinear interpolation.
    #[default]
    Bilinear,
}

/// A colour stop of a gradient.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GradientStop {
    /// Position along the gradient domain, normally `0.0..=1.0`.
    pub offset: f32,
    /// The stop colour; stops may mix colour spaces.
    pub color: Color,
}

/// A linear gradient from `start` to `end`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearGradient {
    /// Line start (the `t = 0` point).
    pub start: Point,
    /// Line end (the `t = 1` point).
    pub end: Point,
    /// Colour stops, sorted by offset.
    pub stops: Vec<GradientStop>,
    /// Behaviour outside `[start, end]`.
    pub extend: Extend,
    /// The space stops are interpolated in.
    pub interpolation: ColorSpace,
}

/// A two-point ("focal") radial gradient: the interpolated circle moves from
/// `(center0, r0)` at `t = 0` to `(center1, r1)` at `t = 1`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RadialGradient {
    /// Centre of the start circle.
    pub center0: Point,
    /// Radius of the start circle (may be `0` for a focal point).
    pub r0: f64,
    /// Centre of the end circle.
    pub center1: Point,
    /// Radius of the end circle.
    pub r1: f64,
    /// Colour stops, sorted by offset.
    pub stops: Vec<GradientStop>,
    /// Behaviour outside the domain.
    pub extend: Extend,
    /// The space stops are interpolated in.
    pub interpolation: ColorSpace,
}

/// A sweep (conical) gradient rotating around `center` from `start_angle` to
/// `end_angle` in radians (clockwise, matching screen coordinates).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SweepGradient {
    /// Centre of rotation.
    pub center: Point,
    /// Angle where `t = 0`, radians.
    pub start_angle: f64,
    /// Angle where `t = 1`, radians.
    pub end_angle: f64,
    /// Colour stops, sorted by offset.
    pub stops: Vec<GradientStop>,
    /// Behaviour outside the angular domain.
    pub extend: Extend,
    /// The space stops are interpolated in.
    pub interpolation: ColorSpace,
}

/// An image pattern paint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImagePaint {
    /// BLAKE3 hash of the image blob in `resources/` (PNG).
    pub image: ResourceHash,
    /// Maps pattern space into the paint's user space.
    pub transform: Affine,
    /// Horizontal edge behaviour.
    pub extend_x: Extend,
    /// Vertical edge behaviour.
    pub extend_y: Extend,
    /// Sampling quality.
    pub sampling: Sampling,
}

/// A paint: what covers the inside of a shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Paint {
    /// A solid colour.
    Solid(Color),
    /// A linear gradient.
    Linear(LinearGradient),
    /// A two-point radial gradient.
    Radial(RadialGradient),
    /// A sweep gradient.
    Sweep(SweepGradient),
    /// An image pattern.
    Image(ImagePaint),
}

impl From<Color> for Paint {
    fn from(color: Color) -> Self {
        Self::Solid(color)
    }
}

/// A stroke style: width, joins, caps and dashes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StrokeStyle {
    /// Stroke width in user-space pixels.
    pub width: f64,
    /// Segment joins.
    pub join: kurbo::Join,
    /// Miter limit for miter joins.
    pub miter_limit: f64,
    /// Cap for the start of open subpaths.
    pub start_cap: kurbo::Cap,
    /// Cap for the end of open subpaths.
    pub end_cap: kurbo::Cap,
    /// Alternating on/off dash lengths (empty = solid).
    pub dash_pattern: Vec<f64>,
    /// Offset of the first dash.
    pub dash_offset: f64,
}

impl Default for StrokeStyle {
    fn default() -> Self {
        Self {
            width: 1.0,
            join: kurbo::Join::Miter,
            miter_limit: 4.0,
            start_cap: kurbo::Cap::Butt,
            end_cap: kurbo::Cap::Butt,
            dash_pattern: Vec::new(),
            dash_offset: 0.0,
        }
    }
}

impl From<&StrokeStyle> for kurbo::Stroke {
    fn from(s: &StrokeStyle) -> Self {
        Self::new(s.width)
            .with_join(s.join)
            .with_miter_limit(s.miter_limit)
            .with_caps(kurbo::Cap::Butt)
            .with_start_cap(s.start_cap)
            .with_end_cap(s.end_cap)
            .with_dashes(s.dash_offset, s.dash_pattern.clone())
    }
}

/// One positioned glyph of a [`GlyphRun`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Glyph {
    /// The glyph id in the font.
    pub id: u32,
    /// Pen-relative x position.
    pub x: f32,
    /// Pen-relative y position.
    pub y: f32,
}

/// A normalized variation coordinate of a variable font.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NormalizedCoord {
    /// The four-character axis tag, e.g. `"wght"`.
    pub tag: String,
    /// The normalized axis value.
    pub value: f32,
}

/// A shaped run of glyphs in one font and style.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GlyphRun {
    /// BLAKE3 hash of the font blob in `resources/`.
    pub font: ResourceHash,
    /// Font index within a TrueType/OpenType collection (`0` for single fonts).
    pub font_index: u32,
    /// Font size in pixels.
    pub size: f32,
    /// Normalized variation coordinates.
    pub normalized_coords: Vec<NormalizedCoord>,
    /// Positioned glyphs (pen-relative).
    pub glyphs: Vec<Glyph>,
    /// The paint used for all glyphs in the run.
    pub paint: Paint,
}

/// A draw command: one item in a layer's ordered item list.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Draw {
    /// Fill a shape.
    Fill {
        /// The shape.
        shape: Shape,
        /// The fill rule.
        rule: FillRule,
        /// The paint.
        paint: Paint,
    },
    /// Stroke a shape's outline.
    Stroke {
        /// The shape.
        shape: Shape,
        /// The stroke style.
        stroke: StrokeStyle,
        /// The paint.
        paint: Paint,
    },
    /// A drop shadow: the shape's coverage blurred and offset, in `color`.
    Shadow {
        /// The shape producing the shadow.
        shape: Shape,
        /// Gaussian blur standard deviation in the shape's units.
        blur_sigma: f64,
        /// Shadow offset in the shape's units.
        offset: [f64; 2],
        /// Shadow colour.
        color: Color,
    },
    /// A shaped glyph run.
    Glyphs(GlyphRun),
    /// An image drawn into a destination rectangle.
    Image {
        /// BLAKE3 hash of the image blob in `resources/` (PNG).
        image: ResourceHash,
        /// Destination rectangle in user space.
        dst: Rect,
        /// Sampling quality.
        sampling: Sampling,
    },
}
