// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Cherenkov → peniko conversions. Everything in here is pure and
//! unit-tested; no GPU objects appear.

use cherenkov::kurbo::{Affine, BezPath, PathEl, Point, Rect, Shape as _};
use cherenkov::{BlendMode, ContinuousRect, FillRule, Interpolation, ShapeData};
use vello::peniko;
use vello::peniko::color::{AlphaColor, ColorSpace, ColorSpaceTag, DynamicColor, LinearSrgb, Srgb};

use crate::error::Unsupported;

/// The target rect covering a whole `width` × `height` surface: vello's
/// `push_layer` requires a clip, so layers without one clip to the target.
#[must_use]
pub fn opaque_clip(width: u32, height: u32) -> BezPath {
    Rect::new(0.0, 0.0, f64::from(width), f64::from(height)).to_path(0.25)
}

/// Working colour (linear Display P3, straight alpha, possibly HDR) →
/// peniko's sRGB `Color`. HDR and out-of-gamut values clamp implicitly in
/// the 8-bit target.
#[must_use]
pub fn color(c: &cherenkov::WorkingColor) -> peniko::Color {
    AlphaColor::<cherenkov::LinearDisplayP3>::new(c.components).convert::<Srgb>()
}

/// A gradient stop: the stop's linear-P3 channels are expressed as linear
/// sRGB (the same primaries vello evaluates linear-space ramps in; the
/// P3↔sRGB matrix commutes with linear interpolation, so this is exact).
#[must_use]
#[expect(clippy::many_single_char_names, reason = "r/g/b/a channel names")]
pub fn stop(s: &cherenkov::ColorStop) -> peniko::ColorStop {
    let [r, g, b, a] = s.color.components;
    let [r, g, b] = cherenkov::LinearDisplayP3::to_linear_srgb([r, g, b]);
    peniko::ColorStop::from((
        s.offset,
        DynamicColor::from_alpha_color(AlphaColor::<LinearSrgb>::new([r, g, b, a])),
    ))
}

/// The colour-space tag a gradient's interpolation space maps to.
///
/// # Errors
/// [`Unsupported::Interpolation`] for any space that has no equivalent
/// `ColorSpaceTag`.
pub const fn interpolation(i: Interpolation) -> Result<ColorSpaceTag, Unsupported> {
    match i {
        // Linear interpolation commutes with the P3↔sRGB matrix, so
        // declaring LinearSrgb to vello is exact, not a remap.
        Interpolation::Working => Ok(ColorSpaceTag::LinearSrgb),
        Interpolation::SrgbEncoded => Ok(ColorSpaceTag::Srgb),
        #[expect(
            unreachable_patterns,
            reason = "Interpolation has two variants; a future variant must become a decision"
        )]
        _ => Err(Unsupported::Interpolation),
    }
}

/// `cherenkov::Extend` → `peniko::Extend`. Peniko has no transparent
/// extend, so [`Unsupported::Extend`] is returned for `Extend::None`.
///
/// # Errors
/// [`Unsupported::Extend`] for `Extend::None`.
pub const fn extend(e: cherenkov::Extend) -> Result<peniko::Extend, Unsupported> {
    match e {
        cherenkov::Extend::Pad => Ok(peniko::Extend::Pad),
        cherenkov::Extend::Repeat => Ok(peniko::Extend::Repeat),
        cherenkov::Extend::Reflect => Ok(peniko::Extend::Reflect),
        cherenkov::Extend::None => Err(Unsupported::Extend),
    }
}

/// `cherenkov::BlendMode` → `peniko::BlendMode`: separable and
/// non-separable modes become a `Mix` over `SrcOver`; Porter-Duff modes
/// become a `Compose` under `Mix::Normal`.
#[must_use]
pub const fn blend(m: BlendMode) -> peniko::BlendMode {
    use peniko::{Compose, Mix};
    let compose = match m {
        BlendMode::Clear => Some(Compose::Clear),
        BlendMode::Src => Some(Compose::Copy),
        BlendMode::Dst => Some(Compose::Dest),
        BlendMode::DestOver => Some(Compose::DestOver),
        BlendMode::SrcIn => Some(Compose::SrcIn),
        BlendMode::DestIn => Some(Compose::DestIn),
        BlendMode::SrcOut => Some(Compose::SrcOut),
        BlendMode::DestOut => Some(Compose::DestOut),
        BlendMode::SrcAtop => Some(Compose::SrcAtop),
        BlendMode::DestAtop => Some(Compose::DestAtop),
        BlendMode::Xor => Some(Compose::Xor),
        BlendMode::PlusLighter => Some(Compose::PlusLighter),
        _ => None,
    };
    if let Some(compose) = compose {
        return peniko::BlendMode::new(Mix::Normal, compose);
    }
    let mix = match m {
        BlendMode::Normal
        | BlendMode::Clear
        | BlendMode::Src
        | BlendMode::Dst
        | BlendMode::DestOver
        | BlendMode::SrcIn
        | BlendMode::DestIn
        | BlendMode::SrcOut
        | BlendMode::DestOut
        | BlendMode::SrcAtop
        | BlendMode::DestAtop
        | BlendMode::Xor
        | BlendMode::PlusLighter => Mix::Normal,
        BlendMode::Multiply => Mix::Multiply,
        BlendMode::Screen => Mix::Screen,
        BlendMode::Overlay => Mix::Overlay,
        BlendMode::Darken => Mix::Darken,
        BlendMode::Lighten => Mix::Lighten,
        BlendMode::ColorDodge => Mix::ColorDodge,
        BlendMode::ColorBurn => Mix::ColorBurn,
        BlendMode::HardLight => Mix::HardLight,
        BlendMode::SoftLight => Mix::SoftLight,
        BlendMode::Difference => Mix::Difference,
        BlendMode::Exclusion => Mix::Exclusion,
        BlendMode::Hue => Mix::Hue,
        BlendMode::Saturation => Mix::Saturation,
        BlendMode::Color => Mix::Color,
        BlendMode::Luminosity => Mix::Luminosity,
    };
    peniko::BlendMode::new(mix, peniko::Compose::SrcOver)
}

/// `cherenkov::FillRule` → `peniko::Fill`.
#[must_use]
pub const fn fill(r: FillRule) -> peniko::Fill {
    match r {
        FillRule::NonZero => peniko::Fill::NonZero,
        FillRule::EvenOdd => peniko::Fill::EvenOdd,
    }
}

/// A layer clip or drawing shape as a `BezPath`.
#[must_use]
pub fn shape_path(shape: &ShapeData) -> BezPath {
    match shape {
        ShapeData::Rect(r) => r.to_path(cherenkov::PATH_TOLERANCE),
        ShapeData::RoundedRect(r) => r.to_path(cherenkov::PATH_TOLERANCE),
        ShapeData::Continuous(c) => continuous_path(c),
        ShapeData::Circle(c) => c.to_path(cherenkov::PATH_TOLERANCE),
        ShapeData::Ellipse(e) => e.to_path(cherenkov::PATH_TOLERANCE),
        ShapeData::Line(l) => {
            let mut p = BezPath::new();
            p.push(PathEl::MoveTo(l.p0));
            p.push(PathEl::LineTo(l.p1));
            p
        }
        ShapeData::Path { elements, .. } => BezPath::from_iter(elements.iter().copied()),
    }
}

/// A shape's fill rule.
#[must_use]
pub const fn shape_rule(shape: &ShapeData) -> FillRule {
    match shape {
        ShapeData::Path { rule, .. } => *rule,
        _ => FillRule::NonZero,
    }
}

/// Expands a [`ContinuousRect`] into a polyline path of Lamé corner arcs.
///
/// Each corner is a quarter Lamé curve `x = r·|cos t|^e`, `y = r·|sin t|^e`
/// with `e = 2/n` and `n = 2 + 2·smoothing`, bisected in `t` until every
/// sampled point lies within `tolerance` of its chord. `smoothing = 0` is a
/// circular corner (matching a [`RoundedRect`]); `1` is the squircle.
/// Per-corner radii are clamped to the rect's half-extents.
#[must_use]
pub fn continuous_path(c: &ContinuousRect) -> BezPath {
    continuous_path_at(c, cherenkov::PATH_TOLERANCE)
}

/// [`continuous_path`] at an explicit tolerance.
#[allow(clippy::many_single_char_names)] // x/y/r/e/n geometry names
fn continuous_path_at(c: &ContinuousRect, tolerance: f64) -> BezPath {
    /// Emits the chord `[t0, t1]` of corner `corner`, bisecting until the
    /// sampled deviation from the chord is within `tol`.
    fn emit(
        corner: usize,
        t0: f64,
        t1: f64,
        arc: &dyn Fn(usize, f64) -> Point,
        tol: f64,
        path: &mut BezPath,
        depth: u32,
    ) {
        const MAX_DEPTH: u32 = 24;
        let (p0, p1) = (arc(corner, t0), arc(corner, t1));
        // Probe the quarter and three-quarter points too: a Lamé arc is
        // steepest near its ends, so one midpoint check can miss it.
        let flat = (1..4).all(|k| {
            let s = f64::from(k) * 0.25;
            let pm = arc(corner, (t1 - t0).mul_add(s, t0));
            let (cx, cy) = (p0.x + s * (p1.x - p0.x), p0.y + s * (p1.y - p0.y));
            (pm.x - cx).hypot(pm.y - cy) <= tol
        });
        if flat || depth >= MAX_DEPTH {
            path.line_to(p1);
        } else {
            let tm = (t1 - t0).mul_add(0.5, t0);
            emit(corner, t0, tm, arc, tol, path, depth + 1);
            emit(corner, tm, t1, arc, tol, path, depth + 1);
        }
    }

    let n = 2.0f64.mul_add(c.smoothing.clamp(0.0, 1.0), 2.0);
    let e = 2.0 / n;
    let hw = c.rect.width() / 2.0;
    let hh = c.rect.height() / 2.0;
    let radii = c.radii;
    // Per-corner radii, each clamped to the half-extents.
    let r = [
        radii.top_right.clamp(0.0, hw.min(hh)),
        radii.bottom_right.clamp(0.0, hw.min(hh)),
        radii.bottom_left.clamp(0.0, hw.min(hh)),
        radii.top_left.clamp(0.0, hw.min(hh)),
    ];
    let Rect { x0, y0, x1, y1 } = c.rect;
    // Corner centres in order top-right, bottom-right, bottom-left, top-left.
    let corners = [
        (x1 - r[0], y0 + r[0]),
        (x1 - r[1], y1 - r[1]),
        (x0 + r[2], y1 - r[2]),
        (x0 + r[3], y0 + r[3]),
    ];
    // Point on corner `c`'s Lamé arc at parameter `t ∈ [0, π/2]`.
    let arc = |c: usize, t: f64| -> Point {
        let (cx, cy) = corners[c];
        let (s, co) = (r[c] * t.sin().powf(e), r[c] * t.cos().powf(e));
        let (dx, dy) = match c {
            0 => (s, -co),  // TR: from (cx, cy-r) to (cx+r, cy)
            1 => (co, s),   // BR: from (cx+r, cy) to (cx, cy+r)
            2 => (-s, co),  // BL: from (cx, cy+r) to (cx-r, cy)
            _ => (-co, -s), // TL: from (cx-r, cy) to (cx, cy-r)
        };
        Point::new(cx + dx, cy + dy)
    };
    let mut path = BezPath::new();
    path.move_to((x0 + r[3], y0));
    for corner in 0..corners.len() {
        path.line_to(arc(corner, 0.0));
        emit(
            corner,
            0.0,
            std::f64::consts::FRAC_PI_2,
            &arc,
            tolerance,
            &mut path,
            0,
        );
    }
    path.close_path();
    path
}

/// Maps a [`cherenkov::Shadow`]-bearing shape to the `(rect, uniform
/// radius)` vello's blurred rounded-rect primitive can express: rects,
/// uniform-radius rounded rects, circles and circles written as ellipses.
/// Returns `None` for anything else (non-uniform radii, ellipses,
/// continuous corners, paths, lines).
#[must_use]
pub fn expressible_shadow(shape: &ShapeData) -> Option<(Rect, f64)> {
    match shape {
        ShapeData::Rect(rect) => Some((*rect, 0.0)),
        ShapeData::Circle(c) => {
            let r = c.radius;
            Some((
                Rect::new(
                    c.center.x - r,
                    c.center.y - r,
                    c.center.x + r,
                    c.center.y + r,
                ),
                r,
            ))
        }
        ShapeData::Ellipse(e) if (e.radii().x - e.radii().y).abs() < f64::EPSILON => {
            let r = e.radii().x;
            let c = e.center();
            Some((Rect::new(c.x - r, c.y - r, c.x + r, c.y + r), r))
        }
        ShapeData::RoundedRect(rrect) => {
            let radii = rrect.radii();
            let uniform = (radii.top_left - radii.top_right).abs() < f64::EPSILON
                && (radii.top_left - radii.bottom_right).abs() < f64::EPSILON
                && (radii.top_left - radii.bottom_left).abs() < f64::EPSILON;
            uniform.then(|| (rrect.rect(), radii.top_left))
        }
        _ => None,
    }
}

/// The transform mapping an `iw` × `ih` image onto the destination rect.
#[must_use]
pub fn image_draw_transform(iw: u32, ih: u32, dst: Rect) -> Affine {
    Affine::translate((dst.x0, dst.y0))
        * Affine::scale_non_uniform(dst.width() / f64::from(iw), dst.height() / f64::from(ih))
}

/// `cherenkov::Sampling` → `peniko::ImageQuality`.
#[must_use]
pub const fn quality(s: cherenkov::Sampling) -> peniko::ImageQuality {
    match s {
        cherenkov::Sampling::Nearest => peniko::ImageQuality::Low,
        cherenkov::Sampling::Linear => peniko::ImageQuality::Medium,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cherenkov::WorkingColor;

    /// Signed area of a path via the line integral over its segments.
    fn path_area(path: &BezPath) -> f64 {
        use cherenkov::kurbo::ParamCurveArea;
        path.segments().map(|seg| seg.signed_area()).sum()
    }

    #[test]
    fn working_color_converts_to_srgb() {
        // Working-space white is sRGB white.
        let white = color(&WorkingColor::WHITE);
        for c in white.components {
            assert!((c - 1.).abs() < 1e-6, "{white:?}");
        }
        // Linear-P3 full red maps into sRGB-encoded P3 red.
        let red = color(&WorkingColor::new([1., 0., 0., 1.]));
        assert!(red.components[0] > 0.9 && red.components[1] < 0.6);
        // Alpha survives unchanged.
        let half = color(&WorkingColor::new([0.5, 0.5, 0.5, 0.25]));
        assert!((half.components[3] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn srgb_working_round_trip() {
        // An sRGB mid-gray converted into working space and back lands on
        // the same encoded value.
        let working = cherenkov::Color::<cherenkov::Srgb>::new([0.5, 0.5, 0.5, 1.]).to_working();
        let back = color(&working);
        assert!((f64::from(back.components[0]) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn every_blend_mode_maps() {
        for m in [
            BlendMode::Normal,
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Overlay,
            BlendMode::Darken,
            BlendMode::Lighten,
            BlendMode::ColorDodge,
            BlendMode::ColorBurn,
            BlendMode::HardLight,
            BlendMode::SoftLight,
            BlendMode::Difference,
            BlendMode::Exclusion,
            BlendMode::Hue,
            BlendMode::Saturation,
            BlendMode::Color,
            BlendMode::Luminosity,
        ] {
            let b = blend(m);
            assert_eq!(b.compose, peniko::Compose::SrcOver);
            if m == BlendMode::Normal {
                assert_eq!(b.mix, peniko::Mix::Normal);
            }
        }
    }

    #[test]
    fn interpolation_maps_the_two_spaces() {
        assert_eq!(
            interpolation(Interpolation::Working),
            Ok(ColorSpaceTag::LinearSrgb)
        );
        assert_eq!(
            interpolation(Interpolation::SrgbEncoded),
            Ok(ColorSpaceTag::Srgb)
        );
    }

    #[test]
    fn continuous_rect_with_zero_smoothing_matches_rounded_rect_area() {
        let rect = Rect::new(0., 0., 100., 60.);
        let continuous = ContinuousRect::new(rect, 20.).with_smoothing(0.);
        let rounded = cherenkov::kurbo::RoundedRect::from_rect(rect, 20.);
        let a = path_area(&continuous_path(&continuous));
        let b = path_area(&rounded.to_path(cherenkov::PATH_TOLERANCE));
        assert!(
            (a - b).abs() / b.abs() < 0.005,
            "area {a} differs from rounded rect {b}"
        );
    }

    #[test]
    fn continuous_rect_per_corner_radii() {
        // Only the top-left corner rounded: the path must start inset by
        // that corner's radius and the other corners stay sharp.
        let c = ContinuousRect::new(
            Rect::new(0., 0., 10., 10.),
            cherenkov::kurbo::RoundedRectRadii::new(4., 0., 0., 0.),
        )
        .with_smoothing(0.);
        let path = continuous_path(&c);
        let bbox = path.bounding_box();
        assert!((bbox.x0 - 0.).abs() < 1e-6 && (bbox.x1 - 10.).abs() < 1e-6);
        // First move lands at the top edge after the top-left arc start.
        match path.elements()[0] {
            PathEl::MoveTo(p) => assert!((p.x - 4.).abs() < 1e-9 && (p.y - 0.).abs() < 1e-9),
            el => panic!("expected move_to, got {el:?}"),
        }
    }

    #[test]
    fn shadow_shapes() {
        assert!(expressible_shadow(&ShapeData::Rect(Rect::new(0., 0., 1., 1.))).is_some());
        let rrect = ShapeData::RoundedRect(cherenkov::kurbo::RoundedRect::from_rect(
            Rect::new(0., 0., 4., 4.),
            cherenkov::kurbo::RoundedRectRadii::new(1., 2., 1., 1.),
        ));
        assert!(expressible_shadow(&rrect).is_none());
        assert!(
            expressible_shadow(&ShapeData::Ellipse(cherenkov::kurbo::Ellipse::new(
                (0., 0.),
                (3., 2.),
                0.
            )))
            .is_none()
        );
    }

    #[test]
    fn line_is_a_two_point_path() {
        let path = shape_path(&ShapeData::Line(cherenkov::kurbo::Line::new(
            (1., 2.),
            (3., 4.),
        )));
        assert_eq!(path.elements().len(), 2);
    }
}
