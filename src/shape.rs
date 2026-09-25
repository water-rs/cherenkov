//! Shapes and their semantic classification.

use std::any::Any;
use std::borrow::Cow;

use kurbo::{Circle, Ellipse, Line, PathEl, Rect, RoundedRect, RoundedRectRadii};
use nami_core::Signal;
use nami_core::watcher::Context;
use serde::{Deserialize, Serialize};

/// Tolerance, in the shape's own units, for converting curves that have no
/// exact Bézier form (such as arcs) into path elements.
pub const PATH_TOLERANCE: f64 = 1e-3;

/// Something the engine can fill, stroke, clip to or cast a shadow from.
///
/// The trait is open: any type can be a shape. Its [`semantic`](Shape::semantic)
/// classification tells the engine whether a fast path applies. Every kurbo
/// shape is a `Shape`, classified through kurbo's own `as_rect`,
/// `as_rounded_rect`, `as_circle` and `as_line`; `kurbo::Ellipse` is
/// recognized by type; anything else is drawn as its path.
pub trait Shape: 'static {
    /// What the engine may draw with a fast path.
    fn semantic(&self) -> Semantic<'_>;
}

/// The engine's fast-path vocabulary: the shapes it draws analytically. Every
/// other shape is a [`Semantic::Path`].
#[derive(Clone, Debug, PartialEq)]
pub enum Semantic<'a> {
    /// An axis-aligned rectangle.
    Rect(Rect),
    /// A rectangle with circular corners.
    RoundedRect(RoundedRect),
    /// A rectangle with continuous (superellipse-blended) corners.
    Continuous(ContinuousRect),
    /// A circle.
    Circle(Circle),
    /// An ellipse, possibly rotated.
    Ellipse(Ellipse),
    /// A line segment.
    Line(Line),
    /// A general path.
    Path(PathRef<'a>),
}

/// A path, borrowed when the shape stores its elements and owned otherwise.
#[derive(Clone, Debug, PartialEq)]
pub struct PathRef<'a> {
    /// The path elements.
    pub elements: Cow<'a, [PathEl]>,
    /// How the interior is decided when the path is filled.
    pub rule: FillRule,
}

/// How the interior of a self-intersecting path is decided.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FillRule {
    /// A point is inside when the winding number is non-zero.
    #[default]
    NonZero,
    /// A point is inside when the winding number is odd.
    EvenOdd,
}

impl<T: kurbo::Shape + 'static> Shape for T {
    fn semantic(&self) -> Semantic<'_> {
        if let Some(ellipse) = (self as &dyn Any).downcast_ref::<Ellipse>() {
            return Semantic::Ellipse(*ellipse);
        }
        if let Some(rect) = self.as_rect() {
            return Semantic::Rect(rect);
        }
        if let Some(rounded) = self.as_rounded_rect() {
            return Semantic::RoundedRect(rounded);
        }
        if let Some(circle) = self.as_circle() {
            return Semantic::Circle(circle);
        }
        if let Some(line) = self.as_line() {
            return Semantic::Line(line);
        }
        let elements = self.as_path_slice().map_or_else(
            || Cow::Owned(self.path_elements(PATH_TOLERANCE).collect()),
            Cow::Borrowed,
        );
        Semantic::Path(PathRef {
            elements,
            rule: FillRule::NonZero,
        })
    }
}

/// A rectangle with continuous corners: each corner blends into the straight
/// edges along a superellipse instead of meeting them at a circular arc's
/// tangent point.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContinuousRect {
    /// The rectangle.
    pub rect: Rect,
    /// The corner radii.
    pub radii: RoundedRectRadii,
    /// How far, as a fraction of each radius, the curvature transition extends
    /// into the straight edges. 0 is a circular corner.
    pub smoothing: f64,
}

impl ContinuousRect {
    /// The smoothing of system continuous corners.
    pub const DEFAULT_SMOOTHING: f64 = 0.6;

    /// Creates a continuous rounded rectangle with the default smoothing.
    #[must_use]
    pub fn new(rect: Rect, radii: impl Into<RoundedRectRadii>) -> Self {
        Self {
            rect,
            radii: radii.into(),
            smoothing: Self::DEFAULT_SMOOTHING,
        }
    }

    /// Returns this shape with a different smoothing.
    #[must_use]
    pub const fn with_smoothing(self, smoothing: f64) -> Self {
        Self { smoothing, ..self }
    }
}

impl Shape for ContinuousRect {
    fn semantic(&self) -> Semantic<'_> {
        Semantic::Continuous(*self)
    }
}

/// A shape filled with the even-odd rule. Only paths can self-intersect, so
/// every other semantic shape is unaffected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvenOdd<S>(pub S);

impl<S: Shape> Shape for EvenOdd<S> {
    fn semantic(&self) -> Semantic<'_> {
        match self.0.semantic() {
            Semantic::Path(path) => Semantic::Path(PathRef {
                rule: FillRule::EvenOdd,
                ..path
            }),
            other => other,
        }
    }
}

/// A shape in the display list: an owned [`Semantic`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ShapeData {
    /// An axis-aligned rectangle.
    Rect(Rect),
    /// A rectangle with circular corners.
    RoundedRect(RoundedRect),
    /// A rectangle with continuous corners.
    Continuous(ContinuousRect),
    /// A circle.
    Circle(Circle),
    /// An ellipse.
    Ellipse(Ellipse),
    /// A line segment.
    Line(Line),
    /// A general path.
    Path {
        /// The path elements.
        elements: Vec<PathEl>,
        /// The fill rule.
        rule: FillRule,
    },
}

impl ShapeData {
    /// Records a shape.
    #[must_use]
    pub fn of(shape: &(impl Shape + ?Sized)) -> Self {
        shape.semantic().into()
    }
}

impl From<Semantic<'_>> for ShapeData {
    fn from(semantic: Semantic<'_>) -> Self {
        match semantic {
            Semantic::Rect(rect) => Self::Rect(rect),
            Semantic::RoundedRect(rounded) => Self::RoundedRect(rounded),
            Semantic::Continuous(continuous) => Self::Continuous(continuous),
            Semantic::Circle(circle) => Self::Circle(circle),
            Semantic::Ellipse(ellipse) => Self::Ellipse(ellipse),
            Semantic::Line(line) => Self::Line(line),
            Semantic::Path(path) => Self::Path {
                elements: path.elements.into_owned(),
                rule: path.rule,
            },
        }
    }
}

nami_core::impl_constant!(ContinuousRect);

impl<S: Clone + 'static> Signal for EvenOdd<S> {
    type Output = Self;
    type Guard = ();

    fn snapshot(&self) -> Self::Output {
        self.clone()
    }

    fn watch(&self, _watcher: impl Fn(Context<Self::Output>) + 'static) -> Self::Guard {}
}

#[cfg(test)]
mod tests {
    use kurbo::{Arc, BezPath, Point, Vec2};

    use super::*;

    #[test]
    fn kurbo_shapes_keep_their_semantics() {
        let rect = Rect::new(0., 0., 10., 20.);
        assert_eq!(rect.semantic(), Semantic::Rect(rect));

        let rounded = RoundedRect::from_rect(rect, 4.);
        assert_eq!(rounded.semantic(), Semantic::RoundedRect(rounded));

        let circle = Circle::new((5., 5.), 3.);
        assert_eq!(circle.semantic(), Semantic::Circle(circle));

        let ellipse = Ellipse::new((5., 5.), (3., 2.), 0.3);
        assert_eq!(ellipse.semantic(), Semantic::Ellipse(ellipse));

        let line = Line::new((0., 0.), (1., 1.));
        assert_eq!(line.semantic(), Semantic::Line(line));

        let continuous = ContinuousRect::new(rect, 6.);
        assert_eq!(continuous.semantic(), Semantic::Continuous(continuous));
    }

    #[test]
    fn stored_paths_are_borrowed_and_other_curves_are_flattened_to_elements() {
        let mut path = BezPath::new();
        path.move_to((0., 0.));
        path.line_to((10., 0.));
        path.line_to((0., 10.));
        path.close_path();
        let Semantic::Path(borrowed) = path.semantic() else {
            panic!("a BezPath is a general path");
        };
        assert!(matches!(borrowed.elements, Cow::Borrowed(_)));

        let arc = Arc::new(Point::ZERO, Vec2::new(4., 4.), 0., 1., 0.);
        let Semantic::Path(owned) = arc.semantic() else {
            panic!("an arc is a general path");
        };
        assert!(matches!(owned.elements, Cow::Owned(_)));
    }

    #[test]
    fn even_odd_applies_only_to_paths() {
        let rect = Rect::new(0., 0., 1., 1.);
        assert_eq!(EvenOdd(rect).semantic(), Semantic::Rect(rect));

        let mut path = BezPath::new();
        path.move_to((0., 0.));
        path.line_to((1., 0.));
        path.close_path();
        let even_odd = EvenOdd(path);
        let Semantic::Path(even_odd) = even_odd.semantic() else {
            panic!("still a path");
        };
        assert_eq!(even_odd.rule, FillRule::EvenOdd);
    }
}
