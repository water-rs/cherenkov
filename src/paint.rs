//! Paint: what fills a shape.

use std::sync::Arc;

use kurbo::{Affine, Point};
use serde::{Deserialize, Serialize};

use crate::color::{Color, ColorSpace, DynColor, WorkingColor};

/// What fills a shape or a glyph.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum Paint {
    /// A single colour.
    Solid(WorkingColor),
    /// A linear gradient.
    Linear(LinearGradient),
    /// A two-point radial gradient.
    Radial(RadialGradient),
    /// A sweep (conic) gradient.
    Sweep(SweepGradient),
    /// A mesh gradient.
    Mesh(MeshGradient),
    /// An image pattern.
    Image(ImagePattern),
    /// A user shader paint. It needs the GPU backend.
    Shader(ShaderPaint),
    /// A paint whose coordinates are mapped into shape space independently
    /// of the shape's geometry. See [`TransformedPaint`].
    Transformed(TransformedPaint),
}

impl Clone for Paint {
    #[expect(
        clippy::inline_always,
        reason = "recording should specialize common solid copies without an out-of-line paint dispatch"
    )]
    #[inline(always)]
    fn clone(&self) -> Self {
        match self {
            Self::Solid(color) => Self::Solid(*color),
            _ => self.clone_resources(),
        }
    }
}

impl Paint {
    // Keep owned gradient/mesh/shader cloning out of each inlined solid
    // recording site. The result and resource ownership remain identical.
    #[inline(never)]
    fn clone_resources(&self) -> Self {
        match self {
            Self::Solid(color) => Self::Solid(*color),
            Self::Linear(gradient) => Self::Linear(gradient.clone()),
            Self::Radial(gradient) => Self::Radial(gradient.clone()),
            Self::Sweep(gradient) => Self::Sweep(gradient.clone()),
            Self::Mesh(mesh) => Self::Mesh(mesh.clone()),
            Self::Image(pattern) => Self::Image(pattern.clone()),
            Self::Shader(shader) => Self::Shader(shader.clone()),
            Self::Transformed(paint) => Self::Transformed(paint.clone()),
        }
    }
}

/// A paint with its own coordinate system.
///
/// `transform` maps the underlying paint's coordinates into shape space.
/// Geometry, stroke width, clipping and coverage are unchanged. Nested
/// transforms compose outside-in: an outer `A` around an inner `B` maps a
/// paint point by `A * B`. For an image pattern this precedes the pattern's
/// texel-to-paint transform. Shader paints transform their sampling coordinates
/// relative to their ordinary, untransformed shape-bounds domain.
///
/// A non-finite or non-invertible transform fails rendering with
/// [`crate::RenderError::Render`], including for solid paints. Reflections are
/// valid. The shared paint keeps stops and mesh data shared when a live signal
/// updates only `transform`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransformedPaint {
    /// Immutable paint, shared across transform updates.
    pub paint: Arc<Paint>,
    /// Paint-to-shape coordinates; identity preserves the original paint.
    pub transform: Affine,
}

impl TransformedPaint {
    /// Shares `paint` and maps its coordinates into shape space.
    #[must_use]
    pub fn new(paint: impl Into<Paint>, transform: Affine) -> Self {
        Self {
            paint: Arc::new(paint.into()),
            transform,
        }
    }
}

impl From<TransformedPaint> for Paint {
    fn from(paint: TransformedPaint) -> Self {
        Self::Transformed(paint)
    }
}

impl Paint {
    /// Maps this paint into shape space without changing geometry or stroke
    /// width. Successive calls compose on the left. See [`TransformedPaint`].
    #[must_use]
    pub fn transformed(self, transform: Affine) -> Self {
        if transform == Affine::IDENTITY {
            return self;
        }
        Self::Transformed(TransformedPaint::new(self, transform))
    }
}

/// One colour stop of a gradient.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColorStop {
    /// Position along the gradient, from 0 to 1.
    pub offset: f32,
    /// The colour at this position.
    pub color: WorkingColor,
}

/// How a gradient or pattern continues outside its defined range.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Extend {
    /// The edge colour continues.
    #[default]
    Pad,
    /// The range repeats.
    Repeat,
    /// The range repeats, mirrored every other time.
    Reflect,
    /// Transparent outside the range.
    None,
}

/// The space in which gradient stops are interpolated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Interpolation {
    /// The linear working space.
    #[default]
    Working,
    /// sRGB-encoded values, as CSS gradients interpolate by default.
    SrgbEncoded,
}

macro_rules! gradient_builders {
    ($ty:ident, $variant:ident) => {
        impl $ty {
            /// Appends a colour stop.
            #[must_use]
            pub fn stop(mut self, offset: f32, color: impl Into<WorkingColor>) -> Self {
                self.stops.push(ColorStop {
                    offset,
                    color: color.into(),
                });
                self
            }

            /// Sets how the gradient continues outside its range.
            #[must_use]
            pub const fn extend(mut self, extend: Extend) -> Self {
                self.extend = extend;
                self
            }

            /// Sets the interpolation space.
            #[must_use]
            pub const fn interpolation(mut self, interpolation: Interpolation) -> Self {
                self.interpolation = interpolation;
                self
            }
        }

        impl From<$ty> for Paint {
            fn from(gradient: $ty) -> Self {
                Self::$variant(gradient)
            }
        }
    };
}

/// A gradient along the line from `start` to `end`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearGradient {
    /// Where offset 0 lies.
    pub start: Point,
    /// Where offset 1 lies.
    pub end: Point,
    /// The colour stops, in ascending offset order.
    pub stops: Vec<ColorStop>,
    /// How the gradient continues outside its range.
    pub extend: Extend,
    /// The interpolation space.
    pub interpolation: Interpolation,
}

impl LinearGradient {
    /// Creates a gradient with no stops.
    #[must_use]
    pub fn new(start: impl Into<Point>, end: impl Into<Point>) -> Self {
        Self {
            start: start.into(),
            end: end.into(),
            stops: Vec::new(),
            extend: Extend::Pad,
            interpolation: Interpolation::Working,
        }
    }
}

/// A gradient between two circles.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RadialGradient {
    /// Centre of the circle at offset 0.
    pub start_center: Point,
    /// Radius of the circle at offset 0.
    pub start_radius: f64,
    /// Centre of the circle at offset 1.
    pub end_center: Point,
    /// Radius of the circle at offset 1.
    pub end_radius: f64,
    /// The colour stops, in ascending offset order.
    pub stops: Vec<ColorStop>,
    /// How the gradient continues outside its range.
    pub extend: Extend,
    /// The interpolation space.
    pub interpolation: Interpolation,
}

impl RadialGradient {
    /// Creates a gradient from the centre outwards to `radius`.
    #[must_use]
    pub fn new(center: impl Into<Point>, radius: f64) -> Self {
        let center = center.into();
        Self::two_point(center, 0., center, radius)
    }

    /// Creates a gradient between two circles.
    #[must_use]
    pub fn two_point(
        start_center: impl Into<Point>,
        start_radius: f64,
        end_center: impl Into<Point>,
        end_radius: f64,
    ) -> Self {
        Self {
            start_center: start_center.into(),
            start_radius,
            end_center: end_center.into(),
            end_radius,
            stops: Vec::new(),
            extend: Extend::Pad,
            interpolation: Interpolation::Working,
        }
    }
}

/// A gradient around a centre, from `start_angle` to `end_angle` (radians).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SweepGradient {
    /// The centre.
    pub center: Point,
    /// The angle of offset 0.
    pub start_angle: f64,
    /// The angle of offset 1.
    pub end_angle: f64,
    /// The colour stops, in ascending offset order.
    pub stops: Vec<ColorStop>,
    /// How the gradient continues outside its range.
    pub extend: Extend,
    /// The interpolation space.
    pub interpolation: Interpolation,
}

impl SweepGradient {
    /// Creates a gradient with no stops.
    #[must_use]
    pub fn new(center: impl Into<Point>, start_angle: f64, end_angle: f64) -> Self {
        Self {
            center: center.into(),
            start_angle,
            end_angle,
            stops: Vec::new(),
            extend: Extend::Pad,
            interpolation: Interpolation::Working,
        }
    }
}

gradient_builders!(LinearGradient, Linear);
gradient_builders!(RadialGradient, Radial);
gradient_builders!(SweepGradient, Sweep);

/// The weights used to interpolate mesh vertex colours in premultiplied
/// linear Display P3. Geometry and patch ownership are unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MeshColorInterpolation {
    /// Bilinear colour weights, matching the existing mesh contract.
    #[default]
    Linear,
    /// Apply `t*t*(3-2*t)` independently to each patch coordinate before
    /// interpolating colours. Endpoint derivatives are zero.
    Smoothstep,
}

impl MeshColorInterpolation {
    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "serde skip_serializing_if requires a borrowed value"
    )]
    const fn is_linear(&self) -> bool {
        matches!(self, Self::Linear)
    }
}

/// A mesh gradient's fields before its grid is validated: the form it
/// deserializes from, so that captured scenes cannot bypass the invariants.
#[derive(Deserialize)]
struct MeshGradientData {
    columns: u32,
    rows: u32,
    points: Vec<Point>,
    colors: Vec<WorkingColor>,
    #[serde(default, skip_serializing_if = "MeshColorInterpolation::is_linear")]
    interpolation: MeshColorInterpolation,
}

/// Why a mesh gradient's grid is malformed.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MeshGradientError {
    /// Grid dimensions overflow addressable storage.
    #[error("mesh grid exceeds addressable storage")]
    GridOverflow,
    /// The grid has no patch.
    #[error("a mesh gradient needs at least one patch, got {columns} x {rows}")]
    Empty {
        /// Patches per row.
        columns: u32,
        /// Patches per column.
        rows: u32,
    },
    /// A per-vertex list does not hold one entry per grid vertex.
    #[error("a mesh gradient needs one {list} per grid vertex: {vertices} vertices, {len} entries")]
    VertexCount {
        /// Which list: points or colours.
        list: &'static str,
        /// Vertices in the grid.
        vertices: usize,
        /// Entries in the list.
        len: usize,
    },
}

impl TryFrom<MeshGradientData> for MeshGradient {
    type Error = MeshGradientError;

    fn try_from(data: MeshGradientData) -> Result<Self, Self::Error> {
        let MeshGradientData {
            columns,
            rows,
            points,
            colors,
            interpolation,
        } = data;
        if columns == 0 || rows == 0 {
            return Err(MeshGradientError::Empty { columns, rows });
        }
        let vertices = usize::try_from(columns)
            .ok()
            .and_then(|n| n.checked_add(1))
            .zip(usize::try_from(rows).ok().and_then(|n| n.checked_add(1)))
            .and_then(|(columns, rows)| columns.checked_mul(rows))
            .ok_or(MeshGradientError::GridOverflow)?;
        for (list, len) in [("point", points.len()), ("colour", colors.len())] {
            if len != vertices {
                return Err(MeshGradientError::VertexCount {
                    list,
                    vertices,
                    len,
                });
            }
        }
        Ok(Self {
            columns,
            rows,
            points,
            colors,
            interpolation,
        })
    }
}

/// A mesh gradient: a grid of `columns` × `rows` patches whose corner points
/// carry colours, interpolated across each patch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "MeshGradientData")]
pub struct MeshGradient {
    columns: u32,
    rows: u32,
    points: Vec<Point>,
    colors: Vec<WorkingColor>,
    #[serde(default, skip_serializing_if = "MeshColorInterpolation::is_linear")]
    interpolation: MeshColorInterpolation,
}

impl MeshGradient {
    /// Creates a mesh gradient. `points` and `colors` list the
    /// `(columns + 1) × (rows + 1)` grid vertices row by row.
    ///
    /// # Panics
    ///
    /// Panics when the grid is empty or when either list does not hold one
    /// entry per vertex.
    #[must_use]
    pub fn new(columns: u32, rows: u32, points: Vec<Point>, colors: Vec<WorkingColor>) -> Self {
        Self::try_from(MeshGradientData {
            columns,
            rows,
            points,
            colors,
            interpolation: MeshColorInterpolation::Linear,
        })
        .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Selects the colour weights without changing the mesh geometry.
    #[must_use]
    pub const fn interpolation(mut self, mode: MeshColorInterpolation) -> Self {
        self.interpolation = mode;
        self
    }

    /// The selected colour interpolation mode.
    #[must_use]
    pub const fn interpolation_mode(&self) -> MeshColorInterpolation {
        self.interpolation
    }

    /// Patches per row.
    #[must_use]
    pub const fn columns(&self) -> u32 {
        self.columns
    }

    /// Patches per column.
    #[must_use]
    pub const fn rows(&self) -> u32 {
        self.rows
    }

    /// Grid vertices, row by row.
    #[must_use]
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    /// Vertex colours, row by row.
    #[must_use]
    pub fn colors(&self) -> &[WorkingColor] {
        &self.colors
    }
}

impl From<MeshGradient> for Paint {
    fn from(mesh: MeshGradient) -> Self {
        Self::Mesh(mesh)
    }
}

/// An image registered with the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImageId(u64);

impl ImageId {
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

/// How an image is sampled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sampling {
    /// Nearest texel.
    Nearest,
    /// Bilinear.
    #[default]
    Linear,
}

/// An image used as a paint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImagePattern {
    /// The image.
    pub image: ImageId,
    /// Maps image texels into the painted shape's space.
    pub transform: Affine,
    /// Horizontal continuation.
    pub extend_x: Extend,
    /// Vertical continuation.
    pub extend_y: Extend,
    /// Sampling.
    pub sampling: Sampling,
}

impl From<ImagePattern> for Paint {
    fn from(pattern: ImagePattern) -> Self {
        Self::Image(pattern)
    }
}

/// A user shader registered with the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShaderId(u64);

impl ShaderId {
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

/// A user shader used as a paint, with its uniform values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShaderPaint {
    /// The shader.
    pub shader: ShaderId,
    /// Uniform values, in the shader's declared order.
    pub uniforms: Vec<f32>,
}

impl From<ShaderPaint> for Paint {
    fn from(shader: ShaderPaint) -> Self {
        Self::Shader(shader)
    }
}

impl From<WorkingColor> for Paint {
    fn from(color: WorkingColor) -> Self {
        Self::Solid(color)
    }
}

impl<CS: ColorSpace> From<Color<CS>> for Paint {
    fn from(color: Color<CS>) -> Self {
        Self::Solid(color.to_working())
    }
}

impl From<DynColor> for Paint {
    fn from(color: DynColor) -> Self {
        Self::Solid(color.to_working())
    }
}

nami_core::impl_constant!(
    Paint,
    LinearGradient,
    RadialGradient,
    SweepGradient,
    MeshGradient,
    ImagePattern,
    ShaderPaint,
    TransformedPaint
);

#[cfg(test)]
mod mesh_overflow_tests {
    #[test]
    fn overflowing_grid_is_rejected_before_vertex_access() {
        let error = super::MeshGradient::try_from(super::MeshGradientData {
            columns: u32::MAX,
            rows: u32::MAX,
            points: vec![],
            colors: vec![],
            interpolation: super::MeshColorInterpolation::Linear,
        })
        .expect_err("unaddressable grid");
        assert_eq!(error, super::MeshGradientError::GridOverflow);
    }
}

#[cfg(test)]
mod mesh_interpolation_tests {
    use super::*;
    #[test]
    fn old_captures_default_to_linear_and_new_modes_round_trip() {
        let original = MeshGradient::new(
            1,
            1,
            vec![
                Point::ZERO,
                Point::new(1., 0.),
                Point::new(0., 1.),
                Point::new(1., 1.),
            ],
            vec![WorkingColor::WHITE; 4],
        );
        let json = serde_json::to_string(&original).unwrap();
        assert!(!json.contains("interpolation"));
        let decoded: MeshGradient = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.interpolation_mode(), MeshColorInterpolation::Linear);
        let smooth = original.interpolation(MeshColorInterpolation::Smoothstep);
        let json = serde_json::to_string(&smooth).unwrap();
        assert!(json.contains("smoothstep"));
        assert_eq!(serde_json::from_str::<MeshGradient>(&json).unwrap(), smooth);
    }
}
