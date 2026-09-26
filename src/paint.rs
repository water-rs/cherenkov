//! Paint: what fills a shape.

use kurbo::{Affine, Point};
use serde::{Deserialize, Serialize};

use crate::color::{Color, ColorSpace, DynColor, WorkingColor};

/// What fills a shape or a glyph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

/// A mesh gradient's fields before its grid is validated: the form it
/// deserializes from, so that captured scenes cannot bypass the invariants.
#[derive(Deserialize)]
struct MeshGradientData {
    columns: u32,
    rows: u32,
    points: Vec<Point>,
    colors: Vec<WorkingColor>,
}

/// Why a mesh gradient's grid is malformed.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MeshGradientError {
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
        } = data;
        if columns == 0 || rows == 0 {
            return Err(MeshGradientError::Empty { columns, rows });
        }
        let vertices = (columns as usize + 1) * (rows as usize + 1);
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
        })
        .unwrap_or_else(|error| panic!("{error}"))
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
    ShaderPaint
);
