// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Stage-1 lowering: a display list becomes a device-independent op
//! stream, patchable in place along [`cherenkov::Dirty`] ranges.

use std::collections::HashMap;
use std::sync::Arc;

use cherenkov::kurbo::{Affine, Line, PathEl, Rect};
use cherenkov::{
    BlendMode, BlendSpace, Command, Extend, FillRule, ImageId, ImagePattern, Interpolation, Paint,
    RenderError, ShapeData,
};
use cherenkov::{GlyphRun, GlyphStyle};

use super::instance::{
    EXTEND_NONE, EXTEND_PAD, EXTEND_REFLECT, EXTEND_REPEAT, FLAG_HAS_INNER, INTERP_SRGB,
    INTERP_WORKING, KIND_FILL, KIND_STROKE_DIST, KIND_STROKE_OFFSET, PAINT_IMAGE, PAINT_LINEAR,
    PAINT_RADIAL, PAINT_SOLID, PAINT_SWEEP, Shape, Stop,
};
use super::path;
use crate::names;
use crate::render::GpuImage;
use crate::render::glyph::FontData;
use skrifa::MetadataProvider as _;
use skrifa::raw::TableProvider as _;

/// Linear Display P3 to linear sRGB (the inverse of the shader's
/// `SRGB_TO_P3`), used to store `SrgbRenderErrord` gradient stops.
const P3_TO_SRGB: [[f32; 3]; 3] = [
    [1.224_940_1, -0.224_940_4, 0.0],
    [-0.042_056_9, 1.042_057_1, 0.0],
    [-0.019_637_6, -0.078_636_1, 1.098_273_5],
];

/// f64 to f32; instance data is f32 by design.
#[expect(clippy::cast_possible_truncation)]
const fn f32_f64(v: f64) -> f32 {
    v as f32
}

/// A `ShapeData` expressed as a centred rounded box.
pub struct Boxed {
    /// Extra local transform (centre translation, plus rotation for
    /// ellipses and stroked lines).
    pub extra: Affine,
    /// The centred shape.
    pub shape: Shape,
    /// The local bounds, centred.
    pub bounds: Rect,
}

/// Converts a semantic shape into a centred rounded box plus the local
/// transform that centres it.
///
/// A `Line` has no area and draws nothing, so it returns `None`.
pub fn box_shape(shape: &ShapeData) -> Result<Option<Boxed>, RenderError> {
    let boxed = match shape {
        ShapeData::Rect(r) => {
            let half = [f32_f64(r.width() / 2.0), f32_f64(r.height() / 2.0)];
            Boxed {
                extra: Affine::translate(r.center().to_vec2()),
                shape: Shape::rect(half),
                bounds: rect_around_origin(half),
            }
        }
        ShapeData::RoundedRect(rr) => {
            let r = rr.rect();
            let half = [f32_f64(r.width() / 2.0), f32_f64(r.height() / 2.0)];
            let radii = clamped_radii(rr.radii(), half);
            Boxed {
                extra: Affine::translate(r.center().to_vec2()),
                shape: Shape {
                    half,
                    aspect: 1.0,
                    exponent: 2.0,
                    radii,
                },
                bounds: rect_around_origin(half),
            }
        }
        ShapeData::Continuous(c) => {
            let r = c.rect;
            let half = [f32_f64(r.width() / 2.0), f32_f64(r.height() / 2.0)];
            let radii = clamped_radii(c.radii, half);
            Boxed {
                extra: Affine::translate(r.center().to_vec2()),
                shape: Shape {
                    half,
                    aspect: 1.0,
                    exponent: f32_f64(c.smoothing).clamp(0.0, 1.0).mul_add(2.0, 2.0),
                    radii,
                },
                bounds: rect_around_origin(half),
            }
        }
        ShapeData::Circle(c) => {
            let r = c.radius;
            if r <= 0.0 {
                return Ok(None);
            }
            let half = [f32_f64(r), f32_f64(r)];
            Boxed {
                extra: Affine::translate(c.center.to_vec2()),
                shape: Shape {
                    half,
                    aspect: 1.0,
                    exponent: 2.0,
                    radii: [f32_f64(r); 4],
                },
                bounds: rect_around_origin(half),
            }
        }
        ShapeData::Ellipse(e) => {
            let radii_v = e.radii();
            let (a, b) = (radii_v.x, radii_v.y);
            if a <= 0.0 {
                return Ok(None);
            }
            let half = [f32_f64(a), f32_f64(b)];
            Boxed {
                extra: Affine::translate(e.center().to_vec2()) * Affine::rotate(e.rotation()),
                shape: Shape {
                    half,
                    aspect: f32_f64(b / a),
                    exponent: 2.0,
                    radii: [f32_f64(a); 4],
                },
                bounds: rect_around_origin(half),
            }
        }
        ShapeData::Line(_) => return Ok(None),
        ShapeData::Path { .. } => {
            return Err(RenderError::Unsupported(names::PATH));
        }
    };
    Ok(Some(boxed))
}

fn rect_around_origin(half: [f32; 2]) -> Rect {
    Rect::new(
        -f64::from(half[0]),
        -f64::from(half[1]),
        f64::from(half[0]),
        f64::from(half[1]),
    )
}

fn clamped_radii(radii: kurbo::RoundedRectRadii, half: [f32; 2]) -> [f32; 4] {
    let limit = f64::from(half[0].min(half[1]));
    [
        f32_f64(radii.top_left.clamp(0.0, limit)),
        f32_f64(radii.top_right.clamp(0.0, limit)),
        f32_f64(radii.bottom_right.clamp(0.0, limit)),
        f32_f64(radii.bottom_left.clamp(0.0, limit)),
    ]
}

/// The blur sigma the shader integrates against, modelling the oracle's
/// pixel-area sampling: `sqrt(sigma² + 1/12)` for a positive sigma.
fn shadow_sigma(sigma: f64) -> f64 {
    if sigma > 0.0 {
        sigma.mul_add(sigma, 1.0 / 12.0).sqrt()
    } else {
        sigma
    }
}

/// Exact `2.0`/`1.0` comparisons: these fields only ever hold the constants
/// [`box_shape`] assigns.
#[expect(clippy::float_cmp)]
fn shape_is_offsettable(shape: Shape) -> bool {
    shape.exponent == 2.0 && shape.aspect == 1.0
}

/// The path cache tag for a fill rule.
const fn fill_tag(rule: FillRule) -> u64 {
    match rule {
        FillRule::NonZero => 0,
        FillRule::EvenOdd => 1,
    }
}

/// The paint data shared by every instance kind.
#[derive(Clone, Default)]
pub struct PaintData {
    /// `PAINT_*`.
    pub kind: u32,
    /// Solid colour.
    pub color: [f32; 4],
    /// Gradient/image coefficients (see `Instance::grad`).
    pub grad: [f32; 4],
    /// Gradient/image coefficients (see `Instance::grad2`).
    pub grad2: [f32; 4],
    /// First stop, relative to [`ResolvedPaint::stops`].
    pub first_stop: u32,
    /// `count | interp << 16 | extend << 20`; for `PAINT_IMAGE`,
    /// `extend_x | extend_y << 4 | sampling << 8`.
    pub packed: u32,
    /// The bound image for `PAINT_IMAGE`.
    pub image: Option<u64>,
}

/// A paint resolved at lowering: device-independent shader data plus its
/// gradient stops.
#[derive(Clone, Default)]
pub struct ResolvedPaint {
    /// The instance paint fields.
    pub data: PaintData,
    /// The gradient stops this paint pushes (`data.first_stop` indexes it).
    pub stops: Vec<Stop>,
}

/// sRGB-encodes one channel, preserving sign.
fn srgb_encode(x: f32) -> f32 {
    let e = if x.abs() <= 0.003_130_8 {
        x.abs() * 12.92
    } else {
        1.055f32.mul_add(x.abs().powf(1.0 / 2.4), -0.055)
    };
    e.copysign(x)
}

fn push_stops(
    stops: &mut Vec<Stop>,
    gradient_stops: &[cherenkov::ColorStop],
    interpolation: Interpolation,
    extend: Extend,
) -> (u32, u32) {
    let first = u32::try_from(stops.len()).unwrap_or(u32::MAX);
    let mut sorted = gradient_stops.to_vec();
    sorted.sort_by(|a, b| a.offset.total_cmp(&b.offset));
    let count = u32::try_from(sorted.len().min(0xffff)).unwrap_or(0xffff);
    for stop in sorted.iter().take(count as usize) {
        let [r, g, b, a] = stop.color.components;
        let color = if interpolation == Interpolation::SrgbRenderErrord {
            let [sr, sg, sb] = [
                P3_TO_SRGB[0][0].mul_add(r, P3_TO_SRGB[0][1].mul_add(g, P3_TO_SRGB[0][2] * b)),
                P3_TO_SRGB[1][0].mul_add(r, P3_TO_SRGB[1][1].mul_add(g, P3_TO_SRGB[1][2] * b)),
                P3_TO_SRGB[2][0].mul_add(r, P3_TO_SRGB[2][1].mul_add(g, P3_TO_SRGB[2][2] * b)),
            ];
            [srgb_encode(sr), srgb_encode(sg), srgb_encode(sb), a]
        } else {
            [r, g, b, a]
        };
        stops.push(Stop {
            color,
            offset: stop.offset,
            pad: [0.0; 3],
        });
    }
    let interp = match interpolation {
        Interpolation::Working => INTERP_WORKING,
        Interpolation::SrgbRenderErrord => INTERP_SRGB,
    };
    (first, count | (interp << 16) | (extend_code(extend) << 20))
}

const fn extend_code(extend: Extend) -> u32 {
    match extend {
        Extend::Pad => EXTEND_PAD,
        Extend::Repeat => EXTEND_REPEAT,
        Extend::Reflect => EXTEND_REFLECT,
        Extend::None => EXTEND_NONE,
    }
}

/// Lowers a paint; `to_local` maps content space to the instance's local
/// (shape-centred) space in which the shader evaluates gradient parameters.
#[expect(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "image dimensions fit f32; affine coefficients are conventionally a..f"
)]
fn paint_data(
    paint: &Paint,
    to_local: Affine,
    stops: &mut Vec<Stop>,
    images: &HashMap<u64, GpuImage>,
) -> Result<PaintData, RenderError> {
    let mut data = PaintData {
        kind: PAINT_SOLID,
        ..PaintData::default()
    };
    match paint {
        Paint::Solid(c) => data.color = c.components,
        Paint::Linear(g) => {
            data.kind = PAINT_LINEAR;
            let start = to_local * g.start;
            let end = to_local * g.end;
            data.grad = [
                f32_f64(start.x),
                f32_f64(start.y),
                f32_f64(end.x),
                f32_f64(end.y),
            ];
            let (first, packed) = push_stops(stops, &g.stops, g.interpolation, g.extend);
            data.first_stop = first;
            data.packed = packed;
        }
        Paint::Radial(g) => {
            data.kind = PAINT_RADIAL;
            let c0 = to_local * g.start_center;
            let c1 = to_local * g.end_center;
            data.grad = [f32_f64(c0.x), f32_f64(c0.y), f32_f64(c1.x), f32_f64(c1.y)];
            data.grad2 = [f32_f64(g.start_radius), f32_f64(g.end_radius), 0.0, 0.0];
            let (first, packed) = push_stops(stops, &g.stops, g.interpolation, g.extend);
            data.first_stop = first;
            data.packed = packed;
        }
        Paint::Sweep(g) => {
            data.kind = PAINT_SWEEP;
            let center = to_local * g.center;
            data.grad = [f32_f64(center.x), f32_f64(center.y), 0.0, 0.0];
            data.grad2 = [f32_f64(g.start_angle), f32_f64(g.end_angle), 0.0, 0.0];
            let (first, packed) = push_stops(stops, &g.stops, g.interpolation, g.extend);
            data.first_stop = first;
            data.packed = packed;
        }
        Paint::Mesh(_) => return Err(RenderError::Unsupported(names::MESH)),
        Paint::Image(pattern) => {
            let img = images.get(&pattern.image.raw()).ok_or_else(|| {
                RenderError::Image(format!("unregistered image {}", pattern.image.raw()))
            })?;
            // `to_local * transform` maps image space to instance-local; the
            // shader needs the inverse.
            let [a, b, c, d, e, f] = (to_local * pattern.transform).inverse().as_coeffs();
            data.kind = PAINT_IMAGE;
            data.grad = [f32_f64(a), f32_f64(b), f32_f64(c), f32_f64(d)];
            let (iw, ih) = (img.width as f32, img.height as f32);
            data.grad2 = [f32_f64(e), f32_f64(f), iw, ih];
            let sampling = match pattern.sampling {
                cherenkov::Sampling::Nearest => 0,
                cherenkov::Sampling::Linear => 1,
            };
            data.packed = extend_code(pattern.extend_x)
                | (extend_code(pattern.extend_y) << 4)
                | (sampling << 8);
            data.image = Some(pattern.image.raw());
        }
        Paint::Shader(_) => return Err(RenderError::Unsupported(names::SHADER)),
    }
    Ok(data)
}

/// Resolves `paint` device-independently; `to_local` is the
/// instance-local transform the shader paints in (the boxed `extra`
/// inverse, or identity for device-space replay).
fn resolve(
    paint: &Paint,
    to_local: Affine,
    images: &HashMap<u64, GpuImage>,
) -> Result<ResolvedPaint, RenderError> {
    let mut stops = Vec::new();
    let data = paint_data(paint, to_local, &mut stops, images)?;
    Ok(ResolvedPaint { data, stops })
}

/// A clip shape, resolved at lowering.
pub enum ClipShape {
    /// A shape with no area (`Line`): the scope draws nothing.
    Empty,
    /// A centred rounded box clip.
    Boxed {
        /// The boxed shape's local transform.
        extra: Affine,
        /// The centred clip shape.
        shape: Shape,
        /// The scene `Rect`, when the source shape was one (the compose
        /// stage computes the device-space candidate).
        rect: Option<Rect>,
    },
    /// A path clip rasterized into a coverage mask at compose.
    Path {
        /// The path elements.
        elements: Arc<[PathEl]>,
        /// The fill rule.
        rule: FillRule,
    },
}

/// The outline a `Path` op rasterizes on a cache miss.
pub enum Outline {
    /// Fill `elements` under `rule`.
    Fill(Arc<[PathEl]>),
    /// Stroke `shape` with `stroke`; the device-dependent tolerance is
    /// mixed into the content hash at compose.
    Stroke {
        /// The stroked shape.
        shape: ShapeData,
        /// The stroke style.
        stroke: kurbo::Stroke,
    },
}

/// A stage-1 draw or scope op: device-independent.
pub enum Op {
    /// A shaped SDF instance (fill/stroke box paths, stroked lines, image
    /// draws).
    Shaped {
        /// `KIND_*`.
        kind: u32,
        /// Content→instance-local: the accumulated ambient transform
        /// (BeginTransform/Picture/COLR placements) times `boxed.extra`.
        local: Affine,
        /// The ambient transform without `boxed.extra` (for the AA
        /// margin at compose).
        ambient: Affine,
        /// The centred shape.
        shape: Shape,
        /// The inner shape of an offset stroke.
        inner: Option<Shape>,
        /// Uninflated local bounds.
        bounds: Rect,
        /// The margin that is not antialiasing (half width for strokes).
        extra_margin: f64,
        /// The resolved paint.
        paint: ResolvedPaint,
        /// `params[0]` (stroke half width).
        param_x: f32,
        /// `meta[3]` flag bits (e.g. `FLAG_HAS_INNER`).
        flags: u32,
    },
    /// A Gaussian-blurred box.
    Shadow {
        /// Content→instance-local (ambient × offset × `boxed.extra`).
        local: Affine,
        /// The ambient transform (AA margin).
        ambient: Affine,
        /// The centred shape.
        shape: Shape,
        /// Uninflated local bounds.
        bounds: Rect,
        /// `sqrt(sigma² + 1/12)`.
        sigma_eff: f64,
        /// The shadow colour.
        color: [f32; 4],
    },
    /// A path fill or stroked outline; the rasterizer runs at compose on
    /// an atlas miss.
    Path {
        /// The ambient transform.
        local: Affine,
        /// The content hash (stroke hashes exclude the tolerance).
        content: u64,
        /// The fill rule.
        rule: FillRule,
        /// The outline builder.
        outline: Outline,
        /// The resolved paint (identity local space — device pixels).
        paint: ResolvedPaint,
    },
    /// A glyph run with the COLR glyphs already expanded into their
    /// picture ops at their placements.
    Glyphs {
        /// The ambient transform.
        local: Affine,
        /// The run, minus its COLR glyphs.
        run: GlyphRun,
        /// The resolved paint (identity local space).
        paint: ResolvedPaint,
    },
    /// Open a clip scope.
    BeginClip {
        /// The ambient transform (the clip's own `extra` is inside
        /// [`ClipShape`]).
        local: Affine,
        /// The clip shape.
        shape: ClipShape,
        /// Op index of the matching `End` op.
        end: u32,
    },
    /// Open an isolation scope (`BeginGroup` with opacity < 1 or a
    /// non-normal blend).
    BeginIsolate {
        /// The group opacity.
        opacity: f32,
        /// The group blend mode.
        blend: BlendMode,
        /// Op index of the matching `End` op.
        end: u32,
    },
    /// Close the innermost clip or isolate scope.
    End,
}

impl cherenkov::lowering::Operation for Op {
    fn end_mut(&mut self) -> Option<&mut u32> {
        match self {
            Self::BeginClip { end, .. } | Self::BeginIsolate { end, .. } => Some(end),
            _ => None,
        }
    }
    fn same_structure(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::BeginClip { shape: a, .. }, Self::BeginClip { shape: b, .. })
                if std::mem::discriminant(a) != std::mem::discriminant(b) =>
            {
                return false;
            }
            (Self::BeginIsolate { blend: a, .. }, Self::BeginIsolate { blend: b, .. })
                if a != b =>
            {
                return false;
            }
            _ => {}
        }
        std::mem::discriminant(self) == std::mem::discriminant(other)
            && !matches!((self, other), (Self::Glyphs { run: a, .. }, Self::Glyphs { run: b, .. }) if a.glyphs.len() != b.glyphs.len())
    }
}

/// Resources needed while resolving content-space operations.
pub struct Lowerer<'a> {
    pub pending: &'a mut Vec<super::glyph::PendingRaster>,
    /// Fonts for colour glyph expansion.
    pub fonts: &'a HashMap<u64, FontData>,
    /// Images for resolving image paint dimensions.
    pub images: &'a HashMap<u64, GpuImage>,
}

impl cherenkov::lowering::Compiler for Lowerer<'_> {
    type Op = Op;
    type Error = RenderError;
    fn draw(
        &mut self,
        command: &Command,
        ambient: Affine,
        ops: &mut Vec<Op>,
    ) -> Result<(), RenderError> {
        match command {
            Command::Fill { shape, paint } => self.fill(ambient, shape, paint, ops),
            Command::Stroke {
                shape,
                stroke,
                paint,
            } => self.stroke(ambient, shape, stroke, paint, ops),
            Command::Shadow { shape, shadow } => Self::shadow(ambient, shape, shadow, ops),
            Command::Glyphs { run, paint } => self.glyph_run(ambient, run, paint, ops),
            Command::Image {
                image,
                dst,
                sampling,
            } => self.image_draw(ambient, *image, dst, *sampling, ops),
            _ => unreachable!("shared walker handles scopes and pictures"),
        }
    }
    fn clip(&mut self, shape: &ShapeData, ambient: Affine) -> Result<Op, RenderError> {
        Ok(Op::BeginClip {
            local: ambient,
            shape: clip_shape(shape)?,
            end: 0,
        })
    }
    fn group(&mut self, group: &cherenkov::Group) -> Result<Option<Op>, RenderError> {
        if group.filter.is_some() {
            return Err(RenderError::Unsupported(names::FILTER).into());
        }
        if group.blend_space != BlendSpace::Linear {
            return Err(RenderError::Unsupported(names::BLEND_SPACE).into());
        }
        Ok(
            (group.opacity < 1.0 || group.blend != BlendMode::Normal).then_some(Op::BeginIsolate {
                opacity: group.opacity,
                blend: group.blend,
                end: 0,
            }),
        )
    }
    fn end(&mut self) -> Op {
        Op::End
    }
}

impl Lowerer<'_> {
    /// `Fill`: a shaped quad; a path goes through the coverage
    /// rasterizer.
    fn fill(
        &self,
        ambient: Affine,
        shape: &ShapeData,
        paint: &Paint,
        ops: &mut Vec<Op>,
    ) -> Result<(), RenderError> {
        if let ShapeData::Path { elements, rule } = shape {
            ops.push(Op::Path {
                local: ambient,
                content: path::hash_elements(elements, fill_tag(*rule)),
                rule: *rule,
                outline: Outline::Fill(Arc::from(elements.as_slice())),
                paint: resolve(paint, Affine::IDENTITY, self.images)?,
            });
            return Ok(());
        }
        let Some(boxed) = box_shape(shape)? else {
            return Ok(());
        };
        ops.push(Op::Shaped {
            kind: KIND_FILL,
            local: ambient * boxed.extra,
            ambient,
            shape: boxed.shape,
            inner: None,
            bounds: boxed.bounds,
            extra_margin: 0.0,
            paint: resolve(paint, boxed.extra.inverse(), self.images)?,
            param_x: 0.0,
            flags: 0,
        });
        Ok(())
    }

    /// `Image`: a fill of `dst` whose paint maps the rect onto the whole
    /// image, pad-extended.
    fn image_draw(
        &self,
        ambient: Affine,
        image: ImageId,
        dst: &Rect,
        sampling: cherenkov::Sampling,
        ops: &mut Vec<Op>,
    ) -> Result<(), RenderError> {
        let img = self
            .images
            .get(&image.raw())
            .ok_or_else(|| RenderError::Image(format!("unregistered image {}", image.raw())))?;
        let (iw, ih) = (f64::from(img.width), f64::from(img.height));
        let (dw, dh) = (dst.x1 - dst.x0, dst.y1 - dst.y0);
        if dw <= 0.0 || dh <= 0.0 {
            return Ok(());
        }
        let transform =
            Affine::translate((dst.x0, dst.y0)) * Affine::scale_non_uniform(dw / iw, dh / ih);
        let paint = Paint::Image(ImagePattern {
            image,
            transform,
            extend_x: Extend::Pad,
            extend_y: Extend::Pad,
            sampling,
        });
        self.fill(ambient, &ShapeData::Rect(*dst), &paint, ops)
    }

    /// `Stroke`: offset strokes for circular-corner boxes, distance
    /// strokes for continuous corners and ellipses, a box fast path for
    /// lines; paths and dashed strokes become `Outline::Stroke`.
    fn stroke(
        &self,
        ambient: Affine,
        shape: &ShapeData,
        stroke: &kurbo::Stroke,
        paint: &Paint,
        ops: &mut Vec<Op>,
    ) -> Result<(), RenderError> {
        if matches!(shape, ShapeData::Path { .. }) || !stroke.dash_pattern.is_empty() {
            if matches!(shape, ShapeData::Continuous(_)) {
                return Err(RenderError::Unsupported(names::PATH));
            }
            ops.push(Op::Path {
                local: ambient,
                content: path::hash_stroke(shape, stroke, 0.0),
                rule: FillRule::NonZero,
                outline: Outline::Stroke {
                    shape: shape.clone(),
                    stroke: stroke.clone(),
                },
                paint: resolve(paint, Affine::IDENTITY, self.images)?,
            });
            return Ok(());
        }
        let hw = stroke.width / 2.0;
        if let ShapeData::Line(line) = shape {
            return self.stroke_line(ambient, line, hw, stroke, paint, ops);
        }
        let Some(boxed) = box_shape(shape)? else {
            return Ok(());
        };
        if shape_is_offsettable(boxed.shape) {
            // Offset stroke: outer minus inner.
            let mut outer = boxed.shape;
            for h in &mut outer.half {
                *h += f32_f64(hw);
            }
            for r in &mut outer.radii {
                if *r > 0.0 {
                    *r += f32_f64(hw);
                } else {
                    *r = match stroke.join {
                        kurbo::Join::Round => f32_f64(hw),
                        kurbo::Join::Miter => {
                            if stroke.miter_limit < 1.415 {
                                return Err(RenderError::Unsupported(names::STROKE_JOIN));
                            }
                            0.0
                        }
                        kurbo::Join::Bevel => {
                            return Err(RenderError::Unsupported(names::STROKE_JOIN));
                        }
                    };
                }
            }
            let mut inner = boxed.shape;
            let has_inner = inner.half.iter().all(|h| *h > f32_f64(hw));
            let inner_opt = if has_inner {
                for h in &mut inner.half {
                    *h -= f32_f64(hw);
                }
                for r in &mut inner.radii {
                    *r = (*r - f32_f64(hw)).max(0.0);
                }
                Some(inner)
            } else {
                None
            };
            let flags = if has_inner { FLAG_HAS_INNER } else { 0 };
            ops.push(Op::Shaped {
                kind: KIND_STROKE_OFFSET,
                local: ambient * boxed.extra,
                ambient,
                shape: outer,
                inner: inner_opt,
                bounds: boxed.bounds,
                extra_margin: hw,
                paint: resolve(paint, boxed.extra.inverse(), self.images)?,
                param_x: f32_f64(hw),
                flags,
            });
        } else {
            ops.push(Op::Shaped {
                kind: KIND_STROKE_DIST,
                local: ambient * boxed.extra,
                ambient,
                shape: boxed.shape,
                inner: None,
                bounds: boxed.bounds,
                extra_margin: hw,
                paint: resolve(paint, boxed.extra.inverse(), self.images)?,
                param_x: f32_f64(hw),
                flags: 0,
            });
        }
        Ok(())
    }

    /// A stroked line as a box in the line's local frame.
    fn stroke_line(
        &self,
        ambient: Affine,
        line: &Line,
        hw: f64,
        stroke: &kurbo::Stroke,
        paint: &Paint,
        ops: &mut Vec<Op>,
    ) -> Result<(), RenderError> {
        if stroke.start_cap != stroke.end_cap {
            return Err(RenderError::Unsupported(names::STROKE_JOIN));
        }
        let d = line.p1 - line.p0;
        let len = d.hypot();
        if len <= 0.0 || hw <= 0.0 {
            return Ok(());
        }
        let mid = line.p0 + d * 0.5;
        let extra = Affine::translate(mid.to_vec2()) * Affine::rotate(d.y.atan2(d.x));
        let half = match stroke.start_cap {
            kurbo::Cap::Butt => [f32_f64(len / 2.0), f32_f64(hw)],
            _ => [f32_f64(len / 2.0 + hw), f32_f64(hw)],
        };
        let radii = if stroke.start_cap == kurbo::Cap::Round {
            [f32_f64(hw); 4]
        } else {
            [0.0; 4]
        };
        let boxed = Boxed {
            extra,
            shape: Shape {
                half,
                aspect: 1.0,
                exponent: 2.0,
                radii,
            },
            bounds: rect_around_origin(half),
        };
        ops.push(Op::Shaped {
            kind: KIND_FILL,
            local: ambient * boxed.extra,
            ambient,
            shape: boxed.shape,
            inner: None,
            bounds: boxed.bounds,
            extra_margin: 0.0,
            paint: resolve(paint, boxed.extra.inverse(), self.images)?,
            param_x: 0.0,
            flags: 0,
        });
        Ok(())
    }

    /// `Shadow`: a Gaussian-blurred rounded box, offset and spread.
    fn shadow(
        ambient: Affine,
        shape: &ShapeData,
        shadow: &cherenkov::Shadow,
        ops: &mut Vec<Op>,
    ) -> Result<(), RenderError> {
        let Some(boxed) = box_shape(shape)? else {
            return Ok(());
        };
        if !shape_is_offsettable(boxed.shape) {
            return Err(RenderError::Unsupported(names::SHADOW));
        }
        let mut s = boxed.shape;
        let spread = f32_f64(shadow.spread);
        for h in &mut s.half {
            *h += spread;
        }
        if s.half.iter().any(|h| *h <= 0.0) {
            return Ok(());
        }
        for r in &mut s.radii {
            if *r > 0.0 {
                *r = (*r + spread).max(0.0);
            }
        }
        ops.push(Op::Shadow {
            local: ambient * Affine::translate(shadow.offset) * boxed.extra,
            ambient,
            shape: s,
            bounds: rect_around_origin(s.half),
            sigma_eff: shadow_sigma(shadow.sigma),
            color: shadow.color.components,
        });
        Ok(())
    }

    /// `Glyphs`: plain glyphs collect into `Glyphs` ops; COLR glyphs
    /// expand their picture at the glyph's placement, in order.
    fn glyph_run(
        &mut self,
        ambient: Affine,
        run: &GlyphRun,
        paint: &Paint,
        ops: &mut Vec<Op>,
    ) -> Result<(), RenderError> {
        if matches!(run.style, GlyphStyle::Stroke(_)) {
            return Err(RenderError::Unsupported(names::GLYPH_STROKE));
        }
        let font = self
            .fonts
            .get(&run.font.raw())
            .ok_or_else(|| RenderError::Font(format!("unregistered font {:?}", run.font)))?;
        let resolved = resolve(paint, Affine::IDENTITY, self.images)?;
        let mut pending: Vec<cherenkov::Glyph> = Vec::new();
        let mut colr_ctx: Option<(skrifa::FontRef<'_>, f64)> = None;
        let mut colr_checked = false;
        for glyph in &run.glyphs {
            if glyph.transform.is_some() {
                return Err(RenderError::Unsupported(names::GLYPH_TRANSFORM));
            }
            if !colr_checked {
                colr_checked = true;
                let font_ref = skrifa::FontRef::from_index(&font.data, font.index)
                    .map_err(|e| RenderError::Font(format!("{e}")))?;
                if font_ref.colr().is_ok() {
                    let upem = font_ref
                        .head()
                        .map_err(|e| RenderError::Font(format!("head: {e}")))?
                        .units_per_em();
                    colr_ctx = Some((font_ref, f64::from(upem)));
                }
            }
            if let Some((font_ref, upem)) = colr_ctx.as_ref()
                && font_ref
                    .color_glyphs()
                    .get(skrifa::GlyphId::new(glyph.id))
                    .is_some()
            {
                if *upem <= 0.0 {
                    return Err(RenderError::Font("zero units_per_em".into()));
                }
                if !pending.is_empty() {
                    ops.push(Op::Glyphs {
                        local: ambient,
                        run: GlyphRun {
                            font: run.font,
                            size: run.size,
                            coords: run.coords.clone(),
                            glyphs: std::mem::take(&mut pending),
                            style: run.style.clone(),
                        },
                        paint: resolved.clone(),
                    });
                }
                let picture = crate::render::colr::glyph_picture(
                    font,
                    run.font.raw(),
                    glyph.id,
                    &run.coords,
                    paint,
                    self.pending,
                )?;
                // `translate(x, y) * scale_non_uniform(size/upem,
                // -size/upem)` places the font-space picture at the
                // glyph's origin.
                let s = f64::from(run.size) / upem;
                let place = Affine::translate((f64::from(glyph.x), f64::from(glyph.y)))
                    * Affine::scale_non_uniform(s, -s);
                let mut lowerer = Lowerer {
                    fonts: self.fonts,
                    images: self.images,
                    pending: self.pending,
                };
                cherenkov::lowering::append(
                    picture.display_list(),
                    ambient * place,
                    &mut lowerer,
                    ops,
                )?;
                continue;
            }
            pending.push(*glyph);
        }
        if !pending.is_empty() {
            ops.push(Op::Glyphs {
                local: ambient,
                run: GlyphRun {
                    font: run.font,
                    size: run.size,
                    coords: run.coords.clone(),
                    glyphs: pending,
                    style: run.style.clone(),
                },
                paint: resolved,
            });
        }
        Ok(())
    }
}

/// The clip shape a `BeginClip` lowers to.
fn clip_shape(shape: &ShapeData) -> Result<ClipShape, RenderError> {
    if let ShapeData::Path { elements, rule } = shape {
        return Ok(ClipShape::Path {
            elements: Arc::from(elements.as_slice()),
            rule: *rule,
        });
    }
    let Some(boxed) = box_shape(shape)? else {
        return Ok(ClipShape::Empty);
    };
    Ok(ClipShape::Boxed {
        extra: boxed.extra,
        shape: boxed.shape,
        rect: match shape {
            ShapeData::Rect(r) => Some(*r),
            _ => None,
        },
    })
}
