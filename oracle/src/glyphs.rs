// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Glyph rendering via `skrifa`: **unhinted** outlines at the exact size.
//!
//! Outlines are expanded into scene [`Item`]s the renderer walks like
//! authored content — `COLRv1` glyphs become nested layers (clips, blend
//! modes, gradient fills), plain glyphs a single fill.
//!
//! Glyph space is font units, y-up; each glyph is placed by
//! `translate(x, y) * scale_non_uniform(size/upem, -size/upem)` so the scene
//! position `x, y` is the glyph's origin on the baseline in y-down scene
//! coordinates.
//!
//! `COLRv1` brush transforms that are not (close to) similarity transforms
//! can't be expressed exactly for radial/sweep gradients — radii are scaled
//! by `sqrt(|det|)` and sweep angles are left unrotated. Font gradients are
//! interpolated in sRGB per the `COLRv1` spec's use of CSS images semantics;
//! [`ColorSpace::Srgb`] is used as the interpolation space.

use cherenkov_scene::{
    BlendMode, Color, ColorSpace, Draw, Extend, FillRule, GlyphRun, GradientStop, Item, Layer,
    LinearGradient, Paint, RadialGradient, Shape, SweepGradient,
};
use kurbo::{Affine, BezPath, Point, Rect, Shape as _};
use read_fonts::types::BoundingBox;
use skrifa::{
    GlyphId, MetadataProvider,
    color::{Brush, ColorPainter, ColorStop},
    instance::{LocationRef, NormalizedCoord as F2Dot14Coord, Size},
    outline::{DrawSettings, OutlinePen},
};

use crate::resources::Resources;

/// Errors from glyph loading or painting.
#[derive(Debug)]
pub enum GlyphError {
    /// The font blob failed to parse.
    Font(String),
    /// A glyph id is out of range for `u16`.
    GlyphId(u32),
    /// The font has no outline for this glyph.
    NoOutline(u16),
    /// A `COLRv1` paint graph failed.
    Paint(String),
    /// A composite mode outside the W3C-16 set was encountered.
    UnsupportedCompositeMode(String),
}

impl std::fmt::Display for GlyphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Font(e) => write!(f, "font parse error: {e}"),
            Self::GlyphId(id) => write!(f, "glyph id {id} does not fit in u16"),
            Self::NoOutline(id) => write!(f, "glyph {id} has no outline"),
            Self::Paint(e) => write!(f, "COLR paint error: {e}"),
            Self::UnsupportedCompositeMode(m) => write!(f, "unsupported COLR composite mode {m}"),
        }
    }
}

impl std::error::Error for GlyphError {}

/// Collects path commands into a [`BezPath`].
struct BezPen(BezPath);

impl OutlinePen for BezPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to((f64::from(x), f64::from(y)));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to((f64::from(x), f64::from(y)));
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.0
            .quad_to((f64::from(cx), f64::from(cy)), (f64::from(x), f64::from(y)));
    }
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.0.curve_to(
            (f64::from(cx0), f64::from(cy0)),
            (f64::from(cx1), f64::from(cy1)),
            (f64::from(x), f64::from(y)),
        );
    }
    fn close(&mut self) {
        self.0.close_path();
    }
}

/// A node of the paint tree a `COLRv1` painter builds, in **font units**.
enum Node {
    /// Fill `shape` (already transform-applied in font space) with `paint`
    /// (gradient coordinates in font space). `shape == None` fills the
    /// enclosing clip region — emitted as a whole-canvas fill, bounded by
    /// whatever clips enclose it.
    Fill {
        shape: Option<BezPath>,
        paint: Paint,
    },
    /// A group: optional clip + blend mode applied to `children`.
    Group {
        clip: Option<BezPath>,
        blend: BlendMode,
        children: Vec<Self>,
    },
}

fn composite_to_blend(mode: skrifa::color::CompositeMode) -> Result<BlendMode, GlyphError> {
    use skrifa::color::CompositeMode as Cm;
    Ok(match mode {
        Cm::Clear => BlendMode::Clear,
        Cm::Src => BlendMode::Src,
        Cm::Dest => BlendMode::Dst,
        Cm::SrcOver => BlendMode::Normal,
        Cm::DestOver => BlendMode::DestOver,
        Cm::SrcIn => BlendMode::SrcIn,
        Cm::DestIn => BlendMode::DestIn,
        Cm::SrcOut => BlendMode::SrcOut,
        Cm::DestOut => BlendMode::DestOut,
        Cm::SrcAtop => BlendMode::SrcAtop,
        Cm::DestAtop => BlendMode::DestAtop,
        Cm::Xor => BlendMode::Xor,
        // COLR `Plus` is the spec's "plus lighter" additive mode.
        Cm::Plus => BlendMode::PlusLighter,
        Cm::Screen => BlendMode::Screen,
        Cm::Overlay => BlendMode::Overlay,
        Cm::Darken => BlendMode::Darken,
        Cm::Lighten => BlendMode::Lighten,
        Cm::ColorDodge => BlendMode::ColorDodge,
        Cm::ColorBurn => BlendMode::ColorBurn,
        Cm::HardLight => BlendMode::HardLight,
        Cm::SoftLight => BlendMode::SoftLight,
        Cm::Difference => BlendMode::Difference,
        Cm::Exclusion => BlendMode::Exclusion,
        Cm::Multiply => BlendMode::Multiply,
        Cm::HslHue => BlendMode::Hue,
        Cm::HslSaturation => BlendMode::Saturation,
        Cm::HslColor => BlendMode::Color,
        Cm::HslLuminosity => BlendMode::Luminosity,
        other => return Err(GlyphError::UnsupportedCompositeMode(format!("{other:?}"))),
    })
}

const fn extend(e: skrifa::color::Extend) -> Extend {
    match e {
        skrifa::color::Extend::Repeat => Extend::Repeat,
        skrifa::color::Extend::Reflect => Extend::Reflect,
        // `Pad` and anything unrecognised pad the edge stops.
        _ => Extend::Pad,
    }
}

/// The `COLRv1` painter: keeps a transform stack (font space), a container
/// stack for clips and composite layers, and emits [`Node`]s.
struct ColrPainter<'a> {
    font: &'a skrifa::FontRef<'a>,
    coords: &'a [F2Dot14Coord],
    palette: Vec<skrifa::color::Color>,
    /// The run's own paint — the COLR "foreground" brush
    /// (`palette_index == 0xFFFF`) for solid brushes.
    foreground: &'a Paint,
    /// `foreground`'s colour, respecting its declared colour space. For a
    /// non-solid run paint — which a gradient stop colour cannot express —
    /// this falls back to opaque sRGB black; solid foreground brushes still
    /// emit the full run paint.
    foreground_color: Color,
    /// Canvas rect in font units — what a `fill` with no glyph clip covers.
    fill_rect_font: BezPath,
    tf: Vec<Affine>,
    containers: Vec<(Option<BezPath>, BlendMode, Vec<Node>)>,
    top: Vec<Node>,
    /// First failure recorded by a `ColorPainter` callback (the trait's
    /// methods cannot return `Result`); checked after `paint()` returns.
    err: Option<GlyphError>,
}

impl ColrPainter<'_> {
    fn cur(&self) -> Affine {
        *self.tf.last().unwrap_or(&Affine::IDENTITY)
    }

    fn palette_color(&self, index: u16, alpha: f32) -> Color {
        if index == 0xFFFF || usize::from(index) >= self.palette.len() {
            // The foreground keeps the run paint's declared colour space.
            let mut c = self.foreground_color;
            c.components[3] *= alpha;
            return c;
        }
        let c = self.palette[usize::from(index)];
        Color {
            space: ColorSpace::Srgb,
            components: [
                f32::from(c.red) / 255.0,
                f32::from(c.green) / 255.0,
                f32::from(c.blue) / 255.0,
                f32::from(c.alpha) / 255.0 * alpha,
            ],
        }
    }

    fn stops(&self, stops: &[ColorStop]) -> Vec<GradientStop> {
        stops
            .iter()
            .map(|s| GradientStop {
                offset: s.offset,
                color: self.palette_color(s.palette_index, s.alpha),
            })
            .collect()
    }

    /// Resolve a COLR brush into a scene [`Paint`]; geometry is in font
    /// units under transform `tf`.
    fn brush_paint(&self, brush: &Brush<'_>, tf: Affine) -> Paint {
        let det = tf.as_coeffs();
        let scale = det[1].mul_add(-det[2], det[0] * det[3]).abs().sqrt();
        match brush {
            Brush::Solid {
                palette_index,
                alpha,
            } => {
                if *palette_index == 0xFFFF {
                    // The foreground brush is the run's own paint — it may
                    // be a gradient or image, not just a solid colour. The
                    // COLR `alpha` applies to it as a paint opacity.
                    let mut paint = self.foreground.clone();
                    paint_opacity(&mut paint, *alpha);
                    paint
                } else {
                    Paint::Solid(self.palette_color(*palette_index, *alpha))
                }
            }
            Brush::LinearGradient {
                p0,
                p1,
                color_stops,
                extend: e,
            } => Paint::Linear(LinearGradient {
                start: tf * Point::new(f64::from(p0.x), f64::from(p0.y)),
                end: tf * Point::new(f64::from(p1.x), f64::from(p1.y)),
                stops: self.stops(color_stops),
                extend: extend(*e),
                interpolation: ColorSpace::Srgb,
            }),
            Brush::RadialGradient {
                c0,
                r0,
                c1,
                r1,
                color_stops,
                extend: e,
            } => Paint::Radial(RadialGradient {
                center0: tf * Point::new(f64::from(c0.x), f64::from(c0.y)),
                r0: f64::from(*r0) * scale,
                center1: tf * Point::new(f64::from(c1.x), f64::from(c1.y)),
                r1: f64::from(*r1) * scale,
                stops: self.stops(color_stops),
                extend: extend(*e),
                interpolation: ColorSpace::Srgb,
            }),
            Brush::SweepGradient {
                c0,
                start_angle,
                end_angle,
                color_stops,
                extend: e,
            } => Paint::Sweep(SweepGradient {
                center: tf * Point::new(f64::from(c0.x), f64::from(c0.y)),
                // skrifa hands degrees, interpreted clockwise in y-up font
                // space; after the y-flip into y-down scene space the same
                // angles read clockwise on screen, which is our convention.
                start_angle: f64::from(*start_angle).to_radians(),
                end_angle: f64::from(*end_angle).to_radians(),
                stops: self.stops(color_stops),
                extend: extend(*e),
                interpolation: ColorSpace::Srgb,
            }),
        }
    }

    fn glyph_path(&self, glyph_id: GlyphId) -> Result<BezPath, GlyphError> {
        let id = u16::try_from(glyph_id.to_u32()).unwrap_or(0);
        let glyph = self
            .font
            .outline_glyphs()
            .get(glyph_id)
            .ok_or(GlyphError::NoOutline(id))?;
        let mut pen = BezPen(BezPath::new());
        glyph
            .draw(
                DrawSettings::unhinted(Size::unscaled(), LocationRef::new(self.coords)),
                &mut pen,
            )
            .map_err(|e| GlyphError::Font(e.to_string()))?;
        Ok(pen.0)
    }
}

impl ColorPainter for ColrPainter<'_> {
    fn push_transform(&mut self, transform: skrifa::color::Transform) {
        let c = transform;
        let affine = Affine::new([
            f64::from(c.xx),
            f64::from(c.yx),
            f64::from(c.xy),
            f64::from(c.yy),
            f64::from(c.dx),
            f64::from(c.dy),
        ]);
        self.tf.push(self.cur() * affine);
    }

    fn pop_transform(&mut self) {
        self.tf.pop();
    }

    fn push_clip_glyph(&mut self, glyph_id: GlyphId) {
        if self.err.is_some() {
            return;
        }
        let path = match self.glyph_path(glyph_id).map(|p| self.cur() * p) {
            Ok(p) => p,
            Err(e) => {
                self.err = Some(e);
                return;
            }
        };
        self.containers
            .push((Some(path), BlendMode::Normal, std::mem::take(&mut self.top)));
    }

    fn push_clip_box(&mut self, clip_box: BoundingBox<f32>) {
        let rect = Rect::new(
            f64::from(clip_box.x_min),
            f64::from(clip_box.y_min),
            f64::from(clip_box.x_max),
            f64::from(clip_box.y_max),
        );
        let path = self.cur() * rect.to_path(1e-9);
        self.containers
            .push((Some(path), BlendMode::Normal, std::mem::take(&mut self.top)));
    }

    fn pop_clip(&mut self) {
        if let Some((clip, blend, mut children)) = self.containers.pop() {
            children.push(Node::Group {
                clip,
                blend,
                children: std::mem::take(&mut self.top),
            });
            self.top = children;
        }
    }

    fn fill(&mut self, brush: Brush<'_>) {
        let paint = self.brush_paint(&brush, self.cur());
        self.top.push(Node::Fill {
            shape: Some(self.cur() * self.fill_rect_font.clone()),
            paint,
        });
    }

    fn fill_glyph(
        &mut self,
        glyph_id: GlyphId,
        brush_transform: Option<skrifa::color::Transform>,
        brush: Brush<'_>,
    ) {
        if self.err.is_some() {
            return;
        }
        let cur = self.cur();
        let shape = match self.glyph_path(glyph_id).map(|p| cur * p) {
            Ok(p) => Some(p),
            Err(e) => {
                self.err = Some(e);
                return;
            }
        };
        let paint = self.brush_paint(
            &brush,
            brush_transform.map_or(cur, |t| {
                cur * Affine::new([
                    f64::from(t.xx),
                    f64::from(t.yx),
                    f64::from(t.xy),
                    f64::from(t.yy),
                    f64::from(t.dx),
                    f64::from(t.dy),
                ])
            }),
        );
        self.top.push(Node::Fill { shape, paint });
    }

    fn push_layer(&mut self, composite_mode: skrifa::color::CompositeMode) {
        if self.err.is_some() {
            return;
        }
        let blend = match composite_to_blend(composite_mode) {
            Ok(b) => b,
            Err(e) => {
                self.err = Some(e);
                return;
            }
        };
        self.containers
            .push((None, blend, std::mem::take(&mut self.top)));
    }

    fn pop_layer_with_mode(&mut self, composite_mode: skrifa::color::CompositeMode) {
        if self.err.is_some() {
            return;
        }
        if let Some((clip, _stored, mut children)) = self.containers.pop() {
            let blend = match composite_to_blend(composite_mode) {
                Ok(b) => b,
                Err(e) => {
                    self.err = Some(e);
                    return;
                }
            };
            children.push(Node::Group {
                clip,
                blend,
                children: std::mem::take(&mut self.top),
            });
            self.top = children;
        }
    }
}

/// Transform a paint's geometry by `t` (for moving from font units into
/// scene space). Image paints compose into `transform`.
fn transform_paint(paint: &mut Paint, t: Affine) {
    match paint {
        Paint::Linear(g) => {
            g.start = t * g.start;
            g.end = t * g.end;
        }
        Paint::Radial(g) => {
            let c = t.as_coeffs();
            let scale = c[1].mul_add(-c[2], c[0] * c[3]).abs().sqrt();
            g.center0 = t * g.center0;
            g.center1 = t * g.center1;
            g.r0 *= scale;
            g.r1 *= scale;
        }
        Paint::Sweep(g) => g.center = t * g.center,
        Paint::Image(i) => i.transform = t * i.transform,
        Paint::Solid(_) => {}
    }
}

/// Multiply a paint's opacity by `alpha`: the colour's alpha for a solid
/// paint, every stop's alpha for a gradient. `Paint::Image` carries no
/// opacity channel in the scene format, so it is left unchanged.
fn paint_opacity(paint: &mut Paint, alpha: f32) {
    let stops = match paint {
        Paint::Linear(g) => Some(&mut g.stops),
        Paint::Radial(g) => Some(&mut g.stops),
        Paint::Sweep(g) => Some(&mut g.stops),
        Paint::Solid(c) => {
            c.components[3] *= alpha;
            None
        }
        Paint::Image(_) => None,
    };
    if let Some(stops) = stops {
        for s in stops {
            s.color.components[3] *= alpha;
        }
    }
}

/// Convert a font-units paint tree into scene [`Item`]s under `place`.
fn node_to_item(node: Node, place: Affine, scene_rect: Rect) -> Item {
    match node {
        Node::Fill { shape, paint } => {
            let mut paint = paint;
            transform_paint(&mut paint, place);
            let shape = shape.map_or_else(
                || Shape::Path {
                    path: scene_rect.to_path(1e-9),
                },
                |p| Shape::Path { path: place * p },
            );
            Item::Draw(Draw::Fill {
                shape,
                rule: FillRule::NonZero,
                paint,
            })
        }
        Node::Group {
            clip,
            blend,
            children,
        } => Item::Layer(Layer {
            transform: Affine::IDENTITY,
            clip: clip.map(|p| Shape::Path { path: place * p }),
            opacity: 1.0,
            blend,
            scroll_offset: kurbo::Vec2::ZERO,
            motion: None,
            items: children
                .into_iter()
                .map(|n| node_to_item(n, place, scene_rect))
                .collect(),
        }),
    }
}

/// Expand a [`GlyphRun`] into scene [`Item`]s.
///
/// Each glyph is drawn at `Size::unscaled()` (font units, **unhinted**) under
/// the run's normalized variation coordinates; the resulting items are
/// placed by the glyph's `(x, y)` origin.
///
/// # Errors
/// `GlyphError` on font parse failures, missing outlines or paint-graph
/// errors; `crate::SceneError` on missing font resources.
pub fn items_for_glyph_run(
    run: &GlyphRun,
    resources: &mut Resources,
    scene_rect: Rect,
) -> Result<Vec<Item>, GlyphError> {
    let data = resources
        .font(run.font)
        .map_err(|e| GlyphError::Font(e.to_string()))?;
    let font = skrifa::FontRef::from_index(data, run.font_index)
        .map_err(|e| GlyphError::Font(e.to_string()))?;
    let metrics = font.metrics(Size::unscaled(), LocationRef::default());
    let upem = f64::from(metrics.units_per_em);
    if upem <= 0.0 {
        return Err(GlyphError::Font("zero units_per_em".into()));
    }
    let s = f64::from(run.size) / upem;

    let coords: Vec<F2Dot14Coord> = run
        .normalized_coords
        .iter()
        .map(|c| F2Dot14Coord::from_f32(c.value))
        .collect();

    let outlines = font.outline_glyphs();
    let color_glyphs = font.color_glyphs();
    let palette: Vec<skrifa::color::Color> = font
        .color_palettes()
        .get(0)
        .map(|p| p.colors().to_vec())
        .unwrap_or_default();
    let foreground_color = match &run.paint {
        Paint::Solid(c) => *c,
        // Non-solid run paints still act as the foreground brush; inside a
        // gradient stop only a colour is expressible — sRGB black.
        _ => Color {
            space: ColorSpace::Srgb,
            components: [0.0, 0.0, 0.0, 1.0],
        },
    };

    let mut items = Vec::new();
    for g in &run.glyphs {
        let gid_u16 = u16::try_from(g.id).map_err(|_| GlyphError::GlyphId(g.id))?;
        let gid = GlyphId::from(gid_u16);
        let place =
            Affine::translate((f64::from(g.x), f64::from(g.y))) * Affine::scale_non_uniform(s, -s);

        if let Some(color_glyph) = color_glyphs.get(gid) {
            // fill_rect_font: canvas rect expressed in font units.
            let inv = place.inverse();
            let corners = [
                inv * Point::new(scene_rect.x0, scene_rect.y0),
                inv * Point::new(scene_rect.x1, scene_rect.y0),
                inv * Point::new(scene_rect.x1, scene_rect.y1),
                inv * Point::new(scene_rect.x0, scene_rect.y1),
            ];
            let mut fill_rect_font = BezPath::new();
            for (i, c) in corners.iter().enumerate() {
                if i == 0 {
                    fill_rect_font.move_to(*c);
                } else {
                    fill_rect_font.line_to(*c);
                }
            }
            fill_rect_font.close_path();

            let mut painter = ColrPainter {
                font: &font,
                coords: &coords,
                palette: palette.clone(),
                foreground: &run.paint,
                foreground_color,
                fill_rect_font,
                tf: vec![Affine::IDENTITY],
                containers: Vec::new(),
                top: Vec::new(),
                err: None,
            };
            color_glyph
                .paint(LocationRef::new(&coords), &mut painter)
                .map_err(|e| GlyphError::Paint(format!("{e}")))?;
            if let Some(e) = painter.err {
                return Err(e);
            }
            let roots = std::mem::take(&mut painter.top);
            items.extend(
                roots
                    .into_iter()
                    .map(|n| node_to_item(n, place, scene_rect)),
            );
        } else {
            let outline = outlines.get(gid).ok_or(GlyphError::NoOutline(gid_u16))?;
            let mut pen = BezPen(BezPath::new());
            outline
                .draw(
                    DrawSettings::unhinted(Size::unscaled(), LocationRef::new(&coords)),
                    &mut pen,
                )
                .map_err(|e| GlyphError::Font(e.to_string()))?;
            if pen.0.elements().is_empty() {
                continue;
            }
            items.push(Item::Draw(Draw::Fill {
                shape: Shape::Path {
                    path: place * pen.0,
                },
                rule: FillRule::NonZero,
                paint: run.paint.clone(),
            }));
        }
    }
    Ok(items)
}
