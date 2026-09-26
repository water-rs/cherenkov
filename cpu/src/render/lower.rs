// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Lowering: a surface's layer tree and display lists become one flat list
//! of rasterization [`Item`]s in device space.

use std::collections::HashMap;
use std::sync::Arc;

use cherenkov::kurbo::{Affine, BezPath, PathEl, Point, Rect};
use cherenkov::{
    BlendMode, BlendSpace, Command, ContinuousRect, DisplayList, FillRule, GlyphRun, GlyphStyle,
    Paint, ShapeData, WorkingColor,
};

use skrifa::MetadataProvider as _;
use skrifa::raw::TableProvider as _;

use crate::error::{RenderError, Unsupported};
use crate::render::coverage::{Coverage, CoverageCache, Operand};
use crate::render::paint::{PaintData, paint_data};
use crate::render::raster::Edge;

/// Curve-to-path and stroke tolerance in device pixels.
pub const FLATTEN_TOL: f64 = 0.02;

/// An integer device-space rectangle (x ∈ `[x0, x1)`, y ∈ `[y0, y1)`).
#[derive(Clone, Copy, Debug)]
pub struct IRect {
    /// Left edge.
    pub x0: i32,
    /// Top edge.
    pub y0: i32,
    /// Right edge.
    pub x1: i32,
    /// Bottom edge.
    pub y1: i32,
}

/// The geometric operands of the active clip intersection.
#[derive(Debug)]
pub struct ClipGeometry {
    /// Every clip retains its own fill rule until the intersection resolves.
    pub operands: Vec<Operand>,
}

/// A clip shared by items; cheap to clone.
pub type ClipRef = Arc<ClipGeometry>;

/// One rasterization item of a lowered frame.
#[derive(Debug)]
pub enum Item {
    /// A filled, flattened polygon.
    Draw {
        /// Prepared exact coverage, including every clip in force.
        coverage: Arc<Coverage>,
        /// Number of source edges, for frame statistics.
        edge_count: usize,
        /// The paint evaluator.
        paint: PaintData,
    },
    /// A rasterized glyph mask instance.
    Glyph {
        /// The mask slot, filled before the band pass.
        slot: crate::render::glyph::GlyphSlot,
        /// The glyph's integer device origin (x).
        x: i32,
        /// The glyph's integer device origin (y).
        y: i32,
        /// The paint evaluator.
        paint: PaintData,
        /// The clip in force.
        clip: Option<ClipRef>,
    },
    /// Start a fresh transparent scratch layer.
    PushIsolate,
    /// Composite the scratch layer onto what lies below it.
    PopIsolate {
        /// The opacity multiplier.
        opacity: f32,
        /// The blend mode to composite with.
        blend: BlendMode,
        /// Space used for the group composite.
        space: BlendSpace,
    },
}

/// A layer node on the render thread's side, handed to the lowering.
pub struct LayerNode {
    /// Local transform.
    pub transform: Affine,
    /// Opacity; below 1.0 isolates.
    pub opacity: f32,
    /// Blend mode onto the parent; non-normal isolates.
    pub blend: BlendMode,
    /// Space used to composite this layer.
    pub blend_space: BlendSpace,
    /// Clip shape.
    pub clip: Option<ShapeData>,
    /// The content.
    pub content: Option<ContentData>,
    /// Child layers, in order.
    pub children: Vec<u64>,
}

/// A layer's content.
pub enum ContentData {
    /// A shared picture.
    Picture(cherenkov::Picture),
}

/// A glyph mask request lowering emits: everything needed to rasterize
/// the mask in parallel, plus the slot the [`Item::Glyph`] reads.
pub struct GlyphReq {
    /// The cache key.
    pub key: crate::render::glyph::GlyphKey,
    /// The engine font id.
    pub font: u64,
    /// The glyph index.
    pub glyph_id: u32,
    /// The run's size.
    pub size: f32,
    /// Stroke in run units, or fill.
    pub style: GlyphStyle,
    /// The exact fractional device-space offset of the glyph origin.
    pub subpixel: (f32, f32),
    /// The device transform's 2x2 as f32.
    pub matrix: [f32; 4],
    /// The run's variation coordinates.
    pub coords: std::sync::Arc<[i16]>,
    /// The slot the emitted item reads.
    pub slot: crate::render::glyph::GlyphSlot,
}

/// Render-thread resources the lowering reads: registered fonts and
/// images (and the COLR picture cache).
pub struct Resources<'a> {
    /// Registered fonts by engine id.
    pub fonts: &'a HashMap<u64, crate::render::FontData>,
    /// Registered images by engine id.
    pub images: &'a HashMap<u64, std::sync::Arc<crate::render::CpuImage>>,
    /// Cached COLR glyph pictures by `(font, glyph, coords hash, paint hash)`.
    pub colr_cache: &'a mut HashMap<(u64, u32, u64, u64), cherenkov::Picture>,
    /// Prepared coverage retained across frames.
    pub coverage_cache: &'a mut CoverageCache,
}

/// The lowering walk state for one surface frame.
pub struct Lowering<'a> {
    items: &'a mut Vec<Item>,
    /// The resources paints and glyph runs resolve against.
    res: &'a mut Resources<'a>,
    /// Glyph mask requests emitted during the walk.
    pub glyphs: Vec<GlyphReq>,
    width: usize,
    height: usize,
    transform: Affine,
    clip: Option<ClipRef>,
}

/// The largest singular value of `t`'s linear part — the worst-case factor
/// by which a user-space distance error can grow under the transform.
#[expect(
    clippy::many_single_char_names,
    reason = "a/b/c/d are the conventional affine matrix coefficient names"
)]
pub(super) fn sigma_max(t: Affine) -> f64 {
    let [a, b, c, d, _, _] = t.as_coeffs();
    let p = a.mul_add(a, b * b) + c.mul_add(c, d * d);
    let det = a.mul_add(d, -(b * c));
    let disc = p.mul_add(p, (-4.0 * det) * det).max(0.0).sqrt();
    p.midpoint(disc).sqrt()
}

/// A `ContinuousRect` as a path of Lamé-corner line segments, like the
/// scene's `ContinuousRect::to_path_at` but with per-corner radii.
///
/// Each corner is a quarter Lamé curve `x = r·|cos t|^e`, `y = r·|sin t|^e`
/// with `e = 2/n` and `n = 2 + 2·smoothing`, recursively bisected in `t`
/// until within `tolerance` of the chord.
#[expect(
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    reason = "x/y/r/e/n geometry names, mirroring the scene expansion"
)]
fn continuous_path(c: &ContinuousRect, tolerance: f64) -> BezPath {
    fn emit(
        centre: Point,
        r: f64,
        c: usize,
        t0: f64,
        t1: f64,
        e: f64,
        tol: f64,
        path: &mut BezPath,
        depth: u32,
    ) {
        const MAX_DEPTH: u32 = 24;
        // Point on corner `c`'s Lamé arc at `t ∈ [0, π/2]`.
        let arc = |t: f64| -> Point {
            let (s, co) = (r * t.sin().powf(e), r * t.cos().powf(e));
            let (dx, dy) = match c {
                0 => (s, -co),  // TR: from (cx, cy-r) to (cx+r, cy)
                1 => (co, s),   // BR: from (cx+r, cy) to (cx, cy+r)
                2 => (-s, co),  // BL: from (cx, cy+r) to (cx-r, cy)
                _ => (-co, -s), // TL: from (cx-r, cy) to (cx, cy-r)
            };
            Point::new(centre.x + dx, centre.y + dy)
        };
        let (p0, p1) = (arc(t0), arc(t1));
        let flat = (1..4).all(|k| {
            let s = f64::from(k) * 0.25;
            let pm = arc((t1 - t0).mul_add(s, t0));
            let (cx, cy) = (p0.x + s * (p1.x - p0.x), p0.y + s * (p1.y - p0.y));
            (pm.x - cx).hypot(pm.y - cy) <= tol
        });
        if flat || depth >= MAX_DEPTH {
            path.line_to(p1);
        } else {
            let tm = (t1 - t0).mul_add(0.5, t0);
            emit(centre, r, c, t0, tm, e, tol, path, depth + 1);
            emit(centre, r, c, tm, t1, e, tol, path, depth + 1);
        }
    }

    let n = 2.0f64.mul_add(c.smoothing.clamp(0.0, 1.0), 2.0);
    let e = 2.0 / n;
    let (rect, radii) = (c.rect, c.radii);
    let half_w = rect.width() / 2.0;
    let half_h = rect.height() / 2.0;
    let r = [
        radii.top_right.clamp(0.0, half_w.min(half_h)),
        radii.bottom_right.clamp(0.0, half_w.min(half_h)),
        radii.bottom_left.clamp(0.0, half_w.min(half_h)),
        radii.top_left.clamp(0.0, half_w.min(half_h)),
    ];
    let Rect { x0, y0, x1, y1 } = rect;
    // Corner centres in order top-right, bottom-right, bottom-left,
    // top-left; a zero radius still emits its corner point, matching the
    // scene's degenerate-radius behaviour.
    let corners = [
        (x1 - r[0], y0 + r[0]),
        (x1 - r[1], y1 - r[1]),
        (x0 + r[2], y1 - r[2]),
        (x0 + r[3], y0 + r[3]),
    ];
    let mut path = BezPath::new();
    path.move_to((x0 + r[3], y0));
    for (c, &(cx, cy)) in corners.iter().enumerate() {
        let centre = Point::new(cx, cy);
        emit(
            centre,
            r[c],
            c,
            0.0,
            std::f64::consts::FRAC_PI_2,
            e,
            tolerance,
            &mut path,
            0,
        );
        // Straight edge to the next corner's start.
        let next = (c + 1) % 4;
        let (nx, ny) = corners[next];
        let rn = r[next];
        let p = match next {
            0 => Point::new(nx, ny - rn),
            1 => Point::new(nx + rn, ny),
            2 => Point::new(nx, ny + rn),
            _ => Point::new(nx - rn, ny),
        };
        path.line_to(p);
    }
    path.close_path();
    path
}

/// A `ShapeData` as a kurbo path in content space, plus its fill rule.
/// `Line` fills draw nothing.
fn shape_path(shape: &ShapeData, tol: f64) -> Option<(BezPath, FillRule)> {
    use kurbo::Shape as _;
    match shape {
        ShapeData::Rect(r) => Some((r.to_path(tol), FillRule::NonZero)),
        ShapeData::RoundedRect(r) => Some((r.to_path(tol), FillRule::NonZero)),
        ShapeData::Continuous(c) => Some((continuous_path(c, tol), FillRule::NonZero)),
        ShapeData::Circle(c) => Some((c.to_path(tol), FillRule::NonZero)),
        ShapeData::Ellipse(e) => Some((e.to_path(tol), FillRule::NonZero)),
        ShapeData::Line(_) => None,
        ShapeData::Path { elements, rule } => Some((BezPath::from_vec(elements.clone()), *rule)),
    }
}

/// Flattens `path` (already in device space) into directed edges.
#[expect(clippy::cast_possible_truncation, reason = "geometry is f32")]
pub(super) fn flatten_edges(path: BezPath, tol: f64) -> Vec<Edge> {
    let mut edges = Vec::new();
    let mut cur = Point::ZERO;
    let mut start = Point::ZERO;
    let close = |edges: &mut Vec<Edge>, cur: Point, start: Point| {
        // Fills implicitly close open subpaths.
        if cur != start {
            edges.push(Edge {
                x0: cur.x as f32,
                y0: cur.y as f32,
                x1: start.x as f32,
                y1: start.y as f32,
            });
        }
    };
    kurbo::flatten(path, tol, |el| match el {
        PathEl::MoveTo(p) => {
            close(&mut edges, cur, start);
            cur = p;
            start = p;
        }
        PathEl::LineTo(p) => {
            edges.push(Edge {
                x0: cur.x as f32,
                y0: cur.y as f32,
                x1: p.x as f32,
                y1: p.y as f32,
            });
            cur = p;
        }
        PathEl::QuadTo(..) | PathEl::CurveTo(..) => {
            // `kurbo::flatten` never emits curves.
            debug_assert!(false, "flatten emits only lines");
        }
        PathEl::ClosePath => {
            close(&mut edges, cur, start);
            cur = start;
        }
    });
    close(&mut edges, cur, start);
    edges
}

/// The bounding box of `edges` as an integer rect intersected with the
/// surface.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "edge coordinates fit i32 on a real surface"
)]
fn bbox_of(edges: &[Edge], w: usize, h: usize) -> IRect {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for e in edges {
        x0 = x0.min(e.x0.min(e.x1));
        y0 = y0.min(e.y0.min(e.y1));
        x1 = x1.max(e.x0.max(e.x1));
        y1 = y1.max(e.y0.max(e.y1));
    }
    IRect {
        x0: (x0.floor() as i32).max(0).min(w as i32),
        y0: (y0.floor() as i32).max(0).min(h as i32),
        x1: (x1.ceil() as i32).max(0).min(w as i32),
        y1: (y1.ceil() as i32).max(0).min(h as i32),
    }
}

/// Box spread adjusts semantic corner radii; other outlines use a miter band.
fn shadow_path(shape: &ShapeData, spread: f64, tolerance: f64) -> Option<(BezPath, FillRule, f64)> {
    use kurbo::Shape as _;
    let rounded = match shape {
        ShapeData::Rect(rect) => kurbo::RoundedRect::from_rect(*rect, 0.0),
        ShapeData::RoundedRect(rect) => *rect,
        _ => return shape_path(shape, tolerance).map(|(path, rule)| (path, rule, spread)),
    };
    let rect = rounded.rect().inflate(spread, spread);
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return None;
    }
    let radius = |r: f64| if r > 0.0 { (r + spread).max(0.0) } else { 0.0 };
    let radii = rounded.radii();
    let path = kurbo::RoundedRect::from_rect(
        rect,
        kurbo::RoundedRectRadii::new(
            radius(radii.top_left),
            radius(radii.top_right),
            radius(radii.bottom_right),
            radius(radii.bottom_left),
        ),
    )
    .to_path(tolerance);
    Some((path, FillRule::NonZero, 0.0))
}

/// Close each authored contour, including an implicit final closing segment.
fn closed_contours(path: &kurbo::BezPath) -> kurbo::BezPath {
    let mut closed = kurbo::BezPath::new();
    let mut open = false;
    for &element in path.elements() {
        match element {
            kurbo::PathEl::MoveTo(_) => {
                if open {
                    closed.close_path();
                }
                open = true;
            }
            kurbo::PathEl::ClosePath => open = false,
            _ => {}
        }
        closed.push(element);
    }
    if open {
        closed.close_path();
    }
    closed
}

/// Lossless binary key words; no float hashing or quantization.
#[expect(
    clippy::cast_possible_truncation,
    reason = "split the exact f64 bits into two u32 words"
)]
fn key_float(key: &mut Vec<u32>, value: f64) {
    let bits = value.to_bits();
    key.extend([bits as u32, (bits >> 32) as u32]);
}

fn key_point(key: &mut Vec<u32>, point: Point) {
    key_float(key, point.x);
    key_float(key, point.y);
}

fn key_path(key: &mut Vec<u32>, path: &BezPath) {
    key.push(u32::try_from(path.elements().len()).expect("path element count"));
    for element in path.elements() {
        match element {
            PathEl::MoveTo(point) | PathEl::LineTo(point) => {
                key.push(u32::from(matches!(element, PathEl::LineTo(_))));
                key_point(key, *point);
            }
            PathEl::QuadTo(first, last) => {
                key.push(2);
                for point in [first, last] {
                    key_point(key, *point);
                }
            }
            PathEl::CurveTo(first, second, last) => {
                key.push(3);
                for point in [first, second, last] {
                    key_point(key, *point);
                }
            }
            PathEl::ClosePath => key.push(4),
        }
    }
}

impl<'a> Lowering<'a> {
    /// Starts a lowering into `items` for a `w` × `h` surface.
    pub const fn new(
        items: &'a mut Vec<Item>,
        res: &'a mut Resources<'a>,
        size: (u32, u32),
    ) -> Self {
        Self {
            items,
            res,
            glyphs: Vec::new(),
            width: size.0 as usize,
            height: size.1 as usize,
            transform: Affine::IDENTITY,
            clip: None,
        }
    }

    /// Lowers a root layer and its clear colour. The clear itself is
    /// applied by the rasterizer to the framebuffer; the lowering emits
    /// only the layer's items.
    pub fn run(
        &mut self,
        root: &LayerNode,
        layers: &HashMap<u64, LayerNode>,
        clear: WorkingColor,
    ) -> Result<(), RenderError> {
        let _ = clear;
        self.layer_node(root, layers)
    }

    /// Retains geometric operands; nested clipping is an intersection.
    fn make_clip(&self, shape: &ShapeData) -> Option<ClipRef> {
        let sm = sigma_max(self.transform).max(1e-12);
        let (path, rule) = shape_path(shape, FLATTEN_TOL / sm)?;
        let edges = flatten_edges(self.transform * path, FLATTEN_TOL);
        let mut operands = self
            .clip
            .as_ref()
            .map_or_else(Vec::new, |clip| clip.operands.clone());
        operands.push(Operand {
            edges: edges.into(),
            rule,
        });
        Some(Arc::new(ClipGeometry { operands }))
    }

    /// Keys source geometry before stroking or flattening. A cache hit skips
    /// both operations; exact transform bits prevent fractional reuse.
    fn prepared(
        &mut self,
        path: BezPath,
        rule: FillRule,
        stroke: Option<&kurbo::Stroke>,
        tolerance: f64,
    ) -> Arc<Coverage> {
        let mut operands = self
            .clip
            .as_ref()
            .map_or_else(Vec::new, |clip| clip.operands.clone());
        let mut key = crate::render::coverage::geometry_key(&operands, self.width, self.height);
        key[0] = if stroke.is_some() { 3 } else { 2 };
        key.push(u32::from(rule == FillRule::EvenOdd));
        for coefficient in self.transform.as_coeffs() {
            key_float(&mut key, coefficient);
        }
        key_path(&mut key, &path);
        if let Some(stroke) = stroke {
            for value in [stroke.width, stroke.miter_limit, stroke.dash_offset] {
                key_float(&mut key, value);
            }
            key.extend([
                stroke.join as u32,
                stroke.start_cap as u32,
                stroke.end_cap as u32,
            ]);
            key.push(u32::try_from(stroke.dash_pattern.len()).expect("dash count"));
            for &dash in &stroke.dash_pattern {
                key_float(&mut key, dash);
            }
        }
        self.res.coverage_cache.get_or_insert(key, || {
            let path = if let Some(stroke) = stroke {
                kurbo::stroke(path, stroke, &kurbo::StrokeOpts::default(), tolerance)
            } else {
                path
            };
            let edges = flatten_edges(self.transform * path, FLATTEN_TOL);
            let bbox = bbox_of(&edges, self.width, self.height);
            if bbox.x0 >= bbox.x1 || bbox.y0 >= bbox.y1 {
                return Coverage::default();
            }
            operands.push(Operand {
                edges: edges.into(),
                rule,
            });
            crate::render::coverage::rasterize(&operands, self.width, self.height)
        })
    }

    /// Applies `clip` around `body`: `None` passes through.
    fn with_clip(
        &mut self,
        shape: Option<&ShapeData>,
        body: impl FnOnce(&mut Self) -> Result<(), RenderError>,
    ) -> Result<(), RenderError> {
        let Some(shape_data) = shape else {
            return body(self);
        };
        let Some(clip) = self.make_clip(shape_data) else {
            // A clip path with no area clips everything away.
            return Ok(());
        };
        let saved = self.clip.replace(clip);
        let result = body(self);
        self.clip = saved;
        result
    }

    /// Emits the isolation pair around `body`.
    fn isolate(
        &mut self,
        opacity: f32,
        blend: BlendMode,
        space: BlendSpace,
        inner_clip: Option<ClipRef>,
        body: impl FnOnce(&mut Self) -> Result<(), RenderError>,
    ) -> Result<(), RenderError> {
        let saved = std::mem::replace(&mut self.clip, inner_clip);
        self.items.push(Item::PushIsolate);
        let result = body(self);
        self.clip = saved;
        self.items.push(Item::PopIsolate {
            opacity,
            blend,
            space,
        });
        result
    }

    /// A layer: push its transform, then clip, then isolate for opacity,
    /// then content followed by children.
    fn layer(&mut self, id: u64, layers: &HashMap<u64, LayerNode>) -> Result<(), RenderError> {
        let Some(node) = layers.get(&id) else {
            return Ok(());
        };
        self.layer_node(node, layers)
    }

    fn layer_node(
        &mut self,
        node: &LayerNode,
        layers: &HashMap<u64, LayerNode>,
    ) -> Result<(), RenderError> {
        let saved = self.transform;
        self.transform = saved * node.transform;
        let result = self.with_clip(node.clip.as_ref(), |s| {
            if node.opacity < 1.0
                || node.blend != BlendMode::Normal
                || node.blend_space != BlendSpace::Linear
            {
                let clip = s.clip.clone();
                s.isolate(node.opacity, node.blend, node.blend_space, clip, |s| {
                    s.layer_items(node, layers)
                })
            } else {
                s.layer_items(node, layers)
            }
        });
        self.transform = saved;
        result
    }

    /// Content first, then children — the engine's layer ordering.
    fn layer_items(
        &mut self,
        node: &LayerNode,
        layers: &HashMap<u64, LayerNode>,
    ) -> Result<(), RenderError> {
        match &node.content {
            Some(ContentData::Picture(p)) => {
                let list = p.display_list();
                self.commands(list, 0, list.len())?;
            }
            None => {}
        }
        for child in &node.children {
            self.layer(*child, layers)?;
        }
        Ok(())
    }

    /// Walks commands `[start, end)` of `list`.
    fn commands(
        &mut self,
        list: &DisplayList,
        mut i: usize,
        end: usize,
    ) -> Result<(), RenderError> {
        let commands = list.commands();
        while i < end {
            match &commands[i] {
                Command::Fill { shape, paint } => self.fill(shape, paint)?,
                Command::Stroke {
                    shape,
                    stroke,
                    paint,
                } => self.stroke(shape, stroke, paint)?,
                Command::Shadow { shape, shadow } => self.shadow(shape, shadow),
                Command::Glyphs { run, paint } => self.glyph_run(run, paint)?,
                Command::Image {
                    image,
                    dst,
                    sampling,
                } => self.image_draw(*image, dst, *sampling)?,
                Command::Picture { picture, transform } => {
                    let saved = self.transform;
                    self.transform = saved * *transform;
                    let list = picture.display_list();
                    let result = self.commands(list, 0, list.len());
                    self.transform = saved;
                    result?;
                }
                Command::BeginClip { shape, end } => {
                    let inner_end = (*end as usize).min(commands.len());
                    self.with_clip(Some(shape), |s| s.commands(list, i + 1, inner_end))?;
                    i = inner_end;
                }
                Command::BeginTransform { transform, end } => {
                    let saved = self.transform;
                    self.transform = saved * *transform;
                    let inner_end = (*end as usize).min(commands.len());
                    let result = self.commands(list, i + 1, inner_end);
                    self.transform = saved;
                    result?;
                    i = inner_end;
                }
                Command::BeginGroup { group, end } => {
                    if group.filter.is_some() {
                        return Err(Unsupported::Filter.into());
                    }
                    let inner_end = (*end as usize).min(commands.len());
                    if group.opacity >= 1.0
                        && group.blend == BlendMode::Normal
                        && group.blend_space == BlendSpace::Linear
                    {
                        self.commands(list, i + 1, inner_end)?;
                    } else {
                        let clip = self.clip.clone();
                        self.isolate(group.opacity, group.blend, group.blend_space, clip, |s| {
                            s.commands(list, i + 1, inner_end)
                        })?;
                    }
                    i = inner_end;
                }
                Command::End => {}
            }
            i += 1;
        }
        Ok(())
    }

    /// `Image`: a fill of `dst` whose paint maps the rect onto the whole
    /// image, pad-extended, like the oracle's `Draw::Image`.
    fn image_draw(
        &mut self,
        image: cherenkov::ImageId,
        dst: &Rect,
        sampling: cherenkov::Sampling,
    ) -> Result<(), RenderError> {
        let Some(img) = self.res.images.get(&image.raw()) else {
            return Err(RenderError::Image(image.raw()));
        };
        let (iw, ih) = (f64::from(img.width), f64::from(img.height));
        let (dw, dh) = (dst.x1 - dst.x0, dst.y1 - dst.y0);
        if dw <= 0.0 || dh <= 0.0 {
            return Ok(());
        }
        let transform =
            Affine::translate((dst.x0, dst.y0)) * Affine::scale_non_uniform(dw / iw, dh / ih);
        let paint = Paint::Image(cherenkov::ImagePattern {
            image,
            transform,
            extend_x: cherenkov::Extend::Pad,
            extend_y: cherenkov::Extend::Pad,
            sampling,
        });
        self.fill(&ShapeData::Rect(*dst), &paint)
    }

    /// `Fill`: a flattened polygon of edges in device space.
    fn fill(&mut self, shape: &ShapeData, paint: &Paint) -> Result<(), RenderError> {
        let sm = sigma_max(self.transform).max(1e-12);
        let tol_u = FLATTEN_TOL / sm;
        let Some((path, rule)) = shape_path(shape, tol_u) else {
            return Ok(());
        };
        let paint = paint_data(paint, self.transform.inverse(), self.res.images)?;
        let coverage = self.prepared(path, rule, None, tol_u);
        let edge_count = coverage.edge_count;
        self.items.push(Item::Draw {
            coverage,
            edge_count,
            paint,
        });
        Ok(())
    }

    /// `Stroke`: `kurbo::stroke` the content-space path, transform,
    /// flatten and fill non-zero — dashes included, like the oracle.
    fn stroke(
        &mut self,
        shape: &ShapeData,
        stroke: &kurbo::Stroke,
        paint: &Paint,
    ) -> Result<(), RenderError> {
        let sm = sigma_max(self.transform).max(1e-12);
        let tol_u = FLATTEN_TOL / sm;
        let path = match shape {
            ShapeData::Line(l) => {
                let mut p = BezPath::new();
                p.move_to(l.p0);
                p.line_to(l.p1);
                p
            }
            shape => {
                let Some((path, _)) = shape_path(shape, tol_u) else {
                    return Ok(());
                };
                path
            }
        };
        let paint = paint_data(paint, self.transform.inverse(), self.res.images)?;
        let coverage = self.prepared(path, FillRule::NonZero, Some(stroke), tol_u);
        let edge_count = coverage.edge_count;
        self.items.push(Item::Draw {
            coverage,
            edge_count,
            paint,
        });
        Ok(())
    }

    /// Compile the shifted caster and clips, then convolve the exact field.
    fn shadow(&mut self, shape: &ShapeData, shadow: &cherenkov::Shadow) {
        use crate::render::coverage::{Combine, rasterize_combined};
        // Shape/stroke approximation and device flattening share one error budget.
        let segment_tolerance = FLATTEN_TOL * 0.5;
        let tolerance = segment_tolerance / sigma_max(self.transform).max(1e-12);
        let Some((path, rule, spread)) = shadow_path(shape, shadow.spread, tolerance) else {
            return;
        };
        let transform = self.transform * Affine::translate(shadow.offset);
        let clips = self
            .clip
            .as_ref()
            .map_or(&[][..], |clip| clip.operands.as_slice());
        let mut key = crate::render::coverage::geometry_key(clips, self.width, self.height);
        key[0] = 1;
        key.push(u32::from(rule == FillRule::EvenOdd));
        key_path(&mut key, &path);
        for value in transform
            .as_coeffs()
            .into_iter()
            .chain([shadow.sigma, spread])
        {
            key_float(&mut key, value);
        }
        let coverage = self.res.coverage_cache.get_or_insert(key, || {
            let mut operands = vec![Operand {
                edges: flatten_edges(transform * path.clone(), segment_tolerance).into(),
                rule,
            }];
            let combine = if spread == 0.0 {
                Combine::Intersection
            } else {
                let band = kurbo::stroke(
                    closed_contours(&path),
                    &kurbo::Stroke::new(2.0 * spread.abs())
                        .with_join(kurbo::Join::Miter)
                        .with_miter_limit(4.0),
                    &kurbo::StrokeOpts::default(),
                    tolerance,
                );
                operands.push(Operand {
                    edges: flatten_edges(transform * band, segment_tolerance).into(),
                    rule: FillRule::NonZero,
                });
                if spread > 0.0 {
                    Combine::Union
                } else {
                    Combine::Difference
                }
            };
            operands.extend_from_slice(clips);
            let caster = rasterize_combined(&operands, self.width, self.height, combine);
            crate::render::raster::blur_coverage(
                &caster,
                self.width,
                self.height,
                shadow.sigma,
                transform,
            )
        });
        let [red, green, blue, alpha] = shadow.color.components;
        self.items.push(Item::Draw {
            coverage,
            edge_count: 0,
            paint: PaintData::Solid([red * alpha, green * alpha, blue * alpha, alpha]),
        });
    }

    /// `Glyphs`: one mask request per positioned glyph. Font lookup and
    /// mask rasterization happen on the render thread before the band
    /// pass; the slot is filled by then.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::many_single_char_names,
        reason = "glyph device coordinates fit i32 on a real surface"
    )]
    fn glyph_run(&mut self, run: &GlyphRun, paint: &Paint) -> Result<(), RenderError> {
        let (font_data, font_index) = {
            let font =
                self.res.fonts.get(&run.font.raw()).ok_or_else(|| {
                    RenderError::Font(format!("unregistered font {:?}", run.font))
                })?;
            (font.data.clone(), font.index)
        };
        let pdata = paint_data(paint, self.transform.inverse(), self.res.images)?;
        let coords: std::sync::Arc<[i16]> = run.coords.clone().into();
        let font_ref = skrifa::FontRef::from_index(&font_data, font_index)
            .map_err(|error| RenderError::Font(error.to_string()))?;
        let colr_upem = if font_ref.colr().is_ok() {
            Some(f64::from(
                font_ref
                    .head()
                    .map_err(|error| RenderError::Font(format!("head: {error}")))?
                    .units_per_em(),
            ))
        } else {
            None
        };
        for glyph in &run.glyphs {
            let placement = self.transform
                * Affine::translate((f64::from(glyph.x), f64::from(glyph.y)))
                * glyph.transform.unwrap_or(Affine::IDENTITY);
            let [a, b, c, d, ..] = placement.as_coeffs();
            let matrix = [a as f32, b as f32, c as f32, d as f32];
            if matches!(run.style, GlyphStyle::Fill)
                && let Some(upem) = colr_upem
                && font_ref
                    .color_glyphs()
                    .get(skrifa::GlyphId::new(glyph.id))
                    .is_some()
            {
                if upem <= 0.0 {
                    return Err(RenderError::Font("zero units_per_em".into()));
                }
                let picture = crate::render::colr::glyph_picture(
                    run.font.raw(),
                    &font_data,
                    font_index,
                    self.res.colr_cache,
                    glyph.id,
                    &run.coords,
                    paint,
                )?;
                // `translate(x, y) * scale_non_uniform(size/upem, -size/upem)`
                // places the font-space picture at the glyph's origin.
                let s = f64::from(run.size) / upem;
                let place = placement * Affine::scale_non_uniform(s, -s);
                let saved = self.transform;
                self.transform = place;
                let list = picture.display_list();
                let result = self.commands(list, 0, list.len());
                self.transform = saved;
                result?;
                continue;
            }
            let o = placement * Point::ORIGIN;
            let (ix, iy) = (o.x.floor(), o.y.floor());
            // The oracle places glyphs at their exact origins: the mask
            // keeps the subpixel fraction unquantized (a 1/4px quantize
            // would shift every edge by up to 0.25px).
            let subpixel = ((o.x - ix) as f32, (o.y - iy) as f32);
            let key = crate::render::glyph::glyph_key(run, &coords, glyph.id, subpixel, placement);
            let slot: crate::render::glyph::GlyphSlot =
                std::sync::Arc::new(std::sync::OnceLock::new());
            self.glyphs.push(GlyphReq {
                key,
                font: run.font.raw(),
                glyph_id: glyph.id,
                size: run.size,
                style: run.style.clone(),
                subpixel,
                matrix,
                coords: coords.clone(),
                slot: slot.clone(),
            });
            self.items.push(Item::Glyph {
                slot,
                x: ix as i32,
                y: iy as i32,
                paint: pdata.clone(),
                clip: self.clip.clone(),
            });
        }
        Ok(())
    }
}
