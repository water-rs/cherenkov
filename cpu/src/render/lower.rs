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
use crate::render::paint::{PaintData, paint_data};
use crate::render::raster::{Edge, coverage_mask};

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

/// A rasterized clip: an integer rect fast path or a full-surface
/// coverage mask.
#[derive(Debug)]
pub enum ClipMask {
    /// Axis-aligned rect with integer edges: coverage is 1 inside, 0
    /// outside.
    Rect(IRect),
    /// Exact-area coverage of a general clip shape, `w * h` cells.
    Cover(Vec<f32>),
}

/// A clip shared by items; cheap to clone.
pub type ClipRef = Arc<ClipMask>;

/// One rasterization item of a lowered frame.
#[derive(Debug)]
pub enum Item {
    /// A filled, flattened polygon.
    Draw {
        /// Directed edges in device space.
        edges: Arc<[Edge]>,
        /// Device-space bounding box of the edges.
        bbox: IRect,
        /// The fill rule.
        rule: FillRule,
        /// The paint evaluator.
        paint: PaintData,
        /// The clip in force.
        clip: Option<ClipRef>,
    },
    /// A Gaussian-blurred, axis-aligned rounded box.
    Shadow {
        /// The device-space axis-aligned rounded box half extents and
        /// centre, `[cx, cy, hx, hy]`.
        rbox: [f32; 4],
        /// Per-corner radii.
        radii: [f32; 4],
        /// The effective blur sigma (`sqrt(sigma² + 1/6)`).
        sigma_eff: f32,
        /// Premultiplied colour.
        color: [f32; 4],
        /// Device-space bounding box.
        bbox: IRect,
        /// The clip in force.
        clip: Option<ClipRef>,
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
    /// The quantized subpixel offset of the glyph origin.
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
fn sigma_max(t: Affine) -> f64 {
    let [a, b, c, d, _, _] = t.as_coeffs();
    let p = a.mul_add(a, b * b) + c.mul_add(c, d * d);
    let det = a.mul_add(d, -(b * c));
    let disc = p.mul_add(p, (-4.0 * det) * det).sqrt();
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
fn flatten_edges(path: BezPath, tol: f64) -> Vec<Edge> {
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
    #[expect(clippy::float_cmp, reason = "horizontal edges carry no area")]
    edges.retain(|e| e.y0 != e.y1);
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

/// Whether the transform's 2x2 is axis-aligned (`b == c == 0`, or a 90°
/// rotation with `a == d == 0`).
fn axis_aligned(transform: Affine) -> bool {
    let [c0, c1, c2, c3, _, _] = transform.as_coeffs();
    (c1 == 0.0 && c2 == 0.0) || (c0 == 0.0 && c3 == 0.0)
}

/// The device-space rectangle of a rect under an axis-aligned transform.
fn device_rect(t: Affine, r: Rect) -> Rect {
    let p0 = t * Point::new(r.x0, r.y0);
    let p1 = t * Point::new(r.x1, r.y1);
    Rect::new(
        p0.x.min(p1.x),
        p0.y.min(p1.y),
        p0.x.max(p1.x),
        p0.y.max(p1.y),
    )
}

/// Whether all four edges of `r` are within `1e-6` of integers.
fn integer_edges(r: Rect) -> Option<IRect> {
    let close = |v: f64| (v - v.round()).abs() <= 1e-6;
    if close(r.x0) && close(r.y0) && close(r.x1) && close(r.y1) {
        Some(IRect {
            #[expect(clippy::cast_possible_truncation)]
            x0: r.x0.round() as i32,
            #[expect(clippy::cast_possible_truncation)]
            y0: r.y0.round() as i32,
            #[expect(clippy::cast_possible_truncation)]
            x1: r.x1.round() as i32,
            #[expect(clippy::cast_possible_truncation)]
            y1: r.y1.round() as i32,
        })
    } else {
        None
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
        self.layer_items(root, layers)
    }

    /// The coverage a clip contributes at `(px, py)`.
    fn clip_cov_at(clip: &ClipMask, px: usize, py: usize, w: usize) -> f32 {
        match clip {
            ClipMask::Rect(r) => f32::from(
                px >= usize::try_from(r.x0).unwrap_or(0)
                    && px < usize::try_from(r.x1).unwrap_or(0)
                    && py >= usize::try_from(r.y0).unwrap_or(0)
                    && py < usize::try_from(r.y1).unwrap_or(0),
            ),
            ClipMask::Cover(mask) => mask[py * w + px],
        }
    }

    /// A clip shape under the current transform becomes a [`ClipRef`],
    /// combined (coverage product) with the clip already in force.
    fn make_clip(&self, shape: &ShapeData) -> Option<ClipRef> {
        // The rect fast path: an axis-aligned rect with integer-ish edges.
        let new_rect = match shape {
            ShapeData::Rect(r) if axis_aligned(self.transform) => device_rect(self.transform, *r),
            _ => Rect::ZERO, // sentinel: not applicable
        };
        let new_is_rect = matches!(shape, ShapeData::Rect(_))
            && axis_aligned(self.transform)
            && integer_edges(new_rect).is_some();
        if let Some(current) = &self.clip {
            if new_is_rect
                && let ClipMask::Rect(cur) = current.as_ref()
                && let Some(r) = integer_edges(new_rect)
            {
                return Some(Arc::new(ClipMask::Rect(IRect {
                    x0: cur.x0.max(r.x0),
                    y0: cur.y0.max(r.y0),
                    x1: cur.x1.min(r.x1),
                    y1: cur.y1.min(r.y1),
                })));
            }
        } else if new_is_rect {
            return integer_edges(new_rect).map(|r| Arc::new(ClipMask::Rect(r)));
        }
        // General path: rasterize the new clip's coverage over the full
        // surface, multiplied by the coverage of the clip already in
        // force (coverage product — the oracle intersects geometry; the
        // product is the accepted approximation, as on the GPU slice).
        let (w, h) = (self.width, self.height);
        let sm = sigma_max(self.transform).max(1e-12);
        let tol_u = FLATTEN_TOL / sm;
        let (path, rule) = shape_path(shape, tol_u)?;
        let edges = flatten_edges(self.transform * path, FLATTEN_TOL);
        let mut mask = coverage_mask(&edges, rule, w, h);
        if let Some(current) = &self.clip {
            for (i, m) in mask.iter_mut().enumerate() {
                let (px, py) = (i % w, i / w);
                *m *= Self::clip_cov_at(current, px, py, w);
            }
        }
        Some(Arc::new(ClipMask::Cover(mask)))
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
        inner_clip: Option<ClipRef>,
        body: impl FnOnce(&mut Self) -> Result<(), RenderError>,
    ) -> Result<(), RenderError> {
        let saved = std::mem::replace(&mut self.clip, inner_clip);
        self.items.push(Item::PushIsolate);
        let result = body(self);
        self.clip = saved;
        self.items.push(Item::PopIsolate { opacity, blend });
        result
    }

    /// A layer: push its transform, then clip, then isolate for opacity,
    /// then content followed by children.
    fn layer(&mut self, id: u64, layers: &HashMap<u64, LayerNode>) -> Result<(), RenderError> {
        let Some(node) = layers.get(&id) else {
            return Ok(());
        };
        let saved = self.transform;
        self.transform = saved * node.transform;
        let result = self.with_clip(node.clip.as_ref(), |s| {
            if node.opacity < 1.0 || node.blend != BlendMode::Normal {
                let clip = s.clip.clone();
                s.isolate(node.opacity, node.blend, clip, |s| {
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
                Command::Shadow { shape, shadow } => self.shadow(shape, shadow)?,
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
                    if group.blend_space != BlendSpace::Linear {
                        return Err(Unsupported::BlendSpace.into());
                    }
                    let inner_end = (*end as usize).min(commands.len());
                    if group.opacity >= 1.0 && group.blend == BlendMode::Normal {
                        self.commands(list, i + 1, inner_end)?;
                    } else {
                        let clip = self.clip.clone();
                        self.isolate(group.opacity, group.blend, clip, |s| {
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
        let edges = flatten_edges(self.transform * path, FLATTEN_TOL);
        if edges.is_empty() {
            return Ok(());
        }
        let paint = paint_data(paint, self.transform.inverse(), self.res.images)?;
        let bbox = bbox_of(&edges, self.width, self.height);
        if bbox.x0 >= bbox.x1 || bbox.y0 >= bbox.y1 {
            return Ok(());
        }
        self.items.push(Item::Draw {
            edges: edges.into(),
            bbox,
            rule,
            paint,
            clip: self.clip.clone(),
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
        let outline = kurbo::stroke(path, stroke, &kurbo::StrokeOpts::default(), tol_u);
        let edges = flatten_edges(self.transform * outline, FLATTEN_TOL);
        if edges.is_empty() {
            return Ok(());
        }
        let paint = paint_data(paint, self.transform.inverse(), self.res.images)?;
        let bbox = bbox_of(&edges, self.width, self.height);
        self.items.push(Item::Draw {
            edges: edges.into(),
            bbox,
            rule: FillRule::NonZero,
            paint,
            clip: self.clip.clone(),
        });
        Ok(())
    }

    /// `Shadow`: a Gaussian-blurred rounded box in device space.
    ///
    /// Only shapes whose corners are circular under the transform —
    /// `Rect`, `RoundedRect`, `Circle` — under an axis-aligned
    /// transform; everything else reports [`Unsupported::Shadow`].
    /// Radii are handled per corner (the closed form works per corner).
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::many_single_char_names,
        reason = "shadow geometry is f32 and surface sizes fit i32"
    )]
    fn shadow(&mut self, shape: &ShapeData, shadow: &cherenkov::Shadow) -> Result<(), RenderError> {
        // The shape as a centred rect plus per-corner radii, in content
        // space.
        let (rect, radii) = match shape {
            ShapeData::Rect(r) => (*r, [0.0; 4]),
            ShapeData::RoundedRect(rr) => {
                let r = rr.rect();
                let limit = r.width().min(r.height()) / 2.0;
                let radii = rr.radii();
                (
                    r,
                    [
                        radii.top_left.clamp(0.0, limit),
                        radii.top_right.clamp(0.0, limit),
                        radii.bottom_right.clamp(0.0, limit),
                        radii.bottom_left.clamp(0.0, limit),
                    ],
                )
            }
            ShapeData::Circle(c) => (
                Rect::from_center_size(c.center, (c.radius * 2.0, c.radius * 2.0)),
                [c.radius; 4],
            ),
            _ => return Err(Unsupported::Shadow.into()),
        };
        if !axis_aligned(self.transform) {
            return Err(Unsupported::Shadow.into());
        }
        let [a, b, c, d, _, _] = self.transform.as_coeffs();
        let (sx, sy) = (a.hypot(b), c.hypot(d));
        let smax = sx.max(sy).max(1e-12);
        // The device-space box: transform the rect, offset by the linear
        // part applied to the shadow offset.
        let dr = device_rect(self.transform, rect);
        let (cx, cy) = (
            c.mul_add(
                shadow.offset.y,
                a.mul_add(shadow.offset.x, dr.x0.midpoint(dr.x1)),
            ),
            d.mul_add(
                shadow.offset.y,
                b.mul_add(shadow.offset.x, dr.y0.midpoint(dr.y1)),
            ),
        );
        let spread = shadow.spread * smax;
        let (hx, hy) = (
            (dr.width() / 2.0 + spread).max(0.0),
            (dr.height() / 2.0 + spread).max(0.0),
        );
        let limit = hx.min(hy);
        let radii = radii.map(|r| {
            if r > 0.0 {
                r.mul_add(smax, spread).clamp(0.0, limit)
            } else {
                0.0
            }
        });
        let sigma_eff =
            ((shadow.sigma * smax).mul_add(shadow.sigma * smax, 1.0 / 6.0)).sqrt() as f32;
        let margin = f64::from(sigma_eff).mul_add(3.0, 1.0);
        let (w, h) = (self.width as i32, self.height as i32);
        let bbox = IRect {
            x0: ((cx - hx - margin).floor() as i32).clamp(0, w),
            y0: ((cy - hy - margin).floor() as i32).clamp(0, h),
            x1: ((cx + hx + margin).ceil() as i32).clamp(0, w),
            y1: ((cy + hy + margin).ceil() as i32).clamp(0, h),
        };
        if bbox.x0 >= bbox.x1 || bbox.y0 >= bbox.y1 {
            return Ok(());
        }
        let [r, g, bl, al] = shadow.color.components;
        self.items.push(Item::Shadow {
            rbox: [cx as f32, cy as f32, hx as f32, hy as f32],
            radii: radii.map(|r| r as f32),
            sigma_eff,
            color: [r * al, g * al, bl * al, al],
            bbox,
            clip: self.clip.clone(),
        });
        Ok(())
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
        if matches!(run.style, GlyphStyle::Stroke(_)) {
            return Err(Unsupported::GlyphStroke.into());
        }
        let (font_data, font_index) = {
            let font =
                self.res.fonts.get(&run.font.raw()).ok_or_else(|| {
                    RenderError::Font(format!("unregistered font {:?}", run.font))
                })?;
            (font.data.clone(), font.index)
        };
        let pdata = paint_data(paint, self.transform.inverse(), self.res.images)?;
        let [a, b, c, d, ..] = self.transform.as_coeffs();
        let matrix = [a as f32, b as f32, c as f32, d as f32];
        let coords: std::sync::Arc<[i16]> = run.coords.clone().into();
        // `upem`/`color_glyphs` are resolved lazily — plain runs never parse
        // the font here (the rasterizer does it on cache miss).
        let mut colr_ctx: Option<(skrifa::FontRef<'_>, f64)> = None;
        let mut colr_checked = false;
        for glyph in &run.glyphs {
            if glyph.transform.is_some() {
                return Err(Unsupported::GlyphTransform.into());
            }
            if !colr_checked {
                colr_checked = true;
                let font_ref = skrifa::FontRef::from_index(&font_data, font_index)
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
                let place = Affine::translate((f64::from(glyph.x), f64::from(glyph.y)))
                    * Affine::scale_non_uniform(s, -s);
                let saved = self.transform;
                self.transform = saved * place;
                let list = picture.display_list();
                let result = self.commands(list, 0, list.len());
                self.transform = saved;
                result?;
                continue;
            }
            let o = self.transform * Point::new(f64::from(glyph.x), f64::from(glyph.y));
            let (ix, iy) = (o.x.floor(), o.y.floor());
            let (fx, fy) = (
                ((o.x - ix) * 4.0).floor() / 4.0,
                ((o.y - iy) * 4.0).floor() / 4.0,
            );
            let subpixel = (fx as f32, fy as f32);
            let key = crate::render::glyph::glyph_key(run, glyph.id, subpixel, self.transform);
            let slot: crate::render::glyph::GlyphSlot =
                std::sync::Arc::new(std::sync::OnceLock::new());
            self.glyphs.push(GlyphReq {
                key,
                font: run.font.raw(),
                glyph_id: glyph.id,
                size: run.size,
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
