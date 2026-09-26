// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Lowering: a surface's layer tree and display lists become one list of
//! instanced-quad passes.

use std::collections::HashMap;
use std::ops::Range;

use cherenkov::kurbo::{Affine, Line, Point, Rect, Vec2};
use cherenkov::{
    BlendMode, BlendSpace, Command, DisplayList, Extend, Interpolation, Paint, ShapeData,
    WorkingColor,
};
use cherenkov::{GlyphRun, GlyphStyle};

use crate::error::{RenderError, Unsupported};
use crate::render::glyph::{Atlas, FontData, glyph_key, rasterize};
use crate::render::instance::{
    EXTEND_PAD, EXTEND_REFLECT, EXTEND_REPEAT, FLAG_HAS_CLIP, FLAG_HAS_INNER, Globals, INTERP_SRGB,
    INTERP_WORKING, Instance, KIND_FILL, KIND_GLYPH, KIND_SHADOW, KIND_STROKE_DIST,
    KIND_STROKE_OFFSET, PAINT_LINEAR, PAINT_RADIAL, PAINT_SOLID, PAINT_TEXTURE, Shape, Stop,
    affine,
};

/// Linear Display P3 to linear sRGB (the inverse of the shader's
/// `SRGB_TO_P3`), used to store `SrgbEncoded` gradient stops.
const P3_TO_SRGB: [[f32; 3]; 3] = [
    [1.224_940_1, -0.224_940_4, 0.0],
    [-0.042_056_9, 1.042_057_1, 0.0],
    [-0.019_637_6, -0.078_636_1, 1.098_273_5],
];

/// The target a pass draws into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// The surface's render texture.
    Surface,
    /// Scratch texture at this isolation depth index.
    Scratch(usize),
}

/// One draw call's instance range and bound source texture.
#[derive(Clone, Debug)]
pub struct DrawRange {
    /// The scratch texture bound as group 1, `None` for the dummy texture.
    pub source: Option<usize>,
    /// Range into the frame's instance buffer.
    pub instances: Range<u32>,
}

/// One render pass.
#[derive(Clone, Debug)]
pub struct Pass {
    /// What it draws into.
    pub target: Target,
    /// Clear colour; `None` loads the previous contents.
    pub clear: Option<[f32; 4]>,
    /// Draw calls.
    pub ranges: Vec<DrawRange>,
}

/// A lowered frame.
#[derive(Default)]
pub struct Frame {
    /// Instance data.
    pub instances: Vec<Instance>,
    /// Gradient stops.
    pub stops: Vec<Stop>,
    /// Passes in submission order.
    pub passes: Vec<Pass>,
    open: Option<OpenPass>,
}

struct OpenPass {
    target: Target,
    clear: Option<[f32; 4]>,
    source: Option<usize>,
    ranges: Vec<DrawRange>,
    seg_start: u32,
}

impl Frame {
    /// Reuses this frame's allocations for the next lowering.
    pub fn reset(&mut self) {
        self.instances.clear();
        self.stops.clear();
        self.passes.clear();
        self.open = None;
    }
}

/// A clip shape with its device-to-clip-local transform, plus its device
/// rectangle when it is axis-aligned.
#[derive(Clone, Copy, Debug)]
struct DeviceClip {
    /// Device to clip-local.
    inv: Affine,
    /// The centred clip shape.
    shape: Shape,
    /// The clip's device-space rectangle, when it is one.
    aligned_rect: Option<Rect>,
}

/// A `ShapeData` expressed as a centred rounded box.
struct Boxed {
    /// Extra local transform (centre translation, plus rotation for
    /// ellipses and stroked lines).
    extra: Affine,
    /// The centred shape.
    shape: Shape,
    /// The local bounds, centred.
    bounds: Rect,
}

/// Converts a semantic shape into a centred rounded box plus the local
/// transform that centres it.
///
/// A `ContinuousRect`'s Lamé exponent follows the scene/oracle model:
/// `2 + 2 * smoothing`, so a circular corner is 2 and a fully continuous one
/// approaches 4.
///
/// A `Line` has no area and draws nothing, so it returns `None`.
fn box_shape(shape: &ShapeData) -> Result<Option<Boxed>, Unsupported> {
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
        ShapeData::Path { .. } => return Err(Unsupported::Path),
    };
    Ok(Some(boxed))
}

/// f64 to f32; instance data is f32 by design.
#[expect(clippy::cast_possible_truncation)]
const fn f32_f64(v: f64) -> f32 {
    v as f32
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

/// A cheap over-estimate of one device pixel in local space, for antialiasing
/// margins: `2 * max(1, 1 / min column length of the 2x2)`.
fn aa_margin(transform: Affine) -> f64 {
    let [c0, c1, c2, c3, _, _] = transform.as_coeffs();
    let l0 = c0.hypot(c1);
    let l1 = c2.hypot(c3);
    2.0 * (1.0 / l0.min(l1)).max(1.0)
}

/// The blur sigma the shader integrates against, modelling the oracle's
/// pixel-area sampling: `sqrt(sigma² + 1/6)` for a positive sigma.
fn shadow_sigma(sigma: f64) -> f64 {
    if sigma > 0.0 {
        sigma.mul_add(sigma, 1.0 / 6.0).sqrt()
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

/// The paint data shared by every instance kind.
#[derive(Default)]
struct PaintData {
    kind: u32,
    color: [f32; 4],
    grad: [f32; 4],
    grad2: [f32; 4],
    first_stop: u32,
    /// `count | interp << 16 | extend << 20`.
    packed: u32,
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
        let color = if interpolation == Interpolation::SrgbEncoded {
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
        Interpolation::SrgbEncoded => INTERP_SRGB,
    };
    let extend = match extend {
        Extend::Pad => EXTEND_PAD,
        Extend::Repeat => EXTEND_REPEAT,
        Extend::Reflect => EXTEND_REFLECT,
    };
    (first, count | (interp << 16) | (extend << 20))
}

/// Lowers a paint; `to_local` maps content space to the instance's local
/// (shape-centred) space in which the shader evaluates gradient parameters.
fn paint_data(
    paint: &Paint,
    to_local: Affine,
    stops: &mut Vec<Stop>,
) -> Result<PaintData, Unsupported> {
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
        Paint::Sweep(_) => return Err(Unsupported::Sweep),
        Paint::Mesh(_) => return Err(Unsupported::Mesh),
        Paint::Image(_) => return Err(Unsupported::Image),
        Paint::Shader(_) => return Err(Unsupported::Shader),
    }
    Ok(data)
}

/// A layer node on the render thread's side, handed to the lowering.
pub struct LayerNode {
    /// Local transform.
    pub transform: Affine,
    /// Opacity; below 1.0 isolates.
    pub opacity: f32,
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
    /// A live display list.
    List(DisplayList),
}

/// GPU resources the lowering needs to emit glyph instances.
pub struct GlyphContext<'a> {
    /// The atlas.
    pub atlas: &'a mut Atlas,
    /// For cell uploads.
    pub queue: &'a wgpu::Queue,
    /// For atlas regrowth.
    pub device: &'a wgpu::Device,
    /// Registered fonts.
    pub fonts: &'a HashMap<u64, FontData>,
}

/// The lowering walk state for one surface frame.
pub struct Lowering<'a> {
    frame: &'a mut Frame,
    width: f32,
    height: f32,
    transform: Affine,
    clip: Option<DeviceClip>,
    depth: usize,
    glyphs: u32,
}

impl<'a> Lowering<'a> {
    /// Starts a lowering into `frame` for a `width` × `height` surface.
    #[expect(clippy::cast_precision_loss, reason = "surface sizes fit f32")]
    pub const fn new(frame: &'a mut Frame, size: (u32, u32)) -> Self {
        Self {
            frame,
            width: size.0 as f32,
            height: size.1 as f32,
            transform: Affine::IDENTITY,
            clip: None,
            depth: 0,
            glyphs: 0,
        }
    }

    /// Glyphs rasterized during this lowering.
    pub const fn glyphs_rasterized(&self) -> u32 {
        self.glyphs
    }

    /// Lowers a root layer and its clear colour into the frame.
    pub fn run(
        &mut self,
        root: &LayerNode,
        layers: &HashMap<u64, LayerNode>,
        clear: WorkingColor,
        glyphs: &mut GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        let [r, g, b, a] = clear.components;
        self.begin_pass(Target::Surface, Some([r * a, g * a, b * a, a]));
        self.layer_items(root, layers, glyphs)?;
        self.finish_pass();
        Ok(())
    }

    /// Ends the open draw range at the current source boundary.
    fn end_segment(&mut self) {
        let Some(open) = &mut self.frame.open else {
            return;
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "instance counts fit u32 in practice"
        )]
        let end = self.frame.instances.len() as u32;
        if end > open.seg_start {
            open.ranges.push(DrawRange {
                source: open.source,
                instances: open.seg_start..end,
            });
            open.seg_start = end;
        }
    }

    fn begin_pass(&mut self, target: Target, clear: Option<[f32; 4]>) {
        self.finish_pass();
        self.frame.open = Some(OpenPass {
            target,
            clear,
            source: None,
            ranges: Vec::new(),
            #[expect(clippy::cast_possible_truncation)]
            seg_start: self.frame.instances.len() as u32,
        });
    }

    fn finish_pass(&mut self) {
        self.end_segment();
        if let Some(open) = self.frame.open.take() {
            self.frame.passes.push(Pass {
                target: open.target,
                clear: open.clear,
                ranges: open.ranges,
            });
        }
    }

    /// Starts a new draw range when `source` changes.
    fn set_source(&mut self, source: Option<usize>) {
        if self.frame.open.as_ref().is_some_and(|o| o.source != source) {
            self.end_segment();
            if let Some(open) = &mut self.frame.open {
                open.source = source;
            }
        }
    }

    /// Renders `body` into an isolated scratch texture, composited back at
    /// `opacity` under the saved outer clip.
    fn isolate(
        &mut self,
        inner_clip: Option<DeviceClip>,
        opacity: f32,
        body: impl FnOnce(&mut Self, &mut GlyphContext<'_>) -> Result<(), RenderError>,
        glyphs: &mut GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        self.depth += 1;
        let scratch = self.depth - 1;
        let outer_clip = self.clip;
        self.clip = inner_clip;
        self.begin_pass(Target::Scratch(scratch), Some([0.0; 4]));
        body(self, glyphs)?;
        self.finish_pass();
        self.depth -= 1;
        self.clip = outer_clip;
        let outer_target = if self.depth == 0 {
            Target::Surface
        } else {
            Target::Scratch(self.depth - 1)
        };
        self.begin_pass(outer_target, None);
        // The composite instance: a full-surface quad sampling the scratch.
        self.emit_composite(scratch, opacity);
        Ok(())
    }

    /// Emits the composite quad for `scratch` onto the current target.
    fn emit_composite(&mut self, scratch: usize, opacity: f32) {
        let half = [self.width / 2.0, self.height / 2.0];
        let mut inst = self.base(
            KIND_FILL,
            affine(Affine::translate((f64::from(half[0]), f64::from(half[1])))),
        );
        inst.bounds = [-half[0], -half[1], half[0], half[1]];
        inst.shape = Shape::rect(half);
        inst.meta[1] = PAINT_TEXTURE;
        inst.params[1] = opacity;
        self.set_source(Some(scratch));
        self.frame.instances.push(inst);
        self.set_source(None);
    }

    /// An instance of `kind` under the current transform and clip.
    const fn base(&self, kind: u32, local_to_device: [f32; 8]) -> Instance {
        let mut inst = Instance::new(kind);
        inst.affine = local_to_device;
        if let Some(clip) = &self.clip {
            inst.clip_inv = affine(clip.inv);
            inst.clip = clip.shape;
            inst.meta[3] |= FLAG_HAS_CLIP << 24;
        }
        inst
    }

    /// Emits a shaped instance: paint evaluated in local space, bounds the
    /// local quad, `margin` inflation.
    #[expect(clippy::too_many_arguments)]
    fn emit(
        &mut self,
        kind: u32,
        boxed: &Boxed,
        shape: Shape,
        inner: Option<Shape>,
        margin: f64,
        paint: &Paint,
        param_x: f32,
        flags: u32,
    ) -> Result<(), RenderError> {
        let b = boxed.bounds.inflate(margin, margin);
        if b.width() <= 0.0 || b.height() <= 0.0 {
            return Ok(());
        }
        let mut inst = self.base(kind, affine(self.transform * boxed.extra));
        inst.bounds = [f32_f64(b.x0), f32_f64(b.y0), f32_f64(b.x1), f32_f64(b.y1)];
        inst.shape = shape;
        if let Some(inner) = inner {
            inst.inner = inner;
        }
        inst.params[0] = param_x;
        let paint = paint_data(paint, boxed.extra.inverse(), &mut self.frame.stops)?;
        inst.color = paint.color;
        inst.grad = paint.grad;
        inst.grad2 = paint.grad2;
        inst.meta[1] = paint.kind;
        inst.meta[2] = paint.first_stop;
        inst.meta[3] |= (paint.packed & 0x00ff_ffff) | (flags << 24);
        self.frame.instances.push(inst);
        Ok(())
    }

    /// Applies `clip` around `body`, merging axis-aligned rects and
    /// isolating for nested non-rect clips.
    fn with_clip(
        &mut self,
        shape: Option<&ShapeData>,
        body: impl FnOnce(&mut Self, &mut GlyphContext<'_>) -> Result<(), RenderError>,
        glyphs: &mut GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        let Some(shape_data) = shape else {
            return body(self, glyphs);
        };
        let Some(boxed) = box_shape(shape_data)? else {
            return Ok(());
        };
        let inv = (self.transform * boxed.extra).inverse();
        let aligned_rect = match shape_data {
            ShapeData::Rect(r) if axis_aligned(self.transform) => {
                Some(device_rect(self.transform, *r))
            }
            _ => None,
        };
        let clip = DeviceClip {
            inv,
            shape: boxed.shape,
            aligned_rect,
        };
        match self.clip {
            None => {
                self.clip = Some(clip);
                body(self, glyphs)?;
                self.clip = None;
                Ok(())
            }
            Some(cur) => match (cur.aligned_rect, clip.aligned_rect) {
                (Some(cr), Some(dr)) => {
                    let merged = cr.intersect(dr);
                    let size = (merged.width().max(0.0), merged.height().max(0.0));
                    let center = merged.center();
                    let half = [
                        f32_f64(size.0 / 2.0).max(0.0),
                        f32_f64(size.1 / 2.0).max(0.0),
                    ];
                    let merged_clip = DeviceClip {
                        inv: Affine::translate(Vec2::new(-center.x, -center.y)),
                        shape: Shape::rect(half),
                        aligned_rect: Some(if size.0 <= 0.0 || size.1 <= 0.0 {
                            Rect::new(center.x, center.y, center.x, center.y)
                        } else {
                            merged
                        }),
                    };
                    self.clip = Some(merged_clip);
                    body(self, glyphs)?;
                    self.clip = Some(cur);
                    Ok(())
                }
                _ => self.isolate(Some(clip), 1.0, body, glyphs),
            },
        }
    }

    /// A layer: push its transform, then clip, then isolate for opacity,
    /// then content followed by children.
    fn layer(
        &mut self,
        id: u64,
        layers: &HashMap<u64, LayerNode>,
        glyphs: &mut GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        let Some(node) = layers.get(&id) else {
            return Ok(());
        };
        let saved = self.transform;
        self.transform = saved * node.transform;
        let result = self.with_clip(
            node.clip.as_ref(),
            |s, glyphs| {
                if node.opacity < 1.0 {
                    let inner = s.clip;
                    s.isolate(
                        inner,
                        node.opacity,
                        |s, glyphs| s.layer_items(node, layers, glyphs),
                        glyphs,
                    )
                } else {
                    s.layer_items(node, layers, glyphs)
                }
            },
            glyphs,
        );
        self.transform = saved;
        result
    }

    /// Content first, then children — the engine's layer ordering.
    fn layer_items(
        &mut self,
        node: &LayerNode,
        layers: &HashMap<u64, LayerNode>,
        glyphs: &mut GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        match &node.content {
            Some(ContentData::Picture(p)) => {
                self.commands(p.display_list(), 0, p.display_list().len(), glyphs)?;
            }
            Some(ContentData::List(list)) => {
                self.commands(list, 0, list.len(), glyphs)?;
            }
            None => {}
        }
        for child in &node.children {
            self.layer(*child, layers, glyphs)?;
        }
        Ok(())
    }

    /// Walks commands `[start, end)` of `list`.
    fn commands(
        &mut self,
        list: &DisplayList,
        mut i: usize,
        end: usize,
        glyphs: &mut GlyphContext<'_>,
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
                Command::Glyphs { run, paint } => self.glyph_run(run, paint, glyphs)?,
                Command::Image { .. } => return Err(Unsupported::Image.into()),
                Command::Picture { picture, transform } => {
                    let saved = self.transform;
                    self.transform = saved * *transform;
                    let list = picture.display_list();
                    let result = self.commands(list, 0, list.len(), glyphs);
                    self.transform = saved;
                    result?;
                }
                Command::BeginClip { shape, end } => {
                    let inner_end = (*end as usize).min(commands.len());
                    self.with_clip(
                        Some(shape),
                        |s, glyphs| s.commands(list, i + 1, inner_end, glyphs),
                        glyphs,
                    )?;
                    i = inner_end;
                }
                Command::BeginTransform { transform, end } => {
                    let saved = self.transform;
                    self.transform = saved * *transform;
                    let inner_end = (*end as usize).min(commands.len());
                    let result = self.commands(list, i + 1, inner_end, glyphs);
                    self.transform = saved;
                    result?;
                    i = inner_end;
                }
                Command::BeginGroup { group, end } => {
                    if group.filter.is_some() {
                        return Err(Unsupported::Filter.into());
                    }
                    if group.blend != BlendMode::Normal {
                        return Err(Unsupported::Blend.into());
                    }
                    if group.blend_space != BlendSpace::Linear {
                        return Err(Unsupported::BlendSpace.into());
                    }
                    let inner_end = (*end as usize).min(commands.len());
                    if group.opacity >= 1.0 {
                        self.commands(list, i + 1, inner_end, glyphs)?;
                    } else {
                        self.isolate(
                            None,
                            group.opacity,
                            |s, glyphs| s.commands(list, i + 1, inner_end, glyphs),
                            glyphs,
                        )?;
                    }
                    i = inner_end;
                }
                Command::End => {}
            }
            i += 1;
        }
        Ok(())
    }

    /// `Fill`: a shaped quad inflated by the antialiasing margin.
    fn fill(&mut self, shape: &ShapeData, paint: &Paint) -> Result<(), RenderError> {
        let Some(boxed) = box_shape(shape)? else {
            return Ok(());
        };
        let margin = aa_margin(self.transform);
        self.emit(KIND_FILL, &boxed, boxed.shape, None, margin, paint, 0.0, 0)
    }

    /// `Stroke`: offset strokes for circular-corner boxes, distance strokes
    /// for continuous corners and ellipses, a box fast path for lines.
    fn stroke(
        &mut self,
        shape: &ShapeData,
        stroke: &kurbo::Stroke,
        paint: &Paint,
    ) -> Result<(), RenderError> {
        if !stroke.dash_pattern.is_empty() {
            return Err(Unsupported::StrokeDash.into());
        }
        let hw = stroke.width / 2.0;
        if let ShapeData::Line(line) = shape {
            return self.stroke_line(line, hw, stroke, paint);
        }
        let Some(boxed) = box_shape(shape)? else {
            return Ok(());
        };
        let margin = hw + aa_margin(self.transform);
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
                                return Err(Unsupported::StrokeJoin.into());
                            }
                            0.0
                        }
                        kurbo::Join::Bevel => return Err(Unsupported::StrokeJoin.into()),
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
            self.emit(
                KIND_STROKE_OFFSET,
                &boxed,
                outer,
                inner_opt,
                margin,
                paint,
                f32_f64(hw),
                flags,
            )
        } else {
            self.emit(
                KIND_STROKE_DIST,
                &boxed,
                boxed.shape,
                None,
                margin,
                paint,
                f32_f64(hw),
                0,
            )
        }
    }

    /// A stroked line as a box in the line's local frame.
    fn stroke_line(
        &mut self,
        line: &Line,
        hw: f64,
        stroke: &kurbo::Stroke,
        paint: &Paint,
    ) -> Result<(), RenderError> {
        if stroke.start_cap != stroke.end_cap {
            return Err(Unsupported::StrokeJoin.into());
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
        self.emit(
            KIND_FILL,
            &boxed,
            boxed.shape,
            None,
            aa_margin(self.transform),
            paint,
            0.0,
            0,
        )
    }

    /// `Shadow`: a Gaussian-blurred rounded box, offset and spread.
    fn shadow(&mut self, shape: &ShapeData, shadow: &cherenkov::Shadow) -> Result<(), RenderError> {
        let Some(boxed) = box_shape(shape)? else {
            return Ok(());
        };
        if !shape_is_offsettable(boxed.shape) {
            return Err(Unsupported::Shadow.into());
        }
        let mut s = boxed.shape;
        let spread = f32_f64(shadow.spread);
        for h in &mut s.half {
            *h += spread;
        }
        for r in &mut s.radii {
            if *r > 0.0 {
                *r += spread;
            }
        }
        let sigma_eff = shadow_sigma(shadow.sigma);
        let boxed = Boxed {
            extra: Affine::translate(shadow.offset) * boxed.extra,
            shape: s,
            bounds: rect_around_origin(s.half),
        };
        let margin = sigma_eff.mul_add(3.0, 1.0) + aa_margin(self.transform);
        let mut inst = self.base(KIND_SHADOW, affine(self.transform * boxed.extra));
        let b = boxed.bounds.inflate(margin, margin);
        inst.bounds = [f32_f64(b.x0), f32_f64(b.y0), f32_f64(b.x1), f32_f64(b.y1)];
        inst.shape = boxed.shape;
        inst.params[0] = f32_f64(sigma_eff);
        inst.color = shadow.color.components;
        inst.meta[1] = PAINT_SOLID;
        self.frame.instances.push(inst);
        Ok(())
    }

    /// `Glyphs`: rasterize missing atlas entries and emit one quad per
    /// glyph.
    fn glyph_run(
        &mut self,
        run: &GlyphRun,
        paint: &Paint,
        glyphs: &mut GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        if matches!(run.style, GlyphStyle::Stroke(_)) {
            return Err(Unsupported::GlyphStroke.into());
        }
        let font = glyphs
            .fonts
            .get(&run.font.raw())
            .ok_or_else(|| RenderError::Font(format!("unregistered font {:?}", run.font)))?;
        for glyph in &run.glyphs {
            if glyph.transform.is_some() {
                return Err(Unsupported::GlyphTransform.into());
            }
            let o = self.transform * Point::new(f64::from(glyph.x), f64::from(glyph.y));
            let ix = o.x.floor();
            let iy = o.y.floor();
            let fx = ((o.x - ix) * 4.0).floor() / 4.0;
            let fy = ((o.y - iy) * 4.0).floor() / 4.0;
            let key = glyph_key(run, glyph.id, (f32_f64(fx), f32_f64(fy)), self.transform);
            let entry = if let Some(entry) = glyphs.atlas.get(&key) {
                entry
            } else {
                self.glyphs += 1;
                rasterize(
                    glyphs.device,
                    glyphs.queue,
                    glyphs.atlas,
                    font,
                    key,
                    glyph.id,
                    run.size,
                    (f32_f64(fx), f32_f64(fy)),
                    self.transform,
                    &run.coords,
                )?
            };
            if entry.w == 0 || entry.h == 0 {
                continue;
            }
            let mut inst = self.base(KIND_GLYPH, affine(self.transform));
            let x0 = f32_f64(ix + f64::from(entry.left));
            let y0 = f32_f64(iy + f64::from(entry.top));
            inst.bounds = [x0, y0, x0 + f32::from(entry.w), y0 + f32::from(entry.h)];
            inst.uv = [f32::from(entry.x), f32::from(entry.y), 0.0, 0.0];
            let paint_data = paint_data(paint, Affine::IDENTITY, &mut self.frame.stops)?;
            inst.color = paint_data.color;
            inst.grad = paint_data.grad;
            inst.grad2 = paint_data.grad2;
            inst.meta[1] = paint_data.kind;
            inst.meta[2] = paint_data.first_stop;
            inst.meta[3] |= paint_data.packed & 0x00ff_ffff;
            self.frame.instances.push(inst);
        }
        Ok(())
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

/// The globals uniform for a surface.
#[expect(clippy::cast_precision_loss)]
pub const fn globals(size: (u32, u32)) -> Globals {
    Globals {
        size: [size.0 as f32, size.1 as f32],
        pad: [0.0; 2],
    }
}
