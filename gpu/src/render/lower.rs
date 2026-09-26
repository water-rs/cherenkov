// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Lowering: a surface's layer tree and display lists become one list of
//! instanced-quad passes.

use std::collections::HashMap;
use std::ops::Range;

use cherenkov::kurbo::{Affine, BezPath, Line, PathEl, Point, Rect, Vec2};
use cherenkov::{
    BlendMode, BlendSpace, Command, DisplayList, Extend, FillRule, ImageId, ImagePattern,
    Interpolation, Paint, ShapeData, WorkingColor,
};
use cherenkov::{GlyphRun, GlyphStyle};

use crate::error::{RenderError, Unsupported};
use crate::render::GpuImage;
use skrifa::MetadataProvider as _;
use skrifa::raw::TableProvider as _;

use crate::render::glyph::{Atlas, FontData, MaskCell, PathEmit, glyph_key, rasterize};
use crate::render::instance::{
    EXTEND_NONE, EXTEND_PAD, EXTEND_REFLECT, EXTEND_REPEAT, FLAG_HAS_CLIP, FLAG_HAS_INNER,
    FLAG_HAS_MASK, Globals, INTERP_SRGB, INTERP_WORKING, Instance, KIND_FILL, KIND_GLYPH,
    KIND_SHADOW, KIND_SPAN, KIND_STROKE_DIST, KIND_STROKE_OFFSET, PAINT_IMAGE, PAINT_LINEAR,
    PAINT_RADIAL, PAINT_SOLID, PAINT_SWEEP, PAINT_TEXTURE, Shape, Stop, affine, blend_code,
};
use crate::render::path;

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

/// The blend pipeline a draw range uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PipelineKind {
    /// Fixed-function source-over compositing.
    #[default]
    SrcOver,
    /// Write the fragment result unblended; blended composites do their
    /// compositing in the shader.
    Replace,
}

/// The specialised fragment pipeline a range draws with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShaderVariant {
    /// Solid fills, spans, and glyphs: coverage + opacity + solid colour.
    #[default]
    Simple,
    /// The shadow kernel + solid colour.
    Shadow,
    /// Everything else: clips, masks, strokes, gradients, composites.
    Full,
}

/// One draw call's instance range and bound source texture.
#[derive(Clone, Debug)]
pub struct DrawRange {
    /// The scratch texture bound as group 1, `None` for the dummy texture.
    pub source: Option<usize>,
    /// The image texture bound for `PAINT_IMAGE` instances.
    pub image: Option<u64>,
    /// The pipeline variant this range draws with.
    pub pipeline: PipelineKind,
    /// The fragment-shader variant this range draws with.
    pub variant: ShaderVariant,
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
    /// Device-space `(x, y, w, h)` of the target this pass covers: the
    /// whole surface for [`Target::Surface`], the tight union bbox of its
    /// content for [`Target::Scratch`].
    pub region: [u32; 4],
    /// When set, the target's region is copied into the backdrop texture
    /// before this pass begins — the blend composite reads it explicitly.
    pub backdrop_copy: Option<[u32; 4]>,
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

#[derive(Clone)]
struct OpenPass {
    target: Target,
    clear: Option<[f32; 4]>,
    source: Option<usize>,
    image: Option<u64>,
    pipeline: PipelineKind,
    variant: ShaderVariant,
    backdrop_copy: Option<[u32; 4]>,
    ranges: Vec<DrawRange>,
    seg_start: u32,
}

/// A rollback point for [`Frame::restore`] (used by speculative
/// pass-through layers).
struct FrameSnapshot {
    instances: usize,
    stops: usize,
    passes: usize,
    open: Option<OpenPass>,
}

impl Frame {
    /// Captures the frame's emission state.
    fn snapshot(&self) -> FrameSnapshot {
        FrameSnapshot {
            instances: self.instances.len(),
            stops: self.stops.len(),
            passes: self.passes.len(),
            open: self.open.clone(),
        }
    }

    /// Reverts every emission since `snap`.
    fn restore(&mut self, snap: FrameSnapshot) {
        self.instances.truncate(snap.instances);
        self.stops.truncate(snap.stops);
        self.passes.truncate(snap.passes);
        self.open = snap.open;
    }
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
    /// A rasterized path-coverage mask.
    mask: Option<MaskCell>,
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

/// One device pixel in local space, for antialiasing margins: `2 / lmin`
/// where `lmin` is the smaller column norm of the 2x2. A degenerate
/// transform (`lmin` ~ 0) draws nothing, so the margin is 0.
fn aa_margin(transform: Affine) -> f64 {
    let [c0, c1, c2, c3, _, _] = transform.as_coeffs();
    let lmin = c0.hypot(c1).min(c2.hypot(c3));
    if lmin <= 1e-9 { 0.0 } else { 2.0 / lmin }
}

/// The specialised fragment pipeline `inst` requires: the uber-shader
/// when it reads rare fields (clip, mask, inner, strokes, non-solid
/// paint), the shadow kernel otherwise for shadows, and the trivial
/// coverage path for solid fills, spans, and glyphs.
const fn variant_of(inst: &Instance) -> ShaderVariant {
    let flags = inst.meta[3] >> 24;
    if (flags & (FLAG_HAS_CLIP | FLAG_HAS_MASK | FLAG_HAS_INNER)) != 0
        || inst.meta[1] != PAINT_SOLID
        || matches!(inst.meta[0], KIND_STROKE_OFFSET | KIND_STROKE_DIST)
    {
        return ShaderVariant::Full;
    }
    if inst.meta[0] == KIND_SHADOW {
        return ShaderVariant::Shadow;
    }
    ShaderVariant::Simple
}

/// The up-to-four border strips of `b` minus the covered box `c`:
/// top and bottom run the full width, left and right fit between them.
/// Empty strips are dropped; `c` need not lie inside `b`.
fn border_strips(b: Rect, c: Rect) -> impl Iterator<Item = Rect> {
    [
        Rect::new(b.x0, b.y0, b.x1, c.y0),
        Rect::new(b.x0, c.y1, b.x1, b.y1),
        Rect::new(b.x0, c.y0, c.x0, c.y1),
        Rect::new(c.x1, c.y0, b.x1, c.y1),
    ]
    .into_iter()
    .filter(|r| r.width() > 0.0 && r.height() > 0.0)
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

/// The paint data shared by every instance kind.
#[derive(Default)]
struct PaintData {
    kind: u32,
    color: [f32; 4],
    grad: [f32; 4],
    grad2: [f32; 4],
    first_stop: u32,
    /// `count | interp << 16 | extend << 20`; for `PAINT_IMAGE`,
    /// `extend_x | extend_y << 4 | sampling << 8`.
    packed: u32,
    /// The bound image for `PAINT_IMAGE`.
    image: Option<u64>,
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
        Paint::Mesh(_) => return Err(Unsupported::Mesh.into()),
        Paint::Image(pattern) => {
            let img = images
                .get(&pattern.image.raw())
                .ok_or_else(|| RenderError::Image(pattern.image.raw()))?;
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
        Paint::Shader(_) => return Err(Unsupported::Shader.into()),
    }
    Ok(data)
}

/// A layer node on the render thread's side, handed to the lowering.
pub struct LayerNode {
    /// Local transform.
    pub transform: Affine,
    /// Opacity; below 1.0 isolates.
    pub opacity: f32,
    /// Blend mode onto the parent; non-normal isolates.
    pub blend: cherenkov::BlendMode,
    /// Clip shape.
    pub clip: Option<ShapeData>,
    /// The content.
    pub content: Option<ContentData>,
    /// Child layers, in order.
    pub children: Vec<u64>,
    /// The layer this node is attached under — lets detach touch one
    /// child list instead of scanning every node's.
    pub parent: Option<u64>,
}

/// A layer's content.
pub enum ContentData {
    /// A shared picture.
    Picture(cherenkov::Picture),
    /// A live display list, shared with its `Content` until it changes.
    List(cherenkov::Picture),
}

/// GPU resources the lowering needs to emit glyph instances.
pub struct GlyphContext<'a> {
    /// The atlas.
    pub atlas: &'a mut Atlas,
    /// For cell uploads.
    pub queue: &'a wgpu::Queue,
    /// Registered fonts.
    pub fonts: &'a HashMap<u64, FontData>,
    /// Registered images, for dimension lookup during lowering.
    pub images: &'a HashMap<u64, GpuImage>,
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
    paths: u32,
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
            paths: 0,
        }
    }

    /// Glyphs rasterized during this lowering.
    pub const fn glyphs_rasterized(&self) -> u32 {
        self.glyphs
    }

    /// Paths rasterized during this lowering (cache misses).
    pub const fn paths_rasterized(&self) -> u32 {
        self.paths
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
                image: open.image,
                pipeline: open.pipeline,
                variant: open.variant,
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
            image: None,
            pipeline: PipelineKind::SrcOver,
            variant: ShaderVariant::Simple,
            backdrop_copy: None,
            ranges: Vec::new(),
            #[expect(clippy::cast_possible_truncation)]
            seg_start: self.frame.instances.len() as u32,
        });
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "surface size is a small positive float"
    )]
    fn finish_pass(&mut self) {
        self.end_segment();
        if let Some(open) = self.frame.open.take() {
            let region = match open.target {
                Target::Surface => [0, 0, self.width as u32, self.height as u32],
                // Scratch regions are tightened in `isolate` once the
                // pass's instance bboxes are known.
                Target::Scratch(_) => [0, 0, 0, 0],
            };
            self.frame.passes.push(Pass {
                target: open.target,
                clear: open.clear,
                ranges: open.ranges,
                region,
                backdrop_copy: open.backdrop_copy,
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

    /// Starts a new draw range when the bound image texture changes.
    fn set_image(&mut self, image: Option<u64>) {
        if self.frame.open.as_ref().is_some_and(|o| o.image != image) {
            self.end_segment();
            if let Some(open) = &mut self.frame.open {
                open.image = image;
            }
        }
    }

    /// Starts a new draw range when the pipeline variant changes.
    fn set_pipeline(&mut self, pipeline: PipelineKind) {
        if self
            .frame
            .open
            .as_ref()
            .is_some_and(|o| o.pipeline != pipeline)
        {
            self.end_segment();
            if let Some(open) = &mut self.frame.open {
                open.pipeline = pipeline;
            }
        }
    }

    /// Starts a new draw range when the shader variant changes.
    fn set_variant(&mut self, variant: ShaderVariant) {
        if self
            .frame
            .open
            .as_ref()
            .is_some_and(|o| o.variant != variant)
        {
            self.end_segment();
            if let Some(open) = &mut self.frame.open {
                open.variant = variant;
            }
        }
    }

    /// Emits `inst` into the current draw range, segmenting on its
    /// specialised fragment variant.
    fn push_instance(&mut self, inst: &Instance) {
        self.set_variant(variant_of(inst));
        self.frame.instances.push(*inst);
    }

    /// Renders `body` into an isolated scratch texture, composited back at
    /// `opacity` under the saved outer clip.
    ///
    /// When the isolation exists only for `opacity < 1`, first runs `body`
    /// non-isolated: if it stays in the current pass and its instances'
    /// device bboxes are pairwise disjoint, the instances cannot overlap
    /// and folding `opacity` into each is identical to the isolated
    /// composite. Otherwise the attempt is rolled back and isolation runs
    /// as before.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "surface size is a small positive float"
    )]
    fn isolate(
        &mut self,
        inner_clip: Option<DeviceClip>,
        opacity: f32,
        blend: cherenkov::BlendMode,
        mut body: impl FnMut(&mut Self, &mut GlyphContext<'_>) -> Result<(), RenderError>,
        glyphs: &mut GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        if opacity < 1.0
            && blend == cherenkov::BlendMode::Normal
            && self.try_passthrough(opacity, &mut body, glyphs)?
        {
            return Ok(());
        }
        self.depth += 1;
        let scratch = self.depth - 1;
        let outer_clip = self.clip;
        self.clip = inner_clip;
        // Nested isolations split this scratch's open pass into segments;
        // every segment at this depth needs the region.
        let passes_start = self.frame.passes.len();
        self.begin_pass(Target::Scratch(scratch), Some([0.0; 4]));
        let inst_start = self.frame.instances.len();
        body(self, glyphs)?;
        self.finish_pass();
        self.depth -= 1;
        self.clip = outer_clip;
        let outer_target = if self.depth == 0 {
            Target::Surface
        } else {
            Target::Scratch(self.depth - 1)
        };
        let region = tight_region(
            &self.frame.instances[inst_start..],
            self.width as u32,
            self.height as u32,
        );
        if region[2] == 0 || region[3] == 0 {
            // Nothing visible in the scratch: drop this depth's segment
            // passes and the composite entirely.
            for i in (passes_start..self.frame.passes.len()).rev() {
                if self.frame.passes[i].target == Target::Scratch(scratch) {
                    self.frame.passes.remove(i);
                }
            }
            self.begin_pass(outer_target, None);
            return Ok(());
        }
        for pass in &mut self.frame.passes[passes_start..] {
            if pass.target == Target::Scratch(scratch) {
                pass.region = region;
            }
        }
        self.begin_pass(outer_target, None);
        if blend != cherenkov::BlendMode::Normal {
            // The blend composite samples the backdrop explicitly: the
            // target's current contents are copied aside before this pass.
            if let Some(open) = &mut self.frame.open {
                open.backdrop_copy = Some(region);
            }
        }
        // The composite instance: a quad over the scratch's region
        // sampling it with `grad.xy` as the texel origin.
        self.emit_composite(scratch, opacity, region, blend);
        Ok(())
    }

    /// Speculative pass-through for [`Lowering::isolate`]. Returns `Ok(true)`
    /// when `body` stayed in the current pass with disjoint device bboxes
    /// (opacity folded per instance), `Ok(false)` after rolling back so the
    /// caller can isolate for real.
    fn try_passthrough(
        &mut self,
        opacity: f32,
        body: &mut impl FnMut(&mut Self, &mut GlyphContext<'_>) -> Result<(), RenderError>,
        glyphs: &mut GlyphContext<'_>,
    ) -> Result<bool, RenderError> {
        let snap = self.frame.snapshot();
        let depth = self.depth;
        let clip = self.clip;
        let transform = self.transform;
        let result = body(self, glyphs);
        let new = &self.frame.instances[snap.instances..];
        if result.is_ok() && self.frame.passes.len() == snap.passes && bboxes_disjoint(new) {
            for inst in &mut self.frame.instances[snap.instances..] {
                inst.params[1] *= opacity;
            }
            return Ok(true);
        }
        self.frame.restore(snap);
        self.depth = depth;
        self.clip = clip;
        self.transform = transform;
        result?;
        Ok(false)
    }

    /// Emits the composite quad for `scratch` onto the current target,
    /// covering `region` (`x, y, w, h` device pixels).
    fn emit_composite(
        &mut self,
        scratch: usize,
        opacity: f32,
        region: [u32; 4],
        blend: cherenkov::BlendMode,
    ) {
        #[expect(clippy::cast_precision_loss, reason = "region fits the surface")]
        let (rx, ry, rw, rh) = (
            region[0] as f32,
            region[1] as f32,
            region[2] as f32,
            region[3] as f32,
        );
        // `KIND_SPAN` coverage is exactly 1: the region edge is the
        // content's edge, not a shape boundary the SDF would antialias
        // into a half-covered rim — wrong under a non-Normal blend.
        // Bounds are device-space for spans and the texture paint reads
        // `pixel`, so the affine is irrelevant.
        let mut inst = self.base(KIND_SPAN, affine(Affine::IDENTITY));
        inst.bounds = [rx, ry, rx + rw, ry + rh];
        inst.meta[1] = PAINT_TEXTURE;
        inst.params[1] = opacity;
        // `grad.xy` carries the scratch region's texel origin.
        inst.grad[0] = rx;
        inst.grad[1] = ry;
        let code = blend_code(blend);
        if code != 0 {
            inst.meta[3] |= code << 16;
            self.set_pipeline(PipelineKind::Replace);
        }
        self.set_source(Some(scratch));
        self.push_instance(&inst);
        self.set_source(None);
        if code != 0 {
            self.set_pipeline(PipelineKind::SrcOver);
        }
    }

    /// An instance of `kind` under the current transform and clip.
    const fn base(&self, kind: u32, local_to_device: [f32; 8]) -> Instance {
        let mut inst = Instance::new(kind);
        inst.affine = local_to_device;
        if let Some(clip) = &self.clip {
            inst.clip_inv = affine(clip.inv);
            inst.clip = clip.shape;
            inst.meta[3] |= FLAG_HAS_CLIP << 24;
            if let Some(mask) = &clip.mask {
                // A masked clip's shape is always a sharp rect, so
                // `aspect`/`exponent` — never read by its SDF — carry the
                // mask cell size for the shader's out-of-cell guard.
                inst.params[2] = mask.device[0];
                inst.params[3] = mask.device[1];
                inst.uv[2] = mask.atlas[0];
                inst.uv[3] = mask.atlas[1];
                inst.clip.aspect = mask.size[0];
                inst.clip.exponent = mask.size[1];
                inst.meta[3] |= FLAG_HAS_MASK << 24;
            }
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
        glyphs: &GlyphContext<'_>,
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
        let paint = paint_data(
            paint,
            boxed.extra.inverse(),
            &mut self.frame.stops,
            glyphs.images,
        )?;
        inst.color = paint.color;
        inst.grad = paint.grad;
        inst.grad2 = paint.grad2;
        inst.meta[1] = paint.kind;
        inst.meta[2] = paint.first_stop;
        inst.meta[3] |= (paint.packed & 0x00ff_ffff) | (flags << 24);
        self.set_image(paint.image);
        self.push_instance(&inst);
        Ok(())
    }

    /// Applies `clip` around `body`, merging axis-aligned rects and
    /// isolating for nested non-rect clips.
    fn with_clip(
        &mut self,
        shape: Option<&ShapeData>,
        mut body: impl FnMut(&mut Self, &mut GlyphContext<'_>) -> Result<(), RenderError>,
        glyphs: &mut GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        let Some(shape_data) = shape else {
            return body(self, glyphs);
        };
        if let ShapeData::Path { elements, rule } = shape_data {
            return self.with_path_clip(elements, *rule, body, glyphs);
        }
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
            mask: None,
        };
        self.run_clipped(clip, body, glyphs)
    }

    /// Runs `body` under `clip`, merging it with the current clip when the
    /// combination stays analytic (or singly masked), isolating otherwise.
    fn run_clipped(
        &mut self,
        clip: DeviceClip,
        mut body: impl FnMut(&mut Self, &mut GlyphContext<'_>) -> Result<(), RenderError>,
        glyphs: &mut GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        /// The merged axis-aligned rect clip for `cr ∩ dr`.
        fn merged_rect(cr: Rect, dr: Rect, mask: Option<MaskCell>) -> DeviceClip {
            let merged = cr.intersect(dr);
            let size = (merged.width().max(0.0), merged.height().max(0.0));
            let center = merged.center();
            let half = [
                f32_f64(size.0 / 2.0).max(0.0),
                f32_f64(size.1 / 2.0).max(0.0),
            ];
            DeviceClip {
                inv: Affine::translate(Vec2::new(-center.x, -center.y)),
                shape: Shape::rect(half),
                aligned_rect: Some(if size.0 <= 0.0 || size.1 <= 0.0 {
                    Rect::new(center.x, center.y, center.x, center.y)
                } else {
                    merged
                }),
                mask,
            }
        }
        match self.clip {
            None => {
                self.clip = Some(clip);
                body(self, glyphs)?;
                self.clip = None;
                Ok(())
            }
            // The current clip is masked: only an aligned rect merges
            // (keeping the mask); anything else isolates.
            Some(cur) if cur.mask.is_some() => {
                if clip.mask.is_none()
                    && let (Some(cr), Some(dr)) = (cur.aligned_rect, clip.aligned_rect)
                {
                    self.clip = Some(merged_rect(cr, dr, cur.mask));
                    body(self, glyphs)?;
                    self.clip = Some(cur);
                    return Ok(());
                }
                self.isolate(Some(clip), 1.0, cherenkov::BlendMode::Normal, body, glyphs)
            }
            Some(cur) => match (clip.mask, cur.aligned_rect, clip.aligned_rect) {
                // A new masked clip merges with an aligned rect clip (or
                // attaches to the current clip's analytic shape).
                (Some(mask), Some(cr), Some(dr)) => {
                    self.clip = Some(merged_rect(cr, dr, Some(mask)));
                    body(self, glyphs)?;
                    self.clip = Some(cur);
                    Ok(())
                }
                (Some(mask), _, _) => {
                    self.clip = Some(DeviceClip {
                        inv: cur.inv,
                        shape: cur.shape,
                        aligned_rect: cur.aligned_rect,
                        mask: Some(mask),
                    });
                    body(self, glyphs)?;
                    self.clip = Some(cur);
                    Ok(())
                }
                (None, Some(cr), Some(dr)) => {
                    self.clip = Some(merged_rect(cr, dr, None));
                    body(self, glyphs)?;
                    self.clip = Some(cur);
                    Ok(())
                }
                _ => self.isolate(Some(clip), 1.0, cherenkov::BlendMode::Normal, body, glyphs),
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
                if node.opacity < 1.0 || node.blend != cherenkov::BlendMode::Normal {
                    let inner = s.clip;
                    s.isolate(
                        inner,
                        node.opacity,
                        node.blend,
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
            Some(ContentData::Picture(p) | ContentData::List(p)) => {
                self.commands(p.display_list(), 0, p.display_list().len(), glyphs)?;
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
                Command::Fill { shape, paint } => self.fill(shape, paint, glyphs)?,
                Command::Stroke {
                    shape,
                    stroke,
                    paint,
                } => self.stroke(shape, stroke, paint, glyphs)?,
                Command::Shadow { shape, shadow } => {
                    let covered = self.shadow_cover(commands.get(i + 1), shape, shadow);
                    self.shadow(shape, shadow, covered)?;
                }
                Command::Glyphs { run, paint } => self.glyph_run(run, paint, glyphs)?,
                Command::Image {
                    image,
                    dst,
                    sampling,
                } => self.image_draw(*image, dst, *sampling, glyphs)?,
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
                    if group.blend_space != BlendSpace::Linear {
                        return Err(Unsupported::BlendSpace.into());
                    }
                    let inner_end = (*end as usize).min(commands.len());
                    if group.opacity >= 1.0 && group.blend == BlendMode::Normal {
                        self.commands(list, i + 1, inner_end, glyphs)?;
                    } else {
                        self.isolate(
                            None,
                            group.opacity,
                            group.blend,
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

    /// `Fill`: a shaped quad inflated by the antialiasing margin; a path
    /// goes through the coverage rasterizer.
    fn fill(
        &mut self,
        shape: &ShapeData,
        paint: &Paint,
        glyphs: &mut GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        if let ShapeData::Path { elements, rule } = shape {
            let content = path::hash_elements(elements, fill_tag(*rule));
            return self.path(
                content,
                *rule,
                || BezPath::from_vec(elements.clone()),
                paint,
                glyphs,
            );
        }
        let Some(boxed) = box_shape(shape)? else {
            return Ok(());
        };
        let margin = aa_margin(self.transform);
        // A large axis-aligned box fill shades a full interior of
        // coverage 1: emit the guaranteed-covered device rect as one
        // `KIND_SPAN` and only the four border strips as `KIND_FILL`s.
        let to_device = self.transform * boxed.extra;
        let [ta, tb, tc, td, te, tf] = to_device.as_coeffs();
        if tb == 0.0 && tc == 0.0 && ta != 0.0 && td != 0.0 {
            let max_r = boxed.shape.radii.iter().copied().fold(0.0, f32::max);
            let inner =
                rect_around_origin(boxed.shape.half).inset(-(f64::from(max_r) + margin + 1.0));
            let (dx0, dx1) = if ta >= 0.0 {
                (ta.mul_add(inner.x0, te), ta.mul_add(inner.x1, te))
            } else {
                (ta.mul_add(inner.x1, te), ta.mul_add(inner.x0, te))
            };
            let (dy0, dy1) = if td >= 0.0 {
                (td.mul_add(inner.y0, tf), td.mul_add(inner.y1, tf))
            } else {
                (td.mul_add(inner.y1, tf), td.mul_add(inner.y0, tf))
            };
            if (dx1 - dx0) * (dy1 - dy0) >= 4096.0 {
                let span = Rect::new(dx0.ceil(), dy0.ceil(), dx1.floor(), dy1.floor());
                if span.width() > 0.0 && span.height() > 0.0 {
                    return self.fill_span(&boxed, to_device, span, margin, paint, glyphs);
                }
            }
        }
        self.emit(
            KIND_FILL,
            &boxed,
            boxed.shape,
            None,
            margin,
            paint,
            0.0,
            0,
            glyphs,
        )
    }

    /// The box `covered` split of `fill`: one device-space `KIND_SPAN` for
    /// the interior plus up to four `KIND_FILL` border strips of `b \ c`,
    /// where `c` is the span's local-space pre-image. Every piece carries
    /// the same shape, clip/mask flags, opacity and paint fields, so the
    /// fragment result is identical — only the fragment count drops.
    fn fill_span(
        &mut self,
        boxed: &Boxed,
        to_device: Affine,
        span: Rect,
        margin: f64,
        paint: &Paint,
        glyphs: &GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        let paint = paint_data(
            paint,
            boxed.extra.inverse(),
            &mut self.frame.stops,
            glyphs.images,
        )?;
        self.set_image(paint.image);
        let apply = |inst: &mut Instance, bounds: [f32; 4]| {
            inst.bounds = bounds;
            inst.shape = boxed.shape;
            inst.color = paint.color;
            inst.grad = paint.grad;
            inst.grad2 = paint.grad2;
            inst.meta[1] = paint.kind;
            inst.meta[2] = paint.first_stop;
            inst.meta[3] |= paint.packed & 0x00ff_ffff;
        };
        let mut inst = self.base(KIND_SPAN, affine(to_device));
        apply(
            &mut inst,
            [
                f32_f64(span.x0),
                f32_f64(span.y0),
                f32_f64(span.x1),
                f32_f64(span.y1),
            ],
        );
        self.push_instance(&inst);
        // The span's device rect back in local space: `to_device` is
        // axis-aligned, so invert each axis independently.
        let [ta, _, _, td, te, tf] = to_device.as_coeffs();
        let (cx0, cx1) = if ta >= 0.0 {
            ((span.x0 - te) / ta, (span.x1 - te) / ta)
        } else {
            ((span.x1 - te) / ta, (span.x0 - te) / ta)
        };
        let (cy0, cy1) = if td >= 0.0 {
            ((span.y0 - tf) / td, (span.y1 - tf) / td)
        } else {
            ((span.y1 - tf) / td, (span.y0 - tf) / td)
        };
        let c = Rect::new(cx0, cy0, cx1, cy1);
        let b = boxed.bounds.inflate(margin, margin);
        let mut inst = self.base(KIND_FILL, affine(to_device));
        apply(&mut inst, [0.0; 4]);
        for strip in border_strips(b, c) {
            inst.bounds = [
                f32_f64(strip.x0),
                f32_f64(strip.y0),
                f32_f64(strip.x1),
                f32_f64(strip.y1),
            ];
            self.push_instance(&inst);
        }
        Ok(())
    }

    /// `Image`: a fill of `dst` whose paint maps the rect onto the whole
    /// image, pad-extended, like the oracle's `Draw::Image`.
    fn image_draw(
        &mut self,
        image: ImageId,
        dst: &Rect,
        sampling: cherenkov::Sampling,
        glyphs: &mut GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        let img = glyphs
            .images
            .get(&image.raw())
            .ok_or_else(|| RenderError::Image(image.raw()))?;
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
        self.fill(&ShapeData::Rect(*dst), &paint, glyphs)
    }

    /// `Stroke`: offset strokes for circular-corner boxes, distance strokes
    /// for continuous corners and ellipses, a box fast path for lines.
    fn stroke(
        &mut self,
        shape: &ShapeData,
        stroke: &kurbo::Stroke,
        paint: &Paint,
        glyphs: &mut GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        // Path strokes and dashed strokes rasterize the stroked outline.
        if matches!(shape, ShapeData::Path { .. }) || !stroke.dash_pattern.is_empty() {
            if matches!(shape, ShapeData::Continuous(_)) {
                return Err(Unsupported::Path.into());
            }
            let tol = path::FLATTEN / path::sigma_max(self.transform).max(1e-12);
            let content = path::hash_stroke(shape, stroke, tol);
            return self.path(
                content,
                FillRule::NonZero,
                || {
                    let outline_path = path::shape_path(shape, tol)
                        .expect("shape_path is Some for non-Continuous shapes");
                    kurbo::stroke(outline_path, stroke, &kurbo::StrokeOpts::default(), tol)
                },
                paint,
                glyphs,
            );
        }
        let hw = stroke.width / 2.0;
        if let ShapeData::Line(line) = shape {
            return self.stroke_line(line, hw, stroke, paint, glyphs);
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
                glyphs,
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
                glyphs,
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
        glyphs: &GlyphContext<'_>,
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
            glyphs,
        )
    }

    /// When `commands[i]` is a `Shadow` immediately followed by an
    /// opaque solid fill of the same shape, the fill covers the shadow
    /// inside the fill's inner box. Returns that box in shadow-local
    /// space (shadow local = translate(offset) * shape local), or `None`
    /// when the next command doesn't qualify.
    #[expect(
        clippy::float_cmp,
        reason = "coverage is exact only for a fully opaque fill"
    )]
    fn shadow_cover(
        &self,
        next: Option<&Command>,
        shape: &ShapeData,
        shadow: &cherenkov::Shadow,
    ) -> Option<Rect> {
        let Some(Command::Fill {
            shape: fill_shape,
            paint,
        }) = next
        else {
            return None;
        };
        if fill_shape != shape {
            return None;
        }
        let Paint::Solid(color) = paint else {
            return None;
        };
        if color.components[3] != 1.0 {
            return None;
        }
        let fb = box_shape(fill_shape).ok().flatten()?;
        let max_r = fb.shape.radii.iter().copied().fold(0.0, f32::max);
        let inner = rect_around_origin(fb.shape.half)
            .inset(-(f64::from(max_r) + aa_margin(self.transform) + 1.0));
        Some(inner - shadow.offset)
    }

    /// `Shadow`: a Gaussian-blurred rounded box, offset and spread.
    fn shadow(
        &mut self,
        shape: &ShapeData,
        shadow: &cherenkov::Shadow,
        covered: Option<Rect>,
    ) -> Result<(), RenderError> {
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
        if s.half[0] <= 0.0 || s.half[1] <= 0.0 {
            // The spread collapsed the box: a zero-area shape casts no
            // shadow.
            return Ok(());
        }
        for r in &mut s.radii {
            if *r > 0.0 {
                *r = (*r + spread).max(0.0);
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
        self.push_shadow_quads(&inst, b, covered);
        Ok(())
    }

    /// Pushes `inst` either as one quad or, when the shadow's local box
    /// `covered` hides its interior, as the up-to-four border strips of
    /// `b \ covered`. The interior behind an opaque card is opaque
    /// shadow: coverage there is already saturated, so skipping it
    /// changes no pixels — the strips' bounds only bound rasterization.
    /// An opacity below 1 disables the split: a translucent group would
    /// composite each strip separately.
    #[expect(clippy::float_cmp, reason = "the split is exact only at full opacity")]
    fn push_shadow_quads(&mut self, inst: &Instance, b: Rect, covered: Option<Rect>) {
        let mut inst = *inst;
        let c = covered
            .map(|c| c.intersect(b))
            .filter(|c| c.width() > 0.0 && c.height() > 0.0)
            .filter(|_| inst.params[1] == 1.0);
        match c {
            None => {
                inst.bounds = [f32_f64(b.x0), f32_f64(b.y0), f32_f64(b.x1), f32_f64(b.y1)];
                self.push_instance(&inst);
            }
            Some(c) => {
                for strip in border_strips(b, c) {
                    inst.bounds = [
                        f32_f64(strip.x0),
                        f32_f64(strip.y0),
                        f32_f64(strip.x1),
                        f32_f64(strip.y1),
                    ];
                    self.push_instance(&inst);
                }
            }
        }
    }

    /// A path or stroked outline: rasterize once per
    /// (content, matrix, subpixel, surface) and replay spans plus atlas
    /// cells. `content` hashes the draw's semantics; `make` builds the
    /// local outline only on a cache miss, so replays cost no `BezPath` or
    /// stroke work. Coverage clipped by the surface is stored under the
    /// offset-specific key: it is only valid at that integer offset.
    fn path(
        &mut self,
        content: u64,
        rule: FillRule,
        make: impl FnOnce() -> BezPath,
        paint: &Paint,
        glyphs: &mut GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "surface sizes fit u32"
        )]
        let surface = (self.width as u32, self.height as u32);
        let pl = path::placement(content, self.transform, surface);
        let stored = if let Some(emit) = glyphs.atlas.path(pl.key) {
            emit.clone()
        } else if let Some(emit) = glyphs.atlas.path(pl.key_exact) {
            emit.clone()
        } else {
            let device = pl.raster * make();
            let (segments, bbox) = path::flatten_segments(&device, path::FLATTEN);
            let (emit, clipped) = if let Some(coverage) = path::rasterize(
                &segments,
                bbox,
                (f64::from(self.width), f64::from(self.height)),
                rule,
            ) {
                self.paths += 1;
                (
                    path::emit(&coverage, glyphs.atlas, glyphs.queue)?,
                    coverage.clipped,
                )
            } else {
                // Missing the surface at this offset says nothing about
                // other offsets: cache it only under the exact key.
                (PathEmit::default(), true)
            };
            let stored = emit.translated(-pl.offset.x, -pl.offset.y);
            let key = if clipped { pl.key_exact } else { pl.key };
            glyphs.atlas.insert_path(key, stored.clone());
            stored
        };
        self.replay(&stored, pl.offset, paint, glyphs)
    }

    /// Replays a cached path emission: `KIND_SPAN` runs and `KIND_GLYPH`
    /// cells at `offset` from their stored rects, painted like glyphs.
    fn replay(
        &mut self,
        emit: &PathEmit,
        offset: Vec2,
        paint: &Paint,
        glyphs: &GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        let paint = paint_data(
            paint,
            Affine::IDENTITY,
            &mut self.frame.stops,
            glyphs.images,
        )?;
        self.set_image(paint.image);
        for rect in &emit.spans {
            let mut inst = self.base(KIND_SPAN, affine(self.transform));
            inst.bounds = [
                f32_f64(f64::from(rect[0]) + offset.x),
                f32_f64(f64::from(rect[1]) + offset.y),
                f32_f64(f64::from(rect[2]) + offset.x),
                f32_f64(f64::from(rect[3]) + offset.y),
            ];
            inst.color = paint.color;
            inst.grad = paint.grad;
            inst.grad2 = paint.grad2;
            inst.meta[1] = paint.kind;
            inst.meta[2] = paint.first_stop;
            inst.meta[3] |= paint.packed & 0x00ff_ffff;
            self.push_instance(&inst);
        }
        for cell in &emit.cells {
            let mut inst = self.base(KIND_GLYPH, affine(self.transform));
            inst.bounds = [
                f32_f64(f64::from(cell.rect[0]) + offset.x),
                f32_f64(f64::from(cell.rect[1]) + offset.y),
                f32_f64(f64::from(cell.rect[2]) + offset.x),
                f32_f64(f64::from(cell.rect[3]) + offset.y),
            ];
            inst.uv = [f32::from(cell.x), f32::from(cell.y), inst.uv[2], inst.uv[3]];
            inst.color = paint.color;
            inst.grad = paint.grad;
            inst.grad2 = paint.grad2;
            inst.meta[1] = paint.kind;
            inst.meta[2] = paint.first_stop;
            inst.meta[3] |= paint.packed & 0x00ff_ffff;
            self.push_instance(&inst);
        }
        Ok(())
    }

    /// A clip with a `Path` shape: the coverage rasterized into one atlas
    /// cell multiplies every instance drawn under it. The mask is cached
    /// like a path draw: replays cost no rasterization or atlas cell.
    #[expect(clippy::cast_possible_truncation)]
    #[expect(clippy::cast_sign_loss)]
    #[expect(clippy::cast_precision_loss)]
    fn with_path_clip(
        &mut self,
        elements: &[PathEl],
        rule: FillRule,
        body: impl FnMut(&mut Self, &mut GlyphContext<'_>) -> Result<(), RenderError>,
        glyphs: &mut GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        let surface = (self.width as u32, self.height as u32);
        let pl = path::placement(path::hash_elements(elements, 2), self.transform, surface);
        let stored = if let Some(mask) = glyphs
            .atlas
            .mask(pl.key)
            .or_else(|| glyphs.atlas.mask(pl.key_exact))
        {
            *mask
        } else {
            let device = pl.raster * BezPath::from_vec(elements.to_vec());
            let (segments, bbox) = path::flatten_segments(&device, path::FLATTEN);
            let Some(coverage) = path::rasterize(
                &segments,
                bbox,
                (f64::from(self.width), f64::from(self.height)),
                rule,
            ) else {
                // Clipping to nothing at this offset draws nothing.
                return Ok(());
            };
            self.paths += 1;
            let (Ok(w), Ok(h)) = (u32::try_from(coverage.w), u32::try_from(coverage.h)) else {
                return Err(Unsupported::PathClipTooLarge.into());
            };
            if !glyphs.atlas.can_ever_fit(w, h) {
                return Err(Unsupported::PathClipTooLarge.into());
            }
            let Some((cx, cy)) = glyphs.atlas.alloc(w, h) else {
                return Err(RenderError::AtlasFull);
            };
            let texels: Vec<u8> = coverage
                .data
                .iter()
                .map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect();
            glyphs.atlas.write(glyphs.queue, cx, cy, w, h, &texels);
            let mask = MaskCell {
                device: [
                    f32_f64(coverage.x - pl.offset.x),
                    f32_f64(coverage.y - pl.offset.y),
                ],
                atlas: [cx as f32, cy as f32],
                size: [w as f32, h as f32],
                rect: [
                    f32_f64(coverage.x - pl.offset.x),
                    f32_f64(coverage.y - pl.offset.y),
                    f32_f64(coverage.x + f64::from(w) - pl.offset.x),
                    f32_f64(coverage.y + f64::from(h) - pl.offset.y),
                ],
            };
            glyphs.atlas.insert_mask(
                if coverage.clipped {
                    pl.key_exact
                } else {
                    pl.key
                },
                mask,
            );
            mask
        };
        // Stored mask rects are relative to the placement offset.
        let mask = stored.translated(pl.offset.x, pl.offset.y);
        let rect = Rect::new(
            f64::from(mask.rect[0]),
            f64::from(mask.rect[1]),
            f64::from(mask.rect[2]),
            f64::from(mask.rect[3]),
        );
        let center = rect.center();
        let clip = DeviceClip {
            inv: Affine::translate(Vec2::new(-center.x, -center.y)),
            shape: Shape::rect([f32_f64(rect.width() / 2.0), f32_f64(rect.height() / 2.0)]),
            aligned_rect: Some(rect),
            mask: Some(mask),
        };
        self.run_clipped(clip, body, glyphs)
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
                let picture =
                    crate::render::colr::glyph_picture(font, glyph.id, &run.coords, paint)?;
                // `translate(x, y) * scale_non_uniform(size/upem, -size/upem)`
                // places the font-space picture at the glyph's origin.
                let s = f64::from(run.size) / upem;
                let place = Affine::translate((f64::from(glyph.x), f64::from(glyph.y)))
                    * Affine::scale_non_uniform(s, -s);
                let saved = self.transform;
                self.transform = saved * place;
                let list = picture.display_list();
                let result = self.commands(list, 0, list.len(), glyphs);
                self.transform = saved;
                result?;
                continue;
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
            let paint_data = paint_data(
                paint,
                Affine::IDENTITY,
                &mut self.frame.stops,
                glyphs.images,
            )?;
            self.set_image(paint_data.image);
            inst.color = paint_data.color;
            inst.grad = paint_data.grad;
            inst.grad2 = paint_data.grad2;
            inst.meta[1] = paint_data.kind;
            inst.meta[2] = paint_data.first_stop;
            inst.meta[3] |= paint_data.packed & 0x00ff_ffff;
            self.push_instance(&inst);
        }
        Ok(())
    }
}

/// The path cache tag for a fill rule.
const fn fill_tag(rule: FillRule) -> u64 {
    match rule {
        FillRule::NonZero => 0,
        FillRule::EvenOdd => 1,
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

/// The globals uniform for one pass: `size` is the target region's pixel
/// size and `origin` its device-space origin.
pub const fn globals(size: [f32; 2], origin: [f32; 2]) -> Globals {
    Globals { size, origin }
}

/// An instance's device-space bounds: `bounds` transformed by `affine`,
/// unless the kind is already device space (`KIND_GLYPH`, `KIND_SPAN`).
#[expect(
    clippy::many_single_char_names,
    reason = "a..f are the conventional affine coefficient names"
)]
fn device_bbox(inst: &Instance) -> [f32; 4] {
    if inst.meta[0] == KIND_GLYPH || inst.meta[0] == KIND_SPAN {
        return inst.bounds;
    }
    let [a, b, c, d, e, f, _, _] = inst.affine;
    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    for (x, y) in [
        (inst.bounds[0], inst.bounds[1]),
        (inst.bounds[2], inst.bounds[1]),
        (inst.bounds[0], inst.bounds[3]),
        (inst.bounds[2], inst.bounds[3]),
    ] {
        let dx = a.mul_add(x, c.mul_add(y, e));
        let dy = b.mul_add(x, d.mul_add(y, f));
        min[0] = min[0].min(dx);
        min[1] = min[1].min(dy);
        max[0] = max[0].max(dx);
        max[1] = max[1].max(dy);
    }
    [min[0], min[1], max[0], max[1]]
}

/// Whether two `[x0, y0, x1, y1]` boxes overlap in area.
fn boxes_overlap(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]
}

/// Whether every instance's device bbox is pairwise non-overlapping —
/// O(n²) up to 256 instances, a sort-by-x sweep beyond.
fn bboxes_disjoint(instances: &[Instance]) -> bool {
    let mut boxes: Vec<[f32; 4]> = instances.iter().map(device_bbox).collect();
    if boxes.len() <= 256 {
        for (i, a) in boxes.iter().enumerate() {
            if boxes[i + 1..].iter().any(|b| boxes_overlap(*a, *b)) {
                return false;
            }
        }
        return true;
    }
    boxes.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let mut reach = f32::NEG_INFINITY;
    let mut y0 = f32::INFINITY;
    let mut y1 = f32::NEG_INFINITY;
    for b in boxes {
        if b[0] >= reach {
            reach = b[2];
            y0 = b[1];
            y1 = b[3];
        } else {
            // x overlaps the running interval: check its y span.
            if b[1] < y1 && b[3] > y0 {
                return false;
            }
            reach = reach.max(b[2]);
            y0 = y0.min(b[1]);
            y1 = y1.max(b[3]);
        }
    }
    true
}

/// The union device bbox of `instances` clipped to the surface, inflated
/// by one pixel and integer-rounded, as `(x, y, w, h)`; `[0, 0, 0, 0]`
/// when empty.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "region coordinates are finite, non-negative and below the surface size"
)]
fn tight_region(instances: &[Instance], width: u32, height: u32) -> [u32; 4] {
    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    for b in instances.iter().map(device_bbox) {
        if b.iter().any(|v| !v.is_finite()) {
            continue;
        }
        min[0] = min[0].min(b[0]);
        min[1] = min[1].min(b[1]);
        max[0] = max[0].max(b[2]);
        max[1] = max[1].max(b[3]);
    }
    let x0 = (min[0].floor() - 1.0).max(0.0);
    let y0 = (min[1].floor() - 1.0).max(0.0);
    let x1 = (max[0].ceil() + 1.0).min(width as f32);
    let y1 = (max[1].ceil() + 1.0).min(height as f32);
    if x1 <= x0 || y1 <= y0 {
        return [0, 0, 0, 0];
    }
    [x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An adapter plus device, or `None` where no GPU exists.
    fn device_and_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .next()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    /// An isolated group's composite quad is a `KIND_SPAN`: full coverage
    /// over its device-space region, not an SDF edge that would half-cover
    /// the rim texels.
    #[test]
    fn a_composite_is_a_full_coverage_span() {
        let Some((device, queue)) = device_and_queue() else {
            return;
        };
        let mut atlas = Atlas::new(&device, u64::MAX);
        let fonts = HashMap::new();
        let images = HashMap::new();
        let mut frame = Frame::default();
        let mut lowering = Lowering::new(&mut frame, (64, 64));
        let mut glyphs = GlyphContext {
            atlas: &mut atlas,
            queue: &queue,
            fonts: &fonts,
            images: &images,
        };
        // Two overlapping rects defeat the pass-through speculation, so
        // the group really isolates into a scratch and composites back.
        lowering
            .isolate(
                None,
                0.5,
                BlendMode::Normal,
                |s, g| {
                    s.fill(
                        &ShapeData::Rect(Rect::new(4.0, 4.0, 20.0, 20.0)),
                        &Paint::Solid(WorkingColor::new([1.0, 0.0, 0.0, 1.0])),
                        g,
                    )?;
                    s.fill(
                        &ShapeData::Rect(Rect::new(12.0, 12.0, 28.0, 28.0)),
                        &Paint::Solid(WorkingColor::new([0.0, 0.0, 1.0, 1.0])),
                        g,
                    )
                },
                &mut glyphs,
            )
            .expect("isolate");
        let composite = frame
            .instances
            .iter()
            .find(|i| i.meta[1] == PAINT_TEXTURE)
            .expect("the composite instance");
        assert_eq!(composite.meta[0], KIND_SPAN, "composites are spans");
    }

    /// A shadow whose negative spread collapses the shape's box emits no
    /// quad — a zero-area shape casts nothing.
    #[test]
    fn a_collapsed_shadow_emits_no_quads() {
        let mut frame = Frame::default();
        let mut lowering = Lowering::new(&mut frame, (64, 64));
        // Half extents [20, 5]: a spread of -20 inverts both.
        let bar = ShapeData::Rect(kurbo::Rect::new(20.0, 20.0, 60.0, 30.0));
        let collapsed =
            cherenkov::Shadow::new(2.0, WorkingColor::new([0.0, 0.0, 0.0, 1.0])).spread(-20.0);
        lowering
            .shadow(&bar, &collapsed, None)
            .expect("collapsed shadow is not an error");
        assert!(
            lowering.frame.instances.is_empty(),
            "a collapsed box emits no quad"
        );
        // A milder negative spread that leaves the box positive still
        // emits — and its radii clamp at zero rather than going negative.
        let shrunk =
            cherenkov::Shadow::new(2.0, WorkingColor::new([0.0, 0.0, 0.0, 1.0])).spread(-4.0);
        lowering.shadow(&bar, &shrunk, None).expect("shadow");
        assert_eq!(lowering.frame.instances.len(), 1);
        assert!(
            lowering.frame.instances[0]
                .shape
                .half
                .iter()
                .all(|h| *h > 0.0)
        );
        assert!(
            lowering.frame.instances[0]
                .shape
                .radii
                .iter()
                .all(|r| *r >= 0.0)
        );
    }
}
