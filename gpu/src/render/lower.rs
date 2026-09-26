// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Lowering: a surface's layer tree and display lists become one list of
//! instanced-quad passes.

use cherenkov::lowering::Realization;
use std::collections::HashMap;
use std::ops::Range;

use super::prepared::{ClipShape, Op, Outline, PaintData, ResolvedPaint, box_shape};
use cherenkov::kurbo::{Affine, BezPath, PathEl, Point, Rect, Vec2};
use cherenkov::{FillRule, GlyphRun, ShapeData, WorkingColor};

use cherenkov::RenderError;

use crate::names;
use cherenkov::{LayerId, SurfaceTree};

use crate::render::GpuImage;

use crate::render::glyph::{self, Atlas, FontData, MaskCell, PathEmit, PendingRaster, glyph_key};
use crate::render::instance::{
    FLAG_HAS_CLIP, FLAG_HAS_INNER, FLAG_HAS_MASK, Globals, Instance, KIND_FILL, KIND_GLYPH,
    KIND_SHADOW, KIND_SPAN, KIND_STROKE_DIST, KIND_STROKE_OFFSET, PAINT_SOLID, PAINT_TEXTURE,
    Shape, Stop, affine, blend_code,
};
use crate::render::path;

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
    mask: Option<ClipMask>,
}

/// A path-clip mask: stored in the atlas, or produced by a pending
/// raster the render thread will store — then `uv.zw` of every instance
/// emitted under it is patched by `Lowering::mask_patches`.
#[derive(Clone, Copy, Debug)]
enum ClipMask {
    /// Stored: the atlas origin is `cell.atlas`.
    Cell(MaskCell),
    /// Pending its raster (index into `Lowering::pending`).
    Pending(MaskCell, u32),
}

impl ClipMask {
    /// The mask data: device rect, size, and the pending-or-stored atlas
    /// origin.
    const fn cell(self) -> MaskCell {
        match self {
            Self::Cell(m) | Self::Pending(m, _) => m,
        }
    }

    /// This mask shifted by `(dx, dy)` device pixels.
    fn translated(self, dx: f64, dy: f64) -> Self {
        match self {
            Self::Cell(m) => Self::Cell(m.translated(dx, dy)),
            Self::Pending(m, i) => Self::Pending(m.translated(dx, dy), i),
        }
    }
}

/// f64 to f32; instance data is f32 by design.
#[expect(clippy::cast_possible_truncation)]
const fn f32_f64(v: f64) -> f32 {
    v as f32
}

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

/// The part of a shadow its following opaque fill hides, in shadow-local
/// space: the fill's rounded box minus its corner squares, as the union of
/// `wide` (corner rows excluded) and `tall` (corner columns excluded).
/// Both are inset by an antialiasing margin so the fill's edge pixels stay.
#[derive(Clone, Copy, PartialEq)]
struct Cover {
    wide: Rect,
    tall: Rect,
}

/// The four border strips of `b` minus the covered box `c`: top and
/// bottom run the full width, left and right fit between them. Strips
/// may be empty; `c` need not lie inside `b`.
const fn border_strips(b: Rect, c: Rect) -> [Rect; 4] {
    [
        Rect::new(b.x0, b.y0, b.x1, c.y0),
        Rect::new(b.x0, c.y1, b.x1, b.y1),
        Rect::new(b.x0, c.y0, c.x0, c.y1),
        Rect::new(c.x1, c.y0, b.x1, c.y1),
    ]
}

/// The up-to-eight strips of `b` minus the cover `c`: full-width top and
/// bottom strips above/below `wide`, the four corner blocks beside `wide`
/// but above/below `tall`, and the two side strips beside `tall`. A cover
/// box that misses `b` degenerates to the four [`border_strips`] of the
/// other; empty strips are dropped.
fn cover_strips(b: Rect, c: Cover) -> impl Iterator<Item = Rect> {
    let nonempty = |r: Rect| r.width() > 0.0 && r.height() > 0.0;
    let w = c.wide.intersect(b);
    let t = c.tall.intersect(b);
    // The strip layout below needs `w` to span the rows `t` does not;
    // swap when `t` reaches higher or lower than `w`.
    let (w, t) = if t.y0 < w.y0 || t.y1 > w.y1 {
        (t, w)
    } else {
        (w, t)
    };
    let mut rects = [Rect::ZERO; 8];
    match (nonempty(w), nonempty(t)) {
        (true, true) => {
            rects = [
                Rect::new(b.x0, b.y0, b.x1, w.y0),
                Rect::new(b.x0, w.y1, b.x1, b.y1),
                Rect::new(b.x0, w.y0, w.x0, t.y0),
                Rect::new(w.x1, w.y0, b.x1, t.y0),
                Rect::new(b.x0, t.y1, w.x0, w.y1),
                Rect::new(w.x1, t.y1, b.x1, w.y1),
                Rect::new(b.x0, t.y0, t.x0, t.y1),
                Rect::new(t.x1, t.y0, b.x1, t.y1),
            ];
        }
        (true, false) => rects[..4].copy_from_slice(&border_strips(b, w)),
        (false, true) => rects[..4].copy_from_slice(&border_strips(b, t)),
        (false, false) => rects[0] = b,
    }
    rects.into_iter().filter(move |r| nonempty(*r))
}

/// A layer's retained content and device output.
pub type ContentData = cherenkov::lowering::Content<Op, Emission>;

/// Per-operation device output under its sampled placement.
pub struct Emission {
    pub(crate) pending_cells: Vec<(u32, u32, u32)>,
    cover: Option<Cover>,
    transform: Affine,
    size: [f32; 2],
    generation: u64,
    pub(crate) instances: Vec<Instance>,
    stops: Vec<Stop>,
    image: Option<u64>,
}

/// GPU resources the lowering needs to emit glyph instances.
pub struct GlyphContext<'a> {
    /// The atlas, read-only here: lookups never mutate, misses become
    /// [`PendingRaster`]s on the `Lowering`, applied serially on the
    /// render thread.
    pub atlas: &'a Atlas,
    /// Registered fonts — a per-worker snapshot, so reads and the COLR
    /// cache stay lock-free.
    pub fonts: &'a HashMap<u64, FontData>,
    /// Registered images, for dimension lookup during lowering.
    pub images: &'a HashMap<u64, GpuImage>,
}

/// One surface's lowering output: the raster counts plus every deferred
/// atlas insert and COLR cache update, in lowering order, for the render
/// thread to commit before encoding.
#[derive(Default)]
pub struct Lowered {
    /// Source commands resolved this frame.
    pub commands: u32,
    /// Layers with new device realizations.
    pub layers: u32,
    /// Glyphs rasterized during the lowering.
    pub glyphs: u32,
    /// Path rasters during the lowering (cache misses).
    pub paths: u32,
    /// Deferred rasters, in lowering order.
    pub pending: Vec<PendingRaster>,
    /// `uv.xy` patches: `(instance, pending index, cell index)`.
    pub cell_patches: Vec<(u32, u32, u32)>,
    /// `uv.zw` patches: `(instance, pending index)`.
    pub mask_patches: Vec<(u32, u32)>,
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
    /// `(instance, pending, cell)` triples whose `uv.xy` are set when the
    /// render thread stores the pending path's cells.
    pub(crate) cell_patches: Vec<(u32, u32, u32)>,
    /// `(instance, pending)` pairs whose `uv.zw` are set when the render
    /// thread stores the pending clip mask.
    pub(crate) mask_patches: Vec<(u32, u32)>,
    /// Atlas writes and cache updates to commit, in lowering order.
    pub(crate) pending: Vec<PendingRaster>,
    pub commands_lowered: u32,
    pub layers_composed: u32,
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
            cell_patches: Vec::new(),
            mask_patches: Vec::new(),
            pending: Vec::new(),
            commands_lowered: 0,
            layers_composed: 0,
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

    /// Lowers a surface's sampled [`SurfaceTree`] and its clear colour
    /// into the frame. `caches` holds each layer's render-side content.
    pub fn run(
        &mut self,
        tree: &SurfaceTree,
        caches: &mut HashMap<LayerId, ContentData>,
        clear: WorkingColor,
        glyphs: &GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        for content in caches.values_mut() {
            self.commands_lowered += content.prepare(&mut super::prepared::Lowerer {
                fonts: glyphs.fonts,
                images: glyphs.images,
                pending: &mut self.pending,
            })?;
        }
        let [r, g, b, a] = clear.components;
        self.begin_pass(Target::Surface, Some([r * a, g * a, b * a, a]));
        self.layer(tree.root(), tree, caches, glyphs)?;
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
    /// specialised fragment variant. Under a pending clip mask the
    /// instance's `uv.zw` is patched after the mask is stored.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a surface emits far fewer than u32::MAX instances"
    )]
    fn push_instance(&mut self, inst: &Instance) {
        self.set_variant(variant_of(inst));
        self.frame.instances.push(*inst);
        if let Some(ClipMask::Pending(_, pending)) = self.clip.and_then(|c| c.mask) {
            self.mask_patches
                .push((self.frame.instances.len() as u32 - 1, pending));
        }
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
        mut body: impl FnMut(&mut Self, &GlyphContext<'_>) -> Result<(), RenderError>,
        glyphs: &GlyphContext<'_>,
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
        body: &mut impl FnMut(&mut Self, &GlyphContext<'_>) -> Result<(), RenderError>,
        glyphs: &GlyphContext<'_>,
    ) -> Result<bool, RenderError> {
        let snap = self.frame.snapshot();
        let patches = (self.cell_patches.len(), self.mask_patches.len());
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
        self.cell_patches.truncate(patches.0);
        self.mask_patches.truncate(patches.1);
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
        Self::apply_clip(&mut inst, self.clip);
        inst
    }

    /// Clip state belongs to composition; coverage and glyph atlas UVs remain retained.
    const fn apply_clip(inst: &mut Instance, clip: Option<DeviceClip>) {
        if let Some(clip) = clip {
            inst.clip_inv = affine(clip.inv);
            inst.clip = clip.shape;
            inst.meta[3] |= FLAG_HAS_CLIP << 24;
            if let Some(mask) = clip.mask {
                let cell = mask.cell();
                // A masked clip's shape is always a sharp rect, so
                // `aspect`/`exponent` — never read by its SDF — carry the
                // mask cell size for the shader's out-of-cell guard.
                inst.params[2] = cell.device[0];
                inst.params[3] = cell.device[1];
                inst.clip.aspect = cell.size[0];
                inst.clip.exponent = cell.size[1];
                inst.meta[3] |= FLAG_HAS_MASK << 24;
                if let ClipMask::Cell(cell) = mask {
                    inst.uv[2] = cell.atlas[0];
                    inst.uv[3] = cell.atlas[1];
                }
            }
        }
    }

    /// Applies `clip` around `body`, merging axis-aligned rects and
    /// isolating for nested non-rect clips.
    fn with_clip(
        &mut self,
        shape: Option<&ShapeData>,
        mut body: impl FnMut(&mut Self, &GlyphContext<'_>) -> Result<(), RenderError>,
        glyphs: &GlyphContext<'_>,
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
        mut body: impl FnMut(&mut Self, &GlyphContext<'_>) -> Result<(), RenderError>,
        glyphs: &GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        /// The merged axis-aligned rect clip for `cr ∩ dr`.
        fn merged_rect(cr: Rect, dr: Rect, mask: Option<ClipMask>) -> DeviceClip {
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

    /// A layer: push its transform, then clip, then isolate for opacity
    /// and blend, then content followed by children. The clip applies in
    /// `transform` space; content and children draw in
    /// `content_transform` space, which is where `scroll_offset` bites.
    fn layer(
        &mut self,
        id: LayerId,
        tree: &SurfaceTree,
        caches: &mut HashMap<LayerId, ContentData>,
        glyphs: &GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        let node = tree.layer(id);
        if node.filter.is_some() {
            return Err(RenderError::Unsupported(names::FILTER));
        }
        if node.backdrop.is_some() {
            return Err(RenderError::Unsupported(names::BACKDROP));
        }
        let saved = self.transform;
        self.transform = saved * node.transform;
        let content_space = saved * node.content_transform();
        let result = self.with_clip(
            node.clip.as_ref(),
            |s, glyphs| {
                s.transform = content_space;
                if node.opacity < 1.0 || node.blend != cherenkov::BlendMode::Normal {
                    let inner = s.clip;
                    s.isolate(
                        inner,
                        node.opacity,
                        node.blend,
                        |s, glyphs| s.layer_items(id, node, tree, caches, glyphs),
                        glyphs,
                    )
                } else {
                    s.layer_items(id, node, tree, caches, glyphs)
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
        id: LayerId,
        node: &cherenkov::LayerNode,
        tree: &SurfaceTree,
        caches: &mut HashMap<LayerId, ContentData>,
        glyphs: &GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        if let Some(content) = caches.get_mut(&id) {
            let (ops, emissions) = content.prepared();
            let changed = self.ops(ops, emissions, 0, ops.len(), glyphs)?;
            self.layers_composed += u32::from(changed);
        }
        for child in &node.children {
            self.layer(*child, tree, caches, glyphs)?;
        }
        Ok(())
    }

    /// Compose retained ops under the sampled layer state. Only invalid leaf
    /// realizations produce new instances or coverage; scopes assemble passes.
    fn ops(
        &mut self,
        ops: &[Op],
        emissions: &mut [Realization<Emission>],
        mut i: usize,
        end: usize,
        glyphs: &GlyphContext<'_>,
    ) -> Result<bool, RenderError> {
        let mut changed = false;
        while i < end {
            match &ops[i] {
                Op::BeginClip { local, shape, end } => {
                    let saved = self.transform;
                    self.transform = saved * *local;
                    let body = |s: &mut Self, g: &GlyphContext<'_>| {
                        s.transform = saved;
                        changed |= s.ops(ops, emissions, i + 1, *end as usize, g)?;
                        Ok(())
                    };
                    match shape {
                        ClipShape::Empty => {}
                        ClipShape::Boxed { extra, shape, rect } => {
                            let clip = DeviceClip {
                                inv: (self.transform * *extra).inverse(),
                                shape: *shape,
                                aligned_rect: rect
                                    .filter(|_| axis_aligned(self.transform))
                                    .map(|r| device_rect(self.transform, r)),
                                mask: None,
                            };
                            self.run_clipped(clip, body, glyphs)?;
                        }
                        ClipShape::Path { elements, rule, .. } => {
                            self.with_path_clip(elements, *rule, body, glyphs)?;
                        }
                    }
                    self.transform = saved;
                    i = *end as usize;
                }
                Op::BeginIsolate {
                    opacity,
                    blend,
                    end,
                } => {
                    self.isolate(
                        None,
                        *opacity,
                        *blend,
                        |s, g| {
                            changed |= s.ops(ops, emissions, i + 1, *end as usize, g)?;
                            Ok(())
                        },
                        glyphs,
                    )?;
                    i = *end as usize;
                }
                Op::End => unreachable!("paired scopes consume their ends"),
                op => changed |= self.leaf(op, ops.get(i + 1), &mut emissions[i], glyphs)?,
            }
            i += 1;
        }
        Ok(changed)
    }

    /// Retain each leaf's instances and gradient stops independently. A dirty
    /// command drops just its entries; atlas resets invalidate device addresses.
    fn leaf(
        &mut self,
        op: &Op,
        next: Option<&Op>,
        cache: &mut Realization<Emission>,
        glyphs: &GlyphContext<'_>,
    ) -> Result<bool, RenderError> {
        let clip = self.clip;
        let cover = self.shadow_cover(op, next);
        let hit = cache.valid
            && cache.data.as_ref().is_some_and(|e| {
                e.cover == cover
                    && e.transform == self.transform
                    && e.size.map(f32::to_bits) == [self.width, self.height].map(f32::to_bits)
                    && e.generation == glyphs.atlas.generation()
            });
        if !hit {
            let mut frame = Frame::default();
            if let Some(previous) = cache.data.take() {
                frame.instances = previous.instances;
                frame.stops = previous.stops;
                frame.reset();
            }
            let mut compose = Lowering::new(&mut frame, (0, 0));
            compose.width = self.width;
            compose.height = self.height;
            compose.transform = self.transform;
            compose.begin_pass(Target::Surface, None);
            compose.realize(op, cover, glyphs)?;
            self.glyphs += compose.glyphs;
            self.paths += compose.paths;
            let image = compose.frame.open.as_ref().expect("leaf pass").image;
            let pending_base = u32::try_from(self.pending.len()).expect("pending count fits u32");
            self.pending.append(&mut compose.pending);
            let pending_cells = compose
                .cell_patches
                .iter()
                .map(|&(i, p, c)| (i, p + pending_base, c))
                .collect();
            cache.data = Some(Emission {
                pending_cells,
                cover,
                transform: self.transform,
                size: [self.width, self.height],
                generation: glyphs.atlas.generation(),
                instances: frame.instances,
                stops: frame.stops,
                image,
            });
        }
        cache.valid = true;
        let emission = cache.data.as_ref().expect("leaf realized");
        let offset = u32::try_from(self.frame.stops.len()).expect("stop count fits u32");
        self.frame.stops.extend_from_slice(&emission.stops);
        self.set_image(emission.image);
        let instance_base =
            u32::try_from(self.frame.instances.len()).expect("instance count fits u32");
        for inst in &emission.instances {
            let mut inst = *inst;
            inst.meta[2] += offset;
            Self::apply_clip(&mut inst, clip);
            self.push_instance(&inst);
        }
        self.cell_patches.extend(
            emission
                .pending_cells
                .iter()
                .map(|&(i, p, c)| (i + instance_base, p, c)),
        );
        Ok(!hit)
    }

    fn resolved_paint(&mut self, paint: &ResolvedPaint) -> PaintData {
        let mut data = paint.data.clone();
        data.first_stop += u32::try_from(self.frame.stops.len()).expect("stop count fits u32");
        self.frame.stops.extend_from_slice(&paint.stops);
        self.set_image(data.image);
        data
    }

    fn realize(
        &mut self,
        op: &Op,
        cover: Option<Cover>,
        glyphs: &GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        match op {
            Op::Shaped {
                kind,
                local,
                ambient,
                shape,
                inner,
                bounds,
                extra_margin,
                paint,
                param_x,
                flags,
            } => {
                let margin = extra_margin + aa_margin(self.transform * *ambient);
                let b = bounds.inflate(margin, margin);
                if b.width() <= 0.0 || b.height() <= 0.0 {
                    return Ok(());
                }
                let mut inst = self.base(*kind, affine(self.transform * *local));
                inst.bounds = [f32_f64(b.x0), f32_f64(b.y0), f32_f64(b.x1), f32_f64(b.y1)];
                inst.shape = *shape;
                if let Some(inner) = inner {
                    inst.inner = *inner;
                }
                inst.params[0] = *param_x;
                let paint = self.resolved_paint(paint);
                inst.color = paint.color;
                inst.grad = paint.grad;
                inst.grad2 = paint.grad2;
                inst.meta[1] = paint.kind;
                inst.meta[2] = paint.first_stop;
                inst.meta[3] |= (paint.packed & 0x00ff_ffff) | (flags << 24);
                self.push_shaped(inst, self.transform * *local, *bounds, margin);
            }
            Op::Shadow {
                local,
                ambient,
                shape,
                bounds,
                sigma_eff,
                color,
            } => {
                let margin = sigma_eff.mul_add(3.0, 1.0) + aa_margin(self.transform * *ambient);
                let b = bounds.inflate(margin, margin);
                let mut inst = self.base(KIND_SHADOW, affine(self.transform * *local));
                inst.bounds = [f32_f64(b.x0), f32_f64(b.y0), f32_f64(b.x1), f32_f64(b.y1)];
                inst.shape = *shape;
                inst.params[0] = f32_f64(*sigma_eff);
                inst.color = *color;
                inst.meta[1] = PAINT_SOLID;
                self.push_shadow_quads(&inst, b, cover);
            }
            Op::Path {
                local,
                content,
                rule,
                outline,
                paint,
            } => {
                self.transform *= *local;
                match outline {
                    Outline::Fill(elements) => self.path(
                        *content,
                        *rule,
                        || BezPath::from_vec(elements.to_vec()),
                        paint,
                        glyphs,
                    )?,
                    Outline::Stroke { shape, stroke } => {
                        let tol = path::FLATTEN / path::sigma_max(self.transform).max(1e-12);
                        let content = path::hash_stroke(shape, stroke, tol);
                        self.path(
                            content,
                            *rule,
                            || {
                                let path =
                                    path::shape_path(shape, tol).expect("supported stroke shape");
                                kurbo::stroke(path, stroke, &kurbo::StrokeOpts::default(), tol)
                            },
                            paint,
                            glyphs,
                        )?;
                    }
                }
            }
            Op::Glyphs { local, run, paint } => {
                self.transform *= *local;
                self.glyph_run(run, paint, glyphs)?;
            }
            _ => unreachable!("scope is composed, never realized as a leaf"),
        }
        Ok(())
    }

    /// A following opaque box hides a rectangle inside each pair of its corner rows.
    #[expect(
        clippy::float_cmp,
        reason = "occlusion requires exact opacity and matching axes"
    )]
    fn shadow_cover(&self, op: &Op, next: Option<&Op>) -> Option<Cover> {
        let Op::Shadow { local, .. } = op else {
            return None;
        };
        let Some(Op::Shaped {
            kind: KIND_FILL,
            local: fill,
            ambient,
            shape,
            bounds,
            paint,
            flags: 0,
            ..
        }) = next
        else {
            return None;
        };
        if paint.data.kind != PAINT_SOLID || paint.data.color[3] != 1.0 {
            return None;
        }
        let relative = local.inverse() * *fill;
        let [scale_x, skew_y, skew_x, scale_y, offset_x, offset_y] = relative.as_coeffs();
        if [scale_x, skew_y, skew_x, scale_y] != [1.0, 0.0, 0.0, 1.0] {
            return None;
        }
        let max_r = f64::from(shape.radii.iter().copied().fold(0.0, f32::max));
        let m = aa_margin(self.transform * *ambient) + 1.0;
        Some(Cover {
            wide: bounds.inset((-m, -(max_r + m))) + Vec2::new(offset_x, offset_y),
            tall: bounds.inset((-(max_r + m), -m)) + Vec2::new(offset_x, offset_y),
        })
    }

    /// Split a large aligned fill into its full-coverage interior and antialiased border.
    fn push_shaped(&mut self, mut inst: Instance, to_device: Affine, bounds: Rect, margin: f64) {
        let [scale_x, skew_y, skew_x, scale_y, offset_x, offset_y] = to_device.as_coeffs();
        if inst.meta[0] != KIND_FILL
            || skew_y != 0.0
            || skew_x != 0.0
            || scale_x == 0.0
            || scale_y == 0.0
        {
            self.push_instance(&inst);
            return;
        }
        let radius = f64::from(inst.shape.radii.iter().copied().fold(0.0, f32::max));
        let inner = bounds.inset(-(radius + margin + 1.0));
        let device = device_rect(to_device, inner);
        let span = Rect::new(
            device.x0.ceil(),
            device.y0.ceil(),
            device.x1.floor(),
            device.y1.floor(),
        );
        if inner.width() <= 0.0
            || inner.height() <= 0.0
            || device.area() < 4096.0
            || span.width() <= 0.0
            || span.height() <= 0.0
        {
            self.push_instance(&inst);
            return;
        }
        inst.meta[0] = KIND_SPAN;
        inst.bounds = [
            f32_f64(span.x0),
            f32_f64(span.y0),
            f32_f64(span.x1),
            f32_f64(span.y1),
        ];
        self.push_instance(&inst);
        let (x0, x1) = if scale_x >= 0.0 {
            (
                (span.x0 - offset_x) / scale_x,
                (span.x1 - offset_x) / scale_x,
            )
        } else {
            (
                (span.x1 - offset_x) / scale_x,
                (span.x0 - offset_x) / scale_x,
            )
        };
        let (y0, y1) = if scale_y >= 0.0 {
            (
                (span.y0 - offset_y) / scale_y,
                (span.y1 - offset_y) / scale_y,
            )
        } else {
            (
                (span.y1 - offset_y) / scale_y,
                (span.y0 - offset_y) / scale_y,
            )
        };
        inst.meta[0] = KIND_FILL;
        for strip in border_strips(bounds.inflate(margin, margin), Rect::new(x0, y0, x1, y1))
            .into_iter()
            .filter(|r| r.width() > 0.0 && r.height() > 0.0)
        {
            inst.bounds = [
                f32_f64(strip.x0),
                f32_f64(strip.y0),
                f32_f64(strip.x1),
                f32_f64(strip.y1),
            ];
            self.push_instance(&inst);
        }
    }

    /// Pushes `inst` either as one quad or, when the shadow's local
    /// `covered` region hides its interior, as the up-to-eight strips of
    /// `b \ covered`. The interior behind an opaque card is opaque
    /// shadow: coverage there is already saturated, so skipping it
    /// changes no pixels — the strips' bounds only bound rasterization.
    /// An opacity below 1 disables the split: a translucent group would
    /// composite each strip separately.
    #[expect(clippy::float_cmp, reason = "the split is exact only at full opacity")]
    fn push_shadow_quads(&mut self, inst: &Instance, b: Rect, covered: Option<Cover>) {
        let mut inst = *inst;
        let c = covered.filter(|_| inst.params[1] == 1.0);
        match c {
            None => {
                inst.bounds = [f32_f64(b.x0), f32_f64(b.y0), f32_f64(b.x1), f32_f64(b.y1)];
                self.push_instance(&inst);
            }
            Some(c) => {
                for strip in cover_strips(b, c) {
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
        paint: &ResolvedPaint,
        glyphs: &GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "surface sizes fit u32"
        )]
        let surface = (self.width as u32, self.height as u32);
        let pl = path::placement(content, self.transform, surface);
        let (stored, pending) = 'stored: {
            if let Some(emit) = glyphs
                .atlas
                .path(pl.key)
                .or_else(|| glyphs.atlas.path(pl.key_exact))
            {
                break 'stored (emit.clone(), None);
            }
            let device = pl.raster * make();
            let (segments, bbox) = path::flatten_segments(&device, path::FLATTEN);
            let Some(coverage) = path::rasterize(
                &segments,
                bbox,
                (f64::from(self.width), f64::from(self.height)),
                rule,
            ) else {
                // Missing the surface at this offset says nothing about
                // other offsets: cache the empty emission only under the
                // exact key.
                let pending = u32::try_from(self.pending.len()).expect("pending count fits u32");
                self.pending.push(PendingRaster::Path {
                    key: pl.key_exact,
                    emit: PathEmit::default(),
                    cells: Vec::new(),
                });
                break 'stored (PathEmit::default(), Some(pending));
            };
            self.paths += 1;
            let (emit, cells) = path::emit(&coverage)?;
            // Emission rects are relative to the placement offset; the
            // stored record keeps that frame.
            let stored = emit.translated(-pl.offset.x, -pl.offset.y);
            let key = if coverage.clipped {
                pl.key_exact
            } else {
                pl.key
            };
            let pending = u32::try_from(self.pending.len()).expect("pending count fits u32");
            self.pending.push(PendingRaster::Path {
                key,
                emit: stored.clone(),
                cells,
            });
            (stored, Some(pending))
        };
        self.replay(&stored, pending, pl.offset, paint);
        Ok(())
    }

    /// Replays a cached path emission: `KIND_SPAN` runs and `KIND_GLYPH`
    /// cells at `offset` from their stored rects, painted like glyphs.
    fn replay(
        &mut self,
        emit: &PathEmit,
        pending: Option<u32>,
        offset: Vec2,
        paint: &ResolvedPaint,
    ) {
        let paint = self.resolved_paint(paint);
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
        for (i, cell) in emit.cells.iter().enumerate() {
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
            if let Some(pending) = pending {
                self.cell_patches.push((
                    u32::try_from(self.frame.instances.len() - 1).expect("instance index fits u32"),
                    pending,
                    u32::try_from(i).expect("cell index fits u32"),
                ));
            }
        }
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
        body: impl FnMut(&mut Self, &GlyphContext<'_>) -> Result<(), RenderError>,
        glyphs: &GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        let surface = (self.width as u32, self.height as u32);
        let pl = path::placement(path::hash_elements(elements, 2), self.transform, surface);
        let stored = 'stored: {
            if let Some(mask) = glyphs
                .atlas
                .mask(pl.key)
                .or_else(|| glyphs.atlas.mask(pl.key_exact))
            {
                break 'stored ClipMask::Cell(*mask);
            }
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
                return Err(RenderError::Unsupported(names::PATH_CLIP_TOO_LARGE));
            };
            let texels: Vec<u8> = coverage
                .data
                .iter()
                .map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect();
            let key = if coverage.clipped {
                pl.key_exact
            } else {
                pl.key
            };
            if let Some(mask) = glyphs.atlas.mask(key) {
                break 'stored ClipMask::Cell(*mask);
            }
            if !glyphs.atlas.can_ever_fit(w, h) {
                return Err(RenderError::Unsupported(names::PATH_CLIP_TOO_LARGE));
            }
            let mask = MaskCell {
                device: [
                    f32_f64(coverage.x - pl.offset.x),
                    f32_f64(coverage.y - pl.offset.y),
                ],
                // Filled by `Atlas::store_mask` on the render thread.
                atlas: [0.0, 0.0],
                size: [w as f32, h as f32],
                rect: [
                    f32_f64(coverage.x - pl.offset.x),
                    f32_f64(coverage.y - pl.offset.y),
                    f32_f64(coverage.x + f64::from(w) - pl.offset.x),
                    f32_f64(coverage.y + f64::from(h) - pl.offset.y),
                ],
            };
            let pending = u32::try_from(self.pending.len()).expect("pending count fits u32");
            self.pending.push(PendingRaster::Mask {
                key,
                mask,
                w,
                h,
                texels,
            });
            ClipMask::Pending(mask, pending)
        };
        // Stored mask rects are relative to the placement offset.
        let stored = stored.translated(pl.offset.x, pl.offset.y);
        let mask = stored.cell();
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
            mask: Some(stored),
        };
        self.run_clipped(clip, body, glyphs)
    }

    /// `Glyphs`: rasterize missing atlas entries and emit one quad per
    /// glyph.
    fn glyph_run(
        &mut self,
        run: &GlyphRun,
        paint: &ResolvedPaint,
        glyphs: &GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        let font = glyphs
            .fonts
            .get(&run.font.raw())
            .ok_or_else(|| RenderError::Font(format!("unregistered font {:?}", run.font)))?;
        for glyph in &run.glyphs {
            let o = self.transform * Point::new(f64::from(glyph.x), f64::from(glyph.y));
            let ix = o.x.floor();
            let iy = o.y.floor();
            let fx = ((o.x - ix) * 4.0).floor() / 4.0;
            let fy = ((o.y - iy) * 4.0).floor() / 4.0;
            let key = glyph_key(run, glyph.id, (f32_f64(fx), f32_f64(fy)), self.transform);
            let (entry, pending) = glyph::entry(
                glyphs.atlas,
                font,
                key,
                glyph.id,
                run.size,
                (f32_f64(fx), f32_f64(fy)),
                self.transform,
                &run.coords,
                &mut self.pending,
            )?;
            self.glyphs += u32::from(pending.is_some());
            if entry.w == 0 || entry.h == 0 {
                continue;
            }
            let mut inst = self.base(KIND_GLYPH, affine(self.transform));
            let x0 = f32_f64(ix + f64::from(entry.left));
            let y0 = f32_f64(iy + f64::from(entry.top));
            inst.bounds = [x0, y0, x0 + f32::from(entry.w), y0 + f32::from(entry.h)];
            inst.uv = [f32::from(entry.x), f32::from(entry.y), 0.0, 0.0];
            let paint_data = self.resolved_paint(paint);
            inst.color = paint_data.color;
            inst.grad = paint_data.grad;
            inst.grad2 = paint_data.grad2;
            inst.meta[1] = paint_data.kind;
            inst.meta[2] = paint_data.first_stop;
            inst.meta[3] |= paint_data.packed & 0x00ff_ffff;
            self.push_instance(&inst);
            if let Some(pending) = pending {
                let inst =
                    u32::try_from(self.frame.instances.len() - 1).expect("instance count fits u32");
                self.cell_patches.push((inst, pending, 0));
            }
        }
        Ok(())
    }
}

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

    fn draw(
        lowering: &mut Lowering<'_>,
        command: &cherenkov::Command,
        glyphs: &GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        use cherenkov::lowering::Compiler as _;
        let mut pending = Vec::new();
        let mut compiler = super::super::prepared::Lowerer {
            fonts: glyphs.fonts,
            images: glyphs.images,
            pending: &mut pending,
        };
        let mut ops = Vec::new();
        compiler.draw(command, Affine::IDENTITY, &mut ops)?;
        for op in ops {
            lowering.realize(&op, None, glyphs)?;
        }
        Ok(())
    }

    /// An isolated group's composite quad is a `KIND_SPAN`: full coverage
    /// over its device-space region, not an SDF edge that would half-cover
    /// the rim texels.
    #[test]
    fn a_composite_is_a_full_coverage_span() {
        let Some((device, _queue)) = device_and_queue() else {
            return;
        };
        let atlas = Atlas::new(&device, u64::MAX);
        let fonts = HashMap::new();
        let images = HashMap::new();
        let mut frame = Frame::default();
        let mut lowering = Lowering::new(&mut frame, (64, 64));
        let glyphs = GlyphContext {
            atlas: &atlas,
            fonts: &fonts,
            images: &images,
        };
        // Two overlapping rects defeat the pass-through speculation, so
        // the group really isolates into a scratch and composites back.
        lowering
            .isolate(
                None,
                0.5,
                cherenkov::BlendMode::Normal,
                |s, g| {
                    draw(
                        s,
                        &cherenkov::Command::Fill {
                            shape: ShapeData::Rect(Rect::new(4.0, 4.0, 20.0, 20.0)),
                            paint: cherenkov::Paint::Solid(WorkingColor::new([1.0, 0.0, 0.0, 1.0])),
                        },
                        g,
                    )?;
                    draw(
                        s,
                        &cherenkov::Command::Fill {
                            shape: ShapeData::Rect(Rect::new(12.0, 12.0, 28.0, 28.0)),
                            paint: cherenkov::Paint::Solid(WorkingColor::new([0.0, 0.0, 1.0, 1.0])),
                        },
                        g,
                    )
                },
                &glyphs,
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
        let Some((device, _queue)) = device_and_queue() else {
            return;
        };
        let atlas = Atlas::new(&device, u64::MAX);
        let fonts = HashMap::new();
        let images = HashMap::new();
        let glyphs = GlyphContext {
            atlas: &atlas,
            fonts: &fonts,
            images: &images,
        };
        // Half extents [20, 5]: a spread of -20 inverts both.
        let bar = ShapeData::Rect(kurbo::Rect::new(20.0, 20.0, 60.0, 30.0));
        let collapsed =
            cherenkov::Shadow::new(2.0, WorkingColor::new([0.0, 0.0, 0.0, 1.0])).spread(-20.0);
        draw(
            &mut lowering,
            &cherenkov::Command::Shadow {
                shape: bar.clone(),
                shadow: collapsed,
            },
            &glyphs,
        )
        .expect("collapsed shadow is not an error");
        assert!(
            lowering.frame.instances.is_empty(),
            "a collapsed box emits no quad"
        );
        // A milder negative spread that leaves the box positive still
        // emits — and its radii clamp at zero rather than going negative.
        let shrunk =
            cherenkov::Shadow::new(2.0, WorkingColor::new([0.0, 0.0, 0.0, 1.0])).spread(-4.0);
        draw(
            &mut lowering,
            &cherenkov::Command::Shadow {
                shape: bar,
                shadow: shrunk,
            },
            &glyphs,
        )
        .expect("shadow");
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

    fn area(r: Rect) -> f64 {
        r.width() * r.height()
    }

    fn disjoint(strips: &[Rect]) -> bool {
        strips.iter().enumerate().all(|(i, a)| {
            strips[i + 1..].iter().all(|b| {
                let i = a.intersect(*b);
                !(i.width() > 0.0 && i.height() > 0.0)
            })
        })
    }

    fn inside(strips: &[Rect], b: Rect) -> bool {
        strips.iter().all(|r| r.intersect(b) == *r)
    }

    #[expect(
        clippy::float_cmp,
        clippy::suboptimal_flops,
        reason = "integer-valued geometry is exact"
    )]
    #[test]
    fn cover_strips_full_cover() {
        let b = Rect::new(-20.0, -20.0, 20.0, 20.0);
        let c = Cover {
            wide: Rect::new(-15.0, -5.0, 15.0, 5.0),
            tall: Rect::new(-5.0, -15.0, 5.0, 15.0),
        };
        let strips: Vec<Rect> = cover_strips(b, c).collect();
        assert_eq!(strips.len(), 8);
        assert!(disjoint(&strips));
        assert!(inside(&strips, b));
        let total: f64 = strips.iter().map(|r| area(*r)).sum();
        assert_eq!(total, 1600.0 - (30.0 * 10.0 + 10.0 * 30.0 - 10.0 * 10.0));
    }

    #[expect(clippy::float_cmp, reason = "integer-valued geometry is exact")]
    #[test]
    fn cover_strips_one_empty_box() {
        let b = Rect::new(-20.0, -20.0, 20.0, 20.0);
        let c = Cover {
            wide: Rect::new(-15.0, -5.0, 15.0, 5.0),
            tall: Rect::new(-5.0, 0.0, 5.0, 0.0),
        };
        let strips: Vec<Rect> = cover_strips(b, c).collect();
        assert_eq!(strips.len(), 4);
        assert!(disjoint(&strips));
        assert!(inside(&strips, b));
        let total: f64 = strips.iter().map(|r| area(*r)).sum();
        assert_eq!(total, 1600.0 - 300.0);
    }

    #[test]
    fn cover_strips_both_empty() {
        let b = Rect::new(-20.0, -20.0, 20.0, 20.0);
        let c = Cover {
            wide: Rect::ZERO,
            tall: Rect::new(0.0, -5.0, 0.0, 5.0),
        };
        let strips: Vec<Rect> = cover_strips(b, c).collect();
        assert_eq!(strips, vec![b]);
    }

    #[expect(clippy::float_cmp, reason = "integer-valued geometry is exact")]
    #[test]
    fn cover_strips_clips_to_b() {
        let b = Rect::new(-20.0, -20.0, 20.0, 20.0);
        let c = Cover {
            wide: Rect::new(-15.0, -5.0, 40.0, 5.0),
            tall: Rect::new(-5.0, -40.0, 5.0, 15.0),
        };
        let strips: Vec<Rect> = cover_strips(b, c).collect();
        assert!(disjoint(&strips));
        assert!(inside(&strips, b));
        let covered = area(c.wide.intersect(b)) + area(c.tall.intersect(b))
            - area(c.wide.intersect(b).intersect(c.tall.intersect(b)));
        let total: f64 = strips.iter().map(|r| area(*r)).sum();
        assert_eq!(total, area(b) - covered);
    }
}
