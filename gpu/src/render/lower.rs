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
    FLAG_HAS_CLIP, FLAG_HAS_INNER, FLAG_HAS_MASK, FLAG_MASK_TEXTURE, Globals, Instance, KIND_FILL,
    KIND_GLYPH, KIND_SHADOW, KIND_SPAN, KIND_STROKE_DIST, KIND_STROKE_OFFSET, PAINT_SOLID,
    PAINT_TEXTURE, Shape, Stop, affine, blend_code,
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

/// A texture identity scoped to its resource owner.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ImageSource {
    Registered(u64),
    Content(LayerId),
    Shader(std::sync::Arc<super::paint::Key>),
}

/// One draw call's instance range and bound source texture.
#[derive(Clone, Debug)]
pub struct DrawRange {
    /// The scratch texture bound as group 1, `None` for the dummy texture.
    pub source: Option<usize>,
    /// The image texture bound for `PAINT_IMAGE` instances.
    pub image: Option<ImageSource>,
    /// The mask texture bound at group-1 binding 3, by key.
    pub mask: Option<u64>,
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
    pub content: Vec<LayerId>,
    pub filters: Vec<(usize, u64)>,
    open: Option<OpenPass>,
}

#[derive(Clone)]
struct OpenPass {
    target: Target,
    clear: Option<[f32; 4]>,
    source: Option<usize>,
    image: Option<ImageSource>,
    /// The mask texture bound at group-1 binding 3, by key.
    mask: Option<u64>,
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
    filters: usize,
    open: Option<OpenPass>,
}

impl Frame {
    /// Captures the frame's emission state.
    fn snapshot(&self) -> FrameSnapshot {
        FrameSnapshot {
            instances: self.instances.len(),
            stops: self.stops.len(),
            passes: self.passes.len(),
            filters: self.filters.len(),
            open: self.open.clone(),
        }
    }

    /// Reverts every emission since `snap`.
    fn restore(&mut self, snap: FrameSnapshot) {
        self.instances.truncate(snap.instances);
        self.stops.truncate(snap.stops);
        self.passes.truncate(snap.passes);
        self.filters.truncate(snap.filters);
        self.open = snap.open;
    }
}

impl Frame {
    /// Reuses this frame's allocations for the next lowering.
    pub fn reset(&mut self) {
        self.instances.clear();
        self.stops.clear();
        self.passes.clear();
        self.content.clear();
        self.filters.clear();
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

/// A path-clip mask: stored in the atlas or on its own texture, or
/// produced by a pending raster the render thread will store — then
/// `uv.zw` of every instance emitted under it is patched by
/// `Lowering::mask_patches`.
#[derive(Clone, Copy, Debug)]
enum ClipMask {
    /// Stored in the atlas: the atlas origin is `cell.atlas`.
    Cell(MaskCell),
    /// Stored on a dedicated texture: the mask's content-hash key,
    /// bound at group-1 binding 3.
    Texture(MaskCell, u64),
    /// Pending its raster (index into `Lowering::pending`).
    Pending(MaskCell, u32),
}

impl ClipMask {
    /// The mask data: device rect, size, and the pending-or-stored atlas
    /// origin.
    const fn cell(self) -> MaskCell {
        match self {
            Self::Cell(m) | Self::Texture(m, _) | Self::Pending(m, _) => m,
        }
    }

    /// This mask shifted by `(dx, dy)` device pixels.
    fn translated(self, dx: f64, dy: f64) -> Self {
        match self {
            Self::Cell(m) => Self::Cell(m.translated(dx, dy)),
            Self::Texture(m, key) => Self::Texture(m.translated(dx, dy), key),
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
pub struct ContentData {
    pub(crate) retained: cherenkov::lowering::Content<Op, Emission>,
    pub(crate) storage: EmissionStorage,
}

impl ContentData {
    pub fn new(list: cherenkov::Picture) -> Self {
        Self {
            retained: cherenkov::lowering::Content::new(list),
            storage: EmissionStorage::default(),
        }
    }

    pub fn replace(&mut self, list: cherenkov::Picture) {
        self.retained.replace(list);
        self.storage.instances.clear();
        self.storage.stops.clear();
        self.storage.templates.clear();
        self.storage.covers.clear();
    }

    pub fn picture(list: cherenkov::Picture) -> Self {
        Self {
            retained: cherenkov::lowering::Content::picture(list),
            storage: EmissionStorage::default(),
        }
    }

    pub fn update(&mut self, updates: Vec<cherenkov::SlotUpdate>) {
        self.retained.update(updates);
    }

    pub fn invalidate(&mut self) {
        self.retained.invalidate();
        self.storage = EmissionStorage::default();
    }

    pub fn trim(&mut self) {
        self.retained.trim();
        self.storage = EmissionStorage::default();
    }
}

/// A layer's device data. Leaf ranges remain independent for dirty updates.
#[derive(Default)]
pub struct EmissionStorage {
    templates: Vec<InstanceTemplate>,
    covers: Vec<Cover>,
    pub(crate) instances: Vec<RetainedInstance>,
    stops: Vec<Stop>,
}

impl EmissionStorage {
    /// Reclaim obsolete ranges after patches without rebuilding valid leaves.
    fn compact(&mut self, emissions: &mut [Realization<Emission>]) {
        if self.templates.is_empty() {
            return;
        }
        let (instances, stops, templates, covers) = emissions
            .iter()
            .filter_map(|e| e.data.as_ref())
            .fold((0, 0, 0, 0), |(i, s, t, c), e| {
                (
                    i + e.instances.len(),
                    s + e.stops.len(),
                    t + 1,
                    c + usize::from(e.cover.is_some()),
                )
            });
        if self.instances.len() <= instances * 2
            && self.stops.len() <= stops * 2
            && self.templates.len() <= templates * 2
            && self.covers.len() <= covers * 2
        {
            return;
        }
        let mut storage = Self {
            templates: Vec::with_capacity(templates),
            covers: Vec::with_capacity(covers),
            instances: Vec::with_capacity(instances),
            stops: Vec::with_capacity(stops),
        };
        for e in emissions.iter_mut().filter_map(|e| e.data.as_mut()) {
            storage.templates.push(self.templates[e.template]);
            e.template = storage.templates.len() - 1;
            if let Some(cover) = &mut e.cover {
                storage.covers.push(self.covers[*cover]);
                *cover = storage.covers.len() - 1;
            }
            let first = storage.instances.len();
            storage
                .instances
                .extend_from_slice(&self.instances[e.instances.clone()]);
            e.instances = first..storage.instances.len();
            let first = storage.stops.len();
            storage
                .stops
                .extend_from_slice(&self.stops[e.stops.clone()]);
            e.stops = first..storage.stops.len();
        }
        *self = storage;
    }
}

/// The non-varying fields of an unclipped leaf. Clip fields, affine padding,
/// mask coordinates and per-quad fields are reconstructed when composed.
#[derive(Clone, Copy)]
struct InstanceTemplate {
    affine: [f32; 6],
    shape: Shape,
    inner: Shape,
    color: [f32; 4],
    grad: [f32; 4],
    grad2: [f32; 4],
    params: [f32; 2],
    paint: u32,
    packed: u32,
}

impl InstanceTemplate {
    fn new(inst: &Instance) -> Self {
        Self {
            affine: inst.affine[..6]
                .try_into()
                .expect("six affine coefficients"),
            shape: inst.shape,
            inner: inst.inner,
            color: inst.color,
            grad: inst.grad,
            grad2: inst.grad2,
            params: [inst.params[0], inst.params[1]],
            paint: inst.meta[1],
            packed: inst.meta[3],
        }
    }

    fn restore(&self) -> Instance {
        let mut inst = Instance::new(0);
        inst.affine[..6].copy_from_slice(&self.affine);
        inst.shape = self.shape;
        inst.inner = self.inner;
        inst.color = self.color;
        inst.grad = self.grad;
        inst.grad2 = self.grad2;
        inst.params[..2].copy_from_slice(&self.params);
        inst.meta[1] = self.paint;
        inst.meta[3] = self.packed;
        inst
    }
}

/// Fields that vary within one realized leaf. Shape, placement and paint are
/// shared by all its quads, including a box's interior/border split.
#[derive(Clone, Copy)]
pub struct RetainedInstance {
    bounds: [f32; 4],
    pub(crate) uv: [f32; 2],
    kind: u32,
    first_stop: u32,
}

impl RetainedInstance {
    fn restore(self, template: &InstanceTemplate, stop_base: u32) -> Instance {
        let mut inst = template.restore();
        inst.bounds = self.bounds;
        inst.uv[..2].copy_from_slice(&self.uv);
        inst.meta[0] = self.kind;
        inst.meta[2] = self.first_stop + stop_base;
        inst
    }
}

/// Per-operation device output under its sampled placement.
pub struct Emission {
    pub(crate) pending_cells: Vec<(u32, u32, u32)>,
    template: usize,
    // Only shadows carry an occlusion key; ordinary leaves keep a small index.
    cover: Option<usize>,
    transform: Affine,
    size: [f32; 2],
    generation: u64,
    pub(crate) instances: Range<usize>,
    stops: Range<usize>,
    image: Option<ImageSource>,
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
    pub content: &'a HashMap<LayerId, super::gpu_content::Slot>,
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
    // The margin depends only on the transform's linear coefficients. Keep
    // their exact bits so signed zero and non-finite inputs retain semantics.
    margin: Option<([u64; 4], f64)>,
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
    /// The open range's bound mask texture key; `None` for the dummy view.
    /// Kept in sync with `clip` by `set_clip`.
    mask_key: Option<u64>,
    /// The current clip's pending mask raster index, if any.
    mask_pending: Option<u32>,
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
            margin: None,
            clip: None,
            mask_key: None,
            mask_pending: None,
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

    /// Reuse only the scalar margin, never a command's device realization.
    fn margin(&mut self, transform: Affine) -> f64 {
        let [a, b, c, d, _, _] = transform.as_coeffs();
        let linear = [a, b, c, d].map(f64::to_bits);
        if let Some((cached, value)) = self.margin
            && cached == linear
        {
            return value;
        }
        let value = aa_margin(transform);
        self.margin = Some((linear, value));
        value
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
            self.commands_lowered += content.retained.prepare(&mut super::prepared::Lowerer {
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
        self.end_segment_at(self.frame.instances.len());
    }

    #[expect(
        clippy::inline_always,
        reason = "keep current-length calls as cheap as the original unsplit hot path"
    )]
    #[inline(always)]
    fn end_segment_at(&mut self, end: usize) {
        let Some(open) = &mut self.frame.open else {
            return;
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "instance counts fit u32 in practice"
        )]
        let end = end as u32;
        if end > open.seg_start {
            open.ranges.push(DrawRange {
                source: open.source,
                image: open.image.clone(),
                mask: open.mask,
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
            mask: None,
            pipeline: PipelineKind::SrcOver,
            variant: ShaderVariant::Simple,
            backdrop_copy: None,
            ranges: Vec::new(),
            #[expect(clippy::cast_possible_truncation)]
            seg_start: self.frame.instances.len() as u32,
        });
        // A masked clip can span passes; the new pass binds its texture.
        self.set_mask(self.mask_key);
    }

    fn finish_pass(&mut self) {
        self.finish_pass_at(self.frame.instances.len());
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "surface size is a small positive float"
    )]
    #[expect(
        clippy::inline_always,
        reason = "keep current-length calls as cheap as the original unsplit hot path"
    )]
    #[inline(always)]
    fn finish_pass_at(&mut self, end: usize) {
        self.end_segment_at(end);
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
    #[expect(
        clippy::inline_always,
        reason = "keep ordinary image identity changes as cheap as the original Copy path"
    )]
    #[inline(always)]
    fn set_image(&mut self, image: Option<ImageSource>) {
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

    /// Sets the current clip and the open range's bound mask texture.
    /// `mask_key`/`mask_pending` derive from the clip, so the per-instance
    /// path only reads scalars.
    fn set_clip(&mut self, clip: Option<DeviceClip>) {
        self.clip = clip;
        let mask = clip.and_then(|c| c.mask);
        self.mask_key = match mask {
            Some(ClipMask::Texture(_, key)) => Some(key),
            _ => None,
        };
        self.mask_pending = match mask {
            Some(ClipMask::Pending(_, p)) => Some(p),
            _ => None,
        };
        self.set_mask(self.mask_key);
    }

    /// Starts a new draw range when the bound mask texture changes.
    #[expect(
        clippy::inline_always,
        reason = "a compare on the hot push_instance path"
    )]
    #[inline(always)]
    fn set_mask(&mut self, mask: Option<u64>) {
        if self.frame.open.as_ref().is_some_and(|o| o.mask != mask) {
            self.end_segment();
            if let Some(open) = &mut self.frame.open {
                open.mask = mask;
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
        if let Some(pending) = self.mask_pending {
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
    /// composite. An unclipped overlapping batch can become the scratch pass
    /// directly; nested or clipped batches are rolled back and isolated again.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "surface size is a small positive float"
    )]
    // Keep isolation's speculative buffers off the ordinary drawing walk's stack.
    #[inline(never)]
    fn isolate(
        &mut self,
        inner_clip: Option<DeviceClip>,
        filter: Option<cherenkov::FilterId>,
        opacity: f32,
        blend: cherenkov::BlendMode,
        mut body: impl FnMut(&mut Self, &GlyphContext<'_>) -> Result<(), RenderError>,
        glyphs: &GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        if filter.is_none()
            && opacity < 1.0
            && blend == cherenkov::BlendMode::Normal
            && self.try_passthrough(opacity, inner_clip.is_none(), &mut body, glyphs)?
        {
            return Ok(());
        }
        self.depth += 1;
        let scratch = self.depth - 1;
        let outer_clip = self.clip;
        self.set_clip(inner_clip);
        // Nested isolations split this scratch's open pass into segments;
        // every segment at this depth needs the region.
        let passes_start = self.frame.passes.len();
        self.begin_pass(Target::Scratch(scratch), Some([0.0; 4]));
        let inst_start = self.frame.instances.len();
        body(self, glyphs)?;
        self.finish_pass();
        self.depth -= 1;
        self.set_clip(outer_clip);
        let outer_target = if self.depth == 0 {
            Target::Surface
        } else {
            Target::Scratch(self.depth - 1)
        };
        let region = if let Some(filter) = filter {
            self.frame
                .filters
                .push((self.frame.passes.len() - 1, filter.raw()));
            [0, 0, self.width as u32, self.height as u32]
        } else if is_destructive(blend) {
            clip_region(
                inner_clip.or(outer_clip),
                self.width as u32,
                self.height as u32,
            )
        } else {
            tight_region(
                &self.frame.instances[inst_start..],
                self.width as u32,
                self.height as u32,
            )
        };
        if region[2] == 0 || region[3] == 0 {
            // Nothing visible in the scratch: drop this depth's segment
            // passes and the composite entirely.
            for i in (passes_start..self.frame.passes.len()).rev() {
                if self.frame.passes[i].target == Target::Scratch(scratch) {
                    self.frame.passes.remove(i);
                    self.frame.filters.retain_mut(|(pass, _)| {
                        if *pass == i {
                            return false;
                        }
                        if *pass > i {
                            *pass -= 1;
                        }
                        true
                    });
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
        // sampling it with `grad.xy` as the texel origin. A destructive
        // composite carries the effective clip itself: its coverage decides
        // where the operator applies across the whole region.
        if is_destructive(blend) {
            self.set_clip(inner_clip.or(outer_clip));
        }
        self.emit_composite(scratch, opacity, region, blend);
        if is_destructive(blend) {
            self.set_clip(outer_clip);
        }
        Ok(())
    }

    /// Speculative pass-through for [`Lowering::isolate`]. Returns `Ok(true)`
    /// when `body` stayed in one pass (folding opacity for disjoint bounds,
    /// or promoting an unclipped overlapping batch into a scratch pass).
    /// Otherwise rolls back and returns `Ok(false)` for nested/clipped isolation.
    fn try_passthrough(
        &mut self,
        opacity: f32,
        inner_unclipped: bool,
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
        if result.is_ok()
            && inner_unclipped
            && clip.is_none()
            && self.frame.passes.len() == snap.passes
            && self.frame.open.is_some()
        {
            self.promote_isolation(snap, opacity);
            return Ok(true);
        }
        self.frame.restore(snap);
        self.cell_patches.truncate(patches.0);
        self.mask_patches.truncate(patches.1);
        self.depth = depth;
        self.set_clip(clip);
        self.transform = transform;
        result?;
        Ok(false)
    }

    /// Reuse an overlapping speculative batch as the isolated pass. The body
    /// stayed in one pass and used the same (empty) inner and outer clip, so
    /// its instances, stops and pending atlas patches already are final.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "surface dimensions and instance indices fit u32"
    )]
    fn promote_isolation(&mut self, snap: FrameSnapshot, opacity: f32) {
        self.end_segment();
        let mut open = self
            .frame
            .open
            .take()
            .expect("speculation has an open pass");
        let first = snap.instances as u32;
        open.ranges.retain_mut(|range| {
            range.instances.start = range.instances.start.max(first);
            range.instances.start < range.instances.end
        });
        let outer_target = open.target;
        self.frame.open = snap.open;
        self.finish_pass_at(snap.instances);
        let region = tight_region(
            &self.frame.instances[snap.instances..],
            self.width as u32,
            self.height as u32,
        );
        if region[2] != 0 && region[3] != 0 {
            self.frame.passes.push(Pass {
                target: Target::Scratch(self.depth),
                clear: Some([0.0; 4]),
                ranges: open.ranges,
                region,
                backdrop_copy: None,
            });
        }
        self.begin_pass(outer_target, None);
        if region[2] != 0 && region[3] != 0 {
            self.emit_composite(self.depth, opacity, region, cherenkov::BlendMode::Normal);
        }
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
                match mask {
                    ClipMask::Cell(cell) => {
                        inst.uv[2] = cell.atlas[0];
                        inst.uv[3] = cell.atlas[1];
                    }
                    ClipMask::Texture(..) => {
                        inst.meta[3] |= FLAG_MASK_TEXTURE << 24;
                    }
                    ClipMask::Pending(..) => {}
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
                self.set_clip(Some(clip));
                body(self, glyphs)?;
                self.set_clip(None);
                Ok(())
            }
            // The current clip is masked: only an aligned rect merges
            // (keeping the mask); anything else isolates.
            Some(cur) if cur.mask.is_some() => {
                if clip.mask.is_none()
                    && let (Some(cr), Some(dr)) = (cur.aligned_rect, clip.aligned_rect)
                {
                    self.set_clip(Some(merged_rect(cr, dr, cur.mask)));
                    body(self, glyphs)?;
                    self.set_clip(Some(cur));
                    return Ok(());
                }
                self.isolate(
                    Some(clip),
                    None,
                    1.0,
                    cherenkov::BlendMode::Normal,
                    body,
                    glyphs,
                )
            }
            Some(cur) => match (clip.mask, cur.aligned_rect, clip.aligned_rect) {
                // A new masked clip merges with an aligned rect clip (or
                // attaches to the current clip's analytic shape).
                (Some(mask), Some(cr), Some(dr)) => {
                    self.set_clip(Some(merged_rect(cr, dr, Some(mask))));
                    body(self, glyphs)?;
                    self.set_clip(Some(cur));
                    Ok(())
                }
                (Some(mask), _, _) => {
                    self.set_clip(Some(DeviceClip {
                        inv: cur.inv,
                        shape: cur.shape,
                        aligned_rect: cur.aligned_rect,
                        mask: Some(mask),
                    }));
                    body(self, glyphs)?;
                    self.set_clip(Some(cur));
                    Ok(())
                }
                (None, Some(cr), Some(dr)) => {
                    self.set_clip(Some(merged_rect(cr, dr, None)));
                    body(self, glyphs)?;
                    self.set_clip(Some(cur));
                    Ok(())
                }
                _ => self.isolate(
                    Some(clip),
                    None,
                    1.0,
                    cherenkov::BlendMode::Normal,
                    body,
                    glyphs,
                ),
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
                if node.filter.is_some()
                    || node.opacity < 1.0
                    || node.blend != cherenkov::BlendMode::Normal
                {
                    let inner = s.clip;
                    s.isolate(
                        inner,
                        node.filter,
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
            let (ops, emissions, source) = content.retained.prepared_source();
            content.storage.compact(emissions);
            let changed = self.ops(
                source,
                ops,
                emissions,
                &mut content.storage,
                0..ops.len(),
                glyphs,
            )?;
            self.layers_composed += u32::from(changed);
        }
        if let Some(slot) = glyphs.content.get(&id) {
            self.frame.content.push(id);
            let bounds = Rect::new(0.0, 0.0, f64::from(slot.size.0), f64::from(slot.size.1));
            if let Some(boxed) = box_shape(&ShapeData::Rect(bounds))? {
                let transform = self.transform * boxed.extra;
                let mut inst = self.base(KIND_FILL, affine(transform));
                let margin = self.margin(self.transform);
                let b = boxed.bounds.inflate(margin, margin);
                inst.bounds = [f32_f64(b.x0), f32_f64(b.y0), f32_f64(b.x1), f32_f64(b.y1)];
                inst.shape = boxed.shape;
                inst.meta[1] = super::instance::PAINT_IMAGE;
                inst.grad = [1.0, 0.0, 0.0, 1.0];
                inst.grad2 = [
                    f32_f64(bounds.width() / 2.0),
                    f32_f64(bounds.height() / 2.0),
                    f32_f64(bounds.width()),
                    f32_f64(bounds.height()),
                ];
                inst.meta[3] |=
                    super::instance::EXTEND_PAD | (super::instance::EXTEND_PAD << 4) | (1 << 8);
                self.set_image(Some(ImageSource::Content(id)));
                self.push_shaped(inst, transform, boxed.bounds, margin);
            }
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
        source: &cherenkov::DisplayList,
        ops: &[Op],
        emissions: &mut [Realization<Emission>],
        storage: &mut EmissionStorage,
        range: Range<usize>,
        glyphs: &GlyphContext<'_>,
    ) -> Result<bool, RenderError> {
        let mut changed = false;
        let mut i = range.start;
        while i < range.end {
            match &ops[i] {
                Op::BeginClip { local, shape, end } => {
                    let saved = self.transform;
                    self.transform = saved * *local;
                    let body = |s: &mut Self, g: &GlyphContext<'_>| {
                        s.transform = saved;
                        changed |=
                            s.ops(source, ops, emissions, storage, i + 1..*end as usize, g)?;
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
                    filter,
                    end,
                } => {
                    self.isolate(
                        None,
                        *filter,
                        *opacity,
                        *blend,
                        |s, g| {
                            changed |=
                                s.ops(source, ops, emissions, storage, i + 1..*end as usize, g)?;
                            Ok(())
                        },
                        glyphs,
                    )?;
                    i = *end as usize;
                }
                Op::End => unreachable!("paired scopes consume their ends"),
                op => {
                    changed |= self.leaf(
                        op,
                        ops.get(i + 1),
                        &mut emissions[i],
                        storage,
                        glyphs,
                        source,
                    )?;
                }
            }
            i += 1;
        }
        Ok(changed)
    }

    /// Retain each leaf's instances and gradient stops independently. A dirty
    /// command drops just its entries; atlas resets invalidate device addresses.
    ///
    /// A miss realizes the leaf straight into this frame, in the form the
    /// retained copy is defined in — unclipped, from an unbound image — and
    /// copies the emitted range into the cache, so each instance is built
    /// once. Unclipped, the realized range already is the composed output;
    /// under a clip the range is rolled back and the retained copy is
    /// composed under it like a hit, because draw ranges segment on the
    /// clipped instances' shader variant.
    fn leaf(
        &mut self,
        op: &Op,
        next: Option<&Op>,
        cache: &mut Realization<Emission>,
        storage: &mut EmissionStorage,
        glyphs: &GlyphContext<'_>,
        source: &cherenkov::DisplayList,
    ) -> Result<bool, RenderError> {
        let cover = self.shadow_cover(op, next);
        let hit = cache.valid
            && cache.data.as_ref().is_some_and(|e| {
                e.cover.map(|index| storage.covers[index]) == cover
                    && e.transform == self.transform
                    && e.size.map(f32::to_bits) == [self.width, self.height].map(f32::to_bits)
                    && e.generation == glyphs.atlas.generation()
            });
        cache.valid = true;
        if hit {
            self.compose(
                cache.data.as_ref().expect("a hit has data"),
                storage,
                self.clip,
            );
            return Ok(false);
        }
        if self.clip.is_some() {
            self.realize_clipped_leaf(op, cover, cache, storage, glyphs, source)?;
        } else {
            self.realize_leaf(op, cover, cache, storage, glyphs, source)?;
        }
        Ok(true)
    }

    /// Clipped misses need a rollback because clipping can change draw variants.
    /// Keep their large clip/snapshot values off the ordinary miss path.
    fn realize_clipped_leaf(
        &mut self,
        op: &Op,
        cover: Option<Cover>,
        cache: &mut Realization<Emission>,
        storage: &mut EmissionStorage,
        glyphs: &GlyphContext<'_>,
        source: &cherenkov::DisplayList,
    ) -> Result<(), RenderError> {
        self.set_image(None);
        let snapshot = self.frame.snapshot();
        let first_patch = self.cell_patches.len();
        let clip = self.clip;
        self.set_clip(None);
        let result = self.realize_leaf(op, cover, cache, storage, glyphs, source);
        self.set_clip(clip);
        result?;
        self.frame.restore(snapshot);
        self.cell_patches.truncate(first_patch);
        self.compose(
            cache.data.as_ref().expect("realized leaf has data"),
            storage,
            clip,
        );
        Ok(())
    }

    fn realize_leaf(
        &mut self,
        op: &Op,
        cover: Option<Cover>,
        cache: &mut Realization<Emission>,
        storage: &mut EmissionStorage,
        glyphs: &GlyphContext<'_>,
        source: &cherenkov::DisplayList,
    ) -> Result<(), RenderError> {
        debug_assert!(self.clip.is_none());
        self.set_image(None);
        let first_instance = self.frame.instances.len();
        let first_stop = self.frame.stops.len();
        let first_patch = self.cell_patches.len();
        // `realize` composes path and glyph ops' local transforms into
        // `self.transform`; the leaf's placement is restored with the clip.
        let transform = self.transform;
        let realized = self.realize(op, cover, glyphs, source);
        self.transform = transform;
        realized?;
        let stop_base = u32::try_from(first_stop).expect("stop count fits u32");
        let instance_base = u32::try_from(first_instance).expect("instance count fits u32");
        let retained_instance = storage.instances.len();
        let retained_stop = storage.stops.len();
        let template = storage.templates.len();
        storage.templates.push(InstanceTemplate::new(
            self.frame
                .instances
                .get(first_instance)
                .unwrap_or(&Instance::new(0)),
        ));
        storage
            .instances
            .extend(self.frame.instances[first_instance..].iter().map(|inst| {
                let retained = RetainedInstance {
                    bounds: inst.bounds,
                    uv: [inst.uv[0], inst.uv[1]],
                    kind: inst.meta[0],
                    first_stop: inst.meta[2].saturating_sub(stop_base),
                };
                // Keep this invariant checked as new leaf emitters are added. Stop
                // indices without gradients are unused but preserve their input too.
                let restored = retained.restore(
                    &storage.templates[template],
                    inst.meta[2] - retained.first_stop,
                );
                debug_assert_eq!(bytemuck::bytes_of(&restored), bytemuck::bytes_of(inst));
                retained
            }));
        storage
            .stops
            .extend_from_slice(&self.frame.stops[first_stop..]);
        cache.data = Some(Emission {
            pending_cells: self.cell_patches[first_patch..]
                .iter()
                .map(|&(i, p, c)| (i - instance_base, p, c))
                .collect(),
            template,
            cover: cover.map(|cover| {
                storage.covers.push(cover);
                storage.covers.len() - 1
            }),
            transform,
            size: [self.width, self.height],
            generation: glyphs.atlas.generation(),
            instances: retained_instance..storage.instances.len(),
            stops: retained_stop..storage.stops.len(),
            image: self
                .frame
                .open
                .as_ref()
                .expect("a leaf lowers into an open pass")
                .image
                .clone(),
        });
        Ok(())
    }

    /// Emits a retained leaf's instances and stops into this frame under
    /// `clip`, rebasing its stop and pending-cell indices.
    fn compose(
        &mut self,
        emission: &Emission,
        storage: &EmissionStorage,
        clip: Option<DeviceClip>,
    ) {
        let offset = u32::try_from(self.frame.stops.len()).expect("stop count fits u32");
        self.frame
            .stops
            .extend_from_slice(&storage.stops[emission.stops.clone()]);
        self.set_image(emission.image.clone());
        let instance_base =
            u32::try_from(self.frame.instances.len()).expect("instance count fits u32");
        for inst in &storage.instances[emission.instances.clone()] {
            let mut inst = inst.restore(&storage.templates[emission.template], offset);
            Self::apply_clip(&mut inst, clip);
            self.push_instance(&inst);
        }
        self.cell_patches.extend(
            emission
                .pending_cells
                .iter()
                .map(|&(i, p, c)| (i + instance_base, p, c)),
        );
    }

    #[expect(
        clippy::inline_always,
        reason = "let leaf emitters eliminate unused paint fields instead of copying the full payload"
    )]
    #[inline(always)]
    fn resolved_paint(&mut self, paint: &ResolvedPaint) -> PaintData {
        let offset = u32::try_from(self.frame.stops.len()).expect("stop count fits u32");
        let data = match paint {
            ResolvedPaint::Shader(_) => unreachable!("shader paint needs draw bounds"),
            ResolvedPaint::Solid(color) => PaintData {
                kind: PAINT_SOLID,
                color: *color,
                first_stop: offset,
                ..PaintData::default()
            },
            ResolvedPaint::Resources(resources) => {
                let (data, stops) = resources.as_ref();
                let mut data = *data;
                data.first_stop += offset;
                self.frame.stops.extend_from_slice(stops);
                data
            }
        };
        self.set_image(data.image.map(ImageSource::Registered));
        data
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "bounded texture extents"
    )]
    #[cold]
    #[inline(never)]
    fn shader_paint(
        &mut self,
        paint: &super::prepared::ResolvedShader,
        bounds: Rect,
        transform: Affine,
    ) -> Result<PaintData, RenderError> {
        let device = transform.transform_rect_bbox(bounds);
        if ![
            device.width(),
            device.height(),
            bounds.width(),
            bounds.height(),
        ]
        .iter()
        .all(|v| v.is_finite())
        {
            return Err(RenderError::Render(
                "shader paint requires finite bounds".into(),
            ));
        }
        let size = (
            device.width().ceil().max(1.0) as u32,
            device.height().ceil().max(1.0) as u32,
        );
        let key = std::sync::Arc::new(super::paint::Key {
            shader: paint.source.shader.raw(),
            uniforms: paint.source.uniforms.iter().map(|v| v.to_bits()).collect(),
            size,
        });
        // Collapsed axes sample their center. Coverage still comes from the
        // ordinary shape/path lowering, including zero-area geometry.
        let x = if bounds.width() == 0.0 {
            0.0
        } else {
            f64::from(size.0) / bounds.width()
        };
        let y = if bounds.height() == 0.0 {
            0.0
        } else {
            f64::from(size.1) / bounds.height()
        };
        let offset_x = if bounds.width() == 0.0 {
            0.5
        } else {
            -bounds.x0 * x
        };
        let offset_y = if bounds.height() == 0.0 {
            0.5
        } else {
            -bounds.y0 * y
        };
        let [xx, yx, xy, yy, tx, ty] =
            (Affine::new([x, 0.0, 0.0, y, offset_x, offset_y]) * paint.sampling).as_coeffs();
        let data = PaintData {
            kind: super::instance::PAINT_IMAGE,
            grad: [f32_f64(xx), f32_f64(yx), f32_f64(xy), f32_f64(yy)],
            grad2: [
                f32_f64(tx),
                f32_f64(ty),
                f32_f64(f64::from(size.0)),
                f32_f64(f64::from(size.1)),
            ],
            packed: super::instance::EXTEND_PAD | (super::instance::EXTEND_PAD << 4) | (1 << 8),
            ..PaintData::default()
        };
        self.set_image(Some(ImageSource::Shader(key)));
        Ok(data)
    }

    #[inline]
    fn shaped_paint(
        &mut self,
        paint: &ResolvedPaint,
        bounds: Rect,
        local: Affine,
        extra_margin: f64,
    ) -> Result<PaintData, RenderError> {
        if let ResolvedPaint::Shader(shader) = paint {
            self.shader_paint(
                shader,
                bounds.inflate(extra_margin, extra_margin),
                self.transform * local,
            )
        } else {
            Ok(self.resolved_paint(paint))
        }
    }

    #[expect(
        clippy::inline_always,
        reason = "preserve dev inlining of common leaf realization as capabilities grow"
    )]
    #[inline(always)]
    fn realize(
        &mut self,
        op: &Op,
        cover: Option<Cover>,
        glyphs: &GlyphContext<'_>,
        source: &cherenkov::DisplayList,
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
                let margin = extra_margin + self.margin(self.transform * *ambient);
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
                let paint = self.shaped_paint(paint, *bounds, *local, *extra_margin)?;
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
                let margin = sigma_eff.mul_add(3.0, 1.0) + self.margin(self.transform * *ambient);
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
                rule,
                outline,
                paint,
            } => {
                self.transform *= *local;
                match outline {
                    Outline::Fill { elements, content } => self.path(
                        *content,
                        *rule,
                        || BezPath::from_vec(elements.to_vec()),
                        paint,
                        glyphs,
                    )?,
                    Outline::Stroke { shape, stroke } => {
                        self.stroke_path(shape, stroke, *rule, paint, glyphs)?;
                    }
                    Outline::Source { command, content } => match &source.commands()[*command] {
                        cherenkov::Command::Fill {
                            shape: ShapeData::Path { elements, .. },
                            ..
                        } => self.path(
                            content.expect("prepared fill has a hash"),
                            *rule,
                            || BezPath::from_vec(elements.clone()),
                            paint,
                            glyphs,
                        )?,
                        cherenkov::Command::Stroke { shape, stroke, .. } => {
                            self.stroke_path(shape, stroke, *rule, paint, glyphs)?;
                        }
                        _ => unreachable!("prepared outline keeps its source kind"),
                    },
                }
            }
            Op::Glyphs { local, run, paint } => {
                self.transform *= *local;
                self.glyph_run(run.get(source), paint, glyphs)?;
            }
            _ => unreachable!("scope is composed, never realized as a leaf"),
        }
        Ok(())
    }

    fn stroke_path(
        &mut self,
        shape: &ShapeData,
        stroke: &cherenkov::kurbo::Stroke,
        rule: FillRule,
        paint: &ResolvedPaint,
        glyphs: &GlyphContext<'_>,
    ) -> Result<(), RenderError> {
        let tol = path::FLATTEN / path::sigma_max(self.transform).max(1e-12);
        self.path(
            path::hash_stroke(shape, stroke, tol),
            rule,
            || {
                let path = path::shape_path(shape, tol);
                kurbo::stroke(path, stroke, &kurbo::StrokeOpts::default(), tol)
            },
            paint,
            glyphs,
        )
    }

    /// A following opaque box hides a rectangle inside each pair of its corner rows.
    #[expect(
        clippy::float_cmp,
        reason = "occlusion requires exact opacity and matching axes"
    )]
    fn shadow_cover(&mut self, op: &Op, next: Option<&Op>) -> Option<Cover> {
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
        if !matches!(paint, ResolvedPaint::Solid(color) if color[3] == 1.0) {
            return None;
        }
        let relative = local.inverse() * *fill;
        let [scale_x, skew_y, skew_x, scale_y, offset_x, offset_y] = relative.as_coeffs();
        if [scale_x, skew_y, skew_x, scale_y] != [1.0, 0.0, 0.0, 1.0] {
            return None;
        }
        let max_r = f64::from(shape.radii.iter().copied().fold(0.0, f32::max));
        let m = self.margin(self.transform * *ambient) + 1.0;
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

    /// Bounds-dependent shader preparation must not expand ordinary path replay.
    #[cold]
    #[inline(never)]
    fn shader_outline(
        &mut self,
        make: impl FnOnce() -> BezPath,
        paint: &super::prepared::ResolvedShader,
    ) -> Result<(BezPath, PaintData), RenderError> {
        let path = make();
        let data = self.shader_paint(paint, kurbo::Shape::bounding_box(&path), self.transform)?;
        Ok((path, data))
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
        let mut make = Some(make);
        let (shader_path, shader_data) = if let ResolvedPaint::Shader(shader) = paint {
            let (path, data) = self.shader_outline(make.take().expect("path factory"), shader)?;
            (Some(path), Some(data))
        } else {
            (None, None)
        };
        if let Some(emit) = glyphs
            .atlas
            .path(pl.key)
            .or_else(|| glyphs.atlas.path(pl.key_exact()))
        {
            self.replay(emit, None, pl.offset, paint, shader_data.as_ref());
            return Ok(());
        }
        let (stored, pending) = 'stored: {
            let device =
                pl.raster * shader_path.unwrap_or_else(|| make.take().expect("path factory")());
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
                    key: pl.key_exact(),
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
                pl.key_exact()
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
        self.replay(&stored, pending, pl.offset, paint, shader_data.as_ref());
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
        shader_data: Option<&PaintData>,
    ) {
        let paint = shader_data
            .copied()
            .unwrap_or_else(|| self.resolved_paint(paint));
        let mut template = self.base(KIND_SPAN, affine(self.transform));
        template.color = paint.color;
        template.grad = paint.grad;
        template.grad2 = paint.grad2;
        template.meta[1] = paint.kind;
        template.meta[2] = paint.first_stop;
        template.meta[3] |= paint.packed & 0x00ff_ffff;
        self.replay_quads(
            &template,
            emit.spans.iter().map(|rect| (*rect, [0.0; 2])),
            offset,
        );
        template.meta[0] = KIND_GLYPH;
        let first = self.frame.instances.len();
        self.replay_quads(
            &template,
            emit.cells
                .iter()
                .map(|cell| (cell.rect, [f32::from(cell.x), f32::from(cell.y)])),
            offset,
        );
        if let Some(pending) = pending {
            self.cell_patches.extend((0..emit.cells.len()).map(|i| {
                (
                    u32::try_from(first + i).expect("instance index fits u32"),
                    pending,
                    u32::try_from(i).expect("cell index fits u32"),
                )
            }));
        }
    }

    /// All quads in a path group share paint, clip and shader variant. Reserve
    /// and segment once, then write their instances directly into the frame.
    fn replay_quads(
        &mut self,
        template: &Instance,
        quads: impl ExactSizeIterator<Item = ([f32; 4], [f32; 2])>,
        offset: Vec2,
    ) {
        if quads.len() == 0 {
            return;
        }
        self.set_variant(variant_of(template));
        let first = self.frame.instances.len();
        self.frame.instances.resize(first + quads.len(), *template);
        for (inst, (rect, uv)) in self.frame.instances[first..].iter_mut().zip(quads) {
            inst.bounds = [
                f32_f64(f64::from(rect[0]) + offset.x),
                f32_f64(f64::from(rect[1]) + offset.y),
                f32_f64(f64::from(rect[2]) + offset.x),
                f32_f64(f64::from(rect[3]) + offset.y),
            ];
            inst.uv[..2].copy_from_slice(&uv);
        }
        if let Some(pending) = self.mask_pending {
            self.mask_patches.extend(
                (first..self.frame.instances.len())
                    .map(|i| (u32::try_from(i).expect("instance index fits u32"), pending)),
            );
        }
    }

    /// A clip with a `Path` shape: the coverage rasterized into an atlas
    /// cell — or, when it exceeds `Atlas::MASK_TEXTURE_TEXELS` or the
    /// atlas cap, a dedicated texture — multiplies every instance drawn
    /// under it. The mask is cached like a path draw: replays cost no
    /// rasterization or storage.
    #[expect(clippy::cast_possible_truncation)]
    #[expect(clippy::cast_sign_loss)]
    #[expect(clippy::cast_precision_loss)]
    #[expect(
        clippy::too_many_lines,
        reason = "atlas and texture storage share the lookup tail"
    )]
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
                .or_else(|| glyphs.atlas.mask(pl.key_exact()))
            {
                break 'stored ClipMask::Cell(*mask);
            }
            if let Some(mask) = glyphs
                .atlas
                .mask_texture(pl.key)
                .or_else(|| glyphs.atlas.mask_texture(pl.key_exact()))
            {
                let key = if glyphs.atlas.mask_texture(pl.key).is_some() {
                    pl.key
                } else {
                    pl.key_exact()
                };
                break 'stored ClipMask::Texture(*mask, key);
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
                pl.key_exact()
            } else {
                pl.key
            };
            if let Some(mask) = glyphs.atlas.mask(key) {
                break 'stored ClipMask::Cell(*mask);
            }
            if let Some(mask) = glyphs.atlas.mask_texture(key) {
                break 'stored ClipMask::Texture(*mask, key);
            }
            let mask = MaskCell {
                device: [
                    f32_f64(coverage.x - pl.offset.x),
                    f32_f64(coverage.y - pl.offset.y),
                ],
                // Filled by `Atlas::store_mask` on the render thread; a
                // texture mask's stays `[0, 0]`.
                atlas: [0.0, 0.0],
                size: [w as f32, h as f32],
                rect: [
                    f32_f64(coverage.x - pl.offset.x),
                    f32_f64(coverage.y - pl.offset.y),
                    f32_f64(coverage.x + f64::from(w) - pl.offset.x),
                    f32_f64(coverage.y + f64::from(h) - pl.offset.y),
                ],
            };
            if glyphs.atlas.mask_in_atlas(w, h) {
                let pending = u32::try_from(self.pending.len()).expect("pending count fits u32");
                self.pending.push(PendingRaster::Mask {
                    key,
                    mask,
                    w,
                    h,
                    texels,
                });
                ClipMask::Pending(mask, pending)
            } else {
                if !glyphs.atlas.mask_texture_fits(w, h) {
                    return Err(RenderError::Unsupported(names::PATH_CLIP_TOO_LARGE));
                }
                self.pending.push(PendingRaster::MaskTexture {
                    key,
                    mask,
                    w,
                    h,
                    texels,
                });
                ClipMask::Texture(mask, key)
            }
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

    #[cold]
    #[inline(never)]
    fn shader_glyph_run(
        &mut self,
        run: &GlyphRun,
        paint: &super::prepared::ResolvedShader,
        glyphs: &GlyphContext<'_>,
        font: &FontData,
    ) -> Result<(), RenderError> {
        let key = glyph_key(run, 0, (0.0, 0.0), self.transform);
        let mut entries = Vec::new();
        let mut bounds = None::<Rect>;
        for glyph in &run.glyphs {
            let origin = self.transform * Point::new(f64::from(glyph.x), f64::from(glyph.y));
            let x = origin.x.floor();
            let y = origin.y.floor();
            let fraction = (
                f32_f64(((origin.x - x) * 4.0).floor() / 4.0),
                f32_f64(((origin.y - y) * 4.0).floor() / 4.0),
            );
            let (entry, pending) = glyph::entry(
                glyphs.atlas,
                font,
                key.at(glyph.id, fraction),
                glyph.id,
                run.size,
                fraction,
                self.transform,
                &run.coords,
                &mut self.pending,
            )?;
            self.glyphs += u32::from(pending.is_some());
            if entry.w == 0 || entry.h == 0 {
                continue;
            }
            let rect = Rect::from_origin_size(
                (x + f64::from(entry.left), y + f64::from(entry.top)),
                (f64::from(entry.w), f64::from(entry.h)),
            );
            bounds = Some(bounds.map_or(rect, |bounds| bounds.union(rect)));
            entries.push((entry, pending, rect));
        }
        let Some(bounds) = bounds else {
            return Ok(());
        };
        let data = self.shader_paint(
            paint,
            self.transform.inverse().transform_rect_bbox(bounds),
            self.transform,
        )?;
        for (entry, pending, rect) in entries {
            let mut inst = self.base(KIND_GLYPH, affine(self.transform));
            inst.grad = data.grad;
            inst.grad2 = data.grad2;
            inst.meta[1] = data.kind;
            inst.meta[3] |= data.packed;
            inst.bounds = [
                f32_f64(rect.x0),
                f32_f64(rect.y0),
                f32_f64(rect.x1),
                f32_f64(rect.y1),
            ];
            inst.uv = [f32::from(entry.x), f32::from(entry.y), 0.0, 0.0];
            self.push_instance(&inst);
            if let Some(pending) = pending {
                self.cell_patches.push((
                    u32::try_from(self.frame.instances.len() - 1).expect("instance count fits u32"),
                    pending,
                    0,
                ));
            }
        }
        Ok(())
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
        if let ResolvedPaint::Shader(shader) = paint {
            return self.shader_glyph_run(run, shader, glyphs, font);
        }
        let key = glyph_key(run, 0, (0.0, 0.0), self.transform);
        let mut template = None;
        for glyph in &run.glyphs {
            let o = self.transform * Point::new(f64::from(glyph.x), f64::from(glyph.y));
            let ix = o.x.floor();
            let iy = o.y.floor();
            let fx = ((o.x - ix) * 4.0).floor() / 4.0;
            let fy = ((o.y - iy) * 4.0).floor() / 4.0;
            let key = key.at(glyph.id, (f32_f64(fx), f32_f64(fy)));
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
            let mut inst = *template.get_or_insert_with(|| {
                let mut inst = self.base(KIND_GLYPH, affine(self.transform));
                let data = self.resolved_paint(paint);
                inst.color = data.color;
                inst.grad = data.grad;
                inst.grad2 = data.grad2;
                inst.meta[1] = data.kind;
                inst.meta[2] = data.first_stop;
                inst.meta[3] |= data.packed & 0x00ff_ffff;
                inst
            });
            let x0 = f32_f64(ix + f64::from(entry.left));
            let y0 = f32_f64(iy + f64::from(entry.top));
            inst.bounds = [x0, y0, x0 + f32::from(entry.w), y0 + f32::from(entry.h)];
            inst.uv = [f32::from(entry.x), f32::from(entry.y), 0.0, 0.0];
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
    if instances.len() <= 256 {
        let mut boxes = Vec::with_capacity(instances.len());
        for inst in instances {
            let a = device_bbox(inst);
            if boxes.iter().any(|b| boxes_overlap(*b, a)) {
                return false;
            }
            boxes.push(a);
        }
        return true;
    }
    let mut boxes = Vec::with_capacity(instances.len());
    for inst in instances {
        let b = device_bbox(inst);
        // Splitting a draw into border strips can put intersecting boxes a
        // few instances apart. Check a bounded prefix pairwise before sorting;
        // after that, checking neighbors keeps the extra work linear.
        let recent = if boxes.len() < 32 {
            boxes.as_slice()
        } else {
            &boxes[boxes.len() - 1..]
        };
        if recent.iter().any(|a| boxes_overlap(*a, b)) {
            return false;
        }
        boxes.push(b);
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
    padded_region(min, max, width, height)
}

/// The `[x, y, w, h]` texel region covering a float bbox: floor−1 / ceil+1,
/// clamped to the surface, `[0, 0, 0, 0]` when empty.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "region coordinates are finite, non-negative and below the surface size"
)]
fn padded_region(min: [f32; 2], max: [f32; 2], width: u32, height: u32) -> [u32; 4] {
    let x0 = (min[0].floor() - 1.0).max(0.0);
    let y0 = (min[1].floor() - 1.0).max(0.0);
    let x1 = (max[0].ceil() + 1.0).min(width as f32);
    let y1 = (max[1].ceil() + 1.0).min(height as f32);
    if x1 <= x0 || y1 <= y0 {
        return [0, 0, 0, 0];
    }
    [x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32]
}

/// Porter-Duff operators where a transparent source writes over the
/// destination instead of leaving it unchanged. These composite over the
/// layer's whole clip (or parent), not the tight content region.
const fn is_destructive(blend: cherenkov::BlendMode) -> bool {
    use cherenkov::BlendMode as B;
    matches!(
        blend,
        B::Clear | B::Src | B::SrcIn | B::SrcOut | B::DestIn | B::DestAtop
    )
}

/// The device region a destructive composite covers: the clip's extent,
/// or the whole parent when there is no clip. Without an axis-aligned
/// rect, the conservative bbox is the clip shape's half extents mapped to
/// device space (`inv` maps device to clip-local, so `inv⁻¹` maps back);
/// the clip shape itself still bounds coverage per pixel.
#[expect(
    clippy::cast_possible_truncation,
    reason = "clip extents fit the f32 surface space"
)]
fn clip_region(clip: Option<DeviceClip>, width: u32, height: u32) -> [u32; 4] {
    let Some(clip) = clip else {
        return [0, 0, width, height];
    };
    if let Some(rect) = clip.aligned_rect {
        return padded_region(
            [rect.x0 as f32, rect.y0 as f32],
            [rect.x1 as f32, rect.y1 as f32],
            width,
            height,
        );
    }
    let inv = clip.inv.inverse();
    let (hx, hy) = (clip.shape.half[0], clip.shape.half[1]);
    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    for (x, y) in [(hx, hy), (-hx, hy), (hx, -hy), (-hx, -hy)] {
        let p = inv * Point::new(f64::from(x), f64::from(y));
        min[0] = min[0].min(p.x as f32);
        min[1] = min[1].min(p.y as f32);
        max[0] = max[0].max(p.x as f32);
        max[1] = max[1].max(p.y as f32);
    }
    padded_region(min, max, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_margin_preserves_exact_transform_results() {
        let mut frame = Frame::default();
        let mut lowering = Lowering::new(&mut frame, (64, 64));
        let transforms = [
            Affine::IDENTITY,
            Affine::scale_non_uniform(2.0, 3.0),
            Affine::rotate(0.7),
            Affine::new([1.0, 0.2, -0.3, 2.0, 4.0, 5.0]),
            Affine::new([-0.0, 0.0, 0.0, -0.0, 0.0, 0.0]),
            Affine::new([f64::MIN_POSITIVE, 0.0, 0.0, 1.0, 0.0, 0.0]),
            Affine::new([f64::INFINITY, 0.0, 0.0, 1.0, 0.0, 0.0]),
            Affine::new([f64::NAN, 0.0, 0.0, f64::NAN, 0.0, 0.0]),
        ];
        for transform in transforms.into_iter().cycle().take(24) {
            let expected = aa_margin(transform).to_bits();
            assert_eq!(lowering.margin(transform).to_bits(), expected);
            assert_eq!(lowering.margin(transform).to_bits(), expected);
            let [a, b, c, d, _, _] = transform.as_coeffs();
            let translated = Affine::new([a, b, c, d, 123.0, -456.0]);
            assert_eq!(lowering.margin(translated).to_bits(), expected);
        }
    }

    #[test]
    fn large_opacity_groups_detect_adjacent_and_distant_overlaps() {
        let mut instances: Vec<_> = (0_u16..300)
            .map(|index| {
                let mut instance = Instance::new(KIND_SPAN);
                let x = f32::from(index) * 2.0;
                instance.bounds = [x, 0.0, x + 1.0, 1.0];
                instance
            })
            .collect();
        assert!(bboxes_disjoint(&instances));
        let last = instances[299].bounds;
        instances[299].bounds = instances[0].bounds;
        assert!(!bboxes_disjoint(&instances));
        instances[299].bounds = last;
        let within_prefix = instances[15].bounds;
        instances[15].bounds = instances[0].bounds;
        assert!(!bboxes_disjoint(&instances));
        instances[15].bounds = within_prefix;
        instances[1].bounds = instances[0].bounds;
        assert!(!bboxes_disjoint(&instances));
    }

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
            lowering.realize(&op, None, glyphs, &cherenkov::DisplayList::default())?;
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
        lowering.begin_pass(Target::Surface, None);
        let glyphs = GlyphContext {
            atlas: &atlas,
            fonts: &fonts,
            images: &images,
            content: &HashMap::new(),
        };
        let prefix = cherenkov::Command::Fill {
            shape: ShapeData::Rect(Rect::new(0.0, 0.0, 2.0, 2.0)),
            paint: cherenkov::Paint::Solid(WorkingColor::new([0.0, 1.0, 0.0, 1.0])),
        };
        draw(&mut lowering, &prefix, &glyphs).expect("prefix");
        let mut calls = 0;
        // Two overlapping rects defeat the pass-through speculation, so
        // the group really isolates into a scratch and composites back.
        lowering
            .isolate(
                None,
                None,
                0.5,
                cherenkov::BlendMode::Normal,
                |s, g| {
                    calls += 1;
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
        assert_eq!(calls, 1, "promote the speculative output without replay");
        draw(&mut lowering, &prefix, &glyphs).expect("suffix");
        lowering.finish_pass();
        assert_eq!(frame.passes.len(), 3);
        assert_eq!(frame.passes[0].target, Target::Surface);
        assert_eq!(frame.passes[0].ranges[0].instances, 0..1);
        assert_eq!(frame.passes[1].target, Target::Scratch(0));
        assert_eq!(frame.passes[1].ranges[0].instances, 1..3);
        assert_eq!(frame.passes[2].target, Target::Surface);
        assert_eq!(frame.passes[2].ranges[0].instances, 3..4);
        assert_eq!(frame.passes[2].ranges[1].instances, 4..5);
        let composite = frame
            .instances
            .iter()
            .find(|i| i.meta[1] == PAINT_TEXTURE)
            .expect("the composite instance");
        assert_eq!(composite.meta[0], KIND_SPAN, "composites are spans");
    }

    fn one_rect_body(s: &mut Lowering<'_>, g: &GlyphContext<'_>) -> Result<(), RenderError> {
        draw(
            s,
            &cherenkov::Command::Fill {
                shape: ShapeData::Rect(Rect::new(4.0, 4.0, 20.0, 20.0)),
                paint: cherenkov::Paint::Solid(WorkingColor::new([1.0, 0.0, 0.0, 1.0])),
            },
            g,
        )
    }

    /// A destructive blend composites over the whole clip (or parent),
    /// not the tight region of the layer's content.
    #[test]
    #[expect(clippy::float_cmp, reason = "integer regions compare exactly")]
    fn destructive_composite_covers_the_whole_parent() {
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
            content: &HashMap::new(),
        };
        let mut frame = Frame::default();
        let mut lowering = Lowering::new(&mut frame, (64, 64));
        lowering.begin_pass(Target::Surface, None);
        lowering
            .isolate(
                None,
                None,
                1.0,
                cherenkov::BlendMode::Clear,
                |s, g| one_rect_body(s, g),
                &glyphs,
            )
            .expect("destructive isolate");
        lowering
            .isolate(
                None,
                None,
                1.0,
                cherenkov::BlendMode::Multiply,
                |s, g| one_rect_body(s, g),
                &glyphs,
            )
            .expect("tight isolate");
        lowering.finish_pass();
        let scratch: Vec<_> = frame
            .passes
            .iter()
            .filter(|p| matches!(p.target, Target::Scratch(_)))
            .map(|p| p.region)
            .collect();
        assert_eq!(scratch, vec![[0, 0, 64, 64], [1, 1, 22, 22]]);
        let composites: Vec<_> = frame
            .instances
            .iter()
            .filter(|i| i.meta[1] == PAINT_TEXTURE)
            .map(|i| i.bounds)
            .collect();
        assert_eq!(
            composites[0],
            [0.0, 0.0, 64.0, 64.0],
            "the destructive composite covers the whole parent"
        );
        assert_eq!(
            composites[1],
            [1.0, 1.0, 23.0, 23.0],
            "the multiply composite keeps the tight region"
        );
    }

    /// A clipped destructive composite covers the padded clip rect and
    /// carries the clip on the composite instance.
    #[test]
    #[expect(clippy::float_cmp, reason = "integer regions compare exactly")]
    fn destructive_composite_is_bounded_by_the_clip() {
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
            content: &HashMap::new(),
        };
        let clip = DeviceClip {
            inv: Affine::translate(Vec2::new(-20.0, -20.0)),
            shape: Shape::rect([10.0, 10.0]),
            aligned_rect: Some(Rect::new(10.0, 10.0, 30.0, 30.0)),
            mask: None,
        };
        let mut frame = Frame::default();
        let mut lowering = Lowering::new(&mut frame, (64, 64));
        lowering.begin_pass(Target::Surface, None);
        lowering
            .isolate(
                Some(clip),
                None,
                1.0,
                cherenkov::BlendMode::DestAtop,
                |s, g| one_rect_body(s, g),
                &glyphs,
            )
            .expect("clipped destructive isolate");
        lowering.finish_pass();
        let scratch = frame
            .passes
            .iter()
            .find(|p| matches!(p.target, Target::Scratch(_)))
            .expect("scratch pass");
        assert_eq!(scratch.region, [9, 9, 22, 22]);
        let composite = frame
            .instances
            .iter()
            .find(|i| i.meta[1] == PAINT_TEXTURE)
            .expect("the composite instance");
        assert_eq!(composite.bounds, [9.0, 9.0, 31.0, 31.0]);
        assert_ne!(
            composite.meta[3] & (FLAG_HAS_CLIP << 24),
            0,
            "the composite carries the effective clip"
        );
    }

    #[test]
    fn is_destructive_lists_the_six_operators() {
        use cherenkov::BlendMode as B;
        let all = [
            B::Normal,
            B::Multiply,
            B::Screen,
            B::Overlay,
            B::Darken,
            B::Lighten,
            B::ColorDodge,
            B::ColorBurn,
            B::HardLight,
            B::SoftLight,
            B::Difference,
            B::Exclusion,
            B::Hue,
            B::Saturation,
            B::Color,
            B::Luminosity,
            B::Clear,
            B::Src,
            B::Dst,
            B::DestOver,
            B::SrcIn,
            B::DestIn,
            B::SrcOut,
            B::DestOut,
            B::SrcAtop,
            B::DestAtop,
            B::Xor,
            B::PlusLighter,
        ];
        let destructive: Vec<_> = all.into_iter().filter(|b| is_destructive(*b)).collect();
        assert_eq!(
            destructive,
            [
                B::Clear,
                B::Src,
                B::SrcIn,
                B::DestIn,
                B::SrcOut,
                B::DestAtop
            ]
        );
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
            content: &HashMap::new(),
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
