// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared command-span bookkeeping for retained backend lowering.

use std::ops::Range;

use kurbo::Affine;

use crate::{BlendMode, Command, Dirty, DisplayList, FillRule, Group, ShapeData};

pub mod shadow;

/// A backend operation whose scope indices can be relocated during a patch.
pub trait Operation {
    /// Matching close index for a scope opener.
    fn end_mut(&mut self) -> Option<&mut u32>;
    /// Whether replacing this operation preserves the command/pass structure.
    fn same_structure(&self, other: &Self) -> bool;
}

/// The backend-specific part of content lowering. Layer state is deliberately
/// absent: it is applied to the retained operations at composition time.
pub trait Compiler {
    /// Retained operation representation.
    type Op: Operation;
    /// Backend lowering error.
    type Error;
    /// Lower one drawing command (scope and picture commands are walked here).
    ///
    /// # Errors
    /// Returns a backend error for unsupported or invalid content.
    fn draw(
        &mut self,
        command: &Command,
        ambient: Affine,
        ops: &mut Vec<Self::Op>,
    ) -> Result<(), Self::Error>;
    /// Lower a command with its index in the root source list, when it belongs
    /// to that list. Nested pictures and expanded glyphs have no root index.
    ///
    /// # Errors
    /// Propagates the backend compiler's error.
    fn draw_at(
        &mut self,
        command: &Command,
        ambient: Affine,
        ops: &mut Vec<Self::Op>,
        _source: Option<usize>,
    ) -> Result<(), Self::Error> {
        self.draw(command, ambient, ops)
    }

    /// Open a clip scope; its matching end is filled by the walker.
    ///
    /// # Errors
    /// Returns a backend error for unsupported or invalid content.
    fn clip(&mut self, shape: &ShapeData, ambient: Affine) -> Result<Self::Op, Self::Error>;
    /// Open a group, or omit isolation when the group is a pass-through.
    /// `isolate` forces isolation for a group that would otherwise compose
    /// in place: a blended descendant group must composite against this
    /// group's own raster, not the parent's.
    ///
    /// # Errors
    /// Returns a backend error for unsupported or invalid content.
    fn group(&mut self, group: &Group, isolate: bool) -> Result<Option<Self::Op>, Self::Error>;
    /// Close a retained scope.
    fn end(&mut self) -> Self::Op;
}

#[derive(Clone)]
struct Span {
    ops: Range<u32>,
    ambient: Affine,
}

/// Retained backend operations and the source commands that produced them.
pub struct Lowered<T> {
    /// Draws and paired scopes in painter order.
    pub ops: Vec<T>,
    spans: Vec<Span>,
}

impl<T: Operation> Lowered<T> {
    /// Resolve all commands once, recording their operation spans.
    ///
    /// # Errors
    /// Propagates the backend compiler's error.
    pub fn full<C: Compiler<Op = T>>(
        list: &DisplayList,
        compiler: &mut C,
    ) -> Result<Self, C::Error> {
        let mut lowered = Self {
            ops: Vec::with_capacity(list.len()),
            spans: Vec::with_capacity(list.len()),
        };
        lowered.refill(list, compiler)?;
        Ok(lowered)
    }

    fn refill<C: Compiler<Op = T>>(
        &mut self,
        list: &DisplayList,
        compiler: &mut C,
    ) -> Result<(), C::Error> {
        self.ops.clear();
        self.spans.clear();
        walk(
            list,
            0..list.len(),
            Affine::IDENTITY,
            compiler,
            &mut self.ops,
            &mut self.spans,
            true,
        )
    }

    /// Patch dirty command spans in place. `None` means the operation count or
    /// pass structure changed and this layer was rebuilt. Otherwise the returned
    /// operation ranges identify precisely which device realizations expired.
    ///
    /// # Errors
    /// Propagates the backend compiler's error.
    ///
    /// # Panics
    /// When dirty ranges refer to another list or the lowered op count exceeds u32.
    pub fn patch<C: Compiler<Op = T>>(
        &mut self,
        list: &DisplayList,
        dirty: &Dirty,
        compiler: &mut C,
    ) -> Result<Option<Vec<Range<usize>>>, C::Error> {
        let mut patches = Vec::new();
        for range in dirty.ranges() {
            let range = range.start as usize..range.end as usize;
            let mut ops = Vec::new();
            let mut spans = Vec::with_capacity(range.len());
            walk(
                list,
                range.clone(),
                self.spans[range.start].ambient,
                compiler,
                &mut ops,
                &mut spans,
                true,
            )?;
            let lo = self.spans[range.start].ops.start as usize;
            let hi = self.spans[range.end - 1].ops.end as usize;
            let old_spans = &self.spans[range.clone()];
            if hi - lo != ops.len()
                || old_spans
                    .iter()
                    .zip(&spans)
                    .any(|(a, b)| a.ops.len() != b.ops.len())
                || self.ops[lo..hi]
                    .iter()
                    .zip(&ops)
                    .any(|(a, b)| !a.same_structure(b))
            {
                *self = Self::full(list, compiler)?;
                return Ok(None);
            }
            let offset = u32::try_from(lo).expect("op index fits u32");
            for op in &mut ops {
                if let Some(end) = op.end_mut() {
                    *end += offset;
                }
            }
            for (dst, src) in self.ops[lo..hi].iter_mut().zip(ops) {
                *dst = src;
            }
            for (dst, mut src) in self.spans[range].iter_mut().zip(spans) {
                src.ops = src.ops.start + offset..src.ops.end + offset;
                *dst = src;
            }
            patches.push(lo..hi);
        }
        Ok(Some(patches))
    }
}

impl<T> Lowered<T> {
    /// Heap bytes of the operation and span buffers, including each
    /// operation's own allocations as reported by `nested`.
    fn heap_bytes(&self, mut nested: impl FnMut(&T) -> u64) -> u64 {
        let mut bytes = (self.ops.capacity() * size_of::<T>()
            + self.spans.capacity() * size_of::<Span>()) as u64;
        for op in &self.ops {
            bytes += nested(op);
        }
        bytes
    }
}

/// Append a nested picture or an expanded colour glyph. Its operations belong
/// to the enclosing source command's span.
///
/// # Errors
/// Propagates the backend compiler's error.
pub fn append<C: Compiler>(
    list: &DisplayList,
    ambient: Affine,
    compiler: &mut C,
    ops: &mut Vec<C::Op>,
) -> Result<(), C::Error> {
    let mut spans = Vec::with_capacity(list.len());
    walk(
        list,
        0..list.len(),
        ambient,
        compiler,
        ops,
        &mut spans,
        false,
    )
}

fn walk<C: Compiler>(
    list: &DisplayList,
    range: Range<usize>,
    ambient: Affine,
    compiler: &mut C,
    ops: &mut Vec<C::Op>,
    spans: &mut Vec<Span>,
    root: bool,
) -> Result<(), C::Error> {
    let mut i = range.start;
    while i < range.end {
        let start = u32::try_from(ops.len()).expect("op index fits u32");
        let scope = match &list.commands()[i] {
            Command::BeginTransform { transform, end } => {
                Some((*end as usize, ambient * *transform, None))
            }
            Command::BeginClip { shape, end } => {
                Some((*end as usize, ambient, Some(compiler.clip(shape, ambient)?)))
            }
            Command::BeginGroup { group, end } => {
                // A pass-through group still isolates when a descendant
                // group blends: without it the descendant's composite would
                // land on the parent's raster, not the group's own.
                let isolate = group.blend == BlendMode::Normal
                    && group.opacity >= 1.0
                    && group.filter.is_none()
                    && blends_within(list, i + 1..*end as usize);
                Some((*end as usize, ambient, compiler.group(group, isolate)?))
            }
            Command::Picture { picture, transform } => {
                append(picture.display_list(), ambient * *transform, compiler, ops)?;
                None
            }
            Command::End => unreachable!("validated scopes consume their end"),
            command => {
                compiler.draw_at(command, ambient, ops, root.then_some(i))?;
                None
            }
        };
        if let Some((end, inner, opener)) = scope {
            let emitted = opener.is_some();
            ops.extend(opener);
            spans.push(Span {
                ops: start..u32::try_from(ops.len()).expect("op index fits u32"),
                ambient,
            });
            walk(list, i + 1..end, inner, compiler, ops, spans, root)?;
            let close = u32::try_from(ops.len()).expect("op index fits u32");
            if emitted {
                *ops[start as usize]
                    .end_mut()
                    .expect("scope opener has an end") = close;
                ops.push(compiler.end());
            }
            spans.push(Span {
                ops: close..u32::try_from(ops.len()).expect("op index fits u32"),
                ambient: inner,
            });
            i = end + 1;
        } else {
            spans.push(Span {
                ops: start..u32::try_from(ops.len()).expect("op index fits u32"),
                ambient,
            });
            i += 1;
        }
    }
    Ok(())
}

/// Whether any command in `range` opens a group with a non-`Normal` blend.
/// Nested pictures count: their contents are walked the same way. Glyphs
/// need no scan — a colour glyph's expansion is itself wrapped in the
/// outer `SrcOver` group this rule isolates.
pub(crate) fn blends_within(list: &DisplayList, range: Range<usize>) -> bool {
    for command in &list.commands()[range] {
        match command {
            Command::BeginGroup { group, .. } => {
                if group.blend != BlendMode::Normal {
                    return true;
                }
            }
            Command::Picture { picture, .. }
                if blends_within(picture.display_list(), 0..picture.display_list().len()) =>
            {
                return true;
            }
            _ => {}
        }
    }
    false
}

/// A retained device realization. Invalidation keeps the storage available for
/// the next patch, while a structural rebuild drops the command layout.
pub struct Realization<T> {
    /// Device data and its placement key.
    pub data: Option<T>,
    /// Whether the source operations still match this realization.
    pub valid: bool,
}

impl<T> Default for Realization<T> {
    fn default() -> Self {
        Self {
            data: None,
            valid: false,
        }
    }
}

/// One layer's display list, accumulated dirty commands, and retained output.
pub struct Content<O, E> {
    live: bool,
    rebuild: bool,
    list: crate::Picture,
    dirty: Dirty,
    lowered: Option<Lowered<O>>,
    emissions: Vec<Realization<E>>,
}

impl<O: Operation, E> Content<O, E> {
    /// Start an unprepared layer.
    #[must_use]
    pub fn new(list: crate::Picture) -> Self {
        Self {
            live: true,
            rebuild: false,
            list,
            dirty: Dirty::default(),
            lowered: None,
            emissions: Vec::new(),
        }
    }

    /// Replace all source commands while keeping the lowering buffers available.
    pub fn replace(&mut self, list: crate::Picture) -> crate::Picture {
        self.live = true;
        let previous = std::mem::replace(&mut self.list, list);
        self.dirty = Dirty::default();
        self.emissions.clear();
        if let Some(lowered) = &mut self.lowered {
            lowered.ops.clear();
            lowered.spans.clear();
        }
        self.rebuild = true;
        previous
    }

    /// Extract the retained source picture.
    #[must_use]
    pub fn into_picture(self) -> crate::Picture {
        self.list
    }

    /// Retain immutable picture content, which cannot accept slot updates.
    #[must_use]
    pub fn picture(picture: crate::Picture) -> Self {
        Self {
            live: false,
            ..Self::new(picture)
        }
    }

    /// Discard compiled resource references after a resource is removed.
    pub fn invalidate(&mut self) {
        self.lowered = None;
        self.emissions.clear();
    }

    /// Whether the current source commands, slot updates applied and nested
    /// pictures included, sample image `id`.
    #[must_use]
    pub fn references_image(&self, id: crate::ImageId) -> bool {
        self.list.display_list().references_image(id)
    }

    /// Discard compiled resource references after image `id`'s pixels were
    /// replaced behind the same id. Lowering resolves an image's dimensions,
    /// and on some backends its storage, into the retained operations, so
    /// content that samples `id` is lowered again. Content with pending slot
    /// updates is discarded too: its operations were compiled from operands
    /// the updates have since replaced, and those may still reference `id`.
    /// Returns whether anything was discarded.
    pub fn invalidate_image(&mut self, id: crate::ImageId) -> bool {
        let stale = self.lowered.is_some() && (!self.dirty.is_empty() || self.references_image(id));
        if stale {
            self.invalidate();
        }
        stale
    }

    /// Accumulate every commit arriving before the next render.
    ///
    /// # Panics
    /// When slot updates target immutable picture content.
    pub fn update(&mut self, updates: Vec<crate::SlotUpdate>) {
        assert!(self.live, "slot update targets immutable picture content");
        self.dirty.union(self.list.apply(updates));
    }

    /// Resolve dirty source commands; return the number lowered.
    ///
    /// # Errors
    /// Propagates the backend compiler's error.
    ///
    /// # Panics
    /// When the command count exceeds u32.
    pub fn prepare<C: Compiler<Op = O>>(&mut self, compiler: &mut C) -> Result<u32, C::Error> {
        let count;
        if self.rebuild
            && let Some(lowered) = &mut self.lowered
        {
            lowered.refill(self.list.display_list(), compiler)?;
            self.emissions
                .resize_with(lowered.ops.len(), Realization::default);
            self.dirty = Dirty::default();
            self.rebuild = false;
            return Ok(
                u32::try_from(self.list.display_list().len()).expect("command count fits u32")
            );
        }
        if let Some(lowered) = &mut self.lowered {
            if self.dirty.is_empty() {
                return Ok(0);
            }
            if let Some(patches) = lowered.patch(self.list.display_list(), &self.dirty, compiler)? {
                for range in patches {
                    for emission in &mut self.emissions[range] {
                        emission.valid = false;
                    }
                }
                count = self.dirty.ranges().iter().map(|r| r.end - r.start).sum();
            } else {
                self.emissions.clear();
                self.emissions
                    .resize_with(lowered.ops.len(), Realization::default);
                count =
                    u32::try_from(self.list.display_list().len()).expect("command count fits u32");
            }
        } else {
            let lowered = Lowered::full(self.list.display_list(), compiler)?;
            self.emissions
                .resize_with(lowered.ops.len(), Realization::default);
            self.lowered = Some(lowered);
            count = u32::try_from(self.list.display_list().len()).expect("command count fits u32");
        }
        self.dirty = Dirty::default();
        self.rebuild = false;
        Ok(count)
    }

    /// The prepared ops and their independently mutable device realizations.
    ///
    /// # Panics
    /// When composition runs before preparation.
    pub fn prepared(&mut self) -> (&[O], &mut [Realization<E>]) {
        let (ops, emissions, _) = self.prepared_source();
        (ops, emissions)
    }

    /// Prepared operations, their device output, and the source they may index.
    ///
    /// # Panics
    /// When composition runs before preparation.
    pub fn prepared_source(&mut self) -> (&[O], &mut [Realization<E>], &DisplayList) {
        (
            &self
                .lowered
                .as_ref()
                .expect("content prepared before composition")
                .ops,
            &mut self.emissions,
            self.list.display_list(),
        )
    }

    /// Retained device realizations, including invalid entries awaiting replacement.
    /// Backends use this read-only view for residency accounting without preparing content.
    #[must_use]
    pub fn realizations(&self) -> &[Realization<E>] {
        &self.emissions
    }

    /// Release device coverage without re-lowering content on the next frame.
    pub fn trim(&mut self) {
        self.emissions
            .iter_mut()
            .for_each(|entry| *entry = Realization::default());
    }

    /// Heap bytes retained by this content: the source list, the lowered
    /// operations and the device realizations. `command`, `op` and
    /// `emission` report each element's own heap allocations.
    ///
    /// The source list is an `Arc`: contents shared with another layer's
    /// picture are counted at each retain.
    pub fn heap_bytes(
        &self,
        mut command: impl FnMut(&Command) -> u64,
        mut op: impl FnMut(&O) -> u64,
        mut emission: impl FnMut(&E) -> u64,
    ) -> u64 {
        let mut bytes = self.list.display_list().heap_bytes(&mut command)
            + (self.emissions.capacity() * size_of::<Realization<E>>()) as u64;
        if let Some(lowered) = &self.lowered {
            bytes += lowered.heap_bytes(&mut op);
        }
        for entry in &self.emissions {
            if let Some(data) = &entry.data {
                bytes += emission(data);
            }
        }
        bytes
    }
}

/// Comparison epsilon for sweep positions and boundary times.
const EPS: f64 = 1e-9;

/// One non-horizontal segment normalized to run top-to-bottom in `f64`.
struct Seg {
    /// X at the top endpoint.
    x0: f64,
    /// Top y.
    y0: f64,
    /// Bottom y.
    y1: f64,
    /// dx/dy.
    slope: f64,
    /// Signed winding deposit: +1 downward, -1 upward.
    dir: f64,
}

impl Seg {
    fn x_at(&self, y: f64) -> f64 {
        self.slope.mul_add(y - self.y0, self.x0)
    }
}

/// Winding resolution before signed-area accumulation.
///
/// The area accumulator reads each pixel's net winding: overlapping
/// same-sign regions clamp (`0.6 + 0.6 -> 1.0`, losing the union's true
/// 0.84) and opposite-sign regions cancel — both wrong under `NonZero`,
/// and the raw winding can exceed any rule's range on self-overlapping
/// outlines (kurbo stroke output overlaps at joins and caps).
/// `resolve_winding` sweeps the flattened device-space segments left to
/// right inside crossing-free horizontal bands and emits each band's
/// inside/outside boundary edges, so every covered region carries
/// winding `1` and the accumulator becomes exact under `NonZero`.
///
/// `None` means the input had no overlap: the caller keeps the original
/// segments and rule untouched, leaving every non-overlapping scene
/// bit-identical.
#[expect(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    reason = "one sweep per design; emitted coordinates fit f32"
)]
pub fn resolve_winding(
    segments: &[(f32, f32, f32, f32)],
    rule: FillRule,
) -> Option<Vec<(f32, f32, f32, f32)>> {
    let mut segs: Vec<Seg> = Vec::with_capacity(segments.len());
    for &(x0, y0, x1, y1) in segments {
        if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
            continue;
        }
        let (x0, y0, x1, y1) = (f64::from(x0), f64::from(y0), f64::from(x1), f64::from(y1));
        #[expect(clippy::float_cmp, reason = "horizontal edges carry no area")]
        if y0 == y1 {
            continue;
        }
        if y0 < y1 {
            segs.push(Seg {
                x0,
                y0,
                y1,
                slope: (x1 - x0) / (y1 - y0),
                dir: 1.0,
            });
        } else {
            segs.push(Seg {
                x0: x1,
                y0: y1,
                y1: y0,
                slope: (x0 - x1) / (y0 - y1),
                dir: -1.0,
            });
        }
    }
    if segs.is_empty() {
        return None;
    }
    let mut ys: Vec<f64> = Vec::with_capacity(2 * segs.len());
    ys.extend(segs.iter().flat_map(|s| [s.y0, s.y1]));
    ys.sort_by(f64::total_cmp);
    ys.dedup();
    segs.sort_by(|a, b| a.y0.total_cmp(&b.y0));

    let mut out: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(segs.len());
    let mut overlap = false;
    // `next` admits segments as the sweep reaches their top; `active`
    // retires them once they end at or above the band's top. No segment
    // endpoint ever lies strictly inside a band — inserted split points
    // don't change admission — so the incremental set matches a fresh
    // `y0 <= ya && y1 >= yb` scan every band.
    let mut next = 0usize;
    let mut active: Vec<usize> = Vec::with_capacity(segs.len());
    // Per-segment open boundary run: (index into `out`, orientation,
    // emitting band). A segment that is a boundary again in the very
    // next band with the same orientation extends its emitted edge
    // instead of starting a new piece; a run not emitted in a band
    // fails the `band - 1` test and closes itself. The merged edge is
    // exactly the union of the per-band pieces of the same line.
    let mut runs: Vec<Option<(usize, bool, usize)>> = vec![None; segs.len()];
    // The runs emitted or extended in the previous band, for the
    // cross-segment collinear merge: (index into `out`, orientation,
    // x at the band's bottom, slope). Rebuilt after each emitting
    // band; a split `continue` leaves it untouched since no band was
    // emitted between. `open` is the band under construction; the two
    // buffers swap so no band allocates.
    let mut prev_open: Vec<(usize, bool, f64, f64)> = Vec::with_capacity(segs.len());
    let mut open: Vec<(usize, bool, f64, f64)> = Vec::new();
    let mut band = 0usize;
    while band + 1 < ys.len() {
        let (ya, yb) = (ys[band], ys[band + 1]);
        while next < segs.len() && segs[next].y0 <= ya + EPS {
            active.push(next);
            next += 1;
        }
        active.retain(|&i| segs[i].y1 > ya + EPS);
        if active.is_empty() {
            prev_open.clear();
            band += 1;
            continue;
        }
        // Order at the band's midpoint: segments that meet at a vertex
        // on ya or yb are separated at ym, so boundary ties cannot hide
        // a crossing. If two non-adjacent segments p < r cross at yc
        // inside the band, the segment q between them at ym must have
        // swapped with p or r somewhere between ym and yc, so an
        // adjacent pair also crosses strictly inside — checking only
        // adjacent pairs is complete. Genuinely coincident segments
        // never cross and give the same regions in any order.
        let ym = ya.midpoint(yb);
        active.sort_by(|a, b| {
            segs[*a]
                .x_at(ym)
                .total_cmp(&segs[*b].x_at(ym))
                .then_with(|| segs[*a].slope.total_cmp(&segs[*b].slope))
        });
        // Split at the smallest crossing strictly inside the band.
        let mut split = None;
        for pair in active.windows(2) {
            let (p, q) = (&segs[pair[0]], &segs[pair[1]]);
            if (p.slope - q.slope).abs() > EPS {
                // p.x0 + p.slope*(y-p.y0) == q.x0 + q.slope*(y-q.y0)
                let yc = q.slope.mul_add(q.y0, p.slope.mul_add(-p.y0, p.x0) - q.x0)
                    / (q.slope - p.slope);
                if yc > ya + EPS && yc < yb - EPS && split.is_none_or(|s| yc < s) {
                    split = Some(yc);
                }
            }
        }
        if let Some(yc) = split {
            ys.insert(band + 1, yc);
            overlap = true;
            continue;
        }
        let mut w = 0.0f64;
        let mut inside = false;
        let mut wmin = 0.0f64;
        let mut wmax = 0.0f64;
        open.clear();
        for &i in &active {
            let seg = &segs[i];
            w += seg.dir;
            wmin = wmin.min(w);
            wmax = wmax.max(w);
            let now = match rule {
                FillRule::NonZero => w != 0.0,
                FillRule::EvenOdd => w.rem_euclid(2.0) > 0.5,
            };
            if now != inside {
                let (xa, xb) = (seg.x_at(ya), seg.x_at(yb));
                // Continue this segment's own open run, or another
                // segment's run that ends exactly where this piece
                // starts (collinear segments share the same boundary
                // line, so the extension is exact).
                // Continue this segment's own open run, or an open
                // run from the previous band that ends where this
                // piece starts and shares its slope — coincident
                // collinear segments form one boundary line, so the
                // extension is exact; at a kink the slope differs and
                // a new edge starts.
                let run = match runs[i] {
                    Some((edge, orient, last)) if orient == now && last + 1 == band => Some(edge),
                    _ => prev_open.iter().find_map(|&(edge, orient, x_end, slope)| {
                        (orient == now
                            && (x_end - xa).abs() <= EPS * (1.0 + xa.abs())
                            && (slope - seg.slope).abs() <= EPS)
                            .then_some(edge)
                    }),
                };
                let edge = if let Some(edge) = run {
                    let piece = &mut out[edge];
                    if now {
                        piece.2 = xb;
                        piece.3 = yb;
                    } else {
                        piece.0 = xb;
                        piece.1 = yb;
                    }
                    edge
                } else {
                    if now {
                        out.push((xa, ya, xb, yb));
                    } else {
                        out.push((xb, yb, xa, ya));
                    }
                    out.len() - 1
                };
                runs[i] = Some((edge, now, band));
                open.push((edge, now, xb, seg.slope));
                inside = now;
            }
        }
        std::mem::swap(&mut open, &mut prev_open);
        // A winding magnitude above one, or both signs in one band,
        // means regions overlap — only then is rewriting needed.
        if wmax >= 2.0 || wmin <= -2.0 || (wmin < 0.0 && wmax > 0.0) {
            overlap = true;
        }
        band += 1;
    }
    if !overlap {
        return None;
    }
    Some(
        out.iter()
            .map(|&(x0, y0, x1, y1)| (x0 as f32, y0 as f32, x1 as f32, y1 as f32))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Signed area of the region whose boundary is `edges`: on the
    /// winding-0/1 emitted set, Σ x̄·(y1−y0) is the shoelace vertical
    /// contribution and horizontal connectors carry no area.
    fn area(edges: &[(f32, f32, f32, f32)]) -> f64 {
        edges
            .iter()
            .map(|&(x0, y0, x1, y1)| f64::midpoint(x0.into(), x1.into()) * f64::from(y1 - y0))
            .sum()
    }

    fn square(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<(f32, f32, f32, f32)> {
        vec![
            (x0, y0, x1, y0),
            (x1, y0, x1, y1),
            (x1, y1, x0, y1),
            (x0, y1, x0, y0),
        ]
    }

    #[test]
    fn a_single_square_needs_no_resolution() {
        let segs = square(0.0, 0.0, 9.0, 9.0);
        assert!(resolve_winding(&segs, FillRule::NonZero).is_none());
        assert!(resolve_winding(&segs, FillRule::EvenOdd).is_none());
    }

    #[test]
    fn overlapping_squares_emit_their_union_boundary() {
        // [0,3]² and [1.5,4.5]², same winding: union 9 + 9 − 1.5² = 15.75.
        let mut segs = square(0.0, 0.0, 3.0, 3.0);
        segs.extend(square(1.5, 1.5, 4.5, 4.5));
        let resolved = resolve_winding(&segs, FillRule::NonZero).expect("overlap present");
        // One merged edge per boundary line: x=0, x=3, x=1.5, x=4.5.
        assert_eq!(resolved.len(), 4, "merged edges: {resolved:?}");
        assert!(
            (area(&resolved).abs() - 15.75).abs() < 1e-3,
            "union area {}",
            area(&resolved)
        );
        // The overlap's interior boundary edges cancel: between x=1.5
        // and x=3 the two squares' shared edges are inside the union and
        // must not appear.
        assert!(
            resolved
                .iter()
                .all(|e| e.0 >= 0.0 && e.1 >= 0.0 && e.2 >= 0.0 && e.3 >= 0.0),
            "edges outside input bounds: {resolved:?}"
        );
    }

    #[test]
    fn stacked_rectangles_emit_one_edge_per_side() {
        // 40 rectangles [0,10]×[0.5i, 0.5i+5], all same winding: many
        // bands and heavy overlap, but the union is one rectangle
        // [0,10]×[0,24.5] — two merged boundary edges, area 245.
        let mut segs = Vec::new();
        for i in 0..40u8 {
            let y = 0.5 * f32::from(i);
            segs.extend(square(0.0, y, 10.0, y + 5.0));
        }
        let resolved = resolve_winding(&segs, FillRule::NonZero).expect("overlap present");
        assert_eq!(resolved.len(), 2, "merged edges: {resolved:?}");
        assert!(
            (area(&resolved).abs() - 245.0).abs() < 1e-3,
            "union area {}",
            area(&resolved)
        );
    }

    #[test]
    fn a_bow_tie_keeps_both_lobes() {
        // Self-crossing quad: mixed-sign windings cancel in the raw
        // accumulator; resolved coverage is the two-triangle union,
        // area 8.0.
        let segs = vec![
            (0.0, 0.0, 4.0, 4.0),
            (4.0, 4.0, 4.0, 0.0),
            (4.0, 0.0, 0.0, 4.0),
            (0.0, 4.0, 0.0, 0.0),
        ];
        let resolved = resolve_winding(&segs, FillRule::NonZero).expect("mixed signs overlap");
        assert!(
            (area(&resolved).abs() - 8.0).abs() < 1e-3,
            "lobes area {}",
            area(&resolved).abs()
        );
    }

    #[test]
    fn even_odd_nested_squares_open_the_hole() {
        let mut segs = square(0.0, 0.0, 4.0, 4.0);
        segs.extend(square(1.0, 1.0, 3.0, 3.0));
        let resolved = resolve_winding(&segs, FillRule::EvenOdd).expect("winding reaches 2");
        // Outer minus inner area: the hole's boundary edges run
        // reversed (the inner square's winding is cancelled).
        assert!(
            (area(&resolved).abs() - 12.0).abs() < 1e-3,
            "ring area {}",
            area(&resolved).abs()
        );
        assert!(
            resolved
                .iter()
                .any(|&(x0, y0, x1, y1)| { x0 >= 1.0 && x1 >= 1.0 && y1 < y0 && (y0 - y1) > 1.0 }),
            "no upward inner edge emitted for the hole: {resolved:?}"
        );
    }

    /// Whether the open interiors of two segments intersect.
    fn edges_cross(e: (f32, f32, f32, f32), f: (f32, f32, f32, f32)) -> bool {
        let cross = |p: (f64, f64), q: (f64, f64), r: (f64, f64)| {
            (q.1 - p.1).mul_add(-(r.0 - p.0), (q.0 - p.0) * (r.1 - p.1))
        };
        let (e0, e1, f0, f1) = (
            (f64::from(e.0), f64::from(e.1)),
            (f64::from(e.2), f64::from(e.3)),
            (f64::from(f.0), f64::from(f.1)),
            (f64::from(f.2), f64::from(f.3)),
        );
        let (d1, d2, d3, d4) = (
            cross(f0, f1, e0),
            cross(f0, f1, e1),
            cross(e0, e1, f0),
            cross(e0, e1, f1),
        );
        ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0)) && d1 != 0.0 && d2 != 0.0
    }

    #[test]
    fn a_later_crossing_still_splits_the_band() {
        // The first inverted pair (0,0)-(4,4) vs (1e-10,0)-(-4,4) crosses
        // at y≈5e-11 — on the band's top boundary, a tie, no valid split.
        // A scan that stops at that pair would walk (5,0)-(4,4) and
        // (6,0)-(3,4) while they cross inside at y=2.
        let segs = vec![
            (0.0f32, 0.0, 4.0, 4.0),
            (1e-10, 0.0, -4.0, 4.0),
            (5.0, 0.0, 4.0, 4.0),
            (6.0, 0.0, 3.0, 4.0),
        ];
        let resolved = resolve_winding(&segs, FillRule::NonZero).expect("crossing present");
        // The crossing splits the band, so no two emitted edges cross
        // strictly inside (the pre-fix walk emitted crossing pieces).
        for (i, first) in resolved.iter().enumerate() {
            for second in &resolved[i + 1..] {
                assert!(
                    !edges_cross(*first, *second),
                    "emitted edges cross: {first:?} x {second:?}"
                );
            }
        }
    }

    #[test]
    fn a_nonadjacent_crossing_still_splits_the_band() {
        // Real geometry from a stroked glyph: at the band top the sweep
        // order is A(+1) B(−1) C(−1) with B just left of C, so the only
        // *adjacent* inversion is B×C — which crossed exactly at the
        // band's top boundary and yields no valid split. C also crosses
        // A inside the band at y≈42.72, non-adjacently; an adjacent-only
        // scan misses it and emits edges that cross inside the band.
        // Verbatim segments from a stroked glyph (the +1 edge, the two
        // diagonals, and the long contour edge sharing its top vertex).
        let segs = vec![
            (110.472_26f32, 43.915_257, 108.866_936, 27.915_59),
            (110.856_94, 27.715_923, 112.462_265, 43.715_59),
            (113.483_76, 27.452_364, 115.089_07, 43.452_03),
            (113.099_07, 43.651_7, 111.493_744, 27.652_03),
            (111.367_424, 42.820_42, 113.994_24, 42.55686),
            (113.994_24, 42.55686, 113.099_07, 43.651_7),
        ];
        let resolved = resolve_winding(&segs, FillRule::NonZero).expect("crossing present");
        assert!(
            resolved
                .iter()
                .any(|e| (e.1 - 42.7206).abs() < 1e-3 || (e.3 - 42.7206).abs() < 1e-3),
            "no edge boundary at the y≈42.72 crossing: {resolved:?}"
        );
    }
}

#[cfg(test)]
mod content_tests {
    use super::{Content, Operation};

    struct TestOp;

    impl Operation for TestOp {
        fn end_mut(&mut self) -> Option<&mut u32> {
            None
        }

        fn same_structure(&self, _other: &Self) -> bool {
            true
        }
    }

    #[test]
    fn replacement_returns_the_previous_picture() {
        let previous = crate::Picture::new(crate::DisplayList::default());
        let replacement = crate::Picture::new(crate::DisplayList::with_capacity(2));
        let mut content = Content::<TestOp, ()>::new(previous.clone());

        assert_eq!(content.replace(replacement.clone()), previous);
        assert_eq!(content.into_picture(), replacement);
    }
}
