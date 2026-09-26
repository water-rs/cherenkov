// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared command-span bookkeeping for retained backend lowering.

use std::ops::Range;

use kurbo::Affine;

use crate::{Command, Dirty, DisplayList, Group, ShapeData};

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
    /// Open a clip scope; its matching end is filled by the walker.
    ///
    /// # Errors
    /// Returns a backend error for unsupported or invalid content.
    fn clip(&mut self, shape: &ShapeData, ambient: Affine) -> Result<Self::Op, Self::Error>;
    /// Open a group, or omit isolation when the group is a pass-through.
    ///
    /// # Errors
    /// Returns a backend error for unsupported or invalid content.
    fn group(&mut self, group: &Group) -> Result<Option<Self::Op>, Self::Error>;
    /// Close a retained scope.
    fn end(&mut self) -> Self::Op;
}

#[derive(Clone)]
struct Span {
    ops: Range<u32>,
    ambient: Affine,
}

impl Default for Span {
    fn default() -> Self {
        Self {
            ops: 0..0,
            ambient: Affine::IDENTITY,
        }
    }
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
        let mut ops = Vec::new();
        let mut spans = vec![Span::default(); list.len()];
        walk(
            list,
            0..list.len(),
            Affine::IDENTITY,
            compiler,
            &mut ops,
            &mut spans,
        )?;
        Ok(Self { ops, spans })
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
            let mut spans = vec![Span::default(); range.len()];
            walk(
                list,
                range.clone(),
                self.spans[range.start].ambient,
                compiler,
                &mut ops,
                &mut spans,
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
    let mut spans = vec![Span::default(); list.len()];
    walk(list, 0..list.len(), ambient, compiler, ops, &mut spans)
}

fn walk<C: Compiler>(
    list: &DisplayList,
    range: Range<usize>,
    ambient: Affine,
    compiler: &mut C,
    ops: &mut Vec<C::Op>,
    spans: &mut [Span],
) -> Result<(), C::Error> {
    let base = range.start;
    let mut i = base;
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
                Some((*end as usize, ambient, compiler.group(group)?))
            }
            Command::Picture { picture, transform } => {
                append(picture.display_list(), ambient * *transform, compiler, ops)?;
                None
            }
            Command::End => unreachable!("validated scopes consume their end"),
            command => {
                compiler.draw(command, ambient, ops)?;
                None
            }
        };
        if let Some((end, inner, opener)) = scope {
            let emitted = opener.is_some();
            ops.extend(opener);
            spans[i - base] = Span {
                ops: start..u32::try_from(ops.len()).expect("op index fits u32"),
                ambient,
            };
            walk(
                list,
                i + 1..end,
                inner,
                compiler,
                ops,
                &mut spans[i + 1 - base..end - base],
            )?;
            let close = u32::try_from(ops.len()).expect("op index fits u32");
            if emitted {
                *ops[start as usize]
                    .end_mut()
                    .expect("scope opener has an end") = close;
                ops.push(compiler.end());
            }
            spans[end - base] = Span {
                ops: close..u32::try_from(ops.len()).expect("op index fits u32"),
                ambient: inner,
            };
            i = end + 1;
        } else {
            spans[i - base] = Span {
                ops: start..u32::try_from(ops.len()).expect("op index fits u32"),
                ambient,
            };
            i += 1;
        }
    }
    Ok(())
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
            list,
            dirty: Dirty::default(),
            lowered: None,
            emissions: Vec::new(),
        }
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
        Ok(count)
    }

    /// The prepared ops and their independently mutable device realizations.
    ///
    /// # Panics
    /// When composition runs before preparation.
    pub fn prepared(&mut self) -> (&[O], &mut [Realization<E>]) {
        (
            &self
                .lowered
                .as_ref()
                .expect("content prepared before composition")
                .ops,
            &mut self.emissions,
        )
    }

    /// Release device coverage without re-lowering content on the next frame.
    pub fn trim(&mut self) {
        self.emissions
            .iter_mut()
            .for_each(|entry| *entry = Realization::default());
    }
}
