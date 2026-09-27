// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Retained content-space operations. Device coverage is realized separately
//! under the sampled layer transform, so property changes never resolve paints
//! or expand pictures again.

use super::paint::{PaintData, paint_data};
use crate::names;
use cherenkov::kurbo::Affine;
use cherenkov::{BlendMode, BlendSpace, Command, GlyphRun, RenderError, Shadow, ShapeData};

/// A retained draw or paired composition scope.
pub enum Op {
    /// Fill geometry and resolved paint in content space.
    Fill {
        local: Affine,
        shape: ShapeData,
        paint: PaintData,
    },
    /// A stroke whose device tolerance is selected during composition.
    Stroke {
        local: Affine,
        shape: ShapeData,
        stroke: kurbo::Stroke,
        paint: PaintData,
    },
    /// Analytic shadow parameters.
    Shadow {
        local: Affine,
        shape: ShapeData,
        shadow: Shadow,
    },
    /// A shaped run and resolved paint.
    Glyphs {
        local: Affine,
        run: GlyphRun,
        paint: PaintData,
    },
    /// Open a clip in its content space.
    BeginClip {
        local: Affine,
        shape: ShapeData,
        end: u32,
    },
    /// Open an opacity group.
    BeginIsolate { opacity: f32, end: u32 },
    /// Close a scope.
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
        std::mem::discriminant(self) == std::mem::discriminant(other)
            && !matches!((self, other), (Self::Glyphs { run: a, .. }, Self::Glyphs { run: b, .. }) if a.glyphs.len() != b.glyphs.len())
    }
}

/// Resolve CPU paints while retaining content-space geometry.
pub struct Lowerer;

impl cherenkov::lowering::Compiler for Lowerer {
    type Op = Op;
    type Error = RenderError;
    fn draw(
        &mut self,
        command: &Command,
        ambient: Affine,
        ops: &mut Vec<Op>,
    ) -> Result<(), RenderError> {
        ops.push(match command {
            Command::Fill { shape, paint } => Op::Fill {
                local: ambient,
                shape: shape.clone(),
                paint: paint_data(paint, Affine::IDENTITY)?,
            },
            Command::Stroke {
                shape,
                stroke,
                paint,
            } => Op::Stroke {
                local: ambient,
                shape: shape.clone(),
                stroke: stroke.clone(),
                paint: paint_data(paint, Affine::IDENTITY)?,
            },
            Command::Shadow { shape, shadow } => Op::Shadow {
                local: ambient,
                shape: shape.clone(),
                shadow: *shadow,
            },
            Command::Glyphs { run, paint } => Op::Glyphs {
                local: ambient,
                run: run.clone(),
                paint: paint_data(paint, Affine::IDENTITY)?,
            },
            Command::Image { .. } => return Err(RenderError::Unsupported(names::IMAGE)),
            _ => unreachable!("shared walker handles scopes and pictures"),
        });
        Ok(())
    }
    fn clip(&mut self, shape: &ShapeData, ambient: Affine) -> Result<Op, RenderError> {
        Ok(Op::BeginClip {
            local: ambient,
            shape: shape.clone(),
            end: 0,
        })
    }
    fn group(&mut self, group: &cherenkov::Group) -> Result<Option<Op>, RenderError> {
        if group.filter.is_some() {
            return Err(RenderError::Unsupported(names::FILTER));
        }
        if group.blend_space != BlendSpace::Linear {
            return Err(RenderError::Unsupported(names::BLEND_SPACE));
        }
        if group.blend != BlendMode::Normal {
            return Err(RenderError::Unsupported(names::BLEND));
        }
        Ok((group.opacity < 1.0).then_some(Op::BeginIsolate {
            opacity: group.opacity,
            end: 0,
        }))
    }
    fn end(&mut self) -> Op {
        Op::End
    }
}
