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
    BeginIsolate {
        opacity: f32,
        blend: BlendMode,
        space: BlendSpace,
        end: u32,
    },
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
pub struct Lowerer<'a> {
    pub images: &'a std::collections::HashMap<u64, std::sync::Arc<super::image::CpuImage>>,
    pub fonts: &'a std::collections::HashMap<u64, cherenkov::FontData>,
}

impl cherenkov::lowering::Compiler for Lowerer<'_> {
    type Op = Op;
    type Error = RenderError;
    fn draw(
        &mut self,
        command: &Command,
        ambient: Affine,
        ops: &mut Vec<Op>,
    ) -> Result<(), RenderError> {
        if let Command::Glyphs { run, paint } = command
            && let cherenkov::GlyphStyle::Stroke(stroke) = &run.style
        {
            self.stroke_glyphs(ambient, run, stroke, paint, ops)?;
            return Ok(());
        }
        ops.push(match command {
            Command::Fill { shape, paint } => Op::Fill {
                local: ambient,
                shape: shape.clone(),
                paint: paint_data(paint, Affine::IDENTITY, self.images)?,
            },
            Command::Stroke {
                shape,
                stroke,
                paint,
            } => Op::Stroke {
                local: ambient,
                shape: shape.clone(),
                stroke: stroke.clone(),
                paint: paint_data(paint, Affine::IDENTITY, self.images)?,
            },
            Command::Shadow { shape, shadow } => Op::Shadow {
                local: ambient,
                shape: shape.clone(),
                shadow: *shadow,
            },
            Command::Glyphs { run, paint } => Op::Glyphs {
                local: ambient,
                run: run.clone(),
                paint: paint_data(paint, Affine::IDENTITY, self.images)?,
            },
            Command::Image {
                image,
                dst,
                sampling,
            } => {
                let source = self.images.get(&image.raw()).ok_or_else(|| {
                    RenderError::Image(format!("unregistered image {}", image.raw()))
                })?;
                let transform = Affine::translate((dst.x0, dst.y0))
                    * Affine::scale_non_uniform(
                        dst.width() / f64::from(source.width),
                        dst.height() / f64::from(source.height),
                    );
                let paint = cherenkov::Paint::Image(cherenkov::ImagePattern {
                    image: *image,
                    transform,
                    extend_x: cherenkov::Extend::Pad,
                    extend_y: cherenkov::Extend::Pad,
                    sampling: *sampling,
                });
                Op::Fill {
                    local: ambient,
                    shape: ShapeData::Rect(*dst),
                    paint: paint_data(&paint, Affine::IDENTITY, self.images)?,
                }
            }
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
        Ok((group.opacity < 1.0
            || group.blend != BlendMode::Normal
            || group.blend_space != BlendSpace::Linear)
            .then_some(Op::BeginIsolate {
                opacity: group.opacity,
                blend: group.blend,
                space: group.blend_space,
                end: 0,
            }))
    }
    fn end(&mut self) -> Op {
        Op::End
    }
}

impl Lowerer<'_> {
    fn stroke_glyphs(
        &self,
        ambient: Affine,
        run: &GlyphRun,
        stroke: &kurbo::Stroke,
        paint: &cherenkov::Paint,
        ops: &mut Vec<Op>,
    ) -> Result<(), RenderError> {
        if run.glyphs.iter().any(|glyph| glyph.transform.is_some()) {
            return Err(RenderError::Unsupported(names::GLYPH_TRANSFORM));
        }
        let font = self
            .fonts
            .get(&run.font.raw())
            .ok_or_else(|| RenderError::Font(format!("unregistered font {:?}", run.font)))?;
        let paint = paint_data(paint, Affine::IDENTITY, self.images)?;
        for path in super::glyph::stroke_outlines(font, run)? {
            ops.push(Op::Stroke {
                local: ambient,
                shape: ShapeData::Path {
                    elements: path.into_elements(),
                    rule: cherenkov::FillRule::NonZero,
                },
                stroke: stroke.clone(),
                paint: paint.clone(),
            });
        }
        Ok(())
    }
}
