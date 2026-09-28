//! Retained content-space operations. Device coverage is realized separately
//! under the sampled layer transform, so property changes never resolve paints
//! or expand pictures again.

use super::paint::{PaintData, paint_data};
use crate::names;
use cherenkov::kurbo::Affine;
use cherenkov::{BlendMode, BlendSpace, Command, GlyphRun, RenderError, Shadow, ShapeData};
use skrifa::MetadataProvider as _;
use skrifa::raw::TableProvider as _;
use skrifa::raw::types::F2Dot14;

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
        if let Command::Glyphs { run, paint } = command {
            if let cherenkov::GlyphStyle::Stroke(stroke) = &run.style {
                self.stroke_glyphs(ambient, run, stroke, paint, ops)?;
                return Ok(());
            }
            if run.glyphs.iter().any(|glyph| glyph.transform.is_some()) {
                self.fill_glyphs(ambient, run, paint, ops)?;
                return Ok(());
            }
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
    fn group(
        &mut self,
        group: &cherenkov::Group,
        isolate: bool,
    ) -> Result<Option<Op>, RenderError> {
        if group.filter.is_some() {
            return Err(RenderError::Unsupported(names::FILTER));
        }
        Ok((isolate
            || group.opacity < 1.0
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

    /// `Glyphs` with per-glyph transforms: pure translations fold into
    /// the position and keep the mask-cache [`Op::Glyphs`] path; every
    /// other transform expands the glyph's outline to an [`Op::Fill`]
    /// path, in glyph order.
    fn fill_glyphs(
        &self,
        ambient: Affine,
        run: &GlyphRun,
        paint: &cherenkov::Paint,
        ops: &mut Vec<Op>,
    ) -> Result<(), RenderError> {
        let font = self
            .fonts
            .get(&run.font.raw())
            .ok_or_else(|| RenderError::Font(format!("unregistered font {:?}", run.font)))?;
        let font_ref = skrifa::FontRef::from_index(&font.data, font.index)
            .map_err(|e| RenderError::Font(e.to_string()))?;
        let upem = font_ref
            .head()
            .map_err(|e| RenderError::Font(format!("head: {e}")))?
            .units_per_em();
        if upem == 0 {
            return Err(RenderError::Font("zero units_per_em".into()));
        }
        let s = f64::from(run.size) / f64::from(upem);
        let font_scale = Affine::scale_non_uniform(s, -s);
        let coords: Vec<F2Dot14> = run.coords.iter().map(|c| F2Dot14::from_bits(*c)).collect();
        let outlines = font_ref.outline_glyphs();
        let paint = paint_data(paint, Affine::IDENTITY, self.images)?;
        let mut pending: Vec<cherenkov::Glyph> = Vec::new();
        for glyph in &run.glyphs {
            let place = match super::glyph::classify(glyph)? {
                super::glyph::GlyphPlacement::Translate(g) => {
                    pending.push(g);
                    continue;
                }
                super::glyph::GlyphPlacement::Outline(place) => place,
            };
            let Some(path) = super::glyph::outline(&outlines, &coords, glyph.id)? else {
                return Err(RenderError::Font(format!(
                    "glyph {} has no outline",
                    glyph.id
                )));
            };
            if path.elements().is_empty() {
                continue;
            }
            if !pending.is_empty() {
                ops.push(Op::Glyphs {
                    local: ambient,
                    run: GlyphRun {
                        font: run.font,
                        size: run.size,
                        coords: run.coords.clone(),
                        glyphs: std::mem::take(&mut pending),
                        style: run.style.clone(),
                    },
                    paint: paint.clone(),
                });
            }
            ops.push(Op::Fill {
                local: ambient,
                shape: ShapeData::Path {
                    elements: (place * font_scale * path).into_elements(),
                    rule: cherenkov::FillRule::NonZero,
                },
                paint: paint.clone(),
            });
        }
        if !pending.is_empty() {
            ops.push(Op::Glyphs {
                local: ambient,
                run: GlyphRun {
                    font: run.font,
                    size: run.size,
                    coords: run.coords.clone(),
                    glyphs: pending,
                    style: run.style.clone(),
                },
                paint,
            });
        }
        Ok(())
    }
}
