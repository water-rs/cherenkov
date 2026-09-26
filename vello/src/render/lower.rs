// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `DisplayList` → `vello::Scene` lowering.

use std::collections::HashMap;

use cherenkov::kurbo::Affine;
use cherenkov::{BlendSpace, Command, DisplayList, GlyphStyle, Paint};
use vello::peniko::{self, Brush, ColorStops, Fill, ImageBrush, ImageSampler};
use vello::{Glyph, Scene};

use crate::error::{RenderError, Unsupported};

use super::convert;

/// The registries lowering resolves ids through.
pub struct Resources<'a> {
    /// Registered fonts, by `FontId::raw`.
    pub fonts: &'a HashMap<u64, peniko::FontData>,
    /// Registered images, by `ImageId::raw`.
    pub images: &'a HashMap<u64, peniko::ImageData>,
    /// The surface size in device pixels (group layers clip to it).
    pub target_size: (u32, u32),
}

/// Lowers `list` into `scene`, appending commands under `transform`.
///
/// # Errors
/// [`RenderError::Unsupported`] for commands this backend does not draw and
/// [`RenderError::Font`]/[`RenderError::Image`] for unregistered resources.
#[expect(clippy::too_many_lines, reason = "one arm per command variant")]
pub fn lower(
    list: &DisplayList,
    scene: &mut Scene,
    transform: Affine,
    resources: &Resources<'_>,
) -> Result<(), RenderError> {
    // Stack entries carry whether the scope pushed a vello layer that must
    // be popped at `End`; `BeginTransform` pushes no layer.
    let mut stack = vec![(transform, false)];
    for command in list.commands() {
        let xf = stack.last().expect("the stack never empties").0;
        match command {
            Command::Fill { shape, paint } => {
                let (brush, brush_transform) = brush_of(paint, resources)?;
                let path = convert::shape_path(shape);
                scene.fill(
                    convert::fill(convert::shape_rule(shape)),
                    xf,
                    &brush,
                    brush_transform,
                    &path,
                );
            }
            Command::Stroke {
                shape,
                stroke,
                paint,
            } => {
                let (brush, brush_transform) = brush_of(paint, resources)?;
                let path = convert::shape_path(shape);
                scene.stroke(stroke, xf, &brush, brush_transform, &path);
            }
            Command::Shadow { shape, shadow } => {
                let Some((rect, radius)) = convert::expressible_shadow(shape) else {
                    return Err(Unsupported::Shadow.into());
                };
                let spread = shadow.spread;
                let rect = rect.inflate(spread, spread);
                let radius = (radius + spread).max(0.0);
                let xf = xf * Affine::translate(shadow.offset);
                scene.draw_blurred_rounded_rect(
                    xf,
                    rect,
                    convert::color(&shadow.color),
                    radius,
                    shadow.sigma,
                );
            }
            Command::Glyphs { run, paint } => {
                let (brush, brush_transform) = match paint {
                    Paint::Shader(_) => return Err(Unsupported::ShaderGlyphs.into()),
                    _ => brush_of(paint, resources)?,
                };
                if run.glyphs.iter().any(|g| g.transform.is_some()) {
                    return Err(Unsupported::GlyphTransform.into());
                }
                let Some(font) = resources.fonts.get(&run.font.raw()) else {
                    return Err(RenderError::Font(format!(
                        "unregistered font {}",
                        run.font.raw()
                    )));
                };
                let style: peniko::StyleRef<'_> = match &run.style {
                    GlyphStyle::Fill => Fill::NonZero.into(),
                    GlyphStyle::Stroke(stroke) => stroke.into(),
                };
                scene
                    .draw_glyphs(font)
                    .font_size(run.size)
                    .normalized_coords(&run.coords)
                    .transform(xf)
                    .brush(&brush)
                    .brush_transform(brush_transform)
                    .draw(
                        style,
                        run.glyphs.iter().map(|g| Glyph {
                            id: g.id,
                            x: g.x,
                            y: g.y,
                        }),
                    );
            }
            Command::Image {
                // A standalone image draw, not an image pattern.
                image,
                dst,
                sampling,
            } => {
                let Some(data) = resources.images.get(&image.raw()) else {
                    return Err(RenderError::Image(format!(
                        "unregistered image {}",
                        image.raw()
                    )));
                };
                let brush = ImageBrush {
                    image: data.clone(),
                    sampler: ImageSampler {
                        x_extend: peniko::Extend::Pad,
                        y_extend: peniko::Extend::Pad,
                        quality: convert::quality(*sampling),
                        alpha: 1.0,
                    },
                };
                scene.draw_image(
                    &brush,
                    xf * convert::image_draw_transform(data.width, data.height, *dst),
                );
            }
            Command::Picture { picture, transform } => {
                lower(picture.display_list(), scene, xf * *transform, resources)?;
            }
            Command::BeginClip { shape, .. } => {
                let path = convert::shape_path(shape);
                scene.push_clip_layer(convert::fill(convert::shape_rule(shape)), xf, &path);
                stack.push((xf, true));
            }
            Command::BeginTransform { transform, .. } => {
                stack.push((xf * *transform, false));
            }
            Command::BeginGroup { group, .. } => {
                if group.filter.is_some() {
                    return Err(Unsupported::GroupFilter.into());
                }
                // Vello blends groups in the encoded 8-bit target. A group
                // that declared working-space blending cannot be honoured
                // when it would actually isolate (a non-normal blend or
                // reduced opacity); blending a plain src-over group is
                // unaffected.
                if group.blend_space == BlendSpace::Linear
                    && (group.blend != cherenkov::BlendMode::Normal || group.opacity < 1.0)
                {
                    return Err(Unsupported::BlendSpace.into());
                }
                let (w, h) = resources.target_size;
                scene.push_layer(
                    Fill::NonZero,
                    convert::blend(group.blend),
                    group.opacity,
                    Affine::IDENTITY,
                    &convert::opaque_clip(w, h),
                );
                stack.push((xf, true));
            }
            Command::End => {
                let (_, layer) = stack.pop().expect("the scope stack never empties");
                if layer {
                    scene.pop_layer();
                }
            }
        }
    }
    Ok(())
}

/// `peniko::ColorStops` from a gradient's stops.
fn stops(stops: &[cherenkov::ColorStop]) -> ColorStops {
    let v: Vec<peniko::ColorStop> = stops.iter().map(convert::stop).collect();
    ColorStops::from(&v[..])
}

/// Resolves a `Paint` to a `peniko::Brush` plus an optional brush-space
/// transform (image patterns only).
#[expect(
    clippy::cast_possible_truncation,
    reason = "peniko gradient geometry is f32; front-end geometry is f64"
)]
fn brush_of(
    paint: &Paint,
    resources: &Resources<'_>,
) -> Result<(Brush, Option<Affine>), RenderError> {
    Ok(match paint {
        Paint::Solid(c) => (Brush::Solid(convert::color(c)), None),
        Paint::Linear(g) => (
            Brush::Gradient(peniko::Gradient {
                kind: peniko::GradientKind::Linear(peniko::LinearGradientPosition {
                    start: g.start,
                    end: g.end,
                }),
                extend: convert::extend(g.extend),
                interpolation_cs: convert::interpolation(g.interpolation)?,
                stops: stops(&g.stops),
                ..peniko::Gradient::default()
            }),
            None,
        ),
        Paint::Radial(g) => (
            Brush::Gradient(peniko::Gradient {
                kind: peniko::GradientKind::Radial(peniko::RadialGradientPosition {
                    start_center: g.start_center,
                    start_radius: g.start_radius as f32,
                    end_center: g.end_center,
                    end_radius: g.end_radius as f32,
                }),
                extend: convert::extend(g.extend),
                interpolation_cs: convert::interpolation(g.interpolation)?,
                stops: stops(&g.stops),
                ..peniko::Gradient::default()
            }),
            None,
        ),
        Paint::Sweep(g) => (
            Brush::Gradient(peniko::Gradient {
                kind: peniko::GradientKind::Sweep(peniko::SweepGradientPosition {
                    center: g.center,
                    start_angle: g.start_angle as f32,
                    end_angle: g.end_angle as f32,
                }),
                extend: convert::extend(g.extend),
                interpolation_cs: convert::interpolation(g.interpolation)?,
                stops: stops(&g.stops),
                ..peniko::Gradient::default()
            }),
            None,
        ),
        Paint::Mesh(_) => return Err(Unsupported::MeshGradient.into()),
        Paint::Image(pattern) => {
            let Some(data) = resources.images.get(&pattern.image.raw()) else {
                return Err(RenderError::Image(format!(
                    "unregistered image {}",
                    pattern.image.raw()
                )));
            };
            (
                Brush::Image(ImageBrush {
                    image: data.clone(),
                    sampler: ImageSampler {
                        x_extend: convert::extend(pattern.extend_x),
                        y_extend: convert::extend(pattern.extend_y),
                        quality: convert::quality(pattern.sampling),
                        alpha: 1.0,
                    },
                }),
                Some(pattern.transform),
            )
        }
        Paint::Shader(_) => return Err(Unsupported::Shader.into()),
    })
}
