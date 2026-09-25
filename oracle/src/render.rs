// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The reference renderer: exact-coverage compositing in premultiplied
//! linear Display P3, `f64` throughout, single-threaded.
//!
//! Per the oracle rules:
//!
//! - **Clipping** is the geometric intersection of shape edges and clip
//!   edges ([`crate::clip`]) — never a product of coverages. Nested clips
//!   apply sequentially: each clip is its own edge set.
//! - **Shading** evaluates the paint at the pixel centre and multiplies by
//!   the exact coverage.
//! - **Items** composite source-over in order; a child layer renders into a
//!   fresh canvas (with the layer clip in force), then composits onto the
//!   parent with the layer's opacity and blend mode (W3C Compositing and
//!   Blending Level 1).
//! - **Shadows** are the shape's exact coverage — clip-intersected, then
//!   offset — convolved with a Gaussian in `f64`, filled with the colour.
//!   Offsetting the edges before integration is exact because convolution
//!   commutes with translation.
//! - **Strokes** expand with `kurbo`'s stroker; glyph outlines come from
//!   `skrifa`, unhinted ([`crate::glyphs`]).

use cherenkov_scene::{BlendMode, Draw, FillRule, Item, Layer, Paint, Scene, Shape};
use kurbo::{Affine, Point, Rect};

use crate::blend::{blend, src_over};
use crate::clip::{Segment, intersect_edges};
use crate::color::to_working;
use crate::coverage::Coverage;
use crate::glyphs;
use crate::image::{F32Image, Image};
use crate::paint::{eval_paint, sample_image};
use crate::path::{edges, shape_polylines, stroke_polylines_device};
use crate::resources::Resources;
use crate::shadow::gaussian_blur;

/// The renderer's error type.
#[derive(Debug)]
pub enum RenderError {
    /// Glyph rendering failed.
    Glyphs(glyphs::GlyphError),
    /// A scene resource failed to load or decode.
    Resource(cherenkov_scene::SceneError),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Glyphs(e) => write!(f, "glyph error: {e}"),
            Self::Resource(e) => write!(f, "resource error: {e}"),
        }
    }
}

impl std::error::Error for RenderError {}

impl From<glyphs::GlyphError> for RenderError {
    fn from(e: glyphs::GlyphError) -> Self {
        Self::Glyphs(e)
    }
}

impl From<cherenkov_scene::SceneError> for RenderError {
    fn from(e: cherenkov_scene::SceneError) -> Self {
        Self::Resource(e)
    }
}

/// Premultiplied linear-P3 `f64` pixels.
struct Canvas {
    pixels: Vec<[f64; 4]>,
    width: usize,
    height: usize,
}

impl Canvas {
    fn new(width: usize, height: usize, clear: [f64; 4]) -> Self {
        Self {
            pixels: vec![clear; width * height],
            width,
            height,
        }
    }
}

/// The oracle renderer.
pub struct Renderer {
    width: usize,
    height: usize,
    scene_rect: Rect,
}

impl Renderer {
    /// A renderer for `width`×`height` output pixels.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "pixel dimensions are far below 2^53"
    )]
    pub const fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            scene_rect: Rect::new(0.0, 0.0, width as f64, height as f64),
        }
    }

    /// Render `scene` into a premultiplied linear-P3 `f32` image.
    /// `scene_dir` locates the `resources/` directory.
    ///
    /// # Errors
    /// `RenderError` on glyph failures or missing resources.
    pub fn render(
        &self,
        scene: &Scene,
        scene_dir: &std::path::Path,
    ) -> Result<F32Image, RenderError> {
        let mut resources = Resources::new(scene_dir.to_path_buf());
        let clear = to_working(&scene.clear);
        let mut canvas = Canvas::new(self.width, self.height, clear);
        self.render_items(
            &scene.root.items,
            Affine::IDENTITY * scene.root.transform,
            &scene_clip_stack(&scene.root, scene.root.transform, self.width, self.height),
            &mut canvas,
            &mut resources,
        )?;
        Ok(F32Image::from_f64(&Image {
            width: self.width,
            height: self.height,
            pixels: canvas.pixels,
        }))
    }

    /// Render `items` (a layer's contents) into `canvas`.
    ///
    /// `tf` maps the items' user space to scene space; `clips` is the stack
    /// of active clip boundary edge sets, already in scene space.
    fn render_items(
        &self,
        items: &[Item],
        tf: Affine,
        clips: &[Vec<Segment>],
        canvas: &mut Canvas,
        resources: &mut Resources,
    ) -> Result<(), RenderError> {
        for item in items {
            match item {
                Item::Draw(draw) => self.render_draw(draw, tf, clips, canvas, resources)?,
                Item::Layer(child) => {
                    self.render_child_layer(child, tf, clips, canvas, resources)?;
                }
            }
        }
        Ok(())
    }

    /// Composite a child layer: render into a fresh canvas under the
    /// accumulated clips plus the child's own clip, then composite with
    /// opacity and blend mode.
    fn render_child_layer(
        &self,
        child: &Layer,
        parent_tf: Affine,
        clips: &[Vec<Segment>],
        canvas: &mut Canvas,
        resources: &mut Resources,
    ) -> Result<(), RenderError> {
        let tf = parent_tf * child.transform;
        let mut child_clips = clips.to_vec();
        if let Some(clip) = &child.clip {
            child_clips.push(shape_edges(clip, tf));
        }

        let mut sub = Canvas::new(canvas.width, canvas.height, [0.0; 4]);
        self.render_items(&child.items, tf, &child_clips, &mut sub, resources)?;

        let opacity = child.opacity;
        for (dst, &src) in canvas.pixels.iter_mut().zip(&sub.pixels) {
            let s = src.map(|v| v * opacity);
            *dst = if child.blend == BlendMode::Normal {
                src_over(*dst, s)
            } else {
                blend(child.blend, *dst, s)
            };
        }
        Ok(())
    }

    /// Exact coverage of `shape` under `tf`, clipped by every clip set.
    fn shape_coverage(
        &self,
        shape: &Shape,
        rule: FillRule,
        tf: Affine,
        clips: &[Vec<Segment>],
    ) -> Vec<f64> {
        let mut segs = edges(&shape_polylines(shape, tf));
        for clip in clips {
            segs = intersect_edges(&segs, rule, clip);
        }
        let mut cov = Coverage::new(self.width, self.height);
        for &s in &segs {
            cov.add_line(s.0, s.1, s.2, s.3);
        }
        cov.finish(rule)
    }

    /// Composite `paint` over `canvas`, multiplied by `coverage`; the paint
    /// is sampled at pixel centres in the items' user space (`inv_tf`).
    /// # Errors
    /// `RenderError` on missing resources.
    fn composite_paint(
        canvas: &mut Canvas,
        coverage: &[f64],
        paint: &Paint,
        inv_tf: Affine,
        resources: &mut Resources,
    ) -> Result<(), RenderError> {
        let (w, h) = (canvas.width, canvas.height);
        #[expect(
            clippy::cast_precision_loss,
            reason = "pixel indices are far below 2^53"
        )]
        for py in 0..h {
            for px in 0..w {
                let c = coverage[py * w + px];
                if c <= 0.0 {
                    continue;
                }
                let p = inv_tf * Point::new(px as f64 + 0.5, py as f64 + 0.5);
                let src = eval_paint(paint, p, resources)?.map(|v| v * c);
                let idx = py * w + px;
                canvas.pixels[idx] = src_over(canvas.pixels[idx], src);
            }
        }
        Ok(())
    }

    #[allow(clippy::many_single_char_names)] // u/v/w/h/x/y are the natural names
    #[expect(
        clippy::cast_precision_loss,
        reason = "pixel and image indices are far below 2^53"
    )]
    fn render_draw(
        &self,
        draw: &Draw,
        tf: Affine,
        clips: &[Vec<Segment>],
        canvas: &mut Canvas,
        resources: &mut Resources,
    ) -> Result<(), RenderError> {
        let inv_tf = tf.inverse();
        match draw {
            Draw::Fill { shape, rule, paint } => {
                let coverage = self.shape_coverage(shape, *rule, tf, clips);
                Self::composite_paint(canvas, &coverage, paint, inv_tf, resources)?;
            }
            Draw::Stroke {
                shape,
                stroke,
                paint,
            } => {
                let mut segs = edges(&stroke_polylines_device(shape, stroke, tf));
                for clip in clips {
                    segs = intersect_edges(&segs, FillRule::NonZero, clip);
                }
                let mut cov = Coverage::new(self.width, self.height);
                for &s in &segs {
                    cov.add_line(s.0, s.1, s.2, s.3);
                }
                let coverage = cov.finish(FillRule::NonZero);
                Self::composite_paint(canvas, &coverage, paint, inv_tf, resources)?;
            }
            Draw::Shadow {
                shape,
                blur_sigma,
                offset,
                color,
            } => {
                // Exact: shift the shape edges by the offset before coverage,
                // then blur (convolution commutes with translation).
                let tf_off = tf * Affine::translate((offset[0], offset[1]));
                let coverage = self.shape_coverage(shape, FillRule::NonZero, tf_off, clips);
                let blurred = gaussian_blur(&coverage, self.width, self.height, *blur_sigma);
                let src = to_working(color);
                for (px, &c) in canvas.pixels.iter_mut().zip(&blurred) {
                    if c > 0.0 {
                        *px = src_over(*px, src.map(|v| v * c));
                    }
                }
            }
            Draw::Glyphs(run) => {
                let items = glyphs::items_for_glyph_run(run, resources, self.scene_rect)?;
                self.render_items(&items, tf, clips, canvas, resources)?;
            }
            Draw::Image {
                image,
                dst,
                sampling,
            } => {
                let (dw, dh) = (dst.x1 - dst.x0, dst.y1 - dst.y0);
                let coverage =
                    self.shape_coverage(&Shape::Rect(*dst), FillRule::NonZero, tf, clips);
                let img = resources.image(*image)?.clone();
                let (w, h) = (canvas.width, canvas.height);
                for py in 0..h {
                    for px in 0..w {
                        let c = coverage[py * w + px];
                        if c <= 0.0 {
                            continue;
                        }
                        let p = inv_tf * Point::new(px as f64 + 0.5, py as f64 + 0.5);
                        // sample_image takes image-space coordinates: the
                        // image spans [0, w] × [0, h] over `dst`.
                        let u = (p.x - dst.x0) / dw * img.width as f64;
                        let v = (p.y - dst.y0) / dh * img.height as f64;
                        let src = sample_image(&img, u, v, *sampling).map(|x| x * c);
                        let idx = py * w + px;
                        canvas.pixels[idx] = src_over(canvas.pixels[idx], src);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Edges of `shape` under `tf` (used for clip boundaries), flattened in
/// device space.
fn shape_edges(shape: &Shape, tf: Affine) -> Vec<Segment> {
    edges(&shape_polylines(shape, tf))
}

/// The clip stack for `layer`: its own clip (in scene space), empty if none.
fn scene_clip_stack(layer: &Layer, tf: Affine, _w: usize, _h: usize) -> Vec<Vec<Segment>> {
    layer
        .clip
        .as_ref()
        .map(|c| vec![shape_edges(c, tf)])
        .unwrap_or_default()
}
