// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The glyph mask cache and mask rasterization.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use cherenkov::GlyphRun;
use cherenkov::kurbo::{Affine, PathEl, Point, Vec2};
use skrifa::MetadataProvider;
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::raw::TableProvider;
use skrifa::raw::types::F2Dot14;

use crate::error::RenderError;
use crate::render::lower::GlyphReq;
use crate::render::raster::{Accum, Edge};

/// A glyph cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GlyphKey {
    /// The engine font id.
    font: u64,
    /// The glyph index.
    glyph: u32,
    /// `(size * 64).round()` — 1/64th-pixel size granularity.
    size_bits: u32,
    /// Quantized subpixel position: `(fx * 4) | ((fy * 4) << 4)`.
    subpixel: u8,
    /// f32 bits of the device transform's 2x2.
    matrix: [u32; 4],
    /// Hash of the run's variation coordinates.
    coords_hash: u64,
}

/// A rasterized coverage mask: `w * h` cells anchored at `left`/`top`
/// relative to the glyph's integer device origin.
#[derive(Debug)]
pub struct GlyphMask {
    /// Mask's left edge offset from the integer origin.
    pub left: i32,
    /// Mask's top edge offset from the integer origin.
    pub top: i32,
    /// Mask width in cells.
    pub w: u32,
    /// Mask height in cells.
    pub h: u32,
    /// Coverage, `w * h` cells.
    pub cov: Vec<f32>,
}

/// A resolved glyph mask awaiting the band pass.
pub type GlyphSlot = Arc<OnceLock<Arc<GlyphMask>>>;

/// The glyph mask cache: keyed masks plus byte accounting against the
/// CPU budget.
#[derive(Default)]
pub struct GlyphCache {
    map: HashMap<GlyphKey, Arc<GlyphMask>>,
    bytes: u64,
    budget: u64,
}

impl GlyphCache {
    /// An empty cache bounded by `budget` bytes.
    pub fn new(budget: u64) -> Self {
        Self {
            map: HashMap::new(),
            bytes: 0,
            budget,
        }
    }

    /// A cached mask, if present.
    pub fn get(&self, key: &GlyphKey) -> Option<Arc<GlyphMask>> {
        self.map.get(key).cloned()
    }

    /// Current cached bytes.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Drops every cached mask (`Trim(Critical)` or an over-budget
    /// batch insert, like the GPU atlas's flush-everything policy).
    pub fn clear(&mut self) {
        self.map.clear();
        self.bytes = 0;
    }

    /// Inserts `masks` (one `(key, mask)` pair per missing request).
    /// When the batch would push the cache over budget, everything is
    /// evicted first.
    pub fn insert_batch(&mut self, masks: Vec<(GlyphKey, Arc<GlyphMask>)>) {
        let batch: u64 = masks
            .iter()
            .map(|(_, m)| u64::from(m.w) * u64::from(m.h) * 4)
            .sum();
        if self.bytes + batch > self.budget {
            self.clear();
        }
        for (key, mask) in masks {
            self.bytes += u64::from(mask.w) * u64::from(mask.h) * 4;
            self.map.insert(key, mask);
        }
    }
}

/// The cache key for a glyph at a quantized device position — the same
/// key the GPU slice computes.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "size and subpixel fractions are small non-negative values"
)]
pub fn glyph_key(run: &GlyphRun, glyph: u32, subpixel: (f32, f32), transform: Affine) -> GlyphKey {
    let mut hasher = DefaultHasher::new();
    run.coords.hash(&mut hasher);
    let [a, b, c, d, ..] = transform.as_coeffs();
    GlyphKey {
        font: run.font.raw(),
        glyph,
        size_bits: (run.size * 64.0).round() as u32,
        subpixel: ((subpixel.0 * 4.0) as u8) | (((subpixel.1 * 4.0) as u8) << 4),
        matrix: [
            (a as f32).to_bits(),
            (b as f32).to_bits(),
            (c as f32).to_bits(),
            (d as f32).to_bits(),
        ],
        coords_hash: hasher.finish(),
    }
}

/// An [`OutlinePen`] collecting a glyph outline into a `kurbo::BezPath`.
struct PathPen {
    path: kurbo::BezPath,
}

impl OutlinePen for PathPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.path.move_to((f64::from(x), f64::from(y)));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.path.line_to((f64::from(x), f64::from(y)));
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.path.quad_to(
            (f64::from(cx0), f64::from(cy0)),
            (f64::from(x), f64::from(y)),
        );
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.path.curve_to(
            (f64::from(cx0), f64::from(cy0)),
            (f64::from(cx1), f64::from(cy1)),
            (f64::from(x), f64::from(y)),
        );
    }

    fn close(&mut self) {
        self.path.close_path();
    }
}

/// An empty mask (missing glyph, empty outline).
const fn empty() -> GlyphMask {
    GlyphMask {
        left: 0,
        top: 0,
        w: 0,
        h: 0,
        cov: Vec::new(),
    }
}

/// Rasterizes one glyph's coverage mask, like the GPU's
/// `glyph::rasterize`: the outline is drawn at `Size::unscaled` in font
/// units, y flipped, scaled by `size / upem`, transformed by the run's
/// 2x2, offset by the quantized subpixel, and flattened to 0.05 device
/// px before exact-area accumulation over the glyph's own bbox.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "glyph mask coordinates are small; a/b/c/d/m are affine names"
)]
pub fn rasterize_mask(
    font: &crate::render::FontData,
    req: &GlyphReq,
) -> Result<GlyphMask, RenderError> {
    let font_ref = skrifa::FontRef::from_index(&font.data, font.index)
        .map_err(|e| RenderError::Font(format!("{e}")))?;
    let upem = font_ref
        .head()
        .map_err(|e| RenderError::Font(format!("head: {e}")))?
        .units_per_em();
    let outlines = font_ref.outline_glyphs();
    let Some(outline) = outlines.get(skrifa::GlyphId::new(req.glyph_id)) else {
        return Ok(empty());
    };
    let location: Vec<F2Dot14> = req.coords.iter().map(|c| F2Dot14::from_bits(*c)).collect();
    let mut pen = PathPen {
        path: kurbo::BezPath::new(),
    };
    let settings = DrawSettings::unhinted(
        skrifa::instance::Size::unscaled(),
        skrifa::instance::LocationRef::new(&location),
    );
    if outline.draw(settings, &mut pen).is_err() || pen.path.is_empty() {
        return Ok(empty());
    }
    // Font units to device pixels: y flips, scale is size per em, then
    // the run's transform's linear part.
    let scale = f64::from(req.size) / f64::from(upem);
    let [a, b, c, d] = req.matrix;
    let m = Affine::new([
        f64::from(a),
        f64::from(b),
        f64::from(c),
        f64::from(d),
        0.0,
        0.0,
    ]) * Affine::scale_non_uniform(scale, -scale);
    let (fx, fy) = req.subpixel;
    let offset = Vec2::new(f64::from(fx), f64::from(fy));
    let mut edges: Vec<Edge> = Vec::new();
    let mut bbox = kurbo::Rect::new(f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    let mut last = Point::ORIGIN;
    let mut start = Point::ORIGIN;
    let mut line = |p0: Point, p1: Point| {
        if p0 == p1 {
            return;
        }
        let a = m * p0 + offset;
        let b = m * p1 + offset;
        bbox = bbox.union_pt(a).union_pt(b);
        edges.push(Edge {
            x0: a.x as f32,
            y0: a.y as f32,
            x1: b.x as f32,
            y1: b.y as f32,
        });
    };
    // Every subpath is closed: an open contour is closed implicitly.
    kurbo::flatten(&pen.path, 0.05 / scale.max(1e-6), |el| match el {
        PathEl::MoveTo(p) => {
            line(last, start);
            start = p;
            last = p;
        }
        PathEl::LineTo(p) => {
            line(last, p);
            last = p;
        }
        PathEl::QuadTo(..) | PathEl::CurveTo(..) => unreachable!("flatten emits lines"),
        PathEl::ClosePath => {
            line(last, start);
            last = start;
        }
    });
    line(last, start);
    if edges.is_empty() || bbox.width() <= 0.0 || bbox.height() <= 0.0 {
        return Ok(empty());
    }
    let left = bbox.x0.floor() as i32 - 1;
    let top = bbox.y0.floor() as i32 - 1;
    let right = bbox.x1.ceil() as i32 + 1;
    let bottom = bbox.y1.ceil() as i32 + 1;
    let (w, h) = ((right - left) as usize, (bottom - top) as usize);
    // Rasterize in mask space.
    let mut acc = Accum::new(w, h);
    let (ox, oy) = (left as f32, top as f32);
    for e in &edges {
        acc.draw_line(e.x0 - ox, e.y0 - oy, e.x1 - ox, e.y1 - oy);
    }
    let mut cov = vec![0.0; w * h];
    for y in 0..h {
        acc.coverage_row(y, cherenkov::FillRule::NonZero, 0, w, |x, c| {
            cov[y * w + x] = c;
        });
    }
    Ok(GlyphMask {
        left,
        top,
        w: w as u32,
        h: h as u32,
        cov,
    })
}
