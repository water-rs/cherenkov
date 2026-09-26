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
use crate::render::coverage::{Operand, rasterize};
use crate::render::lower::GlyphReq;
use crate::render::raster::Edge;

/// A glyph cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GlyphKey {
    /// The engine font id.
    font: u64,
    /// The glyph index.
    glyph: u32,
    /// Exact f32 font size; fractional sizes must not alias in the cache.
    size_bits: u32,
    /// Exact subpixel position: f32 bits of the fractional offset.
    subpixel: [u32; 2],
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
    /// Outline relative to the integer glyph origin, retained for geometric clipping.
    pub edges: Arc<[Edge]>,
    /// Coverage, `w * h` cells.
    pub cov: Vec<f32>,
}

/// A resolved glyph mask awaiting the band pass.
pub type GlyphSlot = Arc<OnceLock<Arc<GlyphMask>>>;

/// A cached mask plus its last-use counter for eviction.
#[derive(Debug)]
struct CacheEntry {
    mask: Arc<GlyphMask>,
    /// The `tick` at which the mask was last fetched or inserted.
    used: u64,
}

/// The glyph mask cache: keyed masks plus byte accounting against the
/// CPU budget. Over-budget batches evict least-recently-used entries.
#[derive(Default)]
pub struct GlyphCache {
    map: HashMap<GlyphKey, CacheEntry>,
    bytes: u64,
    budget: u64,
    /// Monotonically increasing use counter.
    tick: u64,
}

impl GlyphCache {
    /// An empty cache bounded by `budget` bytes.
    pub fn new(budget: u64) -> Self {
        Self {
            map: HashMap::new(),
            bytes: 0,
            budget,
            tick: 0,
        }
    }

    /// A cached mask, if present, marked most recently used.
    pub fn get(&mut self, key: &GlyphKey) -> Option<Arc<GlyphMask>> {
        let entry = self.map.get_mut(key)?;
        self.tick += 1;
        entry.used = self.tick;
        Some(Arc::clone(&entry.mask))
    }

    /// Current cached bytes.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Drops every cached mask (`Trim(Critical)`).
    pub fn clear(&mut self) {
        self.map.clear();
        self.bytes = 0;
    }

    /// Inserts `masks` (one `(key, mask)` pair per missing request),
    /// evicting least-recently-used entries until the batch fits.
    ///
    /// # Errors
    /// [`RenderError::GlyphCacheExhausted`] when the batch alone exceeds
    /// the budget: the frame's glyph set cannot be cached.
    pub fn insert_batch(
        &mut self,
        masks: Vec<(GlyphKey, Arc<GlyphMask>)>,
    ) -> Result<(), RenderError> {
        let batch: u64 = masks.iter().map(|(_, m)| mask_bytes(m)).sum();
        if batch > self.budget {
            return Err(RenderError::GlyphCacheExhausted);
        }
        if self.bytes + batch > self.budget {
            // Evict least-recently-used entries until the batch fits.
            let mut oldest_first: Vec<(u64, GlyphKey)> = self
                .map
                .iter()
                .map(|(key, entry)| (entry.used, *key))
                .collect();
            oldest_first.sort_unstable_by_key(|(used, _)| *used);
            for (_, key) in oldest_first {
                if self.bytes + batch <= self.budget {
                    break;
                }
                if let Some(entry) = self.map.remove(&key) {
                    self.bytes -= mask_bytes(&entry.mask);
                }
            }
        }
        for (key, mask) in masks {
            self.tick += 1;
            self.bytes += mask_bytes(&mask);
            if let Some(previous) = self.map.insert(
                key,
                CacheEntry {
                    mask,
                    used: self.tick,
                },
            ) {
                self.bytes -= mask_bytes(&previous.mask);
            }
        }
        Ok(())
    }
}

fn mask_bytes(mask: &GlyphMask) -> u64 {
    u64::try_from(mask.cov.capacity() * size_of::<f32>() + size_of_val(&*mask.edges))
        .expect("glyph allocation fits u64")
}

/// The cache key for a glyph at an exact subpixel position.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the stored linear transform uses the same f32 precision as glyph rasterization"
)]
pub fn glyph_key(run: &GlyphRun, glyph: u32, subpixel: (f32, f32), transform: Affine) -> GlyphKey {
    let mut hasher = DefaultHasher::new();
    run.coords.hash(&mut hasher);
    let [a, b, c, d, ..] = transform.as_coeffs();
    GlyphKey {
        font: run.font.raw(),
        glyph,
        size_bits: run.size.to_bits(),
        subpixel: [subpixel.0.to_bits(), subpixel.1.to_bits()],
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
fn empty() -> GlyphMask {
    GlyphMask {
        left: 0,
        top: 0,
        w: 0,
        h: 0,
        cov: Vec::new(),
        edges: Arc::from([]),
    }
}

/// Rasterizes one glyph's coverage mask, like the GPU's
/// `glyph::rasterize`: the outline is drawn at `Size::unscaled` in font
/// units, y flipped, scaled by `size / upem`, transformed by the run's
/// 2x2, offset by the glyph's subpixel offset, and flattened to 0.05
/// device px before exact-area accumulation over the glyph's own bbox.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "glyph mask coordinates are small; a/b/c/d/m are affine names"
)]
#[expect(
    clippy::too_many_lines,
    reason = "outline walk plus mask accumulation is one pass"
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
    // Keep the outline at its original precision for clipped instances.
    let local: Arc<[Edge]> = edges
        .iter()
        .map(|e| Edge {
            x0: e.x0 - left as f32,
            y0: e.y0 - top as f32,
            x1: e.x1 - left as f32,
            y1: e.y1 - top as f32,
        })
        .collect();
    let coverage = rasterize(
        &[Operand {
            edges: local,
            rule: cherenkov::FillRule::NonZero,
        }],
        w,
        h,
    );
    let mut cov = vec![0.0; w * h];
    for y in 0..h {
        for span in coverage.row(y) {
            for x in span.columns.clone() {
                cov[y * w + x] = span.at(x);
            }
        }
    }
    Ok(GlyphMask {
        left,
        top,
        w: w as u32,
        h: h as u32,
        edges: edges.into(),
        cov,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u32) -> GlyphKey {
        GlyphKey {
            font: 0,
            glyph: n,
            size_bits: 0,
            subpixel: [0; 2],
            matrix: [0; 4],
            coords_hash: 0,
        }
    }

    /// A mask of `bytes` coverage bytes and no outline.
    fn mask(bytes: usize) -> Arc<GlyphMask> {
        Arc::new(GlyphMask {
            left: 0,
            top: 0,
            w: 0,
            h: 0,
            edges: Arc::from(Vec::new()),
            cov: vec![0.0; bytes / 4],
        })
    }

    #[test]
    fn an_over_budget_batch_evicts_least_recently_used() {
        let mut cache = GlyphCache::new(100);
        cache
            .insert_batch(vec![(key(1), mask(40)), (key(2), mask(40))])
            .expect("fits");
        // Touch key 1 so key 2 is least recently used.
        assert!(cache.get(&key(1)).is_some());
        cache
            .insert_batch(vec![(key(3), mask(40))])
            .expect("fits after eviction");
        assert!(cache.get(&key(1)).is_some(), "recently used stays");
        assert!(cache.get(&key(2)).is_none(), "least recently used evicted");
        assert!(cache.get(&key(3)).is_some(), "new entry cached");
        assert!(cache.bytes() <= 100, "within budget");
    }

    #[test]
    fn a_batch_larger_than_the_budget_errors() {
        let mut cache = GlyphCache::new(100);
        let error = cache
            .insert_batch(vec![(key(1), mask(60)), (key(2), mask(60))])
            .expect_err("frame glyph set exceeds the cache budget");
        assert!(matches!(error, RenderError::GlyphCacheExhausted));
    }
}
