// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The glyph mask cache and mask rasterization.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use cherenkov::kurbo::{Affine, Vec2};
use cherenkov::{GlyphRun, GlyphStyle};
use skrifa::MetadataProvider;
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::raw::TableProvider;
use skrifa::raw::types::F2Dot14;

use crate::error::RenderError;
use crate::render::coverage::{Operand, rasterize};
use crate::render::lower::GlyphReq;
use crate::render::raster::Edge;

/// A glyph cache key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
    /// Full variation coordinates; hash collisions cannot alias outlines.
    coords: Arc<[i16]>,
    /// Full stroke parameters in exact-bit form, or empty for a fill.
    stroke: Vec<u64>,
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
        let batch: u64 = masks
            .iter()
            .map(|(key, mask)| key_bytes(key) + mask_bytes(mask))
            .sum();
        if batch > self.budget {
            return Err(RenderError::GlyphCacheExhausted);
        }
        if self.bytes + batch > self.budget {
            // Evict least-recently-used entries until the batch fits.
            let mut oldest_first: Vec<(u64, GlyphKey)> = self
                .map
                .iter()
                .map(|(key, entry)| (entry.used, key.clone()))
                .collect();
            oldest_first.sort_unstable_by_key(|(used, _)| *used);
            for (_, key) in oldest_first {
                if self.bytes + batch <= self.budget {
                    break;
                }
                if let Some((stored_key, entry)) = self.map.remove_entry(&key) {
                    self.bytes -= key_bytes(&stored_key) + mask_bytes(&entry.mask);
                }
            }
        }
        for (key, mask) in masks {
            self.tick += 1;
            if let Some((old_key, previous)) = self.map.remove_entry(&key) {
                self.bytes -= key_bytes(&old_key) + mask_bytes(&previous.mask);
            }
            self.bytes += key_bytes(&key) + mask_bytes(&mask);
            self.map.insert(
                key,
                CacheEntry {
                    mask,
                    used: self.tick,
                },
            );
        }
        Ok(())
    }
}

fn key_bytes(key: &GlyphKey) -> u64 {
    u64::try_from(size_of_val(&*key.coords) + key.stroke.capacity() * size_of::<u64>())
        .expect("glyph key allocation fits u64")
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
pub fn glyph_key(
    run: &GlyphRun,
    coords: &Arc<[i16]>,
    glyph: u32,
    subpixel: (f32, f32),
    transform: Affine,
) -> GlyphKey {
    let mut stroke_key = Vec::new();
    if let GlyphStyle::Stroke(stroke) = &run.style {
        stroke_key.extend([
            stroke.width.to_bits(),
            stroke.miter_limit.to_bits(),
            stroke.dash_offset.to_bits(),
            stroke.join as u64,
            stroke.start_cap as u64,
            stroke.end_cap as u64,
        ]);
        stroke_key.extend(stroke.dash_pattern.iter().map(|value| value.to_bits()));
    }
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
        coords: Arc::clone(coords),
        stroke: stroke_key,
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
/// 2x2, offset by the glyph's subpixel offset, and flattened to 0.02
/// device px before exact-area accumulation over the glyph's own bbox.
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
    let outline = outlines
        .get(skrifa::GlyphId::new(req.glyph_id))
        .ok_or_else(|| RenderError::Font(format!("missing outline for glyph {}", req.glyph_id)))?;
    let location: Vec<F2Dot14> = req.coords.iter().map(|c| F2Dot14::from_bits(*c)).collect();
    let mut pen = PathPen {
        path: kurbo::BezPath::new(),
    };
    let settings = DrawSettings::unhinted(
        skrifa::instance::Size::unscaled(),
        skrifa::instance::LocationRef::new(&location),
    );
    outline
        .draw(settings, &mut pen)
        .map_err(|error| RenderError::Font(error.to_string()))?;
    if pen.path.is_empty() {
        return Ok(empty());
    }
    // Stroke width, caps and dashes are in run units, before the glyph's
    // local transform. Flatten only after the complete device transform.
    let scale = f64::from(req.size) / f64::from(upem);
    let [a, b, c, d] = req.matrix;
    let linear = Affine::new([
        f64::from(a),
        f64::from(b),
        f64::from(c),
        f64::from(d),
        0.0,
        0.0,
    ]);
    let path = Affine::scale_non_uniform(scale, -scale) * pen.path;
    let path = match &req.style {
        GlyphStyle::Fill => path,
        GlyphStyle::Stroke(stroke) => kurbo::stroke(
            path,
            stroke,
            &kurbo::StrokeOpts::default(),
            0.02 / super::lower::sigma_max(linear).max(1e-12),
        ),
    };
    let offset = Vec2::new(f64::from(req.subpixel.0), f64::from(req.subpixel.1));
    let edges = super::lower::flatten_edges(Affine::translate(offset) * linear * path, 0.02);
    Ok(mask_from_edges(edges))
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "glyph mask bounds fit device pixel indices and f32 geometry"
)]
fn mask_from_edges(edges: Vec<Edge>) -> GlyphMask {
    let bbox = edges.iter().fold(
        kurbo::Rect::new(f64::MAX, f64::MAX, f64::MIN, f64::MIN),
        |bounds, edge| {
            bounds
                .union_pt(kurbo::Point::new(f64::from(edge.x0), f64::from(edge.y0)))
                .union_pt(kurbo::Point::new(f64::from(edge.x1), f64::from(edge.y1)))
        },
    );
    if edges.is_empty() || bbox.width() <= 0.0 || bbox.height() <= 0.0 {
        return empty();
    }
    let left = bbox.x0.floor() as i32 - 1;
    let top = bbox.y0.floor() as i32 - 1;
    let right = bbox.x1.ceil() as i32 + 1;
    let bottom = bbox.y1.ceil() as i32 + 1;
    let (w, h) = ((right - left) as usize, (bottom - top) as usize);
    // Keep the outline at its original precision for clipped instances.
    let local: Arc<[Edge]> = edges
        .iter()
        .map(|edge| Edge {
            x0: edge.x0 - left as f32,
            y0: edge.y0 - top as f32,
            x1: edge.x1 - left as f32,
            y1: edge.y1 - top as f32,
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
    GlyphMask {
        left,
        top,
        w: w as u32,
        h: h as u32,
        edges: edges.into(),
        cov,
    }
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
            coords: Arc::from([]),
            stroke: Vec::new(),
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
    fn mask_identity_retains_stroke_details_and_variations() {
        let mut run = GlyphRun {
            font: cherenkov::FontId::new(1),
            size: 24.0,
            coords: Vec::new(),
            glyphs: Vec::new(),
            style: GlyphStyle::Stroke(kurbo::Stroke::new(2.0)),
        };
        let coords = Arc::from([]);
        let original = glyph_key(&run, &coords, 36, (0.25, 0.5), Affine::IDENTITY);
        for stroke in [
            kurbo::Stroke::new(2.0).with_join(kurbo::Join::Round),
            kurbo::Stroke::new(2.0).with_caps(kurbo::Cap::Round),
            kurbo::Stroke::new(2.0).with_miter_limit(8.0),
            kurbo::Stroke::new(2.0).with_dashes(0.25, [1.0, 2.0]),
        ] {
            run.style = GlyphStyle::Stroke(stroke);
            assert_ne!(
                original,
                glyph_key(&run, &coords, 36, (0.25, 0.5), Affine::IDENTITY)
            );
        }
        run.style = GlyphStyle::Stroke(kurbo::Stroke::new(2.0));
        assert_ne!(
            original,
            glyph_key(&run, &Arc::from([1_i16]), 36, (0.25, 0.5), Affine::IDENTITY)
        );
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
