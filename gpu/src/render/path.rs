// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CPU coverage rasterization of general paths: sparse strips into atlas
//! cells, plus the cache key for replaying them.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use cherenkov::{FillRule, ShapeData};
use kurbo::{Affine, BezPath, PathEl, Point, Rect, Shape as _, Vec2};

use crate::render::glyph::{Atlas, PathCell, PathEmit};
use crate::render::raster::Raster;

use super::Encode;

/// Coverage strip height in device rows.
pub const STRIP_H: usize = 4;

/// A run of full columns shorter than this is emitted as a cell rather than
/// splitting a partial run.
const FULL_RUN_MIN: usize = 8;

/// A path whose raster bbox fits inside this edge length is emitted as one
/// whole-bbox cell, skipping strips.
const SMALL_BBOX: f64 = 32.0;

/// Flattening tolerance in device pixels.
pub const FLATTEN: f64 = 0.02;

/// The largest singular value of `t`'s linear part — the worst-case factor
/// by which a local distance error grows under the transform. Mirrors
/// `cherenkov_oracle::path::sigma_max`.
#[must_use]
#[expect(
    clippy::many_single_char_names,
    reason = "a/b/c/d are the conventional affine coefficient names"
)]
pub fn sigma_max(t: Affine) -> f64 {
    let [a, b, c, d, _, _] = t.as_coeffs();
    let p = a.mul_add(a, b * b) + c.mul_add(c, d * d);
    let det = a.mul_add(d, -(b * c));
    let disc = p.mul_add(p, (-4.0 * det) * det).sqrt();
    p.midpoint(disc).sqrt()
}

/// A semantic shape as a local `BezPath`, or `None` for a `ContinuousRect`,
/// which has no `kurbo` path form.
#[must_use]
pub fn shape_path(shape: &ShapeData, tolerance: f64) -> Option<BezPath> {
    match shape {
        ShapeData::Rect(r) => Some(r.to_path(tolerance)),
        ShapeData::RoundedRect(rr) => Some(rr.to_path(tolerance)),
        ShapeData::Circle(c) => Some(c.to_path(tolerance)),
        ShapeData::Ellipse(e) => Some(e.to_path(tolerance)),
        ShapeData::Line(l) => Some(l.to_path(tolerance)),
        ShapeData::Continuous(_) => None,
        ShapeData::Path { elements, .. } => Some(BezPath::from_vec(elements.clone())),
    }
}

/// Flattens `path` (already in device space) at `tolerance` px into
/// `(x0, y0, x1, y1)` segments and their device bbox. Every subpath is
/// closed: fills close open contours implicitly.
#[expect(clippy::cast_possible_truncation, reason = "device coords are f32")]
pub fn flatten_segments(path: &BezPath, tolerance: f64) -> (Vec<(f32, f32, f32, f32)>, Rect) {
    let mut segments = Vec::new();
    let mut bbox = Rect::new(f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    let mut last = Point::ORIGIN;
    let mut start = Point::ORIGIN;
    let mut line = |p0: Point, p1: Point| {
        if p0 == p1 {
            return;
        }
        bbox = bbox.union_pt(p0).union_pt(p1);
        segments.push((p0.x as f32, p0.y as f32, p1.x as f32, p1.y as f32));
    };
    kurbo::flatten(path.clone(), tolerance, |el| match el {
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
    (segments, bbox)
}

/// A coverage grid over an integer device-space rect.
pub struct Coverage {
    /// Device origin of the grid.
    pub x: f64,
    /// Device origin of the grid.
    pub y: f64,
    /// Columns.
    pub w: usize,
    /// Rows.
    pub h: usize,
    /// True when the rasterization bbox was clipped by the surface rect:
    /// the coverage is only valid at the offset it was rasterized under.
    pub clipped: bool,
    /// Row-major coverage in `[0, 1]`.
    pub data: Vec<f32>,
}

/// Rasterizes flattened device-space `segments` spanning `bbox` under
/// `rule`, over `bbox` inflated by 1 px and clipped to the surface rect.
/// `None` when the path misses the surface entirely.
#[expect(clippy::cast_possible_truncation)]
#[expect(clippy::cast_sign_loss)]
#[expect(clippy::cast_precision_loss)]
pub fn rasterize(
    segments: &[(f32, f32, f32, f32)],
    bbox: Rect,
    surface: (f64, f64),
    rule: FillRule,
) -> Option<Coverage> {
    let clip = Rect::new(0.0, 0.0, surface.0, surface.1);
    let inflated = bbox.inflate(1.0, 1.0);
    let r = inflated.intersect(clip);
    let clipped = r != inflated;
    let x0 = r.x0.floor();
    let y0 = r.y0.floor();
    let w = (r.x1.ceil() - x0).max(0.0) as usize;
    let h = (r.y1.ceil() - y0).max(0.0) as usize;
    if w == 0 || h == 0 {
        return None;
    }
    let (ox, oy) = (x0 as f32, y0 as f32);
    let mut raster = Raster::new(w, h);
    for &(sx0, sy0, sx1, sy1) in segments {
        let (x0, y0, x1, y1) = (sx0 - ox, sy0 - oy, sx1 - ox, sy1 - oy);
        let Some((x0, y0, x1, y1)) = clip_x(x0, y0, x1, y1, w as f32) else {
            continue;
        };
        raster.draw_line(x0, y0, x1, y1);
    }
    Some(Coverage {
        x: x0,
        y: y0,
        w,
        h,
        clipped,
        data: raster.coverage_rule(rule),
    })
}

/// Clips a segment to `0 <= x <= w` in raster space, keeping the same
/// winding deposit the un-clipped edge would have made on the covered
/// columns. `None` for segments outside the range or with non-finite
/// coordinates. Row clipping is `Raster`'s job: `y` is left untouched.
fn clip_x(x0: f32, y0: f32, x1: f32, y1: f32, w: f32) -> Option<(f32, f32, f32, f32)> {
    if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
        return None;
    }
    let dx = x1 - x0;
    if dx.abs() <= f32::EPSILON {
        return (x0 >= 0.0 && x0 <= w).then_some((x0, y0, x1, y1));
    }
    let (enter, exit) = if dx > 0.0 { (0.0, w) } else { (w, 0.0) };
    let t0 = ((enter - x0) / dx).max(0.0);
    let t1 = ((exit - x0) / dx).min(1.0);
    if t0 >= t1 {
        return None;
    }
    let (dy, ix) = (y1 - y0, dx);
    Some((
        ix.mul_add(t0, x0),
        dy.mul_add(t0, y0),
        ix.mul_add(t1, x0),
        dy.mul_add(t1, y0),
    ))
}

/// f32 coverage → `R8Unorm` texels.
#[expect(clippy::cast_possible_truncation)]
#[expect(clippy::cast_sign_loss)]
fn texels(coverage: &[f32]) -> Vec<u8> {
    coverage
        .iter()
        .map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

/// Emits `coverage` as spans and atlas cells: one cell when the grid is
/// small, otherwise strips of [`STRIP_H`] rows split into full-column span
/// runs and partial-column cells.
#[expect(clippy::cast_possible_truncation)]
#[expect(clippy::cast_precision_loss)]
#[expect(
    clippy::float_cmp,
    reason = "a column is 'full' exactly when coverage clamped to 1.0"
)]
pub fn emit(
    coverage: &Coverage,
    atlas: &mut Atlas,
    queue: &wgpu::Queue,
) -> Result<PathEmit, Encode> {
    let mut out = PathEmit::default();
    let upload_cell = |atlas: &mut Atlas,
                       cells: &mut Vec<PathCell>,
                       x: usize,
                       y: usize,
                       w: usize,
                       h: usize|
     -> Result<(), Encode> {
        let (Ok(w32), Ok(h32)) = (u32::try_from(w), u32::try_from(h)) else {
            return Err(Encode::AtlasFull);
        };
        let Some((cx, cy)) = atlas.alloc(w32, h32) else {
            return Err(Encode::AtlasFull);
        };
        let mut rows = Vec::with_capacity(w * h);
        for row in 0..h {
            rows.extend_from_slice(&coverage.data[(y + row) * coverage.w + x..][..w]);
        }
        atlas.write(queue, cx, cy, w32, h32, &texels(&rows));
        cells.push(PathCell {
            rect: [
                (coverage.x + x as f64) as f32,
                (coverage.y + y as f64) as f32,
                (coverage.x + (x + w) as f64) as f32,
                (coverage.y + (y + h) as f64) as f32,
            ],
            x: cx as u16,
            y: cy as u16,
        });
        Ok(())
    };
    if coverage.w as f64 <= SMALL_BBOX && coverage.h as f64 <= SMALL_BBOX {
        upload_cell(atlas, &mut out.cells, 0, 0, coverage.w, coverage.h)?;
        return Ok(out);
    }
    for sy in (0..coverage.h).step_by(STRIP_H) {
        let sh = STRIP_H.min(coverage.h - sy);
        // Classify columns: 0 empty, 1 full, 2 partial.
        let mut class = vec![0u8; coverage.w];
        for (x, c) in class.iter_mut().enumerate() {
            let mut empty = true;
            let mut full = true;
            for row in 0..sh {
                let v = coverage.data[(sy + row) * coverage.w + x];
                empty &= v == 0.0;
                full &= v == 1.0;
            }
            *c = if empty {
                0
            } else if full {
                1
            } else {
                2
            };
        }
        // Demote full runs narrower than `FULL_RUN_MIN` to partial so a
        // partial run is not fragmented by isolated full columns.
        let mut x = 0;
        while x < coverage.w {
            if class[x] == 1 {
                let end = (x + 1..coverage.w)
                    .find(|&i| class[i] != 1)
                    .unwrap_or(coverage.w);
                if end - x < FULL_RUN_MIN {
                    class[x..end].fill(2);
                }
                x = end;
            } else {
                x += 1;
            }
        }
        // Emit runs: all-full runs become spans, the rest cells.
        let mut x = 0;
        while x < coverage.w {
            if class[x] == 0 {
                x += 1;
                continue;
            }
            let start = x;
            if class[x] == 1 {
                while x < coverage.w && class[x] == 1 {
                    x += 1;
                }
                out.spans.push([
                    (coverage.x + start as f64) as f32,
                    (coverage.y + sy as f64) as f32,
                    (coverage.x + x as f64) as f32,
                    (coverage.y + (sy + sh) as f64) as f32,
                ]);
            } else {
                while x < coverage.w && class[x] != 0 {
                    x += 1;
                }
                upload_cell(atlas, &mut out.cells, start, sy, x - start, sh)?;
            }
        }
    }
    Ok(out)
}

/// A stable hash of a shape's field bits, for strokes of non-path shapes
/// which have no element list to hash.
fn hash_shape_fields(hasher: &mut DefaultHasher, shape: &ShapeData) {
    fn v(hasher: &mut DefaultHasher, x: f64) {
        x.to_bits().hash(hasher);
    }
    fn rect(hasher: &mut DefaultHasher, r: &Rect) {
        v(hasher, r.x0);
        v(hasher, r.y0);
        v(hasher, r.x1);
        v(hasher, r.y1);
    }
    match shape {
        ShapeData::Rect(r) => {
            0u8.hash(hasher);
            rect(hasher, r);
        }
        ShapeData::RoundedRect(rr) => {
            1u8.hash(hasher);
            rect(hasher, &rr.rect());
            let radii = rr.radii();
            v(hasher, radii.top_left);
            v(hasher, radii.top_right);
            v(hasher, radii.bottom_right);
            v(hasher, radii.bottom_left);
        }
        ShapeData::Continuous(c) => {
            2u8.hash(hasher);
            rect(hasher, &c.rect);
            let radii = c.radii;
            v(hasher, radii.top_left);
            v(hasher, radii.top_right);
            v(hasher, radii.bottom_right);
            v(hasher, radii.bottom_left);
            v(hasher, c.smoothing);
        }
        ShapeData::Circle(c) => {
            3u8.hash(hasher);
            v(hasher, c.center.x);
            v(hasher, c.center.y);
            v(hasher, c.radius);
        }
        ShapeData::Ellipse(e) => {
            4u8.hash(hasher);
            let center = e.center();
            let radii = e.radii();
            v(hasher, center.x);
            v(hasher, center.y);
            v(hasher, radii.x);
            v(hasher, radii.y);
            v(hasher, e.rotation());
        }
        ShapeData::Line(l) => {
            5u8.hash(hasher);
            v(hasher, l.p0.x);
            v(hasher, l.p0.y);
            v(hasher, l.p1.x);
            v(hasher, l.p1.y);
        }
        ShapeData::Path { elements, rule } => {
            6u8.hash(hasher);
            (*rule as u8).hash(hasher);
            hash_elements_into(hasher, elements);
        }
    }
}

/// A stable hash of a stroked draw: the outline's source shape plus every
/// stroke parameter and the local flatten tolerance, tagged so it never
/// collides with a fill of the same geometry.
pub fn hash_stroke(shape: &ShapeData, stroke: &kurbo::Stroke, tolerance: f64) -> u64 {
    let mut hasher = DefaultHasher::new();
    2u64.hash(&mut hasher);
    hash_shape_fields(&mut hasher, shape);
    stroke.width.to_bits().hash(&mut hasher);
    (stroke.join as u8).hash(&mut hasher);
    stroke.miter_limit.to_bits().hash(&mut hasher);
    (stroke.start_cap as u8).hash(&mut hasher);
    (stroke.end_cap as u8).hash(&mut hasher);
    (stroke.dash_pattern.len() as u64).hash(&mut hasher);
    for v in &stroke.dash_pattern {
        v.to_bits().hash(&mut hasher);
    }
    stroke.dash_offset.to_bits().hash(&mut hasher);
    tolerance.to_bits().hash(&mut hasher);
    hasher.finish()
}

/// A stable hash of a local path's element list, tagged by draw mode so a
/// fill, an even-odd fill and a stroke of the same outline never collide.
pub fn hash_elements(elements: &[PathEl], tag: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    tag.hash(&mut hasher);
    hash_elements_into(&mut hasher, elements);
    hasher.finish()
}

fn hash_elements_into(hasher: &mut DefaultHasher, elements: &[PathEl]) {
    fn point(hasher: &mut DefaultHasher, disc: u8, p: Point) {
        disc.hash(hasher);
        p.x.to_bits().hash(hasher);
        p.y.to_bits().hash(hasher);
    }
    for el in elements {
        match el {
            PathEl::MoveTo(p) => point(hasher, 0, *p),
            PathEl::LineTo(p) => point(hasher, 1, *p),
            PathEl::QuadTo(c, p) => {
                point(hasher, 2, *c);
                point(hasher, 2, *p);
            }
            PathEl::CurveTo(c0, c1, p) => {
                point(hasher, 3, *c0);
                point(hasher, 3, *c1);
                point(hasher, 3, *p);
            }
            PathEl::ClosePath => 4u8.hash(hasher),
        }
    }
}

/// A path draw's cache key and placement.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    /// Cache key: content hash + matrix + quantized subpixel + surface.
    pub key: u64,
    /// `key` plus the integer translation: coverage clipped by the
    /// surface is only valid at this offset.
    pub key_exact: u64,
    /// The transform to rasterize under: the true 2x2 and the translation
    /// snapped to the 1/4 px grid.
    pub raster: Affine,
    /// The integer translation the cache's stored rects are relative to.
    pub offset: Vec2,
}

/// Builds the [`Placement`] for a draw under `transform` on a
/// `surface`-pixel target. The key holds the 2x2, the translation's
/// fractional part quantized to 1/4 px and the surface size, so identical
/// geometry at different integer translations replays the same emission.
#[expect(clippy::cast_possible_truncation)]
#[expect(clippy::cast_sign_loss)]
#[expect(
    clippy::many_single_char_names,
    reason = "a..f are the conventional affine coefficient names"
)]
pub fn placement(content_hash: u64, transform: Affine, surface: (u32, u32)) -> Placement {
    let [a, b, c, d, e, f] = transform.as_coeffs();
    let ix = e.floor();
    let iy = f.floor();
    let qx = ((e - ix) * 4.0).floor() / 4.0;
    let qy = ((f - iy) * 4.0).floor() / 4.0;
    let mut hasher = DefaultHasher::new();
    content_hash.hash(&mut hasher);
    for v in [a, b, c, d] {
        (v as f32).to_bits().hash(&mut hasher);
    }
    ((qx * 4.0) as u8 | (((qy * 4.0) as u8) << 4)).hash(&mut hasher);
    surface.hash(&mut hasher);
    let key = hasher.finish();
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    (ix as i64).hash(&mut hasher);
    (iy as i64).hash(&mut hasher);
    Placement {
        key,
        key_exact: hasher.finish(),
        raster: Affine::new([a, b, c, d, ix + qx, iy + qy]),
        offset: Vec2::new(ix, iy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_odd_folds_winding_into_a_triangle_wave() {
        // Two identical 4x4 squares drawn twice: winding 2 in the overlap,
        // so even-odd leaves a hole where non-zero fills.
        let mut raster = Raster::new(8, 8);
        for _ in 0..2 {
            for (x0, y0, x1, y1) in [
                (1.0, 1.0, 5.0, 1.0),
                (5.0, 1.0, 5.0, 5.0),
                (5.0, 5.0, 1.0, 5.0),
                (1.0, 5.0, 1.0, 1.0),
            ] {
                raster.draw_line(x0, y0, x1, y1);
            }
        }
        let eo = raster.coverage_rule(FillRule::EvenOdd);
        let nz = raster.coverage_rule(FillRule::NonZero);
        assert!((nz[3 * 8 + 3] - 1.0).abs() < 1e-6);
        assert!(
            eo[3 * 8 + 3].abs() < 1e-6,
            "even-odd hole: {}",
            eo[3 * 8 + 3]
        );
        // Partial coverage folds the same way: half coverage stays half.
        let mut edge = Raster::new(4, 4);
        edge.draw_line(0.5, 0.0, 0.5, 3.0);
        let eo = edge.coverage_rule(FillRule::EvenOdd);
        assert!((eo[0] - 0.5).abs() < 1e-6, "edge: {}", eo[0]);
    }
}
