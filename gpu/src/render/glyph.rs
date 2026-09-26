// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The glyph atlas and glyph-run lowering.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use kurbo::{Affine, PathEl, Vec2};
use skrifa::MetadataProvider;
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::raw::TableProvider;
use skrifa::raw::types::F2Dot14;

use crate::error::RenderError;
use crate::render::raster::Raster;

/// Initial atlas edge length.
const ATLAS_START: u32 = 1024;
/// Largest atlas edge length.
const ATLAS_MAX: u32 = 4096;
/// Texels of padding around each cell.
const PAD: u32 = 1;

/// Registered font data: the file bytes and collection index.
pub struct FontData {
    pub data: Arc<[u8]>,
    pub index: u32,
}

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

/// An atlas cell.
#[derive(Clone, Copy, Debug, Default)]
pub struct Entry {
    /// Texel origin.
    pub x: u16,
    /// Texel origin.
    pub y: u16,
    /// Cell size in texels; 0 for an empty outline.
    pub w: u16,
    /// Cell size in texels.
    pub h: u16,
    /// Offset of the cell's left edge from the glyph's integer device origin.
    pub left: i32,
    /// Offset of the cell's top edge from the glyph's integer device origin.
    pub top: i32,
}

/// One shelf of the packer: a row of cells sharing a height class.
struct Shelf {
    y: u32,
    h: u32,
    x: u32,
}

/// The `R8Unorm` coverage atlas.
pub struct Atlas {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: u32,
    /// The largest atlas edge the GPU budget allows.
    cap: u32,
    /// Bumped whenever the texture is recreated, so stale bind groups are
    /// rebuilt.
    generation: u64,
    shelves: Vec<Shelf>,
    map: HashMap<GlyphKey, Entry>,
    /// Sum of cell texels, an approximation of the CPU cache size.
    cpu_bytes: u64,
}

impl Atlas {
    /// A new `ATLAS_START` atlas, capped by the GPU byte budget (one texel
    /// per byte).
    pub fn new(device: &wgpu::Device, budget: u64) -> Self {
        let (texture, view) = Self::allocate(device, ATLAS_START);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            clippy::cast_sign_loss,
            reason = "atlas sizes are small"
        )]
        let cap = ATLAS_MAX
            .min((budget as f64).sqrt() as u32)
            .max(ATLAS_START);
        Self {
            texture,
            view,
            size: ATLAS_START,
            cap,
            generation: 0,
            shelves: Vec::new(),
            map: HashMap::new(),
            cpu_bytes: 0,
        }
    }

    /// Increments whenever the atlas texture is recreated.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    fn allocate(device: &wgpu::Device, size: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("glyph atlas"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// The texture view bound in group 0.
    pub const fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Atlas byte size on the GPU.
    pub fn gpu_bytes(&self) -> u64 {
        u64::from(self.size) * u64::from(self.size)
    }

    /// Approximate CPU-side cache bytes.
    pub const fn cpu_bytes(&self) -> u64 {
        self.cpu_bytes
    }

    /// Clears every entry without freeing the texture.
    pub fn clear(&mut self) {
        self.map.clear();
        self.shelves.clear();
        self.cpu_bytes = 0;
    }

    /// The atlas edge in texels.
    pub const fn size(&self) -> u32 {
        self.size
    }

    /// The largest atlas edge the budget allows.
    pub const fn cap(&self) -> u32 {
        self.cap
    }

    /// Doubles the atlas up to the cap, dropping every cached entry.
    pub fn grow(&mut self, device: &wgpu::Device) {
        let size = (self.size * 2).min(self.cap);
        if size == self.size {
            return;
        }
        let (texture, view) = Self::allocate(device, size);
        self.texture = texture;
        self.view = view;
        self.size = size;
        self.generation += 1;
        self.clear();
    }

    /// Reserves a `w` × `h` cell, or `None` when it does not fit. Never
    /// evicts: growth and clearing are the caller's decision.
    fn alloc(&mut self, w: u32, h: u32) -> Option<(u32, u32)> {
        // Shelf height classes are multiples of 8.
        let class = (h + 2 * PAD).div_ceil(8) * 8;
        for shelf in &mut self.shelves {
            if shelf.h == class && shelf.x + w + 2 * PAD <= self.size {
                let x = shelf.x + PAD;
                shelf.x += w + 2 * PAD;
                return Some((x, shelf.y + PAD));
            }
        }
        let top = self.shelves.last().map_or(0, |s| s.y + s.h);
        if top + class <= self.size {
            self.shelves.push(Shelf {
                y: top,
                h: class,
                x: w + 2 * PAD,
            });
            return Some((PAD, top + PAD));
        }
        None
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

/// Rasterizes one glyph into the atlas and returns its entry, uploading the
/// coverage texels through `queue`.
///
/// Returns [`RenderError::AtlasFull`] when the glyph's cell does not fit.
#[expect(clippy::too_many_arguments)]
#[expect(clippy::too_many_lines)]
#[expect(clippy::many_single_char_names)]
#[expect(clippy::cast_possible_truncation)]
#[expect(clippy::cast_sign_loss)]
#[expect(clippy::cast_precision_loss)]
pub fn rasterize(
    queue: &wgpu::Queue,
    atlas: &mut Atlas,
    font: &FontData,
    key: GlyphKey,
    glyph_id: u32,
    size: f32,
    subpixel: (f32, f32),
    transform: Affine,
    coords: &[i16],
) -> Result<Entry, RenderError> {
    let font_ref = skrifa::FontRef::from_index(&font.data, font.index)
        .map_err(|e| RenderError::Font(format!("{e}")))?;
    let upem = font_ref
        .head()
        .map_err(|e| RenderError::Font(format!("head: {e}")))?
        .units_per_em();
    let outlines = font_ref.outline_glyphs();
    let Some(outline) = outlines.get(skrifa::GlyphId::new(glyph_id)) else {
        let entry = Entry::default();
        atlas.map.insert(key, entry);
        return Ok(entry);
    };
    let location: Vec<F2Dot14> = coords.iter().map(|c| F2Dot14::from_bits(*c)).collect();
    let mut pen = PathPen {
        path: kurbo::BezPath::new(),
    };
    let settings = DrawSettings::unhinted(
        skrifa::instance::Size::unscaled(),
        skrifa::instance::LocationRef::new(&location),
    );
    if outline.draw(settings, &mut pen).is_err() {
        let entry = Entry::default();
        atlas.map.insert(key, entry);
        return Ok(entry);
    }
    if pen.path.is_empty() {
        let entry = Entry::default();
        atlas.map.insert(key, entry);
        return Ok(entry);
    }
    // Font units to device pixels: y flips, scale is size per em, then the
    // run's transform's linear part.
    let scale = f64::from(size) / f64::from(upem);
    let [a, b, c, d, ..] = transform.as_coeffs();
    let m = Affine::new([a, b, c, d, 0.0, 0.0]) * Affine::scale_non_uniform(scale, -scale);
    let (fx, fy) = subpixel;
    // Flatten to 0.05 px in device space.
    let mut segments: Vec<(f32, f32, f32, f32)> = Vec::new();
    let mut bbox = kurbo::Rect::new(f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    let mut last = kurbo::Point::ORIGIN;
    let mut start = kurbo::Point::ORIGIN;
    let offset = Vec2::new(f64::from(fx), f64::from(fy));
    let mut line = |p0: kurbo::Point, p1: kurbo::Point| {
        if p0 == p1 {
            return;
        }
        let a = m * p0 + offset;
        let b = m * p1 + offset;
        bbox = bbox.union_pt(a).union_pt(b);
        segments.push((a.x as f32, a.y as f32, b.x as f32, b.y as f32));
    };
    // Flatten to 0.05 px in device space: `m`'s largest column norm is the
    // worst-case factor a font-unit error grows by.
    let [ma, mb, mc, md, ..] = m.as_coeffs();
    let lmax = ma.hypot(mb).max(mc.hypot(md)).max(1e-9);
    let tol = 0.05 / lmax;
    // Every subpath is closed: an open contour is closed implicitly.
    kurbo::flatten(&pen.path, tol, |el| match el {
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
    if segments.is_empty() || bbox.width() <= 0.0 || bbox.height() <= 0.0 {
        let entry = Entry::default();
        atlas.map.insert(key, entry);
        return Ok(entry);
    }
    let left = bbox.x0.floor() as i32 - 1;
    let top = bbox.y0.floor() as i32 - 1;
    let right = bbox.x1.ceil() as i32 + 1;
    let bottom = bbox.y1.ceil() as i32 + 1;
    let w = (right - left) as u32;
    let h = (bottom - top) as u32;
    let Some((cx, cy)) = atlas.alloc(w, h) else {
        return Err(RenderError::AtlasFull);
    };
    // Rasterize in cell space.
    let mut raster = Raster::new(w as usize, h as usize);
    let ox = left as f32;
    let oy = top as f32;
    for (x0, y0, x1, y1) in segments {
        raster.draw_line(x0 - ox, y0 - oy, x1 - ox, y1 - oy);
    }
    let coverage = raster.coverage();
    let texels: Vec<u8> = coverage
        .iter()
        .map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    // `write_texture` needs rows padded to 256 bytes.
    let pitch = w.div_ceil(256) * 256;
    let mut staging = vec![0u8; (pitch * h) as usize];
    for (row, line) in texels.chunks_exact(w as usize).enumerate() {
        staging[row * pitch as usize..row * pitch as usize + w as usize].copy_from_slice(line);
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &atlas.texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: cx, y: cy, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        &staging,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(pitch),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    atlas.cpu_bytes += u64::from(w) * u64::from(h);
    let entry = Entry {
        x: cx as u16,
        y: cy as u16,
        w: w as u16,
        h: h as u16,
        left,
        top,
    };
    atlas.map.insert(key, entry);
    Ok(entry)
}

/// The cache key for a glyph at a quantized device position.
#[expect(clippy::cast_possible_truncation)]
#[expect(clippy::cast_sign_loss)]
pub fn glyph_key(
    run: &cherenkov::GlyphRun,
    glyph: u32,
    subpixel: (f32, f32),
    transform: Affine,
) -> GlyphKey {
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

/// The atlas's map lookup.
impl Atlas {
    pub fn get(&self, key: &GlyphKey) -> Option<Entry> {
        self.map.get(key).copied()
    }
}
