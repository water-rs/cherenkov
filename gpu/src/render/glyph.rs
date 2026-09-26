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
    /// Built font-space `COLRv1` pictures, per `(glyph id, coords hash,
    /// paint hash)` — content is size-independent, so it is keyed without
    /// the placement. Interior mutability, not shared: parallel lowering
    /// works on a per-thread [`Self::snapshot`].
    pub colr: std::cell::RefCell<HashMap<(u32, u64, u64), cherenkov::Picture>>,
}

impl FontData {
    /// A per-thread copy: shares the font bytes, clones the COLR cache.
    /// Workers each own one so `colr` can stay a plain `RefCell`.
    pub fn snapshot(&self) -> Self {
        Self {
            data: self.data.clone(),
            index: self.index,
            colr: std::cell::RefCell::new(self.colr.borrow().clone()),
        }
    }
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

/// One atlas cell emitted by the path rasterizer: a device-space quad
/// sampling `w` × `h` coverage texels at `(x, y)`.
#[derive(Clone, Copy, Debug)]
pub struct PathCell {
    /// Device-space quad `(x0, y0, x1, y1)`.
    pub rect: [f32; 4],
    /// Atlas texel origin.
    pub x: u16,
    /// Atlas texel origin.
    pub y: u16,
}

/// What a cached path draw replays: full-coverage spans and atlas cells,
/// in device space shifted by the cache's stored offset.
#[derive(Clone, Debug, Default)]
pub struct PathEmit {
    /// Device rectangles of contiguous full-coverage columns.
    pub spans: Vec<[f32; 4]>,
    /// Partial-coverage atlas cells.
    pub cells: Vec<PathCell>,
}

impl PathEmit {
    /// This emission shifted by `(dx, dy)` device pixels.
    #[must_use]
    #[expect(clippy::cast_possible_truncation, reason = "device coords are f32")]
    pub fn translated(&self, dx: f64, dy: f64) -> Self {
        let shift = |r: [f32; 4]| {
            [
                f64::from(r[0]) + dx,
                f64::from(r[1]) + dy,
                f64::from(r[2]) + dx,
                f64::from(r[3]) + dy,
            ]
        };
        Self {
            spans: self
                .spans
                .iter()
                .map(|r| shift(*r).map(|v| v as f32))
                .collect(),
            cells: self
                .cells
                .iter()
                .map(|c| PathCell {
                    rect: shift(c.rect).map(|v| v as f32),
                    x: c.x,
                    y: c.y,
                })
                .collect(),
        }
    }
}

/// A coverage mask in the atlas: a path clip rasterized into one cell.
#[derive(Clone, Copy, Debug)]
pub struct MaskCell {
    /// Device-space origin of the mask.
    pub device: [f32; 2],
    /// Atlas texel origin of the cell.
    pub atlas: [f32; 2],
    /// Cell size in pixels.
    pub size: [f32; 2],
    /// The mask's device-space bounding rect `(x0, y0, x1, y1)` — the
    /// clip's analytic shape shrinks to it.
    pub rect: [f32; 4],
}

impl MaskCell {
    /// This mask shifted by `(dx, dy)` device pixels; the atlas texels
    /// do not move.
    #[must_use]
    #[expect(clippy::cast_possible_truncation, reason = "device coords are f32")]
    pub fn translated(&self, dx: f64, dy: f64) -> Self {
        Self {
            device: [
                (f64::from(self.device[0]) + dx) as f32,
                (f64::from(self.device[1]) + dy) as f32,
            ],
            atlas: self.atlas,
            size: self.size,
            rect: [
                (f64::from(self.rect[0]) + dx) as f32,
                (f64::from(self.rect[1]) + dy) as f32,
                (f64::from(self.rect[2]) + dx) as f32,
                (f64::from(self.rect[3]) + dy) as f32,
            ],
        }
    }
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
    /// Rasterized path emissions, keyed by content hash.
    paths: HashMap<u64, PathEmit>,
    /// Rasterized path-clip masks, keyed by content hash.
    masks: HashMap<u64, MaskCell>,
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
            paths: HashMap::new(),
            masks: HashMap::new(),
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

    /// Drops every cached glyph of `font`. The freed texels stay claimed
    /// in the shelf layout until the next [`Self::grow`] or
    /// [`Self::clear`]; path and mask cells are font-independent and stay.
    pub fn remove_font(&mut self, font: u64) {
        let freed: u64 = self
            .map
            .extract_if(|key, _| key.font == font)
            .map(|(_, entry)| u64::from(entry.w) * u64::from(entry.h))
            .sum();
        self.cpu_bytes = self.cpu_bytes.saturating_sub(freed);
    }

    /// Clears every entry without freeing the texture.
    pub fn clear(&mut self) {
        self.map.clear();
        self.paths.clear();
        self.masks.clear();
        self.shelves.clear();
        self.cpu_bytes = 0;
    }

    /// A cached path emission.
    pub fn path(&self, key: u64) -> Option<&PathEmit> {
        self.paths.get(&key)
    }

    /// Caches a path emission.
    pub fn insert_path(&mut self, key: u64, emit: PathEmit) {
        self.cpu_bytes += (emit.spans.len() * 16 + emit.cells.len() * 24) as u64;
        self.paths.insert(key, emit);
    }

    /// A cached path-clip mask.
    pub fn mask(&self, key: u64) -> Option<&MaskCell> {
        self.masks.get(&key)
    }

    /// Caches a path-clip mask.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "cell sizes are small positive floats"
    )]
    pub fn insert_mask(&mut self, key: u64, mask: MaskCell) {
        self.cpu_bytes += (mask.size[0] * mask.size[1]) as u64;
        self.masks.insert(key, mask);
    }

    /// Uploads `w` × `h` coverage texels into the cell at `(x, y)`.
    pub fn write(&self, queue: &wgpu::Queue, x: u32, y: u32, w: u32, h: u32, texels: &[u8]) {
        // `write_texture` needs rows padded to 256 bytes.
        let pitch = w.div_ceil(256) * 256;
        let mut staging = vec![0u8; (pitch * h) as usize];
        for (row, line) in texels.chunks_exact(w as usize).enumerate() {
            staging[row * pitch as usize..row * pitch as usize + w as usize].copy_from_slice(line);
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
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
    }

    /// The atlas edge in texels.
    pub const fn size(&self) -> u32 {
        self.size
    }

    /// The largest atlas edge the budget allows.
    pub const fn cap(&self) -> u32 {
        self.cap
    }

    /// Whether a `w` × `h` cell fits in an empty atlas at the cap.
    pub const fn can_ever_fit(&self, w: u32, h: u32) -> bool {
        w + 2 * PAD <= self.cap && (h + 2 * PAD).div_ceil(8) * 8 <= self.cap
    }

    /// Stores a rasterized glyph cell under `key`, uploading its texels.
    /// The caller's `x`/`y` patch targets the returned origin. `None`
    /// when the atlas is full; the caller grows/clears and retries.
    #[expect(
        clippy::too_many_arguments,
        reason = "the pending raster record's fields, passed through"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "atlas cells fit u16: the atlas is at most 64kpx per axis"
    )]
    pub fn store_glyph(
        &mut self,
        queue: &wgpu::Queue,
        key: GlyphKey,
        left: i32,
        top: i32,
        w: u32,
        h: u32,
        texels: &[u8],
    ) -> Option<(u32, u32)> {
        if let Some(entry) = self.get(&key) {
            return Some((u32::from(entry.x), u32::from(entry.y)));
        }
        if w == 0 {
            self.map.insert(key, Entry::default());
            return Some((0, 0));
        }
        let (cx, cy) = self.alloc(w, h)?;
        self.write(queue, cx, cy, w, h, texels);
        self.cpu_bytes += u64::from(w) * u64::from(h);
        let entry = Entry {
            x: cx as u16,
            y: cy as u16,
            w: w as u16,
            h: h as u16,
            left,
            top,
        };
        self.map.insert(key, entry);
        Some((cx, cy))
    }

    /// Stores a path emission, allocating each cell and uploading its
    /// texels in emission order. `None` when the atlas is full; `emit`
    /// is only inserted once every cell allocated.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "atlas cells fit u16: the atlas is at most 64kpx per axis"
    )]
    pub fn store_path(
        &mut self,
        queue: &wgpu::Queue,
        key: u64,
        emit: PathEmit,
        cells: &[CellTexels],
    ) -> Option<()> {
        if self.paths.contains_key(&key) {
            return Some(());
        }
        let mut emit = emit;
        for (cell, (w, h, texels)) in emit.cells.iter_mut().zip(cells) {
            let (cx, cy) = self.alloc(*w, *h)?;
            self.write(queue, cx, cy, *w, *h, texels);
            cell.x = cx as u16;
            cell.y = cy as u16;
        }
        self.insert_path(key, emit);
        Some(())
    }

    /// Stores a path-clip mask, setting `mask.atlas` to the cell origin.
    /// `None` when the atlas is full.
    #[expect(
        clippy::cast_precision_loss,
        reason = "atlas coords are well within f32"
    )]
    pub fn store_mask(
        &mut self,
        queue: &wgpu::Queue,
        key: u64,
        mut mask: MaskCell,
        w: u32,
        h: u32,
        texels: &[u8],
    ) -> Option<()> {
        if self.masks.contains_key(&key) {
            return Some(());
        }
        let (cx, cy) = self.alloc(w, h)?;
        self.write(queue, cx, cy, w, h, texels);
        mask.atlas = [cx as f32, cy as f32];
        self.insert_mask(key, mask);
        Some(())
    }

    /// Every cell origin of the cached emission for `key`, in order.
    pub fn path_origins(&self, key: u64) -> Option<Vec<(u32, u32)>> {
        self.paths.get(&key).map(|e| {
            e.cells
                .iter()
                .map(|c| (u32::from(c.x), u32::from(c.y)))
                .collect()
        })
    }

    /// The atlas origin of the cached mask for `key`.
    pub fn mask_origin(&self, key: u64) -> Option<[f32; 2]> {
        self.masks.get(&key).map(|m| m.atlas)
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
    pub(crate) fn alloc(&mut self, w: u32, h: u32) -> Option<(u32, u32)> {
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

/// A glyph's coverage texels, not yet in the atlas.
struct CellRaster {
    /// Bearing: cell's left edge relative to the glyph's snapped origin.
    left: i32,
    /// Bearing: cell's top edge relative to the snapped origin.
    top: i32,
    /// Cell width.
    w: u32,
    /// Cell height.
    h: u32,
    /// `w` × `h` coverage texels.
    texels: Vec<u8>,
}

/// One path cell's raster pending its atlas origin: `(w, h, texels)`.
pub type CellTexels = (u32, u32, Vec<u8>);

/// Work deferred to the render thread by a parallel lowering: every
/// atlas write and COLR cache update a surface wanted, in lowering
/// order, with the bitmaps already rasterized from immutable font data.
/// The render thread applies the batches serially, in dirty-surface
/// order, so the atlas ends in exactly the state serial lowering would
/// produce.
pub enum PendingRaster {
    /// A glyph coverage cell, keyed by `GlyphKey`. `w`/`h` are `0` for
    /// glyphs with no outline: they still must be stored so the miss is
    /// not retried every frame.
    Glyph {
        /// The cache key.
        key: GlyphKey,
        /// Bearing: the cell's left edge relative to the snapped origin.
        left: i32,
        /// Bearing: the cell's top edge relative to the snapped origin.
        top: i32,
        /// Cell width.
        w: u32,
        /// Cell height.
        h: u32,
        /// `w` × `h` coverage texels.
        texels: Vec<u8>,
    },
    /// A path emission whose cells were rasterized cell by cell; each
    /// entry of `cells` pairs one `(w, h, texels)` with the cell in
    /// `emit` at the same index.
    Path {
        /// The content-hash key.
        key: u64,
        /// Spans and cells; each cell's `x`/`y` is set when it is
        /// stored.
        emit: PathEmit,
        /// Per-cell `(width, height, texels)` in emission order.
        cells: Vec<CellTexels>,
    },
    /// A path-clip mask; `mask.atlas` is set when it is stored.
    Mask {
        /// The content-hash key.
        key: u64,
        /// The mask, pending its atlas origin.
        mask: MaskCell,
        /// Cell width.
        w: u32,
        /// Cell height.
        h: u32,
        /// `w` × `h` coverage texels.
        texels: Vec<u8>,
    },
    /// A built COLR picture for one registered font.
    Colr {
        /// `Renderer::fonts` key of the font that produced it.
        font: u64,
        /// The `(glyph id, coords hash, paint hash)` cache key.
        key: (u32, u64, u64),
        /// The built picture.
        picture: cherenkov::Picture,
    },
}

/// The glyph's atlas cell. On a cache hit the entry is returned with
/// `None`; on a miss the cell is rasterized and queued in `pending` as
/// [`PendingRaster::Glyph`], and the returned entry's `x`/`y` stay zero
/// until the render thread stores the cell — the caller records an
/// instance patch against the pending index.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors glyph_key's inputs; grouping them would only rename the bundle"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "pending lists are tiny: u32 indexes them fine"
)]
pub fn entry(
    atlas: &Atlas,
    font: &FontData,
    key: GlyphKey,
    glyph_id: u32,
    size: f32,
    subpixel: (f32, f32),
    transform: Affine,
    coords: &[i16],
    pending: &mut Vec<PendingRaster>,
) -> Result<(Entry, Option<u32>), RenderError> {
    if let Some(entry) = atlas.get(&key) {
        return Ok((entry, None));
    }
    let cell = rasterize_texels(font, glyph_id, size, subpixel, transform, coords)?;
    let (left, top, w, h, texels) = cell.map_or((0, 0, 0, 0, Vec::new()), |c| {
        (c.left, c.top, c.w, c.h, c.texels)
    });
    let idx = pending.len() as u32;
    pending.push(PendingRaster::Glyph {
        key,
        left,
        top,
        w,
        h,
        texels,
    });
    Ok((
        Entry {
            x: 0,
            y: 0,
            w: w as u16,
            h: h as u16,
            left,
            top,
        },
        Some(idx),
    ))
}

/// Rasterizes the glyph's outline into coverage texels — pure CPU work
/// that touches no atlas state. `None` for glyphs with no outline.
#[expect(
    clippy::many_single_char_names,
    reason = "matrix coefficient names follow the Affine convention"
)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "coverage math runs in f64; cells, bearings and segments are bounded by the atlas size"
)]
fn rasterize_texels(
    font: &FontData,
    glyph_id: u32,
    size: f32,
    subpixel: (f32, f32),
    transform: Affine,
    coords: &[i16],
) -> Result<Option<CellRaster>, RenderError> {
    let font_ref = skrifa::FontRef::from_index(&font.data, font.index)
        .map_err(|e| RenderError::Font(format!("{e}")))?;
    let upem = font_ref
        .head()
        .map_err(|e| RenderError::Font(format!("head: {e}")))?
        .units_per_em();
    let outlines = font_ref.outline_glyphs();
    let Some(outline) = outlines.get(skrifa::GlyphId::new(glyph_id)) else {
        return Ok(None);
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
        return Ok(None);
    }
    if pen.path.is_empty() {
        return Ok(None);
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
        return Ok(None);
    }
    let left = bbox.x0.floor() as i32 - 1;
    let top = bbox.y0.floor() as i32 - 1;
    let right = bbox.x1.ceil() as i32 + 1;
    let bottom = bbox.y1.ceil() as i32 + 1;
    let w = (right - left) as u32;
    let h = (bottom - top) as u32;
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
    Ok(Some(CellRaster {
        left,
        top,
        w,
        h,
        texels,
    }))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// An adapter plus device, or `None` where no GPU exists.
    fn device_and_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .next()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    /// Two parallel lowerings that both miss the same glyphs produce the
    /// same atlas as serial lowering: pending rasters commit in
    /// dirty-surface order and a duplicated key resolves to the first
    /// surface's cell.
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the serial reference and the parallel run share the helpers but need both sequences"
    )]
    fn pending_apply_matches_serial_atlas() {
        /// Lower `keys` against `atlas`, collecting the pending rasters.
        fn lower(atlas: &Atlas, font: &FontData, keys: &[GlyphKey]) -> Vec<PendingRaster> {
            let mut pending = Vec::new();
            for &k in keys {
                entry(
                    atlas,
                    font,
                    k,
                    k.glyph,
                    16.0,
                    (0.0, 0.0),
                    Affine::IDENTITY,
                    &[],
                    &mut pending,
                )
                .expect("entry");
            }
            pending
        }
        /// Commit one pending list, the way `apply_raster` does.
        fn apply(atlas: &mut Atlas, queue: &wgpu::Queue, pending: Vec<PendingRaster>) {
            for raster in pending {
                let PendingRaster::Glyph {
                    key,
                    left,
                    top,
                    w,
                    h,
                    texels,
                } = raster
                else {
                    unreachable!("glyph pending only")
                };
                assert!(
                    atlas
                        .store_glyph(queue, key, left, top, w, h, &texels)
                        .is_some()
                );
            }
        }
        let Some((device, queue)) = device_and_queue() else {
            return;
        };
        let font = FontData {
            data: include_bytes!("../../../scenes/fonts/NotoSans.ttf")
                .as_slice()
                .into(),
            index: 0,
            colr: std::cell::RefCell::new(HashMap::new()),
        };
        let key = |glyph: u32| GlyphKey {
            font: 7,
            glyph,
            size_bits: (16.0f32 * 64.0).to_bits(),
            subpixel: 0,
            matrix: [
                1.0f32.to_bits(),
                0.0f32.to_bits(),
                0.0f32.to_bits(),
                1.0f32.to_bits(),
            ],
            coords_hash: 0,
        };
        // Surface A draws glyphs {1, 2, 3}; surface B overlaps on {2, 3, 4}.
        let surfaces = [[key(1), key(2), key(3)], [key(2), key(3), key(4)]];
        // Serial reference: each surface's pending commits before the next
        // lowers, so its lookups hit the earlier surface's cells.
        let mut serial = Atlas::new(&device, u64::MAX);
        for keys in &surfaces {
            let pending = lower(&serial, &font, keys);
            apply(&mut serial, &queue, pending);
        }
        // Parallel: both lowerings read the same empty atlas, so every
        // miss becomes a pending raster — duplicates included — then the
        // render thread commits them in dirty-surface order.
        let mut parallel = Atlas::new(&device, u64::MAX);
        let batches: Vec<_> = surfaces
            .iter()
            .map(|keys| lower(&parallel, &font, keys))
            .collect();
        for pending in batches {
            apply(&mut parallel, &queue, pending);
        }
        for keys in &surfaces {
            for &k in keys {
                assert_eq!(
                    serial.get(&k),
                    parallel.get(&k),
                    "glyph {}'s cell differs",
                    k.glyph
                );
            }
        }
        // A pending mask dedupes the same way.
        let mask = MaskCell {
            device: [0.0, 0.0],
            atlas: [0.0, 0.0],
            size: [2.0, 2.0],
            rect: [0.0, 0.0, 2.0, 2.0],
        };
        assert!(
            parallel
                .store_mask(&queue, 0xdead, mask, 2, 2, &[255u8; 4])
                .is_some()
        );
        let stored = parallel.mask_origin(0xdead).expect("stored");
        assert!(
            parallel
                .store_mask(&queue, 0xdead, mask, 2, 2, &[0u8; 4])
                .is_some(),
            "duplicate mask resolves to the stored cell"
        );
        assert_eq!(parallel.mask_origin(0xdead), Some(stored));
    }
}
