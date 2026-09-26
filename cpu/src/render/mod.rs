// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The render side of the [`Raster`](crate::Raster) backend: sole owner
//! of the framebuffers and the worker pool, driven by the shared front
//! end's render loop.

mod glyph;
mod lower;
mod paint;
mod prepared;
mod raster;

use std::collections::HashMap;

use cherenkov::{
    ContentOp, EngineError, FontData, FontId, Frame, FrameStats, ImageId, ImageUpload, LayerId,
    MemoryUsage, Pressure, Readback, Redraw, RenderError, Renderer, ResourceError, SurfaceError,
    SurfaceId, SurfaceInfo,
};
use lower::{ContentData, Item, Lowering};

use crate::{RasterConfig, RasterInfo, RasterTarget, names};

/// The largest surface dimension the CPU framebuffer supports.
const MAX_SURFACE: u32 = 16384;

/// One surface's render-thread state.
struct SurfaceState {
    size: (u32, u32),
    format: cherenkov::OffscreenFormat,
    /// The f32 premultiplied framebuffer, `width * height` pixels.
    fb: Vec<[f32; 4]>,
    /// Per-layer content caches; the sampled layer state lives in the
    /// front end's [`cherenkov::SurfaceTree`].
    layers: HashMap<LayerId, ContentData>,
}

/// All render-thread state: the [`Raster`](crate::Raster) backend's
/// [`Renderer`] implementation.
pub struct RasterRenderer {
    pool: rayon::ThreadPool,
    surfaces: HashMap<SurfaceId, SurfaceState>,
    fonts: HashMap<u64, FontData>,
    /// The glyph mask cache, bounded by `Budget::cpu`.
    glyph_cache: glyph::GlyphCache,
}

/// Runs on the render thread once: builds the worker pool, returning the
/// backend's [`Renderer`].
///
/// # Errors
/// [`EngineError::Backend`] when the pool cannot be built.
#[expect(
    clippy::needless_pass_by_value,
    reason = "the contract moves the config onto the render thread"
)]
pub fn init(config: RasterConfig) -> Result<(RasterRenderer, RasterInfo), EngineError> {
    let builder = rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads.unwrap_or(0))
        .thread_name(|i| format!("cherenkov-raster-{i}"));
    builder
        .build()
        .map(|pool| {
            let info = RasterInfo {
                threads: pool.current_num_threads(),
                simd: "scalar",
                cpu: cpu_model(),
            };
            (
                RasterRenderer {
                    pool,
                    surfaces: HashMap::new(),
                    fonts: HashMap::new(),
                    glyph_cache: glyph::GlyphCache::new(config.budget.cpu.0),
                },
                info,
            )
        })
        .map_err(|e| EngineError::Backend(format!("rayon pool: {e}")))
}

/// The host CPU model, best effort.
fn cpu_model() -> Option<String> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in cpuinfo.lines() {
        if let Some(v) = line
            .strip_prefix("model name")
            .and_then(|s| s.split(':').nth(1))
        {
            return Some(v.trim().to_owned());
        }
    }
    None
}

/// Validates font data with `skrifa`, rejecting colour fonts.
///
/// Fonts carrying `COLR`, `CBDT`/`CBLC` or `sbix` outlines are colour
/// fonts, which this slice cannot rasterize.
fn validate_font(data: &[u8], index: u32) -> Result<(), ResourceError> {
    use skrifa::raw::TableProvider as _;
    let font = skrifa::FontRef::from_index(data, index)
        .map_err(|e| ResourceError::Font(format!("{e}")))?;
    for tag in [
        skrifa::Tag::new(b"COLR"),
        skrifa::Tag::new(b"CBDT"),
        skrifa::Tag::new(b"sbix"),
    ] {
        if font.data_for_tag(tag).is_some() {
            return Err(ResourceError::Unsupported(names::COLOR_FONT));
        }
    }
    Ok(())
}

impl Renderer for RasterRenderer {
    type Target = RasterTarget;

    fn create_surface(
        &mut self,
        id: SurfaceId,
        target: RasterTarget,
    ) -> Result<SurfaceInfo, SurfaceError> {
        let RasterTarget::Offscreen(offscreen) = target;
        let (size, format) = (offscreen.size, offscreen.format);
        if size.0 > MAX_SURFACE || size.1 > MAX_SURFACE {
            return Err(SurfaceError::TooLarge {
                width: size.0,
                height: size.1,
                max: MAX_SURFACE,
            });
        }
        self.surfaces.insert(
            id,
            SurfaceState {
                size,
                format,
                fb: vec![[0.0; 4]; (size.0 * size.1) as usize],
                layers: HashMap::new(),
            },
        );
        Ok(SurfaceInfo {
            max_dimension: MAX_SURFACE,
            size,
            readable: true,
        })
    }

    fn resize_surface(&mut self, id: SurfaceId, size: (u32, u32)) {
        let Some(state) = self.surfaces.get_mut(&id) else {
            return;
        };
        state.size = size;
        state.fb = vec![[0.0; 4]; (size.0 * size.1) as usize];
    }

    fn destroy_surface(&mut self, id: SurfaceId) {
        self.surfaces.remove(&id);
    }

    fn add_font(&mut self, id: FontId, font: FontData) -> Result<(), ResourceError> {
        validate_font(&font.data, font.index)?;
        self.fonts.insert(id.raw(), font);
        Ok(())
    }

    fn remove_font(&mut self, id: FontId) {
        self.fonts.remove(&id.raw());
        for surface in self.surfaces.values_mut() {
            for content in surface.layers.values_mut() {
                content.invalidate();
            }
        }
    }

    fn add_image(&mut self, _id: ImageId, _image: ImageUpload) -> Result<(), ResourceError> {
        Err(ResourceError::Image(
            "this slice does not draw images".into(),
        ))
    }

    fn remove_image(&mut self, _id: ImageId) {}

    fn set_content(&mut self, surface: SurfaceId, layer: LayerId, content: Option<ContentOp>) {
        let Some(state) = self.surfaces.get_mut(&surface) else {
            return;
        };
        match content {
            Some(ContentOp::Replace(list)) => {
                state.layers.insert(layer, ContentData::new(list));
            }
            Some(ContentOp::Update(updates)) => {
                state
                    .layers
                    .get_mut(&layer)
                    .expect("slot update targets a layer without content")
                    .update(updates);
            }
            Some(ContentOp::Picture(picture)) => {
                state.layers.insert(layer, ContentData::picture(picture));
            }
            None => {
                state.layers.remove(&layer);
            }
        }
    }

    fn remove_layer(&mut self, surface: SurfaceId, layer: LayerId) {
        if let Some(state) = self.surfaces.get_mut(&surface) {
            state.layers.remove(&layer);
        }
    }

    /// Lowers and rasterizes every changed surface.
    fn render(&mut self, frame: &Frame<'_>, stats: &mut FrameStats) -> Result<Redraw, RenderError> {
        for sf in frame.surfaces.iter().filter(|sf| sf.changed) {
            self.render_surface(sf, stats)?;
        }
        // No backend-side redraw sources in this slice.
        Ok(Redraw::None)
    }

    /// Materializes a surface's framebuffer into `Readback` pixels,
    /// rounding through `f16` for
    /// [`OffscreenFormat::LinearF16`](cherenkov::OffscreenFormat::LinearF16).
    fn readback(&mut self, surface: SurfaceId) -> Result<Readback, RenderError> {
        let Some(state) = self.surfaces.get(&surface) else {
            return Err(RenderError::Readback("unknown surface".into()));
        };
        let pixels = match state.format {
            cherenkov::OffscreenFormat::LinearF32 => state.fb.clone(),
            cherenkov::OffscreenFormat::LinearF16 => state
                .fb
                .iter()
                .map(|px| px.map(|v| half::f16::from_f32(v).to_f32()))
                .collect(),
        };
        Ok(Readback {
            width: state.size.0,
            height: state.size.1,
            pixels,
        })
    }

    /// Memory usage across framebuffers and the glyph mask cache.
    fn memory(&self) -> MemoryUsage {
        let framebuffers: u64 = self
            .surfaces
            .values()
            .map(|s| u64::from(s.size.0) * u64::from(s.size.1) * 16)
            .sum();
        MemoryUsage {
            gpu: cherenkov::Bytes(0),
            cpu: cherenkov::Bytes(framebuffers + self.glyph_cache.bytes()),
        }
    }

    fn trim(&mut self, pressure: Pressure) {
        if pressure == Pressure::Critical {
            self.fonts.shrink_to_fit();
            self.glyph_cache.clear();
            for surface in self.surfaces.values_mut() {
                for content in surface.layers.values_mut() {
                    content.trim();
                }
            }
        }
    }
}

impl RasterRenderer {
    /// Lowers and rasterizes one surface's frame.
    #[expect(
        clippy::many_single_char_names,
        reason = "w/h and r/g/b/a are the natural names"
    )]
    fn render_surface(
        &mut self,
        sf: &cherenkov::SurfaceFrame<'_>,
        stats: &mut FrameStats,
    ) -> Result<(), RenderError> {
        let id = sf.id;
        let mut items: Vec<Item> = Vec::new();
        let glyph_reqs;
        // Lowering borrows the layer caches; the surface borrow ends
        // before glyph resolution touches `self.fonts`/`self.glyph_cache`.
        let lowered = {
            let Some(surf) = self.surfaces.get_mut(&id) else {
                return Ok(());
            };
            let mut caches = std::mem::take(&mut surf.layers);
            let mut lowering = Lowering::new(&mut items, surf.size);
            let result = lowering.run(sf.tree, &mut caches);
            stats.commands_lowered += lowering.commands_lowered;
            stats.layers_composed += lowering.layers_composed;
            glyph_reqs = std::mem::take(&mut lowering.glyphs);
            surf.layers = caches;
            result
        };
        lowered?;
        self.resolve_glyphs(&glyph_reqs)?;
        let Some(surf) = self.surfaces.get_mut(&id) else {
            return Ok(());
        };
        let (w, h) = (surf.size.0 as usize, surf.size.1 as usize);
        let [r, g, b, a] = sf.clear.components;
        let clear = [r * a, g * a, b * a, a];
        surf.fb.fill(clear);
        let pool = &self.pool;
        let fb = &mut surf.fb;
        let (draws, edges) = pool.install(|| raster::render_bands(&items, clear, fb, w, h));
        stats.draws += draws;
        stats.instances += edges;
        stats.passes += u32::try_from(h.div_ceil(raster::BAND_H)).unwrap_or(u32::MAX);
        Ok(())
    }

    /// Fills every glyph request's slot: cache hits resolve directly;
    /// misses are rasterized in parallel on the worker pool, then
    /// inserted into the cache under its byte budget.
    fn resolve_glyphs(&mut self, reqs: &[lower::GlyphReq]) -> Result<(), RenderError> {
        use rayon::prelude::*;
        let mut missing = Vec::new();
        for req in reqs {
            if let Some(mask) = self.glyph_cache.get(&req.key) {
                let _ = req.slot.set(mask);
            } else {
                missing.push(req);
            }
        }
        if missing.is_empty() {
            return Ok(());
        }
        let fonts = &self.fonts;
        let pool = &self.pool;
        let masks: Vec<(glyph::GlyphKey, std::sync::Arc<glyph::GlyphMask>)> =
            pool.install(|| {
                missing
                    .par_iter()
                    .map(|req| {
                        let font = fonts.get(&req.font).ok_or_else(|| {
                            RenderError::Font(format!("unregistered font {}", req.font))
                        })?;
                        let mask = std::sync::Arc::new(glyph::rasterize_mask(font, req)?);
                        let _ = req.slot.set(mask.clone());
                        Ok((req.key, mask))
                    })
                    .collect::<Result<Vec<_>, RenderError>>()
            })?;
        self.glyph_cache.insert_batch(masks);
        Ok(())
    }
}
