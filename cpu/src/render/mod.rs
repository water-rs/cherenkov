//! The render side of the [`Raster`](crate::Raster) backend: sole owner
//! of the framebuffers and the worker pool, driven by the shared front
//! end's render loop.

mod account;
mod bitmap;
mod blend;
mod colr;
mod filter;
mod font;
mod glyph;
mod image;
mod mesh;
use image::CpuImage;
use std::sync::Arc;
mod lower;
mod paint;
mod prepared;
pub mod present;
mod raster;

use std::collections::HashMap;

use cherenkov::{
    BackdropId, ContentOp, EngineError, FontData, FontId, Frame, FrameStats, ImageId, ImageUpload,
    LayerId, MemoryUsage, Pressure, Readback, Redraw, RenderError, Renderer, ResourceError,
    SurfaceError, SurfaceId, SurfaceInfo,
};
use lower::{ContentData, Item, Lowering};

use crate::{Band, RasterConfig, RasterInfo, RasterTarget};

/// The largest surface dimension the CPU framebuffer supports.
const MAX_SURFACE: u32 = 16384;

/// A surface's pixel destination.
enum Output {
    /// Full-frame storage in `LinearF32`: the readback buffer.
    F32(Vec<[f32; 4]>),
    /// Full-frame storage in `LinearF16`: the readback buffer in the
    /// host's requested format.
    F16(Vec<[half::f16; 4]>),
    /// No frame storage: finished bands stream to the host's sink in row
    /// order. `emit` is the `LinearF16` conversion buffer.
    Stream {
        format: cherenkov::OffscreenFormat,
        sink: Box<dyn FnMut(Band<'_>) + Send>,
        emit: Vec<[half::f16; 4]>,
    },
}

/// One surface's render-thread state.
struct SurfaceState {
    size: (u32, u32),
    /// Where finished pixels land; only `Offscreen` targets retain a
    /// full-frame buffer, in the format the host asked for.
    output: Output,
    refresh: cherenkov::RefreshRange,
    /// Per-layer content caches; the sampled layer state lives in the
    /// front end's [`cherenkov::SurfaceTree`].
    layers: HashMap<LayerId, ContentData>,
    filters: Vec<u64>,
    /// Backdrop groups referenced by the last frame, by group id.
    groups: Vec<u64>,
    /// The largest live pixel-buffer bytes in any band that ran a
    /// backdrop capture last frame (window, isolation stack and capture
    /// buffers). 0 when no capture ran.
    backdrop_capture_peak: u64,
}

impl SurfaceState {
    /// Heap bytes of the output storage (`Stream` keeps none).
    const fn output_bytes(&self) -> u64 {
        match &self.output {
            Output::F32(fb) => (fb.capacity() * size_of::<[f32; 4]>()) as u64,
            Output::F16(fb) => (fb.capacity() * size_of::<[half::f16; 4]>()) as u64,
            Output::Stream { .. } => 0,
        }
    }

    /// Heap bytes of the surface's own band-format working buffer.
    const fn band_bytes(&self) -> u64 {
        match &self.output {
            Output::Stream { emit, .. } => (emit.capacity() * size_of::<[half::f16; 4]>()) as u64,
            _ => 0,
        }
    }
}

/// All render-thread state: the [`Raster`](crate::Raster) backend's
/// [`Renderer`] implementation.
pub struct RasterRenderer {
    pool: rayon::ThreadPool,
    surfaces: HashMap<SurfaceId, SurfaceState>,
    pub(super) filters: filter::Registry,
    fonts: HashMap<u64, font::Font>,
    bitmap_fonts: HashMap<u64, Arc<bitmap::BitmapFont>>,
    images: HashMap<u64, Arc<CpuImage>>,
    image_budget: u64,
    /// The glyph mask cache, bounded by `Budget::cpu`.
    glyph_cache: glyph::GlyphCache,
    /// Decoded colour-font bitmaps, bounded by `Budget::cpu`.
    bitmap_cache: bitmap::BitmapCache,
}

/// Runs on the render thread once: builds the worker pool, returning the
/// backend's [`Renderer`].
///
/// # Errors
/// [`EngineError::Backend`] when the pool cannot be built.
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
                    filters: filter::Registry::new(config.redraw),
                    fonts: HashMap::new(),
                    bitmap_fonts: HashMap::new(),
                    images: HashMap::new(),
                    image_budget: config.budget.cpu.0,
                    glyph_cache: glyph::GlyphCache::new(config.budget.cpu.0),
                    bitmap_cache: bitmap::BitmapCache::new(config.budget.cpu.0),
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

/// Validates font data and detects colour-glyph sources.
fn validate_font(
    data: &[u8],
    index: u32,
) -> Result<(bool, Option<Arc<bitmap::BitmapFont>>), ResourceError> {
    use skrifa::raw::TableProvider as _;
    let font = skrifa::FontRef::from_index(data, index)
        .map_err(|e| ResourceError::Font(format!("{e}")))?;
    let has_colr = font.colr().is_ok();
    let bitmap = bitmap::BitmapFont::detect(data, index)?.map(Arc::new);
    Ok((has_colr, bitmap))
}

impl Renderer for RasterRenderer {
    type Target = RasterTarget;

    fn create_surface(
        &mut self,
        id: SurfaceId,
        target: RasterTarget,
    ) -> Result<SurfaceInfo, SurfaceError> {
        let (size, output, readable, refresh) = match target {
            RasterTarget::Offscreen(offscreen) => {
                let (size, refresh) = (offscreen.size, offscreen.refresh);
                let pixels = (size.0 * size.1) as usize;
                let output = match offscreen.format {
                    cherenkov::OffscreenFormat::LinearF32 => Output::F32(vec![[0.0; 4]; pixels]),
                    cherenkov::OffscreenFormat::LinearF16 => {
                        Output::F16(vec![[half::f16::ZERO; 4]; pixels])
                    }
                };
                (size, output, true, refresh)
            }
            RasterTarget::Bands(bands) => {
                let (size, format) = (bands.size, bands.format);
                let output = Output::Stream {
                    format: bands.format,
                    sink: bands.sink,
                    // Exactly one band's worth for f16 emission, so
                    // `memory()` is stable. f32 streams borrow the scratch.
                    emit: match format {
                        cherenkov::OffscreenFormat::LinearF16 => Vec::with_capacity(
                            size.0 as usize * raster::BAND_H.min(size.1 as usize),
                        ),
                        cherenkov::OffscreenFormat::LinearF32 => Vec::new(),
                    },
                };
                (size, output, false, cherenkov::DEFAULT_REFRESH)
            }
        };
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
                output,
                refresh,
                layers: HashMap::new(),
                filters: Vec::new(),
                groups: Vec::new(),
                backdrop_capture_peak: 0,
            },
        );
        Ok(SurfaceInfo {
            max_dimension: MAX_SURFACE,
            size,
            readable,
        })
    }

    fn resize_surface(&mut self, id: SurfaceId, size: (u32, u32)) {
        let Some(state) = self.surfaces.get_mut(&id) else {
            return;
        };
        state.size = size;
        let pixels = (size.0 * size.1) as usize;
        match &mut state.output {
            Output::F32(fb) => *fb = vec![[0.0; 4]; pixels],
            Output::F16(fb) => *fb = vec![[half::f16::ZERO; 4]; pixels],
            Output::Stream { format, emit, .. } => {
                *emit = match format {
                    cherenkov::OffscreenFormat::LinearF16 => {
                        Vec::with_capacity(size.0 as usize * raster::BAND_H.min(size.1 as usize))
                    }
                    cherenkov::OffscreenFormat::LinearF32 => Vec::new(),
                };
            }
        }
    }

    fn destroy_surface(&mut self, id: SurfaceId) {
        self.surfaces.remove(&id);
    }

    fn add_font(&mut self, id: FontId, font: FontData) -> Result<(), ResourceError> {
        let (has_colr, bitmap) = validate_font(&font.data, font.index)?;
        let has_bitmap = bitmap.is_some();
        if let Some(bitmap) = bitmap {
            self.bitmap_fonts.insert(id.raw(), bitmap);
        } else {
            self.bitmap_fonts.remove(&id.raw());
        }
        self.fonts.insert(
            id.raw(),
            font::Font {
                data: font,
                has_colr,
                has_bitmap,
                colr: HashMap::new(),
            },
        );
        Ok(())
    }

    fn remove_font(&mut self, id: FontId) {
        self.fonts.remove(&id.raw());
        self.bitmap_fonts.remove(&id.raw());
        self.bitmap_cache.remove_font(id.raw());
        self.refresh_cache_budgets();
        for surface in self.surfaces.values_mut() {
            for content in surface.layers.values_mut() {
                content.invalidate();
            }
        }
    }

    fn add_image(&mut self, id: ImageId, image: ImageUpload) -> Result<(), ResourceError> {
        let resident: u64 = self.images.values().map(|image| image.bytes()).sum();
        let required = u64::from(image.width)
            .checked_mul(u64::from(image.height))
            .and_then(|count| count.checked_mul(16));
        if required.is_none_or(|bytes| bytes > self.image_budget.saturating_sub(resident)) {
            return Err(ResourceError::Image("CPU image budget exhausted".into()));
        }
        let decoded = Arc::new(CpuImage::decode(&image)?);
        self.images.insert(id.raw(), decoded);
        self.refresh_cache_budgets();
        Ok(())
    }

    fn remove_image(&mut self, id: ImageId) {
        self.images.remove(&id.raw());
        self.refresh_cache_budgets();
        for surface in self.surfaces.values_mut() {
            for content in surface.layers.values_mut() {
                content.invalidate();
            }
        }
    }

    fn set_content(
        &mut self,
        surface: SurfaceId,
        layer: LayerId,
        content: Option<ContentOp>,
    ) -> Option<cherenkov::Picture> {
        let state = self.surfaces.get_mut(&surface)?;
        match content {
            Some(ContentOp::Replace(list)) => state
                .layers
                .insert(layer, ContentData::new(list))
                .map(cherenkov::lowering::Content::into_picture),
            Some(ContentOp::Update(updates)) => {
                state
                    .layers
                    .get_mut(&layer)
                    .expect("slot update targets a layer without content")
                    .update(updates);
                None
            }
            Some(ContentOp::Picture(picture)) => state
                .layers
                .insert(layer, ContentData::picture(picture))
                .map(cherenkov::lowering::Content::into_picture),
            None => state
                .layers
                .remove(&layer)
                .map(cherenkov::lowering::Content::into_picture),
        }
    }

    fn remove_layer(&mut self, surface: SurfaceId, layer: LayerId) {
        if let Some(state) = self.surfaces.get_mut(&surface) {
            state.layers.remove(&layer);
        }
    }

    /// Lowers and rasterizes every changed surface.
    #[cfg(not(target_arch = "wasm32"))]
    fn render(&mut self, frame: &Frame<'_>, stats: &mut FrameStats) -> Result<Redraw, RenderError> {
        self.render_frame(frame, stats)
    }

    #[cfg(target_arch = "wasm32")]
    async fn render(
        &mut self,
        frame: &Frame<'_>,
        stats: &mut FrameStats,
    ) -> Result<Redraw, RenderError> {
        self.render_frame(frame, stats)
    }

    /// Materializes a surface's output buffer into `Readback` pixels.
    /// `LinearF16` stores already round once; the readback only widens.
    /// Band-streaming surfaces have no frame buffer to read.
    #[cfg(not(target_arch = "wasm32"))]
    fn readback(&mut self, surface: SurfaceId) -> Result<Readback, RenderError> {
        let Some(state) = self.surfaces.get(&surface) else {
            return Err(RenderError::Readback("unknown surface".into()));
        };
        let pixels = match &state.output {
            Output::F32(fb) => fb.clone(),
            Output::F16(fb) => fb.iter().map(|px| px.map(half::f16::to_f32)).collect(),
            Output::Stream { .. } => {
                return Err(RenderError::Readback(
                    "band-streaming surfaces are not readable".into(),
                ));
            }
        };
        Ok(Readback {
            width: state.size.0,
            height: state.size.1,
            pixels,
        })
    }

    #[cfg(target_arch = "wasm32")]
    async fn readback(&mut self, surface: SurfaceId) -> Result<Readback, RenderError> {
        let Some(state) = self.surfaces.get(&surface) else {
            return Err(RenderError::Readback("unknown surface".into()));
        };
        let pixels = match &state.output {
            Output::F32(fb) => fb.clone(),
            Output::F16(fb) => fb.iter().map(|px| px.map(half::f16::to_f32)).collect(),
            Output::Stream { .. } => {
                return Err(RenderError::Readback(
                    "band-streaming surfaces are not readable".into(),
                ));
            }
        };
        Ok(Readback {
            width: state.size.0,
            height: state.size.1,
            pixels,
        })
    }

    /// Memory usage across output targets, band working buffers, retained
    /// layer content, registered images and the glyph caches.
    fn memory(&self) -> MemoryUsage {
        let mut categories = account::Categories::default();
        for surface in self.surfaces.values() {
            categories.output += surface.output_bytes();
            categories.retained += surface
                .layers
                .values()
                .map(|content| account::content_bytes(content) + lower::silhouette_bytes(content))
                .sum::<u64>();
        }
        categories.bands = self
            .surfaces
            .values()
            .map(SurfaceState::band_bytes)
            .sum::<u64>();
        categories.glyphs = self.glyph_cache.bytes();
        categories.images = self.images.values().map(|image| image.bytes()).sum();
        categories.colr = self.fonts.values().map(font::Font::colr_bytes).sum();
        categories.bitmaps = self.bitmap_cache.bytes();
        tracing::debug!(
            target: "cherenkov_cpu::memory",
            output = categories.output,
            bands = categories.bands,
            retained = categories.retained,
            images = categories.images,
            glyphs = categories.glyphs,
            colr = categories.colr,
            bitmaps = categories.bitmaps,
            "memory usage",
        );
        // Backdrop captures are transient: the reported value is the
        // peak live pixel-buffer bytes of the last frame's capture bands,
        // in premultiplied `f32` (`linear-f32`).
        let backdrop_captures: u64 = self
            .surfaces
            .values()
            .map(|surface| surface.backdrop_capture_peak)
            .sum();
        MemoryUsage {
            gpu: cherenkov::Bytes(0),
            cpu: cherenkov::Bytes(categories.total()),
            backdrop_captures: cherenkov::Bytes(backdrop_captures),
            backdrop_capture_format: (backdrop_captures > 0).then_some("linear-f32"),
        }
    }

    fn trim(&mut self, pressure: Pressure) {
        if pressure == Pressure::Critical {
            for font in self.fonts.values_mut() {
                font.colr.clear();
            }
            self.fonts.shrink_to_fit();
            self.glyph_cache.clear();
            self.bitmap_cache.clear();
            for surface in self.surfaces.values_mut() {
                for content in surface.layers.values_mut() {
                    content.trim();
                }
            }
        }
    }
}

impl RasterRenderer {
    fn render_frame(
        &mut self,
        frame: &Frame<'_>,
        stats: &mut FrameStats,
    ) -> Result<Redraw, RenderError> {
        self.filters.begin_frame(frame.id, frame.time);
        for sf in frame.surfaces {
            let filter_changed = self.surfaces.get(&sf.id).is_some_and(|surface| {
                surface
                    .filters
                    .iter()
                    .any(|id| self.filters.wants_redraw(*id))
                    || surface
                        .groups
                        .iter()
                        .any(|id| self.filters.wants_redraw_group(sf.id, BackdropId::new(*id)))
            });
            if sf.changed || filter_changed {
                stats.frame = Some(frame.id);
                self.render_surface(sf, frame.id, stats)?;
            }
        }
        let used: std::collections::HashSet<u64> = self
            .surfaces
            .values()
            .flat_map(|surface| surface.filters.iter().copied())
            .collect();
        let used_groups: std::collections::HashSet<(u64, u64)> = self
            .surfaces
            .iter()
            .flat_map(|(surface, state)| {
                state
                    .groups
                    .iter()
                    .map(move |group| (surface.raw(), *group))
            })
            .collect();
        self.filters.set_active(&used, &used_groups);
        self.filters.finish_frame(&used, &used_groups);
        let rate = self
            .surfaces
            .iter()
            .filter(|(id, surface)| {
                surface
                    .filters
                    .iter()
                    .any(|fid| self.filters.wants_redraw(*fid))
                    || surface
                        .groups
                        .iter()
                        .any(|gid| self.filters.wants_redraw_group(**id, BackdropId::new(*gid)))
            })
            .map(|(_, surface)| surface)
            .fold(None, |rate: Option<cherenkov::RefreshRange>, surface| {
                Some(rate.map_or_else(
                    || surface.refresh.clone(),
                    |rate| {
                        (*rate.start()).min(*surface.refresh.start())
                            ..=(*rate.end()).max(*surface.refresh.end())
                    },
                ))
            });
        Ok(rate.map_or(Redraw::None, |rate| Redraw::Wanted { rate }))
    }

    fn refresh_cache_budgets(&mut self) {
        let resident: u64 = self.images.values().map(|image| image.bytes()).sum();
        let available = self.image_budget.saturating_sub(resident);
        self.bitmap_cache
            .set_budget(available.saturating_sub(self.glyph_cache.bytes()));
        self.glyph_cache
            .set_budget(available.saturating_sub(self.bitmap_cache.bytes()));
    }
    /// Lowers and rasterizes one surface's frame.
    #[expect(
        clippy::many_single_char_names,
        reason = "w/h and r/g/b/a are the natural names"
    )]
    fn render_surface(
        &mut self,
        sf: &cherenkov::SurfaceFrame<'_>,
        frame: cherenkov::FrameId,
        stats: &mut FrameStats,
    ) -> Result<(), RenderError> {
        let profile = tracing::enabled!(target: "cherenkov_cpu::profile", tracing::Level::DEBUG);
        let start = profile.then(cherenkov::Instant::now);
        let id = sf.id;
        let mut items: Vec<Item> = Vec::new();
        let glyph_reqs;
        let glyphs_rasterized;
        // Lowering borrows the layer caches; the surface borrow ends
        // before glyph resolution touches `self.fonts`/`self.glyph_cache`.
        let lowered = {
            let Some(surf) = self.surfaces.get_mut(&id) else {
                return Ok(());
            };
            let mut caches = std::mem::take(&mut surf.layers);
            let mut lowering = Lowering::new(
                &mut items,
                surf.size,
                Some(&mut self.filters),
                frame,
                &mut self.fonts,
                &self.bitmap_fonts,
                &mut self.bitmap_cache,
            );
            let result = lowering.run(id, sf.tree, &mut caches, &self.images);
            glyphs_rasterized = lowering.glyphs_rasterized;
            stats.glyphs_rasterized += glyphs_rasterized;
            stats.commands_lowered += lowering.commands_lowered;
            stats.layers_composed += lowering.layers_composed;
            glyph_reqs = std::mem::take(&mut lowering.glyphs);
            surf.filters = lowering.take_used_filters().into_iter().collect();
            surf.groups = lowering.take_used_groups().into_iter().collect();
            surf.layers = caches;
            result
        };
        lowered?;
        if glyphs_rasterized > 0 {
            self.refresh_cache_budgets();
        }
        let lowered_at = start.map(|_| cherenkov::Instant::now());
        self.resolve_glyphs(&glyph_reqs)?;
        let resolved_at = start.map(|_| cherenkov::Instant::now());
        let Some(surf) = self.surfaces.get_mut(&id) else {
            return Ok(());
        };
        let (w, h) = (surf.size.0 as usize, surf.size.1 as usize);
        let [r, g, b, a] = sf.clear.components;
        let clear = [r * a, g * a, b * a, a];
        let pool = &self.pool;
        let peak = std::sync::atomic::AtomicU64::new(0);
        let has_backdrop = !surf.groups.is_empty();
        let (draws, edges) = match &mut surf.output {
            Output::F32(fb) => pool.install(|| {
                raster::render_bands(&items, clear, fb, w, h, Some(&peak), has_backdrop)
            })?,
            Output::F16(out) => pool.install(|| {
                raster::render_bands_f16(&items, clear, out, w, h, Some(&peak), has_backdrop)
            })?,
            Output::Stream { format, sink, emit } => raster::render_bands_stream(
                &items,
                clear,
                (w, h),
                emit,
                *format,
                sink.as_mut(),
                Some(&peak),
                has_backdrop,
            )?,
        };
        surf.backdrop_capture_peak = peak.load(std::sync::atomic::Ordering::Relaxed);
        if let (Some(start), Some(lowered), Some(resolved)) = (start, lowered_at, resolved_at) {
            tracing::debug!(target: "cherenkov_cpu::profile",
                lower_ns = lowered.duration_since(start).as_nanos(),
                glyph_ns = resolved.duration_since(lowered).as_nanos(),
                shade_ns = resolved.elapsed().as_nanos(),
                items = items.len(), glyphs = glyph_reqs.len(), "raster phases");
        }
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
                        let mask = std::sync::Arc::new(glyph::rasterize_mask(&font.data, req)?);
                        let _ = req.slot.set(mask.clone());
                        Ok((req.key, mask))
                    })
                    .collect::<Result<Vec<_>, RenderError>>()
            })?;
        self.glyph_cache.insert_batch(masks);
        Ok(())
    }
}
