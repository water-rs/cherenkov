// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The render thread: sole owner of the framebuffers and the worker pool.

mod glyph;
mod lower;
mod paint;
mod raster;

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender};

use cherenkov::ContentChange;

use crate::config::{Bytes, MemoryUsage, Pressure, RasterConfig, RasterInfo};
use crate::error::{EngineError, RenderError, SurfaceError};
use crate::message::{ChangeSet, LayerId, LayerOp, Message, SurfaceId};
use crate::surface::{FrameStats, Next, OffscreenFormat, Readback};
use lower::{ContentData, Item, LayerNode, Lowering};

/// The largest surface dimension the CPU framebuffer supports.
const MAX_SURFACE: u32 = 16384;

/// A registered font's data on the render thread.
struct FontData {
    data: std::sync::Arc<[u8]>,
    index: u32,
}

/// One surface's render-thread state.
struct SurfaceState {
    size: (u32, u32),
    format: OffscreenFormat,
    /// The f32 premultiplied framebuffer, `width * height` pixels.
    fb: Vec<[f32; 4]>,
    layers: HashMap<LayerId, LayerNode>,
    clear: cherenkov::WorkingColor,
    dirty: bool,
}

/// All render-thread state.
struct Renderer {
    pool: rayon::ThreadPool,
    surfaces: HashMap<SurfaceId, SurfaceState>,
    fonts: HashMap<u64, FontData>,
    /// The glyph mask cache, bounded by `Budget::cpu`.
    glyph_cache: glyph::GlyphCache,
}

/// A new default layer node.
const fn node() -> LayerNode {
    LayerNode {
        transform: kurbo::Affine::IDENTITY,
        opacity: 1.0,
        clip: None,
        content: None,
        children: Vec::new(),
    }
}

/// The render-thread entry point: builds the worker pool, replies, then
/// loops over messages until [`Message::Shutdown`].
#[expect(
    clippy::needless_pass_by_value,
    reason = "the thread takes ownership of its channels and config"
)]
pub fn run(
    config: RasterConfig,
    rx: Receiver<Message>,
    init_tx: Sender<Result<Init, EngineError>>,
) {
    let builder = rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads.unwrap_or(0))
        .thread_name(|i| format!("cherenkov-raster-{i}"));
    let renderer = builder.build().map(|pool| {
        let info = RasterInfo {
            threads: pool.current_num_threads(),
            simd: "scalar",
            cpu: cpu_model(),
        };
        (
            Renderer {
                pool,
                surfaces: HashMap::new(),
                fonts: HashMap::new(),
                glyph_cache: glyph::GlyphCache::new(config.budget.cpu.0),
            },
            info,
        )
    });
    let (mut renderer, info) = match renderer {
        Ok(pair) => pair,
        Err(e) => {
            let _ = init_tx.send(Err(EngineError::Thread(format!("rayon pool: {e}"))));
            return;
        }
    };
    let _ = init_tx.send(Ok(Init { info }));
    while let Ok(message) = rx.recv() {
        match message {
            Message::CreateSurface {
                id,
                size,
                format,
                reply,
            } => {
                let _ = reply.send(renderer.create_surface(id, size, format));
            }
            Message::DestroySurface { id } => {
                renderer.surfaces.remove(&id);
            }
            Message::AddFont { id, data, index } => {
                renderer.fonts.insert(id, FontData { data, index });
            }
            Message::Commit { surface, changes } => {
                renderer.commit(surface, changes);
            }
            Message::Render { time, reply } => {
                // The frame time exists for future scheduling; this slice
                // renders immediately.
                let _ = time;
                let _ = reply.send(renderer.render_frame());
            }
            Message::Readback { surface, reply } => {
                let _ = reply.send(renderer.readback(surface));
            }
            Message::Memory { reply } => {
                let _ = reply.send(renderer.memory());
            }
            Message::Trim(pressure) => {
                if pressure == Pressure::Critical {
                    renderer.fonts.shrink_to_fit();
                    renderer.glyph_cache.clear();
                }
            }
            Message::Shutdown => break,
        }
    }
}

/// The render thread's reply to [`crate::Engine::new`].
pub struct Init {
    /// Worker pool info.
    pub info: RasterInfo,
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

impl Renderer {
    /// Creates a surface's framebuffer and layer tree.
    fn create_surface(
        &mut self,
        id: SurfaceId,
        size: (u32, u32),
        format: OffscreenFormat,
    ) -> Result<(), SurfaceError> {
        if size.0 > MAX_SURFACE || size.1 > MAX_SURFACE {
            return Err(SurfaceError::TooLarge {
                width: size.0,
                height: size.1,
                max: MAX_SURFACE,
            });
        }
        let mut layers = HashMap::new();
        layers.insert(0, node());
        self.surfaces.insert(
            id,
            SurfaceState {
                size,
                format,
                fb: vec![[0.0; 4]; (size.0 * size.1) as usize],
                layers,
                clear: cherenkov::WorkingColor::TRANSPARENT,
                dirty: true,
            },
        );
        Ok(())
    }

    /// Applies one surface's change set.
    fn commit(&mut self, surface: SurfaceId, changes: ChangeSet) {
        let Some(state) = self.surfaces.get_mut(&surface) else {
            return;
        };
        if let Some(clear) = changes.clear {
            state.clear = clear;
        }
        for op in changes.ops {
            state.dirty = true;
            match op {
                LayerOp::Create(id) => {
                    state.layers.entry(id).or_insert_with(node);
                }
                LayerOp::Remove(id) => {
                    Self::remove_node(&mut state.layers, id);
                }
                LayerOp::Transform(id, t) => {
                    if let Some(node) = state.layers.get_mut(&id) {
                        node.transform = t;
                    }
                }
                LayerOp::Opacity(id, o) => {
                    if let Some(node) = state.layers.get_mut(&id) {
                        node.opacity = o;
                    }
                }
                LayerOp::Clip(id, clip) => {
                    if let Some(node) = state.layers.get_mut(&id) {
                        node.clip = clip;
                    }
                }
                LayerOp::Content(id, picture) => {
                    if let Some(node) = state.layers.get_mut(&id) {
                        node.content = picture.map(ContentData::Picture);
                    }
                }
                LayerOp::ContentChange(id, change) => {
                    if let Some(node) = state.layers.get_mut(&id) {
                        match change {
                            ContentChange::Replace(picture) => {
                                node.content = Some(ContentData::Picture(picture));
                            }
                            ContentChange::Update(updates) => {
                                if let Some(ContentData::Picture(picture)) = &mut node.content {
                                    let _ = picture.apply(updates);
                                }
                            }
                        }
                    }
                }
                LayerOp::Push { parent, child } => {
                    Self::detach(&mut state.layers, child);
                    if let Some(node) = state.layers.get_mut(&parent) {
                        node.children.push(child);
                    }
                }
                LayerOp::Insert {
                    parent,
                    index,
                    child,
                } => {
                    Self::detach(&mut state.layers, child);
                    if let Some(node) = state.layers.get_mut(&parent) {
                        node.children.insert(index.min(node.children.len()), child);
                    }
                }
                LayerOp::Detach { parent, child } => {
                    if let Some(node) = state.layers.get_mut(&parent) {
                        node.children.retain(|c| *c != child);
                    }
                }
            }
        }
    }

    /// Removes `child` from every child list holding it.
    fn detach(layers: &mut HashMap<LayerId, LayerNode>, child: LayerId) {
        for node in layers.values_mut() {
            node.children.retain(|c| *c != child);
        }
    }

    /// Removes a node and its descendants.
    fn remove_node(layers: &mut HashMap<LayerId, LayerNode>, id: LayerId) {
        Self::detach(layers, id);
        if let Some(node) = layers.remove(&id) {
            for child in node.children {
                Self::remove_node(layers, child);
            }
        }
    }

    /// Memory usage across framebuffers and the glyph mask cache.
    fn memory(&self) -> MemoryUsage {
        let framebuffers = self
            .surfaces
            .values()
            .map(|s| u64::from(s.size.0) * u64::from(s.size.1) * 16)
            .sum();
        MemoryUsage {
            framebuffers: Bytes(framebuffers),
            glyph_cache: Bytes(self.glyph_cache.bytes()),
        }
    }

    /// Lowers and rasterizes every dirty surface.
    fn render_frame(&mut self) -> Result<(Next, FrameStats), RenderError> {
        let mut stats = FrameStats::default();
        let mut dirty: Vec<SurfaceId> = self
            .surfaces
            .iter()
            .filter(|(_, s)| s.dirty)
            .map(|(id, _)| *id)
            .collect();
        dirty.sort_unstable();
        if dirty.is_empty() {
            return Ok((Next::Idle, stats));
        }
        for id in dirty {
            self.render_surface(id, &mut stats)?;
        }
        Ok((Next::Idle, stats))
    }

    /// Lowers and rasterizes one surface's frame.
    #[expect(
        clippy::many_single_char_names,
        reason = "w/h and r/g/b/a are the natural names"
    )]
    fn render_surface(&mut self, id: SurfaceId, stats: &mut FrameStats) -> Result<(), RenderError> {
        let mut items: Vec<Item> = Vec::new();
        let mut glyph_reqs = Vec::new();
        // Lowering borrows the layer map; the surface borrow ends before
        // glyph resolution touches `self.fonts`/`self.glyph_cache`.
        let (lowered, clear_color) = {
            let Some(surf) = self.surfaces.get_mut(&id) else {
                return Ok(());
            };
            let layers = std::mem::take(&mut surf.layers);
            let result = if let Some(root) = layers.get(&0) {
                let mut lowering = Lowering::new(&mut items, surf.size);
                let result = lowering.run(root, &layers, surf.clear);
                glyph_reqs = std::mem::take(&mut lowering.glyphs);
                result
            } else {
                Ok(())
            };
            surf.layers = layers;
            (result, surf.clear)
        };
        lowered?;
        self.resolve_glyphs(&glyph_reqs)?;
        let Some(surf) = self.surfaces.get_mut(&id) else {
            return Ok(());
        };
        let (w, h) = (surf.size.0 as usize, surf.size.1 as usize);
        let [r, g, b, a] = clear_color.components;
        let clear = [r * a, g * a, b * a, a];
        surf.fb.fill(clear);
        let pool = &self.pool;
        let fb = &mut surf.fb;
        let (draws, edges) = pool.install(|| raster::render_bands(&items, clear, fb, w, h));
        stats.draws += draws;
        stats.instances += edges;
        stats.passes += u32::try_from(h.div_ceil(raster::BAND_H)).unwrap_or(u32::MAX);
        surf.dirty = false;
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

    /// Materializes a surface's framebuffer into `Readback` pixels,
    /// rounding through `f16` for [`OffscreenFormat::LinearF16`].
    fn readback(&self, surface: SurfaceId) -> Result<Readback, RenderError> {
        let Some(state) = self.surfaces.get(&surface) else {
            return Err(RenderError::Readback("unknown surface".into()));
        };
        let pixels = match state.format {
            OffscreenFormat::LinearF32 => state.fb.clone(),
            OffscreenFormat::LinearF16 => state
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
}
