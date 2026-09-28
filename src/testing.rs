// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A test backend: [`Null`] draws nothing and reports every render-thread
//! call as an [`Event`] on a channel, so tests and the cross-backend
//! behaviour suite can assert what the front end committed.

use std::collections::HashSet;
use std::sync::mpsc::Sender;

use kurbo::{Affine, Vec2};

use crate::backend::{Backend, Frame, Redraw, Renderer, SurfaceInfo};
use crate::capability::{ShaderPaint, ShaderSource};
use crate::config::MemoryUsage;
use crate::error::{EngineError, RenderError, ResourceError, SurfaceError};
use crate::frame::Readback;
use crate::glyph::FontId;
use crate::image::{ImageUpload, Rgba8, Rgba16F};
use crate::message::{ContentOp, FontData, LayerId, SurfaceId};
use crate::paint::{ImageId, ShaderId};
use crate::{Offscreen, Pressure, Uploads};

/// A render-thread event [`Null`] reports.
#[derive(Debug)]
#[non_exhaustive]
pub enum Event {
    /// `create_surface` ran.
    CreateSurface(SurfaceId),
    /// `resize_surface` ran.
    ResizeSurface(SurfaceId, (u32, u32)),
    /// `destroy_surface` ran.
    DestroySurface(SurfaceId),
    /// `add_font` ran.
    AddFont(FontId),
    /// `remove_font` ran.
    RemoveFont(FontId),
    /// `add_image` ran.
    AddImage(ImageId),
    /// `remove_image` ran.
    RemoveImage(ImageId),
    /// `add_shader` ran.
    AddShader(ShaderId),
    /// `remove_shader` ran.
    RemoveShader(ShaderId),
    /// `set_content` ran.
    SetContent(SurfaceId, LayerId),
    /// `remove_layer` ran.
    RemoveLayer(SurfaceId, LayerId),
    /// One rendered surface, in `frame.surfaces` order.
    Frame(FrameRecord),
}

/// One surface's sampled tree, recorded per rendered frame.
#[derive(Debug)]
pub struct FrameRecord {
    /// The surface.
    pub surface: SurfaceId,
    /// The frame's `changed` flag for this surface.
    pub changed: bool,
    /// Every layer's sampled state.
    pub layers: Vec<LayerSample>,
}

/// One layer's sampled state in a [`FrameRecord`].
#[derive(Debug)]
pub struct LayerSample {
    /// The layer.
    pub id: LayerId,
    /// The sampled transform.
    pub transform: Affine,
    /// The sampled opacity.
    pub opacity: f32,
    /// The sampled, pixel-snapped scroll offset.
    pub scroll_offset: Vec2,
    /// The layer's children, in paint order.
    pub children: Vec<LayerId>,
}

/// A backend that draws nothing and reports every call. The `Config`
/// carries the test's probe channel.
#[derive(Clone, Copy, Debug, Default)]
pub struct Null;

/// Configuration for [`Null`]: the channel events are reported on.
pub struct NullConfig {
    /// The event probe. A test keeps the matching `Receiver`.
    pub events: Sender<Event>,
    /// Registration kinds the backend refuses: the matching `add_*` or
    /// `create_surface` call returns its error without committing.
    pub reject: HashSet<NullReject>,
}

/// A registration [`Null`] refuses, for failure-path tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NullReject {
    /// `create_surface` returns [`SurfaceError::UnsupportedTarget`].
    Surface,
    /// `add_font` returns [`ResourceError::Font`].
    Font,
    /// `add_image` returns [`ResourceError::Image`].
    Image,
    /// `add_shader` returns [`ResourceError::Shader`].
    Shader,
}

/// `Null`'s provenance: nothing to report.
pub type NullInfo = ();

/// The `Null` render-thread state. Registration and removal events are
/// reported only for committed transitions — a `remove_*` for an id the
/// backend never added is a no-op, as in a real backend.
pub struct NullRenderer {
    events: Sender<Event>,
    reject: HashSet<NullReject>,
    surfaces: HashSet<SurfaceId>,
    fonts: HashSet<FontId>,
    images: HashSet<ImageId>,
    shaders: HashSet<ShaderId>,
}

impl NullRenderer {
    fn new(config: NullConfig) -> Self {
        Self {
            events: config.events,
            reject: config.reject,
            surfaces: HashSet::new(),
            fonts: HashSet::new(),
            images: HashSet::new(),
            shaders: HashSet::new(),
        }
    }
}

impl Backend for Null {
    type Config = NullConfig;
    type Info = NullInfo;
    type Target = Offscreen;
    type Renderer = NullRenderer;

    #[cfg(not(target_arch = "wasm32"))]
    fn init(config: NullConfig) -> Result<(NullRenderer, NullInfo), EngineError> {
        Ok((NullRenderer::new(config), ()))
    }

    #[cfg(target_arch = "wasm32")]
    async fn init(config: NullConfig) -> Result<(NullRenderer, NullInfo), EngineError> {
        Ok((NullRenderer::new(config), ()))
    }
}

impl Renderer for NullRenderer {
    type Target = Offscreen;

    fn create_surface(
        &mut self,
        id: SurfaceId,
        target: Offscreen,
    ) -> Result<SurfaceInfo, SurfaceError> {
        if target.size.0 == 0 || target.size.1 == 0 {
            return Err(SurfaceError::ZeroSize);
        }
        if self.reject.contains(&NullReject::Surface) {
            return Err(SurfaceError::UnsupportedTarget("injected rejection".into()));
        }
        self.surfaces.insert(id);
        let _ = self.events.send(Event::CreateSurface(id));
        Ok(SurfaceInfo {
            max_dimension: u32::MAX,
            size: target.size,
            readable: true,
        })
    }

    fn resize_surface(&mut self, id: SurfaceId, size: (u32, u32)) {
        let _ = self.events.send(Event::ResizeSurface(id, size));
    }

    fn destroy_surface(&mut self, id: SurfaceId) {
        if self.surfaces.remove(&id) {
            let _ = self.events.send(Event::DestroySurface(id));
        }
    }

    fn add_font(&mut self, id: FontId, _font: FontData) -> Result<(), ResourceError> {
        if self.reject.contains(&NullReject::Font) {
            return Err(ResourceError::Font("injected rejection".into()));
        }
        self.fonts.insert(id);
        let _ = self.events.send(Event::AddFont(id));
        Ok(())
    }

    fn remove_font(&mut self, id: FontId) {
        if self.fonts.remove(&id) {
            let _ = self.events.send(Event::RemoveFont(id));
        }
    }

    fn add_image(&mut self, id: ImageId, _image: ImageUpload) -> Result<(), ResourceError> {
        if self.reject.contains(&NullReject::Image) {
            return Err(ResourceError::Image("injected rejection".into()));
        }
        self.images.insert(id);
        let _ = self.events.send(Event::AddImage(id));
        Ok(())
    }

    fn remove_image(&mut self, id: ImageId) {
        if self.images.remove(&id) {
            let _ = self.events.send(Event::RemoveImage(id));
        }
    }

    fn set_content(&mut self, surface: SurfaceId, layer: LayerId, _content: Option<ContentOp>) {
        let _ = self.events.send(Event::SetContent(surface, layer));
    }

    fn remove_layer(&mut self, surface: SurfaceId, layer: LayerId) {
        let _ = self.events.send(Event::RemoveLayer(surface, layer));
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn render(
        &mut self,
        frame: &Frame<'_>,
        _stats: &mut crate::FrameStats,
    ) -> Result<Redraw, RenderError> {
        for surface in frame.surfaces {
            let layers = surface
                .tree
                .layers()
                .map(|(id, node)| LayerSample {
                    id,
                    transform: node.transform,
                    opacity: node.opacity,
                    scroll_offset: node.scroll_offset,
                    children: node.children.clone(),
                })
                .collect();
            let _ = self.events.send(Event::Frame(FrameRecord {
                surface: surface.id,
                changed: surface.changed,
                layers,
            }));
        }
        Ok(Redraw::None)
    }

    #[cfg(target_arch = "wasm32")]
    async fn render(
        &mut self,
        frame: &Frame<'_>,
        _stats: &mut crate::FrameStats,
    ) -> Result<Redraw, RenderError> {
        for surface in frame.surfaces {
            let layers = surface
                .tree
                .layers()
                .map(|(id, node)| LayerSample {
                    id,
                    transform: node.transform,
                    opacity: node.opacity,
                    scroll_offset: node.scroll_offset,
                    children: node.children.clone(),
                })
                .collect();
            let _ = self.events.send(Event::Frame(FrameRecord {
                surface: surface.id,
                changed: surface.changed,
                layers,
            }));
        }
        Ok(Redraw::None)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn readback(&mut self, surface: SurfaceId) -> Result<Readback, RenderError> {
        let _ = surface;
        Ok(Readback {
            width: 0,
            height: 0,
            pixels: Vec::new(),
        })
    }

    #[cfg(target_arch = "wasm32")]
    async fn readback(&mut self, surface: SurfaceId) -> Result<Readback, RenderError> {
        let _ = surface;
        Ok(Readback {
            width: 0,
            height: 0,
            pixels: Vec::new(),
        })
    }

    fn memory(&self) -> MemoryUsage {
        MemoryUsage::default()
    }

    fn trim(&mut self, _pressure: Pressure) {}
}

impl ShaderPaint for Null {
    #[cfg(not(target_arch = "wasm32"))]
    fn add_shader(
        r: &mut NullRenderer,
        id: ShaderId,
        _source: ShaderSource,
    ) -> Result<(), ResourceError> {
        if r.reject.contains(&NullReject::Shader) {
            return Err(ResourceError::Shader("injected rejection".into()));
        }
        r.shaders.insert(id);
        let _ = r.events.send(Event::AddShader(id));
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    async fn add_shader(
        r: &mut NullRenderer,
        id: ShaderId,
        _source: ShaderSource,
    ) -> Result<(), ResourceError> {
        if r.reject.contains(&NullReject::Shader) {
            return Err(ResourceError::Shader("injected rejection".into()));
        }
        r.shaders.insert(id);
        let _ = r.events.send(Event::AddShader(id));
        Ok(())
    }

    fn remove_shader(r: &mut NullRenderer, id: ShaderId) {
        if r.shaders.remove(&id) {
            let _ = r.events.send(Event::RemoveShader(id));
        }
    }
}

impl Uploads<Rgba8> for Null {}
impl Uploads<Rgba16F> for Null {}

/// Expands to one cross-backend behaviour suite.
///
/// The suite emits `#[test]` functions exercising the shared front end end
/// to end, observing only `Offscreen` readback pixels, so it is identical
/// for every backend.
///
/// Invoke once in a backend crate's `tests/behaviour.rs`:
///
/// ```ignore
/// cherenkov::behaviour_suite! {
///     backend: cherenkov_vello::Vello,
///     config: || cherenkov_vello::VelloConfig::default(),
///     uploads: true,
/// }
/// ```
///
/// `uploads` gates the image-lifetime test on backends implementing
/// `Uploads<Rgba8>`. GPU backends need the driver environment set by the
/// caller (on this box, lavapipe via `VK_ICD_FILENAMES`/`WGPU_BACKEND`).
/// Available with the `testing` feature.
#[cfg(feature = "testing")]
#[macro_export]
macro_rules! behaviour_suite {
    { backend: $backend:ty, config: $config:expr, uploads: true $(,)? } => {
        $crate::behaviour_suite! { @impl $backend, $config }
        /// Image-lifetime checks for backends implementing
        /// `Uploads<Rgba8>`.
        mod behaviour_suite_uploads {
            use std::time::Duration;
use $crate::Instant;

            use $crate::{Draw as _, Engine, FrameTime, ImageData, Layer, Offscreen, OffscreenFormat, Readback, Rgba8, Surface, WorkingColor};
            use $crate::kurbo::{Affine, Rect, Vec2};

            /// The backend under test.
            type B = $backend;
            const TICK: Duration = Duration::from_nanos(1_000_000_000 / 120);
            fn engine() -> Option<Engine<B>> {
                Engine::<B>::new($config()).ok()
            }

            /// The last `Image` clone's drop queues `remove_image`, which
            /// the backend frees on the next render.
            #[test]
            fn the_last_image_drop_frees_its_memory() {
                let Some(engine) = engine() else { return };
                let _surface = engine
                    .surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))
                    .expect("surface");
                // A first render settles one-off allocations so the
                // baseline is stable.
                engine.render(FrameTime::at(Instant::now())).expect("render");
                let before = engine.memory().gpu.0 + engine.memory().cpu.0;
                let data = vec![255u8; 64 * 64 * 4];
                let image = engine
                    .image(ImageData::<Rgba8>::new(64, 64, data).expect("image data"))
                    .expect("image");
                let clone = image.clone();
                engine
                    .render(FrameTime::at(Instant::now() + TICK))
                    .expect("render");
                let with_image = engine.memory().gpu.0 + engine.memory().cpu.0;
                assert!(with_image > before, "memory {with_image} <= {before}");
                drop(image);
                drop(clone);
                engine
                    .render(FrameTime::at(Instant::now() + TICK * 2))
                    .expect("render");
                let after = engine.memory().gpu.0 + engine.memory().cpu.0;
                assert!(after < with_image, "memory {after} >= {with_image}");
            }
        }
    };
    { backend: $backend:ty, config: $config:expr, uploads: false $(,)? } => {
        $crate::behaviour_suite! { @impl $backend, $config }
    };
    { @impl $backend:ty, $config:expr } => {
        mod behaviour_suite {
            use std::time::Duration;
use $crate::Instant;

            use ::nami::SignalExt as _;
            use $crate::kurbo::{Affine, Rect, Vec2};
            use $crate::{
                Animation, Curve, Decay, Draw as _, Engine, FrameTime, ImageData, Layer, Next,
                Offscreen, OffscreenFormat, Readback, Rgba8, Spring, Surface, WorkingColor,
            };

            /// The backend under test.
            type B = $backend;

            /// One frame at 120 Hz, the suite's sampling step.
            const TICK: Duration = Duration::from_nanos(1_000_000_000 / 120);
            /// An opaque white 16×16 square.
            const WHITE: WorkingColor = WorkingColor::WHITE;

            /// A new engine, or `None` when the backend cannot init here
            /// (a GPU backend without an adapter skips its tests).
            fn engine() -> Option<Engine<B>> {
                Engine::<B>::new($config()).ok()
            }

            /// A 256×64 `Offscreen` surface — wide enough that a scrolled
            /// or translated square stays in view.
            fn surface(engine: &Engine<B>) -> Surface<B> {
                engine
                    .surface(Offscreen::new((256, 64), OffscreenFormat::LinearF16))
                    .expect("surface")
            }

            /// A child layer of `parent` holding an opaque square at
            /// `rect`, in the layer's local space.
            fn square(surface: &Surface<B>, parent: &Layer, rect: Rect) -> Layer {
                let layer = surface.layer();
                let content = surface.record(|c| c.fill(rect, WHITE));
                surface.update(|tx| {
                    tx[parent].push(&layer);
                    tx[&layer].content(content);
                });
                layer
            }

            /// The alpha-weighted centroid of pixels inside `region`
            /// (pixel centres, alpha above 0.25). `None` when the region
            /// holds no opaque pixels.
            fn square_center_in(readback: &Readback, region: Rect) -> Option<(f64, f64)> {
                let (mut sx, mut sy, mut sw) = (0.0, 0.0, 0.0);
                for y in region.min_y().max(0.0) as u32..region.max_y() as u32 {
                    for x in region.min_x().max(0.0) as u32..region.max_x() as u32 {
                        let a = f64::from(readback.pixels[(y * readback.width + x) as usize][3]);
                        if a > 0.25 {
                            sx = a.mul_add(f64::from(x) + 0.5, sx);
                            sy = a.mul_add(f64::from(y) + 0.5, sy);
                            sw += a;
                        }
                    }
                }
                (sw > 0.5).then_some((sx / sw, sy / sw))
            }

            /// The alpha-weighted centroid of every opaque pixel.
            fn square_center(readback: &Readback) -> Option<(f64, f64)> {
                square_center_in(
                    readback,
                    Rect::new(0.0, 0.0, readback.width.into(), readback.height.into()),
                )
            }

            /// The alpha channel of one pixel.
            fn alpha_at(readback: &Readback, x: u32, y: u32) -> f64 {
                f64::from(readback.pixels[(y * readback.width + x) as usize][3])
            }

            /// Renders at `t` and returns the `Next`.
            fn render_at(engine: &Engine<B>, t: Instant) -> Next {
                engine.render(FrameTime::at(t)).expect("render")
            }

            /// Renders frames at `t`, `t + TICK`, … until `Next::Idle`
            /// (cap 2000 frames) and returns the last sampled position.
            fn settle(engine: &Engine<B>, surface: &Surface<B>, t0: Instant) -> (f64, f64) {
                let mut t = t0;
                for _ in 0..2000 {
                    if render_at(engine, t) == Next::Idle {
                        break;
                    }
                    t += TICK;
                }
                square_center(&surface.readback().expect("readback")).expect("a drawn square")
            }

            #[test]
            fn layer_tree_edits_change_what_is_drawn() {
                let Some(engine) = engine() else { return };
                let surface = surface(&engine);
                let window_a = Rect::new(4.0, 24.0, 20.0, 44.0);
                let window_b = Rect::new(64.0, 24.0, 80.0, 44.0);
                let t0 = Instant::now();
                let mut frame = 0u64;
                let mut render = |engine: &Engine<B>| {
                    frame += 1;
                    render_at(engine, t0 + TICK * frame as u32)
                };

                let a = square(&surface, &surface.root(), Rect::new(8.0, 28.0, 16.0, 40.0));
                render(&engine);
                let rb = surface.readback().expect("readback");
                assert!(square_center_in(&rb, window_a).is_some(), "a not drawn");
                assert!(square_center_in(&rb, window_b).is_none(), "b drawn early");

                // `insert` at index 0 puts b first; both are drawn.
                let b = surface.layer();
                let content = surface.record(|c| c.fill(Rect::new(68.0, 28.0, 76.0, 40.0), WHITE));
                surface.update(|tx| {
                    tx[surface.root()].insert(0, &b);
                    tx[&b].content(content);
                });
                render(&engine);
                let rb = surface.readback().expect("readback");
                assert!(square_center_in(&rb, window_a).is_some(), "a missing");
                assert!(square_center_in(&rb, window_b).is_some(), "b missing");

                // `remove` detaches a from the tree; only b is drawn.
                surface.update(|tx| {
                    tx[surface.root()].remove(&a);
                });
                render(&engine);
                let rb = surface.readback().expect("readback");
                assert!(square_center_in(&rb, window_a).is_none(), "a still drawn");
                assert!(square_center_in(&rb, window_b).is_some(), "b missing");

                // Re-attach a with `push`, then drop b: only a remains.
                surface.update(|tx| {
                    tx[surface.root()].push(&a);
                });
                drop(b);
                render(&engine);
                let rb = surface.readback().expect("readback");
                assert!(square_center_in(&rb, window_a).is_some(), "a missing");
                assert!(square_center_in(&rb, window_b).is_none(), "b still drawn");
            }

            #[test]
            fn a_transform_spring_settles_at_its_target() {
                let Some(engine) = engine() else { return };
                let surface = surface(&engine);
                // Square centre starts at (32, 32); the spring targets
                // translate(16, 0) → (48, 32).
                let layer = square(&surface, &surface.root(), Rect::new(24.0, 24.0, 40.0, 40.0));
                surface.update(|tx| {
                    tx[&layer].transform(Affine::IDENTITY);
                });
                surface.update_animated(
                    Spring {
                        response: 0.4,
                        damping: 1.0,
                    },
                    |tx| {
                        tx[&layer].transform(Affine::translate((16.0, 0.0)));
                    },
                );
                let t0 = Instant::now();
                render_at(&engine, t0);
                let (x0, y0) =
                    square_center(&surface.readback().expect("readback")).expect("square");
                assert!((x0 - 32.0).abs() < 0.5 && (y0 - 32.0).abs() < 0.5, "start {x0},{y0}");

                render_at(&engine, t0 + TICK * 6);
                let (x1, _) =
                    square_center(&surface.readback().expect("readback")).expect("square");
                assert!(x1 > x0 + 0.5 && x1 < 48.0, "mid {x1}");

                let (x2, y2) = settle(&engine, &surface, t0 + TICK * 6);
                assert!((x2 - 48.0).abs() < 0.5 && (y2 - 32.0).abs() < 0.5, "end {x2},{y2}");
            }

            #[test]
            fn a_curve_hits_its_endpoints_exactly() {
                let Some(engine) = engine() else { return };
                let surface = surface(&engine);
                let layer = square(&surface, &surface.root(), Rect::new(24.0, 24.0, 40.0, 40.0));
                surface.update(|tx| {
                    tx[&layer].transform(Affine::IDENTITY);
                });
                surface.update_animated(
                    Curve::ease_in_out(Duration::from_millis(160)),
                    |tx| {
                        tx[&layer].transform(Affine::translate((16.0, 0.0)));
                    },
                );
                let t0 = Instant::now();
                render_at(&engine, t0);
                let (x0, _) =
                    square_center(&surface.readback().expect("readback")).expect("square");
                assert!((x0 - 32.0).abs() < 0.5, "start {x0}");

                // At and past the duration the value is exactly the target.
                let next = render_at(&engine, t0 + Duration::from_millis(200));
                assert_eq!(next, Next::Idle, "{next:?}");
                let (x1, _) =
                    square_center(&surface.readback().expect("readback")).expect("square");
                assert!((x1 - 48.0).abs() < 0.01, "end {x1}");
            }

            #[test]
            fn a_retargeted_spring_keeps_its_velocity() {
                let Some(engine) = engine() else { return };
                let surface = surface(&engine);
                let layer = square(&surface, &surface.root(), Rect::new(8.0, 24.0, 24.0, 40.0));
                // A linear curve runs at constant velocity — the
                // pre-retarget samples give v exactly, so the post-retarget
                // step can be checked against v·dt at 1%. (A spring carrier
                // drifts several % per frame across the 2-frame sampling
                // gap, making 1% unreachable through pixel centroids.)
                surface.update(|tx| {
                    tx[&layer].transform(Affine::IDENTITY);
                });
                surface.update_animated(
                    Curve::linear(Duration::from_millis(400)),
                    |tx| {
                        tx[&layer].transform(Affine::translate((160.0, 0.0)));
                    },
                );
                let t0 = Instant::now();
                // Mid-flight at t1: measure the incoming velocity from the
                // two samples just before it.
                let t1 = t0 + TICK * 8;
                render_at(&engine, t1 - TICK);
                let (c0, _) =
                    square_center(&surface.readback().expect("readback")).expect("square");
                render_at(&engine, t1);
                let (c1, _) =
                    square_center(&surface.readback().expect("readback")).expect("square");
                let dt = TICK.as_secs_f64();
                let v = (c1 - c0) / dt;
                assert!(v > 30.0, "not mid-flight: v={v}");

                // Retarget into a very soft, critically damped spring
                // (response 4 s, ω = 2π/4). It starts at the committed
                // position with the inherited velocity v and follows
                // x(t) = x_t + (C1 + C2·t)·e^(−ωt), C1 = x0 − x_t, C2 = v + ω·C1,
                // which over one frame is about 1% short of v·dt: the step
                // is checked against that trajectory, so it tests the
                // inherited velocity rather than the spring's own pull.
                surface.update(|tx| {
                    tx[&layer]
                        .transform(Affine::translate((160.0, 16.0)))
                        .animation(Spring {
                            response: 4.0,
                            damping: 1.0,
                        });
                });
                // The commit frame samples the new track at dt = 0; the
                // step after it must equal the incoming velocity · dt.
                render_at(&engine, t1 + TICK);
                let (c2, _) =
                    square_center(&surface.readback().expect("readback")).expect("square");
                render_at(&engine, t1 + TICK * 2);
                let (c3, _) =
                    square_center(&surface.readback().expect("readback")).expect("square");
                // The square's centre at the target: 16 + 160.
                let target = 176.0;
                let omega = std::f64::consts::TAU / 4.0;
                let offset = c2 - target;
                let expected = omega
                    .mul_add(offset, v)
                    .mul_add(dt, offset)
                    .mul_add((-omega * dt).exp(), -offset);
                // While the transform animates the layer's translation is
                // placed on the ¼-pixel grid, so each measured centre carries
                // up to ~¼ px of quantization and centroid bias; the same
                // uncertainty feeds back through v.
                assert!(
                    (c3 - c2 - expected).abs() <= (v * dt).abs().mul_add(0.01, 0.75),
                    "displacement {} vs the spring's trajectory {expected} (v·dt {})",
                    c3 - c2,
                    v * dt
                );
            }

            #[test]
            fn a_scroll_decay_covers_velocity_over_deceleration() {
                let Some(engine) = engine() else { return };
                let surface = surface(&engine);
                // Square centre (200, 32); a (600, 0)/k=4 decay covers
                // 150 px of scroll, moving the square 150 px left.
                let layer = square(&surface, &surface.root(), Rect::new(192.0, 24.0, 208.0, 40.0));
                surface.update(|tx| {
                    tx[&layer].scroll_offset(Vec2::ZERO);
                });
                surface.update(|tx| {
                    tx[&layer]
                        .scroll_offset(Vec2::ZERO)
                        .animation(Decay {
                            velocity: Vec2::new(600.0, 0.0),
                            deceleration: 4.0,
                            rubber_band: None,
                        });
                });
                let t0 = Instant::now();
                let (x2, y2) = settle(&engine, &surface, t0);
                // Pixel-snapped: 150 ± 1.
                assert!((x2 - 50.0).abs() < 1.0 && (y2 - 32.0).abs() < 1.0, "end {x2},{y2}");
            }

            #[test]
            fn a_rubber_band_returns_to_the_bound_edge() {
                let Some(engine) = engine() else { return };
                let surface = surface(&engine);
                let layer = square(&surface, &surface.root(), Rect::new(192.0, 24.0, 208.0, 40.0));
                // Bounds x ∈ [0, 100]: the decay overshoots to 150 and the
                // rubber band pulls it back to 100.
                let bounds = Rect::new(-10.0, -10.0, 100.0, 100.0);
                surface.update(|tx| {
                    tx[&layer]
                        .scroll_offset(Vec2::ZERO)
                        .animation(Decay::new(Vec2::new(600.0, 0.0)).rubber_band(bounds));
                });
                let t0 = Instant::now();
                let (x2, y2) = settle(&engine, &surface, t0);
                assert!((x2 - 100.0).abs() < 1.0 && (y2 - 32.0).abs() < 1.0, "end {x2},{y2}");
            }

            #[test]
            fn a_bound_signal_updates_opacity_without_a_transaction() {
                let Some(engine) = engine() else { return };
                let surface = surface(&engine);
                let layer = square(&surface, &surface.root(), Rect::new(24.0, 24.0, 40.0, 40.0));
                let opacity = ::nami::binding(1.0f32);
                surface.update(|tx| {
                    tx[&layer].opacity(opacity.clone());
                });
                let t0 = Instant::now();
                render_at(&engine, t0);
                let rb = surface.readback().expect("readback");
                assert!((alpha_at(&rb, 32, 32) - 1.0).abs() < 0.05);

                // A plain signal change snaps the opacity.
                opacity.set(0.4f32);
                let next = render_at(&engine, t0 + TICK);
                assert_eq!(next, Next::Idle, "{next:?}");
                let rb = surface.readback().expect("readback");
                assert!((alpha_at(&rb, 32, 32) - 0.4).abs() < 0.05, "alpha");

                // `with(Animation)` metadata animates the change: a sample
                // one frame in is strictly between the endpoints.
                let layer2 = square(&surface, &surface.root(), Rect::new(24.0, 24.0, 40.0, 40.0));
                surface.update(|tx| {
                    tx[surface.root()].remove(&layer);
                    tx[&layer2].opacity(opacity.clone().with(Animation::from(Spring::smooth())));
                });
                opacity.set(0.0f32);
                // The commit frame samples at dt = 0 → still 0.4.
                render_at(&engine, t0 + TICK * 3);
                let next = render_at(&engine, t0 + TICK * 8);
                let rb = surface.readback().expect("readback");
                let a = alpha_at(&rb, 32, 32);
                assert!(a > 0.02 && a < 0.39, "interpolated alpha {a}");
                assert!(matches!(next, Next::At { .. }), "{next:?}");
            }

            #[test]
            fn next_schedules_the_frame_rate() {
                let Some(engine) = engine() else { return };
                let surface = surface(&engine);
                let layer = square(&surface, &surface.root(), Rect::new(8.0, 24.0, 24.0, 40.0));
                surface.update_animated(
                    Spring {
                        response: 0.4,
                        damping: 1.0,
                    },
                    |tx| {
                        tx[&layer].transform(Affine::translate((160.0, 0.0)));
                    },
                );
                let t0 = Instant::now();
                // A running spring wants the fast class.
                match render_at(&engine, t0 + TICK) {
                    Next::At { rate, .. } => {
                        assert!(*rate.start() <= 60 && *rate.end() >= 120, "rate {rate:?}");
                    }
                    next => panic!("expected At while the spring runs, got {next:?}"),
                }

                // Once only a slow decay remains (under one device pixel
                // per 60 Hz frame), the rate drops to the slow class.
                let decay_layer =
                    square(&surface, &surface.root(), Rect::new(192.0, 24.0, 208.0, 40.0));
                let _ = layer; // keep the settled spring's layer alive
                surface.update(|tx| {
                    tx[&decay_layer]
                        .scroll_offset(Vec2::ZERO)
                        .animation(Decay::new(Vec2::new(30.0, 0.0)));
                });
                // Wait for the spring to settle; the decay outlives it
                // only briefly, so check the class on the first sample.
                let t2 = t0 + TICK * 200;
                let next = render_at(&engine, t2);
                match next {
                    Next::At { rate, .. } => {
                        assert!(*rate.end() <= 60, "rate {rate:?}");
                    }
                    Next::Idle => {}
                }
                // Everything comes to rest eventually.
                let _ = settle(&engine, &surface, t2);
                assert_eq!(render_at(&engine, t2 + TICK * 400), Next::Idle);
            }

        }
    };
}

/// Registration lifetime checks shared by both executors: the committed
/// `Add*`/`Remove*` (`CreateSurface`/`DestroySurface`) events of each
/// resource kind must balance (#150).
#[cfg(test)]
mod balance {
    use std::sync::mpsc::Receiver;

    use super::Event;

    /// The registration transitions in `events`, as `kind raw-id` strings.
    /// Returns `(committed, released)`.
    pub fn transitions(events: &[Event]) -> (Vec<String>, Vec<String>) {
        let (mut commits, mut releases) = (Vec::new(), Vec::new());
        for event in events {
            let (kind, id, committed) = match event {
                Event::AddFont(id) => ("font", id.raw(), true),
                Event::RemoveFont(id) => ("font", id.raw(), false),
                Event::AddImage(id) => ("image", id.raw(), true),
                Event::RemoveImage(id) => ("image", id.raw(), false),
                Event::AddShader(id) => ("shader", id.raw(), true),
                Event::RemoveShader(id) => ("shader", id.raw(), false),
                Event::CreateSurface(id) => ("surface", id.raw(), true),
                Event::DestroySurface(id) => ("surface", id.raw(), false),
                _ => continue,
            };
            let entry = format!("{kind} {id}");
            if committed {
                commits.push(entry);
            } else {
                releases.push(entry);
            }
        }
        (commits, releases)
    }

    /// Drains the probe and asserts every committed registration was
    /// released exactly once, kind and id matching.
    pub fn assert_balanced(rx: &Receiver<Event>) {
        let events: Vec<Event> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        let (mut commits, mut releases) = transitions(&events);
        commits.sort();
        releases.sort();
        assert_eq!(
            commits, releases,
            "unbalanced registration events: {events:?}"
        );
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use crate::Instant;
    use std::time::Duration;

    use super::*;
    use crate::image::ImageData;
    use crate::resource::FontSource;
    use crate::{Decay, Engine, FrameTime, Next, OffscreenFormat, ShaderSource, Spring};

    fn engine() -> (Engine<Null>, std::sync::mpsc::Receiver<Event>) {
        engine_rejecting(HashSet::new())
    }

    fn engine_rejecting(
        reject: HashSet<NullReject>,
    ) -> (Engine<Null>, std::sync::mpsc::Receiver<Event>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let engine = Engine::<Null>::new(NullConfig { events: tx, reject }).expect("init");
        (engine, rx)
    }

    fn frames(rx: &std::sync::mpsc::Receiver<Event>) -> Vec<FrameRecord> {
        let mut out = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let Event::Frame(record) = event {
                out.push(record);
            }
        }
        out
    }

    fn layer(record: &FrameRecord, id: LayerId) -> &LayerSample {
        record
            .layers
            .iter()
            .find(|l| l.id == id)
            .expect("layer in record")
    }

    /// A scroll axis pinned to one value (`x0 == x1` in the bounds) is a
    /// legal bounds rect: `Rect::contains` is half-open, so the rubber
    /// band must not fire for an offset inside such a rect.
    #[test]
    fn rubber_band_with_a_zero_width_bounds_rect() {
        let (engine, rx) = engine();
        let surface = engine
            .surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))
            .expect("surface");
        let layer_handle = surface.layer();
        let bounds = kurbo::Rect::new(0.0, 0.0, 0.0, 300.0);
        surface.update(|tx| {
            tx[surface.root()].push(&layer_handle);
            tx[&layer_handle]
                .scroll_offset(Vec2::new(0.0, 100.0))
                .animation(Decay::new(Vec2::new(0.0, 2000.0)).rubber_band(bounds));
        });
        let t0 = Instant::now();
        let mut t = t0;
        loop {
            t += Duration::from_millis(8);
            let next = engine.render(FrameTime::at(t)).expect("render");
            if matches!(next, Next::Idle) {
                break;
            }
            assert!(t - t0 < Duration::from_secs(10), "never settled");
        }
        let record = frames(&rx).pop().expect("records");
        let offset = layer(&record, layer_handle.id()).scroll_offset;
        assert_eq!(offset, Vec2::new(0.0, 300.0), "settled offset");
    }

    #[test]
    fn decay_starts_at_committed_value_and_stays_where_it_stops() {
        let (engine, rx) = engine();
        let surface = engine
            .surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))
            .expect("surface");
        let layer_handle = surface.layer();
        surface.update(|tx| {
            tx[surface.root()].push(&layer_handle);
            tx[&layer_handle]
                .scroll_offset(Vec2::ZERO)
                .animation(Decay::new(Vec2::new(600.0, 0.0)));
        });
        let t0 = Instant::now();
        let mut t = t0;
        // Unbounded decay: position = v/k = 600/4 = 150, then Idle.
        loop {
            t += Duration::from_millis(8);
            let next = engine.render(FrameTime::at(t)).expect("render");
            if matches!(next, Next::Idle) {
                break;
            }
            assert!(t - t0 < Duration::from_secs(10), "never settled");
        }
        let record = frames(&rx).pop().expect("records");
        let offset = layer(&record, layer_handle.id()).scroll_offset;
        assert!(
            (offset.x - 150.0).abs() < 0.5 && offset.y.abs() < 0.5,
            "offset {offset:?}, expected ≈(150, 0)"
        );
    }

    #[test]
    fn resource_drop_reaches_the_renderer() {
        let (engine, rx) = engine();
        let _surface = engine
            .surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))
            .expect("surface");
        {
            let image = engine
                .image(ImageData::<Rgba8>::new(2, 2, vec![0u8; 16]).expect("image data"))
                .expect("image");
            let _ = image.id();
        }
        engine.render(FrameTime::now()).expect("render");
        let events: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        assert!(
            events.iter().any(|e| matches!(e, Event::RemoveImage(_))),
            "no RemoveImage in {events:?}"
        );
    }

    #[test]
    fn update_animated_fills_the_default() {
        let (engine, rx) = engine();
        let surface = engine
            .surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))
            .expect("surface");
        let layer_handle = surface.layer();
        surface.update_animated(Spring::bouncy(), |tx| {
            tx[surface.root()].push(&layer_handle);
            tx[&layer_handle].opacity(0.5f32);
        });
        let next = engine.render(FrameTime::now()).expect("render");
        assert!(matches!(next, Next::At { .. }), "{next:?}");
        let _ = frames(&rx);
    }

    #[test]
    fn layer_drop_queues_remove() {
        let (engine, rx) = engine();
        let surface = engine
            .surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))
            .expect("surface");
        {
            let layer_handle = surface.layer();
            surface.update(|tx| {
                tx[surface.root()].push(&layer_handle);
            });
        }
        engine.render(FrameTime::now()).expect("render");
        let events: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        assert!(
            events.iter().any(|e| matches!(e, Event::RemoveLayer(_, _))),
            "no RemoveLayer in {events:?}"
        );
    }

    #[test]
    fn live_recorded_operands_wake_the_idle_owner_and_disconnect_on_drop() {
        use crate::{Draw, WorkingColor};
        use std::cell::Cell;
        use std::rc::Rc;
        let (engine, _events) = engine();
        let surface = engine
            .surface(crate::Offscreen::new(
                (8, 8),
                crate::OffscreenFormat::LinearF16,
            ))
            .unwrap();
        let layer = surface.layer();
        let color = nami::binding(WorkingColor::WHITE);
        surface.update(|tx| {
            tx[surface.root()].push(&layer);
            tx[&layer].content(
                surface.record(|r| r.fill(kurbo::Rect::new(0., 0., 8., 8.), color.clone())),
            );
        });
        engine.render(FrameTime::now()).unwrap();
        let count = Rc::new(Cell::new(0));
        let wakes = count.clone();
        engine.set_waker(move || wakes.set(wakes.get() + 1));
        color.set(WorkingColor::BLACK);
        color.set(WorkingColor::WHITE);
        assert_eq!(
            count.get(),
            1,
            "live updates coalesce without host transactions"
        );
        engine.render(FrameTime::now()).unwrap();
        color.set(WorkingColor::BLACK);
        assert_eq!(count.get(), 2, "render re-arms host notification");
        drop(layer);
        engine.render(FrameTime::now()).unwrap();
        let before = count.get();
        color.set(WorkingColor::WHITE);
        assert_eq!(
            count.get(),
            before,
            "removed content cannot wake the engine"
        );
    }

    /// A completed registration releases its backend resource when the
    /// handle drops — the only cancellation window a synchronous engine
    /// has. The async executor's windows are covered by `wasm_tests`.
    #[test]
    fn completed_registrations_release_on_drop() {
        let (engine, rx) = engine();
        let font = engine.font(FontSource::bytes(vec![0u8; 8])).expect("font");
        let image = engine
            .image(ImageData::<Rgba8>::new(1, 1, vec![0u8; 4]).expect("image data"))
            .expect("image");
        let shader = engine
            .shader(ShaderSource::wgsl("@fragment fn f() { }"))
            .expect("shader");
        let surface = engine
            .surface(Offscreen::new((8, 8), OffscreenFormat::LinearF16))
            .expect("surface");
        drop(font);
        drop(image);
        drop(shader);
        drop(surface);
        // The reply lands only after every earlier message was applied.
        let _ = engine.memory();
        balance::assert_balanced(&rx);
    }

    /// A registration the backend refused enqueues no removal: nothing
    /// ever existed to release.
    #[test]
    fn rejected_registrations_enqueue_no_removal() {
        let reject = HashSet::from([
            NullReject::Surface,
            NullReject::Font,
            NullReject::Image,
            NullReject::Shader,
        ]);
        let (engine, rx) = engine_rejecting(reject);
        assert!(
            engine.font(FontSource::bytes(vec![0u8; 8])).is_err(),
            "font"
        );
        assert!(
            engine
                .image(ImageData::<Rgba8>::new(1, 1, vec![0u8; 4]).expect("image data"))
                .is_err(),
            "image"
        );
        assert!(
            engine
                .shader(ShaderSource::wgsl("@fragment fn f() { }"))
                .is_err(),
            "shader"
        );
        assert!(
            engine
                .surface(Offscreen::new((8, 8), OffscreenFormat::LinearF16))
                .is_err(),
            "surface"
        );
        let _ = engine.memory();
        balance::assert_balanced(&rx);
    }
}

/// Retained-lowering equivalence checks for first-party backends.
pub mod incremental;

/// The async executor's registration windows (#150): a registration
/// future owns the backend id from the moment the request is enqueued, so
/// dropping it at any point still releases what the backend committed.
/// Run under the Node `wasm-bindgen-test-runner` (or a browser):
///
/// ```console
/// CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
/// cargo test -p cherenkov --features testing --lib \
///     --target wasm32-unknown-unknown
/// ```
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use std::fmt::Debug;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::mpsc::Receiver;
    use std::task::{Context, Waker};

    use wasm_bindgen_test::wasm_bindgen_test;

    use super::balance;
    use super::{Event, Null, NullConfig, NullReject};
    use crate::image::ImageData;
    use crate::resource::FontSource;
    use crate::{Engine, Offscreen, OffscreenFormat, Rgba8, ShaderSource};

    async fn engine(
        reject: std::collections::HashSet<NullReject>,
    ) -> (Engine<Null>, Receiver<Event>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let engine = Engine::<Null>::new(NullConfig { events: tx, reject })
            .await
            .expect("init");
        (engine, rx)
    }

    /// Runs the registration future until it has enqueued its request and
    /// is parked on the reply.
    fn enqueued<F: Future>(registration: Pin<&mut F>) {
        assert!(
            registration
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending(),
            "registration returned before its reply await"
        );
    }

    /// Everything enqueued before this call has been applied: the
    /// `Memory` reply lands after them in the serial executor.
    async fn flush(engine: &Engine<Null>) {
        let _ = engine.memory().await;
    }

    /// Window (a): dropped after enqueue, before the backend ran.
    async fn drop_after_enqueue<F, O, E>(engine: &Engine<Null>, register: F)
    where
        F: Future<Output = Result<O, E>>,
        E: Debug,
    {
        let mut registration = Box::pin(register);
        enqueued(registration.as_mut());
        drop(registration);
        flush(engine).await;
    }

    /// Window (b): dropped after the backend committed, before the reply
    /// was adopted. The first `flush` returns once the commit is done.
    async fn drop_after_commit<F, O, E>(engine: &Engine<Null>, register: F)
    where
        F: Future<Output = Result<O, E>>,
        E: Debug,
    {
        let mut registration = Box::pin(register);
        enqueued(registration.as_mut());
        flush(engine).await;
        drop(registration);
        flush(engine).await;
    }

    /// Window (c): the reply was adopted and the handle dropped.
    async fn drop_after_reply<F, O, E>(engine: &Engine<Null>, register: F)
    where
        F: Future<Output = Result<O, E>>,
        E: Debug,
    {
        drop(register.await.expect("registration"));
        flush(engine).await;
    }

    /// Every window, for one registration kind.
    async fn all_windows<O, E>(
        engine: &Engine<Null>,
        register: impl for<'a> Fn(&'a Engine<Null>) -> Pin<Box<dyn Future<Output = Result<O, E>> + 'a>>,
    ) where
        E: Debug,
    {
        drop_after_enqueue(engine, register(engine)).await;
        drop_after_commit(engine, register(engine)).await;
        drop_after_reply(engine, register(engine)).await;
    }

    #[wasm_bindgen_test]
    async fn dropped_font_registration_still_releases_it() {
        let (engine, rx) = engine(Default::default()).await;
        all_windows(&engine, |engine| {
            Box::pin(engine.font(FontSource::bytes(vec![0u8; 8])))
        })
        .await;
        balance::assert_balanced(&rx);
    }

    #[wasm_bindgen_test]
    async fn dropped_image_registration_still_releases_it() {
        let (engine, rx) = engine(Default::default()).await;
        all_windows(&engine, |engine| {
            Box::pin(engine.image(ImageData::<Rgba8>::new(1, 1, vec![0u8; 4]).expect("image data")))
        })
        .await;
        balance::assert_balanced(&rx);
    }

    #[wasm_bindgen_test]
    async fn dropped_shader_registration_still_releases_it() {
        let (engine, rx) = engine(Default::default()).await;
        all_windows(&engine, |engine| {
            Box::pin(engine.shader(ShaderSource::wgsl("@fragment fn f() { }")))
        })
        .await;
        balance::assert_balanced(&rx);
    }

    #[wasm_bindgen_test]
    async fn dropped_surface_creation_still_destroys_it() {
        let (engine, rx) = engine(Default::default()).await;
        all_windows(&engine, |engine| {
            Box::pin(engine.surface(Offscreen::new((8, 8), OffscreenFormat::LinearF16)))
        })
        .await;
        balance::assert_balanced(&rx);
    }

    /// A registration the backend refused enqueues no removal — the
    /// resource never existed — whether the caller awaited the error or
    /// dropped the future on the way.
    #[wasm_bindgen_test]
    async fn rejected_registrations_enqueue_no_removal() {
        let reject = std::collections::HashSet::from([
            NullReject::Surface,
            NullReject::Font,
            NullReject::Image,
            NullReject::Shader,
        ]);
        let (engine, rx) = engine(reject).await;
        assert!(
            engine.font(FontSource::bytes(vec![0u8; 8])).await.is_err(),
            "font"
        );
        assert!(
            engine
                .image(ImageData::<Rgba8>::new(1, 1, vec![0u8; 4]).expect("image data"))
                .await
                .is_err(),
            "image"
        );
        assert!(
            engine
                .shader(ShaderSource::wgsl("@fragment fn f() { }"))
                .await
                .is_err(),
            "shader"
        );
        assert!(
            engine
                .surface(Offscreen::new((8, 8), OffscreenFormat::LinearF16))
                .await
                .is_err(),
            "surface"
        );
        // Dropped before the rejected reply: the enqueued removal finds
        // no committed resource and reports nothing.
        drop_after_enqueue(
            &engine,
            engine.image(ImageData::<Rgba8>::new(1, 1, vec![0u8; 4]).expect("image data")),
        )
        .await;
        flush(&engine).await;
        balance::assert_balanced(&rx);
    }
}
