// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cherenkov-vello` adapter: the Cherenkov front end on the Vello backend.
//!
//! Route: the front-end records a display list per content layer, lowered on
//! the render thread into a `vello::Scene` rendered by vello's wgpu renderer
//! into a `Rgba8Unorm` texture (premultiplied sRGB-encoded sRGB — vello
//! blends in the encoded target). GPU time is a real `wgpu` timestamp pair
//! drained inside [`cherenkov_vello::Engine::render`] with
//! `VelloConfig::timestamps` set.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use cherenkov::Draw as _;
use cherenkov::{
    Engine as VelloEngine, FontSource, ImageData, Layer as VelloLayer, Offscreen, OffscreenFormat,
    RenderError, Rgba8, Surface, Transaction,
};
use cherenkov_oracle::color::to_working;
use cherenkov_scene::{
    BlendMode, ColorSpace, Draw as SceneDraw, Extend, Feature, FillRule, GlyphRun as SceneGlyphRun,
    Item, Layer as SceneLayer, Paint as ScenePaint, ResourceHash, Shape,
};
use cherenkov_vello::{Vello, VelloConfig};
use kurbo::{Affine, BezPath, Circle, Ellipse, Line, Rect, RoundedRect, Vec2};

use crate::convert::{self, Blobs, Prepared};
use crate::motion::{Clock, LayerMotion};
use crate::{BenchError, Counters, DeviceInfo, EncodeInput, Engine, EngineInfo, Submit};

/// A scene shape in a form the front-end accepts.
enum ShapeKind {
    Rect(Rect),
    RoundedRect(RoundedRect),
    Continuous(cherenkov::ContinuousRect),
    Circle(Circle),
    Ellipse(Ellipse),
    Line(Line),
    Path(BezPath),
}

/// One recording step of a content layer, resolved in `prepare`.
enum Op {
    /// `Draw::Fill`.
    Fill {
        /// The shape.
        shape: ShapeKind,
        /// The paint.
        paint: cherenkov::Paint,
        /// Even-odd fill rule.
        even_odd: bool,
    },
    /// `Draw::Stroke`.
    Stroke {
        /// The shape.
        shape: ShapeKind,
        /// The stroke style.
        stroke: kurbo::Stroke,
        /// The paint.
        paint: cherenkov::Paint,
    },
    /// `Draw::Shadow`.
    Shadow {
        /// The shape.
        shape: ShapeKind,
        /// The shadow.
        shadow: cherenkov::Shadow,
    },
    /// `Draw::Glyphs`.
    Glyphs {
        /// The run.
        run: cherenkov::GlyphRun,
        /// The paint.
        paint: cherenkov::Paint,
    },
    /// `Draw::Image`.
    Image {
        /// The registered image.
        image: cherenkov::ImageId,
        /// Destination rect.
        dst: Rect,
        /// Sampling quality.
        sampling: cherenkov::Sampling,
    },
}

/// A prepared child item: a draw-item run wrapped in its own layer, or a
/// real child layer.
enum PrepItem {
    /// A maximal run of draw items, drawn as one layer's content.
    Content(Vec<Op>),
    /// A child scene layer.
    Layer(Box<PrepLayer>),
}

/// A scene layer lowered in `prepare`.
struct PrepLayer {
    /// Local transform.
    transform: Affine,
    /// Clip in the layer's own space.
    clip: Option<ShapeKind>,
    /// Group opacity.
    opacity: f64,
    /// Blend mode when compositing onto the parent.
    blend: cherenkov::BlendMode,
    /// Scroll offset applied to content and children.
    scroll_offset: Vec2,
    /// The layer's own content — only when every draw precedes every child.
    own: Vec<Op>,
    /// Ordered children.
    items: Vec<PrepItem>,
    /// The layer's one-time motion.
    motion: Option<LayerMotion>,
}

/// An engine layer plus the ops it records each frame.
struct ContentLayer {
    /// The layer handle.
    layer: VelloLayer,
    /// Its recorded ops.
    ops: Vec<Op>,
    /// The layer's one-time motion, committed on the first encode.
    motion: Option<LayerMotion>,
}

/// `cherenkov-vello` adapter.
pub struct CherenkovVello {
    info: EngineInfo,
    engine: VelloEngine<Vello>,
    surface: Option<Surface<Vello>>,
    /// Registered fonts per `(blob hash, face index)` (keeps the handles
    /// alive).
    fonts: HashMap<(ResourceHash, u32), cherenkov::Font>,
    /// Registered images per blob hash.
    images: HashMap<ResourceHash, cherenkov::Image<Rgba8>>,
    /// Layers holding recorded content, in draw order.
    content_layers: Vec<ContentLayer>,
    /// Decoded texel bytes prepared this scene (`bytes_uploaded`).
    bytes_uploaded: u64,
    /// Whether any layer carries a `motion`.
    has_motion: bool,
    /// Whether the motion commits have been sent (first encode).
    motion_committed: bool,
    /// The fixed frame clock `submit` renders at.
    clock: Clock,
    counters: Counters,
}

/// The features this slice executes faithfully: the honest Vello set,
/// minus what the Cherenkov front end cannot express.
///
/// [`Feature::ExtendNone`] is absent: neither `cherenkov::Extend` nor
/// `peniko::Extend` has a `None` variant. [`Feature::HdrColor`] and
/// [`Feature::WideGamut`] are absent: the backend renders into an
/// `rgba8unorm` sRGB target and clamps. Interpolation declares only what
/// `cherenkov::Interpolation` names (the working space and encoded sRGB),
/// which is still honest: `linear-p3`/`linear-srgb` both mean linear
/// working-space interpolation here, and `srgb` means encoded sRGB.
/// [`Feature::Shadow`] holds for the shapes vello's blurred-rounded-rect
/// primitive expresses (rect, uniform-radius rounded rect, circle); the
/// per-shape check happens in the draw walk and surfaces as
/// [`BenchError::Unsupported`].
fn vello_ad_features() -> Vec<Feature> {
    let mut v = vec![
        Feature::Fill,
        Feature::EvenOdd,
        Feature::Stroke,
        Feature::StrokeDash,
        Feature::Path,
        Feature::ContinuousCorners,
        Feature::LinearGradient,
        Feature::RadialGradient,
        Feature::SweepGradient,
        Feature::Image,
        Feature::ImagePaint,
        Feature::Clip,
        Feature::Opacity,
        Feature::Shadow,
        Feature::Glyphs,
        Feature::FontVariations,
        Feature::Scroll,
        Feature::Animation,
        Feature::InterpolationSpace(ColorSpace::Srgb),
        Feature::InterpolationSpace(ColorSpace::LinearSrgb),
        Feature::InterpolationSpace(ColorSpace::LinearP3),
    ];
    for m in [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::ColorDodge,
        BlendMode::ColorBurn,
        BlendMode::HardLight,
        BlendMode::SoftLight,
        BlendMode::Difference,
        BlendMode::Exclusion,
        BlendMode::Hue,
        BlendMode::Saturation,
        BlendMode::Color,
        BlendMode::Luminosity,
    ] {
        v.push(Feature::Blend(m));
    }
    v
}

/// The upstream API this adapter lacks for a declared scene feature.
const fn missing_api(f: &Feature) -> Option<&'static str> {
    match f {
        Feature::ExtendNone => Some(crate::convert::EXTEND_NONE_API),
        Feature::HdrColor | Feature::WideGamut => {
            Some("vello-family render targets are rgba8unorm sRGB — no HDR or wide-gamut output")
        }
        Feature::InterpolationSpace(_) => {
            Some("cherenkov::Interpolation has only Working and SrgbEncoded variants")
        }
        Feature::Blend(_) => {
            Some("cherenkov::BlendMode has only the 16 CSS mix modes — no Porter-Duff compositing")
        }
        _ => None,
    }
}

/// The scene [`Feature`] a render-time unsupported feature name maps back
/// to. The names are the backend's `RenderError::Unsupported` strings.
fn unsupported_feature(u: &str) -> Feature {
    match u {
        "gradient-interpolation" => Feature::InterpolationSpace(ColorSpace::Srgb),
        "blend-space" => Feature::Blend(BlendMode::Normal),
        "group-filter" | "filter" => Feature::Opacity,
        "image" => Feature::Image,
        "shadow" => Feature::Shadow,
        "glyph-transform" => Feature::Glyphs,
        _ => Feature::Fill,
    }
}

/// Maps a render-time error into a `BenchError`, keeping `Unsupported`
/// scenes reported rather than fatal.
fn render_error(e: RenderError) -> BenchError {
    match e {
        RenderError::Unsupported(u) => BenchError::Unsupported {
            engine: CherenkovVello::NAME,
            feature: unsupported_feature(u),
            api: Some(u),
        },
        e => BenchError::Gpu(format!("cherenkov-vello render: {e}")),
    }
}

/// A scene colour → the front-end's straight-alpha working colour.
#[expect(
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    reason = "the working space is f32 at the engine boundary"
)]
fn working(c: &cherenkov_scene::Color) -> cherenkov::WorkingColor {
    let [r, g, b, a] = to_working(c);
    let (r, g, b) = if a > 1e-12 {
        (r / a, g / a, b / a)
    } else {
        (0.0, 0.0, 0.0)
    };
    cherenkov::WorkingColor::new([r as f32, g as f32, b as f32, a as f32])
}

const fn extend(e: Extend) -> Result<cherenkov::Extend, BenchError> {
    match e {
        Extend::Pad => Ok(cherenkov::Extend::Pad),
        Extend::Repeat => Ok(cherenkov::Extend::Repeat),
        Extend::Reflect => Ok(cherenkov::Extend::Reflect),
        Extend::None => Err(BenchError::Unsupported {
            engine: CherenkovVello::NAME,
            feature: Feature::ExtendNone,
            api: missing_api(&Feature::ExtendNone),
        }),
    }
}

const fn interpolation(space: ColorSpace) -> Result<cherenkov::Interpolation, BenchError> {
    match space {
        ColorSpace::Srgb => Ok(cherenkov::Interpolation::SrgbEncoded),
        ColorSpace::LinearP3 | ColorSpace::LinearSrgb => Ok(cherenkov::Interpolation::Working),
        space => Err(BenchError::Unsupported {
            engine: CherenkovVello::NAME,
            feature: Feature::InterpolationSpace(space),
            api: missing_api(&Feature::InterpolationSpace(space)),
        }),
    }
}

fn blend(m: BlendMode) -> cherenkov::BlendMode {
    match m {
        BlendMode::Normal => cherenkov::BlendMode::Normal,
        BlendMode::Multiply => cherenkov::BlendMode::Multiply,
        BlendMode::Screen => cherenkov::BlendMode::Screen,
        BlendMode::Overlay => cherenkov::BlendMode::Overlay,
        BlendMode::Darken => cherenkov::BlendMode::Darken,
        BlendMode::Lighten => cherenkov::BlendMode::Lighten,
        BlendMode::ColorDodge => cherenkov::BlendMode::ColorDodge,
        BlendMode::ColorBurn => cherenkov::BlendMode::ColorBurn,
        BlendMode::HardLight => cherenkov::BlendMode::HardLight,
        BlendMode::SoftLight => cherenkov::BlendMode::SoftLight,
        BlendMode::Difference => cherenkov::BlendMode::Difference,
        BlendMode::Exclusion => cherenkov::BlendMode::Exclusion,
        BlendMode::Hue => cherenkov::BlendMode::Hue,
        BlendMode::Saturation => cherenkov::BlendMode::Saturation,
        BlendMode::Color => cherenkov::BlendMode::Color,
        BlendMode::Luminosity => cherenkov::BlendMode::Luminosity,
        m => unreachable!("compose mode {m:?} is gated by check_features"),
    }
}

const fn sampling(s: cherenkov_scene::Sampling) -> cherenkov::Sampling {
    match s {
        cherenkov_scene::Sampling::Nearest => cherenkov::Sampling::Nearest,
        cherenkov_scene::Sampling::Bilinear => cherenkov::Sampling::Linear,
    }
}

fn stops(stops: &[cherenkov_scene::GradientStop]) -> Vec<cherenkov::ColorStop> {
    stops
        .iter()
        .map(|s| cherenkov::ColorStop {
            offset: s.offset,
            color: working(&s.color),
        })
        .collect()
}

/// A scene paint → the front-end paint, resolving image patterns against
/// the registered images.
fn front_paint(
    paint: &ScenePaint,
    images: &HashMap<ResourceHash, cherenkov::Image<Rgba8>>,
) -> Result<cherenkov::Paint, BenchError> {
    Ok(match paint {
        ScenePaint::Solid(c) => cherenkov::Paint::Solid(working(c)),
        ScenePaint::Linear(g) => cherenkov::Paint::Linear(cherenkov::LinearGradient {
            start: g.start,
            end: g.end,
            stops: stops(&g.stops),
            extend: extend(g.extend)?,
            interpolation: interpolation(g.interpolation)?,
        }),
        ScenePaint::Radial(g) => cherenkov::Paint::Radial(cherenkov::RadialGradient {
            start_center: g.center0,
            start_radius: g.r0,
            end_center: g.center1,
            end_radius: g.r1,
            stops: stops(&g.stops),
            extend: extend(g.extend)?,
            interpolation: interpolation(g.interpolation)?,
        }),
        ScenePaint::Sweep(g) => cherenkov::Paint::Sweep(cherenkov::SweepGradient {
            center: g.center,
            start_angle: g.start_angle,
            end_angle: g.end_angle,
            stops: stops(&g.stops),
            extend: extend(g.extend)?,
            interpolation: interpolation(g.interpolation)?,
        }),
        ScenePaint::Image(p) => {
            let image = images
                .get(&p.image)
                .ok_or(cherenkov_scene::SceneError::MissingResource(p.image))?;
            cherenkov::Paint::Image(cherenkov::ImagePattern {
                image: image.id(),
                transform: p.transform,
                extend_x: extend(p.extend_x)?,
                extend_y: extend(p.extend_y)?,
                sampling: sampling(p.sampling),
            })
        }
    })
}

/// A scene shape → [`ShapeKind`]; arbitrary paths are expressible through
/// the front end's `ShapeData::Path`.
fn shape_kind(shape: &Shape) -> ShapeKind {
    match shape {
        Shape::Rect(r) => ShapeKind::Rect(*r),
        Shape::RoundedRect(r) => ShapeKind::RoundedRect(*r),
        Shape::Continuous(c) => ShapeKind::Continuous(
            cherenkov::ContinuousRect::new(c.rect, c.corner_radius).with_smoothing(c.smoothing),
        ),
        Shape::Circle(c) => ShapeKind::Circle(*c),
        Shape::Ellipse(e) => ShapeKind::Ellipse(*e),
        Shape::Line(l) => ShapeKind::Line(*l),
        Shape::Path { path } => ShapeKind::Path(path.clone()),
    }
}

/// A shape's clip applied to a layer edit.
fn clip_shape(edit: &mut cherenkov::LayerEdit<Vello>, shape: &ShapeKind) {
    match shape {
        ShapeKind::Rect(r) => drop(edit.clip(*r)),
        ShapeKind::RoundedRect(r) => drop(edit.clip(*r)),
        ShapeKind::Continuous(c) => drop(edit.clip(*c)),
        ShapeKind::Circle(c) => drop(edit.clip(*c)),
        ShapeKind::Ellipse(e) => drop(edit.clip(*e)),
        ShapeKind::Line(l) => drop(edit.clip(*l)),
        ShapeKind::Path(p) => drop(edit.clip(p.clone())),
    }
}

/// A scene draw → an [`Op`].
fn op(
    draw: &SceneDraw,
    fonts: &HashMap<(ResourceHash, u32), cherenkov::Font>,
    images: &HashMap<ResourceHash, cherenkov::Image<Rgba8>>,
    prepared: &Prepared,
) -> Result<Op, BenchError> {
    Ok(match draw {
        SceneDraw::Fill { shape, paint, rule } => Op::Fill {
            shape: shape_kind(shape),
            paint: front_paint(paint, images)?,
            even_odd: matches!(rule, FillRule::EvenOdd),
        },
        SceneDraw::Stroke {
            shape,
            stroke,
            paint,
        } => Op::Stroke {
            shape: shape_kind(shape),
            stroke: convert::stroke(stroke),
            paint: front_paint(paint, images)?,
        },
        SceneDraw::Shadow {
            shape,
            blur_sigma,
            offset,
            color,
        } => Op::Shadow {
            shape: shape_kind(shape),
            shadow: cherenkov::Shadow::new(*blur_sigma, working(color))
                .offset(Vec2::new(offset[0], offset[1])),
        },
        SceneDraw::Glyphs(run) => Op::Glyphs {
            run: glyph_run(run, fonts, prepared)?,
            paint: front_paint(&run.paint, images)?,
        },
        SceneDraw::Image {
            image,
            dst,
            sampling: s,
        } => Op::Image {
            image: images
                .get(image)
                .ok_or(cherenkov_scene::SceneError::MissingResource(*image))?
                .id(),
            dst: *dst,
            sampling: sampling(*s),
        },
    })
}

/// A scene glyph run → a front-end run with the registered font and the
/// resolved `F2Dot14` coordinates.
fn glyph_run(
    run: &SceneGlyphRun,
    fonts: &HashMap<(ResourceHash, u32), cherenkov::Font>,
    prepared: &Prepared,
) -> Result<cherenkov::GlyphRun, BenchError> {
    let font = fonts
        .get(&(run.font, run.font_index))
        .ok_or(cherenkov_scene::SceneError::MissingResource(run.font))?
        .id();
    let coords = prepared.coord_bits(run.font, &run.normalized_coords);
    Ok(cherenkov::GlyphRun {
        font,
        size: run.size,
        coords,
        glyphs: run
            .glyphs
            .iter()
            .map(|g| cherenkov::Glyph {
                id: g.id,
                x: g.x,
                y: g.y,
                transform: None,
            })
            .collect(),
        style: cherenkov::GlyphStyle::Fill,
    })
}

/// Registers every font and image a layer references, once each.
fn register_resources(
    fonts: &mut HashMap<(ResourceHash, u32), cherenkov::Font>,
    images: &mut HashMap<ResourceHash, cherenkov::Image<Rgba8>>,
    engine: &VelloEngine<Vello>,
    prepared: &Prepared,
    layer: &SceneLayer,
    blobs: &Blobs,
) -> Result<(), BenchError> {
    for item in &layer.items {
        match item {
            Item::Layer(l) => register_resources(fonts, images, engine, prepared, l, blobs)?,
            Item::Draw(d) => {
                if let SceneDraw::Glyphs(run) = d {
                    let key = (run.font, run.font_index);
                    if let std::collections::hash_map::Entry::Vacant(e) = fonts.entry(key) {
                        let blob = blobs
                            .get(&run.font)
                            .ok_or(cherenkov_scene::SceneError::MissingResource(run.font))?;
                        let font = engine
                            .font(FontSource::bytes(blob.clone()).with_index(run.font_index))
                            .map_err(|e| {
                                BenchError::Engine(format!("cherenkov-vello font: {e}"))
                            })?;
                        e.insert(font);
                    }
                }
                let image_hash = match d {
                    SceneDraw::Image { image, .. } => Some(*image),
                    SceneDraw::Fill { paint, .. }
                    | SceneDraw::Stroke { paint, .. }
                    | SceneDraw::Glyphs(SceneGlyphRun { paint, .. }) => match paint {
                        ScenePaint::Image(p) => Some(p.image),
                        _ => None,
                    },
                    SceneDraw::Shadow { .. } => None,
                };
                if let Some(hash) = image_hash
                    && let std::collections::hash_map::Entry::Vacant(e) = images.entry(hash)
                {
                    let data = prepared.image(hash)?;
                    let bytes: Arc<[u8]> = Arc::from(data.data.data());
                    let image = engine
                        .image(
                            ImageData::<Rgba8>::new(data.width, data.height, bytes)
                                .map_err(|e| BenchError::Engine(format!("image data: {e}")))?,
                        )
                        .map_err(|e| BenchError::Engine(format!("cherenkov-vello image: {e}")))?;
                    e.insert(image);
                }
            }
        }
    }
    Ok(())
}

/// Lowers a scene layer: one engine layer per scene layer, plus one per
/// draw run that must interleave with child layers.
fn prep_layer(
    layer: &SceneLayer,
    fonts: &HashMap<(ResourceHash, u32), cherenkov::Font>,
    images: &HashMap<ResourceHash, cherenkov::Image<Rgba8>>,
    prepared: &Prepared,
) -> Result<PrepLayer, BenchError> {
    // The engine draws a layer's content before its children, so the draws
    // are the layer's own content only when every draw precedes every child.
    let first_child = layer.items.iter().position(|i| matches!(i, Item::Layer(_)));
    let last_draw = layer.items.iter().rposition(|i| matches!(i, Item::Draw(_)));
    let own = match (first_child, last_draw) {
        (None, _) | (Some(_), None) => true,
        (Some(f), Some(l)) => l < f,
    };
    let mut prep = PrepLayer {
        transform: layer.transform,
        scroll_offset: layer.scroll_offset,
        clip: layer.clip.as_ref().map(shape_kind),
        opacity: layer.opacity,
        blend: blend(layer.blend),
        own: Vec::new(),
        items: Vec::new(),
        motion: layer
            .motion
            .as_ref()
            .map(|m| LayerMotion::from_scene(m, layer.transform)),
    };
    if own {
        for item in &layer.items {
            match item {
                Item::Draw(d) => prep.own.push(op(d, fonts, images, prepared)?),
                Item::Layer(l) => prep.items.push(PrepItem::Layer(Box::new(prep_layer(
                    l, fonts, images, prepared,
                )?))),
            }
        }
    } else {
        let mut run: Vec<Op> = Vec::new();
        for item in &layer.items {
            match item {
                Item::Draw(d) => run.push(op(d, fonts, images, prepared)?),
                Item::Layer(l) => {
                    if !run.is_empty() {
                        prep.items.push(PrepItem::Content(std::mem::take(&mut run)));
                    }
                    prep.items.push(PrepItem::Layer(Box::new(prep_layer(
                        l, fonts, images, prepared,
                    )?)));
                }
            }
        }
        if !run.is_empty() {
            prep.items.push(PrepItem::Content(run));
        }
    }
    Ok(prep)
}

/// Builds one engine layer for `prep` under `parent`, recursing into
/// children in item order.
#[expect(
    clippy::cast_possible_truncation,
    reason = "layer opacity is f32 at the engine boundary"
)]
fn build_layer(
    surface: &Surface<Vello>,
    tx: &mut Transaction<'_, Vello>,
    parent: &VelloLayer,
    prep: PrepLayer,
    content_layers: &mut Vec<ContentLayer>,
) {
    let layer = surface.layer();
    {
        let edit = &mut tx[&layer];
        edit.transform(prep.transform);
        edit.scroll_offset(prep.scroll_offset);
        edit.opacity(prep.opacity as f32);
        edit.blend(prep.blend);
        if let Some(clip) = &prep.clip {
            clip_shape(edit, clip);
        }
    }
    tx[parent].push(&layer);
    for item in prep.items {
        match item {
            PrepItem::Content(ops) => {
                let child = surface.layer();
                tx[&layer].push(&child);
                content_layers.push(ContentLayer {
                    layer: child,
                    ops,
                    motion: None,
                });
            }
            PrepItem::Layer(p) => build_layer(surface, tx, &layer, *p, content_layers),
        }
    }
    content_layers.push(ContentLayer {
        layer,
        ops: prep.own,
        motion: prep.motion,
    });
}

impl CherenkovVello {
    /// Adapter key.
    pub const NAME: &'static str = "cherenkov-vello";

    /// Creates the adapter, initializing the vello engine.
    ///
    /// # Errors
    /// [`BenchError::Gpu`] when no adapter exists or device creation fails.
    pub fn new() -> Result<Self, BenchError> {
        let engine = VelloEngine::<Vello>::new(VelloConfig {
            timestamps: true,
            ..VelloConfig::default()
        })
        .map_err(|e| BenchError::Gpu(format!("cherenkov-vello engine: {e}")))?;
        Ok(Self {
            info: EngineInfo {
                name: Self::NAME,
                engine_crate: "cherenkov-vello",
                crate_version: env!("DEP_CHERENKOV_VELLO_VERSION"),
                source_rev: option_env!("DEP_CHERENKOV_VELLO_SOURCE_REV").map(String::from),
                output_format: "wgpu Rgba8Unorm (premultiplied sRGB)".to_string(),
                precision: "vello tile pipeline; rgba8unorm encoded-space compositing",
                route: "vello classic via cherenkov-vello",
                color_note: "premultiplied sRGB-encoded RGBA8; blends in the encoded target",
                encode_scope: "records `cherenkov::Content` calls (fill/stroke/shadow/glyphs/\
                               image) against fonts, images and the layer tree prepared once",
            },
            engine,
            surface: None,
            fonts: HashMap::new(),
            images: HashMap::new(),
            content_layers: Vec::new(),
            bytes_uploaded: 0,
            has_motion: false,
            motion_committed: false,
            clock: Clock::new(),
            counters: Counters::default(),
        })
    }
}

impl Engine for CherenkovVello {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn supported(&self) -> BTreeSet<Feature> {
        vello_ad_features().into_iter().collect()
    }

    fn prepare(&mut self, input: &EncodeInput<'_>) -> Result<(), BenchError> {
        convert::check_features(Self::NAME, input.scene, &vello_ad_features(), missing_api)?;
        let prepared = Prepared::build(input.scene, input.blobs)?;
        let surface = self
            .engine
            .surface(Offscreen::new(
                (input.scene.width, input.scene.height),
                OffscreenFormat::LinearF16,
            ))
            .map_err(|e| BenchError::Gpu(format!("cherenkov-vello surface: {e}")))?;
        surface.clear_color(working(&input.scene.clear));
        register_resources(
            &mut self.fonts,
            &mut self.images,
            &self.engine,
            &prepared,
            &input.scene.root,
            input.blobs,
        )?;
        self.bytes_uploaded = prepared.texel_bytes();
        let prep = prep_layer(&input.scene.root, &self.fonts, &self.images, &prepared)?;
        self.content_layers.clear();
        let mut content_layers = Vec::new();
        surface.update(|tx| {
            let root = surface.root();
            build_layer(&surface, tx, root, prep, &mut content_layers);
        });
        self.has_motion = content_layers.iter().any(|c| c.motion.is_some());
        self.motion_committed = false;
        self.content_layers = content_layers;
        self.surface = Some(surface);
        Ok(())
    }

    fn encode(&mut self, input: &EncodeInput<'_>) -> Result<(), BenchError> {
        self.counters = Counters::default();
        convert::count_layer(&input.scene.root, &mut self.counters);
        self.counters.bytes_uploaded = Some(self.bytes_uploaded);
        let surface = self
            .surface
            .as_ref()
            .ok_or_else(|| BenchError::Engine("cherenkov-vello: encode before prepare".into()))?;
        let first_motion = self.has_motion && !self.motion_committed;
        if first_motion {
            for cl in &self.content_layers {
                if let Some(motion) = &cl.motion {
                    motion.apply(surface, &cl.layer);
                }
            }
            self.motion_committed = true;
        }
        // Motion scenes record their content once: later encodes only
        // advance the clock. Static scenes keep re-recording each frame
        // so their numbers stay comparable.
        if !self.has_motion || first_motion {
            let contents: Vec<(usize, cherenkov::Content)> = self
                .content_layers
                .iter()
                .enumerate()
                .map(|(i, cl)| {
                    let content = surface.record(|c| {
                        for op in &cl.ops {
                            record_op(c, op);
                        }
                    });
                    (i, content)
                })
                .collect();
            surface.update(|tx| {
                for (i, content) in contents {
                    tx[&self.content_layers[i].layer].content(content);
                }
            });
        }
        self.clock.advance();
        Ok(())
    }

    fn submit(&mut self, readback: bool) -> Result<Submit, BenchError> {
        let surface = self
            .surface
            .as_ref()
            .ok_or_else(|| BenchError::Engine("cherenkov-vello: submit before prepare".into()))?;
        self.engine
            .render(self.clock.time())
            .map_err(render_error)?;
        if readback && self.has_motion {
            // FLIP compares against the oracle's settled scene: render
            // until the animations come to rest (cap 2000 frames).
            let mut settled = false;
            for _ in 0..2000 {
                match self
                    .engine
                    .render(self.clock.time())
                    .map_err(render_error)?
                {
                    cherenkov::Next::Idle => {
                        settled = true;
                        break;
                    }
                    cherenkov::Next::At { .. } => self.clock.advance(),
                }
            }
            if !settled {
                return Err(BenchError::Engine(
                    "cherenkov: motion did not settle in 2000 frames".into(),
                ));
            }
        }
        let gpu_seconds = self.engine.stats().gpu_seconds;
        let image = if readback {
            let rb = surface.readback().map_err(render_error)?;
            Some(cherenkov_oracle::F32Image {
                width: rb.width,
                height: rb.height,
                pixels: rb.pixels,
            })
        } else {
            None
        };
        Ok(Submit {
            image,
            gpu_seconds,
            passes: Vec::new(),
        })
    }

    fn counters(&self) -> Counters {
        let mut counters = self.counters.clone();
        let stats = self.engine.stats();
        counters.dispatches = Some(stats.draws);
        counters.passes = Some(stats.passes);
        counters
    }

    fn device(&self) -> DeviceInfo {
        let info = self.engine.info();
        DeviceInfo {
            adapter: Some(info.name.clone()),
            backend: Some(info.backend.clone()),
            driver: Some(info.driver.clone()),
            driver_info: Some(info.driver_info.clone()),
            vendor: Some(info.vendor),
            device: Some(info.device),
            target_format: Some("Rgba8Unorm".to_string()),
            cpu: crate::cpu_model(),
            thermal_celsius: crate::thermal_celsius(),
        }
    }
}

/// Records one [`Op`] into a recorder — the per-frame engine calls.
fn record_op(c: &mut cherenkov::Recorder, op: &Op) {
    match op {
        Op::Fill {
            shape,
            paint,
            even_odd,
        } => fill(c, shape, paint, *even_odd),
        Op::Stroke {
            shape,
            stroke,
            paint,
        } => match shape {
            ShapeKind::Rect(s) => c.stroke(*s, stroke.clone(), paint.clone()),
            ShapeKind::RoundedRect(s) => c.stroke(*s, stroke.clone(), paint.clone()),
            ShapeKind::Continuous(s) => c.stroke(*s, stroke.clone(), paint.clone()),
            ShapeKind::Circle(s) => c.stroke(*s, stroke.clone(), paint.clone()),
            ShapeKind::Ellipse(s) => c.stroke(*s, stroke.clone(), paint.clone()),
            ShapeKind::Line(s) => c.stroke(*s, stroke.clone(), paint.clone()),
            ShapeKind::Path(s) => c.stroke(s.clone(), stroke.clone(), paint.clone()),
        },
        Op::Shadow { shape, shadow } => match shape {
            ShapeKind::Rect(s) => c.shadow(*s, *shadow),
            ShapeKind::RoundedRect(s) => c.shadow(*s, *shadow),
            ShapeKind::Continuous(s) => c.shadow(*s, *shadow),
            ShapeKind::Circle(s) => c.shadow(*s, *shadow),
            ShapeKind::Ellipse(s) => c.shadow(*s, *shadow),
            ShapeKind::Line(s) => c.shadow(*s, *shadow),
            ShapeKind::Path(s) => c.shadow(s.clone(), *shadow),
        },
        Op::Glyphs { run, paint } => c.glyphs(run, paint.clone()),
        Op::Image {
            image,
            dst,
            sampling,
        } => c.image(*image, *dst, *sampling),
    }
}

/// Records a fill, wrapping the shape in [`cherenkov::EvenOdd`] when the
/// draw declares the even-odd rule.
fn fill(c: &mut cherenkov::Recorder, shape: &ShapeKind, paint: &cherenkov::Paint, even_odd: bool) {
    macro_rules! fill_shape {
        ($s:expr, clone) => {
            if even_odd {
                c.fill(cherenkov::EvenOdd($s.clone()), paint.clone());
            } else {
                c.fill($s.clone(), paint.clone());
            }
        };
        ($s:expr) => {
            if even_odd {
                c.fill(cherenkov::EvenOdd(*$s), paint.clone());
            } else {
                c.fill(*$s, paint.clone());
            }
        };
    }
    match shape {
        ShapeKind::Rect(s) => fill_shape!(s),
        ShapeKind::RoundedRect(s) => fill_shape!(s),
        ShapeKind::Continuous(s) => fill_shape!(s),
        ShapeKind::Circle(s) => fill_shape!(s),
        ShapeKind::Ellipse(s) => fill_shape!(s),
        ShapeKind::Line(s) => fill_shape!(s),
        ShapeKind::Path(s) => fill_shape!(s, clone),
    }
}
