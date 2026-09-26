// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cherenkov-cpu` adapter: the CPU banded exact-area rasterizer.
//!
//! Route: the front-end records a display list per content layer, lowered on
//! the render thread into flattened device-space edge lists drawn by a
//! rayon-banded signed-area coverage rasterizer into an f32 framebuffer
//! (premultiplied linear Display P3 — the suite's working space end to end).
//! Readback is f16-rounded by default; `CHERENKOV_CPU_READBACK=f32` keeps the
//! raw f32s. There is no GPU timestamp source; `gpu_seconds` is always `null`.

use std::collections::{BTreeSet, HashMap};

use cherenkov::Draw as _;
use cherenkov_cpu::{
    Engine as CpuEngine, FrameTime, Layer as CpuLayer, Offscreen, OffscreenFormat, Raster,
    RasterConfig, RenderError, ResourceError, Surface, Transaction, Unsupported,
};
use cherenkov_oracle::color::to_working;
use cherenkov_scene::{
    BlendMode, ColorSpace, Draw as SceneDraw, Extend, Feature, GlyphRun as SceneGlyphRun, Item,
    Layer as SceneLayer, Paint as ScenePaint, ResourceHash, Shape,
};
use kurbo::{Affine, BezPath, Circle, Ellipse, Line, Rect, RoundedRect, Vec2};

use crate::convert::{self, Blobs};
use crate::{BenchError, Counters, DeviceInfo, EncodeInput, Engine, EngineInfo, Submit};

/// A scene shape in a form the front-end accepts.
enum ShapeKind {
    Rect(Rect),
    RoundedRect(RoundedRect),
    Continuous(cherenkov::ContinuousRect),
    Circle(Circle),
    Ellipse(Ellipse),
    Line(Line),
    /// A general path plus the fill rule it records under.
    Path(BezPath, cherenkov::FillRule),
}

/// One recording step of a content layer, resolved in `prepare`.
enum Op {
    /// `Draw::Fill`.
    Fill {
        /// The shape.
        shape: ShapeKind,
        /// The paint.
        paint: cherenkov::Paint,
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
        /// Sampling.
        sampling: cherenkov::Sampling,
    },
}

/// A prepared child item: a draw-item run wrapped in its own layer, or a
/// real child layer.
enum PrepItem {
    /// A maximal run of draw items, drawn as one layer's content.
    Content(Vec<Op>),
    /// A child scene layer.
    Layer(PrepLayer),
}

/// A scene layer lowered in `prepare`.
struct PrepLayer {
    /// Local transform.
    transform: Affine,
    /// Clip in the layer's own space.
    clip: Option<ShapeKind>,
    /// Group opacity.
    opacity: f64,
    /// Blend onto the parent.
    blend: cherenkov::BlendMode,
    /// The layer's own content — only when every draw precedes every child.
    own: Vec<Op>,
    /// Ordered children.
    items: Vec<PrepItem>,
}

/// An engine layer plus the ops it records each frame.
struct ContentLayer {
    /// The layer handle.
    layer: CpuLayer,
    /// Its recorded ops.
    ops: Vec<Op>,
}

/// `cherenkov-cpu` adapter.
pub struct Cherenkov {
    info: EngineInfo,
    engine: CpuEngine<Raster>,
    surface: Option<Surface>,
    /// Registered fonts per `(blob hash, face index)`.
    fonts: HashMap<(ResourceHash, u32), cherenkov::FontId>,
    /// Registered images per blob hash, plus the handles keeping them alive.
    images: HashMap<ResourceHash, cherenkov::ImageId>,
    /// Image handles owning the render-thread pixel stores.
    image_handles: Vec<cherenkov_cpu::Image>,
    /// Layers holding recorded content, in draw order.
    content_layers: Vec<ContentLayer>,
    counters: Counters,
}

/// The features this slice executes faithfully.
fn cherenkov_features() -> Vec<Feature> {
    vec![
        Feature::Fill,
        Feature::Stroke,
        Feature::Path,
        Feature::EvenOdd,
        Feature::StrokeDash,
        Feature::ContinuousCorners,
        Feature::LinearGradient,
        Feature::RadialGradient,
        Feature::Clip,
        Feature::Opacity,
        Feature::Shadow,
        Feature::Glyphs,
        Feature::FontVariations,
        Feature::HdrColor,
        Feature::WideGamut,
        Feature::SweepGradient,
        Feature::ExtendNone,
        Feature::Image,
        Feature::ImagePaint,
        // `sRGB` maps to `SrgbEncoded`; `linear-p3` and `linear-srgb` are
        // both linear interpolation, which is the working space already.
        Feature::InterpolationSpace(ColorSpace::Srgb),
        Feature::InterpolationSpace(ColorSpace::LinearP3),
        Feature::InterpolationSpace(ColorSpace::LinearSrgb),
    ]
    .into_iter()
    .chain(BlendMode::ALL.into_iter().map(Feature::Blend))
    .collect()
}

/// The upstream API this slice lacks for a declared scene feature.
const fn missing_api(f: &Feature) -> Option<&'static str> {
    match f {
        Feature::InterpolationSpace(_) => Some("only srgb / linear interpolation in this slice"),
        _ => None,
    }
}

/// The scene [`Feature`] a render-time [`Unsupported`] maps back to.
const fn unsupported_feature(u: Unsupported) -> Feature {
    match u {
        Unsupported::Shader => Feature::Image,
        Unsupported::Filter => Feature::Opacity,
        Unsupported::ColorFont => Feature::Glyphs,
        _ => Feature::Fill,
    }
}

/// Maps a render-time error into a `BenchError`, keeping `Unsupported`
/// scenes reported rather than fatal.
fn render_error(e: RenderError) -> BenchError {
    match e {
        RenderError::Unsupported(u) => BenchError::Unsupported {
            engine: Cherenkov::NAME,
            feature: unsupported_feature(u),
            api: Some(Box::leak(format!("{u}").into_boxed_str())),
        },
        e => BenchError::Gpu(format!("cherenkov render: {e}")),
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

/// The front-end blend mode matching a scene mode one-for-one by name.
const fn cpu_blend(m: BlendMode) -> cherenkov::BlendMode {
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
        BlendMode::Clear => cherenkov::BlendMode::Clear,
        BlendMode::Src => cherenkov::BlendMode::Src,
        BlendMode::Dst => cherenkov::BlendMode::Dst,
        BlendMode::DestOver => cherenkov::BlendMode::DestOver,
        BlendMode::SrcIn => cherenkov::BlendMode::SrcIn,
        BlendMode::DestIn => cherenkov::BlendMode::DestIn,
        BlendMode::SrcOut => cherenkov::BlendMode::SrcOut,
        BlendMode::DestOut => cherenkov::BlendMode::DestOut,
        BlendMode::SrcAtop => cherenkov::BlendMode::SrcAtop,
        BlendMode::DestAtop => cherenkov::BlendMode::DestAtop,
        BlendMode::Xor => cherenkov::BlendMode::Xor,
        BlendMode::PlusLighter => cherenkov::BlendMode::PlusLighter,
    }
}

const fn extend(e: Extend) -> cherenkov::Extend {
    match e {
        Extend::Pad => cherenkov::Extend::Pad,
        Extend::Repeat => cherenkov::Extend::Repeat,
        Extend::Reflect => cherenkov::Extend::Reflect,
        Extend::None => cherenkov::Extend::None,
    }
}

const fn interpolation(space: ColorSpace) -> Result<cherenkov::Interpolation, BenchError> {
    match space {
        ColorSpace::Srgb => Ok(cherenkov::Interpolation::SrgbEncoded),
        ColorSpace::LinearP3 | ColorSpace::LinearSrgb => Ok(cherenkov::Interpolation::Working),
        space => Err(BenchError::Unsupported {
            engine: Cherenkov::NAME,
            feature: Feature::InterpolationSpace(space),
            api: missing_api(&Feature::InterpolationSpace(space)),
        }),
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

/// A scene paint → the front-end paint.
fn front_paint(
    paint: &ScenePaint,
    images: &HashMap<ResourceHash, cherenkov::ImageId>,
) -> Result<cherenkov::Paint, BenchError> {
    Ok(match paint {
        ScenePaint::Solid(c) => cherenkov::Paint::Solid(working(c)),
        ScenePaint::Linear(g) => cherenkov::Paint::Linear(cherenkov::LinearGradient {
            start: g.start,
            end: g.end,
            stops: stops(&g.stops),
            extend: extend(g.extend),
            interpolation: interpolation(g.interpolation)?,
        }),
        ScenePaint::Radial(g) => cherenkov::Paint::Radial(cherenkov::RadialGradient {
            start_center: g.center0,
            start_radius: g.r0,
            end_center: g.center1,
            end_radius: g.r1,
            stops: stops(&g.stops),
            extend: extend(g.extend),
            interpolation: interpolation(g.interpolation)?,
        }),
        ScenePaint::Sweep(g) => cherenkov::Paint::Sweep(cherenkov::SweepGradient {
            center: g.center,
            start_angle: g.start_angle,
            end_angle: g.end_angle,
            stops: stops(&g.stops),
            extend: extend(g.extend),
            interpolation: interpolation(g.interpolation)?,
        }),
        ScenePaint::Image(p) => cherenkov::Paint::Image(cherenkov::ImagePattern {
            image: *images
                .get(&p.image)
                .ok_or(cherenkov_scene::SceneError::MissingResource(p.image))?,
            transform: p.transform,
            extend_x: extend(p.extend_x),
            extend_y: extend(p.extend_y),
            sampling: match p.sampling {
                cherenkov_scene::Sampling::Nearest => cherenkov::Sampling::Nearest,
                cherenkov_scene::Sampling::Bilinear => cherenkov::Sampling::Linear,
            },
        }),
    })
}

/// A scene shape → [`ShapeKind`]. `rule` is the fill rule in force; it
/// only matters for paths.
fn shape_kind(shape: &Shape, rule: cherenkov::FillRule) -> ShapeKind {
    match shape {
        Shape::Rect(r) => ShapeKind::Rect(*r),
        Shape::RoundedRect(r) => ShapeKind::RoundedRect(*r),
        Shape::Continuous(c) => ShapeKind::Continuous(
            cherenkov::ContinuousRect::new(c.rect, c.corner_radius).with_smoothing(c.smoothing),
        ),
        Shape::Circle(c) => ShapeKind::Circle(*c),
        Shape::Ellipse(e) => ShapeKind::Ellipse(*e),
        Shape::Line(l) => ShapeKind::Line(*l),
        Shape::Path { path } => ShapeKind::Path(path.clone(), rule),
    }
}

/// The scene fill rule as the front-end's.
const fn front_rule(rule: cherenkov_scene::FillRule) -> cherenkov::FillRule {
    match rule {
        cherenkov_scene::FillRule::NonZero => cherenkov::FillRule::NonZero,
        cherenkov_scene::FillRule::EvenOdd => cherenkov::FillRule::EvenOdd,
    }
}

/// Applies a clip shape to a layer edit.
fn clip_shape(edit: &mut cherenkov_cpu::LayerEdit, shape: &ShapeKind) {
    match shape {
        ShapeKind::Rect(r) => drop(edit.clip(*r)),
        ShapeKind::RoundedRect(r) => drop(edit.clip(*r)),
        ShapeKind::Continuous(c) => drop(edit.clip(*c)),
        ShapeKind::Circle(c) => drop(edit.clip(*c)),
        ShapeKind::Ellipse(e) => drop(edit.clip(*e)),
        ShapeKind::Line(l) => drop(edit.clip(*l)),
        ShapeKind::Path(p, _) => drop(edit.clip(p.clone())),
    }
}

/// A scene draw → an [`Op`]; unreachable features still report unsupported
/// rather than silently dropping.
fn op(
    draw: &SceneDraw,
    fonts: &HashMap<(ResourceHash, u32), cherenkov::FontId>,
    images: &HashMap<ResourceHash, cherenkov::ImageId>,
    blobs: &Blobs,
) -> Result<Op, BenchError> {
    Ok(match draw {
        SceneDraw::Fill { shape, rule, paint } => Op::Fill {
            shape: shape_kind(shape, front_rule(*rule)),
            paint: front_paint(paint, images)?,
        },
        SceneDraw::Stroke {
            shape,
            stroke,
            paint,
        } => Op::Stroke {
            shape: shape_kind(shape, cherenkov::FillRule::NonZero),
            stroke: convert::stroke(stroke),
            paint: front_paint(paint, images)?,
        },
        SceneDraw::Shadow {
            shape,
            blur_sigma,
            offset,
            color,
        } => Op::Shadow {
            shape: shape_kind(shape, cherenkov::FillRule::NonZero),
            shadow: cherenkov::Shadow::new(*blur_sigma, working(color))
                .offset(Vec2::new(offset[0], offset[1])),
        },
        SceneDraw::Glyphs(run) => Op::Glyphs {
            run: glyph_run(run, fonts, blobs)?,
            paint: front_paint(&run.paint, images)?,
        },
        SceneDraw::Image {
            image,
            dst,
            sampling,
        } => Op::Image {
            image: *images
                .get(image)
                .ok_or(cherenkov_scene::SceneError::MissingResource(*image))?,
            dst: *dst,
            sampling: match sampling {
                cherenkov_scene::Sampling::Nearest => cherenkov::Sampling::Nearest,
                cherenkov_scene::Sampling::Bilinear => cherenkov::Sampling::Linear,
            },
        },
    })
}

/// A scene glyph run → a front-end run with the registered font and the
/// resolved `F2Dot14` coordinates.
fn glyph_run(
    run: &SceneGlyphRun,
    fonts: &HashMap<(ResourceHash, u32), cherenkov::FontId>,
    blobs: &Blobs,
) -> Result<cherenkov::GlyphRun, BenchError> {
    let font = *fonts
        .get(&(run.font, run.font_index))
        .ok_or(cherenkov_scene::SceneError::MissingResource(run.font))?;
    let coords = blobs
        .get(&run.font)
        .map_or_else(Vec::new, |b| convert::coord_bits(b, &run.normalized_coords));
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

/// Registers every font a glyph run references, once per `(hash, index)`.
/// PNGs carry sRGB data; [`cherenkov_oracle::image::decode_png_rgba8`] is
/// the shared decoder the oracle and every adapter use, and the engine
/// converts to the working space at registration.
fn register_image(
    images: &mut HashMap<ResourceHash, cherenkov::ImageId>,
    handles: &mut Vec<cherenkov_cpu::Image>,
    engine: &CpuEngine<Raster>,
    hash: &ResourceHash,
    blobs: &Blobs,
) -> Result<(), BenchError> {
    if images.contains_key(hash) {
        return Ok(());
    }
    let blob = blobs
        .get(hash)
        .ok_or(cherenkov_scene::SceneError::MissingResource(*hash))?;
    let (width, height, rgba) = cherenkov_oracle::image::decode_png_rgba8(blob)
        .map_err(|e| BenchError::Engine(format!("cherenkov image decode: {e}")))?;
    let image = engine
        .image(cherenkov_cpu::ImageSource {
            width,
            height,
            pixels: rgba,
            color_space: cherenkov_cpu::ImageColorSpace::Srgb,
        })
        .map_err(|e| BenchError::Engine(format!("cherenkov image: {e}")))?;
    images.insert(*hash, image.id());
    handles.push(image);
    Ok(())
}

/// Registers every image referenced by draws or image paints in `layer`.
fn register_images(
    images: &mut HashMap<ResourceHash, cherenkov::ImageId>,
    handles: &mut Vec<cherenkov_cpu::Image>,
    engine: &CpuEngine<Raster>,
    layer: &SceneLayer,
    blobs: &Blobs,
) -> Result<(), BenchError> {
    for item in &layer.items {
        match item {
            Item::Layer(l) => register_images(images, handles, engine, l, blobs)?,
            Item::Draw(SceneDraw::Image { image, .. }) => {
                register_image(images, handles, engine, image, blobs)?;
            }
            Item::Draw(d) => {
                let paint = match d {
                    SceneDraw::Fill { paint, .. } | SceneDraw::Stroke { paint, .. } => Some(paint),
                    SceneDraw::Glyphs(run) => Some(&run.paint),
                    _ => None,
                };
                if let Some(ScenePaint::Image(p)) = paint {
                    register_image(images, handles, engine, &p.image, blobs)?;
                }
            }
        }
    }
    Ok(())
}

fn register_fonts(
    fonts: &mut HashMap<(ResourceHash, u32), cherenkov::FontId>,
    engine: &CpuEngine<Raster>,
    layer: &SceneLayer,
    blobs: &Blobs,
) -> Result<(), BenchError> {
    for item in &layer.items {
        match item {
            Item::Layer(l) => register_fonts(fonts, engine, l, blobs)?,
            Item::Draw(SceneDraw::Glyphs(run)) => {
                if fonts.contains_key(&(run.font, run.font_index)) {
                    continue;
                }
                let blob = blobs
                    .get(&run.font)
                    .ok_or(cherenkov_scene::SceneError::MissingResource(run.font))?;
                let font = engine
                    .font(cherenkov_cpu::FontSource::bytes(blob.clone()).with_index(run.font_index))
                    .map_err(|e| match e {
                        ResourceError::Unsupported(Unsupported::ColorFont) => {
                            BenchError::Unsupported {
                                engine: Cherenkov::NAME,
                                feature: Feature::Glyphs,
                                api: Some(
                                    "colour fonts (COLR/CBDT/sbix) are outside the first slice",
                                ),
                            }
                        }
                        e => BenchError::Engine(format!("cherenkov font: {e}")),
                    })?;
                fonts.insert((run.font, run.font_index), font.id());
            }
            Item::Draw(_) => {}
        }
    }
    Ok(())
}

/// Lowers a scene layer: one engine layer per scene layer, plus one per
/// draw run that must interleave with child layers.
fn prep_layer(
    layer: &SceneLayer,
    fonts: &HashMap<(ResourceHash, u32), cherenkov::FontId>,
    images: &HashMap<ResourceHash, cherenkov::ImageId>,
    blobs: &Blobs,
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
        clip: layer
            .clip
            .as_ref()
            .map(|s| shape_kind(s, cherenkov::FillRule::NonZero)),
        opacity: layer.opacity,
        blend: cpu_blend(layer.blend),
        own: Vec::new(),
        items: Vec::new(),
    };
    if own {
        for item in &layer.items {
            match item {
                Item::Draw(d) => prep.own.push(op(d, fonts, images, blobs)?),
                Item::Layer(l) => prep
                    .items
                    .push(PrepItem::Layer(prep_layer(l, fonts, images, blobs)?)),
            }
        }
    } else {
        let mut run: Vec<Op> = Vec::new();
        for item in &layer.items {
            match item {
                Item::Draw(d) => run.push(op(d, fonts, images, blobs)?),
                Item::Layer(l) => {
                    if !run.is_empty() {
                        prep.items.push(PrepItem::Content(std::mem::take(&mut run)));
                    }
                    prep.items
                        .push(PrepItem::Layer(prep_layer(l, fonts, images, blobs)?));
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
    surface: &Surface,
    tx: &mut Transaction<'_>,
    parent: &CpuLayer,
    prep: PrepLayer,
    content_layers: &mut Vec<ContentLayer>,
) {
    let layer = surface.layer();
    {
        let edit = &mut tx[&layer];
        edit.transform(prep.transform);
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
                content_layers.push(ContentLayer { layer: child, ops });
            }
            PrepItem::Layer(p) => build_layer(surface, tx, &layer, p, content_layers),
        }
    }
    content_layers.push(ContentLayer {
        layer,
        ops: prep.own,
    });
}

impl Cherenkov {
    /// Adapter key.
    pub const NAME: &'static str = "cherenkov-cpu";

    /// The readback format `CHERENKOV_CPU_READBACK` selects (`f16` default,
    /// `f32` for the unrounded framebuffer).
    fn readback_format() -> OffscreenFormat {
        match std::env::var("CHERENKOV_CPU_READBACK").as_deref() {
            Ok("f32") => OffscreenFormat::LinearF32,
            _ => OffscreenFormat::LinearF16,
        }
    }

    /// Creates the adapter, initializing the raster engine.
    ///
    /// # Errors
    /// [`BenchError::Gpu`] when the render thread or worker pool fails.
    pub fn new() -> Result<Self, BenchError> {
        let engine = CpuEngine::<Raster>::new(RasterConfig::default())
            .map_err(|e| BenchError::Gpu(format!("cherenkov engine: {e}")))?;
        Ok(Self {
            info: EngineInfo {
                name: Self::NAME,
                engine_crate: "cherenkov-cpu",
                crate_version: env!("DEP_CHERENKOV_CPU_VERSION"),
                source_rev: option_env!("DEP_CHERENKOV_CPU_SOURCE_REV").map(String::from),
                output_format: match Self::readback_format() {
                    OffscreenFormat::LinearF16 => {
                        "f32 framebuffer read back through f16 (premultiplied linear P3)"
                    }
                    OffscreenFormat::LinearF32 => {
                        "f32 framebuffer, unrounded (premultiplied linear P3)"
                    }
                }
                .to_string(),
                precision: "f32 exact-area coverage bands; f32 working-space framebuffer",
                route: "cpu-raster (rayon bands)",
                color_note: "premultiplied linear Display P3 end to end; HDR channels unclamped",
                encode_scope: "records `cherenkov::Content` calls (fill/stroke/shadow/glyphs) \
                               against fonts and the layer tree prepared once",
            },
            engine,
            surface: None,
            fonts: HashMap::new(),
            images: HashMap::new(),
            image_handles: Vec::new(),
            content_layers: Vec::new(),
            counters: Counters::default(),
        })
    }
}

impl Engine for Cherenkov {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn supported(&self) -> BTreeSet<Feature> {
        cherenkov_features().into_iter().collect()
    }

    fn prepare(&mut self, input: &EncodeInput<'_>) -> Result<(), BenchError> {
        convert::check_features(Self::NAME, input.scene, &cherenkov_features(), missing_api)?;
        let surface = self
            .engine
            .surface(Offscreen::new(
                (input.scene.width, input.scene.height),
                Self::readback_format(),
            ))
            .map_err(|e| BenchError::Gpu(format!("cherenkov surface: {e}")))?;
        surface.clear_color(working(&input.scene.clear));
        register_fonts(
            &mut self.fonts,
            &self.engine,
            &input.scene.root,
            input.blobs,
        )?;
        register_images(
            &mut self.images,
            &mut self.image_handles,
            &self.engine,
            &input.scene.root,
            input.blobs,
        )?;
        let prep = prep_layer(&input.scene.root, &self.fonts, &self.images, input.blobs)?;
        self.content_layers.clear();
        let mut content_layers = Vec::new();
        surface
            .update(|tx| {
                let root = surface.root();
                build_layer(&surface, tx, root, prep, &mut content_layers);
            })
            .map_err(|e| BenchError::Engine(format!("cherenkov update: {e}")))?;
        self.content_layers = content_layers;
        self.surface = Some(surface);
        Ok(())
    }

    fn encode(&mut self, input: &EncodeInput<'_>) -> Result<(), BenchError> {
        self.counters = Counters::default();
        convert::count_layer(&input.scene.root, &mut self.counters);
        let surface = self
            .surface
            .as_ref()
            .ok_or_else(|| BenchError::Engine("cherenkov: encode before prepare".into()))?;
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
        surface
            .update(|tx| {
                for (i, content) in contents {
                    tx[&self.content_layers[i].layer].content(content);
                }
            })
            .map_err(|e| BenchError::Engine(format!("cherenkov update: {e}")))?;
        Ok(())
    }

    fn submit(&mut self, readback: bool) -> Result<Submit, BenchError> {
        let surface = self
            .surface
            .as_ref()
            .ok_or_else(|| BenchError::Engine("cherenkov: submit before prepare".into()))?;
        self.engine.render(FrameTime::now()).map_err(render_error)?;
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
            phases: Vec::new(),
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
            adapter: Some(format!("cherenkov-cpu ({} threads)", info.threads)),
            backend: Some(info.simd.to_string()),
            driver: None,
            driver_info: None,
            vendor: None,
            device: None,
            target_format: Some("f32 RGBA framebuffer".to_string()),
            cpu: info.cpu.clone().or_else(crate::cpu_model),
            thermal_celsius: crate::thermal_celsius(),
        }
    }
}

/// Records one [`Op`] into a recorder — the per-frame engine calls.
fn record_op(c: &mut cherenkov::Recorder, op: &Op) {
    match op {
        Op::Fill { shape, paint } => match shape {
            ShapeKind::Rect(s) => c.fill(*s, paint.clone()),
            ShapeKind::RoundedRect(s) => c.fill(*s, paint.clone()),
            ShapeKind::Continuous(s) => c.fill(*s, paint.clone()),
            ShapeKind::Circle(s) => c.fill(*s, paint.clone()),
            ShapeKind::Ellipse(s) => c.fill(*s, paint.clone()),
            ShapeKind::Line(s) => c.fill(*s, paint.clone()),
            ShapeKind::Path(p, rule) => match rule {
                cherenkov::FillRule::EvenOdd => {
                    c.fill(cherenkov::EvenOdd(p.clone()), paint.clone());
                }
                cherenkov::FillRule::NonZero => c.fill(p.clone(), paint.clone()),
            },
        },
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
            ShapeKind::Path(p, _) => c.stroke(p.clone(), stroke.clone(), paint.clone()),
        },
        Op::Shadow { shape, shadow } => match shape {
            ShapeKind::Rect(s) => c.shadow(*s, *shadow),
            ShapeKind::RoundedRect(s) => c.shadow(*s, *shadow),
            ShapeKind::Continuous(s) => c.shadow(*s, *shadow),
            ShapeKind::Circle(s) => c.shadow(*s, *shadow),
            ShapeKind::Ellipse(s) => c.shadow(*s, *shadow),
            ShapeKind::Line(s) => c.shadow(*s, *shadow),
            ShapeKind::Path(p, _) => c.shadow(p.clone(), *shadow),
        },
        Op::Glyphs { run, paint } => c.glyphs(run, paint.clone()),
        Op::Image {
            image,
            dst,
            sampling,
        } => c.image(*image, *dst, *sampling),
    }
}
