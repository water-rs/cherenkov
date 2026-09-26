// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cherenkov` adapter: the `cherenkov-gpu` backend slice on `wgpu`.
//!
//! Route: the front-end records a display list per content layer, lowered on
//! the render thread into analytic f32 quads drawn by a single WGSL pipeline
//! into a `Rgba16Float` texture (premultiplied linear Display P3 — the
//! suite's working space end to end, so readback needs no conversion). GPU
//! time is a real `wgpu` timestamp pair drained inside
//! [`cherenkov_gpu::Engine::render`] with `GpuConfig::timestamps` set.
//!
//! `CHERENKOV_SCRATCH_FORMAT=rgba8` selects `Rgba8Unorm` isolation targets
//! (default `Rgba16Float`) to compare intermediate precision/bandwidth.

use std::collections::{BTreeSet, HashMap};

use cherenkov::{
    Draw as _, Engine as GpuEngine, ImageData, Layer as GpuLayer, LayerEdit, Offscreen,
    OffscreenFormat, RenderError, ResourceError, Rgba8, Surface, Transaction,
};
use cherenkov_gpu::{Gpu, GpuConfig, ScratchFormat};
use cherenkov_oracle::color::to_working;
use cherenkov_scene::{
    BlendMode, ColorSpace, Draw as SceneDraw, Extend, Feature, GlyphRun as SceneGlyphRun, Item,
    Layer as SceneLayer, Paint as ScenePaint, ResourceHash, Shape,
};
use kurbo::{Affine, BezPath, Circle, Ellipse, Line, Rect, RoundedRect, Vec2};

use crate::convert::{self, Blobs};
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
    /// A general path; the fill rule rides on the op, not the shape.
    Path(BezPath),
}

/// One recording step of a content layer, resolved in `prepare`.
enum Op {
    /// `Draw::Fill`.
    Fill {
        /// The shape.
        shape: ShapeKind,
        /// The fill rule.
        rule: cherenkov_scene::FillRule,
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
    /// Blend onto the parent.
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
    layer: GpuLayer,
    /// Its recorded ops.
    ops: Vec<Op>,
    /// The layer's one-time motion, committed on the first encode.
    motion: Option<LayerMotion>,
}

/// `cherenkov-gpu` adapter.
pub struct Cherenkov {
    info: EngineInfo,
    engine: GpuEngine<Gpu>,
    surface: Option<Surface<Gpu>>,
    /// Registered fonts per `(blob hash, face index)`.
    fonts: HashMap<(ResourceHash, u32), cherenkov::Font>,
    /// Registered images per blob hash (kept alive for the engine).
    images: HashMap<ResourceHash, cherenkov::ImageId>,
    /// The `Image` handles keeping `images` registered.
    image_handles: Vec<cherenkov::Image<cherenkov::Rgba8>>,
    /// Layers holding recorded content, in draw order.
    content_layers: Vec<ContentLayer>,
    /// Whether any layer carries a `motion`.
    has_motion: bool,
    /// Whether the motion commits have been sent (first encode).
    motion_committed: bool,
    /// The fixed frame clock `submit` renders at.
    clock: Clock,
    counters: Counters,
}

/// The features this slice executes faithfully.
fn cherenkov_features() -> Vec<Feature> {
    vec![
        Feature::Fill,
        Feature::Stroke,
        Feature::ContinuousCorners,
        Feature::LinearGradient,
        Feature::RadialGradient,
        Feature::Clip,
        Feature::Opacity,
        Feature::Shadow,
        Feature::Glyphs,
        Feature::FontVariations,
        Feature::Scroll,
        Feature::Animation,
        Feature::HdrColor,
        Feature::WideGamut,
        Feature::Path,
        Feature::EvenOdd,
        Feature::StrokeDash,
        Feature::SweepGradient,
        Feature::Image,
        Feature::ImagePaint,
        Feature::ExtendNone,
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
        Feature::InterpolationSpace(_) => {
            Some("only srgb / linear interpolation in the first slice")
        }
        _ => None,
    }
}

/// The front-end blend mode matching a scene mode one-for-one by name.
const fn gpu_blend(m: BlendMode) -> cherenkov::BlendMode {
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

/// The scene [`Feature`] a render-time unsupported name maps back to.
///
/// `shader-paint`, `mesh-gradient`, `filter` and `blend-space` have no
/// scene feature of their own; they report the nearest declared one
/// (`Fill`) while the `api` string names the real construct. `blend-mode`
/// is unreachable — every blend mode is supported.
fn unsupported_feature(u: &str) -> Feature {
    match u {
        "path" | "path-clip-too-large" => Feature::Path,
        "sweep-gradient" => Feature::SweepGradient,
        "image" => Feature::Image,
        "stroke-dash" => Feature::StrokeDash,
        "stroke-join" => Feature::Stroke,
        "glyph-stroke" | "glyph-transform" | "color-font" => Feature::Glyphs,
        "shadow" => Feature::Shadow,
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
            api: Some(u),
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

/// A scene shape → [`ShapeKind`].
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

/// Applies a clip shape to a layer edit.
fn clip_shape(edit: &mut LayerEdit<Gpu>, shape: &ShapeKind) {
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

/// A scene draw → an [`Op`]; unreachable features still report unsupported
/// rather than silently dropping.
fn op(
    draw: &SceneDraw,
    fonts: &HashMap<(ResourceHash, u32), cherenkov::Font>,
    images: &HashMap<ResourceHash, cherenkov::ImageId>,
    blobs: &Blobs,
) -> Result<Op, BenchError> {
    Ok(match draw {
        SceneDraw::Fill { shape, rule, paint } => Op::Fill {
            shape: shape_kind(shape),
            rule: *rule,
            paint: front_paint(paint, images)?,
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
    fonts: &HashMap<(ResourceHash, u32), cherenkov::Font>,
    blobs: &Blobs,
) -> Result<cherenkov::GlyphRun, BenchError> {
    let font = fonts
        .get(&(run.font, run.font_index))
        .map(cherenkov::Font::id)
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
fn register_fonts(
    fonts: &mut HashMap<(ResourceHash, u32), cherenkov::Font>,
    engine: &GpuEngine<Gpu>,
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
                    .font(cherenkov::FontSource::bytes(blob.clone()).with_index(run.font_index))
                    .map_err(|e| match e {
                        ResourceError::Unsupported("color-font") => BenchError::Unsupported {
                            engine: Cherenkov::NAME,
                            feature: Feature::Glyphs,
                            api: Some("colour fonts (COLR/CBDT/sbix) are outside the first slice"),
                        },
                        e => BenchError::Engine(format!("cherenkov font: {e}")),
                    })?;
                fonts.insert((run.font, run.font_index), font);
            }
            Item::Draw(_) => {}
        }
    }
    Ok(())
}

/// Registers one scene image resource, once per hash.
///
/// PNGs carry sRGB data; [`cherenkov_oracle::image::decode_png_rgba8`] is the
/// shared decoder the oracle and every adapter use, and the engine converts
/// to the working space at upload.
fn register_image(
    images: &mut HashMap<ResourceHash, cherenkov::ImageId>,
    handles: &mut Vec<cherenkov::Image<cherenkov::Rgba8>>,
    engine: &GpuEngine<Gpu>,
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
        .image(
            ImageData::<Rgba8>::new(width, height, rgba)
                .map_err(|e| BenchError::Engine(format!("cherenkov image: {e}")))?
                .color_space(cherenkov::ImageColorSpace::Srgb),
        )
        .map_err(|e| BenchError::Engine(format!("cherenkov image: {e}")))?;
    images.insert(*hash, image.id());
    handles.push(image);
    Ok(())
}

/// Registers every image referenced by draws or image paints in `layer`.
fn register_images(
    images: &mut HashMap<ResourceHash, cherenkov::ImageId>,
    handles: &mut Vec<cherenkov::Image<cherenkov::Rgba8>>,
    engine: &GpuEngine<Gpu>,
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

/// Lowers a scene layer: one engine layer per scene layer, plus one per
/// draw run that must interleave with child layers.
fn prep_layer(
    layer: &SceneLayer,
    fonts: &HashMap<(ResourceHash, u32), cherenkov::Font>,
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
        scroll_offset: layer.scroll_offset,
        clip: layer.clip.as_ref().map(shape_kind),
        opacity: layer.opacity,
        blend: gpu_blend(layer.blend),
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
                Item::Draw(d) => prep.own.push(op(d, fonts, images, blobs)?),
                Item::Layer(l) => prep.items.push(PrepItem::Layer(Box::new(prep_layer(
                    l, fonts, images, blobs,
                )?))),
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
                    prep.items.push(PrepItem::Layer(Box::new(prep_layer(
                        l, fonts, images, blobs,
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
    surface: &Surface<Gpu>,
    tx: &mut Transaction<'_, Gpu>,
    parent: &GpuLayer,
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

impl Cherenkov {
    /// Adapter key.
    pub const NAME: &'static str = "cherenkov";

    /// Creates the adapter, initializing the GPU engine.
    ///
    /// # Errors
    /// [`BenchError::Gpu`] when no adapter exists or device creation fails.
    pub fn new() -> Result<Self, BenchError> {
        let engine = GpuEngine::<Gpu>::new(GpuConfig {
            timestamps: true,
            scratch_format: match std::env::var("CHERENKOV_SCRATCH_FORMAT").as_deref() {
                Ok("rgba8") => ScratchFormat::Rgba8Unorm,
                _ => ScratchFormat::LinearF16,
            },
            ..GpuConfig::default()
        })
        .map_err(|e| BenchError::Gpu(format!("cherenkov engine: {e}")))?;
        Ok(Self {
            info: EngineInfo {
                name: Self::NAME,
                engine_crate: "cherenkov-gpu",
                crate_version: env!("DEP_CHERENKOV_GPU_VERSION"),
                source_rev: option_env!("DEP_CHERENKOV_GPU_SOURCE_REV").map(String::from),
                output_format: "wgpu Rgba16Float texture (premultiplied linear P3)".to_string(),
                precision: "analytic f32 quad instances; f16 working-space target",
                route: "wgpu (gpu slice)",
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
            has_motion: false,
            motion_committed: false,
            clock: Clock::new(),
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
                OffscreenFormat::LinearF16,
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
        let surface = self
            .surface
            .as_ref()
            .ok_or_else(|| BenchError::Engine("cherenkov: encode before prepare".into()))?;
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
            .ok_or_else(|| BenchError::Engine("cherenkov: submit before prepare".into()))?;
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
        let stats = self.engine.stats();
        let gpu_seconds = stats.gpu_seconds;
        let passes = stats
            .passes_timed
            .iter()
            .map(|p| crate::PassSample {
                name: p.name.clone(),
                width: p.width,
                height: p.height,
                format: p.format.to_string(),
                gpu_seconds: p.gpu_seconds,
            })
            .collect();
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
            passes,
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
            target_format: Some("Rgba16Float".to_string()),
            cpu: crate::cpu_model(),
            thermal_celsius: crate::thermal_celsius(),
        }
    }
}

/// Records one [`Op`] into a recorder — the per-frame engine calls.
fn record_op(c: &mut cherenkov::Recorder, op: &Op) {
    match op {
        Op::Fill { shape, rule, paint } => match shape {
            ShapeKind::Rect(s) => c.fill(*s, paint.clone()),
            ShapeKind::RoundedRect(s) => c.fill(*s, paint.clone()),
            ShapeKind::Continuous(s) => c.fill(*s, paint.clone()),
            ShapeKind::Circle(s) => c.fill(*s, paint.clone()),
            ShapeKind::Ellipse(s) => c.fill(*s, paint.clone()),
            ShapeKind::Line(s) => c.fill(*s, paint.clone()),
            ShapeKind::Path(p) => match rule {
                cherenkov_scene::FillRule::EvenOdd => {
                    c.fill(cherenkov::EvenOdd(p.clone()), paint.clone());
                }
                cherenkov_scene::FillRule::NonZero => c.fill(p.clone(), paint.clone()),
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
            ShapeKind::Path(p) => c.stroke(p.clone(), stroke.clone(), paint.clone()),
        },
        Op::Shadow { shape, shadow } => match shape {
            ShapeKind::Rect(s) => c.shadow(*s, *shadow),
            ShapeKind::RoundedRect(s) => c.shadow(*s, *shadow),
            ShapeKind::Continuous(s) => c.shadow(*s, *shadow),
            ShapeKind::Circle(s) => c.shadow(*s, *shadow),
            ShapeKind::Ellipse(s) => c.shadow(*s, *shadow),
            ShapeKind::Line(s) => c.shadow(*s, *shadow),
            ShapeKind::Path(p) => c.shadow(p.clone(), *shadow),
        },
        Op::Glyphs { run, paint } => c.glyphs(run, paint.clone()),
        Op::Image {
            image,
            dst,
            sampling,
        } => c.image(*image, *dst, *sampling),
    }
}
