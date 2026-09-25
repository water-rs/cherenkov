// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared scene → engine conversion helpers.
//!
//! Colour contract (uniform across adapters): a scene [`Color`] is first
//! converted to premultiplied linear Display P3 (the suite working space)
//! via `cherenkov_oracle::color::to_working`, then un-premultiplied,
//! converted to linear sRGB and clamped to the sRGB gamut, `srgb_encode`d
//! and quantized to `rgba8` straight alpha — the input every engine
//! accepts (`peniko::AlphaColor<Srgb>`, `Color4f` on an sRGB surface).
//! Gradient stops are instead passed through as `peniko`/`color`
//! `DynamicColor`s, preserving the declared colour space end to end where
//! the engine supports it.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use cherenkov_oracle::color::{linear_p3_to_linear_srgb, srgb_encode, to_working};
use cherenkov_scene::{
    BlendMode, Color, ColorSpace, Draw, Extend, Feature, FillRule, ImagePaint, Item, Layer,
    NormalizedCoord, Paint, ResourceHash, Scene, Shape, StrokeStyle,
};
use kurbo::{Affine, BezPath};
use peniko::color::{
    AlphaColor, ColorSpaceTag, DisplayP3, DynamicColor, LinearSrgb, Rec2020, Srgb,
};
use peniko::{
    Blob, Brush, ColorStops, Extend as PExtend, FontData, ImageBrush, ImageData, ImageSampler,
};

use crate::{BenchError, Counters};

/// Resource blobs (fonts, images) loaded for a scene, keyed by hash.
pub type Blobs = BTreeMap<ResourceHash, Vec<u8>>;

/// `F2Dot14` bit patterns per distinct normalized-coord set of one font:
/// `(declared coords, resolved bits)` pairs.
type CoordSets = Vec<(Vec<NormalizedCoord>, Vec<i16>)>;

/// Per-scene resources resolved once in [`crate::Engine::prepare`].
///
/// This is everything an application would build once and cache across
/// frames — decoded image payloads, `peniko` font data, resolved
/// variation coordinates, and GPU image handles where the adapter uploads
/// resources — so the timed per-frame `encode` covers only the engine's
/// own recording calls, not resource creation.
pub struct Prepared {
    /// `peniko::FontData` per `(font hash, face index)` — an `Arc`-backed
    /// handle, not a re-parse; the blob bytes are cloned cheaply.
    fonts: HashMap<(ResourceHash, u32), FontData>,
    /// `F2Dot14` design-axis bit patterns per font hash + coord set. The
    /// skrifa `FontRef` parse that orders axes happens in `build`, not per
    /// frame.
    coords: HashMap<ResourceHash, CoordSets>,
    /// Decoded `rgba8` image payloads per image hash.
    images: HashMap<ResourceHash, ImageData>,
    /// GPU image handles per hash where the adapter uploads resources at
    /// prepare time (`vello-hybrid` atlas ids, plus a transparency hint);
    /// empty for adapters that keep image paints CPU-resident.
    gpu_images: HashMap<ResourceHash, (u32, bool)>,
}

impl Prepared {
    /// Builds the resource set for `scene`: wraps every glyph run's font
    /// in `peniko::FontData`, resolves its normalized coords through
    /// skrifa once, and decodes every referenced image.
    ///
    /// # Errors
    /// [`BenchError`] on a missing or undecodable resource.
    pub fn build(scene: &Scene, blobs: &Blobs) -> Result<Self, BenchError> {
        let mut p = Self {
            fonts: HashMap::new(),
            coords: HashMap::new(),
            images: HashMap::new(),
            gpu_images: HashMap::new(),
        };
        visit(&mut p, &scene.root, blobs)?;
        // `Paint::Image` can also appear inside glyph-run paints.
        for layer in paint_images(scene) {
            p.decode(layer, blobs)?;
        }
        Ok(p)
    }

    /// Registers a GPU-resident image handle (`vello-hybrid` atlas id).
    pub fn set_gpu_image(&mut self, hash: ResourceHash, id: u32, may_have_transparency: bool) {
        self.gpu_images.insert(hash, (id, may_have_transparency));
    }

    /// The hash + decoded payload of every prepared image (for adapters
    /// that upload resources to the GPU in `prepare`).
    pub fn image_entries(&self) -> impl Iterator<Item = (ResourceHash, &ImageData)> {
        self.images.iter().map(|(h, d)| (*h, d))
    }

    /// Decoded `peniko::ImageData` for an image hash.
    ///
    /// # Errors
    /// [`BenchError::Scene`] when the image was not prepared.
    pub fn image(&self, hash: ResourceHash) -> Result<&ImageData, BenchError> {
        self.images
            .get(&hash)
            .ok_or_else(|| cherenkov_scene::SceneError::MissingResource(hash).into())
    }

    /// Total decoded texel bytes of every prepared image (`w*h*4`) — the
    /// `bytes_uploaded` counter reports what the GPU receives, not the
    /// compressed PNG size.
    #[must_use]
    pub fn texel_bytes(&self) -> u64 {
        self.images
            .values()
            .map(|d| u64::from(d.width) * u64::from(d.height) * 4)
            .sum()
    }

    /// `peniko::FontData` for `(font hash, face index)`.
    ///
    /// # Errors
    /// [`BenchError::Scene`] when the font was not prepared.
    pub fn font(&self, hash: ResourceHash, index: u32) -> Result<FontData, BenchError> {
        self.fonts
            .get(&(hash, index))
            .cloned()
            .ok_or_else(|| cherenkov_scene::SceneError::MissingResource(hash).into())
    }

    /// Resolved `F2Dot14` design-axis bits for a run's normalized coords,
    /// in the font's axis order. Empty when the font has no axes.
    #[must_use]
    pub fn coord_bits(&self, hash: ResourceHash, coords: &[NormalizedCoord]) -> Vec<i16> {
        self.coords
            .get(&hash)
            .and_then(|v| {
                v.iter()
                    .find(|(cs, _)| cs.as_slice() == coords)
                    .map(|(_, b)| b.clone())
            })
            .unwrap_or_default()
    }

    /// Resolves a scene [`Paint`] to a `peniko::Brush`, images resolved to
    /// the decoded payload.
    ///
    /// # Errors
    /// [`BenchError`] for missing resources or unsupported extends.
    pub fn brush(&self, engine: &'static str, paint: &Paint) -> Result<Brush, BenchError> {
        self.brush_inner(engine, paint)
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "peniko gradient geometry is f32; scene geometry is f64"
    )]
    fn brush_inner(&self, engine: &'static str, paint: &Paint) -> Result<Brush, BenchError> {
        Ok(match paint {
            Paint::Solid(c) => Brush::Solid(peniko_solid(c)),
            Paint::Linear(g) => Brush::Gradient(gradient(
                engine,
                peniko::GradientKind::Linear(peniko::LinearGradientPosition {
                    start: g.start,
                    end: g.end,
                }),
                &g.stops,
                g.extend,
                g.interpolation,
            )?),
            Paint::Radial(g) => Brush::Gradient(gradient(
                engine,
                peniko::GradientKind::Radial(peniko::RadialGradientPosition {
                    start_center: g.center0,
                    start_radius: g.r0 as f32,
                    end_center: g.center1,
                    end_radius: g.r1 as f32,
                }),
                &g.stops,
                g.extend,
                g.interpolation,
            )?),
            Paint::Sweep(g) => Brush::Gradient(gradient(
                engine,
                peniko::GradientKind::Sweep(peniko::SweepGradientPosition {
                    center: g.center,
                    start_angle: g.start_angle as f32,
                    end_angle: g.end_angle as f32,
                }),
                &g.stops,
                g.extend,
                g.interpolation,
            )?),
            Paint::Image(ip) => Brush::Image(ImageBrush {
                image: self.image(ip.image)?.clone(),
                sampler: ImageSampler {
                    x_extend: extend(engine, ip.extend_x)?,
                    y_extend: extend(engine, ip.extend_y)?,
                    quality: match ip.sampling {
                        cherenkov_scene::Sampling::Nearest => peniko::ImageQuality::Low,
                        cherenkov_scene::Sampling::Bilinear => peniko::ImageQuality::Medium,
                    },
                    alpha: 1.0,
                },
            }),
        })
    }

    /// An `ImageSource` for `vello_cpu`/`vello_hybrid` paints: the atlas
    /// `OpaqueId` when the adapter uploaded the image in `prepare`, else
    /// the decoded pixmap (the CPU-resident route).
    ///
    /// # Errors
    /// [`BenchError::Scene`] when the image was not prepared.
    #[cfg(any(feature = "vello-cpu", feature = "vello-hybrid"))]
    pub fn image_source(
        &self,
        hash: ResourceHash,
    ) -> Result<vello_common::paint::ImageSource, BenchError> {
        if let Some(&(id, transp)) = self.gpu_images.get(&hash) {
            return Ok(
                vello_common::paint::ImageSource::opaque_id_with_transparency_hint(
                    vello_common::paint::ImageId::new(id),
                    transp,
                ),
            );
        }
        Ok(vello_common::paint::ImageSource::from_peniko_image_data(
            self.image(hash)?,
        ))
    }

    /// `vello_common::paint::PaintType` — the paint type
    /// `vello_cpu`/`vello_hybrid` `set_paint` takes, with images resolved
    /// through [`Prepared::image_source`].
    ///
    /// # Errors
    /// [`BenchError`] for missing resources or unsupported extends.
    #[cfg(any(feature = "vello-cpu", feature = "vello-hybrid"))]
    pub fn paint_type(
        &self,
        engine: &'static str,
        paint: &Paint,
    ) -> Result<vello_common::paint::PaintType, BenchError> {
        use vello_common::paint::PaintType;
        Ok(match paint {
            Paint::Image(ip) => PaintType::Image(peniko::ImageBrush {
                image: self.image_source(ip.image)?,
                sampler: ImageSampler {
                    x_extend: extend(engine, ip.extend_x)?,
                    y_extend: extend(engine, ip.extend_y)?,
                    quality: match ip.sampling {
                        cherenkov_scene::Sampling::Nearest => peniko::ImageQuality::Low,
                        cherenkov_scene::Sampling::Bilinear => peniko::ImageQuality::Medium,
                    },
                    alpha: 1.0,
                },
            }),
            _ => to_paint_type(self.brush_inner(engine, paint)?),
        })
    }

    fn decode(&mut self, hash: ResourceHash, blobs: &Blobs) -> Result<(), BenchError> {
        if self.images.contains_key(&hash) {
            return Ok(());
        }
        let bytes = blobs
            .get(&hash)
            .ok_or(cherenkov_scene::SceneError::MissingResource(hash))?;
        self.images.insert(hash, image_data(bytes)?);
        Ok(())
    }
}

/// Walks `layer` registering every font (plus coord sets) and every
/// top-level `Paint::Image`/`Draw::Image` payload into `p`.
fn visit(p: &mut Prepared, layer: &Layer, blobs: &Blobs) -> Result<(), BenchError> {
    for item in &layer.items {
        match item {
            Item::Layer(l) => visit(p, l, blobs)?,
            Item::Draw(d) => match d {
                Draw::Glyphs(run) => {
                    let bytes = blobs
                        .get(&run.font)
                        .ok_or(cherenkov_scene::SceneError::MissingResource(run.font))?;
                    p.fonts
                        .entry((run.font, run.font_index))
                        .or_insert_with(|| {
                            FontData::new(Blob::new(Arc::new(bytes.clone())), run.font_index)
                        });
                    let coords = coord_bits(bytes, &run.normalized_coords);
                    let entry = p.coords.entry(run.font).or_default();
                    if entry.iter().all(|(cs, _)| cs != &run.normalized_coords) {
                        entry.push((run.normalized_coords.clone(), coords));
                    }
                }
                Draw::Fill { paint, .. } | Draw::Stroke { paint, .. } => {
                    if let Some(ip) = image_paint(paint) {
                        p.decode(ip.image, blobs)?;
                    }
                }
                Draw::Shadow { .. } => {}
                Draw::Image { image, .. } => {
                    p.decode(*image, blobs)?;
                }
            },
        }
    }
    Ok(())
}

/// The `Paint::Image` inside a draw's paint, when any.
const fn image_paint(paint: &Paint) -> Option<&ImagePaint> {
    match paint {
        Paint::Image(ip) => Some(ip),
        _ => None,
    }
}

/// Image hashes referenced by `Paint::Image` inside glyph-run paints
/// (the draw walk above only inspects top-level paints).
fn paint_images(scene: &Scene) -> Vec<ResourceHash> {
    fn visit(layer: &Layer, out: &mut Vec<ResourceHash>) {
        for item in &layer.items {
            match item {
                Item::Layer(l) => visit(l, out),
                Item::Draw(Draw::Glyphs(run)) => {
                    if let Paint::Image(ip) = &run.paint {
                        out.push(ip.image);
                    }
                }
                Item::Draw(_) => {}
            }
        }
    }
    let mut out = Vec::new();
    visit(&scene.root, &mut out);
    out
}

/// Loads every resource `scene` references from `dir`.
///
/// # Errors
/// [`BenchError::Scene`] on missing blobs or I/O failure.
pub fn load_blobs(scene: &Scene, dir: &std::path::Path) -> Result<Blobs, BenchError> {
    let mut out = Blobs::new();
    for hash in scene.resource_refs() {
        out.insert(hash, Scene::resource(dir, hash)?);
    }
    Ok(out)
}

/// Errors when the scene declares a feature the adapter does not implement.
/// `api_for` names the upstream API the engine lacks for a missing feature.
///
/// # Errors
/// [`BenchError::Unsupported`] naming the first missing feature.
pub fn check_features(
    engine: &'static str,
    scene: &Scene,
    supported: &[Feature],
    api_for: impl Fn(&Feature) -> Option<&'static str>,
) -> Result<(), BenchError> {
    for f in &scene.features {
        if !supported.contains(f) {
            return Err(BenchError::Unsupported {
                engine,
                feature: f.clone(),
                api: api_for(f),
            });
        }
    }
    Ok(())
}

/// Converts a scene colour to `rgba8` straight alpha (sRGB encoded).
///
/// Wide-gamut channels are clamped to the sRGB gamut; HDR channels are
/// clamped to `0.0..=1.0`.
#[must_use]
#[allow(clippy::many_single_char_names)] // r/g/b/a channel names
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "channels are clamped to [0,1] before the deliberate u8 quantize"
)]
pub fn straight_srgb8(c: &Color) -> [u8; 4] {
    let [r, g, b, a] = to_working(c);
    let [lr, lg, lb] = if a > 1e-12 {
        [r / a, g / a, b / a]
    } else {
        [0.0; 3]
    };
    let [sr, sg, sb] = linear_p3_to_linear_srgb([lr, lg, lb]);
    [
        (srgb_encode(sr.clamp(0.0, 1.0)) * 255.0).round() as u8,
        (srgb_encode(sg.clamp(0.0, 1.0)) * 255.0).round() as u8,
        (srgb_encode(sb.clamp(0.0, 1.0)) * 255.0).round() as u8,
        (a.clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

/// Converts a scene colour to a `peniko` `AlphaColor<Srgb>`.
#[must_use]
#[allow(clippy::many_single_char_names)] // r/g/b/a channel names
pub fn peniko_solid(c: &Color) -> AlphaColor<Srgb> {
    let [r, g, b, a] = straight_srgb8(c);
    AlphaColor::from_rgba8(r, g, b, a)
}

/// Preserves a scene colour's declared space in a `DynamicColor` for
/// gradient stops.
#[must_use]
#[allow(clippy::many_single_char_names)] // r/g/b/a channel names
#[expect(
    clippy::cast_possible_truncation,
    reason = "peniko colours are f32; the scene's f64 channel math narrows"
)]
pub fn dynamic_color(c: &Color) -> DynamicColor {
    let [r, g, b, a] = c.components;
    match c.space {
        ColorSpace::Srgb => DynamicColor::from_alpha_color(AlphaColor::<Srgb>::new([r, g, b, a])),
        ColorSpace::DisplayP3 => {
            DynamicColor::from_alpha_color(AlphaColor::<DisplayP3>::new([r, g, b, a]))
        }
        ColorSpace::LinearSrgb => {
            DynamicColor::from_alpha_color(AlphaColor::<LinearSrgb>::new([r, g, b, a]))
        }
        // `peniko`/`color` 0.3 has no LinearP3 tag: express the colour in
        // linear sRGB instead — the same linear-light space vello evaluates
        // gradients in — so the value survives without a gamut shift beyond
        // the P3→sRGB matrix the engines apply anyway.
        ColorSpace::LinearP3 => {
            let [sr, sg, sb] = linear_p3_to_linear_srgb([f64::from(r), f64::from(g), f64::from(b)]);
            DynamicColor::from_alpha_color(AlphaColor::<LinearSrgb>::new([
                sr as f32, sg as f32, sb as f32, a,
            ]))
        }
        ColorSpace::Rec2020 => {
            DynamicColor::from_alpha_color(AlphaColor::<Rec2020>::new([r, g, b, a]))
        }
    }
}

/// The upstream API vello-family adapters lack for [`Extend::None`]:
/// `peniko::Extend` offers only `Pad`/`Repeat`/`Reflect` (peniko 0.6).
///
/// Skia does have it (`SkTileMode::kDecal`), so this is a `peniko` gap, not
/// an inherent engine limitation.
pub const EXTEND_NONE_API: &str =
    "peniko::Extend has no None variant (peniko offers Pad/Repeat/Reflect only)";

/// Maps a scene [`Extend`] to `peniko`'s `Extend` (`Pad`, `Repeat`,
/// `Reflect`). `None` (transparent outside the domain) has no `peniko`
/// equivalent and is reported as unsupported by the adapter.
///
/// # Errors
/// [`BenchError::Unsupported`] for [`Extend::None`].
pub const fn extend(engine: &'static str, e: Extend) -> Result<PExtend, BenchError> {
    match e {
        Extend::Pad => Ok(PExtend::Pad),
        Extend::Repeat => Ok(PExtend::Repeat),
        Extend::Reflect => Ok(PExtend::Reflect),
        Extend::None => Err(BenchError::Unsupported {
            engine,
            feature: Feature::ExtendNone,
            api: Some(EXTEND_NONE_API),
        }),
    }
}

/// `peniko` adapters cannot express `linear-p3` gradient interpolation.
///
/// The upstream API missing for [`Feature::InterpolationSpace`]
/// (LinearP3): `color` 0.3's `ColorSpaceTag` has no linear-P3 variant, so
/// a gradient declaring `linear-p3` interpolation cannot be expressed and
/// must be reported unsupported rather than silently remapped to linear
/// sRGB.
pub const LINEAR_P3_API: &str = "peniko::color::ColorSpaceTag has no linear-P3 variant";

/// Maps the scene's interpolation colour space onto a `ColorSpaceTag`,
/// `None` for [`ColorSpace::LinearP3`] (no tag — see [`LINEAR_P3_API`]).
#[must_use]
pub const fn interpolation_tag(space: ColorSpace) -> Option<ColorSpaceTag> {
    Some(match space {
        ColorSpace::Srgb => ColorSpaceTag::Srgb,
        ColorSpace::DisplayP3 => ColorSpaceTag::DisplayP3,
        ColorSpace::LinearSrgb => ColorSpaceTag::LinearSrgb,
        ColorSpace::LinearP3 => return None,
        ColorSpace::Rec2020 => ColorSpaceTag::Rec2020,
    })
}

/// Builds `peniko` colour stops preserving declared colour spaces.
#[must_use]
pub fn color_stops(stops: &[cherenkov_scene::GradientStop]) -> ColorStops {
    let v: Vec<peniko::ColorStop> = stops
        .iter()
        .map(|s| (s.offset, dynamic_color(&s.color)).into())
        .collect();
    ColorStops::from(&v[..])
}

/// Builds a `peniko` gradient from the scene description.
///
/// # Errors
/// [`BenchError::Unsupported`] for [`Extend::None`].
pub fn gradient(
    engine: &'static str,
    kind: peniko::GradientKind,
    stops: &[cherenkov_scene::GradientStop],
    ext: Extend,
    interpolation: ColorSpace,
) -> Result<peniko::Gradient, BenchError> {
    let interpolation_cs = interpolation_tag(interpolation).ok_or(BenchError::Unsupported {
        engine,
        feature: Feature::InterpolationSpace(interpolation),
        api: Some(LINEAR_P3_API),
    })?;
    Ok(peniko::Gradient {
        kind,
        extend: extend(engine, ext)?,
        interpolation_cs,
        stops: color_stops(stops),
        ..Default::default()
    })
}

/// Decodes a PNG blob into straight `rgba8` pixels — every colour type,
/// via the shared [`cherenkov_oracle::image::decode_png_rgba8`].
///
/// # Errors
/// [`BenchError::Engine`] on decode failure.
pub fn decode_png(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), BenchError> {
    cherenkov_oracle::image::decode_png_rgba8(bytes)
        .map_err(|e| BenchError::Engine(format!("png decode: {e}")))
}

/// Decoded texel byte count (`w*h*4`) of an image blob — the `bytes_uploaded`
/// counter reports what the GPU receives, not the compressed PNG size.
///
/// # Errors
/// [`BenchError::Engine`] on decode failure.
pub fn decoded_texel_bytes(bytes: &[u8]) -> Result<u64, BenchError> {
    let (w, h, _) = decode_png(bytes)?;
    Ok(u64::from(w) * u64::from(h) * 4)
}

/// Builds `peniko::ImageData` (straight alpha RGBA8) from a PNG blob.
///
/// # Errors
/// [`BenchError::Engine`] on decode failure.
pub fn image_data(png_bytes: &[u8]) -> Result<ImageData, BenchError> {
    let (w, h, rgba) = decode_png(png_bytes)?;
    Ok(ImageData {
        data: Blob::new(Arc::new(rgba)),
        format: peniko::ImageFormat::Rgba8,
        alpha_type: peniko::ImageAlphaType::Alpha,
        width: w,
        height: h,
    })
}

/// Resolves an image paint to a `peniko::ImageBrush`.
///
/// # Errors
/// [`BenchError::Engine`] when the blob is missing or undecodable, or
/// [`BenchError::Unsupported`] for [`Extend::None`].
pub fn image_brush(
    engine: &'static str,
    ip: &ImagePaint,
    blobs: &Blobs,
) -> Result<ImageBrush, BenchError> {
    let bytes = blobs.get(&ip.image).ok_or(BenchError::Scene(
        cherenkov_scene::SceneError::MissingResource(ip.image),
    ))?;
    Ok(ImageBrush {
        image: image_data(bytes)?,
        sampler: ImageSampler {
            x_extend: extend(engine, ip.extend_x)?,
            y_extend: extend(engine, ip.extend_y)?,
            quality: match ip.sampling {
                cherenkov_scene::Sampling::Nearest => peniko::ImageQuality::Low,
                cherenkov_scene::Sampling::Bilinear => peniko::ImageQuality::Medium,
            },
            alpha: 1.0,
        },
    })
}

/// Resolves a scene [`Paint`] to a `peniko::Brush`.
///
/// # Errors
/// [`BenchError`] for missing resources or unsupported extends.
#[expect(
    clippy::cast_possible_truncation,
    reason = "peniko gradient geometry is f32; scene geometry is f64"
)]
pub fn brush(engine: &'static str, paint: &Paint, blobs: &Blobs) -> Result<Brush, BenchError> {
    Ok(match paint {
        Paint::Solid(c) => Brush::Solid(peniko_solid(c)),
        Paint::Linear(g) => Brush::Gradient(gradient(
            engine,
            peniko::GradientKind::Linear(peniko::LinearGradientPosition {
                start: g.start,
                end: g.end,
            }),
            &g.stops,
            g.extend,
            g.interpolation,
        )?),
        Paint::Radial(g) => Brush::Gradient(gradient(
            engine,
            peniko::GradientKind::Radial(peniko::RadialGradientPosition {
                start_center: g.center0,
                start_radius: g.r0 as f32,
                end_center: g.center1,
                end_radius: g.r1 as f32,
            }),
            &g.stops,
            g.extend,
            g.interpolation,
        )?),
        Paint::Sweep(g) => Brush::Gradient(gradient(
            engine,
            peniko::GradientKind::Sweep(peniko::SweepGradientPosition {
                center: g.center,
                start_angle: g.start_angle as f32,
                end_angle: g.end_angle as f32,
            }),
            &g.stops,
            g.extend,
            g.interpolation,
        )?),
        Paint::Image(ip) => Brush::Image(image_brush(engine, ip, blobs)?),
    })
}

/// Builds `vello_common::paint::PaintType` — the paint type
/// `vello_cpu`/`vello_hybrid` `set_paint` actually takes (an image brush over
/// `vello_common::paint::ImageSource`, not plain `peniko::ImageData`).
///
/// # Errors
/// [`BenchError`] for missing resources or unsupported extends.
#[cfg(any(feature = "vello-cpu", feature = "vello-hybrid"))]
pub fn paint_type(
    engine: &'static str,
    paint: &Paint,
    blobs: &Blobs,
) -> Result<vello_common::paint::PaintType, BenchError> {
    Ok(to_paint_type(brush(engine, paint, blobs)?))
}

/// Converts a default-generic `peniko::Brush` into
/// `vello_common::paint::PaintType`, converting the image payload.
///
/// # Panics
/// [`vello_common::paint::ImageSource::from_peniko_image_data`] panics for
/// images larger than `u16::MAX` in either dimension.
#[cfg(any(feature = "vello-cpu", feature = "vello-hybrid"))]
#[must_use]
pub fn to_paint_type(brush: Brush) -> vello_common::paint::PaintType {
    match brush {
        Brush::Solid(c) => vello_common::paint::PaintType::Solid(c),
        Brush::Gradient(g) => vello_common::paint::PaintType::Gradient(g),
        Brush::Image(ib) => vello_common::paint::PaintType::Image(peniko::ImageBrush {
            image: vello_common::paint::ImageSource::from_peniko_image_data(&ib.image),
            sampler: ib.sampler,
        }),
    }
}

/// The brush-space transform a paint needs (`ImagePaint::transform` only).
#[must_use]
pub const fn brush_transform(paint: &Paint) -> Option<Affine> {
    match paint {
        Paint::Image(ip) => Some(ip.transform),
        _ => None,
    }
}

/// The transform that scales an `iw`×`ih` image to fill `dst`.
#[must_use]
pub fn image_draw_transform(iw: u32, ih: u32, dst: kurbo::Rect) -> Affine {
    Affine::translate((dst.x0, dst.y0))
        * Affine::scale_non_uniform(dst.width() / f64::from(iw), dst.height() / f64::from(ih))
}

/// Resolves a font blob + index into `peniko::FontData`.
///
/// # Errors
/// [`BenchError::Scene`] when the blob is missing.
pub fn font_data(
    blobs: &Blobs,
    hash: ResourceHash,
    index: u32,
) -> Result<peniko::FontData, BenchError> {
    let bytes = blobs
        .get(&hash)
        .ok_or(cherenkov_scene::SceneError::MissingResource(hash))?;
    Ok(peniko::FontData::new(
        Blob::new(Arc::new(bytes.clone())),
        index,
    ))
}

/// Scene normalized coords → `F2Dot14` bit patterns ordered by the font's
/// variation axes (the order all engines expect).
///
/// Unknown axes fall back to the axis default; axes absent from `coords`
/// fall back to `0` (font default).
#[must_use]
pub fn coord_bits(font_bytes: &[u8], coords: &[NormalizedCoord]) -> Vec<i16> {
    use skrifa::MetadataProvider;
    let Ok(font) = skrifa::FontRef::new(font_bytes) else {
        return Vec::new();
    };
    font.axes()
        .iter()
        .map(|axis| {
            let tag = axis.tag().to_string();
            // An axis absent from `coords` is normalized `0` (the font
            // default) — the axis's user-space default such as `400`
            // would saturate the `F2Dot14` range around `±2`.
            let v = coords
                .iter()
                .find(|c| c.tag == tag)
                .map_or(0.0, |c| c.value);
            skrifa::raw::types::F2Dot14::from_f32(v).to_bits()
        })
        .collect()
}

/// Rasterizes a shape to a `kurbo` path (`Rect`, `RoundedRect`,
/// `Continuous`, `Circle`, `Ellipse`, `Line`, `Path`).
#[must_use]
pub fn shape_path(shape: &Shape) -> BezPath {
    shape.to_path()
}

/// `peniko::ImageSampler` for `Draw::Image` — `Pad` extends and the
/// declared sampling quality. (`Paint::Image` samplers take their
/// extends from the paint instead.)
#[must_use]
pub const fn image_sampler(sampling: cherenkov_scene::Sampling) -> ImageSampler {
    ImageSampler {
        x_extend: peniko::Extend::Pad,
        y_extend: peniko::Extend::Pad,
        quality: match sampling {
            cherenkov_scene::Sampling::Nearest => peniko::ImageQuality::Low,
            cherenkov_scene::Sampling::Bilinear => peniko::ImageQuality::Medium,
        },
        alpha: 1.0,
    }
}

/// Scene stroke style → `kurbo::Stroke`.
#[must_use]
pub fn stroke(style: &StrokeStyle) -> kurbo::Stroke {
    kurbo::Stroke::from(style)
}

/// Scene blend mode → `peniko::BlendMode`: a `Mix` over `SrcOver` for the
/// W3C blend modes, `Mix::Normal` over the matching `Compose` for the
/// Porter-Duff compositing operators and plus-lighter.
#[must_use]
pub const fn blend(m: BlendMode) -> peniko::BlendMode {
    use peniko::{Compose, Mix};
    let compose = match m {
        BlendMode::Clear => Compose::Clear,
        BlendMode::Src => Compose::Copy,
        BlendMode::Dst => Compose::Dest,
        BlendMode::DestOver => Compose::DestOver,
        BlendMode::SrcIn => Compose::SrcIn,
        BlendMode::DestIn => Compose::DestIn,
        BlendMode::SrcOut => Compose::SrcOut,
        BlendMode::DestOut => Compose::DestOut,
        BlendMode::SrcAtop => Compose::SrcAtop,
        BlendMode::DestAtop => Compose::DestAtop,
        BlendMode::Xor => Compose::Xor,
        BlendMode::PlusLighter => Compose::PlusLighter,
        _ => {
            let mix = match m {
                BlendMode::Normal => Mix::Normal,
                BlendMode::Multiply => Mix::Multiply,
                BlendMode::Screen => Mix::Screen,
                BlendMode::Overlay => Mix::Overlay,
                BlendMode::Darken => Mix::Darken,
                BlendMode::Lighten => Mix::Lighten,
                BlendMode::ColorDodge => Mix::ColorDodge,
                BlendMode::ColorBurn => Mix::ColorBurn,
                BlendMode::HardLight => Mix::HardLight,
                BlendMode::SoftLight => Mix::SoftLight,
                BlendMode::Difference => Mix::Difference,
                BlendMode::Exclusion => Mix::Exclusion,
                BlendMode::Hue => Mix::Hue,
                BlendMode::Saturation => Mix::Saturation,
                BlendMode::Color => Mix::Color,
                BlendMode::Luminosity => Mix::Luminosity,
                _ => unreachable!(),
            };
            return peniko::BlendMode::new(mix, Compose::SrcOver);
        }
    };
    peniko::BlendMode::new(Mix::Normal, compose)
}

/// Scene fill rule → `peniko::Fill`.
#[must_use]
pub const fn fill(r: FillRule) -> peniko::Fill {
    match r {
        FillRule::NonZero => peniko::Fill::NonZero,
        FillRule::EvenOdd => peniko::Fill::EvenOdd,
    }
}

/// Converts an `rgba8` premultiplied sRGB buffer (engine readback) into
/// premultiplied linear-P3 `f32` pixels — the suite interchange.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the f32 interchange image deliberately narrows the f64 pipeline"
)]
pub fn rgba8_to_working(width: u32, height: u32, rgba8: &[u8]) -> cherenkov_oracle::F32Image {
    use cherenkov_oracle::color::linear_srgb_to_linear_p3;
    let mut img = cherenkov_oracle::F32Image::new(width, height);
    for (px, out) in rgba8.as_chunks::<4>().0.iter().zip(img.pixels.iter_mut()) {
        let a = f64::from(px[3]) / 255.0;
        let lin = [
            srgb_decode(f64::from(px[0]) / 255.0),
            srgb_decode(f64::from(px[1]) / 255.0),
            srgb_decode(f64::from(px[2]) / 255.0),
        ];
        let p3 = linear_srgb_to_linear_p3(lin);
        *out = [p3[0] as f32, p3[1] as f32, p3[2] as f32, a as f32];
    }
    img
}

/// sRGB transfer-function decode (inverse of `srgb_encode`).
#[must_use]
fn srgb_decode(e: f64) -> f64 {
    if e <= 0.04045 {
        e / 12.92
    } else {
        ((e + 0.055) / 1.055).powf(2.4)
    }
}

/// IEEE 754 binary16 → `f32`, used by the skia-vulkan/`RGBAF16` readback.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "f16→f64→f32 round trip is the conversion's contract"
)]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) & 1;
    let exp = u32::from(bits >> 10) & 0x1f;
    let frac = f64::from(bits & 0x3ff);
    let v = match exp {
        0 => frac * 2f64.powi(-24),
        0x1f => {
            if frac == 0.0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        }
        e => (1.0 + frac / 1024.0) * 2f64.powi(e.cast_signed() - 15),
    };
    (if sign == 1 { -v } else { v }) as f32
}

/// Counts draw commands and nested layers inside a layer tree (adapter
/// counters — every adapter reports what it actually issued).
pub fn count_layer(layer: &Layer, counters: &mut Counters) {
    counters.layers += 1;
    for item in &layer.items {
        match item {
            Item::Layer(l) => count_layer(l, counters),
            Item::Draw(
                Draw::Fill { .. }
                | Draw::Stroke { .. }
                | Draw::Shadow { .. }
                | Draw::Glyphs(_)
                | Draw::Image { .. },
            ) => counters.draw_commands += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A variable font (`wdth,wght`) with only `wght` specified: the
    /// omitted `wdth` axis must resolve to normalized `0` (the font
    /// default), not the axis's user-space default — a value like `75`
    /// would saturate `F2Dot14` around `±2` and render the axis at its
    /// extreme.
    #[test]
    fn coord_bits_omitted_axis_is_normalized_zero() {
        let font = std::fs::read("../scenes/fonts/NotoSans.ttf").expect("test font");
        let bits = coord_bits(
            &font,
            &[NormalizedCoord {
                tag: "wght".into(),
                value: 1.0,
            }],
        );
        assert_eq!(bits.len(), 2, "wdth,wght font should have two axes");
        assert!(
            bits.contains(&0),
            "omitted wdth axis should produce bits 0: {bits:?}"
        );
        let wght = skrifa::raw::types::F2Dot14::from_f32(1.0).to_bits();
        assert!(bits.contains(&wght), "wght=1.0 bits missing: {bits:?}");
    }
}
