// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    BlendMode, Color, ColorSpace, Draw, Extend, Item, Layer, Paint, ResourceHash, SceneError, Shape,
};

/// The file name of the serialized scene inside a scene directory.
pub const SCENE_FILE: &str = "scene.json";
/// The name of the content-addressed resource directory inside a scene
/// directory.
pub const RESOURCES_DIR: &str = "resources";

/// A feature of the scene format that an adapter may or may not support.
///
/// A scene's `features` set is computed from its content (see
/// [`Scene::compute_features`]); an adapter maps this set to its capability
/// table and reports the scene as unsupported rather than emulating.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "feature", content = "value", rename_all = "kebab-case")]
pub enum Feature {
    /// `Fill` draw commands.
    Fill,
    /// Even-odd fill rule.
    EvenOdd,
    /// `Stroke` draw commands.
    Stroke,
    /// Dashed strokes.
    StrokeDash,
    /// Arbitrary paths.
    Path,
    /// Continuous (superellipse) corners.
    ContinuousCorners,
    /// Linear gradients.
    LinearGradient,
    /// Two-point radial gradients.
    RadialGradient,
    /// Sweep gradients.
    SweepGradient,
    /// `Image` draw commands.
    Image,
    /// Image pattern paints.
    ImagePaint,
    /// A non-trivial blend mode (the payload is the mode).
    Blend(BlendMode),
    /// Layer clips.
    Clip,
    /// A non-zero `scroll_offset` on a layer.
    Scroll,
    /// A `motion` on a layer.
    Animation,
    /// Group opacity below `1.0`.
    Opacity,
    /// `Shadow` draw commands.
    Shadow,
    /// `Glyphs` draw commands.
    Glyphs,
    /// Variable-font normalized coordinates.
    FontVariations,
    /// Any colour channel above `1.0`.
    HdrColor,
    /// Colours outside the sRGB gamut (P3, Rec. 2020).
    WideGamut,
    /// A gradient or image pattern with `Extend::None` (transparent outside
    /// the domain). Several paint APIs only offer pad/repeat/reflect.
    ExtendNone,
    /// A gradient whose stops are interpolated in the payload space. Engines
    /// that cannot interpolate in a declared space must report it instead of
    /// silently remapping.
    InterpolationSpace(ColorSpace),
}

/// The scene's working space. Only linear Display P3 exists today; the enum
/// keeps the contract explicit for adapters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorkingSpace {
    /// Linear-light Display P3, premultiplied compositing in `f64` in the
    /// oracle, HDR-capable.
    #[default]
    LinearDisplayP3,
}

/// An engine-neutral scene: a pixel size, a clear colour, and a layer tree.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    /// Width in physical pixels.
    pub width: u32,
    /// Height in physical pixels.
    pub height: u32,
    /// The working space (always linear Display P3 today).
    pub working_space: WorkingSpace,
    /// The colour the scene is cleared to before drawing.
    pub clear: Color,
    /// The features this scene uses.
    pub features: BTreeSet<Feature>,
    /// The root layer.
    pub root: Layer,
}

impl Scene {
    /// Create an empty scene of `width`×`height` cleared to `clear`.
    #[must_use]
    pub fn new(width: u32, height: u32, clear: Color) -> Self {
        Self {
            width,
            height,
            working_space: WorkingSpace::LinearDisplayP3,
            clear,
            features: BTreeSet::new(),
            root: Layer::default(),
        }
    }

    /// Recompute `self.features` from the layer tree. Called by builders;
    /// call again after mutating a scene by hand.
    pub fn compute_features(&mut self) {
        let mut f = BTreeSet::new();
        if self.clear.is_hdr() {
            f.insert(Feature::HdrColor);
        }
        if self.clear.is_wide_gamut() {
            f.insert(Feature::WideGamut);
        }
        collect_layer_features(&self.root, &mut f);
        self.features = f;
    }

    /// Load `scene.json` from a scene directory.
    ///
    /// # Errors
    /// Returns [`SceneError`] on I/O or JSON failures.
    pub fn load(dir: &Path) -> Result<Self, SceneError> {
        let text = std::fs::read_to_string(dir.join(SCENE_FILE))?;
        let mut scene: Self = serde_json::from_str(&text)?;
        // The stored `features` set is advisory input, not truth: recompute
        // it from the layer tree and reject scenes that lie about what they
        // use (a stale or hand-edited file would otherwise bypass an
        // adapter's capability check).
        let declared = std::mem::take(&mut scene.features);
        scene.compute_features();
        if scene.features != declared {
            return Err(SceneError::FeatureMismatch {
                declared: declared.into_iter().collect(),
                computed: scene.features.iter().cloned().collect(),
            });
        }
        Ok(scene)
    }

    /// Write `scene.json` into `dir` (creating it), without touching
    /// `resources/`.
    ///
    /// # Errors
    /// Returns [`SceneError`] on I/O or JSON failures.
    pub fn save(&self, dir: &Path) -> Result<(), SceneError> {
        std::fs::create_dir_all(dir.join(RESOURCES_DIR))?;
        let mut text = serde_json::to_string_pretty(self)?;
        text.push('\n');
        std::fs::write(dir.join(SCENE_FILE), text)?;
        Ok(())
    }

    /// The `resources/` directory of a scene directory.
    #[must_use]
    pub fn resources_dir(dir: &Path) -> PathBuf {
        dir.join(RESOURCES_DIR)
    }

    /// Insert `bytes` into `dir`'s resource store and return the hash.
    ///
    /// # Errors
    /// Returns [`SceneError`] on I/O failure.
    pub fn store_resource(dir: &Path, bytes: &[u8]) -> Result<ResourceHash, SceneError> {
        let hash = ResourceHash::of(bytes);
        let resources = Self::resources_dir(dir);
        std::fs::create_dir_all(&resources)?;
        std::fs::write(resources.join(hash.file_name()), bytes)?;
        Ok(hash)
    }

    /// Read the resource blob `hash` from `dir`.
    ///
    /// # Errors
    /// [`SceneError::MissingResource`] if the blob is absent, [`SceneError::Io`]
    /// on read failure.
    pub fn resource(dir: &Path, hash: ResourceHash) -> Result<Vec<u8>, SceneError> {
        let path = Self::resources_dir(dir).join(hash.file_name());
        match std::fs::read(&path) {
            Ok(b) => Ok(b),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(SceneError::MissingResource(hash))
            }
            Err(e) => Err(SceneError::Io(e)),
        }
    }

    /// All resource hashes referenced by the scene.
    #[must_use]
    pub fn resource_refs(&self) -> Vec<ResourceHash> {
        let mut out = Vec::new();
        collect_resource_refs(&self.root, &mut out);
        out
    }
}

fn collect_resource_refs(layer: &Layer, out: &mut Vec<ResourceHash>) {
    for item in &layer.items {
        match item {
            Item::Layer(l) => collect_resource_refs(l, out),
            Item::Draw(Draw::Glyphs(run)) => {
                out.push(run.font);
                if let Paint::Image(ip) = &run.paint {
                    out.push(ip.image);
                }
            }
            Item::Draw(Draw::Image { image, .. }) => out.push(*image),
            Item::Draw(Draw::Fill { paint, .. } | Draw::Stroke { paint, .. }) => {
                if let Paint::Image(ip) = paint {
                    out.push(ip.image);
                }
            }
            Item::Draw(Draw::Shadow { .. }) => {}
        }
    }
}

fn collect_paint_features(paint: &Paint, f: &mut BTreeSet<Feature>) {
    match paint {
        Paint::Solid(c) => collect_color_features(c, f),
        Paint::Linear(g) => {
            f.insert(Feature::LinearGradient);
            f.insert(Feature::InterpolationSpace(g.interpolation));
            if g.extend == Extend::None {
                f.insert(Feature::ExtendNone);
            }
            collect_stops(&g.stops, f);
        }
        Paint::Radial(g) => {
            f.insert(Feature::RadialGradient);
            f.insert(Feature::InterpolationSpace(g.interpolation));
            if g.extend == Extend::None {
                f.insert(Feature::ExtendNone);
            }
            collect_stops(&g.stops, f);
        }
        Paint::Sweep(g) => {
            f.insert(Feature::SweepGradient);
            f.insert(Feature::InterpolationSpace(g.interpolation));
            if g.extend == Extend::None {
                f.insert(Feature::ExtendNone);
            }
            collect_stops(&g.stops, f);
        }
        Paint::Image(ip) => {
            f.insert(Feature::ImagePaint);
            if ip.extend_x == Extend::None || ip.extend_y == Extend::None {
                f.insert(Feature::ExtendNone);
            }
        }
    }
}

fn collect_stops(stops: &[crate::draw::GradientStop], f: &mut BTreeSet<Feature>) {
    for s in stops {
        collect_color_features(&s.color, f);
    }
}

fn collect_color_features(c: &Color, f: &mut BTreeSet<Feature>) {
    if c.is_hdr() {
        f.insert(Feature::HdrColor);
    }
    if c.is_wide_gamut() {
        f.insert(Feature::WideGamut);
    }
}

fn collect_shape_features(shape: &Shape, f: &mut BTreeSet<Feature>) {
    match shape {
        Shape::Continuous(_) => {
            f.insert(Feature::ContinuousCorners);
        }
        Shape::Path { .. } => {
            f.insert(Feature::Path);
        }
        _ => {}
    }
}

fn collect_layer_features(layer: &Layer, f: &mut BTreeSet<Feature>) {
    if layer.clip.is_some() {
        f.insert(Feature::Clip);
    }
    if layer.opacity < 1.0 {
        f.insert(Feature::Opacity);
    }
    if layer.blend != BlendMode::Normal {
        f.insert(Feature::Blend(layer.blend));
    }
    if layer.scroll_offset != kurbo::Vec2::ZERO {
        f.insert(Feature::Scroll);
    }
    if layer.motion.is_some() {
        f.insert(Feature::Animation);
    }
    for item in &layer.items {
        match item {
            Item::Layer(l) => collect_layer_features(l, f),
            Item::Draw(d) => match d {
                Draw::Fill { shape, rule, paint } => {
                    f.insert(Feature::Fill);
                    if *rule == crate::FillRule::EvenOdd {
                        f.insert(Feature::EvenOdd);
                    }
                    collect_shape_features(shape, f);
                    collect_paint_features(paint, f);
                }
                Draw::Stroke {
                    shape,
                    stroke,
                    paint,
                } => {
                    f.insert(Feature::Stroke);
                    if !stroke.dash_pattern.is_empty() {
                        f.insert(Feature::StrokeDash);
                    }
                    collect_shape_features(shape, f);
                    collect_paint_features(paint, f);
                }
                Draw::Shadow { shape, color, .. } => {
                    f.insert(Feature::Shadow);
                    collect_shape_features(shape, f);
                    collect_color_features(color, f);
                }
                Draw::Glyphs(run) => {
                    f.insert(Feature::Glyphs);
                    if !run.normalized_coords.is_empty() {
                        f.insert(Feature::FontVariations);
                    }
                    collect_paint_features(&run.paint, f);
                }
                Draw::Image { .. } => {
                    f.insert(Feature::Image);
                }
            },
        }
    }
}
