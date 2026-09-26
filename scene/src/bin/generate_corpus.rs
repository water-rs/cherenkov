// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Writes the initial scene corpus to `scenes/corpus/`.
//!
//! Run `prepare-fonts` first: it produces the OFL subsets in `scenes/fonts/`
//! that this binary shapes against with parley. Each scene lands in
//! `scenes/corpus/<name>/` as a `scene.json` plus a `resources/` directory of
//! BLAKE3-addressed blobs (fonts, images).

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;

use cherenkov_scene::corpus;
use cherenkov_scene::kurbo::{
    Affine, BezPath, Ellipse, Line, Point, Rect, RoundedRect, RoundedRectRadii, Vec2,
};
use cherenkov_scene::{
    BlendMode, Color, ColorSpace, Draw, Extend, FillRule, Glyph, GlyphRun, GradientStop, ImagePaint,
    LayerBuilder, LinearGradient, Live, Motion, MotionAnimation, NormalizedCoord, Paint,
    RadialGradient, ResourceHash, Sampling, Scene, SceneError, Shape, StrokeStyle, SweepGradient,
};
use fontique::FontWeight;
use parley::{
    FontContext, LayoutContext, PositionedLayoutItem, StyleProperty, fontique::Blob,
    style::FontFamily,
};
use read_fonts::types::F2Dot14;
use skrifa::MetadataProvider;

/// Padding around text scenes so ascenders/descenders stay inside.
const TEXT_PAD: f32 = 12.0;
/// Maximum line advance for shaped text.
const TEXT_WRAP: f32 = 280.0;

const fn srgb(r: f32, g: f32, b: f32) -> Color {
    Color::srgb(r, g, b)
}

const fn srgba(r: f32, g: f32, b: f32, a: f32) -> Color {
    Color::srgb(r, g, b).with_alpha(a)
}

const fn solid(c: Color) -> Paint {
    Paint::Solid(c)
}

fn stops2() -> Vec<GradientStop> {
    vec![
        GradientStop {
            offset: 0.0,
            color: srgb(1.0, 0.0, 0.0),
        },
        GradientStop {
            offset: 1.0,
            color: srgb(0.0, 0.0, 1.0),
        },
    ]
}

fn stops8() -> Vec<GradientStop> {
    const RAINBOW: [[f32; 3]; 8] = [
        [1.0, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 1.0],
        [1.0, 0.0, 0.5],
    ];
    RAINBOW
        .iter()
        .enumerate()
        .map(|(i, [r, g, b])| GradientStop {
            offset: i as f32 / 7.0,
            color: srgb(*r, *g, *b),
        })
        .collect()
}

/// Fonts loaded from `scenes/fonts/`, plus a parley context to shape with.
struct TextContext {
    fcx: FontContext,
    lcx: LayoutContext<[u8; 4]>,
    /// Family name per subset file name.
    families: BTreeMap<&'static str, String>,
    /// Every registered font blob, keyed by its content hash.
    blobs: BTreeMap<ResourceHash, Vec<u8>>,
}

impl TextContext {
    fn new(fonts_dir: &Path) -> Result<Self, SceneError> {
        let mut fcx = FontContext::new();
        let mut families = BTreeMap::new();
        let mut blobs = BTreeMap::new();
        for spec in corpus::FONTS {
            let path = fonts_dir.join(spec.subset_file);
            let bytes = std::fs::read(&path)?;
            let hash = ResourceHash::of(&bytes);
            let registered = fcx
                .collection
                .register_fonts(Blob::new(Arc::new(bytes.clone())), None);
            let (family_id, _) = registered
                .first()
                .unwrap_or_else(|| panic!("font {} registered no family", spec.subset_file));
            let family_id = *family_id;
            let name = fcx
                .collection
                .family_name(family_id)
                .unwrap_or_else(|| panic!("font {} has no family name", spec.subset_file));
            families.insert(spec.subset_file, name.to_string());
            blobs.insert(hash, bytes);
        }
        Ok(Self {
            fcx,
            lcx: LayoutContext::new(),
            families,
            blobs,
        })
    }

    /// Shape `text` in the font loaded from `file` and return scene glyph
    /// runs positioned with `TEXT_PAD` padding.
    fn shape(
        &mut self,
        file: &str,
        text: &str,
        size: f32,
        weight: FontWeight,
        paint: &Paint,
    ) -> Vec<GlyphRun> {
        let family = self.families[file].clone();
        let mut builder = self.lcx.ranged_builder(&mut self.fcx, text, 1.0, false);
        builder.push_default(StyleProperty::FontFamily(FontFamily::named(
            family.as_str(),
        )));
        builder.push_default(StyleProperty::FontSize(size));
        builder.push_default(StyleProperty::FontWeight(weight));
        let mut layout = builder.build(text);
        layout.break_all_lines(Some(TEXT_WRAP));

        let mut runs = Vec::new();
        for line in layout.lines() {
            for item in line.items() {
                let PositionedLayoutItem::GlyphRun(gr) = item else {
                    continue;
                };
                let run = gr.run();
                let font_data = run.font();
                let data: &[u8] = font_data.data.data();
                let font = skrifa::FontRef::new(data).expect("registered font parses");
                let axes: Vec<String> = font.axes().iter().map(|a| a.tag().to_string()).collect();
                let coords = run
                    .normalized_coords()
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| **v != 0)
                    .map(|(i, v)| NormalizedCoord {
                        tag: axes.get(i).cloned().unwrap_or_default(),
                        value: F2Dot14::from_bits(*v).to_f32(),
                    })
                    .collect();
                let glyphs = gr
                    .positioned_glyphs()
                    .map(|g| Glyph {
                        id: g.id,
                        x: g.x + TEXT_PAD,
                        y: g.y + TEXT_PAD,
                    })
                    .collect();
                runs.push(GlyphRun {
                    font: ResourceHash::of(data),
                    font_index: font_data.index,
                    size,
                    normalized_coords: coords,
                    glyphs,
                    paint: paint.clone(),
                });
            }
        }
        runs
    }
}

/// The font blob a [`GlyphRun`] was shaped from.
fn font_blob<'a>(ctx: &'a TextContext, run: &'a GlyphRun) -> &'a Vec<u8> {
    &ctx.blobs[&run.font]
}

fn encode_png_rgba(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("png header");
        writer.write_image_data(pixels).expect("png data");
    }
    out
}

/// An 8x8 two-colour checker pattern.
fn checker_png() -> Vec<u8> {
    let mut px = Vec::with_capacity(8 * 8 * 4);
    for y in 0..8u8 {
        for x in 0..8u8 {
            if (x + y) % 2 == 0 {
                px.extend_from_slice(&[255, 255, 255, 255]);
            } else {
                px.extend_from_slice(&[0, 32, 200, 255]);
            }
        }
    }
    encode_png_rgba(8, 8, &px)
}

/// A 16x16 gradient with an alpha ramp.
fn gradient_png() -> Vec<u8> {
    let mut px = Vec::with_capacity(16 * 16 * 4);
    for y in 0..16u8 {
        for x in 0..16u8 {
            px.extend_from_slice(&[x * 16, 128, y * 16, (u16::from(x) * 255 / 15) as u8]);
        }
    }
    encode_png_rgba(16, 16, &px)
}

/// A self-intersecting figure-eight-ish cubic path.
fn self_intersecting_path() -> BezPath {
    let mut p = BezPath::new();
    p.move_to((20.0, 64.0));
    p.curve_to((20.0, 10.0), (108.0, 10.0), (108.0, 64.0));
    p.curve_to((108.0, 118.0), (20.0, 118.0), (20.0, 64.0));
    p.curve_to((20.0, 30.0), (108.0, 30.0), (108.0, 64.0));
    p.curve_to((108.0, 98.0), (20.0, 98.0), (20.0, 64.0));
    p.close_path();
    p
}

/// A path mixing lines and curves.
fn curved_path() -> BezPath {
    let mut p = BezPath::new();
    p.move_to((16.0, 96.0));
    p.curve_to((16.0, 30.0), (64.0, 10.0), (64.0, 48.0));
    p.quad_to((64.0, 86.0), (96.0, 96.0));
    p.line_to((112.0, 96.0));
    p.line_to((112.0, 48.0));
    p.curve_to((112.0, 20.0), (88.0, 12.0), (64.0, 24.0));
    p.close_path();
    p
}

/// Two concentric stars for even-odd vs non-zero comparison.
fn star_path(cx: f64, cy: f64, r0: f64, r1: f64) -> BezPath {
    let mut p = BezPath::new();
    for i in 0..10 {
        let angle = f64::from(i) * std::f64::consts::TAU / 10.0 - std::f64::consts::FRAC_PI_2;
        let r = if i % 2 == 0 { r1 } else { r0 };
        let pt = (cx + r * angle.cos(), cy + r * angle.sin());
        if i == 0 {
            p.move_to(pt);
        } else {
            p.line_to(pt);
        }
    }
    p.close_path();
    p
}

/// Clone `run` with all glyph positions shifted by `(dx, dy)`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "glyph offsets stay inside the f32 position range"
)]
fn offset_run(run: &GlyphRun, dx: f64, dy: f64) -> GlyphRun {
    let mut r = run.clone();
    for g in &mut r.glyphs {
        g.x += dx as f32;
        g.y += dy as f32;
    }
    r
}

/// The distinct font blobs a set of glyph runs needs, in hash order.
fn font_blobs(ctx: &TextContext, runs: &[&[GlyphRun]]) -> Vec<Vec<u8>> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for group in runs {
        for r in *group {
            if seen.insert(r.font) {
                out.push(font_blob(ctx, r).clone());
            }
        }
    }
    out
}

/// xorshift64* — deterministic pseudo-random content without a `rand` dep.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform sample in `[0, 1)`.
    #[expect(
        clippy::cast_precision_loss,
        reason = "only the top 53 bits are sampled"
    )]
    fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform sample in `0..n`.
    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n as u64).unwrap_or(0)
    }
}

/// A small right-pointing chevron centred at `(x, y)`.
fn chevron(x: f64, y: f64) -> BezPath {
    let mut p = BezPath::new();
    p.move_to((x - 8.0, y - 12.0));
    p.line_to((x + 8.0, y));
    p.line_to((x - 8.0, y + 12.0));
    p.line_to((x - 4.0, y));
    p.close_path();
    p
}

/// Write `corpus` into `dir`, one `<name>/` per entry.
fn write_corpus(out: &Path, corpus: &Corpus) -> Result<(), SceneError> {
    std::fs::create_dir_all(out)?;
    for entry in &corpus.entries {
        let dir = out.join(&entry.name);
        entry.scene.save(&dir)?;
        for blob in &entry.blobs {
            Scene::store_resource(&dir, blob)?;
        }
    }
    Ok(())
}

/// One emitted scene plus the blobs its `resources/` needs.
struct Entry {
    name: String,
    scene: Scene,
    blobs: Vec<Vec<u8>>,
}

struct Corpus {
    entries: Vec<Entry>,
}

impl Corpus {
    const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Build a `w`x`h` scene via `f` and queue it for output.
    fn scene(
        &mut self,
        name: impl Into<String>,
        w: u32,
        h: u32,
        clear: Color,
        f: impl FnOnce(&mut LayerBuilder),
    ) {
        self.scene_with_blobs(name, w, h, clear, f, Vec::new());
    }

    fn scene_with_blobs(
        &mut self,
        name: impl Into<String>,
        w: u32,
        h: u32,
        clear: Color,
        f: impl FnOnce(&mut LayerBuilder),
        blobs: Vec<Vec<u8>>,
    ) {
        let mut builder = Scene::builder(w, h).clear(clear);
        {
            let mut root = builder.root();
            f(&mut root);
        }
        let scene = builder.build();
        self.entries.push(Entry {
            name: name.into(),
            scene,
            blobs,
        });
    }
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .without_time()
        .init();

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!(%e, "corpus generation failed");
            ExitCode::FAILURE
        }
    }
}

#[allow(clippy::too_many_lines)] // the corpus table is inherently long
fn run() -> Result<(), SceneError> {
    let root = corpus::repo_root();
    let out = corpus::corpus_dir(&root);
    let mut ctx = TextContext::new(&corpus::fonts_dir(&root))?;
    let mut corpus = Corpus::new();

    let white = srgb(0.95, 0.95, 0.95);
    let dark = srgb(0.1, 0.1, 0.12);

    // ---- Basic shapes ------------------------------------------------------

    corpus.scene("rect-plain", 96, 96, white, |l| {
        l.fill(
            Shape::rect(8.0, 8.0, 48.0, 48.0),
            solid(srgb(0.8, 0.1, 0.1)),
        );
        l.fill(
            Shape::rect(40.0, 40.0, 48.0, 48.0),
            solid(srgba(0.1, 0.4, 0.9, 0.7)),
        );
    });

    for (name, r) in [
        ("rounded-r00", 0.0),
        ("rounded-r08", 8.0),
        ("rounded-r24", 24.0),
        ("rounded-r32", 32.0),
    ] {
        corpus.scene(name, 96, 96, white, |l| {
            l.fill(
                Shape::rounded_rect(16.0, 16.0, 64.0, 64.0, r),
                solid(srgb(0.2, 0.6, 0.3)),
            );
        });
    }

    corpus.scene("rounded-mixed", 96, 96, white, |l| {
        let rr = RoundedRect::from_rect(
            Rect::new(12.0, 12.0, 84.0, 84.0),
            RoundedRectRadii::new(4.0, 16.0, 28.0, 40.0),
        );
        l.fill(Shape::RoundedRect(rr), solid(srgb(0.5, 0.2, 0.7)));
    });

    for (name, smoothing) in [
        ("continuous-s00", 0.0),
        ("continuous-s50", 0.5),
        ("continuous-s100", 1.0),
    ] {
        corpus.scene(name, 96, 96, white, |l| {
            l.fill(
                Shape::Continuous(cherenkov_scene::ContinuousRect::new(
                    Rect::new(16.0, 16.0, 80.0, 80.0),
                    28.0,
                    smoothing,
                )),
                solid(srgb(0.9, 0.4, 0.1)),
            );
        });
    }

    corpus.scene("circle", 96, 96, white, |l| {
        l.fill(Shape::circle(48.0, 48.0, 36.0), solid(srgb(0.1, 0.3, 0.8)));
        l.fill(
            Shape::circle(72.0, 30.0, 12.0),
            solid(srgba(1.0, 0.6, 0.0, 0.8)),
        );
    });

    corpus.scene("ellipse", 96, 96, white, |l| {
        l.fill(
            Shape::Ellipse(Ellipse::new((48.0, 48.0), (40.0, 24.0), 0.0)),
            solid(srgb(0.2, 0.5, 0.5)),
        );
    });

    corpus.scene("path-curved", 128, 128, white, |l| {
        l.fill(
            Shape::Path {
                path: curved_path(),
            },
            solid(srgb(0.3, 0.2, 0.7)),
        );
    });

    corpus.scene("path-self-intersect", 128, 128, white, |l| {
        l.fill(
            Shape::Path {
                path: self_intersecting_path(),
            },
            solid(srgb(0.8, 0.2, 0.4)),
        );
    });

    for (name, rule) in [
        ("path-nonzero", FillRule::NonZero),
        ("path-evenodd", FillRule::EvenOdd),
    ] {
        corpus.scene(name, 128, 128, white, |l| {
            let mut p = star_path(64.0, 64.0, 24.0, 56.0);
            p.extend(star_path(64.0, 64.0, 10.0, 24.0));
            l.fill_rule(Shape::Path { path: p }, rule, solid(srgb(0.1, 0.5, 0.6)));
        });
    }

    // ---- Strokes -----------------------------------------------------------

    corpus.scene("stroke-widths", 128, 128, white, |l| {
        for (i, w) in [1.0, 3.0, 8.0].iter().enumerate() {
            let y = (i as f64).mul_add(40.0, 24.0);
            l.stroke(
                Shape::Line(Line::new((12.0, y), (116.0, y))),
                StrokeStyle {
                    width: *w,
                    ..StrokeStyle::default()
                },
                solid(srgb(0.2, 0.2, 0.2)),
            );
        }
    });

    corpus.scene("stroke-joins", 128, 128, white, |l| {
        for (i, join) in [
            cherenkov_scene::kurbo::Join::Miter,
            cherenkov_scene::kurbo::Join::Round,
            cherenkov_scene::kurbo::Join::Bevel,
        ]
        .iter()
        .enumerate()
        {
            let y = (i as f64).mul_add(40.0, 20.0);
            let mut p = BezPath::new();
            p.move_to((16.0, y + 20.0));
            p.line_to((44.0, y));
            p.line_to((72.0, y + 20.0));
            p.line_to((100.0, y));
            l.stroke(
                Shape::Path { path: p },
                StrokeStyle {
                    width: 6.0,
                    join: *join,
                    ..StrokeStyle::default()
                },
                solid(srgb(0.6, 0.1, 0.1)),
            );
        }
    });

    corpus.scene("stroke-caps", 128, 128, white, |l| {
        for (i, cap) in [
            cherenkov_scene::kurbo::Cap::Butt,
            cherenkov_scene::kurbo::Cap::Round,
            cherenkov_scene::kurbo::Cap::Square,
        ]
        .iter()
        .enumerate()
        {
            let y = (i as f64).mul_add(36.0, 28.0);
            l.stroke(
                Shape::Line(Line::new((16.0, y), (112.0, y))),
                StrokeStyle {
                    width: 10.0,
                    start_cap: *cap,
                    end_cap: *cap,
                    ..StrokeStyle::default()
                },
                solid(srgb(0.1, 0.3, 0.6)),
            );
        }
    });

    corpus.scene("stroke-dash", 128, 128, white, |l| {
        l.stroke(
            Shape::Path {
                path: curved_path(),
            },
            StrokeStyle {
                width: 3.0,
                dash_pattern: vec![8.0, 4.0, 2.0, 4.0],
                dash_offset: 2.0,
                ..StrokeStyle::default()
            },
            solid(srgb(0.4, 0.1, 0.5)),
        );
    });

    corpus.scene("stroke-curve", 128, 128, white, |l| {
        l.stroke(
            Shape::rounded_rect(16.0, 16.0, 96.0, 96.0, 24.0),
            StrokeStyle {
                width: 5.0,
                ..StrokeStyle::default()
            },
            solid(srgb(0.1, 0.5, 0.3)),
        );
    });

    // ---- Gradients ---------------------------------------------------------

    let gradient_rect = Shape::rect(8.0, 8.0, 112.0, 112.0);
    for extends in [Extend::Pad, Extend::Repeat, Extend::Reflect, Extend::None] {
        let ename = match extends {
            Extend::Pad => "pad",
            Extend::Repeat => "repeat",
            Extend::Reflect => "reflect",
            Extend::None => "none",
        };
        for (n, stops) in [(2u8, stops2()), (8, stops8())] {
            corpus.scene(format!("grad-linear-{n}-{ename}"), 128, 128, white, |l| {
                l.fill(
                    gradient_rect.clone(),
                    Paint::Linear(LinearGradient {
                        start: Point::new(32.0, 48.0),
                        end: Point::new(96.0, 80.0),
                        stops: stops.clone(),
                        extend: extends,
                        interpolation: ColorSpace::Srgb,
                    }),
                );
            });
            corpus.scene(format!("grad-radial-{n}-{ename}"), 128, 128, white, |l| {
                l.fill(
                    gradient_rect.clone(),
                    Paint::Radial(RadialGradient {
                        center0: Point::new(64.0, 64.0),
                        r0: 8.0,
                        center1: Point::new(80.0, 72.0),
                        r1: 40.0,
                        stops: stops.clone(),
                        extend: extends,
                        interpolation: ColorSpace::Srgb,
                    }),
                );
            });
            corpus.scene(format!("grad-sweep-{n}-{ename}"), 128, 128, white, |l| {
                l.fill(
                    gradient_rect.clone(),
                    Paint::Sweep(SweepGradient {
                        center: Point::new(64.0, 64.0),
                        start_angle: 0.0,
                        end_angle: 1.6 * std::f64::consts::PI,
                        stops: stops.clone(),
                        extend: extends,
                        interpolation: ColorSpace::Srgb,
                    }),
                );
            });
        }
    }

    // ---- Images ------------------------------------------------------------

    let checker = checker_png();
    let grad_img = gradient_png();

    corpus.scene_with_blobs(
        "img-nearest",
        96,
        96,
        white,
        |l| {
            l.image(
                ResourceHash::of(&checker),
                Rect::new(16.0, 16.0, 80.0, 80.0),
                Sampling::Nearest,
            );
        },
        vec![checker.clone()],
    );

    corpus.scene_with_blobs(
        "img-bilinear",
        96,
        96,
        white,
        |l| {
            l.image(
                ResourceHash::of(&grad_img),
                Rect::new(12.0, 12.0, 84.0, 84.0),
                Sampling::Bilinear,
            );
        },
        vec![grad_img.clone()],
    );

    corpus.scene_with_blobs(
        "imgpattern-repeat",
        128,
        128,
        white,
        |l| {
            l.fill(
                Shape::rounded_rect(8.0, 8.0, 112.0, 112.0, 16.0),
                Paint::Image(ImagePaint {
                    image: ResourceHash::of(&checker),
                    transform: Affine::translate((40.0, 40.0)) * Affine::scale(4.0),
                    extend_x: Extend::Repeat,
                    extend_y: Extend::Repeat,
                    sampling: Sampling::Bilinear,
                }),
            );
        },
        vec![checker.clone()],
    );

    // ---- Clips, opacity, transforms ----------------------------------------

    corpus.scene("clip-nested", 128, 128, white, |l| {
        l.layer(|a| {
            a.clip(Shape::rounded_rect(8.0, 8.0, 112.0, 112.0, 24.0));
            a.fill(
                Shape::rect(0.0, 0.0, 128.0, 128.0),
                solid(srgb(0.9, 0.7, 0.1)),
            );
            a.layer(|b| {
                b.clip(Shape::rounded_rect(28.0, 28.0, 72.0, 72.0, 16.0));
                b.fill(
                    Shape::rect(0.0, 0.0, 128.0, 128.0),
                    solid(srgb(0.1, 0.3, 0.7)),
                );
                b.layer(|c| {
                    c.clip(Shape::circle(64.0, 64.0, 26.0));
                    c.fill(
                        Shape::rect(0.0, 0.0, 128.0, 128.0),
                        solid(srgb(0.9, 0.1, 0.2)),
                    );
                });
            });
        });
    });

    corpus.scene("clip-transform", 128, 128, white, |l| {
        l.layer(|a| {
            a.transform(Affine::rotate(0.4) * Affine::translate((-20.0, -10.0)));
            a.clip(Shape::rounded_rect(16.0, 16.0, 96.0, 96.0, 12.0));
            a.fill(
                Shape::rect(0.0, 0.0, 160.0, 160.0),
                solid(srgb(0.2, 0.6, 0.4)),
            );
        });
    });

    corpus.scene("group-opacity", 128, 128, white, |l| {
        l.fill(
            Shape::rect(0.0, 0.0, 128.0, 128.0),
            solid(srgb(0.9, 0.2, 0.2)),
        );
        l.layer(|a| {
            a.opacity(0.5);
            a.fill(Shape::circle(64.0, 64.0, 44.0), solid(srgb(0.1, 0.2, 0.8)));
        });
    });

    corpus.scene("transform-rotate", 128, 128, white, |l| {
        l.layer(|a| {
            a.transform(Affine::rotate_about(0.6, Point::new(64.0, 64.0)));
            a.fill(
                Shape::rounded_rect(24.0, 40.0, 80.0, 48.0, 10.0),
                solid(srgb(0.5, 0.2, 0.6)),
            );
            a.layer(|b| {
                b.transform(Affine::translate((8.0, 8.0)) * Affine::scale_non_uniform(1.0, 0.8));
                b.fill(
                    Shape::circle(64.0, 64.0, 20.0),
                    solid(srgba(0.1, 0.6, 0.8, 0.8)),
                );
            });
        });
    });

    // ---- Blend modes --------------------------------------------------------

    let modes = [
        ("multiply", BlendMode::Multiply),
        ("screen", BlendMode::Screen),
        ("overlay", BlendMode::Overlay),
        ("darken", BlendMode::Darken),
        ("lighten", BlendMode::Lighten),
        ("color-dodge", BlendMode::ColorDodge),
        ("color-burn", BlendMode::ColorBurn),
        ("hard-light", BlendMode::HardLight),
        ("soft-light", BlendMode::SoftLight),
        ("difference", BlendMode::Difference),
        ("exclusion", BlendMode::Exclusion),
        ("hue", BlendMode::Hue),
        ("saturation", BlendMode::Saturation),
        ("color", BlendMode::Color),
        ("luminosity", BlendMode::Luminosity),
        ("normal", BlendMode::Normal),
    ];
    for (mname, mode) in modes {
        corpus.scene(format!("blend-{mname}"), 96, 96, srgb(0.7, 0.5, 0.2), |l| {
            l.fill(
                Shape::rect(0.0, 0.0, 96.0, 96.0),
                Paint::Linear(LinearGradient {
                    start: Point::new(0.0, 0.0),
                    end: Point::new(96.0, 96.0),
                    stops: vec![
                        GradientStop {
                            offset: 0.0,
                            color: srgb(0.9, 0.5, 0.1),
                        },
                        GradientStop {
                            offset: 1.0,
                            color: srgb(0.1, 0.3, 0.8),
                        },
                    ],
                    extend: Extend::Pad,
                    interpolation: ColorSpace::Srgb,
                }),
            );
            l.layer(|a| {
                a.blend(mode);
                a.fill(
                    Shape::circle(48.0, 48.0, 34.0),
                    Paint::Linear(LinearGradient {
                        start: Point::new(14.0, 14.0),
                        end: Point::new(82.0, 82.0),
                        stops: vec![
                            GradientStop {
                                offset: 0.0,
                                color: srgb(0.2, 0.9, 0.4),
                            },
                            GradientStop {
                                offset: 1.0,
                                color: srgb(0.9, 0.2, 0.6),
                            },
                        ],
                        extend: Extend::Pad,
                        interpolation: ColorSpace::Srgb,
                    }),
                );
            });
        });
    }

    // ---- Shadows ------------------------------------------------------------

    for sigma in [0.0, 2.0, 5.0, 10.0, 20.0] {
        corpus.scene(format!("shadow-s{sigma:03.0}"), 128, 128, dark, |l| {
            l.shadow(
                Shape::rounded_rect(32.0, 32.0, 64.0, 64.0, 12.0),
                sigma,
                [0.0, 0.0],
                srgba(0.0, 0.0, 0.0, 0.8),
            );
            l.fill(
                Shape::rounded_rect(32.0, 32.0, 64.0, 64.0, 12.0),
                solid(srgb(0.95, 0.9, 0.3)),
            );
        });
    }

    corpus.scene("shadow-offset", 128, 128, white, |l| {
        l.shadow(
            Shape::circle(56.0, 56.0, 30.0),
            4.0,
            [10.0, 14.0],
            srgba(0.0, 0.0, 0.2, 0.7),
        );
        l.fill(Shape::circle(56.0, 56.0, 30.0), solid(srgb(0.2, 0.5, 0.9)));
    });

    // ---- HDR -----------------------------------------------------------------

    corpus.scene(
        "hdr-bright",
        96,
        96,
        Color::new(ColorSpace::LinearSrgb, [0.02, 0.02, 0.02, 1.0]),
        |l| {
            l.fill(
                Shape::rect(8.0, 8.0, 80.0, 80.0),
                Paint::Linear(LinearGradient {
                    start: Point::new(8.0, 48.0),
                    end: Point::new(88.0, 48.0),
                    stops: vec![
                        GradientStop {
                            offset: 0.0,
                            color: Color::new(ColorSpace::LinearSrgb, [0.4, 0.2, 0.1, 1.0]),
                        },
                        GradientStop {
                            offset: 0.5,
                            color: Color::new(ColorSpace::LinearSrgb, [2.5, 1.8, 0.4, 1.0]),
                        },
                        GradientStop {
                            offset: 1.0,
                            color: Color::new(ColorSpace::Rec2020, [1.5, 0.2, 1.2, 1.0]),
                        },
                    ],
                    extend: Extend::Pad,
                    interpolation: ColorSpace::LinearSrgb,
                }),
            );
            l.fill(
                Shape::circle(48.0, 48.0, 22.0),
                solid(Color::new(ColorSpace::DisplayP3, [1.0, 0.4, 0.0, 1.0])),
            );
        },
    );

    // ---- Text ----------------------------------------------------------------

    let text_specs: &[(&str, &str, &str, f32)] = &[
        ("NotoSans.ttf", "text-latin", corpus::LATIN, 30.0),
        ("NotoSansSC.ttf", "text-cjk", corpus::CJK, 30.0),
        ("NotoSansArabic.ttf", "text-arabic", corpus::ARABIC, 34.0),
        ("NotoSansHebrew.ttf", "text-hebrew", corpus::HEBREW, 34.0),
        (
            "NotoSansDevanagari.ttf",
            "text-devanagari",
            corpus::DEVANAGARI,
            34.0,
        ),
        ("NotoSansThai.ttf", "text-thai", corpus::THAI, 30.0),
        ("NotoEmoji.ttf", "text-emoji", corpus::EMOJI, 48.0),
        ("Nabla.ttf", "text-colr", corpus::COLR, 64.0),
    ];
    for (file, name, text, size) in text_specs {
        let runs = ctx.shape(file, text, *size, FontWeight::NORMAL, &solid(dark));
        let blobs: Vec<Vec<u8>> = runs.iter().map(|r| font_blob(&ctx, r).clone()).collect();
        corpus.scene_with_blobs(
            *name,
            320,
            160,
            white,
            |l| {
                for run in &runs {
                    l.glyphs(run.clone());
                }
            },
            blobs,
        );
    }

    // A variable-weight Latin line: the font is a `wdth,wght` variable font, so
    // a bold weight lands as non-zero normalized coords.
    {
        let text = "AaBbGg 0123456789";
        let runs = ctx.shape("NotoSans.ttf", text, 40.0, FontWeight::BOLD, &solid(dark));
        let blobs: Vec<Vec<u8>> = runs.iter().map(|r| font_blob(&ctx, r).clone()).collect();
        corpus.scene_with_blobs(
            "text-weight",
            320,
            128,
            white,
            |l| {
                for run in &runs {
                    l.glyphs(run.clone());
                }
            },
            blobs,
        );
    }

    // ---- Motion and scrolling ----------------------------------------------
    //
    // Scenes exercising `Layer::scroll_offset` and `Layer::motion`. The
    // oracle renders the settled state; the cherenkov adapters commit the
    // `from` state then the animation/decay, and `render --readback`
    // comparisons run until `Next::Idle`.

    // A card that springs in from above the viewport (Spring 0.5/1.0).
    corpus.scene("anim-spring-card", 256, 256, srgb(0.94, 0.95, 0.98), |l| {
        l.layer(|card| {
            card.transform(Affine::translate((0.0, 0.0)));
            card.motion(Motion::Transform {
                from: Affine::translate((0.0, -200.0)),
                animation: MotionAnimation::Spring {
                    response: 0.5,
                    damping: 1.0,
                },
            });
            let rect = RoundedRect::from_rect(
                Rect::new(32.0, 96.0, 224.0, 208.0),
                RoundedRectRadii::new(16.0, 16.0, 16.0, 16.0),
            );
            card.shadow(
                Shape::RoundedRect(rect),
                8.0,
                [0.0, 6.0],
                srgba(0.0, 0.0, 0.0, 0.25),
            );
            card.fill(Shape::RoundedRect(rect), solid(white));
            card.fill(
                Shape::RoundedRect(RoundedRect::from_rect(
                    Rect::new(48.0, 120.0, 208.0, 140.0),
                    RoundedRectRadii::new(6.0, 6.0, 6.0, 6.0),
                )),
                solid(srgb(0.35, 0.45, 0.85)),
            );
            card.fill(
                Shape::RoundedRect(RoundedRect::from_rect(
                    Rect::new(48.0, 152.0, 168.0, 164.0),
                    RoundedRectRadii::new(4.0, 4.0, 4.0, 4.0),
                )),
                solid(srgb(0.8, 0.82, 0.88)),
            );
            card.fill(
                Shape::RoundedRect(RoundedRect::from_rect(
                    Rect::new(48.0, 174.0, 190.0, 186.0),
                    RoundedRectRadii::new(4.0, 4.0, 4.0, 4.0),
                )),
                solid(srgb(0.8, 0.82, 0.88)),
            );
        });
    });

    // A panel sliding in on a 400 ms ease-in-out curve.
    corpus.scene("anim-curve-slide", 256, 256, srgb(0.92, 0.94, 0.96), |l| {
        l.layer(|panel| {
            panel.transform(Affine::translate((0.0, 0.0)));
            panel.motion(Motion::Transform {
                from: Affine::translate((-180.0, 0.0)),
                animation: MotionAnimation::Curve {
                    duration_ms: 400,
                    x1: 0.42,
                    y1: 0.0,
                    x2: 0.58,
                    y2: 1.0,
                },
            });
            panel.fill(
                Shape::RoundedRect(RoundedRect::from_rect(
                    Rect::new(24.0, 64.0, 232.0, 200.0),
                    RoundedRectRadii::new(12.0, 12.0, 12.0, 12.0),
                )),
                solid(srgb(0.22, 0.5, 0.6)),
            );
            for i in 0u8..4 {
                let y = 88.0 + f64::from(i) * 28.0;
                panel.fill(
                    Shape::RoundedRect(RoundedRect::from_rect(
                        Rect::new(44.0, y, 44.0 + 150.0 - 22.0 * f64::from(i), y + 12.0),
                        RoundedRectRadii::new(4.0, 4.0, 4.0, 4.0),
                    )),
                    solid(srgba(1.0, 1.0, 1.0, 0.75)),
                );
            }
        });
    });

    // A clipped list scrolled to a static offset: rows 3.. are visible.
    corpus.scene("scroll-static", 256, 192, srgb(0.97, 0.97, 0.98), |l| {
        l.layer(|list| {
            list.clip(Shape::Rect(Rect::new(8.0, 16.0, 248.0, 176.0)));
            list.scroll_offset(Vec2::new(0.0, 96.0));
            for i in 0u8..12 {
                let y = f64::from(i) * 48.0;
                list.fill(
                    Shape::RoundedRect(RoundedRect::from_rect(
                        Rect::new(16.0, y, 240.0, y + 40.0),
                        RoundedRectRadii::new(8.0, 8.0, 8.0, 8.0),
                    )),
                    solid(srgb(
                        0.3 + 0.05 * f32::from(i % 4),
                        0.5,
                        0.85 - 0.04 * f32::from(i % 4),
                    )),
                );
            }
        });
    });

    // A 60-row list in a clip: a fling decaying from below the rest
    // position. from = rest - v/k with v = (0, -1800), k = 4.
    corpus.scene("scroll-decay", 256, 320, srgb(0.97, 0.97, 0.98), |l| {
        l.layer(|list| {
            list.clip(Shape::Rect(Rect::new(8.0, 16.0, 248.0, 304.0)));
            list.scroll_offset(Vec2::new(0.0, 600.0));
            list.motion(Motion::Scroll {
                from: Vec2::new(0.0, 1050.0),
                velocity: Vec2::new(0.0, -1800.0),
                deceleration: 4.0,
                bounds: None,
            });
            for i in 0u8..60 {
                let y = f64::from(i) * 48.0;
                list.fill(
                    Shape::RoundedRect(RoundedRect::from_rect(
                        Rect::new(16.0, y, 240.0, y + 40.0),
                        RoundedRectRadii::new(8.0, 8.0, 8.0, 8.0),
                    )),
                    Paint::Linear(LinearGradient {
                        start: Point::new(16.0, y),
                        end: Point::new(240.0, y),
                        extend: Extend::Pad,
                        interpolation: ColorSpace::Srgb,
                        stops: vec![
                            GradientStop {
                                offset: 0.0,
                                color: srgb(
                                    0.25 + 0.01 * f32::from(i % 20),
                                    0.5,
                                    0.8 - 0.02 * f32::from(i % 10),
                                ),
                            },
                            GradientStop {
                                offset: 1.0,
                                color: srgb(0.6, 0.7, 0.9),
                            },
                        ],
                    }),
                );
            }
        });
    });

    // The same list, flung hard enough to overshoot its bounds and
    // rubber-band back; rest = the bound = the static scroll_offset.
    corpus.scene(
        "scroll-rubber-band",
        256,
        320,
        srgb(0.97, 0.97, 0.98),
        |l| {
            l.layer(|list| {
                list.clip(Shape::Rect(Rect::new(8.0, 16.0, 248.0, 304.0)));
                list.scroll_offset(Vec2::new(0.0, 300.0));
                list.motion(Motion::Scroll {
                    from: Vec2::new(0.0, 100.0),
                    velocity: Vec2::new(0.0, 2000.0),
                    deceleration: 4.0,
                    bounds: Some(Rect::new(0.0, 0.0, 0.0, 300.0)),
                });
                for i in 0u8..60 {
                    let y = f64::from(i) * 48.0;
                    list.fill(
                        Shape::RoundedRect(RoundedRect::from_rect(
                            Rect::new(16.0, y, 240.0, y + 40.0),
                            RoundedRectRadii::new(8.0, 8.0, 8.0, 8.0),
                        )),
                        Paint::Linear(LinearGradient {
                            start: Point::new(16.0, y),
                            end: Point::new(240.0, y),
                            extend: Extend::Pad,
                            interpolation: ColorSpace::Srgb,
                            stops: vec![
                                GradientStop {
                                    offset: 0.0,
                                    color: srgb(0.75, 0.55, 0.85),
                                },
                                GradientStop {
                                    offset: 1.0,
                                    color: srgb(0.5, 0.35 + 0.01 * f32::from(i % 20), 0.75),
                                },
                            ],
                        }),
                    );
                }
            });
        },
    );

    // ---- Performance set ---------------------------------------------------
    //
    // Full-resolution scenes (the Pixel 9 Pro viewport, 1024x2216) modelling
    // real app content — a performance set for `measure`, separate from the
    // per-primitive correctness sweeps above. Deterministic throughout; the
    // only pseudo-randomness is the seeded xorshift in the map scene.

    let (pw, ph) = (1024.0_f64, 2216.0_f64);
    let mut perf = Corpus::new();

    // Scrolling UI list: 30 rows, each a shadowed rounded card with an
    // avatar, two text runs and a chevron icon path.
    {
        let title = ctx.shape(
            "NotoSans.ttf",
            "Message from Ada",
            30.0,
            FontWeight::NORMAL,
            &solid(dark),
        );
        let sub = ctx.shape(
            "NotoSans.ttf",
            "See you at the renderer sync-up",
            22.0,
            FontWeight::NORMAL,
            &solid(srgb(0.4, 0.42, 0.45)),
        );
        let blobs = font_blobs(&ctx, &[&title, &sub]);
        perf.scene_with_blobs(
            "ui-list",
            pw as u32,
            ph as u32,
            srgb(0.96, 0.96, 0.97),
            |l| {
                let pitch = 72.0;
                for i in 0u8..30 {
                    let y = 24.0 + f64::from(i) * pitch;
                    let card = RoundedRect::from_rect(
                        Rect::new(24.0, y, 1000.0, y + 64.0),
                        RoundedRectRadii::new(14.0, 14.0, 14.0, 14.0),
                    );
                    l.shadow(
                        Shape::RoundedRect(card),
                        4.0,
                        [0.0, 3.0],
                        srgba(0.0, 0.0, 0.0, 0.22),
                    );
                    l.fill(Shape::RoundedRect(card), solid(white));
                    l.fill(
                        Shape::circle(60.0, y + 32.0, 20.0),
                        solid(srgb(0.3 + 0.02 * f32::from(i), 0.4, 0.8)),
                    );
                    for run in &title {
                        l.glyphs(offset_run(run, 96.0, y + 28.0));
                    }
                    for run in &sub {
                        l.glyphs(offset_run(run, 96.0, y + 54.0));
                    }
                    l.fill(
                        Shape::Path {
                            path: chevron(960.0, y + 32.0),
                        },
                        solid(srgb(0.6, 0.6, 0.62)),
                    );
                }
            },
            blobs,
        );
    }

    // Dense multi-script text page: Latin, CJK, Arabic, Devanagari and emoji
    // lines cycling down the full page height.
    {
        let scripts = [
            ("NotoSans.ttf", corpus::LATIN, 34.0_f32),
            ("NotoSansSC.ttf", corpus::CJK, 34.0),
            ("NotoSansArabic.ttf", corpus::ARABIC, 36.0),
            ("NotoSansDevanagari.ttf", corpus::DEVANAGARI, 34.0),
            ("NotoEmoji.ttf", corpus::EMOJI, 34.0),
        ];
        let shaped: Vec<Vec<GlyphRun>> = scripts
            .iter()
            .map(|(file, text, size)| {
                ctx.shape(file, text, *size, FontWeight::NORMAL, &solid(dark))
            })
            .collect();
        let blobs = font_blobs(&ctx, &shaped.iter().map(Vec::as_slice).collect::<Vec<_>>());
        perf.scene_with_blobs(
            "text-page",
            pw as u32,
            ph as u32,
            white,
            |l| {
                let pitch = 54.0;
                for (i, runs) in (0u8..38).map(|i| (i, &shaped[usize::from(i) % shaped.len()])) {
                    let y = 56.0 + f64::from(i) * pitch;
                    for run in runs {
                        l.glyphs(offset_run(run, 24.0 - f64::from(TEXT_PAD), y));
                    }
                }
            },
            blobs,
        );
    }

    // Map-like page: ~2,000 stroked and filled paths — short segments,
    // closed polygons and curved outlines distributed over the viewport.
    {
        perf.scene("map", pw as u32, ph as u32, srgb(0.93, 0.95, 0.90), |l| {
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
            let palette = [
                srgb(0.45, 0.60, 0.45),
                srgb(0.55, 0.50, 0.38),
                srgb(0.40, 0.55, 0.65),
                srgb(0.70, 0.55, 0.45),
                srgb(0.50, 0.45, 0.60),
            ];
            let thin = StrokeStyle {
                start_cap: kurbo::Cap::Round,
                end_cap: kurbo::Cap::Round,
                ..StrokeStyle::default()
            };
            for i in 0..2000 {
                let (x, y) = (rng.f64() * pw, rng.f64() * ph);
                let color = palette[rng.below(palette.len())];
                match i % 4 {
                    // Polyline segment.
                    0 => {
                        let (dx, dy) = (rng.f64() * 90.0 - 45.0, rng.f64() * 90.0 - 45.0);
                        l.stroke(
                            Shape::Line(Line::new((x, y), (x + dx, y + dy))),
                            StrokeStyle {
                                width: 0.5 + rng.f64() * 4.0,
                                ..thin.clone()
                            },
                            solid(color),
                        );
                    }
                    // Closed polygon (park/parcel fill).
                    1 => {
                        let r = 8.0 + rng.f64() * 48.0;
                        let sides = 3 + rng.below(5);
                        let mut p = BezPath::new();
                        for v in 0..sides {
                            #[expect(
                                clippy::cast_precision_loss,
                                reason = "vertex index is below 8"
                            )]
                            let a = (v as f64) * std::f64::consts::TAU / (sides as f64);
                            let pt = (x + r * a.cos(), y + r * a.sin());
                            if v == 0 {
                                p.move_to(pt);
                            } else {
                                p.line_to(pt);
                            }
                        }
                        p.close_path();
                        l.fill(Shape::Path { path: p }, solid(color));
                    }
                    // Open polyline of 3-6 points (road/river line).
                    2 => {
                        let mut p = BezPath::new();
                        let (mut cx, mut cy) = (x, y);
                        p.move_to((cx, cy));
                        for _ in 0..2 + rng.below(4) {
                            cx += rng.f64() * 120.0 - 60.0;
                            cy += rng.f64() * 60.0 - 30.0;
                            p.line_to((cx, cy));
                        }
                        l.stroke(
                            Shape::Path { path: p },
                            StrokeStyle {
                                width: 1.0 + rng.f64() * 5.0,
                                ..thin.clone()
                            },
                            solid(color),
                        );
                    }
                    // Curved closed path (lake/contour).
                    _ => {
                        let (rx, ry) = (10.0 + rng.f64() * 60.0, 6.0 + rng.f64() * 40.0);
                        l.fill(
                            Shape::Ellipse(Ellipse::new((x, y), (rx, ry), 0.0)),
                            solid(color),
                        );
                    }
                }
            }
        });
    }

    // Chart page: a 2,000-point line chart, a bar chart, axes and labels.
    {
        let label = ctx.shape(
            "NotoSans.ttf",
            "Frame time (ms)",
            28.0,
            FontWeight::NORMAL,
            &solid(dark),
        );
        let tick = ctx.shape(
            "NotoSans.ttf",
            "250",
            24.0,
            FontWeight::NORMAL,
            &solid(dark),
        );
        let blobs = font_blobs(&ctx, &[&label, &tick]);
        perf.scene_with_blobs(
            "chart",
            pw as u32,
            ph as u32,
            white,
            |l| {
                // Axes + grid.
                let axis = StrokeStyle {
                    width: 2.0,
                    ..StrokeStyle::default()
                };
                l.stroke(
                    Shape::Line(Line::new((80.0, 100.0), (80.0, 1400.0))),
                    axis.clone(),
                    solid(dark),
                );
                l.stroke(
                    Shape::Line(Line::new((80.0, 1400.0), (1000.0, 1400.0))),
                    axis,
                    solid(dark),
                );
                let grid = StrokeStyle {
                    width: 0.5,
                    ..StrokeStyle::default()
                };
                for g in 1u8..7 {
                    let gy = 1400.0 - f64::from(g) * 200.0;
                    l.stroke(
                        Shape::Line(Line::new((80.0, gy), (1000.0, gy))),
                        grid.clone(),
                        solid(srgb(0.85, 0.85, 0.85)),
                    );
                    for run in &tick {
                        l.glyphs(offset_run(run, 28.0, gy + 8.0));
                    }
                }
                // Line chart: 2,000 points of a deterministic waveform.
                let mut line = BezPath::new();
                for i in 0u16..2000 {
                    let x = 80.0 + f64::from(i) * (920.0 / 1999.0);
                    let t = f64::from(i) * 0.01;
                    let y = 750.0
                        - 320.0 * (0.5 + 0.3 * t.sin() + 0.2 * (3.1 * t).cos())
                        - 40.0 * (t * 17.3).sin();
                    if i == 0 {
                        line.move_to((x, y));
                    } else {
                        line.line_to((x, y));
                    }
                }
                l.stroke(
                    Shape::Path { path: line },
                    StrokeStyle {
                        width: 2.5,
                        ..StrokeStyle::default()
                    },
                    solid(srgb(0.15, 0.45, 0.85)),
                );
                for run in &label {
                    l.glyphs(offset_run(run, 80.0, 60.0));
                }
                // Bar chart: 44 bars.
                let bw = (920.0 - 40.0 * 8.0) / 44.0;
                for i in 0u8..44 {
                    let h = 120.0 + 380.0 * (f64::from(i) * 0.37).sin().mul_add(0.5, 0.5);
                    let x = 90.0 + f64::from(i) * (bw + 8.0);
                    l.fill(
                        Shape::Rect(Rect::new(x, 2050.0 - h, x + bw, 2050.0)),
                        solid(srgb(0.85, 0.45, 0.25)),
                    );
                }
            },
            blobs,
        );
    }

    // Effects page: 20 shadowed cards under a group-opacity layer.
    {
        perf.scene(
            "effects",
            pw as u32,
            ph as u32,
            srgb(0.98, 0.97, 0.96),
            |l| {
                l.layer(|group| {
                    group.opacity(0.82);
                    for i in 0u8..20 {
                        let (col, row) = (i % 4, i / 4);
                        let (x, y) = (32.0 + f64::from(col) * 250.0, 48.0 + f64::from(row) * 420.0);
                        let card = RoundedRect::from_rect(
                            Rect::new(x, y, x + 226.0, y + 380.0),
                            RoundedRectRadii::new(20.0, 20.0, 20.0, 20.0),
                        );
                        group.shadow(
                            Shape::RoundedRect(card),
                            12.0,
                            [0.0, 10.0],
                            srgba(0.15, 0.1, 0.3, 0.35),
                        );
                        group.fill(
                            Shape::RoundedRect(card),
                            solid(srgb(
                                0.55 + 0.05 * f32::from(col),
                                0.35 + 0.03 * f32::from(row),
                                0.75,
                            )),
                        );
                        group.fill(
                            Shape::RoundedRect(RoundedRect::from_rect(
                                Rect::new(x + 20.0, y + 24.0, x + 206.0, y + 120.0),
                                RoundedRectRadii::new(10.0, 10.0, 10.0, 10.0),
                            )),
                            solid(srgba(1.0, 1.0, 1.0, 0.6)),
                        );
                    }
                });
            },
        );
    }

    // A heavy scrolling list: 120 shadowed card rows inside a clipped,
    // scroll-decaying layer — the frame-time scene for `measure` while a
    // scroll is animating.
    {
        let title = ctx.shape(
            "NotoSans.ttf",
            "Inbox message subject",
            28.0,
            FontWeight::NORMAL,
            &solid(dark),
        );
        let sub = ctx.shape(
            "NotoSans.ttf",
            "A short preview line of the message body",
            20.0,
            FontWeight::NORMAL,
            &solid(srgb(0.4, 0.42, 0.45)),
        );
        let blobs = font_blobs(&ctx, &[&title, &sub]);
        perf.scene_with_blobs(
            "scroll-list",
            pw as u32,
            ph as u32,
            srgb(0.96, 0.96, 0.97),
            |l| {
                l.layer(|list| {
                    list.clip(Shape::Rect(Rect::new(0.0, 0.0, pw, ph)));
                    list.scroll_offset(Vec2::new(0.0, 1200.0));
                    list.motion(Motion::Scroll {
                        from: Vec2::new(0.0, 1700.0),
                        velocity: Vec2::new(0.0, -2000.0),
                        deceleration: 4.0,
                        bounds: Some(Rect::new(0.0, 0.0, 0.0, 120.0 * 96.0 - ph)),
                    });
                    let pitch = 96.0;
                    for i in 0u16..120 {
                        let y = 24.0 + f64::from(i) * pitch;
                        let card = RoundedRect::from_rect(
                            Rect::new(24.0, y, 1000.0, y + 88.0),
                            RoundedRectRadii::new(14.0, 14.0, 14.0, 14.0),
                        );
                        list.shadow(
                            Shape::RoundedRect(card),
                            4.0,
                            [0.0, 3.0],
                            srgba(0.0, 0.0, 0.0, 0.22),
                        );
                        list.fill(Shape::RoundedRect(card), solid(white));
                        list.fill(
                            Shape::circle(64.0, y + 44.0, 24.0),
                            solid(srgb(0.3 + 0.02 * f32::from(i % 16), 0.4, 0.8)),
                        );
                        for run in &title {
                            list.glyphs(offset_run(run, 112.0, y + 40.0));
                        }
                        for run in &sub {
                            list.glyphs(offset_run(run, 112.0, y + 72.0));
                        }
                    }
                });
            },
            blobs,
        );
    }

    // A text-heavy dashboard whose per-frame values ride engine slot
    // updates: one glyph run (a two-digit counter) and one bar of the
    // chart change every frame; everything else is static.
    {
        let title = ctx.shape(
            "NotoSans.ttf",
            "Dashboard",
            40.0,
            FontWeight::BOLD,
            &solid(dark),
        );
        let body = ctx.shape(
            "NotoSans.ttf",
            "Revenue, signups and latency at a glance for the last quarter.",
            22.0,
            FontWeight::NORMAL,
            &solid(srgb(0.4, 0.42, 0.45)),
        );
        let label = ctx.shape(
            "NotoSans.ttf",
            "Active users",
            20.0,
            FontWeight::NORMAL,
            &solid(dark),
        );
        // Same glyph count every frame (two digits) so the run stays a
        // value slot update.
        let digits: Vec<Vec<GlyphRun>> = (0u8..60)
            .map(|n| {
                ctx.shape(
                    "NotoSans.ttf",
                    &format!("{n:02}"),
                    48.0,
                    FontWeight::BOLD,
                    &solid(srgb(0.1, 0.4, 0.9)),
                )
            })
            .collect();
        let blobs = font_blobs(&ctx, &[&title, &body, &label, &digits[0]]);
        perf.scene_with_blobs(
            "live-dashboard",
            pw as u32,
            ph as u32,
            srgb(0.96, 0.96, 0.97),
            |l| {
                for run in &title {
                    l.glyphs(offset_run(run, 40.0, 40.0));
                }
                for (i, run) in body.iter().enumerate() {
                    for line in 0..3 {
                        l.glyphs(offset_run(run, 40.0, 140.0 + 34.0 * (3 * i + line) as f64));
                    }
                }
                // Card grid: 4 columns x 3 rows of shadowed cards.
                for row in 0u8..3 {
                    for col in 0u8..4 {
                        let x = 40.0 + f64::from(col) * 246.0;
                        let y = 280.0 + f64::from(row) * 220.0;
                        let card = RoundedRect::from_rect(
                            Rect::new(x, y, x + 226.0, y + 200.0),
                            RoundedRectRadii::new(14.0, 14.0, 14.0, 14.0),
                        );
                        l.shadow(
                            Shape::RoundedRect(card),
                            4.0,
                            [0.0, 3.0],
                            srgba(0.0, 0.0, 0.0, 0.22),
                        );
                        l.fill(Shape::RoundedRect(card), solid(white));
                        l.stroke(
                            Shape::RoundedRect(card),
                            StrokeStyle {
                                width: 1.0,
                                ..StrokeStyle::default()
                            },
                            solid(srgba(0.0, 0.0, 0.0, 0.12)),
                        );
                        for run in &label {
                            l.glyphs(offset_run(run, x + 20.0, y + 24.0));
                        }
                    }
                }
                // Bar chart: 24 bars inside a framed plot.
                let chart_top = 1000.0;
                let chart_h = 300.0;
                l.stroke(
                    Shape::Rect(Rect::new(40.0, chart_top, 984.0, chart_top + chart_h)),
                    StrokeStyle {
                        width: 1.0,
                        ..StrokeStyle::default()
                    },
                    solid(srgba(0.0, 0.0, 0.0, 0.25)),
                );
                let bar_paint = solid(srgb(0.2, 0.5, 0.9));
                for i in 0u8..24 {
                    let x = 56.0 + f64::from(i) * 39.0;
                    let h = 40.0 + f64::from((i * 37) % 200);
                    l.fill(
                        Shape::Rect(Rect::new(x, chart_top + chart_h - h, x + 28.0, chart_top + chart_h)),
                        bar_paint.clone(),
                    );
                }
                // Live counter, in the same content layer: the digits of
                // "active users" ticking 00..59.
                let counter = l.item_count();
                for run in &digits[0] {
                    l.glyphs(offset_run(run, 60.0, 240.0));
                }
                l.live(Live {
                    item: counter,
                    frames: digits
                        .iter()
                        .map(|runs| Draw::Glyphs(offset_run(&runs[0], 60.0, 240.0)))
                        .collect(),
                });
                // Live bar: the last bar pulses every frame.
                let last = l.item_count();
                let bx = 56.0 + 23.0 * 39.0;
                l.fill(
                    Shape::Rect(Rect::new(bx + 39.0, chart_top + chart_h - 120.0, bx + 39.0 + 28.0, chart_top + chart_h)),
                    bar_paint.clone(),
                );
                l.live(Live {
                    item: last,
                    frames: (0u8..60)
                        .map(|n| {
                            let h = 40.0 + f64::from(n) * 3.0;
                            Draw::Fill {
                                shape: Shape::Rect(Rect::new(
                                    bx + 39.0,
                                    chart_top + chart_h - h,
                                    bx + 39.0 + 28.0,
                                    chart_top + chart_h,
                                )),
                                rule: FillRule::NonZero,
                                paint: bar_paint.clone(),
                            }
                        })
                        .collect(),
                });
            },
            blobs,
        );
    }

    // ---- Write out ---------------------------------------------------------

    let perf_out = corpus::perf_dir(&root);
    write_corpus(&out, &corpus)?;
    write_corpus(&perf_out, &perf)?;
    tracing::info!(count = corpus.entries.len(), dir = %out.display(), "corpus written");
    tracing::info!(count = perf.entries.len(), dir = %perf_out.display(), "perf set written");
    Ok(())
}
