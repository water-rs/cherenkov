// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Native bitmap colour-font rendering and cache behavior.

use cherenkov::kurbo::{Affine, Rect};
use cherenkov::{
    Draw, Engine, EngineError, Extend, FontId, FontSource, FrameTime, Glyph, GlyphRun, GlyphStyle,
    ImageColorSpace, ImageData, ImageId, ImagePattern, Offscreen, OffscreenFormat, Paint, Rgba8,
    Sampling, WorkingColor,
};
use cherenkov_gpu::{Gpu, GpuConfig};
use skrifa::MetadataProvider;
use skrifa::bitmap::{BitmapData, BitmapFormat, BitmapStrikes, Origin};
use skrifa::raw::TableProvider;

const CBDT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../scenes/fonts/NotoColorEmojiSubset.ttf"
);
const SBIX_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../scenes/fonts/CherenkovSbixTest.ttf"
);
const OUTLINE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../scenes/fonts/NotoSans.ttf");

fn engine() -> Option<Engine<Gpu>> {
    match Engine::<Gpu>::new(GpuConfig::default()) {
        Ok(engine) => Some(engine),
        Err(EngineError::Backend(_)) => None,
        Err(error) => panic!("GPU initialization failed: {error}"),
    }
}

fn glyph_id(bytes: &[u8], character: char) -> u32 {
    skrifa::FontRef::from_index(bytes, 0)
        .expect("font")
        .charmap()
        .map(character)
        .expect("fixture glyph")
        .to_u32()
}

fn glyph_run(font: FontId, id: u32, size: f32) -> GlyphRun {
    GlyphRun {
        font,
        size,
        coords: Vec::new(),
        glyphs: vec![Glyph {
            id,
            x: 24.0,
            y: 112.0,
            transform: None,
        }],
        style: GlyphStyle::Fill,
    }
}

fn unregistered_image_error(bytes: &[u8], character: char) -> Option<String> {
    let engine = engine()?;
    let font = engine
        .font(FontSource::bytes(bytes.to_vec()))
        .expect("register font");
    let run = glyph_run(font.id(), glyph_id(bytes, character), 48.0);
    let surface = engine
        .surface(Offscreen::new((160, 120), OffscreenFormat::LinearF16))
        .expect("surface");
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.glyphs(
                run,
                Paint::Image(ImagePattern {
                    image: ImageId::new(u64::MAX),
                    transform: Affine::IDENTITY,
                    extend_x: Extend::Pad,
                    extend_y: Extend::Pad,
                    sampling: Sampling::Linear,
                }),
            );
        }));
    });
    match engine.render(FrameTime::now()) {
        Err(cherenkov::RenderError::Image(message)) => Some(message),
        Err(error) => panic!("expected unregistered-image error, got {error}"),
        Ok(_) => panic!("unregistered-image paint unexpectedly rendered"),
    }
}

fn image_source(
    bytes: &[u8],
    format: BitmapFormat,
    glyph_id: u32,
    ppem: f32,
    size: f32,
) -> (ImageData<Rgba8>, Rect) {
    let font = skrifa::FontRef::from_index(bytes, 0).expect("font");
    let strikes = BitmapStrikes::with_format(&font, format).expect("bitmap strikes");
    let strike = strikes
        .iter()
        .find(|strike| strike.ppem().to_bits() == ppem.to_bits())
        .expect("selected strike");
    let glyph = strike
        .get(skrifa::GlyphId::new(glyph_id))
        .expect("bitmap glyph");
    let BitmapData::Png(png_bytes) = &glyph.data else {
        panic!("fixture uses PNG payloads");
    };
    let mut decoder = png::Decoder::new(std::io::Cursor::new(png_bytes.as_ref()));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().expect("PNG");
    let mut decoded = vec![0; reader.output_buffer_size().expect("PNG output size")];
    let info = reader.next_frame(&mut decoded).expect("PNG frame");
    decoded.truncate(info.buffer_size());
    let rgba = match info.color_type {
        png::ColorType::Rgba => decoded,
        png::ColorType::Rgb => decoded
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
            .collect(),
        png::ColorType::Grayscale => decoded
            .iter()
            .flat_map(|&gray| [gray, gray, gray, 255])
            .collect(),
        png::ColorType::GrayscaleAlpha => decoded
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|pixel| [pixel[0], pixel[0], pixel[0], pixel[1]])
            .collect(),
        other => panic!("unexpected fixture PNG format {other:?}"),
    };
    let upem = f64::from(font.head().expect("head").units_per_em());
    let x0 = f64::from(glyph.bearing_x) / upem
        + f64::from(glyph.inner_bearing_x) / f64::from(glyph.ppem_x);
    let y = f64::from(glyph.bearing_y) / upem
        + f64::from(glyph.inner_bearing_y) / f64::from(glyph.ppem_y);
    let width = f64::from(glyph.width) / f64::from(glyph.ppem_x);
    let height = f64::from(glyph.height) / f64::from(glyph.ppem_y);
    let (y0, y1) = match glyph.placement_origin {
        Origin::TopLeft => (-y, -y + height),
        Origin::BottomLeft => (-y - height, -y),
    };
    let rect = Rect::new(
        24.0 + f64::from(size) * x0,
        112.0 + f64::from(size) * y0,
        24.0 + f64::from(size) * (x0 + width),
        112.0 + f64::from(size) * y1,
    );
    (
        ImageData::<Rgba8>::new(glyph.width, glyph.height, rgba)
            .expect("image data")
            .color_space(ImageColorSpace::Srgb),
        rect,
    )
}

fn equivalent(
    engine: &Engine<Gpu>,
    font: &cherenkov::Font,
    bytes: &[u8],
    format: BitmapFormat,
    gid: u32,
    ppem: f32,
    size: f32,
    transform: Affine,
) {
    let (image_data, rect) = image_source(bytes, format, gid, ppem, size);
    let image = engine.image(image_data).expect("reference image");
    let run = glyph_run(font.id(), gid, size);
    let actual = engine
        .surface(Offscreen::new((320, 220), OffscreenFormat::LinearF16))
        .expect("actual surface");
    let reference = engine
        .surface(Offscreen::new((320, 220), OffscreenFormat::LinearF16))
        .expect("reference surface");
    actual.update(|tx| {
        tx[actual.root()].content(actual.record(|c| {
            c.transform(transform, |c| {
                c.glyphs(run.clone(), WorkingColor::new([1.0, 0.0, 0.0, 1.0]));
            });
        }));
    });
    reference.update(|tx| {
        tx[reference.root()].content(reference.record(|c| {
            c.transform(transform, |c| {
                c.image(image.id(), rect, Sampling::Linear);
            });
        }));
    });
    engine.render(FrameTime::now()).expect("render");
    let actual_pixels = actual.readback().expect("actual readback").pixels;
    let reference_pixels = reference.readback().expect("reference readback").pixels;
    assert!(actual_pixels.iter().any(|pixel| pixel[3] > 0.0));
    assert_eq!(
        actual_pixels
            .iter()
            .map(|pixel| pixel.map(f32::to_bits))
            .collect::<Vec<_>>(),
        reference_pixels
            .iter()
            .map(|pixel| pixel.map(f32::to_bits))
            .collect::<Vec<_>>()
    );
}

#[test]
fn cbdt_and_sbix_glyphs_match_image_draws() {
    let Some(engine) = engine() else {
        return;
    };
    let cbdt = std::fs::read(CBDT_PATH).expect("CBDT fixture");
    let cbdt_id = glyph_id(&cbdt, '😀');
    let cbdt_font = engine
        .font(FontSource::bytes(cbdt.clone()))
        .expect("register CBDT font");
    equivalent(
        &engine,
        &cbdt_font,
        &cbdt,
        BitmapFormat::Cbdt,
        cbdt_id,
        109.0,
        48.0,
        Affine::IDENTITY,
    );
    equivalent(
        &engine,
        &cbdt_font,
        &cbdt,
        BitmapFormat::Cbdt,
        cbdt_id,
        109.0,
        48.0,
        Affine::translate((24.0, 12.0))
            * Affine::rotate(0.22)
            * Affine::scale_non_uniform(1.2, 0.9),
    );

    let sbix = std::fs::read(SBIX_PATH).expect("sbix fixture");
    let sbix_id = glyph_id(&sbix, '😀');
    let sbix_font = engine
        .font(FontSource::bytes(sbix.clone()))
        .expect("register sbix font");
    equivalent(
        &engine,
        &sbix_font,
        &sbix,
        BitmapFormat::Sbix,
        sbix_id,
        32.0,
        20.0,
        Affine::IDENTITY,
    );
    equivalent(
        &engine,
        &sbix_font,
        &sbix,
        BitmapFormat::Sbix,
        sbix_id,
        96.0,
        48.0,
        Affine::IDENTITY,
    );
    equivalent(
        &engine,
        &sbix_font,
        &sbix,
        BitmapFormat::Sbix,
        sbix_id,
        96.0,
        20.0,
        Affine::scale(2.0),
    );
}

#[test]
fn bitmap_cache_reuses_unchanged_glyphs_and_uses_font_identity() {
    let Some(engine) = engine() else {
        return;
    };
    let bytes = std::fs::read(CBDT_PATH).expect("CBDT fixture");
    let small = glyph_id(&bytes, '☕');
    let large = glyph_id(&bytes, '😀');
    let first_font = engine
        .font(FontSource::bytes(bytes.clone()))
        .expect("first font");
    let second_font = engine.font(FontSource::bytes(bytes)).expect("second font");
    let first = nami::Binding::container(glyph_run(first_font.id(), small, 48.0));
    let second = nami::Binding::container(glyph_run(second_font.id(), small, 48.0));
    let surface = engine
        .surface(Offscreen::new((240, 160), OffscreenFormat::LinearF16))
        .expect("surface");
    let content = surface.record(|c| {
        c.glyphs(first.clone(), WorkingColor::WHITE);
        c.glyphs(second.clone(), WorkingColor::WHITE);
    });
    surface.update(|tx| {
        tx[surface.root()].content(content);
    });
    engine.render(FrameTime::now()).expect("initial frame");
    assert_eq!(engine.stats().glyphs_rasterized, 2);
    engine.render(FrameTime::now()).expect("identical frame");
    assert_eq!(engine.stats().glyphs_rasterized, 0);

    let mut changed = glyph_run(first_font.id(), large, 48.0);
    changed.glyphs[0].x = 100.0;
    first.set(changed);
    engine.render(FrameTime::now()).expect("dirty glyph");
    assert_eq!(engine.stats().glyphs_rasterized, 1);

    let only_second = surface.record(|c| {
        c.glyphs(second.clone(), WorkingColor::WHITE);
    });
    surface.update(|tx| {
        tx[surface.root()].content(only_second);
    });
    drop(first_font);
    engine.render(FrameTime::now()).expect("font removal");
    assert_eq!(engine.stats().glyphs_rasterized, 0);
    assert!(
        surface
            .readback()
            .expect("readback")
            .pixels
            .iter()
            .any(|pixel| pixel[3] > 0.0)
    );
}

#[test]
fn missing_notdef_is_empty_and_bitmap_transforms_or_strokes_are_unsupported() {
    let Some(engine) = engine() else {
        return;
    };
    let bytes = std::fs::read(CBDT_PATH).expect("CBDT fixture");
    let font = engine
        .font(FontSource::bytes(bytes))
        .expect("register font");
    let surface = engine
        .surface(Offscreen::new((160, 120), OffscreenFormat::LinearF16))
        .expect("surface");
    let mut missing = glyph_run(font.id(), 0, 48.0);
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.glyphs(missing.clone(), WorkingColor::WHITE);
        }));
    });
    engine
        .render(FrameTime::now())
        .expect("missing glyph frame");
    assert_eq!(engine.stats().glyphs_rasterized, 0);
    assert!(
        surface
            .readback()
            .expect("readback")
            .pixels
            .iter()
            .all(|pixel| pixel[3].to_bits() == 0)
    );

    missing.glyphs[0].id = glyph_id(&std::fs::read(CBDT_PATH).expect("CBDT fixture"), '😀');
    missing.glyphs[0].transform = Some(Affine::rotate(0.2));
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.glyphs(missing.clone(), WorkingColor::WHITE);
        }));
    });
    assert!(matches!(
        engine.render(FrameTime::now()),
        Err(cherenkov::RenderError::Unsupported("glyph-transform"))
    ));

    missing.glyphs[0].transform = None;
    missing.style = GlyphStyle::Stroke(cherenkov::kurbo::Stroke::new(1.0));
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.glyphs(missing.clone(), WorkingColor::WHITE);
        }));
    });
    assert!(matches!(
        engine.render(FrameTime::now()),
        Err(cherenkov::RenderError::Unsupported("glyph-stroke"))
    ));
}

#[test]
fn bitmap_runs_validate_image_paints_like_outline_runs() {
    let cbdt = std::fs::read(CBDT_PATH).expect("CBDT fixture");
    let outline = std::fs::read(OUTLINE_PATH).expect("outline fixture");
    let Some(bitmap_error) = unregistered_image_error(&cbdt, '😀') else {
        return;
    };
    let Some(outline_error) = unregistered_image_error(&outline, 'A') else {
        return;
    };
    assert_eq!(bitmap_error, outline_error);
}
