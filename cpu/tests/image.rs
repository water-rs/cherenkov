// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Shared-front-end image uploads, retained destinations and alpha metadata.
use cherenkov::kurbo::{Affine, Rect};
use cherenkov::{
    Draw, Engine, FrameTime, ImageColorSpace, ImageData, Offscreen, OffscreenFormat, Picture,
    Rgba8, Sampling, WorkingColor,
};
use cherenkov_cpu::{Raster, RasterConfig};
use nami::Binding;

#[test]
fn encoded_premultiplied_upload_matches_straight_alpha() {
    let engine = Engine::<Raster>::new(RasterConfig::default()).expect("engine");
    let straight = engine
        .image(
            ImageData::<Rgba8>::new(1, 1, vec![255, 0, 0, 128])
                .expect("data")
                .color_space(ImageColorSpace::DisplayP3),
        )
        .expect("straight");
    let premul = engine
        .image(
            ImageData::<Rgba8>::new(1, 1, vec![128, 0, 0, 128])
                .expect("data")
                .color_space(ImageColorSpace::DisplayP3)
                .premultiplied(),
        )
        .expect("premul");
    let surface = engine
        .surface(Offscreen::new((4, 2), OffscreenFormat::LinearF32))
        .expect("surface");
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.image(
                straight.id(),
                Rect::new(0.0, 0.0, 2.0, 2.0),
                Sampling::Nearest,
            );
            c.image(
                premul.id(),
                Rect::new(2.0, 0.0, 4.0, 2.0),
                Sampling::Nearest,
            );
        }));
    });
    engine.render(FrameTime::now()).expect("render");
    let pixels = surface.readback().expect("pixels").pixels;
    assert_eq!(pixels[0].map(f32::to_bits), pixels[2].map(f32::to_bits));
    let alpha = 128.0_f32 / 255.0;
    assert!((pixels[0][0] - alpha).abs() < 1e-6 && (pixels[0][3] - alpha).abs() < 1e-6);
}

#[test]
fn image_destination_is_live_and_static_picture_stays_retained() {
    let engine = Engine::<Raster>::new(RasterConfig::default()).expect("engine");
    let image = engine
        .image(ImageData::<Rgba8>::new(1, 1, vec![255, 255, 255, 255]).expect("data"))
        .expect("image");
    let surface = engine
        .surface(Offscreen::new((24, 24), OffscreenFormat::LinearF32))
        .expect("surface");
    let dst = Binding::container(Rect::new(4.0, 4.0, 8.0, 8.0));
    let fixed = Picture::record(|c| c.fill(Rect::new(0.0, 0.0, 2.0, 2.0), WorkingColor::WHITE));
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.picture(&fixed, Affine::IDENTITY);
            c.image(image.id(), dst.clone(), Sampling::Linear);
        }));
    });
    engine.render(FrameTime::now()).expect("initial");
    assert_eq!(engine.stats().commands_lowered, 2);
    dst.set(Rect::new(12.0, 12.0, 16.0, 16.0));
    engine.render(FrameTime::now()).expect("edit");
    assert_eq!(engine.stats().commands_lowered, 1);
    let pixels = surface.readback().expect("pixels").pixels;
    assert!(pixels[13 * 24 + 13][3] > 0.999);
    assert_eq!(pixels[5 * 24 + 5][3].to_bits(), 0.0_f32.to_bits());
    engine.render(FrameTime::now()).expect("idle");
    assert_eq!(engine.stats().commands_lowered, 0);
}
