// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Registered images and image paint on lavapipe.

#![expect(
    clippy::cast_lossless,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::suboptimal_flops,
    reason = "test tolerances and srgb8 conversions"
)]

use cherenkov::kurbo::{Affine, Rect};
use cherenkov::{Draw, Extend, ImagePattern, Paint, Sampling};
use cherenkov_gpu::{
    Engine, EngineError, Gpu, GpuConfig, Image, ImageColorSpace, ImageSource, Offscreen,
    OffscreenFormat,
};

/// An engine, or `None` when no adapter exists.
fn engine() -> Option<Engine<Gpu>> {
    match Engine::<Gpu>::new(GpuConfig::default()) {
        Ok(engine) => Some(engine),
        Err(EngineError::NoAdapter) => None,
        Err(e) => panic!("engine init failed: {e}"),
    }
}

/// sRGB→XYZ→P3 with the same constants the render thread uploads with.
const SRGB_TO_XYZ: [[f32; 3]; 3] = [
    [0.412_390_7, 0.357_584_33, 0.180_480_79],
    [0.212_639, 0.715_168_7, 0.072_192_32],
    [0.019_330_82, 0.119_194_76, 0.950_532_14],
];
const XYZ_TO_P3: [[f32; 3]; 3] = [
    [2.493_497, -0.931_383_6, -0.402_710_77],
    [-0.829_488_93, 1.762_664, 0.023_624_687],
    [0.035_845_827, -0.076_172_38, 0.956_884_5],
];

fn srgb_decode(c: u8) -> f32 {
    let u = c as f32 / 255.0;
    if u <= 0.040_45 {
        u / 12.92
    } else {
        ((u + 0.055) / 1.055).powf(2.4)
    }
}

fn mat_vec(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// The premultiplied linear-P3 value of an sRGB8 texel, as uploaded.
fn premul_p3(c: [u8; 4]) -> [f32; 4] {
    let srgb = [srgb_decode(c[0]), srgb_decode(c[1]), srgb_decode(c[2])];
    let p3 = mat_vec(&XYZ_TO_P3, mat_vec(&SRGB_TO_XYZ, srgb));
    let a = c[3] as f32 / 255.0;
    [p3[0] * a, p3[1] * a, p3[2] * a, a]
}

/// Compare a readback texel (premultiplied linear P3 f32) to an expected
/// premultiplied-P3 colour.
fn close_px(px: [f32; 4], want_p3_premul: [f32; 4]) {
    for (g, w) in px.iter().zip(want_p3_premul) {
        assert!((g - w).abs() < 2e-2, "{px:?} vs {want_p3_premul:?}");
    }
}

/// A 2×2 image: opaque red / opaque green on row 0, opaque blue /
/// half-alpha white on row 1.
fn two_by_two(engine: &Engine<Gpu>) -> Image {
    engine
        .image(ImageSource {
            width: 2,
            height: 2,
            pixels: vec![
                255, 0, 0, 255, // red
                0, 255, 0, 255, // green
                0, 0, 255, 255, // blue
                255, 255, 255, 128, // half white
            ],
            color_space: ImageColorSpace::Srgb,
        })
        .unwrap()
}

#[test]
fn an_image_draws_nearest() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let image = two_by_two(&engine);
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.image(image.id(), Rect::new(0., 0., 64., 64.), Sampling::Nearest);
        }));
    });
    engine.render(cherenkov_gpu::FrameTime::now())?;
    let rb = surface.readback()?;
    let px = |x: u32, y: u32| rb.pixels[(y * rb.width + x) as usize];
    close_px(px(16, 16), premul_p3([255, 0, 0, 255]));
    close_px(px(48, 16), premul_p3([0, 255, 0, 255]));
    close_px(px(16, 48), premul_p3([0, 0, 255, 255]));
    close_px(px(48, 48), premul_p3([255, 255, 255, 128]));
    Ok(())
}

#[test]
fn an_image_interpolates_bilinear() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let image = two_by_two(&engine);
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.image(image.id(), Rect::new(0., 0., 64., 64.), Sampling::Linear);
        }));
    });
    engine.render(cherenkov_gpu::FrameTime::now())?;
    let rb = surface.readback()?;
    // Pixel centre (32,32) sits exactly between the four texels.
    let mut avg = [0f32; 4];
    for c in [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 255, 128],
    ] {
        let p = premul_p3(c);
        for i in 0..4 {
            avg[i] += p[i] * 0.25;
        }
    }
    let px = rb.pixels[(32 * rb.width + 32) as usize];
    for (g, w) in px.iter().zip(avg) {
        assert!((g - w).abs() < 3e-2, "{px:?} vs {avg:?}");
    }
    Ok(())
}

#[test]
fn an_image_pattern_repeats() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let image = two_by_two(&engine);
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(
                Rect::new(0., 0., 64., 64.),
                Paint::Image(ImagePattern {
                    image: image.id(),
                    transform: Affine::IDENTITY,
                    extend_x: Extend::Repeat,
                    extend_y: Extend::Repeat,
                    sampling: Sampling::Nearest,
                }),
            );
        }));
    });
    engine.render(cherenkov_gpu::FrameTime::now())?;
    let rb = surface.readback()?;
    let px = |x: u32, y: u32| rb.pixels[(y * rb.width + x) as usize];
    // Identity transform maps image pixels 1:1; texel (0,0) is red and
    // wraps every 2 px — a pixel centre at x≈40.5 lands on texel 0.
    close_px(px(40, 0), premul_p3([255, 0, 0, 255]));
    close_px(px(41, 1), premul_p3([255, 255, 255, 128]));
    Ok(())
}

#[test]
fn an_image_pattern_with_extend_none_is_transparent() -> Result<(), Box<dyn std::error::Error>> {
    let Some(engine) = engine() else {
        return Ok(());
    };
    let image = two_by_two(&engine);
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(
                Rect::new(0., 0., 64., 64.),
                Paint::Image(ImagePattern {
                    image: image.id(),
                    transform: Affine::IDENTITY,
                    extend_x: Extend::None,
                    extend_y: Extend::None,
                    sampling: Sampling::Nearest,
                }),
            );
        }));
    });
    engine.render(cherenkov_gpu::FrameTime::now())?;
    let rb = surface.readback()?;
    let px = |x: u32, y: u32| rb.pixels[(y * rb.width + x) as usize];
    close_px(px(0, 0), premul_p3([255, 0, 0, 255]));
    assert_eq!(px(8, 8), [0.0; 4], "outside the image must be clear");
    Ok(())
}
