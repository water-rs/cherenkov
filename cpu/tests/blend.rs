// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Group compositing is clipped as an operation, including destination modes.
use cherenkov::kurbo::Rect;
use cherenkov::{
    BlendMode, BlendSpace, Draw, Engine, FrameTime, Group, Offscreen, OffscreenFormat, WorkingColor,
};
use cherenkov_cpu::{Raster, RasterConfig};

const RED: WorkingColor = WorkingColor::new([1.0, 0.0, 0.0, 1.0]);
const BLUE: WorkingColor = WorkingColor::new([0.0, 0.0, 1.0, 1.0]);
const GREEN: WorkingColor = WorkingColor::new([0.0, 1.0, 0.0, 1.0]);

#[test]
fn clear_composite_preserves_destination_outside_clip() {
    let engine = Engine::<Raster>::new(RasterConfig::default()).expect("engine");
    let surface = engine
        .surface(Offscreen::new((8, 8), OffscreenFormat::LinearF32))
        .expect("surface");
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(Rect::new(0.0, 0.0, 8.0, 8.0), WorkingColor::WHITE);
            c.clip(Rect::new(2.0, 2.0, 6.0, 6.0), |c| {
                c.group(Group::new().blend(BlendMode::Clear), |c| {
                    c.fill(Rect::new(0.0, 0.0, 8.0, 8.0), WorkingColor::WHITE);
                });
            });
        }));
    });
    engine.render(FrameTime::now()).expect("render");
    let pixels = surface.readback().expect("pixels").pixels;
    assert_eq!(pixels[0].map(f32::to_bits), [1.0_f32; 4].map(f32::to_bits));
    assert_eq!(
        pixels[3 * 8 + 3].map(f32::to_bits),
        [0.0_f32; 4].map(f32::to_bits)
    );
}

#[test]
fn blended_descendant_isolates_its_normal_group() {
    let engine = Engine::<Raster>::new(RasterConfig::default()).expect("engine");
    let surface = engine
        .surface(Offscreen::new((8, 8), OffscreenFormat::LinearF32))
        .expect("surface");
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(
                Rect::new(0.0, 0.0, 8.0, 8.0),
                WorkingColor::new([1.0, 0.0, 0.0, 1.0]),
            );
            c.group(Group::new(), |c| {
                c.fill(
                    Rect::new(0.0, 0.0, 4.0, 8.0),
                    WorkingColor::new([0.0, 0.0, 1.0, 1.0]),
                );
                c.group(Group::new().blend(BlendMode::Clear), |c| {
                    c.fill(
                        Rect::new(2.0, 0.0, 6.0, 8.0),
                        WorkingColor::new([1.0, 1.0, 1.0, 1.0]),
                    );
                });
            });
        }));
    });
    engine.render(FrameTime::now()).expect("render");
    let pixels = surface.readback().expect("pixels").pixels;
    // `Clear` zeroes the inner group's whole raster — but only inside the
    // outer group's offscreen, which then composites `Normal` over the red
    // background. Without outer isolation the `Clear` reached the scene
    // framebuffer and every pixel came out transparent.
    let red = [1.0_f32, 0.0, 0.0, 1.0].map(f32::to_bits);
    for (i, px) in pixels.iter().enumerate() {
        assert_eq!(px.map(f32::to_bits), red, "pixel {i}");
    }
}

#[test]
fn tree_layer_isolates_blended_child_layer() {
    let engine = Engine::<Raster>::new(RasterConfig::default()).expect("engine");
    let surface = engine
        .surface(Offscreen::new((8, 8), OffscreenFormat::LinearF32))
        .expect("surface");
    let background = surface.layer();
    let pass = surface.layer();
    let cutout = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].push(&background).push(&pass);
        tx[&background].content(surface.record(|c| {
            c.fill(Rect::new(0.0, 0.0, 8.0, 8.0), RED);
        }));
        tx[&pass].content(surface.record(|c| {
            c.fill(Rect::new(0.0, 0.0, 4.0, 8.0), BLUE);
        }));
        tx[&pass].push(&cutout);
        tx[&cutout]
            .blend(BlendMode::DestOut)
            .content(surface.record(|c| {
                c.fill(Rect::new(2.0, 0.0, 6.0, 8.0), WorkingColor::WHITE);
            }));
    });
    engine.render(FrameTime::now()).expect("render");
    let pixels = surface.readback().expect("pixels").pixels;
    let pixel = |x: usize, y: usize| pixels[y * 8 + x];
    assert_eq!(
        pixel(1, 3).map(f32::to_bits),
        [0.0_f32, 0.0, 1.0, 1.0].map(f32::to_bits)
    );
    assert_eq!(
        pixel(3, 3).map(f32::to_bits),
        [1.0_f32, 0.0, 0.0, 1.0].map(f32::to_bits)
    );
}

#[test]
fn tree_layer_isolates_blended_content_group() {
    let engine = Engine::<Raster>::new(RasterConfig::default()).expect("engine");
    let surface = engine
        .surface(Offscreen::new((8, 8), OffscreenFormat::LinearF32))
        .expect("surface");
    let background = surface.layer();
    let pass = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].push(&background).push(&pass);
        tx[&background].content(surface.record(|c| {
            c.fill(Rect::new(0.0, 0.0, 8.0, 8.0), RED);
        }));
        tx[&pass].content(surface.record(|c| {
            c.fill(Rect::new(0.0, 0.0, 4.0, 8.0), BLUE);
            c.group(Group::new().blend(BlendMode::DestOut), |c| {
                c.fill(Rect::new(2.0, 0.0, 6.0, 8.0), WorkingColor::WHITE);
            });
        }));
    });
    engine.render(FrameTime::now()).expect("render");
    let pixels = surface.readback().expect("pixels").pixels;
    let pixel = |x: usize, y: usize| pixels[y * 8 + x];
    assert_eq!(
        pixel(1, 3).map(f32::to_bits),
        [0.0_f32, 0.0, 1.0, 1.0].map(f32::to_bits)
    );
    assert_eq!(
        pixel(3, 3).map(f32::to_bits),
        [1.0_f32, 0.0, 0.0, 1.0].map(f32::to_bits)
    );
}

#[test]
fn nested_tree_layers_isolate_at_the_blending_parent() {
    let engine = Engine::<Raster>::new(RasterConfig::default()).expect("engine");
    let surface = engine
        .surface(Offscreen::new((8, 8), OffscreenFormat::LinearF32))
        .expect("surface");
    let background = surface.layer();
    let outer = surface.layer();
    let inner = surface.layer();
    let cutout = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].push(&background).push(&outer);
        tx[&outer].push(&inner);
        tx[&inner].push(&cutout);
        tx[&background].content(surface.record(|c| {
            c.fill(Rect::new(0.0, 0.0, 8.0, 8.0), RED);
        }));
        tx[&outer].content(surface.record(|c| {
            c.fill(Rect::new(0.0, 0.0, 8.0, 8.0), BLUE);
        }));
        tx[&inner].content(surface.record(|c| {
            c.fill(Rect::new(0.0, 0.0, 4.0, 8.0), GREEN);
        }));
        tx[&cutout]
            .blend(BlendMode::DestOut)
            .content(surface.record(|c| {
                c.fill(Rect::new(2.0, 0.0, 6.0, 8.0), WorkingColor::WHITE);
            }));
    });
    engine.render(FrameTime::now()).expect("render");
    let pixels = surface.readback().expect("pixels").pixels;
    let pixel = |x: usize, y: usize| pixels[y * 8 + x];
    assert_eq!(
        pixel(1, 3).map(f32::to_bits),
        [0.0_f32, 1.0, 0.0, 1.0].map(f32::to_bits)
    );
    assert_eq!(
        pixel(3, 3).map(f32::to_bits),
        [0.0_f32, 0.0, 1.0, 1.0].map(f32::to_bits)
    );
}

#[test]
fn encoded_group_composites_in_encoded_space() {
    let engine = Engine::<Raster>::new(RasterConfig::default()).expect("engine");
    let surface = engine
        .surface(Offscreen::new((4, 4), OffscreenFormat::LinearF32))
        .expect("surface");
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(
                Rect::new(0.0, 0.0, 4.0, 4.0),
                WorkingColor::new([0.2, 0.2, 0.2, 1.0]),
            );
            c.group(
                Group::new()
                    .blend_space(BlendSpace::SrgbEncoded)
                    .opacity(0.5),
                |c| {
                    c.fill(
                        Rect::new(0.0, 0.0, 4.0, 4.0),
                        WorkingColor::new([0.8, 0.8, 0.8, 1.0]),
                    );
                },
            );
        }));
    });
    engine.render(FrameTime::now()).expect("render");
    let pixel = surface.readback().expect("pixels").pixels[5];
    let encoded = 0.2_f64.powf(1.0 / 2.4).midpoint(0.8_f64.powf(1.0 / 2.4));
    let expected = encoded.powf(2.4);
    for channel in &pixel[..3] {
        assert!((f64::from(*channel) - expected).abs() < 1e-5, "{pixel:?}");
    }
    assert_eq!(pixel[3].to_bits(), 1.0_f32.to_bits());
}
