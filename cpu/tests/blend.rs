//! Group compositing is clipped as an operation, including destination modes.
use cherenkov::kurbo::Rect;
use cherenkov::{
    BlendMode, BlendSpace, Draw, Engine, FrameTime, Group, Offscreen, OffscreenFormat, WorkingColor,
};
use cherenkov_cpu::{Raster, RasterConfig};

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
