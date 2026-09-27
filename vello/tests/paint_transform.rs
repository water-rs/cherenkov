//! Coordinate mapping, independently of Vello's existing colour error.
use cherenkov::kurbo::{Affine, Rect, Shape as _};
use cherenkov::{
    Draw, Engine, FrameTime, Offscreen, OffscreenFormat, Paint, RadialGradient, WorkingColor,
};
use cherenkov_vello::{Vello, VelloConfig};

#[test]
fn independent_paint_mapping_matches_transformed_geometry_with_inverse_shape() {
    let engine = Engine::<Vello>::new(VelloConfig::default()).expect("Vello");
    let wrapped = engine
        .surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))
        .expect("surface");
    let reference = engine
        .surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))
        .expect("surface");
    let gradient: Paint = RadialGradient::new((32.0, 32.0), 20.0)
        .stop(0.0, WorkingColor::new([1.0, 0.0, 0.0, 1.0]))
        .stop(1.0, WorkingColor::new([0.0, 0.0, 1.0, 1.0]))
        .into();
    let transform = Affine::new([-1.1, 0.15, 0.25, 0.75, 60.0, 4.0]);
    let rect = Rect::new(2.0, 2.0, 62.0, 62.0);
    wrapped.update(|tx| {
        tx[wrapped.root()].content(wrapped.record(|c| {
            c.fill(rect, gradient.clone().transformed(transform));
        }));
    });
    reference.update(|tx| {
        tx[reference.root()]
            .transform(transform)
            .content(reference.record(|c| {
                c.fill(transform.inverse() * rect.to_path(0.01), gradient.clone());
            }));
    });
    engine.render(FrameTime::now()).expect("render");
    let actual = wrapped.readback().expect("wrapped");
    let expected = reference.readback().expect("reference");
    for y in 6..58 {
        for x in 6..58 {
            for (got, want) in actual.pixels[y * 64 + x]
                .iter()
                .zip(expected.pixels[y * 64 + x])
            {
                assert!((got - want).abs() < 0.003, "({x},{y}): {got} != {want}");
            }
        }
    }
}
