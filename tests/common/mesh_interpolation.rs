// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Known mesh weights and retained mode changes, shared by GPU and CPU tests.
use cherenkov::kurbo::{Point, Rect};
use cherenkov::{
    Backend, Draw, Engine, FrameTime, MeshColorInterpolation, MeshGradient, Offscreen,
    OffscreenFormat, WorkingColor,
};

fn mesh(mode: MeshColorInterpolation) -> MeshGradient {
    MeshGradient::new(
        1,
        1,
        vec![
            Point::new(0.5, 0.5),
            Point::new(16.5, 0.5),
            Point::new(0.5, 16.5),
            Point::new(16.5, 16.5),
        ],
        vec![
            WorkingColor::BLACK,
            WorkingColor::WHITE,
            WorkingColor::BLACK,
            WorkingColor::WHITE,
        ],
    )
    .interpolation(mode)
}

/// The distance between adjacent binary16 values at a normal, positive
/// `value`: the resolution of the `LinearF16` target. A conforming device may
/// round the shader's result to either neighbour of an exact value.
const fn f16_spacing(value: f32) -> f32 {
    f32::from_bits((value.to_bits() & 0x7f80_0000) - (10 << 23))
}

/// Changing only the weight rule patches exactly its recorded command.
pub fn interpolation<B: Backend>(config: B::Config) {
    let engine = Engine::<B>::new(config).expect("backend");
    let surface = engine
        .surface(Offscreen::new((24, 20), OffscreenFormat::LinearF16))
        .expect("surface");
    let value = nami::Binding::container(mesh(MeshColorInterpolation::Linear));
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|r| {
            r.fill(Rect::new(0., 0., 18., 20.), value.clone());
            r.fill(
                Rect::new(20., 0., 24., 20.),
                WorkingColor::new([1., 0., 0., 1.]),
            );
        }));
    });
    engine.render(FrameTime::now()).expect("initial");
    assert_eq!(engine.stats().commands_lowered, 2);
    for (mode, expected) in [
        (MeshColorInterpolation::Smoothstep, 0.15625),
        (MeshColorInterpolation::Linear, 0.25),
    ] {
        value.set(mesh(mode));
        engine.render(FrameTime::now()).expect("change rule");
        assert_eq!(engine.stats().commands_lowered, 1);
        let pixels = surface.readback().expect("pixels").pixels;
        for c in &pixels[8 * 24 + 4][..3] {
            assert!(
                (c - expected).abs() <= f16_spacing(expected),
                "{mode:?}: {c} != {expected}"
            );
        }
        assert!(
            (pixels[8 * 24 + 22][0] - 1.).abs() < 0.0001,
            "unrelated command remains"
        );
        engine.render(FrameTime::now()).expect("idle");
        assert_eq!(engine.stats().commands_lowered, 0);
    }
}
