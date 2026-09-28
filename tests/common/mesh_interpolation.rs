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
            assert!((c - expected).abs() < 0.0001, "{mode:?}: {c} != {expected}");
        }
        assert!(
            (pixels[8 * 24 + 22][0] - 1.).abs() < 0.0001,
            "unrelated command remains"
        );
        engine.render(FrameTime::now()).expect("idle");
        assert_eq!(engine.stats().commands_lowered, 0);
    }
}
