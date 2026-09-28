// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Component animation preserves recorded content on both rendering backends.

use cherenkov::kurbo::{Affine, Circle, Rect, Vec2};
use cherenkov::{
    Animation, Backend, Curve, Draw, Engine, FrameTime, Next, Offscreen, OffscreenFormat, Picture,
    WorkingColor, snap_animating,
};
use nami::SignalExt as _;
use std::cell::Cell;
use std::rc::Rc;
use std::time::{Duration, Instant};

/// Compare engine-sampled live rotation to explicitly placed retained content.
pub fn component_animation<B: Backend>(config: B::Config) {
    let engine = Engine::<B>::new(config).expect("backend required");
    let actual = engine
        .surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))
        .expect("surface");
    let reference = engine
        .surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))
        .expect("reference");
    let layer = actual.layer();
    let fixed = Picture::record(|r| {
        r.fill(
            Rect::new(-3., -15., 5., 2.),
            WorkingColor::new([0.1, 0.5, 0.9, 1.]),
        );
        r.fill(
            Circle::new((5., -12.), 4.),
            WorkingColor::new([0.9, 0.3, 0.1, 1.]),
        );
    });
    let angle = nami::binding(0.0_f64);
    let base = Affine::translate((32., 32.));
    let pivot = Vec2::new(2., -3.);
    let shift = Vec2::new(-2., 1.);
    let scale = Vec2::new(1.1, 0.9);
    let skew = Vec2::new(0.1, -0.05);
    actual.update(|tx| {
        tx[actual.root()].push(&layer);
        tx[&layer]
            .content(fixed.clone())
            .transform(base)
            .translation(shift)
            .pivot(pivot)
            .scale(scale)
            .skew(skew)
            .rotation(
                angle
                    .clone()
                    .with(Animation::from(Curve::linear(Duration::from_secs(1)))),
            );
    });
    reference.update(|tx| {
        tx[reference.root()].content(fixed.clone());
    });
    angle.set(std::f64::consts::TAU);
    let start = Instant::now();
    for step in 0..=8_u32 {
        let fraction = f64::from(step) / 8.;
        let matrix = base
            * Affine::translate(shift + pivot)
            * Affine::rotate(std::f64::consts::TAU * fraction)
            * Affine::new([1., skew.y.tan(), skew.x.tan(), 1., 0., 0.])
            * Affine::scale_non_uniform(scale.x, scale.y)
            * Affine::translate(-pivot);
        // Moving content snaps to the ¼-pixel grid; the settled frame is exact.
        let matrix = if step < 8 {
            snap_animating(matrix)
        } else {
            matrix
        };
        reference.update(|tx| {
            tx[reference.root()].transform(matrix);
        });
        let next = engine
            .render(FrameTime::at(
                start + Duration::from_millis(u64::from(step) * 125),
            ))
            .expect("frame");
        assert_eq!(matches!(next, Next::At { .. }), step < 8);
        if step > 0 {
            assert_eq!(
                engine.stats().commands_lowered,
                0,
                "animation must reuse the recorded commands"
            );
        }
        let a = actual.readback().expect("animated");
        let b = reference.readback().expect("reference");
        for (index, (a, b)) in a.pixels.iter().zip(b.pixels).enumerate() {
            for (a, b) in a.iter().zip(b) {
                assert!(
                    (a - b).abs() < 0.003,
                    "step {step} pixel {index}: {a} != {b}"
                );
            }
        }
    }
    let wakes = Rc::new(Cell::new(0));
    let count = wakes.clone();
    engine.set_waker(move || count.set(count.get() + 1));
    angle.set(0.);
    assert_eq!(
        wakes.get(),
        1,
        "a signal wakes an idle engine without a transaction"
    );
    drop(layer);
    signal_drops_with_the_layer(&engine, &angle, &wakes, start);
}

/// After a bound layer drops, its signal no longer wakes the engine.
fn signal_drops_with_the_layer<B: Backend>(
    engine: &Engine<B>,
    angle: &nami::Binding<f64>,
    wakes: &Cell<usize>,
    start: Instant,
) {
    engine
        .render(FrameTime::at(start + Duration::from_secs(2)))
        .expect("remove bound layer");
    let before = wakes.get();
    angle.set(1.);
    assert_eq!(
        wakes.get(),
        before,
        "dropping layer disconnects component bindings"
    );
}
