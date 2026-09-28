// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Backdrop groups: bounded f16 captures sampled by member layers.

use cherenkov::kurbo::{Rect, RoundedRect};
use cherenkov::{Bytes, Draw, Engine, FrameTime, Offscreen, OffscreenFormat, WorkingColor};
use cherenkov_gpu::{Gpu, GpuConfig};

fn pixel(readback: &cherenkov::Readback, x: usize, y: usize) -> [f32; 4] {
    let p = &readback.pixels[y * readback.width as usize + x];
    [p[0], p[1], p[2], p[3]]
}

fn assert_pixel(actual: [f32; 4], expected: [f32; 4], tolerance: f32) {
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() <= tolerance,
            "pixel {actual:?}, expected {expected:?}"
        );
    }
}

#[test]
fn unfiltered_member_samples_what_is_behind_it() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::<Gpu>::new(GpuConfig::default())?;
    let surface = engine.surface(Offscreen::new((32, 32), OffscreenFormat::LinearF16))?;
    let group = surface.backdrop_group_unfiltered();
    let glass = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|r| {
            r.fill(
                Rect::new(0.0, 0.0, 16.0, 32.0),
                WorkingColor::new([1.0, 0.0, 0.0, 1.0]),
            );
            r.fill(
                Rect::new(16.0, 0.0, 32.0, 32.0),
                WorkingColor::new([0.0, 0.0, 1.0, 1.0]),
            );
        }));
        tx[surface.root()].push(&glass);
        tx[&glass]
            .clip(Rect::new(8.0, 8.0, 24.0, 24.0))
            .backdrop(group.sample())
            .content(surface.record(|r| {
                r.fill(
                    Rect::new(8.0, 8.0, 24.0, 24.0),
                    WorkingColor::new([1.0, 1.0, 1.0, 0.5]),
                );
            }));
    });
    engine.render(FrameTime::now())?;
    let readback = surface.readback()?;
    // src_over(50% white, solid red) = [1.0, 0.5, 0.5, 1.0].
    assert_pixel(pixel(&readback, 12, 12), [1.0, 0.5, 0.5, 1.0], 1e-3);
    // src_over(50% white, solid blue) = [0.5, 0.5, 1.0, 1.0].
    assert_pixel(pixel(&readback, 20, 12), [0.5, 0.5, 1.0, 1.0], 1e-3);
    // Outside the member's clip the surface is untouched.
    assert_pixel(pixel(&readback, 4, 4), [1.0, 0.0, 0.0, 1.0], 1e-3);
    let memory = engine.memory();
    assert_eq!(memory.backdrop_captures, Bytes(16 * 16 * 8));
    assert_eq!(memory.backdrop_capture_format, Some("rgba16float"));
    Ok(())
}

#[test]
fn blurred_backdrop_keeps_extended_range() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::<Gpu>::new(GpuConfig::default())?;
    let surface = engine.surface(Offscreen::new((64, 64), OffscreenFormat::LinearF16))?;
    let group = surface.backdrop_group(filtrate::filters::GaussianBlur(4.0f32));
    let glass = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|r| {
            r.fill(
                Rect::new(0.0, 0.0, 32.0, 64.0),
                WorkingColor::new([16.0, 16.0, 16.0, 1.0]),
            );
            r.fill(
                Rect::new(32.0, 0.0, 64.0, 64.0),
                WorkingColor::new([0.25, 0.25, 0.25, 1.0]),
            );
        }));
        tx[surface.root()].push(&glass);
        tx[&glass]
            .clip(Rect::new(8.0, 8.0, 56.0, 56.0))
            .backdrop(group.sample());
    });
    engine.render(FrameTime::now())?;
    let readback = surface.readback()?;
    // HDR whites survive the capture, blur and sampling unclamped: the
    // blur's support stays inside the left half at x=16.
    let bright = pixel(&readback, 16, 32);
    assert!(bright[..3].iter().all(|c| *c > 15.0), "pixel {bright:?}");
    assert_pixel(pixel(&readback, 48, 32), [0.25, 0.25, 0.25, 1.0], 0.05);
    // The step's midpoint blurs to roughly the mean of both halves.
    let edge = pixel(&readback, 32, 32);
    let expected = f32::midpoint(16.0, 0.25);
    assert!(
        (edge[0] - expected).abs() < 1.0 && (edge[1] - expected).abs() < 1.0,
        "edge pixel {edge:?}, expected about {expected}"
    );
    let memory = engine.memory();
    // footprint = ceil(4 * 3) = 12 → region (8-12..56+12) ∩ 0..64 = 64×64.
    // The filter's input/output intermediates count in `gpu`, not in
    // `backdrop_captures` — only the one capture texture does.
    assert_eq!(memory.backdrop_captures, Bytes(64 * 64 * 8));
    assert_eq!(memory.backdrop_capture_format, Some("rgba16float"));
    Ok(())
}

#[test]
fn nested_groups_capture_in_paint_order() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::<Gpu>::new(GpuConfig::default())?;
    let surface = engine.surface(Offscreen::new((32, 32), OffscreenFormat::LinearF16))?;
    let outer = surface.backdrop_group_unfiltered();
    let inner = surface.backdrop_group_unfiltered();
    let m1 = surface.layer();
    let m2 = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|r| {
            r.fill(
                Rect::new(0.0, 0.0, 32.0, 32.0),
                WorkingColor::new([1.0, 0.0, 0.0, 1.0]),
            );
        }));
        tx[surface.root()].push(&m1);
        tx[&m1]
            .clip(Rect::new(4.0, 4.0, 28.0, 28.0))
            .backdrop(outer.sample())
            .content(surface.record(|r| {
                r.fill(
                    Rect::new(4.0, 4.0, 28.0, 28.0),
                    WorkingColor::new([0.0, 0.0, 1.0, 0.5]),
                );
            }));
        tx[&m1].push(&m2);
        tx[&m2]
            .clip(Rect::new(12.0, 12.0, 20.0, 20.0))
            .backdrop(inner.sample())
            .content(surface.record(|r| {
                r.fill(
                    Rect::new(12.0, 12.0, 20.0, 20.0),
                    WorkingColor::new([0.0, 1.0, 0.0, 0.5]),
                );
            }));
    });
    engine.render(FrameTime::now())?;
    let readback = surface.readback()?;
    // Inside m1 but outside m2: 50% blue over red = [0.5, 0.0, 0.5].
    assert_pixel(pixel(&readback, 8, 8), [0.5, 0.0, 0.5, 1.0], 1e-3);
    // Inside m2 the inner capture saw m1's composite ([0.5, 0.0, 0.5]),
    // then 50% green drew over it: [0.25, 0.5, 0.25].
    assert_pixel(pixel(&readback, 16, 16), [0.25, 0.5, 0.25, 1.0], 1e-3);
    Ok(())
}

#[test]
fn member_inside_clip_only_isolation_sees_the_surface() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::<Gpu>::new(GpuConfig::default())?;
    let surface = engine.surface(Offscreen::new((32, 32), OffscreenFormat::LinearF16))?;
    let group = surface.backdrop_group_unfiltered();
    let p = surface.layer();
    let member = surface.layer();
    surface.update(|tx| {
        // The root's rounded-rect clip cannot merge with P's, so P's body
        // lands in a clip-only scratch — invisible to the capture, which
        // must compose it over the surface copy.
        tx[surface.root()]
            .clip(RoundedRect::new(0.0, 0.0, 32.0, 32.0, 2.0))
            .content(surface.record(|r| {
                r.fill(
                    Rect::new(0.0, 0.0, 32.0, 32.0),
                    WorkingColor::new([1.0, 0.0, 0.0, 1.0]),
                );
            }));
        tx[surface.root()].push(&p);
        tx[&p]
            .clip(RoundedRect::new(0.0, 0.0, 32.0, 32.0, 4.0))
            .content(surface.record(|r| {
                r.fill(
                    Rect::new(8.0, 8.0, 12.0, 24.0),
                    WorkingColor::new([0.0, 0.0, 1.0, 1.0]),
                );
            }));
        tx[&p].push(&member);
        tx[&member]
            .clip(RoundedRect::new(8.0, 8.0, 24.0, 24.0, 3.0))
            .backdrop(group.sample());
    });
    engine.render(FrameTime::now())?;
    let readback = surface.readback()?;
    // The member samples the surface's red plus the clip scratch's blue
    // painted before it, each at the right device pixels.
    assert_pixel(pixel(&readback, 10, 16), [0.0, 0.0, 1.0, 1.0], 1e-3);
    assert_pixel(pixel(&readback, 16, 16), [1.0, 0.0, 0.0, 1.0], 1e-3);
    Ok(())
}

#[test]
fn member_without_clip_is_unsupported() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::<Gpu>::new(GpuConfig::default())?;
    let surface = engine.surface(Offscreen::new((32, 32), OffscreenFormat::LinearF16))?;
    let group = surface.backdrop_group_unfiltered();
    let member = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].push(&member);
        tx[&member].backdrop(group.sample());
    });
    let result = engine.render(FrameTime::now());
    assert!(
        matches!(
            result,
            Err(cherenkov::RenderError::Unsupported(name)) if name == "backdrop-unclipped"
        ),
        "unexpected result {result:?}"
    );
    Ok(())
}

#[test]
fn dropped_group_fails_the_frame() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::<Gpu>::new(GpuConfig::default())?;
    let surface = engine.surface(Offscreen::new((32, 32), OffscreenFormat::LinearF16))?;
    let member = surface.layer();
    {
        let group = surface.backdrop_group_unfiltered();
        surface.update(|tx| {
            tx[surface.root()].push(&member);
            tx[&member]
                .clip(Rect::new(8.0, 8.0, 24.0, 24.0))
                .backdrop(group.sample());
        });
    }
    let result = engine.render(FrameTime::now());
    assert!(
        matches!(result, Err(cherenkov::RenderError::Render(_))),
        "unexpected result {result:?}"
    );
    Ok(())
}

#[test]
fn two_members_share_one_capture() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::<Gpu>::new(GpuConfig::default())?;
    let surface = engine.surface(Offscreen::new((32, 32), OffscreenFormat::LinearF16))?;
    let group = surface.backdrop_group_unfiltered();
    let left = surface.layer();
    let right = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|r| {
            r.fill(
                Rect::new(0.0, 0.0, 32.0, 32.0),
                WorkingColor::new([0.0, 1.0, 0.0, 1.0]),
            );
        }));
        tx[surface.root()].push(&left).push(&right);
        tx[&left]
            .clip(Rect::new(2.0, 2.0, 6.0, 6.0))
            .backdrop(group.sample());
        tx[&right]
            .clip(Rect::new(26.0, 2.0, 30.0, 6.0))
            .backdrop(group.sample());
    });
    engine.render(FrameTime::now())?;
    let readback = surface.readback()?;
    // Both members sample the one capture: the green surface behind them.
    assert_pixel(pixel(&readback, 4, 4), [0.0, 1.0, 0.0, 1.0], 1e-3);
    assert_pixel(pixel(&readback, 28, 4), [0.0, 1.0, 0.0, 1.0], 1e-3);
    let memory = engine.memory();
    // One capture for the group, bounded to the members' union (x 2..30,
    // y 2..6) — not one per member nor the whole surface.
    assert_eq!(memory.backdrop_captures, Bytes(28 * 4 * 8));
    assert_eq!(memory.backdrop_capture_format, Some("rgba16float"));
    Ok(())
}
