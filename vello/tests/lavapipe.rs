// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Pixel tests against a real adapter (lavapipe in CI/dev machines).
//!
//! Every expected pixel is computed from first principles: linear Display
//! P3 → linear sRGB matrix → sRGB encode → quantize to u8 → decode back →
//! linear P3, so the tolerance is 2/255 per channel.

use cherenkov::kurbo::{Affine, Rect};
use cherenkov::{Draw, WorkingColor};
use cherenkov_vello::{Engine, Next, Offscreen, Vello, VelloConfig};

/// sRGB transfer-function encode.
fn srgb_encode(c: f64) -> f64 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055f64.mul_add(c.powf(1.0 / 2.4), -0.055)
    }
}

/// sRGB transfer-function decode.
fn srgb_decode(e: f64) -> f64 {
    if e <= 0.04045 {
        e / 12.92
    } else {
        ((e + 0.055) / 1.055).powf(2.4)
    }
}

/// linear sRGB → linear Display P3 (inverse of the front end's matrix).
const LINEAR_SRGB_TO_LINEAR_P3: [[f64; 3]; 3] = [
    [0.822_461_96, 0.177_538_04, 0.0],
    [0.033_194_2, 0.966_805_8, 0.0],
    [0.017_082_632, 0.072_397_44, 0.910_519_96],
];

const LINEAR_P3_TO_LINEAR_SRGB: [[f64; 3]; 3] = [
    [1.224_940_2, -0.224_940_18, 0.0],
    [-0.042_056_955, 1.042_056_9, 0.0],
    [-0.019_637_555, -0.078_636_04, 1.098_273_6],
];

fn mat_vec(m: &[[f64; 3]; 3], [x, y, z]: [f64; 3]) -> [f64; 3] {
    let dot = |row: &[f64; 3]| row[2].mul_add(z, row[1].mul_add(y, row[0] * x));
    [dot(&m[0]), dot(&m[1]), dot(&m[2])]
}

/// The expected stored-then-decoded pixel of straight working-space colour
/// `src` composited src-over onto `dst` (also straight working space).
///
/// Vello renders into the sRGB-encoded 8-bit target and blends in that
/// encoded space, so the model is: straight linear P3 → linear sRGB → sRGB
/// encode → premultiply → src-over → quantize to u8 → decode → linear P3.
fn expected_pixel(src: [f64; 4], dst: [f64; 4]) -> [f64; 4] {
    let to_premul_encoded = |[r, g, b, a]: [f64; 4]| {
        let [r, g, b] = mat_vec(&LINEAR_P3_TO_LINEAR_SRGB, [r, g, b]);
        [
            srgb_encode(r.clamp(0.0, 1.0)) * a,
            srgb_encode(g.clamp(0.0, 1.0)) * a,
            srgb_encode(b.clamp(0.0, 1.0)) * a,
            a,
        ]
    };
    let s = to_premul_encoded(src);
    let d = to_premul_encoded(dst);
    let s3 = s[3];
    let over = |s: f64, d: f64| (1.0 - s3).mul_add(d, s);
    let out = [
        over(s[0], d[0]),
        over(s[1], d[1]),
        over(s[2], d[2]),
        over(s3, d[3]),
    ];
    // Quantize each encoded channel to u8, decode, convert to linear P3.
    let enc: Vec<f64> = out
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() / 255.0)
        .collect();
    let lin = [
        srgb_decode(enc[0]),
        srgb_decode(enc[1]),
        srgb_decode(enc[2]),
    ];
    let p3 = mat_vec(&LINEAR_SRGB_TO_LINEAR_P3, lin);
    [p3[0], p3[1], p3[2], enc[3]]
}

fn engine() -> Option<Engine<Vello>> {
    match Engine::<Vello>::new(VelloConfig::default()) {
        Ok(engine) => Some(engine),
        Err(e) => {
            eprintln!("no adapter, skipping ({e})");
            None
        }
    }
}

fn assert_pixel(actual: [f32; 4], expected: [f64; 4], what: &str) {
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (f64::from(*a) - e).abs() <= 2.0 / 255.0 + 1e-4,
            "{what}: {actual:?} != {expected:?}"
        );
    }
}

fn px(readback: &cherenkov_vello::Readback, x: u32, y: u32) -> [f32; 4] {
    readback.pixels[(y * readback.width + x) as usize]
}

#[test]
fn solid_fill_matches_expected_pixels() {
    let Some(engine) = engine() else { return };
    let surface = engine.surface(Offscreen::new((64, 64))).expect("surface");
    surface.clear_color(WorkingColor::TRANSPARENT);
    // A P3 colour: mostly red.
    let fill = WorkingColor::new([0.9, 0.1, 0.2, 1.0]);
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| c.fill(Rect::new(8., 8., 40., 40.), fill)));
    });
    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("render");
    assert_eq!(next, Next::Idle);
    let readback = surface.readback().expect("readback");
    // Centre pixel: the fill colour through the full conversion pipeline.
    assert_pixel(
        px(&readback, 24, 24),
        expected_pixel(fill.components.map(f64::from), [0., 0., 0., 0.]),
        "centre",
    );
    // Corner pixel: untouched clear colour.
    assert_pixel(
        px(&readback, 0, 0),
        expected_pixel([0., 0., 0., 0.], [0., 0., 0., 0.]),
        "corner",
    );
    // Just outside the fill.
    assert_pixel(
        px(&readback, 41, 24),
        expected_pixel([0., 0., 0., 0.], [0., 0., 0., 0.]),
        "outside",
    );
}

#[test]
fn layer_tree_composes_transform_opacity_clip() {
    let Some(engine) = engine() else { return };
    let surface = engine.surface(Offscreen::new((64, 64))).expect("surface");
    surface.clear_color(WorkingColor::BLACK);

    let moved = surface.layer();
    let clipped = surface.layer();

    let white = surface.record(|c| {
        c.fill(Rect::new(0., 0., 20., 20.), WorkingColor::WHITE);
    });
    let wide = surface.record(|c| {
        c.fill(Rect::new(0., 0., 64., 64.), WorkingColor::WHITE);
    });

    surface.update(|tx| {
        // `moved`: translate (10,10), opacity 0.5, white square.
        tx[&moved]
            .transform(Affine::translate((10., 10.)))
            .opacity(0.5)
            .content(white);
        // `clipped`: clip to Rect(0,0,20,20), full-width white fill.
        tx[&clipped].clip(Rect::new(0., 0., 20., 20.)).content(wide);
        tx[surface.root()].push(&moved).push(&clipped);
    });

    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("render");
    assert_eq!(next, Next::Idle);
    let readback = surface.readback().expect("readback");

    let half = expected_pixel(
        [1., 1., 1., 0.5],
        WorkingColor::BLACK.components.map(f64::from),
    );
    let full = expected_pixel(
        WorkingColor::WHITE.components.map(f64::from),
        WorkingColor::BLACK.components.map(f64::from),
    );
    let clear = expected_pixel(
        [0., 0., 0., 0.],
        WorkingColor::BLACK.components.map(f64::from),
    );
    // `moved` covers (10..30, 10..30); `clipped` covers (0..20, 0..20).
    // Inside both (15,15): clipped white under 0.5-opacity white = white.
    assert_pixel(px(&readback, 15, 15), full, "inside both layers");
    // Inside `clipped` only: full white.
    assert_pixel(px(&readback, 5, 5), full, "clipped white pixel");
    // Inside `moved` but outside `clipped` (25,25): 0.5 white over black.
    assert_pixel(px(&readback, 25, 25), half, "moved-only pixel");
    // Outside both: untouched black clear.
    assert_pixel(px(&readback, 50, 50), clear, "cleared pixel");
}

/// The expected working-space pixel of stored sRGB-encoded texel `enc`
/// (values in 0..1) composited src-over onto `dst` (straight working P3):
/// quantize, then treat the encoded premultiplied value as the source.
#[expect(
    clippy::many_single_char_names,
    reason = "channel names r/g/b/a are the clearest here"
)]
fn expected_stored(enc: [f64; 4], dst: [f64; 4]) -> [f64; 4] {
    let s: Vec<f64> = enc
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() / 255.0)
        .collect();
    let d: Vec<f64> = {
        let [r, g, b, a] = dst;
        let [r, g, b] = mat_vec(&LINEAR_P3_TO_LINEAR_SRGB, [r, g, b]);
        [
            srgb_encode(r.clamp(0.0, 1.0)) * a,
            srgb_encode(g.clamp(0.0, 1.0)) * a,
            srgb_encode(b.clamp(0.0, 1.0)) * a,
            a,
        ]
        .into_iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() / 255.0)
        .collect()
    };
    let s3 = s[3];
    let over = |s: f64, d: f64| (1.0 - s3).mul_add(d, s);
    let out = [
        over(s[0], d[0]),
        over(s[1], d[1]),
        over(s[2], d[2]),
        over(s3, d[3]),
    ];
    let lin = [
        srgb_decode(out[0]),
        srgb_decode(out[1]),
        srgb_decode(out[2]),
    ];
    let p3 = mat_vec(&LINEAR_SRGB_TO_LINEAR_P3, lin);
    [p3[0], p3[1], p3[2], out[3]]
}

/// `GpuContent` that clears its texture to a constant colour.
struct ClearContent(wgpu::Color);

use cherenkov_vello::interop::wgpu;

impl cherenkov_vello::GpuContent for ClearContent {
    async fn setup(&mut self, _gpu: &wgpu::Context<'_>) {}

    fn render(&mut self, frame: &mut wgpu::Frame<'_>) {
        let mut encoder = frame
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test clear"),
            });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.0),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        frame.queue.submit([encoder.finish()]);
    }
}

#[test]
fn gpu_content_composites_as_an_image() {
    let Some(engine) = engine() else { return };
    let surface = engine.surface(Offscreen::new((64, 64))).expect("surface");
    surface.clear_color(WorkingColor::TRANSPARENT);
    // Opaque colour, so straight == premultiplied in the stored texel.
    let enc = [0.9, 0.1, 0.2];
    let content = engine.gpu_content(
        (20, 20),
        ClearContent(wgpu::Color {
            r: enc[0],
            g: enc[1],
            b: enc[2],
            a: 1.0,
        }),
    );
    let redraw = content.redraw_handle();
    let layer = surface.layer();
    surface.update(|tx| {
        tx[&layer]
            .transform(Affine::translate((10., 10.)))
            .content(content);
        tx[surface.root()].push(&layer);
    });
    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("render");
    assert_eq!(next, Next::Idle);
    let readback = surface.readback().expect("readback");
    assert_pixel(
        px(&readback, 15, 15),
        expected_stored([enc[0], enc[1], enc[2], 1.0], [0., 0., 0., 0.]),
        "content pixel",
    );
    assert_pixel(
        px(&readback, 5, 5),
        expected_stored([0., 0., 0., 0.], [0., 0., 0., 0.]),
        "outside content",
    );
    // Nothing changed: second frame is idle.
    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("render");
    assert_eq!(next, Next::Idle);
    // A redraw request re-renders the content and schedules a frame.
    redraw.request_redraw();
    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("render");
    assert!(matches!(next, Next::At { .. }), "{next:?}");
    assert!(!redraw.is_dirty());
}

#[test]
fn shader_paint_is_sampled_as_an_image() {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "test colours are exact in f32"
    )]
    fn uniforms(enc: [f64; 3]) -> Vec<f32> {
        vec![enc[0] as f32, enc[1] as f32, enc[2] as f32, 1.0]
    }
    let Some(engine) = engine() else { return };
    let shader = engine
        .shader(cherenkov_vello::ShaderSource::wgsl(
            "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { return vec4<f32>(params[0].rgb, params[0].w); }",
        ))
        .expect("shader");
    let animated = engine
        .shader(
            cherenkov_vello::ShaderSource::wgsl(
                "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { return vec4<f32>(params[0].rgb, params[0].w); }",
            )
            .animated(),
        )
        .expect("animated shader");
    let surface = engine.surface(Offscreen::new((64, 64))).expect("surface");
    surface.clear_color(WorkingColor::TRANSPARENT);
    let enc = [0.9, 0.1, 0.2];
    let paint = cherenkov::ShaderPaint {
        shader: shader.id(),
        uniforms: uniforms(enc),
    };
    let anim_paint = cherenkov::ShaderPaint {
        shader: animated.id(),
        uniforms: uniforms(enc),
    };
    let anim_layer = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(Rect::new(8., 8., 40., 40.), paint);
        }));
        tx[&anim_layer]
            .transform(Affine::translate((32., 32.)))
            .content(surface.record(|c| {
                c.fill(Rect::new(0., 0., 16., 16.), anim_paint);
            }));
        tx[surface.root()].push(&anim_layer);
    });
    // The animated shader keeps the frame refresh alive.
    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("render");
    assert!(matches!(next, Next::At { .. }), "{next:?}");
    let readback = surface.readback().expect("readback");
    assert_pixel(
        px(&readback, 24, 24),
        expected_stored([enc[0], enc[1], enc[2], 1.0], [0., 0., 0., 0.]),
        "shader pixel",
    );
    // Drop the animated layer: the static shader alone leaves the engine idle.
    drop(anim_layer);
    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("render");
    assert_eq!(next, Next::Idle);
}

/// A shader paint inside an in-content `transform` must sample the full
/// ramp across the shape: the texture is mapped onto the shape-local
/// bbox (vello composes `xf * brush_transform`), not the device-space
/// one — that would translate the texture a second time.
#[test]
fn shader_paint_maps_in_shape_space_under_transform() {
    let Some(engine) = engine() else { return };
    // Gray ramp along uv.x.
    let ramp = engine
        .shader(cherenkov_vello::ShaderSource::wgsl(
            "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { return vec4<f32>(uv.x, uv.x, uv.x, 1.0); }",
        ))
        .expect("shader");
    let surface = engine.surface(Offscreen::new((64, 64))).expect("surface");
    surface.clear_color(WorkingColor::BLACK);
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.transform(Affine::translate((16., 16.)), |c| {
                c.fill(
                    Rect::new(0., 0., 32., 32.),
                    cherenkov::ShaderPaint {
                        shader: ramp.id(),
                        uniforms: vec![],
                    },
                );
            });
        }));
    });
    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("render");
    assert_eq!(next, Next::Idle);
    let readback = surface.readback().expect("readback");
    let left = px(&readback, 18, 32);
    let right = px(&readback, 46, 32);
    assert!(
        left[0] < 0.05 && left[1] < 0.05 && left[2] < 0.05,
        "ramp start should be near black: {left:?}"
    );
    assert!(
        right[0] > 0.6 && right[1] > 0.6 && right[2] > 0.6,
        "ramp end should be near white: {right:?}"
    );
}

#[test]
fn bad_wgsl_returns_resource_error() {
    let Some(engine) = engine() else { return };
    let result = engine.shader(cherenkov_vello::ShaderSource::wgsl(
        "this is not valid wgsl {{{",
    ));
    assert!(
        matches!(result, Err(cherenkov_vello::ResourceError::Shader(_))),
        "{result:?}"
    );
}

#[test]
fn mesh_paint_reports_unsupported() {
    let Some(engine) = engine() else { return };
    let surface = engine.surface(Offscreen::new((64, 64))).expect("surface");
    let mesh = cherenkov::MeshGradient::new(
        1,
        1,
        vec![
            cherenkov::kurbo::Point::new(0., 0.),
            cherenkov::kurbo::Point::new(64., 0.),
            cherenkov::kurbo::Point::new(0., 64.),
            cherenkov::kurbo::Point::new(64., 64.),
        ],
        vec![WorkingColor::WHITE; 4],
    );
    surface.update(|tx| {
        tx[surface.root()].content(surface.record(|c| {
            c.fill(Rect::new(0., 0., 64., 64.), mesh);
        }));
    });
    let result = engine.render(cherenkov_vello::FrameTime::now());
    assert!(
        matches!(
            result,
            Err(cherenkov_vello::RenderError::Unsupported(
                cherenkov_vello::Unsupported::MeshGradient
            ))
        ),
        "{result:?}"
    );
}

#[test]
fn filter_runs_over_the_layer_texture() {
    let Some(engine) = engine() else { return };
    let filter = engine.filter(filtrate::filters::Invert);
    let surface = engine.surface(Offscreen::new((64, 64))).expect("surface");
    surface.clear_color(WorkingColor::BLACK);
    let fill = WorkingColor::new([0.9, 0.1, 0.2, 1.0]);
    let layer = surface.layer();
    surface.update(|tx| {
        tx[&layer]
            .content(surface.record(|c| {
                c.fill(Rect::new(0., 0., 64., 64.), fill);
            }))
            .filter(&filter)
            .opacity(0.5);
        tx[surface.root()].push(&layer);
    });
    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("render");
    let readback = surface.readback().expect("readback");
    // Invert is `a - rgb` on premultiplied values in the space the
    // executor samples — our encoded texels. So the stored output is
    // `1 - enc` per channel, then the layer's 0.5 opacity scales the
    // premultiplied source over black.
    let [r, g, b, a] = fill.components.map(f64::from);
    let lin = mat_vec(&LINEAR_P3_TO_LINEAR_SRGB, [r, g, b]);
    let enc = [
        srgb_encode(lin[0].clamp(0.0, 1.0)) * a,
        srgb_encode(lin[1].clamp(0.0, 1.0)) * a,
        srgb_encode(lin[2].clamp(0.0, 1.0)) * a,
        a,
    ];
    let inverted = [1.0 - enc[0], 1.0 - enc[1], 1.0 - enc[2], enc[3]];
    assert_pixel(
        px(&readback, 32, 32),
        expected_stored(
            inverted.map(|v| v * 0.5),
            WorkingColor::BLACK.components.map(f64::from),
        ),
        "inverted pixel at 0.5 opacity",
    );
    assert_eq!(next, Next::Idle);
}

/// A layer `clip` must honour the shape's fill rule: a same-winding
/// rect-in-rect path under `EvenOdd` punches a hole in the clipped
/// content, where non-zero winding would leave the centre filled.
#[test]
fn even_odd_layer_clip_punches_a_hole() {
    let Some(engine) = engine() else { return };
    let surface = engine.surface(Offscreen::new((64, 64))).expect("surface");
    surface.clear_color(WorkingColor::BLACK);
    let mut path = cherenkov::kurbo::BezPath::new();
    path.move_to((0., 0.));
    path.line_to((64., 0.));
    path.line_to((64., 64.));
    path.line_to((0., 64.));
    path.close_path();
    path.move_to((16., 16.));
    path.line_to((48., 16.));
    path.line_to((48., 48.));
    path.line_to((16., 48.));
    path.close_path();
    let clipped = surface.layer();
    surface.update(|tx| {
        tx[&clipped]
            .clip(cherenkov::EvenOdd(path))
            .content(surface.record(|c| {
                c.fill(Rect::new(0., 0., 64., 64.), WorkingColor::WHITE);
            }));
        tx[surface.root()].push(&clipped);
    });
    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("render");
    assert_eq!(next, Next::Idle);
    let readback = surface.readback().expect("readback");
    assert_pixel(
        px(&readback, 8, 8),
        expected_pixel(
            WorkingColor::WHITE.components.map(f64::from),
            WorkingColor::BLACK.components.map(f64::from),
        ),
        "inside outer ring",
    );
    assert_pixel(
        px(&readback, 32, 32),
        expected_pixel(
            WorkingColor::TRANSPARENT.components.map(f64::from),
            WorkingColor::BLACK.components.map(f64::from),
        ),
        "punched hole",
    );
}

/// `GpuContent` that counts its renders and always asks for another frame.
struct LoopingContent(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl cherenkov_vello::GpuContent for LoopingContent {
    async fn setup(&mut self, _gpu: &wgpu::Context<'_>) {}

    fn render(&mut self, frame: &mut wgpu::Frame<'_>) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        frame.request_redraw();
    }
}

/// A `filtrate::Effect` whose `redraw_hint` keeps asking for a frame;
/// `encode_render` itself reports no in-flight animation.
struct AlwaysHinting;

impl filtrate::Effect for AlwaysHinting {
    fn setup(
        &mut self,
        _ctx: &filtrate::EffectContext<'_>,
    ) -> impl std::future::Future<Output = filtrate::EffectSetupResult> {
        std::future::ready(Ok(()))
    }

    fn encode_render(
        &mut self,
        _input: &filtrate::EffectInput<'_>,
        _output: &filtrate::EffectOutput<'_>,
        _encoder: &mut wgpu::CommandEncoder,
    ) -> filtrate::EffectRenderResult {
        Ok(false)
    }

    fn redraw_hint(&self) -> bool {
        true
    }
}

/// An animated shader, a looping `GpuContent` and a `redraw_hint` filter
/// must keep the surface rendering: every `render` answers `Next::At` and
/// the content is genuinely re-rendered each frame.
#[test]
fn animated_content_requests_every_frame() {
    let Some(engine) = engine() else { return };
    let surface = engine.surface(Offscreen::new((64, 64))).expect("surface");
    let animated = engine
        .shader(
            cherenkov_vello::ShaderSource::wgsl(
                "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { return vec4<f32>(uv, 0.0, 1.0); }",
            )
            .animated(),
        )
        .expect("animated shader");
    let renders = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let content = engine.gpu_content((8, 8), LoopingContent(std::sync::Arc::clone(&renders)));
    let filter = engine.effect(AlwaysHinting);
    let shader_layer = surface.layer();
    let gpu_layer = surface.layer();
    let filtered_layer = surface.layer();
    surface.update(|tx| {
        tx[&shader_layer].content(surface.record(|c| {
            c.fill(
                Rect::new(0., 0., 8., 8.),
                cherenkov::ShaderPaint {
                    shader: animated.id(),
                    uniforms: vec![],
                },
            );
        }));
        tx[&gpu_layer]
            .transform(Affine::translate((16., 0.)))
            .content(content);
        tx[&filtered_layer]
            .transform(Affine::translate((32., 0.)))
            .filter(&filter)
            .content(surface.record(|c| {
                c.fill(Rect::new(0., 0., 8., 8.), WorkingColor::WHITE);
            }));
        tx[surface.root()]
            .push(&shader_layer)
            .push(&gpu_layer)
            .push(&filtered_layer);
    });
    for frame in 0..3 {
        let next = engine
            .render(cherenkov_vello::FrameTime::now())
            .expect("render");
        assert!(matches!(next, Next::At { .. }), "frame {frame}: {next:?}");
    }
    let rendered = renders.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        rendered >= 3,
        "content rendered {rendered} times in 3 frames"
    );
}

/// An oversized surface must fail fast with `TooLarge` and leave the
/// engine usable — not panic the render thread and turn every later
/// `render` into `RenderError::Thread`.
#[test]
fn oversized_surface_fails_fast_and_the_engine_survives() {
    let Some(engine) = engine() else { return };
    let err = engine.surface(Offscreen::new((u32::MAX, u32::MAX))).err();
    assert!(
        matches!(err, Some(cherenkov_vello::SurfaceError::TooLarge { .. })),
        "expected TooLarge, got {err:?}"
    );
    // The render thread is still alive and healthy surfaces still render.
    let ok = engine.surface(Offscreen::new((8, 8))).expect("surface");
    ok.clear_color(WorkingColor::BLACK);
    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("render after rejected surface");
    assert_eq!(next, Next::Idle);
    let readback = ok.readback().expect("readback");
    assert_pixel(
        px(&readback, 0, 0),
        expected_pixel(
            [0., 0., 0., 0.],
            WorkingColor::BLACK.components.map(f64::from),
        ),
        "black clear",
    );
}

/// A real X window destroyed mid-flight: the next `render` hits
/// `CurrentSurfaceTexture::Lost`, skips the present, and must report
/// `Next::At` so the host retries — not `Idle`, which would leave the
/// stale frame up until an unrelated change arrived. And a dead window
/// must not panic the render thread.
#[cfg(target_os = "linux")]
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "a real X window lifecycle cannot be shortened without losing the scenario"
)]
fn skipped_window_present_retries_next_frame() {
    use raw_window_handle::{RawDisplayHandle, RawWindowHandle, XcbDisplayHandle, XcbWindowHandle};
    use x11rb::COPY_DEPTH_FROM_PARENT;
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::{ConnectionExt as _, CreateWindowAux, WindowClass};
    use x11rb::xcb_ffi::XCBConnection;

    let Ok((conn, screen_num)) = XCBConnection::connect(None) else {
        eprintln!("no X display, skipping");
        return;
    };
    let screen = &conn.setup().roots[screen_num];
    let window = conn.generate_id().expect("window id");
    conn.create_window(
        COPY_DEPTH_FROM_PARENT,
        window,
        screen.root,
        0,
        0,
        64,
        64,
        0,
        WindowClass::INPUT_OUTPUT,
        screen.root_visual,
        &CreateWindowAux::new(),
    )
    .expect("create_window");
    conn.map_window(window).expect("map");
    conn.flush().expect("flush");

    let raw_display = RawDisplayHandle::Xcb(XcbDisplayHandle::new(
        std::ptr::NonNull::new(conn.get_raw_xcb_connection()),
        i32::try_from(screen_num).expect("screen number fits in i32"),
    ));
    let raw_window = RawWindowHandle::Xcb(XcbWindowHandle::new(
        std::num::NonZero::new(window).expect("window id"),
    ));
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Ok(surface) = (unsafe {
        instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(raw_display),
            raw_window_handle: raw_window,
        })
    }) else {
        eprintln!("no surface support, skipping");
        return;
    };
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        compatible_surface: Some(&surface),
        ..wgpu::RequestAdapterOptions::default()
    })) else {
        eprintln!("no adapter, skipping");
        return;
    };
    let Ok((device, queue)) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
    else {
        eprintln!("no device, skipping");
        return;
    };
    let caps = surface.get_capabilities(&adapter);
    let Some(&format) = caps.formats.first() else {
        eprintln!("no surface formats, skipping");
        return;
    };
    let Some(&alpha_mode) = caps.alpha_modes.first() else {
        eprintln!("no alpha modes, skipping");
        return;
    };
    let engine = Engine::<Vello>::with_device(
        VelloConfig::default(),
        cherenkov_vello::interop::wgpu::DeviceSource::new(adapter, device, queue),
    )
    .expect("engine");
    let surface = engine
        .surface(cherenkov_vello::interop::wgpu::Window {
            surface,
            config: wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: 64,
                height: 64,
                present_mode: wgpu::PresentMode::Fifo,
                desired_maximum_frame_latency: 1,
                alpha_mode,
                view_formats: vec![],
            },
            rate: 30..=120,
        })
        .expect("window surface");
    surface.clear_color(WorkingColor::BLACK);
    let _ = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("first render");

    // Destroying the window makes the next acquire report `Lost` — a
    // skipped present.
    conn.destroy_window(window).expect("destroy_window");
    conn.flush().expect("flush");
    // Round-trip so the server has processed the destruction before the
    // render thread acquires.
    let _ = conn.get_input_focus().expect("cookie").reply();
    surface.clear_color(WorkingColor::WHITE);
    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("second render");
    match next {
        Next::At { rate, .. } => assert_eq!(rate, 30..=120),
        other @ Next::Idle => panic!("skipped present must retry next frame, got {other:?}"),
    }
    // The retry keeps asking until the embedder destroys the surface.
    surface.clear_color(WorkingColor::BLACK);
    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("retry render");
    match next {
        Next::At { rate, .. } => assert_eq!(rate, 30..=120),
        other @ Next::Idle => panic!("a dead window must keep retrying, got {other:?}"),
    }
    drop(surface);
    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("render after destroy");
    assert_eq!(next, Next::Idle);
}

/// `Next::At` must report the refresh range the surface's display
/// supports — the caller-configured range for an offscreen target —
/// and a deadline at the fastest end of it, not a hard-coded 60 Hz.
#[test]
fn next_at_reports_the_configured_refresh_range() {
    let Some(engine) = engine() else { return };
    let surface = engine
        .surface(Offscreen::new((8, 8)).rate(30..=144))
        .expect("surface");
    let animated = engine
        .shader(
            cherenkov_vello::ShaderSource::wgsl(
                "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { return vec4<f32>(uv, 0.0, 1.0); }",
            )
            .animated(),
        )
        .expect("animated shader");
    let layer = surface.layer();
    surface.update(|tx| {
        tx[&layer].content(surface.record(|c| {
            c.fill(
                Rect::new(0., 0., 8., 8.),
                cherenkov::ShaderPaint {
                    shader: animated.id(),
                    uniforms: vec![],
                },
            );
        }));
        tx[surface.root()].push(&layer);
    });
    let t0 = std::time::Instant::now();
    let next = engine
        .render(cherenkov_vello::FrameTime::at(t0))
        .expect("render");
    match next {
        Next::At { time, rate } => {
            assert_eq!(rate, 30..=144);
            assert_eq!(
                time,
                t0 + std::time::Duration::from_secs_f64(1.0 / 144.0),
                "deadline must be one tick at the fastest supported rate"
            );
        }
        other @ Next::Idle => panic!("expected At, got {other:?}"),
    }
}

/// An oversized `Surface::resize` must fail fast with `TooLarge` and
/// keep the old size — not report `Ok` while the render thread silently
/// ignores it.
#[test]
fn oversized_resize_fails_fast() {
    let Some(engine) = engine() else { return };
    let mut surface = engine.surface(Offscreen::new((8, 8))).expect("surface");
    let err = surface.resize((u32::MAX, 16)).err();
    assert!(
        matches!(err, Some(cherenkov_vello::SurfaceError::TooLarge { .. })),
        "expected TooLarge, got {err:?}"
    );
    assert_eq!(
        surface.size(),
        (8, 8),
        "a rejected resize keeps the old size"
    );
    // A legal resize still applies and the engine still renders.
    surface.resize((16, 16)).expect("resize");
    surface.clear_color(WorkingColor::BLACK);
    let next = engine
        .render(cherenkov_vello::FrameTime::now())
        .expect("render after resizes");
    assert_eq!(next, Next::Idle);
    let readback = surface.readback().expect("readback");
    assert_eq!(readback.width, 16);
    assert_eq!(readback.height, 16);
}
