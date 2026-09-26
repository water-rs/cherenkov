// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! End-to-end CPU effects compared with independent oracle entry points.

use cherenkov::{
    BlendMode, BlendSpace, Draw, Glyph, GlyphRun, GlyphStyle, Group, MeshGradient, Shadow,
    WorkingColor,
};
use cherenkov_cpu::{
    Engine, FontSource, FrameTime, Offscreen, OffscreenFormat, Raster, RasterConfig,
};
use cherenkov_oracle::{coverage::Coverage, path, shadow};
use cherenkov_scene::FillRule;
use kurbo::{Affine, BezPath, Point, Rect, Stroke};

const SIZE: usize = 48;
const COLOR: WorkingColor = WorkingColor::new([0.75, 0.25, 1.25, 0.5]);

fn engine() -> Engine<Raster> {
    Engine::new(RasterConfig {
        threads: Some(2),
        ..RasterConfig::default()
    })
    .expect("engine")
}

fn render(engine: &Engine<Raster>, draw: impl FnOnce(&mut cherenkov::Recorder)) -> Vec<[f32; 4]> {
    let surface = engine
        .surface(Offscreen::new((48, 48), OffscreenFormat::LinearF32))
        .expect("surface");
    surface
        .update(|tx| {
            tx[surface.root()].content(surface.record(draw));
        })
        .expect("update");
    engine.render(FrameTime::now()).expect("render");
    surface.readback().expect("readback").pixels
}

fn polygon(points: &[(f64, f64)]) -> BezPath {
    let mut outline = BezPath::new();
    outline.move_to(points[0]);
    for &point in &points[1..] {
        outline.line_to(point);
    }
    outline.close_path();
    outline
}

fn assert_pixels(actual: &[[f32; 4]], expected: &[[f64; 4]], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        for channel in 0..4 {
            assert!(
                (f64::from(actual[channel]) - expected[channel]).abs() < tolerance,
                "pixel {index}, channel {channel}: {actual:?} versus {expected:?}"
            );
        }
    }
}

fn colored(field: &[f64], color: WorkingColor) -> Vec<[f64; 4]> {
    let [red, green, blue, alpha] = color.components.map(f64::from);
    field
        .iter()
        .map(|&area| {
            [
                red * alpha * area,
                green * alpha * area,
                blue * alpha * area,
                alpha * area,
            ]
        })
        .collect()
}

#[test]
fn arbitrary_shadow_rotation_offset_clips_and_cache_match_oracle() {
    let engine = engine();
    let caster = polygon(&[
        (4.125, 5.25),
        (31.75, 29.875),
        (3.25, 30.125),
        (32.5, 3.125),
    ]);
    let clips = [
        polygon(&[(0.25, 0.5), (44.75, 4.25), (8.75, 43.5)]),
        polygon(&[(1.75, 42.5), (9.125, 0.25), (45.25, 41.5)]),
    ];
    let boundaries: Vec<_> = clips
        .iter()
        .map(|clip| path::edges(&path::flatten(clip)))
        .collect();
    let transform = Affine::translate((7.0, 1.0)) * Affine::rotate(0.25);
    let offset = (0.375, 1.125);
    let coverage = shadow::spread_coverage(
        &cherenkov_scene::Shape::Path {
            path: caster.clone(),
        },
        FillRule::NonZero,
        transform * Affine::translate(offset),
        0.0,
        &boundaries,
        (SIZE, SIZE),
    );
    for sigma in [0.0, 0.125, 2.25] {
        let expected = colored(
            &shadow::gaussian_blur(&coverage, SIZE, SIZE, sigma, transform),
            COLOR,
        );
        for _ in 0..2 {
            let actual = render(&engine, |recorder| {
                recorder.clip(clips[0].clone(), |recorder| {
                    recorder.clip(clips[1].clone(), |recorder| {
                        recorder.transform(transform, |recorder| {
                            recorder
                                .shadow(caster.clone(), Shadow::new(sigma, COLOR).offset(offset));
                        });
                    });
                });
            });
            assert_pixels(&actual, &expected, 4e-6);
        }
    }
}

#[test]
fn rotated_box_shadows_and_signed_spread_match_oracle() {
    let engine = engine();
    let rect = Rect::new(8.25, 7.125, 30.5, 28.75);
    let transform = Affine::translate((8.0, 1.0)) * Affine::rotate(0.25);
    for spread in [-1.5, 0.0, 2.25] {
        let coverage = shadow::spread_coverage(
            &cherenkov_scene::Shape::Rect(rect),
            FillRule::NonZero,
            transform,
            spread,
            &[],
            (SIZE, SIZE),
        );
        let expected = colored(
            &shadow::gaussian_blur(&coverage, SIZE, SIZE, 1.25, transform),
            COLOR,
        );
        let actual = render(&engine, |recorder| {
            recorder.transform(transform, |recorder| {
                recorder.shadow(rect, Shadow::new(1.25, COLOR).spread(spread));
            })
        });
        // Sharp box spread stays polygonal under rotation.
        assert_pixels(&actual, &expected, 4e-6);
    }
}

#[test]
fn mesh_sampling_transforms_hdr_alpha_and_outside_match_oracle() {
    let mesh = MeshGradient::new(
        2,
        1,
        vec![
            (2.0, 2.0).into(),
            (18.0, 3.0).into(),
            (35.0, 1.0).into(),
            (4.0, 35.0).into(),
            (22.0, 30.0).into(),
            (33.0, 38.0).into(),
        ],
        vec![
            COLOR,
            WorkingColor::new([9.0, 0.0, 0.0, 0.0]),
            WorkingColor::WHITE,
            WorkingColor::new([-0.25, 2.0, 0.0, 0.75]),
            COLOR,
            WorkingColor::WHITE,
        ],
    );
    let transform = Affine::translate((1.25, 2.5)) * Affine::rotate(-0.125);
    let actual = render(&engine(), |recorder| {
        recorder.transform(transform, |recorder| {
            recorder.fill(Rect::new(-100.0, -100.0, 100.0, 100.0), mesh.clone());
        })
    });
    let expected: Vec<_> = (0..48_u32)
        .flat_map(|y| (0..48_u32).map(move |x| (x, y)))
        .map(|(x, y)| {
            cherenkov_oracle::mesh::sample(
                &mesh,
                transform.inverse() * Point::new(f64::from(x) + 0.5, f64::from(y) + 0.5),
            )
        })
        .collect();
    assert_pixels(&actual, &expected, 2e-6);
}

#[test]
fn encoded_groups_apply_opacity_before_compositing() {
    let engine = engine();
    let backdrop = WorkingColor::new([-0.1, 0.4, 1.5, 0.75]);
    for (mode, reference) in [
        (BlendMode::Normal, cherenkov_scene::BlendMode::Normal),
        (BlendMode::Multiply, cherenkov_scene::BlendMode::Multiply),
        (BlendMode::Src, cherenkov_scene::BlendMode::Src),
        (BlendMode::DestIn, cherenkov_scene::BlendMode::DestIn),
    ] {
        let actual = render(&engine, |recorder| {
            recorder.fill(Rect::new(0.0, 0.0, 48.0, 48.0), backdrop);
            recorder.group(
                Group::new()
                    .blend(mode)
                    .blend_space(BlendSpace::SrgbEncoded)
                    .opacity(0.625),
                |recorder| recorder.fill(Rect::new(0.0, 0.0, 48.0, 48.0), COLOR),
            );
        });
        let source = colored(&[0.625], COLOR)[0];
        let destination = colored(&[1.0], backdrop)[0];
        let expected = cherenkov_oracle::blend::in_space(
            reference,
            BlendSpace::SrgbEncoded,
            destination,
            source,
        );
        assert_pixels(&actual, &vec![expected; SIZE * SIZE], 2e-5);
    }
}

#[test]
fn encoded_layer_matches_an_encoded_group() {
    let engine = engine();
    let surface = engine
        .surface(Offscreen::new((48, 48), OffscreenFormat::LinearF32))
        .expect("surface");
    let child = surface.layer();
    surface
        .update(|tx| {
            tx[surface.root()]
                .content(surface.record(|recorder| {
                    recorder.fill(Rect::new(0.0, 0.0, 48.0, 48.0), WorkingColor::WHITE);
                }))
                .push(&child);
            tx[&child]
                .blend_space(BlendSpace::SrgbEncoded)
                .opacity(0.5)
                .content(
                    surface
                        .record(|recorder| recorder.fill(Rect::new(0.0, 0.0, 48.0, 48.0), COLOR)),
                );
        })
        .expect("update");
    engine.render(FrameTime::now()).expect("render");
    let expected = cherenkov_oracle::blend::in_space(
        cherenkov_scene::BlendMode::Normal,
        BlendSpace::SrgbEncoded,
        [1.0; 4],
        colored(&[0.5], COLOR)[0],
    );
    assert_pixels(
        &surface.readback().expect("readback").pixels,
        &vec![expected; SIZE * SIZE],
        2e-5,
    );
}

#[test]
fn transformed_stroked_glyphs_and_cache_identity_match_oracle() {
    let engine = engine();
    let data = include_bytes!("../../scenes/fonts/NotoSans.ttf");
    let font = engine.font(FontSource::bytes(data.to_vec())).expect("font");
    let mut run = GlyphRun {
        font: font.id(),
        size: 30.0,
        coords: Vec::new(),
        glyphs: vec![Glyph {
            id: 36,
            x: 17.125,
            y: 33.75,
            transform: Some(Affine::new([0.75, 0.25, -0.25, 1.0, 1.125, -0.25])),
        }],
        style: GlyphStyle::Fill,
    };
    let clip = polygon(&[(1.25, 1.25), (45.75, 5.125), (30.25, 46.75)]);
    let clip_edges = path::edges(&path::flatten(&clip));
    for style in [
        GlyphStyle::Fill,
        GlyphStyle::Stroke(Stroke::new(1.5).with_join(kurbo::Join::Miter)),
        GlyphStyle::Stroke(
            Stroke::new(2.25)
                .with_join(kurbo::Join::Round)
                .with_caps(kurbo::Cap::Round)
                .with_dashes(0.5, [2.0, 1.0]),
        ),
        GlyphStyle::Fill,
    ] {
        run.style = style;
        let outline = cherenkov_oracle::glyphs::styled_outline(
            data,
            0,
            &run,
            &run.glyphs[0],
            Affine::IDENTITY,
        )
        .expect("oracle outline");
        let edges = cherenkov_oracle::clip::intersect_edges(
            &path::edges(&path::flatten(&outline)),
            FillRule::NonZero,
            &clip_edges,
        );
        let mut coverage = Coverage::new(SIZE, SIZE);
        for (x0, y0, x1, y1) in edges {
            coverage.add_line(x0, y0, x1, y1);
        }
        let expected = colored(&coverage.finish(FillRule::NonZero), COLOR);
        let actual = render(&engine, |recorder| {
            recorder.clip(clip.clone(), |recorder| recorder.glyphs(&run, COLOR))
        });
        assert_pixels(&actual, &expected, 0.025);
    }
}

#[test]
fn a_transformed_colour_glyph_matches_the_oracle_paint_graph() {
    let engine = engine();
    let data = include_bytes!("../../scenes/fonts/Nabla.ttf");
    let font = engine
        .font(FontSource::bytes(data.to_vec()))
        .expect("colour font");
    let local = Affine::rotate(0.125);
    let glyph = Glyph {
        id: 1,
        x: 13.25,
        y: 35.5,
        transform: Some(local),
    };
    let run = GlyphRun {
        font: font.id(),
        size: 30.0,
        coords: Vec::new(),
        glyphs: vec![glyph],
        style: GlyphStyle::Fill,
    };
    let actual = render(&engine, |recorder| recorder.glyphs(&run, COLOR));
    let mut scene = cherenkov_scene::Scene::new(
        48,
        48,
        cherenkov_scene::Color::new(cherenkov_scene::ColorSpace::LinearP3, [0.0; 4]),
    );
    scene.root.transform = Affine::translate((f64::from(glyph.x), f64::from(glyph.y))) * local;
    scene
        .root
        .items
        .push(cherenkov_scene::Item::Draw(cherenkov_scene::Draw::Glyphs(
            cherenkov_scene::GlyphRun {
                font: cherenkov_scene::ResourceHash::of(data),
                font_index: 0,
                size: 30.0,
                normalized_coords: Vec::new(),
                glyphs: vec![cherenkov_scene::Glyph {
                    id: 1,
                    x: 0.0,
                    y: 0.0,
                }],
                paint: cherenkov_scene::Color::new(
                    cherenkov_scene::ColorSpace::LinearP3,
                    COLOR.components,
                )
                .into(),
            },
        )));
    let reference = cherenkov_oracle::render::Renderer::new(SIZE, SIZE)
        .render(&scene, std::path::Path::new("../scenes/corpus/text-colr"))
        .expect("oracle colour glyph");
    let expected: Vec<_> = reference
        .pixels
        .iter()
        .map(|pixel| pixel.map(f64::from))
        .collect();
    assert_pixels(&actual, &expected, 0.04);
}

#[test]
fn root_layer_state_uses_the_same_compositor_as_child_layers() {
    let engine = engine();
    let surface = engine
        .surface(Offscreen::new((48, 48), OffscreenFormat::LinearF32))
        .expect("surface");
    surface
        .update(|tx| {
            tx[surface.root()]
                .blend_space(BlendSpace::SrgbEncoded)
                .opacity(0.5)
                .transform(Affine::translate((2.0, 3.0)))
                .content(
                    surface.record(|recorder| recorder.fill(Rect::new(0.0, 0.0, 1.0, 1.0), COLOR)),
                );
        })
        .expect("update");
    engine.render(FrameTime::now()).expect("render");
    let mut expected = vec![[0.0; 4]; SIZE * SIZE];
    expected[3 * SIZE + 2] = cherenkov_oracle::blend::in_space(
        cherenkov_scene::BlendMode::Normal,
        BlendSpace::SrgbEncoded,
        [0.0; 4],
        colored(&[0.5], COLOR)[0],
    );
    assert_pixels(
        &surface.readback().expect("readback").pixels,
        &expected,
        2e-5,
    );
}

#[test]
fn mixed_radius_box_spread_scales_and_clamps_like_oracle() {
    let engine = engine();
    let rect = kurbo::RoundedRect::from_rect(
        Rect::new(8.0, 8.0, 24.0, 24.0),
        kurbo::RoundedRectRadii::new(0.0, 1.0, 3.0, 6.0),
    );
    let transforms = [
        Affine::scale_non_uniform(1.5, 0.75),
        Affine::new([1.0, 0.25, 0.5, 1.0, 0.0, 0.0]),
    ];
    for transform in transforms {
        for spread in [-8.0, -2.0, 0.0, 2.0] {
            let coverage = shadow::spread_coverage(
                &cherenkov_scene::Shape::RoundedRect(rect),
                FillRule::NonZero,
                transform,
                spread,
                &[],
                (SIZE, SIZE),
            );
            let expected = colored(
                &shadow::gaussian_blur(&coverage, SIZE, SIZE, 0.75, transform),
                COLOR,
            );
            let actual = render(&engine, |recorder| {
                recorder.transform(transform, |recorder| {
                    recorder.shadow(rect, Shadow::new(0.75, COLOR).spread(spread));
                })
            });
            // Rounded outlines retain the coverage compiler's curve tolerance.
            assert_pixels(&actual, &expected, 0.004);
        }
    }
}

#[test]
fn acute_path_spread_uses_miter_limit_four_under_shear_and_clipping() {
    let engine = engine();
    let caster = polygon(&[(12.0, 10.0), (13.0, 25.0), (11.0, 25.0)]);
    let clip = polygon(&[(1.25, 0.5), (40.5, 7.25), (35.75, 43.5), (2.25, 39.75)]);
    let boundaries = [path::edges(&path::flatten(&clip))];
    let transform = Affine::new([1.5, 0.3, 0.25, 0.75, 1.0, 2.0]);
    for spread in [-0.25, 0.0, 2.0] {
        let coverage = shadow::spread_coverage(
            &cherenkov_scene::Shape::Path {
                path: caster.clone(),
            },
            FillRule::NonZero,
            transform,
            spread,
            &boundaries,
            (SIZE, SIZE),
        );
        let expected = colored(
            &shadow::gaussian_blur(&coverage, SIZE, SIZE, 0.25, transform),
            COLOR,
        );
        let actual = render(&engine, |recorder| {
            recorder.clip(clip.clone(), |recorder| {
                recorder.transform(transform, |recorder| {
                    recorder.shadow(caster.clone(), Shadow::new(0.25, COLOR).spread(spread));
                });
            })
        });
        assert_pixels(&actual, &expected, 4e-6);
    }
}
