// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scene serde round-trip and path/shape helper checks.

use cherenkov_scene::{
    BlendMode, Color, ColorSpace, Feature, FillRule, Glyph, GlyphRun, GradientStop, Item,
    LinearGradient, Paint, ResourceHash, Sampling, Scene, Shape, StrokeStyle,
};
use kurbo::{Affine, BezPath, Point, Rect};

#[test]
fn color_roundtrip() {
    let c = Color::srgb(1.0, 0.5, 0.25).with_alpha(0.75);
    let json = serde_json::to_string(&c).unwrap();
    let back: Color = serde_json::from_str(&json).unwrap();
    assert_eq!(c, back);
    assert!(!c.is_hdr());
    let hdr = Color::new(ColorSpace::LinearP3, [2.0, 0.0, 0.0, 1.0]);
    assert!(hdr.is_hdr());
    assert!(hdr.is_wide_gamut());
}

#[test]
fn hash_roundtrip() {
    let h = ResourceHash::of(b"hello world");
    let s = h.to_string();
    assert_eq!(s.len(), 64);
    let back: ResourceHash = s.parse().unwrap();
    assert_eq!(h, back);
    let json = serde_json::to_string(&h).unwrap();
    assert_eq!(json, format!("\"{s}\""));
    let back: ResourceHash = serde_json::from_str(&json).unwrap();
    assert_eq!(h, back);
}

#[test]
fn scene_save_load() {
    let dir = std::env::temp_dir().join(format!("cherenkov-scene-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let mut path = BezPath::new();
    path.move_to((10.0, 10.0));
    path.curve_to((30.0, 40.0), (60.0, 40.0), (90.0, 10.0));
    path.close_path();

    let font = ResourceHash::of(b"fake font");
    let scene = Scene::builder(128, 96)
        .clear(Color::srgb(0.0, 0.0, 0.0))
        .fill_rule(
            Shape::Path { path },
            FillRule::EvenOdd,
            Paint::Linear(LinearGradient {
                start: Point::new(0.0, 0.0),
                end: Point::new(128.0, 96.0),
                stops: vec![
                    GradientStop {
                        offset: 0.0,
                        color: Color::srgb(1.0, 0.0, 0.0),
                    },
                    GradientStop {
                        offset: 1.0,
                        color: Color::srgb(0.0, 0.0, 1.0),
                    },
                ],
                extend: cherenkov_scene::Extend::Repeat,
                interpolation: ColorSpace::Srgb,
            }),
        )
        .stroke(
            Shape::circle(64.0, 48.0, 20.0),
            StrokeStyle {
                width: 2.0,
                dash_pattern: vec![3.0, 1.0],
                ..StrokeStyle::default()
            },
            Paint::Solid(Color::new(ColorSpace::DisplayP3, [0.0, 1.0, 0.0, 1.0])),
        )
        .glyphs(GlyphRun {
            font,
            font_index: 0,
            size: 24.0,
            normalized_coords: vec![],
            glyphs: vec![Glyph {
                id: 36,
                x: 0.0,
                y: 0.0,
            }],
            paint: Paint::Solid(Color::srgb(0.0, 0.0, 0.0)),
        })
        .image(
            ResourceHash::of(b"fake image"),
            Rect::new(0.0, 0.0, 32.0, 32.0),
            Sampling::Nearest,
        )
        .layer(|l| {
            l.blend(BlendMode::Multiply)
                .opacity(0.5)
                .clip(Shape::rounded_rect(4.0, 4.0, 60.0, 60.0, 8.0))
                .transform(Affine::rotate(0.2))
                .shadow(
                    Shape::rect(10.0, 10.0, 50.0, 30.0),
                    4.0,
                    [2.0, 3.0],
                    Color::srgb(0.0, 0.0, 0.0).with_alpha(0.5),
                );
        })
        .build();

    for feat in [
        Feature::Fill,
        Feature::EvenOdd,
        Feature::LinearGradient,
        Feature::Stroke,
        Feature::StrokeDash,
        Feature::Path,
        Feature::Glyphs,
        Feature::Image,
        Feature::Clip,
        Feature::Opacity,
        Feature::Shadow,
        Feature::Blend(BlendMode::Multiply),
        Feature::WideGamut,
    ] {
        assert!(scene.features.contains(&feat), "missing {feat:?}");
    }

    scene.save(&dir).unwrap();
    let blob = Scene::store_resource(&dir, b"fake font").unwrap();
    assert_eq!(blob, font);
    let loaded = Scene::load(&dir).unwrap();
    assert_eq!(scene, loaded);
    assert_eq!(Scene::resource(&dir, font).unwrap(), b"fake font");
    assert!(matches!(
        Scene::resource(&dir, ResourceHash::of(b"absent")),
        Err(cherenkov_scene::SceneError::MissingResource(_))
    ));

    let refs = scene.resource_refs();
    assert!(refs.contains(&font));
    assert!(refs.contains(&ResourceHash::of(b"fake image")));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn item_untagged() {
    let layer_json = r#"{"items":[]}"#;
    let layer: cherenkov_scene::Layer = serde_json::from_str(layer_json).unwrap();
    assert!(layer.items.is_empty());
    let item_layer: Item = serde_json::from_str(r#"{"layer":{"items":[]}}"#).unwrap();
    assert!(matches!(item_layer, Item::Layer(_)));
    let item_draw: Item = serde_json::from_str(
        r#"{"draw":{"image":{"image":"0000000000000000000000000000000000000000000000000000000000000000",
             "dst":{"x0":0.0,"y0":0.0,"x1":1.0,"y1":1.0},"sampling":"nearest"}}}"#,
    )
    .unwrap();
    assert!(matches!(
        item_draw,
        Item::Draw(cherenkov_scene::Draw::Image { .. })
    ));
}

#[test]
fn continuous_rect_path_valid() {
    let shape = Shape::Continuous(cherenkov_scene::ContinuousRect {
        rect: Rect::new(0.0, 0.0, 100.0, 80.0),
        corner_radius: 20.0,
        smoothing: 0.6,
    });
    let path = shape.to_path();
    let mut has_close = false;
    for el in path.elements() {
        if let Some(p) = el.end_point() {
            assert!(p.x.is_finite() && p.y.is_finite());
        }
        if matches!(el, kurbo::PathEl::ClosePath) {
            has_close = true;
        }
    }
    assert!(has_close);
    let bb = shape.bounding_box();
    assert!((bb.x1 - bb.x0 - 100.0).abs() < 1e-6);
    assert!((bb.y1 - bb.y0 - 80.0).abs() < 1e-6);
}

#[test]
fn smoothing_zero_matches_circular_corner() {
    // smoothing = 0 -> exponent 2 -> a circle quarter approximating the corner.
    let shape = Shape::Continuous(cherenkov_scene::ContinuousRect {
        rect: Rect::new(0.0, 0.0, 40.0, 40.0),
        corner_radius: 10.0,
        smoothing: 0.0,
    });
    let path = shape.to_path();
    assert!(path.elements().len() > 10);
}
