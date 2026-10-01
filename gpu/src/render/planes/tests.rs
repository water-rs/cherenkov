use kurbo::{Affine, Rect, RoundedRect, Vec2};
use rustc_hash::FxHashMap;

use cherenkov::testing::LayerOp;
use cherenkov::{BlendMode, FilterId, LayerId, Prop, ShapeData, SurfaceTree};

use super::{Compositor, Ineligible, Level, Plan, plan};
use crate::render::lower::axis_aligned;

/// A compositor that carries axis-aligned transforms and rect or
/// rounded-rect clips, with a budget of two planes.
struct Test;

impl Compositor for Test {
    const BUDGET: usize = 2;
    fn expresses_transform(transform: Affine) -> bool {
        axis_aligned(transform)
    }
    fn expresses_clip(clip: &ShapeData) -> bool {
        matches!(clip, ShapeData::Rect(_) | ShapeData::RoundedRect(_))
    }
    fn shows(_: &crate::interop::ExternalFrame) -> bool {
        true
    }
}

const ROOT: LayerId = LayerId::new(0);
const VIDEO: LayerId = LayerId::new(1);
const ABOVE: LayerId = LayerId::new(2);
const BELOW: LayerId = LayerId::new(3);
const PARENT: LayerId = LayerId::new(4);
const SIZE: (u32, u32) = (320, 180);

const fn prop<T>(target: T) -> Prop<T> {
    Prop {
        target,
        animation: None,
    }
}

/// Root with `BELOW`, `PARENT` (holding `VIDEO`) and `ABOVE`, in that
/// paint order.
fn scene() -> SurfaceTree {
    let mut tree = SurfaceTree::new();
    for id in [VIDEO, ABOVE, BELOW, PARENT] {
        tree.apply(LayerOp::Create(id));
    }
    for (parent, child) in [
        (ROOT, BELOW),
        (ROOT, PARENT),
        (PARENT, VIDEO),
        (ROOT, ABOVE),
    ] {
        tree.apply(LayerOp::Push { parent, child });
    }
    tree
}

fn video() -> FxHashMap<LayerId, (u32, u32)> {
    std::iter::once((VIDEO, SIZE)).collect()
}

fn verdict(tree: &SurfaceTree) -> Result<Plan, Ineligible> {
    let plan = plan::<Test>(tree, &video());
    match plan.rejected.as_slice() {
        [] => Ok(plan),
        [(layer, cause)] => {
            assert_eq!(*layer, VIDEO);
            Err(cause.clone())
        }
        more => panic!("one candidate, {} rejections", more.len()),
    }
}

/// An external frame at the surface level with a default blend, no
/// backdrop and an expressible path is promoted, and the surface splits
/// into the part below it and the part above it.
#[test]
fn an_eligible_external_frame_is_promoted_between_two_parts() {
    let plan = verdict(&scene()).expect("eligible");
    assert_eq!(plan.planes.len(), 1);
    let placement = &plan.planes[0];
    assert_eq!(placement.layer, VIDEO);
    assert_eq!(placement.size, SIZE);
    assert_eq!(
        placement.path.iter().map(|l| l.layer).collect::<Vec<_>>(),
        [ROOT, PARENT, VIDEO]
    );
    assert!(plan.trailing, "ABOVE is painted after the video");
    assert_eq!(plan.parts(), 2);
}

/// Only external-frame layers are candidates: an otherwise eligible layer
/// without one stays in the engine.
#[test]
fn only_candidates_are_promoted() {
    let none = plan::<Test>(&scene(), &FxHashMap::default());
    assert_eq!(none.planes, []);
    assert_eq!(none.parts(), 1);
    let video = plan::<Test>(&scene(), &video());
    assert_eq!(
        video.planes.iter().map(|p| p.layer).collect::<Vec<_>>(),
        [VIDEO],
        "eligible layers without an external frame stay in the engine"
    );
}

/// With nothing painted after the plane, no part exists above it.
#[test]
fn no_part_exists_above_a_topmost_plane() {
    let mut tree = scene();
    tree.remove(ABOVE);
    let plan = verdict(&tree).expect("eligible");
    assert!(!plan.trailing);
    assert_eq!(plan.parts(), 1);
}

#[test]
fn an_isolating_ancestor_keeps_the_layer_in_the_engine() {
    let mut tree = scene();
    tree.apply(LayerOp::Opacity(PARENT, prop(0.5)));
    assert_eq!(verdict(&tree), Err(Ineligible::Isolated(PARENT)));
    let mut tree = scene();
    tree.apply(LayerOp::Filter(PARENT, Some(FilterId::new(1))));
    assert_eq!(verdict(&tree), Err(Ineligible::Isolated(PARENT)));
}

#[test]
fn a_filtered_layer_is_not_promoted() {
    let mut tree = scene();
    tree.apply(LayerOp::Filter(VIDEO, Some(FilterId::new(1))));
    assert_eq!(verdict(&tree), Err(Ineligible::Filter));
}

#[test]
fn a_non_default_blend_is_not_promoted() {
    // Under the root, which renders into the surface and never isolates
    // for a blended child; under PARENT the blend would isolate PARENT.
    let mut tree = scene();
    tree.apply(LayerOp::Push {
        parent: ROOT,
        child: VIDEO,
    });
    tree.apply(LayerOp::Blend(VIDEO, BlendMode::Multiply));
    assert_eq!(verdict(&tree), Err(Ineligible::Blend));
    let mut tree = scene();
    tree.apply(LayerOp::Blend(VIDEO, BlendMode::Multiply));
    assert_eq!(verdict(&tree), Err(Ineligible::Isolated(PARENT)));
    // A child blending onto the frame needs the frame's pixels too.
    let mut tree = scene();
    tree.apply(LayerOp::Create(LayerId::new(9)));
    tree.apply(LayerOp::Push {
        parent: VIDEO,
        child: LayerId::new(9),
    });
    tree.apply(LayerOp::Blend(LayerId::new(9), BlendMode::Screen));
    assert_eq!(verdict(&tree), Err(Ineligible::Blend));
}

#[test]
fn a_surface_level_blend_above_keeps_the_layer_in_the_engine() {
    let mut tree = scene();
    tree.apply(LayerOp::Blend(ABOVE, BlendMode::Multiply));
    assert_eq!(verdict(&tree), Err(Ineligible::BlendAbove(ABOVE)));
    // Below the frame, the blend composites onto the part under it exactly
    // as it would in the engine.
    let mut tree = scene();
    tree.apply(LayerOp::Blend(BELOW, BlendMode::Multiply));
    assert!(verdict(&tree).is_ok());
    // Inside an isolated layer above, the blend stays in its offscreen.
    let mut tree = scene();
    tree.apply(LayerOp::Create(LayerId::new(9)));
    tree.apply(LayerOp::Push {
        parent: ABOVE,
        child: LayerId::new(9),
    });
    tree.apply(LayerOp::Opacity(ABOVE, prop(0.5)));
    tree.apply(LayerOp::Blend(LayerId::new(9), BlendMode::Multiply));
    assert!(verdict(&tree).is_ok());
}

#[test]
fn a_group_opacity_is_not_promoted_but_a_leaf_opacity_is() {
    let mut tree = scene();
    tree.apply(LayerOp::Opacity(VIDEO, prop(0.5)));
    let plan = verdict(&tree).expect("a childless layer's opacity is a plane property");
    assert!((plan.planes[0].opacity - 0.5).abs() < f32::EPSILON);
    tree.apply(LayerOp::Create(LayerId::new(9)));
    tree.apply(LayerOp::Push {
        parent: VIDEO,
        child: LayerId::new(9),
    });
    assert_eq!(verdict(&tree), Err(Ineligible::GroupOpacity));
}

#[test]
fn an_inexpressible_transform_is_not_promoted() {
    let mut tree = scene();
    tree.apply(LayerOp::Transform(PARENT, prop(Affine::rotate(0.3))));
    assert_eq!(verdict(&tree), Err(Ineligible::Transform(PARENT)));
}

#[test]
fn an_inexpressible_clip_is_not_promoted() {
    let mut tree = scene();
    tree.apply(LayerOp::Clip(
        PARENT,
        Some(ShapeData::Circle(kurbo::Circle::new((10.0, 10.0), 5.0))),
    ));
    assert_eq!(verdict(&tree), Err(Ineligible::Clip(PARENT)));
    let mut tree = scene();
    tree.apply(LayerOp::Clip(
        VIDEO,
        Some(ShapeData::Rect(Rect::new(0.0, 0.0, 100.0, 50.0))),
    ));
    assert!(verdict(&tree).is_ok());
}

/// Two shaped clips on the path make the engine isolate the inner one into
/// a clip offscreen; a device-aligned rect nests freely.
#[test]
fn nested_shaped_clips_are_not_promoted() {
    let rounded = ShapeData::RoundedRect(RoundedRect::new(0.0, 0.0, 100.0, 50.0, 8.0));
    let mut tree = scene();
    tree.apply(LayerOp::Clip(PARENT, Some(rounded.clone())));
    tree.apply(LayerOp::Clip(VIDEO, Some(rounded.clone())));
    assert_eq!(verdict(&tree), Err(Ineligible::NestedClip(VIDEO)));
    let mut tree = scene();
    tree.apply(LayerOp::Clip(PARENT, Some(rounded)));
    tree.apply(LayerOp::Clip(
        VIDEO,
        Some(ShapeData::Rect(Rect::new(0.0, 0.0, 100.0, 50.0))),
    ));
    assert!(verdict(&tree).is_ok());
}

#[test]
fn the_budget_goes_to_the_first_candidates_in_paint_order() {
    let tree = scene();
    let candidates = [(BELOW, SIZE), (VIDEO, SIZE), (ABOVE, SIZE)]
        .into_iter()
        .collect();
    let plan = plan::<Test>(&tree, &candidates);
    assert_eq!(
        plan.planes.iter().map(|p| p.layer).collect::<Vec<_>>(),
        [BELOW, VIDEO]
    );
    assert_eq!(plan.rejected, [(ABOVE, Ineligible::Budget(Test::BUDGET))]);
    assert!(plan.trailing);
    assert_eq!(plan.parts(), 3);
}

/// The path carries each level's sampled properties, so the content lands
/// where the engine would draw it.
#[test]
fn the_placement_maps_content_to_device_space() {
    let mut tree = scene();
    tree.apply(LayerOp::Transform(
        PARENT,
        prop(Affine::translate((40.0, 30.0))),
    ));
    tree.apply(LayerOp::ScrollOffset(PARENT, prop(Vec2::new(0.0, 10.0))));
    tree.apply(LayerOp::Transform(VIDEO, prop(Affine::scale(0.5))));
    let plan = verdict(&tree).expect("eligible");
    let placement = &plan.planes[0];
    assert_eq!(
        placement.path[1],
        Level {
            layer: PARENT,
            transform: Affine::translate((40.0, 30.0)),
            clip: None,
            scroll: Vec2::new(0.0, 10.0),
        }
    );
    assert_eq!(
        placement.content_to_device(),
        Affine::translate((40.0, 20.0)) * Affine::scale(0.5)
    );
}

/// A backdrop sampled by the frame itself, or anywhere above it, needs the
/// frame's pixels in the engine; one sampled below it does not.
#[test]
fn a_backdrop_on_or_above_the_layer_keeps_it_in_the_engine() {
    use cherenkov::{Engine, Offscreen, OffscreenFormat};
    let Ok(engine) = Engine::<crate::Gpu>::new(crate::GpuConfig::default()) else {
        return;
    };
    let surface = engine
        .surface(Offscreen::new((8, 8), OffscreenFormat::LinearF16))
        .expect("offscreen surface");
    let group = surface.backdrop_group_unfiltered();
    let mut tree = scene();
    tree.apply(LayerOp::Backdrop(VIDEO, Some(group.sample())));
    assert_eq!(verdict(&tree), Err(Ineligible::Backdrop));
    let mut tree = scene();
    tree.apply(LayerOp::Backdrop(ABOVE, Some(group.sample())));
    assert_eq!(verdict(&tree), Err(Ineligible::BackdropAbove(ABOVE)));
    let mut tree = scene();
    tree.apply(LayerOp::Backdrop(BELOW, Some(group.sample())));
    assert!(verdict(&tree).is_ok());
}
