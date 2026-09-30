//! Projective layers (#84) on both rendering backends.

use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::time::{Duration, Instant};

use cherenkov::kurbo::{Affine, Circle, Rect, Vec2};
use cherenkov::{
    Backdrop, BlendMode, Draw, Engine, FrameTime, Layer, Offscreen, OffscreenFormat, Picture,
    Projective, ProjectiveLayers, RenderError, Surface, WorkingColor,
};

const SIZE: (u32, u32) = (64, 64);
const CLEAR: WorkingColor = WorkingColor::new([0.1, 0.2, 0.6, 1.0]);

/// An asymmetric card: a red block on the left, a green disc on the right
/// and a transparent band at the bottom inside the clip.
fn card() -> Picture {
    Picture::record(|r| {
        r.fill(
            Rect::new(0.0, 0.0, 16.0, 24.0),
            WorkingColor::new([0.9, 0.1, 0.1, 1.0]),
        );
        r.fill(
            Circle::new((26.0, 12.0), 6.0),
            WorkingColor::new([0.1, 0.8, 0.2, 1.0]),
        );
    })
}

const CLIP: Rect = Rect::new(0.0, 0.0, 32.0, 32.0);

struct Scene<B: ProjectiveLayers> {
    engine: Engine<B>,
    surface: Surface<B>,
    layer: Layer,
    frame: u64,
    start: Instant,
}

impl<B: ProjectiveLayers> Scene<B> {
    fn new(config: B::Config, setup: impl FnOnce(&mut cherenkov::LayerEdit<B>)) -> Self {
        let engine = Engine::<B>::new(config).expect("backend required");
        let surface = engine
            .surface(Offscreen::new(SIZE, OffscreenFormat::LinearF32))
            .expect("surface");
        let layer = surface.layer();
        surface.update(|tx| {
            tx[surface.root()].push(&layer);
            let edit = &mut tx[&layer];
            edit.content(card())
                .clip(CLIP)
                .transform(Affine::translate((16.0, 16.0)));
            setup(edit);
        });
        surface.clear_color(CLEAR);
        Self {
            engine,
            surface,
            layer,
            frame: 0,
            start: Instant::now(),
        }
    }

    fn edit(&self, edit: impl FnOnce(&mut cherenkov::LayerEdit<B>)) {
        self.surface.update(|tx| edit(&mut tx[&self.layer]));
    }

    fn render(&mut self) -> Result<Vec<[f32; 4]>, RenderError> {
        self.frame += 1;
        self.engine.render(FrameTime::at(
            self.start + Duration::from_millis(self.frame * 8),
        ))?;
        Ok(self.surface.readback().expect("readback").pixels)
    }
}

/// The backend's own rendering of the clear colour alone, in its storage
/// precision: what an untouched pixel reads back as.
fn clear_pixel<B: ProjectiveLayers>(config: B::Config) -> [f32; 4] {
    let mut scene = Scene::<B>::new(config, |e| {
        e.content(Picture::record(|_| {}));
    });
    let pixels = scene.render().expect("clear");
    assert!(pixels.iter().all(|px| bits(*px) == bits(pixels[0])));
    pixels[0]
}

/// A pixel's exact bits, for exact comparisons.
fn bits(px: [f32; 4]) -> [u32; 4] {
    px.map(f32::to_bits)
}

fn close(a: &[[f32; 4]], b: &[[f32; 4]], tolerance: f32, what: &str) {
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        for c in 0..4 {
            assert!(
                (a[c] - b[c]).abs() <= tolerance,
                "{what}: pixel ({}, {}) channel {c}: {} != {}",
                i % SIZE.0 as usize,
                i / SIZE.0 as usize,
                a[c],
                b[c]
            );
        }
    }
}

/// An identity projection composes exactly like the affine layer, up to
/// the local image's `f16` storage.
pub fn identity_matches_affine<B: ProjectiveLayers>(config: impl Fn() -> B::Config) {
    let mut projected = Scene::<B>::new(config(), |e| {
        e.projection(Projective::IDENTITY);
    });
    let mut affine = Scene::<B>::new(config(), |_| {});
    let a = projected.render().expect("projected");
    assert_eq!(projected.engine.stats().projective_realized, 1);
    let b = affine.render().expect("affine");
    close(&a, &b, 2e-3, "identity projection");
    assert_eq!(affine.engine.stats().projective_realized, 0);
}

/// A card entirely behind the viewer, or seen exactly edge-on, draws
/// nothing: the surface keeps its clear colour bit for bit.
pub fn hidden_and_edge_on_contribute_nothing<B: ProjectiveLayers>(config: impl Fn() -> B::Config) {
    let clear = clear_pixel::<B>(config());
    for (what, edit) in [
        (
            "behind",
            Box::new(|e: &mut cherenkov::LayerEdit<B>| {
                // w = 1 − 150/100 < 0 over the whole card.
                e.projection(Projective::perspective(100.0).unwrap())
                    .depth(150.0);
            }) as Box<dyn Fn(&mut cherenkov::LayerEdit<B>)>,
        ),
        (
            "edge-on",
            Box::new(|e: &mut cherenkov::LayerEdit<B>| {
                e.projection(Projective::perspective(200.0).unwrap())
                    .pivot(Vec2::new(16.0, 16.0))
                    .tilt(Vec2::new(0.0, FRAC_PI_2));
            }),
        ),
    ] {
        let mut scene = Scene::<B>::new(config(), |e| edit(e));
        let pixels = scene.render().expect(what);
        for (i, px) in pixels.iter().enumerate() {
            assert_eq!(bits(*px), bits(clear), "{what}: pixel {i} changed");
        }
    }
}

/// Half a turn about Y shows the back side mirrored about the pivot; a
/// full turn is the untilted card. There is no backface culling.
pub fn turns_keep_winding_and_show_both_sides<B: ProjectiveLayers>(config: impl Fn() -> B::Config) {
    let pivot = Vec2::new(16.0, 16.0);
    let mut flat = Scene::<B>::new(config(), |e| {
        e.pivot(pivot).tilt(Vec2::ZERO);
    });
    let flat = flat.render().expect("flat");
    let mut full = Scene::<B>::new(config(), |e| {
        e.pivot(pivot).tilt(Vec2::new(0.0, TAU));
    });
    close(&full.render().expect("full turn"), &flat, 2e-3, "full turn");
    let mut half = Scene::<B>::new(config(), |e| {
        e.pivot(pivot).tilt(Vec2::new(0.0, PI));
    });
    let mut mirrored = Scene::<B>::new(config(), |e| {
        e.pivot(pivot).scale(Vec2::new(-1.0, 1.0));
    });
    close(
        &half.render().expect("half turn"),
        &mirrored.render().expect("mirror"),
        2e-3,
        "half turn",
    );
}

/// A retained realization and a fresh one of the same state are
/// identical, and a warm matrix-only frame realizes nothing.
pub fn cached_and_fresh_realizations_are_identical<B: ProjectiveLayers>(
    config: impl Fn() -> B::Config,
) {
    let camera = Projective::perspective(120.0).unwrap();
    let pivot = Vec2::new(16.0, 16.0);
    let pose = |tilt: Vec2| {
        move |e: &mut cherenkov::LayerEdit<B>| {
            e.projection(camera).pivot(pivot).tilt(tilt);
        }
    };
    let (a, b) = (Vec2::new(0.3, -0.5), Vec2::new(-0.2, 0.7));
    let mut warm = Scene::<B>::new(config(), pose(a));
    let first = warm.render().expect("cold");
    assert_eq!(warm.engine.stats().projective_realized, 1);
    warm.edit(pose(b));
    warm.render().expect("second pose");
    warm.edit(pose(a));
    let again = warm.render().expect("warm");
    assert_eq!(
        warm.engine.stats().projective_realized,
        0,
        "a matrix-only frame reuses the retained realization"
    );
    let mut fresh = Scene::<B>::new(config(), pose(a));
    let cold = fresh.render().expect("fresh");
    assert_eq!(first, again, "cached realization changed");
    assert_eq!(cold, again, "fresh and cached realizations differ");
    // Changing the content invalidates the retained image.
    warm.edit(|e| {
        e.content(Picture::record(|r| {
            r.fill(CLIP, WorkingColor::WHITE);
        }));
    });
    let changed = warm.render().expect("changed content");
    assert_eq!(warm.engine.stats().projective_realized, 1);
    assert_ne!(changed, again);
}

/// A destructive blend keeps its operator domain — the projected layer
/// clip — where the image is transparent: `Src` clears the parent inside
/// the clip and leaves it untouched outside.
pub fn destructive_blend_keeps_its_operator_domain<B: ProjectiveLayers>(
    config: impl Fn() -> B::Config,
) {
    let mut projected = Scene::<B>::new(config(), |e| {
        e.projection(Projective::IDENTITY).blend(BlendMode::Src);
    });
    let mut affine = Scene::<B>::new(config(), |e| {
        e.blend(BlendMode::Src);
    });
    let p = projected.render().expect("projected");
    let a = affine.render().expect("affine");
    close(&p, &a, 2e-3, "destructive identity");
    let clear = clear_pixel::<B>(config());
    let at = |x: usize, y: usize| p[y * SIZE.0 as usize + x];
    // Inside the clip, below the content: cleared by the operator.
    assert_eq!(bits(at(20, 44)), bits([0.0; 4]));
    // Outside the clip: the parent is untouched.
    assert_eq!(bits(at(4, 4)), bits(clear));
}

/// An invalid composed pose, and a projective layer without a clip, are
/// render errors, never an identity or a transparent layer.
pub fn invalid_poses_are_errors<B: ProjectiveLayers>(config: impl Fn() -> B::Config) {
    let mut singular = Scene::<B>::new(config(), |e| {
        e.tilt(Vec2::new(0.2, 0.0)).scale(Vec2::new(0.0, 1.0));
    });
    assert!(matches!(
        singular.render(),
        Err(RenderError::ProjectivePose {
            error: cherenkov::ProjectiveError::NonInvertible,
            ..
        })
    ));
    let mut unclipped = Scene::<B>::new(config(), |e| {
        e.clear_clip().tilt(Vec2::new(0.2, 0.0));
    });
    assert!(matches!(
        unclipped.render(),
        Err(RenderError::Unsupported(
            cherenkov::lowering::projective::UNCLIPPED
        ))
    ));
}

/// A visible image beyond the backend's dimension limit or byte budget is
/// an explicit error naming the required size, never a capped density.
pub fn limits_are_explicit_errors<B: ProjectiveLayers>(config: impl Fn() -> B::Config) {
    for (clip, what) in [
        (Rect::new(0.0, 0.0, 40_000.0, 32.0), "dimension limit"),
        (Rect::new(0.0, 0.0, 8_000.0, 8_000.0), "are admitted"),
    ] {
        let mut scene = Scene::<B>::new(config(), |e| {
            e.clip(clip).projection(Projective::IDENTITY);
        });
        match scene.render() {
            Err(RenderError::ProjectiveUnsupported { reason, .. }) => {
                assert!(reason.contains(what), "{what}: {reason}");
            }
            other => panic!("{what}: expected ProjectiveUnsupported, got {other:?}"),
        }
    }
}

/// A projective layer cannot be a backdrop member, and one backdrop group
/// cannot span the surface and a projective layer's local space: both are
/// explicit `Unsupported` errors.
pub fn backdrop_spaces_are_checked<B: ProjectiveLayers + Backdrop>(config: impl Fn() -> B::Config) {
    let mut member = Scene::<B>::new(config(), |_| {});
    let group = member.surface.backdrop_group_unfiltered();
    member.edit(|e| {
        e.projection(Projective::IDENTITY).backdrop(group.sample());
    });
    assert!(matches!(
        member.render(),
        Err(RenderError::Unsupported(
            cherenkov::lowering::projective::BACKDROP_MEMBER
        ))
    ));

    let mut spanning = Scene::<B>::new(config(), |e| {
        e.projection(Projective::IDENTITY);
    });
    let group = spanning.surface.backdrop_group_unfiltered();
    let (inside, outside) = (spanning.surface.layer(), spanning.surface.layer());
    spanning.surface.update(|tx| {
        tx[&spanning.layer].push(&inside);
        tx[&inside]
            .clip(Rect::new(0.0, 0.0, 8.0, 8.0))
            .backdrop(group.sample());
        tx[spanning.surface.root()].push(&outside);
        tx[&outside]
            .clip(Rect::new(0.0, 0.0, 8.0, 8.0))
            .backdrop(group.sample());
    });
    assert!(matches!(
        spanning.render(),
        Err(RenderError::Unsupported(
            cherenkov::lowering::projective::BACKDROP_CROSS_SPACE
        ))
    ));
}

/// A tilt animation keeps the engine awake and reuses the recorded
/// content; after `clear_projection` the layer is affine again.
pub fn tilt_animates_and_clears<B: ProjectiveLayers>(config: impl Fn() -> B::Config) {
    let mut scene = Scene::<B>::new(config(), |e| {
        e.projection(Projective::perspective(200.0).unwrap())
            .pivot(Vec2::new(16.0, 16.0));
    });
    scene.render().expect("flat");
    scene.edit(|e| {
        e.tilt(Vec2::new(0.0, TAU))
            .animation(cherenkov::Curve::linear(Duration::from_millis(40)));
    });
    let mut realized = 0;
    for _ in 0..4 {
        scene.render().expect("animating");
        assert_eq!(scene.engine.stats().commands_lowered, 0);
        realized += scene.engine.stats().projective_realized;
    }
    assert!(
        realized <= 3,
        "one realization per density bucket, got {realized}"
    );
    scene.edit(|e| {
        e.clear_projection();
    });
    let cleared = scene.render().expect("cleared");
    assert_eq!(scene.engine.stats().projective_composed, 0);
    let mut affine = Scene::<B>::new(config(), |e| {
        e.pivot(Vec2::new(16.0, 16.0));
    });
    assert_eq!(cleared, affine.render().expect("affine"));
}

/// A card crossing the horizon: pixels whose ray meets the card's plane
/// behind the viewer keep the clear colour exactly, the front part draws,
/// and nothing is NaN.
pub fn horizon_crossing_excludes_the_back_half_space<B: ProjectiveLayers>(
    config: impl Fn() -> B::Config,
) {
    // A 32 × 64 card turned about both axes so its far corner lies
    // behind the camera plane: the diagonal horizon puts pixels whose ray
    // meets the plane behind the viewer inside the front part's bounds.
    let (tilt, distance) = (Vec2::new(0.4, 1.0), 10.0);
    let tall = Rect::new(0.0, 0.0, 32.0, 64.0);
    let mut scene = Scene::<B>::new(config(), |e| {
        e.content(Picture::record(|r| {
            r.fill(tall, WorkingColor::new([0.9, 0.8, 0.1, 1.0]));
        }))
        .clip(tall)
        .projection(Projective::perspective(distance).unwrap())
        .pivot(Vec2::new(16.0, 16.0))
        .tilt(tilt);
    });
    let pixels = scene.render().expect("horizon");
    let clear = clear_pixel::<B>(config());
    // The pose, composed here from public constructors:
    // translate(16 + 16) · perspective · Ry(tilt.y) · Rx(tilt.x) ·
    // translate(−16).
    let translate = |offset: f64| {
        Projective::from_rows([
            [1., 0., 0., offset],
            [0., 1., 0., offset],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
        .unwrap()
    };
    let (sin, cos) = tilt.x.sin_cos();
    let rx = Projective::from_rows([
        [1., 0., 0., 0.],
        [0., cos, -sin, 0.],
        [0., sin, cos, 0.],
        [0., 0., 0., 1.],
    ])
    .unwrap();
    let (sin, cos) = tilt.y.sin_cos();
    let ry = Projective::from_rows([
        [cos, 0., sin, 0.],
        [0., 1., 0., 0.],
        [-sin, 0., cos, 0.],
        [0., 0., 0., 1.],
    ])
    .unwrap();
    let pose = translate(32.0)
        .checked_mul(Projective::perspective(distance).unwrap())
        .and_then(|m| m.checked_mul(ry))
        .and_then(|m| m.checked_mul(rx))
        .and_then(|m| m.checked_mul(translate(-16.0)))
        .unwrap();
    let h = cherenkov::lowering::projective::Homography::plane(&pose).front_inverse();
    let (mut front, mut back) = (0, 0);
    let pixel_centres = (0..SIZE.1)
        .flat_map(|y| (0..SIZE.0).map(move |x| [f64::from(x) + 0.5, f64::from(y) + 0.5]));
    for (px, d) in pixels.iter().zip(pixel_centres) {
        assert!(
            px.iter().all(|v| v.is_finite()),
            "pixel {d:?} is not finite"
        );
        let [x, y, w] = h.map(d[0], d[1]);
        let local = [x / w, y / w];
        // Away from the clip edge, so reconstruction cannot reach across.
        let inside = (1.5..30.5).contains(&local[0]) && (1.5..62.5).contains(&local[1]);
        if w < 0.0 && inside {
            back += 1;
            assert_eq!(
                bits(*px),
                bits(clear),
                "pixel {d:?} samples the back half-space"
            );
        } else if w > 0.0 && inside && bits(*px) != bits(clear) {
            front += 1;
        }
    }
    assert!(back > 0 && front > 0, "front {front}, back {back}");
}
