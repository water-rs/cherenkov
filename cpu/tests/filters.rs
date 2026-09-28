//! CPU filter execution and rendering tests.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use cherenkov::kurbo::Rect;
use cherenkov::{Draw, Engine, FrameTime, Group, Next, Offscreen, OffscreenFormat, WorkingColor};
use cherenkov_cpu::{Raster, RasterConfig, RedrawCallback};
use filtrate::{
    AnimatedCallback, AnimatedTarget, AuxData, AuxImage, AuxSource, CpuFilter, CpuFilterError,
    CpuImage, Filter, FilterExt, FilterImage, FilterParam, Footprint, ImageVisitor, Interpolator,
    OperatingSpace, Placed, SpatialFilter, SpatialStage, StageCollector, WatchGuard, WorkingSpace,
    filters,
};

fn engine() -> Engine<Raster> {
    Engine::<Raster>::new(RasterConfig::default()).expect("engine")
}

#[test]
fn colour_chains_run_on_layers_and_recorded_groups() {
    let engine = engine();
    let filter =
        engine.filter(filters::Brightness(-0.125_f32).then(filters::Brightness(-0.125_f32)));
    let color = WorkingColor::new([0.25, 0.5, 0.75, 1.0]);
    let layer_surface = engine
        .surface(Offscreen::new((8, 8), OffscreenFormat::LinearF32))
        .expect("layer surface");
    let layer = layer_surface.layer();
    layer_surface.update(|tx| {
        tx[layer_surface.root()].push(&layer);
        tx[&layer]
            .filter(&filter)
            .content(layer_surface.record(|r| {
                r.fill(Rect::new(0.0, 0.0, 8.0, 8.0), color);
            }));
    });

    let group_surface = engine
        .surface(Offscreen::new((8, 8), OffscreenFormat::LinearF32))
        .expect("group surface");
    let group_color = nami::binding(color);
    group_surface.update(|tx| {
        tx[group_surface.root()].content(group_surface.record(|r| {
            r.group(Group::new().opacity(0.5), |r| {
                r.group(Group::new().filter(filter.id()), |r| {
                    r.fill(Rect::new(0.0, 0.0, 8.0, 8.0), group_color.clone());
                });
            });
        }));
    });
    engine.render(FrameTime::now()).expect("render");

    let layer_pixel = layer_surface.readback().expect("layer readback").pixels[4 * 8 + 4];
    assert_eq!(
        layer_pixel.map(f32::to_bits),
        [0.0_f32, 0.25, 0.5, 1.0].map(f32::to_bits)
    );
    let group_pixel = group_surface.readback().expect("group readback").pixels[4 * 8 + 4];
    for (actual, expected) in group_pixel.into_iter().zip([0.0, 0.125, 0.25, 0.5]) {
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "filtered group pixel: {group_pixel:?}"
        );
    }

    group_color.set(WorkingColor::new([0.75, 0.5, 0.25, 1.0]));
    engine.render(FrameTime::now()).expect("live patch");
    assert_eq!(engine.stats().commands_lowered, 1);
    let patched = group_surface.readback().expect("patched readback").pixels[4 * 8 + 4];
    for (actual, expected) in patched.into_iter().zip([0.25, 0.125, 0.0, 0.5]) {
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "patched filtered group pixel: {patched:?}"
        );
    }
}

#[test]
fn nested_spatial_filters_match_full_surface_application_at_band_edges() {
    let engine = engine();
    let unfiltered = engine
        .surface(Offscreen::new((32, 96), OffscreenFormat::LinearF32))
        .expect("unfiltered surface");
    let filtered = engine
        .surface(Offscreen::new((32, 96), OffscreenFormat::LinearF32))
        .expect("filtered surface");
    let gaussian_kernel = filters::GaussianBlur(3.0_f32);
    let box_kernel = filters::Blur(5.0_f32);
    let gaussian = engine.filter(gaussian_kernel);
    let box_blur = engine.filter(box_kernel);
    let content = |surface: &cherenkov::Surface<Raster>| {
        surface.record(|r| {
            for y in [15.0, 16.0, 31.0, 32.0, 47.0, 48.0, 63.0, 64.0, 79.0, 80.0] {
                r.fill(
                    Rect::new(0.0, y, 32.0, y + 1.0),
                    WorkingColor::new([0.8, 0.3, 0.1, 0.75]),
                );
            }
        })
    };
    unfiltered.update(|tx| {
        tx[unfiltered.root()].content(content(&unfiltered));
    });
    let outer = filtered.layer();
    let inner = filtered.layer();
    filtered.update(|tx| {
        tx[filtered.root()].push(&outer);
        tx[&outer].push(&inner).filter(&gaussian);
        tx[&inner].filter(&box_blur).content(content(&filtered));
    });
    engine.render(FrameTime::now()).expect("render");

    let size = (32, 96);
    let source = unfiltered.readback().expect("source readback").pixels;
    let mut expected = source;
    let mut image = CpuImage {
        pixels: &mut expected,
        top: 0,
        size,
    };
    box_kernel
        .apply_cpu_image(
            &box_kernel.params(),
            &WorkingSpace::LINEAR_DISPLAY_P3,
            &mut image,
        )
        .expect("CPU box blur");
    gaussian_kernel
        .apply_cpu_image(
            &gaussian_kernel.params(),
            &WorkingSpace::LINEAR_DISPLAY_P3,
            &mut image,
        )
        .expect("CPU gaussian blur");
    let actual = filtered.readback().expect("filtered readback").pixels;
    assert_eq!(actual, expected);
}

#[test]
fn rgba8_filter_images_blend_on_the_cpu() {
    let engine = engine();
    let image = FilterImage::from_rgba8(1, 1, vec![128, 64, 32, 9]);
    let filter = engine.filter(filters::BlendWithImage {
        image,
        amount: 1.0_f32,
        mode: filters::BlendMode::Multiply,
    });
    let surface = engine
        .surface(Offscreen::new((4, 4), OffscreenFormat::LinearF32))
        .expect("surface");
    surface.update(|tx| {
        tx[surface.root()]
            .filter(&filter)
            .content(surface.record(|r| {
                r.fill(
                    Rect::new(0.0, 0.0, 4.0, 4.0),
                    WorkingColor::new([0.5, 0.4, 0.3, 1.0]),
                );
            }));
    });
    engine.render(FrameTime::now()).expect("render");
    let pixel = surface.readback().expect("readback").pixels[2 * 4 + 2];
    for (actual, expected) in pixel.into_iter().zip([
        0.5 * (128.0 / 255.0),
        0.4 * (64.0 / 255.0),
        0.3 * (32.0 / 255.0),
        1.0,
    ]) {
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "blended image pixel: {pixel:?}"
        );
    }
}

#[derive(Clone)]
struct ScriptedParam {
    initial: f32,
    callback: Arc<Mutex<Option<AnimatedCallback>>>,
}

impl ScriptedParam {
    fn new(initial: f32) -> Self {
        Self {
            initial,
            callback: Arc::default(),
        }
    }

    fn fire(&self, value: f32, interpolator: Option<Box<dyn Interpolator>>) {
        self.callback
            .lock()
            .expect("parameter callback mutex poisoned")
            .as_ref()
            .expect("filter watcher installed")(AnimatedTarget {
            value,
            interpolator,
        });
    }
}

impl FilterParam for ScriptedParam {
    fn snapshot(&self) -> f32 {
        self.initial
    }

    fn watch_animated(&self, callback: AnimatedCallback) -> WatchGuard {
        *self
            .callback
            .lock()
            .expect("parameter callback mutex poisoned") = Some(callback);
        WatchGuard::new(())
    }
}

struct LinearRamp(Duration);

impl Interpolator for LinearRamp {
    fn duration(&self) -> Duration {
        self.0
    }

    fn interpolate(&self, from: f32, to: f32, elapsed: Duration) -> f32 {
        let progress = (elapsed.as_secs_f32() / self.0.as_secs_f32()).min(1.0);
        (to - from).mul_add(progress, from)
    }
}

#[test]
fn animated_parameters_rerender_and_request_frames() {
    let wakes = Arc::new(AtomicUsize::new(0));
    let wake_count = Arc::clone(&wakes);
    let engine = Engine::<Raster>::new(RasterConfig {
        redraw: Some(RedrawCallback::new(move || {
            wake_count.fetch_add(1, Ordering::Relaxed);
        })),
        ..RasterConfig::default()
    })
    .expect("engine");
    let surface = engine
        .surface(Offscreen::new((4, 4), OffscreenFormat::LinearF32).rate(30..=60))
        .expect("surface");
    let other_surface = engine
        .surface(Offscreen::new((4, 4), OffscreenFormat::LinearF32).rate(45..=90))
        .expect("other surface");
    let parameter = ScriptedParam::new(0.0);
    let filter = engine.filter(filters::Brightness(parameter.clone()));
    for filtered_surface in [&surface, &other_surface] {
        filtered_surface.update(|tx| {
            tx[filtered_surface.root()]
                .filter(&filter)
                .content(filtered_surface.record(|r| {
                    r.fill(
                        Rect::new(0.0, 0.0, 4.0, 4.0),
                        WorkingColor::new([0.1, 0.1, 0.1, 1.0]),
                    );
                }));
        });
    }
    let start = Instant::now();
    assert_eq!(
        engine.render(FrameTime::at(start)).expect("initial frame"),
        Next::Idle
    );

    parameter.fire(0.2, None);
    assert_eq!(wakes.load(Ordering::Relaxed), 1);
    assert_eq!(
        engine
            .render(FrameTime::at(start + Duration::from_millis(10)))
            .expect("snap frame"),
        Next::Idle
    );
    assert_eq!(engine.stats().commands_lowered, 0);
    let snapped = surface.readback().expect("snap readback").pixels[2 * 4 + 2];
    assert!((snapped[0] - 0.3).abs() < 1.0e-6);

    parameter.fire(0.8, Some(Box::new(LinearRamp(Duration::from_millis(100)))));
    assert_eq!(wakes.load(Ordering::Relaxed), 2);
    let mid_time = start + Duration::from_millis(60);
    let next = engine
        .render(FrameTime::at(mid_time))
        .expect("animated frame");
    assert!(matches!(next, Next::At { .. }));
    if let Next::At { rate, .. } = next {
        assert_eq!(rate, 30..=90);
    }
    let mid = surface.readback().expect("mid readback").pixels[2 * 4 + 2];
    assert!((mid[0] - 0.6).abs() < 1.0e-6, "mid pixel: {mid:?}");

    let repeated = engine
        .render(FrameTime::at(mid_time))
        .expect("repeated frame");
    assert!(matches!(repeated, Next::At { .. }));
    let unchanged = surface.readback().expect("repeated readback").pixels[2 * 4 + 2];
    assert_eq!(mid.map(f32::to_bits), unchanged.map(f32::to_bits));

    assert_eq!(
        engine
            .render(FrameTime::at(start + Duration::from_millis(120)))
            .expect("settled frame"),
        Next::Idle
    );
    let settled = surface.readback().expect("settled readback").pixels[2 * 4 + 2];
    assert!((settled[0] - 0.9).abs() < 1.0e-6);
}

struct GpuOnlyImage;

impl AuxImage for GpuOnlyImage {
    fn width(&self) -> u32 {
        1
    }

    fn height(&self) -> u32 {
        1
    }

    fn data(&self) -> Option<AuxData<'_>> {
        None
    }
}

struct GpuOnlyFilter {
    image: GpuOnlyImage,
}

impl Filter for GpuOnlyFilter {
    type Kind = filtrate::kind::Spatial;
    type Params = [f32; 0];

    fn params(&self) -> Self::Params {
        []
    }

    fn collect_stages<C: StageCollector>(&self, collector: &mut C) {
        const STAGE: SpatialStage = SpatialStage {
            name: "gpu_only_test",
            source: "",
            params: &[],
            space: OperatingSpace::Working,
            shape: None,
            aux: &[AuxSource::Image(0)],
        };
        collector.spatial(Placed::new(&STAGE));
    }

    fn visit_images<V: ImageVisitor>(&self, visitor: &mut V) {
        visitor.visit(0, &self.image);
    }
}

impl SpatialFilter for GpuOnlyFilter {
    fn footprint_of(_params: &Self::Params) -> Footprint {
        Footprint::ZERO
    }
}

impl CpuFilter for GpuOnlyFilter {
    fn cpu_footprint(_params: &Self::Params) -> Footprint {
        Footprint::ZERO
    }

    fn apply_cpu_image(
        &self,
        _params: &Self::Params,
        _space: &WorkingSpace,
        _image: &mut CpuImage<'_>,
    ) -> Result<(), CpuFilterError> {
        let _data = self
            .image
            .data()
            .ok_or(CpuFilterError::GpuImage { index: 0 })?;
        Ok(())
    }
}

#[test]
fn gpu_only_auxiliary_images_return_an_explicit_unsupported_error() {
    let engine = engine();
    let filter = engine.filter(GpuOnlyFilter {
        image: GpuOnlyImage,
    });
    let surface = engine
        .surface(Offscreen::new((4, 4), OffscreenFormat::LinearF32))
        .expect("surface");
    surface.update(|tx| {
        tx[surface.root()]
            .filter(&filter)
            .content(surface.record(|r| {
                r.fill(Rect::new(0.0, 0.0, 4.0, 4.0), WorkingColor::WHITE);
            }));
    });
    assert!(matches!(
        engine.render(FrameTime::now()),
        Err(cherenkov::RenderError::Unsupported("filter-gpu-image"))
    ));
}

#[test]
fn removing_a_registered_filter_does_not_silently_fallback() {
    let engine = engine();
    let filter = engine.filter(filters::Brightness(0.0_f32));
    let surface = engine
        .surface(Offscreen::new((4, 4), OffscreenFormat::LinearF32))
        .expect("surface");
    let layer = surface.layer();
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
        tx[&layer].filter(&filter).content(surface.record(|r| {
            r.fill(Rect::new(0.0, 0.0, 4.0, 4.0), WorkingColor::WHITE);
        }));
    });
    engine.render(FrameTime::now()).expect("initial render");
    drop(filter);
    surface.update(|tx| {
        tx[&layer].opacity(0.5_f32);
    });
    let error = engine
        .render(FrameTime::now())
        .expect_err("stale filter id");
    assert!(
        matches!(error, cherenkov::RenderError::Render(ref message) if message.contains("unregistered filter")),
        "stale filter error: {error}"
    );
}
