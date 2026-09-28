//! Regression test for allocation-free steady-state recording and rendering.

#![cfg(all(feature = "testing", not(target_arch = "wasm32")))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::mpsc;

use cherenkov::kurbo::{PathEl, Point, Rect};
use cherenkov::testing::{Null, NullConfig};
use cherenkov::{
    Draw, Engine, FillRule, Fixed, FrameTime, Glyph, GlyphRun, GlyphStyle, Offscreen,
    OffscreenFormat, ShapeData, WorkingColor,
};

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static FREES: Cell<usize> = const { Cell::new(0) };
}

struct ThreadAllocator;

#[global_allocator]
static ALLOCATOR: ThreadAllocator = ThreadAllocator;

unsafe impl GlobalAlloc for ThreadAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        allocation();
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        allocation();
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        release();
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, size) };
        allocation();
        release();
        pointer
    }
}

fn allocation() {
    if TRACKING.try_with(Cell::get).unwrap_or(false) {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
    }
}

fn release() {
    if TRACKING.try_with(Cell::get).unwrap_or(false) {
        let _ = FREES.try_with(|count| count.set(count.get() + 1));
    }
}

fn start_tracking() {
    ALLOCATIONS.with(|count| count.set(0));
    FREES.with(|count| count.set(0));
    TRACKING.with(|tracking| tracking.set(true));
}

fn stop_tracking() -> (usize, usize) {
    TRACKING.with(|tracking| tracking.set(false));
    (ALLOCATIONS.with(Cell::get), FREES.with(Cell::get))
}

#[test]
fn recording_and_rendering_reuse_ui_thread_allocations() {
    let (events, _receiver) = mpsc::channel();
    let engine = Engine::<Null>::new(NullConfig {
        events,
        reject: HashSet::new(),
    })
    .expect("init");
    let surface = engine
        .surface(Offscreen::new((32, 32), OffscreenFormat::LinearF16))
        .expect("surface");
    let layer = surface.layer();
    let path = ShapeData::Path {
        elements: Arc::from([
            PathEl::MoveTo(Point::new(1.0, 1.0)),
            PathEl::LineTo(Point::new(20.0, 1.0)),
            PathEl::LineTo(Point::new(10.0, 20.0)),
            PathEl::ClosePath,
        ]),
        rule: FillRule::NonZero,
    };
    let run = GlyphRun {
        font: cherenkov::FontId::new(1),
        size: 16.0,
        coords: Arc::from([]),
        glyphs: Arc::from([Glyph {
            id: 1,
            x: 1.0,
            y: 16.0,
            transform: None,
        }]),
        style: GlyphStyle::Fill,
    };

    for frame in 0..12 {
        let time = FrameTime::now();
        start_tracking();
        surface.update(|tx| {
            tx[&layer].record(|recorder| {
                recorder.fill(Fixed(path.clone()), Fixed(WorkingColor::WHITE));
                recorder.fill(
                    Fixed(Rect::new(2.0, 2.0, 24.0, 24.0)),
                    Fixed(WorkingColor::BLACK),
                );
                recorder.glyphs(Fixed(run.clone()), Fixed(WorkingColor::WHITE));
            });
        });
        let result = engine.render(time);
        let counts = stop_tracking();
        result.expect("render");
        if frame >= 3 {
            assert_eq!(counts, (0, 0), "frame {}", frame + 1);
        }
    }
}
