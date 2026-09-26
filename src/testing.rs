// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A test backend: [`Null`] draws nothing and reports every render-thread
//! call as an [`Event`] on a channel, so tests and the cross-backend
//! behaviour suite can assert what the front end committed.

use std::sync::mpsc::Sender;

use kurbo::{Affine, Vec2};

use crate::backend::{Backend, Frame, Redraw, Renderer, SurfaceInfo};
use crate::config::MemoryUsage;
use crate::error::{EngineError, RenderError, ResourceError, SurfaceError};
use crate::frame::Readback;
use crate::glyph::FontId;
use crate::image::{ImageUpload, Rgba8, Rgba16F};
use crate::message::{ContentOp, FontData, LayerId, SurfaceId};
use crate::paint::ImageId;
use crate::{Offscreen, Pressure, Uploads};

/// A render-thread event [`Null`] reports.
#[derive(Debug)]
#[non_exhaustive]
pub enum Event {
    /// `create_surface` ran.
    CreateSurface(SurfaceId),
    /// `resize_surface` ran.
    ResizeSurface(SurfaceId, (u32, u32)),
    /// `destroy_surface` ran.
    DestroySurface(SurfaceId),
    /// `add_font` ran.
    AddFont(FontId),
    /// `remove_font` ran.
    RemoveFont(FontId),
    /// `add_image` ran.
    AddImage(ImageId),
    /// `remove_image` ran.
    RemoveImage(ImageId),
    /// `set_content` ran.
    SetContent(SurfaceId, LayerId),
    /// `remove_layer` ran.
    RemoveLayer(SurfaceId, LayerId),
    /// One rendered surface, in `frame.surfaces` order.
    Frame(FrameRecord),
}

/// One surface's sampled tree, recorded per rendered frame.
#[derive(Debug)]
pub struct FrameRecord {
    /// The surface.
    pub surface: SurfaceId,
    /// The frame's `changed` flag for this surface.
    pub changed: bool,
    /// Every layer's sampled state.
    pub layers: Vec<LayerSample>,
}

/// One layer's sampled state in a [`FrameRecord`].
#[derive(Debug)]
pub struct LayerSample {
    /// The layer.
    pub id: LayerId,
    /// The sampled transform.
    pub transform: Affine,
    /// The sampled opacity.
    pub opacity: f32,
    /// The sampled, pixel-snapped scroll offset.
    pub scroll_offset: Vec2,
    /// The layer's children, in paint order.
    pub children: Vec<LayerId>,
}

/// A backend that draws nothing and reports every call. The `Config`
/// carries the test's probe channel.
#[derive(Clone, Copy, Debug, Default)]
pub struct Null;

/// Configuration for [`Null`]: the channel events are reported on.
pub struct NullConfig {
    /// The event probe. A test keeps the matching `Receiver`.
    pub events: Sender<Event>,
}

/// `Null`'s provenance: nothing to report.
pub type NullInfo = ();

/// The `Null` render-thread state.
pub struct NullRenderer {
    events: Sender<Event>,
}

impl Backend for Null {
    type Config = NullConfig;
    type Info = NullInfo;
    type Target = Offscreen;
    type Renderer = NullRenderer;

    fn init(config: NullConfig) -> Result<(NullRenderer, NullInfo), EngineError> {
        Ok((
            NullRenderer {
                events: config.events,
            },
            (),
        ))
    }
}

impl Renderer for NullRenderer {
    type Target = Offscreen;

    fn create_surface(
        &mut self,
        id: SurfaceId,
        target: Offscreen,
    ) -> Result<SurfaceInfo, SurfaceError> {
        if target.size.0 == 0 || target.size.1 == 0 {
            return Err(SurfaceError::ZeroSize);
        }
        let _ = self.events.send(Event::CreateSurface(id));
        Ok(SurfaceInfo {
            size: target.size,
            readable: true,
        })
    }

    fn resize_surface(&mut self, id: SurfaceId, size: (u32, u32)) {
        let _ = self.events.send(Event::ResizeSurface(id, size));
    }

    fn destroy_surface(&mut self, id: SurfaceId) {
        let _ = self.events.send(Event::DestroySurface(id));
    }

    fn add_font(&mut self, id: FontId, _font: FontData) -> Result<(), ResourceError> {
        let _ = self.events.send(Event::AddFont(id));
        Ok(())
    }

    fn remove_font(&mut self, id: FontId) {
        let _ = self.events.send(Event::RemoveFont(id));
    }

    fn add_image(&mut self, id: ImageId, _image: ImageUpload) -> Result<(), ResourceError> {
        let _ = self.events.send(Event::AddImage(id));
        Ok(())
    }

    fn remove_image(&mut self, id: ImageId) {
        let _ = self.events.send(Event::RemoveImage(id));
    }

    fn set_content(&mut self, surface: SurfaceId, layer: LayerId, _content: Option<ContentOp>) {
        let _ = self.events.send(Event::SetContent(surface, layer));
    }

    fn remove_layer(&mut self, surface: SurfaceId, layer: LayerId) {
        let _ = self.events.send(Event::RemoveLayer(surface, layer));
    }

    fn render(
        &mut self,
        frame: &Frame<'_>,
        _stats: &mut crate::FrameStats,
    ) -> Result<Redraw, RenderError> {
        for surface in frame.surfaces {
            let layers = surface
                .tree
                .layers()
                .map(|(id, node)| LayerSample {
                    id,
                    transform: node.transform,
                    opacity: node.opacity,
                    scroll_offset: node.scroll_offset,
                    children: node.children.clone(),
                })
                .collect();
            let _ = self.events.send(Event::Frame(FrameRecord {
                surface: surface.id,
                changed: surface.changed,
                layers,
            }));
        }
        Ok(Redraw::None)
    }

    fn readback(&mut self, surface: SurfaceId) -> Result<Readback, RenderError> {
        let _ = surface;
        Ok(Readback {
            width: 0,
            height: 0,
            pixels: Vec::new(),
        })
    }

    fn memory(&self) -> MemoryUsage {
        MemoryUsage::default()
    }

    fn trim(&mut self, _pressure: Pressure) {}
}

impl Uploads<Rgba8> for Null {}
impl Uploads<Rgba16F> for Null {}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use nami::{SignalExt, binding};

    use super::*;
    use crate::image::ImageData;
    use crate::{Animation, Curve, Decay, Engine, FrameTime, Next, OffscreenFormat, Spring};

    fn engine() -> (Engine<Null>, std::sync::mpsc::Receiver<Event>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let engine = Engine::<Null>::new(NullConfig { events: tx }).expect("init");
        (engine, rx)
    }

    fn frames(rx: &std::sync::mpsc::Receiver<Event>) -> Vec<FrameRecord> {
        let mut out = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let Event::Frame(record) = event {
                out.push(record);
            }
        }
        out
    }

    fn layer(record: &FrameRecord, id: LayerId) -> &LayerSample {
        record
            .layers
            .iter()
            .find(|l| l.id == id)
            .expect("layer in record")
    }

    #[test]
    fn layer_tree_edits_reach_the_tree() {
        let (engine, rx) = engine();
        let surface = engine
            .surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))
            .expect("surface");
        let child = surface.layer();
        let grandchild = surface.layer();
        surface.update(|tx| {
            tx[surface.root()].push(&child);
            tx[&child].push(&grandchild);
            tx[&child]
                .transform(Affine::translate((3.0, 4.0)))
                .opacity(0.5f32);
        });
        let next = engine.render(FrameTime::now()).expect("render");
        assert_eq!(next, Next::Idle);
        let records = frames(&rx);
        let record = records.last().expect("a frame record");
        assert_eq!(record.layers.len(), 3);
        let child_rec = layer(record, child.id());
        assert_eq!(child_rec.transform, Affine::translate((3.0, 4.0)));
        assert!((child_rec.opacity - 0.5).abs() < f32::EPSILON);
        assert_eq!(child_rec.children, vec![grandchild.id()]);
    }

    #[test]
    fn spring_settles_to_target_then_idle() {
        let (engine, rx) = engine();
        let surface = engine
            .surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))
            .expect("surface");
        let layer_handle = surface.layer();
        surface.update(|tx| {
            tx[surface.root()].push(&layer_handle);
            tx[&layer_handle]
                .opacity(0.25f32)
                .animation(Spring::smooth());
        });
        let t0 = Instant::now();
        let mut next = engine.render(FrameTime::at(t0)).expect("render");
        assert!(
            matches!(next, Next::At { ref rate, .. } if *rate == (60..=120)),
            "animating: {next:?}"
        );
        let mut t = t0;
        loop {
            t += Duration::from_millis(8);
            next = engine.render(FrameTime::at(t)).expect("render");
            if matches!(next, Next::Idle) {
                break;
            }
            assert!(t - t0 < Duration::from_secs(5), "spring never settled");
        }
        let record = frames(&rx).pop().expect("records");
        assert!((layer(&record, layer_handle.id()).opacity - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn curve_endpoints() {
        let (engine, rx) = engine();
        let surface = engine
            .surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))
            .expect("surface");
        let layer_handle = surface.layer();
        surface.update(|tx| {
            tx[surface.root()].push(&layer_handle);
            tx[&layer_handle]
                .opacity(0.7f32)
                .animation(Curve::linear(Duration::from_millis(200)));
        });
        let t0 = Instant::now();
        engine.render(FrameTime::at(t0)).expect("render");
        let _ = frames(&rx);
        // Before the curve ends it is still running.
        let next = engine
            .render(FrameTime::at(t0 + Duration::from_millis(100)))
            .expect("render");
        assert!(matches!(next, Next::At { .. }), "{next:?}");
        let record = frames(&rx).pop().expect("mid-frame");
        let mid = layer(&record, layer_handle.id()).opacity;
        assert!(mid < 1.0 && mid > 0.7, "mid {mid}");
        // Past the duration it lands exactly on the target and idles.
        let next = engine
            .render(FrameTime::at(t0 + Duration::from_millis(300)))
            .expect("render");
        assert_eq!(next, Next::Idle, "{next:?}");
        let record = frames(&rx).pop().expect("end frame");
        assert!((layer(&record, layer_handle.id()).opacity - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn retargeting_a_spring_is_continuous() {
        let (engine, rx) = engine();
        let surface = engine
            .surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))
            .expect("surface");
        let layer_handle = surface.layer();
        surface.update(|tx| {
            tx[surface.root()].push(&layer_handle);
            tx[&layer_handle]
                .transform(Affine::translate((10.0, 0.0)))
                .animation(Spring::snappy());
        });
        let t0 = Instant::now();
        engine.render(FrameTime::at(t0)).expect("render");
        let t1 = t0 + Duration::from_millis(100);
        engine.render(FrameTime::at(t1)).expect("render");
        let _ = frames(&rx);
        let mid_record = {
            engine.render(FrameTime::at(t1)).expect("render");
            frames(&rx).pop().expect("mid")
        };
        let pos_mid = layer(&mid_record, layer_handle.id()).transform.as_coeffs()[4];
        // Retarget mid-flight.
        surface.update(|tx| {
            tx[&layer_handle]
                .transform(Affine::translate((-5.0, 0.0)))
                .animation(Spring::snappy());
        });
        let eps = Duration::from_millis(1);
        engine.render(FrameTime::at(t1 + eps)).expect("render");
        let record = frames(&rx).pop().expect("retarget frame");
        let pos_next = layer(&record, layer_handle.id()).transform.as_coeffs()[4];
        // Position must be (nearly) the last sampled position — continuity,
        // not a snap back to the start.
        assert!((pos_next - pos_mid).abs() < 0.05, "{pos_mid} -> {pos_next}");
    }

    #[test]
    fn scroll_decay_and_rubber_band() {
        let (engine, rx) = engine();
        let surface = engine
            .surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))
            .expect("surface");
        let layer_handle = surface.layer();
        let bounds = kurbo::Rect::new(0.0, 0.0, 100.0, 50.0);
        surface.update(|tx| {
            tx[surface.root()].push(&layer_handle);
            tx[&layer_handle]
                .scroll_offset(Vec2::new(0.0, 10.0))
                .animation(Decay::new(Vec2::new(0.0, 300.0)).rubber_band(bounds));
        });
        let t0 = Instant::now();
        let mut t = t0;
        // The decay flings the offset past the bound; the rubber-band
        // spring pulls it back and the engine settles to Idle.
        loop {
            t += Duration::from_millis(8);
            let next = engine.render(FrameTime::at(t)).expect("render");
            if matches!(next, Next::Idle) {
                break;
            }
            assert!(t - t0 < Duration::from_secs(10), "never settled");
        }
        let record = frames(&rx).pop().expect("records");
        let offset = layer(&record, layer_handle.id()).scroll_offset;
        // The rubber band settles on the bound edge, which `contains`
        // excludes, so compare inclusively.
        assert!(
            (bounds.min_x()..=bounds.max_x()).contains(&offset.x)
                && (bounds.min_y()..=bounds.max_y()).contains(&offset.y),
            "offset {offset:?} outside {bounds:?}"
        );
    }

    #[test]
    fn decay_starts_at_committed_value_and_stays_where_it_stops() {
        let (engine, rx) = engine();
        let surface = engine
            .surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))
            .expect("surface");
        let layer_handle = surface.layer();
        surface.update(|tx| {
            tx[surface.root()].push(&layer_handle);
            tx[&layer_handle]
                .scroll_offset(Vec2::ZERO)
                .animation(Decay::new(Vec2::new(600.0, 0.0)));
        });
        let t0 = Instant::now();
        let mut t = t0;
        // Unbounded decay: position = v/k = 600/4 = 150, then Idle.
        loop {
            t += Duration::from_millis(8);
            let next = engine.render(FrameTime::at(t)).expect("render");
            if matches!(next, Next::Idle) {
                break;
            }
            assert!(t - t0 < Duration::from_secs(10), "never settled");
        }
        let record = frames(&rx).pop().expect("records");
        let offset = layer(&record, layer_handle.id()).scroll_offset;
        assert!(
            (offset.x - 150.0).abs() < 0.5 && offset.y.abs() < 0.5,
            "offset {offset:?}, expected ≈(150, 0)"
        );
    }

    #[test]
    fn bound_signal_updates_without_a_transaction() {
        let (engine, rx) = engine();
        let surface = engine
            .surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))
            .expect("surface");
        let layer_handle = surface.layer();
        let opacity = binding::<f32>(1.0f32);
        surface.update(|tx| {
            tx[surface.root()].push(&layer_handle);
            tx[&layer_handle].opacity(opacity.clone());
        });
        let t0 = Instant::now();
        engine.render(FrameTime::at(t0)).expect("render");
        let _ = frames(&rx);

        // A plain change snaps.
        opacity.set(0.5f32);
        let next = engine
            .render(FrameTime::at(t0 + Duration::from_millis(16)))
            .expect("render");
        assert_eq!(next, Next::Idle);
        let record = frames(&rx).pop().expect("record");
        assert!((layer(&record, layer_handle.id()).opacity - 0.5).abs() < f32::EPSILON);

        // A change carrying Animation metadata interpolates.
        let animated = opacity.with(Animation::from(Spring::smooth()));
        let surface2 = surface.layer();
        surface.update(|tx| {
            tx[surface.root()].push(&surface2);
            tx[&surface2].opacity(animated);
        });
        opacity.set(0.25f32);
        let next = engine
            .render(FrameTime::at(t0 + Duration::from_millis(32)))
            .expect("render");
        assert!(matches!(next, Next::At { .. }), "{next:?}");
    }

    #[test]
    fn resource_drop_reaches_the_renderer() {
        let (engine, rx) = engine();
        let _surface = engine
            .surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))
            .expect("surface");
        {
            let image = engine
                .image(ImageData::<Rgba8>::new(2, 2, vec![0u8; 16]).expect("image data"))
                .expect("image");
            let _ = image.id();
        }
        engine.render(FrameTime::now()).expect("render");
        let events: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        assert!(
            events.iter().any(|e| matches!(e, Event::RemoveImage(_))),
            "no RemoveImage in {events:?}"
        );
    }

    #[test]
    fn update_animated_fills_the_default() {
        let (engine, rx) = engine();
        let surface = engine
            .surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))
            .expect("surface");
        let layer_handle = surface.layer();
        surface.update_animated(Spring::bouncy(), |tx| {
            tx[surface.root()].push(&layer_handle);
            tx[&layer_handle].opacity(0.5f32);
        });
        let next = engine.render(FrameTime::now()).expect("render");
        assert!(matches!(next, Next::At { .. }), "{next:?}");
        let _ = frames(&rx);
    }

    #[test]
    fn layer_drop_queues_remove() {
        let (engine, rx) = engine();
        let surface = engine
            .surface(Offscreen::new((16, 16), OffscreenFormat::LinearF16))
            .expect("surface");
        {
            let layer_handle = surface.layer();
            surface.update(|tx| {
                tx[surface.root()].push(&layer_handle);
            });
        }
        engine.render(FrameTime::now()).expect("render");
        let events: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        assert!(
            events.iter().any(|e| matches!(e, Event::RemoveLayer(_, _))),
            "no RemoveLayer in {events:?}"
        );
    }
}
