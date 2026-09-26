// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Frame-timing attribution for the shared `cherenkov` front end.
//!
//! A render's GPU timing can resolve during a later render, so each resolved
//! [`FrameTiming`] is traced back to the bench frame that rendered it.

use std::collections::HashMap;

use cherenkov::{Backend, Engine, FrameId, FrameStats, FrameTiming, Next, RenderError};

use crate::motion::Clock;
use crate::{BenchError, GpuSample, PassSample};

/// The most renders [`Timings::render_frame`] spends letting motion come
/// to rest.
const SETTLE_FRAMES: u32 = 2000;

/// The bench frame of every engine frame whose GPU timing has not
/// resolved yet.
#[derive(Default)]
pub struct Timings {
    in_flight: HashMap<FrameId, u64>,
}

impl Timings {
    /// Renders bench frame `frame` at `clock`'s time. With `settle`,
    /// keeps rendering, advancing the clock, until the scene's motion
    /// comes to rest — FLIP compares against the oracle's settled scene.
    /// Every render is recorded as `frame`; returns the GPU timings those
    /// renders resolved.
    ///
    /// # Errors
    /// A render failure mapped through `render_error`, or
    /// [`BenchError::Engine`] when the motion does not settle within
    /// `SETTLE_FRAMES` renders.
    pub fn render_frame<B: Backend>(
        &mut self,
        engine: &Engine<B>,
        clock: &mut Clock,
        frame: u64,
        settle: bool,
        render_error: fn(RenderError) -> BenchError,
    ) -> Result<Vec<GpuSample>, BenchError> {
        let mut next = engine.render(clock.time()).map_err(render_error)?;
        let mut gpu = self.record(frame, engine.stats());
        if !settle {
            return Ok(gpu);
        }
        for _ in 0..SETTLE_FRAMES {
            if next == Next::Idle {
                return Ok(gpu);
            }
            clock.advance();
            next = engine.render(clock.time()).map_err(render_error)?;
            gpu.extend(self.record(frame, engine.stats()));
        }
        if next == Next::Idle {
            return Ok(gpu);
        }
        Err(BenchError::Engine(format!(
            "motion did not settle in {SETTLE_FRAMES} frames"
        )))
    }

    /// Records the engine frame one render drew (if it drew anything) as
    /// bench frame `frame`, and returns the timings that render resolved.
    pub fn record(&mut self, frame: u64, stats: FrameStats) -> Vec<GpuSample> {
        if let Some(id) = stats.frame {
            self.in_flight.insert(id, frame);
        }
        self.samples(stats.timings)
    }

    /// Tags each resolved engine frame timing with the bench frame that
    /// rendered it.
    ///
    /// # Panics
    /// When a timing names an engine frame [`Timings::record`] never saw —
    /// the engine times only frames its renders drew.
    pub fn samples(&mut self, timings: Vec<FrameTiming>) -> Vec<GpuSample> {
        timings
            .into_iter()
            .map(|timing| GpuSample {
                frame: self
                    .in_flight
                    .remove(&timing.frame)
                    .expect("the engine times only frames its renders drew"),
                gpu_seconds: timing.gpu_seconds,
                passes: timing
                    .passes
                    .into_iter()
                    .map(|p| PassSample {
                        name: p.name,
                        width: p.width,
                        height: p.height,
                        format: p.format.to_string(),
                        gpu_seconds: p.gpu_seconds,
                    })
                    .collect(),
            })
            .collect()
    }
}
