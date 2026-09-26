// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Serialized report shapes for `render` and `measure` — always written
//! with serde, never concatenated.

use std::collections::BTreeMap;

use cherenkov_oracle::Metrics;
use serde::Serialize;

use crate::{Counters, DeviceInfo, EngineInfo, PassSample};

/// `render` output: the engine provenance, the correctness metrics against
/// the oracle, and the counters of what was submitted.
#[derive(Serialize)]
pub struct RenderReport {
    /// Adapter key.
    pub engine: &'static str,
    /// The adapter's pinned provenance.
    pub info: EngineInfo,
    /// Scene directory name.
    pub scene: String,
    /// Scene pixel size.
    pub width: u32,
    /// Scene pixel size.
    pub height: u32,
    /// Metrics against the oracle reference.
    pub metrics: Metrics,
    /// What the adapter issued.
    pub counters: Counters,
    /// Device/thermal metadata at render time.
    pub device: DeviceInfo,
}

/// One measured frame: raw samples, not percentiles.
#[derive(Clone, Debug, Serialize)]
pub struct FrameSample {
    /// Wall-clock seconds of the CPU encode phase.
    pub encode_seconds: f64,
    /// Wall-clock seconds of submit + device wait. For CPU engines this is
    /// the rasterization cost; for GPU engines it brackets queue submission
    /// and device poll (host-side wall time, not GPU time).
    pub submit_seconds: f64,
    /// GPU seconds from real GPU timestamps where the backend provides
    /// them; `null` otherwise. Never estimated.
    pub gpu_seconds: Option<f64>,
    /// CPU the measuring thread ran on when the timed phases started
    /// (`sched_getcpu`); `null` where the kernel does not report it.
    pub cpu_start: Option<u32>,
    /// CPU the measuring thread ran on when the timed phases ended.
    pub cpu_end: Option<u32>,
    /// True when the thread migrated between `cpu_start` and `cpu_end`.
    pub migrated: bool,
    /// Per-pass GPU timings for this frame, in submission order; omitted
    /// when the backend provides none.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub passes: Vec<PassSample>,
}

/// CPU placement of a `measure` run.
///
/// On big.LITTLE hardware (differing `cpuinfo_max_freq` across CPUs) an
/// unpinned run is uncontrolled — samples spread across clusters and
/// percentiles blend them.
#[derive(Clone, Debug, Serialize)]
pub struct Placement {
    /// Sorted CPU ids passed via `--cpu`; `null` when unpinned.
    pub requested: Option<Vec<u32>>,
    /// Whether the measuring thread was pinned to `requested` before
    /// engine creation (threads spawned later inherit the mask).
    pub controlled: bool,
    /// True when the host's CPUs report differing `cpuinfo_max_freq` —
    /// placement matters; an uncontrolled run blends cluster speeds.
    pub heterogeneous: bool,
    /// CPU the one-time `prepare` phase started on.
    pub prepare_cpu: Option<u32>,
    /// Per-CPU counts: a sample counts once under each CPU it touched,
    /// keyed by CPU id, with that CPU's `cpuinfo_max_freq` kHz where
    /// sysfs exposes it.
    pub cpus: BTreeMap<u32, CpuUse>,
    /// Samples that migrated between their start and end CPU.
    pub migrated: u32,
}

/// Per-CPU tally in [`Placement::cpus`].
#[derive(Clone, Copy, Debug, Serialize)]
pub struct CpuUse {
    /// Timed-phase endpoints observed on this CPU (a migrated sample
    /// contributes one to each of its start and end CPUs).
    pub count: u32,
    /// `cpuinfo_max_freq` kHz where sysfs exposes it.
    pub max_freq_khz: Option<u64>,
}

/// `render`/`measure` output when the scene declares a feature the
/// adapter does not implement — recorded, never emulated.
#[derive(Serialize)]
pub struct UnsupportedReport {
    /// Adapter key.
    pub engine: &'static str,
    /// The adapter's pinned provenance.
    pub info: EngineInfo,
    /// Scene directory name.
    pub scene: String,
    /// The scene feature the adapter cannot execute faithfully.
    pub unsupported: cherenkov_scene::Feature,
    /// The upstream API the engine lacks, when the adapter knows it
    /// (e.g. `peniko::Extend` has no `None` variant). `null` when the
    /// limitation is adapter coverage rather than a named upstream API.
    pub missing_api: Option<&'static str>,
}

/// `measure` output.
#[derive(Serialize)]
pub struct MeasureReport {
    /// Adapter key.
    pub engine: &'static str,
    /// The adapter's pinned provenance.
    pub info: EngineInfo,
    /// Scene directory name.
    pub scene: String,
    /// Scene pixel size.
    pub width: u32,
    /// Scene pixel size.
    pub height: u32,
    /// One-time scene preparation in seconds — resource creation and
    /// scene conversion done once outside the timed loop (fonts,
    /// images, GPU uploads, immutable shaders). Not a per-frame cost.
    pub prepare_seconds: f64,
    /// Warmup frames discarded.
    pub warmup_frames: u32,
    /// Raw per-frame samples (one per measured frame).
    pub samples: Vec<FrameSample>,
    /// CPU placement of the run (see [`Placement`]).
    pub placement: Placement,
    /// p50/p90/p99 over the raw samples, per field.
    pub percentiles: Percentiles,
    /// What the adapter issued per frame.
    pub counters: Counters,
    /// Device/thermal metadata at measure time.
    pub device: DeviceInfo,
}

/// Percentiles of each measured field.
#[derive(Clone, Debug, Serialize)]
pub struct Percentiles {
    /// `encode_seconds` percentiles `[p50, p90, p99]`.
    pub encode_seconds: [f64; 3],
    /// `submit_seconds` percentiles `[p50, p90, p99]`.
    pub submit_seconds: [f64; 3],
    /// `gpu_seconds` percentiles `[p50, p90, p99]`; `null` when the backend
    /// provides no GPU timestamps.
    pub gpu_seconds: Option<[f64; 3]>,
    /// Per-pass GPU percentiles, grouped by pass index within the frame;
    /// omitted when the backend provides no per-pass timings.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub passes: Vec<PassPercentiles>,
}

/// Percentiles of one render pass across the measured frames.
#[derive(Clone, Debug, Serialize)]
pub struct PassPercentiles {
    /// The pass's name (`"surface"`, `"scratch{n}"`, ...).
    pub name: String,
    /// Target width in pixels.
    pub width: u32,
    /// Target height in pixels.
    pub height: u32,
    /// Target texture format.
    pub format: String,
    /// `gpu_seconds` percentiles `[p50, p90, p99]` for this pass.
    pub gpu_seconds: [f64; 3],
}

/// Nearest-rank percentiles of `samples`, `[p50, p90, p99]`.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "sample counts are small positive values"
)]
pub fn percentiles(samples: &[f64]) -> Option<[f64; 3]> {
    if samples.is_empty() {
        return None;
    }
    let mut v = samples.to_vec();
    v.sort_by(f64::total_cmp);
    let at = |p: f64| {
        let idx = (((v.len() as f64) * p).ceil() as usize).saturating_sub(1);
        v[idx.min(v.len() - 1)]
    };
    Some([at(0.50), at(0.90), at(0.99)])
}
