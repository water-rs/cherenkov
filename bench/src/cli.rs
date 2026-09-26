// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cherenkov-bench`: cross-engine correctness and performance runner.
//!
//! - `render --engine E --scene DIR --out FILE` renders one scene against
//!   the oracle and writes the engine's image, an error heatmap and a
//!   metrics JSON. `--corpus DIR --out-dir DIR` sweeps a corpus.
//! - `measure --engine E --scene DIR --frames N --warmup W --out FILE`
//!   records raw per-frame CPU encode, submit and GPU seconds (real
//!   timestamps only) plus p50/p90/p99. `--cpu LIST` pins the run to a
//!   CPU set — required for meaningful numbers on big.LITTLE hosts
//!   (Linux and Android only); the report records placement either way.
//!   `--rate HZ` paces the measured frames to a fixed rate instead of
//!   running flat out; `--energy` brackets the window with the
//!   platform's power meter (Android ODPM rails, root via `su -c`;
//!   macOS `sudo -n powermetrics`) and reports joules per frame.

use std::collections::BTreeMap;
use std::ffi::OsString;
#[cfg(unix)]
use std::ffi::{CStr, OsStr, c_char, c_int};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::convert;
use crate::report::{
    CpuUse, FrameSample, MeasureReport, Pacing, PassPercentiles, Percentiles, PhasePercentiles,
    Placement, RenderReport, UnsupportedReport, percentiles,
};
use crate::{
    BenchError, EncodeInput, Engine, affinity, conditions, create_engine, energy, engine_names,
};
use cherenkov_oracle::{F32Image, Renderer, metrics};
use cherenkov_scene::Scene;
use clap::{Parser, Subcommand};

/// Command line.
#[derive(Parser)]
#[command(
    name = "cherenkov-bench",
    about = "Cross-engine correctness and performance suite"
)]
struct Cli {
    /// Subcommand.
    #[command(subcommand)]
    cmd: Sub,
}

#[derive(Subcommand)]
enum Sub {
    /// Render scene(s) and report correctness metrics vs the oracle.
    Render {
        /// Adapter key (see `cherenkov-bench engines`).
        #[arg(long)]
        engine: String,
        /// One scene directory (`scene.json` + `resources/`).
        #[arg(long, conflicts_with = "corpus", required_unless_present = "corpus")]
        scene: Option<PathBuf>,
        /// Corpus directory; every child holding a `scene.json` is run.
        #[arg(long)]
        corpus: Option<PathBuf>,
        /// Metrics JSON path (with `--scene`).
        #[arg(long, conflicts_with = "out_dir", required_unless_present = "out_dir")]
        out: Option<PathBuf>,
        /// Output directory (with `--corpus`).
        #[arg(long)]
        out_dir: Option<PathBuf>,
    },
    /// Measure encode/submit/GPU frame times.
    Measure {
        /// Adapter key.
        #[arg(long)]
        engine: String,
        /// One scene directory.
        #[arg(long, conflicts_with = "corpus", required_unless_present = "corpus")]
        scene: Option<PathBuf>,
        /// Corpus directory.
        #[arg(long)]
        corpus: Option<PathBuf>,
        /// Measured frame count (after warmup).
        #[arg(long, default_value_t = 60)]
        frames: u32,
        /// Warmup frames discarded before measuring.
        #[arg(long, default_value_t = 5)]
        warmup: u32,
        /// Report JSON path (with `--scene`).
        #[arg(long, conflicts_with = "out_dir", required_unless_present = "out_dir")]
        out: Option<PathBuf>,
        /// Output directory (with `--corpus`).
        #[arg(long)]
        out_dir: Option<PathBuf>,
        /// Pin the measurement to these CPUs — a list like `7`, `4-6` or
        /// `1,3,5-7`. Required for meaningful numbers on big.LITTLE
        /// hardware; Linux and Android only.
        #[arg(long, value_name = "LIST")]
        cpu: Option<String>,
        /// Pace the measured frames to this rate in Hz — each frame
        /// starts on the deadline `start + n/rate` rather than running
        /// flat out. Energy per frame is only comparable at a fixed
        /// rate.
        #[arg(long, value_name = "HZ")]
        rate: Option<f64>,
        /// Measure energy over the measured window: every ODPM rail on
        /// Android (root — run via `su -c`), `sudo -n powermetrics` on
        /// macOS. Fails when the meter cannot be read rather than
        /// reporting nothing.
        #[arg(long)]
        energy: bool,
    },
    /// List compiled-in adapter keys.
    Engines,
}

/// Runs one `cherenkov-bench` invocation — the single code path behind
/// both the `cherenkov-bench` binary and the C entry point the iOS
/// host calls (`cherenkov_bench_run`).
///
/// `args` is the full argv, program name first. Returns the
/// process-style exit code: 0 on success, 1 on a run failure, or the
/// `clap` code (0 for `--help`/`--version`, 2 for a parse error).
#[must_use]
pub fn run_args(args: &[OsString]) -> i32 {
    // The iOS host calls `cherenkov_bench_run` once per argument list
    // inside one process, so a later call finds the global subscriber
    // already installed; `try_init` keeps re-entry clean. Diagnostics
    // go to stderr (the host captures it per run into `run-<n>.log`).
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(e) => {
            let _ = e.print();
            return e.exit_code();
        }
    };
    let code = match run(cli) {
        Ok(()) => 0,
        Err(e) => {
            tracing::error!("{e}");
            1
        }
    };
    // The host keeps this process alive for the next argument list and
    // may redirect our fds per call — leave no partial line buffered.
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    code
}

/// C entry point for hosts that cannot spawn a process.
///
/// The iOS bench app links the `cherenkov-bench` static library and
/// calls this once per argument list; it runs the same [`run_args`] the
/// binary's `main` does.
///
/// Returns the process-style exit code (0 on success, 101 on panic —
/// unwinding cannot cross `extern "C"`, so a panicking run would
/// otherwise abort the host and every queued run with it).
///
/// # Panics
/// `argc` is a count, so a negative value — outside the
/// `main(argc, argv)` contract — panics.
///
/// # Safety
/// `argv` must point to `argc` non-null pointers, each to a valid
/// NUL-terminated C string — the `main(argc, argv)` contract.
#[cfg(unix)]
#[expect(clippy::similar_names, reason = "the argc/argv C contract")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cherenkov_bench_run(argc: c_int, argv: *const *const c_char) -> c_int {
    use std::os::unix::ffi::OsStrExt as _;
    let argc = usize::try_from(argc).expect("argc is non-negative");
    // SAFETY: the caller guarantees `argv` points to `argc` pointers,
    // each to a NUL-terminated C string that outlives this call.
    let args: Vec<OsString> = unsafe {
        std::slice::from_raw_parts(argv, argc)
            .iter()
            .map(|&arg| OsStr::from_bytes(CStr::from_ptr(arg).to_bytes()).to_os_string())
            .collect()
    };
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_args(&args))).unwrap_or(101)
}

fn run(cli: Cli) -> Result<(), BenchError> {
    match cli.cmd {
        Sub::Engines => {
            for e in engine_names() {
                tracing::info!(engine = e, "adapter");
            }
            Ok(())
        }
        Sub::Render {
            engine,
            scene,
            corpus,
            out,
            out_dir,
        } => {
            let mut engine = create_engine(&engine)?;
            for dir in scene_dirs(scene.as_deref(), corpus.as_deref())? {
                let out_path = match (&out, &out_dir) {
                    (Some(o), None) => o.clone(),
                    (None, Some(d)) => d.join(format!(
                        "render-{}-{}.json",
                        engine.info().name,
                        dir.file_name().unwrap_or_default().to_string_lossy()
                    )),
                    _ => return Err(BenchError::Engine("--out or --out-dir required".into())),
                };
                match render_scene(&mut *engine, &dir) {
                    Ok(rendered) => {
                        write_render(&rendered, &out_path)?;
                        tracing::info!(
                            scene = %dir.display(),
                            flip_mean = rendered.report.metrics.flip_mean,
                            flip_max = rendered.report.metrics.flip_max,
                            max_local_error = rendered.report.metrics.max_local_error,
                            out = %out_path.display(),
                            "render"
                        );
                    }
                    Err(BenchError::Unsupported { feature, api, .. }) => {
                        write_unsupported(&*engine, &dir, feature.clone(), api, &out_path)?;
                        tracing::warn!(
                            scene = %dir.display(),
                            ?feature,
                            out = %out_path.display(),
                            "unsupported"
                        );
                    }
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        }
        Sub::Measure {
            engine,
            scene,
            corpus,
            frames,
            warmup,
            out,
            out_dir,
            cpu,
            rate,
            energy,
        } => measure_cmd(
            &engine,
            MeasureOpts {
                scene: scene.as_deref(),
                corpus: corpus.as_deref(),
                frames,
                warmup,
                out: out.as_deref(),
                out_dir: out_dir.as_deref(),
                cpu: cpu.as_deref(),
                rate,
                energy,
            },
        ),
    }
}

/// The `measure` subcommand's fields, borrowed to avoid cloning paths.
#[derive(Clone, Copy)]
struct MeasureOpts<'a> {
    scene: Option<&'a Path>,
    corpus: Option<&'a Path>,
    frames: u32,
    warmup: u32,
    out: Option<&'a Path>,
    out_dir: Option<&'a Path>,
    cpu: Option<&'a str>,
    rate: Option<f64>,
    energy: bool,
}

fn measure_cmd(engine: &str, opts: MeasureOpts<'_>) -> Result<(), BenchError> {
    if let Some(rate) = opts.rate {
        if !(rate.is_finite() && rate > 0.0) {
            return Err(BenchError::Engine(format!(
                "--rate must be a positive, finite Hz value, got {rate}"
            )));
        }
        pacing(rate, opts.frames)?;
    }
    if opts.energy {
        if opts.frames == 0 {
            return Err(BenchError::Engine(
                "--energy requires at least one measured frame (--frames > 0)".into(),
            ));
        }
        // Mandatory: fail here, before the engine and scenes, rather
        // than writing reports without energy.
        energy::Meter::probe()?;
        if opts.rate.is_none() {
            tracing::warn!(
                "--energy without --rate: energy per frame is only comparable at a fixed rate"
            );
        }
    }
    let pinned = opts.cpu.map(affinity::parse_cpu_list).transpose()?;
    if let Some(cpus) = &pinned {
        // Pin before the adapter is created so every thread it spawns
        // inherits the mask.
        affinity::pin_current_thread(cpus)?;
    }
    let mut engine = create_engine(engine)?;
    for dir in scene_dirs(opts.scene, opts.corpus)? {
        let out_path = match (opts.out, opts.out_dir) {
            (Some(o), None) => o.to_path_buf(),
            (None, Some(d)) => d.join(format!(
                "measure-{}-{}.json",
                engine.info().name,
                dir.file_name().unwrap_or_default().to_string_lossy()
            )),
            _ => return Err(BenchError::Engine("--out or --out-dir required".into())),
        };
        match measure_scene(
            &mut *engine,
            &dir,
            opts.frames,
            opts.warmup,
            pinned.as_deref(),
            opts.rate,
            opts.energy,
        ) {
            Ok(report) => {
                write_json(&report, &out_path)?;
                tracing::info!(
                    scene = %dir.display(),
                    prepare_s = report.prepare_seconds,
                    encode_p50 = report.percentiles.encode_seconds[0],
                    submit_p50 = report.percentiles.submit_seconds[0],
                    gpu_p50 = ?report.percentiles.gpu_seconds.map(|g| g[0]),
                    out = %out_path.display(),
                    "measure"
                );
                for pass in &report.percentiles.passes {
                    tracing::info!(
                        scene = %dir.display(),
                        pass = %pass.name,
                        size = %format!("{}x{}", pass.width, pass.height),
                        format = %pass.format,
                        gpu_p50 = ?pass.gpu_seconds.map(|g| g[0]),
                        "pass"
                    );
                }
                for phase in &report.percentiles.phases {
                    tracing::info!(
                        scene = %dir.display(),
                        phase = %phase.name,
                        p50 = phase.seconds[0],
                        p99 = phase.seconds[2],
                        "phase"
                    );
                }
            }
            Err(BenchError::Unsupported { feature, api, .. }) => {
                write_unsupported(&*engine, &dir, feature.clone(), api, &out_path)?;
                tracing::warn!(
                    scene = %dir.display(),
                    ?feature,
                    out = %out_path.display(),
                    "unsupported"
                );
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// `--scene` gives one dir; `--corpus` gives every child holding
/// `scene.json`, sorted by name.
fn scene_dirs(scene: Option<&Path>, corpus: Option<&Path>) -> Result<Vec<PathBuf>, BenchError> {
    if let Some(d) = scene {
        return Ok(vec![d.to_path_buf()]);
    }
    let corpus = corpus.ok_or_else(|| BenchError::Engine("no scene or corpus".into()))?;
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(corpus)? {
        let dir = entry?.path();
        if dir.is_dir() && dir.join("scene.json").is_file() {
            dirs.push(dir);
        }
    }
    dirs.sort();
    if dirs.is_empty() {
        return Err(BenchError::Engine(format!(
            "no scenes under {}",
            corpus.display()
        )));
    }
    Ok(dirs)
}

/// Rendered image + heatmap, written next to the metrics JSON.
struct RenderOutput {
    /// The metrics report.
    report: RenderReport,
    /// Engine image.
    image: F32Image,
    /// Error heatmap, RGB8.
    heatmap: Vec<u8>,
}

fn render_scene(engine: &mut dyn Engine, dir: &Path) -> Result<RenderOutput, BenchError> {
    let scene = Scene::load(dir)?;
    let blobs = convert::load_blobs(&scene, dir)?;
    let reference =
        Renderer::new(scene.width as usize, scene.height as usize).render(&scene, dir)?;
    let input = EncodeInput {
        scene: &scene,
        blobs: &blobs,
    };
    engine.prepare(&input)?;
    engine.encode(&input)?;
    let submit = engine.submit(0, true)?;
    let test = submit
        .image
        .ok_or_else(|| BenchError::Engine("adapter returned no image".into()))?;
    let (metrics_v, heatmap) = metrics::compare(&reference, &test);
    Ok(RenderOutput {
        report: RenderReport {
            engine: engine.info().name,
            info: engine.info().clone(),
            scene: dir
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            width: scene.width,
            height: scene.height,
            metrics: metrics_v,
            counters: engine.counters(),
            device: engine.device(),
        },
        image: test,
        heatmap,
    })
}

/// Per-CPU use counts over the samples plus the prepare call.
fn cpu_use(prepare_cpu: Option<u32>, samples: &[FrameSample]) -> BTreeMap<u32, CpuUse> {
    let mut cpus = BTreeMap::<u32, CpuUse>::new();
    let mut count = |cpu| {
        if let Some(cpu) = cpu {
            cpus.entry(cpu)
                .or_insert_with(|| CpuUse {
                    count: 0,
                    max_freq_khz: affinity::cpu_max_freq_khz(cpu),
                })
                .count += 1;
        }
    };
    count(prepare_cpu);
    for sample in samples {
        count(sample.cpu_start);
        count(sample.cpu_end);
    }
    cpus
}

/// What [`run_frames`] collected.
struct Window {
    samples: Vec<FrameSample>,
    meter: Option<energy::Meter>,
    start: Instant,
    missed_deadlines: u32,
}

/// How far past a deadline a frame may start before it counts as
/// missed — scheduler overshoot under a millisecond is noise.
const PACING_TOLERANCE: Duration = Duration::from_millis(1);

/// The warmup + measured frame loop.
///
/// With `--rate` each measured frame starts on `start + n / rate` —
/// the one sleep in the bench is frame pacing — and the energy meter
/// wraps exactly the measured window.
fn run_frames(
    engine: &mut dyn Engine,
    input: &EncodeInput<'_>,
    frames: u32,
    warmup: u32,
    period: Option<Duration>,
    window_hint: Duration,
    measure_energy: bool,
) -> Result<Window, BenchError> {
    let mut meter = None;
    let mut start = Instant::now();
    let mut missed_deadlines = 0u32;
    let mut samples = Vec::with_capacity(frames as usize);
    for frame in 0..(warmup + frames) {
        if frame == warmup {
            // The energy window and the pacing clock both open
            // immediately before the first measured frame.
            if measure_energy {
                meter = Some(energy::Meter::begin(window_hint)?);
            }
            start = Instant::now();
        }
        if frame >= warmup
            && let Some(period) = period
            && let Some(deadline) = period
                .checked_mul(frame - warmup)
                .and_then(|d| start.checked_add(d))
        {
            let now = Instant::now();
            if now < deadline {
                std::thread::sleep(deadline - now);
                // A wake-up past the tolerance still started the frame
                // late — a missed deadline all the same.
                if Instant::now().saturating_duration_since(deadline) > PACING_TOLERANCE {
                    missed_deadlines += 1;
                }
            } else if frame > warmup {
                missed_deadlines += 1;
            }
        }
        let cpu_start = affinity::current_cpu();
        let t0 = Instant::now();
        engine.encode(input)?;
        let t1 = Instant::now();
        let submit = engine.submit(u64::from(frame), false)?;
        let t2 = Instant::now();
        let cpu_end = affinity::current_cpu();
        if frame >= warmup {
            samples.push(FrameSample {
                encode_seconds: t1.duration_since(t0).as_secs_f64(),
                submit_seconds: t2.duration_since(t1).as_secs_f64(),
                gpu_seconds: None,
                cpu_start,
                cpu_end,
                migrated: matches!((cpu_start, cpu_end), (Some(a), Some(b)) if a != b),
                passes: Vec::new(),
                phases: submit.phases,
            });
        }
        attribute_gpu(&mut samples, warmup, submit.gpu);
    }
    attribute_gpu(&mut samples, warmup, engine.finish_gpu()?);
    Ok(Window {
        samples,
        meter,
        start,
        missed_deadlines,
    })
}

/// Stores each GPU timing on the measured sample of the frame it times;
/// warmup frames' timings are dropped.
fn attribute_gpu(samples: &mut [FrameSample], warmup: u32, gpu: Vec<crate::GpuSample>) {
    for timing in gpu {
        let Some(index) = timing.frame.checked_sub(u64::from(warmup)) else {
            continue;
        };
        let sample = &mut samples[usize::try_from(index).expect("a frame index fits usize")];
        sample.gpu_seconds = timing.gpu_seconds;
        sample.passes = timing.passes;
    }
}

/// `--rate` validation + derivation: the per-frame period must fit a
/// `Duration` (a rate near zero overflows it) and the paced window
/// the pacing clock, else deadlines would silently break mid-run.
/// Returns the period and the whole paced window.
fn pacing(rate: f64, frames: u32) -> Result<(Duration, Duration), BenchError> {
    let period = Duration::try_from_secs_f64(1.0 / rate).map_err(|_| {
        BenchError::Engine(format!(
            "--rate {rate} Hz gives an unrepresentable pacing period"
        ))
    })?;
    let window = period.checked_mul(frames).ok_or_else(|| {
        BenchError::Engine(format!(
            "--rate {rate} Hz over --frames {frames} overflows the pacing clock"
        ))
    })?;
    Ok((period, window))
}

fn measure_scene(
    engine: &mut dyn Engine,
    dir: &Path,
    frames: u32,
    warmup: u32,
    pinned: Option<&[u32]>,
    rate: Option<f64>,
    measure_energy: bool,
) -> Result<MeasureReport, BenchError> {
    let scene = Scene::load(dir)?;
    let blobs = convert::load_blobs(&scene, dir)?;
    let input = EncodeInput {
        scene: &scene,
        blobs: &blobs,
    };
    let heterogeneous = affinity::cpu_freqs_differ();
    if pinned.is_none() && heterogeneous {
        tracing::warn!(
            "CPUs report differing max frequencies; this run's placement is uncontrolled — \
             pass --cpu (e.g. --cpu 7) to pin to one cluster"
        );
    }
    let prepare_cpu = affinity::current_cpu();
    let t_prepare = Instant::now();
    engine.prepare(&input)?;
    let prepare_seconds = t_prepare.elapsed().as_secs_f64();
    let (period, window_hint) = match rate {
        Some(hz) => pacing(hz, frames).map(|(p, w)| (Some(p), w))?,
        None => (None, Duration::from_secs(30)),
    };
    let window = run_frames(
        &mut *engine,
        &input,
        frames,
        warmup,
        period,
        window_hint,
        measure_energy,
    )?;
    let window_end = Instant::now();
    let window_seconds = window_end.duration_since(window.start).as_secs_f64();
    let energy_outcome = window
        .meter
        .map(|m| m.finish(window.start, window_end, frames))
        .transpose()?;
    let missed_deadlines = window.missed_deadlines;
    let samples = window.samples;
    let pacing = rate.map(|requested_hz| Pacing {
        requested_hz,
        achieved_hz: if window_seconds > 0.0 {
            f64::from(frames) / window_seconds
        } else {
            0.0
        },
        missed_deadlines,
        window_seconds,
    });
    let conditions = conditions::collect(
        energy_outcome
            .as_ref()
            .and_then(|o| o.thermal_pressure.clone()),
    );
    let placement = Placement {
        requested: pinned.map(<[u32]>::to_vec),
        controlled: pinned.is_some(),
        heterogeneous,
        prepare_cpu,
        cpus: cpu_use(prepare_cpu, &samples),
        migrated: u32::try_from(samples.iter().filter(|s| s.migrated).count()).unwrap_or(u32::MAX),
    };
    let enc: Vec<f64> = samples.iter().map(|s| s.encode_seconds).collect();
    let sub: Vec<f64> = samples.iter().map(|s| s.submit_seconds).collect();
    let gpu: Vec<f64> = samples.iter().filter_map(|s| s.gpu_seconds).collect();
    let passes = pass_percentiles(&samples);
    let phases = phase_percentiles(&samples);
    Ok(MeasureReport {
        engine: engine.info().name,
        info: engine.info().clone(),
        scene: dir
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        width: scene.width,
        height: scene.height,
        prepare_seconds,
        warmup_frames: warmup,
        samples,
        placement,
        percentiles: Percentiles {
            encode_seconds: percentiles(&enc).unwrap_or([0.0; 3]),
            submit_seconds: percentiles(&sub).unwrap_or([0.0; 3]),
            gpu_seconds: percentiles(&gpu),
            passes,
            phases,
        },
        pacing,
        energy: energy_outcome.map(|o| o.report),
        conditions,
        counters: engine.counters(),
        device: engine.device(),
    })
}

/// Percentiles per pass index, over the frames with the modal pass
/// count.
fn pass_percentiles(samples: &[FrameSample]) -> Vec<PassPercentiles> {
    // The modal pass count; frames with a different count are skipped.
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for s in samples {
        *counts.entry(s.passes.len()).or_default() += 1;
    }
    let Some(mode) = counts.into_iter().max_by_key(|(_, n)| *n).map(|(n, _)| n) else {
        return Vec::new();
    };
    if mode == 0 {
        return Vec::new();
    }
    let frames: Vec<&FrameSample> = samples.iter().filter(|s| s.passes.len() == mode).collect();
    (0..mode)
        .map(|i| {
            let times: Vec<f64> = frames
                .iter()
                .filter_map(|s| s.passes[i].gpu_seconds)
                .collect();
            PassPercentiles {
                name: frames[0].passes[i].name.clone(),
                width: frames[0].passes[i].width,
                height: frames[0].passes[i].height,
                format: frames[0].passes[i].format.clone(),
                gpu_seconds: percentiles(&times),
            }
        })
        .collect()
}

/// Percentiles per phase index, over the frames with the modal phase
/// count.
fn phase_percentiles(samples: &[FrameSample]) -> Vec<PhasePercentiles> {
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for s in samples {
        *counts.entry(s.phases.len()).or_default() += 1;
    }
    let Some(mode) = counts.into_iter().max_by_key(|(_, n)| *n).map(|(n, _)| n) else {
        return Vec::new();
    };
    if mode == 0 {
        return Vec::new();
    }
    let frames: Vec<&FrameSample> = samples.iter().filter(|s| s.phases.len() == mode).collect();
    (0..mode)
        .map(|i| {
            let times: Vec<f64> = frames.iter().map(|s| s.phases[i].seconds).collect();
            PhasePercentiles {
                name: frames[0].phases[i].name.clone(),
                seconds: percentiles(&times).unwrap_or([0.0; 3]),
            }
        })
        .collect()
}

/// Record a scene the adapter cannot execute faithfully.
fn write_unsupported(
    engine: &dyn Engine,
    dir: &Path,
    feature: cherenkov_scene::Feature,
    api: Option<&'static str>,
    out: &Path,
) -> Result<(), BenchError> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_json(
        &UnsupportedReport {
            engine: engine.info().name,
            info: engine.info().clone(),
            scene: dir
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            unsupported: feature,
            missing_api: api,
        },
        out,
    )
}

fn write_render(rendered: &RenderOutput, out: &Path) -> Result<(), BenchError> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let stem = out.parent().map_or_else(
        || out.with_extension(""),
        |p| p.join(out.file_stem().unwrap_or_default()),
    );
    rendered
        .image
        .write_png(&stem.with_extension("engine.png"))?;
    metrics::write_heatmap(
        &rendered.heatmap,
        rendered.image.width,
        rendered.image.height,
        &stem.with_extension("heatmap.png"),
    )?;
    write_json(&rendered.report, out)
}

fn write_json<T: serde::Serialize>(v: &T, path: &Path) -> Result<(), BenchError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut text = serde_json::to_string_pretty(v)
        .map_err(|e| BenchError::Engine(format!("json serialize: {e}")))?;
    text.push('\n');
    std::fs::write(path, text)?;
    Ok(())
}
