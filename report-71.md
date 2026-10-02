# CPU sparse coverage (#71)

## Verdict

Acceptance is not met. The CPU corpus has eight regressions against dev,
including two images whose channels already equal the correctly rounded f16
oracle. The sparse renderer also fails the Pixel performance and memory bar.
These commits are local review material, not an accepted optimization.

Branch: `perf/71-cpu-sparse-coverage`. Dev baseline: `2cae0517`.
The Pixel baseline is `6e9f1375` in `../cherenkov-wt-71-baseline`: dev plus
the same CPU phase timestamps and bench reporting as the candidate, with no
rasterizer changes. Candidate `367942dc` contains the final Rust changes and
the device runner. Subsequent report-only changes do not change its binaries.
The final CPU corpus used the Rust sources at `c6679033`, identical to
`367942dc`. The GPU identity pass used `0ba6cc7c`; subsequent Rust changes
touch only the CPU paint kernel and CPU provenance reporting.

## Implementation and reproductions

- `b190ee5d` preserves horizontal boundaries in `flatten_edges`. Removing
  them before the sparse compiler broke fractional-row connectivity. The
  regression test returned 0.75 instead of 0.5 before the fix.
- `6af3e307` retains geometric clip operands through nested clips and draws,
  compiles exact intersections in f64, and shades nonempty spans with
  `wide` 1.7.1. Four-lane kernels cover solid paints, gradients, image
  interpolation and source-over. Irregular stop/texel lookups, mesh lookup,
  and transcendental operations retain scalar lane evaluation. Source-over
  uses fused multiply-add. Draw compilation materializes only requested
  band rows; the streaming/offscreen and band/full-field tests agree.
- `b15850bb` fixes silhouette coverage at
  `cpu/src/render/lower/silhouette.rs:66` and
  `cpu/src/render/raster.rs:1663`. The old accumulator integrated signed
  area before resolving winding. Two coincident fractional squares produced
  0.5 coverage instead of 0.25. The before-failing test now passes for both
  nonzero and even-odd winding. Convolution still follows coverage and the
  enclosing clip still follows convolution.
- `0ba6cc7c` replaces the repeated scan of all row edges at every vertical
  event with an active-edge sweep (`cpu/src/render/coverage.rs:409`). It
  preserves the previous ordering of coincident crossings. All 420 metric
  records and all 840 PNGs stayed identical to `b15850bb`.
- `ab792b37` fixes SIMD gradient extension semantics
  (`cpu/src/render/paint/batch.rs:43`). Finite transform coefficients can
  overflow a gradient parameter. The vector path incorrectly returned
  transparent before applying padding; scalar evaluation returned the
  endpoint color. The before-failing test now covers all four extension
  modes. Radial parameters without a finite solution remain transparent.
  All 420 metric records and 840 PNGs stayed identical after this fix.
- `c6679033` corrects provenance: `RasterInfo` names `wide::f32x4`, and the
  bench reports f64 geometric coverage and the actual f16/f32 framebuffer.
- `367942dc` adds `bench/scripts/pixel_cpu_ab.py`: exclusive `pixel-adb`
  locking, A/B/A/B order, binary hash verification, CPU affinity, screen-off
  enforcement, battery/status gates, and per-run thermal records. Cooldown
  waits on Android battery-change events, bounded to 30 minutes; no sleeps.

## Verification

Commands ran on the pinned stable toolchain, with sccache enabled and no
target-directory overrides. Every Cargo command used `--locked`; available
capacity was checked before each build. One silhouette Android rebuild
started before an earlier CPU test command exited, violating serialization;
both completed, and subsequent Cargo work was sequential.

| Command | Result |
| --- | --- |
| `gh issue view 71 -R water-rs/cherenkov` | Specification and acceptance criteria read. |
| `cargo check --locked -p cherenkov-cpu -p cherenkov-bench --features cherenkov-bench/cherenkov-cpu` | PASS: `Finished dev profile`. |
| Same packages/features with `cargo clippy --locked --all-targets -- -D warnings` | PASS: `Finished dev profile`; zero warnings. |
| Same packages/features with `cargo test --locked --all-targets` | PASS: 167 tests, zero failures (CPU 127; bench 40). |
| Same three gates in the instrumented baseline worktree | PASS: check, clippy, 153 tests; zero failures. |
| `rustfmt --edition 2024` on changed Rust files only | PASS; status checked before and after; module recursion disabled for root modules. |
| `git diff --check` | PASS. |
| Native release bench build, features `cherenkov,cherenkov-cpu` | PASS: `Finished release profile`. |
| Android release bench builds, target `aarch64-linux-android`, feature `cherenkov-cpu` | PASS for baseline and candidate. |
| CPU render against all 420 cached references | Completed; eight regressions, detailed below. |
| GPU render on Apple M2 Max / Metal | PASS: 420 identical metric records; all 840 engine/heatmap PNGs byte-identical to dev. |
| Pixel runner syntax and thermal boundary checks | PASS: 44.9°C/status 1 allowed; 45°C or status 2 refused. |

The reference command was already parallel at the repository root:
`bench/src/cli.rs:890` uses scoped workers and an atomic scene queue. The
canonical rebuilt command completed with `scenes=420 threads=12`. No serial
reference defect was reproduced and no replacement script was introduced.

The first Android attempt failed because cc-rs could not find
`aarch64-linux-android-clang`. The successful builds set
`CC_aarch64_linux_android` and `CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER` to
the installed NDK 27.0.12077973 `aarch64-linux-android26-clang`, and
`AR_aarch64_linux_android` to its `llvm-ar`. sccache remained enabled.

Correctness commands:

```sh
target/release/cherenkov-bench reference --corpus scenes/corpus \
  --out-dir out/71/rebased/reference
RAYON_NUM_THREADS=6 target/release/cherenkov-bench render \
  --engine cherenkov-cpu --corpus scenes/corpus \
  --reference out/71/rebased/reference --out-dir out/71/rebased/verified-cpu
target/release/cherenkov-bench render --engine cherenkov \
  --corpus scenes/corpus --reference out/71/rebased/reference \
  --out-dir out/71/rebased/active-gpu
```

## CPU correctness gate

Mean scene FLIP: dev **0.0021179351832778916**, candidate
**0.0013968286398738396**. 294 scenes improve at least one tracked metric;
this does not excuse any individual regression.

| Scene | Regressing metric | Dev | Candidate |
| --- | --- | ---: | ---: |
| backdrop-refraction-hdr | maximum local error | 0.32173681259155273 | 0.3218806054857044 |
| clip-path-oversized | mean FLIP | 0.001425045410164627 | 0.0014254371188041517 |
| fill-sharp-subpixel-p3 | mean FLIP | 0.001644982432912621 | 0.0016456663543253495 |
| fill-sharp-subpixel | mean FLIP | 0.0018535358893565497 | 0.0018537414912320957 |
| path-evenodd | mean FLIP | 0.0013458695089649602 | 0.0013474917384078211 |
| path-nonzero | mean FLIP | 0.0013418753294547326 | 0.0013428834639855525 |
| present-glass-highlights | mean FLIP | 0.0025662857665272386 | 0.002607802361705178 |
| shadow-silhouette-path | maximum local error | 0.0002050002415975305 | 0.00020518567827010337 |

The silhouette fix eliminates the affine-path and P3 regressions. Compared
with the inherited ten-regression candidate, it also changes the metrics of
three previously passing scenes: `shadow-silhouette-continuous`,
`shadow-silhouette-ellipse`, and `shadow-silhouette-hdr`. They remain no worse
than dev and their PNGs remain identical, but their metric records violate
the continuation's additional bit-identity constraint. Across the entire
corpus, six metric records and four PNGs change; those four PNGs belong to
the original failing silhouette scenes.

### Target-format rounding conflict

`CHERENKOV_CPU_READBACK=f32` is a diagnostic, not a substitute for the default
f16 gate. Both f32 corpus passes were retained. In the default f16 pass,
every channel of `fill-sharp-subpixel` and `fill-sharp-subpixel-p3` equals
`half::f16::from_f64` applied directly to the independent oracle. Their
correctly rounded images therefore have exactly the candidate FLIP means
above, which exceed dev. Meeting those thresholds requires changing either
the nearest-f16 pixels or the comparison contract. No gate exception,
per-scene branch, metric tolerance, or output-format substitution was made.

The direct quantization audit is recorded in
`out/71/rebased/ideal-f16.json`. The correctly rounded oracle also has
maximum local error `0.00020518567827010337` for
`shadow-silhouette-path`, matching its remaining local-error regression.

Two effects still need an isolated causal diagnosis. Relevant source
differences are the flattened f32 SDF boundary in
`cpu/src/render/raster.rs:469` versus the oracle's analytic distance, and
integrated Gaussian weights in `filtrate/src/cpu.rs:223` versus sampled
Gaussian weights in `oracle/src/render.rs:1233`. These observations alone
do not establish which difference causes each failing metric.

## Final Pixel A/B/A/B

Device: Pixel 9 Pro CPU, screen off, exclusive `~/.local/bin/pixel-adb`
lock for the complete matrix. A = `6e9f1375`; B = `367942dc`.
One worker used `--cpu 7`; eight workers used `--cpu 0-7`.
Each run used 30 warmup frames followed by 120 measured frames, at the
generated 1024×2216 scene size, without pacing. The matrix ran on
October 2, 2026, from **15:56:39 EDT to 16:09:31 EDT**.
All 40 runs started with battery **34.9–39.1°C** and thermal status **0**.
The admission rule was battery below 45°C and thermal status below 2.

Verified executable SHA-256:

```text
A 2ab2d1852fc34f9febcfbfe5cc997438ea2cadc7fdc32d2c075db9ebf76a900d
B 495ba745d9a74af4c287de2e31ebc1bcbe1c2bcd7be13a1f6af0363ef5aad00f
```

Reproduction command:

```sh
~/.local/bin/pixel-adb --run python3 bench/scripts/pixel_cpu_ab.py \
  --baseline ../cherenkov-wt-71-baseline/target/aarch64-linux-android/release/cherenkov-bench \
  --candidate target/aarch64-linux-android/release/cherenkov-bench \
  --baseline-rev 6e9f1375 --candidate-rev 367942dc \
  --corpus scenes/perf --out out/71/rebased/pixel-final-ab \
  --frames 120 --warmup 30
```

Frame time is each sample's `encode_seconds + submit_seconds`, then
percentiled; it is not the sum of independently computed percentiles.
It includes command recording, lowering, glyph resolution and CPU rendering.
Shade is the engine's `encode` phase: band rasterization, filtering,
compositing and output conversion. Percentiles use the harness's nearest-rank
definition. The following table pools the two runs per side, 240 samples.
Times are p50/p99 milliseconds; memory is steady `Engine::memory().cpu`,
in decimal MB. Engine GPU bytes were zero.

| Workers | Scene | Frame A → B, ms | Shade A → B, ms | Engine MB A → B |
| ---: | --- | ---: | ---: | ---: |
| 1 | map | 84.92/90.73 → 255.26/269.62 | 52.46/56.20 → 202.81/212.88 | 22.34 → 62.67 |
| 1 | chart | 45.29/47.53 → 32.17/33.99 | 25.84/27.46 → 29.96/31.53 | 18.57 → 18.86 |
| 1 | text-page | 28.42/29.75 → 26.32/26.88 | 27.82/29.16 → 25.65/26.16 | 18.94 → 18.92 |
| 1 | ui-list | 89.63/114.52 → 94.29/100.65 | 86.12/109.82 → 86.68/93.09 | 19.04 → 21.57 |
| 1 | effects | 194.75/197.41 → 182.56/197.91 | 193.09/195.63 → 178.70/193.87 | 18.27 → 19.85 |
| 8 | map | 261.82/301.55 → 293.99/385.64 | 125.39/158.62 → 154.73/214.08 | 22.34 → 62.67 |
| 8 | chart | 201.42/230.61 → 56.28/80.00 | 98.94/127.09 → 42.19/61.33 | 18.57 → 18.86 |
| 8 | text-page | 42.83/53.72 → 33.76/50.65 | 38.39/49.49 → 30.09/42.17 | 18.94 → 18.92 |
| 8 | ui-list | 82.97/105.17 → 96.79/130.68 | 61.02/75.83 → 65.01/89.52 | 19.04 → 21.57 |
| 8 | effects | 64.36/101.24 → 102.49/137.95 | 55.79/89.38 → 81.88/116.09 | 18.27 → 19.85 |

### Per-round results and process memory

Order is A1, B1, A2, B2 within each worker/scene combination. PSS columns
are steady/sampled peak, in decimal MB. Peak is the harness's maximum
snapshot across preparation, warmup and post-window sampling, not a
continuous process-memory trace. Process GPU-memory readings were
unavailable because Android reported no entry for the CPU bench process.

| Workers | Scene | Run | Frame p50/p99 ms | Shade p50/p99 ms | Engine MB | PSS steady/peak MB |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | map | A1 | 80.42/84.80 | 49.63/52.17 | 22.34 | 31.10/31.10 |
| 1 | map | B1 | 251.35/262.31 | 199.82/209.78 | 62.67 | 77.67/77.67 |
| 1 | map | A2 | 88.09/90.75 | 54.59/56.43 | 22.34 | 30.93/30.93 |
| 1 | map | B2 | 259.78/271.43 | 206.47/214.22 | 62.67 | 77.79/77.79 |
| 1 | chart | A1 | 45.58/47.55 | 26.01/27.64 | 18.57 | 25.18/25.18 |
| 1 | chart | B1 | 31.08/32.20 | 28.89/29.96 | 18.86 | 25.48/25.48 |
| 1 | chart | A2 | 45.11/46.24 | 25.76/27.02 | 18.57 | 25.20/25.20 |
| 1 | chart | B2 | 33.24/34.02 | 31.00/31.62 | 18.86 | 25.65/25.65 |
| 1 | text-page | A1 | 27.33/28.36 | 26.77/27.82 | 18.94 | 27.21/27.61 |
| 1 | text-page | B1 | 26.34/26.85 | 25.65/26.12 | 18.92 | 27.54/27.94 |
| 1 | text-page | A2 | 28.94/29.81 | 28.35/29.17 | 18.94 | 27.23/27.63 |
| 1 | text-page | B2 | 26.31/26.88 | 25.62/26.19 | 18.92 | 27.33/27.73 |
| 1 | ui-list | A1 | 75.88/89.61 | 73.19/86.12 | 19.04 | 26.67/27.24 |
| 1 | ui-list | B1 | 92.43/94.29 | 84.95/86.65 | 21.57 | 30.95/31.11 |
| 1 | ui-list | A2 | 93.95/114.60 | 90.37/110.07 | 19.04 | 26.64/27.20 |
| 1 | ui-list | B2 | 99.30/101.13 | 91.53/93.13 | 21.57 | 30.95/31.11 |
| 1 | effects | A1 | 194.69/197.42 | 193.08/195.66 | 18.27 | 23.86/23.86 |
| 1 | effects | B1 | 182.53/197.91 | 178.60/193.87 | 19.85 | 26.59/26.59 |
| 1 | effects | A2 | 194.79/196.49 | 193.17/194.89 | 18.27 | 23.73/23.73 |
| 1 | effects | B2 | 182.67/197.88 | 178.73/193.86 | 19.85 | 26.75/26.75 |
| 8 | map | A1 | 260.97/293.37 | 125.24/157.90 | 22.34 | 31.25/31.25 |
| 8 | map | B1 | 293.99/384.17 | 154.02/212.12 | 62.67 | 79.33/79.33 |
| 8 | map | A2 | 262.01/307.50 | 125.39/158.62 | 22.34 | 31.40/31.40 |
| 8 | map | B2 | 293.54/385.64 | 154.73/214.08 | 62.67 | 79.04/79.04 |
| 8 | chart | A1 | 200.81/237.61 | 97.92/127.38 | 18.57 | 25.58/25.58 |
| 8 | chart | B1 | 57.05/76.50 | 43.18/58.97 | 18.86 | 26.53/26.53 |
| 8 | chart | A2 | 202.28/227.53 | 99.54/123.00 | 18.57 | 25.61/25.61 |
| 8 | chart | B2 | 55.55/80.00 | 41.00/61.33 | 18.86 | 26.54/26.54 |
| 8 | text-page | A1 | 40.27/53.72 | 36.68/49.49 | 18.94 | 28.46/28.61 |
| 8 | text-page | B1 | 32.45/51.98 | 29.30/47.92 | 18.92 | 28.05/28.39 |
| 8 | text-page | A2 | 43.73/52.30 | 39.23/47.46 | 18.94 | 28.21/28.41 |
| 8 | text-page | B2 | 35.12/45.92 | 31.11/42.15 | 18.92 | 28.18/28.45 |
| 8 | ui-list | A1 | 83.37/105.17 | 60.88/81.17 | 19.04 | 27.37/27.77 |
| 8 | ui-list | B1 | 92.82/127.40 | 63.09/83.12 | 21.57 | 31.73/32.43 |
| 8 | ui-list | A2 | 82.06/102.90 | 61.34/74.74 | 19.04 | 27.42/27.85 |
| 8 | ui-list | B2 | 101.93/130.68 | 67.27/89.52 | 21.57 | 31.90/32.63 |
| 8 | effects | A1 | 60.00/77.79 | 51.35/68.20 | 18.27 | 24.16/24.16 |
| 8 | effects | B1 | 104.13/148.72 | 83.66/123.53 | 19.85 | 28.01/28.01 |
| 8 | effects | A2 | 77.51/103.38 | 66.98/91.38 | 18.27 | 24.17/24.17 |
| 8 | effects | B2 | 101.16/133.45 | 79.43/109.96 | 19.85 | 27.77/27.77 |

Same-binary drift remains visible despite the thermal and screen controls:
for example eight-worker effects baseline p50 moves from 60.00 to 77.51 ms.
Pooled differences alone are not proof of a stable speedup. The map and
memory regressions reject this candidate regardless. The complete final
manifest, per-frame samples and summaries are in
`out/71/rebased/pixel-final-ab/`.

## Profiling and performance findings

Only bounded `sample` captures were used on the Mac: three seconds, 10 ms
sampling interval, one map scene. The three text captures total 426,991
bytes. Their timing and process-footprint fields are not performance
evidence; the measurement-driver JSON files were deleted. Instruments CPU
Counters was not used again. The named `.ktrace` and partial `.trace`
bundle are absent.

The inherited scalar map profile places substantial lowering work in
`resolve_winding`. The sparse port removes that duplicate winding pass.
The new profile instead points to `Compiler::compile`; disassembly maps
its dominant sampled loop to rebuilding the active edges for every event
interval. `0ba6cc7c` removes that quadratic scan while preserving pixels.
It does not establish an overall win: every intersecting band still scans
the complete operands (`cpu/src/render/coverage.rs:249`, called at
`cpu/src/render/raster.rs:1250`). Geometry retention at
`cpu/src/render/lower.rs:1678` and the finer tolerance at
`cpu/src/render/lower.rs:25` also increase the map's retained memory.

In the first complete Pixel matrix, `b15850bb` had one-worker map frame
p50/p99 **308.02/328.65 ms**, versus **81.22/88.39 ms** for dev. Shade was
**259.09/278.90 ms** versus **50.44/54.86 ms**. Engine memory rose from
**22,344,352** to **62,674,720 bytes**. The active-edge matrix at `0ba6cc7c`
still failed: one-worker map frame **390.94/607.13 ms** versus
**105.48/170.06 ms**. Its two baseline map medians were 97.16 and 145.94 ms,
so those wall-clock samples cannot establish a precise speedup for the
active-edge change itself. The per-round JSON is retained.

No accepted Callgrind counts exist. For a Linux instruction-count diagnostic,
build the same revisions with `--locked`, generate the same perf inputs,
and run this from the candidate worktree; run the corresponding command
from the baseline worktree with output names using `6e9f1375`:

```sh
mkdir -p out/71
RAYON_NUM_THREADS=1 valgrind --tool=callgrind \
  --callgrind-out-file=out/71/map-367942dc.callgrind \
  target/release/cherenkov-bench measure --engine cherenkov-cpu \
  --scene scenes/perf/map --warmup 1 --frames 2 \
  --out out/71/map-367942dc.discard-timing.json
callgrind_annotate --inclusive=yes --tree=calling \
  out/71/map-367942dc.callgrind > out/71/map-367942dc.calls.txt
```

This is a whole-run CPU-only diagnostic, not the second-steady-frame Ir
acceptance gate. `bench/scripts/ir_gate.py` currently selects GPU symbols
and the `cherenkov` adapter; CPU phase selection still needs to be added
there before using it as a CPU gate.

The unresolved acceptance work is concrete: settle the f16 comparison
contract and the three changed passing metric records, diagnose the two
effect cases, then redesign coverage compilation and geometry storage to
remove the measured map cost without increasing memory. Repeat all
correctness and paired-device gates. No result here authorizes landing.

Local logs, reference images and device JSON are under ignored
`out/71/rebased/`. Task-owned incremental caches and an unused baseline
wasm target were removed (`du -sh`: 2.2G, 1.1G, and 163M respectively). The
available-capacity check afterwards returned **71,829,600,898 bytes**
(71.83 GB). No branch was pushed and no PR was opened.
