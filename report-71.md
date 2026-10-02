# CPU sparse coverage (#71)

## Checkpoint: October 2, 2026

Acceptance is not met. Work stopped at the mandatory disk floor:
`python3 ~/.cache/cutover/avail.py` returned `59673504386` bytes
(59.67 GB available, including purgeable space). The baseline Cargo build
was interrupted and its process exited. No task-owned jobs remain running.

## Rebase and verification

Branch: `perf/71-cpu-sparse-coverage`.
Baseline: `2cae0517`, fetched from `origin/dev`.
Tested candidate: `deae2648` (the report update changes documentation only).

The three inherited commits rebased to:

- `0ac3d76b`: exact sparse clip coverage compiler.
- `886f018a`: CPU render phase reporting.
- `deae2648`: inherited measurement report.

The phase-reporting conflict in `cpu/src/render/mod.rs` was resolved against
dev's projective-layer implementation: `lower_items` already resolves glyphs
and returns their count and the optional lowering timestamp. Phase reporting
uses that result without resolving glyphs a second time. Lowering includes
projective realization and glyph resolution; encode covers band shading.

| Command | Result |
| --- | --- |
| `gh issue view 71 -R water-rs/cherenkov` | Read the specification and acceptance criteria before other work. |
| `git fetch origin dev` / `git rebase origin/dev` | Successfully rebased onto `2cae0517`. |
| `cargo test --locked -p cherenkov-cpu --all-targets` | PASS: 118 tests across 20 test executables; zero failures. |
| `cargo clippy --locked -p cherenkov-cpu --all-targets -- -D warnings` | PASS: `Finished dev profile`. |
| `rustfmt --edition 2024 cpu/src/render/mod.rs` | PASS: no changes; status checked before and after. |
| `uv run --python 3.12 --with-requirements scenes/fonts/tools/requirements.txt python scenes/tools/generate.py` | PASS: `corpus written count=420`; `perf set written count=9`. |
| `cargo build --locked --release -p cherenkov-bench --features cherenkov,cherenkov-cpu --bin cherenkov-bench` | PASS: `Finished release profile [optimized]`. |
| `target/release/cherenkov-bench reference --corpus scenes/corpus --out-dir out/71/rebased/reference` | PASS: `reference pass totals scenes=420 threads=12`; exit 0. |
| Same release build in the baseline worktree | Interrupted at the disk floor; not a passing build. |
| `git diff --check` | PASS. |

The regenerated corpus contains the inherited 398 scenes plus 22 scenes added
on dev. All 420 reference files exist; there are no missing scenes or temporary
reference files. The reference run completed at 12:06:14 EDT on October 2.
This is oracle generation, not the CPU quality or GPU identity gate.

## Source findings

- `bench/src/cli.rs:890`: `reference_cmd` already uses scoped workers,
  `available_parallelism`, and an atomic scene queue. This implementation is
  present in both `f2209fb4` and `2cae0517` (from `6cab84fd`). The inherited
  report's serial-path diagnosis was not reproduced. The rebuilt executable
  completed the canonical command on 12 threads; no replacement script or
  source change was needed for parallel reference generation.
- `cpu/src/render/lower.rs:1042`: general clips still rasterize each shape
  separately, expand sparse coverage to a full-surface dense mask, and multiply
  it by the existing clip. Although `coverage::rasterize` accepts geometric
  intersections, nested clips do not yet use that capability.
- `cpu/src/render/raster.rs:1287`: draw shading still deposits into `Accum`,
  walks pixel coverage, evaluates paint, and composites each pixel scalarly.
- `bench/src/cherenkov_cpu_ad.rs:1438`: static benchmark scenes re-record their
  contents each frame. A profiling investigation of the full sparse port must
  include compilation cost on that path, not only retained-frame shading.

These are source observations, not a completed profile of the reverted port.
The inherited full draw/shading experiment remains reverted. Its historical
new-only Pixel timings do not establish an old/new comparison.

## Measurements and device

No new frame-time, instruction-count, or memory comparison was completed.
There are no accepted p50/p99 or old/new memory numbers for this checkpoint.

The read-only probe through `~/.local/bin/pixel-adb` acquired the device lock
and returned `Pixel 9 Pro`, battery `temperature: 277` (27.7 degrees C), and
`mHalInteractiveModeEnabled=false`. No benchmark was launched on the device.
The Mac ran correctness tests and oracle generation only, not performance
measurements.

## Resume point

After available disk capacity is restored to at least 60 GB, the next action
is to finish the baseline release build in
`../cherenkov-wt-71-baseline` (branch `verify/71-cpu-baseline`, commit
`2cae0517`). Its generated scene inputs are copies of this task's regenerated
inputs. Both worktrees retain their build caches.

Then render baseline and candidate CPU and Metal corpora against the completed
`out/71/rebased/reference` cache. Compare CPU FLIP mean and maximum local error
per scene, require improvement overall, and compare GPU metrics and PNG bytes
for identity. After those checks, profile and complete sparse draw coverage,
geometric clipping, and SIMD shading while preserving bounded band memory.
Measure dev versus the resulting candidate on the locked, screen-off Pixel in
interleaved A/B/A/B rounds, with one worker on CPU 7 and eight workers on CPUs
0-7; report shade/frame p50/p99 and memory, stopping above 45 degrees C.

Local logs and generated references are under `out/71/rebased/`, ignored by
Git. No branch was pushed and no PR was opened.
