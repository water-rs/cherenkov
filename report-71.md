# CPU sparse coverage (#71)

## Result

The worktree contains a reviewed exact sparse coverage compiler and routes
general clip realization through it. The full draw/shading rasterizer port was
measured and reverted because it regressed frame time. The issue acceptance
target is therefore not met by this worktree: no valid old/new Pixel A/B exists,
the full 398-scene corpus gate did not complete, and GPU identity was not run.

## Design

The compiler uses an f64 event sweep per device row. It splits at edge endpoints
and crossings, resolves each operand's winding rule before integration, and
stores positive coverage as flat row offsets plus constant or sampled spans.
Horizontal edges remain connectivity events. This fixes the geometric error in
the old clip path, which multiplied already-integrated masks.

The historical shadow implementation was not reused: it clips the caster before
blur, while the current oracle clips the post-convolution silhouette. No shared
front end or GPU code was changed.

## Changes

- `cpu/src/render/coverage.rs`: exact sparse compiler and unit tests.
- `cpu/src/render/lower.rs`: general clip realization calls the sparse compiler,
  then expands the result to the existing dense clip representation so current
  compositing semantics remain unchanged.
- `cpu/src/render/mod.rs`, `bench/src/cherenkov_cpu_ad.rs`,
  `bench/README.md`: lower and encode phase reporting.
- `cpu/src/render/raster.rs`: `Edge` equality for compiler tests.

Lowering/front-end changes: only `cpu/src/render/lower.rs` clip realization;
there are no changes under the shared front end (`src/`).

## Correctness

The following passed after the final revert of the measured draw experiment:

```text
cargo test -p cherenkov-cpu --all-targets: PASS
cargo clippy -p cherenkov-cpu --all-targets -- -D warnings: PASS
git diff --check: PASS
```

The cached-reference sample render reused four references from the interrupted
run. Their metrics were unchanged from the earlier pass:

| Scene | FLIP mean | FLIP max | max local error |
| --- | ---: | ---: | ---: |
| anim-curve-slide | 0.0032196 | 0.0185372 | 0.0034519 |
| anim-paint-curve | 0.0026781 | 0.0184113 | 0.0034323 |
| anim-paint-spring | 0.0033652 | 0.0304809 | 0.0062576 |
| anim-spring-card | 0.0031777 | 0.0305410 | 0.0062576 |

The full corpus has 398 generated scenes. The canonical reference command was
started, but it runs serially in this checkout: the first five scenes consumed
71 seconds. It was stopped before completion rather than extrapolating an
incomplete cache. GPU corpus identity was not run; GPU sources were untouched.

## Pixel measurements

Device protocol: Pixel 9 Pro, screen off, CPU 7 for one worker and CPUs 4–7 for
four workers, 30 measured frames after five warmups. Every run was preceded by
`dumpsys battery` and `dumpsys thermalservice`; observed battery temperature was
28.5–37.1 °C and thermal status was 0.

The prior new-only measurements (before the draw experiment) were:

| Scene | 1-thread round | submit p50/p99 (s) | encode p50/p99 (s) |
| --- | --- | ---: | ---: |
| map | 1 | 0.05692 / 0.05771 | 0.0451 / 0.0456 |
| map | 2 | 0.05705 / 0.05820 | 0.0452 / 0.0459 |
| map | 3 | 0.05797 / 0.05901 | 0.0459 / 0.0470 |
| chart | 1 | 0.03585 / 0.03633 | 0.02068 / 0.02097 |
| chart | 2 | 0.03613 / 0.03679 | 0.02080 / 0.02109 |
| chart | 3 | 0.03615 / 0.03677 | 0.02073 / 0.02121 |
| text-page | 1 | 0.02026 / 0.02198 | 0.01975 / 0.02137 |
| text-page | 2 | 0.02023 / 0.02144 | 0.01972 / 0.02084 |
| text-page | 3 | 0.02015 / 0.02169 | 0.01965 / 0.02108 |
| ui-list | 1 | 0.04941 / 0.05089 | 0.04806 / 0.04955 |
| ui-list | 2 | 0.05110 / 0.05522 | 0.04977 / 0.05385 |
| ui-list | 3 | 0.05074 / 0.05356 | 0.04939 / 0.05224 |
| effects | 1 | 0.11045 / 0.11200 | 0.11003 / 0.11159 |
| effects | 2 | 0.11135 / 0.11327 | 0.11094 / 0.11287 |
| effects | 3 | 0.11111 / 0.11269 | 0.11070 / 0.11227 |

Four-worker new-only runs were slower on map, chart, and effects; they are not
claimed as scaling evidence. No old/dev executable was available, so these are
not A/B comparisons.

The attempted full draw integration was measured once on map after rebuilding:
submit p50 was 0.1069–0.1090 s and encode p50 was 0.0659–0.0679 s across three
rounds. The prior path was restored immediately after this regression was
observed.

## Gate lines

```text
SCENE_GENERATION: PASS — 398 corpus scenes and 9 perf scenes generated.
CPU_UNIT_AND_INTEGRATION: PASS.
CLIPPY: PASS — cherenkov-cpu all targets, -D warnings.
ANDROID_RELEASE_BUILD: PASS — aarch64-linux-android with NDK linker.
CPU_CORPUS: INCOMPLETE — representative cached-reference sample passed; full cache run stopped after 5/398 scenes.
GPU_CORPUS_IDENTITY: NOT RUN.
PIXEL_ABAB: NOT RUN — no baseline executable.
CALLGRIND_IR: RESERVED FOR THE EXTERNAL LINUX GATE.
```

## Commits

No commits were created. The work remains uncommitted for review because the
measured draw path regressed and the acceptance gates are incomplete.

## Plan reconciliation

The plan of record's exact geometric intersection design agrees with the
compiler implementation. Its shadow clipping assumption conflicts with the
current oracle semantics; the current post-convolution clip behavior was kept.
Measured Pixel data overrides the plan's expected speedup for the attempted
full draw port.
