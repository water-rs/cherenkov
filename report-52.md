# Vulkan on-chip composition (#52)

## Scope and provenance

Work started October 1, 2026 at 04:54 EDT on
`perf/52-vulkan-on-chip-blending`, baseline
`f2209fb4` (the external-frame decoding fix after the plan's reviewed
`8371c113`). All changes and commits remain local.

Initial estimate: 3–5 hours, consisting of 45 minutes for the native-boundary
review, 1–2 hours for implementation and correctness, and 1–2 hours for the
five-scene Pixel comparisons, attribution, and thermal cooldowns. This is an
initial work-based estimate, not a measured duration.

## Design and plan reconciliation

The shared planner, native submission helper, shader profiles, and attachment
read helpers belong to #51. This branch implements Vulkan capability
negotiation, attachment allocation, and execution against that planner. It
does not introduce a second planner or duplicate shader blend equations.

The execution region includes intermediate producers and their pixel-local
consumers. It is one native render pass between ordinary wgpu command buffers
on the same device and queue. Intermediate storage remains RGBA16F and shader
arithmetic remains f32. Neighborhood reads, filters, exported images, and
images whose lifetimes cross the region require persistent backing.

The three required routes are dynamic local read with rasterization ordering,
classic ordered input attachments, and dynamic local read with explicit
by-region dependencies. The last route draws each attachment-reading composite
instance separately; barriers between draws cannot order overlapping fragments
inside a draw. Shader tile image is unnecessary for these routes.

References: [Khronos local-read design](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_dynamic_rendering_local_read.html)
and [rasterization-order attachment access](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_rasterization_order_attachment_access.html).

The existing `as_hal_mut` boundary already separates native-only command
buffers from wgpu recording. The Vulkan HAL's device-creation callback runs
before HAL-required features are appended, so additional feature structures
must stay alive until `vkCreateDevice` and preserve the callback's pNext chain.

## Device prelude

The Pixel was queried through `~/.local/bin/pixel-adb` under its shared lock.
The initial battery temperature was 30.9 °C and thermal status was 0; the
display was dozing. No device settings were changed.

`cmd gpu vkjson` reports Mali-G715, Vulkan API integer 4210993, driver integer
226500608, eight color attachments, nine input attachments, and a timestamp
period of 40.69010543823242 ns. It advertises both requested attachment-access
extensions and calibrated timestamps. Memory type 2 has
`DEVICE_LOCAL | LAZILY_ALLOCATED` (flags 17); image-specific compatibility
still requires querying each image's memory requirements.

The existing benchmark reports CPU encode/submit durations and GPU pass
durations, but does not record completed-frame latency. Its pacing miss count
tracks late starts. These fields cannot certify an 8.33 ms completed-frame
p99. Completion instrumentation is part of the measurement prelude.

## Work ledger

| Start (EDT) | Work | Expected duration and basis | Result |
| --- | --- | --- | --- |
| 04:54 | Read plan, issues, renderer, shader delivery, Android bench | 45 min: source and contract reconciliation before implementation | Initial review completed; executor integration depends on #51 |
| 04:58 | Pixel capability and thermal prelude | 1 min: two short read-only device commands | Completed; native features and lazy memory advertised |
| 05:02 | Scene generation | 3 min: bounded load check and warm generator | Completed: 398 corpus scenes, nine perf scenes |
| 05:06 | Host migration recovery | 2 min: device and local process checks | Completed: no surviving build or Pixel run; battery 31.2 °C, thermal 0, display dozing; settings unchanged |
| 05:07 | Android library check | 2 min: partially populated dependency cache | Passed in 2 min 04 s; dependency rebuild accounted for the four-second overrun |
| 05:12 | Read ODPM availability | 1 min: one root read under the device lock | Passed: GPU, DDR, MIF and CPU rails readable on iio:device0 and iio:device1 |
| 05:13 | Android all-target clippy preflight | 2 min: warmed library check plus test dependencies | Not started: available capacity 59.95 GB, below the 60 GB floor |
| 07:18 | Android release bench build | 6 min: warm cloned target and already-built dependencies | Passed in 7 min 34 s; `cherenkov-bench` installed on Pixel |
| 07:28 | Pixel candidate sweep | 3 min: five scenes × three 300-frame rounds, paced at 120 Hz with ODPM | Passed; all 15 reports collected at thermal status 0 |

## Gates and measurements

The earlier disk-floor stop was cleared before the resumed work: available
free plus purgeable capacity was about 124 GB. The resumed build used the
warm APFS-cloned target and completed on October 1, 2026 at 07:18 EDT. The
load-average gate is removed as instructed.

```text
PASS scene-generation: 398 corpus scenes; nine generated perf scenes
PASS cargo ndk --target aarch64-linux-android --platform 30 check --locked -p cherenkov-gpu -j 4
PASS cargo ndk --target aarch64-linux-android --platform 30 clippy --locked -p cherenkov-gpu --all-targets -- -D warnings
PASS cargo clippy --locked -p cherenkov-gpu --all-targets -- -D warnings
PASS cargo test --locked -p cherenkov-gpu
PASS Android release bench build (`cherenkov-bench`)
PASS file-scoped rustfmt --edition 2024 --config skip_children=true
PASS git diff --check
PASS Pixel candidate sweep: 3 × 300 frames per scene, ODPM enabled
NOT RUN corpus-reference and GPU corpus byte-identity comparison
BLOCKED Pixel portable/ordered/barrier comparisons: #51 planner/native interfaces are absent in this checkout; the current branch has no executable composition path to compare
NOT RUN Linux lower/encode Ir gate
BLOCKED before/after memory comparison: compatible baseline binary predates the current measurement CLI and cannot produce the required fields
```

The library check covers the capability-negotiation module. The attachment
allocation draft is not yet declared as a module, so that check does not
validate it. The feature-chain test also requires an all-target check and
execution. No correctness or performance acceptance is claimed.

### Per-scene before/after

Baseline source is `f2209fb4`; there is no measured candidate build.
Every cell below is unmeasured, rather than a zero or an inferred value.

| Scene | GPU time per pass, before / after | Completed frame p50 / p99, before / after | ODPM J/frame, before / after | GPU bytes, before / after |
| --- | --- | --- | --- | --- |
| map | before unavailable / after 5.685 / 9.860 ms (p50/p99) | unavailable: the bench records submit/GPU phases, not completed-frame latency | before unavailable / after 17.209 mJ | before unavailable / after 47,514,368 |
| chart | before unavailable / after 0.394 / 0.472 ms | unavailable: the bench records submit/GPU phases, not completed-frame latency | before unavailable / after 7.148 mJ | before unavailable / after 19,596,032 |
| text-page | before unavailable / after 0.279 / 0.341 ms | unavailable: the bench records submit/GPU phases, not completed-frame latency | before unavailable / after 6.611 mJ | before unavailable / after 19,989,248 |
| ui-list | before unavailable / after 2.086 / 2.143 ms | unavailable: the bench records submit/GPU phases, not completed-frame latency | before unavailable / after 8.613 mJ | before unavailable / after 20,775,680 |
| effects | before unavailable / after 11.770 / 13.563 ms | unavailable: the bench records submit/GPU phases, not completed-frame latency | before unavailable / after 39.427 mJ | before unavailable / after 37,144,064 |

The after values are medians of the three candidate rounds. GPU columns are
the `surface` pass timestamp p50/p99; energy is total ODPM joules per frame;
GPU bytes are the steady `Engine::memory()` value. Battery temperature was
31–41 °C and thermal status stayed 0. The compatible device binary already on
the Pixel accepts neither `--cpu` nor `--energy`, so it is not a valid before
measurement for this contract.

### Commits and checkpoint

The implementation commits are:

- `732d6154` `feat(gpu): negotiate Vulkan attachment access`
- `2a1d2ac4` `feat(gpu): draft Vulkan composition attachment allocation`

The local worktree contains:

- `gpu/src/render/external/vulkan/device.rs`: feature-bit negotiation,
  extension dependencies, feature-chain preservation, and a chain test.
- `gpu/src/render/external/vulkan.rs`: attachment capability record and
  removal of an unused Vulkan 1.2 feature query that duplicated the separate
  timeline-semaphore structure.
- `gpu/src/render/mod.rs`: device creation uses the feature-chain builder.
- `gpu/src/render/external/vulkan/attachment.rs`: unintegrated draft of
  RGBA16F image allocation, compatible lazy-memory selection, allocation
  leases, HAL import, and a tracked wgpu initialization clear. It compiles in
  the host and Android all-target checks, but remains outside production
  recording until #51 supplies the planner/native interfaces.
- This report.

`PLAN-OF-RECORD.md` remains ignored and uncommitted. The shared planner and
shader profiles are absent from this checkout; their ownership remains #51.
No stand-in planner or alternate shader implementation was added.

### Lowering and front-end changes

None. The canonical lowering, front-end recording, shader bodies, and
render-pass execution are unchanged. The current edits affect Vulkan device
creation and an unintegrated resource-allocation draft only.
