# AGENTS.md

Cherenkov is the 2D engine under Hydrolysis and WaterUI's self-drawn
components (#1). It has a GPU backend (`gpu/`, on wgpu) and a CPU backend
(`cpu/`, including the microcontroller design point), selected at build time
and never as a runtime fallback. The rules below are the engine's standing
contract; the per-decision log is issue #2.

## Correctness

- **The oracle is the reference.** The corpus (`scenes/corpus`) is rendered
  against the independent f64 oracle. The original 83 scenes keep bit-identical
  metrics and PNG bytes across every change that is not meant to change pixels.
  A change that intends to change pixels (for example #71) must be no worse
  than dev against the oracle on every scene, and better overall.
- **Every new capability gets corpus scenes** on each backend that claims it.
- **Visual review is done by eye.** Images are reviewed by looking at them,
  never by pixel-count, brightness or dominant-colour heuristics.
- **Fail fast.** An unsupported case returns an explicit `Unsupported` error,
  and an invalid input (a non-finite or non-invertible transform, say) is an
  error. Nothing is silently substituted, dropped or rendered transparent.
- **No workarounds in consumers.** A capability a consumer needs and the
  engine lacks becomes an engine issue (#65 collects them). Consumers never
  work around it with local rasterization, glyph outlining, an alternate
  renderer or a dropped feature.

## Performance

- **The goal** is to beat Skia Graphite, Vello classic and Vello hybrid in the
  same harness, on every scene and on the devices of each stage (#1).
- **Frames** target 120fps: 8.33 ms at p99. 60fps is accepted only where no
  renderer can fit the budget in theory.
- **CPU-side changes are accepted on deterministic Callgrind instruction
  counts:** Ir for `lower` and `encode` at the second steady frame of the five
  perf scenes (the #43 harness), at most +1% per scene against dev. Wall-clock
  time on a shared cloud VM is not evidence: identical binaries drifted by up
  to ±80% there.
- **Wall-clock time and energy come only from quiet real devices**, in
  interleaved A/B rounds with per-round results: the Apple M1, the iPad Pro M4
  (Metal) and the Pixel 9 Pro (Vulkan). The iPad's GPU clock is bimodal with
  device state.
- **Memory is a first-class cost.** A cache or retained structure has to pay
  for its memory in measured time or energy.

## Platform

- **wgpu stays** unless it is shown to be the bottleneck that keeps Cherenkov
  from beating Skia. The Rust GPU ecosystem is built on it.
- **Hardware features wgpu does not expose** are reached through passthrough
  shaders and `as_hal` on the same device and queue: on-chip blending, tile
  memory, memoryless attachments (#50–#53).
- **Engine shaders are precompiled at build time** (WGSL through naga to
  metallib or SPIR-V) and loaded as passthrough shaders (#57). naga still links
  through wgpu, and that is accepted.
- **The hardware floor** is on-chip programmable blending plus f16 arithmetic
  on every supported device (#1). There is no low-end tier.
- **Upgrading wgpu** moves the whole stack at once, because an application can
  link only one wgpu (#63).

## API and review

- **Public API changes** follow `docs/api.md`'s style, and every decision is
  recorded as a comment on #2 with its rationale.
- **Untrusted code is reviewed line by line.** Code produced by a weaker model,
  or by sessions whose model is uncertain, is never merged or cherry-picked
  wholesale; each commit gets a recorded verdict.
- **Measurements say what they cover.** Every report names both builds by
  commit and states what each measured interval includes.
