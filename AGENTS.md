# AGENTS.md — filtrate

GPU filter runtime for WaterUI: typed `Filter` graphs compile into wgpu
pipelines (fused fragment passes for color stages, compute passes for
spatial stages, fragment translations of the same spatial bodies on
compute-less devices such as WebGL2).

## Commands

- `cargo test --workspace` — unit + GPU integration tests (needs a GPU
  adapter; CI runners and dev machines have one).
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo fmt --all` (rustfmt via the workspace toolchain)
- `cargo check -p filtrate --target wasm32-unknown-unknown --features webgl`
  — the wasm/WebGL build leg. `webgl` is wasm-only: enabling it elsewhere is
  a `compile_error!`, so never run `--all-features` on a native target.
- `cargo bench -p filtrate --bench gpu_runtime` — GPU pass benchmarks.

## Testing guidance

- **Pixel-level divergence between GPU execution paths is tolerable.** The
  same shader body can run as a compute dispatch or a fragment draw; the
  paths round intermediates differently (storage-texture + blit vs direct
  attachment writes), so a few-LSB per-pixel differences are expected. Assert
  with a small per-channel tolerance, not byte equality.
- **Prefer visual tests.** `gpu_export_filter_gallery_images` renders every
  built-in filter through *both* spatial backends into
  `/tmp/waterui_filter_gallery/` (compute) and
  `/tmp/waterui_filter_gallery_fragment/` (WebGL2 path) — run it and eyeball
  or diff the outputs when touching shaders or the pass planner.
- **Backend parity is contractual.** `SpatialExecution::ForceFragment`
  forces the WebGL2 path on any adapter, which is how the fragment path is
  exercised without a browser. A spatial body that fails the fragment
  translation contract (one `textureStore(output_texture, …)` per own-pixel)
  fails loudly at specialization, not silently at runtime.

## Structure

- `core/` — `filtrate-core`: the `Filter` trait, params, chains (no GPU).
- `derive/` — `filtrate-derive`: the `#[derive(Filter)]` proc macro.
- `src/shaders/` — WGSL sources, `include_str!`'d by the runtime;
  `shared/` holds the preamble pieces composed by `src/runtime/shader.rs`.
- `src/runtime/` — stage fusion, pass planning, pipelines, command encoding.
