# filtrate

GPU texture filter library built on `wgpu`. Filters are declared as pure
data, fused into as few GPU passes as possible, and executed by a runtime
that handles parameter animation, HDR intermediates, and scratch-texture
reuse. `filtrate` powers WaterUI's visual-effect modifiers but has no
WaterUI dependency — it works on images, video frames, or any `wgpu`
texture in any Rust application.

## Crates

| Crate | Role |
| --- | --- |
| `filtrate` | Built-in filter library, WGSL shaders, and the GPU runtime (`FilterAdapter`). |
| `filtrate-core` | `no_std` abstraction layer: the pure-data `Filter` trait, `Chain` composition, and parameter/stage visitors. Stable surface with no `wgpu` dependency. |
| `filtrate-derive` | `#[derive(Filter)]` for the regular single-pass filter shapes. |

## Quick start

```rust
use filtrate::{FilterAdapter, FilterExt};
use filtrate::filters::{Blur, Brightness, Grayscale};

// Chain filters; adjacent color-only filters fuse into one GPU pass.
let chain = Grayscale(1.0).then(Blur(5.0)).then(Brightness(0.2));
let effect = FilterAdapter::new(chain);
// Hand `effect` to your render loop: `Effect::setup`, then
// `Effect::render` once per frame.
```

Reactive frontends implement `FilterParam` for their signal types so filter
parameters animate without rebuilding the pipeline; plain `f32` works for
static values.

## Design notes

- **Fusion**: consecutive `COLOR_ONLY` filters compile into a single
  fragment shader; spatial filters (blurs, convolutions, distortions) each
  get a compute pass with automatic scratch ping-pong.
- **Color contract**: premultiplied alpha end to end; texel values are
  filtered as sampled, with no implicit sRGB conversion.
- **HDR**: intermediates prefer `Rgba16Float` and degrade to LDR only
  where the policy allows (`FilterAdapter::require_hdr` /
  `FilterAdapter::force_ldr`).

## WebGL (wasm)

The optional `webgl` feature turns on wgpu's WebGL2 backend. It only
exists for `wasm32-unknown-unknown`; enabling it on a native target is a
`compile_error!`.

WebGL2 has no compute shaders or storage textures, so spatial stages run
through a **fragment translation** of the same WGSL body: every spatial
shader writes only its own output texel, which the runtime rewrites into a
fullscreen draw. Color stages are unchanged — they already run as fragment
passes. Selection is automatic from device limits
(`SpatialExecution::Auto`); `ForceFragment` exercises the WebGL2 path on
native GPUs for tests and benchmarks.

Honest caveats:

- **Precision**: compute can round the final stage through an
  `Rgba16Float` scratch + blit while the fragment path writes the output
  attachment directly, so outputs may differ by a ±1 u8 step per channel.
- **Performance**: on an Apple-Silicon Metal adapter (`cargo bench
  --bench gpu_runtime`), the fragment path is parity-or-faster because it
  skips the final blit — e.g. a 2048² separable blur measures 2.54 ms
  fragment vs 2.57 ms compute+blit (medians). These numbers exercise the
  same shader path WebGL runs, but cannot model browser/WebGL context
  overhead.
- A spatial body that breaks the one-store-per-own-texel contract fails
  loudly at specialization rather than silently mis-rendering.

## License

MIT OR Apache-2.0, at your option.
