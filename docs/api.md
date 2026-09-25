# Cherenkov public API

This document is the design of Cherenkov's public API: the contract that every implementation direction satisfies, and the only surface through which the correctness oracle and the benchmark harness drive the engine. The decision log behind it is water-rs/cherenkov#2.

Sections marked **Proposal** are not yet agreed; everything else records a decision. Code blocks are signatures and usage, not implementations.

## Principles

- **The API describes what to draw.** Tiles, strips, passes and pipelines never appear in it. wgpu types appear only in the `interop` modules.
- **Semantic primitives are first-class.** A rounded rectangle, a shadow or a glyph run reaches the engine as itself, so its fast path survives. Nothing is lowered to a path at the API boundary.
- **Type safety wherever an invariant is static.** Colour spaces, image storage formats, backend capabilities, thread affinity and paired state are types. Facts that change at run time, such as a display's HDR headroom, stay values.
- **Memory is part of the design.** Shared `Picture`s, typed and compressed image storage, and GPU/CPU budgets with system memory-pressure handling are part of the API.
- **Invisible optimizations are verified invisible.** Layer caching and damage tracking must produce bit-identical output when disabled. This is exact by construction, not by tolerance:
  - Canonical f16 rounding and materialization points are part of the semantics, so a cached and an uncached render round at the same places.
  - Scroll offsets and integer layer translations snap to device pixels as part of the semantics.
  - Content under a fractional transform is re-rasterized rather than resampled from a cache. Promotion to system-compositor planes is compared against in-engine composition with a perceptual tolerance.
- **No runtime fallback.** A backend is chosen deliberately, at build time or once at process start by capability. A failure is an error.

## Crates and backends

| Crate | Directory | Contents |
|---|---|---|
| `cherenkov` | `src/` | Front end: API types, recording, layer tree, animation, CPU geometry. No GPU dependency. |
| `cherenkov-gpu` | `gpu/` | GPU backend `Gpu` and the wgpu, Apple, Android, Windows and Wayland interop. |
| `cherenkov-cpu` | `cpu/` | CPU backends `Raster` (desktop/server: full framebuffer, multi-threaded, SIMD) and `Banded<P>` (microcontroller: banded output, panel pixel formats, flash-resident assets). |
| `cherenkov-shader` | `shader/` | The shared shader composer on naga IR, used by the engine and by filtrate. |
| `filtrate`, `filtrate-core`, `filtrate-derive` | `filtrate/` | Independent filter library: definitions, a thin reference executor, and a derive macro. It keeps its own name and does not depend on the engine crates. |

Capabilities are traits implemented by backend types, so using a missing capability is a compile error:

| Capability trait | `Gpu` | `Raster` | `Banded<P>` |
|---|---|---|---|
| `HdrOutput` | ✓ | | |
| `Backdrop` | ✓ | ✓ | |
| `GpuContent`, `ShaderPaint`, `ExternalFrames` | ✓ | | |
| `Planes` (system-compositor promotion) | ✓ | | |
| `Runs<F>` for a filter `F` | every filter | filters with a CPU kernel | filters with a CPU kernel |

Recorded content (`Picture`, `Content`) is backend-independent. Components such as math, chart, svg and map never name a backend. Only the host names one.

On the native Android backend, the host picks `Gpu` or `Raster` once at process start by querying Vulkan capabilities: the floor is `VK_EXT_rasterization_order_attachment_access` or `VK_KHR_dynamic_rendering_local_read`, plus f16. These two are different synchronization architectures. Ordered attachment access orders overlapping fragments implicitly; local read needs explicit by-region dependencies between overlapping work. They are not interchangeable implementations of the same design. Apple needs no selection: every iOS 26 device and every Apple silicon Mac running macOS 26 meets the floor. macOS 26 also still runs on some Intel Macs, which are outside the floor.

## Threading

- **UI thread: the single state machine.** Layers, backdrop groups, transactions, live recording and nami subscriptions live here, and all of them are `!Send`.
- **Render thread: sole owner of GPU state.** A commit sends an owned change set over a channel.
- **Parallelism over immutable data only.** Recorded `Picture`s are `Send`. Flattening, strip generation and glyph rasterization are data-parallel over owned data.
- **No locks anywhere in the engine.**

## Engine

```rust
let engine: Engine<Gpu> = Engine::new(GpuConfig {
    budget: Budget { gpu: Bytes::mib(512), cpu: Bytes::mib(96) },
    pipeline_cache: Some(cache_dir),
    ..GpuConfig::default()
}).await?; // creates the device and asynchronously precompiles the closed pipeline set

engine.trim(Pressure::Critical);       // system memory warning
let usage: MemoryUsage = engine.memory();
```

- The engine owns the device. `GpuContent` implementations reach wgpu through `cherenkov_gpu::interop::wgpu`.
- The pipeline set is closed and fully precompiled at creation, and the driver cache is persisted. Custom shaders compile when they are registered. Nothing compiles at draw time.
- `Engine` is `!Send` and lives on the UI thread. It spawns and owns the render thread.

## Resources

Resources are RAII handles: they are `Clone`, and the GPU memory is released, deferred, when the last handle drops.

```rust
let font: Font = engine.font(FontSource::mapped(path)?)?;          // memory-mapped; never copied
let photo: Image<Astc4x4> = engine.image(encoded_astc, ImageDesc::new(DisplayP3)).await?;
let hdr: Image<Rgba16F> = engine.image(decoded, ImageDesc::new(Rec2020Pq).hdr(meta)).await?;
let shader: ShaderPaintHandle = engine.shader_paint(wgsl_source)?; // compiled here, GPU only
```

- **`Image<F>`.** `F` is the storage format: `Rgba8`, `Rgba16F`, `Astc4x4`, `Etc2Rgba`, `Bc7`, and panel formats for `Banded`. Only uncompressed formats have `update(region, pixels)`. Compressed formats are uploaded as-is, and compression is an explicit step (`engine.compress::<Astc4x4>(image)`), never implicit.
- **Colour metadata.** Every image carries its colour space and optional HDR metadata. Conversion into the working space happens when the image is sampled.

## Surfaces and output

```rust
let window = engine.surface(interop::apple::LayerTarget::new(ca_layer))?;  // system-compositor parent
let embedded = engine.surface(interop::android::SurfaceControlTarget::new(parent))?;
let plain = engine.surface(interop::window::Target::new(raw_window_handle))?; // single surface
let snapshot = engine.surface(Offscreen::new(size, OffscreenFormat::LinearF16))?;
```

- **System-compositor parents.** Targets that expose one (`CALayer`, `SurfaceControl`, a DirectComposition visual, a Wayland subsurface) let the engine build **planes**. Most layers are composited inside the engine onto one plane. Eligible layers are promoted automatically to their own system layers: video frames, custom GPU content and large stable layers. A layer is not promoted when it is under a backdrop, uses a non-default blend or has a clip the system cannot express. Hardware overlay budgets also limit promotion.
- **Many small surfaces are first-class.** A native backend embeds one surface per self-drawn component, and a list may hold dozens. All surfaces share the engine's pipelines, atlases and caches. Creating and dropping one is cheap. All dirty surfaces render in one submission per frame.
- **Display properties.** Headroom and scale belong to the display, so the host sets them when they change: `surface.display(Display { headroom, scale })`.

## Frame driving

The host owns the event loop and the display link. The engine says when the next frame is needed and at what rate, because only the engine knows every running animation.

```rust
match engine.render(FrameTime::at(target_presentation_time))? {
    Next::Idle => link.pause(),
    Next::At { time, rate } => link.request(time, rate), // rate: RefreshRange, e.g. 60..=120
}
```

## Layer tree

```rust
let card: Layer = surface.layer();               // Layer: !Clone, !Send
surface.update(|tx| {
    tx[&root].push(&card);
    tx[&card]
        .transform(Affine::translate((24.0, 80.0)))
        .clip(ContinuousRect::new(bounds, 16.0))
        .opacity(&opacity_signal)                // bound: later changes need no transaction
        .content(content);
});
drop(card);                                      // removed at the next commit
```

- **Properties:** `transform`, `opacity`, `clip`, `blend`, `filter`, `backdrop`, `scroll_offset`, `content`, and child order (`push`, `insert`, `remove`). Each property accepts a constant or a nami signal. A bound signal keeps updating the layer with no further transactions. The subscription is owned by the layer and released when the layer drops.
- **Stable identity.** The layer handle is the identity. Content versions are internal: setting `content` bumps the version, and caching keys on (layer, version).
- **Content kinds:** recorded `Content`, a shared `Picture`, `ExternalFrame` (video, web views), `GpuContent` (custom GPU pipelines).

## Animation

```rust
surface.update_animated(Spring::smooth(), |tx| {
    tx[&sheet].transform(open);
    tx[&scrim].opacity(0.4).animation(Curve::ease_out(Duration::from_millis(200)));
});
```

- **Two levels.** A transaction-wide animation applies to every property it changes, and `.animation(...)` overrides it for one property.
- **From nami.** A bound signal's change carries WaterUI's `Animation` in its `Context` metadata. The engine reads it and interpolates, so WaterUI's `.animation(...)` reaches the engine with no glue.
- **One set of animation types.** `Spring { response, damping }`, `Curve` (cubic Bézier with a duration), and `Decay { velocity, deceleration }` with optional rubber-banding. WaterUI's `Animation` becomes these types, the same way colours were unified.
- **Out-of-process handoff.** On promoted layers, `transform` and `opacity` animations are handed to Core Animation (Apple) or DirectComposition (Windows) whenever the curve maps exactly: springs map to `CASpringAnimation`, and Bézier curves map to `CAMediaTimingFunction`. Everything else, and everything on Android, is engine-driven.

## Scrolling

```rust
tx[&list].scroll_offset(offset);                                  // tracking a finger
tx[&list].scroll_offset(target).animation(
    Decay::new(velocity).rubber_band(content_bounds));            // fling, engine-driven
```

Scrolled content is never re-recorded. Gesture recognition stays with the host.

## Recording

There are two recorders, and the difference between them is thread affinity:

```rust
/// Any thread. Constants only. Produces a frozen, shareable, Send Picture.
let icon: Picture = Picture::record(|c: &mut StaticRecorder| { /* … */ });

/// UI thread only (the surface is the proof). Accepts nami signals anywhere a value is accepted.
let content: Content = surface.record(|c: &mut Recorder| { /* … */ });
```

Both implement one drawing trait. A generic associated type decides what a parameter accepts:

```rust
pub trait Draw {
    /// `StaticRecorder`: `T`. `Recorder`: any `Signal<Output = T>` (constants are signals).
    type Value<T: 'static>;

    fn fill<S: Shape>(&mut self, shape: impl Into<Self::Value<S>>, paint: impl Into<Self::Value<Paint>>);
    fn stroke<S: Shape>(&mut self, shape: impl Into<Self::Value<S>>, style: impl Into<Self::Value<Stroke>>,
                        paint: impl Into<Self::Value<Paint>>);
    fn shadow<S: Shape>(&mut self, shape: impl Into<Self::Value<S>>, shadow: impl Into<Self::Value<Shadow>>);
    fn text(&mut self, layout: &TextLayout, origin: Point);
    fn glyphs(&mut self, run: &GlyphRun<'_>);
    fn image<F: Format>(&mut self, image: &Image<F>, dst: impl Into<Self::Value<Rect>>, sampling: Sampling);
    fn picture(&mut self, picture: &Picture, transform: impl Into<Self::Value<Affine>>);

    fn clip<S: Shape>(&mut self, shape: impl Into<Self::Value<S>>, body: impl FnOnce(&mut Self));
    fn transform(&mut self, t: impl Into<Self::Value<Affine>>, body: impl FnOnce(&mut Self));
    fn group(&mut self, group: Group, body: impl FnOnce(&mut Self)); // opacity, blend, filter
}
```

- **Numeric changes.** A signal passed to `Recorder` becomes an engine-side value slot. When it changes, only the commands that reference it are regenerated, and damage is exactly those commands. Structural changes re-record.
- **Shape signals.** A signal of a shape (`radius.map(|r| Circle::new(c, r))`) is how geometry becomes reactive. nami's `map` and `zip` compose it, and there is no per-field generic.
- **Paired state is closure scopes only.** There is no ambient mutable state and no push/pop.
- **nami `kurbo` feature.** nami gains a `kurbo` feature that implements constant `Signal` for kurbo types, so `impl Signal<Output = Affine>` accepts a plain `Affine`. The orphan rule prevents Cherenkov from doing this itself. Cherenkov's own types implement constant `Signal` in Cherenkov.

### Canvas (WaterUI)

Canvas and `Content` are fully unified. Canvas is a WaterUI view holding a recording closure that receives `&mut Recorder`; there is no second drawing API and no adapter.

- `DrawingState`, `save`/`restore`, the `set_*` style setters and `push_*`/`pop_layer` are removed.
- Styles are explicit parameters (`fill(shape, paint)`, `stroke(shape, style, paint)`). Transform, clip, opacity and blend are closure scopes. Reactive numbers are nami signals. Text goes through parley layouts.
- Structural changes re-run the closure, and numeric changes flow through bound signals.
- chart and mermaid do not use Canvas; they record directly.

## Shapes

```rust
pub trait Shape: 'static {
    /// What the engine may draw with a fast path. The default for kurbo shapes uses
    /// kurbo's own `as_rect` / `as_rounded_rect` / `as_circle` / `as_line`, and falls
    /// back to the path elements otherwise.
    fn semantic(&self) -> Semantic<'_>;
}

pub enum Semantic<'a> {
    Rect(Rect), RoundedRect(RoundedRect), Continuous(ContinuousRect), Circle(Circle),
    Ellipse(Ellipse), Line(Line), Path(PathRef<'a>),
}

impl<T: kurbo::Shape + 'static> Shape for T { /* … */ }
impl Shape for ContinuousRect { /* Semantic::Continuous */ }
// kurbo's Shape has no ellipse downcast, so the blanket impl recognises
// kurbo::Ellipse by type and gives it Semantic::Ellipse; no separate oval type.
```

- **Custom shapes are open.** `waterui-shape` merges here, and Lyon is removed.
- **The semantic vocabulary is closed.** It is the set of fast paths. Besides the shapes above, it includes `Border` (a stroked rounded or continuous rectangle of a given width) and `InnerShadow`. These are the most common UI elements after the rounded rectangle, and otherwise they would fall to the general path route.
- **Proposal: native path type.** If profiling shows `BezPath`'s f64 storage is a bottleneck for large paths, add an engine-native f32 path type that also implements `Shape`. `BezPath` stays accepted.

## Paint and stroke

```rust
pub enum Paint {
    Solid(WorkingColor),             // colours convert to the working space when recorded
    Linear(LinearGradient), Radial(RadialGradient), Sweep(SweepGradient),
    Mesh(MeshGradient),              // Hydrolysis panics on this today
    Image(ImagePaint),               // pattern: image, transform, extend modes, sampling
    Shader(ShaderPaint),             // user WGSL fragment shader; GPU only (ShaderPaint capability)
}
```

- `impl<CS: ColorSpace> From<Color<CS>> for Paint`, and likewise for each gradient type.
- **Gradients.** Stops are colours in any space. The interpolation space is a gradient property; the default is the working space, and an sRGB-encoded option exists for web compatibility.
- **Stroke** is `kurbo::Stroke`: width, joins, caps, miter limit, dashes.
- **Shader paints** replace `ShaderSurface`, `FlowingGradient` and `ViewEffect`. They inherit the shape, clip, antialiasing and on-chip blending, and they receive time and any signal-bound uniforms.
- **Shader paints follow a portable contract.** Inputs are explicit: coordinates, time, uniforms, declared resources and sampling footprints. Gradients are passed explicitly, and nothing relies on implicit fragment derivatives or on fragment-stage built-ins. This lets the same paint run in a fragment shader, a tile interpreter or a compute shader, so the paint contract never pre-selects the raster architecture.

## Colour

```rust
Color::<DisplayP3>::new([1.0, 0.2, 0.1, 1.0])
Color::<LinearSrgb>::new([4.0, 4.0, 4.0, 1.0])   // four times SDR white
DynColor::from_css(parsed)                         // colour space known only at run time
```

- **Typed colour spaces.** `Color<CS>` converts to the working space (linear Display P3) through a matrix that is constant-folded when monomorphised. HDR is extended values above 1.0, relative to SDR white.
- **The display supplies headroom.** Effects may read it. Output tone-maps to the display's headroom.
- **Blending space** is linear by default. Groups can opt into sRGB-encoded blending for web compatibility.
- **WaterUI unification.** WaterUI's `ResolvedColor` becomes Cherenkov's colour type, with headroom folded into extended values.

## Text

Shaping stays outside the engine: parley, which covers complex scripts (Arabic, Indic, Thai, Hebrew and so on). The engine is responsible for everything between shaped glyphs and pixels, for every script:

```rust
c.text(&layout, origin);          // parley adapter: TextLayout wraps parley::Layout<Paint>
c.glyphs(&GlyphRun {
    font, size: 17.0, coords: variation_coords,
    glyphs,                       // id, position, and an optional per-glyph transform (vertical CJK)
    style: GlyphStyle::Fill,
}, paint);                        // paint is a separate parameter, so it can be bound to a signal
```

- **Large scripts.** CJK text can touch thousands of distinct glyphs per screen.
  - The glyph atlas is budgeted and evicts least-recently-used pages.
  - Subpixel positions are quantized.
  - Glyph rasterization runs data-parallel with SIMD.
  - Glyphs above a size threshold are drawn as paths instead of atlas entries.
- **Colour glyphs.** COLRv0/v1 and bitmap strikes (sbix, CBDT/CBLC) are drawn natively. Emoji ZWJ sequences arrive as single glyphs from shaping.
- **Vertical text.** Glyph runs carry per-glyph transforms, so shaping can emit rotated or upright vertical glyphs.
- **Coverage correction.** Blending coverage in linear space makes text, especially thin CJK strokes, look lighter than users expect. Text coverage therefore gets a perceptual contrast and gamma correction, applied only to glyph coverage and never to geometry.
- **Font data.** Fonts are memory-mapped and never copied, which matters for Noto CJK-sized fallback chains. On `Banded`, glyph subsets are pre-rasterized into flash at build time.
- **Variable fonts** take normalized coordinates on the run.
- **Glyph realization is an experimental axis** (coverage atlas, direct curve evaluation, the path route, or distance fields for validated sizes), decided by the device farm. The CPU exact-area glyph rasterizer is both the correctness reference and the CPU backends' route. COLRv1 glyphs are a paint graph: every realization handles their transforms, gradients and compositing, and cached colour glyphs key on palette and foreground.
- **Test coverage.** The correctness corpus (#3) includes Latin, CJK (horizontal and vertical), Arabic, Hebrew, Devanagari, Thai, emoji ZWJ sequences and COLRv1 glyphs.

## Effects and filters

### Repository and crates

filtrate moves into this repository, with its history, as an independent crate family: `filtrate`, `filtrate-core` and `filtrate-derive` in `filtrate/`. They keep their names, as an independently published brand, and they do not depend on the engine crates.

A shared **shader composer** crate, `cherenkov-shader` in `shader/`, is built on naga IR and used by both filtrate and Cherenkov. Every shader fragment is a naga function: primitive shading, paints, blending, backdrop sampling, YUV conversion and filter stages. Composition, inlining, specialization (constant parameters, f16/f32 and subgroup variants) and dead-code elimination all happen on the IR, so no shader text is built by string splicing. This mirrors Skia Graphite's `ShaderCodeDictionary`, where each snippet is "the ABI of an SkSL module function and its uniform data", but with one composer for the whole repository.

### Contract

1. **Stages are functions.** A colour stage is `fn(color, params) -> color`. A spatial stage is `fn(sampler, uv, params) -> color`.
2. **Filter kinds as types.** `ColorFilter` has a `LINEAR` property. It is necessary, but not sufficient, for pushing the filter down into each primitive's shading. `LINEAR` means a linear map on premultiplied RGBA with an identity alpha row and zero offset. Only such a map commutes with src-over, \(M(a + (1-\alpha_a)b) = Ma + (1-\alpha_a)Mb\). Saturation, hue rotation, grayscale, sepia and multiplicative brightness qualify. Anything that touches alpha, or that adds an offset (an additive brightness, for example), is not `LINEAR`: pushed down, the offset would be applied once per primitive instead of once per group. Push-down is legal only when the operation is `LINEAR` **and** every composition inside the group is premultiplied source-over, with no intermediate clamping, un-premultiplying or rounding boundary. So eligibility is a property of the operation *and* its composition context, and the engine checks both. The correctness corpus includes overlapping translucent primitives inside filtered groups to catch violations. `SpatialFilter` samples neighbours. `Chain<A, B>` is a `ColorFilter` exactly when both halves are.
3. **Footprint.** `SpatialFilter::footprint(&self) -> f32` is the maximum sample radius for the current parameters; while a parameter animates, it is the maximum over its animation track. It sizes intermediates, damage expansion, backdrop regions and band or tile aprons.
4. **Working-space constants and operating space.** Luma and saturation coefficients come from the working space (linear P3) as engine-provided constants. A filter also declares the colour space it operates in. For web compatibility, CSS filter functions operate in sRGB, so the engine converts around them. Extended values (negative components, values above 1) stay extended.
5. **Shape input.** A filter can declare that it needs the clip shape's signed distance field or its mask.
6. **CPU kernels.** A filter may provide a SIMD CPU kernel, which makes it `Runs<Raster>` and `Runs<Banded<_>>`. The oracle cross-checks every kernel against its shader.

### Execution

The composer produces a normalized form: segment boundaries plus the possible materialization points and their cost parameters. The executor chooses among them.

- **filtrate's thin wgpu executor** runs composed programs pass by pass. It serves UI-independent consumers, such as waterkit's export pipeline, and it is the reference for the oracle.
- **Cherenkov** chooses by where the filter sits:
  - **Colour filters.** A `LINEAR` filter is pushed down into each primitive's fragment shading before blending, so it needs no group at all. Any other colour filter is applied when its group resolves in tile memory, using on-chip programmable blending, so nothing goes back to memory. Nesting spills to a texture only when it exceeds on-chip capacity.
  - **Spatial filters.** Layer content is rendered to a transient intermediate covering the content bounds plus the footprint, and processed in fragment passes, which keeps lossless framebuffer compression. Large blurs downsample first. The final pass applies the colour suffix and blends into the parent. Keeping small-footprint spatial filters on chip with Apple tile shaders is a farm axis, not a guarantee.
  - **Backdrops.** A `BackdropGroup` defines one explicit capture point in painter order. Only members that sample the backdrop at that point can share it; an effect at a different paint-order position has a different backdrop. The group resolves its parent once, over the members' bounds plus the footprint, and the capture may be stored sparsely, so two small distant members do not force a capture and blur of the empty space between them. The spatial chain runs once, at reduced resolution where the blur contract permits. Each member samples the shared result with its own effect in its composite shader.
  - **Parameters** are value slots, which may be bound to nami signals. Changing one updates a uniform only; nothing is recompiled or re-recorded.
  - **Compilation** happens when a chain is first registered, and the driver pipeline cache is persisted.
  - **Apple tile passes.** naga cannot express imageblocks, tile render pipelines or raster order groups. On the GPU backend's Apple route, tile and imageblock passes are therefore thin native MSL scaffolding around function bodies emitted by the shared naga composer. Every shared function still comes from the composer; only the tile-pass declarations are native.
  - **CPU backends** apply pushed-down colour functions per span and run spatial kernels with footprint-wide aprons between bands.

```rust
tx[&card].filter(Saturation(1.2).then(Brightness(0.9)));  // ColorFilter: fused, no extra pass
c.group(Group::new().filter(Blur::new(4.0)), |c| { /* … */ });
```

Multi-input operations (blend with an image, displacement, LUT) take `Image<F>` handles as auxiliary inputs.

## Backdrop

Requires the `Backdrop` capability.

```rust
let glass: BackdropGroup = surface.backdrop_group(Blur::new(24.0).then(Saturation(1.8))); // SpatialFilter
tx[&toolbar].backdrop(glass.sample(refraction.clone()));
tx[&tab_bar].backdrop(glass.sample(refraction));
```

- **One capture per group.** A group owns one capture and one spatial filter chain. Each member applies its own per-element effect on the shared result: a colour filter, a shader, or a filter that takes shape input. So there is one capture and one blur per group, whatever the number of members.
- **Handle rules.** `BackdropGroup` is an RAII, `!Send` handle.

## External content

```rust
// Video and web views: zero-copy frames, promoted to hardware overlays when eligible.
tx[&player].content(ExternalFrame::apple(io_surface).color(ColorInfo::rec2020_pq()).hdr(meta));
tx[&player].content(ExternalFrame::android(hardware_buffer).dataspace(dataspace));

// Custom GPU pipelines (particles): GPU backend only.
impl GpuContent for Particles {
    async fn setup(&mut self, gpu: &interop::wgpu::Context<'_>) { /* … */ }
    fn render(&mut self, frame: &mut interop::wgpu::Frame<'_>) { /* … */ }
}
tx[&sparks].content(GpuContentHandle::new(Particles::new()));
```

- **The engine does YUV conversion and tone mapping** for external frames when it composites them itself.
- **Custom GPU content composites like any other layer:** it can be clipped, filtered, animated and used as a backdrop source.
- **The map records `Content`, not `GpuContent`.** Each tile is a frozen `Picture` and the camera is the layer transform, so pinch-zoom and fling run in the engine. Tessellation is refreshed at the new zoom level once the gesture settles.

## Damage (invisible)

- **Damage is computed from the change set,** at three levels:
  - layer properties;
  - content replacement;
  - command level: the old and new display lists are compared, and bound-value slots contribute their commands' bounds.
- **What damage drives:**
  - partial rasterization on every backend;
  - partial present on the GPU (`VK_KHR_incremental_present`, EGL swap-with-damage);
  - partial panel transfer on `Banded` over SPI/QSPI.

## Test hooks and serialization

- **Float readback.** `Offscreen` surfaces read back in linear extended f16: `surface.readback().await`.
- **`Config::invisible_optimizations(Enabled | Disabled)`.** The oracle renders every scene twice and requires bit-identical output. Plane promotion has its own perceptual check.
- **Frame statistics.** `engine.stats()` reports GPU time, pass count, plane count and damage area for the last frame.
- **Captures.** With the `capture` feature, `surface.capture()` produces a serializable `Capture` (serde). It holds the layer tree, the contents with signal values resolved at capture time, and the resources, stored by content hash. The cross-engine suite (#3) records real Hydrolysis frames this way and replays them through every engine's adapter.

## Errors

Errors are `thiserror` enums per operation family: `EngineError`, `SurfaceError`, `ResourceError`, `RenderError` (including device loss). Invariant violations panic with a message. Nothing silently degrades.
