# Exact CPU coverage and sparse compositing

## Contract

Coverage is the integral of the fill predicate over a pixel. With clips it is
`area(pixel ∩ shape ∩ clip₁ ∩ … ∩ clipₙ)`. Each operand retains its fill rule.
Neither multiplying coverage masks nor taking their maximum computes that area.

“Exact” here means analytic integration of the flattened, device-space line
segments, subject to floating-point rounding. The existing curve/stroke tolerance
(0.02 device pixels), glyph flattening tolerance, and f32 framebuffer remain.
This does not claim exact integration of the original Bézier curves. The oracle
uses finer flattening, so the corpus FLIP comparison remains necessary.

## Representation and pipeline

1. Lowering keeps the complete clip geometry, including horizontal edges. It
   builds a lossless binary key from source path elements, transform bits, fill
   rule, stroke parameters, clip operands, and surface dimensions. Paint is not
   part of coverage identity.
2. On a cache miss, kurbo expands strokes, then curves are flattened. The coverage
   compiler bins directed segments into the rows they touch, retaining their
   original endpoint coordinates. Calculations after this conversion use f64.
3. Within a row, segments are sorted by their horizontal projection and partitioned
   into connected groups of overlapping projections. Each operand has an integer
   winding carried from one group to the next. A vertical gap between groups has
   no boundary crossing, so its winding is constant throughout the open row.
4. A group's events are its segment endpoints and its interior segment crossings.
   A broad phase compares only overlapping horizontal projections. Between
   consecutive event heights, active crossings have a fixed order. The compiler
   walks them at the slab midpoint, updates the appropriate operand's winding,
   and emits a boundary only when the joint inside predicate changes.
5. Boundary integrals become sparse column deltas. Sorting and prefix summing
   these deltas produces constant-coverage runs; uncovered columns consume no
   storage. Long opaque interiors occupy one run rather than one value per pixel.
   Coverage is clamped for numerical roundoff and rounded to f32 only on emission.
6. Lowering emits prepared draws. A single frame pass bins draws into intersecting
   16-row bands, preserving painter order. Rayon shades independent framebuffer
   slices. Solid opaque runs are direct stores; other runs evaluate paint and
   composite. Glyph misses are resolved before this pass.

This replaces `Accum`, `deposit_exact`, the per-band edge walk, and clip coverage
products. There is one geometry compiler for primitive fills, general paths,
stroke outlines, and clipped glyphs. A convex primitive does not need a separate
coverage implementation.

Horizontal edges have zero area contribution but **must be retained**. A
fractional horizontal edge can connect two distant vertical edges within a row.
Discarding it would incorrectly assert that winding in the gap is independent
of height, particularly when another operand crosses that gap. Horizontal edges
participate in grouping and endpoint events, never in winding updates.

## Why the result is exact

Inside an event slab, every active crossing is linear in y, and every pairwise
crossing lies on a slab boundary. Consequently the ordering and winding predicate
are constant between adjacent crossings. Updating a count of operands that are
outside identifies transitions of the intersection predicate in constant time.

An entering transition contributes the area to the right of its edge; a leaving
transition subtracts it. Summing these contributions integrates the indicator
function of the intersection, whose values are zero or one, rather than its
possibly larger signed winding. The compiler integrates the linear ramp across
crossed pixel columns analytically. All other columns are constant runs.

Coincident boundaries can produce intermediate enter/leave transitions at the
same position; their area contributions cancel. Opposite winding, repeated
contours, nonzero winding magnitudes greater than one, and even-odd parity are
resolved before integration. The event list uses f64 ordering without an epsilon
step that discards a sliver or a retry loop that can stall at a crossing.

A raw signed-area accumulator followed by `abs`, clamping, or an even-odd fold
cannot replace this predicate resolution. Subpixel regions of winding +1 and -1
can cancel their signed integral while both regions are filled. Similarly,
multiple regions with winding +1 can overlap without covering the whole pixel.
A scalar area and an integer tile winding do not encode that distribution.

## Strokes and classification

Kurbo remains the stroke expander, preserving its joins, caps, miter limit,
dashes, and nonzero outline semantics. Coverage does not union independently
rasterized stroke pieces: that would lose subpixel intersections at joins.

The compiler classifies overlap spatially per row instead of attaching an
unsafe “simple path” label to a whole path. Separated edge groups are resolved
independently; only a group containing overlapping projections pays for local
crossing events. Both the classification result and integrated coverage are
retained in the prepared coverage cache. The source-path key is checked before
stroking and flattening, so unchanged strokes do not repeat expansion either.

This choice also handles self-intersecting strokes without a separate slow
implementation. In particular, chart's hundreds of crossings in a row no longer
force a sort of the complete row at every unrelated endpoint. A dense connected
group can still be expensive on a cache miss; the worst case is stated below.

## Clips, glyphs, and shadows

Clip operands are intersected jointly with each draw, not rasterized to separate
alpha masks. Integer and fractional rectangles use the same semantics. Nested
clips, coincident boundaries, and even-odd operands require no special product
operation.

Glyph masks retain the original flattened outline relative to their integer
origin. Unclipped glyph instances use their cached coverage. Clipped instances
translate that outline and enter the same intersection compiler. The exact
subpixel offset and f32 font size remain part of glyph identity.

Unclipped rounded-box shadows retain the existing analytic quadrature, now
prepared once per geometry key. Clipped shadows follow the oracle's operation
order: intersect the caster with every clip, then convolve that coverage with
pixel-integrated Gaussian taps. Clipping an already blurred image is incorrect
because it cuts off the blur tail. The convolution uses the existing workspace
`libm` implementation for f64 erf, a six-sigma kernel, and the oracle's surface
edge clamping. Shadow rows use dense samples within nonempty runs because their
coverage usually varies at every pixel. Their colour is applied at composition.

## Cache identity, ownership, and memory

The renderer owns the cache; no globals, worker-local caches, or locks are added.
Keys retain their full binary contents, and `HashMap` equality checks them after
hashing. Hash collisions cannot alias geometry. Fractional transforms are keyed
exactly and rerasterized, never reconstructed by resampling cached coverage.
Changing paint reuses coverage but evaluates the new paint. Changing the surface
size, clips, fill rule, geometry, stroke parameters, or transform changes the key.

Half of `Budget::cpu` is reserved for glyph masks and half for prepared coverage.
Coverage accounting includes key capacity, row/run capacities, and varying sample
capacities; hash-table bookkeeping is additional allocator overhead. Values
larger than the retention budget still render but are not retained. Exceeding the
budget clears retained entries; active frame references remain valid. A zero
budget disables retention. Cold, warm, and evicted execution use the same compiler
and rounding points. Memory reporting includes coverage and retained band scratch;
critical pressure clears caches and band storage. Glyph accounting includes the
retained outlines and avoids double-counting replacement entries.

Band item lists and isolation buffers belong to the surface and survive frames.
Push/pop markers reach every band because blend modes such as Clear and `DestIn`
can modify destination pixels even where source coverage is empty. Buffer reuse
zeros each isolation layer before reuse. Framebuffer clearing happens in the band
pass. There are no per-draw accumulators or per-frame clip-mask parallel launches.
The existing persistent renderer-owned Rayon pool remains responsible for worker
lifetime; there is no busy wait or arbitrary sleep.

## Complexity

Let `E` be input segments, `R` the total segment/row visits, `kᵣ` the number of
segments in row r, `m` the number in one connected group, `X` its actual crossings,
`A` the emitted boundary/pixel-column visits, and `P` shaded pixels.

- Row binning is `O(E + R)` time and `O(R)` transient storage.
- Group construction is `O(Σ kᵣ log kᵣ)` time.
- Broad-phase intersection discovery is `O(m²)` in a dense group, less when
  projections separate. There are `O(m + X)` event slabs, each costing at most
  `O(m log m)` for ordering and a winding walk. Thus a deliberately adversarial
  dense group can cost `O(m² + (m + X)m log m)`. This is not a claim of a
  worst-case-linear Boolean rasterizer.
- Boundary integration costs `O(A)` and the row's delta sort `O(Aᵣ log Aᵣ)`.
  Output storage is proportional to retained runs and antialiased samples, not
  the full path bounding-box area.
- A warm source lookup costs key construction/hash/equality, proportional to
  source geometry and clips. Stroking, flattening, event construction and area
  integration are skipped. Composition costs `O(P)` plus item/band references.
- Clipped-shadow cold preparation adds separable convolution work proportional
  to the affected rows/columns times the kernel radius. Warm shadows reuse their
  prepared coverage.

## Validation and measurement expectations

Tests compare the replacement compiler with the oracle for the star, nested
contours, figure-eight, a true interior-crossing bowtie, randomized polygons,
sharp stroke joins/caps, repeated and reversed contours, fractional horizontal
edges, off-surface edges, and nested non-rectangular intersections. Additional
cases distinguish intersection from coverage products, retain each operand's
fill rule, check clipped-shadow convolution, and compare cached/uncached output
and one/four-worker composition bit for bit.

No build, formatter, test, or benchmark was run on the coordinating Mac. Stable
Rust 1.98 compilation, workspace lints, the full corpus, and equal-thread p50/p99
measurements belong to the Linux verification run.

| Scene | Expected effect, pending measurement |
| --- | --- |
| chart | Largest reduction from skipping repeated stroke expansion and geometric sweeps; only covered spans are shaded. |
| map | Less per-band item scanning and bounding-box work, plus prepared paths/strokes; key construction and pixel blending may become dominant. |
| ui-list | Reuses path and shadow coverage; band scratch reuse removes recurring isolation allocations. |
| effects | Warm frames eliminate analytic shadow evaluation; cost shifts to coverage reads and composition. Cold clipped shadows pay convolution. |
| text-page | Band binning and cheaper source-over help; glyph hit frames already skipped coverage, so expect a smaller improvement. |

The p50/p99 target is an acceptance criterion, not a measured result of this
change. Reduced computation and allocation should reduce tails; OS scheduling
jitter, cache eviction, dense cold geometry, and memory bandwidth still need the
VM measurements. Keep cold/changed geometry measurements separate from steady
repeated frames, and compare the full 83-scene FLIP results against the handoff.
