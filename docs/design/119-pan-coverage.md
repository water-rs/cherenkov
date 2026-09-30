# #119 pan coverage: bounded atlas eviction

Panning a path-heavy layer used to re-rasterize every path per frame.
Two mechanisms produced that cost. `snap_animating` quantized only the
transforms the engine itself drives, so a host-side `Motion::Transform`
pan kept arbitrary subpixel translations and `path::placement` hashed
each distinct fraction as a different key — the legitimate part of the
rasterization. The illegitimate part: the glyph atlas's only answer to
an overflowing pending batch was `AtlasPlan::Recycle` — a wholesale
`atlas.clear()` that also bumped `generation`. Every retained `Emission`
carries that generation, so the clear failed every emission's liveness
check and forced a full re-lower of the frame, not just a re-rasterize
of the batch that overflowed. On `scenes/perf/map-pan` the atlas cleared
about every third moving frame and re-rasterized ~85% of paths.

## Design

Replace wholesale recycling with bounded in-place eviction at shelf
granularity (plan-of-record candidate C; A and B landed via #186).

- `Layout` already described the atlas as shelf bands; dead bands are
  now a first-class `vacant` set. `Atlas::plan` dry-runs the pending
  batch on a cloned layout — strict allocation first, then simulated
  eviction — and returns `Fits`, `FitsEviction`, `Grow(size)` or
  `Recycle` exactly matching what the commit would do.
- A commit that needs room calls `enable_evicting`, then `alloc` falls
  through live-fit → vacant best-fit → virgin top → `evict_one`. The
  victim is the coldest shelf by segmented LRU: probationary
  (`hits == 0`) before protected, then least-recently-used within a
  segment. A shelf touched by a hit or written this commit
  (`last_used == tick`) can never be the victim, so eviction frees only
  what the frame itself did not ask for.
- `free_band` coalesces dead bands vertically adjacent to the freed
  one — without it eviction leaves space fragmented by height class
  and tall cells starve even while bytes are free — and a merged band
  reaching the layout frontier hands its rows back to `top` as virgin
  space. The vacant set stays pairwise non-adjacent, which bounds the
  merge scan.
- Eviction does not touch `generation`. Surviving emissions stay live.

## Correctness: shelf epochs and per-leaf refs

A baked cell index is valid only while its band still occupies the same
atlas slot, so every emission that drew atlas cells records `(slot,
epoch)` pairs — `Emission.refs` into the shared `EmissionStorage.refs`
arena — one per band it used. `Shelf.epoch` comes from `Layout.
next_epoch` and is bumped on every band admission (vacant reuse, top
carve, phantom split) and on every band death (`free_band`), so a slot
re-used by a different band always reports a different epoch. Liveness
is one packed compare: `Emission.live_stamp` mirrors
`Atlas::live_stamp()` — texture `generation` in the high half,
eviction `clock` in the low — and hits check equality first; only an
emission whose stamp is stale walks its refs
(`shelf_epoch(slot) == epoch` per ref — no hash lookups), restamping
on success.

Ref collection stays out of the measured `lower` interval. Lowering is
parallel against an immutable `&Atlas`, so a leaf records only the shelf
slots it touches into `Lowering.touches` (glyph and mask admissions,
consecutive-shelf deduped) and, per replayed path emission, one
`TOUCH_RANGE`-flagged index into `Lowering.touched_ranges` —
`PathEmit::slots` is a range into the atlas's append-only `emit_slots`
arena, so a replay pin costs one push instead of copying the whole
slot list, and stays readable even if the record is evicted between lower
and commit. Each realized `Emission` carries the `LIVE_PENDING`
stamp while its `refs` range still addresses `touches` instead of
`storage.refs`. The commit's `apply_pending` folds the touch slots —
decoding flagged indices to their arena ranges — plus the bands its
raster actually landed in (`PendingOrigin::Cells`) into
`(slot, epoch)` pairs, swaps `refs` to them, and stamps the emission.
An emission whose resolving commit was abandoned (`Grow` or
`AtlasExhausted`) keeps `LIVE_PENDING` and never verifies live, so the
next hit check re-lowers it.

Mask cells take part in the same touches → refs path, so they are
covered by the epoch check too.

## Residual behavior

`Recycle` remains as the plan's exhaustion verdict — the commit leaves
`AtlasExhausted` to the surface rather than falling back to a clear,
because a batch that cannot fit even after evicting everything cold is
a real failure the caller must see, not a silent re-lower. `Grow`
still doubles the atlas once per renderer (generation bumps there, as
on dev). Under a continuous pan the steady state is self-cleaning: the
frame's misses are the coldest shelves, so the next commit reclaims
exactly the band the previous frame's churn left behind.
