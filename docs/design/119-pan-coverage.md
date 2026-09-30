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
- The commit's work scales with what changed, not what is retained.
  `fits_strict` dry-runs the batch alone first; a strictly-fitting
  frame goes straight to `begin_commit(&[])` — no emission scan, no
  pin derivation, no eviction. Only a batch that cannot place strictly
  decodes `touches`, scans live-verified emissions for their band
  lists and plans with eviction enabled. The scan's buffers
  (`commit_touches`, `commit_writes`, the dedupe sets and plan cells)
  live on the renderer and are rebuilt in place, so the steady commit
  allocates nothing.
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
epoch)` pairs — `Emission.refs`, one packed `u64` (`start << 32 | len`)
addressing the shared `EmissionStorage.refs` arena — one per band it
used. `Shelf.epoch` comes from `Layout.
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
consecutive-shelf deduped). A replayed path emission records no touch
at all: its band set is already the atlas's — `emit.slots`, a range in
the `emit_slots` arena — so the leaf stores `refs = ARENA_REFS |
(start << 32 | len)` and nothing else, with the `live` key the lookup
hit riding in `live_stamp` (the field is dual-purpose: a stamp for
pair-referenced emissions, the key for arena-referenced ones — a key
colliding with a stamp value is the accepted 2^-64 hash-collision
class of the `live` map's own keys). The emission's liveness is the
key's survival: evicting a shelf drops every key on it, so `live`
still containing the key means every band the replay drew is intact —
a stale check is one hash lookup, no pairs to write or walk. Other
emissions resolve eagerly: the atlas is immutable for the whole
lowering, so the leaf folds its touched slots into `(slot, epoch)`
pairs at realize time and `refs` is a real pair range from birth —
`live_stamp` is a real watermark too, so a not-yet-applied emission
can hit while its commit is in flight (a leaf re-checked after a
shared emission landed finds it already stamped). At apply,
`apply_pending` moves those pairs and each placed cell's band pairs
to the storage tail as one contiguous extent. `ARENA_REFS` emissions
verify identically — when their live key survives the commit they
keep the key in place of a stamp — with nothing to write.

`pending_cells` stays the one frame-scoped field: its pending-raster
indexes are only meaningful for the commit that applies them. Every
path that skips apply — a `Grow` early return, an apply abandoned on
`AtlasExhausted`, or a lowering that returned `Err` — discards that
surface's retained emissions (`discard_surface`), so stale pending
cells can never compose into a later frame.

Mask cells take part in the same touches → refs path, so they are
covered by the epoch check too.

`emit_slots` ranges of evicted emissions are recycled: `evict_one`
pushes the range onto `slot_dead`, `begin_commit` promotes it to
`slot_holes`, and admission reuses a hole before growing the arena —
same-commit readers keep addressing it correctly because a freed range
is never handed out until the next commit begins.

## Residual behavior

`Recycle` remains as the plan's exhaustion verdict — the commit leaves
`AtlasExhausted` to the surface rather than falling back to a clear,
because a batch that cannot fit even after evicting everything cold is
a real failure the caller must see, not a silent re-lower. `Grow`
still doubles the atlas once per renderer (generation bumps there, as
on dev). `scenes/perf/map-pan` pans a linear two device pixels per
frame: the integer step keeps every path's `Placement` key stable, so
a retained `PathEmit` is re-found through `Atlas::path` and replayed
instead of re-rasterized — the case the issue names. Under a pan whose
step is not integer the subpixel fraction changes each frame, keys do
not recur, and the steady state is instead self-cleaning: the frame's
misses are the coldest shelves, so the next commit reclaims exactly the
band the previous frame's churn left behind.
