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
- Eviction does not touch `generation`. Surviving emissions stay live;
  only the victim's keys lose their `live_epoch`.

## Correctness: epochs and per-leaf refs

A baked cell index is valid only while its atlas entry survives, so
every emission records `(key, epoch)` refs for the admissions it drew
from (`Emission.refs`, `clock` for a fast all-live check). Evicting a
key and later re-admitting it under a different origin bumps `epoch`,
so a replayed emission that referenced the old entry fails
`live_epoch == epoch` and re-lowers instead of sampling stale texels.
Hits mark the shelves a frame still uses through `Lowering.touches`,
applied once by `begin_commit` — deferred because lowering runs in
parallel against an immutable `&Atlas`. Mask cells contribute touches
but no refs: their atlas UV is rewritten per frame by `apply_clip`.

## Residual behavior

`Recycle` remains as the plan's exhaustion verdict — the commit leaves
`AtlasExhausted` to the surface rather than falling back to a clear,
because a batch that cannot fit even after evicting everything cold is
a real failure the caller must see, not a silent re-lower. `Grow`
still doubles the atlas once per renderer (generation bumps there, as
on dev). Under a continuous pan the steady state is self-cleaning: the
frame's misses are the coldest shelves, so the next commit reclaims
exactly the band the previous frame's churn left behind.
