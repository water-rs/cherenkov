# filtrate-core

The stable abstraction layer behind [`filtrate`](https://crates.io/crates/filtrate):
the pure-data `Filter` trait, `Chain`, and the parameter composition primitives.

It describes filters without knowing how they are executed — no GPU, no `wgpu`,
no dependencies at all — so a filter definition can be shared by a renderer, a
test, or a tool that only inspects it.

Most users want `filtrate` instead, which pairs these definitions with a GPU
runtime.
