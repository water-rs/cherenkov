// The GPU runtime's async setup nests deeply enough that clippy's `Send`
// analysis overflows the default limit and abandons the check — which then
// leaves the `future_not_send` expectations below it unfulfilled. Raise it so
// the lint actually runs.
#![recursion_limit = "256"]
#![cfg_attr(
    test,
    allow(
        clippy::float_cmp,
        reason = "tests assert exact filter parameter values"
    )
)]
//! GPU filter library built on top of `filtrate-core` abstractions.
//!
//! `filtrate` hosts the built-in filter implementations and their WGSL
//! shaders, and (in upcoming phases) the GPU runtime that compiles a
//! [`Filter`] graph into a wgpu pipeline. It is designed to be usable
//! outside of `WaterUI` for any wgpu-based image, video, or render-target
//! workflow.
//!
//! # Layout
//!
//! - [`filters`]: built-in filter structs (`Brightness`, `Blur`, ...).
//!   Each filter implements [`Filter`] from `filtrate-core` and references
//!   one or more WGSL shader files under `src/shaders/`.
//! - `shaders/` (not a Rust module): WGSL fragments and full-shader files
//!   compiled into the binary via `include_str!` from individual filter
//!   modules.
//!
//! # Example
//!
//! ```rust
//! use filtrate::filters::{Blur, Brightness};
//! use filtrate::{Filter, FilterExt};
//!
//! let chain = Blur(5.0_f32).then(Brightness(0.1_f32));
//! # // A chain's params nest, one array per link, in application order.
//! # assert_eq!(chain.params(), ([5.0, 5.0], [0.1]));
//! ```
//!
//! # WebGL support
//!
//! The optional `webgl` feature enables wgpu's WebGL2 backend for wasm
//! targets (`wasm32-unknown-unknown`). WebGL2 has no compute shaders or
//! storage textures, so spatial filter stages compile to a fragment
//! translation of the same shader body there, selected automatically from
//! the device limits — see [`runtime::SpatialExecution`]. The feature is a
//! compile error on non-wasm targets.

// The `webgl` feature only makes sense where a WebGL2 context exists. On any
// other target it is inert (wgpu gates its WebGL backend to wasm), so turning
// it on natively is a build-graph mistake — fail fast instead of silently
// shipping an unused dependency set.
#[cfg(all(feature = "webgl", not(target_family = "wasm")))]
compile_error!(
    "filtrate's `webgl` feature is only supported on wasm targets \
     (wasm32-unknown-unknown); remove it from this build"
);

mod compiled_shaders;
pub mod effect;
pub mod filters;
pub mod multi_input;
pub mod runtime;

pub use effect::{
    Effect, EffectContext, EffectFrameClock, EffectFrameTiming, EffectInput, EffectOutput,
    EffectRedrawCallback, EffectRenderError, EffectRenderResult, EffectSetupError,
    EffectSetupResult,
};
pub use filtrate_core::{
    AnimatedCallback, AnimatedTarget, AnimationTrack, Chain, Filter, FilterExt, FilterParam,
    Interpolator, MAX_FILTER_PARAM_VEC4S, MAX_FILTER_PARAMS, ParamArray, SignalVisitor,
    StageCollector, WatchGuard,
};
pub use runtime::{FilterAdapter, HdrPolicy};
pub use shaderloom::WgslModuleCache;

/// Procedural derive that generates a [`Filter`] implementation for a tuple
/// struct. See `filtrate-derive` for the supported `#[filter(...)]` shapes.
pub use filtrate_derive::Filter;
