//! The shader composer shared by Cherenkov and filtrate.
//!
//! Every shader fragment is a naga function. A [`Snippet`] is authored as WGSL
//! source, parsed by naga's WGSL front end, and checked against a declared
//! ABI. [`compose`] builds one naga [`Module`](naga::Module) from a chain of
//! snippets by working on the IR: it imports each snippet's function (with
//! its types, constants and helpers), generates the functions that sequence
//! them, rewrites texture samples to fold a colour prefix into a spatial
//! stage, specializes parameters that are constant, compacts the module, and
//! validates the result. No shader text is ever built by concatenation.
//!
//! # Snippet ABI
//!
//! A snippet defines exactly one function named `apply`.
//!
//! A **colour** snippet maps a colour to a colour:
//!
//! ```wgsl
//! struct Params { amount: f32 }
//! fn apply(color: vec4<f32>, params: Params) -> vec4<f32> { /* … */ }
//! ```
//!
//! A **spatial** snippet samples its input around a coordinate:
//!
//! ```wgsl
//! struct Params { radius: f32 }
//! fn apply(input: texture_2d<f32>, input_sampler: sampler, uv: vec2<f32>, params: Params) -> vec4<f32> { /* … */ }
//! ```
//!
//! After the required arguments, a snippet may declare, in any order:
//!
//! - `params: Params`: a struct whose members are `f32`, `vec2<f32>`,
//!   `vec3<f32>` or `vec4<f32>`;
//! - `space: WorkingSpace`: the engine-provided working-space constants,
//!   declared exactly as [`WORKING_SPACE_WGSL`];
//! - spatial snippets only: `shape: texture_2d<f32>` (the clip shape's signed
//!   distance field or mask), and `aux0`, `aux1`, … `: texture_2d<f32>`
//!   (auxiliary images, numbered without gaps).
//!
//! Colours are premultiplied, in the linear working space. Parameters are
//! always `f32`, whatever precision the colour values use.
//!
//! # Variants
//!
//! The `f32` source is required. A snippet may also declare an `f16` variant,
//! in which the colour values (`color`, the result, and a spatial snippet's
//! result) are `vec4<f16>`, and subgroup variants that use subgroup
//! operations. [`compose`] uses a declared variant wherever it matches the
//! requested [`ComposeOptions`] and reports the variant each stage used.
//!
//! # What the composer decides, and what it leaves to the executor
//!
//! The composer makes no execution decision. It returns a normalized
//! [`Composition`]: a sequence of [`Piece`]s whose boundaries are the possible
//! materialization points. Where a colour piece precedes a spatial piece, it
//! also offers a folded alternative, with the cost parameters an executor
//! needs to choose ([`FoldCost`]). Executors (filtrate's reference executor,
//! or Cherenkov itself) wrap the segment functions into entry points and pick
//! among the alternatives.
//!
//! Composed modules call snippet functions rather than splicing their bodies
//! together. Every platform shader compiler inlines small functions and then
//! folds the specialized constants; naga-level inlining would add nothing
//! but work.

mod abi;
mod builder;
mod chain;
mod emit;
mod errors;
mod import;
mod parse;
mod rewrite;

pub use abi::{Param, ParamType, ParamValue, Precision, SnippetKind, WORKING_SPACE_WGSL};
pub use chain::{
    ComposeOptions, Composition, FoldCost, Folded, Piece, Segment, SegmentArg, Stage,
    UniformLayout, UniformMember, compose,
};
pub use emit::{msl, spirv, validate, wgsl};
pub use errors::{ComposeError, EmitError, SnippetError};
/// The naga version modules are built with; the same one wgpu 29 links.
pub use naga;
pub use parse::{SampleCount, Snippet, SnippetSource, Variant};
