//! Cherenkov is a GPU 2D rendering engine for modern hardware.
//!
//! The design of the public API is `docs/api.md`. This crate is the front end:
//! the vocabulary of what to draw (colour, shapes, paint, styles) and the
//! recording of it into display lists.
//!
//! - [`Picture::record`] records constants on any thread into an immutable,
//!   shareable [`Picture`].
//! - [`Content::record`] records on the UI thread and accepts nami signals
//!   anywhere a value is accepted. A signal's later changes become
//!   [`SlotUpdate`]s that regenerate only the commands referencing it.
//! - [`ContentChange`] is the owned, `Send` change set a commit hands to the
//!   render thread.

mod color;
mod display_list;
mod glyph;
mod paint;
mod record;
mod shape;
mod style;

pub use kurbo;

pub use crate::color::{
    Color, ColorSpace, DisplayP3, DynColor, LinearDisplayP3, LinearSrgb, Rec2020, Srgb,
    WorkingColor,
};
pub use crate::display_list::{
    Command, Dirty, DisplayList, Operand, OperandKind, Picture, ScopeError, Slot, SlotUpdate,
};
pub use crate::glyph::{FontId, Glyph, GlyphRun, GlyphStyle};
pub use crate::paint::{
    ColorStop, Extend, ImageId, ImagePattern, Interpolation, LinearGradient, MeshGradient,
    MeshGradientError, Paint,
    RadialGradient, Sampling, ShaderId, ShaderPaint, SweepGradient,
};
pub use crate::record::{Content, ContentChange, Draw, Fixed, Live, Recorder, StaticRecorder};
pub use crate::shape::{
    ContinuousRect, EvenOdd, FillRule, PATH_TOLERANCE, PathRef, Semantic, Shape, ShapeData,
};
pub use crate::style::{BlendMode, BlendSpace, FilterId, Group, Shadow};
pub use kurbo::Stroke;
