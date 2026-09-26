// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Cherenkov is a GPU 2D rendering engine for modern hardware.
//!
//! The design of the public API is `docs/api.md`. This crate is the front
//! end: the vocabulary of what to draw (colour, shapes, paint, styles), the
//! recording of it into display lists, and the shared engine — [`Engine`],
//! [`Surface`]s, the layer tree, transactions, resource handles, animation
//! and the render-thread loop, plus the [`Backend`] contract a backend
//! crate implements. There is no GPU dependency here.
//!
//! - [`Picture::record`] records constants on any thread into an immutable,
//!   shareable [`Picture`].
//! - [`Content::record`] records on the UI thread and accepts nami signals
//!   anywhere a value is accepted. A signal's later changes become
//!   [`SlotUpdate`]s that regenerate only the commands referencing it.
//! - [`Engine::render`] drains every surface's queued change set into one
//!   commit per frame, samples the animations at the frame time and renders
//!   on the render thread.

mod animation;
mod backend;
mod capability;
mod color;
mod config;
mod display_list;
mod engine;
mod error;
mod frame;
mod glyph;
mod image;
pub mod lowering;
mod message;
mod paint;
mod record;
mod resource;
mod shape;
mod style;
mod surface;
mod tree;

#[cfg(feature = "testing")]
pub mod testing;

pub use kurbo;

pub use crate::animation::{Animatable, Animation, Curve, Decay, Lanes, Spring};
pub use crate::backend::{Backend, Display, Frame, Redraw, Renderer, SurfaceFrame, SurfaceInfo};
pub use crate::capability::{
    Backdrop, Effects, ExternalFrames, Filters, GpuContent, HdrOutput, Planes, Runs,
    ShaderPaint as ShaderPaintCapability, ShaderSource, Uploads,
};
pub use crate::color::{
    Color, ColorSpace, DisplayP3, DynColor, LinearDisplayP3, LinearSrgb, Rec2020, Srgb,
    WorkingColor,
};
pub use crate::config::{Budget, Bytes, MemoryUsage, Pressure};
pub use crate::display_list::{
    Command, Dirty, DisplayList, Operand, OperandKind, Picture, ScopeError, Slot, SlotUpdate,
};
pub use crate::engine::Engine;
pub use crate::error::{EngineError, RenderError, ResourceError, SurfaceError};
pub use crate::frame::{
    FrameId, FrameStats, FrameTime, FrameTiming, Next, Offscreen, OffscreenFormat, PassTiming,
    Phases, Readback, RefreshRange,
};
pub use crate::glyph::{FontId, Glyph, GlyphRun, GlyphStyle};
pub use crate::image::{
    Astc4x4, Bc7, Etc2Rgba, Format, ImageColorSpace, ImageData, ImageFormat, ImageUpload, Rgba8,
    Rgba16F,
};
pub use crate::message::{BackdropId, ContentOp, FontData, LayerId, Prop, SurfaceId};
pub use crate::paint::{
    ColorStop, Extend, ImageId, ImagePattern, Interpolation, LinearGradient, MeshGradient,
    MeshGradientError, Paint, RadialGradient, Sampling, ShaderId, ShaderPaint, SweepGradient,
};
pub use crate::record::{Content, ContentChange, Draw, Fixed, Live, Recorder, StaticRecorder};
pub use crate::resource::{Filter, Font, FontSource, Image, Shader};
pub use crate::shape::{
    ContinuousRect, EvenOdd, FillRule, PATH_TOLERANCE, PathRef, Semantic, Shape, ShapeData,
};
pub use crate::style::{BlendMode, BlendSpace, FilterId, Group, Shadow};
pub use crate::surface::{
    ExternalFrameHandle, GpuContentHandle, Layer, LayerContent, LayerEdit, Surface, Transaction,
};
pub use crate::tree::{LayerNode, SurfaceTree};
pub use kurbo::Stroke;
