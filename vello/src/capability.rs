// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Capability markers on the backend type.
//!
//! Engine entry points that need a capability live on impls bounded by these
//! traits: a backend without the trait simply has no such method, so
//! requesting an unsupported feature is a compile error rather than a
//! run-time fallback. All capabilities are declared here so consumers can
//! name them in bounds; [`Vello`](crate::Vello) implements only
//! [`ShaderPaint`], [`GpuContent`] and [`Runs`].

use crate::Backend;

/// The backend draws user WGSL shader paints.
pub trait ShaderPaint: Backend {}

/// The backend composites user GPU-rendered content as layer content.
pub trait GpuContent: Backend {}

/// The backend can run the filtrate filter `F`.
pub trait Runs<F: filtrate::Filter>: Backend {}

/// The backend produces HDR output.
pub trait HdrOutput: Backend {}

/// The backend samples the backdrop behind a layer.
pub trait Backdrop: Backend {}

/// The backend presents on multiple hardware planes.
pub trait Planes: Backend {}

/// The backend consumes externally produced frames (video, camera).
pub trait ExternalFrames: Backend {}

impl ShaderPaint for crate::Vello {}

impl GpuContent for crate::Vello {}

impl<F: filtrate::Filter> Runs<F> for crate::Vello {}
