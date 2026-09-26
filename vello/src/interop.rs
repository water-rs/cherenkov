// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Interop with the wrapped GPU stack.
//!
//! [`interop::wgpu`] is the only place `wgpu` (and, through it, `vello`)
//! types appear in this crate's public API: sharing a device or a surface
//! with the embedder needs the real types.

/// The `wgpu` surface of the engine.
pub mod wgpu {
    use std::time::Duration;

    pub use ::wgpu::*;

    /// An existing device and queue the engine drives instead of creating
    /// its own (the embedder's shared GPU context).
    ///
    /// The engine uses it to create its vello renderer; shader pipelines
    /// and resources it builds are freed with the engine.
    pub struct DeviceSource {
        /// The adapter the device was requested from (for [`crate::GpuInfo`]).
        pub adapter: Adapter,
        /// The device.
        pub device: Device,
        /// The queue.
        pub queue: Queue,
    }

    impl std::fmt::Debug for DeviceSource {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("DeviceSource").finish_non_exhaustive()
        }
    }

    impl DeviceSource {
        /// A source over an existing adapter, device and queue.
        #[must_use]
        pub const fn new(adapter: Adapter, device: Device, queue: Queue) -> Self {
            Self {
                adapter,
                device,
                queue,
            }
        }
    }

    /// A window target: the embedder's configured surface.
    ///
    /// The engine configures `surface` with `config` on the render thread
    /// and reconfigures it on [`crate::Surface::resize`].
    pub struct Window {
        /// The wgpu surface.
        pub surface: Surface<'static>,
        /// The configuration the engine applies.
        pub config: SurfaceConfiguration,
    }

    impl std::fmt::Debug for Window {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Window").finish_non_exhaustive()
        }
    }

    /// The GPU context handed to GPU content implementations at setup time.
    pub struct Context<'a> {
        /// The device.
        pub device: &'a Device,
        /// The queue.
        pub queue: &'a Queue,
        /// The texture format content renders into.
        pub format: TextureFormat,
    }

    /// One frame's drawing context handed to GPU content implementations.
    ///
    /// The content draws into `texture`/`view`, a `w` × `h` `Rgba8Unorm`
    /// texture whose values are premultiplied sRGB-encoded.
    pub struct Frame<'a> {
        /// The device.
        pub device: &'a Device,
        /// The queue.
        pub queue: &'a Queue,
        /// The texture being drawn into.
        pub texture: &'a Texture,
        /// Its view.
        pub view: &'a TextureView,
        /// The texture format.
        pub format: TextureFormat,
        /// Width in pixels.
        pub width: u32,
        /// Height in pixels.
        pub height: u32,
        /// Scale factor of the target.
        pub scale: f32,
        /// Time since the engine started.
        pub elapsed: Duration,
        /// Time since the previous frame, capped at 100 ms.
        pub delta: Duration,
        redraw: bool,
    }

    impl Frame<'_> {
        /// Requests another frame after this one.
        pub const fn request_redraw(&mut self) {
            self.redraw = true;
        }

        /// Whether a redraw was requested this frame.
        #[must_use]
        pub const fn redraw_requested(&self) -> bool {
            self.redraw
        }
    }
}
