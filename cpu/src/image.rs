// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Image registration: [`ImageSource`] pixels are validated on the caller
//! thread and converted to premultiplied linear Display P3 on the render
//! thread.

use std::rc::Rc;
use std::sync::mpsc::Sender;

use crate::error::ResourceError;
use crate::message::Message;

/// The gamma-encoded colour space of an [`ImageSource`]'s pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageColorSpace {
    /// sRGB primaries and transfer function.
    Srgb,
    /// Display P3 primaries with the sRGB transfer function.
    DisplayP3,
}

/// The data of an image to register with the engine: straight-alpha RGBA8,
/// row-major, `width * height * 4` bytes.
pub struct ImageSource {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The pixel data.
    pub pixels: Vec<u8>,
    /// The encoded colour space.
    pub color_space: ImageColorSpace,
}

impl ImageSource {
    /// Validates the buffer against its dimensions.
    ///
    /// # Errors
    /// [`ResourceError::Image`] for a zero dimension or a length mismatch.
    pub fn validate(&self) -> Result<(), ResourceError> {
        if self.width == 0 || self.height == 0 {
            return Err(ResourceError::Image("zero-size image".into()));
        }
        let want = usize::try_from(self.width)
            .ok()
            .and_then(|w| w.checked_mul(usize::try_from(self.height).ok()?))
            .and_then(|px| px.checked_mul(4));
        if want != Some(self.pixels.len()) {
            return Err(ResourceError::Image(format!(
                "{}x{} needs {} bytes, got {}",
                self.width,
                self.height,
                want.unwrap_or(0),
                self.pixels.len()
            )));
        }
        Ok(())
    }
}

/// The shared image state; dropping the last clone releases the pixels.
struct ImageInner {
    id: cherenkov::ImageId,
    tx: Sender<Message>,
}

impl Drop for ImageInner {
    fn drop(&mut self) {
        let _ = self.tx.send(Message::DestroyImage { id: self.id.raw() });
    }
}

/// An image registered with an engine. Cloning is cheap; the pixel store
/// is released when the last clone drops.
#[derive(Clone)]
pub struct Image {
    inner: Rc<ImageInner>,
}

impl Image {
    /// A handle for the registered image `id`.
    #[must_use]
    pub fn new(id: cherenkov::ImageId, tx: Sender<Message>) -> Self {
        Self {
            inner: Rc::new(ImageInner { id, tx }),
        }
    }

    /// The identifier paints and image draws reference.
    #[must_use]
    pub fn id(&self) -> cherenkov::ImageId {
        self.inner.id
    }
}
