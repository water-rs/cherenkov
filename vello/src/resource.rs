// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Resource registration: fonts and images are validated on the caller
//! thread, then handed to the render thread by their handles. Dropping a
//! handle unregisters the resource.

use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::Sender;

use skrifa::raw::TableProvider;
use vello::peniko;

use crate::error::{ResourceError, Unsupported};
use crate::message::Message;

/// The data of a font to register with the engine.
#[derive(Clone)]
pub struct FontSource {
    /// The raw font data.
    pub data: Arc<[u8]>,
    /// The font index inside a collection.
    pub index: u32,
}

impl std::fmt::Debug for FontSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FontSource")
            .field("len", &self.data.len())
            .field("index", &self.index)
            .finish()
    }
}

impl FontSource {
    /// A font already in memory.
    pub fn bytes(bytes: impl Into<Arc<[u8]>>) -> Self {
        Self {
            data: bytes.into(),
            index: 0,
        }
    }

    /// A font read from a file.
    ///
    /// This reads the whole file into memory; it does not memory-map yet,
    /// so very large fonts are copied once.
    ///
    /// # Errors
    /// Any [`std::io::Error`] from reading the file.
    pub fn mapped(path: impl AsRef<Path>) -> std::io::Result<Self> {
        std::fs::read(path).map(Self::bytes)
    }

    /// Selects a font index inside a collection.
    #[must_use]
    pub fn with_index(self, index: u32) -> Self {
        Self { index, ..self }
    }
}

/// A font registered with an engine. Dropping it unregisters the font.
#[derive(Debug)]
pub struct Font {
    id: cherenkov::FontId,
    tx: Sender<Message>,
}

impl Font {
    /// A handle for the registered font `id`.
    #[must_use]
    pub const fn new(id: cherenkov::FontId, tx: Sender<Message>) -> Self {
        Self { id, tx }
    }

    /// The identifier glyph runs reference.
    #[must_use]
    pub const fn id(&self) -> cherenkov::FontId {
        self.id
    }
}

impl Drop for Font {
    fn drop(&mut self) {
        let _ = self.tx.send(Message::RemoveFont { id: self.id.raw() });
    }
}

/// Validates font data with `skrifa`, rejecting colour fonts.
///
/// Fonts carrying `COLR`, `CBDT`/`CBLC` or `sbix` outlines are colour fonts,
/// which vello rasterizes through a different path this backend does not
/// validate.
pub fn validate_font(data: &[u8], index: u32) -> Result<(), ResourceError> {
    let font = skrifa::FontRef::from_index(data, index)
        .map_err(|e| ResourceError::Font(format!("{e}")))?;
    for tag in [
        skrifa::Tag::new(b"COLR"),
        skrifa::Tag::new(b"CBDT"),
        skrifa::Tag::new(b"sbix"),
    ] {
        if font.data_for_tag(tag).is_some() {
            return Err(Unsupported::ColorFont.into());
        }
    }
    Ok(())
}

/// The data of an image to register with the engine.
#[derive(Debug)]
pub struct ImageSource {
    /// The image width in pixels.
    pub width: u32,
    /// The image height in pixels.
    pub height: u32,
    /// RGBA texels, sRGB-encoded, row-major.
    pub data: Arc<[u8]>,
    premultiplied: bool,
}

impl ImageSource {
    /// An `width` × `height` image of straight-alpha sRGB-encoded RGBA8
    /// texels.
    ///
    /// # Panics
    /// Panics when `data` is not exactly `width * height * 4` bytes.
    #[must_use]
    pub fn rgba8(width: u32, height: u32, data: impl Into<Arc<[u8]>>) -> Self {
        let data = data.into();
        assert_eq!(
            data.len(),
            (width * height * 4) as usize,
            "an rgba8 image needs width * height * 4 texel bytes"
        );
        Self {
            width,
            height,
            data,
            premultiplied: false,
        }
    }

    /// Marks the texels as premultiplied rather than straight alpha.
    #[must_use]
    pub fn premultiplied(self) -> Self {
        Self {
            premultiplied: true,
            ..self
        }
    }

    /// Builds the `peniko` image payload the render thread registers.
    #[must_use]
    pub fn into_image_data(self) -> peniko::ImageData {
        peniko::ImageData {
            data: peniko::Blob::new(std::sync::Arc::new(crate::message::SharedBytes(
                self.data.clone(),
            ))),
            format: peniko::ImageFormat::Rgba8,
            alpha_type: if self.premultiplied {
                peniko::ImageAlphaType::AlphaPremultiplied
            } else {
                peniko::ImageAlphaType::Alpha
            },
            width: self.width,
            height: self.height,
        }
    }
}

/// An image registered with an engine. Dropping it unregisters the image.
#[derive(Debug)]
pub struct Image {
    id: cherenkov::ImageId,
    tx: Sender<Message>,
}

impl Image {
    /// A handle for the registered image `id`.
    #[must_use]
    pub const fn new(id: cherenkov::ImageId, tx: Sender<Message>) -> Self {
        Self { id, tx }
    }

    /// The identifier image draws and image paints reference.
    #[must_use]
    pub const fn id(&self) -> cherenkov::ImageId {
        self.id
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        let _ = self.tx.send(Message::RemoveImage { id: self.id.raw() });
    }
}

/// A user shader's WGSL fragment source.
///
/// The engine prepends a prelude declaring `uniforms` (time, resolution),
/// `params` (up to 16 `vec4<f32>` of [`ShaderPaint`](cherenkov::ShaderPaint)
/// uniforms, zero-padded) and a fullscreen-triangle vertex shader; the
/// source supplies `@fragment fn main(@location(0) uv: vec2<f32>) ->
/// @location(0) vec4<f32>` returning a straight-alpha colour.
#[derive(Clone, Debug)]
pub struct ShaderSource {
    /// The fragment source, without the prelude.
    pub source: std::borrow::Cow<'static, str>,
    /// Whether the shader animates: when true, it is re-rendered every
    /// frame so `uniforms.time` advances and the engine keeps refreshing.
    pub animated: bool,
}

impl ShaderSource {
    /// A static shader from a WGSL fragment body.
    pub fn wgsl(fragment: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        Self {
            source: fragment.into(),
            animated: false,
        }
    }

    /// Marks the shader as animated (re-rendered each frame).
    #[must_use]
    pub fn animated(self) -> Self {
        Self {
            source: self.source,
            animated: true,
        }
    }
}

/// A shader registered with an engine. Dropping it unregisters the shader.
#[derive(Debug)]
pub struct Shader {
    id: cherenkov::ShaderId,
    tx: Sender<Message>,
}

impl Shader {
    /// A handle for the registered shader `id`.
    #[must_use]
    pub const fn new(id: cherenkov::ShaderId, tx: Sender<Message>) -> Self {
        Self { id, tx }
    }

    /// The identifier [`ShaderPaint`](cherenkov::ShaderPaint) references.
    #[must_use]
    pub const fn id(&self) -> cherenkov::ShaderId {
        self.id
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        let _ = self.tx.send(Message::RemoveShader { id: self.id.raw() });
    }
}

/// A filter effect registered with an engine. Dropping it unregisters the
/// filter.
#[derive(Debug)]
pub struct Filter {
    id: cherenkov::FilterId,
    tx: Sender<Message>,
}

impl Filter {
    /// A handle for the registered filter `id`.
    #[must_use]
    pub const fn new(id: cherenkov::FilterId, tx: Sender<Message>) -> Self {
        Self { id, tx }
    }

    /// The identifier [`LayerEdit::filter`](crate::LayerEdit::filter)
    /// references.
    #[must_use]
    pub const fn id(&self) -> cherenkov::FilterId {
        self.id
    }
}

impl Drop for Filter {
    fn drop(&mut self) {
        let _ = self.tx.send(Message::RemoveFilter { id: self.id.raw() });
    }
}
