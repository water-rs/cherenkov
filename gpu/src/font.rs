// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Font registration: [`FontSource`] bytes are validated on the caller
//! thread, then handed to the render thread by [`Font`].

use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::mpsc::Sender;

use skrifa::MetadataProvider;
use skrifa::raw::TableProvider;

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
    /// This slice reads the whole file into memory; it does not memory-map
    /// yet, so very large fonts are copied once.
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

/// The shared font state; dropping the last clone releases the font.
struct FontInner {
    id: cherenkov::FontId,
    tx: Sender<Message>,
}

impl Drop for FontInner {
    fn drop(&mut self) {
        let _ = self.tx.send(Message::RemoveFont { id: self.id.raw() });
    }
}

/// A font registered with an engine. Cloning is cheap; the font data and
/// its render-thread caches are released when the last clone drops.
#[derive(Clone)]
pub struct Font {
    inner: Rc<FontInner>,
}

impl Font {
    /// A handle for the registered font `id`.
    #[must_use]
    pub fn new(id: cherenkov::FontId, tx: Sender<Message>) -> Self {
        Self {
            inner: Rc::new(FontInner { id, tx }),
        }
    }

    /// The identifier glyph runs reference.
    #[must_use]
    pub fn id(&self) -> cherenkov::FontId {
        self.inner.id
    }
}

/// Validates font data with `skrifa`, rejecting bitmap-only colour fonts.
///
/// `COLR` fonts render through the colour-glyph lowering; fonts carrying
/// `CBDT`/`CBLC` or `sbix` bitmaps without outline glyphs cannot rasterize.
pub fn validate_font(data: &[u8], index: u32) -> Result<(), ResourceError> {
    let font = skrifa::FontRef::from_index(data, index)
        .map_err(|e| ResourceError::Font(format!("{e}")))?;
    if font.outline_glyphs().iter().next().is_none()
        && [skrifa::Tag::new(b"CBDT"), skrifa::Tag::new(b"sbix")]
            .iter()
            .any(|tag| font.data_for_tag(*tag).is_some())
    {
        return Err(Unsupported::ColorFont.into());
    }
    Ok(())
}
