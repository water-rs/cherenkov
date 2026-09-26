// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Font registration: [`FontSource`] bytes are validated on the caller
//! thread, then handed to the render thread by [`Font`].

use std::path::Path;
use std::sync::Arc;

use skrifa::MetadataProvider;
use skrifa::raw::TableProvider;

use crate::error::{ResourceError, Unsupported};

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

/// A font registered with an engine. Cloning is cheap; ids are unique per
/// engine.
#[derive(Clone, Debug)]
pub struct Font {
    id: cherenkov::FontId,
}

impl Font {
    /// A handle for the registered font `id`.
    #[must_use]
    pub const fn new(id: cherenkov::FontId) -> Self {
        Self { id }
    }

    /// The identifier glyph runs reference.
    #[must_use]
    pub const fn id(&self) -> cherenkov::FontId {
        self.id
    }
}

/// Validates font data with `skrifa`, rejecting colour fonts the
/// rasterizer cannot draw.
///
/// `COLR` fonts render through the colour-glyph lowering; fonts carrying
/// `CBDT`/`CBLC` or `sbix` bitmaps without outline glyphs cannot
/// rasterize, and an `SVG ` table (SVG-in-OpenType) has no SVG glyph
/// support at all.
pub fn validate_font(data: &[u8], index: u32) -> Result<(), ResourceError> {
    let font = skrifa::FontRef::from_index(data, index)
        .map_err(|e| ResourceError::Font(format!("{e}")))?;
    if font.data_for_tag(skrifa::Tag::new(b"SVG ")).is_some() {
        return Err(Unsupported::ColorFont.into());
    }
    if font.outline_glyphs().iter().next().is_none()
        && [skrifa::Tag::new(b"CBDT"), skrifa::Tag::new(b"sbix")]
            .iter()
            .any(|tag| font.data_for_tag(*tag).is_some())
    {
        return Err(Unsupported::ColorFont.into());
    }
    Ok(())
}
