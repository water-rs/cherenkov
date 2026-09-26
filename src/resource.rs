// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Resource handles: [`Font`], [`Image`], [`Shader`] and [`Filter`] are
//! `Clone` over an `Rc`; the last drop queues the `remove_*` op on the
//! render thread.

use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use crate::ShaderId;
use crate::glyph::FontId;
use crate::image::Format;
use crate::paint::ImageId;
use crate::style::FilterId;

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

/// The shared state of a resource handle: the last `Rc` drop runs
/// `on_drop`, which queues the resource's `remove_*` op.
struct Inner<I> {
    id: I,
    on_drop: Option<Box<dyn FnOnce()>>,
}

impl<I> std::fmt::Debug for Inner<I>
where
    I: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl<I> Drop for Inner<I> {
    fn drop(&mut self) {
        if let Some(on_drop) = self.on_drop.take() {
            on_drop();
        }
    }
}

fn handle<I>(id: I, on_drop: impl FnOnce() + 'static) -> Rc<Inner<I>> {
    Rc::new(Inner {
        id,
        on_drop: Some(Box::new(on_drop)),
    })
}

/// A font registered with an engine. Dropping the last clone unregisters
/// the font.
#[derive(Debug)]
pub struct Font {
    inner: Rc<Inner<FontId>>,
}

impl Clone for Font {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl Font {
    pub(crate) fn new(id: FontId, on_drop: impl FnOnce() + 'static) -> Self {
        Self {
            inner: handle(id, on_drop),
        }
    }

    /// The identifier glyph runs reference.
    #[must_use]
    pub fn id(&self) -> FontId {
        self.inner.id
    }
}

/// An image registered with an engine, typed by its storage [`Format`].
/// Dropping the last clone unregisters the image.
#[derive(Debug)]
pub struct Image<F: Format> {
    inner: Rc<Inner<ImageId>>,
    format: std::marker::PhantomData<F>,
}

impl<F: Format> Clone for Image<F> {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
            format: std::marker::PhantomData,
        }
    }
}

impl<F: Format> Image<F> {
    pub(crate) fn new(id: ImageId, on_drop: impl FnOnce() + 'static) -> Self {
        Self {
            inner: handle(id, on_drop),
            format: std::marker::PhantomData,
        }
    }

    /// The identifier image draws and image paints reference.
    #[must_use]
    pub fn id(&self) -> ImageId {
        self.inner.id
    }
}

/// A shader registered with an engine. Dropping the last clone unregisters
/// the shader.
#[derive(Debug)]
pub struct Shader {
    inner: Rc<Inner<ShaderId>>,
}

impl Clone for Shader {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl Shader {
    pub(crate) fn new(id: ShaderId, on_drop: impl FnOnce() + 'static) -> Self {
        Self {
            inner: handle(id, on_drop),
        }
    }

    /// The identifier [`ShaderPaint`](crate::paint::ShaderPaint) references.
    #[must_use]
    pub fn id(&self) -> ShaderId {
        self.inner.id
    }
}

/// A filter or effect registered with an engine. Dropping the last clone
/// unregisters it.
#[derive(Debug)]
pub struct Filter {
    inner: Rc<Inner<FilterId>>,
}

impl Clone for Filter {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl Filter {
    pub(crate) fn new(id: FilterId, on_drop: impl FnOnce() + 'static) -> Self {
        Self {
            inner: handle(id, on_drop),
        }
    }

    /// The identifier [`LayerEdit::filter`](crate::LayerEdit::filter)
    /// references.
    #[must_use]
    pub fn id(&self) -> FilterId {
        self.inner.id
    }
}
