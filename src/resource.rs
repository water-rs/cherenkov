//! Resource handles: [`Font`], [`Image`], [`Shader`] and [`Filter`] are
//! `Clone` over an `Rc`; the last drop queues the `remove_*` op on the
//! render thread. An [`Image`] also queues in-place pixel replacements.

use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use crate::ShaderId;
use crate::error::ResourceError;
use crate::glyph::FontId;
use crate::image::{Format, ImageData, ImageUpload};
use crate::message::BackdropId;
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

/// Queues a replacement of an image's pixels on the render thread and
/// answers with the backend's result.
#[cfg(not(target_arch = "wasm32"))]
pub type ReplaceImage = Rc<dyn Fn(ImageId, ImageUpload) -> Result<(), ResourceError>>;
/// Queues a replacement of an image's pixels on the local executor and
/// resolves with the backend's result.
#[cfg(target_arch = "wasm32")]
pub type ReplaceImage = Rc<
    dyn Fn(
        ImageId,
        ImageUpload,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), ResourceError>>>>,
>;

/// The shared state of a resource handle: the last `Rc` drop runs
/// `on_drop`, which queues the resource's `remove_*` op. `ops` carries the
/// render-thread operations a kind queues while it lives (an image's
/// replacement); kinds without any use `()`.
struct Inner<I, O = ()> {
    id: I,
    ops: O,
    on_drop: Option<Box<dyn FnOnce()>>,
}

impl<I, O> std::fmt::Debug for Inner<I, O>
where
    I: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl<I, O> Drop for Inner<I, O> {
    fn drop(&mut self) {
        if let Some(on_drop) = self.on_drop.take() {
            on_drop();
        }
    }
}

fn handle<I>(id: I, on_drop: impl FnOnce() + 'static) -> Rc<Inner<I>> {
    handle_with(id, (), on_drop)
}

fn handle_with<I, O>(id: I, ops: O, on_drop: impl FnOnce() + 'static) -> Rc<Inner<I, O>> {
    Rc::new(Inner {
        id,
        ops,
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
/// [`Image::replace`] swaps its pixels behind the same id. Dropping the last
/// clone unregisters the image.
#[derive(Debug)]
pub struct Image<F: Format> {
    inner: Rc<Inner<ImageId, ReplaceImage>>,
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
    pub(crate) fn new(
        id: ImageId,
        replace: ReplaceImage,
        on_drop: impl FnOnce() + 'static,
    ) -> Self {
        Self {
            inner: handle_with(id, replace, on_drop),
            format: std::marker::PhantomData,
        }
    }

    /// The identifier image draws and image paints reference.
    #[must_use]
    pub fn id(&self) -> ImageId {
        self.inner.id
    }

    /// Replaces the image's pixels in place, blocking until the render
    /// thread has applied the replacement.
    ///
    /// The id is unchanged, so every recording that names this image draws
    /// the new pixels from the next frame on, without re-recording. The
    /// replacement is ordered with frames on the render thread, so no frame
    /// samples a partly written image. The same dimensions reuse the
    /// backing storage; different dimensions reallocate it behind the same
    /// id. The next render redraws the surfaces whose content draws this
    /// image, and the engine's waker fires when there is at least one.
    ///
    /// `image` is validated by [`ImageData::new`]; the backend may still
    /// reject it, as it may in [`Engine::image`](crate::Engine::image).
    ///
    /// # Errors
    /// [`ResourceError::Image`] when the backend rejects the data (the image
    /// keeps its previous pixels), [`ResourceError::Lost`] when the render
    /// thread is gone.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn replace(&self, image: ImageData<F>) -> Result<(), ResourceError> {
        (self.inner.ops)(self.inner.id, image.into_upload())
    }

    /// Replaces the image's pixels in place, resolving once the local
    /// executor has applied the replacement.
    ///
    /// The id is unchanged, so every recording that names this image draws
    /// the new pixels from the next frame on, without re-recording. The
    /// replacement is ordered with frames on the executor, so no frame
    /// samples a partly written image. The same dimensions reuse the
    /// backing storage; different dimensions reallocate it behind the same
    /// id. The next render redraws the surfaces whose content draws this
    /// image, and the engine's waker fires when there is at least one.
    ///
    /// `image` is validated by [`ImageData::new`]; the backend may still
    /// reject it, as it may in [`Engine::image`](crate::Engine::image).
    ///
    /// # Errors
    /// [`ResourceError::Image`] when the backend rejects the data (the image
    /// keeps its previous pixels), [`ResourceError::Lost`] when the executor
    /// is gone.
    #[cfg(target_arch = "wasm32")]
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    pub async fn replace(&self, image: ImageData<F>) -> Result<(), ResourceError> {
        (self.inner.ops)(self.inner.id, image.into_upload()).await
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

/// A backdrop group: one capture and one spatial filter chain shared by
/// its members. `!Send`; dropping it unregisters the group, and a member
/// still sampling it makes the frame fail.
#[derive(Debug)]
pub struct BackdropGroup {
    inner: Rc<Inner<BackdropId>>,
}

impl BackdropGroup {
    pub(crate) fn new(id: BackdropId, on_drop: impl FnOnce() + 'static) -> Self {
        Self {
            inner: handle(id, on_drop),
        }
    }

    /// The group's identifier.
    #[must_use]
    pub fn id(&self) -> BackdropId {
        self.inner.id
    }

    /// A sample of this group for [`LayerEdit::backdrop`](crate::LayerEdit::backdrop).
    #[must_use]
    pub fn sample(&self) -> BackdropSample {
        BackdropSample {
            group: self.id(),
            effect: None,
        }
    }

    /// A sample of this group with a per-member effect, evaluated in the
    /// member's composite against the shared filtered capture.
    #[must_use]
    pub fn sample_with(&self, effect: impl Into<crate::BackdropEffect>) -> BackdropSample {
        BackdropSample {
            group: self.id(),
            effect: Some(effect.into()),
        }
    }
}

/// A sample of a [`BackdropGroup`], attached to a layer by
/// [`LayerEdit::backdrop`](crate::LayerEdit::backdrop).
#[derive(Clone, Debug, PartialEq)]
pub struct BackdropSample {
    /// The sampled group.
    group: BackdropId,
    /// The per-member effect applied in the member's composite.
    effect: Option<crate::BackdropEffect>,
}

impl BackdropSample {
    /// The sampled group.
    #[must_use]
    pub const fn group(&self) -> BackdropId {
        self.group
    }

    /// The per-member effect, when the sample was made with
    /// [`BackdropGroup::sample_with`].
    #[must_use]
    pub const fn effect(&self) -> Option<&crate::BackdropEffect> {
        self.effect.as_ref()
    }
}

/// A backdrop effect shader registered with an engine.
///
/// Made by [`Engine::backdrop_shader`](crate::Engine::backdrop_shader).
/// Dropping the last clone unregisters it; a member still sampling it
/// makes the frame fail.
#[derive(Debug)]
pub struct BackdropShader {
    inner: Rc<Inner<crate::message::BackdropShaderId>>,
    reach: f32,
}

impl Clone for BackdropShader {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
            reach: self.reach,
        }
    }
}

impl BackdropShader {
    pub(crate) fn new(
        id: crate::message::BackdropShaderId,
        reach: f32,
        on_drop: impl FnOnce() + 'static,
    ) -> Self {
        Self {
            inner: handle(id, on_drop),
            reach,
        }
    }

    /// The identifier [`BackdropShaderEffect`] references.
    #[must_use]
    pub fn id(&self) -> crate::message::BackdropShaderId {
        self.inner.id
    }

    /// A [`BackdropEffect::Shader`] for
    /// [`BackdropGroup::sample_with`], with `uniforms` in the shader's
    /// declared order (at most 64 finite values, packed four per `vec4`).
    #[must_use]
    pub fn effect(&self, uniforms: Vec<f32>) -> crate::BackdropShaderEffect {
        crate::BackdropShaderEffect {
            shader: self.id(),
            uniforms,
            reach: self.reach,
        }
    }
}
