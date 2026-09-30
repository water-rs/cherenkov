//! Target-appropriate rendering: a native thread or the owning browser JS thread.
#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(not(target_arch = "wasm32"))]
mod native;
pub mod thread;
#[cfg(target_arch = "wasm32")]
pub use browser::Engine;
#[cfg(not(target_arch = "wasm32"))]
pub use native::Engine;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::backend::Visibility;

/// The host wake-up: called at most once between two
/// [`Engine::render`]s, the first time something is queued on a visible
/// surface.
///
/// It also counts the live and the visible surfaces: while every live
/// surface is hidden it calls nothing, and [`Engine::render`] fails with
/// [`RenderError::Hidden`](crate::RenderError::Hidden).
pub struct Waker {
    callback: RefCell<Option<Box<dyn Fn()>>>,
    armed: Cell<bool>,
    /// Live surface handles.
    surfaces: Cell<usize>,
    /// Live surface handles that are visible.
    visible: Cell<usize>,
}

impl Waker {
    pub(super) fn new() -> Self {
        Self {
            callback: RefCell::new(None),
            armed: Cell::new(true),
            surfaces: Cell::new(0),
            visible: Cell::new(0),
        }
    }

    /// Calls the callback once if armed and a surface is visible, then
    /// disarms until the next [`Engine::render`] re-arms.
    pub fn wake(&self) {
        if self.visible.get() == 0 || !self.armed.replace(false) {
            return;
        }
        if let Some(callback) = self.callback.borrow().as_ref() {
            callback();
        }
    }

    pub(super) fn arm(&self) {
        self.armed.set(true);
    }

    /// Whether a surface is alive and every live surface is hidden.
    pub(super) const fn all_hidden(&self) -> bool {
        self.surfaces.get() > 0 && self.visible.get() == 0
    }

    /// One visible surface fewer. Once none is visible the host renders
    /// nothing, so the next surface shown must be able to wake it: the
    /// waker re-arms.
    fn hide_one(&self) {
        let visible = self.visible.get() - 1;
        self.visible.set(visible);
        if visible == 0 {
            self.arm();
        }
    }
}

/// One surface's gate on the engine's [`Waker`]: every wake the surface's
/// ops, bound signals and live operands raise passes through it, and none
/// passes while the surface is hidden.
pub struct SurfaceWaker {
    engine: Rc<Waker>,
    visibility: Cell<Visibility>,
    /// Set when the surface handle dropped; its lingering layers wake
    /// nothing.
    retired: Cell<bool>,
}

impl SurfaceWaker {
    /// A visible surface's gate.
    pub(crate) fn new(engine: Rc<Waker>) -> Self {
        engine.surfaces.set(engine.surfaces.get() + 1);
        engine.visible.set(engine.visible.get() + 1);
        Self {
            engine,
            visibility: Cell::new(Visibility::Visible),
            retired: Cell::new(false),
        }
    }

    /// Wakes the host unless the surface is hidden or dropped.
    pub fn wake(&self) {
        if self.visibility.get() == Visibility::Visible && !self.retired.get() {
            self.engine.wake();
        }
    }

    /// The surface's current visibility.
    pub(crate) const fn visibility(&self) -> Visibility {
        self.visibility.get()
    }

    /// Records a visibility change. Showing the surface requests the
    /// frame that draws it.
    pub(crate) fn set_visibility(&self, visibility: Visibility) {
        if self.retired.get() || self.visibility.replace(visibility) == visibility {
            return;
        }
        match visibility {
            Visibility::Hidden => self.engine.hide_one(),
            Visibility::Visible => {
                self.engine.visible.set(self.engine.visible.get() + 1);
                self.engine.wake();
            }
        }
    }

    /// The surface handle dropped: it no longer counts, visible or not.
    pub(crate) fn retire(&self) {
        if self.retired.replace(true) {
            return;
        }
        self.engine.surfaces.set(self.engine.surfaces.get() - 1);
        if self.visibility.get() == Visibility::Visible {
            self.engine.hide_one();
        }
    }
}
