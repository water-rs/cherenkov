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
/// The host wake-up: called at most once between two
/// [`Engine::render`]s, the first time something is queued.
pub struct Waker {
    callback: RefCell<Option<Box<dyn Fn()>>>,
    armed: Cell<bool>,
}

impl Waker {
    pub(super) fn new() -> Self {
        Self {
            callback: RefCell::new(None),
            armed: Cell::new(true),
        }
    }

    /// Calls the callback once if armed, then disarms until the next
    /// [`Engine::render`] re-arms.
    pub fn wake(&self) {
        if !self.armed.replace(false) {
            return;
        }
        if let Some(callback) = self.callback.borrow().as_ref() {
            callback();
        }
    }

    pub(super) fn arm(&self) {
        self.armed.set(true);
    }
}
