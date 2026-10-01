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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
/// The host wake-up: called at most once between two
/// [`Engine::render`]s, the first time something is queued.
///
/// `wake` may run on the engine's thread or on the main thread — a
/// completion queued there by the render thread (a promoted plane's
/// attach) fires it — so the interior is guarded and the callback a
/// host registers must tolerate either.
pub struct Waker {
    callback: Mutex<Option<Box<dyn Fn()>>>,
    armed: AtomicBool,
}

impl Waker {
    pub(super) fn new() -> Self {
        Self {
            callback: Mutex::new(None),
            armed: AtomicBool::new(true),
        }
    }

    /// Calls the callback once if armed, then disarms until the next
    /// [`Engine::render`] re-arms.
    pub fn wake(&self) {
        if !self.armed.swap(false, Ordering::Relaxed) {
            return;
        }
        // Out from under the lock: the callback may itself touch the
        // engine, and a `set_waker` it makes wins over re-arming this
        // one.
        let callback = self.callback.lock().expect("waker poisoned").take();
        let Some(callback) = callback else { return };
        callback();
        let mut slot = self.callback.lock().expect("waker poisoned");
        if slot.is_none() {
            *slot = Some(callback);
        }
    }

    pub(super) fn arm(&self) {
        self.armed.store(true, Ordering::Relaxed);
    }
}

/// A [`Waker`] carried to the render thread so a completion the main
/// queue runs can wake the host for the next frame.
///
/// The render side stores and clones it but never calls `wake`: only the
/// block the main queue executes does.
#[derive(Clone)]
pub struct MainWaker(Arc<Waker>);
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "`wake` is invoked only on the main thread, where the engine's \
        own wake path runs, and `Waker`'s interior is guarded (`Mutex`, \
        `AtomicBool`); `Arc` owns the cross-thread hand-off"
)]
// SAFETY: as the expect reason states.
unsafe impl Send for MainWaker {}

impl MainWaker {
    /// Wraps the engine's waker. Called on the engine's thread.
    pub(crate) fn new(waker: &Arc<Waker>) -> Self {
        Self(Arc::clone(waker))
    }

    /// Fires the host's wake callback if armed. Main thread only.
    pub fn wake(&self) {
        self.0.wake();
    }
}
