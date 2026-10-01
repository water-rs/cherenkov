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
/// host registers must be `Send`.
/// The callback a host registers with `set_waker`: `Send` where a wake
/// may be fired by a completion another thread queued on the main queue
/// (a promoted plane's attach), a plain `Fn()` on the single-threaded
/// wasm engine.
#[cfg(not(target_family = "wasm"))]
type Wake = dyn Fn() + Send;
#[cfg(target_family = "wasm")]
type Wake = dyn Fn();

pub struct Waker {
    callback: Mutex<Option<Box<Wake>>>,
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
