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
/// The callback is transferable on native targets and local on wasm32.
#[cfg(not(target_arch = "wasm32"))]
type Wake = dyn Fn() + Send;
#[cfg(target_arch = "wasm32")]
type Wake = dyn Fn();

/// The host wake-up, coalesced between engine renders.
/// Native callbacks may run on any thread; the callback slot is synchronized.
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

/// A transferable completion notification for backend work.
///
/// On native targets this can wake the host from any thread. On wasm32 it
/// retains the engine's single-threaded callback contract.
#[derive(Clone)]
pub struct CompletionWaker(Arc<Waker>);

impl CompletionWaker {
    /// Wraps the engine's waker. Called on the engine's thread.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn new(waker: &Arc<Waker>) -> Self {
        Self(Arc::clone(waker))
    }

    /// Fires the host's wake callback if armed.
    pub fn wake(&self) {
        self.0.wake();
    }
}
