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
/// Native it may run on any thread while the engine holds it, so it is
/// shared: `Send` to cross threads, `Sync` to be shared across them.
#[cfg(not(target_arch = "wasm32"))]
type Wake = dyn Fn() + Send + Sync;
#[cfg(target_arch = "wasm32")]
type Wake = dyn Fn();

/// The host wake-up, coalesced between engine renders.
/// Native callbacks may run on any thread; the callback slot is synchronized.
pub struct Waker {
    callback: Mutex<Option<Arc<Wake>>>,
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
        // Cloned under the lock, called outside it: the callback may
        // itself touch the engine — a `set_waker` it makes replaces the
        // slot — and a panicking callback must not lose the installed
        // one.
        let callback = self.callback.lock().expect("waker poisoned").clone();
        let Some(callback) = callback else { return };
        callback();
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
