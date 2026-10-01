//! The `AHardwareBuffer` pool: each buffer is filled, imported as a new
//! frame generation, and reused only once its merged release fence
//! (engine + compositor) has signalled.

use std::os::fd::AsRawFd;
use std::os::fd::OwnedFd;

use ash::vk::{self, Handle as _};
use cherenkov_gpu::interop::RgbAlpha;
use cherenkov_gpu::interop::vulkan::{
    self, Ahb, Frame, FrameSource, NativeError, ReleaseSync, Wait,
};
use ndk_sys::AHardwareBuffer;

use crate::ahb;
use crate::logcat;
use crate::scenario::Spec;

/// Four buffers cover the engine's plus `SurfaceFlinger`'s release latency
/// with slack; a deeper stall simply pauses the video.
const POOL: usize = 4;

/// A ring of `AHardwareBuffer`s feeding one video layer.
pub struct Pool {
    device: vulkan::Device,
    spec: Spec,
    slots: Vec<Slot>,
    /// Frames handed to the engine.
    pub produced: u64,
    /// Release fences observed signalled.
    pub signalled: u64,
    /// The pre-signalled timeline acquire (`timeline` pools only).
    semaphore: Option<vk::Semaphore>,
    ash: ash::Device,
    stalled: bool,
}

struct Slot {
    ahb: *mut AHardwareBuffer,
    flight: Option<Flight>,
}

struct Flight {
    /// The live generation; its `release_fd` becomes available once the
    /// release submission runs.
    frame: Frame,
    fd: Option<OwnedFd>,
}

impl Pool {
    /// Allocates the pool's buffers.
    ///
    /// # Errors
    /// When `spec.timeline` is set but the device has no timeline
    /// semaphores, or Vulkan refuses the semaphore.
    pub fn new(device: &vulkan::Device, spec: Spec) -> Result<Self, String> {
        let ash = device.shared.vk.device.clone();
        let semaphore = if spec.timeline {
            if !device.caps().timeline_semaphore {
                return Err("timeline semaphores unsupported".into());
            }
            let mut kind =
                vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
            let semaphore = unsafe {
                ash.create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                    None,
                )
            }
            .map_err(|e| format!("timeline semaphore: {e}"))?;
            // Host-signal value 1 once: every frame's acquire at value 1
            // is immediately satisfied.
            unsafe {
                ash.signal_semaphore(
                    &vk::SemaphoreSignalInfo::default()
                        .semaphore(semaphore)
                        .value(1),
                )
            }
            .map_err(|e| format!("timeline signal: {e}"))?;
            Some(semaphore)
        } else {
            None
        };
        let slots = (0..POOL)
            .map(|_| Slot {
                ahb: ahb::alloc(spec.format, spec.overlay),
                flight: None,
            })
            .collect();
        Ok(Self {
            device: device.clone(),
            spec,
            slots,
            produced: 0,
            signalled: 0,
            semaphore,
            ash,
            stalled: false,
        })
    }

    /// Fills a free buffer, imports it as the next generation and returns
    /// it for installation. `None` while every buffer is in flight.
    pub fn produce(&mut self) -> Option<Frame> {
        self.drain();
        let Some(slot) = self.slots.iter_mut().find(|slot| slot.flight.is_none()) else {
            if !self.stalled {
                self.stalled = true;
                logcat::warn("producer stalled: every AHB is still in flight");
            }
            return None;
        };
        self.stalled = false;
        unsafe { ahb::fill(self.spec.format, slot.ahb, self.produced) };
        let sync = self.semaphore.map(|semaphore| Wait::Timeline {
            semaphore: semaphore.as_raw(),
            value: 1,
        });
        let frame = self.device.import(FrameSource::Ahb(Box::new(Ahb {
            buffer: slot.ahb.cast(),
            sync,
            release: Some(ReleaseSync::FenceFd),
            color: self.spec.color,
            alpha: RgbAlpha::Opaque,
            hdr: self.spec.hdr,
        })));
        match frame {
            Ok(frame) => {
                self.produced += 1;
                slot.flight = Some(Flight {
                    frame: frame.clone(),
                    fd: None,
                });
                Some(frame)
            }
            Err(e) => {
                logcat::error(&format!("frame import failed: {e}"));
                None
            }
        }
    }

    /// Counts signalled release fences and frees their buffers.
    fn drain(&mut self) {
        for slot in &mut self.slots {
            let Some(flight) = &mut slot.flight else {
                continue;
            };
            if flight.fd.is_none() {
                match flight.frame.release_fd() {
                    Ok(fd) => flight.fd = Some(fd),
                    Err(NativeError::Unready) => continue,
                    Err(e) => {
                        logcat::warn(&format!("release_fd failed: {e}"));
                        continue;
                    }
                }
            }
            let Some(fd) = &flight.fd else {
                continue;
            };
            let mut pfd = libc::pollfd {
                fd: fd.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            let rc = unsafe { libc::poll(&raw mut pfd, 1, 0) };
            if rc > 0 {
                slot.flight = None;
                self.signalled += 1;
            }
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        for slot in &self.slots {
            unsafe { ahb::release(slot.ahb) };
        }
        if let Some(semaphore) = self.semaphore {
            unsafe { self.ash.destroy_semaphore(semaphore, None) };
        }
    }
}
