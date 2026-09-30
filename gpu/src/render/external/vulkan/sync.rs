//! The acquisition and release state machine for one frame generation.
//!
//! One immutable generation has one shared acquisition state regardless of
//! how many layers or surfaces reference it. The first consuming submission
//! records the acquire barrier — the producer's actual layout to the
//! sampling layout, and the queue-family transfer from the producer's real
//! family (`FOREIGN_EXT` for the Android contract) to the engine's — and
//! registers the producer semaphore waits immediately before the consuming
//! `queue.submit`. A cancelled, unsubmitted plan leaves the frame
//! unacquired. Retirement schedules a release submission that transitions
//! back and signals the producer's release mechanism; every object is
//! retained until that submission completes. There is no
//! `vkQueueWaitIdle`, no device wait and no render-thread fence wait.

use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
use std::sync::Arc;

use ash::vk;
use ash::vk::Handle as _;
use rustc_hash::FxHashMap;

use super::{Native, NativeError, QueueFamily, ReleaseSync, Shared, Wait, ycbcr};

/// The acquisition state of one frame generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Imported, never consumed.
    Registered,
    /// Acquire barrier + wait staged for the encoder being built.
    AcquisitionPlanned,
    /// Consuming submission accepted; the wait executes on the GPU.
    AcquisitionSubmitted,
    /// Acquisition complete on the queue; later reads need nothing.
    OwnedForRead,
    /// Last retained owner dropped; release is queued.
    Retiring,
    /// The release submission is in flight.
    ReleaseSubmitted,
    /// The release submission completed; objects destroyed.
    Released,
}

/// A wait payload resolved to its `VkSemaphore` form, ready for
/// `add_wait_semaphore` at submit time.
pub struct PendingWait {
    pub semaphore: vk::Semaphore,
    /// `Some(value)` waits a timeline payload; `None` a binary one.
    pub value: Option<u64>,
    /// Whether the engine created (and so destroys) the semaphore.
    pub owned: bool,
}

/// One generation staged for acquisition in the encoder being built.
pub struct PendingAcquire {
    pub generation: Arc<Generation>,
    /// The resolved wait, registered on the queue at submit.
    pub wait: Option<PendingWait>,
}

/// How the frame's planes are bound; kept for the params/repr contract.
#[derive(Clone, Copy)]
pub enum Views {
    /// `fs_external`: integer plane views (`y`, `uv`).
    Planes { y: vk::ImageView, uv: vk::ImageView },
    /// `fs_external_format`: the conversion-carrying sampled view.
    /// `Repr::ExternalFormat` — the conversion-attached image view.
    /// Constructed only on Android, read by `write_set1` everywhere.
    #[allow(dead_code)]
    ExternalFormat { view: vk::ImageView },
    /// `Repr::Rgb` wraps the plane as a `wgpu::Texture`; nothing native.
    Wrapped,
}

/// The producer lease: handles whose lifetime the frame's ownership
/// contract ties to, released last — after every Vulkan object built on
/// them is destroyed.
pub enum Lease {
    /// Nothing extra; the image's own memory import consumed the fds.
    None,
    /// File descriptors kept open through the frame's lifetime (a sync-fd
    /// payload whose fd stays caller-owned, an AHB leak on Android).
    /// Open dmabuf descriptors retained until release completes —
    /// constructed by the Linux dmabuf importer's failures path only on
    /// platforms where `NativeFd::Owned` is possible.
    #[allow(dead_code)]
    Fds(Vec<OwnedFd>),
    /// A retained `AHardwareBuffer`, released on teardown.
    #[cfg(target_os = "android")]
    Ahb(*mut ndk_sys::AHardwareBuffer),
}

impl std::fmt::Debug for Lease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Fds(fds) => write!(f, "Fds({} fds)", fds.len()),
            #[cfg(target_os = "android")]
            Self::Ahb(_) => write!(f, "Ahb"),
        }
    }
}

#[cfg(target_os = "android")]
impl Drop for Lease {
    fn drop(&mut self) {
        if let Self::Ahb(ptr) = self {
            unsafe { ndk_sys::AHardwareBuffer_release(*ptr) };
        }
    }
}

/// Everything the release submission and the eventual destroy need,
/// packaged at the point the last retained owner drops.
pub struct Release {
    pub image: vk::Image,
    /// Destroy order: descriptor sets (via the pool), then views, then the
    /// conversion's sampler/conversion/layout, then the image, memory, and
    /// finally the semaphore payloads and producer lease.
    pub pool: Option<vk::DescriptorPool>,
    pub views: Vec<vk::ImageView>,
    pub conv: Option<Arc<ycbcr::Conv>>,
    pub memory: Vec<vk::DeviceMemory>,
    pub semaphores: Vec<vk::Semaphore>,
    /// How the release submission acknowledges the producer.
    pub sync_payload: Option<ReleaseSync>,
    /// The `FenceFd` export semaphore, signalled by the release submission
    /// and exported by [`Generation::release_fd`].
    pub fence_semaphore: Option<vk::Semaphore>,
    /// Set when the release submission is accepted; `release_fd` exports
    /// only after this. Shared with the generation's flag.
    pub submitted_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// The generation's state record, so the submit path can advance it
    /// through `ReleaseSubmitted` and `Released` after the generation's
    /// `Arc` has unwound.
    pub state: Option<Arc<std::sync::Mutex<State>>>,
    /// Whether the acquisition barrier ran — the release barrier only
    /// unwinds state the acquire established.
    pub acquired: bool,
    pub producer_layout: vk::ImageLayout,
    pub producer_family: QueueFamily,
    pub aspects: vk::ImageAspectFlags,
    pub lease: Lease,
}

impl Release {
    /// Records the release barrier — sampling layout back to the producer's
    /// layout, and ownership released to its queue family — into `cb`.
    ///
    /// # Safety
    /// `cb` must be a recording `VkCommandBuffer` on the shared device.
    pub unsafe fn encode_barrier(&self, shared: &Shared, cb: vk::CommandBuffer) {
        if !self.acquired {
            return;
        }
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::empty())
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .new_layout(self.producer_layout)
            .src_queue_family_index(shared.vk.queue_family)
            .dst_queue_family_index(self.producer_family.vk())
            .image(self.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: self.aspects,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        unsafe {
            shared.vk.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    /// The semaphores the release submission must signal — the release
    /// sync's payload plus, for `FenceFd`, the export semaphore itself.
    pub fn signals(&self) -> Vec<(vk::Semaphore, Option<u64>)> {
        let mut out = Vec::new();
        match &self.sync_payload {
            Some(ReleaseSync::FenceFd) => {
                if let Some(sem) = self.fence_semaphore {
                    out.push((sem, None));
                }
            }
            Some(ReleaseSync::Timeline { semaphore, value }) => {
                out.push((vk::Semaphore::from_raw(*semaphore), Some(*value)));
            }
            None => {}
        }
        out
    }

    /// Destroys the objects in dependency order. Runs once the release
    /// submission has completed, on whichever thread observed completion —
    /// always the render thread in practice, which serializes device calls.
    pub fn destroy(self, shared: &Shared) {
        let dev = &shared.vk.device;
        unsafe {
            if let Some(pool) = self.pool {
                // Destroying the pool frees every set allocated from it.
                dev.destroy_descriptor_pool(pool, None);
            }
            for view in self.views {
                dev.destroy_image_view(view, None);
            }
        }
        // The conversion's layout, sampler and conversion outlive the sets
        // and views that referenced them.
        if let Some(conv) = self.conv
            && let Ok(conv) = Arc::try_unwrap(conv)
        {
            conv.destroy(&shared.vk);
        }
        // Shared by another live generation: its release owns the last
        // reference and performs the destroy.
        unsafe {
            dev.destroy_image(self.image, None);
            for memory in self.memory {
                dev.free_memory(memory, None);
            }
            for semaphore in self.semaphores {
                dev.destroy_semaphore(semaphore, None);
            }
            if let Some(semaphore) = self.fence_semaphore {
                dev.destroy_semaphore(semaphore, None);
            }
        }
        if let Some(state) = &self.state {
            *state.lock().expect("generation state") = State::Released;
        }
        // The producer lease drops after every Vulkan object built on it.
        drop(self.lease);
    }
}

/// One imported frame generation — the shared acquisition record every
/// layer attachment deduplicates against.
pub struct Generation {
    pub shared: Arc<Shared>,
    /// Pixel extent of the frame.
    pub size: (u32, u32),
    /// The decode contract baked into the params uniform.
    pub color: crate::interop::FrameColor,
    /// The alpha contract for RGB planes.
    pub alpha: crate::interop::RgbAlpha,
    /// How the planes bind.
    pub repr: super::Repr,
    /// The imported image; barriers run on it.
    pub image: vk::Image,
    /// The views the native operation binds.
    pub views: Views,
    /// The external-format conversion object, for `Repr::ExternalFormat`.
    pub conv: Option<Arc<ycbcr::Conv>>,
    /// The `Repr::Rgb` wgpu wrapper texture, bound on the ordinary path.
    pub rgb_wrap: Option<wgpu::Texture>,
    /// Imported allocation bytes, charged separately in `Engine::memory`.
    pub bytes: u64,
    /// The producer's actual release-time layout.
    pub producer_layout: vk::ImageLayout,
    /// The producer's actual queue family (`FOREIGN_EXT` for Android).
    pub producer_family: QueueFamily,
    /// The aspects the plane views and barriers cover.
    pub aspects: vk::ImageAspectFlags,
    /// The pool this generation's set-1 descriptors allocate from.
    pub pool: Option<vk::DescriptorPool>,
    /// Set-1 descriptors cached per mask texture key.
    pub sets: std::sync::Mutex<FxHashMap<u64, vk::DescriptorSet>>,
    /// The shared acquisition state.
    pub state: Arc<std::sync::Mutex<State>>,
    /// The un-imported producer wait descriptor, taken at first use.
    pub wait: std::sync::Mutex<Option<Wait>>,
    /// The resolved wait semaphore, restored by a cancelled plan and
    /// surrendered to the release submission once consumed.
    pub resolved_wait: std::sync::Mutex<Option<PendingWait>>,
    /// The producer's release mechanism.
    pub release_sync: std::sync::Mutex<Option<ReleaseSync>>,
    /// The `FenceFd` export semaphore, created at import when requested.
    pub fence_semaphore: std::sync::Mutex<Option<vk::Semaphore>>,
    /// True once the release submission has been accepted; shared with
    /// the `Release` parts so the submit path can set it.
    pub release_submitted: Arc<std::sync::atomic::AtomicBool>,
    /// Engine-side retained references — the slots currently holding this
    /// generation. The producer's own `Frame` handles are not counted, so
    /// they may keep the generation alive for `release_fd` without
    /// delaying retirement.
    pub leases: std::sync::atomic::AtomicUsize,
    /// The destroy-time object set, moved out at retirement.
    pub parts: std::sync::Mutex<Option<Release>>,
}

impl Generation {
    /// The resolved wait payload for the first consuming submission, or
    /// `None` when the frame carries no synchronization.
    ///
    /// fd payloads import here — once, at first use — into a fresh binary
    /// semaphore owned by the generation; timeline payloads carry the
    /// host's semaphore.
    fn resolve_wait(&self) -> Result<Option<PendingWait>, NativeError> {
        let pending = self.resolved_wait.lock().expect("resolved wait").take();
        if let Some(pending) = pending {
            return Ok(Some(pending));
        }
        let Some(wait) = self.wait.lock().expect("frame wait").take() else {
            return Ok(None);
        };
        let dev = &self.shared.vk.device;
        let pending = match wait {
            Wait::Timeline { semaphore, value } => PendingWait {
                semaphore: vk::Semaphore::from_raw(semaphore),
                value: Some(value),
                owned: false,
            },
            Wait::OpaqueFd { fd } => {
                let semaphore =
                    unsafe { dev.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                        .map_err(NativeError::from)?;
                let info = vk::ImportSemaphoreFdInfoKHR::default()
                    .semaphore(semaphore)
                    .flags(vk::SemaphoreImportFlags::TEMPORARY)
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
                    .fd(fd.as_raw_fd());
                let res = unsafe {
                    self.shared
                        .vk
                        .external_semaphore_fd
                        .as_ref()
                        .expect("opaque-fd support checked at import")
                        .import_semaphore_fd(&info)
                };
                match res {
                    Ok(()) => {
                        // The driver consumed the fd.
                        let _ = fd.into_raw_fd();
                    }
                    Err(err) => {
                        unsafe { dev.destroy_semaphore(semaphore, None) };
                        return Err(err.into());
                    }
                }
                PendingWait {
                    semaphore,
                    value: None,
                    owned: true,
                }
            }
            Wait::SyncFd { fd } => {
                let semaphore =
                    unsafe { dev.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                        .map_err(NativeError::from)?;
                let info = vk::ImportSemaphoreFdInfoKHR::default()
                    .semaphore(semaphore)
                    .flags(vk::SemaphoreImportFlags::TEMPORARY)
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
                    .fd(fd.as_raw_fd());
                let res = unsafe {
                    self.shared
                        .vk
                        .external_semaphore_fd
                        .as_ref()
                        .expect("sync-fd support checked at import")
                        .import_semaphore_fd(&info)
                };
                if let Err(err) = res {
                    unsafe { dev.destroy_semaphore(semaphore, None) };
                    return Err(err.into());
                }
                PendingWait {
                    semaphore,
                    value: None,
                    owned: true,
                }
            }
        };
        Ok(Some(pending))
    }

    /// Records the acquire barrier into `cb` and returns the staged wait.
    ///
    /// # Safety
    /// `cb` must be a recording `VkCommandBuffer` on the shared device.
    #[allow(clippy::significant_drop_tightening)]
    pub unsafe fn acquire(
        self: &Arc<Self>,
        cb: vk::CommandBuffer,
    ) -> Result<Option<PendingWait>, NativeError> {
        let mut state = self.state.lock().expect("generation state");
        match *state {
            State::OwnedForRead | State::AcquisitionSubmitted => return Ok(None),
            State::Registered | State::AcquisitionPlanned => {}
            _ => return Err(NativeError::Invalid("external frame read while retiring")),
        }
        let wait = self.resolve_wait()?;
        let needs_barrier = self.producer_layout != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            || self.producer_family.vk() != self.shared.vk.queue_family;
        if needs_barrier {
            let barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(self.producer_layout)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(self.producer_family.vk())
                .dst_queue_family_index(self.shared.vk.queue_family)
                .image(self.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: self.aspects,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            unsafe {
                self.shared.vk.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }
        }
        *state = State::AcquisitionPlanned;
        Ok(wait)
    }
}

impl Generation {
    /// One engine-side reference (a slot install) began or ended.
    /// Retirement fires when the last engine reference ends; the
    /// producer's own `Frame` handles keep the generation alive for
    /// `release_fd` without being counted here.
    pub fn lease(&self) {
        self.leases
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }

    /// Drops one engine-side reference; the last one retires the frame.
    pub fn unlease(&self) {
        if self
            .leases
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel)
            == 1
        {
            self.retire_inner();
        }
    }

    /// Packages the destroy-time object set and queues the release
    /// submission. Idempotent: the first call wins; a generation dropped
    /// while `parts` are already out queues nothing.
    fn retire_inner(&self) {
        let mut parts = self.parts.lock().expect("release parts").take();
        if let Some(release) = parts.as_mut() {
            // A resolved-but-unconsumed wait semaphore dies with the frame.
            if let Some(wait) = self.resolved_wait.lock().expect("resolved wait").take()
                && wait.owned
            {
                release.semaphores.push(wait.semaphore);
            }
            release.acquired = matches!(
                *self.state.lock().expect("generation state"),
                State::OwnedForRead | State::AcquisitionSubmitted | State::Retiring
            );
            release.sync_payload = self.release_sync.lock().expect("release sync").take();
            release.fence_semaphore = self.fence_semaphore.lock().expect("fence semaphore").take();
            release.submitted_flag = Some(Arc::clone(&self.release_submitted));
            release.state = Some(Arc::clone(&self.state));
            for (_, set) in self.sets.lock().expect("frame sets").drain() {
                // Sets are freed with the pool; the map drain is bookkeeping.
                let _ = set;
            }
        }
        if let Some(release) = parts.take() {
            *self.state.lock().expect("generation state") = State::Retiring;
            self.shared.retire(release);
        }
    }
}

impl Drop for Generation {
    /// A generation that was never leased — imported but never installed
    /// — still retires its lease and objects; a leased generation already
    /// retired when its last slot dropped.
    fn drop(&mut self) {
        self.retire_inner();
    }
}

impl Generation {
    /// Exports the `FenceFd` release payload. The spec requires the export
    /// to run while the semaphore still has a pending signal, so the call
    /// is valid from the accepted release submission until it executes.
    #[allow(clippy::significant_drop_tightening)]
    pub fn release_fd(&self) -> Result<OwnedFd, NativeError> {
        if !self
            .release_submitted
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(NativeError::Unready);
        }
        let guard = self.fence_semaphore.lock().expect("fence semaphore");
        let Some(&semaphore) = guard.as_ref() else {
            return Err(NativeError::Unsupported("frame has no fence release"));
        };
        let fd = unsafe {
            self.shared
                .vk
                .external_semaphore_fd
                .as_ref()
                .expect("sync-fd support checked at import")
                .get_semaphore_fd(
                    &vk::SemaphoreGetFdInfoKHR::default()
                        .semaphore(semaphore)
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD),
                )
        }
        .map_err(NativeError::from)?;
        // SAFETY: `get_semaphore_fd` returned a new fd owned by the caller.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}

/// Stages `generation`'s acquisition into the encoder: records the barrier and
/// returns the pending-wait entry to stage.
///
/// # Safety
/// `cb` must be a recording `VkCommandBuffer` on the shared device.
pub unsafe fn stage_acquire(
    generation: &Arc<Generation>,
    cb: vk::CommandBuffer,
) -> Result<Option<PendingAcquire>, NativeError> {
    let wait = unsafe { generation.acquire(cb) }?;
    let staged = wait.map(|wait| PendingAcquire {
        generation: Arc::clone(generation),
        wait: Some(wait),
    });
    if staged.is_none()
        && *generation.state.lock().expect("generation state") == State::AcquisitionPlanned
    {
        // No wait payload — still mark acquisition as staged so the submit
        // finalises it.
        return Ok(Some(PendingAcquire {
            generation: Arc::clone(generation),
            wait: None,
        }));
    }
    Ok(staged)
}

/// Registers every staged generation's wait on the queue immediately
/// before the consuming submission.
pub fn submit_waits(native: &Native, queue: &wgpu::hal::vulkan::Queue) {
    for pending in &native.staged {
        if let Some(wait) = &pending.wait {
            queue.add_wait_semaphore(
                wait.semaphore,
                wait.value,
                // Conservative stage covering the acquire and all dependent
                // work; narrowing it is a separately measured change.
                vk::PipelineStageFlags::ALL_COMMANDS,
            );
        }
    }
}

/// Finalises staged generations after the consuming submission is
/// accepted: binary waits are consumed once for the generation; timeline
/// waits may legally repeat but are unnecessary on the ordered queue.
pub fn mark_submitted(native: &mut Native) {
    for pending in native.staged.drain(..) {
        if let Some(wait) = pending.wait
            && wait.owned
        {
            // The consumed binary semaphore stays alive through the
            // release submission, then dies with the frame's objects.
            let mut parts = pending.generation.parts.lock().expect("release parts");
            if let Some(release) = parts.as_mut() {
                release.semaphores.push(wait.semaphore);
            } else {
                unsafe {
                    native
                        .shared
                        .vk
                        .device
                        .destroy_semaphore(wait.semaphore, None);
                };
            }
        }
        let mut state = pending.generation.state.lock().expect("generation state");
        if *state == State::AcquisitionPlanned || *state == State::AcquisitionSubmitted {
            *state = State::OwnedForRead;
        }
    }
}

/// Cancels an unsubmitted plan: frames go back to unacquired and staged
/// queue waits are dropped — nothing was registered on the queue yet.
pub fn cancel_staged(native: &mut Native) {
    for pending in native.staged.drain(..) {
        let mut state = pending.generation.state.lock().expect("generation state");
        if *state == State::AcquisitionPlanned {
            *state = State::Registered;
        }
        drop(state);
        if let Some(wait) = pending.wait {
            // The resolved payload goes back to the generation so a later
            // plan reuses it — an imported fd cannot be imported twice.
            *pending
                .generation
                .resolved_wait
                .lock()
                .expect("resolved wait") = Some(wait);
        }
    }
}

/// Drains the retire queue into release submissions. Called before each
/// submission so an idle engine still processes a pending retirement the
/// next time anything is submitted; the renderer calls it from
/// `flush_releases` with a fresh encoder.
pub fn drain_releases(native: &Native) -> Vec<Release> {
    native
        .shared
        .vk
        .pending_release
        .lock()
        .expect("pending release")
        .drain(..)
        .collect()
}
