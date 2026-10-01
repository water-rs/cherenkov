//! Pixel-side `AHardwareBuffer` external-frame verification (issue #166,
//! plan E7). Built in CI for aarch64-linux-android; run the artifact on
//! a Pixel 9 Pro over adb:
//!
//! ```text
//! cargo ndk --target aarch64-linux-android --platform 30 \
//!     build -p cherenkov-gpu --test vulkan_external_android
//! BIN=$(ls -t target/aarch64-linux-android/debug/deps/vulkan_external_android-* | head -1)
//! adb push "$BIN" /data/local/tmp/vulkan_external_android
//! adb shell 'cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp \
//!     ./vulkan_external_android --nocapture --test-threads=1'
//! ```
//!
//! Native `AHardwareBuffer` success is a required result here: the Pixel
//! exposes the AHB, YCbCr and fd-semaphore extensions the import is built
//! on, so a missing capability is a failure, not a skip.
#![cfg(target_os = "android")]

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ash::vk;
use ash::vk::Handle as _;
use cherenkov::kurbo::{BezPath, Rect};
use cherenkov::{Draw, Engine, FrameTime, Next, WorkingColor};
use cherenkov_gpu::{
    Gpu, GpuConfig,
    interop::{
        ExternalFrame, FrameColor, OutputAlpha, OutputColor, Presenter, RgbAlpha, SharedDevice,
        TextureOutput, TextureTarget,
        vulkan::{self, FrameSource},
        wgpu,
    },
};

/// The engine device and the native import context — the Pixel must
/// provide every capability the import relies on, so absence fails.
fn setup() -> (SharedDevice, vulkan::Device) {
    let shared = SharedDevice::create(&GpuConfig::default()).expect("shared device");
    assert_eq!(
        shared.adapter.get_info().backend,
        wgpu::Backend::Vulkan,
        "the Pixel test must run on Vulkan"
    );
    let device = vulkan::Device::new(&shared).expect("vulkan import device");
    let caps = device.caps();
    eprintln!("device caps: {caps:?}");
    assert!(caps.timeline_semaphore, "timeline semaphores required");
    assert!(
        caps.sampler_ycbcr_conversion,
        "YCbCr conversion required for the external-format path"
    );
    assert!(caps.external_semaphore_sync_fd, "SYNC_FD export required");
    (shared, device)
}

/// Raw handles the producer side of a test uses.
fn raw(shared: &SharedDevice) -> (ash::Device, vk::Queue, u32) {
    let hal = unsafe { shared.device.as_hal::<wgpu::hal::vulkan::Api>() };
    let hal = hal.as_ref().expect("vulkan device");
    let device = hal.raw_device().clone();
    let queue = hal.raw_queue();
    (device, queue, hal.queue_family_index())
}

/// Records `record` on a one-shot buffer and waits for it — the producer
/// side may wait on the host; the engine path never does.
fn run_once(
    dev: &ash::Device,
    queue: vk::Queue,
    family: u32,
    record: impl FnOnce(vk::CommandBuffer),
) {
    let pool = unsafe {
        dev.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )
    }
    .expect("command pool");
    let cb = unsafe {
        dev.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .expect("command buffer")[0];
    unsafe {
        dev.begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .expect("begin");
        record(cb);
        dev.end_command_buffer(cb).expect("end");
        dev.queue_submit(
            queue,
            &[vk::SubmitInfo::default().command_buffers(&[cb])],
            vk::Fence::null(),
        )
        .expect("submit");
        dev.queue_wait_idle(queue).expect("producer wait");
        dev.destroy_command_pool(pool, None);
    }
}

/// Allocates an `AHardwareBuffer`; the returned pointer is owned by the
/// caller (one acquire reference).
fn alloc_ahb(
    format: u32,
    width: u32,
    height: u32,
    usage: u64,
) -> Option<*mut ndk_sys::AHardwareBuffer> {
    let desc = ndk_sys::AHardwareBuffer_Desc {
        width,
        height,
        layers: 1,
        format,
        usage,
        stride: 0,
        rfu0: 0,
        rfu1: 0,
    };
    if unsafe { ndk_sys::AHardwareBuffer_isSupported(&raw const desc) } == 0 {
        eprintln!("AHB format {format:#x} not supported at {width}x{height}");
        return None;
    }
    let mut buffer = std::ptr::null_mut();
    if unsafe { ndk_sys::AHardwareBuffer_allocate(&raw const desc, &raw mut buffer) } != 0 {
        return None;
    }
    Some(buffer)
}

/// Fills an `RGBA_8888` AHB with `rgba` through a CPU lock — this is the
/// producer writing content, not a consumer copy.
fn make_ahb_rgb(w: u32, h: u32, rgba: [u8; 4]) -> *mut ndk_sys::AHardwareBuffer {
    let usage = ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE.0
        | ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN.0;
    let buffer = alloc_ahb(
        ndk_sys::AHardwareBuffer_Format::AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM.0,
        w,
        h,
        usage,
    )
    .expect("RGB AHB");
    let mut addr = std::ptr::null_mut();
    let rc = unsafe {
        ndk_sys::AHardwareBuffer_lock(
            buffer,
            ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN.0,
            -1,
            std::ptr::null(),
            &raw mut addr,
        )
    };
    assert_eq!(rc, 0, "AHB lock");
    let mut desc = unsafe { std::mem::zeroed::<ndk_sys::AHardwareBuffer_Desc>() };
    unsafe { ndk_sys::AHardwareBuffer_describe(buffer, &raw mut desc) };
    unsafe {
        let stride = desc.stride as usize;
        let base = addr.cast::<u8>();
        for row in 0..h as usize {
            for col in 0..w as usize {
                base.add(row * stride * 4 + col * 4).write_bytes(rgba[0], 1);
                base.add(row * stride * 4 + col * 4 + 1)
                    .write_bytes(rgba[1], 1);
                base.add(row * stride * 4 + col * 4 + 2)
                    .write_bytes(rgba[2], 1);
                base.add(row * stride * 4 + col * 4 + 3)
                    .write_bytes(rgba[3], 1);
            }
        }
        ndk_sys::AHardwareBuffer_unlock(buffer, std::ptr::null_mut());
    }
    buffer
}

/// Fills a `Y8Cb8Cr8_420` AHB with a flat `luma`/`chroma` through
/// `AHardwareBuffer_lockPlanes` — the producer's own content write.
fn make_ahb_nv12(
    width: u32,
    height: u32,
    luma: u8,
    chroma: (u8, u8),
) -> *mut ndk_sys::AHardwareBuffer {
    let usage = ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE.0
        | ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN.0;
    let buffer = alloc_ahb(
        ndk_sys::AHardwareBuffer_Format::AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420.0,
        width,
        height,
        usage,
    )
    .expect("NV12 AHB");
    let mut planes = unsafe { std::mem::zeroed::<ndk_sys::AHardwareBuffer_Planes>() };
    let rc = unsafe {
        ndk_sys::AHardwareBuffer_lockPlanes(
            buffer,
            ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN.0,
            -1,
            std::ptr::null(),
            &raw mut planes,
        )
    };
    assert_eq!(rc, 0, "AHB lockPlanes");
    unsafe {
        let luma_plane = planes.planes[0];
        let luma_base = luma_plane.data.cast::<u8>();
        for row in 0..height as usize {
            luma_base
                .add(row * luma_plane.rowStride as usize)
                .write_bytes(luma, width as usize);
        }
        if planes.planeCount >= 3 {
            // Separate U/V planes: fill both.
            let cb = planes.planes[1];
            let cr = planes.planes[2];
            for row in 0..(height as usize) / 2 {
                cb.data
                    .cast::<u8>()
                    .add(row * cb.rowStride as usize)
                    .write_bytes(chroma.0, (width as usize) / 2);
                cr.data
                    .cast::<u8>()
                    .add(row * cr.rowStride as usize)
                    .write_bytes(chroma.1, (width as usize) / 2);
            }
        } else {
            // Interleaved UV: pairs of (U, V).
            let uv = planes.planes[1];
            let uv_base = uv.data.cast::<u8>();
            for row in 0..(height as usize) / 2 {
                for col in 0..(width as usize) / 2 {
                    *uv_base.add(row * uv.rowStride as usize + col * uv.pixelStride as usize) =
                        chroma.0;
                    *uv_base.add(row * uv.rowStride as usize + col * uv.pixelStride as usize + 1) =
                        chroma.1;
                }
            }
        }
        ndk_sys::AHardwareBuffer_unlock(buffer, std::ptr::null_mut());
    }
    buffer
}

/// A producer sync chain: a binary semaphore whose `SYNC_FD` export is
/// the frame's fence payload. `delay_ms` > 0 inserts a host-signalled
/// timeline semaphore ahead of the binary signal, so the fence completes
/// only once the delayed host signal lands — all on the GPU.
///
/// The spec requires the semaphore to have a *pending* signal — a signal
/// submitted to a queue — before `SYNC_FD` export transplants it into the
/// fence; exporting a never-submitted semaphore is invalid use and the
/// Mali driver answers `ERROR_OUT_OF_HOST_MEMORY`. `fire` therefore
/// submits first and exports after.
struct ProducerFence {
    /// The sync-file fd the frame waits on, filled by `fire`.
    fd: Mutex<Option<OwnedFd>>,
    /// The host-signalled timeline semaphore delaying the producer
    /// (kept alive until `fire`).
    timeline: vk::Semaphore,
    /// The binary semaphore the producer submit signals.
    binary: vk::Semaphore,
    /// The `VK_KHR_external_semaphore_fd` device-level functions.
    loader: ash::khr::external_semaphore_fd::Device,
    /// The device that owns both semaphores.
    dev: ash::Device,
}

impl ProducerFence {
    /// Arms the fence; `fire(delay_ms)` completes it.
    fn new(device: &vulkan::Device) -> Self {
        let dev = device.shared.vk.device.clone();
        let loader = device
            .shared
            .vk
            .external_semaphore_fd
            .clone()
            .expect("sync-fd support checked at setup");
        let binary = unsafe { dev.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
            .expect("binary semaphore");
        let mut type_info =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let timeline = unsafe {
            dev.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut type_info),
                None,
            )
        }
        .expect("timeline semaphore");
        Self {
            fd: Mutex::new(None),
            timeline,
            binary,
            loader,
            dev,
        }
    }

    /// The exported sync-file fd; `fire` must have run first.
    fn fd(&self) -> OwnedFd {
        let guard = self.fd.lock().expect("fence fd");
        let fd = guard.as_ref().expect("fence exported by fire");
        unsafe { OwnedFd::from_raw_fd(libc::dup(fd.as_raw_fd())) }
    }

    /// Schedules the producer's signal: a submit that waits on the
    /// host-signalled timeline and then signals the binary payload —
    /// the fence completes `delay_ms` from now, entirely on the GPU.
    /// The `SYNC_FD` export runs after the submit so the semaphore has a
    /// pending signal, as the spec requires.
    fn fire(&self, queue: vk::Queue, family: u32, delay_ms: u64) {
        let dev = self.dev.clone();
        let timeline = self.timeline;
        let binary = self.binary;
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(delay_ms));
            unsafe {
                dev.signal_semaphore(
                    &vk::SemaphoreSignalInfo::default()
                        .semaphore(timeline)
                        .value(1),
                )
                .expect("host timeline signal");
            }
        });
        let dev = self.dev.clone();
        let timeline = self.timeline;
        run_once(&self.dev, queue, family, |_| {});
        // The signalling submit must run AFTER the timeline signal is
        // scheduled but is itself a GPU wait on it.
        let pool = unsafe {
            dev.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                None,
            )
        }
        .expect("pool");
        let cb = unsafe {
            dev.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .expect("cb")[0];
        unsafe {
            dev.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
                .expect("begin");
            dev.end_command_buffer(cb).expect("end");
            let mut timeline_info =
                vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&[1]);
            dev.queue_submit(
                queue,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&[timeline])
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::ALL_COMMANDS])
                    .signal_semaphores(&[binary])
                    .command_buffers(&[cb])
                    .push_next(&mut timeline_info)],
                vk::Fence::null(),
            )
            .expect("producer submit");
            dev.destroy_command_pool(pool, None);
            // The signal is pending now: the SYNC_FD export transplants it
            // into the returned fence.
            let fd = self
                .loader
                .get_semaphore_fd(
                    &vk::SemaphoreGetFdInfoKHR::default()
                        .semaphore(binary)
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD),
                )
                .expect("sync-fd export");
            *self.fd.lock().expect("fence fd") = Some(OwnedFd::from_raw_fd(fd));
        }
    }
}

impl Drop for ProducerFence {
    fn drop(&mut self) {
        unsafe {
            self.dev.destroy_semaphore(self.timeline, None);
            self.dev.destroy_semaphore(self.binary, None);
        }
    }
}

/// A sendable ash device handle for the signal thread.
#[expect(dead_code)]
struct SendDevice(ash::Device);
unsafe impl Send for SendDevice {}

/// Presents `texture` into a fresh destination and returns its f32
/// pixels (the same helper pattern as `host_contracts`).
fn read_pixels(
    engine: &Engine<Gpu>,
    shared: &SharedDevice,
    output: &wgpu::Texture,
) -> Result<Vec<[f32; 4]>, Box<dyn std::error::Error>> {
    let (target, destinations) = TextureTarget::new((16, 16));
    let destination = engine.surface(target)?;
    let destination_texture = destinations.try_recv()?;
    let delivery = cherenkov_gpu::interop::shader_delivery(wgpu::Backend::Vulkan, &shared.device)?;
    let mut presenter = Presenter::new(&shared.device, delivery);
    presenter.texture(
        &shared.device,
        &shared.queue,
        &output.create_view(&wgpu::TextureViewDescriptor::default()),
        TextureOutput {
            texture: &destination_texture,
            color: OutputColor::LinearDisplayP3,
            alpha: OutputAlpha::Premultiplied,
            headroom: 1.0,
        },
    );
    Ok(destination.readback()?.pixels)
}

/// The count of open fds in this process.
fn fd_count() -> usize {
    std::fs::read_dir("/proc/self/fd").map_or(0, std::iter::Iterator::count)
}

#[test]
fn ahb_rgb_import_decodes_known_pixels() {
    let (shared, device) = setup();
    let buffer = make_ahb_rgb(16, 16, [0xe0, 0x40, 0x20, 0xff]);
    let frame = device
        .import(FrameSource::Ahb(Box::new(vulkan::Ahb {
            buffer: buffer.cast(),
            sync: None,
            release: None,
            color: FrameColor::SRGB,
            alpha: RgbAlpha::Opaque,
        })))
        .expect("RGB AHB import");
    eprintln!(
        "RGB frame: size {:?} repr {:?} bytes {}",
        frame.size(),
        frame.repr(),
        frame.imported_bytes()
    );
    let engine = Engine::<Gpu>::new(GpuConfig {
        device: Some(shared.clone()),
        ..GpuConfig::default()
    })
    .expect("engine");
    let (target, textures) = TextureTarget::new((16, 16));
    let surface = engine.surface(target).expect("surface");
    let output = textures.try_recv().expect("output texture");
    let layer = surface.layer();
    let handle = engine.external_frame(ExternalFrame::native(frame).expect("external"));
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
        tx[&layer].content(handle);
    });
    assert!(matches!(engine.render(FrameTime::now()), Ok(Next::Idle)));
    let pixels = read_pixels(&engine, &shared, &output).expect("readback");
    let pixel = pixels[8 * 16 + 8];
    assert!(
        pixel[0] > 0.5 && pixel[1] < 0.35 && pixel[2] < 0.3 && pixel[3] > 0.99,
        "AHB RGB decode: {pixel:?}"
    );
}

#[test]
fn ahb_yuv_external_format_decodes_neutral() {
    let (shared, device) = setup();
    // Grey content (luma 0x50, neutral chroma) must survive the
    // conversion path to a flat grey — SDR, real external format.
    let buffer = make_ahb_nv12(16, 16, 0x50, (0x80, 0x80));
    let frame = device
        .import(FrameSource::Ahb(Box::new(vulkan::Ahb {
            buffer: buffer.cast(),
            sync: None,
            release: None,
            color: FrameColor::BT709_VIDEO,
            alpha: RgbAlpha::Opaque,
        })))
        .expect("NV12 AHB import");
    eprintln!(
        "YUV frame: size {:?} repr {:?} bytes {}",
        frame.size(),
        frame.repr(),
        frame.imported_bytes()
    );
    match frame.repr() {
        vulkan::Repr::ExternalFormat { id } => {
            eprintln!("external-format representation, id {id}");
        }
        repr => eprintln!("known-format representation: {repr:?}"),
    }
    let engine = Engine::<Gpu>::new(GpuConfig {
        device: Some(shared.clone()),
        ..GpuConfig::default()
    })
    .expect("engine");
    let (target, textures) = TextureTarget::new((16, 16));
    let surface = engine.surface(target).expect("surface");
    let output = textures.try_recv().expect("output texture");
    let layer = surface.layer();
    let handle = engine.external_frame(ExternalFrame::native(frame).expect("external"));
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
        tx[&layer].content(handle);
    });
    assert!(matches!(engine.render(FrameTime::now()), Ok(Next::Idle)));
    let pixels = read_pixels(&engine, &shared, &output).expect("readback");
    let pixel = pixels[8 * 16 + 8];
    let max_dev = pixel[..3]
        .iter()
        .fold(0.0f32, |d, c| d.max((c - pixel[0]).abs()));
    assert!(
        pixel[0] > 0.02 && max_dev < 0.08 && pixel[3] > 0.99,
        "YUV neutral chroma decodes to opaque grey: {pixel:?}"
    );
}

#[test]
fn producer_fence_and_delayed_signal_stay_on_gpu() {
    let (shared, device) = setup();
    let (dev, queue, family) = raw(&shared);
    let _ = dev;
    let fence = ProducerFence::new(&device);
    // The fence completes 200 ms from now — fired before the import so
    // the SYNC_FD export finds a pending signal, but the fence itself is
    // still unsigned when the engine consumes it below.
    fence.fire(queue, family, 200);
    let buffer = make_ahb_rgb(16, 16, [0x20, 0x90, 0x30, 0xff]);
    let frame = device
        .import(FrameSource::Ahb(Box::new(vulkan::Ahb {
            buffer: buffer.cast(),
            // The real producer fence: a sync_file fd signalling on the
            // GPU once the producer's submit completes. The import takes
            // ownership of the fd it is given — `fence.fd()` returns a
            // duplicate so the fence keeps its own reference.
            sync: Some(vulkan::Wait::SyncFd { fd: fence.fd() }),
            release: Some(vulkan::ReleaseSync::Timeline {
                semaphore: fence.timeline.as_raw(),
                value: 2,
            }),
            color: FrameColor::SRGB,
            alpha: RgbAlpha::Opaque,
        })))
        .expect("fenced AHB import");
    let engine = Engine::<Gpu>::new(GpuConfig {
        device: Some(shared),
        ..GpuConfig::default()
    })
    .expect("engine");
    let (target, _) = TextureTarget::new((16, 16));
    let surface = engine.surface(target).expect("surface");
    let layer = surface.layer();
    let handle = engine.external_frame(ExternalFrame::native(frame).expect("external"));
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
        tx[&layer].content(handle);
    });
    // The fence is still unsigned: the consuming submission must return
    // immediately — a CPU wait would block for the whole delay.
    let start = Instant::now();
    assert!(matches!(engine.render(FrameTime::now()), Ok(Next::Idle)));
    assert!(
        start.elapsed() < Duration::from_millis(100),
        "submission CPU-waited on the producer fence: {:?}",
        start.elapsed()
    );
}

#[test]
fn two_layers_replace_retire_and_release_fence() {
    let (shared, device) = setup();
    let buffer = make_ahb_rgb(16, 16, [0x30, 0x30, 0xa0, 0xff]);
    let frame = device
        .import(FrameSource::Ahb(Box::new(vulkan::Ahb {
            buffer: buffer.cast(),
            sync: None,
            release: Some(vulkan::ReleaseSync::FenceFd),
            color: FrameColor::SRGB,
            alpha: RgbAlpha::Opaque,
        })))
        .expect("AHB import");
    let engine = Engine::<Gpu>::new(GpuConfig {
        device: Some(shared),
        ..GpuConfig::default()
    })
    .expect("engine");
    let (target, textures) = TextureTarget::new((16, 16));
    let surface = engine.surface(target).expect("surface");
    textures.try_recv().expect("output texture");
    let (a, b) = (surface.layer(), surface.layer());
    // Two attachments of one generation deduplicate the acquisition.
    let (ha, hb) = (
        engine.external_frame(ExternalFrame::native(frame.clone()).expect("a")),
        engine.external_frame(ExternalFrame::native(frame.clone()).expect("b")),
    );
    surface.update(|tx| {
        tx[surface.root()].push(&a);
        tx[surface.root()].push(&b);
        tx[&a].content(ha);
        tx[&b].content(hb);
    });
    for _ in 0..3 {
        assert!(matches!(engine.render(FrameTime::now()), Ok(Next::Idle)));
    }
    assert_eq!(frame.generation.lease_count(), 2);
    // Replace one attachment, then drop the other — the last retire
    // submits the release and exports the fence.
    surface.update(|tx| {
        tx[surface.root()].remove(&a);
    });
    drop(a);
    assert!(matches!(engine.render(FrameTime::now()), Ok(Next::Idle)));
    surface.update(|tx| {
        tx[surface.root()].remove(&b);
    });
    drop(b);
    assert!(matches!(engine.render(FrameTime::now()), Ok(Next::Idle)));
    let fd = frame.release_fd().expect("release fence after submission");
    // The fence signals once the release submission executes; a sync-file
    // fd becomes readable at signal — `poll` is the producer's wait, never
    // the engine's.
    let mut pfd = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    let rc = unsafe { libc::poll(&raw mut pfd, 1, 2000) };
    assert_eq!(rc, 1, "release fence did not signal");
}

#[test]
fn native_op_survives_engine_buffer_and_atlas_regrowth() {
    let (shared, device) = setup();
    let buffer = make_ahb_rgb(16, 16, [0xe0, 0x40, 0x20, 0xff]);
    let frame = device
        .import(FrameSource::Ahb(Box::new(vulkan::Ahb {
            buffer: buffer.cast(),
            sync: None,
            release: None,
            color: FrameColor::SRGB,
            alpha: RgbAlpha::Opaque,
        })))
        .expect("AHB import");
    let engine = Engine::<Gpu>::new(GpuConfig {
        device: Some(shared.clone()),
        ..GpuConfig::default()
    })
    .expect("engine");
    let (target, textures) = TextureTarget::new((16, 16));
    let surface = engine.surface(target).expect("surface");
    let output = textures.try_recv().expect("output texture");
    let layer = surface.layer();
    let handle = engine.external_frame(ExternalFrame::native(frame).expect("external"));
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
        tx[&layer].content(handle);
    });
    assert!(matches!(engine.render(FrameTime::now()), Ok(Next::Idle)));
    // Regrow the engine's per-draw buffers: enough fills to exceed the
    // initial instance/stops sizing replaces the globals, instances and
    // stops buffers the native op's shared set 0 keys on, so the cached
    // set is rewritten, never allocated a second time.
    let fills = surface.layer();
    surface.update(|tx| {
        // Beneath the external layer: detach, append fills, re-attach.
        tx[surface.root()].remove(&layer);
        tx[surface.root()].push(&fills);
        tx[surface.root()].push(&layer);
        tx[&fills].content(surface.record(|r| {
            for i in 0..40 {
                r.fill(
                    Rect::new(
                        f64::from(i) * 0.2,
                        0.0,
                        f64::from(i).mul_add(0.2, 1.0),
                        16.0,
                    ),
                    WorkingColor::new([0.1, 0.2, 0.3, 1.0]),
                );
            }
        }));
    });
    assert!(matches!(engine.render(FrameTime::now()), Ok(Next::Idle)));
    let pixels = read_pixels(&engine, &shared, &output).expect("readback after regrow");
    let pixel = pixels[8 * 16 + 8];
    assert!(
        pixel[0] > 0.5 && pixel[3] > 0.99,
        "native op after buffer regrow: {pixel:?}"
    );
    // Regrow the atlas: fresh path cells past the starting allocation
    // replace the mask texture the per-generation set-1 keys on — the
    // generation-number invalidation must catch the recycled view.
    surface.update(|tx| {
        tx[&fills].content(surface.record(|r| {
            for i in 0..144u32 {
                let mut path = BezPath::new();
                let cx = 0.5 + f64::from(i % 8);
                let cy = 0.5 + f64::from(i / 8);
                for point in 0..5u32 {
                    let angle = f64::from(point)
                        .mul_add(144.0 + f64::from(i), -90.0)
                        .to_radians();
                    let p = (angle.cos().mul_add(0.4, cx), angle.sin().mul_add(0.4, cy));
                    if point == 0 {
                        path.move_to(p);
                    } else {
                        path.line_to(p);
                    }
                }
                path.close_path();
                r.fill(path, WorkingColor::new([0.8, 0.3, 0.1, 1.0]));
            }
        }));
    });
    assert!(matches!(engine.render(FrameTime::now()), Ok(Next::Idle)));
    let pixels = read_pixels(&engine, &shared, &output).expect("readback after atlas grow");
    let pixel = pixels[8 * 16 + 8];
    assert!(
        pixel[0] > 0.5 && pixel[3] > 0.99,
        "native op after atlas regrow: {pixel:?}"
    );
}

#[test]
fn next_generation_after_release() {
    let (shared, device) = setup();
    // A reused producer buffer is a new generation with new
    // synchronization, never a mutation of the retired frame.
    let buffer1 = make_ahb_rgb(16, 16, [0xa0, 0x30, 0x30, 0xff]);
    let frame1 = device
        .import(FrameSource::Ahb(Box::new(vulkan::Ahb {
            buffer: buffer1.cast(),
            sync: None,
            release: None,
            color: FrameColor::SRGB,
            alpha: RgbAlpha::Opaque,
        })))
        .expect("gen1");
    let buffer2 = make_ahb_rgb(16, 16, [0x30, 0xa0, 0x30, 0xff]);
    let frame2 = device
        .import(FrameSource::Ahb(Box::new(vulkan::Ahb {
            buffer: buffer2.cast(),
            sync: None,
            release: None,
            color: FrameColor::SRGB,
            alpha: RgbAlpha::Opaque,
        })))
        .expect("gen2");
    assert!(!std::sync::Arc::ptr_eq(
        &frame1.generation,
        &frame2.generation
    ));
    let engine = Engine::<Gpu>::new(GpuConfig {
        device: Some(shared.clone()),
        ..GpuConfig::default()
    })
    .expect("engine");
    let (target, textures) = TextureTarget::new((16, 16));
    let surface = engine.surface(target).expect("surface");
    let output = textures.try_recv().expect("output texture");
    let layer = surface.layer();
    let h2 = engine.external_frame(ExternalFrame::native(frame2).expect("external"));
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
        tx[&layer].content(h2);
    });
    assert!(matches!(engine.render(FrameTime::now()), Ok(Next::Idle)));
    let pixels = read_pixels(&engine, &shared, &output).expect("readback");
    let pixel = pixels[8 * 16 + 8];
    assert!(
        pixel[1] > 0.3 && pixel[0] < 0.3,
        "next generation reads its own content: {pixel:?}"
    );
}

#[test]
fn cancellation_and_teardown() {
    let (shared, device) = setup();
    let buffer = make_ahb_rgb(16, 16, [0x60, 0x60, 0x60, 0xff]);
    let frame = device
        .import(FrameSource::Ahb(Box::new(vulkan::Ahb {
            buffer: buffer.cast(),
            sync: None,
            release: None,
            color: FrameColor::SRGB,
            alpha: RgbAlpha::Opaque,
        })))
        .expect("AHB import");
    let generation = frame.generation.clone();
    let (dev, _, family) = raw(&shared);
    let pool = unsafe {
        dev.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(family),
            None,
        )
    }
    .expect("pool");
    let cb = unsafe {
        dev.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .expect("cb")[0];
    unsafe {
        dev.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
            .expect("begin");
    }
    let mut native = vulkan::Native::new(device.shared).expect("native");
    if let Some(pending) = unsafe { vulkan::stage_acquire(&generation, cb) }.expect("stage") {
        native.staged.push(pending);
    }
    vulkan::cancel_staged(&mut native);
    unsafe { dev.destroy_command_pool(pool, None) };
    assert_eq!(
        generation.state(),
        vulkan::State::Registered,
        "a cancelled plan leaves the frame unacquired"
    );
    // Engine teardown with the frame still held must not leak or hang:
    // the engine's final flush retires it.
    let engine = Engine::<Gpu>::new(GpuConfig {
        device: Some(shared),
        ..GpuConfig::default()
    })
    .expect("engine");
    let (target, _) = TextureTarget::new((16, 16));
    let surface = engine.surface(target).expect("surface");
    let layer = surface.layer();
    let handle = engine.external_frame(ExternalFrame::native(frame).expect("external"));
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
        tx[&layer].content(handle);
    });
    assert!(matches!(engine.render(FrameTime::now()), Ok(Next::Idle)));
    drop(engine);
}

#[test]
fn report_counts_and_timings() {
    let (shared, device) = setup();
    let before = fd_count();
    let import_start = Instant::now();
    let buffer = make_ahb_rgb(16, 16, [0x44, 0x44, 0x44, 0xff]);
    let frame = device
        .import(FrameSource::Ahb(Box::new(vulkan::Ahb {
            buffer: buffer.cast(),
            sync: None,
            release: None,
            color: FrameColor::SRGB,
            alpha: RgbAlpha::Opaque,
        })))
        .expect("AHB import");
    let import_us = import_start.elapsed().as_micros();
    let engine = Engine::<Gpu>::new(GpuConfig {
        device: Some(shared),
        ..GpuConfig::default()
    })
    .expect("engine");
    let (target, textures) = TextureTarget::new((16, 16));
    let surface = engine.surface(target).expect("surface");
    textures.try_recv().expect("output texture");
    let layer = surface.layer();
    let imported = frame.imported_bytes();
    let handle = engine.external_frame(ExternalFrame::native(frame).expect("external"));
    surface.update(|tx| {
        tx[surface.root()].push(&layer);
        tx[&layer].content(handle);
    });
    // Cold vs warm: the first render prepares pipeline state; the second
    // is steady-state.
    let cold = Instant::now();
    assert!(matches!(engine.render(FrameTime::now()), Ok(Next::Idle)));
    let cold_ms = cold.elapsed().as_secs_f64() * 1e3;
    let mut warm = Vec::new();
    for _ in 0..8 {
        let t = Instant::now();
        assert!(matches!(engine.render(FrameTime::now()), Ok(Next::Idle)));
        warm.push(t.elapsed().as_secs_f64() * 1e3);
    }
    warm.sort_by(f64::total_cmp);
    let after = fd_count();
    eprintln!("== #166 Pixel report ==");
    eprintln!("import: {import_us}us  bytes: {imported}");
    eprintln!("first render (cold): {cold_ms:.2}ms");
    eprintln!(
        "steady renders ms: p50={:.2} p99={:.2}",
        warm[warm.len() / 2],
        warm[warm.len() - 1]
    );
    eprintln!("fds: before={before} after={after}");
    eprintln!("memory: {:?}", engine.memory());
    drop(engine);
    let released = fd_count();
    eprintln!("fds after teardown: {released}");
    assert!(released <= before + 1, "leaked fds: {before} -> {released}");
}
