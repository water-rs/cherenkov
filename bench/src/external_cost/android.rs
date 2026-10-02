//! `AHardwareBuffer` producer: a ring of GPU-sampleable AHBs, CPU-filled
//! per frame — the Android video-decode output model, the same buffer
//! kind `gpu/tests/vulkan_external_android.rs` imports.

use super::{BenchError, ExternalFrame, FrameColor, RING, Ramps, SharedDevice, Spec, wgpu};

use cherenkov_gpu::interop::{HdrMetadata, RgbAlpha, vulkan};

/// The producer ring.
pub struct Producer {
    buffers: Vec<std::ptr::NonNull<ndk_sys::AHardwareBuffer>>,
    /// The engine's native Vulkan device — per-frame AHB import.
    vulkan: vulkan::Device,
    ramps: Ramps,
    spec: Spec,
}

/// One allocated `AHardwareBuffer` of `spec`'s format — the usage set
/// `gpu/tests/vulkan_external_android.rs::alloc_ahb` asks for, plus CPU
/// read for the path-`c` copy.
fn alloc_ahb(spec: &Spec) -> Result<std::ptr::NonNull<ndk_sys::AHardwareBuffer>, BenchError> {
    let usage = ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE.0
        | ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN.0
        | ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN.0;
    let format = if spec.bits == 8 {
        ndk_sys::AHardwareBuffer_Format::AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420.0
    } else {
        ndk_sys::AHardwareBuffer_Format::AHARDWAREBUFFER_FORMAT_YCbCr_P010.0
    };
    let desc = ndk_sys::AHardwareBuffer_Desc {
        width: spec.width,
        height: spec.height,
        layers: 1,
        format,
        usage,
        stride: 0,
        rfu0: 0,
        rfu1: 0,
    };
    // SAFETY: `desc` is fully initialized.
    if unsafe { ndk_sys::AHardwareBuffer_isSupported(&raw const desc) } == 0 {
        return Err(BenchError::Engine(format!(
            "external-cost: AHB format {format:#x} unsupported at {}x{}",
            spec.width, spec.height
        )));
    }
    let mut buffer = std::ptr::null_mut();
    // SAFETY: `buffer` receives the allocated buffer on success.
    if unsafe { ndk_sys::AHardwareBuffer_allocate(&raw const desc, &raw mut buffer) } != 0
        || buffer.is_null()
    {
        return Err(BenchError::Engine(
            "external-cost: AHardwareBuffer_allocate failed".into(),
        ));
    }
    Ok(std::ptr::NonNull::new(buffer).expect("non-null after check"))
}

/// The locked planes of `buffer` — `body` runs under one lock.
fn planes(
    buffer: *mut ndk_sys::AHardwareBuffer,
    write: bool,
    body: impl Fn(&ndk_sys::AHardwareBuffer_Planes),
) {
    let usage = if write {
        ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN.0
    } else {
        ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN.0
    };
    let mut planes = ndk_sys::AHardwareBuffer_Planes {
        planeCount: 0,
        planes: [ndk_sys::AHardwareBuffer_Plane {
            data: std::ptr::null_mut(),
            pixelStride: 0,
            rowStride: 0,
        }; 4],
    };
    // SAFETY: `planes` is fully initialized and `buffer` is live.
    let rc = unsafe {
        ndk_sys::AHardwareBuffer_lockPlanes(
            buffer,
            u64::from(usage),
            -1,
            std::ptr::null_mut(),
            &raw mut planes,
        )
    };
    assert_eq!(rc, 0, "external-cost: AHardwareBuffer_lockPlanes");
    body(&planes);
    // SAFETY: locked above.
    let _ = unsafe { ndk_sys::AHardwareBuffer_unlock(buffer, std::ptr::null_mut()) };
}

/// SAFETY invariants of the helpers below: every `data` pointer comes
/// from a live locked plane, `rowStride`/`pixelStride` describe its
/// layout, and the slice never crosses the plane's `rows * rowStride`.
unsafe fn plane_row<'a>(
    plane: &ndk_sys::AHardwareBuffer_Plane,
    row: usize,
    len: usize,
) -> &'a mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(
            plane.data.cast::<u8>().add(row * plane.rowStride as usize),
            len,
        )
    }
}

/// The packed interleaved `(cb, cr)` rows of a locked semi-planar or
/// tri-planar chroma buffer — `width / 2` pairs of `bytes` each.
///
/// Semi-planar AHBs expose the pair in plane 1 (`pixelStride == 2`
/// codes on plane 1, possibly split as pixelStride-2 planes 1/2
/// pointing at cb/cr of the same interleaved row). Tri-planar
/// (`pixelStride == 1`) is de-interleaved per pixel.
fn chroma_pairs(
    planes: &ndk_sys::AHardwareBuffer_Planes,
    spec: &Spec,
    row: usize,
    dst: &mut [u8],
) -> Result<(), BenchError> {
    let bytes = spec.code_bytes();
    let pair = 2 * bytes;
    let cols = spec.width.div_ceil(2) as usize;
    let p1 = planes.planes[1];
    if p1.pixelStride as usize == pair
        || (planes.planeCount == 3 && p1.pixelStride as usize == pair)
    {
        // Interleaved UV in plane 1.
        let len = cols * pair;
        // SAFETY: plane 1 row `row` holds `cols` packed pairs.
        dst[..len].copy_from_slice(unsafe { plane_row(&p1, row, len) });
        return Ok(());
    }
    if planes.planeCount == 3 && p1.pixelStride as usize == bytes {
        // Separate Cb/Cr planes.
        let p2 = planes.planes[2];
        let cb_row = unsafe { plane_row(&p1, row, p1.rowStride as usize) };
        let cr_row = unsafe { plane_row(&p2, row, p2.rowStride as usize) };
        for col in 0..cols {
            dst[col * pair..col * pair + bytes]
                .copy_from_slice(&cb_row[col * bytes..col * bytes + bytes]);
            dst[col * pair + bytes..col * pair + pair]
                .copy_from_slice(&cr_row[col * bytes..col * bytes + bytes]);
        }
        return Ok(());
    }
    Err(BenchError::Engine(format!(
        "external-cost: unhandled AHB chroma layout (planes {}, pixelStride {})",
        planes.planeCount, p1.pixelStride
    )))
}

impl Producer {
    /// The ring over `shared`'s Vulkan device.
    pub fn new(spec: &Spec, shared: &SharedDevice) -> Result<Self, BenchError> {
        let vulkan = vulkan::Device::new(shared)
            .map_err(|e| BenchError::Gpu(format!("external-cost vulkan device: {e}")))?;
        let buffers = (0..RING)
            .map(|_| alloc_ahb(spec))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            buffers,
            vulkan,
            ramps: Ramps::new(spec),
            spec: *spec,
        })
    }

    /// Writes frame `frame`'s gradient into buffer `frame % RING`.
    pub fn fill(&self, frame: u32) {
        let buffer = self.buffers[frame as usize % RING].as_ptr();
        let (ramps, spec) = (&self.ramps, &self.spec);
        planes(buffer, true, |planes| {
            assert!(planes.planeCount >= 2, "external-cost: 1-plane AHB");
            let luma = planes.planes[0];
            for row in 0..spec.height as usize {
                let dst = unsafe { plane_row(&luma, row, spec.width as usize * spec.code_bytes()) };
                ramps.luma_row(frame, u32::try_from(row).expect("row fits u32"), dst);
            }
            let pair = 2 * spec.code_bytes();
            let mut row_buf = vec![0u8; spec.width as usize * spec.code_bytes()];
            for row in 0..spec.height.div_ceil(2) as usize {
                ramps.chroma_row(
                    frame,
                    u32::try_from(row).expect("row fits u32"),
                    &mut row_buf,
                );
                let p1 = planes.planes[1];
                if p1.pixelStride as usize == pair {
                    let dst =
                        unsafe { plane_row(&p1, row, spec.width.div_ceil(2) as usize * pair) };
                    dst.copy_from_slice(&row_buf[..dst.len()]);
                } else if planes.planeCount == 3 && p1.pixelStride as usize == spec.code_bytes() {
                    let p2 = planes.planes[2];
                    for col in 0..spec.width.div_ceil(2) as usize {
                        let bytes = spec.code_bytes();
                        let cb = unsafe {
                            p1.data
                                .cast::<u8>()
                                .add(row * p1.rowStride as usize + col * p1.pixelStride as usize)
                        };
                        let cr = unsafe {
                            p2.data
                                .cast::<u8>()
                                .add(row * p2.rowStride as usize + col * p2.pixelStride as usize)
                        };
                        unsafe {
                            cb.copy_from(row_buf[col * pair..].as_ptr(), bytes);
                            cr.copy_from(row_buf[col * pair + bytes..].as_ptr(), bytes);
                        }
                    }
                } else {
                    panic!(
                        "external-cost: unhandled AHB chroma layout (planes {}, pixelStride {})",
                        planes.planeCount, p1.pixelStride
                    );
                }
            }
        });
    }

    /// Path `c`: copies the filled buffer's planes into `y`/`uv` —
    /// the semi-planar rows copy straight through `write_texture`;
    /// a tri-planar AHB is interleaved into `scratch` first, which is
    /// the same total bytes moved.
    pub fn upload(&self, frame: u32, queue: &wgpu::Queue, y: &wgpu::Texture, uv: &wgpu::Texture) {
        let buffer = self.buffers[frame as usize % RING].as_ptr();
        let spec = self.spec;
        planes(buffer, false, |planes| {
            let luma = planes.planes[0];
            // SAFETY: the locked luma plane is `rowStride * height`.
            let y_data = unsafe {
                std::slice::from_raw_parts(
                    luma.data.cast::<u8>(),
                    luma.rowStride as usize * spec.height as usize,
                )
            };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: y,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                y_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(luma.rowStride),
                    rows_per_image: Some(spec.height),
                },
                wgpu::Extent3d {
                    width: spec.width,
                    height: spec.height,
                    depth_or_array_layers: 1,
                },
            );
            let pair = 2 * spec.code_bytes();
            let cols = spec.width.div_ceil(2) as usize;
            let rows = spec.height.div_ceil(2) as usize;
            let p1 = planes.planes[1];
            if p1.pixelStride as usize == pair {
                // Interleaved UV plane: write it strided.
                let uv_data = unsafe {
                    std::slice::from_raw_parts(p1.data.cast::<u8>(), p1.rowStride as usize * rows)
                };
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: uv,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    uv_data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(p1.rowStride),
                        rows_per_image: Some(spec.height.div_ceil(2)),
                    },
                    wgpu::Extent3d {
                        width: u32::try_from(cols).expect("cols fit u32"),
                        height: spec.height.div_ceil(2),
                        depth_or_array_layers: 1,
                    },
                );
            } else {
                // Tri-planar: interleave row by row.
                let mut uv_data = vec![0u8; cols * pair * rows];
                for row in 0..rows {
                    chroma_pairs(planes, &spec, row, &mut uv_data[row * cols * pair..])
                        .expect("chroma layout checked at fill");
                }
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: uv,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &uv_data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(u32::try_from(cols * pair).expect("row bytes fit u32")),
                        rows_per_image: Some(spec.height.div_ceil(2)),
                    },
                    wgpu::Extent3d {
                        width: u32::try_from(cols).expect("cols fit u32"),
                        height: spec.height.div_ceil(2),
                        depth_or_array_layers: 1,
                    },
                );
            }
        });
    }

    /// Path `e`: the filled buffer imported as a Vulkan-native
    /// [`ExternalFrame`] — `gpu/tests/vulkan_external_android.rs`'s
    /// `FrameSource::Ahb` path.
    pub fn external(&self, frame: u32, color: FrameColor) -> Result<ExternalFrame, BenchError> {
        let buffer = self.buffers[frame as usize % RING];
        let native = self
            .vulkan
            .import(vulkan::FrameSource::Ahb(Box::new(vulkan::Ahb {
                buffer: buffer.as_ptr().cast(),
                sync: None,
                release: None,
                color,
                alpha: RgbAlpha::Opaque,
                hdr: HdrMetadata::default(),
            })))
            .map_err(|e| BenchError::Engine(format!("external-cost: AHB import failed: {e}")))?;
        ExternalFrame::native(native).map_err(|e| {
            BenchError::Engine(format!("external-cost: invalid native frame: {e:?}"))
        })
    }
}

impl Drop for Producer {
    fn drop(&mut self) {
        for buffer in self.buffers.drain(..) {
            // SAFETY: each buffer was allocated once and released once.
            unsafe { ndk_sys::AHardwareBuffer_release(buffer.as_ptr()) };
        }
    }
}
