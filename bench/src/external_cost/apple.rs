//! `CVPixelBuffer` producer: a ring of `IOSurface`-backed buffers,
//! CPU-filled per frame — the Apple video-decode output model.

use super::{BenchError, ExternalFrame, FrameColor, RING, Ramps, SharedDevice, Spec, wgpu};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::{CFDictionary, CFRetained, CFString, CFType};
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferCreate, CVPixelBufferGetBaseAddressOfPlane,
    CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferGetHeightOfPlane, CVPixelBufferGetIOSurface,
    CVPixelBufferGetWidthOfPlane, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
    CVPixelBufferUnlockBaseAddress, kCVPixelBufferIOSurfacePropertiesKey,
    kCVPixelBufferMetalCompatibilityKey, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
    kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, kCVReturnSuccess,
};
use objc2_metal::{MTLDevice, MTLPixelFormat, MTLTextureDescriptor, MTLTextureUsage};

/// The producer ring.
pub struct Producer {
    buffers: Vec<CFRetained<CVPixelBuffer>>,
    /// The raw `MTLDevice` the engine's `wgpu::Device` wraps — plane
    /// textures are created on it.
    mtl: Retained<ProtocolObject<dyn MTLDevice>>,
    device: wgpu::Device,
    ramps: Ramps,
    spec: Spec,
}

/// A Metal-compatible, `IOSurface`-backed pixel buffer — the same
/// construction `gpu/tests/planes.rs::surface_buffer` uses.
fn surface_buffer(spec: &Spec) -> Result<CFRetained<CVPixelBuffer>, BenchError> {
    let format = if spec.bits == 8 {
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
    } else {
        kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
    };
    let empty = CFDictionary::<CFString, CFType>::from_slices(&[], &[]);
    // SAFETY: CoreVideo's attribute keys and the boolean are immutable
    // statics.
    let attributes = unsafe {
        CFDictionary::<CFString, CFType>::from_slices(
            &[
                kCVPixelBufferIOSurfacePropertiesKey,
                kCVPixelBufferMetalCompatibilityKey,
            ],
            &[
                &empty,
                objc2_core_foundation::kCFBooleanTrue.expect("kCFBooleanTrue"),
            ],
        )
    };
    let mut out = std::ptr::null_mut();
    // SAFETY: `out` receives a +1 pixel buffer.
    let status = unsafe {
        CVPixelBufferCreate(
            None,
            spec.width as usize,
            spec.height as usize,
            format,
            Some(attributes.as_opaque()),
            std::ptr::NonNull::from(&mut out),
        )
    };
    if status != kCVReturnSuccess || out.is_null() {
        return Err(BenchError::Engine(format!(
            "CVPixelBufferCreate failed (CVReturn {status})"
        )));
    }
    // SAFETY: the create call returned a +1 pixel buffer.
    Ok(unsafe { CFRetained::from_raw(std::ptr::NonNull::new_unchecked(out)) })
}

/// Runs `body(plane, base, row_stride, rows)` under one lock.
fn planes(buffer: &CVPixelBuffer, body: impl Fn(usize, *mut u8, usize, usize)) {
    // SAFETY: locked once, unlocked below with the same flags.
    unsafe { CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(0)) };
    for plane in 0..2 {
        let base = CVPixelBufferGetBaseAddressOfPlane(buffer, plane).cast::<u8>();
        let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane);
        let rows = CVPixelBufferGetHeightOfPlane(buffer, plane);
        assert!(!base.is_null(), "locked plane has a base address");
        body(plane, base, stride, rows);
    }
    // SAFETY: locked above.
    unsafe { CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(0)) };
}

impl Producer {
    /// The ring on `shared`'s Metal device.
    pub fn new(spec: &Spec, shared: &SharedDevice) -> Result<Self, BenchError> {
        // SAFETY: the guard is dropped before the device.
        let mtl = unsafe { shared.device.as_hal::<wgpu::hal::metal::Api>() }
            .ok_or_else(|| BenchError::Gpu("external-cost: the engine device is not Metal".into()))?
            .raw_device()
            .clone();
        let buffers = (0..RING)
            .map(|_| surface_buffer(spec))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            buffers,
            mtl,
            device: shared.device.clone(),
            ramps: Ramps::new(spec),
            spec: *spec,
        })
    }

    /// Writes frame `frame`'s gradient into buffer `frame % RING`.
    pub fn fill(&self, frame: u32) {
        let buffer = &self.buffers[frame as usize % RING];
        let ramps = &self.ramps;
        planes(buffer, |plane, base, stride, rows| {
            for row in 0..rows {
                // SAFETY: the plane is locked and `rows * stride` bytes
                // long; `width * bytes <= stride`.
                let dst = unsafe { std::slice::from_raw_parts_mut(base.add(row * stride), stride) };
                let row = u32::try_from(row).expect("row index fits u32");
                if plane == 0 {
                    ramps.luma_row(frame, row, dst);
                } else {
                    ramps.chroma_row(frame, row, dst);
                }
            }
        });
    }

    /// Path `c`: copies the filled buffer's planes into `y`/`uv`
    /// through `Queue::write_texture` — the hand-off
    /// water-rs/video-gpu's software path runs
    /// (`codec/src/frame/gpu.rs` `write_biplanar`, called by
    /// `runtime_player.rs::upload_frame_texture`).
    pub fn upload(&self, frame: u32, queue: &wgpu::Queue, y: &wgpu::Texture, uv: &wgpu::Texture) {
        let buffer = &self.buffers[frame as usize % RING];
        let spec = self.spec;
        planes(buffer, |plane, base, stride, rows| {
            let texels = if plane == 0 {
                spec.width
            } else {
                spec.width.div_ceil(2)
            };
            // SAFETY: `stride * rows` bytes are locked and readable.
            let data = unsafe { std::slice::from_raw_parts(base.cast_const(), stride * rows) };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: if plane == 0 { y } else { uv },
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(u32::try_from(stride).expect("stride fits u32")),
                    rows_per_image: Some(u32::try_from(rows).expect("rows fit u32")),
                },
                wgpu::Extent3d {
                    width: texels,
                    height: u32::try_from(rows).expect("rows fit u32"),
                    depth_or_array_layers: 1,
                },
            );
        });
    }

    /// Plane `plane` of `buffer`'s `IOSurface` as a texture on the
    /// engine's device — `gpu/tests/planes.rs::plane_texture`.
    fn plane_texture(&self, buffer: &CVPixelBuffer, plane: usize) -> wgpu::Texture {
        let (mtl_format, format) = match (self.spec.bits, plane) {
            (8, 0) => (MTLPixelFormat::R8Uint, wgpu::TextureFormat::R8Uint),
            (8, _) => (MTLPixelFormat::RG8Uint, wgpu::TextureFormat::Rg8Uint),
            (_, 0) => (MTLPixelFormat::R16Uint, wgpu::TextureFormat::R16Uint),
            (_, _) => (MTLPixelFormat::RG16Uint, wgpu::TextureFormat::Rg16Uint),
        };
        let surface = CVPixelBufferGetIOSurface(Some(buffer)).expect("an IOSurface-backed buffer");
        // SAFETY: the descriptor is fully specified.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                mtl_format,
                CVPixelBufferGetWidthOfPlane(buffer, plane),
                CVPixelBufferGetHeightOfPlane(buffer, plane),
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::ShaderRead);
        let raw = self
            .mtl
            .newTextureWithDescriptor_iosurface_plane(&descriptor, &surface, plane)
            .expect("an IOSurface plane texture");
        // SAFETY: the texture is on the engine's device and holds
        // `format`.
        unsafe { cherenkov_gpu::interop::metal::import_texture(&self.device, raw, format) }
    }

    /// Path `e`: the filled buffer's planes wrapped as an
    /// [`ExternalFrame`].
    pub fn external(&self, frame: u32, color: FrameColor) -> Result<ExternalFrame, BenchError> {
        let buffer = &self.buffers[frame as usize % RING];
        ExternalFrame::yuv(
            self.plane_texture(buffer, 0),
            self.plane_texture(buffer, 1),
            color,
        )
        .map_err(|e| BenchError::Engine(format!("external-cost: invalid frame: {e:?}")))
    }
}
