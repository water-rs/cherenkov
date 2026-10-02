//! Immutable, colour-tagged `IOSurface`s for recorded layer pixels.

use cherenkov::RenderError;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_foundation::{CFDictionary, CFRetained, CFString, CFType};
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferCreate, CVPixelBufferGetIOSurface,
    kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferMetalCompatibilityKey,
    kCVPixelFormatType_64RGBAHalf, kCVReturnSuccess,
};
use objc2_io_surface::IOSurfaceRef;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLDevice, MTLPixelFormat, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage,
};
use objc2_quartz_core::CALayer;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub(super) struct Buffer {
    pub texture: wgpu::Texture,
    pub surface: CFRetained<IOSurfaceRef>,
    pub generation: u64,
    pub ready: Arc<AtomicBool>,
    pub headroom: f32,
}

impl Buffer {
    pub fn new(
        device: &wgpu::Device,
        size: (u32, u32),
        generation: u64,
        headroom: f32,
    ) -> Result<Self, RenderError> {
        let empty = CFDictionary::<CFString, CFType>::from_slices(&[], &[]);
        // SAFETY: immutable framework keys and the CFBoolean singleton.
        let attributes = unsafe {
            CFDictionary::<CFString, CFType>::from_slices(
                &[
                    kCVPixelBufferIOSurfacePropertiesKey,
                    kCVPixelBufferMetalCompatibilityKey,
                ],
                &[
                    &empty,
                    objc2_core_foundation::kCFBooleanTrue.expect("CFBoolean"),
                ],
            )
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: valid attributes, and an out pointer for the retained buffer.
        let status = unsafe {
            CVPixelBufferCreate(
                None,
                size.0 as usize,
                size.1 as usize,
                kCVPixelFormatType_64RGBAHalf,
                Some(attributes.as_opaque()),
                std::ptr::NonNull::from(&mut raw),
            )
        };
        if status != kCVReturnSuccess {
            return Err(RenderError::Render(format!(
                "static plane pixel buffer: {status}"
            )));
        }
        // SAFETY: success transfers one reference to the caller.
        let buffer = unsafe {
            CFRetained::<CVPixelBuffer>::from_raw(
                std::ptr::NonNull::new(raw).expect("successful pixel-buffer allocation"),
            )
        };
        let surface =
            CVPixelBufferGetIOSurface(Some(&buffer)).expect("IOSurface-backed pixel buffer");
        // SAFETY: the colour-space name and IOSurface key are immutable statics.
        let space = objc2_core_graphics::CGColorSpace::with_name(Some(unsafe {
            objc2_core_graphics::kCGColorSpaceExtendedLinearDisplayP3
        }))
        .expect("extended linear P3");
        let property = space.property_list().expect("serializable colour space");
        unsafe { surface.set_value(objc2_io_surface::kIOSurfaceColorSpace, &property) };
        // SAFETY: Metal is the Apple backend and the descriptor is fully specified.
        let metal = unsafe { device.as_hal::<wgpu::hal::metal::Api>() }.expect("Metal device");
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA16Float,
                size.0 as usize,
                size.1 as usize,
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let raw = metal
            .raw_device()
            .newTextureWithDescriptor_iosurface_plane(&descriptor, &surface, 0)
            .ok_or_else(|| RenderError::Render("Metal refused a static plane IOSurface".into()))?;
        let extent = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        };
        // SAFETY: this texture belongs to the engine's device, has the declared
        // format and extent, and wgpu retains it through every GPU submission.
        let texture = unsafe {
            let texture = wgpu::hal::metal::Device::texture_from_raw(
                raw,
                wgpu::TextureFormat::Rgba16Float,
                MTLTextureType::Type2D,
                1,
                1,
                extent.into(),
                None,
            );
            device.create_texture_from_hal::<wgpu::hal::metal::Api>(
                texture,
                &wgpu::TextureDescriptor {
                    label: Some("static plane IOSurface"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::wgt::TextureUses::UNINITIALIZED,
            )
        };
        Ok(Self {
            texture,
            surface,
            generation,
            headroom,
            ready: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn completed(
        &self,
        queue: &wgpu::Queue,
        waker: cherenkov::CompletionWaker,
    ) -> Result<(), RenderError> {
        // wgpu completion callbacks require a subsequent submit or device
        // poll. This surface can now be idle: a native queue completion
        // must wake it without another engine frame. The empty command
        // buffer follows the conversion on the very same Metal queue.
        // SAFETY: no resource or queue state is modified outside wgpu;
        // the marker only observes completion of preceding submissions.
        let metal = unsafe { queue.as_hal::<wgpu::hal::metal::Api>() }.expect("Metal queue");
        let marker = metal.as_raw().commandBuffer().ok_or_else(|| {
            RenderError::Render("Metal refused a capture completion marker".into())
        })?;
        let ready = Arc::clone(&self.ready);
        let block = block2::RcBlock::new(
            move |_: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                ready.store(true, Ordering::Release);
                waker.wake();
            },
        );
        // SAFETY: Metal retains the callback until completion. Its captured
        // state is thread safe and it never accesses a native layer.
        unsafe { marker.addCompletedHandler(block2::RcBlock::as_ptr(&block)) };
        marker.commit();
        Ok(())
    }

    pub fn bytes(&self) -> u64 {
        self.surface.alloc_size() as u64
    }
}

/// `IOSurface` is reference counted and its immutable contents may cross threads.
pub(super) struct Contents(pub CFRetained<IOSurfaceRef>);
// SAFETY: the reference holds an immutable IOSurface, shown only after its GPU fence.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "IOSurface is thread safe; only immutable GPU-completed contents cross to main"
)]
unsafe impl Send for Contents {}

impl Contents {
    pub fn set(self, layer: &CALayer) {
        // SAFETY: CALayer accepts an IOSurface as contents and retains it.
        let value: &AnyObject = unsafe { &*CFRetained::as_ptr(&self.0).as_ptr().cast() };
        unsafe { layer.setContents(Some(value)) };
        layer.setPreferredDynamicRange(unsafe { objc2_quartz_core::CADynamicRangeHigh });
    }
}
