//! Native-compatible composition images and their allocation ownership.

use std::sync::Arc;

use ash::vk;

use super::{NativeError, Shared};

/// A dedicated image allocation. Native submissions retain an `Arc` lease;
/// imported wgpu textures retain another through their destruction callback.
struct Allocation {
    shared: Arc<Shared>,
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    bytes: u64,
    lazy: bool,
    size: (u32, u32),
}

impl Drop for Allocation {
    fn drop(&mut self) {
        // SAFETY: the wrapper and every submitted native operation hold
        // leases, so the last lease retires only after the last use. Shared
        // keeps the Vulkan device alive through these calls.
        unsafe {
            let device = &self.shared.vk.device;
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Persistent backing whose wgpu handle tracks ordinary passes and transfers.
struct Persistent {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    allocation: Arc<Allocation>,
}

impl Allocation {
    /// Allocates one epoch-local RGBA16F input/color attachment. The caller
    /// discards its contents at epoch exit and retains it through completion.
    fn transient(shared: &Arc<Shared>, size: (u32, u32)) -> Result<Arc<Self>, NativeError> {
        Self::create(
            shared,
            size,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::INPUT_ATTACHMENT
                | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        )
    }

    fn create(
        shared: &Arc<Shared>,
        size: (u32, u32),
        usage: vk::ImageUsageFlags,
    ) -> Result<Arc<Self>, NativeError> {
        let device = &shared.vk.device;
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .extent(vk::Extent3D {
                width: size.0,
                height: size.1,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: the shared context owns this device; the image is not
        // exposed until allocation, binding, and view creation succeed.
        let image = unsafe { device.create_image(&info, None) }?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let properties = unsafe {
            shared
                .instance
                .get_physical_device_memory_properties(shared.physical_device)
        };
        let transient = usage.contains(vk::ImageUsageFlags::TRANSIENT_ATTACHMENT);
        let Some((memory_type, lazy)) =
            memory_type(&properties, requirements.memory_type_bits, transient)
        else {
            unsafe { device.destroy_image(image, None) };
            return Err(NativeError::Unsupported(
                "no device-local attachment memory type",
            ));
        };
        // Dedicated images are pooled by epoch layout, not reallocated per
        // frame. No host-visible allocation or staging copy is involved.
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let allocation_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut dedicated);
        let memory = match unsafe { device.allocate_memory(&allocation_info, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { device.destroy_image(image, None) };
                return Err(error.into());
            }
        };
        if let Err(error) = unsafe { device.bind_image_memory(image, memory, 0) } {
            unsafe {
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
            return Err(error.into());
        }
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(info.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let view = match unsafe { device.create_image_view(&view_info, None) } {
            Ok(view) => view,
            Err(error) => {
                unsafe {
                    device.destroy_image(image, None);
                    device.free_memory(memory, None);
                }
                return Err(error.into());
            }
        };
        Ok(Arc::new(Self {
            shared: Arc::clone(shared),
            image,
            view,
            memory,
            bytes: requirements.size,
            lazy,
            size,
        }))
    }
}

impl Persistent {
    /// Imports an INPUT_ATTACHMENT-capable image and records its first clear
    /// through wgpu. The caller submits `initialize` before native writes;
    /// this marks the image initialized in wgpu's tracking as well as Vulkan.
    fn new(
        shared: &Arc<Shared>,
        size: (u32, u32),
        initialize: &mut wgpu::CommandEncoder,
    ) -> Result<Self, NativeError> {
        let allocation = Allocation::create(
            shared,
            size,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::INPUT_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        )?;
        let extent = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        };
        let uses = wgpu::wgt::TextureUses::COLOR_TARGET
            | wgpu::wgt::TextureUses::RESOURCE
            | wgpu::wgt::TextureUses::COPY_SRC
            | wgpu::wgt::TextureUses::COPY_DST;
        let lease = Arc::clone(&allocation);
        // SAFETY: this image was created on the same device with a superset
        // of every declared use. Its allocation is retained by the drop
        // callback and native submission leases, separately from the wrapper.
        let texture = unsafe {
            let device = shared
                .wgpu
                .as_hal::<wgpu::hal::vulkan::Api>()
                .expect("Vulkan device");
            let imported = device.texture_from_raw(
                allocation.image,
                &wgpu::hal::TextureDescriptor {
                    label: Some("composition persistent"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: uses,
                    memory_flags: wgpu::hal::MemoryFlags::empty(),
                    view_formats: Vec::new(),
                },
                Some(Box::new(move || drop(lease))),
                wgpu::hal::vulkan::TextureMemory::External,
            );
            shared
                .wgpu
                .create_texture_from_hal::<wgpu::hal::vulkan::Api>(
                    imported,
                    &wgpu::TextureDescriptor {
                        label: Some("composition persistent"),
                        size: extent,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba16Float,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_SRC
                            | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                    wgpu::wgt::TextureUses::UNINITIALIZED,
                )
        };
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        {
            let _pass = initialize.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("initialize composition persistent"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..wgpu::RenderPassDescriptor::default()
            });
        }
        Ok(Self {
            texture,
            view,
            allocation,
        })
    }
}

/// Selects a compatible local type, preferring lazy memory only for transient
/// attachments. A backed transient is reported honestly through `lazy`.
fn memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    compatible: u32,
    transient: bool,
) -> Option<(u32, bool)> {
    let mut backed = None;
    for (index, memory) in properties.memory_types_as_slice().iter().enumerate() {
        let index = u32::try_from(index).expect("Vulkan memory type index");
        let flags = memory.property_flags;
        if compatible & (1 << index) == 0
            || !flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            || flags.contains(vk::MemoryPropertyFlags::PROTECTED)
        {
            continue;
        }
        let lazy = flags.contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED);
        if lazy && transient {
            return Some((index, true));
        }
        if !lazy && backed.is_none() {
            backed = Some((index, false));
        }
    }
    backed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "android")]
    #[expect(
        clippy::too_many_lines,
        reason = "the Android handoff test keeps allocation, native transition, wgpu suffix, and retirement in one scenario"
    )]
    fn pixel_native_allocation_and_wgpu_handoff() {
        let context = crate::interop::SharedDevice::create(&crate::GpuConfig::default())
            .expect("Pixel Vulkan device");
        let shared = super::super::shared_for(&context).expect("native context");
        assert!(shared.caps.attachment_access.local_read);
        assert!(shared.caps.attachment_access.raster_order);
        let size = (32, 4);
        let transient = Allocation::transient(&shared, size).expect("transient image");
        assert!(transient.lazy, "Pixel must expose compatible lazy memory");
        assert!(transient.bytes >= u64::from(size.0 * size.1) * 8);
        assert_eq!(transient.size, size);

        let mut prefix = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let persistent = Persistent::new(&shared, size, &mut prefix).expect("persistent image");
        assert!(!persistent.allocation.lazy);
        assert_eq!(persistent.view.texture(), &persistent.texture);
        let mut native = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let image = persistent.allocation.image;
        // SAFETY: prefix initializes this image as COLOR_ATTACHMENT_OPTIMAL;
        // this separate raw encoder changes it to transfer destination and
        // restores the precise wgpu handoff state with a memory dependency.
        unsafe {
            native.as_hal_mut::<wgpu::hal::vulkan::Api, _, _>(|encoder| {
                let cb = encoder.expect("Vulkan encoder").raw_handle();
                let device = &shared.vk.device;
                let barrier = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .subresource_range(range)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
                device.cmd_clear_color_image(
                    cb,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue {
                        float32: [2.0, 0.25, 0.5, 1.0],
                    },
                    &[range],
                );
                let barrier = barrier
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::COLOR_ATTACHMENT_READ
                            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    );
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            });
        }
        let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("native handoff readback"),
            size: 1024,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut suffix = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        suffix.copy_texture_to_buffer(
            persistent.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );
        let lease = Arc::clone(&persistent.allocation);
        let weak = Arc::downgrade(&lease);
        let submission = context
            .queue
            .submit([prefix.finish(), native.finish(), suffix.finish()]);
        context.queue.on_submitted_work_done(move || drop(lease));
        drop(persistent);
        let (send, receive) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                send.send(result).unwrap();
            });
        context
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .expect("readback completion");
        receive.recv().unwrap().expect("readback mapping");
        let mapped = buffer.slice(..).get_mapped_range().expect("mapped bytes");
        for pixel in mapped.as_chunks::<8>().0 {
            assert_eq!(*pixel, [0, 64, 0, 52, 0, 56, 0, 60]);
        }
        drop(mapped);
        buffer.unmap();
        context
            .device
            .poll(wgpu::PollType::Poll)
            .expect("retire resources");
        assert!(weak.upgrade().is_none(), "retired allocation was leaked");
    }

    #[test]
    fn memory_selection_respects_usage_and_image_compatibility() {
        let mut properties = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 3,
            ..vk::PhysicalDeviceMemoryProperties::default()
        };
        properties.memory_types[0].property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE;
        properties.memory_types[1].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        properties.memory_types[2].property_flags =
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::LAZILY_ALLOCATED;
        assert_eq!(memory_type(&properties, 0b111, true), Some((2, true)));
        assert_eq!(memory_type(&properties, 0b111, false), Some((1, false)));
        assert_eq!(memory_type(&properties, 0b011, true), Some((1, false)));
        assert_eq!(memory_type(&properties, 0b100, false), None);
        assert_eq!(memory_type(&properties, 0b001, true), None);
    }
}
