//! Additional device features shared by native import and composition.

use std::ffi::CStr;

use ash::vk;

/// Attachment access negotiated when the engine creates its Vulkan device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttachmentAccess {
    /// Fragment color-attachment reads observe prior overlapping fragments.
    pub raster_order: bool,
    /// Dynamic rendering supports input attachments and by-region barriers.
    pub local_read: bool,
}

/// Feature structures whose addresses remain valid through `vkCreateDevice`.
///
/// wgpu-hal appends its own feature structures *after* invoking the callback.
/// These fields must therefore outlive both the callback and the HAL open call.
pub struct Features {
    ycbcr: vk::PhysicalDeviceSamplerYcbcrConversionFeatures<'static>,
    raster: vk::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT<'static>,
    dynamic: vk::PhysicalDeviceDynamicRenderingFeatures<'static>,
    local: vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR<'static>,
    extensions: Vec<&'static CStr>,
}

impl Features {
    /// Queries feature bits, then requests only the supported combinations.
    pub fn query(adapter: &wgpu::hal::vulkan::Adapter) -> Self {
        let instance = adapter.shared_instance().raw_instance();
        let physical = adapter.raw_physical_device();
        let caps = adapter.physical_device_capabilities();
        let has = |extension: &CStr| caps.supports_extension(extension);
        let mut features = Self {
            ycbcr: vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default(),
            raster: vk::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT::default(),
            dynamic: vk::PhysicalDeviceDynamicRenderingFeatures::default(),
            local: vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR::default(),
            extensions: super::extra_device_extensions()
                .into_iter()
                .filter(|extension| has(extension))
                .collect(),
        };
        let mut query = vk::PhysicalDeviceFeatures2::default();
        if has(ash::khr::sampler_ycbcr_conversion::NAME) {
            query = query.push_next(&mut features.ycbcr);
        }
        if has(ash::ext::rasterization_order_attachment_access::NAME) {
            query = query.push_next(&mut features.raster);
        }
        if has(ash::khr::dynamic_rendering::NAME) {
            query = query.push_next(&mut features.dynamic);
        }
        if has(ash::khr::dynamic_rendering_local_read::NAME) {
            query = query.push_next(&mut features.local);
        }
        // SAFETY: the physical device belongs to this instance, and every
        // queried structure is supported and borrowed for the duration.
        unsafe { instance.get_physical_device_features2(physical, &mut query) };

        let raster = features.raster.rasterization_order_color_attachment_access == vk::TRUE;
        let local = features.local.dynamic_rendering_local_read == vk::TRUE
            && features.dynamic.dynamic_rendering == vk::TRUE;
        // Vulkan 1.2 promotes the dependencies of dynamic rendering. On
        // earlier versions request the dependency closure explicitly.
        let api = unsafe { instance.get_physical_device_properties(physical) }.api_version;
        let dependencies = [
            (vk::API_VERSION_1_2, ash::khr::depth_stencil_resolve::NAME),
            (vk::API_VERSION_1_2, ash::khr::create_renderpass2::NAME),
            (vk::API_VERSION_1_1, ash::khr::multiview::NAME),
            (vk::API_VERSION_1_1, ash::khr::maintenance2::NAME),
        ];
        let local = local
            && dependencies
                .iter()
                .all(|(promoted, extension)| api >= *promoted || has(extension));
        if raster {
            features
                .extensions
                .push(ash::ext::rasterization_order_attachment_access::NAME);
        }
        if local {
            features.extensions.extend([
                ash::khr::dynamic_rendering::NAME,
                ash::khr::dynamic_rendering_local_read::NAME,
            ]);
            features.extensions.extend(
                dependencies
                    .into_iter()
                    .filter(|(promoted, _)| api < *promoted)
                    .map(|(_, extension)| extension),
            );
        }
        // Query structures contain supported bits and links. Rebuild the
        // enabled structures so no unused depth/stencil feature or query
        // chain accidentally reaches device creation.
        features.ycbcr = vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default()
            .sampler_ycbcr_conversion(features.ycbcr.sampler_ycbcr_conversion == vk::TRUE);
        features.raster =
            vk::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT::default()
                .rasterization_order_color_attachment_access(raster);
        features.dynamic =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(local);
        features.local = vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR::default()
            .dynamic_rendering_local_read(local);
        features
    }

    /// Prepends the requested features, retaining any existing callback chain.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the HAL callback passes an owned FnOnce argument with mutable fields"
    )]
    pub fn apply(&mut self, args: wgpu::hal::vulkan::CreateDeviceCallbackArgs<'_, '_, '_>) {
        for extension in &self.extensions {
            if !args.extensions.contains(extension) {
                args.extensions.push(extension);
            }
        }
        args.create_info.p_next = self.chain(args.create_info.p_next);
    }

    fn chain(&mut self, tail: *const core::ffi::c_void) -> *const core::ffi::c_void {
        self.ycbcr.p_next = core::ptr::null_mut();
        self.raster.p_next = core::ptr::null_mut();
        self.dynamic.p_next = core::ptr::null_mut();
        self.local.p_next = core::ptr::null_mut();
        // Only the pNext address is transferred: the lifetime belongs to
        // `self`, which the caller retains until `open_with_callback` returns.
        let mut info = vk::DeviceCreateInfo {
            p_next: tail,
            ..vk::DeviceCreateInfo::default()
        };
        if self.ycbcr.sampler_ycbcr_conversion == vk::TRUE {
            info = info.push_next(&mut self.ycbcr);
        }
        if self.raster.rasterization_order_color_attachment_access == vk::TRUE {
            info = info.push_next(&mut self.raster);
        }
        if self.local.dynamic_rendering_local_read == vk::TRUE {
            info = info.push_next(&mut self.dynamic).push_next(&mut self.local);
        }
        info.p_next
    }
}

/// Queries attachment features for an engine-created (or equivalently
/// configured) device. Extension presence alone never establishes support.
pub fn attachment_access(
    device: &wgpu::hal::vulkan::Device,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> AttachmentAccess {
    let enabled = device.enabled_device_extensions();
    let mut raster = vk::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT::default();
    let mut dynamic = vk::PhysicalDeviceDynamicRenderingFeatures::default();
    let mut local = vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR::default();
    let mut query = vk::PhysicalDeviceFeatures2::default();
    if enabled.contains(&ash::ext::rasterization_order_attachment_access::NAME) {
        query = query.push_next(&mut raster);
    }
    if enabled.contains(&ash::khr::dynamic_rendering_local_read::NAME)
        && enabled.contains(&ash::khr::dynamic_rendering::NAME)
    {
        query = query.push_next(&mut dynamic).push_next(&mut local);
    }
    // SAFETY: the query chain contains only enabled extensions for physical.
    unsafe { instance.get_physical_device_features2(physical, &mut query) };
    AttachmentAccess {
        raster_order: raster.rasterization_order_color_attachment_access == vk::TRUE,
        local_read: dynamic.dynamic_rendering == vk::TRUE
            && local.dynamic_rendering_local_read == vk::TRUE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_chain_preserves_existing_tail_and_import_features() {
        let tail = vk::PhysicalDeviceShaderDrawParametersFeatures::default();
        let tail_ptr = core::ptr::from_ref(&tail).cast();
        let mut features = Features {
            ycbcr: vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default()
                .sampler_ycbcr_conversion(true),
            raster: vk::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT::default()
                .rasterization_order_color_attachment_access(true),
            dynamic: vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true),
            local: vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR::default()
                .dynamic_rendering_local_read(true),
            extensions: Vec::new(),
        };
        let head = features.chain(tail_ptr);
        let mut cursor = head.cast::<vk::BaseInStructure<'_>>();
        let mut types = Vec::new();
        while !cursor.is_null() {
            let header = unsafe { &*cursor };
            types.push(header.s_type);
            cursor = header.p_next;
        }
        assert_eq!(types, [
            vk::StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR,
            vk::StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
            vk::StructureType::PHYSICAL_DEVICE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_FEATURES_EXT,
            vk::StructureType::PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
            vk::StructureType::PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES,
        ]);
        features.local.dynamic_rendering_local_read = vk::FALSE;
        features.raster.rasterization_order_color_attachment_access = vk::FALSE;
        assert_eq!(
            features.chain(tail_ptr),
            core::ptr::from_ref(&features.ycbcr).cast()
        );
        assert_eq!(features.ycbcr.p_next.cast_const(), tail_ptr);
    }
}
