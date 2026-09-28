//! Loading of the engine's fixed shader modules (issue #57).
//!
//! `build.rs` compiles every fixed WGSL module once with naga; the runtime
//! embeds the results and loads them through wgpu's passthrough shader API,
//! so no naga translation, bounds checks or loop bounding run at device or
//! pipeline creation. The passthrough path is safe here because the sources
//! are fixed strings shipped with the crate and validated at build time —
//! unlike the user-supplied shader sources in `paint` and `interop`, which
//! keep the checked WGSL path.
//!
//! Delivery by backend:
//!
//! - **Vulkan** gets `spirv-opt -O` SPIR-V.
//! - **Metal** gets a `.metallib` compiled by `xcrun` during the build.
//! - **Everything else keeps WGSL** as a backend property, not a fallback:
//!   wgpu 29 has no GLSL producer to feed GL's passthrough input, DX12
//!   passthrough takes runtime-compiled HLSL or build-time DXIL that this
//!   build does not produce, and wasm/WebGPU keeps WGSL by design.

use std::borrow::Cow;

use cherenkov::EngineError;

/// One fixed module's source and precompiled artifacts.
struct Fixed {
    /// WGSL source — the fallback text and the input `build.rs` compiled.
    wgsl: &'static str,
    /// `spirv-opt -O` output, as little-endian SPIR-V bytes.
    spirv: &'static [u8],
    /// The `xcrun metallib` output, present only in Apple builds.
    metallib: &'static [u8],
}

/// The `VARIANT = 0` specialization of `shader.wgsl` (simple).
const ENGINE_WGSL0: &str = concat!("const VARIANT: u32 = 0u;\n", include_str!("shader.wgsl"));
/// The `VARIANT = 1` specialization (shadow).
const ENGINE_WGSL1: &str = concat!("const VARIANT: u32 = 1u;\n", include_str!("shader.wgsl"));
/// The `VARIANT = 2` specialization (full).
const ENGINE_WGSL2: &str = concat!("const VARIANT: u32 = 2u;\n", include_str!("shader.wgsl"));

// The passthrough artifacts are embedded only where they can be loaded:
// wasm keeps WGSL, and `.metallib` files exist only in Apple builds
// (`build.rs` refuses to produce them otherwise, and a Metal backend cannot
// appear on a non-Apple build), so the empty slices are unreachable.
#[cfg(not(target_arch = "wasm32"))]
const ENGINE_SPV: [&[u8]; 3] = [
    include_bytes!(concat!(env!("OUT_DIR"), "/engine0.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/engine1.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/engine2.spv")),
];
#[cfg(target_arch = "wasm32")]
const ENGINE_SPV: [&[u8]; 3] = [&[], &[], &[]];
#[cfg(not(target_arch = "wasm32"))]
const PRESENT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/present.spv"));
#[cfg(target_arch = "wasm32")]
const PRESENT_SPV: &[u8] = &[];
#[cfg(target_vendor = "apple")]
const ENGINE_METALLIB: [&[u8]; 3] = [
    include_bytes!(concat!(env!("OUT_DIR"), "/engine0.metallib")),
    include_bytes!(concat!(env!("OUT_DIR"), "/engine1.metallib")),
    include_bytes!(concat!(env!("OUT_DIR"), "/engine2.metallib")),
];
#[cfg(not(target_vendor = "apple"))]
const ENGINE_METALLIB: [&[u8]; 3] = [&[], &[], &[]];
#[cfg(target_vendor = "apple")]
const PRESENT_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/present.metallib"));
#[cfg(not(target_vendor = "apple"))]
const PRESENT_METALLIB: &[u8] = &[];

/// The three `VARIANT` specializations of `shader.wgsl`, indexed by
/// `variant_index`.
const ENGINE: [Fixed; 3] = [
    Fixed {
        wgsl: ENGINE_WGSL0,
        spirv: ENGINE_SPV[0],
        metallib: ENGINE_METALLIB[0],
    },
    Fixed {
        wgsl: ENGINE_WGSL1,
        spirv: ENGINE_SPV[1],
        metallib: ENGINE_METALLIB[1],
    },
    Fixed {
        wgsl: ENGINE_WGSL2,
        spirv: ENGINE_SPV[2],
        metallib: ENGINE_METALLIB[2],
    },
];

/// `present.wgsl`, the presenter's module.
const PRESENT: Fixed = Fixed {
    wgsl: include_str!("present.wgsl"),
    spirv: PRESENT_SPV,
    metallib: PRESENT_METALLIB,
};

/// How the fixed engine modules reach the device — a property of the
/// selected backend, not a runtime fallback.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShaderDelivery {
    /// Optimized SPIR-V through `create_shader_module_passthrough`.
    Spirv,
    /// A compiled `.metallib` through `create_shader_module_passthrough`.
    Metallib,
    /// WGSL source.
    Wgsl,
}

/// Selects the delivery for `backend`.
///
/// # Errors
/// `EngineError::Backend` when the backend takes passthrough shaders
/// (Vulkan, Metal) but the device was created without
/// `Features::PASSTHROUGH_SHADERS` — possible only for an externally
/// supplied device, since engine-created devices request it.
pub fn delivery(
    backend: wgpu::Backend,
    device: &wgpu::Device,
) -> Result<ShaderDelivery, EngineError> {
    let delivery = match backend {
        // On wasm this is dead code — `Backend` is always BrowserWebGpu —
        // but the match still type-checks both targets.
        wgpu::Backend::Vulkan => ShaderDelivery::Spirv,
        wgpu::Backend::Metal => ShaderDelivery::Metallib,
        _ => return Ok(ShaderDelivery::Wgsl),
    };
    if device
        .features()
        .contains(wgpu::Features::PASSTHROUGH_SHADERS)
    {
        Ok(delivery)
    } else {
        Err(EngineError::Backend(format!(
            "{backend:?} backend but the device was created without \
             Features::PASSTHROUGH_SHADERS: cherenkov's fixed shaders are \
             precompiled for this backend (issue #57), so a shared device \
             passed to cherenkov-gpu must request the feature"
        )))
    }
}

impl ShaderDelivery {
    /// The engine `VARIANT` modules, indexed by `variant_index`.
    #[must_use]
    pub fn engine_module(self, device: &wgpu::Device, variant: usize) -> wgpu::ShaderModule {
        self.module(device, "cherenkov", &ENGINE[variant])
    }

    /// The present module.
    #[must_use]
    pub fn present_module(self, device: &wgpu::Device) -> wgpu::ShaderModule {
        self.module(device, "present", &PRESENT)
    }

    fn module(
        self,
        device: &wgpu::Device,
        label: &'static str,
        fixed: &Fixed,
    ) -> wgpu::ShaderModule {
        match self {
            Self::Wgsl => {
                // The engine sources are fixed strings validated at build
                // time — trusted, like the passthrough binaries, so the WGSL
                // path also skips naga's runtime checks.
                unsafe {
                    device.create_shader_module_trusted(
                        wgpu::ShaderModuleDescriptor {
                            label: Some(label),
                            source: wgpu::ShaderSource::Wgsl(fixed.wgsl.into()),
                        },
                        wgpu::ShaderRuntimeChecks::unchecked(),
                    )
                }
            }
            Self::Spirv => {
                assert!(
                    !fixed.spirv.is_empty(),
                    "no SPIR-V artifact is embedded in a wasm build"
                );
                // SAFETY: `fixed.spirv` is naga+spirv-opt output embedded at
                // build time — trusted SPIR-V matching the pipeline layout.
                unsafe {
                    device.create_shader_module_passthrough(
                        wgpu::ShaderModuleDescriptorPassthrough {
                            label: Some(label),
                            entry_points: entry_points(),
                            spirv: Some(words(fixed.spirv)),
                            ..wgpu::ShaderModuleDescriptorPassthrough::default()
                        },
                    )
                }
            }
            Self::Metallib => {
                assert!(
                    !fixed.metallib.is_empty(),
                    "a Metal backend implies an Apple build with the \
                     metallib compiled in"
                );
                // SAFETY: `fixed.metallib` is `xcrun metallib` output
                // embedded at build time — trusted code matching the
                // pipeline layout.
                unsafe {
                    device.create_shader_module_passthrough(
                        wgpu::ShaderModuleDescriptorPassthrough {
                            label: Some(label),
                            entry_points: entry_points(),
                            metallib: Some(Cow::Borrowed(fixed.metallib)),
                            ..wgpu::ShaderModuleDescriptorPassthrough::default()
                        },
                    )
                }
            }
        }
    }
}

/// Decodes the little-endian SPIR-V byte file into words.
///
/// Compiled for wasm too — the `Spirv` arm asserts unreachable there — so
/// the match arms stay uniform.
fn words(spirv: &[u8]) -> Cow<'static, [u32]> {
    assert_eq!(spirv.len() % 4, 0, "SPIR-V artifact truncated");
    Cow::Owned(
        spirv
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect(),
    )
}

// Every fixed module contains this vertex/fragment pair. Passthrough entry
// points are explicit in wgpu 30; graphics stages have no workgroup size.
const fn entry_points() -> Cow<'static, [wgpu::PassthroughShaderEntryPoint<'static>]> {
    Cow::Borrowed(&[
        wgpu::PassthroughShaderEntryPoint {
            name: Cow::Borrowed("vs_main"),
            workgroup_size: (0, 0, 0),
        },
        wgpu::PassthroughShaderEntryPoint {
            name: Cow::Borrowed("fs_main"),
            workgroup_size: (0, 0, 0),
        },
    ])
}
