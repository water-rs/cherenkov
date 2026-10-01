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

use cherenkov::{EngineError, ResourceError};

/// One fixed module's source and precompiled artifacts.
struct Fixed {
    /// WGSL source — the fallback text and the input `build.rs` compiled.
    wgsl: &'static str,
    /// `spirv-opt -O` output, as little-endian SPIR-V bytes.
    spirv: &'static [u8],
    /// The `xcrun metallib` output, present only in Apple builds.
    metallib: &'static [u8],
    /// The module's graphics entry points, declared explicitly for
    /// passthrough creation in wgpu 30.
    entries: &'static [wgpu::PassthroughShaderEntryPoint<'static>],
}

/// `vs_main` + `fs_main`, the engine and present modules' entry points.
const VS_FS_MAIN: &[wgpu::PassthroughShaderEntryPoint<'static>] = &[
    wgpu::PassthroughShaderEntryPoint {
        name: Cow::Borrowed("vs_main"),
        workgroup_size: (0, 0, 0),
    },
    wgpu::PassthroughShaderEntryPoint {
        name: Cow::Borrowed("fs_main"),
        workgroup_size: (0, 0, 0),
    },
];

/// `vs_main` + `fs_external`, the external module's entry points.
const VS_FS_EXTERNAL: &[wgpu::PassthroughShaderEntryPoint<'static>] = &[
    wgpu::PassthroughShaderEntryPoint {
        name: Cow::Borrowed("vs_main"),
        workgroup_size: (0, 0, 0),
    },
    wgpu::PassthroughShaderEntryPoint {
        name: Cow::Borrowed("fs_external"),
        workgroup_size: (0, 0, 0),
    },
];

/// `vs_main` + `fs_projective`, the projective composite's entry points.
const VS_FS_PROJECTIVE: &[wgpu::PassthroughShaderEntryPoint<'static>] = &[
    wgpu::PassthroughShaderEntryPoint {
        name: Cow::Borrowed("vs_main"),
        workgroup_size: (0, 0, 0),
    },
    wgpu::PassthroughShaderEntryPoint {
        name: Cow::Borrowed("fs_projective"),
        workgroup_size: (0, 0, 0),
    },
];

/// The `VARIANT = 0` specialization of `shader.wgsl` (simple).
const ENGINE_WGSL0: &str = concat!(
    "const VARIANT: u32 = 0u;\n",
    include_str!("shared.wgsl"),
    include_str!("shader.wgsl"),
    "\n",
    include_str!("blend.wgsl")
);
/// The `VARIANT = 1` specialization (shadow).
const ENGINE_WGSL1: &str = concat!(
    "const VARIANT: u32 = 1u;\n",
    include_str!("shared.wgsl"),
    include_str!("shader.wgsl"),
    "\n",
    include_str!("blend.wgsl")
);
/// The `VARIANT = 2` specialization (full).
const ENGINE_WGSL2: &str = concat!(
    "const VARIANT: u32 = 2u;\n",
    include_str!("shared.wgsl"),
    include_str!("shader.wgsl"),
    "\n",
    include_str!("blend.wgsl")
);

/// The effect-module text for a user `backdrop_effect` source.
///
/// The full engine module — `ENGINE_WGSL2`, the same `const VARIANT` +
/// `shared` prelude + tail pieces the engine composes — with the stub
/// `backdrop_effect` removed and the user source appended. The stub sits
/// between two `// backdrop-effect-stub` marker lines, so removal is a
/// plain string split.
///
/// # Panics
///
/// If the stub markers are missing from `shader.wgsl` (a build bug).
#[must_use]
pub fn backdrop_effect_text(user: &str) -> Cow<'static, str> {
    const MARK: &str = "// backdrop-effect-stub";
    let (head, rest) = ENGINE_WGSL2
        .split_once(MARK)
        .expect("the backdrop-effect stub marker is part of shader.wgsl");
    let (_, tail) = rest
        .split_once(MARK)
        .expect("the backdrop-effect stub has a closing marker");
    format!("{head}{tail}\n{user}").into()
}

/// Parses and validates WGSL `text` on the caller thread, with the
/// capabilities of a core WebGPU device. Errors carry naga's diagnostic
/// against the text.
///
/// # Errors
/// [`ResourceError::Shader`] when the text fails to parse or validate.
pub fn validate_wgsl(text: &str) -> Result<naga::Module, ResourceError> {
    let module = naga::front::wgsl::parse_str(text)
        .map_err(|error| ResourceError::Shader(error.emit_to_string(text)))?;
    // The engine's floor, not the device's capabilities: a user shader has
    // to run on every device the engine supports, so a construct that only
    // a host-supplied `SharedDevice` with extra features could compile is
    // rejected here, identically on every device. What depends on the
    // actual device (limits, the driver's compiler) is left to pipeline
    // creation on the render thread, which reports it as a rejection.
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .map_err(|error| ResourceError::Shader(error.emit_to_string(text)))?;
    Ok(module)
}

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
#[cfg(not(target_arch = "wasm32"))]
const EXTERNAL_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/external.spv"));
#[cfg(target_arch = "wasm32")]
const EXTERNAL_SPV: &[u8] = &[];
#[cfg(not(target_arch = "wasm32"))]
const PROJECTIVE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/projective.spv"));
#[cfg(target_arch = "wasm32")]
const PROJECTIVE_SPV: &[u8] = &[];
#[cfg(not(target_arch = "wasm32"))]
const MIP_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mip.spv"));
#[cfg(target_arch = "wasm32")]
const MIP_SPV: &[u8] = &[];
/// `external_native.spv` — the Vulkan native module: `vs_main`,
/// `fs_external` and `fs_external_format`, with the external-format pair
/// merged into a combined sampled image by the build's restricted
/// lowering.
#[cfg(all(unix, not(target_vendor = "apple")))]
pub const EXTERNAL_NATIVE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/external_native.spv"));
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
#[cfg(target_vendor = "apple")]
const EXTERNAL_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/external.metallib"));
#[cfg(not(target_vendor = "apple"))]
const EXTERNAL_METALLIB: &[u8] = &[];
#[cfg(target_vendor = "apple")]
const PROJECTIVE_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/projective.metallib"));
#[cfg(not(target_vendor = "apple"))]
const PROJECTIVE_METALLIB: &[u8] = &[];
#[cfg(target_vendor = "apple")]
const MIP_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mip.metallib"));
#[cfg(not(target_vendor = "apple"))]
const MIP_METALLIB: &[u8] = &[];

/// The three `VARIANT` specializations of `shader.wgsl`, indexed by
/// `variant_index`.
const ENGINE: [Fixed; 3] = [
    Fixed {
        wgsl: ENGINE_WGSL0,
        spirv: ENGINE_SPV[0],
        metallib: ENGINE_METALLIB[0],
        entries: VS_FS_MAIN,
    },
    Fixed {
        wgsl: ENGINE_WGSL1,
        spirv: ENGINE_SPV[1],
        metallib: ENGINE_METALLIB[1],
        entries: VS_FS_MAIN,
    },
    Fixed {
        wgsl: ENGINE_WGSL2,
        spirv: ENGINE_SPV[2],
        metallib: ENGINE_METALLIB[2],
        entries: VS_FS_MAIN,
    },
];

/// `present.wgsl`, the presenter's module.
const PRESENT: Fixed = Fixed {
    wgsl: include_str!("present.wgsl"),
    spirv: PRESENT_SPV,
    metallib: PRESENT_METALLIB,
    entries: VS_FS_MAIN,
};

/// `shared.wgsl` plus `external.wgsl`, the external-frame module.
const EXTERNAL: Fixed = Fixed {
    wgsl: concat!(include_str!("shared.wgsl"), include_str!("external.wgsl")),
    spirv: EXTERNAL_SPV,
    metallib: EXTERNAL_METALLIB,
    entries: VS_FS_EXTERNAL,
};

/// `shared.wgsl`, `blend.wgsl` and `projective.wgsl`, the projective
/// composite module (#84).
const PROJECTIVE: Fixed = Fixed {
    wgsl: concat!(
        include_str!("shared.wgsl"),
        "\n",
        include_str!("blend.wgsl"),
        "\n",
        include_str!("projective.wgsl")
    ),
    spirv: PROJECTIVE_SPV,
    metallib: PROJECTIVE_METALLIB,
    entries: VS_FS_PROJECTIVE,
};

/// `mip.wgsl`, the projective local image's mip level module (#84).
const MIP: Fixed = Fixed {
    wgsl: include_str!("mip.wgsl"),
    spirv: MIP_SPV,
    metallib: MIP_METALLIB,
    entries: VS_FS_MAIN,
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

    /// The external-frame module.
    #[must_use]
    pub fn external_module(self, device: &wgpu::Device) -> wgpu::ShaderModule {
        self.module(device, "cherenkov external", &EXTERNAL)
    }

    /// The projective composite module.
    #[must_use]
    pub fn projective_module(self, device: &wgpu::Device) -> wgpu::ShaderModule {
        self.module(device, "cherenkov projective", &PROJECTIVE)
    }

    /// The projective mip level module.
    #[must_use]
    pub fn mip_module(self, device: &wgpu::Device) -> wgpu::ShaderModule {
        self.module(device, "cherenkov mip", &MIP)
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
                            entry_points: Cow::Borrowed(fixed.entries),
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
                            entry_points: Cow::Borrowed(fixed.entries),
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
