//! Precompiles the engine's fixed WGSL modules (issue #57): naga parses,
//! validates and translates them once at build time instead of wgpu doing it
//! per pipeline at runtime. Each module produces
//!
//! - `<name>.spv` — naga SPIR-V run through the `spirv-opt -O` recipe minus
//!   `simplify-instructions` (it reassociates floating-point math, breaking
//!   bit-identical output) and validated by `spirv-val` against `vulkan1.0`;
//! - `<name>.metal` — naga MSL at wgpu-hal's argument slots;
//! - `<name>.metallib` — the `.metal` compiled by `xcrun metal`/`metallib`,
//!   on Apple targets only.
//!
//! The runtime embeds these with `include_bytes!` and hands them to
//! `Device::create_shader_module_passthrough`, so the binaries are trusted
//! inputs: SPIR-V and MSL are emitted with naga's bounds checks and loop
//! bounding off (passthrough carries no runtime checks — this supersedes
//! the #55 runtime-checks work).
//!
//! Metal needs two accommodations of wgpu-hal's slot protocol, both encoded
//! in `src/render/bindings.rs`, which this script shares with the crate:
//! argument slots are per-stage counters over the bind group layout, and a
//! passthrough module is never written a runtime-array-sizes buffer, so the
//! MSL must not take one — the engine shaders never call `arrayLength`, so
//! their `array<T>` storage buffers are pinned to `array<T, 1>` for the MSL
//! emission only, which keeps the signature free of `_mslBufferSizes`.

#[path = "src/render/bindings.rs"]
mod bindings;

use std::env;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::process::Command;

use naga::back::{msl, spv};
use naga::valid::{Capabilities, ValidationFlags, Validator};

/// One fixed engine module to compile.
struct Spec {
    /// Output file stem, e.g. `engine0`.
    name: String,
    /// WGSL text (`VARIANT` already prepended for the engine shader).
    source: String,
    /// The pipeline's bind group layouts, in group order.
    groups: &'static [&'static [bindings::Entry]],
}

/// Passthrough shaders carry no naga runtime checks: the source is
/// controlled and validated here at build time.
const UNCHECKED: naga::proc::BoundsCheckPolicies = naga::proc::BoundsCheckPolicies {
    index: naga::proc::BoundsCheckPolicy::Unchecked,
    buffer: naga::proc::BoundsCheckPolicy::Unchecked,
    image_load: naga::proc::BoundsCheckPolicy::Unchecked,
    binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
};

/// The `spirv-opt -O` pass recipe (spirv-tools 2022.2) minus
/// `simplify-instructions`, which reassociates floating-point math and so
/// broke bit-identical corpus rendering. See `write_spirv`.
const SPIRV_OPT_PASSES: &[&str] = &[
    "--wrap-opkill",
    "--eliminate-dead-branches",
    "--merge-return",
    "--inline-entry-points-exhaustive",
    "--eliminate-dead-functions",
    "--eliminate-dead-code-aggressive",
    "--private-to-local",
    "--eliminate-local-single-block",
    "--eliminate-local-single-store",
    "--eliminate-dead-code-aggressive",
    "--scalar-replacement=100",
    "--convert-local-access-chains",
    "--eliminate-local-single-block",
    "--eliminate-local-single-store",
    "--eliminate-dead-code-aggressive",
    "--ssa-rewrite",
    "--eliminate-dead-code-aggressive",
    "--ccp",
    "--eliminate-dead-code-aggressive",
    "--loop-unroll",
    "--eliminate-dead-branches",
    "--redundancy-elimination",
    "--combine-access-chains",
    "--scalar-replacement=100",
    "--convert-local-access-chains",
    "--eliminate-local-single-block",
    "--eliminate-local-single-store",
    "--eliminate-dead-code-aggressive",
    "--ssa-rewrite",
    "--eliminate-dead-code-aggressive",
    "--vector-dce",
    "--eliminate-dead-inserts",
    "--eliminate-dead-branches",
    "--if-conversion",
    "--copy-propagate-arrays",
    "--reduce-load-size",
    "--eliminate-dead-code-aggressive",
    "--merge-blocks",
    "--redundancy-elimination",
    "--eliminate-dead-branches",
    "--merge-blocks",
];

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_path = manifest.join("src/render/shader.wgsl");
    let present_path = manifest.join("src/render/present.wgsl");
    for path in [&shader_path, &present_path] {
        println!("cargo::rerun-if-changed={}", path.display());
    }
    println!(
        "cargo::rerun-if-changed={}",
        manifest.join("src/render/bindings.rs").display()
    );

    let shader = std::fs::read_to_string(&shader_path)
        .unwrap_or_else(|e| panic!("{}: {e}", shader_path.display()));
    let present = std::fs::read_to_string(&present_path)
        .unwrap_or_else(|e| panic!("{}: {e}", present_path.display()));

    // The three VARIANT specializations of shader.wgsl plus present.wgsl —
    // the fixed module set `render` creates at init.
    let mut specs: Vec<Spec> = (0..3u32)
        .map(|variant| Spec {
            name: format!("engine{variant}"),
            source: format!("const VARIANT: u32 = {variant}u;\n{shader}"),
            groups: bindings::ENGINE_GROUPS,
        })
        .collect();
    specs.push(Spec {
        name: "present".into(),
        source: present,
        groups: bindings::PRESENT_GROUPS,
    });

    // wasm builds embed no passthrough artifacts, so the toolchain the
    // SPIR-V and MSL emission needs (spirv-tools, `xcrun`) is not required
    // for them; the WGSL is still parsed and validated here.
    let wasm = env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "wasm32";
    let sdk = apple_sdk();
    for spec in &specs {
        compile(&out_dir, spec, sdk, wasm);
    }
}

/// Parses, validates and compiles one module.
fn compile(out_dir: &Path, spec: &Spec, sdk: Option<&'static str>, wasm: bool) {
    let module = naga::front::wgsl::parse_str(&spec.source).unwrap_or_else(|e| {
        panic!(
            "{}: WGSL parse failed:\n{}",
            spec.name,
            e.emit_to_string(&spec.source)
        )
    });
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}: WGSL validation failed: {e:?}", spec.name));
    if wasm {
        return;
    }
    write_spirv(out_dir, spec, &module, &info);
    write_metal(out_dir, spec, &module, sdk);
}

/// wgpu-hal Vulkan maps a WGSL `@binding` to its entry's ordinal position
/// in the bind group layout; this SPIR-V must match.
fn binding_map(spec: &Spec, module: &naga::Module) -> spv::BindingMap {
    module
        .global_variables
        .iter()
        .filter_map(|(_, var)| var.binding)
        .map(|br| {
            let slot = bindings::vulkan_slot(spec.groups[br.group as usize], br.binding);
            (
                br,
                spv::BindingInfo {
                    descriptor_set: br.group,
                    binding: slot,
                    binding_array_size: None,
                },
            )
        })
        .collect()
}

fn write_spirv(out_dir: &Path, spec: &Spec, module: &naga::Module, info: &naga::valid::ModuleInfo) {
    let options = spv::Options {
        // SPIR-V 1.0 is legal for every Vulkan 1.x driver.
        lang_version: (1, 0),
        flags: spv::WriterFlags::empty(),
        fake_missing_bindings: false,
        binding_map: binding_map(spec, module),
        capabilities: None,
        bounds_check_policies: UNCHECKED,
        zero_initialize_workgroup_memory: spv::ZeroInitializeWorkgroupMemoryMode::Native,
        force_loop_bounding: false,
        ray_query_initialization_tracking: false,
        trace_ray_argument_validation: false,
        // naga 29 guarded integer division unconditionally; preserve it.
        emit_int_div_checks: true,
        use_storage_input_output_16: false,
        debug_info: None,
        task_dispatch_limits: None,
        mesh_shader_primitive_indices_clamp: false,
    };
    let mut writer = spv::Writer::new(&options)
        .unwrap_or_else(|e| panic!("{}: SPIR-V writer failed: {e:?}", spec.name));
    let mut words = Vec::new();
    writer
        .write(module, info, None, &None, &mut words)
        .unwrap_or_else(|e| panic!("{}: SPIR-V emission failed: {e:?}", spec.name));
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let unoptimized = out_dir.join(format!("{}.unopt.spv", spec.name));
    std::fs::write(&unoptimized, bytes).unwrap();
    let optimized = out_dir.join(format!("{}.spv", spec.name));
    // `spirv-opt -O` minus `simplify-instructions`: that pass reassociates
    // floating-point math (`C * (1/x)` → `C/x`, `(2*z)*k` → `z*(k/2)`),
    // which is IEEE-valid but not bit-identical and broke the corpus oracle
    // (issue #57). The pinned recipe is `spirv-opt -O` as of spirv-tools
    // 2022.2, expanded, with the offending pass dropped; keeping the pass
    // list explicit also makes the result independent of the installed
    // spirv-tools version's idea of `-O`.
    run(
        Command::new("spirv-opt")
            .args(SPIRV_OPT_PASSES)
            .arg(&unoptimized)
            .arg("-o")
            .arg(&optimized),
        &spec.name,
        "spirv-opt (from the spirv-tools package) is required to build \
         cherenkov-gpu: engine shaders are precompiled (issue #57). Install \
         it, e.g. `apt-get install spirv-tools` or `brew install spirv-tools`.",
    );
    run(
        Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.0")
            .arg(&optimized),
        &spec.name,
        "spirv-val (from the spirv-tools package) is required to build \
         cherenkov-gpu: engine shaders are precompiled (issue #57).",
    );
}

/// Emits the Metal source and, on Apple targets, the compiled library.
///
/// wgpu-hal passes a runtime-array-sizes buffer to naga-compiled modules,
/// but writes none for passthrough modules — and never calls `arrayLength`
/// sites exist in these shaders anyway. Emitting the storage arrays as
/// `array<T, 1>` keeps naga from putting a `_mslBufferSizes` argument in
/// the entry point signature, which would otherwise be left unbound.
fn write_metal(out_dir: &Path, spec: &Spec, module: &naga::Module, sdk: Option<&'static str>) {
    let module = pin_runtime_arrays(module);
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}: WGSL validation failed: {e:?}", spec.name));
    let options = msl::Options {
        // The floor Metal language version wgpu selects on supported
        // hardware; the shader test covers the same value.
        lang_version: (2, 0),
        per_entry_point_map: resource_map(spec, &module),
        inline_samplers: Vec::new(),
        spirv_cross_compatibility: false,
        fake_missing_bindings: false,
        bounds_check_policies: UNCHECKED,
        zero_initialize_workgroup_memory: false,
        force_loop_bounding: false,
        task_dispatch_limits: None,
        mesh_shader_primitive_indices_clamp: false,
        ray_query_initialization_tracking: false,
        // Preserve naga 29's integer division semantics.
        emit_int_div_checks: true,
    };
    // The engine's pipelines declare no vertex buffers, so wgpu-hal's
    // `vertex_pulling_transform` never applies; it is still passed so the
    // emission matches the hal's option set.
    let pipeline_options = msl::PipelineOptions {
        entry_point: None,
        allow_and_force_point_size: false,
        vertex_pulling_transform: true,
        vertex_buffer_mappings: Vec::new(),
        binding_array_length_map: naga::FastHashMap::default(),
    };
    let (source, translation_info) = msl::write_string(&module, &info, &options, &pipeline_options)
        .unwrap_or_else(|e| panic!("{}: MSL emission failed: {e:?}", spec.name));
    // wgpu-hal looks up pipeline functions by `entry_point` name verbatim,
    // so the emitted names must be the WGSL names.
    let emitted: Vec<&str> = translation_info
        .entry_point_names
        .iter()
        .map(|name| name.as_deref().unwrap_or("<error>"))
        .collect();
    let expected: Vec<&str> = module
        .entry_points
        .iter()
        .map(|ep| ep.name.as_str())
        .collect();
    assert_eq!(
        emitted, expected,
        "{}: MSL entry point names differ from the WGSL names",
        spec.name
    );

    let metal = out_dir.join(format!("{}.metal", spec.name));
    std::fs::write(&metal, &source).unwrap();

    if let Some(sdk) = sdk {
        let air = out_dir.join(format!("{}.air", spec.name));
        let metallib = out_dir.join(format!("{}.metallib", spec.name));
        run(
            Command::new("xcrun")
                .args(["-sdk", sdk, "metal", "-c", "-o"])
                .arg(&air)
                .arg(&metal),
            &spec.name,
            "xcrun metal is required to build cherenkov-gpu for Apple \
             targets: engine shaders are precompiled (issue #57).",
        );
        run(
            Command::new("xcrun")
                .args(["-sdk", sdk, "metallib", "-o"])
                .arg(&metallib)
                .arg(&air),
            &spec.name,
            "xcrun metallib is required to build cherenkov-gpu for Apple \
             targets: engine shaders are precompiled (issue #57).",
        );
    }
}

/// Pins each runtime-sized `array<T>` to `array<T, 1>` in a cloned module.
/// Metal cannot express runtime array types; naga's own backend uses the
/// bound-1 form plus a sizes argument, and the pinned form alone is what a
/// passthrough signature needs.
fn pin_runtime_arrays(module: &naga::Module) -> naga::Module {
    let mut module = module.clone();
    let dynamic: Vec<_> = module
        .types
        .iter()
        .filter(|(_, ty)| {
            matches!(
                ty.inner,
                naga::TypeInner::Array {
                    size: naga::ArraySize::Dynamic,
                    ..
                }
            )
        })
        .map(|(handle, _)| handle)
        .collect();
    for handle in dynamic {
        let mut ty = module.types.get_handle(handle).unwrap().clone();
        // `base` and `stride` are preserved; only the bound is pinned.
        if let naga::TypeInner::Array { ref mut size, .. } = ty.inner {
            *size = naga::ArraySize::Constant(NonZeroU32::new(1).unwrap());
        }
        module.types.replace(handle, ty);
    }
    module
}

/// The argument slots naga must emit, reproducing wgpu-hal's
/// `create_pipeline_layout` assignment: per-stage counters over the groups
/// in order.
fn resource_map(spec: &Spec, module: &naga::Module) -> msl::EntryPointResourceMap {
    let mut map = msl::EntryPointResourceMap::new();
    for ep in &module.entry_points {
        let stage = match ep.stage {
            naga::ShaderStage::Vertex => bindings::VERTEX,
            naga::ShaderStage::Fragment => bindings::FRAGMENT,
            naga::ShaderStage::Compute => bindings::COMPUTE,
            other => panic!("{}: unsupported shader stage {other:?}", spec.name),
        };
        let plan = bindings::metal_plan(spec.groups, stage);
        let resources = plan
            .targets
            .iter()
            .map(|&(group, binding, target)| {
                (
                    naga::ResourceBinding { group, binding },
                    msl::BindTarget {
                        buffer: target.buffer,
                        texture: target.texture,
                        sampler: target.sampler.map(msl::BindSamplerTarget::Resource),
                        external_texture: None,
                        mutable: target.mutable,
                    },
                )
            })
            .collect();
        map.insert(
            ep.name.clone(),
            msl::EntryPointResources {
                resources,
                immediates_buffer: None,
                sizes_buffer: plan.sizes_buffer,
            },
        );
    }
    map
}

/// The Metal SDK for an Apple target, or `None` for other targets. A
/// non-Apple host cannot produce a `.metallib`, so building for Apple there
/// is an explicit error — never a silent WGSL fallback.
fn apple_sdk() -> Option<&'static str> {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let sdk = match target_os.as_str() {
        "macos" => "macosx",
        "ios" if target_env == "sim" => "iphonesimulator",
        "ios" => "iphoneos",
        "tvos" | "watchos" | "visionos" => panic!(
            "cherenkov-gpu precompiles Metal shaders for Apple targets (issue \
             #57): no Metal SDK mapping exists for target-os {target_os}"
        ),
        _ => return None,
    };
    let target = env::var("TARGET").unwrap();
    let host = env::var("HOST").unwrap();
    assert!(
        host.ends_with("apple-darwin"),
        "cherenkov-gpu precompiles Metal shaders with xcrun (issue #57): \
         building for {target} needs an Apple host, but the host is {host}. \
         Build for Apple targets on macOS."
    );
    Some(sdk)
}

/// Runs `command`, failing the build with `hint` when the tool is missing
/// and with the tool's own output when it fails.
fn run(command: &mut Command, name: &str, hint: &str) {
    let output = match command.output() {
        Ok(output) => output,
        Err(e) => panic!(
            "{name}: {hint}\nspawning {}: {e}",
            command.get_program().display()
        ),
    };
    assert!(
        output.status.success(),
        "{name}: {} failed ({})\nstdout:\n{}\nstderr:\n{}",
        command.get_program().display(),
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
