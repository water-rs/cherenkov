//! Shader source specialization: token substitution, preambles, and
//! format capability helpers shared by the runtime's pipeline builders.

use filtrate_core::MAX_FILTER_PARAM_VEC4S;

extern crate alloc;

pub(super) const SPATIAL_OUTPUT_FORMAT_TOKEN: &str = "OUTPUT_STORAGE_FORMAT";
/// Token in both shader preambles for the parameter array row count, so the
/// WGSL declaration can never drift from `filtrate_core::MAX_FILTER_PARAMS`.
pub(super) const PARAM_VEC4S_TOKEN: &str = "PARAM_VEC4S";
/// Token in the color preamble for the display-range clamp bound
/// (`const COLOR_CLAMP_MAX`) used by LDR-styled fragments (photo-effect
/// presets). Substituted with 1.0 for LDR targets and the f16 maximum for
/// HDR targets so those fragments never crush an HDR chain's highlights.
pub(super) const COLOR_CLAMP_MAX_TOKEN: &str = "CLAMP_MAX_BOUND";
pub(super) const F16_MAX_WGSL: &str = "65504.0";

/// Workgroup shape shared by every spatial compute shader. The WGSL sources
/// reference the `WORKGROUP_X` / `WORKGROUP_Y` tokens, which
/// [`specialize_spatial_shader`] substitutes with these values so shader
/// declarations and `dispatch_workgroups` can never drift apart.
/// 256 threads benchmarks at the memory-bandwidth floor on tiler GPUs.
pub(super) const SPATIAL_WORKGROUP_X: u32 = 16;
pub(super) const SPATIAL_WORKGROUP_Y: u32 = 16;
pub(super) const SPATIAL_WORKGROUP_X_TOKEN: &str = "WORKGROUP_X";
pub(super) const SPATIAL_WORKGROUP_Y_TOKEN: &str = "WORKGROUP_Y";

/// Policy controlling whether intermediate (scratch) textures use HDR
/// formats. Chosen per adapter via [`super::FilterAdapter::require_hdr`] /
/// [`super::FilterAdapter::force_ldr`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HdrPolicy {
    /// Require HDR-capable intermediate pipeline; fail setup if unavailable.
    RequireHdr,
    /// Prefer HDR intermediates and automatically downgrade to LDR when unsupported.
    #[default]
    PreferHdr,
    /// Force LDR intermediates even on HDR-capable devices.
    ForceLdr,
}

/// Policy controlling how spatial (`Filter::COLOR_ONLY == false`) stages are
/// executed. Chosen per adapter via
/// [`super::FilterAdapter::spatial_execution`].
///
/// WebGL2 devices expose no compute shaders or storage textures
/// (`max_compute_workgroups_per_dimension == 0`), so
/// [`SpatialExecution::Auto`] resolves to the fragment path there: each
/// spatial body is recompiled as a fragment shader that writes its own pixel
/// through a render attachment instead of `textureStore`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpatialExecution {
    /// Use compute passes when the device supports them, fragment passes
    /// otherwise (WebGL2).
    #[default]
    Auto,
    /// Always compile spatial stages as fragment passes, even on
    /// compute-capable devices. Used to validate and benchmark the WebGL path
    /// on native adapters.
    ForceFragment,
}

/// Resolved execution strategy for spatial passes on this device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SpatialBackend {
    /// One `@compute` dispatch per stage writing a storage texture.
    Compute,
    /// One fullscreen fragment pass per stage writing a render attachment.
    Fragment,
}

impl SpatialBackend {
    /// Resolve `policy` against the device's actual limits. WebGL2 devices
    /// report zero compute/storage capacity, which is what `Auto` keys on.
    pub(super) const fn resolve(policy: SpatialExecution, limits: &wgpu::Limits) -> Self {
        let compute_supported = limits.max_compute_workgroups_per_dimension > 0
            && limits.max_storage_textures_per_shader_stage > 0;
        match (policy, compute_supported) {
            (SpatialExecution::ForceFragment, _) | (SpatialExecution::Auto, false) => {
                Self::Fragment
            }
            (SpatialExecution::Auto, true) => Self::Compute,
        }
    }

    /// Usages every scratch texture must carry under this backend.
    pub(super) fn scratch_texture_usage(self) -> wgpu::TextureUsages {
        let mut usage =
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT;
        if matches!(self, Self::Compute) {
            usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        usage
    }
}

pub(super) const fn is_hdr_texture_format(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rgba32Float
    )
}

pub(super) const fn preferred_scratch_format(
    input_format: wgpu::TextureFormat,
    output_format: wgpu::TextureFormat,
) -> wgpu::TextureFormat {
    if is_hdr_texture_format(input_format) || is_hdr_texture_format(output_format) {
        wgpu::TextureFormat::Rgba16Float
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    }
}

pub(super) const fn storage_format_to_wgsl(
    format: wgpu::TextureFormat,
) -> Result<&'static str, &'static str> {
    match format {
        wgpu::TextureFormat::Rgba8Unorm => Ok("rgba8unorm"),
        wgpu::TextureFormat::Rgba16Float => Ok("rgba16float"),
        wgpu::TextureFormat::Rgba32Float => Ok("rgba32float"),
        _ => Err("unsupported storage texture format for spatial filter"),
    }
}

/// Whether `format` supports hardware linear filtering without extra device
/// features (float32 filtering is feature-gated in WebGPU; float16 and unorm
/// formats are filterable in core).
pub(super) const fn is_filterable_texture_format(format: wgpu::TextureFormat) -> bool {
    !matches!(format, wgpu::TextureFormat::Rgba32Float)
}

const SPATIAL_BINDINGS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/shaders/shared/spatial_bindings.wgsl"
));
const SPATIAL_OUTPUT_COMPUTE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/shaders/shared/spatial_output_compute.wgsl"
));
const SPATIAL_OUTPUT_FRAGMENT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/shaders/shared/spatial_output_fragment.wgsl"
));
const SPATIAL_HELPERS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/shaders/shared/spatial_helpers.wgsl"
));
const SPATIAL_FRAGMENT_POSTAMBLE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/shaders/shared/spatial_fragment_postamble.wgsl"
));

fn compose_spatial_shader(parts: &[&str]) -> alloc::string::String {
    let mut combined =
        alloc::string::String::with_capacity(parts.iter().map(|part| part.len() + 1).sum());
    for (index, part) in parts.iter().enumerate() {
        if index > 0 {
            combined.push('\n');
        }
        combined.push_str(part);
    }
    combined.replace(PARAM_VEC4S_TOKEN, &MAX_FILTER_PARAM_VEC4S.to_string())
}

/// Assembles the compute variant of a spatial stage (bindings + storage
/// output + helpers + body) and substitutes its token contract.
pub(super) fn specialize_spatial_shader(
    shader_source: &str,
    storage_format: wgpu::TextureFormat,
) -> Result<alloc::string::String, &'static str> {
    let storage_ty = storage_format_to_wgsl(storage_format)?;
    Ok(compose_spatial_shader(&[
        SPATIAL_BINDINGS,
        SPATIAL_OUTPUT_COMPUTE,
        SPATIAL_HELPERS,
        shader_source,
    ])
    .replace(SPATIAL_OUTPUT_FORMAT_TOKEN, storage_ty)
    .replace(SPATIAL_WORKGROUP_X_TOKEN, &SPATIAL_WORKGROUP_X.to_string())
    .replace(SPATIAL_WORKGROUP_Y_TOKEN, &SPATIAL_WORKGROUP_Y.to_string()))
}

/// The exact compute entry declaration every spatial body starts with.
const SPATIAL_COMPUTE_ENTRY: &str = "@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>)";
const SPATIAL_FRAGMENT_ENTRY: &str = "fn main(gid: vec3<u32>)";

/// Rewrites a spatial body for fragment execution: drops the `@compute`
/// entry attributes so `fs_main` can call `main(gid)` as a plain function,
/// and translates `textureStore(output_texture, coord, value)` calls —
/// including ones wrapped across lines — into `store_output(coord, value)`.
/// Fails fast on any body that doesn't match the contract.
fn spatial_body_for_fragment(body: &str) -> alloc::string::String {
    const STORE: &str = "textureStore";
    let mut out = body.replace(SPATIAL_COMPUTE_ENTRY, SPATIAL_FRAGMENT_ENTRY);
    assert!(
        !out.contains("@compute") && !out.contains("global_invocation_id"),
        "spatial shader body does not match the expected compute entry declaration"
    );
    // Translate textureStore(output_texture, …) → store_output(…). The first
    // argument is always the `output_texture` sentinel; consume it strictly so
    // a store to anything else is a loud failure, not silent corruption.
    let mut rewritten = alloc::string::String::with_capacity(out.len());
    while let Some(start) = out.find(STORE) {
        rewritten.push_str(&out[..start]);
        let rest = &out[start + STORE.len()..];
        let rest = rest
            .trim_start()
            .strip_prefix('(')
            .and_then(|s| s.trim_start().strip_prefix("output_texture"))
            .and_then(|s| s.trim_start().strip_prefix(','))
            .map(str::trim_start)
            .expect(
                "spatial shader body calls textureStore on something other than `output_texture`",
            );
        rewritten.push_str("store_output(");
        out = rest.to_string();
    }
    rewritten.push_str(&out);
    rewritten
}

/// Assembles the fragment variant of a spatial stage: same bindings and
/// helpers as the compute variant, but the output binding is the
/// `store_output` sentinel and the module ends with fullscreen-triangle
/// vertex/fragment entry points.
pub(super) fn specialize_spatial_fragment_shader(shader_source: &str) -> alloc::string::String {
    let body = spatial_body_for_fragment(shader_source);
    compose_spatial_shader(&[
        SPATIAL_BINDINGS,
        SPATIAL_OUTPUT_FRAGMENT,
        SPATIAL_HELPERS,
        &body,
        SPATIAL_FRAGMENT_POSTAMBLE,
    ])
}

/// Assembles the fused color shader (preamble + fragments + postamble) and
/// substitutes the token contract for the pass's target format.
pub(super) fn specialize_color_shader(
    fragments: &str,
    target_format: wgpu::TextureFormat,
) -> alloc::string::String {
    let preamble = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/shaders/shared/fragment_preamble.wgsl"
    ));
    let postamble = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/shaders/shared/fragment_postamble.wgsl"
    ));
    let mut shader_source =
        alloc::string::String::with_capacity(preamble.len() + fragments.len() + postamble.len());
    shader_source.push_str(preamble);
    shader_source.push_str(fragments);
    shader_source.push_str(postamble);
    let clamp_max = if is_hdr_texture_format(target_format) {
        F16_MAX_WGSL
    } else {
        "1.0"
    };
    shader_source
        .replace(PARAM_VEC4S_TOKEN, &MAX_FILTER_PARAM_VEC4S.to_string())
        .replace(COLOR_CLAMP_MAX_TOKEN, clamp_max)
}
