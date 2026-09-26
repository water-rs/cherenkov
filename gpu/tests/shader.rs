// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The WGSL source translates through every naga backend wgpu uses, for
//! each `VARIANT` pipeline, so a Metal or D3D regression shows on Linux.

use naga::back::{hlsl, msl, spv};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{Module, front::wgsl};

const SHADER: &str = include_str!("../src/render/shader.wgsl");

/// The oldest Metal language version wgpu selects on a supported macOS
/// (10.13 → 2.0), which is where `instance_id` and friends became legal.
const MSL_VERSION: (u8, u8) = (2, 0);

fn composed(variant: u32) -> (Module, naga::valid::ModuleInfo) {
    let source = format!("const VARIANT: u32 = {variant}u;\n{SHADER}");
    let module = wgsl::parse_str(&source).unwrap_or_else(|e| panic!("variant {variant}: {e}"));
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|e| panic!("variant {variant}: {e:?}"));
    (module, info)
}

#[test]
fn every_variant_emits_msl() {
    for variant in 0..3 {
        let (module, info) = composed(variant);
        let options = msl::Options {
            lang_version: MSL_VERSION,
            ..msl::Options::default()
        };
        msl::write_string(&module, &info, &options, &msl::PipelineOptions::default())
            .unwrap_or_else(|e| panic!("variant {variant}: msl: {e}"));
    }
}

#[test]
fn every_variant_emits_spirv() {
    for variant in 0..3 {
        let (module, info) = composed(variant);
        let mut writer = spv::Writer::new(&spv::Options::default())
            .unwrap_or_else(|e| panic!("variant {variant}: spv: {e}"));
        let mut words = Vec::new();
        writer
            .write(&module, &info, None, &None, &mut words)
            .unwrap_or_else(|e| panic!("variant {variant}: spv: {e}"));
        assert!(!words.is_empty());
    }
}

#[test]
fn every_variant_emits_hlsl() {
    for variant in 0..3 {
        let (module, info) = composed(variant);
        let options = hlsl::Options::default();
        let mut out = String::new();
        let pipeline_options = hlsl::PipelineOptions::default();
        let mut writer = hlsl::Writer::new(&mut out, &options, &pipeline_options);
        writer
            .write(&module, &info, None)
            .unwrap_or_else(|e| panic!("variant {variant}: hlsl: {e}"));
        assert!(!out.is_empty());
    }
}
