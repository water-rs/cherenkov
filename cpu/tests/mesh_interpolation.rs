// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Mesh colour weight selection on this backend.
#[path = "../../tests/common/mesh_interpolation.rs"]
mod common;
#[test]
fn mesh_modes_patch_only_their_own_command() {
    common::interpolation::<cherenkov_cpu::Raster>(cherenkov_cpu::RasterConfig::default());
}
