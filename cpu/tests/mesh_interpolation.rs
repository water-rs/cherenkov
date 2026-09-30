//! Mesh colour weight selection on this backend.
#![cfg(not(target_arch = "wasm32"))]
#[path = "../../tests/common/mesh_interpolation.rs"]
mod common;
#[test]
fn mesh_modes_patch_only_their_own_command() {
    common::interpolation::<cherenkov_cpu::Raster>(cherenkov_cpu::RasterConfig::default());
}
