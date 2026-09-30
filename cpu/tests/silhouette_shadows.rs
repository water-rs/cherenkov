//! General silhouette convolution.
#![cfg(not(target_arch = "wasm32"))]
#[path = "../../tests/common/silhouette_shadows.rs"]
mod common;
#[test]
fn live_shadows_keep_offscreen_contributors_and_outer_clips() {
    common::retained_and_padded::<cherenkov_cpu::Raster>(cherenkov_cpu::RasterConfig::default());
}

#[test]
fn invalid_silhouette_inputs_fail_explicitly() {
    common::invalid::<cherenkov_cpu::Raster>(cherenkov_cpu::RasterConfig::default);
}
