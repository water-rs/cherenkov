//! Component transforms reuse recorded content on the cpu backend.
#![cfg(not(target_arch = "wasm32"))]
#[path = "../../tests/common/component_animation.rs"]
mod common;

#[test]
fn live_components_reuse_recorded_content() {
    common::component_animation::<cherenkov_cpu::Raster>(cherenkov_cpu::RasterConfig::default());
}
