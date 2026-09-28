//! General silhouette convolution.
#[path = "../../tests/common/silhouette_shadows.rs"]
mod common;
#[test]
fn live_shadows_keep_offscreen_contributors_and_outer_clips() {
    common::retained_and_padded::<cherenkov_cpu::Raster>(cherenkov_cpu::RasterConfig::default());
}
