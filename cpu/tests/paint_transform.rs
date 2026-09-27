//! Independent paint coordinates on the cpu backend.
#[path = "../../tests/support/paint_transform.rs"]
mod common;

#[test]
fn live_paint_transforms_preserve_geometry_and_retained_output() {
    common::retained::<cherenkov_cpu::Raster>(cherenkov_cpu::RasterConfig::default());
}
#[test]
fn nested_paint_transforms_compose_and_sample_analytically() {
    common::composition::<cherenkov_cpu::Raster>(cherenkov_cpu::RasterConfig::default());
}
#[test]
fn singular_and_non_finite_paint_transforms_fail() {
    common::invalid::<cherenkov_cpu::Raster>(cherenkov_cpu::RasterConfig::default());
}
