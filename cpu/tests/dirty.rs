//! Exact retained/full equivalence on the CPU raster backend.

use cherenkov::Backend;

#[test]
fn randomized_incremental_matches_full_lowering() {
    let (mut renderer, _) =
        cherenkov_cpu::Raster::init(cherenkov_cpu::RasterConfig::default()).expect("CPU renderer");
    cherenkov::testing::incremental::equivalence(&mut renderer);
}
