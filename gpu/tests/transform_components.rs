// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Component transforms reuse recorded content on the gpu backend.
#[path = "../../tests/common/component_animation.rs"]
mod common;

#[test]
fn live_components_reuse_recorded_content() {
    common::component_animation::<cherenkov_gpu::Gpu>(cherenkov_gpu::GpuConfig::default());
}
