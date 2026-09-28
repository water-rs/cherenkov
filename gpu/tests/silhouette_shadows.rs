// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! General silhouette convolution.
#[path = "../../tests/common/silhouette_shadows.rs"]
mod common;
#[test]
fn live_shadows_keep_offscreen_contributors_and_outer_clips() {
    common::retained_and_padded::<cherenkov_gpu::Gpu>(cherenkov_gpu::GpuConfig::default());
}
