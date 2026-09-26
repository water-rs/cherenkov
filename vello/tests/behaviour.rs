// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The shared cross-backend behaviour suite against `cherenkov-vello`.
//! Run with the lavapipe environment (`VK_ICD_FILENAMES` +
//! `WGPU_BACKEND=vulkan`); without an adapter every test returns early.

cherenkov::behaviour_suite! {
    backend: cherenkov_vello::Vello,
    config: cherenkov_vello::VelloConfig::default,
    uploads: true,
}
