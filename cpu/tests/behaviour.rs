// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The shared cross-backend behaviour suite against `cherenkov-cpu`.

cherenkov::behaviour_suite! {
    backend: cherenkov_cpu::Raster,
    config: cherenkov_cpu::RasterConfig::default,
    uploads: false,
}
