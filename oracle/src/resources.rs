// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Decoded scene resources: PNG images become premultiplied linear
//! Display P3 `f64` images; font blobs stay as byte caches for `skrifa`.

use std::collections::HashMap;
use std::path::PathBuf;

use cherenkov_scene::{ResourceHash, Scene, SceneError};

use crate::color::{linear_srgb_to_linear_p3, srgb_decode};
use crate::image::Image;

/// Lazily decoded `resources/` blobs for one scene directory.
pub struct Resources {
    dir: PathBuf,
    images: HashMap<ResourceHash, Image>,
    fonts: HashMap<ResourceHash, Vec<u8>>,
}

impl Resources {
    /// Resources under `scene_dir` (its `resources/` directory).
    #[must_use]
    pub fn new(scene_dir: PathBuf) -> Self {
        Self {
            dir: scene_dir,
            images: HashMap::new(),
            fonts: HashMap::new(),
        }
    }

    fn blob(&self, hash: ResourceHash) -> Result<Vec<u8>, SceneError> {
        Scene::resource(&self.dir, hash)
    }

    /// The decoded image for `hash`, decoding on first use.
    ///
    /// PNGs carry sRGB data; every colour type is handled by
    /// [`crate::image::decode_png_rgba8`], then decoded to linear and
    /// converted linear-sRGB → linear-P3 into the premultiplied `f64`
    /// working image — the same conversion the bench adapters apply on
    /// readback.
    ///
    /// # Errors
    /// [`SceneError::MissingResource`] if the blob is absent, [`SceneError::Io`]
    /// on read or decode failure.
    /// # Panics
    /// Never — the cache entry was just inserted on the miss path.
    pub fn image(&mut self, hash: ResourceHash) -> Result<&Image, SceneError> {
        if !self.images.contains_key(&hash) {
            let bytes = self.blob(hash)?;
            let (width, height, rgba) = crate::image::decode_png_rgba8(&bytes).map_err(|e| {
                SceneError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e.to_string(),
                ))
            })?;
            let mut pixels = Vec::with_capacity(width as usize * height as usize);
            for px in rgba.as_chunks::<4>().0 {
                let a = f64::from(px[3]) / 255.0;
                let lin_p3 = linear_srgb_to_linear_p3([
                    srgb_decode(f64::from(px[0]) / 255.0),
                    srgb_decode(f64::from(px[1]) / 255.0),
                    srgb_decode(f64::from(px[2]) / 255.0),
                ]);
                pixels.push([a * lin_p3[0], a * lin_p3[1], a * lin_p3[2], a]);
            }
            self.images.insert(
                hash,
                Image {
                    width: width as usize,
                    height: height as usize,
                    pixels,
                },
            );
        }
        Ok(self.images.get(&hash).unwrap())
    }

    /// The font bytes for `hash`, loading on first use.
    ///
    /// # Errors
    /// [`SceneError`] on missing/unreadable resource.
    /// # Panics
    /// Never — the cache entry was just inserted on the miss path.
    pub fn font(&mut self, hash: ResourceHash) -> Result<&Vec<u8>, SceneError> {
        if !self.fonts.contains_key(&hash) {
            let bytes = self.blob(hash)?;
            self.fonts.insert(hash, bytes);
        }
        Ok(self.fonts.get(&hash).unwrap())
    }
}
