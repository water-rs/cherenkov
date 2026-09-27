// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Registered CPU images: immutable, shared by retained paint operands.

use cherenkov::{ImageColorSpace, ImageFormat, ImageUpload, ResourceError};

#[derive(Debug)]
pub struct CpuImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Box<[[f32; 4]]>,
}

impl CpuImage {
    pub fn bytes(&self) -> u64 {
        u64::try_from(size_of_val(&*self.pixels)).expect("image allocation fits u64")
    }

    /// Decode once on registration. Alpha is unpremultiplied in the encoded
    /// domain before transfer/primary conversion, then premultiplied in P3.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "decoded working pixels use f32"
    )]
    pub fn decode(image: &ImageUpload) -> Result<Self, ResourceError> {
        if image.format != ImageFormat::Rgba8 || image.width == 0 || image.height == 0 {
            return Err(ResourceError::Image(
                "CPU uploads require nonempty RGBA8".into(),
            ));
        }
        let count = usize::try_from(image.width)
            .ok()
            .zip(usize::try_from(image.height).ok())
            .and_then(|(width, height)| width.checked_mul(height))
            .and_then(|count| count.checked_mul(4));
        if count != Some(image.data.len()) {
            return Err(ResourceError::Image(
                "image dimensions and byte length disagree".into(),
            ));
        }
        let mut pixels = Vec::with_capacity(image.data.len() / 4);
        for pixel in image.data.as_chunks::<4>().0 {
            let alpha = f64::from(pixel[3]) / 255.0;
            if alpha == 0.0 {
                pixels.push([0.0; 4]);
                continue;
            }
            let linear = std::array::from_fn(|channel| {
                let encoded = f64::from(pixel[channel]) / 255.0;
                let straight = if image.premultiplied {
                    (encoded / alpha).min(1.0)
                } else {
                    encoded
                };
                if image.color_space == ImageColorSpace::LinearSrgb {
                    straight
                } else if straight <= 0.04045 {
                    straight / 12.92
                } else {
                    ((straight + 0.055) / 1.055).powf(2.4)
                }
            });
            let working = if image.color_space == ImageColorSpace::DisplayP3 {
                linear
            } else {
                mul(&XYZ_TO_P3, mul(&SRGB_TO_XYZ, linear))
            };
            pixels.push([
                (working[0] * alpha) as f32,
                (working[1] * alpha) as f32,
                (working[2] * alpha) as f32,
                alpha as f32,
            ]);
        }
        Ok(Self {
            width: image.width,
            height: image.height,
            pixels: pixels.into_boxed_slice(),
        })
    }
}

const SRGB_TO_XYZ: [[f64; 3]; 3] = [
    [
        0.412_390_799_265_959_5,
        0.357_584_339_383_878,
        0.180_480_788_401_834_3,
    ],
    [
        0.212_639_005_871_510_4,
        0.715_168_678_767_756,
        0.072_192_315_360_733_7,
    ],
    [
        0.019_330_818_715_591_9,
        0.119_194_779_794_626,
        0.950_532_152_249_660_7,
    ],
];
const XYZ_TO_P3: [[f64; 3]; 3] = [
    [
        2.493_496_911_941_425,
        -0.931_383_617_919_123_9,
        -0.402_710_784_450_716_8,
    ],
    [
        -0.829_488_969_561_574_7,
        1.762_664_060_318_346_3,
        0.023_624_685_841_943_6,
    ],
    [
        0.035_845_830_243_784_5,
        -0.076_172_389_268_041_8,
        0.956_884_524_007_687_2,
    ],
];
fn mul(matrix: &[[f64; 3]; 3], value: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| row[2].mul_add(value[2], row[1].mul_add(value[1], row[0] * value[0])))
}
