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
    /// `Rgba16F` texels convert f16 -> f64 directly — never through 8 bits —
    /// and keep values outside `[0, 1]`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "decoded working pixels use f32"
    )]
    pub fn decode(image: &ImageUpload) -> Result<Self, ResourceError> {
        if image.width == 0 || image.height == 0 {
            return Err(ResourceError::Image(
                "CPU uploads require nonempty images".into(),
            ));
        }
        let bytes_per_texel = match image.format {
            ImageFormat::Rgba8 => 4,
            ImageFormat::Rgba16F => 8,
            format => {
                return Err(ResourceError::Image(format!(
                    "unsupported image format {format:?}"
                )));
            }
        };
        // u8 data can be premultiplied-encodable only up to 1.0; f16 texels
        // keep extended values, so their un-premultiply is unbounded.
        let unpremul_max = match image.format {
            ImageFormat::Rgba8 => 1.0,
            _ => f64::INFINITY,
        };
        let count = usize::try_from(image.width)
            .ok()
            .zip(usize::try_from(image.height).ok())
            .and_then(|(width, height)| width.checked_mul(height))
            .and_then(|count| count.checked_mul(bytes_per_texel));
        if count != Some(image.data.len()) {
            return Err(ResourceError::Image(
                "image dimensions and byte length disagree".into(),
            ));
        }
        let mut pixels = Vec::with_capacity(image.data.len() / bytes_per_texel);
        let mut texel = |encoded: [f64; 4]| {
            let alpha = encoded[3];
            if alpha == 0.0 {
                pixels.push([0.0; 4]);
                return;
            }
            let linear = std::array::from_fn(|channel| {
                let straight = if image.premultiplied {
                    (encoded[channel] / alpha).min(unpremul_max)
                } else {
                    encoded[channel]
                };
                if image.color_space == ImageColorSpace::LinearSrgb
                    || image.color_space == ImageColorSpace::LinearP3
                {
                    straight
                } else if straight <= 0.04045 {
                    straight / 12.92
                } else {
                    ((straight + 0.055) / 1.055).powf(2.4)
                }
            });
            let working = if image.color_space == ImageColorSpace::DisplayP3
                || image.color_space == ImageColorSpace::LinearP3
            {
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
        };
        match image.format {
            ImageFormat::Rgba8 => {
                for pixel in image.data.as_chunks::<4>().0 {
                    texel(pixel.map(|v| f64::from(v) / 255.0));
                }
            }
            ImageFormat::Rgba16F => {
                for pixel in image.data.as_chunks::<8>().0 {
                    texel(std::array::from_fn(|channel| {
                        f64::from(half::f16::from_le_bytes([
                            pixel[2 * channel],
                            pixel[2 * channel + 1],
                        ]))
                    }));
                }
            }
            _ => unreachable!("format validated above"),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A `LinearP3` upload is the working space already: no transfer
    /// function and no primaries matrix — the byte value passes through.
    #[expect(
        clippy::float_cmp,
        reason = "255/255 decodes to exactly 1.0 — an exact assertion"
    )]
    #[test]
    fn linear_p3_upload_decodes_as_identity() {
        let image = ImageUpload {
            width: 1,
            height: 1,
            data: vec![255, 0, 0, 255].into(),
            color_space: ImageColorSpace::LinearP3,
            premultiplied: false,
            format: ImageFormat::Rgba8,
        };
        let decoded = CpuImage::decode(&image).expect("decode");
        assert_eq!(decoded.pixels[0], [1.0, 0.0, 0.0, 1.0]);
    }
}
