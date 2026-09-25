//! Auxiliary images: the second inputs of blends, masks, maps, LUTs and
//! transitions.

extern crate alloc;

use alloc::sync::Arc;
use alloc::vec::Vec;

use filtrate_core::AuxImage;

/// An immutable RGBA8 image, bound to a stage's `aux` argument.
///
/// The executor uploads it once, as RGBA8 texels exactly as given.
#[derive(Clone, Debug)]
pub struct FilterImage {
    width: u32,
    height: u32,
    rgba8: Arc<[u8]>,
}

impl FilterImage {
    /// Creates an image from raw RGBA8 bytes, row-major.
    ///
    /// # Panics
    ///
    /// Panics when `rgba8` does not contain exactly `width * height * 4` bytes.
    #[must_use]
    pub fn from_rgba8(width: u32, height: u32, rgba8: Vec<u8>) -> Self {
        let expected_len = width as usize * height as usize * 4;
        assert_eq!(
            rgba8.len(),
            expected_len,
            "FilterImage::from_rgba8: expected {expected_len} bytes for {width}x{height} RGBA8 image, got {}",
            rgba8.len()
        );
        Self {
            width,
            height,
            rgba8: Arc::from(rgba8),
        }
    }

    /// Decodes an encoded image and converts it to RGBA8 pixels.
    ///
    /// # Errors
    ///
    /// Returns the decode error when the bytes cannot be parsed as an image.
    pub fn from_encoded(bytes: &[u8]) -> Result<Self, image::ImageError> {
        let decoded = image::load_from_memory(bytes)?;
        Ok(Self::from_dynamic_image(&decoded))
    }

    /// Converts a dynamic image into a filter image.
    #[must_use]
    pub fn from_dynamic_image(image: &image::DynamicImage) -> Self {
        let rgba = image.to_rgba8();
        let width = rgba.width();
        let height = rgba.height();
        Self {
            width,
            height,
            rgba8: Arc::from(rgba.into_raw()),
        }
    }
}

impl AuxImage for FilterImage {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn rgba8(&self) -> &[u8] {
        &self.rgba8
    }
}

/// A 3D LUT packed into the common 2D strip layout: `size` slices of
/// `size` x `size` texels side by side, blue selecting the slice.
#[derive(Clone, Debug)]
pub struct LutImage {
    image: FilterImage,
    size: u32,
}

impl LutImage {
    /// Creates a LUT from a strip image.
    ///
    /// # Panics
    ///
    /// Panics when `size < 2` or when `image` is not `size * size` by `size`
    /// texels.
    #[must_use]
    pub fn new(image: FilterImage, size: u32) -> Self {
        assert!(
            size >= 2,
            "LutImage::new: lut size must be >= 2, got {size}"
        );
        let expected_width = size * size;
        assert_eq!(
            image.width, expected_width,
            "LutImage::new: expected width {expected_width} for size {size}, got {}",
            image.width
        );
        assert_eq!(
            image.height, size,
            "LutImage::new: expected height {size} for size {size}, got {}",
            image.height
        );
        Self { image, size }
    }

    /// Creates a LUT from RGBA8 strip bytes.
    ///
    /// # Panics
    ///
    /// Panics when the bytes do not form a `size * size` by `size` strip, or
    /// when `size < 2`.
    #[must_use]
    pub fn from_rgba8(size: u32, rgba8: Vec<u8>) -> Self {
        Self::new(FilterImage::from_rgba8(size * size, size, rgba8), size)
    }

    /// Decodes a LUT strip from encoded image bytes.
    ///
    /// # Errors
    ///
    /// Returns the decode error when the bytes cannot be parsed as an image.
    ///
    /// # Panics
    ///
    /// Panics when the decoded image is not a `size * size` by `size` strip,
    /// or when `size < 2`.
    pub fn from_encoded(size: u32, encoded: &[u8]) -> Result<Self, image::ImageError> {
        Ok(Self::new(FilterImage::from_encoded(encoded)?, size))
    }

    /// The LUT cube size.
    #[must_use]
    pub const fn size(&self) -> u32 {
        self.size
    }

    /// The strip image.
    #[must_use]
    pub const fn image(&self) -> &FilterImage {
        &self.image
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_image_rejects_invalid_buffer_len() {
        let result = std::panic::catch_unwind(|| FilterImage::from_rgba8(2, 2, vec![0; 3]));
        assert!(result.is_err());
    }

    #[test]
    fn lut_image_rejects_invalid_dimensions() {
        let bad_image = FilterImage::from_rgba8(16, 15, vec![0; 16 * 15 * 4]);
        let result = std::panic::catch_unwind(|| LutImage::new(bad_image, 4));
        assert!(result.is_err());
    }
}
