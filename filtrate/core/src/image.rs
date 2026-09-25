//! Auxiliary images a filter provides to its stages.

/// An image a filter binds to one of its stages' `aux` arguments.
///
/// The executor uploads it once, as RGBA8 texels exactly as given (no colour
/// conversion and no premultiplication), and samples it with nearest
/// filtering.
pub trait AuxImage {
    /// Width in pixels.
    fn width(&self) -> u32;
    /// Height in pixels.
    fn height(&self) -> u32;
    /// `width * height` RGBA8 texels, row-major.
    fn rgba8(&self) -> &[u8];
}

/// Sink for [`Filter::visit_images`](crate::Filter::visit_images).
pub trait ImageVisitor {
    /// Visits image `index` of the flattened image list.
    fn visit<I: AuxImage + ?Sized>(&mut self, index: usize, image: &I);
}
