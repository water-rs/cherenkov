//! Four-pixel source-over kernels over nonempty coverage spans.

use cherenkov::BlendSpace;
use wide::f32x4;

use super::coverage::{Coverage, Span, SpanKind};
use super::paint::{PaintData, convert_pixel};

#[expect(
    clippy::cast_precision_loss,
    reason = "device pixel coordinates fit f32"
)]
pub fn shade(
    pixels: &mut [[f32; 4]],
    paint: &PaintData,
    origin: (usize, usize),
    space: BlendSpace,
    span: &Span,
    coverage: &Coverage,
) {
    if let (PaintData::Solid(color), SpanKind::Constant(area)) = (paint, &span.kind)
        && (color[3] * area).to_bits() == 1.0_f32.to_bits()
    {
        let color = if space == BlendSpace::Linear {
            *color
        } else {
            convert_pixel(*color, true)
        };
        pixels.fill(color);
        return;
    }
    let samples = coverage.samples(span);
    let area_at = |index| match span.kind {
        SpanKind::Constant(area) => area,
        SpanKind::Samples(_) => samples[index],
    };
    let (chunks, remainder) = pixels.as_chunks_mut::<4>();
    for (block, pixels) in chunks.iter_mut().enumerate() {
        let index = block * 4;
        let x = f32x4::from(std::array::from_fn(|lane| {
            (origin.0 + index + lane) as f32 + 0.5
        }));
        let mut source = paint.eval4(x, f32x4::splat(origin.1 as f32 + 0.5));
        let area = f32x4::from(std::array::from_fn(|lane| area_at(index + lane)));
        source = source.map(|channel| channel * area);
        if space == BlendSpace::SrgbEncoded {
            let rgba = f32x4::transpose(source).map(|pixel| convert_pixel(pixel.to_array(), true));
            source = f32x4::transpose(rgba.map(f32x4::from));
        }
        let inverse = f32x4::ONE - source[3];
        let destination = f32x4::transpose(std::array::from_fn(|lane| f32x4::from(pixels[lane])));
        let result = f32x4::transpose(std::array::from_fn(|channel| {
            destination[channel].mul_add(inverse, source[channel])
        }));
        for (pixel, result) in pixels.iter_mut().zip(result) {
            *pixel = result.to_array();
        }
    }
    let start = span.len as usize - remainder.len();
    for (offset, pixel) in remainder.iter_mut().enumerate() {
        let index = start + offset;
        let mut source = paint.eval((origin.0 + index) as f32 + 0.5, origin.1 as f32 + 0.5);
        source = source.map(|channel| channel * area_at(index));
        if space == BlendSpace::SrgbEncoded {
            source = convert_pixel(source, true);
        }
        let inverse = 1.0 - source[3];
        for channel in 0..4 {
            pixel[channel] = pixel[channel].mul_add(inverse, source[channel]);
        }
    }
}
