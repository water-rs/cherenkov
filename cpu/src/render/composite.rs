// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SIMD source-over, preserving the scalar multiplication/addition order.
//!
//! Solid paints blend packed RGBA directly, expanding varying coverage over
//! each pixel's channels. Isolation uses matching channel-vector permutations. Partial vectors use the same
//! scalar operation.

use pulp::Simd;

use super::blend::src_over;

#[expect(
    clippy::inline_always,
    reason = "SIMD operations must inline into pulp's target-feature context"
)]
#[inline(always)]
fn blocks<S: Simd>(pixels: &mut [[f32; 4]]) -> (&mut [[S::f32s; 4]], &mut [[f32; 4]]) {
    let count = pixels.len() / S::F32_LANES * S::F32_LANES;
    let (full, tail) = pixels.split_at_mut(count);
    let (vectors, _) = S::as_mut_simd_f32s(pulp::bytemuck::cast_slice_mut(full));
    (pulp::as_arrays_mut(vectors).0, tail)
}

#[expect(
    clippy::inline_always,
    reason = "SIMD operations must inline into pulp's target-feature context"
)]
#[inline(always)]
fn over<S: Simd>(simd: S, destination: [S::f32s; 4], source: [S::f32s; 4]) -> [S::f32s; 4] {
    let inverse = simd.sub_f32s(simd.splat_f32s(1.0), source[3]);
    std::array::from_fn(|channel| {
        simd.add_f32s(
            simd.mul_f32s(destination[channel], inverse),
            source[channel],
        )
    })
}

/// A constant source has one inverse alpha for every channel, so its pixels
/// stay interleaved. There is no reason to transpose the destination.
#[expect(
    clippy::inline_always,
    reason = "packed source-over must inline into the SIMD target-feature context"
)]
#[inline(always)]
pub fn constant<S: Simd>(simd: S, pixels: &mut [[f32; 4]], color: [f32; 4]) {
    let (pixels, tail) = blocks::<S>(pixels);
    let source = simd.interleave_shfl_f32s(color.map(|channel| simd.splat_f32s(channel)));
    if color[3].to_bits() == 1.0_f32.to_bits() {
        pixels.fill(source);
        tail.fill(color);
    } else {
        let inverse = simd.splat_f32s(1.0 - color[3]);
        for pixel in pixels {
            for (destination, source) in pixel.iter_mut().zip(source) {
                *destination = simd.add_f32s(simd.mul_f32s(*destination, inverse), source);
            }
        }
        for pixel in tail {
            *pixel = src_over(*pixel, color);
        }
    }
}

/// Repeat each coverage value over its packed RGBA channels. Keeping pixels
/// packed avoids transposing the much larger destination read/write stream.
#[expect(
    clippy::inline_always,
    reason = "coverage expansion must inline into the SIMD target-feature context"
)]
#[inline(always)]
fn packed_coverage<S: Simd>(simd: S, coverage: S::f32s) -> [S::f32s; 4] {
    let mut pixels = [simd.splat_f32s(0.0); 4];
    let values: &[f32] = pulp::bytemuck::cast_slice(std::slice::from_ref(&coverage));
    let channels: &mut [[f32; 4]] = pulp::bytemuck::cast_slice_mut(&mut pixels);
    for (pixel, &value) in channels.iter_mut().zip(values) {
        *pixel = [value; 4];
    }
    pixels
}

/// Composite a solid colour through a scalar coverage span.
/// Coverage and destination have matching lengths by construction.
#[expect(
    clippy::inline_always,
    reason = "SIMD operations must inline into pulp's target-feature context"
)]
#[inline(always)]
pub fn solid_span<S: Simd>(simd: S, pixels: &mut [[f32; 4]], coverage: &[f32], color: [f32; 4]) {
    let (pixels, tail) = blocks::<S>(pixels);
    let count = pixels.len() * S::F32_LANES;
    let (coverage_vectors, _) = S::as_simd_f32s(&coverage[..count]);
    let colors = simd.interleave_shfl_f32s(color.map(|channel| simd.splat_f32s(channel)));
    let alpha = simd.splat_f32s(color[3]);
    let one = simd.splat_f32s(1.0);
    let zero = simd.splat_f32s(0.0);
    for (pixel, &coverage) in pixels.iter_mut().zip(coverage_vectors) {
        let coverage = packed_coverage(simd, coverage);
        for ((destination, color), coverage) in pixel.iter_mut().zip(colors).zip(coverage) {
            let source = simd.mul_f32s(color, coverage);
            let inverse = simd.sub_f32s(one, simd.mul_f32s(alpha, coverage));
            let result = simd.add_f32s(simd.mul_f32s(*destination, inverse), source);
            let active = simd.greater_than_f32s(coverage, zero);
            *destination = simd.select_f32s(active, result, *destination);
        }
    }
    for (pixel, &coverage) in tail.iter_mut().zip(&coverage[count..]) {
        if coverage > 0.0 {
            *pixel = src_over(*pixel, color.map(|channel| channel * coverage));
        }
    }
}

/// Composite an isolated linear source-over layer with its group opacity.
#[expect(
    clippy::inline_always,
    reason = "SIMD operations must inline into pulp's target-feature context"
)]
#[inline(always)]
pub fn isolate<S: Simd>(simd: S, pixels: &mut [[f32; 4]], source: &[[f32; 4]], opacity: f32) {
    let (pixels, tail) = blocks::<S>(pixels);
    let count = pixels.len() * S::F32_LANES;
    let (source_vectors, _) = S::as_simd_f32s(pulp::bytemuck::cast_slice(&source[..count]));
    let (source_blocks, _) = pulp::as_arrays::<4, _>(source_vectors);
    let alpha = simd.splat_f32s(opacity);
    for (pixel, &source) in pixels.iter_mut().zip(source_blocks) {
        let destination = simd.deinterleave_shfl_f32s(*pixel);
        let source = simd
            .deinterleave_shfl_f32s(source)
            .map(|channel| simd.mul_f32s(channel, alpha));
        let transparent = simd.equal_f32s(source[3], simd.splat_f32s(0.0));
        let result = over(simd, destination, source);
        *pixel = simd.interleave_shfl_f32s(std::array::from_fn::<_, 4, _>(|channel| {
            simd.select_f32s(transparent, destination[channel], result[channel])
        }));
    }
    for (pixel, source) in tail.iter_mut().zip(&source[count..]) {
        let source = source.map(|channel| channel * opacity);
        if source[3] != 0.0 {
            *pixel = src_over(*pixel, source);
        }
    }
}
