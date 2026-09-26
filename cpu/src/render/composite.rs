// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Channel-wise SIMD source-over, preserving the scalar multiplication/addition
//! order.
//!
//! Each vector holds one channel from consecutive pixels; shuffles only
//! transpose their layout. Partial vectors use the same scalar operation.

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

/// Put coverage in the same lane order as a four-channel pixel block.
/// Pulp's shuffle transpose may permute pixels within each channel vector.
/// Transposing coverage through that same layout keeps each value with its pixel.
#[expect(
    clippy::inline_always,
    reason = "the coverage transpose must inline into the SIMD target-feature context"
)]
#[inline(always)]
fn coverage_lanes<S: Simd>(simd: S, coverage: S::f32s) -> S::f32s {
    let mut pixels = [simd.splat_f32s(0.0); 4];
    let values: &[f32] = pulp::bytemuck::cast_slice(std::slice::from_ref(&coverage));
    let channels: &mut [[f32; 4]] = pulp::bytemuck::cast_slice_mut(&mut pixels);
    for (pixel, &value) in channels.iter_mut().zip(values) {
        pixel[0] = value;
    }
    simd.deinterleave_shfl_f32s(pixels)[0]
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
    let colors = color.map(|channel| simd.splat_f32s(channel));
    for (pixel, &coverage) in pixels.iter_mut().zip(coverage_vectors) {
        let coverage = coverage_lanes(simd, coverage);
        let destination = simd.deinterleave_shfl_f32s(*pixel);
        let source = colors.map(|channel| simd.mul_f32s(channel, coverage));
        let active = simd.greater_than_f32s(coverage, simd.splat_f32s(0.0));
        let result = over(simd, destination, source);
        *pixel = simd.interleave_shfl_f32s(std::array::from_fn::<_, 4, _>(|channel| {
            simd.select_f32s(active, result[channel], destination[channel])
        }));
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
