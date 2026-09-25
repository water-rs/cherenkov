//! SIMD CPU kernels for colour filters.
//!
//! Every kernel here is a linear map on premultiplied RGBA, so each builds a
//! 4x4 matrix from its parameters and runs the shared SIMD loop: one `f32x4`
//! per pixel, the output accumulated column by column with fused
//! multiply-adds.

use filtrate_core::WorkingSpace;
use wide::f32x4;

/// A 4x4 matrix on premultiplied RGBA, stored as columns: the output is
/// `c[0] * r + c[1] * g + c[2] * b + c[3] * a`.
struct Matrix([f32x4; 4]);

impl Matrix {
    fn apply(&self, pixels: &mut [[f32; 4]]) {
        let [red, green, blue, alpha] = self.0;
        for pixel in pixels {
            let [r, g, b, a] = *pixel;
            let out = alpha * f32x4::splat(a);
            let out = blue.mul_add(f32x4::splat(b), out);
            let out = green.mul_add(f32x4::splat(g), out);
            *pixel = red.mul_add(f32x4::splat(r), out).to_array();
        }
    }

    /// `rgb' = keep * rgb + toward * luma(rgb)`, alpha unchanged: the shape
    /// of saturation and grayscale.
    fn luma_mix(space: &WorkingSpace, keep: f32, toward: f32) -> Self {
        let column = |channel: usize| {
            let mut column = [toward * space.luma[channel]; 4];
            column[channel] += keep;
            column[3] = 0.0;
            f32x4::new(column)
        };
        Self([
            column(0),
            column(1),
            column(2),
            f32x4::new([0.0, 0.0, 0.0, 1.0]),
        ])
    }
}

/// [`Brightness`](crate::filters::Brightness): `rgb + amount * a`.
pub fn brightness(params: [f32; 1], _space: &WorkingSpace, pixels: &mut [[f32; 4]]) {
    let amount = params[0];
    Matrix([
        f32x4::new([1.0, 0.0, 0.0, 0.0]),
        f32x4::new([0.0, 1.0, 0.0, 0.0]),
        f32x4::new([0.0, 0.0, 1.0, 0.0]),
        f32x4::new([amount, amount, amount, 1.0]),
    ])
    .apply(pixels);
}

/// [`Saturation`](crate::filters::Saturation): `mix(luma, rgb, amount)`.
pub fn saturation(params: [f32; 1], space: &WorkingSpace, pixels: &mut [[f32; 4]]) {
    let amount = params[0];
    Matrix::luma_mix(space, amount, 1.0 - amount).apply(pixels);
}

/// [`Grayscale`](crate::filters::Grayscale): `mix(rgb, luma, intensity)`.
pub fn grayscale(params: [f32; 1], space: &WorkingSpace, pixels: &mut [[f32; 4]]) {
    let intensity = params[0];
    Matrix::luma_mix(space, 1.0 - intensity, intensity).apply(pixels);
}

/// [`ColorMatrix`](crate::filters::ColorMatrix): the 3x4 matrix on
/// straight-alpha RGB, whose bias column scales with alpha on premultiplied
/// colour.
pub fn color_matrix(params: [f32; 12], _space: &WorkingSpace, pixels: &mut [[f32; 4]]) {
    let column = |index: usize, alpha: f32| {
        f32x4::new([params[index], params[4 + index], params[8 + index], alpha])
    };
    Matrix([
        column(0, 0.0),
        column(1, 0.0),
        column(2, 0.0),
        column(3, 1.0),
    ])
    .apply(pixels);
}
