//! Procedural frames in `AHardwareBuffer`s — no media files, no assets.
//!
//! The picture is four colour quadrants, a white bar sweeping right and a
//! strip holding the frame counter in binary, so a torn or stale buffer
//! is visible immediately.

use std::mem::MaybeUninit;
use std::ptr;

use ndk_sys::{AHardwareBuffer, AHardwareBuffer_Desc, AHardwareBuffer_Planes};

use crate::scenario::Format;

/// Width of every video frame.
pub const WIDTH: u32 = 1920;
/// Height of every video frame.
pub const HEIGHT: u32 = 1080;

const BAR: u32 = 24;
const STRIP: u32 = 40;
const CELL: u32 = 24;

/// `AHardwareBuffer` usage for a video: GPU-sampled and CPU-written;
/// `overlay` adds the `COMPOSER_OVERLAY` bit plane promotion requires.
#[must_use]
pub const fn usage(overlay: bool) -> u64 {
    use ndk_sys::AHardwareBuffer_UsageFlags as U;
    let mut flags =
        U::AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE.0 | U::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN.0;
    if overlay {
        flags |= U::AHARDWAREBUFFER_USAGE_COMPOSER_OVERLAY.0;
    }
    flags
}

/// Allocates one buffer, returning the owned reference.
///
/// # Panics
/// When the device refuses a format or usage this harness requires.
#[must_use]
pub fn alloc(format: Format, overlay: bool) -> *mut AHardwareBuffer {
    let format_code = match format {
        Format::Nv12 => ndk_sys::AHardwareBuffer_Format::AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420.0,
        Format::P010 => ndk_sys::AHardwareBuffer_Format::AHARDWAREBUFFER_FORMAT_YCbCr_P010.0,
    };
    let desc = AHardwareBuffer_Desc {
        width: WIDTH,
        height: HEIGHT,
        layers: 1,
        format: format_code,
        usage: usage(overlay),
        stride: 0,
        rfu0: 0,
        rfu1: 0,
    };
    assert_ne!(
        unsafe { ndk_sys::AHardwareBuffer_isSupported(&raw const desc) },
        0,
        "{format:?} AHB is not supported at {WIDTH}x{HEIGHT}"
    );
    let mut buffer = ptr::null_mut();
    assert_eq!(
        unsafe { ndk_sys::AHardwareBuffer_allocate(&raw const desc, &raw mut buffer) },
        0,
        "{format:?} AHB allocation failed"
    );
    buffer
}

/// Releases the reference `alloc` returned.
///
/// # Safety
/// `buffer` is a live, owned `AHardwareBuffer` reference.
pub unsafe fn release(buffer: *mut AHardwareBuffer) {
    unsafe { ndk_sys::AHardwareBuffer_release(buffer) };
}

/// Writes frame `frame` of the pattern into `buffer`.
///
/// # Safety
/// `buffer` is a live `AHardwareBuffer` of `format` allocated with CPU
/// write usage, and nothing else is writing it.
pub unsafe fn fill(format: Format, buffer: *mut AHardwareBuffer, frame: u64) {
    match format {
        Format::Nv12 => unsafe { fill_nv12(buffer, frame) },
        Format::P010 => unsafe { fill_p010(buffer, frame) },
    }
}

/// The quadrant colours as linear 0..1 RGB: top-left, top-right,
/// bottom-left, bottom-right.
const QUADRANTS: [(f64, f64, f64); 4] = [
    (0.85, 0.08, 0.08),
    (0.08, 0.70, 0.08),
    (0.08, 0.15, 0.85),
    (0.55, 0.55, 0.55),
];

/// BT.709 video-range `Y'CbCr` codes for `rgb` (0..1).
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops,
    reason = "clamped video-range codes; the textbook coefficient order stays readable"
)]
fn code709(rgb: (f64, f64, f64)) -> (u8, u8, u8) {
    let (r, g, b) = rgb;
    let clamp = |v: f64, lo: f64, hi: f64| -> u8 { v.clamp(lo, hi).round() as u8 };
    (
        clamp(16.0 + 65.481 * r + 128.553 * g + 24.966 * b, 16.0, 235.0),
        clamp(128.0 - 37.797 * r - 74.203 * g + 112.0 * b, 16.0, 240.0),
        clamp(128.0 + 112.0 * r - 93.786 * g - 18.214 * b, 16.0, 240.0),
    )
}

/// BT.2020 video-range `Y'CbCr` codes for `rgb` (0..1), 10-bit left-aligned
/// in u16 as P010 stores them.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops,
    reason = "clamped video-range codes; the textbook coefficient order stays readable"
)]
fn code2020(rgb: (f64, f64, f64)) -> (u16, u16, u16) {
    let (r, g, b) = rgb;
    let yn = 0.2627 * r + 0.6780 * g + 0.0593 * b;
    let clamp = |v: f64, lo: f64, hi: f64| -> u16 { (v.clamp(lo, hi).round() as u16) << 6 };
    (
        clamp(64.0 + 876.0 * yn, 64.0, 940.0),
        clamp(
            512.0 + 896.0 * (b - yn) / (2.0 * (1.0 - 0.0593)),
            64.0,
            960.0,
        ),
        clamp(
            512.0 + 896.0 * (r - yn) / (2.0 * (1.0 - 0.2627)),
            64.0,
            960.0,
        ),
    )
}

/// The white bar's left edge on frame `frame`.
fn bar_x(frame: u64) -> usize {
    usize::try_from(frame * 8 % u64::from(WIDTH - BAR)).expect("the bar stays inside the frame")
}

/// Writes one luma row (u8 codes) of `frame` at `row` into `base`.
///
/// # Safety
/// `base` has room for `WIDTH` bytes.
unsafe fn luma_row8(base: *mut u8, row: usize, frame: u64, luma: [u8; 4]) {
    let width = WIDTH as usize;
    let strip = row >= (HEIGHT - STRIP) as usize;
    if strip {
        // Counter strip: `CELL`-wide cells, bit i of the frame index at
        // cell i — lit white when set.
        let mut x = 0usize;
        while x < width {
            let bit = frame >> ((x / CELL as usize) % 64) & 1;
            let cell = (CELL as usize).min(width - x);
            unsafe {
                base.add(x)
                    .write_bytes(if bit == 1 { 235 } else { 24 }, cell);
            };
            x += cell;
        }
        return;
    }
    let top = row < (HEIGHT / 2) as usize;
    let (left, right) = if top {
        (luma[0], luma[1])
    } else {
        (luma[2], luma[3])
    };
    unsafe {
        base.write_bytes(left, width / 2);
        base.add(width / 2).write_bytes(right, width / 2);
        let bar = bar_x(frame);
        base.add(bar).write_bytes(235, BAR as usize);
    }
}

/// Writes one luma row (u16 P010 codes) of `frame` at `row` into `base`.
///
/// # Safety
/// `base` has room for `WIDTH` u16s.
unsafe fn luma_row10(base: *mut u16, row: usize, frame: u64, luma: [u16; 4]) {
    let width = WIDTH as usize;
    let strip = row >= (HEIGHT - STRIP) as usize;
    if strip {
        for x in 0..width {
            let bit = frame >> ((x / CELL as usize) % 64) & 1;
            unsafe { *base.add(x) = if bit == 1 { 940 << 6 } else { 80 << 6 } };
        }
        return;
    }
    let top = row < (HEIGHT / 2) as usize;
    let (left, right) = if top {
        (luma[0], luma[1])
    } else {
        (luma[2], luma[3])
    };
    unsafe {
        for x in 0..width / 2 {
            *base.add(x) = left;
        }
        for x in width / 2..width {
            *base.add(x) = right;
        }
        let bar = bar_x(frame);
        for x in bar..bar + BAR as usize {
            *base.add(x) = 940 << 6;
        }
    }
}

/// Fills a `Y8Cb8Cr8_420` buffer through `AHardwareBuffer_lockPlanes`,
/// covering both the semi-planar (NV12, 2 planes) and tri-planar (3
/// planes) layouts the flexible format resolves to.
///
/// # Safety
/// As [`fill`].
#[expect(
    clippy::too_many_lines,
    reason = "the fill is a straight-line walk over the planes"
)]
unsafe fn fill_nv12(buffer: *mut AHardwareBuffer, frame: u64) {
    let mut planes = MaybeUninit::<AHardwareBuffer_Planes>::uninit();
    let rc = unsafe {
        ndk_sys::AHardwareBuffer_lockPlanes(
            buffer,
            ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN.0,
            -1,
            ptr::null_mut(),
            planes.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "AHB lockPlanes");
    let planes = unsafe { planes.assume_init() };
    assert!(
        planes.planeCount == 2 || planes.planeCount == 3,
        "Y8Cb8Cr8_420 resolved to {} planes",
        planes.planeCount
    );
    let luma: [u8; 4] = QUADRANTS.map(|rgb| code709(rgb).0);
    let chroma: [(u8, u8); 4] = QUADRANTS.map(|rgb| {
        let (_, cb, cr) = code709(rgb);
        (cb, cr)
    });
    let y_plane = planes.planes[0];
    unsafe {
        let y_base = y_plane.data.cast::<u8>();
        let y_stride = y_plane.rowStride as usize;
        for row in 0..HEIGHT as usize {
            luma_row8(y_base.add(row * y_stride), row, frame, luma);
        }
    }
    let bar = bar_x(frame);
    if planes.planeCount == 3 {
        // Tri-planar: separate Cb and Cr planes, each WIDTH/2 per row.
        for plane_at in 1..=2 {
            let chroma_plane = planes.planes[plane_at];
            let base = chroma_plane.data.cast::<u8>();
            let stride = chroma_plane.rowStride as usize;
            for row in 0..(HEIGHT / 2) as usize {
                let luma_row = row * 2;
                let strip = luma_row >= (HEIGHT - STRIP) as usize;
                unsafe {
                    let dst = base.add(row * stride);
                    if strip {
                        dst.write_bytes(128, (WIDTH / 2) as usize);
                        continue;
                    }
                    let top = luma_row < (HEIGHT / 2) as usize;
                    let (l, r) = if top {
                        (chroma[0], chroma[1])
                    } else {
                        (chroma[2], chroma[3])
                    };
                    let value = if plane_at == 1 { l.0 } else { l.1 };
                    let other = if plane_at == 1 { r.0 } else { r.1 };
                    dst.write_bytes(value, (WIDTH / 4) as usize);
                    dst.add((WIDTH / 4) as usize)
                        .write_bytes(other, (WIDTH / 4) as usize);
                    // Neutral chroma under the bar.
                    let start = bar / 2;
                    dst.add(start).write_bytes(128, (BAR / 2) as usize);
                }
            }
        }
    } else {
        // Semi-planar: interleaved (Cb, Cr) pairs at `pixelStride`.
        let uv_plane = planes.planes[1];
        let uv_stride = uv_plane.rowStride as usize;
        let uv_pixel = uv_plane.pixelStride as usize;
        unsafe {
            let uv_base = uv_plane.data.cast::<u8>();
            for row in 0..(HEIGHT / 2) as usize {
                let luma_row = row * 2;
                let strip = luma_row >= (HEIGHT - STRIP) as usize;
                let dst = uv_base.add(row * uv_stride);
                if strip {
                    for col in 0..(WIDTH / 2) as usize {
                        *dst.add(col * uv_pixel) = 128;
                        *dst.add(col * uv_pixel + 1) = 128;
                    }
                    continue;
                }
                let top = luma_row < (HEIGHT / 2) as usize;
                let (l, r) = if top {
                    (chroma[0], chroma[1])
                } else {
                    (chroma[2], chroma[3])
                };
                for col in 0..(WIDTH / 4) as usize {
                    *dst.add(col * uv_pixel) = l.0;
                    *dst.add(col * uv_pixel + 1) = l.1;
                }
                for col in (WIDTH / 4) as usize..(WIDTH / 2) as usize {
                    *dst.add(col * uv_pixel) = r.0;
                    *dst.add(col * uv_pixel + 1) = r.1;
                }
                let bar_cols = bar / 2..bar / 2 + (BAR / 2) as usize;
                for col in bar_cols {
                    *dst.add(col * uv_pixel) = 128;
                    *dst.add(col * uv_pixel + 1) = 128;
                }
            }
        }
    }
    unsafe { ndk_sys::AHardwareBuffer_unlock(buffer, ptr::null_mut()) };
}

/// Fills a `YCbCr_P010` buffer. P010 is a fixed format, so
/// `AHardwareBuffer_lockPlanes` does not apply: the layout is the
/// semiplanar one every P010 gralloc produces — a u16 luma plane at
/// `stride` pixels per row followed by an interleaved (Cb, Cr) u16 plane
/// — taken from `AHardwareBuffer_describe`.
///
/// # Safety
/// As [`fill`].
#[expect(
    clippy::cast_ptr_alignment,
    reason = "gralloc's mapped address is page-aligned and the UV offset is even"
)]
unsafe fn fill_p010(buffer: *mut AHardwareBuffer, frame: u64) {
    let mut addr = ptr::null_mut();
    let rc = unsafe {
        ndk_sys::AHardwareBuffer_lock(
            buffer,
            ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN.0,
            -1,
            ptr::null(),
            &raw mut addr,
        )
    };
    assert_eq!(rc, 0, "P010 AHB lock");
    let mut desc = AHardwareBuffer_Desc {
        width: 0,
        height: 0,
        layers: 0,
        format: 0,
        usage: 0,
        stride: 0,
        rfu0: 0,
        rfu1: 0,
    };
    unsafe { ndk_sys::AHardwareBuffer_describe(buffer, &raw mut desc) };
    // `stride` is the luma row stride in pixels (u16s).
    let stride = desc.stride as usize;
    let luma: [u16; 4] = QUADRANTS.map(|rgb| code2020(rgb).0);
    let chroma: [(u16, u16); 4] = QUADRANTS.map(|rgb| {
        let (_, cb, cr) = code2020(rgb);
        (cb, cr)
    });
    unsafe {
        let y_base = addr.cast::<u16>();
        for row in 0..HEIGHT as usize {
            luma_row10(y_base.add(row * stride), row, frame, luma);
        }
        // The interleaved chroma plane follows the luma plane.
        let uv_base = addr
            .cast::<u8>()
            .add(stride * 2 * HEIGHT as usize)
            .cast::<u16>();
        let bar = bar_x(frame);
        for row in 0..(HEIGHT / 2) as usize {
            let luma_row = row * 2;
            let strip = luma_row >= (HEIGHT - STRIP) as usize;
            let dst = uv_base.add(row * stride);
            if strip {
                for col in 0..(WIDTH / 2) as usize {
                    *dst.add(col * 2) = 512 << 6;
                    *dst.add(col * 2 + 1) = 512 << 6;
                }
                continue;
            }
            let top = luma_row < (HEIGHT / 2) as usize;
            let (l, r) = if top {
                (chroma[0], chroma[1])
            } else {
                (chroma[2], chroma[3])
            };
            for col in 0..(WIDTH / 4) as usize {
                *dst.add(col * 2) = l.0;
                *dst.add(col * 2 + 1) = l.1;
            }
            for col in (WIDTH / 4) as usize..(WIDTH / 2) as usize {
                *dst.add(col * 2) = r.0;
                *dst.add(col * 2 + 1) = r.1;
            }
            let bar_cols = bar / 2..bar / 2 + (BAR / 2) as usize;
            for col in bar_cols {
                *dst.add(col * 2) = 512 << 6;
                *dst.add(col * 2 + 1) = 512 << 6;
            }
        }
        ndk_sys::AHardwareBuffer_unlock(buffer, ptr::null_mut());
    }
}
