//! The `cherenkov-planes` heartbeat and engine-side messages through the
//! Android logger.

use std::ffi::{CStr, CString, c_char, c_int};

// ndk-sys declares the symbol without a `#[link]`; this crate supplies it.
#[link(name = "log")]
unsafe extern "C" {
    fn __android_log_write(priority: c_int, tag: *const c_char, text: *const c_char) -> c_int;
}

const TAG: &CStr = c"cherenkov-planes";
const ENGINE_TAG: &CStr = c"cherenkov";

const INFO: c_int = 4;
const WARN: c_int = 5;
const ERROR: c_int = 6;

/// Writes `line` under the `cherenkov-planes` tag at INFO: the per-second
/// scenario heartbeat the verification reads.
pub fn line(line: &str) {
    write(INFO, TAG, line);
}

/// Writes `line` under `cherenkov` at WARN.
pub fn warn(line: &str) {
    write(WARN, ENGINE_TAG, line);
}

/// Writes `line` under `cherenkov` at ERROR.
pub fn error(line: &str) {
    write(ERROR, ENGINE_TAG, line);
}

fn write(priority: c_int, tag: &CStr, line: &str) {
    let Ok(text) = CString::new(line) else {
        return;
    };
    unsafe {
        __android_log_write(priority, tag.as_ptr(), text.as_ptr());
    }
}
