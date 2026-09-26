// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `cherenkov-bench` binary entry — forwards argv unchanged to
//! [`cherenkov_bench::cherenkov_bench_run`], the same entry point the
//! iOS host app calls, so binary and embedded library share one path.

use std::ffi::{CString, c_char, c_int};
use std::process::ExitCode;

fn main() -> ExitCode {
    let c_args: Vec<CString> = std::env::args_os()
        .map(|arg| {
            CString::new(arg.as_encoded_bytes()).expect("process argv contains an interior NUL")
        })
        .collect();
    let argv: Vec<*const c_char> = c_args.iter().map(|arg| arg.as_ptr()).collect();
    // SAFETY: `argv` points to `argv.len()` NUL-terminated C strings
    // kept alive by `c_args` for the duration of the call.
    let code = unsafe {
        cherenkov_bench::cherenkov_bench_run(
            c_int::try_from(argv.len()).expect("argc overflows c_int"),
            argv.as_ptr(),
        )
    };
    ExitCode::from(u8::try_from(code).unwrap_or(u8::MAX))
}
