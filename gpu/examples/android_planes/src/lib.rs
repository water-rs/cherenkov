//! On-device verification harness for cherenkov's Android
//! system-compositor planes (issue #90).

#![cfg(target_os = "android")]

mod ahb;
mod app;
mod logcat;
mod observe;
mod producer;
mod scenario;
mod text;
