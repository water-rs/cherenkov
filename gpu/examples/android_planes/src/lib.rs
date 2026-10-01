//! On-device verification harness for cherenkov's Android
//! system-compositor planes (issue #90).

pub mod scenario;

#[cfg(target_os = "android")]
mod ahb;
#[cfg(target_os = "android")]
mod app;
#[cfg(target_os = "android")]
mod logcat;
#[cfg(target_os = "android")]
mod observe;
#[cfg(target_os = "android")]
mod producer;
#[cfg(target_os = "android")]
mod text;
