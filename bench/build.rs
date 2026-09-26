// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Emits `DEP_*_VERSION` environment variables with the exact dependency
//! versions resolved in `Cargo.lock`, for adapter provenance reporting.

use std::path::Path;

/// Reads `(name, version, git-rev)` tuples out of a `Cargo.lock`; the rev is
/// the `#sha` fragment of a `git+` source, `None` for registry packages.
fn lockfile_packages(text: &str) -> Vec<(String, String, Option<String>)> {
    let mut out = Vec::new();
    let (mut name, mut version, mut rev) = (None::<String>, None::<String>, None);
    let mut flush =
        |name: &mut Option<String>, version: &mut Option<String>, rev: &mut Option<String>| {
            if let (Some(n), Some(v)) = (name.take(), version.take()) {
                out.push((n, v, rev.take()));
            }
        };
    for line in text.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            flush(&mut name, &mut version, &mut rev);
        } else if let Some(v) = line
            .strip_prefix("name = \"")
            .and_then(|s| s.strip_suffix('"'))
        {
            name = Some(v.to_owned());
        } else if let Some(v) = line
            .strip_prefix("version = \"")
            .and_then(|s| s.strip_suffix('"'))
        {
            version = Some(v.to_owned());
        } else if let Some(v) = line.strip_prefix("source = \"git+") {
            rev = v
                .rsplit('#')
                .next()
                .and_then(|s| s.strip_suffix('"'))
                .map(str::to_owned);
        }
    }
    flush(&mut name, &mut version, &mut rev);
    out
}

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let lock = Path::new(&manifest).join("../Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    let Ok(text) = std::fs::read_to_string(&lock) else {
        return;
    };
    let packages = lockfile_packages(&text);
    for (pkg, env) in [
        ("vello", "DEP_VELLO"),
        ("vello_hybrid", "DEP_VELLO_HYBRID"),
        ("vello_cpu", "DEP_VELLO_CPU"),
        ("skia-safe", "DEP_SKIA_SAFE"),
        ("wgpu", "DEP_WGPU"),
        ("cherenkov-cpu", "DEP_CHERENKOV_CPU"),
    ] {
        if let Some((_, v, rev)) = packages.iter().find(|(n, _, _)| n == pkg) {
            println!("cargo:rustc-env={env}_VERSION={v}");
            if let Some(rev) = rev {
                println!("cargo:rustc-env={env}_SOURCE_REV={rev}");
            }
        }
    }
}
