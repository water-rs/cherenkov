// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! HDR presentation preview (#97): renders a scene through the `f64`
//! oracle, presents it to an extended linear-P3 display at `headroom`,
//! then writes a PNG normalized to display white (pixel/headroom) — what
//! a headroom-`h` panel would show.
//!
//! `cargo run -p cherenkov-oracle --example tone_preview -- <scene-dir> <headroom> <out.png>`

use cherenkov_oracle::present::present_linear_p3;
use cherenkov_oracle::{F32Image, Renderer};
use cherenkov_scene::Scene;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().ok_or("usage: scene-dir")?);
    let headroom: f64 = args.next().ok_or("headroom")?.parse()?;
    let out = std::path::PathBuf::from(args.next().ok_or("out.png")?);
    let scene = Scene::load(&dir)?;
    let image =
        Renderer::new(scene.width as usize, scene.height as usize).render_image(&scene, &dir)?;
    let presented = present_linear_p3(headroom, &image);
    // Display-normalized: headroom h maps peak white onto 1.0.
    let normalized = cherenkov_oracle::Image {
        pixels: presented
            .pixels
            .iter()
            .map(|p| [p[0] / headroom, p[1] / headroom, p[2] / headroom, p[3]])
            .collect(),
        ..presented
    };
    F32Image::from_f64(&normalized).write_png(&out)?;
    println!("wrote {}", out.display());
    Ok(())
}
