// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::*;
use cherenkov::{Draw as _, FrameTime, Offscreen, SurfaceTree};

struct Harness {
    renderer: VelloRenderer,
    tree: SurfaceTree,
    surface: SurfaceId,
}

impl Harness {
    fn new() -> Option<Self> {
        let (mut renderer, _) = init(VelloConfig::default()).ok()?;
        let surface = SurfaceId::new(1);
        renderer
            .create_surface(
                surface,
                Offscreen::new((32, 32), OffscreenFormat::LinearF16).into(),
            )
            .expect("surface");
        Some(Self {
            renderer,
            tree: SurfaceTree::new(),
            surface,
        })
    }

    fn content(&mut self, draw: impl FnOnce(&mut cherenkov::StaticRecorder)) {
        let picture = cherenkov::Picture::record(draw);
        self.renderer.set_content(
            self.surface,
            self.tree.root(),
            Some(ContentOp::Replace(picture.display_list().clone())),
        );
    }

    fn render(&mut self) -> Result<Redraw, RenderError> {
        let frames = [SurfaceFrame {
            id: self.surface,
            size: (32, 32),
            display: cherenkov::Display::default(),
            clear: cherenkov::WorkingColor::TRANSPARENT,
            changed: true,
            tree: &self.tree,
        }];
        self.renderer.render(
            &cherenkov::Frame {
                time: FrameTime::now(),
                surfaces: &frames,
            },
            &mut FrameStats::default(),
        )
    }

    fn cache(&self) -> &LayerCache {
        &self.renderer.surfaces[&self.surface].layers[&self.tree.root()]
    }

    fn bound(&mut self, image: &peniko::ImageData) -> bool {
        let old = self.renderer.vello.override_image(image, None);
        let bound = old.is_some();
        self.renderer.vello.override_image(image, old);
        bound
    }
}

#[test]
fn removing_a_shader_invalidates_cached_fragments() {
    let Some(mut h) = Harness::new() else {
        return;
    };
    h.renderer.shaders.add(&h.renderer.device, 7, &ShaderSpec {
        source: "@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { return vec4<f32>(uv, 0.0, 1.0); }".into(),
        animated: false,
    }).expect("shader");
    h.content(|c| {
        c.fill(
            kurbo::Rect::new(0., 0., 8., 8.),
            cherenkov::ShaderPaint {
                shader: cherenkov::ShaderId::new(7),
                uniforms: vec![],
            },
        )
    });
    h.render().expect("render");
    assert!(h.cache().fragment.is_some());
    let image = h.cache().shader_uses[0].image.clone();
    assert!(h.bound(&image));
    h.renderer.remove_shader(cherenkov::ShaderId::new(7));
    assert!(h.cache().fragment.is_none());
    assert!(h.cache().shader_uses.is_empty());
    assert!(!h.bound(&image));
    assert!(matches!(h.render(), Err(RenderError::Shader(_))));
}

#[test]
fn removing_an_image_invalidates_cached_fragments() {
    let Some(mut h) = Harness::new() else {
        return;
    };
    h.renderer.images.insert(
        9,
        peniko::ImageData {
            data: peniko::Blob::new(Arc::new(SharedBytes(Arc::from([255u8; 4])))),
            format: peniko::ImageFormat::Rgba8,
            alpha_type: peniko::ImageAlphaType::AlphaPremultiplied,
            width: 1,
            height: 1,
        },
    );
    h.content(|c| {
        c.image(
            cherenkov::ImageId::new(9),
            kurbo::Rect::new(0., 0., 8., 8.),
            cherenkov::Sampling::Nearest,
        )
    });
    h.render().expect("render");
    assert!(h.cache().fragment.is_some());
    h.renderer.remove_image(ImageId::new(9));
    assert!(h.cache().fragment.is_none());
    assert!(matches!(h.render(), Err(RenderError::Image(_))));
}

struct NoopContent;
impl crate::interop::GpuContent for NoopContent {
    async fn setup(&mut self, _gpu: &interop::wgpu::Context<'_>) {}
    fn render(&mut self, _frame: &mut interop::wgpu::Frame<'_>) {}
}

fn bound_gpu_image(h: &mut Harness) -> peniko::ImageData {
    h.renderer
        .set_gpu_content(h.surface, h.tree.root(), (8, 8), NoopContent.into());
    h.render().expect("render");
    let Some(ContentData::Gpu(slot)) = &h.cache().content else {
        panic!("gpu content");
    };
    slot.ready.as_ref().expect("rendered").image.clone()
}

#[test]
fn destroying_a_surface_unbinds_its_overrides() {
    let Some(mut h) = Harness::new() else {
        return;
    };
    let image = bound_gpu_image(&mut h);
    assert!(h.bound(&image));
    h.renderer.destroy_surface(h.surface);
    assert!(!h.bound(&image));
}

#[test]
fn replace_releases_the_old_contents_binding() {
    let Some(mut h) = Harness::new() else {
        return;
    };
    let image = bound_gpu_image(&mut h);
    assert!(h.bound(&image));
    h.content(|_| {});
    assert!(!h.bound(&image));
}
