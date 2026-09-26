//! Shared-device native presentation preserves extended color and resize ownership.

use cherenkov::{Engine, FrameTime, WorkingColor};
use cherenkov_gpu::{
    Gpu, GpuConfig,
    interop::{
        OutputAlpha, OutputColor, Presenter, SharedDevice, TextureOutput, TextureTarget, wgpu,
    },
};

#[test]
fn exported_texture_preserves_hdr_and_updates_after_resize()
-> Result<(), Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;
    let engine = Engine::<Gpu>::new(GpuConfig {
        device: Some(SharedDevice {
            instance,
            adapter,
            device: device.clone(),
            queue: queue.clone(),
        }),
        ..GpuConfig::default()
    })?;
    let (target, textures) = TextureTarget::new((16, 16));
    let source = engine.surface(target)?;
    source.clear_color(WorkingColor::new([4.0, 0.5, 0.0, 0.5]));
    let texture = textures.try_recv()?;
    let (target, destinations) = TextureTarget::new((16, 16));
    let destination = engine.surface(target)?;
    let output = destinations.try_recv()?;
    engine.render(FrameTime::now())?;
    let mut presenter = Presenter::new(&device);
    presenter.texture(
        &device,
        &queue,
        &texture.create_view(&wgpu::TextureViewDescriptor::default()),
        TextureOutput {
            texture: &output,
            color: OutputColor::LinearDisplayP3,
            alpha: OutputAlpha::Premultiplied,
        },
    );
    let pixel = destination.readback()?.pixels[0];
    for (actual, expected) in pixel.into_iter().zip([2.0, 0.25, 0.0, 0.5]) {
        assert!(
            (actual - expected).abs() < 0.001,
            "native output {actual} != {expected}"
        );
    }
    source.clear_color(WorkingColor::new([1.0, 1.0, 1.0, 0.5]));
    engine.render(FrameTime::now())?;
    presenter.texture(
        &device,
        &queue,
        &texture.create_view(&wgpu::TextureViewDescriptor::default()),
        TextureOutput {
            texture: &output,
            color: OutputColor::Srgb,
            alpha: OutputAlpha::Premultiplied,
        },
    );
    let pixel = destination.readback()?.pixels[0];
    for actual in pixel {
        assert!(
            (actual - 0.5).abs() < 0.001,
            "sRGB premultiplication follows transfer encoding: {actual}"
        );
    }
    source.resize((8, 4))?;
    engine.render(FrameTime::now())?;
    let resized = textures.try_recv()?;
    assert_eq!((resized.width(), resized.height()), (8, 4));
    assert_eq!(
        (texture.width(), texture.height()),
        (16, 16),
        "host retains old texture until released"
    );
    assert!(
        textures.try_recv().is_err(),
        "only allocation changes publish a texture"
    );
    Ok(())
}
