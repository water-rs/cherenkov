//! System-compositor planes on macOS (#90), through the public API a host
//! uses: an eligible external frame is realized on a display layer between
//! the engine's parts, the layer shows the frame's own `IOSurface` with the
//! colour the frame declares, ineligible frames stay in the engine, and the
//! system's composition of the realized tree matches the engine's own.
//!
//! The binary owns the process main thread: a view is main-thread state, and
//! a display layer makes its frames ready through the main queue. Every case
//! therefore runs on the main thread, one at a time.

use libtest_mimic::{Arguments, Trial};

fn main() {
    let mut args = Arguments::from_args();
    args.test_threads = Some(1);
    libtest_mimic::run(&args, trials()).exit();
}

#[cfg(not(target_os = "macos"))]
const fn trials() -> Vec<Trial> {
    Vec::new()
}

#[cfg(target_os = "macos")]
fn trials() -> Vec<Trial> {
    macos::trials()
}

#[cfg(target_os = "macos")]
mod macos {
    use std::ptr::NonNull;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    use cherenkov::kurbo::{Affine, Rect, RoundedRect};
    use cherenkov::{
        Display, Draw as _, Engine, FrameTime, Layer, Offscreen, OffscreenFormat, Surface,
        WorkingColor,
    };
    use cherenkov_gpu::interop::wgpu::rwh::{
        AppKitWindowHandle, DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle,
        RawWindowHandle, WindowHandle,
    };
    use cherenkov_gpu::interop::{
        ExternalFrame, FrameColor, RgbAlpha, SharedDevice, YuvRange, metal::import_texture, wgpu,
    };
    use cherenkov_gpu::{Gpu, GpuConfig, WindowTarget};
    use dispatch2::DispatchQueue;
    use libtest_mimic::Trial;
    use objc2::rc::Retained;
    use objc2::runtime::{AnyObject, NSObjectProtocol as _, ProtocolObject};
    use objc2::{MainThreadMarker, MainThreadOnly as _};
    use objc2_app_kit::NSView;
    use objc2_av_foundation::AVSampleBufferDisplayLayer;
    use objc2_core_foundation::{
        CFDictionary, CFRetained, CFRunLoop, CFString, CFType, CGAffineTransform, CGPoint, CGRect,
        CGSize, kCFRunLoopDefaultMode,
    };
    use objc2_core_video::{
        CVPixelBuffer, CVPixelBufferCreate, CVPixelBufferGetBaseAddressOfPlane,
        CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferGetHeightOfPlane,
        CVPixelBufferGetIOSurface, CVPixelBufferGetWidthOfPlane, CVPixelBufferLockBaseAddress,
        CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress, kCVImageBufferChromaLocation_Left,
        kCVImageBufferChromaLocationTopFieldKey, kCVImageBufferColorPrimaries_ITU_R_2020,
        kCVImageBufferColorPrimariesKey, kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ,
        kCVImageBufferTransferFunctionKey, kCVImageBufferYCbCrMatrix_ITU_R_2020,
        kCVImageBufferYCbCrMatrixKey, kCVPixelBufferIOSurfacePropertiesKey,
        kCVPixelBufferMetalCompatibilityKey, kCVPixelFormatType_32BGRA,
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, kCVReturnSuccess,
    };
    use objc2_metal::{
        MTLCommandBuffer as _, MTLCommandQueue, MTLDevice, MTLPixelFormat, MTLRegion,
        MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
    };
    use objc2_quartz_core::{CALayer, CAMetalLayer, CARenderer, CATransaction};

    /// The surface in device pixels, its scale, and the video in pixels.
    const SIZE: (u32, u32) = (96, 64);
    const SCALE: f64 = 2.0;
    const VIDEO: (usize, usize) = (48, 32);

    pub fn trials() -> Vec<Trial> {
        let case = |name: &str, run: fn()| {
            Trial::test(name, move || {
                run();
                Ok(())
            })
        };
        vec![
            case(
                "the_realized_tree_puts_the_plane_between_its_parts",
                the_realized_tree_puts_the_plane_between_its_parts,
            ),
            case(
                "a_promoted_frame_shows_its_own_surface_and_declared_colour",
                a_promoted_frame_shows_its_own_surface_and_declared_colour,
            ),
            case(
                "a_frame_whose_surface_disagrees_with_its_range_stays_in_the_engine",
                a_frame_whose_surface_disagrees_with_its_range_stays_in_the_engine,
            ),
            case(
                "planes_of_two_surfaces_stay_in_the_engine",
                planes_of_two_surfaces_stay_in_the_engine,
            ),
            case(
                "a_frame_without_an_iosurface_stays_in_the_engine",
                a_frame_without_an_iosurface_stays_in_the_engine,
            ),
            case(
                "only_opaque_rgb_frames_are_promoted",
                only_opaque_rgb_frames_are_promoted,
            ),
            case(
                "promoted_composition_matches_engine_composition",
                promoted_composition_matches_engine_composition,
            ),
            case(
                "a_bt709_frame_stays_in_the_engine_and_matches",
                a_bt709_frame_stays_in_the_engine_and_matches,
            ),
            case(
                "a_translucent_layer_above_stays_in_the_engine_and_matches",
                a_translucent_layer_above_stays_in_the_engine_and_matches,
            ),
        ]
    }

    /// The engine's device, shared with the test so frames live on it.
    struct Metal {
        shared: SharedDevice,
        raw: Retained<ProtocolObject<dyn MTLDevice>>,
    }

    fn metal() -> Metal {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("a Metal adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::PASSTHROUGH_SHADERS,
            ..wgpu::DeviceDescriptor::default()
        }))
        .expect("a Metal device");
        // SAFETY: the guard is dropped before the device.
        let raw = unsafe { device.as_hal::<wgpu::hal::metal::Api>() }
            .expect("a Metal device")
            .raw_device()
            .clone();
        Metal {
            shared: SharedDevice {
                instance,
                adapter,
                device,
                queue,
            },
            raw,
        }
    }

    /// A Metal-compatible, `IOSurface`-backed pixel buffer of `format`.
    fn surface_buffer(width: usize, height: usize, format: u32) -> CFRetained<CVPixelBuffer> {
        let empty = CFDictionary::<CFString, CFType>::from_slices(&[], &[]);
        // SAFETY: CoreVideo's attribute keys and the boolean are immutable
        // statics.
        let attributes = unsafe {
            CFDictionary::<CFString, CFType>::from_slices(
                &[
                    kCVPixelBufferIOSurfacePropertiesKey,
                    kCVPixelBufferMetalCompatibilityKey,
                ],
                &[
                    &empty,
                    objc2_core_foundation::kCFBooleanTrue.expect("kCFBooleanTrue"),
                ],
            )
        };
        let mut out = std::ptr::null_mut();
        // SAFETY: the attributes are a CoreVideo attribute dictionary and
        // `out` receives a +1 pixel buffer.
        let status = unsafe {
            CVPixelBufferCreate(
                None,
                width,
                height,
                format,
                Some(attributes.as_opaque()),
                NonNull::from(&mut out),
            )
        };
        assert_eq!(status, kCVReturnSuccess, "CVPixelBufferCreate");
        // SAFETY: the create call returned a +1 pixel buffer.
        unsafe { CFRetained::from_raw(NonNull::new(out).expect("a pixel buffer")) }
    }

    /// Writes each row of each of `planes` planes through `fill(plane, row)`.
    fn fill(buffer: &CVPixelBuffer, planes: usize, fill: impl Fn(usize, &mut [u8])) {
        // SAFETY: the buffer is unlocked and locked once here.
        unsafe { CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(0)) };
        for plane in 0..planes {
            let base = CVPixelBufferGetBaseAddressOfPlane(buffer, plane).cast::<u8>();
            let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane);
            for y in 0..CVPixelBufferGetHeightOfPlane(buffer, plane) {
                // SAFETY: the plane is locked and `height * stride` bytes long.
                let row = unsafe { std::slice::from_raw_parts_mut(base.add(y * stride), stride) };
                fill(plane, row);
            }
        }
        // SAFETY: locked above with the same flags.
        unsafe { CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(0)) };
    }

    /// Plane `plane` of `buffer`'s `IOSurface` as a texture on the engine's
    /// device.
    fn plane_texture(
        metal: &Metal,
        buffer: &CVPixelBuffer,
        plane: usize,
        (mtl, format): (MTLPixelFormat, wgpu::TextureFormat),
    ) -> wgpu::Texture {
        let surface = CVPixelBufferGetIOSurface(Some(buffer)).expect("an IOSurface-backed buffer");
        // SAFETY: the descriptor is fully specified.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                mtl,
                CVPixelBufferGetWidthOfPlane(buffer, plane),
                CVPixelBufferGetHeightOfPlane(buffer, plane),
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::ShaderRead);
        let raw = metal
            .raw
            .newTextureWithDescriptor_iosurface_plane(&descriptor, &surface, plane)
            .expect("an IOSurface plane texture");
        // SAFETY: the texture is on the engine's device and holds `format`.
        unsafe { import_texture(&metal.shared.device, raw, format) }
    }

    const LUMA: (MTLPixelFormat, wgpu::TextureFormat) =
        (MTLPixelFormat::R8Uint, wgpu::TextureFormat::R8Uint);
    const CHROMA: (MTLPixelFormat, wgpu::TextureFormat) =
        (MTLPixelFormat::RG8Uint, wgpu::TextureFormat::Rg8Uint);

    /// An 8-bit studio-range NV12 buffer: luma ramps left to right and chroma
    /// is a fixed warm tint, every pixel inside the BT.709 gamut.
    fn nv12_buffer((width, height): (usize, usize)) -> CFRetained<CVPixelBuffer> {
        let buffer = surface_buffer(
            width,
            height,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        );
        fill(&buffer, 2, |plane, row| {
            if plane == 0 {
                for (x, v) in row[..width].iter_mut().enumerate() {
                    *v = u8::try_from(48 + x * 152 / width).expect("studio luma");
                }
            } else {
                for pair in row[..width.div_ceil(2) * 2].as_chunks_mut::<2>().0 {
                    *pair = [118, 140];
                }
            }
        });
        buffer
    }

    /// The frame over `buffer`'s two planes, declared as `color`.
    fn nv12(metal: &Metal, buffer: &CVPixelBuffer, color: FrameColor) -> ExternalFrame {
        ExternalFrame::yuv(
            plane_texture(metal, buffer, 0, LUMA),
            plane_texture(metal, buffer, 1, CHROMA),
            color,
        )
        .expect("a valid NV12 frame")
    }

    /// A view that is never put in a window, as a window handle.
    struct View(NonNull<NSView>);

    // SAFETY: the engine reads the handle once, on the main thread, in
    // `WindowTarget::new`; the fixture keeps the view alive and releases it
    // on the main thread.
    unsafe impl Send for View {}
    // SAFETY: as above.
    unsafe impl Sync for View {}

    impl HasWindowHandle for View {
        fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
            let raw = RawWindowHandle::AppKit(AppKitWindowHandle::new(self.0.cast()));
            // SAFETY: the view outlives the handle.
            Ok(unsafe { WindowHandle::borrow_raw(raw) })
        }
    }

    impl HasDisplayHandle for View {
        fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
            Ok(DisplayHandle::appkit())
        }
    }

    fn sublayers(layer: &CALayer) -> Vec<Retained<CALayer>> {
        // SAFETY: the sublayers array is read on the thread that owns it.
        unsafe { layer.sublayers() }.map_or_else(Vec::new, |a| a.to_vec())
    }

    /// Every display layer under `layer`.
    fn displays(layer: &CALayer) -> Vec<Retained<AVSampleBufferDisplayLayer>> {
        sublayers(layer)
            .into_iter()
            .flat_map(|l| match l.downcast::<AVSampleBufferDisplayLayer>() {
                Ok(display) => vec![display],
                Err(l) => displays(&l),
            })
            .collect()
    }

    /// Drives the main run loop until `done`, or fails once `deadline`
    /// passes.
    fn drive(deadline: Instant, done: &dyn Fn() -> bool, what: &dyn Fn() -> String) {
        // SAFETY: the mode is an immutable static.
        let mode = unsafe { kCFRunLoopDefaultMode };
        while !done() {
            assert!(Instant::now() < deadline, "{}", what());
            CFRunLoop::run_in_mode(mode, 0.005, true);
        }
    }

    /// Drives the main run loop until `flag` is set — the completion
    /// signal a queued block, like a display layer's attach, leaves
    /// behind.
    fn settle_flag(flag: &AtomicBool, what: &str) {
        drive(
            Instant::now() + Duration::from_secs(10),
            &|| flag.load(Ordering::Acquire),
            &|| what.into(),
        );
    }

    /// Commits this thread's implicit transaction, then drives the main run
    /// loop until every display layer has its first frame ready and has laid
    /// the frame out.
    ///
    /// A display layer reports ready before the layout of its video sublayer
    /// has run: that layout is queued on the main queue first, so a block
    /// queued behind the readiness runs after it.
    fn settle(layer: &CALayer) {
        CATransaction::flush();
        let deadline = Instant::now() + Duration::from_secs(10);
        for display in displays(layer) {
            drive(
                deadline,
                // SAFETY: the display layer is read on the main thread.
                &|| unsafe { display.isReadyForDisplay() },
                &|| {
                    // SAFETY: as above.
                    let r = unsafe { display.sampleBufferRenderer() };
                    format!(
                        "the display layer never became ready: {:?} error={:?} \
                        bounds={:?} hidden={:?}",
                        unsafe { r.status() },
                        unsafe { r.error() },
                        display.bounds(),
                        display.isHidden(),
                    )
                },
            );
        }
        let drained = Arc::new(AtomicBool::new(false));
        let mark = Arc::clone(&drained);
        DispatchQueue::main().exec_async(move || mark.store(true, Ordering::Release));
        drive(deadline, &|| drained.load(Ordering::Acquire), &|| {
            "the main queue never drained".into()
        });
        CATransaction::flush();
    }

    /// Core Animation's own renderer drawing a layer tree into an extended
    /// sRGB texture, standing in for the window server.
    ///
    /// The target is sRGB because the engine's window parts present
    /// sRGB-encoded content with a layer's default colour space, which Core
    /// Animation reads in the target's space; frames on planes carry their
    /// own and are converted.
    struct SystemCompositor {
        renderer: Retained<CARenderer>,
        target: Retained<ProtocolObject<dyn MTLTexture>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        stage: Retained<CALayer>,
    }

    impl SystemCompositor {
        /// Stages `host`, sized in points, at the surface's scale so one unit
        /// of the renderer's bounds is one pixel, and commits it.
        ///
        /// Core Animation sends a renderer only what is committed after it is
        /// attached, so this runs before the engine builds its layers.
        fn attach(metal: &Metal, host: &CALayer) -> Self {
            // SAFETY: the descriptor is fully specified.
            let descriptor = unsafe {
                MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                    MTLPixelFormat::RGBA16Float,
                    SIZE.0 as usize,
                    SIZE.1 as usize,
                    false,
                )
            };
            descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
            descriptor.setStorageMode(MTLStorageMode::Shared);
            let target = metal
                .raw
                .newTextureWithDescriptor(&descriptor)
                .expect("composite target");
            let queue = metal.raw.newCommandQueue().expect("queue");
            // SAFETY: the name is an immutable static.
            let space = objc2_core_graphics::CGColorSpace::with_name(Some(unsafe {
                objc2_core_graphics::kCGColorSpaceExtendedSRGB
            }))
            .expect("extended sRGB");
            // SAFETY: a `CGColorSpace` is toll-free bridged to an Objective-C
            // object.
            let space: &AnyObject = unsafe { &*CFRetained::as_ptr(&space).as_ptr().cast() };
            let queue_object: &AnyObject = AsRef::<AnyObject>::as_ref(&*queue);
            // SAFETY: the option keys are immutable statics.
            let keys = unsafe {
                [
                    objc2_quartz_core::kCARendererColorSpace,
                    objc2_quartz_core::kCARendererMetalCommandQueue,
                ]
            };
            let options = objc2_foundation::NSDictionary::<
                objc2_foundation::NSString,
                AnyObject,
            >::from_slices(&keys, &[space, queue_object]);
            // SAFETY: a string key is an object key; the target outlives the
            // renderer.
            let renderer = unsafe {
                CARenderer::rendererWithMTLTexture_options(&target, Some(options.cast_unchecked()))
            };
            let pixels = CGRect::new(
                CGPoint::new(0.0, 0.0),
                CGSize::new(f64::from(SIZE.0), f64::from(SIZE.1)),
            );
            // The view's layer keeps the geometry AppKit gives it; a layer
            // between it and the stage scales points to pixels.
            let stage = CALayer::new();
            stage.setBounds(pixels);
            stage.setAnchorPoint(CGPoint::new(0.0, 0.0));
            stage.setPosition(CGPoint::new(0.0, 0.0));
            let points = CALayer::new();
            points.setBounds(host.frame());
            points.setAnchorPoint(CGPoint::new(0.0, 0.0));
            points.setPosition(CGPoint::new(0.0, 0.0));
            points.setAffineTransform(CGAffineTransform {
                a: SCALE,
                b: 0.0,
                c: 0.0,
                d: SCALE,
                tx: 0.0,
                ty: 0.0,
            });
            stage.addSublayer(&points);
            points.addSublayer(host);
            renderer.setLayer(Some(&stage));
            renderer.setBounds(pixels);
            CATransaction::flush();
            Self {
                renderer,
                target,
                queue,
                stage,
            }
        }

        /// Composites what is committed, returning premultiplied linear
        /// Display P3 pixels, row 0 at the top of the screen.
        ///
        /// The renderer writes layer space bottom-up (row 0 is `y = 0`, the
        /// bottom of an unflipped layer), so rows are reversed.
        fn composite(&self) -> Vec<[f32; 4]> {
            settle(&self.stage);
            let bounds = self.renderer.bounds();
            // SAFETY: a null timestamp is allowed.
            unsafe {
                self.renderer.beginFrameAtTime_timeStamp(
                    objc2_quartz_core::CACurrentMediaTime(),
                    std::ptr::null_mut(),
                );
            }
            self.renderer.addUpdateRect(bounds);
            self.renderer.render();
            self.renderer.endFrame();
            let fence = self.queue.commandBuffer().expect("fence");
            fence.commit();
            fence.waitUntilCompleted();
            let mut halves = vec![0u16; SIZE.0 as usize * SIZE.1 as usize * 4];
            // SAFETY: `halves` holds the whole RGBA16Float target.
            unsafe {
                self.target.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                    NonNull::new(halves.as_mut_ptr().cast()).expect("bytes"),
                    SIZE.0 as usize * 8,
                    MTLRegion {
                        origin: objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
                        size: objc2_metal::MTLSize {
                            width: SIZE.0 as usize,
                            height: SIZE.1 as usize,
                            depth: 1,
                        },
                    },
                    0,
                );
            }
            let decode = |c: f64| {
                let linear = if c.abs() <= 0.040_45 {
                    c.abs() / 12.92
                } else {
                    ((c.abs() + 0.055) / 1.055).powf(2.4)
                };
                linear.copysign(c)
            };
            let pixels: Vec<[f32; 4]> = halves
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| {
                    let [r, g, b, a] = p.map(|h| f64::from(half::f16::from_bits(h).to_f32()));
                    let straight = [r, g, b].map(|c| if a > 0.0 { decode(c / a) } else { 0.0 });
                    let [r, g, b] = cherenkov_oracle::color::linear_srgb_to_linear_p3(straight);
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "the target holds half floats"
                    )]
                    [r * a, g * a, b * a, a].map(|c| c as f32)
                })
                .collect();
            pixels
                .as_chunks::<{ SIZE.0 as usize }>()
                .0
                .iter()
                .rev()
                .flatten()
                .copied()
                .collect()
        }
    }

    /// An engine with a window surface over a view that is never shown, the
    /// view's layer staged for the system compositor.
    struct Fixture {
        metal: Metal,
        engine: Engine<Gpu>,
        window: Surface<Gpu>,
        system: SystemCompositor,
        view: Retained<NSView>,
        /// Set by the engine's wake callback: an attach landing on the
        /// main queue asks for the frame that promotes its candidate.
        woke: Arc<AtomicBool>,
    }

    impl Fixture {
        fn new() -> Self {
            let mtm = MainThreadMarker::new().expect("the cases run on the main thread");
            let metal = metal();
            let engine = Engine::<Gpu>::new(GpuConfig {
                device: Some(metal.shared.clone()),
                ..GpuConfig::default()
            })
            .expect("an engine");
            let view = NSView::initWithFrame(
                NSView::alloc(mtm),
                CGRect::new(
                    CGPoint::new(0.0, 0.0),
                    CGSize::new(f64::from(SIZE.0) / SCALE, f64::from(SIZE.1) / SCALE),
                ),
            );
            view.setWantsLayer(true);
            let host = view.layer().expect("a layer-backed view");
            host.setContentsScale(SCALE);
            let system = SystemCompositor::attach(&metal, &host);
            let window = engine
                .surface(WindowTarget::new(View(NonNull::from(&*view)), SIZE))
                .expect("a window surface");
            window
                .display(Display {
                    scale: SCALE,
                    headroom: 1.0,
                })
                .expect("the display");
            let woke = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&woke);
            engine.set_waker(move || flag.store(true, Ordering::Release));
            Self {
                metal,
                engine,
                window,
                system,
                view,
                woke,
            }
        }

        fn host(&self) -> Retained<CALayer> {
            self.view.layer().expect("a layer-backed view")
        }

        /// The engine's root layer under the host.
        fn root(&self) -> Retained<CALayer> {
            let layers = sublayers(&self.host());
            let [root] = &layers[..] else {
                panic!("one engine root under the host, found {}", layers.len());
            };
            root.clone()
        }

        fn render(&self) {
            self.engine.render(FrameTime::now()).expect("rendered");
            settle(&self.host());
        }

        /// The frames a window produces while a candidate's attach lands:
        /// the first render composites the candidate in-engine —
        /// asserted, the pending contract — the attach block's
        /// completion wake is both the drain's done signal and the
        /// redraw request, and the second render promotes it.
        fn promote(&self) {
            self.woke.store(false, Ordering::Relaxed);
            self.engine.render(FrameTime::now()).expect("rendered");
            assert!(
                displays(&self.root()).is_empty(),
                "the pending candidate stays engine-composited"
            );
            settle_flag(&self.woke, "the queued attach never completed");
            self.engine.render(FrameTime::now()).expect("rendered");
            settle(&self.host());
        }
    }

    /// A full-surface backdrop, a holder translated into the surface with a
    /// rounded clip and the video scaled into it, and a translucent control
    /// bar painted above the video. The layers live as long as the handles.
    #[must_use = "dropping the handles removes the layers"]
    fn scene(engine: &Engine<Gpu>, surface: &Surface<Gpu>, frame: ExternalFrame) -> [Layer; 4] {
        scene_bar(engine, surface, frame, 0.5)
    }

    /// `scene` with the control bar's alpha: `bar_alpha` 1.0 makes it
    /// opaque — the composite then has no translucency for the platform
    /// to resolve differently.
    fn scene_bar(
        engine: &Engine<Gpu>,
        surface: &Surface<Gpu>,
        frame: ExternalFrame,
        bar_alpha: f32,
    ) -> [Layer; 4] {
        let below = surface.layer();
        let holder = surface.layer();
        let player = surface.layer();
        let above = surface.layer();
        let backdrop = surface.record(|c| {
            c.fill(
                Rect::new(0.0, 0.0, 96.0, 64.0),
                WorkingColor::new([0.1, 0.3, 0.6, 1.0]),
            );
        });
        let bar = surface.record(|c| {
            c.fill(
                Rect::new(8.0, 44.0, 88.0, 58.0),
                WorkingColor::new([0.5, 0.5, 0.5, bar_alpha]),
            );
        });
        let video = engine.external_frame(frame);
        surface.update(|tx| {
            tx[surface.root()].push(&below).push(&holder).push(&above);
            tx[&below].content(backdrop);
            tx[&holder]
                .push(&player)
                .transform(Affine::translate((12.0, 8.0)))
                .clip(RoundedRect::new(0.0, 0.0, 72.0, 48.0, 6.0));
            tx[&player].transform(Affine::scale(1.5)).content(video);
            tx[&above].content(bar);
        });
        [below, holder, player, above]
    }

    fn is<T: objc2::ClassType>(layer: &CALayer) -> bool {
        layer.isKindOfClass(T::class())
    }

    /// The layers directly under the engine root: parts and planes in paint
    /// order.
    fn stack(fixture: &Fixture) -> Vec<Retained<CALayer>> {
        sublayers(&fixture.root())
    }

    /// The plane sits between the part painted below it and the part
    /// painted above it, nested in one layer per tree level, with the level's
    /// transform, rounded clip, and the frame's size and opacity.
    fn the_realized_tree_puts_the_plane_between_its_parts() {
        let fixture = Fixture::new();
        let buffer = nv12_buffer(VIDEO);
        let _scene = scene_bar(
            &fixture.engine,
            &fixture.window,
            nv12(&fixture.metal, &buffer, FrameColor::BT2020_PQ),
            1.0,
        );
        fixture.promote();

        let root = fixture.root();
        assert!(root.isGeometryFlipped(), "engine space is y-down");
        let stack = stack(&fixture);
        assert_eq!(stack.len(), 3, "part, plane, part");
        assert!(is::<CAMetalLayer>(&stack[0]) && is::<CAMetalLayer>(&stack[2]));
        assert!(!is::<CAMetalLayer>(&stack[1]));
        // Pixel space, then root → holder → player, each transform → [clip]
        // → scroll.
        let top = &stack[1];
        let t = top.affineTransform();
        assert!((t.a - 1.0 / SCALE).abs() < 1e-12 && (t.d - 1.0 / SCALE).abs() < 1e-12);
        let root_node = &sublayers(top)[0];
        let root_scroll = &sublayers(root_node)[0];
        let holder_node = &sublayers(root_scroll)[0];
        assert!((holder_node.affineTransform().tx - 12.0).abs() < 1e-12);
        let holder_clip = &sublayers(holder_node)[0];
        assert!(holder_clip.masksToBounds());
        assert!((holder_clip.cornerRadius() - 6.0).abs() < 1e-12);
        let holder_scroll = &sublayers(holder_clip)[0];
        let player_node = &sublayers(holder_scroll)[0];
        assert!((player_node.affineTransform().a - 1.5).abs() < 1e-12);
        let player_scroll = &sublayers(player_node)[0];
        let [display] = &sublayers(player_scroll)[..] else {
            panic!("one display layer");
        };
        assert!(is::<AVSampleBufferDisplayLayer>(display));
        let bounds = display.bounds();
        assert_eq!((bounds.size.width, bounds.size.height), (48.0, 32.0));
        assert!(
            (display.opacity() - 1.0).abs() < f32::EPSILON,
            "the player's opacity"
        );
    }

    fn same(buffer: &CVPixelBuffer, key: &CFString, expected: &CFString) -> bool {
        // SAFETY: the attachment is read, not retained past the buffer.
        unsafe { buffer.attachment(key, std::ptr::null_mut()) }
            .is_some_and(|v| v.downcast_ref::<CFString>().is_some_and(|s| s == expected))
    }

    /// The display layer shows the frame's own `IOSurface`, without a copy,
    /// tagged with exactly the colour the frame declares, so the system
    /// decodes and tone-maps it the way the engine would.
    fn a_promoted_frame_shows_its_own_surface_and_declared_colour() {
        // SAFETY: CoreVideo's constants are immutable statics.
        let cases = unsafe {
            [(
                FrameColor::BT2020_PQ,
                kCVImageBufferColorPrimaries_ITU_R_2020,
                kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ,
                kCVImageBufferYCbCrMatrix_ITU_R_2020,
            )]
        };
        for (color, primaries, transfer, matrix) in cases {
            let fixture = Fixture::new();
            let buffer = nv12_buffer(VIDEO);
            let _scene = scene_bar(
                &fixture.engine,
                &fixture.window,
                nv12(&fixture.metal, &buffer, color),
                1.0,
            );
            fixture.promote();
            // The renderer records the buffer it displayed when it draws.
            let _ = fixture.system.composite();
            let [display] = &displays(&fixture.root())[..] else {
                panic!("one display layer");
            };
            // SAFETY: the renderer is read on the main thread.
            let shown = unsafe { display.sampleBufferRenderer().copyDisplayedPixelBuffer() }
                .expect("the display layer shows a frame");
            let own = CVPixelBufferGetIOSurface(Some(&buffer)).expect("IOSurface");
            let displayed = CVPixelBufferGetIOSurface(Some(&shown)).expect("IOSurface");
            // The renderer holds its own reference to the surface; the
            // surface's identity is its ID.
            assert_eq!(own.id(), displayed.id(), "the frame's own surface, no copy");
            // SAFETY: CoreVideo's keys are immutable statics.
            unsafe {
                assert!(same(&shown, kCVImageBufferColorPrimariesKey, primaries));
                assert!(same(&shown, kCVImageBufferTransferFunctionKey, transfer));
                assert!(same(&shown, kCVImageBufferYCbCrMatrixKey, matrix));
                assert!(same(
                    &shown,
                    kCVImageBufferChromaLocationTopFieldKey,
                    kCVImageBufferChromaLocation_Left
                ));
            }
        }
    }

    /// Renders `frame` in the scene and asserts the engine composed it
    /// itself: one part, no plane.
    fn stays_in_the_engine(fixture: &Fixture, frame: ExternalFrame) {
        let _scene = scene_bar(&fixture.engine, &fixture.window, frame, 1.0);
        fixture.render();
        let stack = stack(fixture);
        assert_eq!(stack.len(), 1, "one part");
        assert!(is::<CAMetalLayer>(&stack[0]));
        assert!(displays(&fixture.root()).is_empty(), "no plane");
    }

    /// A studio-range surface declared full-range would be decoded
    /// differently by the system than by the engine.
    fn a_frame_whose_surface_disagrees_with_its_range_stays_in_the_engine() {
        let fixture = Fixture::new();
        let buffer = nv12_buffer(VIDEO);
        let full = FrameColor {
            range: YuvRange::Full,
            ..FrameColor::BT709_VIDEO
        };
        let frame = nv12(&fixture.metal, &buffer, full);
        stays_in_the_engine(&fixture, frame);
    }

    fn planes_of_two_surfaces_stay_in_the_engine() {
        let fixture = Fixture::new();
        let one = nv12_buffer(VIDEO);
        let two = nv12_buffer(VIDEO);
        let frame = ExternalFrame::yuv(
            plane_texture(&fixture.metal, &one, 0, LUMA),
            plane_texture(&fixture.metal, &two, 1, CHROMA),
            FrameColor::BT709_VIDEO,
        )
        .expect("valid");
        stays_in_the_engine(&fixture, frame);
    }

    fn a_frame_without_an_iosurface_stays_in_the_engine() {
        let fixture = Fixture::new();
        let plane = |format, (width, height): (usize, usize)| {
            fixture
                .metal
                .shared
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: u32::try_from(width).expect("small"),
                        height: u32::try_from(height).expect("small"),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
        };
        let frame = ExternalFrame::yuv(
            plane(wgpu::TextureFormat::R8Uint, VIDEO),
            plane(wgpu::TextureFormat::Rg8Uint, (VIDEO.0 / 2, VIDEO.1 / 2)),
            FrameColor::BT709_VIDEO,
        )
        .expect("valid");
        stays_in_the_engine(&fixture, frame);
    }

    /// A display layer shows opaque video; alpha stays with the engine.
    fn only_opaque_rgb_frames_are_promoted() {
        let bgra = |fixture: &Fixture, alpha| {
            let buffer = surface_buffer(VIDEO.0, VIDEO.1, kCVPixelFormatType_32BGRA);
            let plane = plane_texture(
                &fixture.metal,
                &buffer,
                0,
                (MTLPixelFormat::BGRA8Unorm, wgpu::TextureFormat::Bgra8Unorm),
            );
            ExternalFrame::rgb(plane, alpha, FrameColor::SRGB).expect("valid")
        };
        let straight = Fixture::new();
        stays_in_the_engine(&straight, bgra(&straight, RgbAlpha::Straight));
        let opaque = Fixture::new();
        let _scene = scene_bar(
            &opaque.engine,
            &opaque.window,
            bgra(&opaque, RgbAlpha::Opaque),
            1.0,
        );
        opaque.promote();
        assert_eq!(displays(&opaque.root()).len(), 1, "promoted");
    }

    /// The system compositor's result for the promoted stack matches the
    /// engine's own composition of the same tree within a perceptual
    /// tolerance: FLIP mean at most 0.05 and no local error above 0.25.
    ///
    /// The frame is opaque BGRA declared sRGB — the platform and the
    /// engine produce identical pixels from it, so the comparison
    /// measures promotion (order, geometry, blending), not the
    /// platform's YCbCr decoder, whose studio-range expansion differs
    /// from the engine's on this target.
    fn promoted_composition_matches_engine_composition() {
        let fixture = Fixture::new();
        let buffer = surface_buffer(VIDEO.0, VIDEO.1, kCVPixelFormatType_32BGRA);
        fill(&buffer, 1, |_, row| {
            for (x, px) in row[..VIDEO.0 * 4]
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .enumerate()
            {
                *px = [
                    u8::try_from(40 + x * 3).expect("blue"),
                    96,
                    u8::try_from(30 + x * 4).expect("red"),
                    255,
                ];
            }
        });
        let bgra = |metal: &Metal| {
            ExternalFrame::rgb(
                plane_texture(
                    metal,
                    &buffer,
                    0,
                    (MTLPixelFormat::BGRA8Unorm, wgpu::TextureFormat::Bgra8Unorm),
                ),
                RgbAlpha::Opaque,
                FrameColor::SRGB,
            )
            .expect("a valid BGRA frame")
        };
        let offscreen = fixture
            .engine
            .surface(Offscreen::new(SIZE, OffscreenFormat::LinearF16))
            .expect("offscreen");
        let _engine_scene = scene_bar(&fixture.engine, &offscreen, bgra(&fixture.metal), 1.0);
        let _window_scene = scene_bar(&fixture.engine, &fixture.window, bgra(&fixture.metal), 1.0);
        fixture.promote();
        assert_eq!(displays(&fixture.root()).len(), 1, "promoted");
        let engine = offscreen.readback().expect("engine composition");
        let system = fixture.system.composite();
        let image = |pixels: Vec<[f32; 4]>| cherenkov_oracle::F32Image {
            width: SIZE.0,
            height: SIZE.1,
            pixels,
        };
        let (metrics, _) =
            cherenkov_oracle::metrics::compare(&image(engine.pixels), &image(system));
        assert!(
            metrics.flip_mean <= 0.05 && metrics.max_local_error <= 0.25,
            "promoted vs engine composition: {metrics:?}"
        );
    }

    /// The pixels a window and an offscreen engine surface produce from
    /// the same scene, compared the way
    /// `promoted_composition_matches_engine_composition` does.
    fn engine_parity(fixture: &Fixture, offscreen: &Surface<Gpu>, what: &str) {
        let engine = offscreen.readback().expect("engine composition");
        let system = fixture.system.composite();
        let image = |pixels: Vec<[f32; 4]>| cherenkov_oracle::F32Image {
            width: SIZE.0,
            height: SIZE.1,
            pixels,
        };
        let (metrics, _) =
            cherenkov_oracle::metrics::compare(&image(engine.pixels), &image(system));
        assert!(
            metrics.flip_mean <= 0.05 && metrics.max_local_error <= 0.25,
            "{what} vs engine composition: {metrics:?}"
        );
    }

    /// The platform decodes `ITU_R_709_2` with the inverse OETF while the
    /// engine applies BT.1886 gamma 2.4 and no colour tag reproduces
    /// gamma 2.4, so `shows` keeps a BT.709 frame in the engine: no plane,
    /// and the window shows the engine's own composition.
    fn a_bt709_frame_stays_in_the_engine_and_matches() {
        let fixture = Fixture::new();
        let buffer = nv12_buffer(VIDEO);
        let offscreen = fixture
            .engine
            .surface(Offscreen::new(SIZE, OffscreenFormat::LinearF16))
            .expect("offscreen");
        let _engine_scene = scene_bar(
            &fixture.engine,
            &offscreen,
            nv12(&fixture.metal, &buffer, FrameColor::BT709_VIDEO),
            1.0,
        );
        let _window_scene = scene_bar(
            &fixture.engine,
            &fixture.window,
            nv12(&fixture.metal, &buffer, FrameColor::BT709_VIDEO),
            1.0,
        );
        fixture.render();
        assert!(
            displays(&fixture.root()).is_empty(),
            "the plan keeps BT.709 in the engine"
        );
        engine_parity(&fixture, &offscreen, "engine-composited BT.709");
    }

    /// A layer painted above a plane that is not known to be opaque is
    /// blended by the platform in its own space, not the engine's linear
    /// blend, so the plan keeps the video in the engine: no plane, and
    /// the window shows the engine's own composition.
    fn a_translucent_layer_above_stays_in_the_engine_and_matches() {
        let fixture = Fixture::new();
        let buffer = surface_buffer(VIDEO.0, VIDEO.1, kCVPixelFormatType_32BGRA);
        let bgra = |metal: &Metal| {
            ExternalFrame::rgb(
                plane_texture(
                    metal,
                    &buffer,
                    0,
                    (MTLPixelFormat::BGRA8Unorm, wgpu::TextureFormat::Bgra8Unorm),
                ),
                RgbAlpha::Opaque,
                FrameColor::SRGB,
            )
            .expect("a valid BGRA frame")
        };
        let offscreen = fixture
            .engine
            .surface(Offscreen::new(SIZE, OffscreenFormat::LinearF16))
            .expect("offscreen");
        let _engine_scene = scene(&fixture.engine, &offscreen, bgra(&fixture.metal));
        let _window_scene = scene(&fixture.engine, &fixture.window, bgra(&fixture.metal));
        fixture.render();
        assert!(
            displays(&fixture.root()).is_empty(),
            "the plan keeps a video under a translucent layer in the engine"
        );
        engine_parity(
            &fixture,
            &offscreen,
            "engine-composited video under a translucent layer",
        );
    }
}
