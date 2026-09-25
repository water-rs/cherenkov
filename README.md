# Cherenkov

A GPU 2D rendering engine in Rust, built for modern hardware.

Cherenkov is being designed to render WaterUI's self-drawn backend, Hydrolysis, in place of Vello, and later to serve as the 2D engine of a web browser. It targets current GPUs only. The points below are the working hypotheses; the cross-engine device farm (#3) decides between them and the alternative architectures by measurement:

- **Geometry on the CPU, pixels in the GPU.** The CPU produces geometry whose size is fixed before submission, and fragment shaders do the per-pixel work. On tile-based GPUs, this keeps intermediate results in on-chip memory.
- **On-chip programmable blending is required.** On Metal this means raster order groups and imageblocks; on Vulkan, rasterization-order attachment access or dynamic-rendering local read.
- **Colour.** Rendering happens in linear, extended-range colour with a wide-gamut working space and native HDR output.
- **Layers.** The engine keeps a retained layer tree, and animates layer properties itself at present time.
- **Semantic primitives are first-class.** Rounded rectangles with continuous corners, shadows, glyph runs, images and gradients are drawn directly, not lowered to generic paths.

The CPU side is data-parallel and SIMD-vectorised, and covered by microbenchmarks.

## Status

The project is in the design phase and there is nothing to use yet. The public API is designed first, followed by a cross-engine correctness and performance suite that measures Cherenkov against Skia and Vello.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
