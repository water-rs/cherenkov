// Compute execution: each invocation stores its own output texel via
// textureStore into a storage texture (requires wgpu::TextureUsages::
// STORAGE_BINDING on the target — unavailable on WebGL2).

@group(0) @binding(1)
var output_texture: texture_storage_2d<OUTPUT_STORAGE_FORMAT, write>;
