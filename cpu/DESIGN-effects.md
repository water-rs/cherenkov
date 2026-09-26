# CPU effects

## Reference entry points

The oracle remains the specification. Its frontend reference entry points are
`mesh::sample`, `glyphs::styled_outline`, `blend::in_space`, and
`shadow::spread_coverage` followed by `shadow::gaussian_blur`. They accept geometry and
frontend values directly, so tests can specify mesh grids, glyph styles and
per-glyph transforms without encoding backend resource IDs into scene files.
The existing scene renderer continues to provide the independent colour-glyph
paint-graph reference. These functions are CPU-independent; the production CPU
crate uses the oracle only as a development dependency.

Group filters are outside this change. Filter registration and `Runs<F>` depend
on the shared frontend (#31) and resource handles (#26). `Unsupported::Filter`
continues to identify that capability boundary.

## Shadows

For shape `S`, drawing transform `T`, local offset `o`, and device-space clips,
coverage is `area(pixel ∩ ((T · translate(o)) S) ∩ clip₁ ∩ … ∩ clipₙ)`.
Shape/stroke approximation and device-space flattening each receive half of
the 0.02-pixel curve budget. Shift the geometry before integration. Intersect
clips before convolution;
clipping the blurred image instead would remove its tail. Arbitrary contours,
self-intersections, holes, rotations and shears use the same coverage compiler.

Sigma, offset and spread are in the shape's units. With linear transform `A`,
the local isotropic Gaussian becomes a device Gaussian with covariance
`Σ = sigma² A Aᵀ`. Translation moves the caster without changing covariance.
Uniform scale scales the blur; nonuniform scale and shear generally make it
anisotropic. Rotation of an isotropic Gaussian alone leaves it isotropic. The
former device-sigma oracle was inconsistent with `Shadow::sigma`; the oracle
now follows that API contract.

For rectangles and rounded rectangles, spread grows each half-extent by its
signed value. A positive corner radius `r` becomes `max(0, r + spread)`; a sharp
corner remains sharp. A nonpositive resulting extent is empty. This is the
CSS box-shadow corner rule. It is applied to semantic box geometry before the
transform, including rotated boxes.

Other shapes offset their closed authored contours with miter joins, miter
limit 4 (the SVG default), and bevels beyond that limit. Construct a band of
width `2 * abs(spread)` in local space. Positive spread unions it with the fill;
negative spread subtracts it. All authored contours participate, including
internal contours. The coverage compiler resolves union/difference and clip
intersections before integrating area. The independent oracle uses geometric
intersections and `area(A ∪ B) = area(A) + area(B) − area(A ∩ B)` and
`area(A − B) = area(A) − area(A ∩ B)`.

For each device offset `(i, j)`, the tap is the Gaussian probability of
`[i−0.5, i+0.5] × [j−0.5, j+0.5]`. Retain offsets through
`rx = ceil(6 * sqrt(Σxx))` and `ry = ceil(6 * sqrt(Σyy))`, inclusive; normalize
the retained mass. The outside mass is below `4e-9` by the two marginal tail
bounds. Clamp samples to the surface, never the caster bounds. For diagonal
covariance, each axis has the original integrated tap
`(erf((d+0.5)/(s*sqrt(2))) − erf((d−0.5)/(s*sqrt(2)))) / 2`, where `s` is that
axis's marginal sigma. Nonpositive sigma gives the identity. Every positive
sigma uses integrated taps, with no sharpness threshold or added variance.

For correlated covariance write `X = sx Z`, `Y = b Z + c W`, with independent
standard normal variables `Z` and `W`. Integrate the conditional normal interval
probability against the density of `Z`. Compute `c` from the determinant rather
than subtracting almost equal variances. Split quadrature at integer standard
normal coordinates and both conditional transitions (including their tails),
so near-singular transforms cannot hide narrow mass between quadrature nodes.
The oracle uses adaptive Simpson integration with absolute tolerance `2e-15`
per panel; CPU uses independently refined eight-point Gauss-Legendre integration
with `2e-14` per panel. Integration outside twelve standard deviations can
contribute less than `4e-33`. This numerical bound is below the integration
precision; the specified retained kernel remains the six-sigma pixel rectangle.
Rank-one covariance integrates the resulting line measure analytically, and
zero variance is a point mass. Neither requires an inverse transform.

Diagonal covariance uses two separable passes. Correlated covariance uses the
full integrated two-dimensional kernel, preserving the same pixel-footprint
reference. With width `W`, caster height `Hc` and output halo height `Ho`, the
separable cost is `O(W Hc (2rx+1) + W Ho (2ry+1))`, with horizontal work further
restricted to occupied intervals and their halos. Correlated convolution costs
`O(W Ho (2rx+1)(2ry+1))`. Both store `O(W Hc + W)` temporary scalar pixels, plus
output; kernel storage is linear in the radii for the separable case and their
product for the correlated case. Kernel integration occurs once per cache miss.
Intermediates are `f64`; coverage storage and final output are `f32`.

Colour is applied after convolution, so colour changes reuse the cached scalar
field. Cache identity includes realized spread geometry, fill rule, residual
contour-band spread, clips, transform, offset, sigma and surface size. Warm
draws read prepared fields. Spread uses the existing row/group event compiler;
each crossing updates a constant-time Boolean predicate.

## Glyph styles and placement

The reference outline is unhinted. Scale font units to run units, flipping y;
expand the stroke in run units; apply the optional glyph transform; translate
to the glyph's `(x, y)` origin; then apply the enclosing drawing transform.
Widths, caps, joins, miter limits and dash parameters therefore transform with
the glyph. A translation inside the glyph transform is local to that origin.
Fill and stroke paint remains in the enclosing drawing's coordinate system.

Filled colour glyphs retain their complete paint graph and transform it as a
whole. Stroked colour glyphs select the font's monochrome base outline; a missing
base outline is a font error. Palette graph fills are not individually stroked.

Flatten in device space at 0.02 pixels, then use the exact coverage compiler.
This also fixes the former glyph flattening tolerance, which accounted for
font size but ignored magnification by the drawing transform. Clipped glyphs
retain the realized outline and intersect it with clips geometrically.

Mask identity includes font, glyph, size, exact subpixel position, the realized
linear transform, full variation coordinates and every stroke parameter.
Variation and stroke identity are equality checked, not hash-only surrogates.
Identical misses within a frame share one realization before cache insertion;
all instance slots receive the same mask. The existing LRU policy remains.
Retained stroke-key allocation and variation payload are charged to the glyph
budget; shared variation payload is conservatively counted for each key.

## Mesh paint

A cell's corners are ordered `00, 10, 01, 11`. For `u, v` in `[0, 1]`, geometry
and premultiplied linear-P3 colour use weights
`(1−u)(1−v), u(1−v), (1−u)v, uv`. Premultiplication precedes interpolation;
transparent coloured vertices cannot leak their hidden RGB. Extended channels
remain extended.

Invert the bilinear geometry at the pixel centre in paint space. Eliminating
one parameter gives a quadratic; evaluate every real in-range inverse. For a
fold, choose greatest v, then greatest u. A zero-Jacobian inverse contributes
nothing. Overlapping cells select the last row-major patch, and samples outside
all cells are transparent. Shared grid edges interpolate their shared vertices,
so ownership does not introduce an ordinary seam. Patch ownership selects one
paint sample; it does not composite overlapping translucent patches.

The CPU prepares f64 geometry, premultiplied colours and conservative bounds in
an immutable shared allocation. Paint clones share that allocation. Each sample
rejects bounds before inversion and rounds only its final colour to f32.
Preparation/storage is `O(cells)`; sampling is worst-case `O(cells)` with early
exit at the last containing cell. Shape coverage is supplied independently by
the ordinary draw compiler, as with other pixel-centre-sampled paints.

## Group and layer blend spaces

Isolated content is rendered in premultiplied linear P3. Apply group opacity to
all four channels, then composite the isolated result into its parent in the
selected space. A non-linear space forces isolation even for opaque normal
source-over groups. Layer edits expose the same choice through `blend_space`.
Root layer state now follows the same transform, clip and isolation traversal
as child layers; previously the root's state was skipped.

For encoded sRGB, unpremultiply, convert primaries to linear sRGB, apply the
sign-preserving sRGB transfer curve, and premultiply again. Apply both the blend
function and Porter-Duff composition in that space. Unpremultiply the result,
decode, convert to linear P3 and premultiply for storage. Alpha is never encoded;
no gamut clamp is added. Zero-alpha conversion yields transparent black.
Nested groups perform these conversions at their own boundaries.

Solid coverage spans, solid glyph rows and linear source-over isolation use
portable SIMD selected once by the renderer. Each owned framebuffer band carries
its concrete SIMD type through dispatch. Constant spans retain packed RGBA,
using one uniform inverse alpha without channel shuffles. Varying coverage is
expanded over each pixel's four packed channels, so destination pixels need no
transpose. Isolation transposes source and destination into matching channel
vectors, composites and transposes back. Incomplete vectors use scalar arithmetic. Multiplication and addition remain separate and in the same
order as scalar composition. Zero coverage and zero isolated alpha preserve the
destination exactly. Native and scalar paths are checked bit for bit, including
extended channels, partial vectors, partial bands and isolation.

The oracle's pre-existing sRGB transfer implementation produced NaNs for
negative channels outside its linear segment. Signed encoding and decoding now
preserve the extended range. The largest-singular-value calculation also clamps
its discriminant to zero against floating-point cancellation at similarities.

## GPU differences on the base tree

At `ab47a3f`, GPU lowering rejects mesh paint, stroked glyphs, per-glyph
transforms and non-linear blend spaces. It therefore has no alternative
implemented semantics for those four features.

GPU shadow conformance requires the following changes:

- Replace the `sqrt(sigma² + 1/12)` variance adjustment and the shader's
  `sigma < 0.25` sharp-coverage branch with integrated taps for every positive
  sigma. The adjustment currently makes that branch unreachable for positive
  authored sigma, creating a discontinuity at zero. The sixteen-row midpoint
  corner integration also differs from the exact-coverage convolution reference.
- Replace the shader's 3σ integration cut and lowering's corresponding bound
  with the normalized six-sigma integrated kernel and its full device halo.
  Include coverage beyond 3σ rather than suppressing its tail.
- Preserve local sigma semantics while accounting for transformed pixel
  footprints: the GPU already evaluates the Gaussian in local coordinates,
  but its pixel-variance adjustment does not implement the pushed-forward
  pixel-integrated covariance above.
- Add general-path spread with miter limit 4 and bevels beyond the limit;
  `box_shape` currently rejects those casters. The rectangle/rounded-rectangle
  spread-corner disagreement is resolved by this oracle correction: GPU
  half-extents and positive radii grow by spread, and sharp corners stay sharp.
  Although lowering can store a negative radius after shrinkage, `corner_inset`
  and the distance functions already treat it as zero. There is no remaining
  box-corner rounding difference in those formulas.

GPU implementation changes are separate work; this change modifies no GPU files.

## Validation

Oracle tests cover bilinear interpolation and premultiplied alpha, folds,
reflection, degeneracy, signed extended-range colour conversion, encoded
source-over, glyph origin transforms, stroke width units and signed spread area.
Shadow checks additionally cover analytic correlated quadrant probabilities,
impulse covariance, scale/rotation/reflection invariance, tiny local sigma under
magnification, singular line/point kernels, mixed sharp/round box corners,
negative spread through radius zero, collapsed boxes and beveled acute miters.
CPU unit tests compare all blend modes in both spaces, folded/overlapping mesh
patches, and the independent coverage/clip/convolution shadow pipeline.
End-to-end tests add rotated/path shadows, offset and nested clips, signed
spread under nonuniform scale and shear, warm cache reuse, transformed mesh
paint, group opacity, layer blend space, styled glyph cache transitions, and
transformed colour glyph paint graphs.

Exact area is relative to flattened geometry, as in the coverage design.
Oracle curves are finer than CPU curves, so curved glyph/spread comparisons use
bounds appropriate to the 0.02-pixel flattening tolerance. Polygon and mesh
comparisons use floating-point rounding tolerances. Compilation, tests and benchmarks run on the build VM. The coordinating Mac
only edits, performs file-scoped formatting and reviews the returned evidence.
