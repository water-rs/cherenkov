# CPU effects

## Reference entry points

The oracle remains the specification. Its frontend reference entry points are
`mesh::sample`, `glyphs::styled_outline`, `blend::in_space`, and
`shadow::spread_coverage` followed by `shadow::gaussian_blur`. They accept the
frontend types directly, so tests can specify mesh grids, glyph styles and
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
Shift the geometry before integration. Intersect clips before convolution;
clipping the blurred image instead would remove its tail. Arbitrary contours,
self-intersections, holes, rotations and shears use the same coverage compiler.

The oracle's sigma is in device pixels: transformed geometry is followed by an
isotropic device-space convolution. The frontend's former “shape units” sigma
comment contradicted `oracle/src/render.rs`; the comment now states the oracle
contract. Offsets and spread remain in shape units.

Spread is explicitly a contour operation. Stroke the authored contours with
round joins/caps and width `2 * abs(spread)` before the drawing transform.
Positive spread unions that band with the fill; negative spread subtracts it.
All authored contours participate, including internal contours. This definition
is distinct from eroding only the exposed boundary of a previously unioned
silhouette. The coverage compiler resolves union/difference and all clip
intersections before integrating any area. The oracle independently uses
`area(A ∪ B) = area(A) + area(B) − area(A ∩ B)` and
`area(A − B) = area(A) − area(A ∩ B)` with geometric intersections.

For positive sigma the Gaussian has radius `ceil(6 * sigma)`, with tap `d`
proportional to
`(erf((d + 0.5)/(sigma * sqrt(2))) − erf((d − 0.5)/(sigma * sqrt(2)))) / 2`.
Normalize the finite kernel, convolve horizontally, then vertically. Clamp
sample coordinates to the surface, never to the caster bounds. Sigma at or
below `1e-9` returns the input field. Empty fields remain empty.

Separable convolution is the oracle algorithm, with no box approximation or
point-sampled taps. Intermediates use f64; coverage storage and the final field
use f32. The old rounded-box quadrature, extra variance and three-sigma cutoff
are removed entirely. Colour is applied after convolution, so colour changes
reuse the cached scalar field. Cache identity includes source geometry, fill
rule, clips, transform, offset, spread, sigma and surface size.

With kernel radius `r`, width `W`, caster height `Hc`, and halo height `Ho`,
convolution costs at most `O(W(Hc + Ho)(2r + 1))` and `O(WHc + W)` temporary
scalar storage, plus output. Horizontal work is restricted to occupied row
intervals and their halos. Warm draws read prepared fields. Union/difference
spread uses the same row/group event compiler as intersections; each crossing
updates a constant-time Boolean predicate.

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

The oracle's pre-existing sRGB transfer implementation produced NaNs for
negative channels outside its linear segment. Signed encoding and decoding now
preserve the extended range. The largest-singular-value calculation also clamps
its discriminant to zero against floating-point cancellation at similarities.

## GPU differences on the base tree

At `ab47a3f`, GPU lowering rejects mesh paint, stroked glyphs, per-glyph
transforms and non-linear blend spaces. It therefore has no alternative
implemented semantics for those four features.

GPU shadows differ: rounded-box analytic integration uses
`sqrt(sigma² + 1/12)` and a three-sigma bound, in the box's local coordinates.
Its spread grows box half-extents and existing corner radii, keeping a sharp
box sharp. CPU/oracle shadows use six-sigma integrated pixel taps in device
space and the explicitly round contour spread defined above. These differ at
small sigma, scaled transforms, tails and spread corners. This change does not
alter GPU code or silently treat the two formulas as equivalent.

## Validation

Oracle tests cover bilinear interpolation and premultiplied alpha, folds,
reflection, degeneracy, signed extended-range colour conversion, encoded
source-over, glyph origin transforms, stroke width units and signed spread area.
CPU unit tests compare all blend modes in both spaces, folded/overlapping mesh
patches, and the independent coverage/clip/convolution shadow pipeline.
End-to-end tests add rotated/path shadows, offset and nested clips, signed
spread, warm cache reuse, transformed mesh paint, group opacity, layer blend
space, styled glyph cache transitions, and transformed colour glyph paint graphs.

Exact area is relative to flattened geometry, as in the coverage design.
Oracle curves are finer than CPU curves, so curved glyph/spread comparisons use
bounds appropriate to the 0.02-pixel flattening tolerance. Polygon and mesh
comparisons use floating-point rounding tolerances. No build, formatter, test
or benchmark was run on the coordinating Mac; the VM provides that evidence.
