//! Retained layer pixels and measured admission inputs.

use cherenkov::{Command, DisplayList, GlyphStyle, RenderError};
use kurbo::{Affine, Rect, Shape};
use rustc_hash::FxHashMap;
use skrifa::MetadataProvider;

use crate::render::{
    bitmap, glyph,
    prepared::{Op, Outline, ResolvedPaint},
};

/// A local source domain. Its integer origin and dimensions include the
/// rasterizer's antialiasing footprint; native placement removes that offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Domain {
    pub origin: kurbo::Vec2,
    pub size: (u32, u32),
    pub density: f64,
}

impl Domain {
    pub fn raster(self) -> Affine {
        Affine::translate(self.origin) * Affine::scale(self.density.recip())
    }
}

/// Engine-side capture. A native realization retains its own immutable
/// presentation buffer until content changes or promotion ends.
pub struct Capture {
    pub domain: Domain,
    /// Released after native publication; the capture generation remains.
    pub source: Option<(wgpu::Texture, wgpu::TextureView)>,
    pub generation: u64,
    pub dirty: bool,
}

/// Content observations are separate from pixel allocations: ineligible
/// candidates consume no plane memory.
pub struct Observation {
    pub stamp: u64,
    pub resources: (u64, u64),
    pub domain: Option<Domain>,
    pub density: f64,
    pub quiet_frames: u64,
    pub capture: Option<Capture>,
}

impl Capture {
    pub fn bytes(&self) -> u64 {
        self.source.as_ref().map_or(0, |_| {
            u64::from(self.domain.size.0)
                * u64::from(self.domain.size.1)
                * crate::render::texel_bytes(crate::render::TARGET_FORMAT)
        })
    }
}

fn stroke_margin(stroke: &kurbo::Stroke) -> f64 {
    stroke.width.abs() * 0.5 * stroke.miter_limit.max(1.0)
}

fn union(bounds: &mut Option<Rect>, rect: Rect) {
    if !rect.is_zero_area() {
        *bounds = Some(bounds.map_or(rect, |bounds| bounds.union(rect)));
    }
}

fn glyph_bounds(
    run: &cherenkov::GlyphRun,
    fonts: &FxHashMap<u64, glyph::FontData>,
) -> Result<Rect, RenderError> {
    let data = fonts
        .get(&run.font.raw())
        .ok_or_else(|| RenderError::Font(format!("unregistered font {}", run.font.raw())))?;
    let font = skrifa::FontRef::from_index(&data.data, data.index)
        .map_err(|error| RenderError::Font(error.to_string()))?;
    let units = f64::from(
        font.metrics(
            skrifa::instance::Size::unscaled(),
            skrifa::instance::LocationRef::default(),
        )
        .units_per_em,
    );
    let scale = f64::from(run.size) / units;
    let coords: Vec<_> = run
        .coords
        .iter()
        .map(|value| skrifa::raw::types::F2Dot14::from_bits(*value))
        .collect();
    let outlines = font.outline_glyphs();
    let mut bounds = None;
    for item in run.glyphs.iter() {
        let transform = Affine::translate((f64::from(item.x), f64::from(item.y)))
            * glyph::checked_transform(item)?;
        if let Some(bitmap_font) = &data.bitmap {
            if let Some(decoded) = bitmap::decode(
                &data.data,
                data.index,
                bitmap_font,
                bitmap_font.select(run.size),
                item.id,
            )? {
                union(
                    &mut bounds,
                    transform.transform_rect_bbox(
                        Affine::scale(f64::from(run.size)).transform_rect_bbox(decoded.em),
                    ),
                );
            }
        } else if let Some(outline) = glyph::outline(&outlines, &coords, item.id)? {
            let mut rect = Affine::scale_non_uniform(scale, -scale)
                .transform_rect_bbox(outline.bounding_box());
            if let GlyphStyle::Stroke(stroke) = &run.style {
                rect = rect.inflate(stroke_margin(stroke), stroke_margin(stroke));
            }
            union(&mut bounds, transform.transform_rect_bbox(rect));
        }
    }
    Ok(bounds.unwrap_or(Rect::ZERO))
}

/// Conservative finite bounds from the actual retained display list. A
/// filter or shader can change without a source edit and is not static.
fn path_bounds(outline: &Outline, list: &DisplayList) -> Rect {
    match outline {
        Outline::Fill { elements, .. } => {
            kurbo::BezPath::from_vec(elements.to_vec()).bounding_box()
        }
        Outline::Stroke { shape, stroke } => shape
            .bounds()
            .inflate(stroke_margin(stroke), stroke_margin(stroke)),
        Outline::Source { command, .. } => match &list.commands()[*command] {
            Command::Fill { shape, .. } => shape.bounds(),
            Command::Stroke { shape, stroke, .. } => shape
                .bounds()
                .inflate(stroke_margin(stroke), stroke_margin(stroke)),
            _ => unreachable!("prepared path source"),
        },
    }
}

fn bounds(
    ops: &[Op],
    list: &DisplayList,
    fonts: &FxHashMap<u64, glyph::FontData>,
) -> Result<Option<Rect>, RenderError> {
    let mut result = None;
    let mut margins = Vec::new();
    for op in ops {
        let (local, rect) = match op {
            Op::Shaped {
                local,
                bounds,
                extra_margin,
                paint,
                ..
            } => {
                if matches!(paint, ResolvedPaint::Shader(_)) {
                    return Ok(None);
                }
                (*local, bounds.inflate(*extra_margin, *extra_margin))
            }
            Op::Shadow {
                local,
                bounds,
                sigma_eff,
                ..
            } => {
                let margin = sigma_eff.mul_add(3.0, 1.0);
                (*local, bounds.inflate(margin, margin))
            }
            Op::Path {
                local,
                outline,
                paint,
                ..
            } => {
                if matches!(paint, ResolvedPaint::Shader(_)) {
                    return Ok(None);
                }
                (*local, path_bounds(outline, list))
            }
            Op::Glyphs { local, run, paint } => {
                if matches!(paint, ResolvedPaint::Shader(_)) {
                    return Ok(None);
                }
                (*local, glyph_bounds(run.get(list), fonts)?)
            }
            Op::BitmapGlyph {
                local,
                font,
                glyph,
                origin,
                size,
            } => {
                let font = &fonts[font];
                let bitmap = font.bitmap.as_ref().expect("prepared bitmap font");
                let Some(decoded) =
                    bitmap::decode(&font.data, font.index, bitmap, bitmap.select(*size), *glyph)?
                else {
                    continue;
                };
                (
                    *local
                        * Affine::translate((f64::from(origin[0]), f64::from(origin[1])))
                        * Affine::scale(f64::from(*size)),
                    decoded.em,
                )
            }
            Op::BeginShadow { parameters, .. } => {
                let margin = 6.0_f64.mul_add(parameters.sigma, parameters.spread.max(0.0));
                let [a, b, c, d, _, _] = parameters.transform.as_coeffs();
                margins.push(kurbo::Vec2::new(a.hypot(c) * margin, b.hypot(d) * margin));
                continue;
            }
            Op::BeginClip { .. } => {
                margins.push(kurbo::Vec2::ZERO);
                continue;
            }
            Op::BeginIsolate { filter, .. } => {
                if filter.is_some() {
                    return Ok(None);
                }
                margins.push(kurbo::Vec2::ZERO);
                continue;
            }
            Op::End => {
                margins.pop().expect("prepared scopes pair");
                continue;
            }
        };
        if rect.is_zero_area() {
            continue;
        }
        let margin: kurbo::Vec2 = margins.iter().copied().sum();
        union(
            &mut result,
            local.transform_rect_bbox(rect).inflate(margin.x, margin.y),
        );
    }
    Ok(result)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "finite positive extents bounded by the device limit"
)]
pub fn domain(
    ops: &[Op],
    list: &DisplayList,
    fonts: &FxHashMap<u64, glyph::FontData>,
    density: f64,
    max: u32,
) -> Result<Option<Domain>, RenderError> {
    let Some(bounds) = bounds(ops, list, fonts)? else {
        return Ok(None);
    };
    if bounds.is_zero_area() {
        return Ok(None);
    }
    let bounds = Affine::scale(density)
        .transform_rect_bbox(bounds)
        .inflate(1.0, 1.0)
        .expand();
    if !bounds.is_finite() || bounds.width() > f64::from(max) || bounds.height() > f64::from(max) {
        return Ok(None);
    }
    Ok(Some(Domain {
        origin: bounds.origin().to_vec2() / density,
        density,
        size: (bounds.width() as u32, bounds.height() as u32),
    }))
}

