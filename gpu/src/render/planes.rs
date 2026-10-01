//! System-compositor planes (#90).
//!
//! On a target that exposes a system-compositor parent, the engine promotes
//! eligible layers to their own system layers: the platform compositor, not
//! the engine, composites their content, so a full-screen video that plays
//! on a hardware overlay costs no per-frame engine composition. Everything
//! else is still composited inside the engine, onto one texture per *part*:
//! a promoted layer splits the surface into the part painted below it and
//! the part painted above it, so content drawn after the layer (controls,
//! overlays, its own children) stays above it.
//!
//! The decision is deterministic per frame and invisible to callers:
//! [`plan`] reads the sampled tree and the platform's [`Compositor`] limits
//! and returns the promoted layers in paint order, with the named cause for
//! every candidate it kept in the engine. Promotion is a realization choice
//! made before rendering, never a fallback after a failure: a platform that
//! rejects a plane the plan chose reports an error naming the cause.
//!
//! A platform implements [`SystemPlanes`] and realizes the ordered stack of
//! engine parts and promoted planes a [`Composition`] describes.

use kurbo::{Affine, Vec2};
use rustc_hash::{FxHashMap, FxHashSet};

use cherenkov::{BlendMode, Display, LayerId, RenderError, ShapeData, SurfaceTree};

use crate::interop::ExternalFrame;
use crate::render::lower::axis_aligned;
use crate::render::present::Presenter;

/// What a platform's system compositor can express, which bounds promotion.
pub trait Compositor {
    /// The most layers promoted on one surface. Every promoted layer adds
    /// an engine part above it, a full-surface texture, so the budget bounds
    /// memory as well as the hardware overlays the system can scan out.
    const BUDGET: usize;

    /// Whether a system layer carries `transform`, a layer's local matrix,
    /// exactly.
    fn expresses_transform(transform: Affine) -> bool;

    /// Whether a system layer clips its sublayers to `clip`, in the layer's
    /// own space, exactly.
    fn expresses_clip(clip: &ShapeData) -> bool;

    /// Whether a system layer shows `frame` itself, with the colour the
    /// frame declares: its planes are a buffer the system compositor can
    /// scan out. Only such frames are candidates.
    fn shows(frame: &ExternalFrame) -> bool;
}

/// Why a candidate layer stays composited in the engine.
///
/// Each cause names the rule it failed. Opportunistic promotion keeps such
/// a layer in the engine; content that can only be shown on a plane turns
/// the same cause into a render error.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Ineligible {
    /// An ancestor composites its subtree through an offscreen (opacity
    /// below one, a filter, or a blend), so the layer never reaches the
    /// surface level a plane sits at.
    #[error("ancestor layer {0:?} isolates its subtree into an offscreen")]
    Isolated(LayerId),
    /// The layer carries a filter, which must process its pixels.
    #[error("it carries a filter")]
    Filter,
    /// The layer composites with a non-default blend mode, or a child blends
    /// onto its content.
    #[error("it uses a non-default blend mode")]
    Blend,
    /// A layer painted above it blends with a non-default mode onto the
    /// surface, which would need the layer's pixels.
    #[error("layer {0:?} painted above it blends with a non-default mode")]
    BlendAbove(LayerId),
    /// The layer samples a backdrop.
    #[error("it samples a backdrop")]
    Backdrop,
    /// A layer painted above it samples a backdrop, whose capture would need
    /// the layer's pixels.
    #[error("layer {0:?} painted above it samples a backdrop")]
    BackdropAbove(LayerId),
    /// Its opacity is below one and it has child layers: the opacity applies
    /// to the group, which a plane and the part above it cannot share.
    #[error("its opacity applies to a group of child layers")]
    GroupOpacity,
    /// A transform on the path to the layer is not expressible by the
    /// system layer.
    #[error("layer {0:?}'s transform is not expressible by the system compositor")]
    Transform(LayerId),
    /// A clip on the path to the layer is not expressible by the system
    /// layer.
    #[error("layer {0:?}'s clip is not expressible by the system compositor")]
    Clip(LayerId),
    /// A clip on the path to the layer nests inside another clip that is not
    /// a device-aligned rectangle; the engine composites that pair through a
    /// clip offscreen, which the layer would then sit in.
    #[error("layer {0:?}'s clip nests inside another non-rectangular clip")]
    NestedClip(LayerId),
    /// The surface's plane budget is spent.
    #[error("the plane budget of {0} per surface is spent")]
    Budget(usize),
}

/// One level of the path from the root to a promoted layer: the sampled
/// properties of that tree layer, with the tree's semantics. The level's own
/// space is its parent's content space times `transform`; `clip` applies in
/// that space; content and children sit in that space translated by
/// `-scroll`.
#[derive(Clone, Debug, PartialEq)]
pub struct Level {
    /// The tree layer this level mirrors.
    pub layer: LayerId,
    /// The local transform.
    pub transform: Affine,
    /// The clip, in the level's own space.
    pub clip: Option<ShapeData>,
    /// The snapped scroll offset.
    pub scroll: Vec2,
}

impl Level {
    /// `transform * translate(-scroll)`: the space of the level's content
    /// and children.
    #[must_use]
    #[cfg_attr(
        not(any(test, target_os = "android")),
        expect(
            dead_code,
            reason = "a flattened placement, for realizations without nested layers"
        )
    )]
    pub fn content_transform(&self) -> Affine {
        self.transform * Affine::translate(-self.scroll)
    }
}

/// Where a promoted layer sits.
#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    /// The promoted layer.
    pub layer: LayerId,
    /// The content rectangle `(0, 0, w, h)` in the layer's content space.
    pub size: (u32, u32),
    /// The layer's opacity; every ancestor is opaque by eligibility.
    pub opacity: f32,
    /// The root first, the promoted layer last.
    pub path: Vec<Level>,
}

impl Placement {
    /// Content space to device space: the product of every level's content
    /// transform.
    #[must_use]
    #[cfg_attr(
        not(any(test, target_os = "android")),
        expect(
            dead_code,
            reason = "a flattened placement, for realizations without nested layers"
        )
    )]
    pub fn content_to_device(&self) -> Affine {
        self.path.iter().fold(Affine::IDENTITY, |acc, level| {
            acc * level.content_transform()
        })
    }
}

/// A surface's promotion decision for one frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    /// The promoted layers, in paint order.
    pub planes: Vec<Placement>,
    /// The candidates kept in the engine, in paint order, with their cause.
    pub rejected: Vec<(LayerId, Ineligible)>,
    /// Whether any layer is painted after the last promoted one, so an
    /// engine part exists above it.
    pub trailing: bool,
}

impl Plan {
    /// The number of engine parts: one below each plane, plus the trailing
    /// part.
    #[must_use]
    pub fn parts(&self) -> usize {
        self.planes.len() + usize::from(self.trailing || self.planes.is_empty())
    }
}

/// Whether a layer composites its subtree through an offscreen: the rule the
/// lowering applies (`Lowering::layer`). A projective layer's subtree renders
/// into its local image — the same offscreen isolation — and the root renders
/// into the surface target and never isolates for blended children.
fn isolates(tree: &SurfaceTree, id: LayerId) -> bool {
    let node = tree.layer(id);
    node.filter.is_some()
        || node.opacity < 1.0
        || node.blend != BlendMode::Normal
        || tree.projective_pose(id).is_some()
        || (id != tree.root() && node.blends_within())
}

/// One layer in paint order.
struct Visit {
    id: LayerId,
    /// The index in `order` of each ancestor, root first.
    ancestors: Vec<usize>,
}

/// Decides which of `candidates` (layer to content size) are promoted this
/// frame on a surface realized by compositor `C`.
///
/// Candidates are judged in paint order and the budget goes to the first
/// eligible ones, so the decision depends only on the tree and the
/// candidate set.
#[must_use]
pub fn plan<C: Compositor>(
    tree: &SurfaceTree,
    candidates: &FxHashMap<LayerId, (u32, u32)>,
) -> Plan {
    if candidates.is_empty() {
        return Plan::default();
    }
    let order = paint_order(tree);
    // The first layer at or after each index that samples a backdrop or
    // blends onto the surface, scanning from the top of the paint order.
    let mut backdrop_above = vec![None; order.len() + 1];
    let mut blend_above = vec![None; order.len() + 1];
    for (i, visit) in order.iter().enumerate().rev() {
        let node = tree.layer(visit.id);
        backdrop_above[i] = if node.backdrop.is_some() {
            Some(visit.id)
        } else {
            backdrop_above[i + 1]
        };
        let at_surface = visit
            .ancestors
            .iter()
            .all(|&a| !isolates(tree, order[a].id));
        blend_above[i] = if at_surface && node.blend != BlendMode::Normal {
            Some(visit.id)
        } else {
            blend_above[i + 1]
        };
    }
    let mut plan = Plan::default();
    let mut last = None;
    for (i, visit) in order.iter().enumerate() {
        let Some(&size) = candidates.get(&visit.id) else {
            continue;
        };
        let verdict = judge::<C>(tree, &order, i, backdrop_above[i + 1], blend_above[i + 1]);
        match verdict {
            Ok(()) if plan.planes.len() >= C::BUDGET => plan
                .rejected
                .push((visit.id, Ineligible::Budget(C::BUDGET))),
            Ok(()) => {
                last = Some(i);
                plan.planes.push(placement(tree, &order, i, size));
            }
            Err(cause) => plan.rejected.push((visit.id, cause)),
        }
    }
    plan.trailing = last.is_some_and(|i| i + 1 < order.len());
    plan
}

/// The eligibility rules for the candidate at `order[i]`, in a fixed order
/// so the named cause is deterministic.
fn judge<C: Compositor>(
    tree: &SurfaceTree,
    order: &[Visit],
    i: usize,
    backdrop_above: Option<LayerId>,
    blend_above: Option<LayerId>,
) -> Result<(), Ineligible> {
    let visit = &order[i];
    let node = tree.layer(visit.id);
    if let Some(&a) = visit
        .ancestors
        .iter()
        .find(|&&a| isolates(tree, order[a].id))
    {
        return Err(Ineligible::Isolated(order[a].id));
    }
    if node.filter.is_some() {
        return Err(Ineligible::Filter);
    }
    if node.blend != BlendMode::Normal || node.blends_within() {
        return Err(Ineligible::Blend);
    }
    if node.backdrop.is_some() {
        return Err(Ineligible::Backdrop);
    }
    if let Some(layer) = backdrop_above {
        return Err(Ineligible::BackdropAbove(layer));
    }
    if let Some(layer) = blend_above {
        return Err(Ineligible::BlendAbove(layer));
    }
    if node.opacity < 1.0 && !node.children.is_empty() {
        return Err(Ineligible::GroupOpacity);
    }
    // The engine merges nested clips in place only while at most one of
    // them is not a device-aligned rectangle (`Lowering::run_clipped`).
    let mut shaped_clip = false;
    let mut space = Affine::IDENTITY;
    for id in visit
        .ancestors
        .iter()
        .map(|&a| order[a].id)
        .chain([visit.id])
    {
        let level = tree.layer(id);
        // A projective level's placement is its pose, not `transform` —
        // a plane's affine levels cannot carry it. An isolating ancestor
        // was already named `Isolated`, so this can only be the candidate.
        if tree.projective_pose(id).is_some() || !C::expresses_transform(level.transform) {
            return Err(Ineligible::Transform(id));
        }
        let own = space * level.transform;
        if let Some(clip) = &level.clip {
            if !C::expresses_clip(clip) {
                return Err(Ineligible::Clip(id));
            }
            if !(matches!(clip, ShapeData::Rect(_)) && axis_aligned(own)) {
                if shaped_clip {
                    return Err(Ineligible::NestedClip(id));
                }
                shaped_clip = true;
            }
        }
        space = own * Affine::translate(-level.scroll_offset);
    }
    Ok(())
}

fn placement(tree: &SurfaceTree, order: &[Visit], i: usize, size: (u32, u32)) -> Placement {
    let visit = &order[i];
    let path = visit
        .ancestors
        .iter()
        .map(|&a| order[a].id)
        .chain([visit.id])
        .map(|id| {
            let node = tree.layer(id);
            Level {
                layer: id,
                transform: node.transform,
                clip: node.clip.clone(),
                scroll: node.scroll_offset,
            }
        })
        .collect();
    Placement {
        layer: visit.id,
        size,
        opacity: tree.layer(visit.id).opacity,
        path,
    }
}

/// Every layer in paint order: a layer's content, then its children.
fn paint_order(tree: &SurfaceTree) -> Vec<Visit> {
    let mut order = Vec::new();
    let mut stack = vec![(tree.root(), Vec::new())];
    while let Some((id, ancestors)) = stack.pop() {
        let index = order.len();
        let children = &tree.layer(id).children;
        for &child in children.iter().rev() {
            let mut path = ancestors.clone();
            path.push(index);
            stack.push((child, path));
        }
        order.push(Visit { id, ancestors });
    }
    order
}

/// Whether a frame whose only committed change is new external frames on
/// `layers` can present through the planes alone: every changed layer is
/// promoted by the surface's committed `plan`, and the plan recomputed
/// over the frame's tree with the fresh `candidates` is the committed
/// one — a new frame's different size, or a new frame no plane can show,
/// changes the candidate set and fails the check, keeping those layers on
/// the full path like any other change (#90).
#[must_use]
pub fn frames_only<C: Compositor>(
    plan: &Plan,
    tree: &SurfaceTree,
    candidates: &FxHashMap<LayerId, (u32, u32)>,
    layers: &FxHashSet<LayerId>,
) -> bool {
    !layers.is_empty()
        && layers
            .iter()
            .all(|layer| plan.planes.iter().any(|plane| plane.layer == *layer))
        && self::plan::<C>(tree, candidates) == *plan
}

/// The content a plane shows.
#[derive(Debug)]
#[cfg_attr(
    not(any(target_vendor = "apple", target_os = "android")),
    expect(
        dead_code,
        reason = "read by the platform realizations of `SystemPlanes`"
    )
)]
pub enum PlaneContent<'a> {
    /// A retained external frame, handed to the system compositor instead of
    /// being sampled by the engine.
    Frame {
        /// The installed frame.
        frame: &'a ExternalFrame,
        /// Bumped every time a new frame is installed on the layer.
        generation: u64,
    },
}

/// One promoted plane of a [`Composition`].
#[derive(Debug)]
#[cfg_attr(
    not(any(target_vendor = "apple", target_os = "android")),
    expect(
        dead_code,
        reason = "read by the platform realizations of `SystemPlanes`"
    )
)]
pub struct Plane<'a> {
    /// Where it sits.
    pub placement: &'a Placement,
    /// What it shows.
    pub content: PlaneContent<'a>,
}

/// One engine part of a [`Composition`]: premultiplied linear Display P3 at
/// the surface size.
#[derive(Debug)]
#[cfg_attr(
    not(any(target_vendor = "apple", target_os = "android")),
    expect(
        dead_code,
        reason = "read by the platform realizations of `SystemPlanes`"
    )
)]
pub struct Part<'a> {
    /// The engine texture holding the part.
    pub view: &'a wgpu::TextureView,
}

/// A surface's stack for one frame, bottom first: `parts[0]`, `planes[0]`,
/// `parts[1]`, `planes[1]`, … and, when [`Plan::trailing`] holds, a last
/// part above the last plane. Without promoted planes there is exactly one
/// part, the whole surface.
#[cfg_attr(
    not(any(target_vendor = "apple", target_os = "android")),
    expect(
        dead_code,
        reason = "read by the platform realizations of `SystemPlanes`"
    )
)]
pub struct Composition<'a> {
    /// The engine's device.
    pub device: &'a wgpu::Device,
    /// The engine's queue.
    pub queue: &'a wgpu::Queue,
    /// Presents an engine part into a platform drawable, tone-mapping to the
    /// display's headroom.
    pub presenter: &'a mut Presenter,
    /// The surface size in device pixels.
    pub size: (u32, u32),
    /// The display the surface is on.
    pub display: Display,
    /// The engine parts, bottom first.
    pub parts: &'a [Part<'a>],
    /// The promoted planes, bottom first.
    pub planes: &'a [Plane<'a>],
}

/// A platform's realization of a surface's planes under the host's
/// system-compositor parent.
///
/// The render thread calls [`SystemPlanes::compose`] once per rendered
/// frame with the whole stack; the realization makes the system tree match
/// it, atomically where the platform allows, and presents the parts.
pub trait SystemPlanes: Compositor {
    /// Realizes `composition`. Returns false when a part's drawable was not
    /// available (the window is occluded or the acquire timed out): nothing
    /// changed on screen, and the engine composes again on the next frame.
    ///
    /// # Errors
    /// A [`RenderError`] naming the cause when the system rejects a plane
    /// or a part cannot be presented.
    fn compose(&mut self, composition: Composition<'_>) -> Result<bool, RenderError>;

    /// Presents only the promoted planes' new frames: `frames` carries
    /// every promoted layer whose frame changed this frame, inside the
    /// transaction or equivalent atomic update the platform composes, and
    /// every part's shown buffer stays in place — the engine did no work
    /// for this frame, so there is nothing to blit and no buffer to
    /// acquire. Called only for a frame [`frames_only`] admitted, so the
    /// stack itself is the one `compose` last realized (#90).
    ///
    /// # Errors
    /// A [`RenderError`] naming the cause when the system rejects a plane.
    fn refresh(&mut self, frames: &[Plane<'_>]) -> Result<(), RenderError>;

    /// The surface was resized to `size` device pixels.
    fn resize(&mut self, size: (u32, u32));

    /// A display move or a scale change re-runs every part's output
    /// negotiation — each part's [`WindowSurface::reselect`].
    fn reselect(&mut self, adapter: &wgpu::Adapter, device: &wgpu::Device);
}

#[cfg(target_vendor = "apple")]
pub mod apple;

/// The realization on this platform: Core Animation layer planes.
#[cfg(target_vendor = "apple")]
pub type Platform = apple::LayerPlanes;
/// The realization on this platform: child surface controls on Android.
#[cfg(target_os = "android")]
pub type Platform = super::surface_control::planes::Planes;
/// The realization on this platform.
#[cfg(not(any(target_vendor = "apple", target_os = "android")))]
pub type Platform = NoPlanes;

/// Stands in for [`SystemPlanes`] on platforms without a realization yet:
/// no value exists, so a surface there never has planes.
#[cfg(not(any(target_vendor = "apple", target_os = "android")))]
#[derive(Debug)]
pub enum NoPlanes {}

#[cfg(not(any(target_vendor = "apple", target_os = "android")))]
impl Compositor for NoPlanes {
    const BUDGET: usize = 0;
    fn expresses_transform(_: Affine) -> bool {
        false
    }
    fn expresses_clip(_: &ShapeData) -> bool {
        false
    }
    fn shows(_: &ExternalFrame) -> bool {
        false
    }
}

#[cfg(not(any(target_vendor = "apple", target_os = "android")))]
impl SystemPlanes for NoPlanes {
    fn compose(&mut self, _: Composition<'_>) -> Result<bool, RenderError> {
        unreachable!("no `NoPlanes` value exists")
    }
    fn refresh(&mut self, _: &[Plane<'_>]) -> Result<(), RenderError> {
        unreachable!("no `NoPlanes` value exists")
    }
    fn resize(&mut self, _: (u32, u32)) {
        unreachable!("no `NoPlanes` value exists")
    }
    fn reselect(&mut self, _: &wgpu::Adapter, _: &wgpu::Device) {
        unreachable!("no `NoPlanes` value exists")
    }
}

#[cfg(test)]
mod tests;
