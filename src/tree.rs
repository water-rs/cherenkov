// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The render thread's copy of a surface's layer tree: the layer graph,
//! every layer property, its animation track and the sampled value for the
//! current frame.
//!
//! The backend never receives property ops; it reads the sampled tree
//! through [`SurfaceFrame`](crate::SurfaceFrame).

use std::collections::HashMap;
use std::time::Instant;

use kurbo::{Affine, Vec2};

use crate::animation::{
    Animatable, Animation, Lanes, clamp_to_rect, curve_value, decay_step, rubber_band_spring,
    settled, spring_step,
};
use crate::backend::Display;
use crate::frame::RefreshRange;
use crate::message::{BackdropId, LayerId, LayerOp, Prop};
use crate::shape::ShapeData;
use crate::style::{BlendMode, FilterId};

/// The fast rate class: springs, curves and fast decays run here.
pub const RATE_FAST: RefreshRange = 60..=120;
/// The slow rate class: only decays slower than one device pixel per frame
/// at 60 Hz remain.
pub const RATE_SLOW: RefreshRange = 30..=60;

/// A surface's layer tree on the render thread.
pub struct SurfaceTree {
    nodes: HashMap<u64, LayerNode>,
    root: LayerId,
}

/// One layer's sampled state for the current frame.
pub struct LayerNode {
    /// The local transform.
    pub transform: Affine,
    /// The opacity.
    pub opacity: f32,
    /// The scroll offset, snapped to the display's device-pixel grid.
    pub scroll_offset: Vec2,
    /// The clip shape, in the layer's own space.
    pub clip: Option<ShapeData>,
    /// The blend mode the layer composites onto its parent with.
    pub blend: BlendMode,
    /// The filter applied to this layer's subtree.
    pub filter: Option<FilterId>,
    /// The backdrop group this layer samples.
    pub backdrop: Option<BackdropId>,
    /// The child layers, in paint order.
    pub children: Vec<LayerId>,
    parent: Option<LayerId>,
    transform_track: Option<Track<Affine>>,
    opacity_track: Option<Track<f32>>,
    scroll_track: Option<Track<Vec2>>,
}

impl std::fmt::Debug for LayerNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerNode")
            .field("transform", &self.transform)
            .field("opacity", &self.opacity)
            .field("scroll_offset", &self.scroll_offset)
            .field("clip", &self.clip)
            .field("blend", &self.blend)
            .field("filter", &self.filter)
            .field("backdrop", &self.backdrop)
            .field("children", &self.children)
            .finish_non_exhaustive()
    }
}

impl LayerNode {
    fn new() -> Self {
        Self {
            transform: Affine::IDENTITY,
            opacity: 1.0,
            scroll_offset: Vec2::ZERO,
            clip: None,
            blend: BlendMode::default(),
            filter: None,
            backdrop: None,
            children: Vec::new(),
            parent: None,
            transform_track: None,
            opacity_track: None,
            scroll_track: None,
        }
    }

    /// `transform * translate(-scroll_offset)`: the space of the content
    /// and children. The clip applies in `transform` space, so scrolling
    /// moves content and children inside the clip and never re-records
    /// anything.
    #[must_use]
    pub fn content_transform(&self) -> Affine {
        self.transform * Affine::translate(-self.scroll_offset)
    }
}

/// One running animation track.
struct Track<T: Animatable> {
    /// The position the track started from (its retarget snapshot).
    from: T::Lanes,
    /// The velocity the track started with.
    velocity: T::Lanes,
    /// The value the track moves toward.
    target: T,
    /// The animation driving the track.
    animation: Animation,
    /// The time the track started; `None` until first sampled, so a track
    /// committed between frames starts at the next presentation time.
    start: Option<Instant>,
    /// The last sampled `(time, position, velocity)`, for retarget
    /// continuity.
    last: Option<(Instant, T::Lanes, T::Lanes)>,
}

impl<T: Animatable> Track<T> {
    fn new(from: T::Lanes, velocity: T::Lanes, target: T, animation: Animation) -> Self {
        debug_assert!(
            !matches!(animation, Animation::Decay(_)) || T::Lanes::N == 2,
            "Decay is only legal on scroll_offset"
        );
        Self {
            from,
            velocity,
            target,
            animation,
            start: None,
            last: None,
        }
    }

    /// Evaluates the track at `t`, returning `(position, velocity, settled)`
    /// in lane space. A settled spring reports its target exactly.
    fn sample(&mut self, t: Instant) -> (T::Lanes, T::Lanes, bool) {
        let start = *self.start.get_or_insert(t);
        let dt = t.duration_since(start).as_secs_f64();
        let target = self.target.into_lanes();
        let (pos, vel, done) = match &self.animation {
            Animation::Spring(spring) => {
                let (pos, vel) = spring_step(self.from, self.velocity, target, spring, dt);
                if settled(pos, vel, target) {
                    (target, T::Lanes::zero(), true)
                } else {
                    (pos, vel, false)
                }
            }
            Animation::Curve(curve) => {
                let duration = curve.duration.as_secs_f64();
                let t01 = if duration <= 0.0 { 1.0 } else { dt / duration };
                let delta = target.sub(&self.from);
                let pos = self.from.add_scaled(&delta, curve_value(curve, t01));
                // v = Δ·e′(t)/duration so a retarget can inherit it.
                let vel = if duration > 0.0 {
                    delta.scale(crate::animation::curve_slope(curve, t01) / duration)
                } else {
                    T::Lanes::zero()
                };
                (pos, vel, t01 >= 1.0)
            }
            Animation::Decay(decay) => {
                let (pos, vel) = decay_step(self.from, self.velocity, decay.deceleration, dt);
                (pos, vel, vel.max_abs() < 1e-3)
            }
        };
        self.last = Some((t, pos, vel));
        (pos, vel, done)
    }

    /// Whether the still-running track needs the fast rate class. A decay
    /// needs only the slow class once it runs slower than one device pixel
    /// per frame at 60 Hz (`scale * 60` logical pixels per second).
    fn is_fast(&self, scale: f64) -> bool {
        match self.animation {
            Animation::Decay(_) => {
                self.last.map_or(self.velocity, |(_, _, vel)| vel).max_abs() >= scale * 60.0
            }
            Animation::Spring(_) | Animation::Curve(_) => true,
        }
    }
}

/// The outcome of [`SurfaceTree::sample`].
#[derive(Clone, Debug)]
pub struct Sampling {
    /// Whether any animation step touched the tree.
    pub stepped: bool,
    /// The refresh rate still-running animations need.
    pub rate: Option<RefreshRange>,
}

impl Default for SurfaceTree {
    fn default() -> Self {
        Self::new()
    }
}

impl SurfaceTree {
    /// An empty tree holding only its root layer (`LayerId(0)`).
    #[must_use]
    pub fn new() -> Self {
        let mut nodes = HashMap::new();
        nodes.insert(0, LayerNode::new());
        Self {
            nodes,
            root: LayerId::new(0),
        }
    }

    /// The root layer.
    #[must_use]
    pub const fn root(&self) -> LayerId {
        self.root
    }

    /// The sampled node of layer `id`.
    ///
    /// # Panics
    /// Panics when `id` is not in the tree.
    #[must_use]
    pub fn layer(&self, id: LayerId) -> &LayerNode {
        self.nodes
            .get(&id.raw())
            .unwrap_or_else(|| panic!("layer {} is not in the tree", id.raw()))
    }

    /// Every layer in the tree. Order is unspecified.
    pub fn layers(&self) -> impl Iterator<Item = (LayerId, &LayerNode)> + '_ {
        self.nodes
            .iter()
            .map(|(id, node)| (LayerId::new(*id), node))
    }

    /// Removes `id` and its descendants. Returns every removed id.
    ///
    /// # Panics
    /// Panics when `id` is not in the tree or is the root.
    pub fn remove(&mut self, id: LayerId) -> Vec<LayerId> {
        assert!(
            self.nodes.contains_key(&id.raw()),
            "removing unknown layer {}",
            id.raw()
        );
        assert_ne!(id, self.root, "the root layer cannot be removed");
        self.detach(id);
        let mut removed = Vec::new();
        let mut stack = vec![id];
        while let Some(current) = stack.pop() {
            let Some(node) = self.nodes.remove(&current.raw()) else {
                continue;
            };
            stack.extend(node.children.iter().copied());
            removed.push(current);
        }
        removed
    }

    /// Applies one committed layer op.
    ///
    /// # Panics
    /// Panics on an op naming a layer that is not in the tree: after
    /// `Create` ordering is respected, an unknown layer is an invariant
    /// violation. Attaching the root or closing a cycle also panics before
    /// mutating the tree.
    pub fn apply(&mut self, op: LayerOp) {
        match op {
            LayerOp::Create(id) => {
                assert!(
                    self.nodes.insert(id.raw(), LayerNode::new()).is_none(),
                    "layer {} created twice",
                    id.raw()
                );
            }
            LayerOp::Remove(_) => unreachable!("Remove is handled by the render loop"),
            LayerOp::Transform(id, prop) => {
                let node = self.node_mut(id);
                set_prop(&mut node.transform_track, &mut node.transform, &prop);
            }
            LayerOp::Opacity(id, prop) => {
                let node = self.node_mut(id);
                set_prop(&mut node.opacity_track, &mut node.opacity, &prop);
            }
            LayerOp::ScrollOffset(id, prop) => {
                let node = self.node_mut(id);
                set_prop(&mut node.scroll_track, &mut node.scroll_offset, &prop);
            }
            LayerOp::Clip(id, clip) => self.node_mut(id).clip = clip,
            LayerOp::Blend(id, blend) => self.node_mut(id).blend = blend,
            LayerOp::Filter(id, filter) => self.node_mut(id).filter = filter,
            LayerOp::Backdrop(id, backdrop) => self.node_mut(id).backdrop = backdrop,
            LayerOp::Content(id, _) => {
                // Validated here; forwarded to the renderer by the loop.
                assert!(
                    self.nodes.contains_key(&id.raw()),
                    "content on unknown layer {}",
                    id.raw()
                );
            }
            LayerOp::Push { parent, child } => {
                assert!(
                    self.nodes.contains_key(&child.raw()),
                    "pushing unknown layer {}",
                    child.raw()
                );
                self.assert_attachment(parent, child);
                self.detach(child);
                self.node_mut(child).parent = Some(parent);
                self.node_mut(parent).children.push(child);
            }
            LayerOp::Insert {
                parent,
                index,
                child,
            } => {
                assert!(
                    self.nodes.contains_key(&child.raw()),
                    "inserting unknown layer {}",
                    child.raw()
                );
                self.assert_attachment(parent, child);
                self.detach(child);
                self.node_mut(child).parent = Some(parent);
                let node = self.node_mut(parent);
                let index = index.min(node.children.len());
                node.children.insert(index, child);
            }
            LayerOp::Detach { parent, child } => {
                assert_eq!(
                    self.layer(child).parent,
                    Some(parent),
                    "detaching from the wrong parent"
                );
                self.detach(child);
            }
        }
    }

    fn assert_attachment(&self, parent: LayerId, child: LayerId) {
        assert_ne!(child, self.root, "the root layer cannot be attached");
        let mut cursor = Some(parent);
        while let Some(id) = cursor {
            assert_ne!(id, child, "layer attachment would create a cycle");
            cursor = self.layer(id).parent;
        }
    }

    fn detach(&mut self, child: LayerId) {
        if let Some(parent) = self.node_mut(child).parent.take() {
            self.node_mut(parent).children.retain(|c| *c != child);
        }
    }

    fn node_mut(&mut self, id: LayerId) -> &mut LayerNode {
        self.nodes
            .get_mut(&id.raw())
            .unwrap_or_else(|| panic!("layer {} is not in the tree", id.raw()))
    }

    /// Samples every animation track at `time`, updating the layers'
    /// sampled properties. `display.scale` snaps the scroll offset to the
    /// device-pixel grid and classifies slow decays.
    pub fn sample(&mut self, time: Instant, display: Display) -> Sampling {
        let mut stepped = false;
        let mut fast = false;
        let mut slow = false;
        for node in self.nodes.values_mut() {
            if let Some(track) = &mut node.transform_track {
                stepped = true;
                let (pos, _vel, done) = track.sample(time);
                node.transform = Affine::from_lanes(pos);
                if done {
                    node.transform = track.target;
                    node.transform_track = None;
                }
            }
            if let Some(track) = &mut node.opacity_track {
                stepped = true;
                let (pos, _vel, done) = track.sample(time);
                node.opacity = f32::from_lanes(pos);
                if done {
                    node.opacity = track.target;
                    node.opacity_track = None;
                }
            }
            if let Some(track) = &mut node.scroll_track {
                stepped = true;
                let (pos, vel, done) = track.sample(time);
                // Rubber-band handoff: the moment the offset leaves the
                // bounds, the remaining motion becomes a critically damped
                // spring back to the nearest point of the bounds, so the
                // overshoot and the return are one continuous motion.
                if let Animation::Decay(decay) = track.animation
                    && let Some(bounds) = decay.rubber_band
                {
                    let offset = Vec2::from_lanes(pos);
                    // "Outside the bounds" = clamping changes the offset:
                    // `Rect::contains` is half-open and never holds on a
                    // degenerate edge (a zero-width bounds rect), while a
                    // scroll axis exactly on the bound must stay in.
                    let clamped = clamp_to_rect(offset, bounds);
                    if clamped != offset {
                        *track = Track {
                            from: pos,
                            velocity: vel,
                            target: clamped,
                            animation: Animation::Spring(rubber_band_spring()),
                            start: Some(time),
                            last: track.last,
                        };
                    }
                }
                // A decay keeps where it stopped; a settled spring (including
                // the rubber-band handoff) reports its target exactly.
                node.scroll_offset = snap(Vec2::from_lanes(pos), display.scale);
                if done {
                    node.scroll_track = None;
                }
            }
            // Rate classification of the tracks that remain.
            for running_track in [
                node.transform_track
                    .as_ref()
                    .map(|t| t.is_fast(display.scale)),
                node.opacity_track
                    .as_ref()
                    .map(|t| t.is_fast(display.scale)),
                node.scroll_track.as_ref().map(|t| t.is_fast(display.scale)),
            ]
            .into_iter()
            .flatten()
            {
                if running_track {
                    fast = true;
                } else {
                    slow = true;
                }
            }
        }
        let rate = if fast {
            Some(RATE_FAST)
        } else if slow {
            Some(RATE_SLOW)
        } else {
            None
        };
        Sampling { stepped, rate }
    }
}

/// Applies a property change to a track: an animated change retargets from
/// the old track's last sampled position and velocity (its start state when
/// never sampled), so a spring retargeted mid-flight is continuous in
/// position and velocity and a curve restarts from its current value. A
/// change without an animation snaps and drops the track.
fn set_prop<T: Animatable>(track: &mut Option<Track<T>>, value: &mut T, prop: &Prop<T>) {
    match prop.animation {
        None => {
            *value = prop.target;
            *track = None;
        }
        Some(animation @ Animation::Decay(decay)) => {
            // A decay is a fling: it starts AT the committed value with its
            // own velocity; the previous track is irrelevant.
            let mut velocity = T::Lanes::zero();
            for (i, v) in [decay.velocity.x, decay.velocity.y]
                .into_iter()
                .take(T::Lanes::N)
                .enumerate()
            {
                velocity.set(i, v);
            }
            *track = Some(Track::new(
                prop.target.into_lanes(),
                velocity,
                prop.target,
                animation,
            ));
        }
        Some(animation) => {
            let (from, velocity, last) = track.as_ref().map_or_else(
                || ((*value).into_lanes(), T::Lanes::zero(), None),
                |old| {
                    (
                        old.last.map_or(old.from, |(_, pos, _)| pos),
                        old.last.map_or(old.velocity, |(_, _, vel)| vel),
                        old.last,
                    )
                },
            );
            let mut next = Track::new(from, velocity, prop.target, animation);
            next.last = last;
            *track = Some(next);
        }
    }
}

/// Snaps an offset to the device-pixel grid: `round(offset · scale) /
/// scale`. The track itself is not snapped, so a slow decay still settles
/// smoothly.
fn snap(offset: Vec2, scale: f64) -> Vec2 {
    if scale <= 0.0 {
        return offset;
    }
    Vec2::new(
        (offset.x * scale).round() / scale,
        (offset.y * scale).round() / scale,
    )
}

#[cfg(test)]
mod hierarchy_tests {
    use super::*;

    fn tree() -> SurfaceTree {
        let mut tree = SurfaceTree::new();
        for id in 1..=3 {
            tree.apply(LayerOp::Create(LayerId::new(id)));
        }
        tree.apply(LayerOp::Push {
            parent: tree.root(),
            child: LayerId::new(1),
        });
        tree.apply(LayerOp::Push {
            parent: LayerId::new(1),
            child: LayerId::new(2),
        });
        tree.apply(LayerOp::Push {
            parent: tree.root(),
            child: LayerId::new(3),
        });
        tree
    }

    #[test]
    #[should_panic(expected = "layer attachment would create a cycle")]
    fn attaching_a_layer_under_its_subtree_is_a_cycle() {
        let mut tree = tree();
        tree.apply(LayerOp::Push {
            parent: LayerId::new(2),
            child: LayerId::new(1),
        });
    }

    #[test]
    #[should_panic(expected = "layer attachment would create a cycle")]
    fn inserting_a_layer_under_itself_is_a_cycle() {
        let mut tree = tree();
        tree.apply(LayerOp::Insert {
            parent: LayerId::new(1),
            child: LayerId::new(1),
            index: 0,
        });
    }

    #[test]
    #[should_panic(expected = "the root layer cannot be attached")]
    fn attaching_the_root_is_rejected() {
        let mut tree = tree();
        tree.apply(LayerOp::Push {
            parent: LayerId::new(2),
            child: tree.root(),
        });
    }

    #[test]
    fn detach_and_remove_preserve_parent_links() {
        let mut tree = tree();
        tree.apply(LayerOp::Push {
            parent: LayerId::new(3),
            child: LayerId::new(1),
        });
        assert_eq!(tree.layer(tree.root()).children, [LayerId::new(3)]);
        assert_eq!(tree.layer(LayerId::new(1)).parent, Some(LayerId::new(3)));
        assert_eq!(
            tree.remove(LayerId::new(1)),
            [LayerId::new(1), LayerId::new(2)]
        );
        assert!(tree.layer(LayerId::new(3)).children.is_empty());
    }
}
