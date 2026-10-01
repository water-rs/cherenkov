//! The render thread's loop: owns the backend renderer and one
//! [`SurfaceTree`] per surface, applies commits, samples animations at the
//! frame time, renders, and answers with [`Next`] and the [`FrameStats`].

use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::backend::{Backend, Display, Frame, Redraw, Renderer, SurfaceFrame, SurfaceInfo};
use crate::error::{EngineError, RenderError, ResourceError, SurfaceError};
use crate::frame::{FrameId, FrameStats, Next};
use crate::image::ImageUpload;
use crate::message::{BackdropShaderId, ChangeSet, LayerOp, Message, Op, ResOp, SurfaceId};
use crate::paint::ImageId;
use crate::resource::ResourceId;
use crate::tree::SurfaceTree;
use crate::{BackdropEffect, WorkingColor};

/// Whether a surface's frames can ask the backend to present, and the
/// pending presentation state when they can. Only a surface the backend
/// reported as presenting carries the flag — a pending present cannot
/// exist for an offscreen target (#98).
enum Presentation {
    /// The surface retains pixels; there is no swapchain to present to,
    /// so a `Display` update never marks a present.
    Retained,
    /// The surface presents to a display; `pending` asks for a frame even
    /// without `changed` — a headroom-only `Display` update reaches the
    /// swapchain without touching the layer tree or any content cache.
    Presenting {
        /// Whether the surface's window should present without new
        /// content.
        pending: bool,
    },
}

/// One surface's render-thread state.
struct SurfaceState {
    tree: SurfaceTree,
    size: (u32, u32),
    display: Display,
    clear: WorkingColor,
    /// Whether a property op, a content op or an animation step touched the
    /// surface since the last render.
    changed: bool,
    /// Whether the surface presents, and whether a present is pending (#98).
    presentation: Presentation,
    /// Whether the host announced the surface moved to another display
    /// since the previous frame — the frame's `display_moved` (#98).
    display_moved: bool,
    /// Whether the surface's recorded contents still run operand animations
    /// on the UI thread. The tracks live there; they need the next frame's
    /// sample at the fast rate class.
    content_animating: bool,
}

impl SurfaceState {
    /// Whether the frame should ask the backend to present.
    const fn present_pending(&self) -> bool {
        matches!(
            self.presentation,
            Presentation::Presenting { pending: true }
        )
    }

    /// Marks the next frame for presentation; a no-op on a retained
    /// surface, which cannot hold the flag.
    const fn mark_present(&mut self) {
        if let Presentation::Presenting { pending } = &mut self.presentation {
            *pending = true;
        }
    }

    /// Consumes the pending present after a render.
    const fn presented(&mut self) {
        if let Presentation::Presenting { pending } = &mut self.presentation {
            *pending = false;
        }
    }
}

/// A rejection the backend reported after the resource's handle was
/// returned.
struct Rejection {
    reason: Arc<ResourceError>,
    /// Whether the backend still holds the resource: true after a rejected
    /// image replacement, which keeps the previous pixels, false after a
    /// rejected registration, which committed nothing.
    held: bool,
}

/// A released resource that installed content still draws. Its backend
/// removal waits until no surface's installed content draws it (#199).
struct PendingRelease<B: Backend> {
    remove: ResOp<B>,
    /// The surfaces whose installed content draws the resource.
    surfaces: FxHashSet<SurfaceId>,
}

/// The render loop's per-resource bookkeeping, shared by the native render
/// thread and the browser executor.
///
/// - A rejection the backend reported fails every render that draws the
///   resource with [`RenderError::Rejected`], until a replacement succeeds
///   or the resource is freed.
/// - A released resource is freed only once no surface's installed
///   content draws it: the release waits while one does, and is carried
///   out when a commit, a layer removal or a surface's destruction leaves
///   no surface drawing it. Ids are never reused, so a pending id cannot
///   name another resource.
struct Resources<B: Backend> {
    rejections: FxHashMap<ResourceId, Rejection>,
    pending: FxHashMap<ResourceId, PendingRelease<B>>,
}

impl<B: Backend> Default for Resources<B> {
    fn default() -> Self {
        Self {
            rejections: FxHashMap::default(),
            pending: FxHashMap::default(),
        }
    }
}

impl<B: Backend> Resources<B> {
    /// Records the outcome of registering `resource`.
    fn register(&mut self, resource: ResourceId, result: Result<(), ResourceError>) {
        debug_assert!(
            !self.pending.contains_key(&resource),
            "{resource} registered while its release is pending: ids are never reused"
        );
        if let Err(reason) = result {
            tracing::debug!(%resource, %reason, "backend rejected a registration");
            self.rejections.insert(
                resource,
                Rejection {
                    reason: Arc::new(reason),
                    held: false,
                },
            );
        }
    }

    /// Releases `resource`, whose last handle dropped: frees it now when no
    /// surface's installed content draws it, and otherwise records the
    /// release as pending.
    fn release(
        &mut self,
        renderer: &mut B::Renderer,
        surfaces: &FxHashMap<SurfaceId, SurfaceState>,
        resource: ResourceId,
        remove: ResOp<B>,
    ) {
        let drawing: FxHashSet<SurfaceId> = surfaces
            .iter()
            .filter(|(surface, state)| draws(renderer, **surface, state, resource))
            .map(|(surface, _)| *surface)
            .collect();
        if drawing.is_empty() {
            free::<B>(&mut self.rejections, renderer, resource, remove);
        } else {
            tracing::debug!(%resource, surfaces = drawing.len(), "release waits for installed content");
            self.pending.insert(
                resource,
                PendingRelease {
                    remove,
                    surfaces: drawing,
                },
            );
        }
    }

    /// Updates the pending releases after commits: a surface that changed
    /// is drawing a pending resource exactly when its installed content now
    /// names it. Frees every resource no surface draws any more.
    fn settle(
        &mut self,
        renderer: &mut B::Renderer,
        surfaces: &FxHashMap<SurfaceId, SurfaceState>,
    ) {
        if self.pending.is_empty() {
            return;
        }
        for (resource, pending) in &mut self.pending {
            for (surface, state) in surfaces.iter().filter(|(_, state)| state.changed) {
                if draws(renderer, *surface, state, *resource) {
                    pending.surfaces.insert(*surface);
                } else {
                    pending.surfaces.remove(surface);
                }
            }
        }
        self.free_settled(renderer);
    }

    /// Surface `id` is destroyed: it draws nothing any more.
    fn surface_destroyed(&mut self, renderer: &mut B::Renderer, id: SurfaceId) {
        if self.pending.is_empty() {
            return;
        }
        for pending in self.pending.values_mut() {
            pending.surfaces.remove(&id);
        }
        self.free_settled(renderer);
    }

    /// Carries out every pending release that no surface draws any more.
    fn free_settled(&mut self, renderer: &mut B::Renderer) {
        let Self {
            rejections,
            pending,
        } = self;
        for (resource, release) in pending.extract_if(|_, release| release.surfaces.is_empty()) {
            tracing::debug!(%resource, "pending release carried out");
            free::<B>(rejections, renderer, resource, release.remove);
        }
    }

    /// Fails when a surface that changed since the last render draws a
    /// rejected resource. An unchanged surface cannot: a rejected
    /// registration's id reaches content only in a commit, and a rejected
    /// replacement marks every surface sampling the image changed.
    fn check(
        &self,
        renderer: &B::Renderer,
        surfaces: &FxHashMap<SurfaceId, SurfaceState>,
    ) -> Result<(), RenderError> {
        if self.rejections.is_empty() {
            return Ok(());
        }
        for (surface, state) in surfaces.iter().filter(|(_, state)| state.changed) {
            for (resource, rejection) in &self.rejections {
                if draws(renderer, *surface, state, *resource) {
                    return Err(RenderError::Rejected {
                        resource: *resource,
                        reason: Arc::clone(&rejection.reason),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Frees `resource`: clears its rejection and runs the backend's removal
/// unless the backend never committed the resource.
fn free<B: Backend>(
    rejections: &mut FxHashMap<ResourceId, Rejection>,
    renderer: &mut B::Renderer,
    resource: ResourceId,
    remove: ResOp<B>,
) {
    if rejections
        .remove(&resource)
        .is_none_or(|rejection| rejection.held)
    {
        remove(renderer);
    }
}

/// Whether surface `surface`'s installed content draws `resource`. The
/// backend answers for content; backdrop shaders are sampled through the
/// layer tree.
fn draws<R: Renderer>(
    renderer: &R,
    surface: SurfaceId,
    state: &SurfaceState,
    resource: ResourceId,
) -> bool {
    match resource {
        ResourceId::BackdropShader(id) => samples_backdrop_shader(&state.tree, id),
        resource => renderer.samples(surface, resource),
    }
}

/// Whether a layer of `tree` samples its backdrop through backdrop shader
/// `id`.
fn samples_backdrop_shader(tree: &SurfaceTree, id: BackdropShaderId) -> bool {
    tree.layers().any(|(_, node)| {
        matches!(
            node.backdrop.as_ref().and_then(crate::BackdropSample::effect),
            Some(BackdropEffect::Shader(effect)) if effect.shader == id
        )
    })
}

/// The render loop: runs on the `"cherenkov-render"` thread until
/// [`Message::Shutdown`] or channel disconnect.
#[cfg(not(target_arch = "wasm32"))]
pub fn run<B: Backend>(
    config: B::Config,
    rx: &Receiver<Message<B>>,
    init_reply: &Sender<Result<B::Info, EngineError>>,
) {
    let (mut renderer, info) = match B::init(config) {
        Ok(pair) => pair,
        Err(error) => {
            let _ = init_reply.send(Err(error));
            return;
        }
    };
    let _ = init_reply.send(Ok(info));
    let mut surfaces: FxHashMap<SurfaceId, SurfaceState> = FxHashMap::default();
    let mut resources = Resources::<B>::default();
    let mut next_frame = 0u64;
    while let Ok(message) = rx.recv() {
        match message {
            Message::CreateSurface { id, target, reply } => {
                let _ = reply.send(create_surface::<B>(
                    &mut renderer,
                    &mut surfaces,
                    id,
                    target,
                ));
            }
            Message::ResizeSurface { id, size } => {
                resize_surface::<B>(&mut renderer, &mut surfaces, id, size);
            }
            Message::DestroySurface { id } => {
                destroy_surface::<B>(&mut renderer, &mut surfaces, &mut resources, id);
            }
            Message::Display { id, display } => set_display(&mut surfaces, id, display),
            Message::DisplayMoved { id } => set_display_moved(&mut surfaces, id),
            Message::Resource(op) => op(&mut renderer),
            Message::Register { resource, op } => {
                resources.register(resource, op(&mut renderer));
            }
            Message::Release { resource, op } => {
                resources.release(&mut renderer, &surfaces, resource, op);
            }
            Message::ReplaceImage { id, image } => {
                replace_image::<B>(&mut renderer, &mut surfaces, &mut resources, id, image);
            }
            Message::Render {
                time,
                mut commits,
                reply,
            } => {
                let id = FrameId(next_frame);
                next_frame += 1;
                let result = render::<B>(
                    &mut renderer,
                    &mut surfaces,
                    &mut resources,
                    id,
                    time.0,
                    &mut commits,
                );
                let sender = reply.clone();
                let _ = sender.send(crate::message::RenderReply {
                    result,
                    commits,
                    sender: reply,
                });
            }
            Message::FinishTimings { reply } => {
                let _ = reply.send(renderer.finish_timings());
            }
            Message::Readback { surface, reply } => {
                let _ = reply.send(renderer.readback(surface));
            }
            Message::Memory { reply } => {
                let sender = reply.clone();
                let _ = sender.send(crate::message::MemoryReply {
                    usage: renderer.memory(),
                    sender: reply,
                });
            }
            Message::Trim(pressure) => renderer.trim(pressure),
            Message::Shutdown => break,
        }
    }
}

/// Creates surface `id`'s render-side state and its layer tree.
fn create_surface<B: Backend>(
    renderer: &mut B::Renderer,
    surfaces: &mut FxHashMap<SurfaceId, SurfaceState>,
    id: SurfaceId,
    target: B::Target,
) -> Result<SurfaceInfo, SurfaceError> {
    let info = renderer.create_surface(id, target)?;
    surfaces.insert(
        id,
        SurfaceState {
            tree: SurfaceTree::new(),
            size: info.size,
            display: Display::default(),
            clear: WorkingColor::TRANSPARENT,
            changed: true,
            presentation: if info.presents {
                Presentation::Presenting { pending: false }
            } else {
                Presentation::Retained
            },
            display_moved: false,
            content_animating: false,
        },
    );
    Ok(info)
}

fn resize_surface<B: Backend>(
    renderer: &mut B::Renderer,
    surfaces: &mut FxHashMap<SurfaceId, SurfaceState>,
    id: SurfaceId,
    size: (u32, u32),
) {
    renderer.resize_surface(id, size);
    if let Some(state) = surfaces.get_mut(&id) {
        state.size = size;
        state.changed = true;
    } else {
        tracing::trace!(surface = id.raw(), "resize of unknown surface");
    }
}

/// Destroys surface `id`, then carries out the pending releases only its
/// content still drew.
fn destroy_surface<B: Backend>(
    renderer: &mut B::Renderer,
    surfaces: &mut FxHashMap<SurfaceId, SurfaceState>,
    resources: &mut Resources<B>,
    id: SurfaceId,
) {
    if surfaces.remove(&id).is_some() {
        renderer.destroy_surface(id);
        resources.surface_destroyed(renderer, id);
    } else {
        // Nothing was committed — a dropped `Engine::surface` future whose
        // create failed may still send this (#150).
        tracing::trace!(surface = id.raw(), "destroy of unknown surface");
    }
}

fn set_display(surfaces: &mut FxHashMap<SurfaceId, SurfaceState>, id: SurfaceId, display: Display) {
    if let Some(state) = surfaces.get_mut(&id) {
        if state.display != display {
            // A scale change reshapes the content; a headroom-only update
            // re-presents without touching it (#98). Only a presenting
            // surface can be pending a present.
            state.changed |= state.display.scale.to_bits() != display.scale.to_bits();
            state.mark_present();
            state.display = display;
        }
    } else {
        tracing::trace!(surface = id.raw(), "display of unknown surface");
    }
}

fn set_display_moved(surfaces: &mut FxHashMap<SurfaceId, SurfaceState>, id: SurfaceId) {
    if let Some(state) = surfaces.get_mut(&id) {
        // A move re-enumerates output negotiation, where a headroom-only
        // `Display` update never does — and presents, since a
        // reconfigured swapchain must be shown (#98).
        state.display_moved = true;
        state.mark_present();
    } else {
        tracing::trace!(surface = id.raw(), "display move of unknown surface");
    }
}

/// Replaces image `id`'s pixels and marks changed only the surfaces whose
/// content samples the image; the next render redraws those with the new
/// pixels and leaves every other surface's skip intact. An image whose
/// registration was rejected is registered with the new pixels instead. A
/// rejection is recorded, and the marked surfaces then fail their render.
fn replace_image<B: Backend>(
    renderer: &mut B::Renderer,
    surfaces: &mut FxHashMap<SurfaceId, SurfaceState>,
    resources: &mut Resources<B>,
    id: ImageId,
    image: ImageUpload,
) {
    let resource = ResourceId::Image(id);
    debug_assert!(
        !resources.pending.contains_key(&resource),
        "{resource} replaced after its release: a replacement needs a live handle"
    );
    let rejections = &mut resources.rejections;
    let held = rejections
        .get(&resource)
        .is_none_or(|rejection| rejection.held);
    let result = if held {
        renderer.replace_image(id, image)
    } else {
        renderer.add_image(id, image)
    };
    match result {
        Ok(()) => {
            rejections.remove(&resource);
        }
        Err(reason) => {
            tracing::debug!(%resource, %reason, "backend rejected an image replacement");
            rejections.insert(
                resource,
                Rejection {
                    reason: Arc::new(reason),
                    held,
                },
            );
        }
    }
    for (surface, state) in surfaces {
        if renderer.samples(*surface, resource) {
            state.changed = true;
        }
    }
}

/// Applies one surface's committed change set into its tree, forwarding
/// content ops and install closures to the renderer in order.
fn commit<B: Backend>(
    renderer: &mut B::Renderer,
    state: &mut SurfaceState,
    surface: SurfaceId,
    changes: &mut ChangeSet<B>,
) {
    let ChangeSet {
        clear,
        ops,
        recycled,
        animating,
    } = changes;
    if let Some(clear) = clear.take() {
        state.clear = clear;
        state.changed = true;
    }
    state.content_animating = *animating;
    for op in ops.drain(..) {
        match op {
            Op::Layer(LayerOp::Remove(layer)) => {
                for removed in state.tree.remove(layer) {
                    renderer.remove_layer(surface, removed);
                }
                state.changed = true;
            }
            Op::Layer(LayerOp::Content(layer, content)) => {
                state.tree.apply(LayerOp::Content(layer, None));
                state.tree.note_content(layer, content.as_ref());
                if let Some(mut old) = renderer.set_content(surface, layer, content)
                    && old.clear_unique()
                {
                    recycled.push((layer, old));
                }
                state.changed = true;
            }
            Op::Layer(op) => {
                state.tree.apply(op);
                state.changed = true;
            }
            Op::Install(install) => {
                install(&mut *renderer);
                state.changed = true;
            }
        }
    }
}

/// Applies every surface's commit, then carries out the pending releases
/// no installed content draws any more, before the frame can draw them.
///
/// # Errors
/// [`RenderError::Rejected`] when a changed surface draws a resource the
/// backend rejected.
fn apply_commits<B: Backend>(
    renderer: &mut B::Renderer,
    surfaces: &mut FxHashMap<SurfaceId, SurfaceState>,
    resources: &mut Resources<B>,
    commits: &mut [(SurfaceId, ChangeSet<B>)],
) -> Result<(), RenderError> {
    for (surface, changes) in commits.iter_mut() {
        if let Some(state) = surfaces.get_mut(surface) {
            commit(renderer, state, *surface, changes);
        } else {
            // A dropped surface may still have queued ops: legal, ignore.
            tracing::trace!(surface = surface.raw(), "commit for unknown surface");
            changes.clear = None;
            changes.ops.clear();
        }
    }
    resources.settle(renderer, surfaces);
    resources.check(renderer, surfaces)
}

/// One frame: apply every commit, sample, render, answer.
#[cfg(not(target_arch = "wasm32"))]
fn render<B: Backend>(
    renderer: &mut B::Renderer,
    surfaces: &mut FxHashMap<SurfaceId, SurfaceState>,
    resources: &mut Resources<B>,
    id: FrameId,
    time: crate::Instant,
    commits: &mut [(SurfaceId, ChangeSet<B>)],
) -> Result<(Next, FrameStats), RenderError> {
    apply_commits(renderer, surfaces, resources, commits)?;
    let mut frames: Vec<SurfaceFrame<'_>> = Vec::with_capacity(surfaces.len());
    // The fast class wins when any surface needs it.
    let mut rate = None;
    for (id, state) in &mut *surfaces {
        let sampling = state.tree.sample(time, state.display);
        let changed = state.changed || sampling.stepped;
        // Operand animations run on the UI thread and are springs or
        // curves only: they always need the fast class.
        let running = if state.content_animating {
            Some(crate::tree::RATE_FAST)
        } else {
            sampling.rate
        };
        match running {
            Some(r) if r == crate::tree::RATE_FAST => rate = Some(crate::tree::RATE_FAST),
            Some(r) => rate = rate.or(Some(r)),
            None => {}
        }
        frames.push(SurfaceFrame {
            id: *id,
            size: state.size,
            display: state.display,
            clear: state.clear,
            changed,
            present_pending: state.present_pending(),
            display_moved: state.display_moved,
            tree: &state.tree,
        });
    }
    let mut stats = FrameStats::default();
    let redraw = renderer.render(
        &Frame {
            id,
            time: crate::frame::FrameTime(time),
            surfaces: &frames,
        },
        &mut stats,
    )?;
    for state in surfaces.values_mut() {
        state.changed = false;
        state.display_moved = false;
        state.presented();
    }
    if let Redraw::Wanted { rate: backend_rate } = redraw {
        rate = Some(rate.map_or_else(
            || backend_rate.clone(),
            |r| (*r.start()).min(*backend_rate.start())..=(*r.end()).max(*backend_rate.end()),
        ));
    }
    let next = rate.map_or(Next::Idle, |rate| Next::At {
        time: time + Duration::from_secs_f64(1.0 / f64::from(*rate.end())),
        rate,
    });
    Ok((next, stats))
}
#[cfg(target_arch = "wasm32")]
#[expect(
    clippy::future_not_send,
    reason = "the browser engine is single-threaded and its futures run on the page's event loop"
)]
async fn render_local<B: Backend>(
    renderer: &mut B::Renderer,
    surfaces: &mut FxHashMap<SurfaceId, SurfaceState>,
    resources: &mut Resources<B>,
    id: FrameId,
    time: crate::Instant,
    commits: &mut [(SurfaceId, ChangeSet<B>)],
) -> Result<(Next, FrameStats), RenderError> {
    apply_commits(renderer, surfaces, resources, commits)?;
    let mut frames: Vec<SurfaceFrame<'_>> = Vec::with_capacity(surfaces.len());
    // The fast class wins when any surface needs it.
    let mut rate = None;
    for (id, state) in &mut *surfaces {
        let sampling = state.tree.sample(time, state.display);
        let changed = state.changed || sampling.stepped;
        // Operand animations run on the UI thread and are springs or
        // curves only: they always need the fast class.
        let running = if state.content_animating {
            Some(crate::tree::RATE_FAST)
        } else {
            sampling.rate
        };
        match running {
            Some(r) if r == crate::tree::RATE_FAST => rate = Some(crate::tree::RATE_FAST),
            Some(r) => rate = rate.or(Some(r)),
            None => {}
        }
        frames.push(SurfaceFrame {
            id: *id,
            size: state.size,
            display: state.display,
            clear: state.clear,
            changed,
            present_pending: state.present_pending(),
            display_moved: state.display_moved,
            tree: &state.tree,
        });
    }
    let mut stats = FrameStats::default();
    let redraw = renderer
        .render(
            &Frame {
                id,
                time: crate::frame::FrameTime(time),
                surfaces: &frames,
            },
            &mut stats,
        )
        .await?;
    for state in surfaces.values_mut() {
        state.changed = false;
        state.display_moved = false;
        state.presented();
    }
    if let Redraw::Wanted { rate: backend_rate } = redraw {
        rate = Some(rate.map_or_else(
            || backend_rate.clone(),
            |r| (*r.start()).min(*backend_rate.start())..=(*r.end()).max(*backend_rate.end()),
        ));
    }
    let next = rate.map_or(Next::Idle, |rate| Next::At {
        time: time + Duration::from_secs_f64(1.0 / f64::from(*rate.end())),
        rate,
    });
    Ok((next, stats))
}

#[cfg(all(test, feature = "testing", not(target_arch = "wasm32")))]
mod tests {
    use super::{Presentation, SurfaceState, commit};
    use crate::WorkingColor;
    use crate::backend::{Backend, Display};
    use crate::display_list::{Command, DisplayList, Picture};
    use crate::message::{ChangeSet, ContentOp, LayerId, LayerOp, Op, SurfaceId};
    use crate::testing::{Null, NullConfig};
    use crate::tree::SurfaceTree;

    #[test]
    fn caller_shared_picture_is_not_recycled() {
        let (events, _receiver) = std::sync::mpsc::channel();
        let (mut renderer, ()) = <Null as Backend>::init(NullConfig {
            events,
            reject: std::collections::HashSet::new(),
        })
        .expect("null backend");
        let surface = SurfaceId::new(1);
        let layer = LayerId::new(0);
        let mut list = DisplayList::with_capacity(1);
        list.push(Command::End);
        let caller_picture = Picture::new(list);
        let mut state = SurfaceState {
            tree: SurfaceTree::new(),
            size: (1, 1),
            display: Display::default(),
            clear: WorkingColor::TRANSPARENT,
            changed: false,
            presentation: Presentation::Retained,
            display_moved: false,
            content_animating: false,
        };

        let mut first = ChangeSet::<Null> {
            clear: None,
            ops: vec![Op::Layer(LayerOp::Content(
                layer,
                Some(ContentOp::Picture(caller_picture.clone())),
            ))],
            recycled: Vec::new(),
            animating: false,
        };
        commit(&mut renderer, &mut state, surface, &mut first);
        let mut second = ChangeSet::<Null> {
            clear: None,
            ops: vec![Op::Layer(LayerOp::Content(
                layer,
                Some(ContentOp::Picture(Picture::new(DisplayList::default()))),
            ))],
            recycled: Vec::new(),
            animating: false,
        };

        commit(&mut renderer, &mut state, surface, &mut second);

        assert_eq!(second.recycled, []);
        assert_eq!(caller_picture.display_list().len(), 1);
    }
}

#[cfg(target_arch = "wasm32")]
pub(super) async fn local<B: Backend>(
    config: B::Config,
) -> Result<(crate::local::Sender<Message<B>>, B::Info), EngineError> {
    use std::cell::RefCell;
    use std::rc::Rc;
    let (renderer, info) = B::init(config).await?;
    let state = Rc::new(RefCell::new(Some(LocalState::<B> {
        renderer,
        surfaces: FxHashMap::default(),
        resources: Resources::default(),
        next_frame: 0,
    })));
    let tx = crate::local::Sender::new(move |message| {
        let mut owned = state.borrow_mut().take().expect("serial local executor");
        let state = Rc::clone(&state);
        Box::pin(async move {
            let live = owned.apply(message).await;
            *state.borrow_mut() = Some(owned);
            live
        })
    });
    Ok((tx, info))
}

#[cfg(target_arch = "wasm32")]
struct LocalState<B: Backend> {
    renderer: B::Renderer,
    surfaces: FxHashMap<SurfaceId, SurfaceState>,
    resources: Resources<B>,
    next_frame: u64,
}
#[cfg(target_arch = "wasm32")]
impl<B: Backend> LocalState<B> {
    #[expect(
        clippy::future_not_send,
        reason = "the browser engine is single-threaded and its futures run on the page's event loop"
    )]
    async fn apply(&mut self, message: Message<B>) -> bool {
        let Self {
            renderer,
            surfaces,
            resources,
            next_frame,
        } = self;
        match message {
            Message::CreateSurface { id, target, reply } => {
                let _ = reply.send(create_surface::<B>(renderer, surfaces, id, target));
            }
            Message::ResizeSurface { id, size } => {
                resize_surface::<B>(renderer, surfaces, id, size);
            }
            Message::DestroySurface { id } => {
                destroy_surface::<B>(renderer, surfaces, resources, id);
            }
            Message::Display { id, display } => set_display(surfaces, id, display),
            Message::DisplayMoved { id } => set_display_moved(surfaces, id),
            Message::Resource(op) => op(renderer),
            Message::Register { resource, op } => {
                let result = op(renderer).await;
                resources.register(resource, result);
            }
            Message::Release { resource, op } => {
                resources.release(renderer, surfaces, resource, op);
            }
            Message::ReplaceImage { id, image } => {
                replace_image::<B>(renderer, surfaces, resources, id, image);
            }
            Message::Render {
                time,
                mut commits,
                reply,
            } => {
                let id = FrameId(*next_frame);
                *next_frame += 1;
                let result =
                    render_local::<B>(renderer, surfaces, resources, id, time.0, &mut commits)
                        .await;
                let _ = reply.send(crate::message::RenderReply { result, commits });
            }
            Message::FinishTimings { reply } => {
                let _ = reply.send(renderer.finish_timings().await);
            }
            Message::Readback { surface, reply } => {
                let _ = reply.send(renderer.readback(surface).await);
            }
            Message::Memory { reply } => {
                let _ = reply.send(crate::message::MemoryReply {
                    usage: renderer.memory(),
                });
            }
            Message::Trim(pressure) => renderer.trim(pressure),
            Message::Shutdown => return false,
        }
        true
    }
}
