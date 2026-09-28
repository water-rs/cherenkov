// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The render thread's loop: owns the backend renderer and one
//! [`SurfaceTree`] per surface, applies commits, samples animations at the
//! frame time, renders, and answers with [`Next`] and the [`FrameStats`].

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use crate::WorkingColor;
use crate::backend::{Backend, Display, Frame, Redraw, Renderer, SurfaceFrame};
use crate::error::{EngineError, RenderError};
use crate::frame::{FrameId, FrameStats, Next};
use crate::message::{ChangeSet, LayerOp, Message, Op, SurfaceId};
use crate::tree::SurfaceTree;

/// One surface's render-thread state.
struct SurfaceState {
    tree: SurfaceTree,
    size: (u32, u32),
    display: Display,
    clear: WorkingColor,
    /// Whether a property op, a content op or an animation step touched the
    /// surface since the last render.
    changed: bool,
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
    let mut surfaces: HashMap<SurfaceId, SurfaceState> = HashMap::new();
    let mut next_frame = 0u64;
    while let Ok(message) = rx.recv() {
        match message {
            Message::CreateSurface { id, target, reply } => {
                let result = renderer.create_surface(id, target);
                if let Ok(info) = &result {
                    surfaces.insert(
                        id,
                        SurfaceState {
                            tree: SurfaceTree::new(),
                            size: info.size,
                            display: Display::default(),
                            clear: WorkingColor::TRANSPARENT,
                            changed: true,
                        },
                    );
                }
                let _ = reply.send(result);
            }
            Message::ResizeSurface { id, size } => {
                renderer.resize_surface(id, size);
                if let Some(state) = surfaces.get_mut(&id) {
                    state.size = size;
                    state.changed = true;
                } else {
                    tracing::trace!(surface = id.raw(), "resize of unknown surface");
                }
            }
            Message::DestroySurface { id } => {
                if surfaces.remove(&id).is_some() {
                    renderer.destroy_surface(id);
                } else {
                    // Nothing was committed — a dropped `Engine::surface`
                    // future whose create failed may still send this (#150).
                    tracing::trace!(surface = id.raw(), "destroy of unknown surface");
                }
            }
            Message::Display { id, display } => {
                if let Some(state) = surfaces.get_mut(&id) {
                    state.display = display;
                    state.changed = true;
                } else {
                    tracing::trace!(surface = id.raw(), "display of unknown surface");
                }
            }
            Message::Resource(op) => op(&mut renderer),
            Message::Render {
                time,
                mut commits,
                reply,
            } => {
                let id = FrameId(next_frame);
                next_frame += 1;
                let result = render::<B>(&mut renderer, &mut surfaces, id, time.0, &mut commits);
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

/// Applies one surface's committed change set into its tree, forwarding
/// content ops and install closures to the renderer in order.
fn commit<B: Backend>(
    renderer: &mut B::Renderer,
    state: &mut SurfaceState,
    surface: SurfaceId,
    changes: &mut ChangeSet<B>,
) {
    if let Some(clear) = changes.clear.take() {
        state.clear = clear;
        state.changed = true;
    }
    for op in changes.ops.drain(..) {
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
                renderer.set_content(surface, layer, content);
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

/// One frame: apply every commit, sample, render, answer.
#[cfg(not(target_arch = "wasm32"))]
fn render<B: Backend>(
    renderer: &mut B::Renderer,
    surfaces: &mut HashMap<SurfaceId, SurfaceState>,
    id: FrameId,
    time: crate::Instant,
    commits: &mut [(SurfaceId, ChangeSet<B>)],
) -> Result<(Next, FrameStats), RenderError> {
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
    let mut frames: Vec<SurfaceFrame<'_>> = Vec::with_capacity(surfaces.len());
    // The fast class wins when any surface needs it.
    let mut rate = None;
    for (id, state) in &mut *surfaces {
        let sampling = state.tree.sample(time, state.display);
        let changed = state.changed || sampling.stepped;
        match sampling.rate {
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
    surfaces: &mut HashMap<SurfaceId, SurfaceState>,
    id: FrameId,
    time: crate::Instant,
    commits: &mut [(SurfaceId, ChangeSet<B>)],
) -> Result<(Next, FrameStats), RenderError> {
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
    let mut frames: Vec<SurfaceFrame<'_>> = Vec::with_capacity(surfaces.len());
    // The fast class wins when any surface needs it.
    let mut rate = None;
    for (id, state) in &mut *surfaces {
        let sampling = state.tree.sample(time, state.display);
        let changed = state.changed || sampling.stepped;
        match sampling.rate {
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
pub(super) async fn local<B: Backend>(
    config: B::Config,
) -> Result<(crate::local::Sender<Message<B>>, B::Info), EngineError> {
    use std::cell::RefCell;
    use std::rc::Rc;
    let (renderer, info) = B::init(config).await?;
    let state = Rc::new(RefCell::new(Some(LocalState::<B> {
        renderer,
        surfaces: HashMap::new(),
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
    surfaces: HashMap<SurfaceId, SurfaceState>,
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
            next_frame,
        } = self;
        match message {
            Message::CreateSurface { id, target, reply } => {
                let result = renderer.create_surface(id, target);
                if let Ok(info) = &result {
                    surfaces.insert(
                        id,
                        SurfaceState {
                            tree: SurfaceTree::new(),
                            size: info.size,
                            display: Display::default(),
                            clear: WorkingColor::TRANSPARENT,
                            changed: true,
                        },
                    );
                }
                let _ = reply.send(result);
            }
            Message::ResizeSurface { id, size } => {
                renderer.resize_surface(id, size);
                if let Some(state) = surfaces.get_mut(&id) {
                    state.size = size;
                    state.changed = true;
                } else {
                    tracing::trace!(surface = id.raw(), "resize of unknown surface");
                }
            }
            Message::DestroySurface { id } => {
                if surfaces.remove(&id).is_some() {
                    renderer.destroy_surface(id);
                } else {
                    // Nothing was committed — a dropped `Engine::surface`
                    // future whose create failed may still send this (#150).
                    tracing::trace!(surface = id.raw(), "destroy of unknown surface");
                }
            }
            Message::Display { id, display } => {
                if let Some(state) = surfaces.get_mut(&id) {
                    state.display = display;
                    state.changed = true;
                } else {
                    tracing::trace!(surface = id.raw(), "display of unknown surface");
                }
            }
            Message::Resource(op) => op(renderer),
            Message::AsyncResource(op) => op(renderer).await,
            Message::Render {
                time,
                mut commits,
                reply,
            } => {
                let id = FrameId(*next_frame);
                *next_frame += 1;
                let result = render_local::<B>(renderer, surfaces, id, time.0, &mut commits).await;
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
