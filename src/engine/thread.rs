// Copyright 2026 the Cherenkov Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The render thread's loop: owns the backend renderer and one
//! [`SurfaceTree`] per surface, applies commits, samples animations at the
//! frame time, renders, and answers with [`Next`] and the [`FrameStats`].

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use crate::WorkingColor;
use crate::backend::{Backend, Display, Frame, Redraw, Renderer, SurfaceFrame};
use crate::error::{EngineError, RenderError};
use crate::frame::{FrameStats, Next};
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
                if surfaces.remove(&id).is_none() {
                    tracing::trace!(surface = id.raw(), "destroy of unknown surface");
                }
                renderer.destroy_surface(id);
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
                commits,
                reply,
            } => {
                let result = render::<B>(&mut renderer, &mut surfaces, time.0, commits);
                let _ = reply.send(result);
            }
            Message::Readback { surface, reply } => {
                let _ = reply.send(renderer.readback(surface));
            }
            Message::Memory { reply } => {
                let _ = reply.send(renderer.memory());
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
    changes: ChangeSet<B>,
) {
    if let Some(clear) = changes.clear {
        state.clear = clear;
        state.changed = true;
    }
    for op in changes.ops {
        match op {
            Op::Layer(LayerOp::Remove(layer)) => {
                for removed in state.tree.remove(layer) {
                    renderer.remove_layer(surface, removed);
                }
                state.changed = true;
            }
            Op::Layer(LayerOp::Content(layer, content)) => {
                state.tree.apply(LayerOp::Content(layer, None));
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
fn render<B: Backend>(
    renderer: &mut B::Renderer,
    surfaces: &mut HashMap<SurfaceId, SurfaceState>,
    time: std::time::Instant,
    commits: Vec<(SurfaceId, ChangeSet<B>)>,
) -> Result<(Next, FrameStats), RenderError> {
    for (surface, changes) in commits {
        match surfaces.get_mut(&surface) {
            Some(state) => commit(renderer, state, surface, changes),
            // A dropped surface may still have queued ops: legal, ignore.
            None => {
                tracing::trace!(surface = surface.raw(), "commit for unknown surface");
            }
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
            time: crate::frame::FrameTime(time),
            surfaces: &frames,
        },
        &mut stats,
    )?;
    for state in surfaces.values_mut() {
        state.changed = false;
    }
    let next = match (rate, redraw) {
        (Some(rate), _) => Next::At {
            time: time + Duration::from_secs_f64(1.0 / f64::from(*rate.end())),
            rate,
        },
        (None, Redraw::Wanted) => Next::At {
            time: time + Duration::from_secs_f64(1.0 / 120.0),
            rate: crate::tree::RATE_FAST,
        },
        (None, Redraw::None) => Next::Idle,
    };
    Ok((next, stats))
}
