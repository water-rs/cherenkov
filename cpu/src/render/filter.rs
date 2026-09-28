use std::time::Duration;
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender, channel},
    },
};

use cherenkov::{FilterId, FrameId, FrameTime, RenderError};
use filtrate_core::{
    AnimatedTarget, AnimationTrack, CpuFilter, CpuFilterError, CpuImage, ParamArray, SignalVisitor,
    WatchGuard, WorkingSpace,
};

use crate::RedrawCallback;

pub(super) trait Erased: Send + Sync {
    fn footprint(&self, params: &[f32], size: (usize, usize)) -> f32;
    fn apply(
        &self,
        params: &[f32],
        space: &WorkingSpace,
        image: &mut CpuImage<'_>,
    ) -> Result<(), CpuFilterError>;
}

pub(super) type Prepared = (Arc<dyn Erased + Send + Sync>, Arc<[f32]>, f32);

impl<F> Erased for F
where
    F: CpuFilter + Send + Sync,
{
    #[expect(
        clippy::cast_precision_loss,
        reason = "surface dimensions are bounded to 16384 pixels"
    )]
    fn footprint(&self, params: &[f32], size: (usize, usize)) -> f32 {
        let params = F::Params::read_from(params);
        F::cpu_footprint(&params).resolve((size.0 as f32, size.1 as f32))
    }

    fn apply(
        &self,
        params: &[f32],
        space: &WorkingSpace,
        image: &mut CpuImage<'_>,
    ) -> Result<(), CpuFilterError> {
        self.apply_cpu_image(&F::Params::read_from(params), space, image)
    }
}

struct Entry {
    filter: Arc<dyn Erased + Send + Sync>,
    tracks: Vec<AnimationTrack>,
    events: Receiver<(usize, AnimatedTarget)>,
    pending_events: VecDeque<(usize, AnimatedTarget)>,
    dirty: Arc<AtomicBool>,
    active: Arc<AtomicBool>,
    _guards: Vec<WatchGuard>,
    sequence: Option<FrameId>,
    params: Arc<[f32]>,
}

impl Entry {
    fn wants_redraw(&self) -> bool {
        self.dirty.load(Ordering::Acquire) || self.tracks.iter().any(AnimationTrack::is_active)
    }

    fn prepare(&mut self, sequence: FrameId, delta: Duration) {
        if self.sequence == Some(sequence) {
            return;
        }
        let events: Vec<_> = std::mem::take(&mut self.pending_events)
            .into_iter()
            .chain(self.events.try_iter())
            .collect();
        self.dirty.store(false, Ordering::Release);
        let mut changed = !events.is_empty();
        for (index, target) in events {
            self.tracks[index].set_target(target.value, target.interpolator);
        }
        for track in &mut self.tracks {
            changed |= track.is_active();
            track.advance(delta);
        }
        if changed {
            self.dirty.store(true, Ordering::Release);
        }
        self.params = self.tracks.iter().map(AnimationTrack::value).collect();
        self.sequence = Some(sequence);
    }
}

struct WatcherInstaller<'a> {
    events: Sender<(usize, AnimatedTarget)>,
    dirty: Arc<AtomicBool>,
    active: Arc<AtomicBool>,
    redraw: Option<RedrawCallback>,
    guards: &'a mut Vec<WatchGuard>,
}

impl SignalVisitor for WatcherInstaller<'_> {
    fn visit<P: filtrate_core::FilterParam + ?Sized>(&mut self, index: usize, param: &P) {
        let events = self.events.clone();
        let dirty = Arc::clone(&self.dirty);
        let active = Arc::clone(&self.active);
        let redraw = self.redraw.clone();
        self.guards
            .push(param.watch_animated(Box::new(move |target| {
                if events.send((index, target)).is_err() {
                    return;
                }
                dirty.store(true, Ordering::Release);
                if active.load(Ordering::Acquire)
                    && let Some(redraw) = &redraw
                {
                    redraw.wake();
                }
            })));
    }
}

#[derive(Default)]
pub struct Registry {
    entries: std::collections::HashMap<u64, Entry>,
    redraw: Option<RedrawCallback>,
    last_frame: Option<(FrameId, cherenkov::Instant)>,
    delta: Duration,
}

impl Registry {
    pub(super) fn new(redraw: Option<RedrawCallback>) -> Self {
        let mut registry = Self::default();
        registry.redraw = redraw;
        registry
    }

    pub fn add<F>(&mut self, id: FilterId, filter: F)
    where
        F: CpuFilter + cherenkov::RenderTransfer + Send + Sync,
    {
        let mut initial = vec![0.0; F::Params::LEN];
        filter.params().write_to(&mut initial);
        let initial: Arc<[f32]> = initial.into();
        let (event_sender, events) = channel();
        let dirty = Arc::new(AtomicBool::new(false));
        let active = Arc::new(AtomicBool::new(false));
        let mut guards = Vec::with_capacity(F::Params::LEN);
        filter.visit_signals(&mut WatcherInstaller {
            events: event_sender,
            dirty: Arc::clone(&dirty),
            active: Arc::clone(&active),
            redraw: self.redraw.clone(),
            guards: &mut guards,
        });
        let filter: Arc<dyn Erased + Send + Sync> = Arc::new(filter);
        self.entries.insert(
            id.raw(),
            Entry {
                filter,
                tracks: initial.iter().copied().map(AnimationTrack::new).collect(),
                events,
                pending_events: VecDeque::new(),
                dirty,
                active,
                _guards: guards,
                sequence: None,
                params: initial,
            },
        );
    }

    pub fn remove(&mut self, id: FilterId) {
        if let Some(entry) = self.entries.remove(&id.raw()) {
            entry.active.store(false, Ordering::Release);
        }
    }

    pub(super) fn wants_redraw(&self, id: u64) -> bool {
        self.entries.get(&id).is_some_and(Entry::wants_redraw)
    }

    pub(super) fn begin_frame(&mut self, id: FrameId, time: FrameTime) -> Duration {
        if let Some((last_id, last_time)) = self.last_frame {
            if last_id == id {
                self.delta = Duration::ZERO;
            } else {
                self.delta = time.0.saturating_duration_since(last_time);
                self.last_frame = Some((id, time.0));
            }
        } else {
            self.last_frame = Some((id, time.0));
            self.delta = Duration::ZERO;
        }
        self.delta
    }

    pub(super) fn prepare(
        &mut self,
        id: FilterId,
        sequence: FrameId,
        size: (usize, usize),
    ) -> Result<Prepared, RenderError> {
        let entry = self
            .entries
            .get_mut(&id.raw())
            .ok_or_else(|| RenderError::Render(format!("unregistered filter {}", id.raw())))?;
        entry.prepare(sequence, self.delta);
        let bounds: Vec<f32> = entry
            .tracks
            .iter()
            .map(AnimationTrack::magnitude_bound)
            .collect();
        let footprint = entry.filter.footprint(&bounds, size);
        Ok((
            Arc::clone(&entry.filter),
            Arc::clone(&entry.params),
            footprint,
        ))
    }

    pub(super) fn set_active(&self, used: &std::collections::HashSet<u64>) {
        for (id, entry) in &self.entries {
            entry.active.store(used.contains(id), Ordering::Release);
        }
    }

    pub(super) fn finish_frame(&mut self, used: &std::collections::HashSet<u64>) {
        for id in used {
            let Some(entry) = self.entries.get_mut(id) else {
                continue;
            };
            if !entry.pending_events.is_empty() {
                entry.dirty.store(true, Ordering::Release);
                continue;
            }
            entry.dirty.store(false, Ordering::Release);
            if let Ok(event) = entry.events.try_recv() {
                entry.pending_events.push_back(event);
                entry.dirty.store(true, Ordering::Release);
            }
        }
    }
}

impl Drop for Registry {
    fn drop(&mut self) {
        for entry in self.entries.values() {
            entry.active.store(false, Ordering::Release);
        }
    }
}
