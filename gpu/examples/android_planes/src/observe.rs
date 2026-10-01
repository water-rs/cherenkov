//! Captures the plane pipeline's `cherenkov::planes` tracing events so
//! each heartbeat line can report the live promotion decision.
//!
//! The engine emits two events per candidate:
//! `external frame eligibility` (`layer`, `reason` — `None` when the
//! frame's buffer contract holds) at install time, and `plane decision`
//! (`layer`, `decision` — `promoted` or the [`Ineligible`] reason) on
//! every planned frame.

use std::collections::HashMap;
use std::fmt::Debug;
use std::io::Write;
use std::sync::{Arc, Mutex};

use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;

use crate::logcat;

/// What the engine has decided about each video layer, by raw `LayerId`.
#[derive(Default)]
pub struct Decisions {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    /// `layer` → `None` when the frame's contract holds, `Some(reason)`
    /// when the plane contract rejects it.
    eligibility: HashMap<u64, Option<String>>,
    /// `layer` → the plan's verdict: `promoted` or the rejection reason.
    verdicts: HashMap<u64, String>,
}

impl Decisions {
    /// The current decision string for `layer`'s heartbeat.
    #[must_use]
    pub fn decision(&self, layer: u64) -> String {
        let inner = self.inner.lock().expect("decisions lock");
        if let Some(verdict) = inner.verdicts.get(&layer) {
            return verdict.clone();
        }
        match inner.eligibility.get(&layer) {
            Some(Some(reason)) => reason.clone(),
            Some(None) => "pending".into(),
            None => "unseen".into(),
        }
    }
}

/// Installs the global subscriber — the capture layer plus a fmt layer
/// mirroring the engine's INFO+ messages into logcat — and returns the
/// shared decision record.
#[must_use]
pub fn install() -> Arc<Decisions> {
    let decisions = Arc::new(Decisions::default());
    let capture = Capture {
        decisions: Arc::clone(&decisions),
    };
    let engine_log = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(Logcat)
        .with_filter(filter_fn(|meta| {
            meta.target().starts_with("cherenkov") && *meta.level() <= Level::INFO
        }));
    tracing_subscriber::registry()
        .with(capture)
        .with(engine_log)
        .init();
    decisions
}

struct Capture {
    decisions: Arc<Decisions>,
}

impl<S: Subscriber> Layer<S> for Capture {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        if event.metadata().target() != "cherenkov::planes" {
            return;
        }
        let mut fields = Fields::default();
        event.record(&mut fields);
        let Some(layer) = fields.layer else {
            return;
        };
        let mut inner = self.decisions.inner.lock().expect("decisions lock");
        if let Some(reason) = fields.reason {
            inner.eligibility.insert(layer, parse_option(&reason));
        }
        if let Some(verdict) = fields.decision {
            inner.verdicts.insert(layer, verdict);
        }
    }
}

/// The `Option<Ineligible>` debug form (`None` / `Some(Reason)`).
fn parse_option(debug: &str) -> Option<String> {
    debug
        .strip_prefix("Some(")
        .and_then(|rest| rest.strip_suffix(')'))
        .map(ToOwned::to_owned)
}

/// `LayerId(N)` → `N`.
fn parse_layer_id(debug: &str) -> Option<u64> {
    debug
        .strip_prefix("LayerId(")
        .and_then(|rest| rest.strip_suffix(')'))
        .and_then(|raw| raw.parse().ok())
}

#[derive(Default)]
struct Fields {
    layer: Option<u64>,
    reason: Option<String>,
    decision: Option<String>,
}

impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
        let rendered = format!("{value:?}");
        match field.name() {
            "layer" => self.layer = parse_layer_id(&rendered),
            "reason" => self.reason = Some(rendered),
            "decision" => self.decision = Some(rendered),
            _ => {}
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "decision" {
            self.decision = Some(value.to_owned());
        }
    }
}

/// Writes fmt output into logcat under the `cherenkov` tag.
struct Logcat;

impl MakeWriter<'_> for Logcat {
    type Writer = LogcatWriter;

    fn make_writer(&self) -> Self::Writer {
        LogcatWriter
    }
}

struct LogcatWriter;

impl Write for LogcatWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let text = String::from_utf8_lossy(buf);
        logcat::warn(text.trim_end());
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
