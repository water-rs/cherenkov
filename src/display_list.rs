//! The display list: the owned, serializable form of recorded content, and the
//! slot updates that patch it.

use std::ops::Range;
use std::sync::Arc;

use kurbo::{Affine, Rect, Stroke};
use serde::{Deserialize, Serialize};

use crate::glyph::GlyphRun;
use crate::paint::{ImageId, Paint, Sampling};
use crate::shape::ShapeData;
use crate::style::{Group, Shadow};

/// One recorded command.
///
/// Scopes are flat: a `Begin*` command records the index of its matching
/// [`Command::End`], so a consumer can skip or regenerate a whole scope.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Command {
    /// Fill a shape.
    Fill {
        /// The shape.
        shape: ShapeData,
        /// The paint.
        paint: Paint,
    },
    /// Stroke a shape.
    Stroke {
        /// The shape.
        shape: ShapeData,
        /// The stroke style.
        stroke: Stroke,
        /// The paint.
        paint: Paint,
    },
    /// Cast a shadow from a shape.
    Shadow {
        /// The shape.
        shape: ShapeData,
        /// The shadow.
        shadow: Shadow,
    },
    /// Draw a glyph run.
    Glyphs {
        /// The run.
        run: GlyphRun,
        /// The paint.
        paint: Paint,
    },
    /// Draw an image into a rectangle.
    Image {
        /// The image.
        image: ImageId,
        /// Destination rectangle.
        dst: Rect,
        /// Sampling.
        sampling: Sampling,
    },
    /// Draw a shared picture.
    Picture {
        /// The picture.
        picture: Picture,
        /// Where it is placed.
        transform: Affine,
    },
    /// Clip the scope to a shape.
    BeginClip {
        /// The clip shape.
        shape: ShapeData,
        /// Index of the matching `End`.
        end: u32,
    },
    /// Transform the scope.
    BeginTransform {
        /// The transform.
        transform: Affine,
        /// Index of the matching `End`.
        end: u32,
    },
    /// Isolate the scope as a group.
    BeginGroup {
        /// The group style.
        group: Group,
        /// Index of the matching `End`.
        end: u32,
    },
    /// Close the innermost scope.
    End,
}

/// Which value of a command a slot addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperandKind {
    /// The shape of a fill, stroke, shadow or clip.
    Shape,
    /// The paint of a fill, stroke or glyph run.
    Paint,
    /// The stroke style of a stroke.
    Stroke,
    /// The shadow of a shadow command.
    Shadow,
    /// The transform of a picture or a transform scope.
    Transform,
    /// The group style of a group scope.
    Group,
    /// The destination rectangle of an image.
    Rect,
}

/// A value that replaces one operand of a command.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Operand {
    /// A shape.
    Shape(ShapeData),
    /// A paint.
    Paint(Paint),
    /// A stroke style.
    Stroke(Stroke),
    /// A shadow.
    Shadow(Shadow),
    /// A transform.
    Transform(Affine),
    /// A group style.
    Group(Group),
    /// A rectangle.
    Rect(Rect),
}

impl Operand {
    /// Which operand this value replaces.
    #[must_use]
    pub const fn kind(&self) -> OperandKind {
        match self {
            Self::Shape(_) => OperandKind::Shape,
            Self::Paint(_) => OperandKind::Paint,
            Self::Stroke(_) => OperandKind::Stroke,
            Self::Shadow(_) => OperandKind::Shadow,
            Self::Transform(_) => OperandKind::Transform,
            Self::Group(_) => OperandKind::Group,
            Self::Rect(_) => OperandKind::Rect,
        }
    }
}

/// The address of a value in a display list: one operand of one command.
///
/// A value recorded from a signal owns a slot; when the signal changes, the
/// slot's new value is sent as a [`SlotUpdate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Slot {
    /// Index of the command.
    pub command: u32,
    /// Which operand of the command.
    pub operand: OperandKind,
}

/// A new value for a slot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SlotUpdate {
    /// Index of the command.
    pub command: u32,
    /// The new value. Its kind selects the operand.
    pub value: Operand,
}

impl SlotUpdate {
    /// The slot this update addresses.
    #[must_use]
    pub const fn slot(&self) -> Slot {
        Slot {
            command: self.command,
            operand: self.value.kind(),
        }
    }
}

/// Commands that must be regenerated after updates, as sorted, disjoint
/// ranges of command indices.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Dirty {
    ranges: Vec<Range<u32>>,
}

impl Dirty {
    /// The dirty ranges, sorted and disjoint.
    #[must_use]
    pub fn ranges(&self) -> &[Range<u32>] {
        &self.ranges
    }

    /// Whether nothing needs regenerating.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Whether a command needs regenerating.
    #[must_use]
    pub fn contains(&self, command: u32) -> bool {
        self.ranges.iter().any(|range| range.contains(&command))
    }

    fn from_unsorted(mut ranges: Vec<Range<u32>>) -> Self {
        ranges.sort_unstable_by_key(|range| range.start);
        let mut merged: Vec<Range<u32>> = Vec::with_capacity(ranges.len());
        for range in ranges {
            match merged.last_mut() {
                Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
                _ => merged.push(range),
            }
        }
        Self { ranges: merged }
    }
}

/// Recorded content: an ordered list of commands.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DisplayList {
    commands: Vec<Command>,
}

impl DisplayList {
    /// The commands.
    #[must_use]
    pub fn commands(&self) -> &[Command] {
        &self.commands
    }

    /// Number of commands.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.commands.len()
    }

    /// Whether the list has no commands.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Applies slot updates and returns the commands to regenerate. An update
    /// to a leaf command dirties that command; an update to a scope dirties the
    /// whole scope through its `End`.
    ///
    /// # Panics
    ///
    /// Panics when an update addresses a command that does not exist or an
    /// operand the command does not have. Updates come from the recorder that
    /// produced this list, so either is a bug.
    pub fn apply(&mut self, updates: impl IntoIterator<Item = SlotUpdate>) -> Dirty {
        let mut dirty = Vec::new();
        for SlotUpdate { command, value } in updates {
            let index = command as usize;
            let len = self.commands.len();
            let target = self
                .commands
                .get_mut(index)
                .unwrap_or_else(|| panic!("slot update for command {index} of {len}"));
            let end = match (target, value) {
                (
                    Command::Fill { shape, .. }
                    | Command::Stroke { shape, .. }
                    | Command::Shadow { shape, .. },
                    Operand::Shape(new),
                ) => {
                    *shape = new;
                    command
                }
                (
                    Command::Fill { paint, .. }
                    | Command::Stroke { paint, .. }
                    | Command::Glyphs { paint, .. },
                    Operand::Paint(new),
                ) => {
                    *paint = new;
                    command
                }
                (Command::Stroke { stroke, .. }, Operand::Stroke(new)) => {
                    *stroke = new;
                    command
                }
                (Command::Shadow { shadow, .. }, Operand::Shadow(new)) => {
                    *shadow = new;
                    command
                }
                (Command::Image { dst, .. }, Operand::Rect(new)) => {
                    *dst = new;
                    command
                }
                (Command::Picture { transform, .. }, Operand::Transform(new)) => {
                    *transform = new;
                    command
                }
                (Command::BeginClip { shape, end }, Operand::Shape(new)) => {
                    *shape = new;
                    *end
                }
                (Command::BeginTransform { transform, end }, Operand::Transform(new)) => {
                    *transform = new;
                    *end
                }
                (Command::BeginGroup { group, end }, Operand::Group(new)) => {
                    *group = new;
                    *end
                }
                (target, value) => panic!(
                    "slot update {:?} does not match command {index}: {target:?}",
                    value.kind()
                ),
            };
            dirty.push(command..end + 1);
        }
        Dirty::from_unsorted(dirty)
    }

    pub(crate) fn push(&mut self, command: Command) -> u32 {
        let index = u32::try_from(self.commands.len())
            .expect("a display list holds at most u32::MAX commands");
        self.commands.push(command);
        index
    }

    /// Closes the scope opened by the `Begin*` command at `begin`.
    pub(crate) fn end(&mut self, begin: u32) {
        let end = self.push(Command::End);
        match &mut self.commands[begin as usize] {
            Command::BeginClip { end: slot, .. }
            | Command::BeginTransform { end: slot, .. }
            | Command::BeginGroup { end: slot, .. } => *slot = end,
            other => unreachable!("command {begin} opens no scope: {other:?}"),
        }
    }
}

/// Immutable recorded content, shared by reference: cloning is cheap, and a
/// picture can be sent to and shared between threads.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Picture(Arc<DisplayList>);

impl Picture {
    pub(crate) fn new(list: DisplayList) -> Self {
        Self(Arc::new(list))
    }

    /// The recorded commands.
    #[must_use]
    pub fn display_list(&self) -> &DisplayList {
        &self.0
    }
}
