//! Snippets: parsed, ABI-checked WGSL functions.

use std::collections::HashSet;

use naga::{
    Block, Expression, Function, FunctionArgument, Handle, ImageClass, ImageDimension, Module,
    ScalarKind, Statement, Type, TypeInner, VectorSize,
    valid::{Capabilities, ValidationFlags, Validator},
};

use crate::{
    abi::{self, FoldBlocker, Param, ParamType, Precision, SamplerFilter, SnippetKind},
    errors::SnippetError,
    rewrite::map_expression,
};

/// Which variant of a snippet a source is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Variant {
    /// The precision of colour values.
    pub precision: Precision,
    /// Whether the source uses subgroup operations.
    pub subgroups: bool,
}

impl Variant {
    /// The required base variant: `f32`, no subgroup operations.
    pub const BASE: Self = Self {
        precision: Precision::F32,
        subgroups: false,
    };

    pub(crate) fn capabilities(self) -> Capabilities {
        let mut capabilities = Capabilities::empty();
        if self.precision == Precision::F16 {
            capabilities |= Capabilities::SHADER_FLOAT16;
        }
        if self.subgroups {
            capabilities |= Capabilities::SUBGROUP;
        }
        capabilities
    }

    fn label(self) -> String {
        let precision = precision_name(self.precision);
        if self.subgroups {
            format!("{precision}, subgroups")
        } else {
            precision.to_owned()
        }
    }
}

/// The WGSL sources of one snippet.
#[derive(Debug, Clone)]
pub struct SnippetSource<'a> {
    name: &'a str,
    kind: SnippetKind,
    base: &'a str,
    others: Vec<(Variant, &'a str)>,
}

impl<'a> SnippetSource<'a> {
    /// A snippet with its required `f32` source.
    #[must_use]
    pub const fn new(name: &'a str, kind: SnippetKind, f32_source: &'a str) -> Self {
        Self {
            name,
            kind,
            base: f32_source,
            others: Vec::new(),
        }
    }

    /// Declares another variant. Declaring [`Variant::BASE`] replaces the
    /// `f32` source; declaring any variant twice keeps the later source.
    #[must_use]
    pub fn variant(mut self, variant: Variant, source: &'a str) -> Self {
        if variant == Variant::BASE {
            self.base = source;
        } else {
            self.others.retain(|(existing, _)| *existing != variant);
            self.others.push((variant, source));
        }
        self
    }
}

/// How many times a spatial snippet samples its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleCount {
    /// A fixed number of samples per output pixel.
    Static(u32),
    /// The number depends on control flow (a loop) or on a helper function.
    Dynamic,
}

/// The ABI a snippet declares, identical across its variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Abi {
    pub kind: SnippetKind,
    pub params: Vec<Param>,
    pub working_space: bool,
    pub shape: bool,
    pub aux: u32,
    /// Spatial snippets: the filter mode `input_sampler` declares.
    pub sampler: Option<SamplerFilter>,
}

/// Where each optional ABI argument sits in the `apply` signature.
#[derive(Debug, Clone, Default)]
pub struct ArgSlots {
    pub params: Option<usize>,
    pub working_space: Option<usize>,
    pub shape: Option<usize>,
    pub aux: Vec<usize>,
}

/// One parsed variant.
#[derive(Debug, Clone)]
pub struct Parsed {
    pub module: Module,
    pub apply: Handle<Function>,
    pub slots: ArgSlots,
    /// Spatial snippets: how often `input` is sampled.
    pub samples: SampleCount,
    /// Spatial snippets: why folding a colour prefix into `apply`'s accesses
    /// of `input` is not equivalent; `None` when the stage is foldable.
    pub not_foldable: Option<FoldBlocker>,
    /// The number of computed expressions in `apply` and every helper it
    /// calls, transitively — a proxy for the ALU cost of one invocation.
    pub ops: u32,
}

/// A parsed, ABI-checked snippet with all its declared variants.
#[derive(Debug, Clone)]
pub struct Snippet {
    name: String,
    abi: Abi,
    base: Parsed,
    others: Vec<(Variant, Parsed)>,
}

impl Snippet {
    /// Parses and checks every declared variant of a snippet.
    ///
    /// # Errors
    ///
    /// Returns an error when a source does not parse or validate, breaks the
    /// ABI, uses a construct snippets may not use, or declares a different
    /// ABI than the `f32` source.
    pub fn parse(source: &SnippetSource<'_>) -> Result<Self, SnippetError> {
        let reader = Reader {
            name: source.name,
            kind: source.kind,
        };
        let (abi, base) = reader.variant(Variant::BASE, source.base)?;
        let mut others = Vec::with_capacity(source.others.len());
        for &(variant, wgsl) in &source.others {
            let (variant_abi, parsed) = reader.variant(variant, wgsl)?;
            if variant_abi != abi {
                return Err(SnippetError::VariantMismatch {
                    name: source.name.to_owned(),
                    variant: variant.label(),
                });
            }
            others.push((variant, parsed));
        }
        Ok(Self {
            name: source.name.to_owned(),
            abi,
            base,
            others,
        })
    }

    /// The snippet name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// What the snippet does to pixels.
    #[must_use]
    pub const fn kind(&self) -> SnippetKind {
        self.abi.kind
    }

    /// The declared parameters, in `Params` member order.
    #[must_use]
    pub fn params(&self) -> &[Param] {
        &self.abi.params
    }

    /// Whether the snippet takes the working-space constants.
    #[must_use]
    pub const fn needs_working_space(&self) -> bool {
        self.abi.working_space
    }

    /// Whether the snippet takes the clip shape as input.
    #[must_use]
    pub const fn needs_shape(&self) -> bool {
        self.abi.shape
    }

    /// The number of auxiliary images the snippet takes.
    #[must_use]
    pub const fn aux_images(&self) -> u32 {
        self.abi.aux
    }

    /// The filter mode a spatial snippet declares for `input_sampler`
    /// (`None` for colour snippets).
    #[must_use]
    pub const fn sampler_filter(&self) -> Option<SamplerFilter> {
        self.abi.sampler
    }

    /// The declared variants, the base first.
    pub fn variants(&self) -> impl Iterator<Item = Variant> + '_ {
        core::iter::once(Variant::BASE).chain(self.others.iter().map(|(variant, _)| *variant))
    }

    pub(crate) const fn abi(&self) -> &Abi {
        &self.abi
    }

    /// The best declared variant for a request: the exact variant when
    /// declared, otherwise without subgroups, otherwise at `f32`.
    pub(crate) fn select(&self, want: Variant) -> (Variant, &Parsed) {
        let candidates = [
            want,
            Variant {
                subgroups: false,
                ..want
            },
            Variant {
                precision: Precision::F32,
                ..want
            },
        ];
        candidates
            .iter()
            .find_map(|candidate| {
                self.others
                    .iter()
                    .find(|(variant, _)| variant == candidate)
                    .map(|(variant, parsed)| (*variant, parsed))
            })
            .unwrap_or((Variant::BASE, &self.base))
    }
}

/// Parses the variants of one snippet.
struct Reader<'a> {
    name: &'a str,
    kind: SnippetKind,
}

impl Reader<'_> {
    fn variant(&self, variant: Variant, wgsl: &str) -> Result<(Abi, Parsed), SnippetError> {
        let label = variant.label();
        let module = naga::front::wgsl::parse_str(wgsl).map_err(|error| SnippetError::Parse {
            name: self.name.to_owned(),
            variant: label.clone(),
            message: error.emit_to_string(wgsl),
        })?;
        Validator::new(ValidationFlags::all(), variant.capabilities())
            .validate(&module)
            .map_err(|error| SnippetError::Invalid {
                name: self.name.to_owned(),
                variant: label.clone(),
                message: error.emit_to_string(wgsl),
            })?;
        check_module(&module, variant).map_err(|construct| SnippetError::Unsupported {
            name: self.name.to_owned(),
            variant: label.clone(),
            construct,
        })?;

        let abi_error = |reason: String| SnippetError::Abi {
            name: self.name.to_owned(),
            variant: label.clone(),
            reason,
        };
        let apply = module
            .functions
            .iter()
            .find(|(_, function)| function.name.as_deref() == Some("apply"))
            .map(|(handle, _)| handle)
            .ok_or_else(|| abi_error("defines no function named `apply`".to_owned()))?;
        let function = &module.functions[apply];
        let (required, filter) = self
            .check_required(&module, function, variant.precision)
            .map_err(abi_error)?;
        let (slots, params) = self
            .read_optional(&module, &function.arguments[required..], required)
            .map_err(abi_error)?;

        let (samples, not_foldable) = match self.kind {
            SnippetKind::Spatial => count_samples(function, filter.expect("spatial declares one")),
            SnippetKind::Color => (SampleCount::Static(0), None),
        };
        let ops = total_ops(&module, apply);

        let abi = Abi {
            kind: self.kind,
            params,
            working_space: slots.working_space.is_some(),
            shape: slots.shape.is_some(),
            aux: u32::try_from(slots.aux.len()).unwrap_or(u32::MAX),
            sampler: filter,
        };
        Ok((
            abi,
            Parsed {
                module,
                apply,
                slots,
                samples,
                not_foldable,
                ops,
            },
        ))
    }
}

fn check_module(module: &Module, variant: Variant) -> Result<(), &'static str> {
    if !module.global_variables.is_empty() {
        return Err("global variables (inputs are function arguments)");
    }
    if !module.overrides.is_empty() {
        return Err("override declarations");
    }
    if !module.entry_points.is_empty() {
        return Err("entry points");
    }
    for (_, function) in module.functions.iter() {
        check_constructs(function, variant.subgroups)?;
    }
    Ok(())
}

impl Reader<'_> {
    /// Checks the required leading arguments and the result. Returns the
    /// number of required arguments and, for spatial snippets, the filter
    /// mode the sampler argument declares.
    ///
    /// The sampler argument declares its filter mode by name:
    /// `input_point_sampler` requires the executor to bind a nearest (point)
    /// sampler, `input_sampler` allows a filtering one.
    fn check_required(
        &self,
        module: &Module,
        function: &Function,
        precision: Precision,
    ) -> Result<(usize, Option<SamplerFilter>), String> {
        let colour = colour_type(precision);
        let uv = ParamType::Vec2.inner();
        let texture = texture_2d();
        // (argument index, name, type); the spatial sampler argument is
        // checked separately because its name carries the filter mode.
        let required: Vec<(usize, &str, &TypeInner)> = match self.kind {
            SnippetKind::Color => vec![(0, "color", &colour)],
            SnippetKind::Spatial => vec![(0, "input", &texture), (2, "uv", &uv)],
        };
        let count = required.last().map_or(0, |(index, ..)| index + 1);
        if function.arguments.len() < count {
            return Err(format!(
                "`apply` takes {} arguments; the {:?} ABI requires at least {}",
                function.arguments.len(),
                self.kind,
                count
            ));
        }
        for (index, name, expected) in &required {
            let argument = &function.arguments[*index];
            if argument.name.as_deref() != Some(*name)
                || module.types[argument.ty].inner != **expected
            {
                return Err(format!("argument {index} must be `{name}` of the ABI type"));
            }
        }
        let sampler = if self.kind == SnippetKind::Spatial {
            let argument = &function.arguments[1];
            let filter = match argument.name.as_deref() {
                Some("input_point_sampler") => SamplerFilter::Point,
                Some("input_sampler") => SamplerFilter::Filtered,
                other => {
                    return Err(format!(
                        "argument 1 must be `input_point_sampler` (nearest sampling) or `input_sampler` (filtering allowed), found `{}`",
                        other.unwrap_or("?")
                    ));
                }
            };
            if module.types[argument.ty].inner != (TypeInner::Sampler { comparison: false }) {
                return Err("argument 1 must be a `sampler`".to_owned());
            }
            Some(filter)
        } else {
            None
        };
        let result_ok = function
            .result
            .as_ref()
            .is_some_and(|result| module.types[result.ty].inner == colour);
        if !result_ok {
            return Err(format!(
                "`apply` must return vec4<{}>",
                precision_name(precision)
            ));
        }
        Ok((count, sampler))
    }

    /// Reads the optional arguments after the required ones.
    fn read_optional(
        &self,
        module: &Module,
        arguments: &[FunctionArgument],
        first: usize,
    ) -> Result<(ArgSlots, Vec<Param>), String> {
        let mut slots = ArgSlots::default();
        let mut params = Vec::new();
        for (offset, argument) in arguments.iter().enumerate() {
            let slot = first + offset;
            let inner = &module.types[argument.ty].inner;
            let spatial = self.kind == SnippetKind::Spatial;
            match argument.name.as_deref() {
                Some("params") if slots.params.is_none() => {
                    params = read_params(module, inner)?;
                    slots.params = Some(slot);
                }
                Some("space") if slots.working_space.is_none() => {
                    if !abi::is_working_space(module, argument.ty) {
                        return Err(format!(
                            "`space` must be a struct exactly equivalent to `{}` — same members, offsets and span; the name is free",
                            abi::WORKING_SPACE_WGSL
                        ));
                    }
                    slots.working_space = Some(slot);
                }
                Some("shape") if spatial && slots.shape.is_none() => {
                    if !is_texture_2d(inner) {
                        return Err("`shape` must be texture_2d<f32>".to_owned());
                    }
                    slots.shape = Some(slot);
                }
                Some(other) if spatial && other.starts_with("aux") => {
                    let expected = format!("aux{}", slots.aux.len());
                    if other != expected || !is_texture_2d(inner) {
                        return Err(format!(
                            "auxiliary images are `aux0`, `aux1`, … of type texture_2d<f32>, in order; found `{other}`, expected `{expected}`"
                        ));
                    }
                    slots.aux.push(slot);
                }
                other => {
                    return Err(format!("unexpected argument `{}`", other.unwrap_or("?")));
                }
            }
        }
        Ok((slots, params))
    }
}

fn read_params(module: &Module, inner: &TypeInner) -> Result<Vec<Param>, String> {
    let TypeInner::Struct { members, .. } = inner else {
        return Err("`params` must be a struct".to_owned());
    };
    members
        .iter()
        .map(|member| {
            let name = member
                .name
                .clone()
                .ok_or_else(|| "parameters must be named".to_owned())?;
            let ty = ParamType::from_inner(&module.types[member.ty].inner).ok_or_else(|| {
                format!("parameter `{name}` must be f32, vec2<f32>, vec3<f32> or vec4<f32>")
            })?;
            Ok(Param { name, ty })
        })
        .collect()
}

pub const fn colour_type(precision: Precision) -> TypeInner {
    TypeInner::Vector {
        size: VectorSize::Quad,
        scalar: precision.scalar(),
    }
}

pub fn colour_type_handle(module: &mut Module, precision: Precision) -> Handle<Type> {
    module.types.insert(
        Type {
            name: None,
            inner: colour_type(precision),
        },
        naga::Span::UNDEFINED,
    )
}

const fn precision_name(precision: Precision) -> &'static str {
    match precision {
        Precision::F32 => "f32",
        Precision::F16 => "f16",
    }
}

pub const fn texture_2d() -> TypeInner {
    TypeInner::Image {
        dim: ImageDimension::D2,
        arrayed: false,
        class: ImageClass::Sampled {
            kind: ScalarKind::Float,
            multi: false,
        },
    }
}

pub fn is_texture_2d(inner: &TypeInner) -> bool {
    *inner == texture_2d()
}

/// Rejects every construct outside the snippet contract.
fn check_constructs(function: &Function, subgroups: bool) -> Result<(), &'static str> {
    for (_, expression) in function.expressions.iter() {
        match *expression {
            Expression::AtomicResult { .. } => return Err("atomics"),
            Expression::WorkGroupUniformLoadResult { .. } => return Err("workgroup memory"),
            Expression::ArrayLength(_) => return Err("runtime-sized arrays"),
            Expression::RayQueryVertexPositions { .. }
            | Expression::RayQueryProceedResult
            | Expression::RayQueryGetIntersection { .. } => return Err("ray queries"),
            Expression::CooperativeLoad { .. } | Expression::CooperativeMultiplyAdd { .. } => {
                return Err("cooperative matrices");
            }
            Expression::SubgroupBallotResult | Expression::SubgroupOperationResult { .. }
                if !subgroups =>
            {
                return Err("subgroup operations outside a subgroup variant");
            }
            _ => {}
        }
    }
    check_block(&function.body, subgroups)
}

fn check_block(block: &Block, subgroups: bool) -> Result<(), &'static str> {
    for statement in block {
        match statement {
            Statement::Block(inner) => check_block(inner, subgroups)?,
            Statement::If { accept, reject, .. } => {
                check_block(accept, subgroups)?;
                check_block(reject, subgroups)?;
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    check_block(&case.body, subgroups)?;
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                check_block(body, subgroups)?;
                check_block(continuing, subgroups)?;
            }
            Statement::Kill => return Err("discard"),
            Statement::ControlBarrier(_) | Statement::MemoryBarrier(_) => return Err("barriers"),
            Statement::ImageStore { .. } => return Err("image stores"),
            Statement::Atomic { .. } | Statement::ImageAtomic { .. } => return Err("atomics"),
            Statement::WorkGroupUniformLoad { .. } => return Err("workgroup memory"),
            Statement::RayQuery { .. } | Statement::RayPipelineFunction(_) => {
                return Err("ray tracing");
            }
            Statement::CooperativeStore { .. } => return Err("cooperative matrices"),
            Statement::SubgroupBallot { .. }
            | Statement::SubgroupGather { .. }
            | Statement::SubgroupCollectiveOperation { .. }
                if !subgroups =>
            {
                return Err("subgroup operations outside a subgroup variant");
            }
            Statement::Emit(_)
            | Statement::Break
            | Statement::Continue
            | Statement::Return { .. }
            | Statement::Store { .. }
            | Statement::Call { .. }
            | Statement::SubgroupBallot { .. }
            | Statement::SubgroupGather { .. }
            | Statement::SubgroupCollectiveOperation { .. } => {}
        }
    }
    Ok(())
}

/// The expression that reads argument `index`, if the function reads it.
pub fn argument_expression(function: &Function, index: u32) -> Option<Handle<Expression>> {
    function
        .expressions
        .iter()
        .find(|(_, expression)| **expression == Expression::FunctionArgument(index))
        .map(|(handle, _)| handle)
}

/// Counts how often `apply` accesses argument 0 (`input`), and whether
/// folding a colour prefix into every access is equivalent.
///
/// Folding is equivalent only when every access returns an exact texel of
/// `input`: an `ImageLoad`, or an `ImageSample` that does not gather, does
/// not compare a depth reference, and goes through the declared point
/// sampler. A filtered sample computes `prefix(lerp(a, b))` where
/// materializing first computes `lerp(prefix(a), prefix(b))`; a gather would
/// apply the prefix to a gathered component vector; and an `ImageQuery` can
/// observe the materialized intermediate's extent, which may differ from
/// `input`'s (aprons).
fn count_samples(function: &Function, filter: SamplerFilter) -> (SampleCount, Option<FoldBlocker>) {
    let Some(input) = argument_expression(function, 0) else {
        return (SampleCount::Static(0), None);
    };
    let declared = argument_expression(function, 1);
    let blocker = function
        .expressions
        .iter()
        .find_map(|(_, expression)| access_blocker(expression, input, declared, filter));
    let mut counter = SampleCounter {
        function,
        input,
        count: 0,
        in_loop: false,
        dynamic: false,
        escapes: false,
    };
    counter.block(&function.body);
    let samples = if counter.dynamic {
        SampleCount::Dynamic
    } else {
        SampleCount::Static(counter.count)
    };
    let blocker = blocker.or_else(|| counter.escapes.then_some(FoldBlocker::HelperCall));
    (samples, blocker)
}

/// Why `expression`, if it accesses `input`, makes folding non-equivalent.
fn access_blocker(
    expression: &Expression,
    input: Handle<Expression>,
    declared: Option<Handle<Expression>>,
    filter: SamplerFilter,
) -> Option<FoldBlocker> {
    match *expression {
        Expression::ImageQuery { image, .. } if image == input => Some(FoldBlocker::ImageQuery),
        Expression::ImageLoad { image, .. } if image == input => None,
        Expression::ImageSample {
            image,
            gather,
            depth_ref,
            sampler: used,
            ..
        } if image == input => {
            if gather.is_some() {
                Some(FoldBlocker::Gather)
            } else if depth_ref.is_some() {
                Some(FoldBlocker::DepthComparison)
            } else if Some(used) != declared {
                Some(FoldBlocker::UnknownUse)
            } else if filter == SamplerFilter::Filtered {
                Some(FoldBlocker::FilteringSampler)
            } else {
                None
            }
        }
        ref other if reads(other, input) => Some(FoldBlocker::UnknownUse),
        _ => None,
    }
}

/// Whether `expression` reads `operand`, in any operand position.
pub fn reads(expression: &Expression, operand: Handle<Expression>) -> bool {
    let mut found = false;
    let _ = map_expression(expression, |handle| {
        found |= handle == operand;
        handle
    });
    found
}

/// The number of computed expressions in `apply` and every function it
/// calls, transitively.
fn total_ops(module: &Module, apply: Handle<Function>) -> u32 {
    let mut ops = 0u32;
    let mut seen = HashSet::new();
    let mut queue = vec![apply];
    while let Some(handle) = queue.pop() {
        if !seen.insert(handle) {
            continue;
        }
        let function = &module.functions[handle];
        ops = ops.saturating_add(
            u32::try_from(
                function
                    .expressions
                    .iter()
                    .filter(|(_, expression)| !expression.needs_pre_emit())
                    .count(),
            )
            .unwrap_or(u32::MAX),
        );
        collect_calls(&function.body, &mut queue);
    }
    ops
}

fn collect_calls(block: &Block, calls: &mut Vec<Handle<Function>>) {
    for statement in block {
        match statement {
            Statement::Call { function, .. } => calls.push(*function),
            Statement::Block(inner) => collect_calls(inner, calls),
            Statement::If { accept, reject, .. } => {
                collect_calls(accept, calls);
                collect_calls(reject, calls);
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_calls(&case.body, calls);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                collect_calls(body, calls);
                collect_calls(continuing, calls);
            }
            _ => {}
        }
    }
}

struct SampleCounter<'f> {
    function: &'f Function,
    input: Handle<Expression>,
    count: u32,
    in_loop: bool,
    dynamic: bool,
    escapes: bool,
}

impl SampleCounter<'_> {
    fn block(&mut self, block: &Block) {
        for statement in block {
            match statement {
                Statement::Emit(range) => {
                    for handle in range.clone() {
                        if samples_input(&self.function.expressions[handle], self.input) {
                            self.count += 1;
                            if self.in_loop {
                                self.dynamic = true;
                            }
                        }
                    }
                }
                Statement::Call { arguments, .. } => {
                    if arguments.contains(&self.input) {
                        self.escapes = true;
                        self.dynamic = true;
                    }
                }
                Statement::Block(inner) => self.block(inner),
                Statement::If { accept, reject, .. } => {
                    self.block(accept);
                    self.block(reject);
                }
                Statement::Switch { cases, .. } => {
                    for case in cases {
                        self.block(&case.body);
                    }
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    let outer = self.in_loop;
                    self.in_loop = true;
                    self.block(body);
                    self.block(continuing);
                    self.in_loop = outer;
                }
                _ => {}
            }
        }
    }
}

/// Whether `expression` samples or loads the texture read by `input`.
pub fn samples_input(expression: &Expression, input: Handle<Expression>) -> bool {
    match *expression {
        Expression::ImageSample { image, .. } | Expression::ImageLoad { image, .. } => {
            image == input
        }
        _ => false,
    }
}
