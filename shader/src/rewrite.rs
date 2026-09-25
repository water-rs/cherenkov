//! IR rewrites: mapping expression handles, folding a colour prefix into
//! every sample a spatial function takes of its input, and substituting the
//! implementation of filtered samples.

use std::collections::{HashMap, HashSet};

use naga::{
    Block, CooperativeData, Expression, Function, FunctionArgument, GatherMode, Handle, ImageQuery,
    Literal, Module, Range, SampleLevel, Statement, SwitchCase,
};

use crate::{import::GENERATED, parse};

/// Rebuilds `expression` with every expression operand passed through `map`.
/// Handles into other arenas (types, constants, functions) are kept.
#[expect(
    clippy::too_many_lines,
    reason = "one arm per naga expression variant; the match stays exhaustive so a new variant fails to compile"
)]
pub fn map_expression(
    expression: &Expression,
    mut map: impl FnMut(Handle<Expression>) -> Handle<Expression>,
) -> Expression {
    match *expression {
        Expression::Literal(_)
        | Expression::Constant(_)
        | Expression::Override(_)
        | Expression::ZeroValue(_)
        | Expression::FunctionArgument(_)
        | Expression::GlobalVariable(_)
        | Expression::LocalVariable(_)
        | Expression::CallResult(_)
        | Expression::AtomicResult { .. }
        | Expression::WorkGroupUniformLoadResult { .. }
        | Expression::RayQueryProceedResult
        | Expression::SubgroupBallotResult
        | Expression::SubgroupOperationResult { .. } => expression.clone(),
        Expression::Compose { ty, ref components } => Expression::Compose {
            ty,
            components: components.iter().map(|&component| map(component)).collect(),
        },
        Expression::Access { base, index } => Expression::Access {
            base: map(base),
            index: map(index),
        },
        Expression::AccessIndex { base, index } => Expression::AccessIndex {
            base: map(base),
            index,
        },
        Expression::Splat { size, value } => Expression::Splat {
            size,
            value: map(value),
        },
        Expression::Swizzle {
            size,
            vector,
            pattern,
        } => Expression::Swizzle {
            size,
            vector: map(vector),
            pattern,
        },
        Expression::Load { pointer } => Expression::Load {
            pointer: map(pointer),
        },
        Expression::ImageSample {
            image,
            sampler,
            gather,
            coordinate,
            array_index,
            offset,
            level,
            depth_ref,
            clamp_to_edge,
        } => {
            let level = match level {
                SampleLevel::Auto => SampleLevel::Auto,
                SampleLevel::Zero => SampleLevel::Zero,
                SampleLevel::Exact(value) => SampleLevel::Exact(map(value)),
                SampleLevel::Bias(value) => SampleLevel::Bias(map(value)),
                SampleLevel::Gradient { x, y } => SampleLevel::Gradient {
                    x: map(x),
                    y: map(y),
                },
            };
            Expression::ImageSample {
                image: map(image),
                sampler: map(sampler),
                gather,
                coordinate: map(coordinate),
                array_index: array_index.map(&mut map),
                offset: offset.map(&mut map),
                level,
                depth_ref: depth_ref.map(&mut map),
                clamp_to_edge,
            }
        }
        Expression::ImageLoad {
            image,
            coordinate,
            array_index,
            sample,
            level,
        } => Expression::ImageLoad {
            image: map(image),
            coordinate: map(coordinate),
            array_index: array_index.map(&mut map),
            sample: sample.map(&mut map),
            level: level.map(&mut map),
        },
        Expression::ImageQuery { image, query } => Expression::ImageQuery {
            image: map(image),
            query: match query {
                ImageQuery::Size { level } => ImageQuery::Size {
                    level: level.map(&mut map),
                },
                other => other,
            },
        },
        Expression::Unary { op, expr } => Expression::Unary {
            op,
            expr: map(expr),
        },
        Expression::Binary { op, left, right } => Expression::Binary {
            op,
            left: map(left),
            right: map(right),
        },
        Expression::Select {
            condition,
            accept,
            reject,
        } => Expression::Select {
            condition: map(condition),
            accept: map(accept),
            reject: map(reject),
        },
        Expression::Derivative { axis, ctrl, expr } => Expression::Derivative {
            axis,
            ctrl,
            expr: map(expr),
        },
        Expression::Relational { fun, argument } => Expression::Relational {
            fun,
            argument: map(argument),
        },
        Expression::Math {
            fun,
            arg,
            arg1,
            arg2,
            arg3,
        } => Expression::Math {
            fun,
            arg: map(arg),
            arg1: arg1.map(&mut map),
            arg2: arg2.map(&mut map),
            arg3: arg3.map(&mut map),
        },
        Expression::As {
            expr,
            kind,
            convert,
        } => Expression::As {
            expr: map(expr),
            kind,
            convert,
        },
        Expression::ArrayLength(array) => Expression::ArrayLength(map(array)),
        Expression::RayQueryVertexPositions { query, committed } => {
            Expression::RayQueryVertexPositions {
                query: map(query),
                committed,
            }
        }
        Expression::RayQueryGetIntersection { query, committed } => {
            Expression::RayQueryGetIntersection {
                query: map(query),
                committed,
            }
        }
        Expression::CooperativeLoad {
            columns,
            rows,
            role,
            data,
        } => Expression::CooperativeLoad {
            columns,
            rows,
            role,
            data: CooperativeData {
                pointer: map(data.pointer),
                stride: map(data.stride),
                row_major: data.row_major,
            },
        },
        Expression::CooperativeMultiplyAdd { a, b, c } => Expression::CooperativeMultiplyAdd {
            a: map(a),
            b: map(b),
            c: map(c),
        },
    }
}

/// Builds a copy of the spatial function `spatial` in which every sample of
/// argument `input` is passed through `prefix` before use.
///
/// The copy gains `extra` as trailing arguments; they are passed to `prefix`
/// after the sampled colour, in order.
pub fn fold_prefix(
    spatial: &Function,
    input: u32,
    prefix: Handle<Function>,
    extra: &[FunctionArgument],
    name: String,
) -> Function {
    let mut folded = Function {
        name: Some(name),
        arguments: spatial.arguments.clone(),
        result: spatial.result.clone(),
        local_variables: spatial.local_variables.clone(),
        diagnostic_filter_leaf: spatial.diagnostic_filter_leaf,
        ..Function::default()
    };
    let first_extra = u32::try_from(folded.arguments.len()).expect("argument count fits in u32");
    folded.arguments.extend_from_slice(extra);

    let input = parse::argument_expression(spatial, input);
    let mut rebuild = Rebuild {
        map: Vec::with_capacity(spatial.expressions.len()),
        replaced: HashMap::new(),
    };
    for (handle, expression) in spatial.expressions.iter() {
        let copy = map_expression(expression, |operand| rebuild.operand(operand));
        let span = spatial.expressions.get_span(handle);
        let new = folded.expressions.append(copy, span);
        rebuild.map.push(new);
        if input.is_some_and(|input| parse::samples_input(expression, input)) {
            let call = folded
                .expressions
                .append(Expression::CallResult(prefix), GENERATED);
            rebuild.replaced.insert(handle, call);
        }
    }
    let extra_arguments: Vec<_> = (0..extra.len())
        .map(|offset| {
            let index = first_extra + u32::try_from(offset).expect("argument count fits in u32");
            folded
                .expressions
                .append(Expression::FunctionArgument(index), GENERATED)
        })
        .collect();

    for (_, local) in folded.local_variables.iter_mut() {
        local.init = local.init.map(|init| rebuild.operand(init));
    }
    for (old, name) in &spatial.named_expressions {
        folded
            .named_expressions
            .insert(rebuild.map[old.index()], name.clone());
    }
    folded.body = rebuild.block(&spatial.body, prefix, &extra_arguments);
    folded
}

struct Rebuild {
    /// Old expression index to its copy.
    map: Vec<Handle<Expression>>,
    /// Old sample expression to the result of the prefix applied to it.
    replaced: HashMap<Handle<Expression>, Handle<Expression>>,
}

impl Rebuild {
    /// Where a use of `old` points after the rewrite.
    fn operand(&self, old: Handle<Expression>) -> Handle<Expression> {
        self.replaced
            .get(&old)
            .copied()
            .unwrap_or_else(|| self.map[old.index()])
    }

    /// Where the definition of `old` lives after the rewrite.
    fn defined(&self, old: Handle<Expression>) -> Handle<Expression> {
        self.map[old.index()]
    }

    fn block(
        &self,
        block: &Block,
        prefix: Handle<Function>,
        extra: &[Handle<Expression>],
    ) -> Block {
        let mut out = Block::with_capacity(block.len());
        for (statement, span) in block.span_iter() {
            match statement {
                Statement::Emit(range) => self.emit(range, &mut out, *span, prefix, extra),
                other => out.push(self.statement(other, prefix, extra), *span),
            }
        }
        out
    }

    /// Re-emits `range`, splitting it after every folded sample to call the
    /// prefix on that sample.
    fn emit(
        &self,
        range: &Range<Expression>,
        out: &mut Block,
        span: naga::Span,
        prefix: Handle<Function>,
        extra: &[Handle<Expression>],
    ) {
        let mut run: Option<(Handle<Expression>, Handle<Expression>)> = None;
        for old in range.clone() {
            let new = self.defined(old);
            run = Some(run.map_or((new, new), |(first, _)| (first, new)));
            if let Some(&call) = self.replaced.get(&old) {
                let (first, last) = run.take().expect("the run holds the sample");
                out.push(Statement::Emit(Range::new_from_bounds(first, last)), span);
                let mut arguments = Vec::with_capacity(1 + extra.len());
                arguments.push(new);
                arguments.extend_from_slice(extra);
                out.push(
                    Statement::Call {
                        function: prefix,
                        arguments,
                        result: Some(call),
                    },
                    GENERATED,
                );
            }
        }
        if let Some((first, last)) = run {
            out.push(Statement::Emit(Range::new_from_bounds(first, last)), span);
        }
    }

    fn statement(
        &self,
        statement: &Statement,
        prefix: Handle<Function>,
        extra: &[Handle<Expression>],
    ) -> Statement {
        let op = |handle: Handle<Expression>| self.operand(handle);
        let block = |block: &Block| self.block(block, prefix, extra);
        match *statement {
            Statement::Emit(_) => unreachable!("emits are rebuilt by `Rebuild::emit`"),
            Statement::Block(ref inner) => Statement::Block(block(inner)),
            Statement::If {
                condition,
                ref accept,
                ref reject,
            } => Statement::If {
                condition: op(condition),
                accept: block(accept),
                reject: block(reject),
            },
            Statement::Switch {
                selector,
                ref cases,
            } => Statement::Switch {
                selector: op(selector),
                cases: cases
                    .iter()
                    .map(|case| SwitchCase {
                        value: case.value,
                        body: block(&case.body),
                        fall_through: case.fall_through,
                    })
                    .collect(),
            },
            Statement::Loop {
                ref body,
                ref continuing,
                break_if,
            } => Statement::Loop {
                body: block(body),
                continuing: block(continuing),
                break_if: break_if.map(op),
            },
            Statement::Break => Statement::Break,
            Statement::Continue => Statement::Continue,
            Statement::Return { value } => Statement::Return {
                value: value.map(op),
            },
            Statement::Store { pointer, value } => Statement::Store {
                pointer: op(pointer),
                value: op(value),
            },
            Statement::Call {
                function,
                ref arguments,
                result,
            } => Statement::Call {
                function,
                arguments: arguments.iter().map(|&argument| op(argument)).collect(),
                result: result.map(|result| self.defined(result)),
            },
            Statement::SubgroupBallot { result, predicate } => Statement::SubgroupBallot {
                result: self.defined(result),
                predicate: predicate.map(op),
            },
            Statement::SubgroupGather {
                mode,
                argument,
                result,
            } => Statement::SubgroupGather {
                mode: gather_mode(mode, op),
                argument: op(argument),
                result: self.defined(result),
            },
            Statement::SubgroupCollectiveOperation {
                op: operation,
                collective_op,
                argument,
                result,
            } => Statement::SubgroupCollectiveOperation {
                op: operation,
                collective_op,
                argument: op(argument),
                result: self.defined(result),
            },
            Statement::Kill
            | Statement::ControlBarrier(_)
            | Statement::MemoryBarrier(_)
            | Statement::ImageStore { .. }
            | Statement::Atomic { .. }
            | Statement::ImageAtomic { .. }
            | Statement::WorkGroupUniformLoad { .. }
            | Statement::RayQuery { .. }
            | Statement::RayPipelineFunction(_)
            | Statement::CooperativeStore { .. } => {
                unreachable!("snippet parsing rejects {statement:?}")
            }
        }
    }
}

fn gather_mode(
    mode: GatherMode,
    op: impl Fn(Handle<Expression>) -> Handle<Expression>,
) -> GatherMode {
    match mode {
        GatherMode::BroadcastFirst => GatherMode::BroadcastFirst,
        GatherMode::Broadcast(index) => GatherMode::Broadcast(op(index)),
        GatherMode::Shuffle(index) => GatherMode::Shuffle(op(index)),
        GatherMode::ShuffleDown(delta) => GatherMode::ShuffleDown(op(delta)),
        GatherMode::ShuffleUp(delta) => GatherMode::ShuffleUp(op(delta)),
        GatherMode::ShuffleXor(mask) => GatherMode::ShuffleXor(op(mask)),
        GatherMode::QuadBroadcast(index) => GatherMode::QuadBroadcast(op(index)),
        GatherMode::QuadSwap(direction) => GatherMode::QuadSwap(direction),
    }
}

// ============================================================================
// Sample substitution: `textureSampleLevel(input, input_sampler, uv)` runs as
// a manual bilinear of texel loads.
// ============================================================================

/// Replaces `function`'s filtered samples of `input` by calls to `manual`.
///
/// `input` is the index of the texture argument samples are taken of, and
/// `sampler` — when `Some` — the index of the sampler argument those samples
/// must go through. `manual` is a function `fn(texture_2d<f32>, vec2<f32>) ->
/// vec4<f32>` implementing a filtered sample out of texel loads.
///
/// The substitution is transitive: a callee that receives `input` is cloned
/// once per combination of argument positions `input` binds to, with its own
/// samples replaced, and the call is retargeted.
///
/// Returns the function to call instead of `function` — `function` itself
/// when nothing needed substituting — or `None` when a use of `input` has no
/// manual equivalent: a gather, a depth reference, an array index or offset,
/// a level other than mip 0, a sampler that is not the function's own, a
/// store or atomic, or a call into a function that cannot itself be
/// substituted.
pub fn substitute_samples(
    module: &mut Module,
    function: Handle<Function>,
    input: u32,
    sampler: Option<u32>,
    manual: Handle<Function>,
) -> Option<Handle<Function>> {
    let names: HashSet<String> = module
        .functions
        .iter()
        .filter_map(|(_, function)| function.name.clone())
        .collect();
    let mut substitution = Substitution {
        module,
        manual,
        functions: HashMap::new(),
        in_progress: HashSet::new(),
        names,
    };
    substitution.substitute(function, &[input], sampler)
}

/// A substitution key: a callee and the sorted argument positions the
/// caller's `input` binds to.
type Key = (Handle<Function>, Vec<u32>);

/// One call `input` flows into, found while scanning a function's body.
struct BoundCall {
    /// The callee.
    function: Handle<Function>,
    /// The callee argument positions `input` binds to, sorted.
    bound: Vec<u32>,
    /// The call's `CallResult` expression, when the call returns a value.
    result: Option<Handle<Expression>>,
}

/// A substitution in progress: clones are memoized per callee and bound
/// argument positions.
struct Substitution<'a> {
    /// The module new functions are appended to.
    module: &'a mut Module,
    /// The manual sample implementation calls are retargeted to.
    manual: Handle<Function>,
    /// `(function, argument positions `input` binds to)` to the function to
    /// call instead, or `None` when it cannot be substituted.
    functions: HashMap<Key, Option<Handle<Function>>>,
    /// Keys currently being substituted — a cycle is not substitutable.
    in_progress: HashSet<Key>,
    /// Function names already taken, for the clones' unique names.
    names: HashSet<String>,
}

/// What one replaced sample expression becomes: the manual call's
/// `CallResult`, and the source operands the call takes.
#[derive(Clone, Copy)]
struct SubstitutedSample {
    /// The `CallResult` of the call to `manual` that replaces the sample.
    result: Handle<Expression>,
    /// The sample's image operand — the `input` expression.
    image: Handle<Expression>,
    /// The sample's coordinate operand.
    coordinate: Handle<Expression>,
}

impl Substitution<'_> {
    /// The substitute of `function` when `inputs`' argument positions bind to
    /// the caller's `input` — `function` itself when nothing changes, or
    /// `None` when it cannot be substituted.
    fn substitute(
        &mut self,
        function: Handle<Function>,
        inputs: &[u32],
        sampler: Option<u32>,
    ) -> Option<Handle<Function>> {
        let mut positions = inputs.to_vec();
        positions.sort_unstable();
        let key = (function, positions);
        if let Some(&substituted) = self.functions.get(&key) {
            return substituted;
        }
        // WGSL admits no recursion; bail rather than looping on a cycle.
        if !self.in_progress.insert(key.clone()) {
            return None;
        }
        let substituted = self.try_substitute(function, &key.1, sampler);
        self.in_progress.remove(&key);
        self.functions.insert(key, substituted);
        substituted
    }

    /// Clones `function` with its substitutable samples of `inputs` replaced;
    /// `sampler` is the declared sampler argument a sample must use (`None`
    /// for a callee, whose sampler arguments all derive from the caller's).
    fn try_substitute(
        &mut self,
        function: Handle<Function>,
        inputs: &[u32],
        sampler: Option<u32>,
    ) -> Option<Handle<Function>> {
        let source = self.module.functions[function].clone();
        let inputs: Vec<Handle<Expression>> = inputs
            .iter()
            .map(|&arg| parse::argument_expression(&source, arg))
            .collect::<Option<_>>()?;
        let sampler = match sampler {
            Some(arg) => Some(parse::argument_expression(&source, arg)?),
            None => None,
        };

        let replaced = substitutable_samples(&source, &inputs, sampler)?;
        let calls = bound_calls(&source, &inputs)?;

        // `(callee, bound positions)` to the substituted callee.
        let mut call_fns: HashMap<Key, Handle<Function>> = HashMap::new();
        // A call's `CallResult` expression to its substituted callee.
        let mut call_results: HashMap<Handle<Expression>, Handle<Function>> = HashMap::new();
        for call in calls {
            let substituted = self.substitute(call.function, &call.bound, None)?;
            call_fns.insert((call.function, call.bound), substituted);
            if let Some(result) = call.result {
                call_results.insert(result, substituted);
            }
        }
        if replaced.is_empty() && call_fns.iter().all(|(&(callee, _), &call)| callee == call) {
            // Nothing samples `input` and no call needed retargeting.
            return Some(function);
        }
        Some(self.rebuild(&source, &inputs, replaced, &call_fns, &call_results))
    }

    /// Clones `source` into the module: every replaced sample becomes a call
    /// to `manual`, every bound call is retargeted to its substituted callee.
    fn rebuild(
        &mut self,
        source: &Function,
        inputs: &[Handle<Expression>],
        replaced: Vec<(Handle<Expression>, Handle<Expression>)>,
        call_fns: &HashMap<Key, Handle<Function>>,
        call_results: &HashMap<Handle<Expression>, Handle<Function>>,
    ) -> Handle<Function> {
        let replaced: HashMap<Handle<Expression>, Handle<Expression>> =
            replaced.into_iter().collect();
        let mut rebuilt = Function {
            name: Some(self.name_for(source)),
            arguments: source.arguments.clone(),
            result: source.result.clone(),
            local_variables: source.local_variables.clone(),
            diagnostic_filter_leaf: source.diagnostic_filter_leaf,
            ..Function::default()
        };

        // As in `fold_prefix`: `map` indexes every source expression to its
        // copy, and `substituted` records a replaced sample's call along with
        // the operands the call takes.
        let mut substitute = Substitute {
            map: Vec::with_capacity(source.expressions.len()),
            substituted: HashMap::new(),
            manual: self.manual,
            inputs,
            call_fns,
        };
        for (handle, expression) in source.expressions.iter() {
            if let Some(&coordinate) = replaced.get(&handle) {
                // The sample itself is never copied or emitted; the call's
                // `CallResult` takes its slot so uses remap to it.
                let call = rebuilt
                    .expressions
                    .append(Expression::CallResult(self.manual), GENERATED);
                let Expression::ImageSample { image, .. } = *expression else {
                    unreachable!("a substituted expression is an ImageSample")
                };
                substitute.map.push(call);
                substitute.substituted.insert(
                    handle,
                    SubstitutedSample {
                        result: call,
                        image,
                        coordinate,
                    },
                );
                continue;
            }
            let copy = match *expression {
                Expression::CallResult(_) => call_results.get(&handle).map_or_else(
                    || expression.clone(),
                    |&target| Expression::CallResult(target),
                ),
                _ => map_expression(expression, |operand| substitute.operand(operand)),
            };
            let new = rebuilt
                .expressions
                .append(copy, source.expressions.get_span(handle));
            substitute.map.push(new);
        }

        for (_, local) in rebuilt.local_variables.iter_mut() {
            local.init = local.init.map(|init| substitute.operand(init));
        }
        for (old, name) in &source.named_expressions {
            rebuilt
                .named_expressions
                .insert(substitute.defined(*old), name.clone());
        }
        rebuilt.body = substitute.block(&source.body);

        self.module.functions.append(rebuilt, GENERATED)
    }

    /// A unique name for `source`'s clone.
    fn name_for(&mut self, source: &Function) -> String {
        let base = format!("{}_manual", source.name.as_deref().unwrap_or("function"));
        let mut name = base.clone();
        let mut suffix = 0;
        while self.names.contains(&name) {
            suffix += 1;
            name = format!("{base}_{suffix}");
        }
        self.names.insert(name.clone());
        name
    }
}

/// Statement rebuilding for [`substitute_samples`], mirroring the fold's
/// [`Rebuild`]: the same `map`, but a replaced sample becomes a `Call` to the
/// manual bilinear at its position in the emit range, and calls are
/// retargeted to substituted callees.
struct Substitute<'a> {
    /// Every source expression to its copy; a replaced sample's entry is the
    /// call's `CallResult`.
    map: Vec<Handle<Expression>>,
    /// Source sample expression to its replacement.
    substituted: HashMap<Handle<Expression>, SubstitutedSample>,
    /// The manual bilinear function samples call.
    manual: Handle<Function>,
    /// The clone's own `input` argument expressions, for call retargeting.
    inputs: &'a [Handle<Expression>],
    /// `(callee, bound argument positions)` to the substituted callee.
    call_fns: &'a HashMap<(Handle<Function>, Vec<u32>), Handle<Function>>,
}

impl Substitute<'_> {
    /// Where a use of `old` points after the rewrite.
    fn operand(&self, old: Handle<Expression>) -> Handle<Expression> {
        self.substituted
            .get(&old)
            .map_or_else(|| self.map[old.index()], |substituted| substituted.result)
    }

    /// Where the definition of `old` lives after the rewrite.
    fn defined(&self, old: Handle<Expression>) -> Handle<Expression> {
        self.map[old.index()]
    }

    fn block(&self, block: &Block) -> Block {
        let mut out = Block::with_capacity(block.len());
        for (statement, span) in block.span_iter() {
            match statement {
                Statement::Emit(range) => self.emit(range, &mut out, *span),
                other => out.push(self.statement(other), *span),
            }
        }
        out
    }

    /// Re-emits `range`, splitting it at each replaced sample: the sample
    /// becomes a `Call` to `manual` on its image and coordinate operands.
    fn emit(&self, range: &Range<Expression>, out: &mut Block, span: naga::Span) {
        let mut run: Option<(Handle<Expression>, Handle<Expression>)> = None;
        for old in range.clone() {
            if let Some(&substituted) = self.substituted.get(&old) {
                if let Some((first, last)) = run.take() {
                    out.push(Statement::Emit(Range::new_from_bounds(first, last)), span);
                }
                out.push(
                    Statement::Call {
                        function: self.manual,
                        arguments: vec![
                            self.operand(substituted.image),
                            self.operand(substituted.coordinate),
                        ],
                        result: Some(substituted.result),
                    },
                    GENERATED,
                );
            } else {
                let new = self.defined(old);
                run = Some(run.map_or((new, new), |(first, _)| (first, new)));
            }
        }
        if let Some((first, last)) = run {
            out.push(Statement::Emit(Range::new_from_bounds(first, last)), span);
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one arm per naga statement variant; the match stays exhaustive so a new variant fails to compile"
    )]
    fn statement(&self, statement: &Statement) -> Statement {
        let op = |handle: Handle<Expression>| self.operand(handle);
        let block = |block: &Block| self.block(block);
        match *statement {
            Statement::Emit(_) => unreachable!("emits are rebuilt by `Substitute::emit`"),
            Statement::Block(ref inner) => Statement::Block(block(inner)),
            Statement::If {
                condition,
                ref accept,
                ref reject,
            } => Statement::If {
                condition: op(condition),
                accept: block(accept),
                reject: block(reject),
            },
            Statement::Switch {
                selector,
                ref cases,
            } => Statement::Switch {
                selector: op(selector),
                cases: cases
                    .iter()
                    .map(|case| SwitchCase {
                        value: case.value,
                        body: block(&case.body),
                        fall_through: case.fall_through,
                    })
                    .collect(),
            },
            Statement::Loop {
                ref body,
                ref continuing,
                break_if,
            } => Statement::Loop {
                body: block(body),
                continuing: block(continuing),
                break_if: break_if.map(op),
            },
            Statement::Break => Statement::Break,
            Statement::Continue => Statement::Continue,
            Statement::Return { value } => Statement::Return {
                value: value.map(op),
            },
            Statement::Store { pointer, value } => Statement::Store {
                pointer: op(pointer),
                value: op(value),
            },
            Statement::Call {
                function,
                ref arguments,
                result,
            } => {
                let mut bound: Vec<u32> = arguments
                    .iter()
                    .enumerate()
                    .filter(|&(_, argument)| self.inputs.contains(argument))
                    .map(|(index, _)| u32::try_from(index).expect("argument index fits in u32"))
                    .collect();
                let function = if bound.is_empty() {
                    function
                } else {
                    bound.sort_unstable();
                    self.call_fns[&(function, bound)]
                };
                Statement::Call {
                    function,
                    arguments: arguments.iter().map(|&argument| op(argument)).collect(),
                    result: result.map(|result| self.defined(result)),
                }
            }
            Statement::SubgroupBallot { result, predicate } => Statement::SubgroupBallot {
                result: self.defined(result),
                predicate: predicate.map(op),
            },
            Statement::SubgroupGather {
                mode,
                argument,
                result,
            } => Statement::SubgroupGather {
                mode: gather_mode(mode, op),
                argument: op(argument),
                result: self.defined(result),
            },
            Statement::SubgroupCollectiveOperation {
                op: operation,
                collective_op,
                argument,
                result,
            } => Statement::SubgroupCollectiveOperation {
                op: operation,
                collective_op,
                argument: op(argument),
                result: self.defined(result),
            },
            Statement::Kill
            | Statement::ControlBarrier(_)
            | Statement::MemoryBarrier(_)
            | Statement::ImageStore { .. }
            | Statement::Atomic { .. }
            | Statement::ImageAtomic { .. }
            | Statement::WorkGroupUniformLoad { .. }
            | Statement::RayQuery { .. }
            | Statement::RayPipelineFunction(_)
            | Statement::CooperativeStore { .. } => {
                unreachable!("snippet parsing rejects {statement:?}")
            }
        }
    }
}

/// The `ImageSample` expressions of `inputs` a manual call replaces, each
/// keyed to its coordinate operand — `None` when an expression has no
/// manual equivalent.
fn substitutable_samples(
    source: &Function,
    inputs: &[Handle<Expression>],
    sampler: Option<Handle<Expression>>,
) -> Option<Vec<(Handle<Expression>, Handle<Expression>)>> {
    let mut replaced = Vec::new();
    for (handle, expression) in source.expressions.iter() {
        match *expression {
            Expression::ImageSample {
                image,
                sampler: used,
                coordinate,
                gather,
                array_index,
                offset,
                level,
                depth_ref,
                clamp_to_edge: _,
            } if inputs.contains(&image) => {
                if gather.is_some()
                    || depth_ref.is_some()
                    || array_index.is_some()
                    || offset.is_some()
                    || !manual_level(source, level)
                    || !manual_sampler(source, used, sampler)
                {
                    return None;
                }
                replaced.push((handle, coordinate));
            }
            // Texel loads and queries reproduce as they are.
            Expression::ImageLoad { image, .. } | Expression::ImageQuery { image, .. }
                if inputs.contains(&image) => {}
            // Any other read of `input` has no manual equivalent.
            ref other if inputs.iter().any(|&input| parse::reads(other, input)) => {
                return None;
            }
            _ => {}
        }
    }
    Some(replaced)
}

/// The calls `input` flows into — `None` when `input` is stored to or
/// atomically accessed, which no manual equivalent covers.
fn bound_calls(source: &Function, inputs: &[Handle<Expression>]) -> Option<Vec<BoundCall>> {
    let mut calls = Vec::new();
    let mut stored = false;
    read_block(&source.body, &mut |statement| match *statement {
        Statement::Call {
            function: callee,
            ref arguments,
            result,
        } => {
            let bound: Vec<u32> = arguments
                .iter()
                .enumerate()
                .filter(|&(_, argument)| inputs.contains(argument))
                .map(|(index, _)| u32::try_from(index).expect("argument index fits in u32"))
                .collect();
            if !bound.is_empty() {
                calls.push(BoundCall {
                    function: callee,
                    bound,
                    result,
                });
            }
        }
        Statement::ImageStore { image, .. } | Statement::ImageAtomic { image, .. }
            if inputs.contains(&image) =>
        {
            stored = true;
        }
        _ => {}
    });
    if stored { None } else { Some(calls) }
}

/// Whether `level` resolves to mip 0 — the only level a manual sample
/// reproduces.
fn manual_level(function: &Function, level: SampleLevel) -> bool {
    match level {
        SampleLevel::Auto | SampleLevel::Zero => true,
        SampleLevel::Exact(value) => zero_level(function, value),
        SampleLevel::Bias(_) | SampleLevel::Gradient { .. } => false,
    }
}

/// Whether `used` is the function's own sampler argument: `declared` for the
/// stage's `apply`, or any sampler argument of a substituted helper — its
/// samplers all derive from `apply`'s.
fn manual_sampler(
    function: &Function,
    used: Handle<Expression>,
    declared: Option<Handle<Expression>>,
) -> bool {
    declared.map_or_else(
        || matches!(function.expressions[used], Expression::FunctionArgument(_)),
        |declared| used == declared,
    )
}

/// Whether `value` is the literal `0.0`.
fn zero_level(function: &Function, value: Handle<Expression>) -> bool {
    match function.expressions[value] {
        Expression::Literal(Literal::F32(v)) => v == 0.0,
        Expression::Literal(Literal::F16(v)) => v.to_bits() == 0,
        Expression::Literal(Literal::AbstractFloat(v)) => v == 0.0,
        _ => false,
    }
}

/// Walks every statement of `block` and its nested blocks.
fn read_block(block: &Block, each: &mut impl FnMut(&Statement)) {
    for statement in block {
        each(statement);
        match statement {
            Statement::Block(inner) => read_block(inner, each),
            Statement::If { accept, reject, .. } => {
                read_block(accept, each);
                read_block(reject, each);
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    read_block(&case.body, each);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                read_block(body, each);
                read_block(continuing, each);
            }
            _ => {}
        }
    }
}
