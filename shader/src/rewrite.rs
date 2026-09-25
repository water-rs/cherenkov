//! IR rewrites: mapping expression handles, and folding a colour prefix into
//! every sample a spatial function takes of its input.

use std::collections::HashMap;

use naga::{
    Block, CooperativeData, Expression, Function, FunctionArgument, GatherMode, Handle, ImageQuery,
    Range, SampleLevel, Statement, SwitchCase,
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
