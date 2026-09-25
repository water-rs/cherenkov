//! Building generated functions expression by expression, with the emit
//! bookkeeping naga requires.

use naga::{
    Expression, Function, FunctionArgument, FunctionResult, Handle, Literal, Range, ScalarKind,
    Statement, Type,
};

use crate::{abi::Precision, import::GENERATED};

/// Builds one function. Expressions that need emitting are grouped into
/// `Emit` statements automatically.
pub struct FunctionBuilder {
    function: Function,
    pending: Option<(Handle<Expression>, Handle<Expression>)>,
}

impl FunctionBuilder {
    pub(crate) fn new(name: String) -> Self {
        Self {
            function: Function {
                name: Some(name),
                ..Function::default()
            },
            pending: None,
        }
    }

    /// Adds an argument and returns the expression that reads it.
    pub(crate) fn argument(&mut self, name: &str, ty: Handle<Type>) -> Handle<Expression> {
        let index =
            u32::try_from(self.function.arguments.len()).expect("argument count fits in u32");
        self.function.arguments.push(FunctionArgument {
            name: Some(name.to_owned()),
            ty,
            binding: None,
        });
        self.expression(Expression::FunctionArgument(index))
    }

    /// Appends an expression, emitting it when naga requires that.
    pub(crate) fn expression(&mut self, expression: Expression) -> Handle<Expression> {
        if expression.needs_pre_emit() {
            self.flush();
            return self.function.expressions.append(expression, GENERATED);
        }
        let handle = self.function.expressions.append(expression, GENERATED);
        self.pending = Some(
            self.pending
                .map_or((handle, handle), |(first, _)| (first, handle)),
        );
        handle
    }

    /// An `f32` literal.
    pub(crate) fn f32(&mut self, value: f32) -> Handle<Expression> {
        self.expression(Expression::Literal(Literal::F32(value)))
    }

    /// Converts a colour to `precision` when it has the other precision.
    pub(crate) fn convert(
        &mut self,
        colour: Handle<Expression>,
        from: Precision,
        to: Precision,
    ) -> Handle<Expression> {
        if from == to {
            return colour;
        }
        self.expression(Expression::As {
            expr: colour,
            kind: ScalarKind::Float,
            convert: Some(to.width()),
        })
    }

    /// Calls `function` and returns its result.
    pub(crate) fn call(
        &mut self,
        function: Handle<Function>,
        arguments: Vec<Handle<Expression>>,
    ) -> Handle<Expression> {
        self.flush();
        let result = self
            .function
            .expressions
            .append(Expression::CallResult(function), GENERATED);
        self.function.body.push(
            Statement::Call {
                function,
                arguments,
                result: Some(result),
            },
            GENERATED,
        );
        result
    }

    /// Returns `value` and finishes the function.
    pub(crate) fn finish(mut self, value: Handle<Expression>, result: Handle<Type>) -> Function {
        self.flush();
        self.function
            .body
            .push(Statement::Return { value: Some(value) }, GENERATED);
        self.function.result = Some(FunctionResult {
            ty: result,
            binding: None,
        });
        self.function
    }

    fn flush(&mut self) {
        if let Some((first, last)) = self.pending.take() {
            self.function.body.push(
                Statement::Emit(Range::new_from_bounds(first, last)),
                GENERATED,
            );
        }
    }
}
