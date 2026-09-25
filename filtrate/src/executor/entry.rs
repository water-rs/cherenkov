//! Wrapping a pass's segment into a fragment entry point, on the IR.
//!
//! A pass draws one full-screen triangle into a target of its input's size.
//! Its fragment entry point reads the input at the fragment's pixel — a texel
//! load for a colour segment, a normalized coordinate for a spatial one —
//! gathers the segment's other arguments from bindings, and returns the
//! segment's result.

extern crate alloc;

use alloc::{borrow::ToOwned, string::ToString, vec::Vec};

use cherenkov_shader::{
    FunctionBuilder, SegmentArg,
    naga::{
        AddressSpace, BinaryOperator, Binding, BuiltIn, EntryPoint, Expression, GlobalVariable,
        Handle, ImageClass, ImageDimension, ImageQuery, Literal, MemoryDecorations, Module,
        ResourceBinding, ScalarKind, ShaderStage, Span, SwizzleComponent, Type, TypeInner,
        VectorSize, valid::Capabilities,
    },
    validate,
};

use super::plan::PassPlan;
use crate::effect::EffectSetupError;

/// The bind group layout every pass follows; a pass binds only the entries
/// its segment takes.
pub(super) mod binding {
    /// The pass's input texture.
    pub const INPUT: u32 = 0;
    /// The input sampler of a spatial pass.
    pub const SAMPLER: u32 = 1;
    /// The segment's uniform block.
    pub const PARAMS: u32 = 2;
    /// The working-space constants.
    pub const SPACE: u32 = 3;
    /// The clip shape.
    pub const SHAPE: u32 = 4;
    /// Auxiliary image `n` binds at `AUX + n`.
    pub const AUX: u32 = 5;
}

/// The name of every pass's fragment entry point.
pub(super) const ENTRY_POINT: &str = "main";

/// Adds `pass`'s fragment entry point, and the bindings it reads, to a copy
/// of the composed module, and validates the result.
pub(super) fn pass_module(
    composed: &Module,
    capabilities: Capabilities,
    index: usize,
    pass: &PassPlan,
) -> Result<Module, EffectSetupError> {
    let mut module = composed.clone();
    let segment = module
        .functions
        .iter()
        .find(|(_, function)| function.name.as_deref() == Some(pass.segment.function.as_str()))
        .map(|(handle, _)| handle)
        .expect("the composition defines every segment it describes");
    let argument_types: Vec<Handle<Type>> = module.functions[segment]
        .arguments
        .iter()
        .map(|argument| argument.ty)
        .collect();
    let result_type = module.functions[segment]
        .result
        .as_ref()
        .expect("a segment returns a colour")
        .ty;

    let texture = insert_type(
        &mut module,
        TypeInner::Image {
            dim: ImageDimension::D2,
            arrayed: false,
            class: ImageClass::Sampled {
                kind: ScalarKind::Float,
                multi: false,
            },
        },
    );
    let position_type = insert_type(
        &mut module,
        TypeInner::Vector {
            size: VectorSize::Quad,
            scalar: cherenkov_shader::naga::Scalar::F32,
        },
    );
    let input = global(
        &mut module,
        "input",
        binding::INPUT,
        texture,
        AddressSpace::Handle,
    );

    let mut builder = FunctionBuilder::new("fragment");
    let position = builder.bound_argument(
        "position",
        position_type,
        Binding::BuiltIn(BuiltIn::Position { invariant: false }),
    );
    let pixel = builder.expression(Expression::Swizzle {
        size: VectorSize::Bi,
        vector: position,
        pattern: [
            SwizzleComponent::X,
            SwizzleComponent::Y,
            SwizzleComponent::X,
            SwizzleComponent::X,
        ],
    });
    let input_expression = builder.expression(Expression::GlobalVariable(input));

    let mut inputs = Inputs {
        builder: &mut builder,
        module: &mut module,
        input: input_expression,
        pixel,
    };
    let arguments = pass
        .segment
        .args
        .iter()
        .zip(&argument_types)
        .map(|(&arg, &ty)| inputs.argument(arg, ty))
        .collect();

    let result = builder.call(segment, arguments);
    let function = builder.finish_bound(
        result,
        result_type,
        Binding::Location {
            location: 0,
            interpolation: None,
            sampling: None,
            blend_src: None,
            per_primitive: false,
        },
    );
    module.entry_points.push(EntryPoint {
        name: ENTRY_POINT.to_owned(),
        stage: ShaderStage::Fragment,
        early_depth_test: None,
        workgroup_size: [0; 3],
        workgroup_size_overrides: None,
        function,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    });

    validate(&module, capabilities).map_err(|error| EffectSetupError::PassValidation {
        pass: index,
        message: error.to_string(),
    })?;
    Ok(module)
}

/// Builds the expressions a pass's segment arguments read.
struct Inputs<'a> {
    builder: &'a mut FunctionBuilder,
    module: &'a mut Module,
    /// The pass's input texture.
    input: Handle<Expression>,
    /// The fragment's pixel centre, in pixels.
    pixel: Handle<Expression>,
}

impl Inputs<'_> {
    /// The expression for one segment argument of type `ty`.
    fn argument(&mut self, arg: SegmentArg, ty: Handle<Type>) -> Handle<Expression> {
        match arg {
            SegmentArg::Color => {
                let coordinate = self.builder.expression(Expression::As {
                    expr: self.pixel,
                    kind: ScalarKind::Sint,
                    convert: Some(4),
                });
                let level = self
                    .builder
                    .expression(Expression::Literal(Literal::I32(0)));
                self.builder.expression(Expression::ImageLoad {
                    image: self.input,
                    coordinate,
                    array_index: None,
                    sample: None,
                    level: Some(level),
                })
            }
            SegmentArg::Input => self.input,
            SegmentArg::InputSampler => {
                self.resource("input_sampler", binding::SAMPLER, ty, AddressSpace::Handle)
            }
            SegmentArg::Uv => {
                let size = self.builder.expression(Expression::ImageQuery {
                    image: self.input,
                    query: ImageQuery::Size { level: None },
                });
                let size = self.builder.expression(Expression::As {
                    expr: size,
                    kind: ScalarKind::Float,
                    convert: Some(4),
                });
                self.builder.expression(Expression::Binary {
                    op: BinaryOperator::Divide,
                    left: self.pixel,
                    right: size,
                })
            }
            SegmentArg::Params => {
                let pointer = self.resource("params", binding::PARAMS, ty, AddressSpace::Uniform);
                self.builder.expression(Expression::Load { pointer })
            }
            SegmentArg::WorkingSpace => {
                let pointer = self.resource("space", binding::SPACE, ty, AddressSpace::Uniform);
                self.builder.expression(Expression::Load { pointer })
            }
            SegmentArg::Shape => self.resource("shape", binding::SHAPE, ty, AddressSpace::Handle),
            SegmentArg::Aux(n) => self.resource(
                &alloc::format!("aux{n}"),
                binding::AUX + n,
                ty,
                AddressSpace::Handle,
            ),
        }
    }

    /// Declares a bound global and returns the expression that reads it.
    fn resource(
        &mut self,
        name: &str,
        binding: u32,
        ty: Handle<Type>,
        space: AddressSpace,
    ) -> Handle<Expression> {
        let variable = global(self.module, name, binding, ty, space);
        self.builder
            .expression(Expression::GlobalVariable(variable))
    }
}

fn insert_type(module: &mut Module, inner: TypeInner) -> Handle<Type> {
    module
        .types
        .insert(Type { name: None, inner }, Span::UNDEFINED)
}

fn global(
    module: &mut Module,
    name: &str,
    binding: u32,
    ty: Handle<Type>,
    space: AddressSpace,
) -> Handle<GlobalVariable> {
    module.global_variables.append(
        GlobalVariable {
            name: Some(name.to_owned()),
            space,
            binding: Some(ResourceBinding { group: 0, binding }),
            ty,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    )
}
