//! Composition against the sequential application of the same snippets, and
//! emission for every back end.

mod support;

use std::collections::HashMap;

use cherenkov_shader::{
    ComposeError, ComposeOptions, Composition, Piece, Precision, SampleCount, Segment, SegmentArg,
    Snippet, SnippetError, SnippetKind, SnippetSource, Stage, Variant, compose, msl,
    naga::{self, valid::Capabilities},
    spirv, validate, wgsl,
};
use support::{Eval, Value, assert_close};

const BRIGHTNESS: &str = include_str!("snippets/brightness.wgsl");
const BRIGHTNESS_F16: &str = include_str!("snippets/brightness_f16.wgsl");
const BRIGHTNESS_SUBGROUPS: &str = include_str!("snippets/brightness_subgroups.wgsl");
const SATURATION: &str = include_str!("snippets/saturation.wgsl");
const CONTRAST: &str = include_str!("snippets/contrast.wgsl");
const INVERT: &str = include_str!("snippets/invert.wgsl");
const BLUR: &str = include_str!("snippets/blur.wgsl");
const MISSING_COLOR: &str = include_str!("snippets/missing_color.wgsl");

/// Luma coefficients of linear Display P3 (the Y row of its RGB to XYZ matrix).
const LUMA: [f32; 3] = [0.228_974_6, 0.691_738_5, 0.079_286_9];
const PIVOT: [f32; 3] = [0.18, 0.18, 0.18];

fn parse(name: &str, kind: SnippetKind, source: &str) -> Snippet {
    Snippet::parse(&SnippetSource::new(name, kind, source)).expect("the test snippet is valid")
}

fn module(source: &str) -> naga::Module {
    naga::front::wgsl::parse_str(source).expect("the test snippet parses")
}

fn space() -> Value {
    Value::Struct(vec![Value::vec(&LUMA)])
}

/// The params block for a segment, from per-(stage, parameter) values.
fn block(segment: &Segment, values: &HashMap<(usize, &str), Value>) -> Value {
    Value::Struct(
        segment
            .uniform
            .members
            .iter()
            .map(|member| values[&(member.stage, member.param.as_str())].clone())
            .collect(),
    )
}

fn only_colour(composition: &Composition) -> &Segment {
    match composition.pieces() {
        [Piece::Color(segment)] => segment,
        other => panic!("expected one colour piece, got {other:?}"),
    }
}

fn run(composition: &Composition, segment: &Segment, args: Vec<Value>) -> Value {
    let eval = Eval::new(composition.module());
    eval.call(eval.function(&segment.function), args)
}

fn apply(source: &str, args: Vec<Value>) -> Value {
    let module = module(source);
    let eval = Eval::new(&module);
    eval.call(eval.function("apply"), args)
}

#[test]
fn colour_chain_matches_sequential_application() {
    let snippets = [
        parse("brightness", SnippetKind::Color, BRIGHTNESS),
        parse("saturation", SnippetKind::Color, SATURATION),
        parse("contrast", SnippetKind::Color, CONTRAST),
        parse("invert", SnippetKind::Color, INVERT),
    ];
    let stages = [
        Stage::new(&snippets[0]),
        Stage::new(&snippets[1]),
        Stage::new(&snippets[2]).constant("pivot", PIVOT),
        Stage::new(&snippets[3]),
    ];
    let composition = compose(&stages, ComposeOptions::default()).expect("the chain composes");
    let segment = only_colour(&composition);
    assert_eq!(
        segment.args,
        [
            SegmentArg::Color,
            SegmentArg::Params,
            SegmentArg::WorkingSpace
        ]
    );

    let values = HashMap::from([
        ((0, "amount"), Value::Float(0.1)),
        ((1, "amount"), Value::Float(0.8)),
        ((2, "amount"), Value::Float(1.3)),
    ]);
    for colour in [
        [0.2, 0.4, 0.6, 1.0],
        [0.05, 0.3, 0.1, 0.5],
        [0.9, 0.9, 0.0, 0.8],
    ] {
        let composed = run(
            &composition,
            segment,
            vec![Value::vec(&colour), block(segment, &values), space()],
        );

        let mut expected = Value::vec(&colour);
        expected = apply(
            BRIGHTNESS,
            vec![expected, Value::Struct(vec![Value::Float(0.1)])],
        );
        expected = apply(
            SATURATION,
            vec![expected, Value::Struct(vec![Value::Float(0.8)]), space()],
        );
        expected = apply(
            CONTRAST,
            vec![
                expected,
                Value::Struct(vec![Value::Float(1.3), Value::vec(&PIVOT)]),
            ],
        );
        expected = apply(INVERT, vec![expected]);

        assert_close(&composed, &expected, 1e-6);
    }
}

#[test]
fn constant_parameters_leave_the_uniform_block() {
    let brightness = parse("brightness", SnippetKind::Color, BRIGHTNESS);
    let contrast = parse("contrast", SnippetKind::Color, CONTRAST);

    let dynamic = compose(
        &[Stage::new(&brightness), Stage::new(&contrast)],
        ComposeOptions::default(),
    )
    .expect("the chain composes");
    let dynamic_segment = only_colour(&dynamic);
    let names: Vec<_> = dynamic_segment
        .uniform
        .members
        .iter()
        .map(|member| (member.stage, member.param.as_str(), member.offset))
        .collect();
    assert_eq!(
        names,
        [(0, "amount", 0), (1, "amount", 4), (1, "pivot", 16)]
    );
    assert_eq!(dynamic_segment.uniform.size, 32);

    let specialized = compose(
        &[
            Stage::new(&brightness).constant("amount", 0.25),
            Stage::new(&contrast).constant("pivot", PIVOT),
        ],
        ComposeOptions::default(),
    )
    .expect("the chain composes");
    let specialized_segment = only_colour(&specialized);
    let names: Vec<_> = specialized_segment
        .uniform
        .members
        .iter()
        .map(|member| (member.stage, member.param.as_str()))
        .collect();
    assert_eq!(names, [(1, "amount")]);
    assert_eq!(specialized_segment.uniform.size, 16);

    // Specializing changes where a value comes from, not the result.
    let colour = Value::vec(&[0.3, 0.2, 0.1, 1.0]);
    let from_block = run(
        &dynamic,
        dynamic_segment,
        vec![
            colour.clone(),
            block(
                dynamic_segment,
                &HashMap::from([
                    ((0, "amount"), Value::Float(0.25)),
                    ((1, "amount"), Value::Float(1.5)),
                    ((1, "pivot"), Value::vec(&PIVOT)),
                ]),
            ),
        ],
    );
    let from_constants = run(
        &specialized,
        specialized_segment,
        vec![
            colour,
            block(
                specialized_segment,
                &HashMap::from([((1, "amount"), Value::Float(1.5))]),
            ),
        ],
    );
    assert_close(&from_constants, &from_block, 1e-6);
}

#[test]
fn colour_prefix_folds_into_every_sample() {
    let brightness = parse("brightness", SnippetKind::Color, BRIGHTNESS);
    let blur = parse("blur", SnippetKind::Spatial, BLUR);
    let composition = compose(
        &[Stage::new(&brightness), Stage::new(&blur)],
        ComposeOptions::default(),
    )
    .expect("the chain composes");

    let [Piece::Color(_), Piece::Spatial { plain, folded }] = composition.pieces() else {
        panic!("expected a colour piece then a spatial piece");
    };
    let folded = folded.as_ref().expect("the blur samples only in `apply`");
    assert_eq!(folded.cost.samples, SampleCount::Static(3));
    assert!(folded.cost.prefix_ops > 0);
    assert_eq!(folded.segment.stages, 0..2);
    assert_eq!(
        folded.segment.args,
        [
            SegmentArg::Input,
            SegmentArg::InputSampler,
            SegmentArg::Uv,
            SegmentArg::Params
        ]
    );

    let amount = 0.2;
    let step = [0.1, 0.0];
    let source = |uv: [f32; 2]| [uv[0], uv[1], 0.5, 0.75];
    let brightened = move |uv: [f32; 2]| {
        let [r, g, b, a] = source(uv);
        [r + amount * a, g + amount * a, b + amount * a, a]
    };
    let values = HashMap::from([
        ((0, "amount"), Value::Float(amount)),
        ((1, "step"), Value::vec(&step)),
    ]);

    for uv in [[0.5, 0.5], [0.2, 0.7]] {
        let mut folded_eval = Eval::new(composition.module());
        let texture = folded_eval.texture(Box::new(source));
        let from_folded = folded_eval.call(
            folded_eval.function(&folded.segment.function),
            vec![
                texture,
                Value::Sampler,
                Value::vec(&uv),
                block(&folded.segment, &values),
            ],
        );

        let mut plain_eval = Eval::new(composition.module());
        let texture = plain_eval.texture(Box::new(brightened));
        let from_plain = plain_eval.call(
            plain_eval.function(&plain.function),
            vec![
                texture,
                Value::Sampler,
                Value::vec(&uv),
                block(plain, &values),
            ],
        );

        assert_close(&from_folded, &from_plain, 1e-6);
    }
}

#[test]
fn every_composed_module_emits_for_every_back_end() {
    let brightness = Snippet::parse(
        &SnippetSource::new("brightness", SnippetKind::Color, BRIGHTNESS).variant(
            Variant {
                precision: Precision::F16,
                subgroups: false,
            },
            BRIGHTNESS_F16,
        ),
    )
    .expect("both variants are valid");
    let saturation = parse("saturation", SnippetKind::Color, SATURATION);
    let blur = parse("blur", SnippetKind::Spatial, BLUR);
    let invert = parse("invert", SnippetKind::Color, INVERT);
    let composition = compose(
        &[
            Stage::new(&brightness),
            Stage::new(&saturation),
            Stage::new(&blur),
            Stage::new(&invert),
        ],
        ComposeOptions {
            precision: Precision::F16,
            subgroups: false,
        },
    )
    .expect("the chain composes");

    let [
        Piece::Color(first),
        Piece::Spatial {
            folded: Some(folded),
            ..
        },
        Piece::Color(_),
    ] = composition.pieces()
    else {
        panic!("unexpected pieces {:?}", composition.pieces());
    };
    assert_eq!(first.variants[0].precision, Precision::F16);
    assert_eq!(first.variants[1].precision, Precision::F32);
    assert_eq!(folded.segment.stages, 0..3);
    assert!(
        composition
            .capabilities()
            .contains(Capabilities::SHADER_FLOAT16)
    );

    let info = validate(composition.module(), composition.capabilities()).expect("valid");
    let source = wgsl(composition.module(), &info).expect("WGSL emits");
    let reparsed = naga::front::wgsl::parse_str(&source).expect("emitted WGSL parses");
    validate(&reparsed, composition.capabilities()).expect("emitted WGSL validates");

    let metal = msl(composition.module(), &info, (2, 4)).expect("MSL emits");
    assert!(metal.contains(&folded.segment.function));

    let words = spirv(composition.module(), &info, (1, 3)).expect("SPIR-V emits");
    assert_eq!(words.first(), Some(&0x0723_0203), "SPIR-V magic number");
}

#[test]
fn subgroup_variants_are_used_where_declared() {
    let brightness = Snippet::parse(
        &SnippetSource::new("brightness", SnippetKind::Color, BRIGHTNESS).variant(
            Variant {
                precision: Precision::F32,
                subgroups: true,
            },
            BRIGHTNESS_SUBGROUPS,
        ),
    )
    .expect("both variants are valid");
    let invert = parse("invert", SnippetKind::Color, INVERT);
    let stages = [Stage::new(&brightness), Stage::new(&invert)];

    let with = compose(
        &stages,
        ComposeOptions {
            precision: Precision::F32,
            subgroups: true,
        },
    )
    .expect("the chain composes");
    assert_eq!(
        only_colour(&with).variants,
        [
            Variant {
                precision: Precision::F32,
                subgroups: true
            },
            Variant::BASE
        ]
    );
    assert!(with.capabilities().contains(Capabilities::SUBGROUP));

    let without = compose(&stages, ComposeOptions::default()).expect("the chain composes");
    assert_eq!(
        only_colour(&without).variants,
        [Variant::BASE, Variant::BASE]
    );
    assert!(!without.capabilities().contains(Capabilities::SUBGROUP));
}

#[test]
fn contract_violations_are_reported() {
    let missing = Snippet::parse(&SnippetSource::new(
        "missing",
        SnippetKind::Color,
        MISSING_COLOR,
    ));
    assert!(
        matches!(missing, Err(SnippetError::Abi { .. })),
        "{missing:?}"
    );

    let brightness = parse("brightness", SnippetKind::Color, BRIGHTNESS);
    let unknown = compose(
        &[Stage::new(&brightness).constant("radius", 1.0)],
        ComposeOptions::default(),
    );
    assert!(
        matches!(unknown, Err(ComposeError::UnknownParam { .. })),
        "{unknown:?}"
    );

    let mistyped = compose(
        &[Stage::new(&brightness).constant("amount", [1.0, 2.0])],
        ComposeOptions::default(),
    );
    assert!(
        matches!(mistyped, Err(ComposeError::ParamType { .. })),
        "{mistyped:?}"
    );

    assert!(matches!(
        compose(&[], ComposeOptions::default()),
        Err(ComposeError::EmptyChain)
    ));
}
