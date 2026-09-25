//! Procedural macros for `filtrate`.
//!
//! This crate exposes `#[derive(Filter)]`, which generates a complete
//! single-stage filter from a struct attributed with `#[filter(...)]`: the
//! `filtrate_core::Filter` implementation, the kind trait (`ColorFilter` or
//! `SpatialFilter`), and, when declared, the `CpuKernel`. Filters with several
//! stages (separable blurs, for example) or with non-parameter fields are
//! written by hand.
//!
//! # Attributes
//!
//! `#[filter(...)]` takes a kind marker, the stage's shader, and the kind's
//! properties:
//!
//! - `color, shader = "<path>", linear = <bool>` declares a colour filter.
//!   `linear` is required: it is the filter's `ColorFilter::LINEAR`
//!   classification, which executors trust when pushing a filter down.
//!   `cpu = <path>` names a CPU kernel,
//!   `fn(&[f32; N], &WorkingSpace, &mut [[f32; 4]])`, and implements
//!   `CpuKernel` with it.
//! - `spatial, shader = "<path>"` declares a spatial filter, with either
//!   `footprint = <expr>` (a constant `f32`) or `footprint_fn = <path>`
//!   (`fn(&[f32; N]) -> f32`), and optionally `shape = sdf` or
//!   `shape = mask` when the snippet reads the clip shape.
//!
//! Both kinds accept `space = srgb` for a stage that operates in sRGB (the
//! default is the linear working space), and `constants = [<f32>, …]`.
//!
//! The shader path is resolved relative to the declaring crate's
//! `src/shaders/` directory and included with `include_str!`; nothing runs
//! at build time.
//!
//! # Parameters
//!
//! Tuple structs and named-field structs are both supported. Each field is
//! typed `T` or `[T; N]`, where `T` is a generic parameter bound to
//! `FilterParam` (or a concrete type implementing it). Fields flatten into the
//! parameter array in declaration order, and the snippet's `Params` struct
//! declares one `f32` member per flattened field, in the same order, followed
//! by one `f32` member per entry of `constants`, which the composer
//! specializes into the shader.
//!
//! # Example
//!
//! The example is `ignore`d because it cannot compile here: the shader must
//! live in the crate that writes the `#[derive]`, and a proc-macro crate has
//! no `src/shaders/`. The compiled version of this example lives in
//! `filtrate::filters`, next to the shaders it names.
//!
//! ```ignore
//! use filtrate::Filter;
//!
//! #[derive(Filter)]
//! #[filter(color, shader = "color/adjustment/brightness.wgsl", linear = true)]
//! pub struct Brightness<T>(pub T);
//!
//! #[derive(Filter)]
//! #[filter(spatial, shader = "image/convolution/gradient.wgsl", footprint = 1.0, constants = [1.0, 1.0, 2.0])]
//! pub struct Sobel;
//! ```

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    Attribute, Data, DataStruct, DeriveInput, Expr, ExprArray, ExprLit, GenericParam, Ident, Lit,
    Member, Path, Type, TypeArray, TypePath, meta::ParseNestedMeta, parse_macro_input, parse_quote,
};

/// Resolves the path of the crate providing the `Filter` machinery.
///
/// The derive is re-exported by `filtrate`, so a consumer may depend on
/// `filtrate-core` directly or only on `filtrate` (whose root re-exports every
/// name the generated code references). Emitting a hard-coded
/// `::filtrate_core` would break the latter, standard, arrangement.
fn core_path() -> syn::Result<TokenStream2> {
    for candidate in ["filtrate-core", "filtrate"] {
        match proc_macro_crate::crate_name(candidate) {
            Ok(proc_macro_crate::FoundCrate::Itself) => return Ok(quote! { crate }),
            Ok(proc_macro_crate::FoundCrate::Name(name)) => {
                let ident = Ident::new(&name, proc_macro2::Span::call_site());
                return Ok(quote! { ::#ident });
            }
            Err(_) => {}
        }
    }
    Err(syn::Error::new(
        proc_macro2::Span::call_site(),
        "#[derive(Filter)] requires a dependency on `filtrate` or `filtrate-core`",
    ))
}

/// Derives a complete single-stage filter from a `#[filter(...)]`
/// attribute; see the crate docs for the supported shapes.
#[proc_macro_derive(Filter, attributes(filter))]
pub fn derive_filter(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// How a spatial filter reports its footprint.
enum Footprint {
    /// A constant expression.
    Constant(Expr),
    /// A function of the parameters.
    Function(Path),
}

/// The kind-specific part of the attribute.
enum KindAttrs {
    Color {
        linear: bool,
        cpu: Option<Path>,
    },
    Spatial {
        footprint: Footprint,
        shape: Option<Ident>,
    },
}

struct FilterAttrs {
    kind: KindAttrs,
    shader_path: String,
    srgb: bool,
    constants: Vec<Expr>,
}

fn expand(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let attrs = parse_filter_attr(input)?;
    let fields: Vec<&syn::Field> = match &input.data {
        Data::Struct(DataStruct { fields, .. }) => fields.iter().collect(),
        _ => {
            return Err(syn::Error::new_spanned(
                &input.ident,
                "Filter derive requires a struct (enums and unions are not supported)",
            ));
        }
    };

    let generic_type_params: Vec<Ident> = input
        .generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(ty) => Some(ty.ident.clone()),
            _ => None,
        })
        .collect();

    let core = core_path()?;
    let layout = analyze_fields(&fields, &generic_type_params)?;
    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let mut where_clause = where_clause.cloned().unwrap_or_else(|| parse_quote!(where));
    for bound in &layout.bound_idents {
        where_clause
            .predicates
            .push(parse_quote!(#bound: #core::FilterParam));
    }

    let total_params = layout.total_params;
    let params_array = layout.build_params_array_tokens(&core);
    let visit_calls = layout.build_visit_signals_tokens();

    let shader_path = &attrs.shader_path;
    let source = quote! {
        ::core::include_str!(::core::concat!(
            ::core::env!("CARGO_MANIFEST_DIR"),
            "/src/shaders/",
            #shader_path
        ))
    };
    let name = ident.to_string();
    let bindings = (0..total_params)
        .map(|index| quote! { #core::ParamSource::Param(#index) })
        .chain(
            attrs
                .constants
                .iter()
                .map(|value| quote! { #core::ParamSource::Constant(&[#value]) }),
        );
    let space = if attrs.srgb {
        quote! { #core::OperatingSpace::Srgb }
    } else {
        quote! { #core::OperatingSpace::Working }
    };

    let (kind, stage, kind_impls) = match &attrs.kind {
        KindAttrs::Color { linear, cpu } => {
            let stage = quote! {
                const STAGE: #core::ColorStage = #core::ColorStage {
                    name: #name,
                    source: #source,
                    params: &[#(#bindings),*],
                    space: #space,
                };
                collector.color(#core::Placed::new(&STAGE));
            };
            let kernel = cpu.as_ref().map(|path| {
                quote! {
                    impl #impl_generics #core::CpuKernel for #ident #ty_generics #where_clause {
                        fn apply_cpu(
                            params: &[f32; #total_params],
                            space: &#core::WorkingSpace,
                            pixels: &mut [[f32; 4]],
                        ) {
                            #path(params, space, pixels);
                        }
                    }
                }
            });
            let impls = quote! {
                impl #impl_generics #core::ColorFilter for #ident #ty_generics #where_clause {
                    const LINEAR: bool = #linear;
                }
                #kernel
            };
            (quote! { #core::kind::Color }, stage, impls)
        }
        KindAttrs::Spatial { footprint, shape } => {
            let shape = match shape {
                None => quote! { ::core::option::Option::None },
                Some(shape) if shape == "sdf" => {
                    quote! { ::core::option::Option::Some(#core::ShapeInput::Sdf) }
                }
                Some(_) => quote! { ::core::option::Option::Some(#core::ShapeInput::Mask) },
            };
            let stage = quote! {
                const STAGE: #core::SpatialStage = #core::SpatialStage {
                    name: #name,
                    source: #source,
                    params: &[#(#bindings),*],
                    space: #space,
                    shape: #shape,
                    aux: &[],
                };
                collector.spatial(#core::Placed::new(&STAGE));
            };
            let footprint = match footprint {
                Footprint::Constant(value) => quote! {
                    fn footprint_of(_params: &[f32; #total_params]) -> f32 {
                        #value
                    }
                },
                Footprint::Function(path) => quote! {
                    fn footprint_of(params: &[f32; #total_params]) -> f32 {
                        #path(params)
                    }
                },
            };
            let impls = quote! {
                impl #impl_generics #core::SpatialFilter for #ident #ty_generics #where_clause {
                    #footprint
                }
            };
            (quote! { #core::kind::Spatial }, stage, impls)
        }
    };

    Ok(quote! {
        impl #impl_generics #core::Filter for #ident #ty_generics #where_clause {
            type Kind = #kind;
            type Params = [f32; #total_params];

            #[inline]
            fn params(&self) -> [f32; #total_params] {
                #params_array
            }

            fn collect_stages<__C: #core::StageCollector>(&self, collector: &mut __C) {
                #stage
            }

            fn visit_signals<__V: #core::SignalVisitor>(&self, visitor: &mut __V) {
                #visit_calls
            }
        }

        #kind_impls
    })
}

/// The attribute's arguments as they are parsed, before the kind decides
/// which ones are required.
#[derive(Default)]
struct RawAttrs {
    kind: Option<(Path, bool)>,
    shader: Option<String>,
    linear: Option<bool>,
    cpu: Option<Path>,
    footprint: Option<Footprint>,
    shape: Option<Ident>,
    srgb: Option<bool>,
    constants: Option<Vec<Expr>>,
}

impl RawAttrs {
    fn parse(&mut self, meta: &ParseNestedMeta<'_>) -> syn::Result<()> {
        let key = meta
            .path
            .get_ident()
            .map(ToString::to_string)
            .unwrap_or_default();
        match key.as_str() {
            "color" | "spatial" => {
                let spatial = key == "spatial";
                if let Some((_, existing)) = &self.kind {
                    return Err(meta.error(if *existing == spatial {
                        "duplicate filter kind marker"
                    } else {
                        "conflicting filter kind markers; declare exactly one of `color` or `spatial`"
                    }));
                }
                self.kind = Some((meta.path.clone(), spatial));
            }
            "shader" => {
                let value = string_literal(meta)?;
                set_once(meta, &mut self.shader, value)?;
            }
            "linear" => {
                let value: syn::LitBool = meta.value()?.parse()?;
                set_once(meta, &mut self.linear, value.value)?;
            }
            "cpu" => {
                let value: Path = meta.value()?.parse()?;
                set_once(meta, &mut self.cpu, value)?;
            }
            "footprint" => {
                let value: Expr = meta.value()?.parse()?;
                set_once(meta, &mut self.footprint, Footprint::Constant(value))?;
            }
            "footprint_fn" => {
                let value: Path = meta.value()?.parse()?;
                set_once(meta, &mut self.footprint, Footprint::Function(value))?;
            }
            "shape" => {
                let value: Ident = meta.value()?.parse()?;
                if value != "sdf" && value != "mask" {
                    return Err(syn::Error::new_spanned(
                        value,
                        "expected `shape = sdf` or `shape = mask`",
                    ));
                }
                set_once(meta, &mut self.shape, value)?;
            }
            "space" => {
                let value: Ident = meta.value()?.parse()?;
                let srgb = if value == "srgb" {
                    true
                } else if value == "working" {
                    false
                } else {
                    return Err(syn::Error::new_spanned(
                        value,
                        "expected `space = working` or `space = srgb`",
                    ));
                };
                set_once(meta, &mut self.srgb, srgb)?;
            }
            "constants" => {
                let value: ExprArray = meta.value()?.parse()?;
                set_once(meta, &mut self.constants, value.elems.into_iter().collect())?;
            }
            _ => {
                return Err(meta.error(
                    "unknown #[filter(...)] argument; expected `color`, `spatial`, `shader`, `linear`, `cpu`, `footprint`, `footprint_fn`, `shape`, `space` or `constants`",
                ));
            }
        }
        Ok(())
    }

    fn finish(self, attr: &Attribute) -> syn::Result<FilterAttrs> {
        let (marker, spatial) = self.kind.ok_or_else(|| {
            syn::Error::new_spanned(
                attr,
                "missing #[filter(color)] or #[filter(spatial)] marker",
            )
        })?;
        let shader_path = self.shader.ok_or_else(|| {
            syn::Error::new_spanned(attr, "missing #[filter(shader = \"...\")] path")
        })?;
        let misplaced = |name: &str, kind: &str| {
            syn::Error::new_spanned(&marker, format!("`{name}` only applies to {kind} filters"))
        };
        let kind = if spatial {
            if self.linear.is_some() {
                return Err(misplaced("linear", "colour"));
            }
            if self.cpu.is_some() {
                return Err(misplaced("cpu", "colour"));
            }
            KindAttrs::Spatial {
                footprint: self.footprint.ok_or_else(|| {
                    syn::Error::new_spanned(
                        attr,
                        "spatial filters declare `footprint = <f32>` or `footprint_fn = <path>`",
                    )
                })?,
                shape: self.shape,
            }
        } else {
            if self.footprint.is_some() {
                return Err(misplaced("footprint", "spatial"));
            }
            if self.shape.is_some() {
                return Err(misplaced("shape", "spatial"));
            }
            KindAttrs::Color {
                linear: self.linear.ok_or_else(|| {
                    syn::Error::new_spanned(
                        attr,
                        "colour filters declare `linear = true` or `linear = false`",
                    )
                })?,
                cpu: self.cpu,
            }
        };
        Ok(FilterAttrs {
            kind,
            shader_path,
            srgb: self.srgb.unwrap_or(false),
            constants: self.constants.unwrap_or_default(),
        })
    }
}

fn string_literal(meta: &ParseNestedMeta<'_>) -> syn::Result<String> {
    match meta.value()?.parse()? {
        Expr::Lit(ExprLit {
            lit: Lit::Str(value),
            ..
        }) => Ok(value.value()),
        _ => Err(meta.error("expected a string literal path")),
    }
}

fn set_once<T>(meta: &ParseNestedMeta<'_>, slot: &mut Option<T>, value: T) -> syn::Result<()> {
    if slot.replace(value).is_some() {
        return Err(meta.error("duplicate #[filter(...)] argument"));
    }
    Ok(())
}

fn parse_filter_attr(input: &DeriveInput) -> syn::Result<FilterAttrs> {
    let mut filter_attrs = input.attrs.iter().filter(|a| a.path().is_ident("filter"));
    let attr: &Attribute = filter_attrs.next().ok_or_else(|| {
        syn::Error::new_spanned(
            &input.ident,
            "Filter derive requires a `#[filter(...)]` attribute",
        )
    })?;
    if let Some(duplicate) = filter_attrs.next() {
        return Err(syn::Error::new_spanned(
            duplicate,
            "duplicate #[filter(...)] attribute; declare the whole filter in one attribute",
        ));
    }

    let mut raw = RawAttrs::default();
    attr.parse_nested_meta(|meta| raw.parse(&meta))?;
    raw.finish(attr)
}

struct FieldLayout {
    fields: Vec<FieldEntry>,
    total_params: usize,
    /// Generic type parameters that need a `FilterParam` where-bound.
    bound_idents: Vec<Ident>,
}

enum FieldEntry {
    Scalar { member: Member },
    Array { member: Member, len: usize },
}

fn element_ident(ty: &Type, message: &str) -> syn::Result<Ident> {
    match ty {
        Type::Path(TypePath {
            qself: None, path, ..
        }) => path
            .get_ident()
            .cloned()
            .ok_or_else(|| syn::Error::new_spanned(ty, message)),
        _ => Err(syn::Error::new_spanned(ty, message)),
    }
}

fn analyze_fields(
    fields: &[&syn::Field],
    generic_type_params: &[Ident],
) -> syn::Result<FieldLayout> {
    let mut layout = FieldLayout {
        fields: Vec::new(),
        total_params: 0,
        bound_idents: Vec::new(),
    };

    let record_element = |layout: &mut FieldLayout, ident: &Ident| {
        // Only generic parameters need an explicit where-bound; concrete
        // element types (e.g. `f32`) resolve `FilterParam` directly and
        // fail with a normal trait error if they don't implement it.
        if generic_type_params.contains(ident) && !layout.bound_idents.contains(ident) {
            layout.bound_idents.push(ident.clone());
        }
    };

    for (field_idx, field) in fields.iter().enumerate() {
        let member = field.ident.clone().map_or_else(
            || Member::Unnamed(syn::Index::from(field_idx)),
            Member::Named,
        );
        match &field.ty {
            Type::Array(TypeArray { elem, len, .. }) => {
                let ident = element_ident(
                    elem,
                    "Filter derive expects each array element type to be a single type ident",
                )?;
                let len_value = match len {
                    Expr::Lit(ExprLit {
                        lit: Lit::Int(int), ..
                    }) => int.base10_parse::<usize>()?,
                    _ => {
                        return Err(syn::Error::new_spanned(
                            len,
                            "Filter derive expects array lengths to be integer literals",
                        ));
                    }
                };
                record_element(&mut layout, &ident);
                layout.fields.push(FieldEntry::Array {
                    member,
                    len: len_value,
                });
                layout.total_params += len_value;
            }
            other => {
                let ident = element_ident(
                    other,
                    "Filter derive supports only fields of type `T` or `[T; N]`",
                )?;
                record_element(&mut layout, &ident);
                layout.fields.push(FieldEntry::Scalar { member });
                layout.total_params += 1;
            }
        }
    }

    Ok(layout)
}

impl FieldLayout {
    fn build_params_array_tokens(&self, core: &TokenStream2) -> TokenStream2 {
        let snapshots = self.fields.iter().flat_map(|entry| match entry {
            FieldEntry::Scalar { member } => {
                vec![quote! { #core::FilterParam::snapshot(&self.#member) }]
            }
            FieldEntry::Array { member, len } => (0..*len)
                .map(|i| {
                    let element = syn::Index::from(i);
                    quote! { #core::FilterParam::snapshot(&self.#member[#element]) }
                })
                .collect(),
        });
        quote! { [ #( #snapshots ),* ] }
    }

    fn build_visit_signals_tokens(&self) -> TokenStream2 {
        let mut current = 0usize;
        let calls = self.fields.iter().map(|entry| match entry {
            FieldEntry::Scalar { member } => {
                let param_idx = current;
                current += 1;
                quote! { visitor.visit(#param_idx, &self.#member); }
            }
            FieldEntry::Array { member, len } => {
                let base = current;
                current += *len;
                quote! {
                    for __i in 0..#len {
                        visitor.visit(#base + __i, &self.#member[__i]);
                    }
                }
            }
        });
        quote! { #( #calls )* }
    }
}
