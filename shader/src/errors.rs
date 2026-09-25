//! Errors.

/// A snippet violates the snippet contract.
#[derive(Debug, thiserror::Error)]
pub enum SnippetError {
    /// The WGSL source does not parse.
    #[error("snippet `{name}` ({variant}) does not parse:\n{message}")]
    Parse {
        /// The snippet name.
        name: String,
        /// The variant whose source failed.
        variant: String,
        /// naga's rendered diagnostic.
        message: String,
    },
    /// The module is not valid.
    #[error("snippet `{name}` ({variant}) is not valid:\n{message}")]
    Invalid {
        /// The snippet name.
        name: String,
        /// The variant whose source failed.
        variant: String,
        /// naga's rendered diagnostic.
        message: String,
    },
    /// The source breaks the ABI.
    #[error("snippet `{name}` ({variant}): {reason}")]
    Abi {
        /// The snippet name.
        name: String,
        /// The variant whose source failed.
        variant: String,
        /// What is wrong.
        reason: String,
    },
    /// The source uses a construct snippets may not use.
    #[error("snippet `{name}` ({variant}) uses {construct}, which snippets may not use")]
    Unsupported {
        /// The snippet name.
        name: String,
        /// The variant whose source failed.
        variant: String,
        /// The construct.
        construct: &'static str,
    },
    /// Two variants disagree on the ABI.
    #[error("snippet `{name}`: variant {variant} declares a different ABI than the f32 source")]
    VariantMismatch {
        /// The snippet name.
        name: String,
        /// The mismatching variant.
        variant: String,
    },
}

/// A chain cannot be composed.
#[derive(Debug, thiserror::Error)]
pub enum ComposeError {
    /// The chain has no stages.
    #[error("the chain is empty")]
    EmptyChain,
    /// A constant names a parameter the snippet does not declare.
    #[error("stage {stage} (`{snippet}`) has no parameter `{param}`")]
    UnknownParam {
        /// The stage index.
        stage: usize,
        /// The snippet name.
        snippet: String,
        /// The parameter name.
        param: String,
    },
    /// A constant has the wrong type.
    #[error("stage {stage} (`{snippet}`) parameter `{param}` is {expected:?}, not {found:?}")]
    ParamType {
        /// The stage index.
        stage: usize,
        /// The snippet name.
        snippet: String,
        /// The parameter name.
        param: String,
        /// The declared type.
        expected: crate::ParamType,
        /// The type of the given constant.
        found: crate::ParamType,
    },
    /// A parameter is given two constants.
    #[error("stage {stage} (`{snippet}`) parameter `{param}` is specialized twice")]
    DuplicateConstant {
        /// The stage index.
        stage: usize,
        /// The snippet name.
        snippet: String,
        /// The parameter name.
        param: String,
    },
    /// The composed module does not validate. This is a composer defect.
    #[error("the composed module is not valid: {0}")]
    Invalid(String),
}

/// A module cannot be emitted for a back end.
#[derive(Debug, thiserror::Error)]
pub enum EmitError {
    /// The module does not validate with the given capabilities.
    #[error("the module is not valid: {0}")]
    Invalid(String),
    /// The WGSL back end failed.
    #[error(transparent)]
    Wgsl(#[from] naga::back::wgsl::Error),
    /// The MSL back end failed.
    #[error(transparent)]
    Msl(#[from] naga::back::msl::Error),
    /// The SPIR-V back end failed.
    #[error(transparent)]
    Spirv(#[from] naga::back::spv::Error),
}
