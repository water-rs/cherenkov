/// Errors loading or saving a scene.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SceneError {
    /// An independent paint transform cannot be inverted to sample its paint.
    #[error("paint transform must be finite and invertible")]
    PaintTransform,
    /// An I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// `scene.json` could not be parsed.
    #[error("scene JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// A referenced resource blob is missing.
    #[error("missing resource {0}")]
    MissingResource(crate::ResourceHash),
    /// An image's declared encoding cannot decode its bytes.
    #[error("invalid image encoding {0:?}")]
    InvalidImageEncoding(crate::ImageEncoding),
    /// The `features` set stored in `scene.json` does not match the features
    /// recomputed from the layer tree.
    #[error("stored features {declared:?} do not match recomputed {computed:?}")]
    FeatureMismatch {
        /// The feature set stored in the file.
        declared: Vec<crate::Feature>,
        /// The feature set recomputed from the layer tree.
        computed: Vec<crate::Feature>,
    },
}
