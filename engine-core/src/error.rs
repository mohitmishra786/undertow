use crate::store::ExpertKey;

pub type Result<T> = std::result::Result<T, EngineError>;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("expert not found: layer {} expert {}", .0.layer, .0.expert)]
    ExpertNotFound(ExpertKey),

    #[error("tensor not found: {0}")]
    TensorNotFound(String),

    #[error("shape mismatch for {name}: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("invalid model config: {0}")]
    InvalidConfig(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}
