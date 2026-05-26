//! Error types surfaced by rigging traits.

/// Errors returned by [`QueryFiltered::query_filtered`](crate::QueryFiltered::query_filtered).
///
/// Stability: Stable. `#[non_exhaustive]` so variants can be added in
/// minor releases.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum QueryError {
    /// Query embedding had a dimension different from the vault's.
    #[error("embedding dim mismatch: expected {expected}, got {got}")]
    EmbeddingDimMismatch {
        /// Dimension the vault was built with.
        expected: u32,
        /// Dimension of the query embedding.
        got: u32,
    },
    /// Underlying index is corrupted in a way the search cannot recover from.
    #[error("index corrupted: {0}")]
    IndexCorrupted(String),
    /// I/O failure during search.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors returned by [`open_readonly`](crate::open_readonly).
///
/// Stability: Stable. `#[non_exhaustive]`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum OpenError {
    /// No vault at the given path.
    #[error("vault not found")]
    NotFound,
    /// Permission denied opening one of the vault's files.
    #[error("permission denied")]
    PermissionDenied,
    /// On-disk format error from skeg-hull.
    #[error("format error: {0}")]
    FormatError(#[from] skeg_hull::Error),
    /// Generic I/O failure.
    #[error("I/O error: {0}")]
    Io(std::io::Error),
}

impl From<std::io::Error> for OpenError {
    fn from(e: std::io::Error) -> Self {
        use std::io::ErrorKind::*;
        match e.kind() {
            NotFound => Self::NotFound,
            PermissionDenied => Self::PermissionDenied,
            _ => Self::Io(e),
        }
    }
}
