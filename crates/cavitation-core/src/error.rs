//! `Error`: the failures a caller can recover from.

use std::fmt;
use std::path::PathBuf;

/// A failure the caller can recover from: bad data, bad files, unsupported requests.
///
/// Shape errors inside recorded operations are programming errors and panic instead.
///
/// # Examples
///
/// ```
/// use cavitation_core::Error;
/// let e = Error::MissingKey("layer0.weight".to_string());
/// assert_eq!(e.to_string(), "missing key `layer0.weight`");
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A shape did not match what was required.
    ShapeMismatch {
        /// The dimensions that were required.
        expected: Vec<usize>,
        /// The dimensions that were given.
        found: Vec<usize>,
    },
    /// Reading or writing a file failed.
    Io {
        /// The file involved.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// A file was not in the expected format.
    Format(String),
    /// A required key was absent from a weight file.
    MissingKey(String),
    /// A weight file held a key the model does not have.
    UnexpectedKey(String),
    /// A stored tensor had an unsupported element type.
    Dtype {
        /// The key of the tensor.
        key: String,
        /// The element type that was found.
        found: String,
    },
    /// A value that must be finite was NaN or infinite.
    NonFinite {
        /// The training step at which it appeared.
        step: u64,
        /// Where it appeared, for example a parameter name.
        location: String,
    },
    /// The request is valid but not supported.
    Unsupported(String),
}

impl fmt::Display for Error {
    /// Writes a one-line description naming the offending values.
    ///
    /// ```
    /// let e = cavitation_core::Error::Unsupported("f16".to_string());
    /// assert_eq!(e.to_string(), "unsupported: f16");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ShapeMismatch { expected, found } => {
                write!(f, "shape mismatch: expected {expected:?}, found {found:?}")
            }
            Error::Io { path, source } => write!(f, "I/O error on {}: {source}", path.display()),
            Error::Format(msg) => write!(f, "bad format: {msg}"),
            Error::MissingKey(key) => write!(f, "missing key `{key}`"),
            Error::UnexpectedKey(key) => write!(f, "unexpected key `{key}`"),
            Error::Dtype { key, found } => {
                write!(f, "key `{key}` has unsupported element type {found}")
            }
            Error::NonFinite { step, location } => {
                write!(f, "non-finite value at step {step} in {location}")
            }
            Error::Unsupported(msg) => write!(f, "unsupported: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    /// The underlying I/O error for [`Error::Io`], `None` for every other variant.
    ///
    /// ```
    /// use std::error::Error as _;
    /// let e = cavitation_core::Error::Format("x".to_string());
    /// assert!(e.source().is_none());
    /// ```
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
