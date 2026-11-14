//! `Error`: the failures a caller can recover from.

use std::fmt;
use std::path::PathBuf;

/// The most keys of one kind that a [`KeyErrors`] message lists; the rest are counted.
const LISTED: usize = 10;

/// A failure the caller can recover from: bad data, bad files, unsupported requests.
///
/// Shape errors inside recorded operations are programming errors and panic instead.
///
/// # Examples
///
/// ```
/// use cavitation_core::Error;
/// let e = Error::Format("not a safetensors file".to_string());
/// assert_eq!(e.to_string(), "bad format: not a safetensors file");
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
    /// A weight file does not fit the model: the keys that are missing, unexpected, of the
    /// wrong shape or of the wrong element type, all listed together.
    Keys(Box<KeyErrors>),
    /// A value that must be finite was NaN or infinite.
    NonFinite {
        /// The training step at which it appeared.
        step: u64,
        /// Where it appeared, for example a parameter name.
        location: String,
        /// What the health checks found out about the cause, when they looked: the first
        /// non-finite input value, or that the inputs were finite. `None` when they did not
        /// look, as for a gradient.
        detail: Option<String>,
    },
    /// Training data that does not suit the model or the loss, for example class labels
    /// for a loss that needs target values, or a label the model has no output for.
    InvalidData(String),
    /// The request is valid but not supported.
    Unsupported(String),
    /// A network operation failed, for example listening on an address already in use.
    Net {
        /// The address involved, as `ip:port`.
        address: String,
        /// The underlying error.
        source: std::io::Error,
    },
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
            Error::Keys(keys) => keys.fmt(f),
            Error::NonFinite {
                step,
                location,
                detail,
            } => {
                write!(f, "non-finite value at step {step} in {location}")?;
                match detail {
                    Some(detail) => write!(f, ": {detail}"),
                    None => Ok(()),
                }
            }
            Error::InvalidData(msg) => write!(f, "invalid data: {msg}"),
            Error::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Error::Net { address, source } => write!(f, "network error on {address}: {source}"),
        }
    }
}

impl std::error::Error for Error {
    /// The underlying I/O error for [`Error::Io`] and [`Error::Net`], `None` for every other
    /// variant.
    ///
    /// ```
    /// use std::error::Error as _;
    /// let e = cavitation_core::Error::Format("x".to_string());
    /// assert!(e.source().is_none());
    /// ```
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } | Error::Net { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Everything wrong between a weight file and the model it was loaded into, found in one
/// pass, so a single error shows all of it. Carried by [`Error::Keys`].
///
/// The keys are the ones the model uses: when a load renames the file's keys first, the
/// renamed keys are listed.
///
/// # Examples
///
/// ```
/// use cavitation_core::{Error, KeyErrors};
/// let keys = KeyErrors {
///     missing: vec!["0.weight".to_string()],
///     unexpected: vec!["fc1.weight".to_string()],
///     wrong_shape: vec![],
///     wrong_dtype: vec![],
///     file_keys: vec!["fc1.weight".to_string()],
/// };
/// let text = Error::Keys(Box::new(keys)).to_string();
/// assert!(text.contains("missing keys (1): 0.weight"));
/// assert!(text.contains("the file holds: fc1.weight"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KeyErrors {
    /// Keys the model needs that the file lacks, in the model's parameter order.
    pub missing: Vec<String>,
    /// Keys in the file that the model lacks, sorted. Empty when the load ignores them.
    pub unexpected: Vec<String>,
    /// Keys whose stored shape differs from the parameter's, as `(key, expected, found)`.
    pub wrong_shape: Vec<(String, Vec<usize>, Vec<usize>)>,
    /// Keys stored with an element type a load cannot read (anything but `F32`, `F16`,
    /// `BF16` and `F64`), as `(key, found type)`.
    pub wrong_dtype: Vec<(String, String)>,
    /// When no key the model needs was found, the first five keys of the file (after any
    /// renaming), sorted, to show what it holds; empty otherwise.
    pub file_keys: Vec<String>,
}

/// Writes `title (n): a, b, c` with at most [`LISTED`] items and `and N more` for the rest.
fn write_list(f: &mut fmt::Formatter<'_>, title: &str, items: &[String]) -> fmt::Result {
    write!(
        f,
        "\n  {title} ({}): {}",
        items.len(),
        items[..items.len().min(LISTED)].join(", ")
    )?;
    if items.len() > LISTED {
        write!(f, ", and {} more", items.len() - LISTED)?;
    }
    Ok(())
}

impl fmt::Display for KeyErrors {
    /// Writes one line per kind of problem, listing up to ten keys of each and counting the
    /// rest, the file's own first keys when no key the model needs was found, and a hint
    /// when a shape is the expected one reversed.
    ///
    /// ```
    /// use cavitation_core::KeyErrors;
    /// let keys = KeyErrors {
    ///     wrong_shape: vec![("0.weight".to_string(), vec![256, 64], vec![64, 256])],
    ///     ..KeyErrors::default()
    /// };
    /// let text = keys.to_string();
    /// assert!(text.contains("key `0.weight` has shape [64, 256], expected [256, 64]"));
    /// assert!(text.contains("this looks transposed"));
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the weight file does not match the model:")?;
        if !self.missing.is_empty() {
            write_list(f, "missing keys", &self.missing)?;
        }
        if !self.unexpected.is_empty() {
            write_list(f, "unexpected keys", &self.unexpected)?;
        }
        let shapes: Vec<String> = self
            .wrong_shape
            .iter()
            .map(|(key, expected, found)| {
                format!("key `{key}` has shape {found:?}, expected {expected:?}")
            })
            .collect();
        if !shapes.is_empty() {
            write_list(f, "wrong shapes", &shapes)?;
        }
        let types: Vec<String> = self
            .wrong_dtype
            .iter()
            .map(|(key, found)| {
                format!("key `{key}` has element type {found}, expected F32, F16, BF16 or F64")
            })
            .collect();
        if !types.is_empty() {
            write_list(f, "wrong element types", &types)?;
        }
        let transposed = self.wrong_shape.iter().any(|(_, expected, found)| {
            expected.len() == 2
                && expected.iter().rev().eq(found.iter())
                && expected[0] != expected[1]
        });
        if transposed {
            write!(
                f,
                "\n  this looks transposed: PyTorch and Cavitation store Linear weights as [outputs, inputs]"
            )?;
        }
        if !self.file_keys.is_empty() {
            write!(f, "\n  the file holds: {}", self.file_keys.join(", "))?;
        }
        Ok(())
    }
}

/// Tests for the `Error` variants that carry an underlying error.
#[cfg(test)]
mod tests {
    use super::Error;
    use std::error::Error as _;

    /// `Error::Net` names the address in its message and gives the I/O error as its source.
    #[test]
    fn net_error_names_address_and_has_source() {
        let e = Error::Net {
            address: "127.0.0.1:7878".to_string(),
            source: std::io::Error::new(std::io::ErrorKind::AddrInUse, "address in use"),
        };
        assert_eq!(
            e.to_string(),
            "network error on 127.0.0.1:7878: address in use"
        );
        let source = e.source().expect("a source");
        assert_eq!(source.to_string(), "address in use");
    }
}
