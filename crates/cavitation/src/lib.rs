//! A fast, glass-box neural network library for the CPU.
//!
//! This crate is the one to depend on. It re-exports the tensor and autograd
//! core and the layers, optimisers and training code.
//!
//! ```
//! println!("cavitation {}", cavitation::VERSION);
//! assert!(!cavitation::VERSION.is_empty());
//! ```

/// The version of this crate, as set in `Cargo.toml`.
///
/// ```
/// assert_eq!(cavitation::VERSION, env!("CARGO_PKG_VERSION"));
/// ```
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
