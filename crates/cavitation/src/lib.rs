//! A fast, glass-box neural network library for the CPU.
//!
//! This is the main crate of the cavitation workspace, the one to depend on. It
//! builds on the tensor and autograd core and the layers, optimisers and training
//! crates.
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
