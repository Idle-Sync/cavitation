//! Datasets for the cavitation examples, tests and benchmark. This crate is not published.
//!
//! - [`digits`]: the 1,797 8 by 8 digit images from scikit-learn, shuffled with
//!   [`Mulberry32`] (seed 42) into 1,437 training and 360 test rows.
//! - [`mnist`]: loads MNIST from its four uncompressed IDX files, which
//!   `scripts/data/get-mnist.sh` downloads.
//! - [`mulberry32`]: the mulberry32 generator, which shuffles the digits split.

pub mod digits;
pub mod mnist;
pub mod mulberry32;

pub use mulberry32::Mulberry32;
