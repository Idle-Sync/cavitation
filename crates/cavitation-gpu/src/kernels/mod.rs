//! The GPU kernels, one module per family, each checked against its CPU twin in
//! `cavitation_core::kernels`.

mod pooling;

pub use pooling::{AvgPool2d, MaxPool2d};
