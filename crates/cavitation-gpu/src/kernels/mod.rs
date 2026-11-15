//! The GPU kernels, one module per family, each with the CPU twin its tests check against.

mod gemm;

pub use gemm::{Gemm, GemmCall, GemmDims};
