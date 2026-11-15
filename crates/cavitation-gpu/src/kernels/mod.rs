//! The GPU kernels, one module per family, each with the CPU twin its tests check against.

mod elementwise;
mod gemm;

pub use elementwise::{Elementwise, ElementwiseCall, ElementwiseKernels};
pub use gemm::{Gemm, GemmCall, GemmDims};
