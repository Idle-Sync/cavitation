//! The GPU kernels, one module per family, each with the CPU twin its tests check against.

mod gemm;
mod reduce;

pub use gemm::{Gemm, GemmCall, GemmDims};
pub use reduce::{BiasAdd, BiasAddCall, ChannelDims, ChannelSums, ChannelSumsCall};
