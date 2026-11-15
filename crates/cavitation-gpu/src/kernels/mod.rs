//! The GPU kernels, one module per family, each with the CPU twin its tests check against.

mod batch_norm;
mod gemm;
mod reduce;

pub use batch_norm::{BatchNorm2d, BatchNormBackward, BatchNormForward, BatchStats, RunningUpdate};
pub use gemm::{Gemm, GemmCall, GemmDims};
pub use reduce::{BiasAdd, BiasAddCall, ChannelDims, ChannelSums, ChannelSumsCall};
