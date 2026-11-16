//! The GPU kernels, one module per family, each checked against its CPU twin in
//! `cavitation_core::kernels`.

mod batch_norm;
mod bindings;
mod conv;
mod elementwise;
mod gemm;
mod health;
mod loss;
mod optimizer;
mod pooling;
mod reduce;

pub use batch_norm::{BatchNorm2d, BatchNormBackward, BatchNormForward, BatchStats, RunningUpdate};
pub use conv::{
    ConvCall, ConvDims, ConvForward, ConvInputGrad, ConvWeightGrad, ConvWeightGradCall,
    WEIGHT_GRAD_BLOCK,
};
pub use elementwise::{Elementwise, ElementwiseCall, ElementwiseKernels};
pub use gemm::{Gemm, GemmCall, GemmDims};
pub use health::{DeadRelu, DeadReluCall, FiniteFlag, FiniteFlagCall, SumSquares, SumSquaresCall};
pub use loss::{CrossEntropyCall, MeanSquaredError, MseCall, SoftmaxCrossEntropy};
pub use optimizer::{OptimizerKernel, OptimizerRule};
pub use pooling::{AvgPool2d, MaxPool2d};
pub use reduce::{BiasAdd, BiasAddCall, ChannelDims, ChannelSums, ChannelSumsCall};

use crate::{PooledBuffer, StepEncoder, wgpu};

/// A counted bind group for `pipeline`'s group 0 with `buffers` at bindings 0, 1, ….
pub(crate) fn bind_buffers(
    steps: &mut StepEncoder,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&PooledBuffer],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.binding(),
        })
        .collect();
    steps.bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}
