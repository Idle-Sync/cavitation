//! Training and inference on an NVIDIA card through CUDA, behind the `cuda` feature:
//! [`CudaModel`], a [`Sequential`](crate::Sequential) compiled for one batch size, with its
//! parameters, batch-norm buffers and neuron masks held on the card, which answers batches in
//! evaluation mode and measures data sets; [`CudaTrainer`], which trains it there a batch at a
//! time; and the run [`fit`](crate::fit) makes through the two for
//! [`Device::Cuda`](crate::Device).
//!
//! The layers are those the wgpu engine runs, from the same plan: the convolution's passes go
//! through cuDNN and `Linear`'s through cuBLAS, and batch norm, pooling, the activations,
//! dropout, the masks, the losses, the health values and the optimisers are
//! `cavitation-cuda`'s own kernels.

mod fit;
#[cfg(test)]
mod fit_tests;
mod forward;
#[cfg(test)]
mod health_tests;
mod model;
#[cfg(test)]
mod model_tests;
mod pass;
#[cfg(test)]
mod short_batch_tests;
#[cfg(test)]
mod test_util;
mod trainer;
#[cfg(test)]
mod trainer_tests;

pub(crate) use fit::{fit_on_cuda, trainer_for};
pub use model::CudaModel;
pub use trainer::{CudaOptimizerState, CudaStepProfile, CudaTrainer};
