//! Training and inference on the GPU, behind the `gpu` feature: [`GpuModel`], a
//! [`Sequential`](crate::Sequential) compiled for one batch size, with its parameters,
//! batch-norm buffers and neuron masks held on the GPU, which answers batches in evaluation
//! mode and measures data sets; [`GpuTrainer`], which trains it there a batch at a time; and
//! the run [`fit`](crate::fit) makes through the two for [`Device::Gpu`](crate::Device).
//!
//! The GPU runs a fixed list of steps built when the model is compiled, not a tape: the
//! layer kinds come from [`Layer::inference`](crate::Layer::inference) and the crate's
//! sealed layer hooks, and the parameter ids from [`Layer::parameters`](crate::Layer::parameters)
//! and [`Layer::buffers`](crate::Layer::buffers).

mod fit;
#[cfg(test)]
mod fit_tests;
mod forward;
#[cfg(test)]
mod forward_tests;
#[cfg(test)]
mod health_tests;
#[cfg(test)]
mod memory_tests;
mod model;
#[cfg(test)]
mod profile_tests;
#[cfg(test)]
mod short_batch_tests;
mod trainer;
#[cfg(test)]
mod trainer_tests;

pub(crate) use fit::fit_on_gpu;
pub use model::GpuModel;
pub use trainer::{GpuOptimizerState, GpuStepProfile, GpuTrainer};
