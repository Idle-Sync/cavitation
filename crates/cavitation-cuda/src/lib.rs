//! NVIDIA's CUDA for the cavitation neural network library: the driver, cuDNN, cuBLAS and
//! NVRTC, found and loaded when the program runs.
//!
//! This crate is the base of a second GPU engine, for NVIDIA cards, beside the wgpu one in
//! `cavitation-gpu`. cuDNN and cuBLAS are NVIDIA's own libraries for convolutions and
//! matrix products, and they work on memory that wgpu's buffers cannot share, so the engine
//! is built from here up and not as a kernel inside the other.
//!
//! Nothing is linked when the crate is built, and it has no dependencies: the libraries
//! are opened with the operating system's loader, so a program built with it starts on a
//! machine that has none of them, and [`Cuda::open`] says which are missing and what to
//! install.
//!
//! What is here:
//!
//! - [`Cuda`]: the driver, the first NVIDIA card with its primary context, and a cuDNN and
//!   a cuBLAS handle. [`Cuda::new`] returns `None` when CUDA cannot be used, and
//!   [`Cuda::open`] returns the reason as a [`CudaError`].
//! - [`CudaInfo`]: the card, each library's version and the files they were loaded from,
//!   for printing.
//! - [`DeviceBuffer`]: memory on the card, zeroed, written from and read into slices of a
//!   [`Scalar`] (`f32`, `u32` or `i32`), copied from another buffer on the card, and zeroed
//!   again.
//! - [`Module`], [`Kernel`] and [`Arg`]: kernels written in CUDA C++, compiled at run time
//!   by NVRTC and launched on the card. The sources are compiled without fused
//!   multiply-add, so their arithmetic is the CPU's for the same expressions.
//!
//! - [`Conv2d`] and [`ConvShape`]: the 2-D convolution through cuDNN (forward with a bias,
//!   input gradient, weight gradient, bias gradient), with deterministic algorithms that
//!   use no tensor cores, chosen once per shape ([`ConvAlgorithms`]). cuDNN publishes no
//!   error bound; each pass's distance from the exact result is measured in
//!   `tests/conv.rs`.
//!
//! - [`Linear`] and [`LinearShape`]: the fully connected layer's passes through cuBLAS in
//!   pedantic math (f32 throughout, no tensor cores): forward with a bias, input gradient,
//!   weight gradient, bias gradient; measured in `tests/linear.rs`.
//!
//! - [`Elementwise`] and [`ElementwiseOp`]: add, multiply, scale and the activations (ReLU,
//!   leaky ReLU, tanh, the sigmoid, GELU in both forms, ELU, SELU, CELU, SiLU, Mish,
//!   softplus, softsign, hardtanh, the hard sigmoid and hard swish, log-sigmoid), forward and
//!   backward, as this crate's own kernels, each the twin of its CPU code
//!   (`tests/elementwise.rs`).
//!
//! - [`Dropout`]: dropout and the neuron mask (`Tape::keep_where`) with a bit-packed `u32`
//!   mask, forward and backward, as this crate's own kernels, each the CPU's bits
//!   (`tests/dropout.rs`).
//! - [`RowSoftmax`] and [`RowSoftmaxOp`]: softmax and log-softmax over each row, forward and
//!   backward, as this crate's own kernels, one thread per row in the CPU's order: the
//!   softmax's backward pass the CPU's bits, the others within the WGSL kernels' stated
//!   bounds (`tests/softmax.rs`).
//! - [`Prelu`] and [`PreluOp`]: PReLU with one slope per column or one shared slope, as this
//!   crate's own kernels: forward and the input's gradient the CPU's bits, and the slopes'
//!   gradient in a fixed tree within a stated bound (`tests/prelu.rs`).
//!
//! - [`MaxPool2d`], [`AvgPool2d`] and [`PoolShape`]: 2-D max and average pooling, forward and
//!   backward, as this crate's own kernels, each the CPU's bits (`tests/pooling.rs`).
//!
//! - [`BatchNorm2d`], [`BatchStats`], [`RunningUpdate`] and [`ChannelDims`]: 2-D batch
//!   normalisation, forward from the batch's or the running statistics, backward and the
//!   running statistics' update, as this crate's own kernels: the sums in a fixed tree
//!   within a stated bound, everything else the CPU's bits (`tests/batch_norm.rs`).
//!
//! - [`SoftmaxCrossEntropy`] and [`MeanSquaredError`]: the training losses with
//!   `Reduction::Mean`, the loss and its gradient, as this crate's own kernels: the mean
//!   squared error's gradient the CPU's bits and its loss the fixed tree's, the
//!   cross-entropy within the WGSL kernel's stated bounds (`tests/loss.rs`).
//!
//! - [`OptimizerKernel`] and [`OptimizerRule`]: SGD (with momentum and weight decay), Adam
//!   and AdamW over a model's parameters in flat buffers, one launch per step, skipped when
//!   the step's finite flag is set, as this crate's own kernels: the bits of the kernels'
//!   `f32` expressions, within the WGSL kernels' stated bounds of the CPU optimisers'
//!   arithmetic (`tests/optimizer.rs`).
//!
//! - [`SumSquares`], [`FiniteFlag`] and [`DeadRelu`]: the health values a training step
//!   reads back, as this crate's own kernels: each gradient's sum of squares in the fixed
//!   tree, the step's finite flag the optimiser and the running statistics read, and the
//!   dead-ReLU bits, the flag and the bits exact (`tests/health.rs`).
//!
//! `cavitation-nn`'s `cuda` feature trains through this crate: its `CudaModel` and
//! `CudaTrainer`, and `fit` with `Device::Cuda`.
//!
//! # What must be installed
//!
//! - An NVIDIA card and NVIDIA's display driver, which brings the CUDA driver library.
//! - cuDNN 9.
//! - cuBLAS and NVRTC from the CUDA toolkit of the major version that cuDNN was built for
//!   (12 or 13).
//!
//! The libraries are looked for in the folders of `CAVITATION_CUDA_PATH`, `CUDNN_PATH` and
//! `CUDA_PATH` (each with its `bin`, `bin/x64`, `lib`, `lib64` and `lib/x64`), then where
//! the operating system's loader looks: the folders of `PATH` on Windows, and of
//! `LD_LIBRARY_PATH` and the linker's configuration on Linux. Neither the libraries loaded
//! here nor the ones they import are looked for in the folder the program was started in;
//! the libraries NVIDIA's code loads by name later and that are known (cuDNN's parts,
//! cuBLASLt, NVRTC's built-ins, `zlibwapi.dll`) are loaded here first from beside the ones
//! that need them. macOS has no CUDA.
//!
//! Two environment variables steer this, mainly for tests and benchmarks:
//!
//! - `CAVITATION_CUDA_PATH`: folders to look in first, separated as `PATH` is.
//! - `CAVITATION_CUDA_REQUIRED=1` turns "CUDA cannot be used" into a panic in
//!   [`Cuda::new`], so a test run that must use CUDA fails instead of skipping.
//!
//! # Examples
//!
//! Without an NVIDIA card or the libraries, [`Cuda::new`] returns `None` and the example
//! does nothing:
//!
//! ```
//! use cavitation_cuda::Cuda;
//!
//! if let Some(cuda) = Cuda::new() {
//!     println!("{}", cuda.info());
//! }
//! ```

mod batch_norm;
mod buffer;
mod conv;
mod device;
mod dropout;
mod elementwise;
mod error;
mod find;
mod health;
mod library;
mod linear;
mod loss;
mod module;
mod optimizer;
mod pooling;
mod prelu;
mod reduce;
mod softmax;
mod sys;

pub use batch_norm::{BatchNorm2d, BatchStats, RunningUpdate};
pub use buffer::{DeviceBuffer, Scalar};
pub use conv::{Conv2d, ConvAlgorithms, ConvCandidates, ConvShape, MAX_WORKSPACE};
pub use device::{Cuda, CudaInfo};
pub use dropout::Dropout;
pub use elementwise::{Elementwise, ElementwiseOp};
pub use error::CudaError;
pub use health::{DeadRelu, FiniteFlag, SumSquares};
pub use linear::{Linear, LinearShape};
pub use loss::{MeanSquaredError, SoftmaxCrossEntropy};
pub use module::{Arg, Kernel, MAX_KERNEL_ARGS, Module};
pub use optimizer::{OptimizerKernel, OptimizerRule};
pub use pooling::{AvgPool2d, MaxPool2d, PoolShape};
pub use prelu::{Prelu, PreluOp};
pub use reduce::ChannelDims;
pub use softmax::{RowSoftmax, RowSoftmaxOp};
