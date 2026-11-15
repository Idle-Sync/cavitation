//! GPU access through wgpu for the cavitation neural network library.
//!
//! This crate is the GPU side of the library, reached through the `gpu` feature of
//! `cavitation` (or of `cavitation-nn`), which is off by default: a default build has no
//! wgpu in it.
//!
//! What is here:
//!
//! - [`Gpu`]: one GPU adapter with its device and queue, chosen among the hardware adapters
//!   wgpu offers on Vulkan, Metal or DX12. Software renderers, CPU and virtual adapters,
//!   paravirtual devices and OpenGL are left out. [`Gpu::new`] picks the platform's default
//!   backend (Metal on macOS, Vulkan elsewhere, then DX12 on Windows), and
//!   [`Gpu::with_backend`] asks for one.
//! - [`GpuInfo`]: what was chosen, for printing.
//! - [`BufferPool`] and [`PooledBuffer`]: GPU buffers by size class, reused once handed
//!   back, with a count of the buffers created.
//! - [`PipelineCache`] and [`Shader`]: compute pipelines built once per shader, entry point
//!   and `override` values, with a count of the pipelines built.
//! - [`StepEncoder`], [`Step`] and [`StepOutput`]: one step's uploads, dispatches and
//!   readbacks recorded in one command encoder and submitted once, through staging and
//!   readback buffers made once, with GPU timestamps per span when profiling.
//! - [`kernels::Gemm`]: the tiled matrix product `C = A·B + β·C` with either operand
//!   transposed, within `2(k + 3) · u · Σ|terms|` of the exact product and exact on small
//!   integers.
//! - [`kernels::BiasAdd`] and [`kernels::ChannelSums`]: a bias added per channel, with the
//!   CPU's bits, and each channel's sum in a fixed tree, over `[outer][channels][inner]` tensors (an
//!   `M × N` matrix or an NCHW batch).
//! - [`kernels::BatchNorm2d`]: 2-D batch normalisation as `Tape::batch_norm2d` computes it,
//!   forward and backward, from the batch's statistics or running ones, and the running
//!   statistics' update, skipped when the step's finite flag is set.
//! - [`kernels::SoftmaxCrossEntropy`] and [`kernels::MeanSquaredError`]: the training losses
//!   as the tape's `Reduction::Mean` computes them, with their gradients.
//! - [`kernels::SumSquares`], [`kernels::FiniteFlag`] and [`kernels::DeadRelu`]: the health
//!   values a step reads back: each gradient's sum of squares, the finite flag (tested by
//!   bits) and the dead-ReLU bits.
//! - [`Backend`] and [`GpuError`].
//! - [`wgpu`] itself, re-exported, so its types match the version this crate uses.
//!
//! Two environment variables steer the choice, mainly for tests and benchmarks:
//!
//! - `CAVITATION_GPU_BACKEND` (`vulkan`, `dx12` or `metal`) restricts every [`Gpu`] to that
//!   backend: [`Gpu::new`] looks only there, and [`Gpu::with_backend`] for any other
//!   backend finds nothing.
//! - `CAVITATION_GPU_REQUIRED=1` turns "no GPU found" into a panic in [`Gpu::new`], and in
//!   [`Gpu::with_backend`] for a backend this platform tries, so a test run that must use
//!   the GPU fails instead of skipping.
//!
//! On DX12, shaders are compiled by DXC when `dxcompiler.dll` can be loaded and by the
//! older FXC otherwise, as wgpu decides; `WGPU_DX12_COMPILER=dxc` makes DXC required, and
//! `fxc` forces FXC. DXC builds a large shader in about a second where FXC can take tens of
//! seconds.
//!
//! # Examples
//!
//! Without a GPU, [`Gpu::new`] returns `None` and the example does nothing:
//!
//! ```
//! use cavitation_gpu::Gpu;
//!
//! if let Some(gpu) = Gpu::new() {
//!     println!("{}", gpu.info());
//! }
//! ```

mod buffer;
mod device;
pub mod kernels;
mod pipeline;
mod step;

pub use buffer::{BufferPool, PooledBuffer};
pub use device::{Backend, Gpu, GpuError, GpuInfo};
pub use pipeline::{PipelineCache, Shader};
pub use step::{MAX_WORKGROUPS_PER_DIMENSION, Step, StepEncoder, StepOutput, StepSizes};
/// The wgpu crate this one is built on, for the types [`Gpu::device`] and [`Gpu::queue`]
/// work with.
///
/// ```
/// use cavitation_gpu::{Gpu, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let encoder = gpu.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
///     gpu.queue().submit([encoder.finish()]);
/// }
/// ```
pub use wgpu;
