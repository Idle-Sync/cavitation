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
//! - [`kernels::ElementwiseKernels`]: add, multiply, scale, ReLU, leaky ReLU, tanh, the
//!   sigmoid, GELU (both forms) forward and backward, and dropout with a bit-packed mask.
//! - [`kernels::MaxPool2d`] and [`kernels::AvgPool2d`]: max and average pooling, forward
//!   and backward, max pooling bit for bit with the CPU's (winners, ties and NaNs) and
//!   average pooling within a stated rounding bound.
//! - [`kernels::OptimizerKernel`] and [`kernels::OptimizerRule`]: SGD (with momentum and
//!   weight decay), Adam and AdamW over a model's parameters in one dispatch, with
//!   per-parameter update counts, skipped when the step's finite flag is set.
//! - [`kernels::Gemm`]: the tiled matrix product `C = A·B + β·C` with either operand
//!   transposed, within `2(k + 3) · u · Σ|terms|` of the exact product and exact on small
//!   integers.
//! - [`kernels::ConvForward`]: the 2-D convolution forward pass as an implicit GEMM, with
//!   stride, padding, dilation and a bias, within the GEMM's bound of the CPU reference.
//! - [`kernels::ConvInputGrad`]: the convolution's input gradient, gathered as an implicit
//!   GEMM through the same core, within the GEMM's bound of the CPU reference.
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
//! # Rounding
//!
//! The kernels' error bounds, in units of `u = 2⁻²⁴`, take each f32 operation WGSL calls
//! correctly rounded to be rounded to nearest, so within `u` of its exact result, and the
//! kernels stated to match the CPU bit for bit take ties to even as well, as the CPU
//! rounds. WGSL leaves the rounding mode open: a correctly rounded result may be either
//! float beside the exact one, up to `2u` off. The test suite checks round to nearest, ties
//! to even, on its own (`tests/rounding.rs`). On a backend that rounds another way, a
//! bound's `u` for each rounding becomes `2u`, and a bit-for-bit match can be off by one
//! ulp per rounding.
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
