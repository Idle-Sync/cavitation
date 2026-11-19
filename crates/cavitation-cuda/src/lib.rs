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
//!   [`Scalar`] (`f32`, `u32` or `i32`).
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
//! What is not here yet: the other layers. Nothing calls cuBLAS beyond creating its handle,
//! and no part of `cavitation-nn` trains through this crate.
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

mod buffer;
mod conv;
mod device;
mod error;
mod find;
mod library;
mod module;
mod sys;

pub use buffer::{DeviceBuffer, Scalar};
pub use conv::{Conv2d, ConvAlgorithms, ConvShape, MAX_WORKSPACE};
pub use device::{Cuda, CudaInfo};
pub use error::CudaError;
pub use module::{Arg, Kernel, MAX_KERNEL_ARGS, Module};
