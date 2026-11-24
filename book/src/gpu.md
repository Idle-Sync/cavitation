# Training on a GPU

The same `fit` that trains on the CPU trains on a GPU: the same network, the same batches in the same order, the same learning rates, dropout masks and health checks. Only `FitConfig::device` changes. This chapter covers:

- the two GPU features and what each needs;
- choosing a device, by name or automatically;
- what a GPU run gives back and how close it is to the CPU's;
- what a GPU does not train yet;
- when a GPU is not worth it.

The program is `crates/cavitation/examples/gpu_digits.rs`. It trains the digits network on the CPU and then on a GPU, from the same weights:

```sh
cargo run --release --example gpu_digits -p cavitation --features gpu
```

## Two features

GPU support is off by default and comes in two features of the `cavitation` crate:

- `gpu` runs one set of WGSL kernels through wgpu, on Vulkan (Linux and Windows), DX12 (Windows) and Metal (macOS). It needs only the GPU's own driver.
- `cuda` runs on NVIDIA cards through NVIDIA's CUDA, cuDNN and cuBLAS. These libraries are found and loaded when the program runs, so a build with the feature still starts on a machine without them. Install an NVIDIA driver, cuDNN 9, and cuBLAS and NVRTC from the CUDA toolkit (12 or 13) that cuDNN was built for. The libraries are looked for in the folders of `CAVITATION_CUDA_PATH`, `CUDNN_PATH` and `CUDA_PATH`, then where the operating system looks.

```toml
[dependencies]
cavitation = { git = "https://github.com/Idle-Sync/cavitation", features = ["gpu"] }
# Or both:
# cavitation = { git = "https://github.com/Idle-Sync/cavitation", features = ["gpu", "cuda"] }
```

With them, `cavitation::cavitation_gpu` and `cavitation::cavitation_cuda` are the GPU crates.

## Choosing a device

`devices()` lists what this machine and this build can train on, best first, and marks the one `Device::Auto` would pick:

```rust,ignore
{{#include ../../crates/cavitation/examples/gpu_digits.rs:devices}}
```

On a laptop with an Intel integrated GPU this prints:

```text
gpu  Intel(R) UHD Graphics (JSL), vulkan  <- chosen
cpu  CPU, sse4.2, 4 threads
```

`cargo run --release -p cavitation-bench --features gpu -- devices` prints the same list from the command line, and with the `cuda` feature it also says what is missing when CUDA does not open.

A run names its device in `FitConfig::device`:

- `Device::Cpu`, the default.
- `Device::Gpu(&gpu)`, for a GPU opened with `cavitation_gpu::Gpu::new()`. `Gpu::new` tries Metal on macOS, Vulkan and then DX12 on Windows, and Vulkan elsewhere. Set `CAVITATION_GPU_BACKEND` to `vulkan`, `dx12` or `metal` to use that backend alone. It returns `None` when no GPU opens, and `Gpu::open` says why.
- `Device::Cuda(&cuda)`, for a card opened with `cavitation_cuda::Cuda::new()` (or `Cuda::open`, which says what is missing).
- `Device::Auto`: CUDA when it opens and can train the run, then a GPU, then the CPU. The environment variable `CAVITATION_DEVICE` (`auto`, `cuda`, `gpu` or `cpu`) restricts what it picks.

The example's training function takes the device as an argument and is otherwise the CPU program:

```rust,ignore
{{#include ../../crates/cavitation/examples/gpu_digits.rs:run}}
```

and the GPU run opens a GPU first:

```rust,ignore
{{#include ../../crates/cavitation/examples/gpu_digits.rs:gpu}}
```

Each `Device::Auto` run opens CUDA or the GPU anew. A program that calls `fit` many times should open the device once and pass `Device::Gpu` or `Device::Cuda`.

## What comes back

The parameters live on the GPU during the run. `fit` copies them back into the `ParamStore` before each `on_epoch` call, when early stopping keeps an epoch as the best so far, before it returns an error met during training, and at the end of the run. So after `fit` returns, `params` holds the trained values, and everything that runs on the CPU (`accuracy`, `InferenceModel`, `save`) works on them as after a CPU run.

`History::device` says where the run trained, and `History::device_note` says why a device that opened was left, for example when `Device::Auto` met a layer the GPU has no kernel for.

A GPU run agrees with the CPU run within the GPU kernels' stated error bounds, not bit for bit. On CUDA, cuDNN and cuBLAS publish no error bounds, so the agreement is within measured tolerances; their algorithms are deterministic, so two CUDA runs give the same bits. On the Intel GPU above, the example's two runs end with the same test accuracy (97.22%), and the GPU's last training loss is 3.8e-7 above the CPU's, relative. A run that must give the same bits on every machine names its device.

One difference from the CPU is deliberate. A step whose loss or gradient is NaN or infinite is never applied on a GPU, so under `OnNonFinite::Stop` the returned parameters are those from before that step. The CPU has already applied a non-finite gradient that came with a finite loss.

For a hand-written loop, `GpuTrainer` (and `CudaTrainer`) are the step-by-step trainers that `fit` uses, and `GpuModel` (and `CudaModel`) run a compiled model in evaluation mode: `infer_batch`, `evaluate` and `accuracy`.

## What a GPU does not train yet

A GPU run is refused with `Error::Unsupported`, before any training, for what its trainer has no kernel for:

- the loss `KlDiv`;
- the optimisers `RmsProp` and `Adagrad`;
- `OnNonFinite::Warn`;
- hooks registered with `fit_with_hooks`, and `fit_controlled`;
- layers of your own.

The layers of the library, the other activations and losses, SGD, Adam and AdamW, the learning-rate schedules, validation, early stopping and the other health checks all train on every device. `Device::Auto` falls back to the CPU for a run a GPU cannot train, and says so in `History::device_note`.

## When a GPU is not worth it

A GPU step pays for an upload, its kernel launches and a readback whatever the network's size. On a small network those weigh as much as the arithmetic. On an Intel UHD Graphics (JSL) through Vulkan, the example's GPU run took 5.3 s, compiling the model's kernels included, against the CPU's 0.07 s. On an NVIDIA GTX 1660 Super, a 64 → 256 → 128 → 10 network at batch 32 trains an epoch of the digits in 19.9 to 20.8 ms on the GPU and in 18.9 to 19.5 ms on an AMD Ryzen 7 2700X. A CNN gains: an epoch of 12,800 CIFAR-10 images takes 2.54 s on that GPU through Vulkan, 1.48 s through CUDA, and 14.21 s on the CPU. The figures are measured with `cifar-cnn --cav-only --rows 12800 --epochs 3` and `--device gpu`, `cuda` and `cpu`.

`Device::Auto` does not weigh the model's size, so name `Device::Cpu` for a network that small.

On DX12, compiling a model's kernels takes a few seconds with Microsoft's DXC compiler and much longer with the older FXC, which DX12 falls back to when `dxcompiler.dll` is not found.
