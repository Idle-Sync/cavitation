# Changelog

Notable changes by release. Releases are git tags `vX.Y.Z`. New work goes under Unreleased.

## Unreleased

The first release: a CPU neural network library for multilayer perceptrons whose every value and gradient can be read during training.

### Workspace

- Five crates: `cavitation` (the one to depend on: re-exports the core and nn crates and gathers the everyday names in `cavitation::prelude`), `cavitation-core`, `cavitation-nn`, `cavitation-datasets` and `cavitation-bench`. Every crate is licensed `MIT OR Apache-2.0`.

### Tensors and autograd (`cavitation-core`)

- `Real` (implemented for `f32` and `f64`, with `Real::erf` ported from FreeBSD msun), `Shape` (`Shape::new` panics if the element count overflows `usize`), `Tensor` with `TensorView` and `TensorViewMut`, and `Error` for the failures a caller can recover from (bad data, bad files, unsupported requests, non-finite values during training).
- `Rng` (xoshiro256++) seeded through `SplitMix64`; `Rng::range(lo, hi)` is below `hi` when `lo < hi`; equal bounds give `lo`.
- Reverse-mode automatic differentiation: `Graph`, `Tape` and `Var`, and the `ParamStore` that holds parameters and their gradients. `zero_grad` marks gradients stale without writing them; `ParamStore::iter_mut` gives `None` for a gradient nothing has contributed to since. A parameter whose value was reshaped through `value_mut` panics in `iter_mut` and in the backward pass, naming the two shapes.
- Tape operations, following PyTorch's definitions and boundary conventions:
  - arithmetic with NumPy-style broadcasting: `add`, `sub`, `mul`, `scale`;
  - `matmul` and `matmul_nt`, with gradients computed on strided views;
  - reductions: `sum`, `sum_axis`, `mean`, `mean_axis`;
  - activations: `relu`, `leaky_relu`, `tanh`, `sigmoid`, `gelu` (exact and tanh forms, `GeluApprox`), `elu`, `selu`, `celu`, `silu`, `mish`, `softplus`, `softsign`, `hardtanh`, `relu6`, `hardsigmoid`, `hardswish`, `log_sigmoid`, `prelu`, and `glu` with a `Gate` (GLU, GeGLU, ReGLU, SwiGLU); NaN passes through every one of them;
  - `softmax` and `log_softmax`, and `softmax_row` and `log_softmax_row` for one row;
  - losses, each taking a `Reduction` (`Mean`, `Sum`, `None`, `BatchMean`): fused `softmax_cross_entropy` in log-sum-exp form, `nll`, `mse`, `l1`, `huber`, `smooth_l1`, `binary_cross_entropy` (with `Tape::clamped_logs`, the count of logarithms clamped to -100), `binary_cross_entropy_with_logits` and `kl_div`; NaN input gives a NaN loss;
  - `keep_where`, the select behind neuron masks, matching `torch.where(keep != 0, x, 0)`: exactly 0 and no gradient where `keep` is 0, even for an infinite or NaN value or gradient there.
- User-defined operations: the `CustomOp` trait (forward, backward and output shape, generic over `Real`) and `Tape::custom`, with up to 4 inputs.
- `gradcheck` and `gradcheck_params`: a numerical gradient check in `f64` by central differences, with a report of every element outside the tolerance. Both take the inputs first and the closure last.
- Kernels, chosen at run time: `kernels::Level` (reference, SSE4.2, AVX2 with FMA, NEON), `set_level` and the `CAVITATION_KERNEL` override; the thread count (`threads`, `set_threads`, `CAVITATION_THREADS`); a persistent thread pool (`kernels::pool::run` and `run_chunks`) that allocates nothing per call; the packed f32 GEMM `kernels::gemm::gemm_f32` (Goto-style blocking and packing, a microkernel per level, bit-identical results for any thread count); and `kernels::matvec::matvec_f32`, one input times a transposed weight matrix with an optional bias and fused ReLU. Every fast kernel has a reference twin and tests that they agree.
- Every tape operation passes the numerical gradient check and matches a PyTorch 2.15.0 fixture (written by `scripts/fixtures/make_fixtures.py`) on every kernel level the CPU supports.

### Layers, training and inference (`cavitation-nn`)

- `Init` (Xavier and He initialisers, zeros, constant), the object-safe `Layer` trait, `Linear` (weight `[outputs, inputs]` like PyTorch), a layer for every built-in activation plus `Softmax`, `LogSoftmax`, `Glu` and `Prelu`, `Activation` and `ActivationLayer` for activations of your own, and `Sequential` with `Trace` and PyTorch-style parameter names (`0.weight`, `2.bias`). `Sequential` checks the width between layers and names the layer that does not fit.
- Optimisers: `Sgd` (momentum, weight decay), `Adam` and `AdamW`, matching `torch.optim` over ten-step trajectories. Each updates its parameters in chunks of 16,384 elements over the thread pool, bit-identical to the plain loop that runs at the reference level, and collects each gradient's norm and finiteness (`GradStats`) in the same pass. A parameter that received no gradient since `zero_grad` is skipped, as PyTorch skips `grad is None`; Adam's bias correction counts each parameter's own updates.
- Data: `Targets`, `Dataset`, `Batch`, `BatchView` and `Batcher` (shuffled batches whose smaller last batch reuses the full-size buffer).
- Training: `Trainer` (one step at a time; `step` returns the batch's loss or the health checks' error at the step that meets it), `Loss` (cross-entropy, NLL, MSE, L1, Huber, smooth L1, BCE, BCE with logits, KL divergence), `fit` and `fit_with_hooks` with `FitConfig`, `OptimizerConfig`, `LrSchedule` (constant, or cosine as PyTorch's `CosineAnnealingLR` stepped once per epoch), `History` and `EpochReport`, and `accuracy`. `fit` checks the data against the loss and the model before it trains and returns `Error::InvalidData` or `Error::ShapeMismatch` instead of panicking. The same seeds give bit-identical parameters for any thread count on the same machine and kernel level.
- Hooks: `Hooks` (with `HookId`) runs closures on chosen layers after each forward or backward pass; each sees a `LayerEvent` with the layer's input, output, local derivative and the gradients with respect to its output and input.
- Neuron masks: `NeuronMask` with `Sequential::set_mask`, `clear_mask` and `mask`; an off neuron outputs exactly 0 and passes no gradient.
- Health checks: `Health` (with `OnNonFinite`), `HealthMonitor` and `Warning`. Every step checks that the loss and each gradient are finite (by default training stops with `Error::NonFinite`, naming the step and `loss` or the parameter), warns when a gradient norm is above 1e3 or below 1e-7 and when binary cross-entropy clamps a logarithm, and each epoch reports ReLU and ReLU6 neurons that output 0 for every sample. Warnings go to a sink closure, or to standard error, where each layer's dead-neuron report is printed only the first time.
- Inference: `InferenceModel::compile` turns a `Sequential` and its parameters into a model that runs without the training graph; `infer` runs one input through the matvec kernels on the calling thread and `infer_batch` a batch through the GEMM. `InferenceLayer` and `Layer::inference` give every built-in layer an inference form.
- `save`, `load` and `load_lenient`: weights in the safetensors format, named as a PyTorch `state_dict`. Files written by `save` load in PyTorch with `safetensors.torch.load_file` (checked by `make_fixtures.py --check-rust-save` against a committed file that `save` must reproduce byte for byte), and files saved by PyTorch load here. Loading checks every key, shape and the F32 type before it writes anything.
- After a warm-up, a training step, a hooked or masked step, the health checks and both inference paths allocate nothing.

### Data, examples, guide and benchmark

- `cavitation-datasets`: loaders for the scikit-learn 8×8 digits (included in `data/digits/`) and for MNIST (downloaded by `scripts/data/get-mnist.sh`).
- The `xor`, `digits` and `mnist` examples, and known-result tests: XOR, 97.50% (351 of 360) on the held-out digits, and 98.63% on the MNIST test images (ignored by default, as it needs the data).
- The guide (`book/`, built with mdBook), with its first chapter, "Backprop by hand".
- `cavitation-bench` (`all`, `mnist`, `digits`, `latency`, `gemm`, `info`) times training and inference against a numpy baseline measured in the same run on the same network, batch, data and step count. Measured with `cargo run --release -p cavitation-bench -- all` on an Intel Celeron N5105 (4 cores, SSE4.2, no AVX), release build, kernel level `sse4.2`, 4 threads, plain SGD, numpy 2.4.4, rustc 1.99.1: an MNIST epoch of the 784 → 1024 → 256 → 10 network takes 6.17 s at batch 128 (numpy 9.35 s) and 10.19 s at batch 32 (numpy 14.61 s); the digits network trains at 116,904 samples/s (numpy 86,787); one input through the digits network in inference mode takes 6.77 µs (numpy 32.35 µs).
