# Cavitation

A fast, glass-box neural network library in Rust. Every value, local derivative and gradient inside a network can be read, changed and visualised, and every gradient is checked against a numerical estimate and against PyTorch.

## What it is

Cavitation trains and runs small and medium neural networks on the CPU, inside a Rust program, with no Python and one direct runtime dependency (`safetensors`, for saving and loading weights).

- **Glass box.** A training step is a tape of recorded operations you can read. Hooks on any layer show its input z, its output a, the local derivative f′(z), and the gradients ∂L/∂a and ∂L/∂z. Neuron masks turn single neurons off in both the forward and the backward pass.
- **Checked.** Every operation passes a numerical gradient check (central differences in `f64`) and a comparison with PyTorch on stored fixtures. Every fast kernel has a plain reference version, and tests check that the two agree. Known-result tests train real networks with fixed seeds and check the accuracy.
- **Fast on ordinary CPUs.** The kernel level (SSE4.2, AVX2 with FMA, or NEON) is chosen at run time, matrix products are split across threads, and a training step allocates nothing after a warm-up epoch.
- **Compatible with PyTorch.** Operations follow PyTorch's definitions. `Linear` stores its weight as `[outputs, inputs]`, and weights save and load as safetensors named like a PyTorch `state_dict` (`0.weight`, `2.bias`).

What v0.1 contains:

- An f32 tensor and reverse-mode autograd over a recorded tape.
- Operations: matmul, add, subtract, multiply (with broadcasting), sum and mean, 17 activations plus the learnable PReLU layer (ReLU, tanh, sigmoid, GELU, SiLU, Mish and the rest of PyTorch's common set), the gated units GLU, GeGLU, ReGLU and SwiGLU, softmax and log-softmax, and nine losses, each with mean, sum, none and batch-mean reductions.
- `Linear`, activation layers and `Sequential`; Xavier and He initialisation; SGD (with momentum and weight decay), Adam and AdamW; cosine learning-rate decay.
- `fit`, a step-by-step `Trainer`, and the manual loop.
- Hooks, neuron masks, and health checks that warn about NaN or infinity, exploding and vanishing gradients and dead ReLU neurons.
- An inference mode that runs a trained network without the training tape, for one input or a batch.
- User-defined activations, layers and tape operations, each covered by the gradient check.

## Using it

The crate is not on crates.io yet. Depend on the repository:

```toml
[dependencies]
cavitation = { git = "https://github.com/Lazy-Coder-03/cavitation" }
```

It needs Rust 1.99 or newer.

## Three levels of use

Each level is built from the one below it, so you can start with `fit` and open it up as far as you want.

**1. `fit`.** Give it a network, its parameters, a dataset and a configuration. It shuffles, batches, trains and checks the run's health.

```rust
use cavitation::prelude::*;

let (x, y) = (
    Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], &[4, 1]).unwrap(),
    Tensor::from_vec(vec![1.0f32, 3.0, 5.0, 7.0], &[4, 1]).unwrap(),
);
let data = Dataset { x, targets: Targets::Values(y) };
let mut params = ParamStore::new();
let net = Sequential::new().push(Linear::new(&mut params, 1, 1, &mut Rng::seed(0)));
let config = FitConfig {
    epochs: 200,
    batch: 4,
    optimizer: OptimizerConfig::Sgd { lr: 0.05, momentum: 0.0, weight_decay: 0.0 },
    loss: Loss::Mse,
    ..FitConfig::default()
};
let history = fit(&net, &mut params, &data, config).unwrap();
assert!(history.epochs[199].mean_loss < history.epochs[0].mean_loss);
```

**2. The manual loop.** Each step written out: open a tape on the parameter values, record the inputs, run the network forward, record the loss, run `backward` into the gradients, drop the tape, step the optimiser and zero the gradients. Every value on the tape can be read between these lines.

```rust
use cavitation::prelude::*;

let x = Tensor::from_vec(vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]).unwrap();
let y = Tensor::from_vec(vec![0.0f32, 1.0, 1.0, 0.0], &[4, 1]).unwrap();
let mut params = ParamStore::new();
let mut rng = Rng::seed(1);
let net = Sequential::new()
    .push(Linear::new(&mut params, 2, 8, &mut rng))
    .push(Tanh)
    .push(Linear::new(&mut params, 8, 1, &mut rng))
    .push(Sigmoid);
let mut optimizer = Sgd::new(0.5);
let mut graph = Graph::new();
let mut last = f32::INFINITY;
for _ in 0..1000 {
    // The tape reads `params.values`; `backward` writes `params.grads`.
    let mut tape = graph.tape(&params.values);
    let input = tape.input(x.view());
    let target = tape.input(y.view());
    let out = net.forward(&mut tape, input);
    let loss = tape.mse(out, target, Reduction::Mean);
    last = tape.value(loss).data()[0];
    tape.backward(loss, &mut params.grads);
    drop(tape); // releases `params.values` so the optimiser can change them
    optimizer.step(&mut params);
    params.zero_grad();
}
assert!(last < 0.01);
```

The `xor` example is this loop with health checks and hooks added, and the guide's first chapter walks through it.

**3. Your own parts.** An `Activation` needs only its value and its derivative. A `Layer` records its forward pass from tape operations. A `CustomOp` is a new tape operation with its own forward and backward. Each is generic over the number type, so `gradcheck` can check it in `f64`.

```rust
use cavitation::prelude::*;

/// `x³`.
struct Cube;

impl<T: Real> Activation<T> for Cube {
    fn name(&self) -> &str {
        "cube"
    }
    fn f(&self, x: T) -> T {
        x * x * x
    }
    // The derivative at x, given y = f(x).
    fn df(&self, x: T, _y: T) -> T {
        T::from_f64(3.0) * x * x
    }
}

let mut params = ParamStore::new();
let net = Sequential::new()
    .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
    .push(ActivationLayer::new(Cube));
assert_eq!(net.layer(1).name(), "cube");
```

## Examples

```sh
cargo run --release --example xor -p cavitation      # the manual loop, then every layer's values through hooks
cargo run --release --example digits -p cavitation   # fit on the 8x8 digits, then one inference
scripts/data/get-mnist.sh                            # downloads MNIST into data/mnist/ (once)
cargo run --release --example mnist -p cavitation    # fit on MNIST with Adam and cosine decay
```

The digits data ships in the repository. MNIST is downloaded by the script.

## Results

Known results, from tests with fixed seeds (`crates/cavitation/tests/known_results.rs`, and the XOR test in `crates/cavitation-nn/src/train/fit.rs`):

- XOR: a 2 → 8 tanh → 1 sigmoid network ends within 0.1 of every target after 2,000 full-batch epochs of SGD.
- scikit-learn 8×8 digits: the 64 → 256 → 128 → 10 network (50,826 parameters) reaches 97.50% (351 of 360) on the held-out digits after 20 epochs of Adam.
- MNIST: the 784 → 1024 → 256 → 10 network (1,068,810 parameters) reaches 98.63% on the 10,000 test images after 10 epochs of Adam with cosine decay. This test is ignored by default because it needs the data; run it with `cargo test --release -p cavitation -- --ignored mnist_known_result --nocapture`.

Speed, measured on an Intel Celeron N5105 (4 cores, SSE4.2, no AVX) with `cargo run --release -p cavitation-bench -- all`: release build, kernel level `sse4.2`, 4 threads, plain SGD for training, with numpy 2.4.4 (Python 3.14) measured in the same run on the same network, batch, data and step count; built with rustc 1.99.1.

- MNIST, 784 → 1024 → 256 → 10, batch 128: measured 6.17 s per epoch of 60,000 images; numpy measured 9.35 s (1.52× faster). The v0.1 target is ≤ 10 s.
- The same network at batch 32: measured 10.19 s per epoch; numpy measured 14.61 s (1.43× faster).
- Digits, 64 → 256 → 128 → 10, batch 128: measured 116,904 samples/s; numpy measured 86,787 samples/s (1.35× faster). The v0.1 target is ≥ 94,000 samples/s.
- One input through the digits network in inference mode: measured 6.77 µs; numpy measured 32.35 µs (4.78× faster). The v0.1 target is ≤ 7 µs.

Each training figure is the median of 3 timed epochs (MNIST) or 100 epochs (digits); the inference figure is the median of 10 rounds of 10,000 calls. Your numbers will differ on other machines.

## Running the benchmark

```sh
scripts/data/get-mnist.sh                        # the MNIST runs need the data
cargo run --release -p cavitation-bench -- info  # CPU, kernel levels and thread count
cargo run --release -p cavitation-bench -- all   # everything above
cargo run --release -p cavitation-bench -- mnist --batch 32          # one run; add --adam to time Adam
cargo run --release -p cavitation-bench -- digits
cargo run --release -p cavitation-bench -- latency
cargo run --release -p cavitation-bench -- gemm --shapes mnist128    # the matrix products of one step
```

The numpy baseline (`scripts/bench/numpy_mlp.py`) runs with `python3` from `PATH`, which needs numpy; `--python <path>` or the `CAVITATION_BENCH_PYTHON` variable names another interpreter. Results are also written to `target/bench/results.json`.

Two environment variables change how the library runs, for the benchmark and for your own programs:

- `CAVITATION_KERNEL` forces a kernel level: `reference`, `sse4.2`, `avx2` or `neon`. The default is the best level the CPU supports.
- `CAVITATION_THREADS` sets the number of threads for matrix products. The default is the number of logical cores.

## Tests

```sh
cargo test --workspace --release
```

This runs the gradient checks, the PyTorch fixture comparisons on every kernel level the CPU supports, the reference-twin kernel tests, the allocation tests and the known results that need no download.

## Guide

The guide is in `book/`. Its first chapter, "Backprop by hand", works through the XOR network's forward pass, local derivatives and gradients and reads each of them through hooks. Build it with [mdBook](https://rust-lang.github.io/mdBook/):

```sh
mdbook serve book   # then open http://localhost:3000
```

Status: in development. No release yet.

## Licence

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT licence ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 licence, shall be dual licensed as above, without any additional terms or conditions.
