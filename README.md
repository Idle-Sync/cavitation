# Cavitation

A fast, glass-box neural network library in Rust. Every value, local derivative and gradient inside a network can be read, changed and visualised, and every gradient is checked against a numerical estimate and against PyTorch.

## What it is

Cavitation trains and runs small and medium neural networks on the CPU, inside a Rust program, with no Python and one direct runtime dependency (`safetensors`, for saving and loading weights).

- **Glass box.** A training step is a tape of recorded operations you can read. Hooks on any layer show its input z, its output a, the local derivative f′(z), and the gradients ∂L/∂a and ∂L/∂z. Neuron masks turn single neurons off in both the forward and the backward pass.
- **Checked.** Every operation passes a numerical gradient check (central differences in `f64`) and a comparison with PyTorch on stored fixtures. Every fast kernel has a plain reference version, and tests check that the two agree. Known-result tests train real networks with fixed seeds and check the accuracy.
- **Fast on ordinary CPUs.** The kernel level (SSE4.2, AVX2 with FMA, NEON, or Apple's Accelerate framework for matrix products on Apple Silicon) is chosen at run time, matrix products are split across threads (on Apple Silicon each one is a single call into Accelerate), and a training step allocates nothing after a warm-up epoch.
- **Compatible with PyTorch.** Operations follow PyTorch's definitions. `Linear` stores its weight as `[outputs, inputs]`, and weights save and load as safetensors named like a PyTorch `state_dict` (`0.weight`, `2.bias`).

What it contains:

- An f32 tensor and reverse-mode autograd over a recorded tape.
- Operations: matmul, add, subtract, multiply (with broadcasting), sum and mean, 17 activations plus the learnable PReLU layer (ReLU, tanh, sigmoid, GELU, SiLU, Mish and the rest of PyTorch's common set), the gated units GLU, GeGLU, ReGLU and SwiGLU, softmax and log-softmax, and nine losses, each with mean, sum, none and batch-mean reductions.
- `Linear`, activation layers and `Sequential`; Xavier and He initialisation; SGD (with momentum and weight decay), Adam and AdamW; cosine learning-rate decay.
- `fit`, a step-by-step `Trainer`, and the manual loop.
- Hooks, neuron masks, and health checks that warn about NaN or infinity, exploding and vanishing gradients and dead ReLU neurons.
- An inference mode that runs a trained network without the training tape, for one input or a batch.
- User-defined activations, layers and tape operations, each covered by the gradient check.
- Held-out evaluation (loss, accuracy, R²), dataset splits, and a `Standardizer` whose scaling folds into the first or last `Linear`.
- Shrinking: an ablation sweep that ranks neurons by what switching each off costs, a `shrink` tool that removes neurons step by step with retraining and stops on a validation set, and compaction, which rebuilds a masked network as a smaller dense one.
- Linear and logistic regression in `f64`, fitted exactly (QR, Newton) or by gradient descent on one `Linear` (the `ml` feature, on by default).
- The inspector: a live view of a training run in the browser, from a server inside your program (the `inspect` feature, off by default).

## Using it

The crate is not on crates.io yet. Depend on the repository:

```toml
[dependencies]
cavitation = { git = "https://github.com/Idle-Sync/cavitation" }
```

It needs Rust 1.99 or newer.

Two features choose the optional parts:

- `ml` (on by default) adds `cavitation::cavitation_ml`: least squares, `LinearRegression` and `LogisticRegression`.
- `inspect` (off by default) adds `cavitation::cavitation_inspect`: the inspector.

```toml
[dependencies]
# With the inspector:
cavitation = { git = "https://github.com/Idle-Sync/cavitation", features = ["inspect"] }
# Or only the neural network parts, without the classic models:
# cavitation = { git = "https://github.com/Idle-Sync/cavitation", default-features = false }
```

With either feature on, `cavitation::prelude` also has its main types: `LinearRegression`, `LogisticRegression`, `LinearModel` and `LogisticModel` with `ml`, and `Inspector` and `InspectorConfig` with `inspect`.

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

## The inspector

The inspector shows a training run in the browser while it trains. Turn on the `inspect` feature, start an `Inspector`, open the address it gives, and train with `cavitation_inspect::fit` in place of `fit`:

```rust,ignore
use cavitation::prelude::*;
use cavitation::cavitation_inspect;

let mut inspector = Inspector::start(InspectorConfig::default())?;
println!("open {}", inspector.url()); // http://127.0.0.1:7878/?token=...
let history = cavitation_inspect::fit(&mut net, &mut params, &data, config, &mut inspector)?;
inspector.export_html("run.html")?; // the page with the run in it, to open without a server
```

The page shows:

- samples per second, GFLOPS, loss, accuracy or R², the epoch and the step, with pause and resume and a learning-rate override;
- one input row through every layer as heat maps of a, z, f′(z) and ∂L/∂z, with switched-off neurons crossed out;
- one neuron in detail: its values and gradients, the chain rule with its numbers, its activation curve with the tangent at z, its incoming weights (as an image for image inputs) and its largest input contributions, and a switch to turn it off;
- where a training step's time goes, layer by layer.

After training, `Inspector::explore` serves the trained network to the page: switch neurons, rank a hidden layer's neurons by what each one's absence costs, and shrink the network level by level while you watch its validation accuracy.

The server binds `127.0.0.1:7878` by default, so only this machine can reach it. Every request for the run's data or a command needs the per-run token in the address, requests from other sites are refused, and the page loads nothing from outside the server. With no page open, nothing is captured, and a run with no commands from the page trains to the same parameters as `fit`, bit for bit.

## Shrinking and compaction

`cavitation_nn::shrink` finds how many neurons a trained network can do without. `ablate` switches each neuron of a hidden layer off in turn and ranks them by what their absence costs. `shrink` removes neurons a step at a time with neuron masks, retrains after each step, and keeps the smallest network whose accuracy on a validation set has not dropped by more than the limit (1 point by default). `compact` then rebuilds the masked network as a smaller dense one in a new parameter store, whose outputs through `infer` match the masked network's bit for bit.

```sh
cargo run --release --example shrink_digits -p cavitation
```

The example trains the 64 → 256 → 128 → 10 digits network on 1,149 training rows, shrinks it with the other 288 as the validation set, compacts it and saves it.

Measured on an Intel Celeron N5105 (4 cores, kernel level `sse4.2`, rustc 1.99.1), pinned to one core with `taskset`, with `cargo run --release -p cavitation-bench -- shrink` on a machine that was otherwise mostly idle (other processes used under 30% of one core): shrinking with the defaults removed 75% of the hidden neurons, leaving 64 + 32, with 351 of 360 test digits right (the unpruned network had 352). One input through the compacted network in inference mode took 1.006 µs, against 6.808 µs for the full network: 6.77× faster (median of 301 rounds of 1,000 calls). The guide's chapter "Shrinking a network" walks through it.

## Linear and logistic regression

`cavitation_ml` (the `ml` feature, on by default) fits linear regression, ridge and logistic regression in `f64`, exactly or by gradient descent. The exact paths solve least squares by Householder QR, and logistic regression by Newton's method. The gradient paths train one `Linear` through the same tape as a neural network. Both give a model that converts to a one-layer `Sequential`, which saves in PyTorch's `nn.Linear` layout.

```rust
use cavitation::prelude::*;

let x = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], &[4, 1]).unwrap();
let y = Tensor::from_vec(vec![1.0f32, 3.0, 5.0, 7.0], &[4, 1]).unwrap();
let data = Dataset { x, targets: Targets::Values(y) };
let model = LinearRegression::default().fit(&data).unwrap();
assert!((model.coef.data()[0] - 2.0).abs() < 1e-12); // y = 2x + 1
assert!((model.intercept[0] - 1.0).abs() < 1e-12);
assert!((model.score(&data).unwrap() - 1.0).abs() < 1e-12); // R²
```

Both paths are tested against scikit-learn 1.9.1. Measured on an Intel Celeron N5105 (4 cores) with `cargo test --release -p cavitation-ml --test sklearn --test gradient_path -- --nocapture`, the largest gaps, relative to max(1, |value|), are:

- `LinearRegression` and ridge against scikit-learn's `LinearRegression` and `Ridge`: 2.6e-14 (the tests allow 1e-9);
- `LogisticRegression`'s Newton path against `newton-cholesky`: 1.6e-12 (allowed 1e-8), and against `lbfgs`: 5.3e-7 (allowed 1e-5);
- the gradient path against the exact one: 3.2e-6 on the standardised Concrete data and 4.8e-7 on the standardised Breast Cancer data (allowed 1e-3).

## Examples

```sh
cargo run --release --example xor -p cavitation      # the manual loop, then every layer's values through hooks
cargo run --release --example digits -p cavitation   # fit on the 8x8 digits, then one inference
cargo run --release --example shrink_digits -p cavitation  # shrink the digits network, compact it and save it
cargo run --release --example linear_regression -p cavitation    # Concrete: exact and gradient OLS and ridge
cargo run --release --example logistic_regression -p cavitation  # Breast Cancer: Newton and gradient paths
scripts/data/get-mnist.sh                            # downloads MNIST into data/mnist/ (once)
cargo run --release --example mnist -p cavitation    # fit on MNIST with Adam and cosine decay
```

The digits, Concrete and Breast Cancer data ship in the repository. MNIST is downloaded by its script, and CIFAR-10 (binary version, 50,000 training and 10,000 test images of 32 by 32 pixels in three colours) by `scripts/data/get-cifar10.sh` into `data/cifar10/`; `cavitation_datasets::cifar10::load` reads it, for the benchmark below.

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
scripts/data/get-cifar10.sh                                          # the cifar run needs the data
cargo run --release -p cavitation-bench -- cifar                     # 3072 → 1024 → 512 → 10, Adam, batch 128 and 32
cargo run --release -p cavitation-bench -- cifar --batch 128 --steps 100  # steps per second, for slow machines
cargo run --release -p cavitation-bench -- digits
cargo run --release -p cavitation-bench -- latency
cargo run --release -p cavitation-bench -- gemm --shapes mnist128    # the matrix products of one step
cargo run --release -p cavitation-bench -- shrink                    # shrink the digits network twice, time it compacted
```

`all` leaves out `cifar`: an epoch of its network on 50,000 images takes minutes on a small machine, so it is its own command. It is sized so that matrix products, not per-step overheads, take most of the time, which the MNIST run is too short for on a fast machine.

The numpy baseline (`scripts/bench/numpy_mlp.py`) runs with `python3` from `PATH`, which needs numpy; `--python <path>` or the `CAVITATION_BENCH_PYTHON` variable names another interpreter. Results are also written to `target/bench/results.json`, or to a `bench` folder under `CARGO_TARGET_DIR` when that is set.

Two environment variables change how the library runs, for the benchmark and for your own programs:

- `CAVITATION_KERNEL` forces a kernel level: `reference`, `sse4.2`, `avx2`, `neon` or `accelerate`. The default is the best level the CPU supports: `accelerate` on Apple Silicon, which runs matrix products through Apple's Accelerate framework (part of macOS, so nothing to install) and everything else through the NEON kernels. Its results match NEON's within the matrix-product tolerance, not bit for bit; `neon` runs the NEON matrix product instead.
- `CAVITATION_THREADS` sets the number of threads the threaded kernels use: the matrix products (in training and in `infer_batch`) and the optimiser update. At the `accelerate` level each matrix product is one Accelerate call, so the variable does not limit the threads Accelerate starts itself (Apple's `VECLIB_MAXIMUM_THREADS` does); it still governs the optimiser update and, with `CAVITATION_KERNEL=neon`, every product. The default is the number of performance cores on Apple Silicon (the efficiency cores slow training down) and the number of logical cores elsewhere; `infer` always runs on the calling thread.

## Tests

```sh
cargo test --workspace --release
```

This runs the gradient checks, the PyTorch fixture comparisons on every kernel level the CPU supports, the reference-twin kernel tests, the allocation tests and the known results that need no download.

## Guide

The guide is in `book/`, in five chapters:

- "Backprop by hand" works through the XOR network's forward pass, local derivatives and gradients and reads each of them through hooks.
- "Shrinking a network" finds how many neurons the digits network can do without and removes them.
- "Watching a network train" runs the inspector on the digits network, reads each panel of the page, and measures what watching costs.
- "Regression, exactly and by gradient" fits linear and logistic regression both ways and shows why the two meet.
- "Coming from PyTorch" lists what matches PyTorch, so a model moved between the two gives the same outputs, and where a PyTorch habit gives a different answer or none.

Build it with [mdBook](https://rust-lang.github.io/mdBook/):

```sh
mdbook serve book   # then open http://localhost:3000
```

Status: in development. v0.2.0 is the latest tag in the repository; nothing is published on crates.io yet.

## Licence

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT licence ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 licence, shall be dual licensed as above, without any additional terms or conditions.
