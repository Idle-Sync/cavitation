# Benchmarks

Measured results for the two CNN benchmarks, `cavitation-bench mnist-cnn` and `cavitation-bench cifar-cnn`. Each one trains the same network in Cavitation and in PyTorch (CPU), from the same initial weights and the same batch order, and reports test accuracy and time per epoch for both.

- **mnist-cnn:** a LeNet-5-style network on MNIST, 5 epochs of 60,000 images at batch 64, Adam with learning rate 1e-3.
- **cifar-cnn:** four 3 × 3 convolutions (32, 32, 64 and 64 channels), each followed by batch norm and ReLU, with max pooling after each pair, then dropout and two linear layers, on CIFAR-10, 10 epochs of 50,000 images at batch 128, Adam with learning rate 1e-3.

Every run uses 4 threads (`CAVITATION_THREADS=4`, and `torch.set_num_threads(4)` on the PyTorch side). "Epoch time" is the median of the epochs after the first. "Ratio" is Cavitation's epoch time divided by PyTorch's, so below 1 means Cavitation is faster.

## Machines

| Machine | CPU | OS | Kernel level | rustc | PyTorch |
|---|---|---|---|---|---|
| A | Intel Celeron N5105, 4 cores | Ubuntu 24.04.5 | `sse4.2` | 1.99.1 | 2.15.0+cpu |
| B | AMD Ryzen 7 2700X, 8 cores | Windows 10 | `avx2` | 1.100.0 | 2.15.0+cpu |
| C | Apple M4 (Mac mini) | macOS | `accelerate` | 1.99.0 | 2.15.0 |

## mnist-cnn

| Machine | Seed | Commit | Cavitation, test % | PyTorch, test % | Cavitation epoch s | PyTorch epoch s | Ratio |
|---|---|---|---|---|---|---|---|
| A | 0 | 85b60f9 | 98.85 | 98.46 | 13.17 | 15.14 | 0.87 |
| A | 1 | 85b60f9 | 98.98 | 98.89 | 13.25 | 14.38 | 0.92 |
| A | 2 | 85b60f9 | 98.70 | 98.79 | 13.75 | 14.53 | 0.95 |
| B | 0 | e232af8 | 98.84 | 98.54 | 6.09 | 6.04 | 1.01 |
| B | 1 | e232af8 | 99.01 | 98.88 | 6.09 | 6.28 | 0.97 |
| B | 2 | v0.3.0 | 98.67 | 98.84 | 6.01 | 6.35 | 0.95 |
| C | 0 | v0.3.0 | 98.79 | 98.83 | 1.73 | 4.71 | 0.37 |

## cifar-cnn

| Machine | Seed | Commit | Cavitation, test % | PyTorch, test % | Cavitation epoch s | PyTorch epoch s | Ratio |
|---|---|---|---|---|---|---|---|
| A | 0 | 85b60f9 | 76.33 | 74.06 | 336.69 | 307.80 | 1.09 |
| A | 1 | 85b60f9 | 68.94 | 71.19 | 395.61 | 305.60 | 1.29 |
| A | 0, 2 epochs, quiet | c5f69ca | 59.38 | 60.73 | 341.8 | 302.8 | 1.13 |
| B | 0 | e232af8 | 75.34 | 74.11 | 113.16 | 86.84 | 1.30 |
| B | 1 | v0.3.0 | 71.13 | 72.94 | 116.38 | 86.92 | 1.34 |
| B | 2 | v0.3.0 | 73.03 | 74.90 | 114.06 | 87.86 | 1.30 |
| C | 0 | v0.3.0 | 76.63 | 74.66 | 30.26 | 43.13 | 0.70 |
| B, `--device cuda`, NVIDIA GTX 1660 Super, heuristic algorithms (default) | 0 | 2149877 | 76.89 | 71.34 | 4.89 | 4.80 | 1.02 |
| B, `--device cuda`, NVIDIA GTX 1660 Super, `CAVITATION_CUDNN_BENCHMARK=1` | 0 | 2149877 | 75.27 | 70.63 | 4.42 | 4.82 | 0.92 |

## Means over seeds

| Machine | Benchmark | Seeds | Cavitation, test % | PyTorch, test % |
|---|---|---|---|---|
| A | mnist-cnn | 0–2 | 98.84 | 98.71 |
| A | cifar-cnn | 0–1 | 72.64 | 72.63 |
| B | mnist-cnn | 0–2 | 98.84 | 98.75 |
| B | cifar-cnn | 0–2 | 73.17 | 73.98 |

## Notes

- On machine A, the mnist-cnn runs and cifar-cnn seed 1 ran while other work was using the machine, so their times are less reliable than the "quiet" row. That row is the median epoch-2 time of the two rounds that started on an idle machine, out of three (one round started at 42% busy; `cifar-cnn --epochs 2`).
- Single seeds of cifar-cnn swing by about 2 points either way. The network has dropout, and its random masks differ between the two libraries. The means above average that out.
- On machine B, cifar-cnn seed 1 overlapped Windows background maintenance (.NET native-image compilation and antivirus scans) from epoch 7 of Cavitation's half through PyTorch's half. Cavitation's epochs 7–10 took 115–120 s, against 112–113 s before. Shorter bursts touched epochs 4–6 of seed 2 and the first PyTorch epochs of seed 2.
- Machine B's seeds 0 and 1 of mnist-cnn and seed 0 of cifar-cnn ran at e232af8, the rest at v0.3.0. The benchmark code is the same apart from one line, and cifar-cnn seed 1's first nine Cavitation epochs give identical losses at both commits.
- The two `--device cuda` rows train both sides on machine B's NVIDIA GTX 1660 Super (driver 591.86), one after the other in one session: Cavitation through `CudaTrainer` (cuDNN 9.28.1, cuBLAS 13.2.2), by default with cuDNN's heuristic choice of convolution algorithms and in the second row with them timed (`CAVITATION_CUDNN_BENCHMARK=1`); PyTorch 2.14.1+cu130 (cuDNN 9.24.0) with `--torch-device cuda`, `cudnn.benchmark` False and deterministic off, so its accuracy differs between the rows. Command: `cargo run --release -p cavitation-bench --features cuda -- cifar-cnn --seed 0 --device cuda --torch-device cuda --python <python with CUDA torch>`. A game was running on the card throughout (35–37% utilisation before the first run, 92% before the second); PyTorch's median step was 11.83 ms in both runs, against 11.04 ms on the idle card. The commit is the last code change the runs were measured on.
- On machine C, a background program used 30–45% of one core during the runs. The PyTorch build there reports `mkldnn False`.

## PyTorch as a fixed baseline

The PyTorch numbers above are the baseline for each machine and seed. With the same network, data, seed, batch size, epochs, learning rate, thread count and PyTorch version, PyTorch's run would be the same again, so a later measurement of Cavitation runs only Cavitation's half (`--cav-only`) and compares it with these rows. A new PyTorch version, or a change to the benchmark's settings, needs a new baseline. Compare epoch times only on a machine as quiet as it was for the baseline row.

## Running them

```sh
scripts/data/get-cifar10.sh                       # CIFAR-10 into data/cifar10/
cargo run --release -p cavitation-bench -- mnist-cnn --seed 0 --python <python with torch>
cargo run --release -p cavitation-bench -- cifar-cnn --seed 0 --python <python with torch>
cargo run --release -p cavitation-bench -- cifar-cnn --seed 0 --cav-only    # Cavitation's half only
cargo run --release -p cavitation-bench -- cifar-cnn --profile 15            # time per layer, no training run
cargo run --release -p cavitation-bench --features gpu -- cifar-cnn --profile 15 --device gpu   # the same on the GPU
```

The PyTorch side is `scripts/bench/torch_cnn.py`; `scripts/fixtures/requirements.txt` lists what its Python needs. Each run writes its results to `target/bench/cnn/<benchmark>-seed<N>/result.json`.
