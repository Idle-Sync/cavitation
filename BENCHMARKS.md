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
| B | 2 | v0.3.0 | running | | | | |
| C | 0 | v0.3.0 | 98.79 | 98.83 | 1.73 | 4.71 | 0.37 |

## cifar-cnn

| Machine | Seed | Commit | Cavitation, test % | PyTorch, test % | Cavitation epoch s | PyTorch epoch s | Ratio |
|---|---|---|---|---|---|---|---|
| A | 0 | 85b60f9 | 76.33 | 74.06 | 336.69 | 307.80 | 1.09 |
| A | 1 | 85b60f9 | 68.94 | 71.19 | 395.61 | 305.60 | 1.29 |
| A | 0, 2 epochs, quiet | c5f69ca | 59.38 | 60.73 | 341.8 | 302.8 | 1.13 |
| B | 0 | e232af8 | 75.34 | 74.11 | 113.16 | 86.84 | 1.30 |
| B | 1 | v0.3.0 | running | | | | |
| B | 2 | v0.3.0 | queued | | | | |
| C | 0 | v0.3.0 | running | | | | |

## Notes

- On machine A, the mnist-cnn runs and cifar-cnn seed 1 ran while other work was using the machine, so their times are less reliable than the "quiet" row. That row is the median epoch-2 time of the two rounds that started on an idle machine, out of three (one round started at 42% busy; `cifar-cnn --epochs 2`).
- Single seeds of cifar-cnn swing by about 2 points either way. The network has dropout, and its random masks differ between the two libraries. Over seeds 0 and 1 on machine A, the mean test accuracy is 72.64% for Cavitation and 72.63% for PyTorch.
- On machine C, a background program used 30–45% of one core during the runs.

## Running them

```sh
scripts/data/get-cifar10.sh                       # CIFAR-10 into data/cifar10/
cargo run --release -p cavitation-bench -- mnist-cnn --seed 0 --python <python with torch>
cargo run --release -p cavitation-bench -- cifar-cnn --seed 0 --python <python with torch>
```

The PyTorch side is `scripts/bench/torch_cnn.py`; `scripts/fixtures/requirements.txt` lists what its Python needs. Each run writes `result.json` under `target/bench/cnn/<benchmark>-seed<N>/`.
