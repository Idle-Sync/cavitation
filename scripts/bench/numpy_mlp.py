# The numpy baseline of cavitation-bench: trains or runs a ReLU MLP in plain numpy and prints
# one JSON line with the timings. `cavitation-bench` runs it with the same network, batch,
# data, warm-up and step counts as its own run and compares the two.
#
#   python3 -I scripts/bench/numpy_mlp.py --mode train --net 784,1024,256,10 --batch 128 \
#       --data mnist --epochs 3 --warmup 20
#   python3 -I scripts/bench/numpy_mlp.py --mode infer --net 64,256,128,10 \
#       --calls 10000 --rounds 10 --warmup 1000
#
# Training is plain SGD (learning rate --lr, default 0.01) or Adam (--optimizer adam, default
# learning rate 1e-3), softmax cross-entropy, He normal weights and zero biases, and no input
# gradient for the first layer. The data is read from --data-dir (default: data/ at the
# repository root): the four MNIST IDX files, or the digits CSV (first 1,437 rows). Batches
# are consecutive slices of the data; the last batch of an epoch is the remainder. The JSON
# line has "kind", "units" (steps per epoch, or calls per round), "samples" (seconds per
# epoch, or per round), "median", "numpy_version" and "blas_threads" (the BLAS libraries and
# their thread counts when threadpoolctl is installed, otherwise the thread environment
# variables). The timed region is the gradient step only (no data copy, loss value or checks).
import argparse
import json
import os
import time

import numpy as np


def load_mnist(directory):
    """Reads the 60,000 training images (scaled to 0..1) and labels from the IDX files."""
    with open(os.path.join(directory, "mnist", "train-images-idx3-ubyte"), "rb") as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    with open(os.path.join(directory, "mnist", "train-labels-idx1-ubyte"), "rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    x = (images.reshape(len(labels), -1) / 255.0).astype(np.float32)
    return x, labels.astype(np.int64)


def load_digits(directory):
    """Reads the first 1,437 rows of the digits CSV: 64 pixel counts (scaled by 1/16), a label."""
    rows = np.loadtxt(os.path.join(directory, "digits", "digits.csv"), delimiter=",")
    rows = rows[:1437]
    return (rows[:, :64] / 16.0).astype(np.float32), rows[:, 64].astype(np.int64)


def init_weights(sizes, rng):
    """He normal weights stored as [inputs, outputs], and zero biases."""
    w = [
        (rng.standard_normal((i, o)) * np.sqrt(2.0 / i)).astype(np.float32)
        for i, o in zip(sizes, sizes[1:])
    ]
    b = [np.zeros(o, np.float32) for o in sizes[1:]]
    return w, b


def train(args, sizes, rng):
    """Times whole epochs of training and returns (steps per epoch, seconds per epoch)."""
    x_all, y_all = (load_mnist if args.data == "mnist" else load_digits)(args.data_dir)
    n = len(y_all)
    w, b = init_weights(sizes, rng)
    lr = args.lr if args.lr is not None else (1e-3 if args.optimizer == "adam" else 0.01)
    params = w + b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    layers = len(w)

    def step(x, y):
        nonlocal t
        acts = [x]
        for j in range(layers):
            z = acts[-1] @ w[j] + b[j]
            acts.append(np.maximum(z, 0) if j < layers - 1 else z)
        z = acts[-1]
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        p[np.arange(len(y)), y] -= 1
        g = p / len(y)
        grads = [None] * (2 * layers)
        for j in range(layers - 1, -1, -1):
            grads[j] = acts[j].T @ g
            grads[layers + j] = g.sum(0)
            if j > 0:
                g = (g @ w[j].T) * (acts[j] > 0)
        t += 1
        if args.optimizer == "sgd":
            for p_, g_ in zip(params, grads):
                p_ -= lr * g_
        else:
            c1 = 1 - 0.9**t
            c2 = 1 - 0.999**t
            for p_, g_, m_, v_ in zip(params, grads, m, v):
                m_ *= 0.9
                m_ += 0.1 * g_
                v_ *= 0.999
                v_ += 0.001 * g_ * g_
                p_ -= lr * (m_ / c1) / (np.sqrt(v_ / c2) + 1e-8)

    batches = [(i, min(i + args.batch, n)) for i in range(0, n, args.batch)]
    for k in range(args.warmup):
        lo, hi = batches[k % len(batches)]
        step(x_all[lo:hi], y_all[lo:hi])
    samples = []
    for _ in range(args.epochs):
        start = time.perf_counter()
        for lo, hi in batches:
            step(x_all[lo:hi], y_all[lo:hi])
        samples.append(time.perf_counter() - start)
    return len(batches), samples


def infer(args, sizes, rng):
    """Times rounds of single-input forward passes; returns (calls per round, seconds per round)."""
    w, _ = init_weights(sizes, rng)
    x = rng.standard_normal((1, sizes[0])).astype(np.float32)
    layers = len(w)

    def forward(x):
        for j in range(layers):
            x = x @ w[j]
            if j < layers - 1:
                x = np.maximum(x, 0)
        return x

    for _ in range(args.warmup):
        forward(x)
    samples = []
    for _ in range(args.rounds):
        start = time.perf_counter()
        for _ in range(args.calls):
            forward(x)
        samples.append(time.perf_counter() - start)
    return args.calls, samples


def blas_threads():
    """Describes the BLAS thread setting: threadpoolctl's view if installed, else the env."""
    try:
        from threadpoolctl import threadpool_info

        found = [f"{i.get('internal_api')}:{i.get('num_threads')}" for i in threadpool_info()]
        if found:
            return " ".join(found)
    except ImportError:
        pass
    return " ".join(
        f"{k}={os.environ.get(k, 'unset')}"
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    )


def main():
    """Parses the arguments, runs the timing and prints the JSON line."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
    p = argparse.ArgumentParser(description="numpy MLP baseline for cavitation-bench")
    p.add_argument("--mode", choices=["train", "infer"], required=True)
    p.add_argument("--net", required=True, help="layer widths, input first: 784,1024,256,10")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--data", choices=["mnist", "digits"], default="mnist")
    p.add_argument("--data-dir", default=root)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=3, help="timed epochs (train)")
    p.add_argument("--rounds", type=int, default=10, help="timed rounds (infer)")
    p.add_argument("--calls", type=int, default=10000, help="calls per round (infer)")
    p.add_argument("--warmup", type=int, default=20, help="untimed steps (train) or calls (infer)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    sizes = [int(s) for s in args.net.split(",")]
    rng = np.random.default_rng(args.seed)
    units, samples = (train if args.mode == "train" else infer)(args, sizes, rng)
    print(
        json.dumps(
            {
                "kind": args.mode,
                "units": units,
                "samples": samples,
                "median": float(np.median(samples)),
                "numpy_version": np.__version__,
                "blas_threads": blas_threads(),
            }
        )
    )


if __name__ == "__main__":
    main()
