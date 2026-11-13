# The numpy baseline of cavitation-bench: trains or runs a ReLU MLP in plain numpy and prints
# one JSON line with the timings. `cavitation-bench` runs it with the same network, batch,
# data, warm-up and step counts as its own run and compares the two.
#
#   python3 -I scripts/bench/numpy_mlp.py --mode train --net 784,1024,256,10 --batch 128 \
#       --data mnist --epochs 3 --warmup 20
#   python3 -I scripts/bench/numpy_mlp.py --mode train --net 3072,1024,512,10 --batch 128 \
#       --data cifar --optimizer adam --epochs 3 --steps 100 --warmup 20
#   python3 -I scripts/bench/numpy_mlp.py --mode infer --net 64,256,128,10 \
#       --calls 10000 --rounds 10 --warmup 1000
#
# Training is plain SGD (learning rate --lr, default 0.01) or Adam (--optimizer adam, default
# learning rate 1e-3), softmax cross-entropy, He normal weights and zero biases, and no input
# gradient for the first layer. The data is read from --data-dir (default: data/ at the
# repository root): the four MNIST IDX files, the five CIFAR-10 training batches of the
# binary version (pixel bytes in file order, divided by 255 in float32), or the digits CSV
# (first 1,437 rows). Batches are consecutive slices of the data; the last batch of an epoch
# is the remainder. With --steps N each timed sample is N steps instead of an epoch, step k
# using batch k modulo the batch count.
#
# The timed region holds the same steps as cavitation-bench's own (`Trainer::step` with
# `Health::default()`, then `Trainer::end_epoch`): for each step, the copy of the batch's
# rows and labels into a reused buffer, the forward pass, the loss value and its finite
# check, dead-ReLU tracking over each hidden layer's output, the backward pass, the update,
# and each parameter's gradient norm checked against 1e3 and 1e-7; at the end of each timed
# epoch or run, the dead-neuron report and the reset of the per-epoch state. Inference rounds
# time single-input forward passes with the bias added, as `InferenceModel::infer` does.
#
# The JSON line has "kind", "units" (steps per epoch or per timed sample, or calls per
# round), "samples" (seconds per epoch or timed sample, or per round), "median", "warnings"
# (health warnings raised in training), "numpy_version", "blas" (the BLAS numpy runs on:
# library, version, OpenBLAS core and thread count) and "blas_core" (the OpenBLAS core name,
# or "unknown"). The BLAS is read with threadpoolctl if it is installed, otherwise from
# numpy.show_config() and the OpenBLAS library's own functions through ctypes.
import argparse
import ctypes
import glob
import json
import math
import os
import time

import numpy as np

# Gradient norms above this are exploding (Health::default().grad_norm_high).
GRAD_NORM_HIGH = 1e3
# Gradient norms below this are vanishing (Health::default().grad_norm_low).
GRAD_NORM_LOW = 1e-7
# A latest loss at most this share of the first is converged: no vanishing warning.
CONVERGED_LOSS_RATIO = 1e-4


def load_mnist(directory):
    """Reads the 60,000 training images (scaled to 0..1) and labels from the IDX files."""
    with open(os.path.join(directory, "mnist", "train-images-idx3-ubyte"), "rb") as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    with open(os.path.join(directory, "mnist", "train-labels-idx1-ubyte"), "rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    x = (images.reshape(len(labels), -1) / 255.0).astype(np.float32)
    return x, labels.astype(np.int64)


def load_cifar(directory):
    """Reads the 50,000 CIFAR-10 training images (scaled to 0..1) and labels from the batches."""
    parts = []
    for i in range(1, 6):
        with open(os.path.join(directory, "cifar10", f"data_batch_{i}.bin"), "rb") as f:
            parts.append(np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 3073))
    records = np.concatenate(parts)
    x = records[:, 1:].astype(np.float32)
    x /= np.float32(255.0)
    return x, records[:, 0].astype(np.int64)


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


class Health:
    """The checks of cavitation's `Health::default()`, counting warnings instead of printing
    them: a non-finite loss or gradient stops the run; a gradient norm above GRAD_NORM_HIGH
    or below GRAD_NORM_LOW (unless the loss has converged) warns once per parameter and
    epoch; a hidden neuron that output 0 for every sample of an epoch is reported at the
    epoch's end."""

    def __init__(self, hidden_widths, parameters):
        """Per-epoch state for the hidden layers' neurons and the parameters' warnings."""
        self.alive = [np.zeros(width, dtype=bool) for width in hidden_widths]
        self.warned = [0] * parameters
        self.first_loss = None
        self.latest_loss = None
        self.negative_warned = False
        self.warnings = 0

    def after_loss(self, step, loss):
        """The loss check: stops on a NaN or infinity, records the loss, warns once per epoch
        if it is negative (cross-entropy cannot be)."""
        if not math.isfinite(loss):
            raise SystemExit(f"non-finite value at step {step} in loss")
        self.first_loss = loss if self.first_loss is None else self.first_loss
        self.latest_loss = loss
        if loss < 0.0 and not self.negative_warned:
            self.negative_warned = True
            self.warnings += 1

    def track_relu(self, layer, out):
        """Marks each neuron of hidden layer `layer` that output anything but 0 as alive."""
        np.logical_or(self.alive[layer], out.any(axis=0), out=self.alive[layer])

    def after_update(self, step, grads):
        """The gradient checks: each gradient's finiteness and its norm against the bounds,
        from its sum of squares (a float32 dot product, redone in float64 only when that
        overflows on finite values)."""
        converged = (
            self.first_loss is not None
            and self.first_loss > 0.0
            and self.latest_loss <= self.first_loss * CONVERGED_LOSS_RATIO
        )
        for k, g in enumerate(grads):
            squares = float(np.vdot(g, g))
            if not math.isfinite(squares):
                if not np.isfinite(g).all():
                    raise SystemExit(f"non-finite value at step {step} in parameter {k}")
                flat = g.ravel().astype(np.float64)
                squares = float(np.dot(flat, flat))
            norm = math.sqrt(squares)
            if norm > GRAD_NORM_HIGH:
                bit = 2
            elif norm < GRAD_NORM_LOW and not converged:
                bit = 4
            else:
                continue
            if not self.warned[k] & bit:
                self.warned[k] |= bit
                self.warnings += 1

    def end_epoch(self):
        """Reports the layers with dead neurons and starts a new epoch."""
        for alive in self.alive:
            dead = np.flatnonzero(~alive)
            if dead.size:
                self.warnings += 1
            alive.fill(False)
        self.warned = [0] * len(self.warned)
        self.negative_warned = False


def train(args, sizes, rng):
    """Times epochs (or runs of --steps steps); returns (steps per sample, seconds per
    sample, health warnings)."""
    loaders = {"mnist": load_mnist, "cifar": load_cifar, "digits": load_digits}
    x_all, y_all = loaders[args.data](args.data_dir)
    n = len(y_all)
    w, b = init_weights(sizes, rng)
    lr = args.lr if args.lr is not None else (1e-3 if args.optimizer == "adam" else 0.01)
    params = w + b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    layers = len(w)
    health = Health(sizes[1:-1], len(params))
    x_buffer = np.empty((args.batch, sizes[0]), np.float32)
    y_buffer = np.empty(args.batch, np.int64)

    def step(lo, hi):
        nonlocal t
        rows = hi - lo
        x_buffer[:rows] = x_all[lo:hi]
        y_buffer[:rows] = y_all[lo:hi]
        x, y = x_buffer[:rows], y_buffer[:rows]
        t += 1
        acts = [x]
        for j in range(layers):
            z = acts[-1] @ w[j] + b[j]
            if j < layers - 1:
                z = np.maximum(z, 0)
                health.track_relu(j, z)
            acts.append(z)
        z = acts[-1]
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        sums = p.sum(1, keepdims=True)
        picked = np.arange(rows)
        loss = float(np.mean(np.log(sums[:, 0]) - z[picked, y]))
        health.after_loss(t, loss)
        p /= sums
        p[picked, y] -= 1
        g = p / rows
        grads = [None] * (2 * layers)
        for j in range(layers - 1, -1, -1):
            grads[j] = acts[j].T @ g
            grads[layers + j] = g.sum(0)
            if j > 0:
                g = (g @ w[j].T) * (acts[j] > 0)
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
        health.after_update(t, grads)
        return loss

    batches = [(i, min(i + args.batch, n)) for i in range(0, n, args.batch)]
    for k in range(args.warmup):
        step(*batches[k % len(batches)])
    count = args.steps if args.steps is not None else len(batches)
    samples = []
    for e in range(args.epochs):
        start = time.perf_counter()
        for k in range(e * count, (e + 1) * count):
            step(*batches[k % len(batches)])
        health.end_epoch()
        samples.append(time.perf_counter() - start)
    return count, samples, health.warnings


def infer(args, sizes, rng):
    """Times rounds of single-input forward passes; returns (calls per round, seconds per
    round, 0)."""
    w, b = init_weights(sizes, rng)
    x = rng.standard_normal((1, sizes[0])).astype(np.float32)
    layers = len(w)

    def forward(x):
        for j in range(layers):
            x = x @ w[j] + b[j]
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
    return args.calls, samples, 0


def describe(library, version, core, threads, extra=None):
    """One line naming a BLAS: `openblas 0.3.27, core Haswell, 4 threads (pthreads)`."""
    parts = [f"{library} {version}" if version else library]
    if core:
        parts.append(f"core {core}")
    if threads is not None:
        parts.append(f"{threads} thread" + ("" if threads == 1 else "s"))
    text = ", ".join(parts)
    return f"{text} ({extra})" if extra else text


def blas_from_threadpoolctl():
    """The BLAS libraries threadpoolctl finds loaded, as (description, OpenBLAS core), or
    None if threadpoolctl is not installed or finds none."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return None
    found = [i for i in threadpool_info() if i.get("user_api") == "blas"]
    if not found:
        return None
    lines, core = [], None
    for i in found:
        arch = i.get("architecture")
        if i.get("internal_api") == "openblas" and arch and core is None:
            core = arch
        lines.append(
            describe(
                i.get("internal_api", "unknown"),
                i.get("version"),
                arch,
                i.get("num_threads"),
                i.get("threading_layer"),
            )
        )
    return "; ".join(lines), core


def openblas_files():
    """Paths of the OpenBLAS libraries this process has loaded (from /proc/self/maps), else
    those shipped beside numpy (its wheels put them in numpy.libs or numpy/.dylibs)."""
    try:
        with open("/proc/self/maps") as f:
            loaded = {
                line.split()[-1] for line in f if "openblas" in line.lower() and "/" in line
            }
        if loaded:
            return sorted(loaded)
    except OSError:
        pass
    here = os.path.dirname(np.__file__)
    found = []
    for folder in (os.path.join(here, "..", "numpy.libs"), os.path.join(here, ".dylibs")):
        found += glob.glob(os.path.join(folder, "*openblas*"))
    return sorted(found)


def openblas_call(lib, name, restype):
    """Calls OpenBLAS's `name` (no arguments) under the plain, `scipy_` and 64-bit-integer
    spellings the numpy wheels use; None if no spelling is exported."""
    for prefix in ("", "scipy_"):
        for suffix in ("", "64_", "_64_"):
            func = getattr(lib, f"{prefix}{name}{suffix}", None)
            if func is not None:
                func.restype = restype
                func.argtypes = []
                return func()
    return None


def blas_from_numpy():
    """The BLAS as numpy.show_config() names it, with the OpenBLAS core and thread count read
    from the library through ctypes; (description, core or None)."""
    library, version, config = "unknown", None, None
    try:
        deps = np.show_config(mode="dicts").get("Build Dependencies", {})
        blas = deps.get("blas", {})
        library = blas.get("name", library)
        version = blas.get("version")
        config = blas.get("openblas configuration")
    except (TypeError, AttributeError, ValueError):
        pass
    core = threads = None
    for path in openblas_files():
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        raw = openblas_call(lib, "openblas_get_corename", ctypes.c_char_p)
        core = raw.decode() if raw else None
        threads = openblas_call(lib, "openblas_get_num_threads", ctypes.c_int)
        if config is None:
            raw = openblas_call(lib, "openblas_get_config", ctypes.c_char_p)
            config = raw.decode().strip() if raw else None
        if core or threads is not None:
            break
    notes = [config] if config else []
    if threads is None:
        notes += [
            f"{k}={os.environ[k]}"
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            if k in os.environ
        ]
    return describe(library, version, core, threads, "; ".join(notes) or None), core


def blas_info():
    """The BLAS numpy runs on, as (description, OpenBLAS core or "unknown")."""
    found = blas_from_threadpoolctl() or blas_from_numpy()
    return found[0], found[1] or "unknown"


def main():
    """Parses the arguments, runs the timing and prints the JSON line."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
    p = argparse.ArgumentParser(description="numpy MLP baseline for cavitation-bench")
    p.add_argument("--mode", choices=["train", "infer"], required=True)
    p.add_argument("--net", required=True, help="layer widths, input first: 784,1024,256,10")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--data", choices=["mnist", "cifar", "digits"], default="mnist")
    p.add_argument("--data-dir", default=root)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=3, help="timed epochs or runs (train)")
    p.add_argument("--steps", type=int, default=None, help="steps per timed run (train)")
    p.add_argument("--rounds", type=int, default=10, help="timed rounds (infer)")
    p.add_argument("--calls", type=int, default=10000, help="calls per round (infer)")
    p.add_argument("--warmup", type=int, default=20, help="untimed steps (train) or calls (infer)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    sizes = [int(s) for s in args.net.split(",")]
    rng = np.random.default_rng(args.seed)
    units, samples, warnings = (train if args.mode == "train" else infer)(args, sizes, rng)
    blas, core = blas_info()
    print(
        json.dumps(
            {
                "kind": args.mode,
                "units": units,
                "samples": samples,
                "median": float(np.median(samples)),
                "warnings": warnings,
                "numpy_version": np.__version__,
                "blas": blas,
                "blas_core": core,
            }
        )
    )


if __name__ == "__main__":
    main()
