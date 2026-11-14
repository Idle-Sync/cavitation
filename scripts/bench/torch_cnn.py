# The PyTorch side of `cavitation-bench mnist-cnn` and `cifar-cnn`: trains the same v0.3
# benchmark network from the initial weights and the batch order the Rust side wrote, with
# Adam at lr 1e-3 and cross-entropy, and prints one line per epoch:
#
#   epoch <n> <mean loss> <test accuracy from 0 to 1> <seconds of the training steps>
#
# The test pass after each epoch runs in evaluation mode and is not timed. Lines starting
# with `torch` describe the run (version, threads, CPU capability). The data are read from
# the same files the Rust side reads (MNIST's IDX files, CIFAR-10's binary batches), each
# pixel divided by 255.
#
#   python3 -I scripts/bench/torch_cnn.py --mode train --net cifar --data-dir data/cifar10 \
#       --init target/bench/cnn/cifar-cnn-seed0/init.safetensors \
#       --order target/bench/cnn/cifar-cnn-seed0/order.bin --epochs 10 --batch 128 \
#       --rows 50000 --threads 4 --seed 0
import argparse
import os
import struct
import time

import numpy as np
import torch
from safetensors.torch import load_file

nn = torch.nn


def mnist_net():
    """LeNet-5's layout, as `cnn::mnist_network` builds it."""
    return nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
        nn.Linear(400, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10),
    )


def cifar_net():
    """The batch-norm CIFAR-10 network, as `cnn::cifar_network` builds it."""

    def block(i, o):
        return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU()]

    return nn.Sequential(
        *block(3, 32), *block(32, 32), nn.MaxPool2d(2),
        *block(32, 64), *block(64, 64), nn.MaxPool2d(2),
        nn.Dropout(0.25), nn.Flatten(), nn.Linear(4096, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 10),
    )


def read_idx(path):
    """The array in an uncompressed IDX file."""
    with open(path, "rb") as f:
        data = f.read()
    _, kind, dims = struct.unpack(">HBB", data[:4])
    if kind != 0x08:
        raise ValueError(f"{path}: not unsigned bytes")
    shape = struct.unpack(">" + "I" * dims, data[4 : 4 + 4 * dims])
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)


def load_mnist(folder):
    """MNIST's train and test images as [n, 1, 28, 28] floats in [0, 1] and their labels."""

    def split(prefix):
        x = read_idx(os.path.join(folder, f"{prefix}-images-idx3-ubyte"))
        y = read_idx(os.path.join(folder, f"{prefix}-labels-idx1-ubyte"))
        x = torch.from_numpy(x.astype(np.float32) / np.float32(255)).reshape(-1, 1, 28, 28)
        return x, torch.from_numpy(y.astype(np.int64))

    return split("train"), split("t10k")


def load_cifar(folder):
    """CIFAR-10's train and test images as [n, 3, 32, 32] floats in [0, 1] and labels."""

    def files(names):
        raw = np.concatenate(
            [np.fromfile(os.path.join(folder, n), dtype=np.uint8).reshape(-1, 3073) for n in names]
        )
        x = torch.from_numpy(raw[:, 1:].astype(np.float32) / np.float32(255)).reshape(-1, 3, 32, 32)
        return x, torch.from_numpy(raw[:, 0].astype(np.int64))

    train = files([f"data_batch_{i}.bin" for i in range(1, 6)])
    return train, files(["test_batch.bin"])


def accuracy(model, x, y):
    """The share of `x` the model classifies as `y`, in evaluation mode, 500 at a time."""
    model.eval()
    right = 0
    with torch.no_grad():
        for i in range(0, len(x), 500):
            right += (model(x[i : i + 500]).argmax(1) == y[i : i + 500]).sum().item()
    model.train()
    return right / len(x)


def main():
    parser = argparse.ArgumentParser(description="PyTorch side of cavitation-bench mnist-cnn and cifar-cnn")
    parser.add_argument("--mode", choices=["train"], required=True)
    parser.add_argument("--net", choices=["mnist", "cifar"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--init", required=True, help="the initial weights, as the Rust side saved them")
    parser.add_argument("--order", required=True, help="each epoch's row order, little-endian u32s")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True, help="train on the first this many rows")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="seeds the dropout masks")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    (x, y), (tx, ty) = (load_cifar if args.net == "cifar" else load_mnist)(args.data_dir)
    x, y = x[: args.rows], y[: args.rows]
    model = cifar_net() if args.net == "cifar" else mnist_net()
    state = load_file(args.init)
    model.load_state_dict(state, strict=True)
    order = np.fromfile(args.order, dtype="<u4").astype(np.int64)
    if len(order) != args.epochs * args.rows:
        raise ValueError(f"{args.order}: {len(order)} entries, not {args.epochs} epochs of {args.rows}")
    order = torch.from_numpy(order).reshape(args.epochs, args.rows)
    print(
        f"torch {torch.__version__}, {torch.get_num_threads()} threads, "
        f"cpu {torch.backends.cpu.get_cpu_capability()}, mkldnn {torch.backends.mkldnn.is_available()}",
        flush=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.epochs):
        total, steps = 0.0, 0
        start = time.perf_counter()
        for i in range(0, args.rows, args.batch):
            idx = order[epoch, i : i + args.batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
            total += loss.item()
            steps += 1
        seconds = time.perf_counter() - start
        acc = accuracy(model, tx, ty)
        print(f"epoch {epoch + 1} {total / steps:.6f} {acc:.6f} {seconds:.6f}", flush=True)


if __name__ == "__main__":
    main()
