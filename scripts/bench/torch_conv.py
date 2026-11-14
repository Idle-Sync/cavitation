# The PyTorch side of `cavitation-bench conv`: times the four convolutions of the CIFAR-10
# CNN benchmark network (3 x 3 kernels with padding 1: 3 -> 32 and 32 -> 32 channels on
# 32 x 32 images, then 32 -> 64 and 64 -> 64 on 16 x 16) at the same batch, as
# `cavitation-bench conv` does: the median of 11 forward passes and of 11 forward and
# backward passes after 3 warm-up runs, the first layer without the input gradient, with
# the same GFLOPS count.
#
#   python3 -I scripts/bench/torch_conv.py --threads 4 --batch 128
import argparse
import statistics
import time

import torch
import torch.nn.functional as F

LAYERS = [
    ("conv1 3->32 32x32", 3, 32, 32),
    ("conv2 32->32 32x32", 32, 32, 32),
    ("conv3 32->64 16x16", 32, 64, 16),
    ("conv4 64->64 16x16", 64, 64, 16),
]


def time_layer(cin, cout, size, batch, first, samples=11):
    """Median seconds of the forward pass and of the forward and backward passes."""
    torch.manual_seed(11)
    x = (torch.rand(batch, cin, size, size) * 2 - 1).requires_grad_(not first)
    w = (torch.rand(cout, cin, 3, 3) * 2 - 1).requires_grad_(True)
    b = (torch.rand(cout) * 2 - 1).requires_grad_(True)
    up = torch.rand(batch, cout, size, size) * 2 - 1

    def run(backward):
        start = time.perf_counter()
        if backward:
            y = F.conv2d(x, w, b, padding=1)
            (y * up).sum().backward()
        else:
            with torch.no_grad():
                y = F.conv2d(x, w, b, padding=1)
        t = time.perf_counter() - start
        for t_ in (x, w, b):
            t_.grad = None
        return t

    for _ in range(3):
        run(True)
    forward = statistics.median(run(False) for _ in range(samples))
    step = statistics.median(run(True) for _ in range(samples))
    return forward, step


def main():
    parser = argparse.ArgumentParser(description="PyTorch side of cavitation-bench conv")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    print(
        f"torch conv: batch {args.batch}, torch {torch.__version__}, {args.threads} threads, "
        f"cpu {torch.backends.cpu.get_cpu_capability()}, mkldnn {torch.backends.mkldnn.is_available()}, "
        "medians of 11 runs"
    )
    print(f"{'layer':<20} {'forward ms':>12} {'GFLOPS':>9} {'step ms':>12} {'GFLOPS':>9}")
    total_f = total_s = 0.0
    for i, (name, cin, cout, size) in enumerate(LAYERS):
        first = i == 0
        fwd, step = time_layer(cin, cout, size, args.batch, first)
        flops = 2.0 * args.batch * cout * size * size * cin * 9
        passes = 2.0 if first else 3.0
        total_f += fwd
        total_s += step
        print(f"{name:<20} {fwd * 1e3:>12.2f} {flops / fwd / 1e9:>9.2f} {step * 1e3:>12.2f} {passes * flops / step / 1e9:>9.2f}")
    print(f"all four: forward {total_f * 1e3:.2f} ms, forward and backward {total_s * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
