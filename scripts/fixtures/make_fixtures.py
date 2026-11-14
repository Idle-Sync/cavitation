"""Writes the PyTorch reference fixtures that the Rust tests compare against.

For every operation on the autograd tape it writes one JSON file per case under
`crates/cavitation-core/tests/fixtures/ops/` (hyphenated names, for example
`softmax-cross-entropy-mean.json`). A file holds the operation and its parameters, the f32
inputs, the upstream gradient, PyTorch's f32 output and input gradients (the gradients of
`sum(output * upstream)`), and the same computation in float64 (`output_f64`, `grads_f64`).
Operations that sum are compared with the float64 result, because PyTorch's f32 sums use a
different order than ours.

Layer fixtures (`nn.Linear`) go under `crates/cavitation-nn/tests/fixtures/layers/` in the same
form, with the layer's weights as inputs after the data input.

The state-dict fixture goes under `crates/cavitation-nn/tests/fixtures/state-dict/`: the
`state_dict` of `nn.Sequential(Linear(4, 8), ReLU(), Linear(8, 3))` (seed 0) saved with
`safetensors.torch.save_file` as `mlp.safetensors`, and `mlp-io.json` with five inputs and the
network's outputs in f32 and float64. `--only state-dict` writes just these two files.

The bias-free fixture goes in the same folder: the `state_dict` of
`nn.Sequential(Linear(4, 8, bias=False), ReLU(), Linear(8, 3))` (seed 0, so its keys are
`0.weight`, `2.weight` and `2.bias`) saved as `mlp-no-bias.safetensors`, and
`mlp-no-bias-io.json` with five inputs, an upstream gradient, the network's outputs and the
gradients of `sum(output * upstream)` with respect to each parameter, in f32 and float64.
Every hidden pre-activation is at least 1e-3 from ReLU's kink, so both sides take the same
branch. After writing, the script loads the file back into a new network with
`load_state_dict(strict=True)` and requires the same outputs. `--only no-bias` writes just
these two files.

`--check-rust-save [PATH]` writes nothing. It checks the other direction: that a file written by the
library's `save` loads in PyTorch. The Rust test `save_writes_the_file_pytorch_was_checked_against`
(in `crates/cavitation-nn/src/io.rs`) keeps `from-rust.safetensors` in the same folder equal,
byte for byte, to what `save` writes today, and `from-rust-io.json` beside it holds that model's
parameter values, five inputs and the library's outputs on them. The check loads the file with
`safetensors.torch.load_file`, loads it into `nn.Sequential(Linear(4, 8), ReLU(), Linear(8, 3))`
with `load_state_dict(strict=True)`, requires every tensor to be F32 and bit-for-bit the stored
values, runs the network and requires both PyTorch's f32 outputs and the library's to be within
`2 * (k + 2) * 1.19e-7 * sum|terms|` of the float64 result (`k` terms per layer, the bias
included, and the first layer's bound carried through the second layer's weights). To rewrite
the two files after a deliberate change to `save`, run
`CAVITATION_WRITE_FIXTURES=1 cargo test --release -p cavitation-nn --lib save_writes_the_file`,
then this check, run from the repository root (`--out` defaults to `crates`).

The same check runs on a second pair, `from-rust-no-bias.safetensors` and
`from-rust-no-bias-io.json`, which the Rust test
`save_writes_the_bias_free_file_pytorch_was_checked_against` keeps equal to what `save` writes
for `Linear(4, 8, bias=False), ReLU, Linear(8, 3)`: the file must hold only `0.weight` for the
first layer and load with strict keys into that PyTorch network. Rewrite it with
`CAVITATION_WRITE_FIXTURES=1 cargo test --release -p cavitation-nn --lib save_writes_the_bias_free_file`.
The command:

    scripts/fixtures/.venv/bin/python -I scripts/fixtures/make_fixtures.py --check-rust-save

Give a path to check another copy of the file (for example one altered on purpose, to see the
check fail); `from-rust-io.json` is still read from the committed folder. Every check exits
with status 1 and a message, also under `python -O`.

Inputs come from `torch.manual_seed(0)` with one thread, in a fixed order, so a rerun writes
identical files. Inputs are kept away from kinks (zero for ReLU, the clamp edges, the points
where Huber and smooth L1 change branch) so both sides take the same branch.

PyTorch has no `batchmean` reduction except for `kl_div`. For every other loss the fixture
for `batchmean` is computed here as the `sum` reduction divided by the input's first
dimension, and its `params` say so.

Run it from the repository root, inside the virtual environment in `scripts/fixtures/.venv`
(created with `uv venv scripts/fixtures/.venv --python 3.14` and filled from
`scripts/fixtures/requirements.txt`, which picks the CPU build of PyTorch on Linux and the
standard build on macOS):

    scripts/fixtures/.venv/bin/python -I scripts/fixtures/make_fixtures.py --out crates

Floats are written with 9 significant digits for f32 data and 17 for float64 data, enough to
read back the exact values.
"""

import argparse
import copy
import json
import math
import os
import sys
import warnings

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

F32_DIGITS = 9
F64_DIGITS = 17
SHAPES = [(1, 1), (3, 5), (7, 1), (16, 33)]
REDUCTIONS = ["mean", "sum", "none", "batchmean"]

# PyTorch warns that kl_div's 'mean' will change; the fixtures use today's meaning, the
# element mean, which is what the library's Mean does.
warnings.filterwarnings("ignore", message="reduction: 'mean' divides")

torch.set_num_threads(1)
torch.manual_seed(0)

TORCH_VERSION = torch.__version__.split("+")[0]
CASES = []
LAYER_CASES = []
OPTIM_CASES = []


def check(condition, message):
    """Exits with status 1 and `message` unless `condition` holds. Unlike `assert`, it still
    runs under `python -O`."""
    if not condition:
        raise SystemExit(message)


def rnd(*shape, scale=1.0):
    """Normal values times `scale`, as f32."""
    return torch.randn(shape) * scale


def uniform(shape, lo, hi):
    """Uniform values in [lo, hi), as f32."""
    return torch.rand(shape) * (hi - lo) + lo


def away(x, points, margin=0.05):
    """Moves every element of `x` that is within `margin` of a point in `points` to `margin * 2` beyond it."""
    x = x.clone()
    for p in points:
        near = (x - p).abs() < margin
        side = torch.where(x >= p, 1.0, -1.0)
        x = torch.where(near, p + side * margin * 2, x)
    for p in points:
        check(bool(((x - p).abs() >= margin).all()), "an input is on a kink")
    return x


def away_diff(pred, target, points, margin=0.05):
    """Moves `pred` so that `pred - target` is away from each of `points`."""
    return target + away(pred - target, points, margin)


def number(value, digits):
    """A float rounded to `digits` significant digits, as a JSON number."""
    return float(format(float(value), f".{digits}g"))


def tensor_json(t, digits):
    """The `{shape, data}` form of a tensor."""
    return {
        "shape": list(t.shape),
        "data": [number(v, digits) for v in t.detach().reshape(-1).tolist()],
    }


def evaluate(fn, inputs, dtype, upstream=None):
    """Runs `fn` on `inputs` cast to `dtype`; returns the output, upstream and gradients."""
    xs = [x.to(dtype).clone().requires_grad_(True) for x in inputs]
    out = fn(*xs)
    if upstream is None:
        upstream = torch.randn(out.shape)
    loss = (out * upstream.to(dtype)).sum()
    loss.backward()
    grads = [x.grad if x.grad is not None else torch.zeros_like(x) for x in xs]
    return out.detach(), upstream, grads


def add_case(name, op, params, inputs, fn, cases=None):
    """Evaluates one case in f32 and float64 and records it (in `cases`, `CASES` by default)."""
    cases = CASES if cases is None else cases
    inputs = [x.float() for x in inputs]
    out, upstream, grads = evaluate(fn, inputs, torch.float32)
    out64, _, grads64 = evaluate(fn, inputs, torch.float64, upstream)
    cases.append(
        (
            name,
            {
                "torch_version": TORCH_VERSION,
                "op": op,
                "params": params,
                "inputs": [tensor_json(x, F32_DIGITS) for x in inputs],
                "upstream": tensor_json(upstream, F32_DIGITS),
                "output": tensor_json(out, F32_DIGITS),
                "grads": [tensor_json(g, F32_DIGITS) for g in grads],
                "output_f64": tensor_json(out64, F64_DIGITS),
                "grads_f64": [tensor_json(g, F64_DIGITS) for g in grads64],
            },
        )
    )


def shape_name(shape):
    """`16x33` for `(16, 33)`."""
    return "x".join(str(d) for d in shape)


def hyphen(op):
    """File-name form of an operation name."""
    return op.replace("_", "-")


TINY_NEGATIVES = [-1e-10, -1e-5, -0.0, -1e-40]


def unary(
    op, fn, params=None, shape=(16, 33), scale=2.5, kinks=(), suffix="", margin=0.05, tiny=False
):
    """A case of an operation of one input.

    With `tiny`, the first four elements are replaced by `TINY_NEGATIVES` (a value near zero,
    negative zero and an f32 subnormal), where `exp(x) - 1` loses precision. They replace
    values instead of being appended, so the random stream, and with it every other case, is
    unchanged.
    """
    x = away(rnd(*shape, scale=scale), kinks, margin) if kinks else rnd(*shape, scale=scale)
    if tiny:
        x.view(-1)[: len(TINY_NEGATIVES)] = torch.tensor(TINY_NEGATIVES)
    add_case(hyphen(op) + suffix, op, params or {}, [x], fn)


def make_arithmetic():
    """add, sub, mul (including broadcasting) and scale."""
    ops = {"add": torch.add, "sub": torch.sub, "mul": torch.mul}
    pairs = [((3, 4), (4,)), ((3, 1), (1, 4)), ((), (5,)), ((5,), ())]
    for op, fn in ops.items():
        for shape in SHAPES:
            add_case(f"{op}-{shape_name(shape)}", op, {}, [rnd(*shape), rnd(*shape)], fn)
        for a, b in pairs:
            name = f"{op}-broadcast-{shape_name(a) or 'scalar'}-with-{shape_name(b) or 'scalar'}"
            add_case(name, op, {}, [rnd(*a), rnd(*b)], fn)
    s = 0.37
    add_case("scale", "scale", {"s": s}, [rnd(16, 33)], lambda x: x * s)


def make_matmul():
    """matmul and matmul_nt over the standard shapes."""
    for m, k, n in [(1, 1, 1), (3, 5, 4), (7, 1, 6), (16, 33, 16)]:
        add_case(f"matmul-{m}x{k}x{n}", "matmul", {}, [rnd(m, k), rnd(k, n)], torch.matmul)
        add_case(
            f"matmul-nt-{m}x{k}x{n}",
            "matmul_nt",
            {},
            [rnd(m, k), rnd(n, k)],
            lambda a, b: a @ b.T,
        )


def make_reductions():
    """sum, mean, sum_axis and mean_axis."""
    for shape in SHAPES:
        add_case(f"sum-{shape_name(shape)}", "sum", {}, [rnd(*shape)], torch.sum)
        add_case(f"mean-{shape_name(shape)}", "mean", {}, [rnd(*shape)], torch.mean)
    for op, fn in [("sum_axis", torch.sum), ("mean_axis", torch.mean)]:
        for shape, axis, keep in [
            ((3, 5), 0, False),
            ((3, 5), 1, True),
            ((16, 33), 1, False),
            ((16, 33), 0, True),
        ]:
            name = f"{hyphen(op)}-{shape_name(shape)}-axis{axis}-{'keep' if keep else 'drop'}"
            params = {"axis": axis, "keep_dim": keep}
            add_case(
                name, op, params, [rnd(*shape)], lambda x, a=axis, k=keep, f=fn: f(x, dim=a, keepdim=k)
            )


def make_activations():
    """Every activation function, prelu and glu."""
    for shape in SHAPES:
        sfx = "-" + shape_name(shape)
        unary("relu", torch.relu, shape=shape, kinks=[0.0], suffix=sfx)
        unary("tanh", torch.tanh, shape=shape, suffix=sfx)
        unary("sigmoid", torch.sigmoid, shape=shape, suffix=sfx)
    slope = 0.1
    unary("leaky_relu", lambda x: F.leaky_relu(x, slope), {"slope": slope}, kinks=[0.0])
    unary(
        "gelu", lambda x: F.gelu(x, approximate="none"), {"approximate": "none"}, suffix="-exact"
    )
    unary(
        "gelu", lambda x: F.gelu(x, approximate="tanh"), {"approximate": "tanh"}, suffix="-tanh"
    )
    alpha = 1.3
    unary("elu", lambda x: F.elu(x, alpha), {"alpha": alpha}, kinks=[0.0], tiny=True)
    unary("selu", F.selu, kinks=[0.0], tiny=True)
    alpha = 0.7
    unary("celu", lambda x: F.celu(x, alpha), {"alpha": alpha}, kinks=[0.0], tiny=True)
    unary("silu", F.silu)
    unary("mish", F.mish)
    beta, threshold = 1.5, 20.0
    unary(
        "softplus",
        lambda x: F.softplus(x, beta, threshold),
        {"beta": beta, "threshold": threshold},
    )
    # A low threshold, so the linear branch is taken for part of the inputs.
    beta, threshold = 2.0, 3.0
    unary(
        "softplus",
        lambda x: F.softplus(x, beta, threshold),
        {"beta": beta, "threshold": threshold},
        kinks=[threshold / beta],
        suffix="-threshold",
    )
    unary("softsign", F.softsign)
    lo, hi = -0.5, 0.8
    unary(
        "hardtanh",
        lambda x: F.hardtanh(x, lo, hi),
        {"min": lo, "max": hi},
        scale=1.2,
        kinks=[lo, hi],
    )
    unary("relu6", F.relu6, scale=3.0, kinks=[0.0, 6.0])
    unary("hardsigmoid", F.hardsigmoid, kinks=[-3.0, 3.0])
    unary("hardswish", F.hardswish, kinks=[-3.0, 3.0])
    unary("log_sigmoid", F.logsigmoid)
    x = away(rnd(16, 33, scale=2.0), [0.0])
    add_case("prelu-1", "prelu", {}, [x, uniform((1,), 0.05, 0.5)], F.prelu)
    x = away(rnd(16, 33, scale=2.0), [0.0])
    add_case("prelu-channels", "prelu", {}, [x, uniform((33,), 0.05, 0.5)], F.prelu)

    def glu_with(gate):
        def f(x):
            a, b = x.chunk(2, dim=-1)
            return a * gate(b)

        return f

    gates = {
        "sigmoid": lambda x: F.glu(x, dim=-1),
        "gelu_exact": glu_with(lambda b: F.gelu(b, approximate="none")),
        "gelu_tanh": glu_with(lambda b: F.gelu(b, approximate="tanh")),
        "relu": glu_with(F.relu),
        "silu": glu_with(F.silu),
    }
    for gate, fn in gates.items():
        x = rnd(7, 10, scale=2.0)
        if gate == "relu":
            half = x.shape[-1] // 2
            x = torch.cat([x[..., :half], away(x[..., half:], [0.0])], dim=-1)
        add_case(f"glu-{gate.replace('_', '-')}", "glu", {"gate": gate}, [x], fn)


def make_softmax():
    """softmax and log_softmax."""
    for shape in SHAPES:
        add_case(
            f"softmax-{shape_name(shape)}",
            "softmax",
            {},
            [rnd(*shape, scale=2.0)],
            lambda x: F.softmax(x, dim=-1),
        )
        add_case(
            f"log-softmax-{shape_name(shape)}",
            "log_softmax",
            {},
            [rnd(*shape, scale=2.0)],
            lambda x: F.log_softmax(x, dim=-1),
        )


def reduced(call, reduction, first_dim, native_batchmean=False):
    """Applies `call(reduction=...)`; `batchmean` is `sum / first_dim` unless PyTorch has it."""
    if reduction == "batchmean" and not native_batchmean:
        return call(reduction="sum") / first_dim
    return call(reduction=reduction)


def loss_params(reduction, extra=None, native_batchmean=False):
    """The `params` of a loss fixture."""
    params = dict(extra or {})
    params["reduction"] = reduction
    if reduction == "batchmean" and not native_batchmean:
        params["torch_reduction"] = "sum"
        params["divisor"] = "sum divided by the first dimension of the input, computed in the script"
    return params


def add_loss(op, reduction, inputs, call, extra=None, native_batchmean=False):
    """Records one loss under one reduction; `call(*inputs, reduction=...)` is PyTorch's loss."""
    first_dim = inputs[0].shape[0]
    params = loss_params(reduction, extra, native_batchmean)
    name = f"{hyphen(op)}-{reduction}"
    add_case(
        name,
        op,
        params,
        inputs,
        lambda *xs: reduced(lambda reduction: call(*xs, reduction=reduction), reduction, first_dim, native_batchmean),
    )


def make_losses():
    """Every loss under every reduction."""
    batch, classes = 6, 5
    for r in REDUCTIONS:
        labels = torch.randint(0, classes, (batch,))
        add_loss(
            "softmax_cross_entropy",
            r,
            [rnd(batch, classes, scale=2.0)],
            lambda z, reduction, y=labels: F.cross_entropy(z, y, reduction=reduction),
            {"labels": labels.tolist()},
        )
    for r in REDUCTIONS:
        labels = torch.randint(0, classes, (batch,))
        add_loss(
            "nll",
            r,
            [F.log_softmax(rnd(batch, classes, scale=2.0), dim=-1)],
            lambda lp, reduction, y=labels: F.nll_loss(lp, y, reduction=reduction),
            {"labels": labels.tolist()},
        )
    shape = (4, 7)
    for r in REDUCTIONS:
        add_loss("mse", r, [rnd(*shape), rnd(*shape)], F.mse_loss)
    for r in REDUCTIONS:
        t = rnd(*shape)
        add_loss("l1", r, [away_diff(rnd(*shape), t, [0.0]), t], F.l1_loss)
    delta = 0.7
    for r in REDUCTIONS:
        t = rnd(*shape)
        p = away_diff(rnd(*shape, scale=1.5), t, [-delta, delta])
        add_loss(
            "huber",
            r,
            [p, t],
            lambda p, t, reduction, d=delta: F.huber_loss(p, t, reduction=reduction, delta=d),
            {"delta": delta},
        )
    beta = 0.6
    for r in REDUCTIONS:
        t = rnd(*shape)
        p = away_diff(rnd(*shape, scale=1.5), t, [-beta, beta])
        add_loss(
            "smooth_l1",
            r,
            [p, t],
            lambda p, t, reduction, b=beta: F.smooth_l1_loss(p, t, reduction=reduction, beta=b),
            {"beta": beta},
        )
    for r in REDUCTIONS:
        add_loss(
            "binary_cross_entropy",
            r,
            [uniform(shape, 0.05, 0.95), uniform(shape, 0.0, 1.0)],
            F.binary_cross_entropy,
        )
    for r in REDUCTIONS:
        add_loss(
            "binary_cross_entropy_with_logits",
            r,
            [rnd(*shape, scale=3.0), uniform(shape, 0.0, 1.0)],
            F.binary_cross_entropy_with_logits,
        )
    for r in REDUCTIONS:
        add_loss(
            "kl_div",
            r,
            [F.log_softmax(rnd(*shape), dim=-1), F.softmax(rnd(*shape), dim=-1)],
            F.kl_div,
            native_batchmean=True,
        )


def make_layers():
    """`nn.Linear(7, 5)` on a [3, 7] input, seeded 0 independently of the cases above.

    The inputs are the layer input, its weight ([out, in]) and its bias, in that order; the
    output is `x @ weight.T + bias`.
    """
    torch.manual_seed(0)
    layer = torch.nn.Linear(7, 5)
    x = rnd(3, 7)
    add_case(
        "linear",
        "linear",
        {"inputs": 7, "outputs": 5},
        [x, layer.weight.detach(), layer.bias.detach()],
        lambda x, w, b: F.linear(x, w, b),
        cases=LAYER_CASES,
    )


def make_optimizers():
    """Ten steps of each optimiser on a [4, 3] weight and a [3] bias.

    The gradients are fixed per step (drawn from a generator seeded 0, independent of the
    cases above) and set directly, so the fixture holds exactly the numbers `torch.optim`
    was given: the starting values, the ten gradients and the parameters after each step.
    """
    configs = [
        ("sgd", torch.optim.SGD, {"lr": 0.1}),
        ("sgd-momentum", torch.optim.SGD, {"lr": 0.1, "momentum": 0.9}),
        ("adam", torch.optim.Adam, {"lr": 0.01}),
        ("adamw", torch.optim.AdamW, {"lr": 0.01, "weight_decay": 0.01}),
        ("adam-weight-decay", torch.optim.Adam, {"lr": 0.01, "weight_decay": 0.1}),
        ("sgd-weight-decay", torch.optim.SGD, {"lr": 0.1, "weight_decay": 0.05}),
        (
            "sgd-momentum-weight-decay",
            torch.optim.SGD,
            {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.05},
        ),
    ]
    steps = 10
    for name, cls, kwargs in configs:
        gen = torch.Generator().manual_seed(0)
        weight = torch.nn.Parameter(torch.randn(4, 3, generator=gen))
        bias = torch.nn.Parameter(torch.randn(3, generator=gen))
        init = {"weight": tensor_json(weight, F32_DIGITS), "bias": tensor_json(bias, F32_DIGITS)}
        opt = cls([weight, bias], **kwargs)
        grads, after = [], []
        for _ in range(steps):
            gw = torch.randn(4, 3, generator=gen)
            gb = torch.randn(3, generator=gen)
            # The fixture stores the rounded gradients, so torch must use the same values.
            gw = torch.tensor(tensor_json(gw, F32_DIGITS)["data"], dtype=torch.float32).reshape(4, 3)
            gb = torch.tensor(tensor_json(gb, F32_DIGITS)["data"], dtype=torch.float32)
            weight.grad, bias.grad = gw.clone(), gb.clone()
            opt.step()
            grads.append({"weight": tensor_json(gw, F32_DIGITS), "bias": tensor_json(gb, F32_DIGITS)})
            after.append({"weight": tensor_json(weight, F32_DIGITS), "bias": tensor_json(bias, F32_DIGITS)})
        OPTIM_CASES.append(
            (
                name,
                {
                    "torch_version": TORCH_VERSION,
                    "optimizer": cls.__name__,
                    "params": kwargs,
                    "initial": init,
                    "grads": grads,
                    "after": after,
                },
            )
        )


def make_select():
    """keep_where: the input where `keep` is non-zero and 0 elsewhere, with `keep` a row
    broadcast over the rows, a value of the input's shape, and a scalar.

    Built last, so its random draws do not shift the inputs of the cases before it.
    """

    def keep_where(x, keep):
        return torch.where(keep != 0, x, torch.zeros((), dtype=x.dtype))

    row = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
    add_case("keep-where-3x5-with-5", "keep_where", {}, [rnd(3, 5), row], keep_where)
    full = (torch.rand(16, 33) < 0.5).float()
    add_case("keep-where-16x33", "keep_where", {}, [rnd(16, 33), full], keep_where)
    add_case("keep-where-7x1-with-scalar", "keep_where", {}, [rnd(7, 1), torch.tensor(0.0)], keep_where)


def write_state_dict(crates_dir):
    """Saves a seeded `Linear(4, 8), ReLU(), Linear(8, 3)` and its outputs for five inputs."""
    out_dir = os.path.join(crates_dir, "cavitation-nn", "tests", "fixtures", "state-dict")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
    save_file(model.state_dict(), os.path.join(out_dir, "mlp.safetensors"))
    x = rnd(5, 4)
    with torch.no_grad():
        out32 = model(x)
        out64 = model.double()(x.double())
    body = {
        "torch_version": TORCH_VERSION,
        "keys": list(model.state_dict().keys()),
        "input": tensor_json(x, F32_DIGITS),
        "output": tensor_json(out32, F32_DIGITS),
        "output_f64": tensor_json(out64, F64_DIGITS),
    }
    with open(os.path.join(out_dir, "mlp-io.json"), "w") as f:
        json.dump(body, f, separators=(",", ":"))
        f.write("\n")
    print(f"wrote the state-dict fixture to {out_dir} with torch {TORCH_VERSION}", file=sys.stderr)


def write_no_bias_network(crates_dir):
    """Saves a seeded `Linear(4, 8, bias=False), ReLU(), Linear(8, 3)`, with its outputs and
    parameter gradients for five inputs and an upstream gradient, then checks that the file
    loads back strictly and gives the same outputs."""
    out_dir = os.path.join(crates_dir, "cavitation-nn", "tests", "fixtures", "state-dict")
    os.makedirs(out_dir, exist_ok=True)

    def network():
        return torch.nn.Sequential(
            torch.nn.Linear(4, 8, bias=False), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        )

    torch.manual_seed(0)
    model = network()
    x = rnd(5, 4)
    upstream = rnd(5, 3)
    keys = list(model.state_dict().keys())
    check(keys == ["0.weight", "2.weight", "2.bias"], f"unexpected keys {keys}")
    pre = x.double() @ model[0].weight.double().T
    check(bool((pre.abs() >= 1e-3).all()), "a hidden pre-activation is within 1e-3 of ReLU's kink")

    def run(dtype):
        net = copy.deepcopy(model).to(dtype)
        out = net(x.to(dtype))
        (out * upstream.to(dtype)).sum().backward()
        grads = {name: p.grad.detach() for name, p in net.named_parameters()}
        return out.detach(), grads

    out32, grads32 = run(torch.float32)
    out64, grads64 = run(torch.float64)
    path = os.path.join(out_dir, "mlp-no-bias.safetensors")
    save_file(model.state_dict(), path)
    body = {
        "torch_version": TORCH_VERSION,
        "keys": keys,
        "input": tensor_json(x, F32_DIGITS),
        "upstream": tensor_json(upstream, F32_DIGITS),
        "output": tensor_json(out32, F32_DIGITS),
        "grads": {k: tensor_json(grads32[k], F32_DIGITS) for k in keys},
        "output_f64": tensor_json(out64, F64_DIGITS),
        "grads_f64": {k: tensor_json(grads64[k], F64_DIGITS) for k in keys},
    }
    with open(os.path.join(out_dir, "mlp-no-bias-io.json"), "w") as f:
        json.dump(body, f, separators=(",", ":"))
        f.write("\n")

    fresh = network()
    fresh.load_state_dict(load_file(path), strict=True)
    with torch.no_grad():
        again = fresh(x)
    check(torch.equal(again, out32), "the saved bias-free network gives other outputs")
    print(
        f"wrote the bias-free fixture to {out_dir} with torch {TORCH_VERSION}; it loads back "
        f"with strict keys {keys} and gives the same outputs",
        file=sys.stderr,
    )


def mlp_bound(model, x):
    """Per-output bound on the distance of an f32 forward pass of `model` (Linear, ReLU, Linear)
    on `x` from the float64 result: `2 * (k + 2) * 1.19e-7 * sum|terms|` for each layer, with
    `k` its terms (inputs plus the bias), and the first layer's bound carried through the
    second layer's absolute weights."""
    eps = 1.19e-7
    w1 = model[0].weight.double()
    # A layer without a bias adds no term: its bias counts as 0.
    b1 = model[0].bias.double() if model[0].bias is not None else torch.zeros(w1.shape[0], dtype=torch.float64)
    w2, b2 = model[2].weight.double(), model[2].bias.double()
    x = x.double()
    hidden_terms = (x.abs() @ w1.abs().T) + b1.abs()
    hidden_err = 2 * (w1.shape[1] + 3) * eps * hidden_terms
    h = torch.relu(x @ w1.T + b1)
    out_terms = (h.abs() @ w2.abs().T) + b2.abs()
    return 2 * (w2.shape[1] + 3) * eps * out_terms + hidden_err @ w2.abs().T + 1e-30


def check_rust_save(crates_dir, saved=None):
    """Checks both library-written files: `from-rust.safetensors` (or the file `saved`) in
    `Linear(4, 8), ReLU, Linear(8, 3)`, then `from-rust-no-bias.safetensors` in
    `Linear(4, 8, bias=False), ReLU, Linear(8, 3)`; exits with a message on any difference."""
    check_rust_file(crates_dir, "from-rust", True, saved)
    check_rust_file(crates_dir, "from-rust-no-bias", False)


def check_rust_file(crates_dir, stem, first_bias, saved=None):
    """Loads the library-written safetensors file `{stem}.safetensors` (or the file `saved`)
    into the PyTorch network whose first `Linear` has a bias exactly when `first_bias` is
    true, and checks its keys, values and outputs against the committed `{stem}-io.json`;
    exits with a message on any difference."""
    in_dir = os.path.join(crates_dir, "cavitation-nn", "tests", "fixtures", "state-dict")
    state = load_file(saved or os.path.join(in_dir, stem + ".safetensors"))
    with open(os.path.join(in_dir, stem + "-io.json")) as f:
        io = json.load(f)

    def tensor(t):
        return torch.tensor(t["data"], dtype=torch.float32).reshape(t["shape"])

    check(sorted(state) == sorted(io["keys"]), f"keys {sorted(state)} != {sorted(io['keys'])}")
    for key in io["keys"]:
        check(state[key].dtype == torch.float32, f"{key}: {state[key].dtype}")
        check(torch.equal(state[key], tensor(io["params"][key])), f"{key}: values differ")
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8, bias=first_bias), torch.nn.ReLU(), torch.nn.Linear(8, 3)
    )
    check(
        sorted(state) == sorted(model.state_dict()),
        f"{stem}: keys {sorted(state)} != PyTorch's {sorted(model.state_dict())}",
    )
    model.load_state_dict(state, strict=True)
    x = tensor(io["input"])
    ours = tensor(io["output"])
    with torch.no_grad():
        theirs = model(x)
        bound = mlp_bound(model, x)
        exact = model.double()(x.double())
    for name, out in (("PyTorch", theirs), ("the library", ours)):
        diff = (out.double() - exact).abs()
        worst = (diff / bound).max().item()
        check(bool((diff <= bound).all()), f"{stem}: {name}'s outputs are outside the bound (worst {worst:.3g} of it)")
        print(f"{stem}: {name}: every output within the bound (worst {worst:.3g} of it)", file=sys.stderr)
    largest = (theirs - ours).abs().max().item()
    print(
        f"{stem}.safetensors loads in torch {TORCH_VERSION} with strict keys {list(state)}, F32 "
        f"values equal to the library's; largest output difference {largest:.3g}",
        file=sys.stderr,
    )


def main():
    """Builds every case and writes the files."""
    parser = argparse.ArgumentParser(description="Write the PyTorch fixtures for the operations.")
    parser.add_argument(
        "--out",
        default="crates",
        help="the crates directory (default: crates, relative to the working directory; files go under "
        "cavitation-core/tests/fixtures/ops, and --check-rust-save reads under cavitation-nn/tests/fixtures)",
    )
    parser.add_argument(
        "--only", choices=["state-dict", "no-bias"], help="write only this fixture family"
    )
    parser.add_argument(
        "--check-rust-save",
        nargs="?",
        const="",
        metavar="PATH",
        help="write nothing; check that the library-written state-dict file (PATH, default the committed "
        "from-rust.safetensors under --out) loads and predicts in PyTorch",
    )
    args = parser.parse_args()
    if args.check_rust_save is not None:
        check_rust_save(args.out, args.check_rust_save or None)
        return
    if args.only == "state-dict":
        write_state_dict(args.out)
        return
    if args.only == "no-bias":
        write_no_bias_network(args.out)
        return
    make_arithmetic()
    make_matmul()
    make_reductions()
    make_activations()
    make_softmax()
    make_losses()
    make_layers()
    make_optimizers()
    make_select()
    out_dir = os.path.join(args.out, "cavitation-core", "tests", "fixtures", "ops")
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):
        if old.endswith(".json"):
            os.remove(os.path.join(out_dir, old))
    names = set()
    for name, body in CASES:
        check(name not in names, f"duplicate fixture name {name}")
        names.add(name)
        for key in ("output_f64",):
            check(all(math.isfinite(v) for v in body[key]["data"]), f"{name}: not finite")
        with open(os.path.join(out_dir, name + ".json"), "w") as f:
            json.dump(body, f, separators=(",", ":"))
            f.write("\n")
    print(f"wrote {len(CASES)} fixtures to {out_dir} with torch {TORCH_VERSION}", file=sys.stderr)
    layer_dir = os.path.join(args.out, "cavitation-nn", "tests", "fixtures", "layers")
    os.makedirs(layer_dir, exist_ok=True)
    for name, body in LAYER_CASES:
        with open(os.path.join(layer_dir, name + ".json"), "w") as f:
            json.dump(body, f, separators=(",", ":"))
            f.write("\n")
    print(f"wrote {len(LAYER_CASES)} fixtures to {layer_dir} with torch {TORCH_VERSION}", file=sys.stderr)
    optim_dir = os.path.join(args.out, "cavitation-nn", "tests", "fixtures", "optim")
    os.makedirs(optim_dir, exist_ok=True)
    for name, body in OPTIM_CASES:
        with open(os.path.join(optim_dir, name + ".json"), "w") as f:
            json.dump(body, f, separators=(",", ":"))
            f.write("\n")
    print(f"wrote {len(OPTIM_CASES)} fixtures to {optim_dir} with torch {TORCH_VERSION}", file=sys.stderr)
    write_state_dict(args.out)
    write_no_bias_network(args.out)


if __name__ == "__main__":
    main()
