# Runs the speed benchmarks of `cavitation-bench` against other libraries on this machine at the
# checked-out commit and rewrites the GitHub issue that tracks them, so the issue always shows
# the latest measurement of each machine, device and benchmark.
#
#   python scripts/bench/benchmark_issue.py --machine B --python <python with torch and numpy> \
#       --devices cpu,cuda,vulkan,dx12 [--suites cnn,numpy,conv] [--nets NAMES] [--epochs N] \
#       [--rows N] [--conv-batch N] [--hardware "AMD Ryzen 7 2700X; NVIDIA GTX 1660 Super"] \
#       [--os "Windows 10"] [--issue 361] [--dry-run] [--render-only] [--note TEXT] \
#       [--note-about TEXT] [--todo TEXT] [--remove-note TEXT] [--remove-todo TEXT] \
#       [--import FILE] [--replace-body]
#
# Suites (`--suites`, default `cnn`):
# - `cnn`: `cavitation-bench mnist-cnn` and `cifar-cnn` on each device named, against PyTorch
#   (`scripts/bench/torch_cnn.py`, which the benchmark runs itself). Rows `mnist-cnn` and
#   `cifar-cnn`, timed by the epoch.
# - `numpy`: the v0.1 MLP benchmarks on the CPU against numpy (`scripts/bench/numpy_mlp.py`,
#   which the benchmark runs itself, at numpy's default BLAS settings): `mnist --batch 128`
#   (row `mlp-mnist-b128`, timed by the epoch), `mnist --batch 32` (`mlp-mnist-b32`), `digits`
#   (`mlp-digits`, timed by the training sample) and `latency` (`latency`, one input through
#   the digits network, timed by the call). They need the MNIST data in `data/mnist/`.
# - `numpy-cifar`: `cifar --batch 128` and `cifar --batch 32` (`mlp-cifar-b128`,
#   `mlp-cifar-b32`), the MLP on CIFAR-10 against numpy. Not part of `numpy` because its epochs
#   take minutes; it is measured only when asked for and never listed as missing.
# - `conv`: `cavitation-bench conv --batch N` and then `scripts/bench/torch_conv.py --threads 4
#   --batch N` (the script runs both; the benchmark does not run PyTorch itself), the four
#   convolutions of the cifar-cnn network on the CPU. One row per layer and pass, `conv1-fwd`
#   to `conv4-fwd` (the forward pass) and `conv1-fwd-bwd` to `conv4-fwd-bwd` (forward and
#   backward), timed by the pass. `--conv-batch` sets N (default 128, the network's batch); a
#   run at another batch is kept apart, with the mode `batch N`.
# `--nets` picks among the `cnn`, `numpy` and `numpy-cifar` benchmarks of the suites chosen
# (default all of them); `conv` always times the four layers. The numpy and conv suites run on
# the CPU only, so they need `cpu` among `--devices`.
#
# Devices: `cpu` (against PyTorch on the CPU), `cuda` (against PyTorch on CUDA), `metal`
# (against PyTorch on MPS), `vulkan` and `dx12` (Cavitation only: PyTorch has no such engine).
# Every run uses 4 threads (`CAVITATION_THREADS=4`), the CNNs seed 0 and their own epochs and
# rows unless `--epochs` or `--rows` says otherwise. A CNN epoch's time is the median of the
# epochs after the first; the numpy and conv benchmarks report their own medians.
#
# Run it by hand, not in CI: a full run takes about an hour of the machine, which must be
# otherwise idle. Rerun a device after work on its speed; other changes leave the rows standing.
#
# The issue body keeps everything it shows as JSON in a hidden comment at its end: the machines,
# the current row of each machine, device, benchmark and mode, the earlier rows those replaced,
# the notes and the extra to-do items. A run makes its new rows current and moves the rows they
# replace to "Earlier measurements"; nothing is deleted. A row's `cav_s` and `torch_s` are the
# time Cavitation and the other library took for one `unit` (`epoch`, `sample`, `call`, `step`
# or `pass`), in seconds, so smaller is faster whatever the benchmark printed; `baseline` names
# the other library (`pytorch` or `numpy`; a row without one is against PyTorch).
# `--note` and `--todo` add a note or a to-do item, and `--remove-note` and `--remove-todo`
# take one out by its exact text. A note is a row of the Notes table: `--note-about` gives the
# "About" column of every `--note` of the same call, and a `--note` whose text is already there
# replaces that note in its place, so it can give an old note its About. `--import FILE` merges
# rows, earlier rows, notes and to-do items from a JSON file of the same shape (a row whose
# machine, device, benchmark and mode already have one becomes current only if its date is not
# older; the other goes to the earlier rows). A note in the data is a string (shown with "–" as
# its About) or {"about": ..., "text": ...}; an imported note whose text is already there
# replaces that note in its place, and `--remove-note` matches the text of either form.
# Removals run before the import and the additions. A row may carry a `mode` (a variant such as
# `CAVITATION_CUDNN_BENCHMARK=1`, shown after the device), a `source` (where it was measured,
# such as a PR) and a `note`; a back-filled row needs `machine`, `device`, `net` and `cav_s`, and
# its other fields may be null (`unit` and `baseline` default to the benchmark's). Importing a
# row the issue already holds changes nothing. The script refuses to rewrite an issue that has
# text but no data comment, since that text would be lost; `--replace-body` overrides that.
# Before each rewrite it reads the issue again and applies its own changes to what it finds, so
# an edit made during a run is kept. `--render-only` rewrites the issue from what it holds;
# `--dry-run` prints the new body instead of editing the issue. CUDA runs in its default mode
# (`CAVITATION_CUDNN_BENCHMARK` unset), the repeatable one.

import argparse
import datetime
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

CONV_LAYERS = [("conv1", 3, 32, 32), ("conv2", 32, 32, 32), ("conv3", 32, 64, 16), ("conv4", 64, 64, 16)]
CONV_PASSES = [("fwd", "Forward"), ("fwd-bwd", "Forward and backward")]
SUITES = {
    "cnn": ["mnist-cnn", "cifar-cnn"],
    "numpy": ["mlp-mnist-b128", "mlp-mnist-b32", "mlp-digits", "latency"],
    "numpy-cifar": ["mlp-cifar-b128", "mlp-cifar-b32"],
    "conv": [f"{layer[0]}-{p}" for p, _ in CONV_PASSES for layer in CONV_LAYERS],
}
SUITE_OF = {net: suite for suite, nets in SUITES.items() for net in nets}
NETS = [net for nets in SUITES.values() for net in nets]
BASELINE = {"cnn": "pytorch", "numpy": "numpy", "numpy-cifar": "numpy", "conv": "pytorch"}
BASELINE_NAMES = {"pytorch": "PyTorch", "numpy": "numpy"}
CPU_SUITES = ["numpy", "numpy-cifar", "conv"]
# The `cavitation-bench` arguments of each numpy benchmark, and the unit its row is timed by.
NUMPY_RUNS = {
    "mlp-mnist-b128": (["mnist", "--batch", "128"], "epoch"),
    "mlp-mnist-b32": (["mnist", "--batch", "32"], "epoch"),
    "mlp-digits": (["digits"], "sample"),
    "latency": (["latency"], "call"),
    "mlp-cifar-b128": (["cifar", "--batch", "128"], "epoch"),
    "mlp-cifar-b32": (["cifar", "--batch", "32"], "epoch"),
}
UNITS = ["epoch", "sample", "call", "step", "pass"]
CONV_BATCH = 128
DEVICES = {
    "cpu": {"args": ["--device", "cpu", "--torch-device", "cpu"], "feature": None, "torch": True},
    "cuda": {"args": ["--device", "cuda", "--torch-device", "cuda"], "feature": "cuda", "torch": True},
    "metal": {
        "args": ["--device", "gpu", "--backend", "metal", "--torch-device", "mps"],
        "feature": "gpu",
        "torch": True,
    },
    "vulkan": {"args": ["--device", "gpu", "--backend", "vulkan", "--cav-only"], "feature": "gpu", "torch": False},
    "dx12": {"args": ["--device", "gpu", "--backend", "dx12", "--cav-only"], "feature": "gpu", "torch": False},
}
DEVICE_NAMES = {"cpu": "CPU", "cuda": "CUDA", "metal": "Metal", "vulkan": "Vulkan", "dx12": "DX12"}
TORCH_ENGINE = {"cpu": "CPU", "cuda": "CUDA", "metal": "MPS"}
DATA_START = "<!-- benchmark-data"
DATA_END = "-->"


def run(cmd, **kw):
    out = subprocess.run(cmd, check=True, capture_output=True, **kw).stdout
    return out.decode("utf-8", errors="replace")


def output(cmd, **kw):
    """A command's standard output; its errors go to the terminal."""
    out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, **kw).stdout
    return out.decode("utf-8", errors="replace")


def repo_root():
    return run(["git", "rev-parse", "--show-toplevel"]).strip()


def target_dir(root):
    meta = json.loads(run(["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=root))
    return meta["target_directory"]


def epoch_seconds(epochs):
    times = [e["seconds"] for e in epochs]
    if not times:
        return None
    return statistics.median(times[1:] if len(times) > 1 else times)


def bench_env(tdir):
    env = dict(os.environ, CAVITATION_THREADS="4", CARGO_TARGET_DIR=tdir)
    for name in ["CAVITATION_CUDNN_BENCHMARK", "CAVITATION_GPU_BACKEND", "CAVITATION_DEVICE", "CAVITATION_KERNEL"]:
        env.pop(name, None)
    return env


def shown_command(cmd, python, label):
    shown = " ".join(cmd)
    return shown.replace(python, label) if python else shown


def measure(root, binary, tdir, device, net, args):
    spec = DEVICES[device]
    result = os.path.join(tdir, "bench", "cnn", f"{net}-seed0", "result.json")
    if os.path.exists(result):
        os.remove(result)
    cmd = [binary, net, "--seed", "0"] + spec["args"]
    if spec["torch"]:
        cmd += ["--python", args.python]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if args.rows:
        cmd += ["--rows", str(args.rows)]
    print("running:", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=root, env=bench_env(tdir))
    with open(result, encoding="utf-8") as f:
        r = json.load(f)
    cav = epoch_seconds(r["cav"])
    torch = epoch_seconds(r["torch"]) if spec["torch"] else None
    if cav is None or (spec["torch"] and torch is None):
        sys.exit(f"{net} on {device}: the run wrote no epoch times")
    return [{
        "net": net,
        "baseline": "pytorch",
        "unit": "epoch",
        "cav_s": cav,
        "torch_s": torch,
        "epochs": r["epochs"],
        "rows": r["rows"],
        "kernel": r["kernel"],
        "threads": r["threads"],
        "torch_about": r["torch_about"][0] if r["torch_about"] else None,
        "command": "cavitation-bench " + shown_command(cmd[1:], args.python, "<python with torch>"),
    }]


def numpy_seconds(value, unit_label):
    """Seconds per unit from a value `cavitation-bench` wrote in `unit_label`, and that unit."""
    if value is None or value <= 0:
        raise ValueError(f"a {unit_label} value of {value}")
    if unit_label == "s/epoch":
        return value, "epoch"
    if unit_label == "samples/s":
        return 1 / value, "sample"
    if unit_label == "steps/s":
        return 1 / value, "step"
    if unit_label == "us/call":
        return value * 1e-6, "call"
    raise ValueError(f"unknown unit {unit_label}")


def numpy_row(results, net, command):
    """The row of numpy benchmark `net` from the `results.json` that its run wrote."""
    runs = results.get("runs") or []
    if len(runs) != 1:
        raise ValueError(f"results.json holds {len(runs)} runs, not 1")
    r = runs[0]
    cav, unit = numpy_seconds(r["cavitation"], r["unit"])
    base, _ = numpy_seconds(r["numpy"], r["unit"])
    want = NUMPY_RUNS[net][1]
    if unit != want:
        raise ValueError(f"{net} reported {r['unit']}, not a time per {want}")
    batch = re.search(r"batch (\d+)", r["name"])
    return {
        "net": net,
        "baseline": "numpy",
        "unit": unit,
        "cav_s": cav,
        "torch_s": base,
        "epochs": None,
        "rows": None,
        "batch": int(batch.group(1)) if batch else None,
        "kernel": results.get("kernel"),
        "threads": results.get("threads"),
        "torch_about": f"numpy {results.get('numpy_version')}; {results.get('blas_default')}",
        "command": command,
    }


def measure_numpy(root, binary, tdir, net, args):
    result = os.path.join(tdir, "bench", "results.json")
    if os.path.exists(result):
        os.remove(result)
    cmd = [binary] + NUMPY_RUNS[net][0] + ["--python", args.python]
    print("running:", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=root, env=bench_env(tdir))
    with open(result, encoding="utf-8") as f:
        results = json.load(f)
    command = "cavitation-bench " + shown_command(cmd[1:], args.python, "<python with numpy>")
    return [numpy_row(results, net, command)]


CONV_LINE = re.compile(r"^(conv[1-4]) \S+ \S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")


def conv_times(text, batch, who):
    """Seconds of each layer's forward pass and of its forward and backward passes, from the
    table `cavitation-bench conv` or `torch_conv.py` printed. Each time is worked out from the
    printed GFLOPS, which keep more digits than the milliseconds for a small layer."""
    times = {}
    for line in text.splitlines():
        m = CONV_LINE.match(line.strip())
        if not m:
            continue
        name = m.group(1)
        _, cin, cout, size = next(l for l in CONV_LAYERS if l[0] == name)
        flops = 2.0 * batch * cout * size * size * cin * 9
        passes = 2.0 if name == "conv1" else 3.0
        fwd_ms, fwd_g, step_ms, step_g = (float(m.group(i)) for i in range(2, 6))
        fwd = flops / (fwd_g * 1e9) if fwd_g > 0 else fwd_ms / 1e3
        step = passes * flops / (step_g * 1e9) if step_g > 0 else step_ms / 1e3
        times[name] = {"fwd": fwd, "fwd-bwd": step}
    if sorted(times) != [l[0] for l in CONV_LAYERS]:
        raise ValueError(f"{who} printed {len(times)} of the four layers")
    return times


def conv_rows(cav_text, torch_text, batch, cav_command, torch_command):
    """The eight conv rows from the output of `cavitation-bench conv` and `torch_conv.py`."""
    cav = conv_times(cav_text, batch, "cavitation-bench conv")
    torch = conv_times(torch_text, batch, "torch_conv.py")
    head = re.search(r"kernel (\S+), (\d+) threads", cav_text)
    about = re.search(r"^torch conv: batch \d+, (.*), medians of", torch_text, re.M)
    rows = []
    for p, _ in CONV_PASSES:
        for name, *_ in CONV_LAYERS:
            rows.append({
                "net": f"{name}-{p}",
                "baseline": "pytorch",
                "unit": "pass",
                "mode": "" if batch == CONV_BATCH else f"batch {batch}",
                "cav_s": cav[name][p],
                "torch_s": torch[name][p],
                "epochs": None,
                "rows": None,
                "batch": batch,
                "kernel": head.group(1) if head else None,
                "threads": int(head.group(2)) if head else None,
                "torch_about": about.group(1) if about else None,
                "command": cav_command,
                "baseline_command": torch_command,
            })
    return rows


def measure_conv(root, binary, tdir, args):
    batch = args.conv_batch
    cmd = [binary, "conv", "--batch", str(batch)]
    print("running:", " ".join(cmd[1:]), flush=True)
    cav_text = output(cmd, cwd=root, env=bench_env(tdir))
    print(cav_text, end="", flush=True)
    script = os.path.join(root, "scripts", "bench", "torch_conv.py")
    tcmd = [args.python, "-I", script, "--threads", "4", "--batch", str(batch)]
    print("running: torch_conv.py --threads 4 --batch", batch, flush=True)
    torch_text = output(tcmd, cwd=root)
    print(torch_text, end="", flush=True)
    torch_command = f"<python with torch> -I scripts/bench/torch_conv.py --threads 4 --batch {batch}"
    return conv_rows(cav_text, torch_text, batch, "cavitation-bench " + " ".join(cmd[1:]), torch_command)


def default_fields(row):
    """Fills `mode`, `baseline` and `unit` where a row leaves them out (older rows and
    back-filled ones): no mode, the benchmark's library and the benchmark's unit."""
    row["mode"] = row.get("mode") or ""
    suite = SUITE_OF.get(row.get("net"))
    if suite:
        row["baseline"] = row.get("baseline") or BASELINE[suite]
        default_unit = NUMPY_RUNS[row["net"]][1] if row["net"] in NUMPY_RUNS else "pass" if suite == "conv" else "epoch"
        row["unit"] = row.get("unit") or default_unit


def normalise(data):
    data.setdefault("machines", {})
    for k in ["rows", "history", "notes", "todo"]:
        data.setdefault(k, [])
    for row in data["rows"] + data["history"]:
        default_fields(row)
    for note in data["notes"]:
        check_note(note)
    return data


def check_note(note):
    if isinstance(note, str):
        return
    if not isinstance(note, dict) or not isinstance(note.get("text"), str):
        sys.exit(f"a note is neither text nor {{\"about\", \"text\"}}: {note}")
    if not isinstance(note.get("about") or "", str):
        sys.exit(f"a note's about is not text: {note}")


def note_text(note):
    return note if isinstance(note, str) else note["text"]


def check_row(row):
    for field in ["machine", "device", "net"]:
        if not isinstance(row.get(field), str):
            sys.exit(f"a row has no {field}: {row}")
    default_fields(row)
    if row["device"] not in DEVICES or row["net"] not in NETS:
        sys.exit(f"row {key(row)}: unknown device or benchmark")
    suite = SUITE_OF[row["net"]]
    if suite in CPU_SUITES and row["device"] != "cpu":
        sys.exit(f"row {key(row)}: the {suite} benchmarks run on the CPU only")
    if row["baseline"] != BASELINE[suite]:
        sys.exit(f"row {key(row)}: {row['net']} is measured against {BASELINE[suite]}, not {row['baseline']}")
    if row["unit"] not in UNITS:
        sys.exit(f"row {key(row)}: unknown unit {row['unit']}; choose from {', '.join(UNITS)}")
    if not isinstance(row.get("cav_s"), (int, float)):
        sys.exit(f"row {key(row)}: cav_s, Cavitation's time in seconds, is required")


def read_issue(issue, replace_body):
    body = run(["gh", "issue", "view", str(issue), "--json", "body", "-q", ".body"])
    return parse_body(body, issue, replace_body)


def parse_body(body, issue, replace_body):
    start = body.find(DATA_START)
    if start < 0:
        if body.strip() and not replace_body:
            sys.exit(
                f"issue #{issue} has text but no benchmark data comment; rewriting it would lose "
                "that text. Put its measurements in a file for --import, or pass --replace-body."
            )
        return normalise({})
    end = body.find(DATA_END, start)
    if end < 0:
        sys.exit(f"issue #{issue}: the benchmark data comment has no end")
    return normalise(json.loads(body[start + len(DATA_START) : end]))


def key(row):
    return (row["machine"], row["device"], row["net"], row.get("mode", ""))


def add_row(data, row):
    """Makes `row` current unless the current row for its key is newer; keeps the other as
    an earlier row. Dates are ISO strings, so they compare as text; a row without one counts
    as the oldest, and of two rows with the same date the one added last is current. Adding a
    row the data already holds changes nothing."""
    check_row(row)
    if row in data["rows"] or row in data["history"]:
        return
    old = next((r for r in data["rows"] if key(r) == key(row)), None)
    if old is None:
        data["rows"].append(row)
        return
    if (old.get("date") or "") > (row.get("date") or ""):
        earlier = row
    else:
        data["rows"] = [r for r in data["rows"] if key(r) != key(row)]
        data["rows"].append(row)
        earlier = old
    if earlier not in data["history"]:
        data["history"].append(earlier)


def add_note(data, note):
    """Adds `note`; a note with the same text is replaced in its place, so an import can give
    an existing note its About."""
    for i, have in enumerate(data["notes"]):
        if note_text(have) == note_text(note):
            data["notes"][i] = note
            return
    data["notes"].append(note)


def import_file(data, path):
    with open(path, encoding="utf-8") as f:
        new = normalise(json.load(f))
    for name, m in new["machines"].items():
        unknown = [d for d in m.get("devices", []) if d not in DEVICES]
        if unknown:
            sys.exit(f"machine {name}: unknown devices {', '.join(unknown)}")
        have = data["machines"].setdefault(name, {"hardware": "", "os": "", "devices": []})
        have["hardware"] = m.get("hardware") or have["hardware"]
        have["os"] = m.get("os") or have["os"]
        have["devices"] = [d for d in DEVICES if d in have["devices"] or d in m.get("devices", [])]
    for row in new["rows"] + new["history"]:
        check_row(row)
        if row["machine"] not in data["machines"]:
            sys.exit(f"row {key(row)}: machine {row['machine']} is not listed")
    for row in new["rows"]:
        add_row(data, row)
    for row in new["history"]:
        if row not in data["history"] and row not in data["rows"]:
            data["history"].append(row)
    for note in new["notes"]:
        add_note(data, note)
    for text in new["todo"]:
        if text not in data["todo"]:
            data["todo"].append(text)
    for row in data["rows"]:
        machine = data["machines"][row["machine"]]
        if row["device"] not in machine["devices"]:
            machine["devices"] = [d for d in DEVICES if d in machine["devices"] or d == row["device"]]


def ratio_cell(cav, other):
    if other is None:
        return "❌"
    if cav <= 0 or other <= 0:
        return "n/a (a run took no measurable time)"
    r = cav / other
    if r < 0.97:
        return f"🟢 {1 / r:.2f}× faster"
    if r > 1.03:
        return f"🔴 {r:.2f}× slower"
    if f"{1 / r:.2f}" != "1.00" and r < 1:
        return f"🟡 same ({1 / r:.2f}× faster)"
    if f"{r:.2f}" != "1.00" and r > 1:
        return f"🟡 same ({r:.2f}× slower)"
    return "🟡 same"


def speed(row):
    return ratio_cell(row["cav_s"], row.get("torch_s"))


def table_text(text):
    """Text for a table cell: a `|` would end the cell and a newline the row."""
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def duration(seconds):
    if seconds >= 1:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds * 1e6:.2f} µs"


def time_text(seconds, unit):
    """A time per unit as the tables show it, with the rate beside a time per sample or step."""
    text = f"{duration(seconds)}/{unit}"
    if unit in ("sample", "step") and seconds > 0:
        text += f" ({1 / seconds:,.0f} {unit}s/s)"
    return text


def cpu_machines(data):
    return [name for name in sorted(data["machines"]) if "cpu" in data["machines"][name]["devices"]]


def modes_of(rows, name, nets):
    return [""] + sorted({r["mode"] for r in rows if r["machine"] == name and r["net"] in nets and r["mode"]})


def render(data):
    rows = data["rows"]
    out = []
    out.append(
        "Where Cavitation stands against PyTorch and numpy, per machine, device and benchmark. "
        "`scripts/bench/benchmark_issue.py` writes this issue from fresh runs; rerun it on a machine "
        "after work on a device's speed to bring its rows up to date. Each row gives the commit "
        "and date it was measured at."
    )
    out.append("")
    out.append(
        "Each cell says how much faster or slower Cavitation is than the other library for the same "
        "work on the same machine (an epoch, a pass, a sample or a call): \"1.27× faster\" means the "
        "other library takes 1.27 times as long as Cavitation, \"2.46× slower\" means Cavitation "
        "takes 2.46 times as long. "
        "🟢 Cavitation's time more than 3% shorter, 🟡 within 3% either way, 🔴 more than 3% longer, "
        "⬜ not measured, ❌ nothing to compare with."
    )
    out.append("")
    out.append(
        "**Why Vulkan and DX12 have ❌:** PyTorch has no Vulkan or DX12 engine, so there is no "
        "PyTorch run on the same engine to compare against. Their rows give Cavitation's own epoch "
        "time. Metal is compared with PyTorch on MPS, Apple's GPU engine."
    )
    out.append("")
    out.append("## Machines")
    out.append("")
    out.append("| Machine | Hardware | OS | Devices |")
    out.append("|---|---|---|---|")
    for name in sorted(data["machines"]):
        m = data["machines"][name]
        devs = ", ".join(DEVICE_NAMES[d] for d in m["devices"])
        out.append(f"| {name} | {table_text(m['hardware'])} | {table_text(m['os'])} | {devs} |")
    out.append("")
    missing = []
    speed_table(out, data, missing)
    numpy_table(out, data, missing)
    conv_table(out, data, missing)
    out.append("## Measurements")
    out.append("")
    if not rows:
        out.append("None yet.")
        out.append("")
    else:
        measurement_table(out, rows)
        out.append(
            "Commands, as each row ran them (the release build of `cavitation-bench`, with "
            "`CAVITATION_THREADS=4`; it runs the PyTorch or numpy side itself, except `conv`, whose "
            "PyTorch side is `torch_conv.py`):"
        )
        out.append("")
        out.append("```")
        seen = []
        for row in sorted(rows, key=sort_key):
            for command in [row.get("command"), row.get("baseline_command")]:
                if command and command not in seen:
                    seen.append(command)
        out.extend(seen)
        out.append("```")
        out.append("")
    if data["notes"]:
        out.append("## Notes")
        out.append("")
        out.append("| About | Note |")
        out.append("|---|---|")
        for note in data["notes"]:
            about = "–" if isinstance(note, str) else (note.get("about") or "–")
            out.append(f"| {table_text(about)} | {table_text(note_text(note))} |")
        out.append("")
    if data["history"]:
        out.append("## Earlier measurements")
        out.append("")
        out.append("Rows that a later measurement of the same machine, device, benchmark and mode replaced.")
        out.append("")
        measurement_table(out, data["history"])
    out.append("## Not measured yet")
    out.append("")
    for m in missing:
        out.append(f"- [ ] {m}")
    for text in data["todo"]:
        out.append(f"- [ ] {text}")
    if not missing and not data["todo"]:
        out.append("Every machine, device and benchmark listed above has a row.")
    out.append("")
    out.append(DATA_START)
    saved = {k: data[k] for k in ["machines", "rows", "history", "notes", "todo"]}
    out.append(json.dumps(saved, ensure_ascii=False, indent=1).replace(">", "\\u003e"))
    out.append(DATA_END)
    return "\n".join(out) + "\n"


def speed_table(out, data, missing):
    rows, nets = data["rows"], SUITES["cnn"]
    out.append("## Speed")
    out.append("")
    out.append("The CNN benchmarks against PyTorch: one epoch of training.")
    out.append("")
    out.append("| Machine | Device | " + " | ".join(nets) + " |")
    out.append("|---|---|" + "---|" * len(nets))
    for name in sorted(data["machines"]):
        for device in data["machines"][name]["devices"]:
            modes = [""] + sorted(
                {r["mode"] for r in rows if (r["machine"], r["device"]) == (name, device) and r["mode"] and r["net"] in nets}
            )
            for mode in modes:
                cells = []
                for net in nets:
                    row = next((r for r in rows if key(r) == (name, device, net, mode)), None)
                    if row is None:
                        cells.append("❌ (Cavitation not measured yet)" if not DEVICES[device]["torch"] else "⬜")
                        if not mode:
                            missing.append(f"{net} on machine {name}, {DEVICE_NAMES[device]}")
                        continue
                    cell = speed(row)
                    if row.get("torch_s") is None:
                        cell += f" (Cavitation {row['cav_s']:.2f} s an epoch)"
                    cells.append(cell)
                out.append(f"| {name} | {device_label(device, mode)} | " + " | ".join(cells) + " |")
    out.append("")


def numpy_table(out, data, missing):
    """Machine by numpy benchmark, for the machines with a CPU. The CIFAR-10 MLP has columns
    only once one of its rows exists, and is never listed as missing."""
    rows = data["rows"]
    machines = cpu_machines(data)
    nets = SUITES["numpy"] + [n for n in SUITES["numpy-cifar"] if any(r["net"] == n for r in rows)]
    if not machines:
        return
    out.append("## Against numpy")
    out.append("")
    out.append(
        "The v0.1 MLP benchmarks on the CPU against the same network, batch, data and steps in numpy "
        "(`scripts/bench/numpy_mlp.py`): an epoch of the MNIST MLP (784 → 1024 → 256 → 10, plain "
        "SGD) at batch 128 and 32, a training sample of the digits MLP (64 → 256 → 128 → 10) at "
        "batch 128, and one input through the digits MLP in inference mode (`latency`)."
    )
    out.append("")
    out.append("| Machine | Device | " + " | ".join(nets) + " |")
    out.append("|---|---|" + "---|" * len(nets))
    for name in machines:
        for mode in modes_of(rows, name, nets):
            cells = []
            for net in nets:
                row = next((r for r in rows if key(r) == (name, "cpu", net, mode)), None)
                if row is None:
                    cells.append("⬜")
                    if not mode and net in SUITES["numpy"]:
                        missing.append(f"{net} against numpy on machine {name}, CPU")
                    continue
                cells.append(speed(row))
            out.append(f"| {name} | {device_label('cpu', mode)} | " + " | ".join(cells) + " |")
    out.append("")


def conv_table(out, data, missing):
    """Machine and pass by convolution layer, for the machines with a CPU, with the four layers
    together when all four have a row."""
    rows = data["rows"]
    machines = cpu_machines(data)
    if not machines:
        return
    layers = [l[0] for l in CONV_LAYERS]
    out.append("## Per layer against PyTorch")
    out.append("")
    out.append(
        "The four convolutions of the cifar-cnn network on the CPU (3 × 3 kernels with padding 1: "
        "3 → 32 and 32 → 32 channels on 32 × 32 images, 32 → 64 and 64 → 64 on 16 × 16) against "
        "PyTorch's `conv2d` at the same batch (`scripts/bench/torch_conv.py`): the median of 11 "
        "forward passes, and of 11 forward and backward passes (the first layer without the "
        "input's gradient). \"All four\" compares the sums of the four layers' times."
    )
    out.append("")
    out.append("| Machine | Device | Pass | " + " | ".join(layers) + " | All four |")
    out.append("|---|---|---|" + "---|" * (len(layers) + 1))
    for name in machines:
        nets = SUITES["conv"]
        for mode in modes_of(rows, name, nets):
            if not mode and not any(key(r) == (name, "cpu", n, "") for r in rows for n in nets):
                missing.append(f"the four cifar-cnn convolutions (`conv`) against PyTorch on machine {name}, CPU")
            for p, label in CONV_PASSES:
                found = [next((r for r in rows if key(r) == (name, "cpu", f"{l}-{p}", mode)), None) for l in layers]
                cells = [speed(r) if r else "⬜" for r in found]
                if all(r and r.get("torch_s") is not None for r in found):
                    cells.append(ratio_cell(sum(r["cav_s"] for r in found), sum(r["torch_s"] for r in found)))
                else:
                    cells.append("⬜")
                out.append(f"| {name} | {device_label('cpu', mode)} | {label} | " + " | ".join(cells) + " |")
    out.append("")


def device_label(device, mode):
    return f"{DEVICE_NAMES[device]}, {table_text(mode)}" if mode else DEVICE_NAMES[device]


def sort_key(r):
    suites = list(SUITES)
    order = {d: i for i, d in enumerate(DEVICES)}
    return (r["machine"], suites.index(SUITE_OF[r["net"]]), order[r["device"]], r["mode"], NETS.index(r["net"]), r.get("date") or "")


def measurement_table(out, rows):
    out.append(
        "| Machine | Device | Benchmark | Against | Cavitation | Other library | Epochs × rows, batch "
        "| Threads, kernel | Commit | Date | Other library's build | Source and notes |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    def shown(v, fmt="{}"):
        return "–" if v is None else fmt.format(v)

    for row in sorted(rows, key=sort_key):
        unit = row["unit"]
        other = time_text(row["torch_s"], unit) if row.get("torch_s") is not None else "❌"
        size = "–"
        if row.get("epochs") is not None and row.get("rows") is not None:
            size = f"{row['epochs']} × {row['rows']:,}"
        elif row.get("batch") is not None:
            size = f"batch {row['batch']}"
        threads = ", ".join(shown(row.get(k)) for k in ["threads", "kernel"])
        source = table_text("; ".join(t for t in [row.get("source"), row.get("note")] if t) or "–")
        out.append(
            f"| {row['machine']} | {device_label(row['device'], row['mode'])} | {row['net']} "
            f"| {BASELINE_NAMES[row['baseline']]} | {time_text(row['cav_s'], unit)} | {other} | {size} "
            f"| {threads} | {table_text(shown(row.get('commit')))} "
            f"| {shown(row.get('date'))} | {table_text(row.get('torch_about') or '–')} | {source} |"
        )
    out.append("")


def main():
    p = argparse.ArgumentParser(
        description="Measure Cavitation's speed against PyTorch and numpy and rewrite the speed issue."
    )
    p.add_argument("--machine", help="the machine's letter in the issue, such as B")
    p.add_argument("--python", help="a Python with torch (with CUDA for --devices cuda) and numpy")
    p.add_argument("--devices", default="cpu", help="comma-separated: " + ", ".join(DEVICES))
    p.add_argument(
        "--suites",
        default="cnn",
        help="comma-separated: cnn (the CNNs against PyTorch), numpy (the MLPs and latency against "
        "numpy, CPU), numpy-cifar (the CIFAR-10 MLP against numpy, CPU, slow), conv (the four "
        "cifar-cnn convolutions against PyTorch, CPU); default cnn",
    )
    p.add_argument(
        "--nets",
        help="comma-separated benchmarks of the cnn, numpy and numpy-cifar suites chosen (default "
        "all of them): " + ", ".join(n for s in ["cnn", "numpy", "numpy-cifar"] for n in SUITES[s]),
    )
    p.add_argument("--epochs", type=int, help="epochs of the CNNs")
    p.add_argument("--rows", type=int, help="training rows of the CNNs")
    p.add_argument("--conv-batch", type=int, default=CONV_BATCH, help="batch of the conv suite (default 128)")
    p.add_argument("--hardware", help="CPU and GPU names, kept for the machine once given")
    p.add_argument("--os", help="operating system, kept for the machine once given")
    p.add_argument("--issue", type=int, default=361)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--note", action="append", default=[], help="add a note (repeatable)")
    p.add_argument("--note-about", help="the About column of every --note given with it")
    p.add_argument("--todo", action="append", default=[], help="add a to-do item (repeatable)")
    p.add_argument("--remove-note", action="append", default=[], help="remove a note by its exact text")
    p.add_argument("--remove-todo", action="append", default=[], help="remove a to-do item by its exact text")
    p.add_argument("--import", dest="import_path", help="merge rows, notes and to-do items from a JSON file")
    p.add_argument("--replace-body", action="store_true", help="rewrite an issue that has no data comment")
    args = p.parse_args()
    if args.note_about is not None and not args.note:
        sys.exit("--note-about gives the About of the --note texts; give at least one --note")
    if args.conv_batch < 1:
        sys.exit("--conv-batch must be a positive number")

    root = repo_root()
    # Every change is kept as a step and applied to a fresh read of the issue before each
    # publish, so an edit made to the issue during an hour-long run is not overwritten. The
    # steps run once here first, so a bad file or note stops the script before it measures.
    changes = edits(args)
    data = read_issue(args.issue, args.replace_body)
    for change in changes:
        change(data, True)

    if args.render_only or (changes and not args.machine):
        publish(changes, args, root)
        return
    if not args.machine:
        sys.exit("--machine is needed to measure")
    devices = [d for d in args.devices.split(",") if d]
    suites = [s for s in args.suites.split(",") if s]
    for d in devices:
        if d not in DEVICES:
            sys.exit(f"unknown device {d}; choose from {', '.join(DEVICES)}")
    for s in suites:
        if s not in SUITES:
            sys.exit(f"unknown suite {s}; choose from {', '.join(SUITES)}")
    chosen = [n for s in suites if s != "conv" for n in SUITES[s]]
    nets = chosen if args.nets is None else [n for n in args.nets.split(",") if n]
    for n in nets:
        if n not in chosen:
            sys.exit(f"unknown benchmark {n} for the suites {', '.join(suites)}; choose from {', '.join(chosen)}")
    cpu_suites = [s for s in suites if s in CPU_SUITES]
    if cpu_suites and "cpu" not in devices:
        sys.exit(f"the {', '.join(cpu_suites)} suites run on the CPU only: add cpu to --devices")
    if "cnn" not in suites and devices != ["cpu"]:
        sys.exit("only the cnn suite runs on devices other than the CPU: give --devices cpu")
    if not args.python and (cpu_suites or ("cnn" in suites and any(DEVICES[d]["torch"] for d in devices))):
        sys.exit("--python is needed for cpu, cuda and metal and for the numpy and conv suites")
    watched = ["crates", "scripts/bench", "Cargo.toml", "Cargo.lock"]
    if run(["git", "status", "--porcelain", "--"] + watched, cwd=root).strip():
        sys.exit("uncommitted changes in " + ", ".join(watched) + "; measure a commit")
    machine = data["machines"].get(args.machine, {"hardware": "", "os": "", "devices": []})
    if not (args.hardware or machine["hardware"]) or not (args.os or machine["os"]):
        sys.exit(f"machine {args.machine} is new: give --hardware and --os")

    def add_machine(d, first):
        m = d["machines"].setdefault(args.machine, {"hardware": "", "os": "", "devices": []})
        m["hardware"] = args.hardware or m["hardware"]
        m["os"] = args.os or m["os"]
        m["devices"] = [x for x in DEVICES if x in m["devices"] or x in devices]

    changes.append(add_machine)
    if len(changes) > 1:
        publish(changes, args, root)

    cnn_devices = devices if "cnn" in suites else []
    features = sorted({DEVICES[d]["feature"] for d in cnn_devices if DEVICES[d]["feature"]})
    build = ["cargo", "build", "--release", "-p", "cavitation-bench"]
    if features:
        build += ["--features", ",".join(features)]
    subprocess.run(build, cwd=root, check=True)
    tdir = target_dir(root)
    binary = os.path.join(tdir, "release", "cavitation-bench" + (".exe" if os.name == "nt" else ""))
    if not shutil.which(binary) and not os.path.exists(binary):
        sys.exit(f"no binary at {binary}")
    commit = run(["git", "rev-parse", "--short", "HEAD"], cwd=root).strip()
    date = datetime.date.today().isoformat()
    jobs = [(d, n, lambda d=d, n=n: measure(root, binary, tdir, d, n, args)) for d in cnn_devices for n in nets if n in SUITES["cnn"]]
    jobs += [("cpu", n, lambda n=n: measure_numpy(root, binary, tdir, n, args)) for n in nets if n not in SUITES["cnn"]]
    if "conv" in suites:
        jobs.append(("cpu", "conv", lambda: measure_conv(root, binary, tdir, args)))
    for device, name, job in jobs:
        try:
            new = job()
        except (subprocess.CalledProcessError, OSError, KeyError, ValueError) as e:
            sys.exit(f"stopped: {name} on {device}: {e}; the rows measured before it are in the issue")
        for row in new:
            row.update(machine=args.machine, device=device, commit=commit, date=date)
            row.setdefault("mode", "")
            changes.append(lambda d, first, row=row: add_row(d, dict(row)))
        publish(changes, args, root)


def edits(args):
    """The steps for --import, --note, --todo and their removals. Each takes the issue's data
    and whether it is the first time it runs: a removal of text that is not there stops the
    script the first time, and is skipped on a later read where someone already removed it.
    Removals run first, so one call can take out old notes and import their replacements."""
    steps = []
    for kind, removed in [("notes", args.remove_note), ("todo", args.remove_todo)]:
        for text in removed:
            def remove(d, first, kind=kind, text=text):
                texts = [note_text(x) for x in d[kind]]
                if text in texts:
                    d[kind] = [x for x in d[kind] if note_text(x) != text]
                elif first:
                    sys.exit(f"no {kind} entry reads exactly: {text}")
            steps.append(remove)
    if args.import_path:
        steps.append(lambda d, first: import_file(d, args.import_path))
    for text in args.note:
        note = {"about": args.note_about or "", "text": text}
        steps.append(lambda d, first, note=note: add_note(d, dict(note)))
    for text in args.todo:
        def add(d, first, text=text):
            if text not in d["todo"]:
                d["todo"].append(text)
        steps.append(add)
    return steps


def publish(changes, args, root):
    data = read_issue(args.issue, args.replace_body)
    for change in changes:
        change(data, False)
    body = render(data)
    if args.dry_run:
        sys.stdout.reconfigure(encoding="utf-8")
        print(body)
        return
    path = os.path.join(target_dir(root), "bench", "issue-body.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    subprocess.run(["gh", "issue", "edit", str(args.issue), "--body-file", path], cwd=root, check=True)
    print(f"issue #{args.issue} updated")


if __name__ == "__main__":
    main()
