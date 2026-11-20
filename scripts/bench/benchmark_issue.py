# Runs `cavitation-bench mnist-cnn` and `cifar-cnn` on this machine's devices at the checked-out
# commit and rewrites the GitHub issue that tracks speed against PyTorch, so the issue always
# shows the latest measurement of each machine, device and network.
#
#   python scripts/bench/benchmark_issue.py --machine B --python <python with torch> \
#       --devices cpu,cuda,vulkan,dx12 [--nets mnist-cnn,cifar-cnn] [--epochs N] [--rows N] \
#       [--hardware "AMD Ryzen 7 2700X; NVIDIA GTX 1660 Super"] [--os "Windows 10"] \
#       [--issue 361] [--dry-run] [--render-only] [--note TEXT] [--todo TEXT] \
#       [--remove-note TEXT] [--remove-todo TEXT] [--import FILE] [--replace-body]
#
# Devices: `cpu` (against PyTorch on the CPU), `cuda` (against PyTorch on CUDA), `metal`
# (against PyTorch on MPS), `vulkan` and `dx12` (Cavitation only: PyTorch has no such engine).
# Every run uses 4 threads (`CAVITATION_THREADS=4`), seed 0, and the benchmark's own epochs and
# rows unless `--epochs` or `--rows` says otherwise. An epoch's time is the median of the epochs
# after the first.
#
# Run it by hand, not in CI: a full run takes about an hour of the machine, which must be
# otherwise idle. Rerun a device after work on its speed; other changes leave the rows standing.
#
# The issue body keeps everything it shows as JSON in a hidden comment at its end: the machines,
# the current row of each machine, device, network and mode, the earlier rows those replaced, the
# notes and the extra to-do items. A run makes its new rows current and moves the rows they
# replace to "Earlier measurements"; nothing is deleted. `--note` and `--todo` add a note or a
# to-do item, and `--remove-note` and `--remove-todo` take one out by its exact text.
# `--import FILE` merges rows, earlier rows, notes and to-do items from a JSON file of the same
# shape (a row whose machine, device, network and mode already have one becomes current only if
# its date is not older; the other goes to the earlier rows). A row may carry a `mode` (a
# variant such as `CAVITATION_CUDNN_BENCHMARK=1`, shown after the device), a `source` (where it
# was measured, such as a PR) and a `note`; a back-filled row needs `machine`, `device`, `net`
# and `cav_s`, and its other fields may be null. Importing a row the issue already holds changes
# nothing. The script refuses to rewrite an issue that has text but no data comment, since that
# text would be lost; `--replace-body` overrides that. Before each rewrite it reads the issue
# again and applies its own changes to what it finds, so an edit made during a run is kept. `--render-only` rewrites the issue from what it
# holds; `--dry-run` prints the new body instead of editing the issue. CUDA runs in its default
# mode (`CAVITATION_CUDNN_BENCHMARK` unset), the repeatable one.

import argparse
import datetime
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

NETS = ["mnist-cnn", "cifar-cnn"]
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
    env = dict(os.environ, CAVITATION_THREADS="4", CARGO_TARGET_DIR=tdir)
    for name in ["CAVITATION_CUDNN_BENCHMARK", "CAVITATION_GPU_BACKEND", "CAVITATION_DEVICE", "CAVITATION_KERNEL"]:
        env.pop(name, None)
    print("running:", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=root, env=env)
    with open(result, encoding="utf-8") as f:
        r = json.load(f)
    shown = " ".join(cmd[1:])
    if args.python:
        shown = shown.replace(args.python, "<python with torch>")
    cav = epoch_seconds(r["cav"])
    torch = epoch_seconds(r["torch"]) if spec["torch"] else None
    if cav is None or (spec["torch"] and torch is None):
        sys.exit(f"{net} on {device}: the run wrote no epoch times")
    return {
        "cav_s": cav,
        "torch_s": torch,
        "epochs": r["epochs"],
        "rows": r["rows"],
        "kernel": r["kernel"],
        "threads": r["threads"],
        "torch_about": r["torch_about"][0] if r["torch_about"] else None,
        "command": "cavitation-bench " + shown,
    }


def normalise(data):
    data.setdefault("machines", {})
    for key in ["rows", "history", "notes", "todo"]:
        data.setdefault(key, [])
    for row in data["rows"] + data["history"]:
        row["mode"] = row.get("mode") or ""
    return data


def check_row(row):
    row["mode"] = row.get("mode") or ""
    for field in ["machine", "device", "net"]:
        if not isinstance(row.get(field), str):
            sys.exit(f"a row has no {field}: {row}")
    if row["device"] not in DEVICES or row["net"] not in NETS:
        sys.exit(f"row {key(row)}: unknown device or network")
    if not isinstance(row.get("cav_s"), (int, float)):
        sys.exit(f"row {key(row)}: cav_s, Cavitation's epoch in seconds, is required")


def read_issue(issue, replace_body):
    body = run(["gh", "issue", "view", str(issue), "--json", "body", "-q", ".body"])
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
    for kind in ["notes", "todo"]:
        for text in new[kind]:
            if text not in data[kind]:
                data[kind].append(text)
    for row in data["rows"]:
        machine = data["machines"][row["machine"]]
        if row["device"] not in machine["devices"]:
            machine["devices"] = [d for d in DEVICES if d in machine["devices"] or d == row["device"]]


def speed(row):
    if row.get("torch_s") is None:
        return "❌"
    if row["cav_s"] <= 0 or row["torch_s"] <= 0:
        return "n/a (an epoch took no measurable time)"
    r = row["cav_s"] / row["torch_s"]
    if r < 0.97:
        return f"🟢 {1 / r:.2f}× faster"
    if r > 1.03:
        return f"🔴 {r:.2f}× slower"
    if f"{1 / r:.2f}" != "1.00" and r < 1:
        return f"🟡 same ({1 / r:.2f}× faster)"
    if f"{r:.2f}" != "1.00" and r > 1:
        return f"🟡 same ({r:.2f}× slower)"
    return "🟡 same"


def table_text(text):
    """Text for a table cell: a `|` would end the cell and a newline the row."""
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render(data):
    rows = data["rows"]
    out = []
    out.append(
        "Where Cavitation stands against PyTorch, per machine, device and network. "
        "`scripts/bench/benchmark_issue.py` writes this issue from fresh runs; rerun it on a machine "
        "after work on a device's speed to bring its rows up to date. Each row gives the commit "
        "and date it was measured at."
    )
    out.append("")
    out.append(
        "Each cell says how much faster or slower Cavitation is than PyTorch for an epoch on the "
        "same machine: \"1.27× faster\" means PyTorch takes 1.27 times as long as Cavitation, "
        "\"2.46× slower\" means Cavitation takes 2.46 times as long as PyTorch. "
        "🟢 Cavitation's epoch more than 3% shorter than PyTorch's, 🟡 within 3% either way, 🔴 more than 3% longer, "
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
    out.append("## Speed")
    out.append("")
    out.append("| Machine | Device | " + " | ".join(NETS) + " |")
    out.append("|---|---|" + "---|" * len(NETS))
    missing = []
    for name in sorted(data["machines"]):
        for device in data["machines"][name]["devices"]:
            modes = [""] + sorted(
                {r["mode"] for r in rows if (r["machine"], r["device"]) == (name, device) and r["mode"]}
            )
            for mode in modes:
                cells = []
                for net in NETS:
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
    out.append("## Measurements")
    out.append("")
    if not rows:
        out.append("None yet.")
        out.append("")
    else:
        measurement_table(out, rows)
        out.append(
            "Commands, as each row ran them (the release build of `cavitation-bench`, with "
            "`CAVITATION_THREADS=4`):"
        )
        out.append("")
        out.append("```")
        seen = []
        for row in rows:
            if row.get("command") and row["command"] not in seen:
                seen.append(row["command"])
        out.extend(seen)
        out.append("```")
        out.append("")
    if data["notes"]:
        out.append("## Notes")
        out.append("")
        for text in data["notes"]:
            out.append(f"- {text}")
        out.append("")
    if data["history"]:
        out.append("## Earlier measurements")
        out.append("")
        out.append("Rows that a later measurement of the same machine, device, network and mode replaced.")
        out.append("")
        measurement_table(out, data["history"])
    out.append("## Not measured yet")
    out.append("")
    for m in missing:
        out.append(f"- [ ] {m}")
    for text in data["todo"]:
        out.append(f"- [ ] {text}")
    if not missing and not data["todo"]:
        out.append("Every machine, device and network listed above has a row.")
    out.append("")
    out.append(DATA_START)
    saved = {k: data[k] for k in ["machines", "rows", "history", "notes", "todo"]}
    out.append(json.dumps(saved, ensure_ascii=False, indent=1).replace(">", "\\u003e"))
    out.append(DATA_END)
    return "\n".join(out) + "\n"


def device_label(device, mode):
    return f"{DEVICE_NAMES[device]}, {table_text(mode)}" if mode else DEVICE_NAMES[device]


def measurement_table(out, rows):
    out.append(
        "| Machine | Device | Network | Cavitation epoch | PyTorch epoch | Epochs × rows "
        "| Threads, kernel | Commit | Date | PyTorch | Source and notes |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    order = {d: i for i, d in enumerate(DEVICES)}

    def shown(v, fmt="{}"):
        return "–" if v is None else fmt.format(v)

    def sort_key(r):
        return (r["machine"], order[r["device"]], r["mode"], NETS.index(r["net"]), r.get("date") or "")

    for row in sorted(rows, key=sort_key):
        torch = f"{row['torch_s']:.2f} s" if row.get("torch_s") is not None else "❌"
        size = "–"
        if row.get("epochs") is not None and row.get("rows") is not None:
            size = f"{row['epochs']} × {row['rows']:,}"
        threads = ", ".join(shown(row.get(k)) for k in ["threads", "kernel"])
        source = table_text("; ".join(t for t in [row.get("source"), row.get("note")] if t) or "–")
        out.append(
            f"| {row['machine']} | {device_label(row['device'], row['mode'])} | {row['net']} "
            f"| {row['cav_s']:.2f} s | {torch} | {size} | {threads} | {table_text(shown(row.get('commit')))} "
            f"| {shown(row.get('date'))} | {table_text(row.get('torch_about') or '–')} | {source} |"
        )
    out.append("")


def main():
    p = argparse.ArgumentParser(description="Measure the CNN benchmarks and rewrite the speed issue.")
    p.add_argument("--machine", help="the machine's letter in the issue, such as B")
    p.add_argument("--python", help="a Python with torch (with CUDA for --devices cuda)")
    p.add_argument("--devices", default="cpu", help="comma-separated: " + ", ".join(DEVICES))
    p.add_argument("--nets", default=",".join(NETS))
    p.add_argument("--epochs", type=int)
    p.add_argument("--rows", type=int)
    p.add_argument("--hardware", help="CPU and GPU names, kept for the machine once given")
    p.add_argument("--os", help="operating system, kept for the machine once given")
    p.add_argument("--issue", type=int, default=361)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--note", action="append", default=[], help="add a note (repeatable)")
    p.add_argument("--todo", action="append", default=[], help="add a to-do item (repeatable)")
    p.add_argument("--remove-note", action="append", default=[], help="remove a note by its exact text")
    p.add_argument("--remove-todo", action="append", default=[], help="remove a to-do item by its exact text")
    p.add_argument("--import", dest="import_path", help="merge rows, notes and to-do items from a JSON file")
    p.add_argument("--replace-body", action="store_true", help="rewrite an issue that has no data comment")
    args = p.parse_args()

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
    nets = [n for n in args.nets.split(",") if n]
    for d in devices:
        if d not in DEVICES:
            sys.exit(f"unknown device {d}; choose from {', '.join(DEVICES)}")
    for n in nets:
        if n not in NETS:
            sys.exit(f"unknown network {n}; choose from {', '.join(NETS)}")
    if any(DEVICES[d]["torch"] for d in devices) and not args.python:
        sys.exit("--python is needed for cpu, cuda and metal")
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

    features = sorted({DEVICES[d]["feature"] for d in devices if DEVICES[d]["feature"]})
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
    for device in devices:
        for net in nets:
            try:
                row = measure(root, binary, tdir, device, net, args)
            except (subprocess.CalledProcessError, OSError, KeyError, ValueError) as e:
                sys.exit(f"stopped: {net} on {device}: {e}; the rows measured before it are in the issue")
            row.update(machine=args.machine, device=device, net=net, mode="", commit=commit, date=date)
            changes.append(lambda d, first, row=row: add_row(d, dict(row)))
            publish(changes, args, root)


def edits(args):
    """The steps for --import, --note, --todo and their removals. Each takes the issue's data
    and whether it is the first time it runs: a removal of text that is not there stops the
    script the first time, and is skipped on a later read where someone already removed it."""
    steps = []
    if args.import_path:
        steps.append(lambda d, first: import_file(d, args.import_path))
    for kind, added, removed in [("notes", args.note, args.remove_note), ("todo", args.todo, args.remove_todo)]:
        for text in removed:
            def remove(d, first, kind=kind, text=text):
                if text in d[kind]:
                    d[kind].remove(text)
                elif first:
                    sys.exit(f"no {kind} entry reads exactly: {text}")
            steps.append(remove)
        for text in added:
            def add(d, first, kind=kind, text=text):
                if text not in d[kind]:
                    d[kind].append(text)
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
