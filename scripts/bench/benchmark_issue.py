# Runs `cavitation-bench mnist-cnn` and `cifar-cnn` on this machine's devices at the checked-out
# commit and rewrites the GitHub issue that tracks speed against PyTorch, so the issue always
# shows the latest measurement of each machine, device and network.
#
#   python scripts/bench/benchmark_issue.py --machine B --python <python with torch> \
#       --devices cpu,cuda,vulkan,dx12 [--nets mnist-cnn,cifar-cnn] [--epochs N] [--rows N] \
#       [--hardware "AMD Ryzen 7 2700X; NVIDIA GTX 1660 Super"] [--os "Windows 10"] \
#       [--issue 361] [--dry-run] [--render-only]
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
# The issue body keeps every row as JSON in a hidden comment at its end. A run replaces the rows
# of its machine, device and network and leaves the others. `--render-only` rewrites the issue
# from the rows it already holds; `--dry-run` prints the new body instead of editing the issue.
# CUDA runs in its default mode (`CAVITATION_CUDNN_BENCHMARK` unset), the repeatable one.

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


def read_issue(issue):
    body = run(["gh", "issue", "view", str(issue), "--json", "body", "-q", ".body"])
    start = body.find(DATA_START)
    if start < 0:
        return {"machines": {}, "rows": []}
    end = body.find(DATA_END, start)
    if end < 0:
        sys.exit(f"issue #{issue}: the benchmark data comment has no end")
    return json.loads(body[start + len(DATA_START) : end])


def speed(row):
    if row["torch_s"] is None:
        return "❌"
    if row["cav_s"] <= 0 or row["torch_s"] <= 0:
        return "n/a (an epoch took no measurable time)"
    r = row["cav_s"] / row["torch_s"]
    if r < 0.97:
        return f"🟢 {1 / r:.2f}× faster"
    if r > 1.03:
        return f"🔴 {r:.2f}× slower"
    return "🟡 same (within 3%)"


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
        out.append(f"| {name} | {m['hardware']} | {m['os']} | {devs} |")
    out.append("")
    out.append("## Speed")
    out.append("")
    out.append("| Machine | Device | " + " | ".join(NETS) + " |")
    out.append("|---|---|" + "---|" * len(NETS))
    missing = []
    for name in sorted(data["machines"]):
        for device in data["machines"][name]["devices"]:
            cells = []
            for net in NETS:
                row = next((r for r in rows if (r["machine"], r["device"], r["net"]) == (name, device, net)), None)
                if row is None:
                    cells.append("❌" if not DEVICES[device]["torch"] else "⬜")
                    missing.append(f"{net} on machine {name}, {DEVICE_NAMES[device]}")
                    continue
                cell = speed(row)
                if row["torch_s"] is None:
                    cell += f" (Cavitation {row['cav_s']:.2f} s an epoch)"
                cells.append(cell)
            out.append(f"| {name} | {DEVICE_NAMES[device]} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("## Measurements")
    out.append("")
    out.append("| Machine | Device | Network | Cavitation epoch | PyTorch epoch | Epochs × rows | Threads, kernel | Commit | Date | PyTorch |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    order = {d: i for i, d in enumerate(DEVICES)}
    for row in sorted(rows, key=lambda r: (r["machine"], order[r["device"]], NETS.index(r["net"]))):
        torch = f"{row['torch_s']:.2f} s" if row["torch_s"] is not None else "❌"
        about = row["torch_about"] or "–"
        out.append(
            f"| {row['machine']} | {DEVICE_NAMES[row['device']]} | {row['net']} | {row['cav_s']:.2f} s | {torch} "
            f"| {row['epochs']} × {row['rows']:,} | {row['threads']}, {row['kernel']} | {row['commit']} "
            f"| {row['date']} | {about} |"
        )
    out.append("")
    out.append("Commands, as each row ran them (the release build of `cavitation-bench`, with `CAVITATION_THREADS=4`):")
    out.append("")
    out.append("```")
    seen = []
    for row in rows:
        if row["command"] not in seen:
            seen.append(row["command"])
    out.extend(seen)
    out.append("```")
    out.append("")
    out.append("## Not measured yet")
    out.append("")
    for m in missing:
        out.append(f"- [ ] {m}")
    if not missing:
        out.append("Every machine, device and network listed above has a row.")
    out.append("")
    out.append(DATA_START)
    out.append(json.dumps({"machines": data["machines"], "rows": rows}, ensure_ascii=False, indent=1).replace(">", "\\u003e"))
    out.append(DATA_END)
    return "\n".join(out) + "\n"


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
    args = p.parse_args()

    root = repo_root()
    data = read_issue(args.issue)

    if not args.render_only:
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
        if args.hardware:
            machine["hardware"] = args.hardware
        if args.os:
            machine["os"] = args.os
        if not machine["hardware"] or not machine["os"]:
            sys.exit(f"machine {args.machine} is new: give --hardware and --os")
        machine["devices"] = [d for d in DEVICES if d in machine["devices"] or d in devices]
        data["machines"][args.machine] = machine

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
        failed = None
        for device in devices:
            for net in nets:
                try:
                    row = measure(root, binary, tdir, device, net, args)
                except (subprocess.CalledProcessError, OSError, KeyError, ValueError) as e:
                    failed = f"{net} on {device}: {e}"
                    break
                row.update(machine=args.machine, device=device, net=net, commit=commit, date=date)
                data["rows"] = [
                    r for r in data["rows"] if (r["machine"], r["device"], r["net"]) != (args.machine, device, net)
                ]
                data["rows"].append(row)
                publish(data, args, root)
            if failed:
                break
        if failed:
            sys.exit(f"stopped: {failed}; the rows measured before it are in the issue")
        return

    publish(data, args, root)


def publish(data, args, root):
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
