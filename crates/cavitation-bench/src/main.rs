//! Benchmark runner for the cavitation neural network library.
//!
//! `cavitation-bench <all|mnist|cifar|digits|latency|gemm|subnormal|small|shrink|inspect|info>`:
//!
//! - `info` prints the CPU, the kernel levels, the thread count and the compiler.
//! - `mnist [--batch 32|128] [--adam]` times epochs of the 1M network on MNIST;
//!   `digits` times epochs of the 50k network on the digits; `latency` times one input
//!   through the 50k network. Each also runs `scripts/bench/numpy_mlp.py` with the same
//!   network, batch, data and step counts, and prints cavitation against numpy with the
//!   ratio. The results are also written to `target/bench/results.json`.
//! - `cifar [--batch 32|128] [--steps N]` times 3 epochs of the 3072 → 1024 → 512 → 10
//!   network (ReLU, cross-entropy, Adam lr 1e-3) on CIFAR-10's 50,000 training images, at
//!   batch 128 and then 32 (or only the batch given), against the same numpy baseline. With
//!   `--steps N` each timed sample is N steps instead of an epoch and the result is steps
//!   per second, for machines on which an epoch takes minutes.
//! - `all` runs `info`, `mnist` at batch 128 and 32, `digits` and `latency`; not `cifar`,
//!   whose epochs take minutes on a small machine.
//! - `gemm --shapes mnist128|digits128` times the eight matrix products of one batch-128
//!   training step of the named network at the current kernel level and thread count
//!   (`CAVITATION_KERNEL`, `CAVITATION_THREADS`), and prints each product's median time,
//!   GFLOPS (`2·m·n·k` / time) and threads used, then the rate over all eight.
//! - `subnormal [--steps N]` (default 2,000) times the optimiser update of one parameter of
//!   1,068,810 elements whose gradient is 1e-3 on the first step and 0 after, for SGD,
//!   SGD with momentum 0.9 and Adam (lr 1e-3), so their state decays through the subnormal
//!   range. It prints the median update time of each block of 100 steps, and the ratio of
//!   the blocks from step 901 on to the blocks of steps 1 to 700.
//! - `small [--rounds N]` (default 3) trains a small network (4 → 32 → tanh → 32 → tanh →
//!   1, mean squared error, Adam lr 1e-3, batch 64, 20,000 seeded rows, 10 timed epochs) at
//!   1 and at 4 threads, in interleaved rounds, each run in a child process with
//!   `CAVITATION_THREADS` set. It prints each run's wall-clock and CPU seconds, then the
//!   medians and the 4-thread figures over the 1-thread ones, and checks that every run
//!   ended on the same loss. CPU time is read from `/proc/self/stat`, so it runs on Linux
//!   only; its ticks are taken as 1/100 s (`USER_HZ` = 100, as on x86 and Arm Linux).
//!   `small --child` is the child's side: one run at the current thread count.
//! - `shrink` trains the digits network (64 → 256 → 128 → 10, the digits known result's
//!   recipe) on the first 1,149 training digits and shrinks it twice with the last 288 as
//!   the validation set: once ranking by outgoing-weight norm with `dead_first` off and
//!   every level run, once with the defaults. It prints both curves (validation and test
//!   accuracy at every level), checks that the compacted networks give the masked ones'
//!   outputs on the 360 test digits through `infer` and `infer_batch`, then times `infer`
//!   on one test digit for the full network, the full network with the first run's masks,
//!   and the first run's network compacted: 20,000 warm-up calls each, then 301 interleaved
//!   rounds of 1,000 calls of each, and prints the medians and the speed-ups.
//! - `inspect [--rounds N]` (default 8, a multiple of 4) measures what the inspector costs
//!   on the `digits` training (the 50k network, SGD lr 0.01, batch 128, 100 epochs per run,
//!   timed around the `fit` call) with the 360 test digits as the probe set and the default
//!   `eval_rows` and `snapshot_hz`. It first trains, untimed, for at least 30 s in every mode
//!   in turn, past the package power-limit window, so every round runs at the same power
//!   limit. Each round runs four modes, starting one mode later every round: off
//!   (`cavitation_nn::fit`), off again (the A/A pair), on with one client reading `/events`
//!   on its own thread, and on with one client connected and not reading (both
//!   `cavitation_inspect::fit`). It prints every round's times and ratios (off's time over
//!   the mode's: 1 is no cost, 0.98 is 2% slower), the median ratios, the off/off spread (the
//!   median over rounds of `|1 − off / off again|`: how far a typical round strays from 1,
//!   the machine's noise) and whether each passes.
//!
//! The baseline is run with `python3` from `PATH`; `--python <path>` or the
//! `CAVITATION_BENCH_PYTHON` variable names another interpreter (it needs numpy). Both sides
//! of a training comparison time the same steps (see [`runs::TRAIN_TIMED_STEPS`]). The
//! baseline reports the BLAS numpy runs on (library, version, OpenBLAS core, threads), which
//! is printed beside numpy's numbers.
//!
//! With `--numpy-best` (on `mnist`, `cifar`, `digits`, `latency` and `all`), numpy is also
//! timed at one BLAS thread, and, when OpenBLAS runs kernels older than the CPU can (an
//! SSE4.2 CPU on `Katmai`, say), at the newest core the CPU's features allow (`Nehalem` for
//! SSE4.2, `Sandybridge` for AVX, `Haswell` for AVX2 with FMA) with the default thread count
//! and with one thread; each setting goes to the child as environment variables. The
//! comparison quotes the fastest setting, with the default one beside it.

mod machine;
mod numpy;
mod runs;

use std::path::PathBuf;
use std::process::ExitCode;

use cavitation_core::kernels::{detected, level, threads};

use runs::{Comparison, TrainSpec, Unit};

/// How to call the runner, printed on a usage error.
const USAGE: &str = "usage: cavitation-bench <all|mnist|cifar|digits|latency|gemm|subnormal|small|shrink|inspect|info> [options]\n\
    \x20 mnist   [--batch 32|128] [--adam] [--python PATH] [--numpy-best]\n\
    \x20 cifar   [--batch 32|128] [--steps N] [--python PATH] [--numpy-best]\n\
    \x20 digits  [--python PATH] [--numpy-best]\n\
    \x20 latency [--python PATH] [--numpy-best]\n\
    \x20 all     [--python PATH] [--numpy-best]\n\
    \x20 gemm    [--shapes mnist128|digits128]\n\
    \x20 subnormal [--steps N]\n\
    \x20 small   [--rounds N]\n\
    \x20 shrink\n\
    \x20 inspect [--rounds N]\n\
    \x20 info";

/// Entry point of the benchmark runner.
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let rest = &args[1..];
    let result = match command.as_str() {
        "info" => rest
            .is_empty()
            .then(info)
            .ok_or_else(|| "info takes no arguments".to_string()),
        "gemm" => gemm(rest),
        "subnormal" => subnormal(rest),
        "small" => small(rest),
        "inspect" => inspect(rest),
        "shrink" => {
            if rest.is_empty() {
                shrink()
            } else {
                Err("shrink takes no arguments".to_string())
            }
        }
        "mnist" | "cifar" | "digits" | "latency" | "all" => Options::parse(rest).and_then(|o| {
            let mut rows = Vec::new();
            if command == "all" {
                info();
            }
            for run in plan(command, &o)? {
                rows.push(execute(run, &o)?);
            }
            report(&rows, o.numpy_best)
        }),
        _ => Err(format!("unknown command `{command}`")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("cavitation-bench {command}: {e}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// Prints the machine and the kernel settings.
fn info() {
    let machine = machine::detect();
    println!("cpu: {}", machine.cpu_model);
    println!("logical cores: {}", machine.logical_cores);
    println!("detected kernel level: {}", detected());
    println!("current kernel level: {}", level());
    println!("threads: {}", threads());
    println!("rustc: {}", machine::rustc_version());
}

/// The options of the training and latency commands.
struct Options {
    /// `--batch`: 32 or 128 (`mnist` and `cifar` only).
    batch: Option<usize>,
    /// `--adam`: time Adam instead of plain SGD (`mnist` only).
    adam: bool,
    /// `--steps`: steps per timed sample instead of whole epochs (`cifar` only).
    steps: Option<usize>,
    /// The Python interpreter that runs the baseline.
    python: String,
    /// `--numpy-best`: also time numpy at the other BLAS settings and compare against the
    /// fastest.
    numpy_best: bool,
}

impl Options {
    /// Parses the options after the command name.
    ///
    /// # Errors
    ///
    /// Returns a message for an unknown option, a missing value, a batch other than 32
    /// or 128, or a `--steps` that is not a positive number.
    fn parse(args: &[String]) -> Result<Options, String> {
        let mut o = Options {
            batch: None,
            adam: false,
            steps: None,
            python: std::env::var("CAVITATION_BENCH_PYTHON").unwrap_or_else(|_| "python3".into()),
            numpy_best: false,
        };
        let mut rest = args.iter();
        while let Some(arg) = rest.next() {
            match arg.as_str() {
                "--batch" => {
                    let v = rest.next().ok_or("--batch needs a value")?;
                    o.batch = match v.as_str() {
                        "32" => Some(32),
                        "128" => Some(128),
                        _ => return Err(format!("--batch must be 32 or 128, got `{v}`")),
                    };
                }
                "--steps" => {
                    let v = rest.next().ok_or("--steps needs a value")?;
                    o.steps =
                        Some(v.parse().ok().filter(|&n| n > 0).ok_or_else(|| {
                            format!("--steps must be a positive number, got `{v}`")
                        })?);
                }
                "--adam" => o.adam = true,
                "--numpy-best" => o.numpy_best = true,
                "--python" => o.python = rest.next().ok_or("--python needs a value")?.clone(),
                other => return Err(format!("unknown argument `{other}`")),
            }
        }
        Ok(o)
    }
}

/// One run of a command.
enum Run {
    /// A training comparison.
    Train(TrainSpec),
    /// The latency comparison.
    Latency,
}

/// The runs a command stands for.
///
/// # Errors
///
/// Returns a message if an option does not apply to the command.
fn plan(command: &str, o: &Options) -> Result<Vec<Run>, String> {
    let mnist = |batch: usize, adam: bool| {
        Run::Train(TrainSpec {
            name: format!("mnist batch {batch}{}", if adam { " (Adam)" } else { "" }),
            data: "mnist",
            widths: runs::MNIST_NET,
            batch,
            adam,
            epochs: 3,
            steps: None,
            unit: Unit::EpochSeconds,
        })
    };
    let cifar = |batch: usize| {
        let steps = o.steps.map_or(String::new(), |n| format!(", {n} steps"));
        Run::Train(TrainSpec {
            name: format!("cifar batch {batch}{steps}"),
            data: "cifar",
            widths: runs::CIFAR_NET,
            batch,
            adam: true,
            epochs: 3,
            steps: o.steps,
            unit: o.steps.map_or(Unit::EpochSeconds, |_| Unit::StepsPerSecond),
        })
    };
    let digits = Run::Train(TrainSpec {
        name: "digits batch 128".to_string(),
        data: "digits",
        widths: runs::DIGITS_NET,
        batch: 128,
        adam: false,
        epochs: 100,
        steps: None,
        unit: Unit::SamplesPerSecond,
    });
    if command != "mnist" && o.adam {
        return Err(format!("--adam applies to mnist only, not {command}"));
    }
    if command != "mnist" && command != "cifar" && o.batch.is_some() {
        return Err(format!(
            "--batch applies to mnist and cifar only, not {command}"
        ));
    }
    if command != "cifar" && o.steps.is_some() {
        return Err(format!("--steps applies to cifar only, not {command}"));
    }
    Ok(match command {
        "mnist" => vec![mnist(o.batch.unwrap_or(128), o.adam)],
        "cifar" => match o.batch {
            Some(batch) => vec![cifar(batch)],
            None => vec![cifar(128), cifar(32)],
        },
        "digits" => vec![digits],
        "latency" => vec![Run::Latency],
        _ => vec![mnist(128, false), mnist(32, false), digits, Run::Latency],
    })
}

/// Runs one comparison.
fn execute(run: Run, o: &Options) -> Result<Comparison, String> {
    match run {
        Run::Train(spec) => {
            eprintln!("{}: cavitation, then numpy", spec.name);
            runs::compare_training(&spec, &o.python, o.numpy_best)
        }
        Run::Latency => {
            eprintln!("latency: cavitation, then numpy");
            runs::compare_latency(&o.python, o.numpy_best)
        }
    }
}

/// Prints the table of cavitation against numpy and writes `target/bench/results.json`.
/// With `numpy_best`, numpy's column is its fastest setting and the default one is beside it.
///
/// # Errors
///
/// Returns a message if the results file cannot be written.
fn report(rows: &[Comparison], numpy_best: bool) -> Result<(), String> {
    let load = machine::load_average();
    println!(
        "kernel {}, {} threads, load average at the end {load}",
        level(),
        threads()
    );
    if let Some(first) = rows.first() {
        println!(
            "numpy {}; BLAS at numpy's default settings: {}",
            first.numpy_version, first.blas_default
        );
    }
    if numpy_best {
        println!(
            "{:<28} {:>12} {:>12} {:>9} {:>14} {:>9}  unit",
            "run", "cavitation", "numpy best", "ratio", "numpy default", "ratio"
        );
        for row in rows {
            println!(
                "{:<28} {:>12.2} {:>12.2} {:>8.2}x {:>14.2} {:>8.2}x  {}",
                row.name,
                row.value(runs::median(&row.ours)),
                row.value(runs::median(&row.numpy)),
                row.ratio(),
                row.value(runs::median(&row.numpy_default)),
                row.default_ratio(),
                row.unit_label()
            );
        }
        println!(
            "ratio = numpy time over cavitation time; medians of the timed epochs or rounds; \
             numpy best = the fastest numpy setting tried."
        );
        for row in rows {
            println!(
                "{}: numpy best at {} (BLAS: {}); tried: {}",
                row.name,
                row.numpy_setting,
                row.blas,
                row.tried_summary()
            );
        }
    } else {
        println!(
            "{:<28} {:>12} {:>12} {:>10}  unit",
            "run", "cavitation", "numpy", "ratio"
        );
        for row in rows {
            println!(
                "{:<28} {:>12.2} {:>12.2} {:>9.2}x  {}",
                row.name,
                row.value(runs::median(&row.ours)),
                row.value(runs::median(&row.numpy)),
                row.ratio(),
                row.unit_label()
            );
        }
        println!(
            "ratio = numpy time over cavitation time; medians of the timed epochs or rounds; \
             numpy at its default settings (--numpy-best also tries others)."
        );
    }
    for row in rows {
        println!("{}: {}; {}", row.name, row.detail, row.timing_note);
    }
    for row in rows.iter().filter(|r| !r.health.is_empty()) {
        let numpy = row
            .numpy_warnings
            .map_or(String::new(), |n| format!("; numpy's checks: {n} warnings"));
        println!("{}: {}{numpy}", row.name, row.health);
    }
    write_results(rows, &load, numpy_best)
}

/// The JSON text of a list of numbers.
fn json_list(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|v| format!("{v:e}")).collect();
    format!("[{}]", items.join(", "))
}

/// Writes `target/bench/results.json` at the repository root.
///
/// # Errors
///
/// Returns a message if the directory or file cannot be written.
fn write_results(rows: &[Comparison], load: &str, numpy_best: bool) -> Result<(), String> {
    let machine = machine::detect();
    let escape = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let runs: Vec<String> = rows
        .iter()
        .map(|r| {
            format!(
                "    {{\"name\": \"{}\", \"detail\": \"{}\", \"health\": \"{}\", \"timing\": \"{}\", \"unit\": \"{}\", \
                 \"cavitation\": {:e}, \"numpy\": {:e}, \"ratio\": {:e},\n     \
                 \"numpy_setting\": \"{}\", \"blas\": \"{}\", \"numpy_default\": {:e}, \"default_ratio\": {:e},\n     \
                 \"numpy_tried\": \"{}\",\n     \
                 \"cavitation_seconds\": {}, \"numpy_seconds\": {}, \"numpy_default_seconds\": {}}}",
                escape(&r.name),
                escape(&r.detail),
                escape(&r.health),
                escape(&r.timing_note),
                r.unit_label(),
                r.value(runs::median(&r.ours)),
                r.value(runs::median(&r.numpy)),
                r.ratio(),
                escape(&r.numpy_setting),
                escape(&r.blas),
                r.value(runs::median(&r.numpy_default)),
                r.default_ratio(),
                escape(&r.tried_summary()),
                json_list(&r.ours),
                json_list(&r.numpy),
                json_list(&r.numpy_default)
            )
        })
        .collect();
    let text = format!(
        "{{\n  \"cpu\": \"{}\",\n  \"logical_cores\": {},\n  \"kernel\": \"{}\",\n  \
         \"threads\": {},\n  \"rustc\": \"{}\",\n  \"numpy_version\": \"{}\",\n  \"blas_default\": \"{}\",\n  \"numpy_best\": {numpy_best},\n  \"load_average\": \"{}\",\n  \"runs\": [\n{}\n  ]\n}}\n",
        escape(&machine.cpu_model),
        machine.logical_cores,
        level(),
        threads(),
        escape(&machine::rustc_version()),
        escape(rows.first().map_or("", |r| &r.numpy_version)),
        escape(rows.first().map_or("", |r| &r.blas_default)),
        load,
        runs.join(",\n")
    );
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/bench");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join("results.json");
    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}

/// Runs the `gemm` subcommand with its arguments (after `gemm`).
///
/// # Errors
///
/// Returns a message for an unknown argument or shape set.
fn gemm(args: &[String]) -> Result<(), String> {
    let mut set = "mnist128".to_string();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--shapes" => {
                set = rest.next().ok_or("--shapes needs a value")?.clone();
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    let products = runs::shapes(&set)?;
    let (level, threads) = (level(), threads());
    println!("gemm {set}: kernel {level}, {threads} threads, median of 31 samples");
    println!(
        "{:<22} {:>16} {:>10} {:>8} {:>8}",
        "product", "m x n x k", "time us", "GFLOPS", "threads"
    );
    let (mut flops, mut seconds) = (0.0, 0.0);
    for product in products {
        let t = runs::time_product(level, threads, product);
        flops += product.flops();
        seconds += t.seconds;
        println!(
            "{:<22} {:>16} {:>10.1} {:>8.2} {:>8}",
            product.name,
            format!("{}x{}x{}", product.m, product.n, product.k),
            t.seconds * 1e6,
            product.flops() / t.seconds / 1e9,
            t.threads
        );
    }
    println!(
        "all eight: {:.1} us, {:.2} GFLOPS (total FLOPs over the sum of the median times)",
        seconds * 1e6,
        flops / seconds / 1e9
    );
    Ok(())
}

/// Runs the `subnormal` subcommand with its arguments (after `subnormal`).
///
/// # Errors
///
/// Returns a message for an unknown argument or a `--steps` that is not a positive number.
fn subnormal(args: &[String]) -> Result<(), String> {
    let mut steps = 2000usize;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--steps" => {
                let v = rest.next().ok_or("--steps needs a value")?;
                steps = v
                    .parse()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| format!("--steps must be a positive number, got `{v}`"))?;
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    println!(
        "subnormal: one parameter of {} elements, gradient 1e-3 on step 1 and 0 after, \
         kernel {}, {} threads",
        runs::SUBNORMAL_ELEMENTS,
        level(),
        threads()
    );
    println!(
        "median update time in ms per block of {} steps",
        runs::BLOCK
    );
    for run in runs::subnormal(steps) {
        println!("{}", run.name);
        for (b, t) in run.blocks.iter().enumerate() {
            let (first, last) = (b * runs::BLOCK + 1, (b + 1) * runs::BLOCK);
            println!("  steps {first:>5}-{last:<5} {:>8.3}", t * 1e3);
        }
        match run.ratio {
            Some(r) => println!("  steps 901+ over steps 1-700: {r:.2}x"),
            None => println!("  steps 901+ over steps 1-700: needs at least 1,000 steps"),
        }
    }
    Ok(())
}

/// The thread counts the `small` run compares; the first is the baseline.
const SMALL_THREADS: [usize; 2] = [1, 4];

/// Runs the `small` subcommand with its arguments (after `small`).
///
/// # Errors
///
/// Returns a message for an unknown argument, a `--rounds` that is not a positive number,
/// a child run that fails, or runs that ended on different losses.
fn small(args: &[String]) -> Result<(), String> {
    let mut rounds = 3usize;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--child" => {
                let run = runs::small_once()?;
                println!(
                    "wall {:e} cpu {:e} loss {:08x}",
                    run.wall, run.cpu, run.loss_bits
                );
                return Ok(());
            }
            "--rounds" => {
                let v = rest.next().ok_or("--rounds needs a value")?;
                rounds = v
                    .parse()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| format!("--rounds must be a positive number, got `{v}`"))?;
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    let net = runs::SMALL_NET.map(|w| w.to_string()).join(" → ");
    println!(
        "small: {net} with tanh, MSE, Adam lr 1e-3, batch {}, {} rows, {} timed epochs, \
         kernel {}, {rounds} interleaved rounds",
        runs::SMALL_BATCH,
        runs::SMALL_ROWS,
        runs::SMALL_EPOCHS,
        level()
    );
    let mut results: Vec<Vec<runs::SmallRun>> = vec![Vec::new(); SMALL_THREADS.len()];
    for round in 1..=rounds {
        for (i, &threads) in SMALL_THREADS.iter().enumerate() {
            let run = small_child(threads)?;
            println!(
                "round {round}, {threads} thread{}: wall {:.3} s, cpu {:.2} s, loss bits {:08x}",
                if threads == 1 { "" } else { "s" },
                run.wall,
                run.cpu,
                run.loss_bits
            );
            results[i].push(run);
        }
    }
    let medians: Vec<(f64, f64)> = results
        .iter()
        .map(|r| {
            let wall: Vec<f64> = r.iter().map(|x| x.wall).collect();
            let cpu: Vec<f64> = r.iter().map(|x| x.cpu).collect();
            (runs::median(&wall), runs::median(&cpu))
        })
        .collect();
    for (&threads, &(wall, cpu)) in SMALL_THREADS.iter().zip(&medians) {
        let s = if threads == 1 { "" } else { "s" };
        println!("median, {threads} thread{s}: wall {wall:.3} s, cpu {cpu:.2} s");
    }
    let (base_wall, base_cpu) = medians[0];
    for (&threads, &(wall, cpu)) in SMALL_THREADS.iter().zip(&medians).skip(1) {
        println!(
            "{threads} threads over 1: wall {:.3}x, cpu {:.3}x",
            wall / base_wall,
            cpu / base_cpu
        );
    }
    let first = results[0][0].loss_bits;
    if results.iter().flatten().any(|r| r.loss_bits != first) {
        return Err("the runs ended on different losses".to_string());
    }
    println!("every run ended on the same loss");
    Ok(())
}

/// Runs `small --child` in a new process of this program with `CAVITATION_THREADS` set to
/// `threads`, and reads back its result line.
///
/// # Errors
///
/// Returns a message if the child cannot start, fails, or prints something unexpected.
fn small_child(threads: usize) -> Result<runs::SmallRun, String> {
    let exe = std::env::current_exe().map_err(|e| format!("this program's path: {e}"))?;
    let out = std::process::Command::new(exe)
        .args(["small", "--child"])
        .env("CAVITATION_THREADS", threads.to_string())
        .output()
        .map_err(|e| format!("starting the {threads}-thread run: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    if !out.status.success() {
        return Err(format!(
            "the {threads}-thread run failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    parse_small_line(text.trim())
        .ok_or_else(|| format!("the {threads}-thread run printed `{}`", text.trim()))
}

/// Reads the line `small --child` prints: `wall <seconds> cpu <seconds> loss <hex bits>`.
fn parse_small_line(line: &str) -> Option<runs::SmallRun> {
    match line.split_whitespace().collect::<Vec<_>>()[..] {
        ["wall", wall, "cpu", cpu, "loss", loss] => Some(runs::SmallRun {
            wall: wall.parse().ok()?,
            cpu: cpu.parse().ok()?,
            loss_bits: u32::from_str_radix(loss, 16).ok()?,
        }),
        _ => None,
    }
}

/// Runs the `inspect` subcommand with its arguments (after `inspect`): the four modes in
/// rotating order for the rounds asked, then the medians and the criteria.
///
/// # Errors
///
/// Returns a message for an unknown argument, a `--rounds` that is not a positive multiple
/// of 4, a run that fails, or runs that ended on different losses.
fn inspect(args: &[String]) -> Result<(), String> {
    use runs::InspectMode;
    let mut rounds = runs::INSPECT_ROUNDS;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--rounds" => {
                let v = rest.next().ok_or("--rounds needs a value")?;
                rounds = v
                    .parse()
                    .ok()
                    .filter(|&n: &usize| n > 0 && n % 4 == 0)
                    .ok_or_else(|| {
                        format!(
                            "--rounds must be a positive multiple of 4, so every mode runs in \
                             every position equally often; got `{v}`"
                        )
                    })?;
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    let whole = std::time::Instant::now();
    let cpu_start = machine::process_cpu_seconds();
    let machine = machine::detect();
    println!(
        "inspect: {} ({} logical cores), kernel {}, {} threads, {}",
        machine.cpu_model,
        machine.logical_cores,
        level(),
        threads(),
        machine::rustc_version()
    );
    println!("load average at the start {}", machine::load_average());
    let data = runs::inspect_data()?;
    println!(
        "training: the digits benchmark, {} network (ReLU, He normal from seed 1), SGD lr 0.01, \
         batch 128, {} rows in order, Health::quiet(), {} epochs per run, timed around fit",
        runs::DIGITS_NET.map(|w| w.to_string()).join(","),
        data.train.len(),
        runs::INSPECT_EPOCHS
    );
    println!(
        "inspector: probe set the {} test digits, eval_rows {} ({} used), snapshot_hz {}, \
         a new inspector and client for each run, connected before the clock starts",
        data.test.len(),
        cavitation_inspect::InspectorConfig::default().eval_rows,
        cavitation_inspect::InspectorConfig::default()
            .eval_rows
            .min(data.test.len()),
        cavitation_inspect::InspectorConfig::default().snapshot_hz
    );
    let warm = std::time::Instant::now();
    let mut warm_runs = 0;
    while warm.elapsed() < runs::INSPECT_WARMUP || warm_runs < InspectMode::ALL.len() {
        let mode = InspectMode::ALL[warm_runs % InspectMode::ALL.len()];
        runs::inspect_once(mode, &data, runs::INSPECT_EPOCHS)?;
        warm_runs += 1;
    }
    println!(
        "warm-up: {warm_runs} untimed runs, the modes in turn, {:.1} s (at least {} s, past the \
         power-limit window)",
        warm.elapsed().as_secs_f64(),
        runs::INSPECT_WARMUP.as_secs()
    );
    let samples = (data.train.len() * runs::INSPECT_EPOCHS) as f64;
    let mut seconds = [const { Vec::new() }; 4];
    let mut loss_bits = None;
    let timed = std::time::Instant::now();
    for round in 0..rounds {
        let order: Vec<InspectMode> = (0..4).map(|k| InspectMode::ALL[(round + k) % 4]).collect();
        let mut parts = Vec::new();
        let mut notes = Vec::new();
        for &mode in &order {
            let run = runs::inspect_once(mode, &data, runs::INSPECT_EPOCHS)?;
            if *loss_bits.get_or_insert(run.loss_bits) != run.loss_bits {
                return Err(format!(
                    "round {}: {} ended on another loss",
                    round + 1,
                    mode.name()
                ));
            }
            seconds[mode as usize].push(run.seconds);
            parts.push(format!(
                "{} {:.4} s ({:.0} samples/s)",
                mode.name(),
                run.seconds,
                samples / run.seconds
            ));
            if mode.inspected() {
                notes.push(format!(
                    "{}: {} snapshots, {} client at the end",
                    mode.name(),
                    run.snapshots,
                    run.clients_at_end
                ));
            }
        }
        let off = seconds[0][round];
        let ratios: Vec<String> = InspectMode::ALL[1..]
            .iter()
            .map(|&m| format!("{} {:.4}", m.name(), off / seconds[m as usize][round]))
            .collect();
        println!("round {}: {}", round + 1, parts.join(", "));
        println!(
            "  ratio to off: {}; {}",
            ratios.join(", "),
            notes.join("; ")
        );
    }
    let timed_seconds = timed.elapsed().as_secs_f64();
    let ratio = |m: InspectMode| -> Vec<f64> {
        seconds[0]
            .iter()
            .zip(&seconds[m as usize])
            .map(|(off, t)| off / t)
            .collect()
    };
    println!("ratio = off's time over the mode's time in the same round (below 1 is slower)");
    for mode in InspectMode::ALL {
        let t = &seconds[mode as usize];
        println!(
            "median {}: {:.4} s ({:.0} samples/s)",
            mode.name(),
            runs::median(t),
            samples / runs::median(t)
        );
    }
    let aa = ratio(InspectMode::OffAgain);
    let spread = runs::aa_spread(&aa);
    let (lo, hi) = aa
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &r| {
            (lo.min(r), hi.max(r))
        });
    println!(
        "off / off again: median ratio {:.4}; spread (median over rounds of |1 - ratio|) \
         {:.2}%; rounds {lo:.4} to {hi:.4}",
        runs::median(&aa),
        spread * 100.0
    );
    let verdict = |ok: bool| if ok { "met" } else { "not met" };
    let mut worst = f64::INFINITY;
    for mode in [InspectMode::Reading, InspectMode::Connected] {
        let r = ratio(mode);
        let m = runs::median(&r);
        worst = worst.min(m);
        let (lo, hi) = r
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                (lo.min(x), hi.max(x))
            });
        println!(
            "{}: median ratio {m:.4} ({:+.2}% time; rounds {lo:.4} to {hi:.4}); at least 0.98: {}",
            mode.name(),
            (1.0 / m - 1.0) * 100.0,
            verdict(m >= 0.98)
        );
    }
    println!("off/off spread at most 1%: {}", verdict(spread <= 0.01));
    println!(
        "worse \"on\" mode: median ratio {worst:.4}, {:.2}% slower than off",
        (1.0 - worst) * 100.0
    );
    println!(
        "rounds {timed_seconds:.1} s; whole run {:.1} s",
        whole.elapsed().as_secs_f64()
    );
    if let (Some(a), Some(b)) = (cpu_start, machine::process_cpu_seconds()) {
        println!("cpu time of this process {:.1} s", b - a);
    }
    println!("load average at the end {}", machine::load_average());
    Ok(())
}

/// The hidden neurons of the digits network: 256 + 128.
const HIDDEN: usize = 384;

/// Prints one `shrink` run's curve: a line per level with the neurons kept, the parameters
/// after compaction, validation and test accuracy, validation loss and whether the step met
/// the criterion; then the chosen step.
fn print_curve(name: &str, run: &runs::ShrinkRun, start: &runs::ShrinkStart) {
    let (val_rows, test_rows) = (start.validation.len(), start.test.len());
    let base = run.baseline.accuracy.unwrap_or(f64::NAN);
    println!(
        "{name}: unpruned validation {}/{val_rows} ({:.2}%), loss {:.4}",
        runs::rows_right(base, &start.validation),
        base * 100.0,
        run.baseline.loss
    );
    println!(
        "  {:>7} {:>8} {:>7} {:>16} {:>9} {:>16}  met",
        "removed", "kept", "params", "validation", "val loss", "test"
    );
    for point in &run.curve {
        let step = &point.step;
        let val = step.validation.accuracy.unwrap_or(f64::NAN);
        println!(
            "  {:>6.1}% {:>8} {:>7} {:>7} {:>7.2}% {:>9.4} {:>7} {:>7.2}%  {}",
            f64::from(step.removed_bp) / 100.0,
            step.kept
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("+"),
            step.parameters,
            format!("{}/{val_rows}", runs::rows_right(val, &start.validation)),
            val * 100.0,
            step.validation.loss,
            format!("{}/{test_rows}", point.test_right),
            point.test_right as f64 * 100.0 / test_rows as f64,
            if step.met { "yes" } else { "no" }
        );
    }
}

/// What a `shrink` run chose, in one line: the level, the share of hidden neurons removed and
/// the test accuracy against the unpruned network's. Returns the share removed and the test
/// accuracy points lost.
fn print_choice(
    name: &str,
    run: &runs::ShrinkRun,
    unpruned_right: usize,
    test_rows: usize,
) -> (f64, f64) {
    let Some(i) = run.chosen else {
        println!("{name}: no step met the criterion; the unpruned network is kept");
        return (0.0, 0.0);
    };
    let point = &run.curve[i];
    let kept: usize = point.step.kept.iter().sum();
    let removed = (HIDDEN - kept) as f64 / HIDDEN as f64 * 100.0;
    let lost = (unpruned_right as f64 - point.test_right as f64) * 100.0 / test_rows as f64;
    println!(
        "{name}: chose the {:.1}% level, {} of {HIDDEN} hidden neurons removed ({removed:.2}%), \
         test {}/{test_rows} against {unpruned_right}/{test_rows} unpruned ({:+.2} points)",
        f64::from(point.step.removed_bp) / 100.0,
        HIDDEN - kept,
        point.test_right,
        -lost
    );
    (removed, lost)
}

/// Compacts `masked` and compares the compacted network's outputs with the masked one's on
/// the test digits through `infer` (one row at a time) and `infer_batch` (all rows at once),
/// and prints the count of values whose bits differ. Returns the compacted network.
///
/// # Errors
///
/// Returns a message if compaction or compilation fails.
fn check_compaction(
    what: &str,
    masked: (&cavitation_nn::Sequential, &cavitation_core::ParamStore),
    test: &cavitation_nn::Dataset,
) -> Result<(cavitation_nn::Sequential, cavitation_core::ParamStore), String> {
    let (small, small_params) =
        cavitation_nn::shrink::compact(masked.0, masked.1).map_err(|e| e.to_string())?;
    let compile = |net, params| {
        cavitation_nn::InferenceModel::compile(net, params).map_err(|e| e.to_string())
    };
    let (mut a, mut b) = (
        compile(masked.0, masked.1)?,
        compile(&small, &small_params)?,
    );
    let one = runs::bit_diff(
        &runs::infer_rows(&mut b, test),
        &runs::infer_rows(&mut a, test),
    );
    let all = runs::bit_diff(
        &runs::infer_all(&mut b, test),
        &runs::infer_all(&mut a, test),
    );
    let values = test.len() * 10;
    println!(
        "compaction, {what}: infer {} of {values} values differ in their bits (largest \
         difference {:e}); infer_batch {} of {values} (largest {:e})",
        one.0, one.1, all.0, all.1
    );
    Ok((small, small_params))
}

/// Runs the `shrink` subcommand: trains, shrinks twice, checks compaction and times the
/// full and the compacted network.
///
/// # Errors
///
/// Returns a message if a step fails.
fn shrink() -> Result<(), String> {
    use cavitation_nn::shrink::{Ranking, ShrinkConfig};
    let machine = machine::detect();
    println!(
        "shrink: {} ({} logical cores), kernel {}, {} threads, {}",
        machine.cpu_model,
        machine.logical_cores,
        level(),
        threads(),
        machine::rustc_version()
    );
    println!("load average at the start {}", machine::load_average());
    println!(
        "network 64 → 256 → 128 → 10 (ReLU), trained as the digits known result on the first \
         {} training digits; validation: the last {}",
        runs::SHRINK_FIT_ROWS,
        1437 - runs::SHRINK_FIT_ROWS
    );
    let start = runs::shrink_start()?;
    let test_rows = start.test.len();
    let unpruned = cavitation_nn::accuracy(&start.net, &start.params, &start.test, 128);
    let unpruned_right = runs::rows_right(unpruned, &start.test);
    println!(
        "unpruned test accuracy: {unpruned_right}/{test_rows} ({:.2}%)",
        unpruned * 100.0
    );

    let first_name = "run 1 (Ranking::OutNorm, dead_first off, every level)";
    let first = runs::shrink_run(
        &start,
        ShrinkConfig {
            ranking: Ranking::OutNorm,
            dead_first: false,
            patience: None,
            ..ShrinkConfig::default()
        },
    )?;
    print_curve(first_name, &first, &start);
    let second_name = "run 2 (ShrinkConfig::default())";
    let second = runs::shrink_run(&start, ShrinkConfig::default())?;
    print_curve(second_name, &second, &start);
    let (removed1, lost1) = print_choice("run 1", &first, unpruned_right, test_rows);
    let (removed2, lost2) = print_choice("run 2", &second, unpruned_right, test_rows);
    let verdict = |ok: bool| if ok { "met" } else { "not met" };
    println!(
        "run 1: at least 75% removed and at most 1 point lost: {}",
        verdict(removed1 >= 75.0 && lost1 <= 1.0)
    );
    println!(
        "run 2: at least 70% removed and at most 1 point lost: {}",
        verdict(removed2 >= 70.0 && lost2 <= 1.0)
    );

    if let Some(point) = first.curve.iter().find(|p| p.step.removed_bp == 8000) {
        let (masks, store) = point
            .step
            .checkpoint
            .as_ref()
            .expect("checkpoints are kept");
        let mut at = start.net.clone();
        for (linear, mask) in masks {
            at.set_neuron_mask(*linear, mask.clone());
        }
        let kept = point
            .step
            .kept
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>();
        check_compaction(
            &format!("run 1's 80% step ({})", kept.join("+")),
            (&at, store),
            &start.test,
        )?;
    }
    let (small, small_params) =
        check_compaction("run 1's choice", (&first.net, &first.params), &start.test)?;

    let compile = |net, params| {
        cavitation_nn::InferenceModel::compile(net, params).map_err(|e| e.to_string())
    };
    let mut models = vec![
        compile(&start.net, &start.params)?,
        compile(&first.net, &first.params)?,
        compile(&small, &small_params)?,
    ];
    let names = ["full", "full, run 1's masks on", "run 1's choice compacted"];
    let sizes = [
        runs::elements(&start.net, &start.params),
        runs::elements(&first.net, &first.params),
        runs::elements(&small, &small_params),
    ];
    let input = &start.test.x.data()[..start.test.features()];
    println!(
        "latency: one test digit through InferenceModel::infer, {} warm-up calls each, {} \
         interleaved rounds of {} calls each",
        runs::SHRINK_WARMUP,
        runs::SHRINK_ROUNDS,
        runs::SHRINK_CALLS
    );
    let times = runs::time_infer(&mut models, input);
    println!(
        "  {:<26} {:>10} {:>11} {:>19} {:>9}",
        "network", "parameters", "median us", "p10-p90 us", "speed-up"
    );
    let full = runs::median(&times.per_call[0]);
    for ((name, size), t) in names.iter().zip(sizes).zip(&times.per_call) {
        let m = runs::median(t);
        println!(
            "  {name:<26} {size:>10} {:>11.3} {:>9.3}-{:<9.3} {:>8.2}x",
            m * 1e6,
            runs::quantile(t, 0.1) * 1e6,
            runs::quantile(t, 0.9) * 1e6,
            full / m
        );
    }
    println!(
        "speed-up of run 1's choice compacted over the full network: {:.2}x",
        full / runs::median(&times.per_call[2])
    );
    println!("load average at the end {}", machine::load_average());
    Ok(())
}

/// Tests for the command-line side of the runner.
#[cfg(test)]
mod tests {
    use super::*;

    /// The child's result line reads back into the numbers it was printed from.
    #[test]
    fn small_child_line_round_trips() {
        let run = parse_small_line("wall 1.25e0 cpu 4.5e0 loss 3f800000").unwrap();
        assert_eq!((run.wall, run.cpu, run.loss_bits), (1.25, 4.5, 0x3f80_0000));
        assert!(parse_small_line("wall 1 cpu 2").is_none());
        assert!(parse_small_line("wall x cpu 2 loss 0").is_none());
    }

    /// The training specs of a command line, or its error message.
    fn specs(command: &str, args: &[&str]) -> Result<Vec<TrainSpec>, String> {
        let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
        let o = Options::parse(&args)?;
        Ok(plan(command, &o)?
            .into_iter()
            .filter_map(|run| match run {
                Run::Train(spec) => Some(spec),
                Run::Latency => None,
            })
            .collect())
    }

    /// `cifar` runs Adam on the CIFAR-10 network at batch 128 then 32 in seconds per epoch,
    /// or only the batch given, and in steps per second with `--steps`.
    #[test]
    fn cifar_plan_runs_both_batches_or_steps() {
        let both = specs("cifar", &[]).unwrap();
        let batches: Vec<usize> = both.iter().map(|s| s.batch).collect();
        assert_eq!(batches, [128, 32]);
        assert!(both.iter().all(|s| s.adam
            && s.data == "cifar"
            && s.widths == runs::CIFAR_NET
            && s.steps.is_none()
            && s.unit == Unit::EpochSeconds));

        let one = specs("cifar", &["--batch", "32", "--steps", "50"]).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!((one[0].batch, one[0].steps), (32, Some(50)));
        assert_eq!(one[0].unit, Unit::StepsPerSecond);
        assert_eq!(one[0].name, "cifar batch 32, 50 steps");
    }

    /// `--steps` belongs to `cifar`, `--adam` to `mnist`, `--batch` to both; `all` leaves
    /// `cifar` out.
    #[test]
    fn options_apply_only_where_they_mean_something() {
        assert!(
            specs("mnist", &["--steps", "10"])
                .unwrap_err()
                .contains("cifar only")
        );
        assert!(
            specs("cifar", &["--adam"])
                .unwrap_err()
                .contains("mnist only")
        );
        assert!(specs("digits", &["--batch", "32"]).is_err());
        assert!(
            specs("cifar", &["--steps", "0"])
                .unwrap_err()
                .contains("positive")
        );
        assert_eq!(specs("mnist", &["--batch", "32"]).unwrap()[0].batch, 32);
        assert!(specs("all", &[]).unwrap().iter().all(|s| s.data != "cifar"));
    }

    /// `--numpy-best` is off unless given, and parses on every compared command.
    #[test]
    fn numpy_best_option_parses() {
        let parse = |args: &[&str]| {
            let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
            Options::parse(&args).unwrap()
        };
        assert!(!parse(&[]).numpy_best);
        let o = parse(&["--numpy-best", "--python", "py"]);
        assert!(o.numpy_best && o.python == "py");
        for command in ["mnist", "cifar", "digits", "latency", "all"] {
            assert!(plan(command, &o).is_ok());
        }
    }
}
