//! Benchmark runner for the cavitation neural network library.
//!
//! `cavitation-bench <all|mnist|digits|latency|gemm|info>`:
//!
//! - `info` prints the CPU, the kernel levels, the thread count and the compiler.
//! - `mnist [--batch 32|128] [--adam]` times epochs of the 1M network on MNIST;
//!   `digits` times epochs of the 50k network on the digits; `latency` times one input
//!   through the 50k network. Each also runs `scripts/bench/numpy_mlp.py` with the same
//!   network, batch, data and step counts, and prints cavitation against numpy with the
//!   ratio. The results are also written to `target/bench/results.json`.
//! - `all` runs `info`, `mnist` at batch 128 and 32, `digits` and `latency`.
//! - `gemm --shapes mnist128|digits128` times the eight matrix products of one batch-128
//!   training step of the named network at the current kernel level and thread count
//!   (`CAVITATION_KERNEL`, `CAVITATION_THREADS`), and prints each product's median time,
//!   GFLOPS (`2·m·n·k` / time) and threads used, then the rate over all eight.
//!
//! The baseline is run with `python3` from `PATH`; `--python <path>` or the
//! `CAVITATION_BENCH_PYTHON` variable names another interpreter (it needs numpy).

mod machine;
mod numpy;
mod runs;

use std::path::PathBuf;
use std::process::ExitCode;

use cavitation_core::kernels::{detected, level, threads};

use runs::{Comparison, TrainSpec, Unit};

/// How to call the runner, printed on a usage error.
const USAGE: &str = "usage: cavitation-bench <all|mnist|digits|latency|gemm|info> [options]\n\
    \x20 mnist   [--batch 32|128] [--adam] [--python PATH]\n\
    \x20 digits  [--python PATH]\n\
    \x20 latency [--python PATH]\n\
    \x20 all     [--python PATH]\n\
    \x20 gemm    [--shapes mnist128|digits128]\n\
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
        "mnist" | "digits" | "latency" | "all" => Options::parse(rest).and_then(|o| {
            let mut rows = Vec::new();
            if command == "all" {
                info();
            }
            for run in plan(command, &o)? {
                rows.push(execute(run, &o.python)?);
            }
            report(&rows)
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
    /// `--batch`: 32 or 128 (`mnist` only).
    batch: usize,
    /// `--adam`: time Adam instead of plain SGD (`mnist` only).
    adam: bool,
    /// The Python interpreter that runs the baseline.
    python: String,
}

impl Options {
    /// Parses the options after the command name.
    ///
    /// # Errors
    ///
    /// Returns a message for an unknown option, a missing value or a batch other than 32
    /// or 128.
    fn parse(args: &[String]) -> Result<Options, String> {
        let mut o = Options {
            batch: 128,
            adam: false,
            python: std::env::var("CAVITATION_BENCH_PYTHON").unwrap_or_else(|_| "python3".into()),
        };
        let mut rest = args.iter();
        while let Some(arg) = rest.next() {
            match arg.as_str() {
                "--batch" => {
                    let v = rest.next().ok_or("--batch needs a value")?;
                    o.batch = match v.as_str() {
                        "32" => 32,
                        "128" => 128,
                        _ => return Err(format!("--batch must be 32 or 128, got `{v}`")),
                    };
                }
                "--adam" => o.adam = true,
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
            unit: Unit::EpochSeconds,
        })
    };
    let digits = Run::Train(TrainSpec {
        name: "digits batch 128".to_string(),
        data: "digits",
        widths: runs::DIGITS_NET,
        batch: 128,
        adam: false,
        epochs: 100,
        unit: Unit::SamplesPerSecond,
    });
    if command != "mnist" && (o.adam || o.batch != 128) {
        return Err(format!(
            "--batch and --adam apply to mnist only, not {command}"
        ));
    }
    Ok(match command {
        "mnist" => vec![mnist(o.batch, o.adam)],
        "digits" => vec![digits],
        "latency" => vec![Run::Latency],
        _ => vec![mnist(128, false), mnist(32, false), digits, Run::Latency],
    })
}

/// Runs one comparison.
fn execute(run: Run, python: &str) -> Result<Comparison, String> {
    match run {
        Run::Train(spec) => {
            eprintln!("{}: cavitation, then numpy", spec.name);
            runs::compare_training(&spec, python)
        }
        Run::Latency => {
            eprintln!("latency: cavitation, then numpy");
            runs::compare_latency(python)
        }
    }
}

/// Prints the table of cavitation against numpy and writes `target/bench/results.json`.
///
/// # Errors
///
/// Returns a message if the results file cannot be written.
fn report(rows: &[Comparison]) -> Result<(), String> {
    let load = std::fs::read_to_string("/proc/loadavg")
        .map(|l| l.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|_| "unknown".to_string());
    println!(
        "kernel {}, {} threads, load average at the end {load}",
        level(),
        threads()
    );
    if let Some(first) = rows.first() {
        println!(
            "numpy {}, BLAS threads: {}",
            first.numpy_version, first.blas_threads
        );
    }
    println!(
        "{:<24} {:>12} {:>12} {:>10}  unit",
        "run", "cavitation", "numpy", "ratio"
    );
    for row in rows {
        println!(
            "{:<24} {:>12.2} {:>12.2} {:>9.2}x  {}",
            row.name,
            row.value(runs::median(&row.ours)),
            row.value(runs::median(&row.numpy)),
            row.ratio(),
            row.unit_label()
        );
    }
    println!("ratio = numpy time over cavitation time; medians of the timed epochs or rounds.");
    for row in rows {
        println!("{}: {}; {}", row.name, row.detail, row.timing_note);
    }
    write_results(rows, &load)
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
fn write_results(rows: &[Comparison], load: &str) -> Result<(), String> {
    let machine = machine::detect();
    let escape = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let runs: Vec<String> = rows
        .iter()
        .map(|r| {
            format!(
                "    {{\"name\": \"{}\", \"detail\": \"{}\", \"timing\": \"{}\", \"unit\": \"{}\", \
                 \"cavitation\": {:e}, \"numpy\": {:e}, \"ratio\": {:e},\n     \
                 \"cavitation_seconds\": {}, \"numpy_seconds\": {}}}",
                escape(&r.name),
                escape(&r.detail),
                escape(&r.timing_note),
                r.unit_label(),
                r.value(runs::median(&r.ours)),
                r.value(runs::median(&r.numpy)),
                r.ratio(),
                json_list(&r.ours),
                json_list(&r.numpy)
            )
        })
        .collect();
    let text = format!(
        "{{\n  \"cpu\": \"{}\",\n  \"logical_cores\": {},\n  \"kernel\": \"{}\",\n  \
         \"threads\": {},\n  \"rustc\": \"{}\",\n  \"numpy_version\": \"{}\",\n  \"blas_threads\": \"{}\",\n  \"load_average\": \"{}\",\n  \"runs\": [\n{}\n  ]\n}}\n",
        escape(&machine.cpu_model),
        machine.logical_cores,
        level(),
        threads(),
        escape(&machine::rustc_version()),
        escape(rows.first().map_or("", |r| &r.numpy_version)),
        escape(rows.first().map_or("", |r| &r.blas_threads)),
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
