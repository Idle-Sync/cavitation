//! Benchmark runner for the cavitation neural network library.
//!
//! `cavitation-bench gemm --shapes mnist128|digits128` times the eight matrix products of
//! one batch-128 training step of the named network at the current kernel level and thread
//! count (`CAVITATION_KERNEL`, `CAVITATION_THREADS`), and prints each product's median time,
//! GFLOPS (`2·m·n·k` / time) and threads used, then the rate over all eight.

mod runs;

use std::process::ExitCode;

use cavitation_core::kernels::{level, threads};

/// How to call the runner, printed on a usage error.
const USAGE: &str = "usage: cavitation-bench gemm [--shapes mnist128|digits128]";

/// Entry point of the benchmark runner.
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("gemm") => match gemm(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("cavitation-bench gemm: {e}\n{USAGE}");
                ExitCode::from(2)
            }
        },
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
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
