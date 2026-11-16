//! `cavitation-bench gemm-gpu`: the tiled GPU GEMM's speed on square products, timed on the
//! GPU with timestamps, after a warm-up by time, with the GPU clock read from `nvidia-smi`
//! before and after when it is there.

use std::process::Command;
use std::time::{Duration, Instant};

use cavitation_gpu::kernels::{Gemm, GemmDims};
use cavitation_gpu::{Backend, BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};

/// The timed steps per shape; the median is reported.
const TIMED: usize = 20;

/// The options of `gemm-gpu`.
#[derive(Debug, PartialEq)]
struct Options {
    /// The backend to run on, or the platform's default.
    backend: Option<Backend>,
    /// The square sizes to time.
    sizes: Vec<u32>,
    /// The warm-up before each size's timed steps.
    warm_up: Duration,
}

/// Reads `[--backend vulkan|dx12|metal] [--sizes 1024,4096] [--warm-up SECONDS]`.
fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        backend: None,
        sizes: vec![1024, 2048, 4096],
        warm_up: Duration::from_secs(1),
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--backend" => o.backend = Some(value()?.parse().map_err(|e| format!("{e}"))?),
            "--sizes" => {
                o.sizes = value()?
                    .split(',')
                    .map(|s| s.trim().parse::<u32>().ok().filter(|&n| n > 0))
                    .collect::<Option<Vec<u32>>>()
                    .ok_or("--sizes takes positive whole numbers separated by commas")?;
            }
            "--warm-up" => {
                let seconds: f64 = value()?
                    .parse()
                    .map_err(|_| "--warm-up takes a number of seconds".to_string())?;
                o.warm_up = Duration::from_secs_f64(seconds.max(0.0));
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(o)
}

/// The SM clock `nvidia-smi` reports, or `None` without it.
fn sm_clock() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=clocks.sm", "--format=csv,noheader"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// `n` seeded values in [-1, 1), as little-endian bytes.
fn seeded_bytes(seed: u64, n: u64) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .flat_map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let unit = (s.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32;
            (2.0 * unit - 1.0).to_le_bytes()
        })
        .collect()
}

/// Times the GEMM on each size in `o`, plain and with `B` transposed (as a convolution's
/// forward product is), and prints the median GPU time and GFLOPS of each.
fn time(gpu: &Gpu, o: &Options) -> Result<(), String> {
    let largest = u64::from(*o.sizes.iter().max().expect("at least one size"));
    let mut cache = PipelineCache::new(gpu);
    let mut pool = BufferPool::new(gpu);
    let mut steps = StepEncoder::new(
        gpu,
        StepSizes {
            upload_bytes: 4 * largest * largest,
            ..StepSizes::default()
        },
    );
    steps.set_profiling(true);
    let gemm = Gemm::new(gpu, &mut cache).map_err(|e| e.to_string())?;
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    for &n in &o.sizes {
        let len = 4 * u64::from(n) * u64::from(n);
        let (a, b, c) = (
            pool.take(len, usage),
            pool.take(len, usage),
            pool.take(len, usage),
        );
        for (buffer, seed) in [(&a, 1), (&b, 2)] {
            let mut step = steps.begin();
            step.upload(
                buffer.buffer(),
                0,
                &seeded_bytes(seed, u64::from(n) * u64::from(n)),
            );
            let _ = step.finish();
        }
        for trans_b in [false, true] {
            let dims = GemmDims {
                trans_b,
                ..GemmDims::new(n, n, n)
            };
            let call = gemm.call(&mut steps, &mut pool, dims, &a, &b, &c);
            let run = |steps: &mut StepEncoder| {
                let mut step = steps.begin();
                step.span("gemm");
                gemm.record(&mut step, &call);
                step.finish().spans()[0].1
            };
            let start = Instant::now();
            while start.elapsed() < o.warm_up {
                run(&mut steps);
            }
            let mut times: Vec<f64> = (0..TIMED)
                .map(|_| {
                    run(&mut steps).ok_or(
                        "the GPU gave no timestamps, or a pass's end read before its beginning",
                    )
                })
                .collect::<Result<_, _>>()?;
            times.sort_by(f64::total_cmp);
            let median = (times[TIMED / 2 - 1] + times[TIMED / 2]) / 2.0;
            println!(
                "gemm {n} x {n} x {n} {}: median of {TIMED} {:.3} ms (min {:.3}, max {:.3}) = {:.0} GFLOPS",
                if trans_b {
                    "B transposed"
                } else {
                    "plain      "
                },
                median * 1e-6,
                times[0] * 1e-6,
                times[TIMED - 1] * 1e-6,
                dims.flops() / (median * 1e-9) / 1e9
            );
            call.release(&mut pool);
        }
        for buffer in [a, b, c] {
            pool.give(buffer);
        }
    }
    Ok(())
}

/// Runs `gemm-gpu` with `args`: opens the GPU, prints it and the clock, times each size.
pub fn run(args: &[String]) -> Result<(), String> {
    let o = parse(args)?;
    crate::running();
    let gpu = match o.backend {
        Some(backend) => Gpu::with_backend(backend),
        None => Gpu::new(),
    }
    .ok_or("no GPU found; gemm-gpu needs one")?;
    println!("{}", gpu.info());
    println!(
        "warm-up {:.1} s per shape; GPU SM clock before: {}",
        o.warm_up.as_secs_f64(),
        sm_clock().unwrap_or_else(|| "not available".to_string())
    );
    time(&gpu, &o)?;
    println!(
        "GPU SM clock after: {}",
        sm_clock().unwrap_or_else(|| "not available".to_string())
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    //! The option parsing; the timing itself needs a GPU and is run by hand.

    use super::*;

    /// Turns string slices into owned arguments.
    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    /// The defaults, every option, and the errors.
    #[test]
    fn options() {
        let d = parse(&[]).unwrap();
        assert_eq!(
            (d.backend, d.sizes.clone(), d.warm_up),
            (None, vec![1024, 2048, 4096], Duration::from_secs(1))
        );
        let o = parse(&args(&[
            "--backend",
            "dx12",
            "--sizes",
            "512, 1024",
            "--warm-up",
            "0.5",
        ]))
        .unwrap();
        assert_eq!(o.backend, Some(Backend::Dx12));
        assert_eq!(o.sizes, vec![512, 1024]);
        assert_eq!(o.warm_up, Duration::from_millis(500));
        for bad in [
            &["--sizes", "0"][..],
            &["--sizes", "x"],
            &["--backend", "gl"],
            &["--warm-up"],
            &["--what"],
        ] {
            assert!(parse(&args(bad)).is_err(), "{bad:?}");
        }
    }
}
