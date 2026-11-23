//! Times cuDNN's convolution passes on the `cifar-cnn` network's four convolutions at batch
//! 128 (3 × 3 kernels, padding 1, f32): the forward pass without and with the bias, the input
//! gradient and the weight gradient, each on its own, as the median of 51 calls after 20
//! warm-up calls, waiting for the card before the clock starts and before it stops. Prints
//! the algorithm each pass runs.
//!
//! Run with `cargo run --release -p cavitation-cuda --example conv_time`.

use std::time::Instant;

use cavitation_cuda::{Conv2d, ConvShape, Cuda, CudaError, DeviceBuffer};

/// Warm-up calls before timing.
const WARM: usize = 20;
/// Timed calls; the median is printed.
const RUNS: usize = 51;

/// The median time in milliseconds of `f`, waiting for the card around each call.
fn timed(cuda: &Cuda, mut f: impl FnMut() -> Result<(), CudaError>) -> f64 {
    for _ in 0..WARM {
        f().expect("run the pass");
    }
    let mut times: Vec<f64> = (0..RUNS)
        .map(|_| {
            cuda.synchronize().expect("synchronize");
            let start = Instant::now();
            f().expect("run the pass");
            cuda.synchronize().expect("synchronize");
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[RUNS / 2]
}

/// A buffer of `n` floats, all 0.01.
fn filled(cuda: &Cuda, n: usize) -> DeviceBuffer<'_> {
    let mut buffer = cuda.buffer(n * 4).expect("allocate");
    buffer.write(&vec![0.01f32; n]).expect("write");
    buffer
}

/// Opens CUDA, times each pass as the file's first lines describe and prints the medians;
/// when CUDA is not available, says so and points to `--example cuda_info` for the reason.
fn main() {
    let Some(cuda) = Cuda::new() else {
        println!("CUDA is not available: run `--example cuda_info` for the reason");
        return;
    };
    println!("{}", cuda.info());
    println!(
        "| layer | forward ms | forward with bias ms | input gradient ms | weight gradient ms | \
         algorithms |"
    );
    println!("|---|---|---|---|---|---|");
    let mut totals = [0.0f64; 4];
    for (cin, cout, size) in [(3, 32, 32), (32, 32, 32), (32, 64, 16), (64, 64, 16)] {
        let shape = ConvShape {
            padding: (1, 1),
            ..ConvShape::new(128, cin, cout, (size, size), (3, 3))
        };
        let conv = Conv2d::new(&cuda, shape).expect("set up the convolution");
        let (x, w, b, dy) = (
            filled(&cuda, shape.x_len()),
            filled(&cuda, shape.w_len()),
            filled(&cuda, cout as usize),
            filled(&cuda, shape.y_len()),
        );
        let (mut y, mut dx, mut dw) = (
            filled(&cuda, shape.y_len()),
            filled(&cuda, shape.x_len()),
            filled(&cuda, shape.w_len()),
        );
        let forward = timed(&cuda, || conv.forward(&x, &w, None, &mut y));
        let biased = timed(&cuda, || conv.forward(&x, &w, Some(&b), &mut y));
        let input = timed(&cuda, || conv.input_grad(&w, &dy, &mut dx));
        let weight = timed(&cuda, || conv.weight_grad(&x, &dy, &mut dw));
        for (total, t) in totals.iter_mut().zip([forward, biased, input, weight]) {
            *total += t;
        }
        println!(
            "| {cin} -> {cout} at {size} x {size} | {forward:.3} | {biased:.3} | {input:.3} | \
             {weight:.3} | {} |",
            conv.algorithms()
        );
    }
    println!(
        "| all four | {:.3} | {:.3} | {:.3} | {:.3} | |",
        totals[0], totals[1], totals[2], totals[3]
    );
}
