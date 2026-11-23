//! Times cuBLAS's fully connected passes on the `cifar-cnn` network's two layers at batch
//! 128 (4096 → 256 and 256 → 10, f32): the forward pass with the bias, the input gradient,
//! the weight gradient and the bias gradient, each on its own, as the median of 51 calls
//! after 20 warm-up calls, waiting for the card before the clock starts and before it stops.
//!
//! Run with `cargo run --release -p cavitation-cuda --example linear_time`.

use std::time::Instant;

use cavitation_cuda::{Cuda, CudaError, DeviceBuffer, Linear, LinearShape};

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
        "| layer | forward with bias ms | input gradient ms | weight gradient ms | bias \
         gradient ms |"
    );
    println!("|---|---|---|---|---|");
    for (inputs, outputs) in [(4096, 256), (256, 10)] {
        let shape = LinearShape {
            batch: 128,
            inputs,
            outputs,
        };
        let layer = Linear::new(&cuda, shape).expect("set up the layer");
        let (x, w, b, dy) = (
            filled(&cuda, shape.x_len()),
            filled(&cuda, shape.w_len()),
            filled(&cuda, outputs as usize),
            filled(&cuda, shape.y_len()),
        );
        let (mut y, mut dx, mut dw, mut db) = (
            filled(&cuda, shape.y_len()),
            filled(&cuda, shape.x_len()),
            filled(&cuda, shape.w_len()),
            filled(&cuda, outputs as usize),
        );
        let forward = timed(&cuda, || layer.forward(&x, &w, Some(&b), &mut y));
        let input = timed(&cuda, || layer.input_grad(&w, &dy, &mut dx));
        let weight = timed(&cuda, || layer.weight_grad(&x, &dy, &mut dw));
        let bias = timed(&cuda, || layer.bias_grad(&dy, &mut db));
        println!("| {inputs} -> {outputs} | {forward:.3} | {input:.3} | {weight:.3} | {bias:.3} |");
    }
}
