//! Times the max-pooling kernels on the `cifar-cnn` network's two pools at batch 128 (2 × 2
//! windows on 32 channels of 32 × 32 and 64 channels of 16 × 16): the forward pass with its
//! winners and the backward pass, each as the median of 51 calls after 20 warm-up calls,
//! waiting for the card before the clock starts and before it stops.
//!
//! Run with `cargo run --release -p cavitation-cuda --example pooling_time`.

use std::time::Instant;

use cavitation_cuda::{Cuda, CudaError, DeviceBuffer, MaxPool2d, PoolShape};

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

/// A buffer of `n` floats counting up from 0 in steps of 0.01.
fn filled(cuda: &Cuda, n: usize) -> DeviceBuffer<'_> {
    let mut buffer = cuda.buffer(n * 4).expect("allocate");
    let data: Vec<f32> = (0..n).map(|i| (i % 1000) as f32 * 0.01).collect();
    buffer.write(&data).expect("write");
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
    println!("| pool | forward ms | backward ms |");
    println!("|---|---|---|");
    for (channels, size) in [(32, 32), (64, 16)] {
        let shape = PoolShape::new(128, channels, (size, size), (2, 2));
        let pool = MaxPool2d::new(&cuda, shape).expect("compile the pool");
        let x = filled(&cuda, shape.x_len());
        let dy = filled(&cuda, shape.y_len());
        let (mut y, mut winners) = (filled(&cuda, shape.y_len()), filled(&cuda, shape.y_len()));
        let mut dx = filled(&cuda, shape.x_len());
        let forward = timed(&cuda, || pool.forward(&x, &mut y, &mut winners));
        let backward = timed(&cuda, || pool.backward(&dy, &winners, &mut dx));
        println!("| {channels} x {size} x {size} | {forward:.3} | {backward:.3} |");
    }
}
