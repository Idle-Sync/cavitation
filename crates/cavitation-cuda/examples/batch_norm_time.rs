//! Times batch norm on the `cifar-cnn` network's two batch norms at batch 128 (32 channels of
//! 32 × 32 and 64 channels of 16 × 16), affine, through the batch's statistics: the forward
//! pass (the statistics and the normalisation) and the backward pass (`dγ`, `dβ` and the input
//! gradient), each as the median of 51 calls after 20 warm-up calls, waiting for the card
//! before the clock starts and before it stops.
//!
//! Run with `cargo run --release -p cavitation-cuda --example batch_norm_time`.

use std::time::Instant;

use cavitation_cuda::{BatchNorm2d, BatchStats, ChannelDims, Cuda, CudaError, DeviceBuffer};

/// Warm-up calls before timing.
const WARM: usize = 20;
/// Timed calls; the median is printed.
const RUNS: usize = 51;
/// PyTorch's default eps.
const EPS: f64 = 1e-5;

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

fn main() {
    let Some(cuda) = Cuda::new() else {
        println!("CUDA is not available: run `--example cuda_info` for the reason");
        return;
    };
    println!("{}", cuda.info());
    println!("| batch norm | forward ms | backward ms |");
    println!("|---|---|---|");
    for (channels, size) in [(32, 32), (64, 16)] {
        let dims = ChannelDims::new(128, channels, size * size);
        let mut bn = BatchNorm2d::new(&cuda, dims).expect("compile batch norm");
        let (n, c) = (dims.len() as usize, channels as usize);
        let (x, dy) = (filled(&cuda, n), filled(&cuda, n));
        let (mut y, mut dx, mut stats) = (filled(&cuda, n), filled(&cuda, n), filled(&cuda, 3 * c));
        let (w, b) = (filled(&cuda, c), filled(&cuda, c));
        let (mut dw, mut db) = (filled(&cuda, c), filled(&cuda, c));
        let forward = timed(&cuda, || {
            bn.forward(
                EPS,
                BatchStats::Batch,
                &x,
                Some((&w, &b)),
                &mut y,
                &mut stats,
            )
        });
        let backward = timed(&cuda, || {
            bn.backward(&x, &stats, &dy, &mut dx, Some((&w, &mut dw, &mut db)))
        });
        println!("| {channels} x {size} x {size} | {forward:.3} | {backward:.3} |");
    }
}
