//! A `Cuda` opened on one thread, shared by two others that use it at the same time, and
//! dropped on a fourth. Each test skips without CUDA, unless `CAVITATION_CUDA_REQUIRED=1`
//! makes that a failure.

use std::sync::{Arc, Barrier};
use std::thread;

use cavitation_cuda::{Cuda, Elementwise, ElementwiseOp};

/// Round trips each worker makes.
const ROUNDS: usize = 200;

/// Elements per buffer: several blocks of a launch.
const N: usize = 4096;

/// One worker's rounds: each allocates three buffers, writes two of them, runs `op` on the
/// card and reads the result back, which must be the CPU's bits. `seed` makes each worker's
/// values its own.
fn work(cuda: &Cuda, op: ElementwiseOp, seed: u32, start: &Barrier) {
    let kernels = Elementwise::new(cuda).expect("the kernels compile");
    start.wait();
    for round in 0..ROUNDS {
        let base = (seed * 1_000_003 + round as u32) as f32;
        let a: Vec<f32> = (0..N).map(|i| base + i as f32 * 0.25).collect();
        let b: Vec<f32> = (0..N).map(|i| 1.5 - i as f32 * 0.125).collect();
        let mut x = cuda.buffer(N * 4).unwrap();
        let mut y = cuda.buffer(N * 4).unwrap();
        let mut out = cuda.buffer(N * 4).unwrap();
        x.write(&a).unwrap();
        y.write(&b).unwrap();
        kernels.run(op, &x, Some(&y), &mut out, N, 0.0).unwrap();
        let mut back = vec![0.0f32; N];
        out.read(&mut back).unwrap();
        for i in 0..N {
            let want = match op {
                ElementwiseOp::Add => a[i] + b[i],
                _ => a[i] * b[i],
            };
            assert_eq!(
                back[i].to_bits(),
                want.to_bits(),
                "{op:?}, round {round}, element {i}"
            );
        }
    }
}

/// Opened on one thread, moved out of it, shared through an `Arc` by two threads that each
/// allocate, write, launch and read back at the same time with correct results, and
/// dropped on a fourth thread, which is the last owner.
#[test]
fn shared_between_threads() {
    let Some(cuda) = thread::spawn(Cuda::new).join().unwrap() else {
        println!("CUDA cannot be used here; skipped");
        return;
    };
    let cuda = Arc::new(cuda);
    let start = Arc::new(Barrier::new(2));
    let workers: Vec<_> = [(ElementwiseOp::Add, 1), (ElementwiseOp::Mul, 2)]
        .into_iter()
        .map(|(op, seed)| {
            let cuda = Arc::clone(&cuda);
            let start = Arc::clone(&start);
            thread::spawn(move || work(&cuda, op, seed, &start))
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    let cuda = Arc::try_unwrap(cuda).expect("the workers have given their handles back");
    thread::spawn(move || {
        cuda.synchronize().unwrap();
        drop(cuda);
    })
    .join()
    .unwrap();
}
