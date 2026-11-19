//! Prints whether CUDA can be used on this machine: the card, the versions of cuDNN, cuBLAS
//! and NVRTC and the files they were loaded from, or what is missing and what to install.
//!
//! Run with `cargo run --release -p cavitation-cuda --example cuda_info`.

use cavitation_cuda::Cuda;

fn main() {
    match Cuda::open() {
        Ok(cuda) => {
            println!("CUDA is available.");
            println!("{}", cuda.info());
            match cuda.free_memory() {
                Ok(free) => println!("free memory: {:.1} GiB", free as f64 / (1u64 << 30) as f64),
                Err(why) => println!("free memory: {why}"),
            }
        }
        Err(why) => println!("CUDA is not available: {why}"),
    }
}
