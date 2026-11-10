//! Timed runs: the matrix products of one training step, each timed on its own.

use std::hint::black_box;
use std::time::Instant;

use cavitation_core::kernels::gemm::{gemm_f32_with, threads_used};
use cavitation_core::kernels::{Level, MatMut, MatRef};

/// One matrix product of a training step, `C [m, n] = A [m, k] · B [k, n]`, with operands
/// stored the way the step stores them.
#[derive(Clone, Copy, Debug)]
pub struct Product {
    /// What the product is in the step.
    pub name: &'static str,
    /// Rows of `A` and `C`.
    pub m: usize,
    /// Columns of `B` and `C`.
    pub n: usize,
    /// Columns of `A`, rows of `B`: the length of each sum.
    pub k: usize,
    /// `A` is the transpose of a row-major `[k, m]` matrix (a weight gradient's `dYᵀ`).
    pub a_transposed: bool,
    /// `B` is the transpose of a row-major `[n, k]` matrix (a forward pass's `Wᵀ`).
    pub b_transposed: bool,
}

impl Product {
    /// The floating-point operations of one product: `2 · m · n · k`.
    pub fn flops(&self) -> f64 {
        2.0 * self.m as f64 * self.n as f64 * self.k as f64
    }
}

/// The eight products of one batch-128 training step of an MLP with layer widths `widths`
/// (input first): the forward products, the weight gradients, and the input gradients of
/// every layer but the first (the first layer's input gradient is never needed).
fn step_products(widths: [usize; 4], names: &'static [&'static str; 8]) -> [Product; 8] {
    let batch = 128;
    let [w0, w1, w2, w3] = widths;
    let p = |i: usize, m, n, k, a_transposed, b_transposed| Product {
        name: names[i],
        m,
        n,
        k,
        a_transposed,
        b_transposed,
    };
    [
        p(0, batch, w1, w0, false, true),
        p(1, batch, w2, w1, false, true),
        p(2, batch, w3, w2, false, true),
        p(3, w3, w2, batch, true, false),
        p(4, w2, w1, batch, true, false),
        p(5, w1, w0, batch, true, false),
        p(6, batch, w2, w3, false, false),
        p(7, batch, w1, w2, false, false),
    ]
}

/// The step's product names, in the order [`step_products`] lists them.
const NAMES: [&str; 8] = [
    "forward 1   X·W1ᵀ",
    "forward 2   H1·W2ᵀ",
    "forward 3   H2·W3ᵀ",
    "weights 3   dY3ᵀ·H2",
    "weights 2   dY2ᵀ·H1",
    "weights 1   dY1ᵀ·X",
    "input 3     dY3·W3",
    "input 2     dY2·W2",
];

/// The products of one batch-128 step of a named network: `mnist128` (784 → 1024 → 256 →
/// 10, 1,068,810 parameters) or `digits128` (64 → 256 → 128 → 10, 50,826 parameters).
///
/// # Errors
///
/// Returns a message naming `name` and the accepted names if it is neither.
pub fn shapes(name: &str) -> Result<[Product; 8], String> {
    match name {
        "mnist128" => Ok(step_products([784, 1024, 256, 10], &NAMES)),
        "digits128" => Ok(step_products([64, 256, 128, 10], &NAMES)),
        _ => Err(format!(
            "unknown shape set `{name}`; expected mnist128 or digits128"
        )),
    }
}

/// The timing of one product.
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    /// The median time of one call, in seconds.
    pub seconds: f64,
    /// The threads the call used.
    pub threads: usize,
}

/// Times `product` at `level` on `threads` threads: warms up for about 50 ms, then takes 31
/// samples, each the mean of enough back-to-back calls to last about 2 ms, and returns the
/// median sample. Inputs are fixed pseudo-random values; `beta` is 0, as for the first
/// contribution to a gradient.
pub fn time_product(level: Level, threads: usize, product: Product) -> Timing {
    let Product { m, n, k, .. } = product;
    let fill = |len: usize, seed: usize| -> Vec<f32> {
        (0..len)
            .map(|i| ((i * 7919 + seed) % 2001) as f32 / 1000.0 - 1.0)
            .collect()
    };
    let a = fill(m * k, 1);
    let b = fill(k * n, 2);
    let mut c = vec![0.0f32; m * n];
    let a_view = if product.a_transposed {
        MatRef::new(&a, k, m, m, 1).unwrap().transpose()
    } else {
        MatRef::new(&a, m, k, k, 1).unwrap()
    };
    let b_view = if product.b_transposed {
        MatRef::new(&b, n, k, k, 1).unwrap().transpose()
    } else {
        MatRef::new(&b, k, n, n, 1).unwrap()
    };
    let mut call = || {
        gemm_f32_with(
            level,
            threads,
            black_box(1.0),
            black_box(a_view),
            black_box(b_view),
            black_box(0.0),
            MatMut::new(&mut c, m, n, n, 1).unwrap(),
        );
        black_box(&mut c);
    };
    let start = Instant::now();
    let mut calls = 0usize;
    while start.elapsed().as_secs_f64() < 0.05 || calls < 5 {
        call();
        calls += 1;
    }
    let per_call = start.elapsed().as_secs_f64() / calls as f64;
    let reps = ((2e-3 / per_call).ceil() as usize).max(1);
    let mut samples: Vec<f64> = (0..31)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..reps {
                call();
            }
            t.elapsed().as_secs_f64() / reps as f64
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    Timing {
        seconds: samples[samples.len() / 2],
        threads: threads_used(level, threads, m, n, k),
    }
}

/// Tests for the timed runs.
#[cfg(test)]
mod tests {
    use super::*;

    /// The 1M network's eight products do 4,799,488 FLOPs per sample (6 × weights minus the
    /// first layer's input gradient), and the digits network's 269,824.
    #[test]
    fn step_flops_per_sample() {
        let per_sample = |name| {
            shapes(name)
                .unwrap()
                .iter()
                .map(Product::flops)
                .sum::<f64>()
                / 128.0
        };
        assert_eq!(per_sample("mnist128"), 4_799_488.0);
        assert_eq!(per_sample("digits128"), 269_824.0);
        assert!(shapes("cifar").is_err());
    }
}
