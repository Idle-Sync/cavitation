//! What the pooling tests use, ported from `cavitation-gpu`'s twin harness: seeded inputs for
//! the exact and the rounding twins, the two assertions they end in, and buffers moved to and
//! from the card.
//!
//! The exact twins compare bits, subnormals included: NVRTC keeps subnormals, so nothing is
//! flushed and no flushed reference is accepted. A NaN matches any NaN (CUDA's arithmetic
//! gives its own NaN, without the operand's payload). The rounding twins compare against an
//! `f64` reference within a stated bound per element, in units of [`U`] = 2⁻²⁴, plus
//! [`FLOOR`] = 2⁻¹²⁶.

#![allow(dead_code)]

use cavitation_cuda::{Cuda, DeviceBuffer};

/// f32's unit roundoff, 2⁻²⁴.
pub const U: f64 = 1.0 / 16_777_216.0;

/// The absolute floor every rounding bound carries, 2⁻¹²⁶, the smallest normal f32.
pub const FLOOR: f64 = 1.0 / (1u128 << 126) as f64;

/// A small seeded generator (xorshift64*), the one `cavitation-gpu`'s tests use, so the same
/// seed gives the same inputs there and here.
#[derive(Clone, Debug)]
pub struct Seeded(u64);

impl Seeded {
    /// A generator from `seed`; every seed, 0 included, gives a usable sequence.
    pub fn new(seed: u64) -> Seeded {
        Seeded(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A uniform value in `[lo, hi)`, rounded to f32.
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * unit
    }
}

/// `n` values for an exact twin: each one of -2, -1, 1 and 2, drawn from `seed`.
pub fn exact_values(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = Seeded::new(seed);
    (0..n)
        .map(|_| [-2.0, -1.0, 1.0, 2.0][(rng.next_u64() >> 62) as usize])
        .collect()
}

/// `n` values for a rounding twin: uniform in `[lo, hi)`, drawn from `seed`.
pub fn seeded_values(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut rng = Seeded::new(seed);
    (0..n).map(|_| rng.uniform(lo, hi)).collect()
}

/// Asserts that `got` has `want`'s bits, subnormals included, a NaN matching any NaN; on
/// failure, names `what`, the number of mismatches and the first few.
pub fn assert_exact(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    let mismatches: Vec<usize> = (0..got.len())
        .filter(|&i| {
            let (g, w) = (got[i], want[i]);
            g.to_bits() != w.to_bits() && !(g.is_nan() && w.is_nan())
        })
        .collect();
    if !mismatches.is_empty() {
        let shown: Vec<String> = mismatches
            .iter()
            .take(5)
            .map(|&i| {
                format!(
                    "[{i}] got {:e} (0x{:08x}), want {:e} (0x{:08x})",
                    got[i],
                    got[i].to_bits(),
                    want[i],
                    want[i].to_bits()
                )
            })
            .collect();
        panic!(
            "{what}: {} of {} elements differ; first: {}",
            mismatches.len(),
            got.len(),
            shown.join("; ")
        );
    }
}

/// The worst element of a rounding twin, as [`assert_within`] reports it.
#[derive(Clone, Copy, Debug)]
pub struct Worst {
    /// Its index.
    pub index: usize,
    /// The card's value.
    pub got: f32,
    /// The f64 reference.
    pub reference: f64,
    /// `|got - reference|`.
    pub error: f64,
    /// Its bound, the floor included.
    pub bound: f64,
}

impl Worst {
    /// The error as a fraction of the bound: at most 1 when the twin passes.
    pub fn ratio(&self) -> f64 {
        self.error / self.bound
    }
}

/// Asserts that every `got[i]` is within `bound(i)` + [`FLOOR`] of `reference[i]`, and that
/// it is NaN exactly where the reference is; returns the worst element. On failure, names
/// `what` and the worst element.
pub fn assert_within(
    got: &[f32],
    reference: &[f64],
    bound: impl Fn(usize) -> f64,
    what: &str,
) -> Worst {
    assert_eq!(got.len(), reference.len(), "{what}: lengths differ");
    let mut worst = Worst {
        index: 0,
        got: 0.0,
        reference: 0.0,
        error: 0.0,
        bound: FLOOR,
    };
    for (i, (&g, &r)) in got.iter().zip(reference).enumerate() {
        assert_eq!(
            g.is_nan(),
            r.is_nan(),
            "{what}: element {i} is {g:e} where the reference is {r:e}"
        );
        if r.is_nan() {
            continue;
        }
        let error = if f64::from(g) == r {
            0.0
        } else {
            (f64::from(g) - r).abs()
        };
        let candidate = Worst {
            index: i,
            got: g,
            reference: r,
            error,
            bound: bound(i) + FLOOR,
        };
        if candidate.ratio() > worst.ratio() {
            worst = candidate;
        }
    }
    assert!(
        worst.error <= worst.bound,
        "{what}: element {} is {:e}, the reference {:e}: error {:e} above its bound {:e} ({:.2}x)",
        worst.index,
        worst.got,
        worst.reference,
        worst.error,
        worst.bound,
        worst.ratio()
    );
    worst
}

/// A buffer holding `data` (`f32`, `u32` or `i32`).
pub fn upload<'c, T: cavitation_cuda::Scalar>(cuda: &'c Cuda, data: &[T]) -> DeviceBuffer<'c> {
    let mut buffer = cuda.buffer(size_of_val(data)).expect("allocate");
    buffer.write(data).expect("write");
    buffer
}

/// A buffer of `n` NaNs, so a test sees every value a kernel should write written.
pub fn nans(cuda: &Cuda, n: usize) -> DeviceBuffer<'_> {
    upload(cuda, &vec![f32::NAN; n])
}

/// The first `n` floats of `buffer`.
pub fn read(buffer: &DeviceBuffer<'_>, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    buffer.read(&mut out).expect("read");
    out
}

/// The first `n` `u32`s of `buffer`.
pub fn read_u32(buffer: &DeviceBuffer<'_>, n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n];
    buffer.read(&mut out).expect("read");
    out
}

/// The panic message of `body`, which must panic.
pub fn panic_message(body: impl FnOnce()) -> String {
    let payload =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).expect_err("the call panics");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_default()
}
