//! The f32 activation kernels: `tanh` and the logistic sigmoid over a slice, in place, on
//! the SIMD kernel of the current level.
//!
//! Both are built from one exponential, `e^y` for `y <= 0`: `y` is split as `n · ln 2 + r`
//! with `|r| <= ln 2 / 2`, `e^r` comes from a degree-7 polynomial (the coefficients of the
//! Cephes library's `expf`), and the result is scaled by `2^n` in two steps, so results
//! below the smallest normal f32 round once into the subnormal range instead of flushing to
//! zero.
//!
//! - `tanh(x)` is `a + a·s·Q(s)` with `a = |x|`, `s = a²` and `Q` a degree-4 polynomial
//!   for `a < 0.55`, and `(1 - e) / (1 + e)` with `e = e^(-2a)` above; the sign of `x` is
//!   copied onto the result. The polynomial avoids the cancellation in `1 - e` near zero.
//! - `sigmoid(x)` is `1 / (1 + e)` for `x >= 0` and `e / (1 + e)` below, with
//!   `e = e^(-|x|)`, so nothing overflows.
//!
//! Every lane computes both branches and keeps one, so the time does not depend on the
//! values. NaN stays NaN (the input's NaN is returned), `tanh(±∞) = ±1`, `tanh(-0) = -0`,
//! `sigmoid(+∞) = 1` and `sigmoid(-∞) = 0`.
//!
//! **Accuracy.** Against the correctly rounded result (from f64), checked for every finite
//! f32, the reference loop and the SSE4.2 kernel are within 2 ulp for both functions. The
//! bound stated and tested at every level is 3 ulp, since the AVX2 and NEON kernels fuse
//! multiply-adds and round differently from the reference (they are tested on a sweep of
//! one f32 in 1,009 and the edge values). The SSE4.2 kernel performs the reference's
//! operations in the same order and matches it bit for bit.
//!
//! # Examples
//!
//! ```
//! use cavitation_core::kernels::activation::{sigmoid_f32, tanh_f32};
//! let mut t = [0.0f32, 0.5, -3.0, f32::INFINITY];
//! tanh_f32(&mut t);
//! assert_eq!(t[0], 0.0);
//! assert!((t[1] - 0.5f64.tanh() as f32).abs() <= 2.0 * f32::EPSILON);
//! assert!((t[2] - (-3.0f64).tanh() as f32).abs() <= 2.0 * f32::EPSILON);
//! assert_eq!(t[3], 1.0);
//! let mut s = [0.0f32, -200.0, 200.0];
//! sigmoid_f32(&mut s);
//! assert_eq!(s, [0.5, 0.0, 1.0]);
//! ```

use super::{Level, supported};

/// The input below which the exponential's argument is clamped: `e^-104` is below half the
/// smallest subnormal f32, so it rounds to zero, and `2^n` for the `n` it gives stays
/// within two normal scale factors.
pub(super) const EXP_MIN: f32 = -104.0;
/// `log2(e)`, to find `n = floor(y · log2(e) + 1/2)`.
pub(super) const LOG2E: f32 = std::f32::consts::LOG2_E;
/// The high part of `ln 2`, 0.693359375 exactly (the f32 this literal names), with trailing
/// zero bits, so `n · LN2_HI` is exact.
pub(super) const LN2_HI: f32 = 0.693_359_4;
/// The low part of `ln 2`: `LN2_HI + LN2_LO` is `ln 2` to about 1e-11.
pub(super) const LN2_LO: f32 = -2.121_944_4e-4;
/// The coefficients of `e^r ≈ 1 + r + r²·P(r)`, highest degree first (Cephes `expf`).
pub(super) const EXP_P: [f32; 6] = [
    1.987_569_1e-4,
    1.398_199_9e-3,
    8.333_452e-3,
    4.166_579_6e-2,
    1.666_666_5e-1,
    0.5,
];
/// Below this `|x|`, `tanh` uses its polynomial instead of the exponential.
pub(super) const TANH_SMALL: f32 = 0.55;
/// The coefficients of `Q` in `tanh(a) ≈ a + a·s·Q(s)`, `s = a²`, highest degree first:
/// a fit minimising the largest relative error on `a < 0.55`, which is about 1e-9 before
/// rounding to f32.
pub(super) const TANH_Q: [f32; 5] = [
    -6.274_173_3e-3,
    2.107_163_5e-2,
    -5.385_23e-2,
    1.333_258_6e-1,
    -3.333_332e-1,
];

/// The sign bit of an f32.
pub(super) const SIGN: u32 = 0x8000_0000;

/// `2^n` for `-126 <= n <= 127`, built from its exponent bits.
fn pow2(n: i32) -> f32 {
    f32::from_bits(((n + 127) as u32) << 23)
}

/// `e^y` for `y <= 0` (larger `y` gives wrong results; NaN gives a meaningless value), as
/// the kernels compute it, one rounding per operation and no fused multiply-add.
pub(super) fn exp_neg(y: f32) -> f32 {
    let y = if y < EXP_MIN { EXP_MIN } else { y };
    let n = (y * LOG2E + 0.5).floor();
    let r = y - n * LN2_HI;
    let r = r - n * LN2_LO;
    let mut p = EXP_P[0];
    for &c in &EXP_P[1..] {
        p = p * r + c;
    }
    let p = p * r * r + r + 1.0;
    // `n` is a whole number in [-150, 0]; each half is in [-75, 0], a normal power of two.
    let n = n as i32;
    let n1 = n >> 1;
    p * pow2(n1) * pow2(n - n1)
}

/// `tanh(x)` as the kernels compute it, one element at a time: the reference twin of the
/// SIMD kernels, and the [`Level::Reference`] kernel.
pub(super) fn tanh_one(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let a = x.abs();
    let s = a * a;
    let mut q = TANH_Q[0];
    for &c in &TANH_Q[1..] {
        q = q * s + c;
    }
    let small = a + a * (s * q);
    let e = exp_neg(-2.0 * a);
    let big = (1.0 - e) / (1.0 + e);
    let r = if a < TANH_SMALL { small } else { big };
    f32::from_bits(r.to_bits() | (x.to_bits() & SIGN))
}

/// The logistic sigmoid of `x` as the kernels compute it, one element at a time: the
/// reference twin of the SIMD kernels, and the [`Level::Reference`] kernel.
pub(super) fn sigmoid_one(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let e = exp_neg(-x.abs());
    let d = 1.0 + e;
    if x >= 0.0 { 1.0 / d } else { e / d }
}

/// Which function a kernel computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Func {
    /// The hyperbolic tangent.
    Tanh,
    /// The logistic sigmoid.
    Sigmoid,
}

/// Replaces every value with its hyperbolic tangent, with the kernel for the current
/// [`level`](super::level). Each result is within 3 ulp of the correctly rounded `tanh`
/// (see the [module docs](self)); a value's result depends only on the value and the level,
/// not on its place in the slice. It runs on the calling thread and allocates nothing.
///
/// # Panics
///
/// Panics on first use, as [`level`](super::level) does, if `CAVITATION_KERNEL` is invalid.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::tanh_f32;
/// let mut v = [-0.0f32, 20.0, f32::NAN];
/// tanh_f32(&mut v);
/// assert!(v[0] == 0.0 && v[0].is_sign_negative());
/// assert_eq!(v[1], 1.0);
/// assert!(v[2].is_nan());
/// ```
pub fn tanh_f32(values: &mut [f32]) {
    run(super::level(), Func::Tanh, values);
}

/// [`tanh_f32`] with the kernel level given explicitly, instead of the process-wide
/// setting. [`Level::Reference`] runs [`reference::tanh_f32`](super::reference::tanh_f32),
/// [`Level::Sse42`] gives the same bits as it, and [`Level::Avx2Fma`], [`Level::Neon`] and
/// [`Level::Accelerate`] (whose activations are NEON's) fuse multiply-adds.
///
/// # Panics
///
/// Panics if this CPU cannot run `level`, naming the level and the detected features.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::tanh_f32_with;
/// use cavitation_core::kernels::{Level, detected};
/// let (mut fast, mut slow) = ([0.25f32; 9], [0.25f32; 9]);
/// tanh_f32_with(detected(), &mut fast);
/// tanh_f32_with(Level::Reference, &mut slow);
/// for (f, s) in fast.iter().zip(&slow) {
///     assert!((f - s).abs() <= 4.0 * f32::EPSILON * s.abs());
/// }
/// ```
pub fn tanh_f32_with(level: Level, values: &mut [f32]) {
    run(level, Func::Tanh, values);
}

/// Replaces every value with its logistic sigmoid `1 / (1 + e^-x)`, with the kernel for
/// the current [`level`](super::level). Each result is within 3 ulp of the correctly
/// rounded sigmoid (see the [module docs](self)); a value's result depends only on the
/// value and the level. It runs on the calling thread and allocates nothing.
///
/// # Panics
///
/// Panics on first use, as [`level`](super::level) does, if `CAVITATION_KERNEL` is invalid.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::sigmoid_f32;
/// let mut v = [0.0f32, f32::NEG_INFINITY, f32::INFINITY];
/// sigmoid_f32(&mut v);
/// assert_eq!(v, [0.5, 0.0, 1.0]);
/// ```
pub fn sigmoid_f32(values: &mut [f32]) {
    run(super::level(), Func::Sigmoid, values);
}

/// [`sigmoid_f32`] with the kernel level given explicitly, instead of the process-wide
/// setting; the levels differ as for [`tanh_f32_with`].
///
/// # Panics
///
/// Panics if this CPU cannot run `level`, naming the level and the detected features.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::activation::sigmoid_f32_with;
/// use cavitation_core::kernels::Level;
/// let mut v = [2.0f32];
/// sigmoid_f32_with(Level::Reference, &mut v);
/// assert!((v[0] - 0.880_797_1).abs() <= 3.0 * f32::EPSILON);
/// ```
pub fn sigmoid_f32_with(level: Level, values: &mut [f32]) {
    run(level, Func::Sigmoid, values);
}

/// Runs `func` over `values` with the kernel for `level`.
///
/// # Panics
///
/// Panics if this CPU cannot run `level`.
fn run(level: Level, func: Func, values: &mut [f32]) {
    assert!(
        supported(level),
        "activation kernel: {}",
        super::cannot_run(level)
    );
    // Each arm runs a kernel only after checking that this CPU supports it.
    match level {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: this CPU supports AVX2 and FMA (the guard).
        Level::Avx2Fma if supported(Level::Avx2Fma) => unsafe {
            super::x86::activation_avx2(func, values)
        },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: this CPU supports SSE4.2 (the guard).
        Level::Sse42 if supported(Level::Sse42) => unsafe {
            super::x86::activation_sse(func, values)
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: this CPU supports NEON (the guard; Accelerate's support check includes
        // NEON).
        Level::Neon | Level::Accelerate if supported(level) => unsafe {
            super::aarch64::activation_neon(func, values)
        },
        _ => match func {
            Func::Tanh => super::reference::tanh_f32(values),
            Func::Sigmoid => super::reference::sigmoid_f32(values),
        },
    }
}

/// Splits `values` for a `W`-lane kernel: the whole vectors, and the last
/// `values.len() mod W` values copied into a zero-padded vector, with their count. The
/// kernel runs the padded vector like any other and copies the count back, so a value's
/// result does not depend on whether it falls in the tail.
pub(super) fn split<const W: usize>(values: &mut [f32]) -> (&mut [[f32; W]], &mut [f32], [f32; W]) {
    let (chunks, tail) = values.as_chunks_mut::<W>();
    let mut pad = [0.0f32; W];
    pad[..tail.len()].copy_from_slice(tail);
    (chunks, tail, pad)
}

/// Tests for the activation kernels: each level against the reference twin, and the twin
/// and each level against the correctly rounded result.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::Level;

    /// The levels this CPU can run.
    fn levels() -> Vec<Level> {
        Level::ALL.into_iter().filter(|&l| supported(l)).collect()
    }

    /// The distance between two f32 in units in the last place, counting across zero (so
    /// -0 and +0 are 0 apart). Both must be non-NaN.
    fn ulps(a: f32, b: f32) -> u64 {
        let key = |v: f32| {
            let bits = v.to_bits() as i64;
            if bits & SIGN as i64 != 0 {
                -(bits & 0x7fff_ffff)
            } else {
                bits
            }
        };
        (key(a) - key(b)).unsigned_abs()
    }

    /// `tanh` correctly rounded to f32 (f64 `tanh` has far more precision than needed).
    fn tanh_exact(x: f32) -> f32 {
        (x as f64).tanh() as f32
    }

    /// The sigmoid correctly rounded to f32, from f64.
    fn sigmoid_exact(x: f32) -> f32 {
        let x = x as f64;
        let s = if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            x.exp() / (1.0 + x.exp())
        };
        s as f32
    }

    /// One f32 in every `stride` bit patterns, both signs, finite only.
    fn sweep(stride: u32) -> Vec<f32> {
        (0..0x7f80_0000u32)
            .step_by(stride as usize)
            .flat_map(|b| [f32::from_bits(b), f32::from_bits(b | SIGN)])
            .collect()
    }

    /// Values where the formulas change or saturate, and the specials.
    fn edges() -> Vec<f32> {
        let mut v = vec![
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            1e-45,
            1e-30,
            1e-4,
            TANH_SMALL,
            TANH_SMALL.next_down(),
            TANH_SMALL.next_up(),
            1.0,
            8.0,
            9.01,
            10.0,
            51.0,
            52.0,
            87.0,
            88.8,
            103.0,
            104.0,
            150.0,
            f32::MAX,
            f32::INFINITY,
        ];
        let neg: Vec<f32> = v.iter().map(|x| -x).collect();
        v.extend(neg);
        v
    }

    /// Runs `func` on `xs` at `level`, returning the results.
    fn apply(level: Level, func: Func, xs: &[f32]) -> Vec<f32> {
        let mut v = xs.to_vec();
        run(level, func, &mut v);
        v
    }

    /// The largest ulp error of `func` at `level` against `exact` over `xs`, with the input
    /// that gives it.
    fn worst(level: Level, func: Func, xs: &[f32], exact: fn(f32) -> f32) -> (u64, f32) {
        let got = apply(level, func, xs);
        xs.iter()
            .zip(&got)
            .map(|(&x, &g)| (ulps(g, exact(x)), x))
            .max_by_key(|&(u, _)| u)
            .unwrap_or((0, 0.0))
    }

    /// `tanh` and `sigmoid` are within 3 ulp of the correctly rounded result,
    /// at every level, over a sweep of the finite f32 and the edge values.
    #[test]
    fn within_the_stated_ulp_bounds() {
        let mut xs = sweep(1009);
        xs.extend(edges());
        for level in levels() {
            let (t, tx) = worst(level, Func::Tanh, &xs, tanh_exact);
            assert!(t <= 3, "{level}: tanh is {t} ulp off at {tx:e}");
            let (s, sx) = worst(level, Func::Sigmoid, &xs, sigmoid_exact);
            assert!(s <= 3, "{level}: sigmoid is {s} ulp off at {sx:e}");
        }
    }

    /// Every finite f32, at every level, within the stated bounds. Slow (minutes), so run
    /// by hand: `cargo test --release -p cavitation-core --lib every_f32 -- --ignored`.
    #[test]
    #[ignore = "checks all 4 billion f32 values; minutes in release"]
    fn every_f32_within_the_stated_ulp_bounds() {
        const BLOCK: u32 = 1 << 20;
        for level in levels() {
            let (mut t_max, mut s_max) = ((0, 0.0f32), (0, 0.0f32));
            let mut start = 0u32;
            loop {
                let xs: Vec<f32> = (start..start.saturating_add(BLOCK).min(0x7f80_0000))
                    .flat_map(|b| [f32::from_bits(b), f32::from_bits(b | SIGN)])
                    .collect();
                if xs.is_empty() {
                    break;
                }
                let t = worst(level, Func::Tanh, &xs, tanh_exact);
                if t.0 > t_max.0 {
                    t_max = t;
                }
                let s = worst(level, Func::Sigmoid, &xs, sigmoid_exact);
                if s.0 > s_max.0 {
                    s_max = s;
                }
                start += BLOCK;
            }
            println!(
                "{level}: tanh {} ulp at {:e}, sigmoid {} ulp at {:e}",
                t_max.0, t_max.1, s_max.0, s_max.1
            );
            assert!(t_max.0 <= 3 && s_max.0 <= 3, "{level}");
        }
    }

    /// The reference twin itself is within 2 ulp for both, the bound measured for every
    /// finite f32 by `every_f32_within_the_stated_ulp_bounds`.
    #[test]
    fn the_reference_is_within_its_measured_bounds() {
        let mut xs = sweep(1009);
        xs.extend(edges());
        let (t, tx) = worst(Level::Reference, Func::Tanh, &xs, tanh_exact);
        assert!(t <= 2, "tanh is {t} ulp off at {tx:e}");
        let (s, sx) = worst(Level::Reference, Func::Sigmoid, &xs, sigmoid_exact);
        assert!(s <= 2, "sigmoid is {s} ulp off at {sx:e}");
    }

    /// Every level is within 2 ulp of the reference twin, and SSE4.2 matches it bit for
    /// bit.
    #[test]
    fn levels_agree_with_the_reference() {
        let mut xs = sweep(4099);
        xs.extend(edges());
        for func in [Func::Tanh, Func::Sigmoid] {
            let want = apply(Level::Reference, func, &xs);
            for level in levels() {
                let got = apply(level, func, &xs);
                for ((&x, &g), &w) in xs.iter().zip(&got).zip(&want) {
                    if level == Level::Sse42 {
                        assert_eq!(g.to_bits(), w.to_bits(), "{level} {func:?} at {x:e}");
                    } else {
                        assert!(ulps(g, w) <= 2, "{level} {func:?} at {x:e}: {g:e} vs {w:e}");
                    }
                }
            }
        }
    }

    /// NaN stays NaN with its payload, the infinities and signed zeros give the documented
    /// values, at every level.
    #[test]
    fn specials() {
        let nan = f32::from_bits(0x7fc0_1234);
        let xs = [nan, -nan, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];
        for level in levels() {
            let t = apply(level, Func::Tanh, &xs);
            assert_eq!(t[0].to_bits(), nan.to_bits(), "{level}");
            assert_eq!(t[1].to_bits(), (-nan).to_bits(), "{level}");
            assert_eq!(&t[2..4], &[1.0, -1.0], "{level}");
            assert_eq!((t[4].to_bits(), t[5].to_bits()), (0, SIGN), "{level}");
            let s = apply(level, Func::Sigmoid, &xs);
            assert_eq!(s[0].to_bits(), nan.to_bits(), "{level}");
            assert_eq!(&s[2..], &[1.0, 0.0, 0.5, 0.5], "{level}");
        }
    }

    /// `tanh` is odd, bit for bit, and results stay in [-1, 1] and [0, 1].
    #[test]
    fn symmetry_and_range() {
        let xs = sweep(65_537);
        for level in levels() {
            let t = apply(level, Func::Tanh, &xs);
            let s = apply(level, Func::Sigmoid, &xs);
            for (i, &x) in xs.iter().enumerate() {
                assert!(
                    (-1.0..=1.0).contains(&t[i]),
                    "{level} tanh({x:e}) = {:e}",
                    t[i]
                );
                assert!(
                    (0.0..=1.0).contains(&s[i]),
                    "{level} sigmoid({x:e}) = {:e}",
                    s[i]
                );
                let neg = apply(level, Func::Tanh, &[-x])[0];
                assert_eq!(
                    neg.to_bits(),
                    (-t[i]).to_bits(),
                    "{level} tanh is odd at {x:e}"
                );
            }
        }
    }

    /// A value gives the same bits wherever it sits in the slice, tail included.
    #[test]
    fn results_do_not_depend_on_position() {
        let xs: Vec<f32> = (0..37).map(|i| (i as f32 - 18.0) * 0.37).collect();
        for level in levels() {
            for func in [Func::Tanh, Func::Sigmoid] {
                let whole = apply(level, func, &xs);
                for (i, &x) in xs.iter().enumerate() {
                    let one = apply(level, func, &[x])[0];
                    assert_eq!(one.to_bits(), whole[i].to_bits(), "{level} {func:?} at {i}");
                }
            }
        }
    }

    /// An empty slice is left alone.
    #[test]
    fn empty() {
        for level in levels() {
            assert!(apply(level, Func::Tanh, &[]).is_empty());
        }
    }
}
