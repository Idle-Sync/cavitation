//! The f32 inference matvec: one input vector times a weight matrix stored transposed, plus
//! an optional bias and an optional ReLU, on the SIMD kernel of the current level.
//!
//! Each output is a sum over the input in order from zero, one lane of a vector per output,
//! so the kernels never sum across lanes. The kernels read the weights in storage order, 8
//! rows at a time, adding each group of rows into every output vector and keeping the
//! partial sums in the output between groups. The SSE4.2 kernel multiplies and adds separately,
//! exactly as the reference loop does; the AVX2 and NEON kernels fuse each step into one
//! multiply-add (one rounding instead of two).

use super::{Level, MatRef, detected_features, reference, supported};

/// One input times a weight matrix stored transposed, with the kernel for the current
/// [`level`](super::level): `out[j] = bias[j] + Σᵢ x[i] · w_t[i][j]`, then
/// `out[j] = max(out[j], 0)` when `relu` is true (NaN stays NaN, as in the ReLU layer).
///
/// `w_t` is `x.len()` by `out.len()`. The sum runs over `i` in order from zero; the bias, if
/// present, is added to the finished sum. Results match [`reference::matvec`] (followed by
/// ReLU) within `2 · (k + 2) · 1.19e-7 · (Σᵢ |xᵢ| · |wᵢⱼ| + |biasⱼ|)` for an input of
/// width `k`, and the same call at the same level gives the same bits. It runs on the
/// calling thread and allocates nothing.
///
/// # Panics
///
/// Panics, naming the lengths, if `x.len() != w_t.rows()`, `out.len() != w_t.cols()`, or
/// the bias length differs from `out.len()`; and on first use, as
/// [`level`](super::level) does, if `CAVITATION_KERNEL` is invalid.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::MatRef;
/// use cavitation_core::kernels::matvec::matvec_f32;
/// let w_t = [1.0f32, 2.0, 3.0, 4.0];
/// let mut out = [0.0f32; 2];
/// let w = MatRef::new(&w_t, 2, 2, 2, 1).unwrap();
/// matvec_f32(w, &[1.0, 1.0], Some(&[0.5, -10.0]), true, &mut out);
/// assert_eq!(out, [4.5, 0.0]);
/// ```
pub fn matvec_f32(w_t: MatRef<f32>, x: &[f32], bias: Option<&[f32]>, relu: bool, out: &mut [f32]) {
    matvec_f32_with(super::level(), w_t, x, bias, relu, out);
}

/// [`matvec_f32`] with the kernel level given explicitly, instead of the process-wide
/// setting.
///
/// [`Level::Sse42`] runs 4-wide vectors with a separate multiply and add, [`Level::Avx2Fma`]
/// 8-wide vectors with fused multiply-add and [`Level::Neon`] 4-wide vectors with fused
/// multiply-add; [`Level::Reference`] runs [`reference::matvec`]. A weight matrix whose
/// columns are not contiguous (`col_stride` other than 1) runs the reference loop at every
/// level.
///
/// # Panics
///
/// Panics if this CPU cannot run `level` (the message names the level and the detected
/// features), or, naming the lengths, if `x.len() != w_t.rows()`, `out.len() != w_t.cols()`
/// or the bias length differs from `out.len()`. Nothing is computed before these checks.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::{detected, MatRef};
/// use cavitation_core::kernels::matvec::matvec_f32_with;
/// let w_t = [1.0f32, -2.0];
/// let mut out = [0.0f32; 2];
/// let w = MatRef::new(&w_t, 1, 2, 2, 1).unwrap();
/// matvec_f32_with(detected(), w, &[3.0], None, false, &mut out);
/// assert_eq!(out, [3.0, -6.0]);
/// ```
pub fn matvec_f32_with(
    level: Level,
    w_t: MatRef<f32>,
    x: &[f32],
    bias: Option<&[f32]>,
    relu: bool,
    out: &mut [f32],
) {
    assert!(
        supported(level),
        "matvec_f32_with: this CPU cannot run kernel level {level} (detected features: {})",
        detected_features()
    );
    reference::check_matvec_shapes(&w_t, x, bias, out);
    let contiguous = w_t.col_stride() == 1 || w_t.cols() <= 1;
    let job = Job {
        w: w_t.data,
        ld: w_t.row_stride(),
        x,
        bias,
        relu,
    };
    // Each arm runs a kernel only after checking that this CPU supports it.
    match level {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: this CPU supports AVX2 and FMA (the guard), and `job` satisfies
        // `Job`'s invariant: the shapes were checked above against a `MatRef`, whose
        // constructor checked the bounds, and its columns are contiguous (the guard).
        Level::Avx2Fma if contiguous && supported(Level::Avx2Fma) => unsafe {
            super::x86::matvec_avx2(&job, out)
        },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: this CPU supports SSE4.2 (the guard); `job` as for AVX2 above.
        Level::Sse42 if contiguous && supported(Level::Sse42) => unsafe {
            super::x86::matvec_sse(&job, out)
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: this CPU supports NEON (the guard); `job` as for AVX2 above.
        Level::Neon if contiguous && supported(Level::Neon) => unsafe {
            super::aarch64::matvec_neon(&job, out)
        },
        _ => {
            reference::matvec(w_t, x, bias, out);
            if relu {
                for v in out.iter_mut() {
                    *v = relu_value(*v);
                }
            }
        }
    }
}

/// ReLU as the ReLU layer computes it: 0 at or below 0 (so -0 becomes +0), `v` above 0, NaN
/// stays NaN.
pub(super) fn relu_value(v: f32) -> f32 {
    if v <= 0.0 { 0.0 } else { v }
}

/// One matvec, checked, for a SIMD kernel.
///
/// Invariant (established by [`matvec_f32_with`]): with `k = x.len()` and `n` the length of
/// the kernel's `out`, element `(i, j)` of the weight matrix is `w[i · ld + j]` and lies
/// inside `w` for every `i < k` and `j < n`; `bias`, if present, holds `n` values.
pub(super) struct Job<'a> {
    /// The weight matrix's storage, row-major with row stride `ld`.
    pub(super) w: &'a [f32],
    /// The distance in `w` between the starts of consecutive rows.
    pub(super) ld: usize,
    /// The input vector.
    pub(super) x: &'a [f32],
    /// The bias, one value per output.
    pub(super) bias: Option<&'a [f32]>,
    /// Whether to apply ReLU to each finished output.
    pub(super) relu: bool,
}

impl Job<'_> {
    /// Computes outputs `j0..j0 + out.len()` (at most [`MAX_TAIL`]) with scalar arithmetic,
    /// in the reference's order: one sum per output over the input in order, the outputs'
    /// sums advanced together row by row so their additions overlap. For the columns a
    /// kernel's vectors do not cover.
    ///
    /// # Panics
    ///
    /// Panics if `out` holds more than [`MAX_TAIL`] values, or if an element it reads is
    /// outside `w` (which the invariant rules out).
    pub(super) fn scalar_tail(&self, j0: usize, out: &mut [f32]) {
        let cols = out.len();
        assert!(cols <= MAX_TAIL, "matvec: a tail of {cols} columns");
        if cols == 0 {
            return;
        }
        let mut acc = [0.0f32; MAX_TAIL];
        for (i, &xi) in self.x.iter().enumerate() {
            let row = &self.w[i * self.ld + j0..i * self.ld + j0 + cols];
            for (a, &w) in acc.iter_mut().zip(row) {
                *a += xi * w;
            }
        }
        for (q, (o, &a)) in out.iter_mut().zip(&acc).enumerate() {
            let v = match self.bias {
                Some(b) => b[j0 + q] + a,
                None => a,
            };
            *o = if self.relu { relu_value(v) } else { v };
        }
    }
}

/// The most columns [`Job::scalar_tail`] computes: one less than the widest vector.
pub(super) const MAX_TAIL: usize = 7;
