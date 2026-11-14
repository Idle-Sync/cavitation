//! aarch64 microkernels: the NEON GEMM tile (8 by 12, fused multiply-add by vector lane) and
//! the NEON inference matvec kernel, and the NEON `tanh`, sigmoid, softplus and ELU kernels.

use std::arch::aarch64::{
    float32x4_t, vabsq_f32, vaddq_f32, vaddq_s32, vandq_u32, vbslq_f32, vceqq_f32, vcgeq_f32,
    vcgtq_f32, vcleq_f32, vcltq_f32, vcvtq_s32_f32, vdivq_f32, vdupq_n_f32, vdupq_n_s32,
    vdupq_n_u32, vfmaq_f32, vfmaq_laneq_f32, vfmsq_f32, vld1q_f32, vmaxq_f32, vminq_f32, vmulq_f32,
    vnegq_f32, vorrq_u32, vreinterpretq_f32_s32, vreinterpretq_f32_u32, vreinterpretq_u32_f32,
    vrndmq_f32, vshlq_n_s32, vshrq_n_s32, vst1q_f32, vsubq_f32, vsubq_s32,
};

use super::MatRef;
use super::activation::{
    EXP_MIN, EXP_P, EXPM1_MIN, Func, LN2_HI, LN2_LO, LOG_P, LOG2E, SIGN, TANH_Q, TANH_SMALL, split,
};
use super::gemm::{Dest, Kernel, pack_a, pack_b};
use super::matvec::Job;

/// The NEON GEMM kernel: 8 by 12 tiles held in 24 four-wide accumulators, updated with
/// fused multiply-add (with the three `B` vectors and two vectors of `A`, 29 of the 32 NEON
/// registers in use, nothing spilled).
///
/// Packed `A` holds each value once (`A_REP = 1`): NEON's multiply-add by lane
/// (`fmla v.4s, v.4s, v.s[lane]`) takes its scalar from any lane of a vector register, so
/// the kernel loads 8 values of `A` as two vectors and uses each lane in turn, with no
/// broadcast at all. Both panels use the general packing routines.
///
/// Block sizes `kc = 256`, `mc = 128`, `nc = 2048` are estimates, not tuned: with `kc = 256`
/// one `A` panel (8 KB) and one `B` panel (12 KB) fit in a 64 KB L1 data cache, and one
/// task's packed rows of `A` (128 KB) in L2.
pub(super) struct Neon;

// SAFETY: `tile` reads `8·kc` values from `a` and `12·kc` from `b`, with `vld1q_f32`, which
// needs only the 4-byte alignment of f32, and reads and writes C only at the `dest.rows` by
// `dest.cols` elements its contract names.
unsafe impl Kernel for Neon {
    const MR: usize = 8;
    const NR: usize = 12;
    const KC: usize = 256;
    const MC: usize = 128;
    const NC: usize = 2048;
    const A_REP: usize = 1;

    fn pack_a(a: &MatRef<f32>, rows: (usize, usize), ks: (usize, usize), dst: &mut [f32]) {
        pack_a::<1>(a, rows, ks, 8, dst, |_, _, _, _| false);
    }

    fn pack_b(b: &MatRef<f32>, cols: (usize, usize), ks: (usize, usize), dst: &mut [f32]) {
        pack_b(b, cols, ks, 12, dst);
    }

    unsafe fn tile(kc: usize, a: *const f32, b: *const f32, alpha: f32, beta: f32, dest: &Dest) {
        // SAFETY: the caller upholds this function's contract, which is `tile_8x12`'s; NEON
        // is part of the aarch64 baseline, so every aarch64 CPU runs it.
        unsafe { tile_8x12(kc, a, b, alpha, beta, dest) }
    }
}

/// Adds `a[LANE] · b` to each of the three accumulators of one row, one fused multiply-add
/// each. Safe to call from code compiled with NEON, as all aarch64 code is.
#[inline]
#[target_feature(enable = "neon")]
fn fma_row<const LANE: i32>(row: &mut [float32x4_t; 3], b: &[float32x4_t; 3], a: float32x4_t) {
    for (acc, &bv) in row.iter_mut().zip(b) {
        *acc = vfmaq_laneq_f32::<LANE>(*acc, bv, a);
    }
}

/// The 8 by 12 NEON tile; see [`Kernel::tile`] for the contract. Each step of the sum is one
/// fused multiply-add (one rounding instead of two).
///
/// # Safety
///
/// As [`Kernel::tile`] with `MR = 8`, `NR = 12` and `A_REP = 1`; `a` and `b` need only the
/// alignment of f32.
#[target_feature(enable = "neon")]
unsafe fn tile_8x12(kc: usize, a: *const f32, b: *const f32, alpha: f32, beta: f32, dest: &Dest) {
    let mut acc: [[float32x4_t; 3]; 8] = [[vdupq_n_f32(0.0); 3]; 8];
    for p in 0..kc {
        // SAFETY: `p < kc`, so floats `12p..12p + 12` of the B panel and `8p..8p + 8` of the
        // A panel are inside them; `vld1q_f32` needs no more than f32 alignment.
        let (bv, a0, a1) = unsafe {
            let bp = b.add(12 * p);
            let ap = a.add(8 * p);
            (
                [vld1q_f32(bp), vld1q_f32(bp.add(4)), vld1q_f32(bp.add(8))],
                vld1q_f32(ap),
                vld1q_f32(ap.add(4)),
            )
        };
        fma_row::<0>(&mut acc[0], &bv, a0);
        fma_row::<1>(&mut acc[1], &bv, a0);
        fma_row::<2>(&mut acc[2], &bv, a0);
        fma_row::<3>(&mut acc[3], &bv, a0);
        fma_row::<0>(&mut acc[4], &bv, a1);
        fma_row::<1>(&mut acc[5], &bv, a1);
        fma_row::<2>(&mut acc[6], &bv, a1);
        fma_row::<3>(&mut acc[7], &bv, a1);
    }
    let va = vdupq_n_f32(alpha);
    for row in &mut acc {
        for v in row.iter_mut() {
            *v = vmulq_f32(va, *v);
        }
    }
    if dest.rows == 8 && dest.cols == 12 && dest.cs == 1 {
        let vb = vdupq_n_f32(beta);
        for (r, row) in acc.iter().enumerate() {
            for (q, &v) in row.iter().enumerate() {
                // SAFETY: row `r < 8 = dest.rows`, columns `4q..4q + 4 <= 12 = dest.cols`
                // are elements of C, contiguous since `cs = 1`; the caller gives this call
                // sole access to them.
                unsafe {
                    let p = dest.ptr.add(r * dest.rs + 4 * q);
                    if beta == 0.0 {
                        vst1q_f32(p, v);
                    } else {
                        // A separate multiply and add, as the edge path below does, so a
                        // value does not depend on whether its tile lies on an edge.
                        vst1q_f32(p, vaddq_f32(v, vmulq_f32(vb, vld1q_f32(p))));
                    }
                }
            }
        }
    } else {
        // An edge tile, or C with non-unit column stride: the same arithmetic, one element
        // at a time, storing only the valid part.
        let mut t = [0.0f32; 96];
        for (r, row) in acc.iter().enumerate() {
            for (q, &v) in row.iter().enumerate() {
                // SAFETY: `t` has room for 4 floats at `12r + 4q` (r < 8, q < 3).
                unsafe { vst1q_f32(t.as_mut_ptr().add(12 * r + 4 * q), v) };
            }
        }
        for r in 0..dest.rows {
            for j in 0..dest.cols {
                // SAFETY: `r < dest.rows` and `j < dest.cols`, so this is an element of C
                // the caller gives this call sole access to.
                unsafe {
                    let p = dest.ptr.add(r * dest.rs + j * dest.cs);
                    *p = if beta == 0.0 {
                        t[12 * r + j]
                    } else {
                        t[12 * r + j] + beta * *p
                    };
                }
            }
        }
    }
}

/// The NEON inference matvec, streaming the weights in storage order as the SSE4.2 one does
/// (8 rows at a time into every four-wide vector of outputs, partial sums kept in `out`),
/// with one fused multiply-add per step. The last `n mod 4` outputs are computed one at a
/// time, and the bias and ReLU are applied in a final pass.
///
/// # Safety
///
/// `job` must satisfy [`Job`]'s invariant for `out.len()` outputs. NEON is part of the
/// aarch64 baseline, so every aarch64 CPU runs it.
#[target_feature(enable = "neon")]
pub(super) unsafe fn matvec_neon(job: &Job, out: &mut [f32]) {
    let n = out.len();
    let nv = n / 4 * 4;
    let (k, vecs) = (job.x.len(), &mut out[..nv]);
    let mut i = 0;
    while i < k {
        let rows = if i + 8 <= k { 8 } else { 1 };
        // SAFETY: rows `i..i + rows` are below `k` and the columns `0..nv` are outputs, so
        // the weights read are inside `job.w` by the `Job` invariant this function's caller
        // upholds; `vecs` holds `nv` values, a multiple of the vector width.
        unsafe {
            match (rows, i == 0) {
                (8, true) => neon_rows::<8, true>(job, i, vecs),
                (8, false) => neon_rows::<8, false>(job, i, vecs),
                (_, true) => neon_rows::<1, true>(job, i, vecs),
                _ => neon_rows::<1, false>(job, i, vecs),
            }
        }
        i += rows;
    }
    if k == 0 {
        vecs.fill(0.0);
    }
    let zero = vdupq_n_f32(0.0);
    for (j, o) in vecs.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        // SAFETY: `o` holds 4 floats; `vld1q_f32` needs no more than f32 alignment.
        let mut v = unsafe { vld1q_f32(o.as_ptr()) };
        if let Some(b) = job.bias {
            let b = &b[4 * j..4 * j + 4];
            // SAFETY: `b` holds 4 floats; `vld1q_f32` needs no more than f32 alignment.
            v = vaddq_f32(unsafe { vld1q_f32(b.as_ptr()) }, v);
        }
        if job.relu {
            // 0 where v <= 0 (so -0 becomes +0), v elsewhere; NaN fails the comparison and
            // stays.
            v = vbslq_f32(vcleq_f32(v, zero), zero, v);
        }
        // SAFETY: `o` holds 4 floats; `vst1q_f32` needs no more than f32 alignment.
        unsafe { vst1q_f32(o.as_mut_ptr(), v) };
    }
    job.scalar_tail(nv, &mut out[nv..]);
}

/// Adds rows `i0..i0 + R` of the NEON matvec into every four-wide vector of `out`, in row
/// order, one fused multiply-add per row. With `FIRST` the sums start from zero instead of
/// `out`.
///
/// # Safety
///
/// Weights `(i, j)` must lie inside `job.w` for `i0 <= i < i0 + R` and `j < out.len()` (a
/// multiple of 4), and `i0 + R <= job.x.len()`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn neon_rows<const R: usize, const FIRST: bool>(job: &Job, i0: usize, out: &mut [f32]) {
    let xs: [float32x4_t; R] = std::array::from_fn(|r| vdupq_n_f32(job.x[i0 + r]));
    let w = job.w.as_ptr();
    let ld = job.ld;
    for (j, o) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let mut a = if FIRST {
            vdupq_n_f32(0.0)
        } else {
            // SAFETY: `o` holds 4 floats; `vld1q_f32` needs no more than f32 alignment.
            unsafe { vld1q_f32(o.as_ptr()) }
        };
        for (r, &xv) in xs.iter().enumerate() {
            // SAFETY: row `i0 + r` and columns `4j..4j + 4` are inside `job.w` (this
            // function's contract), contiguous since a row's columns are adjacent;
            // `vld1q_f32` needs no more than f32 alignment.
            let wv = unsafe { vld1q_f32(w.add((i0 + r) * ld + 4 * j)) };
            a = vfmaq_f32(a, xv, wv);
        }
        // SAFETY: `o` holds 4 floats; `vst1q_f32` needs no more than f32 alignment.
        unsafe { vst1q_f32(o.as_mut_ptr(), a) };
    }
}

/// The NEON `tanh` or sigmoid kernel over `values`, in place, four lanes at a time (the
/// tail through a padded vector), the reference's algorithm with fused multiply-adds.
///
/// # Safety
///
/// The CPU must support NEON (every aarch64 CPU does).
#[target_feature(enable = "neon")]
pub(super) unsafe fn activation_neon(func: Func, values: &mut [f32]) {
    let lanes = |v: float32x4_t| match func {
        Func::Tanh => tanh_neon(v),
        Func::Sigmoid => sigmoid_neon(v),
    };
    let (chunks, tail, mut pad) = split::<4>(values);
    for c in chunks {
        // SAFETY: `c` holds 4 floats; `vld1q_f32` and `vst1q_f32` need only f32 alignment.
        unsafe { vst1q_f32(c.as_mut_ptr(), lanes(vld1q_f32(c.as_ptr()))) };
    }
    if !tail.is_empty() {
        // SAFETY: `pad` holds 4 floats, as above.
        unsafe { vst1q_f32(pad.as_mut_ptr(), lanes(vld1q_f32(pad.as_ptr()))) };
        let n = tail.len();
        tail.copy_from_slice(&pad[..n]);
    }
}

/// `e^y` in each lane for `y <= 0`, the reference's `exp_neg` with fused multiply-adds
/// (NaN lanes give a meaningless value, which the callers replace).
#[inline]
#[target_feature(enable = "neon")]
fn exp_neg_neon(y: float32x4_t) -> float32x4_t {
    let y = vmaxq_f32(y, vdupq_n_f32(EXP_MIN));
    let n = vrndmq_f32(vfmaq_f32(vdupq_n_f32(0.5), y, vdupq_n_f32(LOG2E)));
    let r = vfmsq_f32(y, n, vdupq_n_f32(LN2_HI));
    let r = vfmsq_f32(r, n, vdupq_n_f32(LN2_LO));
    let mut p = vdupq_n_f32(EXP_P[0]);
    for &c in &EXP_P[1..] {
        p = vfmaq_f32(vdupq_n_f32(c), p, r);
    }
    let p = vaddq_f32(vfmaq_f32(r, vmulq_f32(p, r), r), vdupq_n_f32(1.0));
    // `n` is a whole number in [-150, 0] (or NaN, which converts to 0), so the conversion
    // is exact; each half is in [-75, 0], a normal power of two.
    let n = vcvtq_s32_f32(n);
    let n1 = vshrq_n_s32::<1>(n);
    let n2 = vsubq_s32(n, n1);
    let bias = vdupq_n_s32(127);
    let s1 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n1, bias)));
    let s2 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n2, bias)));
    vmulq_f32(vmulq_f32(p, s1), s2)
}

/// `e^y − 1` in each lane for `y <= 0`, the reference's `expm1_neg` with fused
/// multiply-adds (NaN lanes give a meaningless value, which the callers replace).
#[inline]
#[target_feature(enable = "neon")]
fn expm1_neg_neon(y: float32x4_t) -> float32x4_t {
    let low = vcltq_f32(y, vdupq_n_f32(EXPM1_MIN));
    let y = vmaxq_f32(y, vdupq_n_f32(EXPM1_MIN));
    let n = vrndmq_f32(vfmaq_f32(vdupq_n_f32(0.5), y, vdupq_n_f32(LOG2E)));
    let r = vfmsq_f32(y, n, vdupq_n_f32(LN2_HI));
    let r = vfmsq_f32(r, n, vdupq_n_f32(LN2_LO));
    let mut p = vdupq_n_f32(EXP_P[0]);
    for &c in &EXP_P[1..] {
        p = vfmaq_f32(vdupq_n_f32(c), p, r);
    }
    let q = vfmaq_f32(r, vmulq_f32(p, r), r);
    // `n` is a whole number in [-25, 0] (or NaN, which converts to 0), so the conversion is
    // exact and `2^n` a normal power of two.
    let n = vcvtq_s32_f32(n);
    let s = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n, vdupq_n_s32(127))));
    let v = vfmaq_f32(vsubq_f32(s, vdupq_n_f32(1.0)), q, s);
    vbslq_f32(low, vdupq_n_f32(-1.0), v)
}

/// `ln(1 + t)` in each lane for `t` in `[0, 1]`, the reference's `log1p_unit` with fused
/// multiply-adds in the polynomial; `1 + t` and its rounding error stay separate
/// operations, as the correction needs.
#[inline]
#[target_feature(enable = "neon")]
fn log1p_unit_neon(t: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let m = vaddq_f32(one, t);
    let c = vsubq_f32(t, vsubq_f32(m, one));
    let big = vcgtq_f32(m, vdupq_n_f32(std::f32::consts::SQRT_2));
    let f = vbslq_f32(big, vmulq_f32(m, vdupq_n_f32(0.5)), m);
    let k = vreinterpretq_f32_u32(vandq_u32(big, vreinterpretq_u32_f32(one)));
    let x = vsubq_f32(f, one);
    let z = vmulq_f32(x, x);
    let mut p = vdupq_n_f32(LOG_P[0]);
    for &cc in &LOG_P[1..] {
        p = vfmaq_f32(vdupq_n_f32(cc), p, x);
    }
    let y = vmulq_f32(vmulq_f32(x, z), p);
    let y = vfmaq_f32(y, k, vdupq_n_f32(LN2_LO));
    let y = vfmsq_f32(y, vdupq_n_f32(0.5), z);
    let r = vaddq_f32(x, y);
    let r = vfmaq_f32(r, k, vdupq_n_f32(LN2_HI));
    vaddq_f32(r, vdivq_f32(c, m))
}

/// Softplus in each lane, the reference's `softplus_one` with fused multiply-adds.
#[inline]
#[target_feature(enable = "neon")]
fn softplus_neon(x: float32x4_t) -> float32x4_t {
    let zero = vdupq_n_f32(0.0);
    let l = log1p_unit_neon(exp_neg_neon(vnegq_f32(vabsq_f32(x))));
    let m = vbslq_f32(vcgtq_f32(x, zero), x, zero);
    let r = vaddq_f32(m, l);
    // `x == x` is false only in NaN lanes, which keep `x`.
    vbslq_f32(vceqq_f32(x, x), r, x)
}

/// ELU with `alpha` in each lane, the reference's `elu_one` with fused multiply-adds.
#[inline]
#[target_feature(enable = "neon")]
fn elu_neon(alpha: float32x4_t, x: float32x4_t) -> float32x4_t {
    let zero = vdupq_n_f32(0.0);
    let neg = vmulq_f32(alpha, expm1_neg_neon(vminq_f32(x, zero)));
    let r = vbslq_f32(vcgtq_f32(x, zero), x, neg);
    vbslq_f32(vceqq_f32(x, x), r, x)
}

/// Runs `lanes` over `values` four at a time, the tail through a zero-padded vector.
#[inline]
#[target_feature(enable = "neon")]
fn neon_lanes(values: &mut [f32], lanes: impl Fn(float32x4_t) -> float32x4_t) {
    let (chunks, tail, mut pad) = split::<4>(values);
    for c in chunks {
        // SAFETY: `c` holds 4 floats; `vld1q_f32` and `vst1q_f32` need only f32 alignment.
        unsafe { vst1q_f32(c.as_mut_ptr(), lanes(vld1q_f32(c.as_ptr()))) };
    }
    if !tail.is_empty() {
        // SAFETY: `pad` holds 4 floats, as above.
        unsafe { vst1q_f32(pad.as_mut_ptr(), lanes(vld1q_f32(pad.as_ptr()))) };
        let n = tail.len();
        tail.copy_from_slice(&pad[..n]);
    }
}

/// The NEON softplus over `values`, in place.
///
/// # Safety
///
/// The CPU must support NEON.
#[target_feature(enable = "neon")]
pub(super) unsafe fn softplus_neon_slice(values: &mut [f32]) {
    neon_lanes(values, |v| softplus_neon(v));
}

/// The NEON ELU with `alpha` over `values`, in place.
///
/// # Safety
///
/// The CPU must support NEON.
#[target_feature(enable = "neon")]
pub(super) unsafe fn elu_neon_slice(alpha: f32, values: &mut [f32]) {
    let a = vdupq_n_f32(alpha);
    neon_lanes(values, |v| elu_neon(a, v));
}

/// `tanh` in each lane, the reference's `tanh_one` with fused multiply-adds.
#[inline]
#[target_feature(enable = "neon")]
fn tanh_neon(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let a = vabsq_f32(x);
    let s = vmulq_f32(a, a);
    let mut q = vdupq_n_f32(TANH_Q[0]);
    for &c in &TANH_Q[1..] {
        q = vfmaq_f32(vdupq_n_f32(c), q, s);
    }
    let small = vfmaq_f32(a, a, vmulq_f32(s, q));
    let e = exp_neg_neon(vmulq_f32(vdupq_n_f32(-2.0), a));
    let big = vdivq_f32(vsubq_f32(one, e), vaddq_f32(one, e));
    let r = vbslq_f32(vcltq_f32(a, vdupq_n_f32(TANH_SMALL)), small, big);
    let sign = vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(SIGN));
    let r = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(r), sign));
    // `x == x` is false only in NaN lanes, which keep `x`.
    vbslq_f32(vceqq_f32(x, x), r, x)
}

/// The logistic sigmoid in each lane, the reference's `sigmoid_one` with fused
/// multiply-adds.
#[inline]
#[target_feature(enable = "neon")]
fn sigmoid_neon(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let e = exp_neg_neon(vnegq_f32(vabsq_f32(x)));
    let d = vaddq_f32(one, e);
    let r = vbslq_f32(
        vcgeq_f32(x, vdupq_n_f32(0.0)),
        vdivq_f32(one, d),
        vdivq_f32(e, d),
    );
    vbslq_f32(vceqq_f32(x, x), r, x)
}
