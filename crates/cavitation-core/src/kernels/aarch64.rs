//! aarch64 microkernels: the NEON GEMM tile (8 by 12, fused multiply-add by vector lane).

use std::arch::aarch64::{
    float32x4_t, vaddq_f32, vdupq_n_f32, vfmaq_laneq_f32, vld1q_f32, vmulq_f32, vst1q_f32,
};

use super::MatRef;
use super::gemm::{Dest, Kernel, pack_a, pack_b};

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
