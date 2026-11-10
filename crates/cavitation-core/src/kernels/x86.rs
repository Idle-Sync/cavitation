//! x86_64 microkernels: the SSE4.2 GEMM tile (6 by 8, multiply then add; SSE has no fused
//! multiply-add) and its packing routines.

use std::arch::x86_64::{
    __m128, _mm_add_ps, _mm_load_ps, _mm_loadu_ps, _mm_movehl_ps, _mm_movelh_ps, _mm_mul_ps,
    _mm_set1_ps, _mm_setzero_ps, _mm_shuffle_ps, _mm_storeu_ps, _mm_unpackhi_ps, _mm_unpacklo_ps,
};

use super::MatRef;
use super::gemm::{Dest, Kernel, pack_a, pack_b};

/// The SSE4.2 GEMM kernel: 6 by 8 tiles held in 12 four-wide accumulators. Packed `A` holds
/// each value four times over, so the kernel loads it already broadcast instead of
/// shuffling: on cores with one vector multiply per cycle the shuffles (and register copies)
/// compete with the multiplies. With this layout the tile runs at 22.2 GFLOPS on one core of
/// an Intel Celeron N5105 with its data in L1, against 22.4 for back-to-back multiplies and
/// adds alone (measured, `kc` 64 to 256).
///
/// Block sizes `kc = 256`, `mc = 96`, `nc = 2048`, with one compute task per thread (set in
/// the driver), chosen by measurement on the eight products of a batch-128 training step of
/// a 784-1024-256-10 and a 64-256-128-10 network, 4 threads, Intel Celeron N5105, best of 3
/// runs (GFLOPS over all eight): `kc` 256 gave 55.6 and 41.9, 192 gave 57.3 and 37.3, 384
/// gave 51.2 and 42.1; `mc` 72 gave 56.7 and 41.5, 144 gave 49.4 and 28.6; 1 task per thread
/// gave 56.5 and 49.5, against 55.6 and 41.9 for 2 and 53.6 and 35.0 for 3. Other work ran
/// during the sweep (load average 0.9 to 2.7), so differences of a few percent are noise.
pub(super) struct Sse;

// SAFETY: `tile` reads `24·kc` values from `a` and `8·kc` from `b` (aligned loads at multiples
// of 4 floats from 16-byte aligned starts), and reads and writes C only at the
// `dest.rows` by `dest.cols` elements its contract names.
unsafe impl Kernel for Sse {
    const MR: usize = 6;
    const NR: usize = 8;
    const KC: usize = 256;
    const MC: usize = 96;
    const NC: usize = 2048;
    const A_REP: usize = 4;

    fn pack_a(a: &MatRef<f32>, rows: (usize, usize), ks: (usize, usize), dst: &mut [f32]) {
        pack_a::<4>(a, rows, ks, 6, dst, pack_a_6);
    }

    fn pack_b(b: &MatRef<f32>, cols: (usize, usize), ks: (usize, usize), dst: &mut [f32]) {
        if cols.1 < 8 || !pack_b_8(b, cols.0, ks, dst) {
            pack_b(b, cols, ks, 8, dst);
        }
    }

    unsafe fn tile(kc: usize, a: *const f32, b: *const f32, alpha: f32, beta: f32, dest: &Dest) {
        // SAFETY: the caller upholds this function's contract, which is `tile_6x8`'s, and
        // runs this kernel only on CPUs with SSE4.2.
        unsafe { tile_6x8(kc, a, b, alpha, beta, dest) }
    }
}

/// Loads 4 floats from the start of `v`.
///
/// # Panics
///
/// Panics if `v` has fewer than 4.
fn load4(v: &[f32]) -> __m128 {
    let v = &v[..4];
    // SAFETY: `v` holds 4 floats; the load is unaligned.
    unsafe { _mm_loadu_ps(v.as_ptr()) }
}

/// Stores `x` into the first 4 floats of `out`.
///
/// # Panics
///
/// Panics if `out` has fewer than 4.
fn store4(out: &mut [f32], x: __m128) {
    let out = &mut out[..4];
    // SAFETY: `out` holds 4 floats; the store is unaligned.
    unsafe { _mm_storeu_ps(out.as_mut_ptr(), x) }
}

/// Writes the four lanes of `v`, each broadcast to 4 floats, at `out[0]`, `out[stride]`,
/// `out[2·stride]` and `out[3·stride]`.
fn spread(out: &mut [f32], stride: usize, v: __m128) {
    // SAFETY: SSE is part of the x86_64 baseline, so every x86_64 CPU runs these shuffles.
    let lanes = unsafe {
        [
            _mm_shuffle_ps::<0x00>(v, v),
            _mm_shuffle_ps::<0x55>(v, v),
            _mm_shuffle_ps::<0xAA>(v, v),
            _mm_shuffle_ps::<0xFF>(v, v),
        ]
    };
    for (q, lane) in lanes.into_iter().enumerate() {
        store4(&mut out[q * stride..], lane);
    }
}

/// `x` in all four lanes.
fn splat(x: f32) -> __m128 {
    // SAFETY: SSE is part of the x86_64 baseline.
    unsafe { _mm_set1_ps(x) }
}

/// Packs one full 6-row panel of `A` (rows `r0..r0 + 6`, columns `p0..p0 + kc`) into `out`
/// as `kc` groups of 6 values each broadcast to 4 floats, with vector loads and shuffles,
/// when `A`'s rows or columns are contiguous. Returns false, writing nothing, otherwise.
fn pack_a_6(a: &MatRef<f32>, r0: usize, (p0, kc): (usize, usize), out: &mut [f32]) -> bool {
    let (rs, cs) = (a.row_stride(), a.col_stride());
    let data = a.data;
    if cs == 1 {
        // Rows contiguous along k: 4 steps of k from each row at a time.
        let starts: [usize; 6] = std::array::from_fn(|r| (r0 + r) * rs + p0);
        let mut p = 0;
        while p + 4 <= kc {
            for (r, &s) in starts.iter().enumerate() {
                spread(&mut out[p * 24 + 4 * r..], 24, load4(&data[s + p..]));
            }
            p += 4;
        }
        for p in p..kc {
            for (r, &s) in starts.iter().enumerate() {
                store4(&mut out[p * 24 + 4 * r..], splat(data[s + p]));
            }
        }
        true
    } else if rs == 1 {
        // Columns contiguous along the rows: rows 0 to 3 in one load, 4 and 5 alone.
        for p in 0..kc {
            let s = (p0 + p) * cs + r0;
            let group = &mut out[p * 24..(p + 1) * 24];
            spread(group, 4, load4(&data[s..]));
            store4(&mut group[16..], splat(data[s + 4]));
            store4(&mut group[20..], splat(data[s + 5]));
        }
        true
    } else {
        false
    }
}

/// The transpose of the 4 by 4 matrix whose rows are `r`.
fn transpose4(r: [__m128; 4]) -> [__m128; 4] {
    // SAFETY: SSE is part of the x86_64 baseline.
    unsafe {
        let t0 = _mm_unpacklo_ps(r[0], r[1]);
        let t1 = _mm_unpacklo_ps(r[2], r[3]);
        let t2 = _mm_unpackhi_ps(r[0], r[1]);
        let t3 = _mm_unpackhi_ps(r[2], r[3]);
        [
            _mm_movelh_ps(t0, t1),
            _mm_movehl_ps(t1, t0),
            _mm_movelh_ps(t2, t3),
            _mm_movehl_ps(t3, t2),
        ]
    }
}

/// Packs one full 8-column panel of `B` (columns `j0..j0 + 8`, rows `p0..p0 + kc`) into
/// `out` as `kc` groups of 8 values, with vector loads, when `B`'s rows or columns are
/// contiguous. Returns false, writing nothing, otherwise.
fn pack_b_8(b: &MatRef<f32>, j0: usize, (p0, kc): (usize, usize), out: &mut [f32]) -> bool {
    let (rs, cs) = (b.row_stride(), b.col_stride());
    let data = b.data;
    if cs == 1 {
        // Rows contiguous along n: two vectors per step of k.
        for p in 0..kc {
            let s = (p0 + p) * rs + j0;
            store4(&mut out[8 * p..], load4(&data[s..]));
            store4(&mut out[8 * p + 4..], load4(&data[s + 4..]));
        }
        true
    } else if rs == 1 {
        // Columns contiguous along k: 4 steps of k from 4 columns, transposed in registers.
        let starts: [usize; 8] = std::array::from_fn(|j| (j0 + j) * cs + p0);
        let mut p = 0;
        while p + 4 <= kc {
            for half in 0..2 {
                let rows = std::array::from_fn(|q| load4(&data[starts[4 * half + q] + p..]));
                for (i, v) in transpose4(rows).into_iter().enumerate() {
                    store4(&mut out[8 * (p + i) + 4 * half..], v);
                }
            }
            p += 4;
        }
        for p in p..kc {
            for (j, &s) in starts.iter().enumerate() {
                out[8 * p + j] = data[s + p];
            }
        }
        true
    } else {
        false
    }
}

/// The 6 by 8 SSE tile; see [`Kernel::tile`] for the contract.
///
/// # Safety
///
/// As [`Kernel::tile`] with `MR = 6`, `NR = 8` and `A_REP = 4`, on a CPU with SSE4.2.
#[target_feature(enable = "sse4.2")]
unsafe fn tile_6x8(kc: usize, a: *const f32, b: *const f32, alpha: f32, beta: f32, dest: &Dest) {
    let mut acc: [[__m128; 2]; 6] = [[_mm_setzero_ps(); 2]; 6];
    for p in 0..kc {
        // SAFETY: `p < kc`, so floats `8p..8p + 8` of the B panel are inside it, and `8p`
        // floats from a 16-byte aligned start is 16-byte aligned.
        let (b0, b1) = unsafe { (_mm_load_ps(b.add(8 * p)), _mm_load_ps(b.add(8 * p + 4))) };
        for (r, row) in acc.iter_mut().enumerate() {
            // SAFETY: `24p + 4r + 4 <= 24·kc`, inside the A panel (each value stored four
            // times), and a multiple of 4 floats from a 16-byte aligned start, so `pa` is
            // aligned for `__m128`. The value is loaded twice, once per multiply: volatile
            // keeps the two loads from being merged into one load and a register copy, and
            // the copy would take a slot from the multiplies (measured: 17.3 against 22.2
            // GFLOPS for the tile in L1).
            let (x0, x1) = unsafe {
                let pa = a.add(24 * p + 4 * r).cast::<__m128>();
                (std::ptr::read_volatile(pa), std::ptr::read_volatile(pa))
            };
            row[0] = _mm_add_ps(row[0], _mm_mul_ps(x0, b0));
            row[1] = _mm_add_ps(row[1], _mm_mul_ps(x1, b1));
        }
    }
    let va = _mm_set1_ps(alpha);
    for row in &mut acc {
        row[0] = _mm_mul_ps(va, row[0]);
        row[1] = _mm_mul_ps(va, row[1]);
    }
    if dest.rows == 6 && dest.cols == 8 && dest.cs == 1 {
        let vb = _mm_set1_ps(beta);
        for (r, row) in acc.iter().enumerate() {
            // SAFETY: row `r < 6 = dest.rows` and its 8 columns are elements of C, contiguous
            // since `cs = 1`; the caller gives this call sole access to them.
            unsafe {
                let p = dest.ptr.add(r * dest.rs);
                if beta == 0.0 {
                    _mm_storeu_ps(p, row[0]);
                    _mm_storeu_ps(p.add(4), row[1]);
                } else {
                    let c0 = _mm_mul_ps(vb, _mm_loadu_ps(p));
                    let c1 = _mm_mul_ps(vb, _mm_loadu_ps(p.add(4)));
                    _mm_storeu_ps(p, _mm_add_ps(row[0], c0));
                    _mm_storeu_ps(p.add(4), _mm_add_ps(row[1], c1));
                }
            }
        }
    } else {
        // An edge tile, or C with non-unit column stride: the same arithmetic, one element
        // at a time, storing only the valid part.
        let mut t = [0.0f32; 48];
        for (r, row) in acc.iter().enumerate() {
            // SAFETY: `t` has room for 8 floats at `8r` (r < 6).
            unsafe {
                _mm_storeu_ps(t.as_mut_ptr().add(8 * r), row[0]);
                _mm_storeu_ps(t.as_mut_ptr().add(8 * r + 4), row[1]);
            }
        }
        for r in 0..dest.rows {
            for j in 0..dest.cols {
                // SAFETY: `r < dest.rows` and `j < dest.cols`, so this is an element of C
                // the caller gives this call sole access to.
                unsafe {
                    let p = dest.ptr.add(r * dest.rs + j * dest.cs);
                    *p = if beta == 0.0 {
                        t[8 * r + j]
                    } else {
                        t[8 * r + j] + beta * *p
                    };
                }
            }
        }
    }
}
