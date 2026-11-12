//! The matrix product of [`Level::Accelerate`](super::Level::Accelerate): Apple's
//! Accelerate framework (`cblas_sgemm`), called once per product on the calling thread.
//!
//! [`plan`] maps the strided views to BLAS's row-major arguments (a transpose flag and a
//! leading dimension per operand). A view qualifies when one of its strides is 1 and the
//! other covers a whole row (or column); a column-major `C` is computed as
//! `Cᵀ = Bᵀ · Aᵀ`. Anything else (both strides above 1, overlapping rows, sizes beyond
//! `i32`, or `alpha = 0`, where BLAS skips `A · B` and a NaN in it would not reach `C`) gets
//! no plan, and the caller runs the NEON GEMM instead. The planning is plain Rust and is
//! tested on every target; only the call into Accelerate is compiled on Apple Silicon.
//!
//! Accelerate computes each product in one call, which Cavitation never splits across its
//! thread pool, so the result does not depend on the thread count. It rounds differently
//! from the NEON kernel, so the two agree within the GEMM tolerance, not bit for bit.

use super::{MatMut, MatRef, required_len};

/// `CblasRowMajor` in `<cblas.h>`.
#[cfg(all(target_vendor = "apple", target_arch = "aarch64"))]
const ROW_MAJOR: i32 = 101;
/// `CblasNoTrans` in `<cblas.h>`.
#[cfg(all(target_vendor = "apple", target_arch = "aarch64"))]
const NO_TRANS: i32 = 111;
/// `CblasTrans` in `<cblas.h>`.
#[cfg(all(target_vendor = "apple", target_arch = "aarch64"))]
const TRANS: i32 = 112;

/// One operand as BLAS reads it in row-major order: the matrix itself (`transposed` false,
/// stored row by row with rows `ld` apart) or the transpose of a row-major matrix
/// (`transposed` true, stored column by column with columns `ld` apart).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Operand {
    /// Whether BLAS reads the stored matrix transposed (`CblasTrans`).
    pub(super) transposed: bool,
    /// The leading dimension: the distance between rows of the stored matrix.
    pub(super) ld: i32,
}

/// The arguments of one row-major `cblas_sgemm` call for `C = alpha · A · B + beta · C`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Plan {
    /// Whether the call computes `Cᵀ = Bᵀ · Aᵀ`: the first operand is then `B`'s storage and
    /// the second `A`'s, and `m` and `n` are swapped.
    pub(super) swapped: bool,
    /// Rows of the result the call writes.
    pub(super) m: i32,
    /// Columns of the result the call writes.
    pub(super) n: i32,
    /// The length of each sum.
    pub(super) k: i32,
    /// The first operand (`A`, or `Bᵀ` when swapped).
    pub(super) first: Operand,
    /// The second operand (`B`, or `Aᵀ` when swapped).
    pub(super) second: Operand,
    /// The leading dimension of the result's rows.
    pub(super) ldc: i32,
}

/// How BLAS can read a `rows` by `cols` view with the given strides, or `None` if it cannot:
/// row-major when its columns are contiguous and rows do not overlap, transposed when its
/// rows are contiguous and columns do not overlap. A stride is irrelevant along a dimension
/// of extent 1, and the view must fit in `i32` indices.
fn layout(rows: usize, cols: usize, row_stride: usize, col_stride: usize) -> Option<Operand> {
    let span = required_len(rows, cols, row_stride, col_stride)?;
    i32::try_from(span).ok()?;
    let cols_contiguous = col_stride == 1 || cols <= 1;
    let rows_contiguous = row_stride == 1 || rows <= 1;
    let (transposed, ld) = if cols_contiguous && (rows <= 1 || row_stride >= cols) {
        (false, if rows <= 1 { cols.max(1) } else { row_stride })
    } else if rows_contiguous && (cols <= 1 || col_stride >= rows) {
        (true, if cols <= 1 { rows.max(1) } else { col_stride })
    } else {
        return None;
    };
    Some(Operand {
        transposed,
        ld: i32::try_from(ld).ok()?,
    })
}

/// The `cblas_sgemm` call for `C = alpha · A · B + beta · C`, or `None` if BLAS cannot
/// compute it as the reference does: a layout it cannot express, a size beyond `i32`, an
/// empty product (the caller handles those) or `alpha = 0`.
pub(super) fn plan(alpha: f32, a: &MatRef<f32>, b: &MatRef<f32>, c: &MatMut<f32>) -> Option<Plan> {
    let (m, n, k) = (a.rows, b.cols, a.cols);
    if m == 0 || n == 0 || k == 0 || alpha == 0.0 {
        return None;
    }
    let la = layout(a.rows, a.cols, a.row_stride, a.col_stride)?;
    let lb = layout(b.rows, b.cols, b.row_stride, b.col_stride)?;
    let lc = layout(c.rows, c.cols, c.row_stride, c.col_stride)?;
    let (m, n, k) = (
        i32::try_from(m).ok()?,
        i32::try_from(n).ok()?,
        i32::try_from(k).ok()?,
    );
    let flip = |o: Operand| Operand {
        transposed: !o.transposed,
        ld: o.ld,
    };
    Some(if lc.transposed {
        // C is column-major: Cᵀ is row-major, and Cᵀ = Bᵀ · Aᵀ reads the same storage with
        // each operand's transpose flag flipped.
        Plan {
            swapped: true,
            m: n,
            n: m,
            k,
            first: flip(lb),
            second: flip(la),
            ldc: lc.ld,
        }
    } else {
        Plan {
            swapped: false,
            m,
            n,
            k,
            first: la,
            second: lb,
            ldc: lc.ld,
        }
    })
}

/// `C = alpha · A · B + beta · C` through Accelerate's `cblas_sgemm`, in one call on this
/// thread, when [`plan`] has a call for it; otherwise returns `c` untouched for the caller to
/// compute another way. Allocates nothing (on the Rust side).
#[cfg(all(target_vendor = "apple", target_arch = "aarch64"))]
pub(super) fn gemm<'c>(
    alpha: f32,
    a: MatRef<f32>,
    b: MatRef<f32>,
    beta: f32,
    c: MatMut<'c, f32>,
) -> Result<(), MatMut<'c, f32>> {
    // SAFETY: the declaration matches `cblas_sgemm` in Accelerate's `<cblas.h>` (the enums
    // are C `int`s, the sizes `int` in the LP64 interface this symbol is), and the
    // Accelerate framework, linked by the build script on Apple Silicon, exports it.
    unsafe extern "C" {
        /// `C = alpha · op(A) · op(B) + beta · C` in single precision, reading `A`, `B` and
        /// `C` through their leading dimensions in the given order.
        fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
    let Some(p) = plan(alpha, &a, &b, &c) else {
        return Err(c);
    };
    let (first, second) = if p.swapped {
        (b.data.as_ptr(), a.data.as_ptr())
    } else {
        (a.data.as_ptr(), b.data.as_ptr())
    };
    let flag = |o: Operand| if o.transposed { TRANS } else { NO_TRANS };
    let c_ptr = c.data.as_mut_ptr();
    // SAFETY: `plan` described each operand exactly as its view lays it out (the transpose
    // flag and leading dimension cover the same elements as the strides, and `m`, `n`, `k`
    // are the view's sizes), and each view was bounds-checked against its slice when it was
    // built, so BLAS reads and writes only inside the three slices. `C` is a `MatMut`, which
    // never aliases itself, and Rust's borrows keep it apart from `A` and `B`. Every size and
    // leading dimension fits in `i32` (checked by `plan`), and the call returns before the
    // borrows end.
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            flag(p.first),
            flag(p.second),
            p.m,
            p.n,
            p.k,
            alpha,
            first,
            p.first.ld,
            second,
            p.second.ld,
            beta,
            c_ptr,
            p.ldc,
        );
    }
    Ok(())
}

/// Tests for the planning: the plans BLAS gets for each layout, checked on every target by
/// running them through a plain-Rust stand-in for row-major `cblas_sgemm`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;
    use crate::kernels::reference;

    /// Row-major `cblas_sgemm` written out: `C = alpha · op(X) · op(Y) + beta · C`, reading
    /// each operand only through the plan's flags and leading dimensions. With `beta = 0`, C
    /// is overwritten without being read, as BLAS specifies.
    fn blas_stand_in(p: &Plan, alpha: f32, x: &[f32], y: &[f32], beta: f32, c: &mut [f32]) {
        let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
        let at = |o: Operand, data: &[f32], i: usize, j: usize| {
            let ld = o.ld as usize;
            if o.transposed {
                data[j * ld + i]
            } else {
                data[i * ld + j]
            }
        };
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for q in 0..k {
                    acc += at(p.first, x, i, q) * at(p.second, y, q, j);
                }
                let out = &mut c[i * p.ldc as usize + j];
                *out = if beta == 0.0 {
                    alpha * acc
                } else {
                    beta * *out + alpha * acc
                };
            }
        }
    }

    /// A buffer holding the row-major `rows` by `cols` values `v` with the given strides,
    /// NaN everywhere else.
    fn store(v: &[f32], rows: usize, cols: usize, rs: usize, cs: usize) -> Vec<f32> {
        let mut d = vec![f32::NAN; required_len(rows, cols, rs, cs).unwrap().max(1)];
        for i in 0..rows {
            for j in 0..cols {
                d[i * rs + j * cs] = v[i * cols + j];
            }
        }
        d
    }

    /// The strides of a `rows` by `cols` view: row-major or column-major, with `pad` unused
    /// values after each row or column.
    fn strides(rows: usize, cols: usize, column_major: bool, pad: usize) -> (usize, usize) {
        if column_major {
            (1, rows + pad)
        } else {
            (cols + pad, 1)
        }
    }

    /// Every combination of row-major and column-major A, B and C, with and without padding,
    /// on shapes with dimensions of 1 and larger ones, gets a plan, and the plan run through
    /// the stand-in gives the reference's result exactly (both sum each element in the same
    /// order).
    #[test]
    fn plans_reproduce_the_reference() {
        let mut rng = Rng::seed(29);
        let shapes = [
            (1, 1, 1),
            (1, 7, 5),
            (6, 1, 4),
            (5, 3, 1),
            (1, 1, 9),
            (7, 9, 11),
            (16, 12, 8),
        ];
        let mut planned = 0;
        for (m, n, k) in shapes {
            let a: Vec<f32> = (0..m * k).map(|_| rng.range(-1.0, 1.0) as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|_| rng.range(-1.0, 1.0) as f32).collect();
            let c0: Vec<f32> = (0..m * n).map(|_| rng.range(-1.0, 1.0) as f32).collect();
            for bits in 0..16u32 {
                let (ta, tb, tc, pad) = (
                    bits & 1 != 0,
                    bits & 2 != 0,
                    bits & 4 != 0,
                    (bits >> 3) as usize * 2,
                );
                let (ars, acs) = strides(m, k, ta, pad);
                let (brs, bcs) = strides(k, n, tb, pad);
                let (crs, ccs) = strides(m, n, tc, pad);
                let sa = store(&a, m, k, ars, acs);
                let sb = store(&b, k, n, brs, bcs);
                for beta in [0.0f32, 0.5] {
                    let start = if beta == 0.0 {
                        vec![f32::NAN; m * n]
                    } else {
                        c0.clone()
                    };
                    let mut want = store(&start, m, n, crs, ccs);
                    let mut got = want.clone();
                    let av = MatRef::new(&sa, m, k, ars, acs).unwrap();
                    let bv = MatRef::new(&sb, k, n, brs, bcs).unwrap();
                    reference::gemm(
                        -0.75,
                        av,
                        bv,
                        beta,
                        MatMut::new(&mut want, m, n, crs, ccs).unwrap(),
                    );
                    let p = plan(
                        -0.75,
                        &av,
                        &bv,
                        &MatMut::new(&mut got, m, n, crs, ccs).unwrap(),
                    )
                    .unwrap_or_else(|| panic!("no plan for {m}x{n}x{k}, bits {bits}"));
                    let (x, y) = if p.swapped { (&sb, &sa) } else { (&sa, &sb) };
                    blas_stand_in(&p, -0.75, x, y, beta, &mut got);
                    let same = want
                        .iter()
                        .zip(&got)
                        .all(|(w, g)| w.to_bits() == g.to_bits() || (w.is_nan() && g.is_nan()));
                    assert!(
                        same,
                        "{m}x{n}x{k}, bits {bits}, beta {beta}: {got:?} != {want:?}"
                    );
                    planned += 1;
                }
            }
        }
        assert_eq!(planned, shapes.len() * 32);
    }

    /// Layouts BLAS cannot express get no plan, so the caller falls back: both strides above
    /// 1, rows that overlap, a broadcast row, `alpha = 0` and an empty product.
    #[test]
    fn unexpressible_products_get_no_plan() {
        let d = [1.0f32; 64];
        let mut cd = [0.0f32; 64];
        let rm = |r, c| MatRef::new(&d, r, c, c, 1).unwrap();
        let c = MatMut::new(&mut cd, 3, 3, 3, 1).unwrap();
        let strided = MatRef::new(&d, 3, 3, 7, 2).unwrap();
        let overlapping = MatRef::new(&d, 3, 3, 2, 1).unwrap();
        let broadcast = MatRef::new(&d, 3, 3, 0, 1).unwrap();
        assert!(plan(1.0, &rm(3, 3), &rm(3, 3), &c).is_some());
        for bad in [strided, overlapping, broadcast] {
            assert_eq!(plan(1.0, &bad, &rm(3, 3), &c), None, "{bad:?} as A");
            assert_eq!(plan(1.0, &rm(3, 3), &bad, &c), None, "{bad:?} as B");
        }
        assert_eq!(plan(0.0, &rm(3, 3), &rm(3, 3), &c), None, "alpha 0");
        let c_strided = MatMut::new(&mut cd, 3, 3, 8, 2).unwrap();
        assert_eq!(
            plan(1.0, &rm(3, 3), &rm(3, 3), &c_strided),
            None,
            "strided C"
        );
        let mut empty = [0.0f32; 0];
        let c_empty = MatMut::new(&mut empty, 0, 3, 3, 1).unwrap();
        assert_eq!(plan(1.0, &rm(0, 3), &rm(3, 3), &c_empty), None, "m = 0");
    }

    /// The layouts the tape's products use (row-major A, B row-major or transposed, padded
    /// rows, row-major C) all get a plan with the expected flags and leading dimensions.
    #[test]
    fn training_layouts_get_the_expected_plan() {
        let d = vec![0.0f32; 1024 * 784];
        let mut cd = vec![0.0f32; 128 * 1024];
        let x = MatRef::new(&d, 128, 784, 784, 1).unwrap();
        let w_t = MatRef::new(&d, 784, 1024, 1, 784).unwrap();
        let c = MatMut::new(&mut cd, 128, 1024, 1024, 1).unwrap();
        let p = plan(1.0, &x, &w_t, &c).unwrap();
        assert_eq!(
            p,
            Plan {
                swapped: false,
                m: 128,
                n: 1024,
                k: 784,
                first: Operand {
                    transposed: false,
                    ld: 784
                },
                second: Operand {
                    transposed: true,
                    ld: 784
                },
                ldc: 1024,
            }
        );
    }
}
