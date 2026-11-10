//! Fast kernels against their reference twins. Each test names the kernel level directly,
//! so tests never depend on, or race on, the process-wide level setting.

use super::gemm::{gemm_f32, gemm_f32_with, threads_used};
use super::{Level, MatMut, MatRef, reference, supported};
use crate::Rng;

/// A matrix stored in its own buffer, with the layout a test chose for it.
struct Stored {
    /// The values, `rows * cols` of them plus any padding.
    data: Vec<f32>,
    /// Number of rows.
    rows: usize,
    /// Number of columns.
    cols: usize,
    /// Distance between vertically adjacent elements.
    row_stride: usize,
    /// Distance between horizontally adjacent elements.
    col_stride: usize,
}

impl Stored {
    /// Stores the row-major `rows` by `cols` values `logical`, row-major when `transposed`
    /// is false and column-major when it is true, with `pad` unused values after every row
    /// (or column). Padding holds NaN, so reading it shows up in the result.
    fn new(logical: &[f32], rows: usize, cols: usize, transposed: bool, pad: usize) -> Stored {
        let (outer, inner) = if transposed {
            (cols, rows)
        } else {
            (rows, cols)
        };
        let lead = inner + pad;
        let mut data = vec![f32::NAN; outer * lead];
        for i in 0..rows {
            for j in 0..cols {
                let at = if transposed {
                    j * lead + i
                } else {
                    i * lead + j
                };
                data[at] = logical[i * cols + j];
            }
        }
        let (row_stride, col_stride) = if transposed { (1, lead) } else { (lead, 1) };
        Stored {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
        }
    }

    /// Stores the row-major `rows` by `cols` values `logical` with neither stride 1: column
    /// stride 2 and row stride `2·cols + 1`, the gaps holding NaN.
    fn strided(logical: &[f32], rows: usize, cols: usize) -> Stored {
        let (row_stride, col_stride) = (2 * cols + 1, 2);
        let mut data = vec![f32::NAN; rows * row_stride];
        for i in 0..rows {
            for j in 0..cols {
                data[i * row_stride + j * col_stride] = logical[i * cols + j];
            }
        }
        Stored {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
        }
    }

    /// A read-only view of the matrix.
    fn view(&self) -> MatRef<'_, f32> {
        MatRef::new(
            &self.data,
            self.rows,
            self.cols,
            self.row_stride,
            self.col_stride,
        )
        .unwrap()
    }

    /// A writable view of the matrix.
    fn view_mut(&mut self) -> MatMut<'_, f32> {
        MatMut::new(
            &mut self.data,
            self.rows,
            self.cols,
            self.row_stride,
            self.col_stride,
        )
        .unwrap()
    }

    /// The logical values, row-major.
    fn logical(&self) -> Vec<f32> {
        let v = self.view();
        (0..self.rows * self.cols)
            .map(|x| v.get(x / self.cols, x % self.cols))
            .collect()
    }
}

/// `n` uniform values in `[-1, 1)`.
fn random(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.range(-1.0, 1.0) as f32).collect()
}

/// Row-major view helper.
fn rm(d: &[f32], r: usize, c: usize) -> MatRef<'_, f32> {
    MatRef::new(d, r, c, c, 1).unwrap()
}

/// Row-major mutable view helper.
fn rm_mut(d: &mut [f32], r: usize, c: usize) -> MatMut<'_, f32> {
    MatMut::new(d, r, c, c, 1).unwrap()
}

/// The GEMM tolerance for every element of `C = alpha·A·B + beta·C0` (all row-major
/// logical values), computed in f64:
/// `2·(k + 2)·1.19e-7·(|alpha|·Σₚ |aᵢₚ|·|bₚⱼ| + |beta|·|c0ᵢⱼ|) + 1e-30`.
fn bounds(
    a: &[f32],
    b: &[f32],
    c0: &[f32],
    (m, n, k): (usize, usize, usize),
    alpha: f32,
    beta: f32,
) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for p in 0..k {
                s += f64::from(a[i * k + p]).abs() * f64::from(b[p * n + j]).abs();
            }
            let cterm = if beta == 0.0 {
                0.0
            } else {
                f64::from(beta).abs() * f64::from(c0[i * n + j]).abs()
            };
            out[i * n + j] =
                2.0 * (k as f64 + 2.0) * 1.19e-7 * (f64::from(alpha).abs() * s + cterm) + 1e-30;
        }
    }
    out
}

/// Asserts that `got` matches `want` element by element within `bound`, naming `what` and
/// the first element outside it.
fn assert_within(got: &[f32], want: &[f32], bound: &[f64], n: usize, what: &str) {
    for (x, ((g, w), b)) in got.iter().zip(want).zip(bound).enumerate() {
        let diff = (f64::from(*g) - f64::from(*w)).abs();
        assert!(
            diff <= *b,
            "{what}: element ({}, {}) is {g}, reference {w}, difference {diff:e} above {b:e}",
            x / n,
            x % n
        );
    }
}

/// Runs the fast kernel at `level` and `threads` and the reference on the same logical
/// inputs, for every transpose combination, and checks every element within the GEMM
/// tolerance. With `beta = 0` the fast kernel's C starts as NaN, which must not leak.
#[allow(clippy::too_many_arguments)]
fn check_case(
    rng: &mut Rng,
    level: Level,
    threads: usize,
    (m, n, k): (usize, usize, usize),
    alphas: &[f32],
    betas: &[f32],
    pad: usize,
    c_transposed: bool,
) {
    let a = random(rng, m * k);
    let b = random(rng, k * n);
    let c0 = random(rng, m * n);
    for &alpha in alphas {
        for &beta in betas {
            let mut want = c0.clone();
            reference::gemm(
                alpha,
                rm(&a, m, k),
                rm(&b, k, n),
                beta,
                rm_mut(&mut want, m, n),
            );
            let bound = bounds(&a, &b, &c0, (m, n, k), alpha, beta);
            // The four transpose combinations, then A, B and C all with general strides.
            for (ta, tb, general) in [
                (false, false, false),
                (true, false, false),
                (false, true, false),
                (true, true, false),
                (false, false, true),
            ] {
                let start = if beta == 0.0 {
                    vec![f32::NAN; m * n]
                } else {
                    c0.clone()
                };
                let (sa, sb, mut sc) = if general {
                    (
                        Stored::strided(&a, m, k),
                        Stored::strided(&b, k, n),
                        Stored::strided(&start, m, n),
                    )
                } else {
                    (
                        Stored::new(&a, m, k, ta, pad),
                        Stored::new(&b, k, n, tb, pad),
                        Stored::new(&start, m, n, c_transposed, pad),
                    )
                };
                gemm_f32_with(
                    level,
                    threads,
                    alpha,
                    sa.view(),
                    sb.view(),
                    beta,
                    sc.view_mut(),
                );
                let what = format!(
                    "{m}x{n}x{k}, A transposed {ta}, B transposed {tb}, C transposed \
                     {c_transposed}, general strides {general}, alpha {alpha}, beta {beta}, \
                     {threads} threads"
                );
                assert_within(&sc.logical(), &want, &bound, n, &what);
            }
        }
    }
}

/// The SSE4.2 GEMM matches the reference within the GEMM tolerance on 400 shapes sampled
/// (seed 3) from every combination of sizes around the tile edges, for all four transpose
/// combinations, alpha in {1, -0.5} and beta in {0, 1, 0.25}.
#[test]
fn sse_matches_reference() {
    if !supported(Level::Sse42) {
        return;
    }
    let sizes = [
        1usize, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 65, 127, 129, 255, 257,
    ];
    let mut cases = Vec::new();
    for &m in &sizes {
        for &n in &sizes {
            for &k in &sizes {
                cases.push((m, n, k));
            }
        }
    }
    let mut rng = Rng::seed(3);
    rng.shuffle(&mut cases);
    for (i, &shape) in cases[..400].iter().enumerate() {
        check_case(
            &mut rng,
            Level::Sse42,
            1 + i % 4,
            shape,
            &[1.0, -0.5],
            &[0.0, 1.0, 0.25],
            i % 3,
            i % 2 == 1,
        );
    }
}

/// A 512 x 512 x 512 product on 4 threads matches the reference within the tolerance.
#[test]
fn sse_large_square() {
    if !supported(Level::Sse42) {
        return;
    }
    let mut rng = Rng::seed(5);
    check_case(
        &mut rng,
        Level::Sse42,
        4,
        (512, 512, 512),
        &[1.0],
        &[0.0, 0.25],
        0,
        false,
    );
}

/// Shapes with one dimension of 1 (a row, a column, an outer product) and 1 x 1 x 1 match
/// the reference.
#[test]
fn sse_vector_shapes() {
    if !supported(Level::Sse42) {
        return;
    }
    let mut rng = Rng::seed(7);
    let shapes = [
        (1, 1, 1),
        (1, 300, 77),
        (300, 1, 77),
        (37, 300, 1),
        (1, 1, 500),
        (1, 9, 1),
        (9, 1, 1),
    ];
    for (i, &shape) in shapes.iter().enumerate() {
        check_case(
            &mut rng,
            Level::Sse42,
            1 + i % 4,
            shape,
            &[1.0, -0.5],
            &[0.0, 1.0],
            i % 2,
            i % 2 == 0,
        );
    }
}

/// With m or n zero nothing is written; with k zero, C becomes beta·C (zeros for beta 0,
/// even over NaN).
#[test]
fn sse_empty_dims() {
    if !supported(Level::Sse42) {
        return;
    }
    let a = [1.0f32; 12];
    let b = [1.0f32; 12];
    // m = 0 and n = 0: C's storage is untouched.
    let mut c = [3.0f32; 6];
    let empty_c = MatMut::new(&mut c, 0, 3, 3, 1).unwrap();
    gemm_f32_with(
        Level::Sse42,
        4,
        1.0,
        rm(&a, 0, 4),
        rm(&b, 4, 3),
        0.0,
        empty_c,
    );
    assert_eq!(c, [3.0; 6]);
    let empty_c = MatMut::new(&mut c, 2, 0, 3, 1).unwrap();
    gemm_f32_with(
        Level::Sse42,
        4,
        1.0,
        rm(&a, 2, 4),
        rm(&b, 4, 0),
        0.0,
        empty_c,
    );
    assert_eq!(c, [3.0; 6]);
    // k = 0: C = beta·C.
    let mut c = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    gemm_f32_with(
        Level::Sse42,
        4,
        1.0,
        MatRef::new(&[], 2, 0, 0, 1).unwrap(),
        MatRef::new(&[], 0, 3, 3, 1).unwrap(),
        2.0,
        rm_mut(&mut c, 2, 3),
    );
    assert_eq!(c, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    let mut c = [f32::NAN; 6];
    gemm_f32_with(
        Level::Sse42,
        1,
        1.0,
        MatRef::new(&[], 2, 0, 0, 1).unwrap(),
        MatRef::new(&[], 0, 3, 3, 1).unwrap(),
        0.0,
        rm_mut(&mut c, 2, 3),
    );
    assert_eq!(c, [0.0; 6]);
}

/// The bits of `alpha·A·B + beta·C` for one product at `threads` threads.
#[allow(clippy::too_many_arguments)]
fn product_bits(
    level: Level,
    threads: usize,
    a: MatRef<f32>,
    b: MatRef<f32>,
    beta: f32,
    c0: &[f32],
) -> Vec<u32> {
    let (m, n) = (a.rows(), b.cols());
    let mut c = c0.to_vec();
    gemm_f32_with(level, threads, 0.75, a, b, beta, rm_mut(&mut c, m, n));
    c.into_iter().map(f32::to_bits).collect()
}

/// 1, 2, 3, 4 and 7 threads give bit-identical results on 300 x 700 x 500 and on the
/// training shapes 128 x 1024 x 784 (B transposed, as in a forward pass) and
/// 32 x 256 x 1024.
#[test]
fn threads_bit_identical() {
    if !supported(Level::Sse42) {
        return;
    }
    let mut rng = Rng::seed(11);
    for (m, n, k, b_transposed) in [
        (300, 700, 500, false),
        (128, 1024, 784, true),
        (32, 256, 1024, false),
    ] {
        let a = random(&mut rng, m * k);
        let b = Stored::new(&random(&mut rng, k * n), k, n, b_transposed, 0);
        let c0 = random(&mut rng, m * n);
        for beta in [0.0, 1.0] {
            let one = product_bits(Level::Sse42, 1, rm(&a, m, k), b.view(), beta, &c0);
            for threads in [2, 3, 4, 7] {
                let many = product_bits(Level::Sse42, threads, rm(&a, m, k), b.view(), beta, &c0);
                assert!(
                    one == many,
                    "{m}x{n}x{k}, beta {beta}: {threads} threads differ from 1 thread"
                );
            }
        }
    }
}

/// The same call twice gives the same bits.
#[test]
fn run_to_run_identical() {
    if !supported(Level::Sse42) {
        return;
    }
    let mut rng = Rng::seed(13);
    let (m, n, k) = (200, 300, 400);
    let a = random(&mut rng, m * k);
    let b = random(&mut rng, k * n);
    let c0 = random(&mut rng, m * n);
    let first = product_bits(Level::Sse42, 4, rm(&a, m, k), rm(&b, k, n), 0.5, &c0);
    for _ in 0..3 {
        let again = product_bits(Level::Sse42, 4, rm(&a, m, k), rm(&b, k, n), 0.5, &c0);
        assert!(first == again, "a repeated call gave different bits");
    }
}

/// The reported thread count is the one the call runs on: never more than its tasks.
#[test]
fn threads_used_matches_the_work() {
    if !supported(Level::Sse42) {
        return;
    }
    // One 6 by 8 tile: one task, so one thread, however many are offered.
    assert_eq!(threads_used(Level::Sse42, 4, 6, 8, 4096), 1);
    assert_eq!(threads_used(Level::Sse42, 4, 128, 1024, 784), 4);
    assert_eq!(threads_used(Level::Sse42, 4, 0, 1024, 784), 1);
    assert!(threads_used(Level::Sse42, 1000, 1024, 1024, 1024) <= 256);
}

/// Asking for a level this CPU cannot run panics, naming the level, before any kernel runs.
#[test]
#[should_panic(expected = "cannot run kernel level")]
fn gemm_with_unsupported_level_panics() {
    let foreign = if cfg!(target_arch = "aarch64") {
        Level::Avx2Fma
    } else {
        Level::Neon
    };
    let (a, b, mut c) = ([1.0f32; 4], [1.0f32; 4], [0.0f32; 4]);
    gemm_f32_with(
        foreign,
        1,
        1.0,
        rm(&a, 2, 2),
        rm(&b, 2, 2),
        0.0,
        rm_mut(&mut c, 2, 2),
    );
}

/// A thread count of 0 panics.
#[test]
#[should_panic(expected = "threads must be at least 1")]
fn gemm_with_zero_threads_panics() {
    let (a, b, mut c) = ([1.0f32; 4], [1.0f32; 4], [0.0f32; 4]);
    gemm_f32_with(
        Level::Reference,
        0,
        1.0,
        rm(&a, 2, 2),
        rm(&b, 2, 2),
        0.0,
        rm_mut(&mut c, 2, 2),
    );
}

/// `gemm_f32` at the current level matches the reference on a small product.
#[test]
fn gemm_f32_uses_a_supported_level() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];
    let mut c = [f32::NAN; 4];
    gemm_f32(1.0, rm(&a, 2, 2), rm(&b, 2, 2), 0.0, rm_mut(&mut c, 2, 2));
    assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
}
