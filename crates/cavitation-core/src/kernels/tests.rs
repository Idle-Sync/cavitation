//! Fast kernels against their reference twins. Each test names the kernel level directly,
//! so tests never depend on, or race on, the process-wide level setting.

use super::gemm::{Grid, gemm_f32, gemm_f32_with, threads_used};
use super::matvec::{matvec_f32, matvec_f32_with};
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

/// Every level with a packed kernel, and the `MR` by `NR` tile its kernel computes.
const FAST: [(Level, usize, usize); 3] = [
    (Level::Sse42, 6, 8),
    (Level::Avx2Fma, 6, 16),
    (Level::Neon, 8, 12),
];

/// The fast levels this CPU runs, with their tiles. Prints each level it skips, so the test
/// output shows which kernels were not tested here.
fn fast_levels(test: &str) -> Vec<(Level, usize, usize)> {
    FAST.into_iter()
        .filter(|&(level, ..)| {
            let ok = supported(level);
            if !ok {
                println!("{test}: skipped kernel level {level}, not supported by this CPU");
            }
            ok
        })
        .collect()
}

/// Every fast kernel matches the reference within the GEMM tolerance on 400 shapes sampled
/// (seed 3) from every combination of sizes around the tile edges, for all four transpose
/// combinations, alpha in {1, -0.5} and beta in {0, 1, 0.25}.
#[test]
fn fast_matches_reference() {
    let sizes = [
        1usize, 2, 3, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 31, 33, 64, 65, 127, 129, 255, 257,
    ];
    let mut cases = Vec::new();
    for &m in &sizes {
        for &n in &sizes {
            for &k in &sizes {
                cases.push((m, n, k));
            }
        }
    }
    for (level, ..) in fast_levels("fast_matches_reference") {
        let mut cases = cases.clone();
        let mut rng = Rng::seed(3);
        rng.shuffle(&mut cases);
        for (i, &shape) in cases[..400].iter().enumerate() {
            check_case(
                &mut rng,
                level,
                1 + i % 4,
                shape,
                &[1.0, -0.5],
                &[0.0, 1.0, 0.25],
                i % 3,
                i % 2 == 1,
            );
        }
    }
}

/// A 512 x 512 x 512 product on 4 threads matches the reference within the tolerance, at
/// every fast level.
#[test]
fn fast_large_square() {
    for (level, ..) in fast_levels("fast_large_square") {
        let mut rng = Rng::seed(5);
        check_case(
            &mut rng,
            level,
            4,
            (512, 512, 512),
            &[1.0],
            &[0.0, 0.25],
            0,
            false,
        );
    }
}

/// Shapes with one dimension of 1 (a row, a column, an outer product) and 1 x 1 x 1 match
/// the reference, at every fast level.
#[test]
fn fast_vector_shapes() {
    let shapes = [
        (1, 1, 1),
        (1, 300, 77),
        (300, 1, 77),
        (37, 300, 1),
        (1, 1, 500),
        (1, 9, 1),
        (9, 1, 1),
        (1, 13, 1),
        (13, 1, 1),
    ];
    for (level, ..) in fast_levels("fast_vector_shapes") {
        let mut rng = Rng::seed(7);
        for (i, &shape) in shapes.iter().enumerate() {
            check_case(
                &mut rng,
                level,
                1 + i % 4,
                shape,
                &[1.0, -0.5],
                &[0.0, 1.0],
                i % 2,
                i % 2 == 0,
            );
        }
    }
}

/// With m or n zero nothing is written; with k zero, C becomes beta·C (zeros for beta 0,
/// even over NaN). At every fast level.
#[test]
fn fast_empty_dims() {
    for (level, ..) in fast_levels("fast_empty_dims") {
        let a = [1.0f32; 12];
        let b = [1.0f32; 12];
        // m = 0 and n = 0: C's storage is untouched.
        let mut c = [3.0f32; 6];
        let empty_c = MatMut::new(&mut c, 0, 3, 3, 1).unwrap();
        gemm_f32_with(level, 4, 1.0, rm(&a, 0, 4), rm(&b, 4, 3), 0.0, empty_c);
        assert_eq!(c, [3.0; 6], "{level}");
        let empty_c = MatMut::new(&mut c, 2, 0, 3, 1).unwrap();
        gemm_f32_with(level, 4, 1.0, rm(&a, 2, 4), rm(&b, 4, 0), 0.0, empty_c);
        assert_eq!(c, [3.0; 6], "{level}");
        // k = 0: C = beta·C.
        let mut c = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        gemm_f32_with(
            level,
            4,
            1.0,
            MatRef::new(&[], 2, 0, 0, 1).unwrap(),
            MatRef::new(&[], 0, 3, 3, 1).unwrap(),
            2.0,
            rm_mut(&mut c, 2, 3),
        );
        assert_eq!(c, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0], "{level}");
        let mut c = [f32::NAN; 6];
        gemm_f32_with(
            level,
            1,
            1.0,
            MatRef::new(&[], 2, 0, 0, 1).unwrap(),
            MatRef::new(&[], 0, 3, 3, 1).unwrap(),
            0.0,
            rm_mut(&mut c, 2, 3),
        );
        assert_eq!(c, [0.0; 6], "{level}");
    }
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
/// 32 x 256 x 1024, at every fast level. The test binary re-runs itself as a child that runs
/// only `threads_bit_identical_child`, so nothing else holds the thread pool and the
/// multi-threaded products really use its workers.
#[test]
fn threads_bit_identical() {
    let out = crate::test_child::run_alone("kernels::twin_tests::threads_bit_identical_child", &[]);
    crate::test_child::assert_passed(&out);
    let stdout = String::from_utf8_lossy(&out.stdout);
    print!("{stdout}");
}

/// The body of `threads_bit_identical`, run in a child process after checking that the pool
/// runs tasks on its workers. Ignored by default.
#[test]
#[ignore = "run in a child process by threads_bit_identical"]
fn threads_bit_identical_child() {
    crate::test_child::assert_pool_runs_on_workers();
    for (level, ..) in fast_levels("threads_bit_identical") {
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
                let one = product_bits(level, 1, rm(&a, m, k), b.view(), beta, &c0);
                for threads in [2, 3, 4, 7] {
                    let many = product_bits(level, threads, rm(&a, m, k), b.view(), beta, &c0);
                    assert!(
                        one == many,
                        "{level}, {m}x{n}x{k}, beta {beta}: {threads} threads differ from 1"
                    );
                }
            }
        }
    }
}

/// The same call twice gives the same bits, at every fast level.
#[test]
fn run_to_run_identical() {
    for (level, ..) in fast_levels("run_to_run_identical") {
        let mut rng = Rng::seed(13);
        let (m, n, k) = (200, 300, 400);
        let a = random(&mut rng, m * k);
        let b = random(&mut rng, k * n);
        let c0 = random(&mut rng, m * n);
        let first = product_bits(level, 4, rm(&a, m, k), rm(&b, k, n), 0.5, &c0);
        for _ in 0..3 {
            let again = product_bits(level, 4, rm(&a, m, k), rm(&b, k, n), 0.5, &c0);
            assert!(
                first == again,
                "{level}: a repeated call gave different bits"
            );
        }
    }
}

/// The reported thread count is the one the call runs on: never more than its tasks. Each
/// level runs its own kernel: one of its tiles is one task, two side by side are two.
#[test]
fn threads_used_matches_the_work() {
    for (level, mr, nr) in fast_levels("threads_used_matches_the_work") {
        assert_eq!(threads_used(level, 4, mr, nr, 4096), 1, "{level}: one tile");
        assert_eq!(
            threads_used(level, 4, mr, 2 * nr, 4096),
            2,
            "{level}: two tiles"
        );
        assert_eq!(threads_used(level, 4, 128, 1024, 784), 4, "{level}");
        assert_eq!(threads_used(level, 4, 0, 1024, 784), 1, "{level}");
        assert!(
            threads_used(level, 1000, 1024, 1024, 1024) <= 256,
            "{level}"
        );
    }
}

/// When a block of the product has more tasks than threads, the compute grid has a whole
/// number of tasks per thread (checked on 2, 3, 4 and 8 threads), so no thread runs a last
/// task alone while the others wait.
/// The shapes are the products of the 784 → 1024 → 256 → 10 and 64 → 256 → 128 → 10
/// networks at batch 32 and 128 (each after the driver's swap), in 6 by 8 tiles with
/// 96-row tasks, as the SSE4.2 kernel blocks them.
#[test]
fn grid_gives_every_thread_the_same_number_of_tasks() {
    let shapes = [
        (1024, 784),
        (256, 1024),
        (1024, 256),
        (10, 256),
        (128, 1024),
        (32, 1024),
        (256, 64),
        (128, 256),
        (128, 128),
    ];
    for threads in [2, 3, 4, 8] {
        for (m, n) in shapes {
            let tasks = Grid::new(m, n, 6, 8, 96, threads).tasks();
            assert!(
                tasks <= threads || tasks.is_multiple_of(threads),
                "{m} by {n} on {threads} threads: {tasks} tasks"
            );
        }
    }
}

/// On a CPU without AVX2, `CAVITATION_KERNEL=avx2` makes the first kernel call panic: the
/// test binary re-runs itself as a child with the variable set, and the child fails with a
/// message naming `avx2` and the detected features.
#[test]
fn forced_unsupported_level_panics_cleanly() {
    if supported(Level::Avx2Fma) {
        println!("forced_unsupported_level_panics_cleanly: skipped, this CPU supports avx2");
        return;
    }
    let out = crate::test_child::run_alone_without(
        "kernels::twin_tests::forced_level_child",
        &[("CAVITATION_KERNEL", "avx2")],
        &["CAVITATION_THREADS"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    // A filter that matches nothing also exits 0; "1 failed" shows the helper ran.
    assert!(
        !out.status.success() && stdout.contains("1 failed"),
        "the child should run the helper and fail; stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let features = format!("detected features: {}", super::detected_features());
    assert!(
        stderr.contains("cannot run kernel level avx2") && stderr.contains(&features),
        "stderr:\n{stderr}"
    );
}

/// Helper for `forced_unsupported_level_panics_cleanly`: one product at the level the
/// environment gives. Ignored by default; it is run in a child process.
#[test]
#[ignore = "run in a child process by forced_unsupported_level_panics_cleanly"]
fn forced_level_child() {
    let (a, b, mut c) = ([1.0f32; 4], [1.0f32; 4], [0.0f32; 4]);
    gemm_f32(1.0, rm(&a, 2, 2), rm(&b, 2, 2), 0.0, rm_mut(&mut c, 2, 2));
    assert_eq!(c, [2.0; 4]);
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

/// Every level this CPU runs: the reference loops and each supported fast level. Prints each
/// fast level it skips.
fn every_level(test: &str) -> Vec<Level> {
    std::iter::once(Level::Reference)
        .chain(fast_levels(test).into_iter().map(|(level, ..)| level))
        .collect()
}

/// The matvec tolerance for each output `j` of `bias + x · w_t` (`w_t` row-major, `k` by
/// `n`), computed in f64: `2·(k + 2)·1.19e-7·(Σᵢ |xᵢ|·|wᵢⱼ| + |biasⱼ|) + 1e-30`.
fn matvec_bounds(w_t: &[f32], x: &[f32], bias: Option<&[f32]>, n: usize) -> Vec<f64> {
    let k = x.len();
    (0..n)
        .map(|j| {
            let s: f64 = (0..k)
                .map(|i| f64::from(x[i]).abs() * f64::from(w_t[i * n + j]).abs())
                .sum();
            let b = bias.map_or(0.0, |b| f64::from(b[j]).abs());
            2.0 * (k as f64 + 2.0) * 1.19e-7 * (s + b) + 1e-30
        })
        .collect()
}

/// The fast matvec at every supported level matches the reference (followed by ReLU when
/// asked) within the matvec tolerance, and the SSE4.2 one bit for bit, for input widths 1, 3, 7, 64, 785 and 1023 by output
/// widths 1, 5, 10, 256 and 1024, with and without a bias and ReLU. The weights are stored
/// with 3 NaN values of padding after each row, so reading past a row shows up as NaN; the
/// output starts as NaN, so an element left unwritten shows up too.
#[test]
fn matvec_levels_match_reference() {
    let mut rng = Rng::seed(17);
    for level in every_level("matvec_levels_match_reference") {
        for k in [1usize, 3, 7, 64, 785, 1023] {
            for n in [1usize, 5, 10, 256, 1024] {
                let w = random(&mut rng, k * n);
                let x = random(&mut rng, k);
                let b = random(&mut rng, n);
                let stored = Stored::new(&w, k, n, false, 3);
                for (bias, relu) in [
                    (None, false),
                    (Some(&b[..]), false),
                    (Some(&b[..]), true),
                    (None, true),
                ] {
                    let mut want = vec![0.0f32; n];
                    reference::matvec(rm(&w, k, n), &x, bias, &mut want);
                    if relu {
                        for v in &mut want {
                            *v = if *v <= 0.0 { 0.0 } else { *v };
                        }
                    }
                    let mut got = vec![f32::NAN; n];
                    matvec_f32_with(level, stored.view(), &x, bias, relu, &mut got);
                    let bound = matvec_bounds(&w, &x, bias, n);
                    let what = format!(
                        "{level}: matvec {k} -> {n}, bias {}, relu {relu}",
                        bias.is_some()
                    );
                    assert_within(&got, &want, &bound, n, &what);
                    if matches!(level, Level::Reference | Level::Sse42) {
                        // A separate multiply and add in the reference's order: the same bits.
                        let same = got
                            .iter()
                            .zip(&want)
                            .all(|(g, w)| g.to_bits() == w.to_bits());
                        assert!(same, "{what}: not bit-identical to the reference");
                    }
                }
            }
        }
    }
}

/// A weight matrix whose columns are not contiguous (a transposed view, and general
/// strides) still gives the reference's answer, at every supported level.
#[test]
fn matvec_strided_weights_match_reference() {
    let mut rng = Rng::seed(19);
    let (k, n) = (37, 21);
    let w = random(&mut rng, k * n);
    let x = random(&mut rng, k);
    let b = random(&mut rng, n);
    let mut want = vec![0.0f32; n];
    reference::matvec(rm(&w, k, n), &x, Some(&b), &mut want);
    let bound = matvec_bounds(&w, &x, Some(&b), n);
    for level in every_level("matvec_strided_weights_match_reference") {
        for stored in [Stored::new(&w, k, n, true, 2), Stored::strided(&w, k, n)] {
            let mut got = vec![f32::NAN; n];
            matvec_f32_with(level, stored.view(), &x, Some(&b), false, &mut got);
            assert_within(&got, &want, &bound, n, &format!("{level}: strided matvec"));
        }
    }
}

/// The fused ReLU keeps NaN and turns negative values into +0, and an empty input
/// gives the bias (then ReLU), at every supported level; the same call twice gives the same
/// bits.
#[test]
fn matvec_relu_edges_and_repeats() {
    for level in every_level("matvec_relu_edges_and_repeats") {
        let w = [1.0f32, -1.0, f32::NAN, 0.0, 2.0, -0.0, 1.0, 0.5, 3.0];
        let x = [1.0f32, 0.0, 0.0];
        let mut out = [9.0f32; 3];
        // Row 1 and row 2 are multiplied by 0; the NaN in row 0 reaches output 2.
        matvec_f32_with(level, rm(&w, 3, 3), &x, None, true, &mut out);
        assert_eq!(out[0].to_bits(), 1.0f32.to_bits(), "{level}");
        assert_eq!(out[1].to_bits(), 0.0f32.to_bits(), "{level}");
        assert!(out[2].is_nan(), "{level}");
        let mut empty = [9.0f32; 2];
        let empty_w = MatRef::new(&[], 0, 2, 2, 1).unwrap();
        matvec_f32_with(level, empty_w, &[], Some(&[-0.5, 0.5]), true, &mut empty);
        assert_eq!(empty, [0.0, 0.5], "{level}");
        let mut rng = Rng::seed(23);
        let (k, n) = (300, 77);
        let w = random(&mut rng, k * n);
        let x = random(&mut rng, k);
        let (mut first, mut again) = (vec![0.0f32; n], vec![0.0f32; n]);
        matvec_f32_with(level, rm(&w, k, n), &x, None, false, &mut first);
        matvec_f32_with(level, rm(&w, k, n), &x, None, false, &mut again);
        assert!(
            first
                .iter()
                .zip(&again)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{level}"
        );
    }
}

/// `matvec_f32` at the current level matches a hand-worked product.
#[test]
fn matvec_f32_uses_a_supported_level() {
    let w_t = [1.0f32, 2.0, 3.0, 4.0];
    let mut out = [f32::NAN; 2];
    matvec_f32(
        rm(&w_t, 2, 2),
        &[1.0, 1.0],
        Some(&[0.5, -10.0]),
        true,
        &mut out,
    );
    assert_eq!(out, [4.5, 0.0]);
}

/// Asking the matvec for a level this CPU cannot run panics, naming the level.
#[test]
#[should_panic(expected = "cannot run kernel level")]
fn matvec_with_unsupported_level_panics() {
    let foreign = if cfg!(target_arch = "aarch64") {
        Level::Avx2Fma
    } else {
        Level::Neon
    };
    let mut out = [0.0f32; 2];
    matvec_f32_with(
        foreign,
        rm(&[1.0; 4], 2, 2),
        &[1.0, 1.0],
        None,
        false,
        &mut out,
    );
}

/// A wrong input length panics with the reference's message, at a fast level too.
#[test]
#[should_panic(expected = "input has 3 values")]
fn matvec_with_wrong_input_length_panics() {
    let mut out = [0.0f32; 2];
    matvec_f32(rm(&[1.0; 4], 2, 2), &[1.0; 3], None, false, &mut out);
}
