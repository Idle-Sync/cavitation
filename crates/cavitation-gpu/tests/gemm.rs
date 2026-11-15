//! The tiled GEMM against its CPU twin on every backend this machine has: exact on
//! small-integer inputs up to k = 131,072, within `2(k + 3) · u · (Σ|a·b| + |β·c|)` of an
//! f64 product on seeded inputs for square, tall, wide, k = 1, odd and CIFAR-shaped
//! products in all four transpose combinations, with β, and the same bits on every run.
//! Each test skips without a GPU, unless `CAVITATION_GPU_REQUIRED=1` makes that a failure.

mod common;

use cavitation_gpu::PooledBuffer;
use cavitation_gpu::kernels::{Gemm, GemmDims};
use common::{Kit, U, assert_exact, assert_within, exact_values, for_each_gpu, seeded_values};

/// `A[r][j]` of an `m × k` `A` stored as `dims` says.
fn a_at(a: &[f32], d: &GemmDims, r: usize, j: usize) -> f32 {
    if d.trans_a {
        a[j * d.m as usize + r]
    } else {
        a[r * d.k as usize + j]
    }
}

/// `B[j][c]` of a `k × n` `B` stored as `dims` says.
fn b_at(b: &[f32], d: &GemmDims, j: usize, c: usize) -> f32 {
    if d.trans_b {
        b[c * d.k as usize + j]
    } else {
        b[j * d.n as usize + c]
    }
}

/// The f64 reference of `A·B + β·C` and, per output, `Σ|a·b| + |β·c|`.
fn reference(a: &[f32], b: &[f32], c: &[f32], d: &GemmDims) -> (Vec<f64>, Vec<f64>) {
    let (m, n, k) = (d.m as usize, d.n as usize, d.k as usize);
    let mut value = vec![0.0; m * n];
    let mut abs = vec![0.0; m * n];
    for r in 0..m {
        for col in 0..n {
            let (mut s, mut sa) = (0.0f64, 0.0f64);
            for j in 0..k {
                let p = f64::from(a_at(a, d, r, j)) * f64::from(b_at(b, d, j, col));
                s += p;
                sa += p.abs();
            }
            if d.beta != 0.0 {
                let bc = f64::from(d.beta) * f64::from(c[r * n + col]);
                s += bc;
                sa += bc.abs();
            }
            value[r * n + col] = s;
            abs[r * n + col] = sa;
        }
    }
    (value, abs)
}

/// Runs the GEMM `dims` on `a`, `b` and the old `c` and returns the new `C`.
fn run(kit: &mut Kit, gemm: &Gemm, dims: GemmDims, a: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
    let (ba, bb, bc) = (kit.upload(a), kit.upload(b), kit.upload(c));
    let call = gemm.call(&mut kit.steps, &mut kit.pool, dims, &ba, &bb, &bc);
    let mut step = kit.steps.begin();
    gemm.record(&mut step, &call);
    let _ = step.finish();
    let got = kit.read(&bc);
    call.release(&mut kit.pool);
    for buffer in [ba, bb, bc] {
        kit.pool.give(buffer);
    }
    got
}

/// The four transpose combinations of `dims`.
fn transposes(dims: GemmDims) -> [GemmDims; 4] {
    [(false, false), (false, true), (true, false), (true, true)].map(|(trans_a, trans_b)| {
        GemmDims {
            trans_a,
            trans_b,
            ..dims
        }
    })
}

/// Exact twins: on values from {-2, -1, 1, 2} every product and partial sum is an integer
/// below 2^24, so the GPU's result is the CPU's bit for bit, in every transpose combination,
/// with β = 1 adding the old `C`, up to k = 131,072 (the CIFAR weight gradient's length).
#[test]
fn exact_twins() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 4 * 64 * 131_072);
        let gemm = Gemm::new(gpu, &mut kit.cache).expect("build");
        let shapes = [
            GemmDims::new(64, 64, 16),
            GemmDims::new(67, 33, 129),
            GemmDims::new(1, 1, 1),
            GemmDims {
                beta: 1.0,
                ..GemmDims::new(130, 70, 50)
            },
            GemmDims::new(64, 64, 131_072),
        ];
        for (s, shape) in shapes.into_iter().enumerate() {
            let combos: Vec<GemmDims> = if shape.k > 10_000 {
                vec![
                    shape,
                    GemmDims {
                        trans_a: true,
                        ..shape
                    },
                ]
            } else {
                transposes(shape).to_vec()
            };
            for (t, dims) in combos.into_iter().enumerate() {
                let seed = 10 * s as u64 + t as u64;
                let a = exact_values(seed, dims.a_len() as usize);
                let b = exact_values(seed + 100, dims.b_len() as usize);
                let c = exact_values(seed + 200, dims.c_len() as usize);
                let (want, _) = reference(&a, &b, &c, &dims);
                let want: Vec<f32> = want.iter().map(|&w| w as f32).collect();
                let got = run(&mut kit, &gemm, dims, &a, &b, &c);
                assert_exact(&got, &want, &format!("gemm exact {dims:?}"));
            }
        }
    });
}

/// Rounding twins on seeded inputs in [-1, 1): every output within
/// `2(k + 3) · u · (Σ|a·b| + |β·c|)` of the f64 product, for square, tall, wide, k = 1 and
/// odd shapes in all four transpose combinations, and for the CIFAR network's convolutions
/// written as GEMMs at a batch of 8.
#[test]
fn rounding_twins() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 4 * 8192 * 576);
        let gemm = Gemm::new(gpu, &mut kit.cache).expect("build");
        let mut shapes: Vec<GemmDims> = Vec::new();
        for base in [
            GemmDims::new(256, 256, 256),
            GemmDims::new(1000, 16, 300),
            GemmDims::new(16, 1000, 300),
            GemmDims::new(129, 65, 1),
            GemmDims {
                beta: 0.5,
                ..GemmDims::new(77, 55, 91)
            },
        ] {
            shapes.extend(transposes(base));
        }
        // CIFAR at batch 8: the four convolutions' forward products (rows = pixels, k =
        // channels · 3 · 3), and conv2's weight gradient (k = the batch's pixels).
        shapes.extend([
            GemmDims {
                trans_b: true,
                ..GemmDims::new(8192, 32, 27)
            },
            GemmDims {
                trans_b: true,
                ..GemmDims::new(8192, 32, 288)
            },
            GemmDims {
                trans_b: true,
                ..GemmDims::new(2048, 64, 288)
            },
            GemmDims {
                trans_b: true,
                ..GemmDims::new(2048, 64, 576)
            },
            GemmDims {
                trans_a: true,
                ..GemmDims::new(32, 288, 8192)
            },
        ]);
        for (s, dims) in shapes.into_iter().enumerate() {
            let seed = 1000 + s as u64;
            let a = seeded_values(seed, dims.a_len() as usize, -1.0, 1.0);
            let b = seeded_values(seed + 1, dims.b_len() as usize, -1.0, 1.0);
            let c = seeded_values(seed + 2, dims.c_len() as usize, -1.0, 1.0);
            let (want, abs) = reference(&a, &b, &c, &dims);
            let got = run(&mut kit, &gemm, dims, &a, &b, &c);
            let k = dims.k as f64;
            let worst = assert_within(
                &got,
                &want,
                |i| 2.0 * (k + 3.0) * U * abs[i],
                &format!("gemm rounding {dims:?}"),
            );
            println!(
                "{:>5} x {:>5} x {:>5} ta={} tb={} beta={}: worst error {:.4} of its bound",
                dims.m,
                dims.n,
                dims.k,
                dims.trans_a,
                dims.trans_b,
                dims.beta,
                worst.ratio()
            );
        }
    });
}

/// With β = 0 the old `C` is never read: NaNs in it do not reach the result.
#[test]
fn beta_zero_ignores_old_c() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 16);
        let gemm = Gemm::new(gpu, &mut kit.cache).expect("build");
        let dims = GemmDims::new(20, 30, 40);
        let a = exact_values(1, dims.a_len() as usize);
        let b = exact_values(2, dims.b_len() as usize);
        let nans = vec![f32::NAN; dims.c_len() as usize];
        let (want, _) = reference(&a, &b, &nans, &dims);
        let want: Vec<f32> = want.iter().map(|&w| w as f32).collect();
        assert_exact(
            &run(&mut kit, &gemm, dims, &a, &b, &nans),
            &want,
            "beta 0 with NaN in C",
        );
    });
}

/// Two runs on the same inputs give the same bits.
#[test]
fn runs_are_identical() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 4 * 512 * 512);
        let gemm = Gemm::new(gpu, &mut kit.cache).expect("build");
        let dims = GemmDims::new(300, 200, 500);
        let a = seeded_values(5, dims.a_len() as usize, -1.0, 1.0);
        let b = seeded_values(6, dims.b_len() as usize, -1.0, 1.0);
        let c = vec![0.0; dims.c_len() as usize];
        let first = run(&mut kit, &gemm, dims, &a, &b, &c);
        let second = run(&mut kit, &gemm, dims, &a, &b, &c);
        assert_exact(&second, &first, "two runs");
    });
}

/// A buffer too small for its matrix, offsets, leading dimensions and batch included, is
/// refused when the GEMM is prepared, and the same buffers are accepted for the packed shape.
#[test]
fn short_buffers_are_refused() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 16);
        let gemm = Gemm::new(gpu, &mut kit.cache).expect("build");
        let (a, b, c) = (kit.zeros(64), kit.zeros(64), kit.zeros(64));
        let small = kit.zeros(32);
        let packed = GemmDims::new(8, 8, 8);
        gemm.call(&mut kit.steps, &mut kit.pool, packed, &a, &b, &c)
            .release(&mut kit.pool);
        let refuses = |kit: &mut Kit, dims: GemmDims, c: &PooledBuffer| {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                gemm.call(&mut kit.steps, &mut kit.pool, dims, &a, &b, c)
            }));
            assert!(result.is_err(), "{dims:?} was accepted");
        };
        refuses(&mut kit, packed, &small);
        refuses(&mut kit, GemmDims { lda: 9, ..packed }, &c);
        refuses(
            &mut kit,
            GemmDims {
                a_offset: 1,
                ..packed
            },
            &c,
        );
        refuses(&mut kit, GemmDims { ldb: 9, ..packed }, &c);
        refuses(&mut kit, GemmDims { ldc: 9, ..packed }, &c);
        // Three products with their own A and C, 64 elements apart, need 192 of each.
        let three = GemmDims {
            batch: 3,
            a_batch: 64,
            c_batch: 64,
            ..packed
        };
        refuses(&mut kit, three, &c);
        refuses(
            &mut kit,
            GemmDims {
                a_batch: 0,
                ..three
            },
            &c,
        );
    });
}

/// C's rows overlapping (`ldc` below `n`), or a batch's Cs overlapping, is refused: the
/// products' writes would race.
#[test]
fn overlapping_outputs_are_refused() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 16);
        let gemm = Gemm::new(gpu, &mut kit.cache).expect("build");
        let (a, b, c) = (kit.zeros(64), kit.zeros(64), kit.zeros(1024));
        let shared = GemmDims {
            batch: 3,
            ..GemmDims::new(8, 8, 8)
        };
        for dims in [
            GemmDims {
                ldc: 7,
                ..GemmDims::new(8, 8, 8)
            },
            shared,
            GemmDims {
                c_batch: 1,
                ..shared
            },
            GemmDims {
                c_batch: 63,
                ..shared
            },
        ] {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                gemm.call(&mut kit.steps, &mut kit.pool, dims, &a, &b, &c)
            }));
            assert!(result.is_err(), "{dims:?} was accepted");
        }
        gemm.call(
            &mut kit.steps,
            &mut kit.pool,
            GemmDims {
                c_batch: 64,
                ..shared
            },
            &a,
            &b,
            &c,
        )
        .release(&mut kit.pool);
    });
}

/// The `dims` of one packed product with the same `m`, `n`, `k`, transposes and β.
fn packed(d: &GemmDims) -> GemmDims {
    GemmDims {
        trans_a: d.trans_a,
        trans_b: d.trans_b,
        beta: d.beta,
        ..GemmDims::new(d.m, d.n, d.k)
    }
}

/// A batch of three products on strided sub-matrices: leading dimensions wider than the
/// rows, offsets, per-product strides (B shared by every product in one case), β = 1, in all
/// four transpose combinations. On exact values the whole of C's buffer must equal the CPU's:
/// the products' outputs, and every element between them left as it was.
#[test]
fn strided_batched_products_are_exact() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 22);
        let gemm = Gemm::new(gpu, &mut kit.cache).expect("build");
        let (m, n, k) = (37u32, 29u32, 45u32);
        for (t, (trans_a, trans_b)) in [(false, false), (false, true), (true, false), (true, true)]
            .into_iter()
            .enumerate()
        {
            let (a_rows, a_cols) = if trans_a { (k, m) } else { (m, k) };
            let (b_rows, b_cols) = if trans_b { (n, k) } else { (k, n) };
            let (lda, ldb, ldc) = (a_cols + 7, b_cols + 3, n + 5);
            let dims = GemmDims {
                trans_a,
                trans_b,
                beta: 1.0,
                lda,
                ldb,
                ldc,
                batch: 3,
                a_offset: 11,
                b_offset: 13,
                c_offset: 17,
                a_batch: a_rows * lda + 2,
                // The first two cases share one B; the others give each product its own.
                b_batch: if t < 2 { 0 } else { b_rows * ldb + 6 },
                c_batch: m * ldc + 9,
                ..GemmDims::new(m, n, k)
            };
            let a = exact_values(300 + t as u64, dims.a_extent() as usize);
            let b = exact_values(400 + t as u64, dims.b_extent() as usize);
            let c = exact_values(500 + t as u64, dims.c_extent() as usize + 23);
            let mut want = c.clone();
            for z in 0..dims.batch as usize {
                let a0 = dims.a_offset as usize + z * dims.a_batch as usize;
                let b0 = dims.b_offset as usize + z * dims.b_batch as usize;
                let c0 = dims.c_offset as usize + z * dims.c_batch as usize;
                let az: Vec<f32> = (0..a_rows as usize)
                    .flat_map(|r| a[a0 + r * lda as usize..][..a_cols as usize].to_vec())
                    .collect();
                let bz: Vec<f32> = (0..b_rows as usize)
                    .flat_map(|r| b[b0 + r * ldb as usize..][..b_cols as usize].to_vec())
                    .collect();
                let cz: Vec<f32> = (0..m as usize)
                    .flat_map(|r| c[c0 + r * ldc as usize..][..n as usize].to_vec())
                    .collect();
                let (value, _) = reference(&az, &bz, &cz, &packed(&dims));
                for r in 0..m as usize {
                    for col in 0..n as usize {
                        want[c0 + r * ldc as usize + col] = value[r * n as usize + col] as f32;
                    }
                }
            }
            let got = run(&mut kit, &gemm, dims, &a, &b, &c);
            assert_exact(&got, &want, &format!("strided batch {dims:?}"));
        }
    });
}

/// Writing C into the buffer A or B is read from is refused when the product is prepared.
#[test]
fn aliasing_is_refused() {
    for_each_gpu(|gpu| {
        let mut kit = Kit::new(gpu, 1 << 16);
        let gemm = Gemm::new(gpu, &mut kit.cache).expect("build");
        let dims = GemmDims::new(8, 8, 8);
        let (a, b) = (kit.zeros(64), kit.zeros(64));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gemm.call(&mut kit.steps, &mut kit.pool, dims, &a, &b, &a)
        }));
        assert!(result.is_err());
    });
}
