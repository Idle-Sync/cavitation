//! The SIMD kernels of the direct 3 × 3 convolution ([`conv3x3_direct`](super::conv::conv3x3_direct))
//! and of its weight gradient
//! ([`conv3x3_direct_weight_grad`](super::conv::conv3x3_direct_weight_grad)): SSE4.2 (four
//! lanes, multiply then add), AVX2 (eight lanes, fused multiply-add) and NEON (four lanes,
//! fused multiply-add), each over one zero-padded image.
//!
//! Every output is a sum that starts at zero and adds `x · w` over the input channels in
//! order, and within each channel over the kernel's rows and then its columns; at SSE4.2
//! each step is a multiply and then an add, at AVX2 and NEON one fused multiply-add. The
//! scalar twin ([`conv3x3_direct_twin`](super::conv::conv3x3_direct_twin)) sums in the same
//! order, so each kernel matches it bit for bit.
//!
//! Each weight-gradient sum, for one output channel, input channel, tap and image, runs one
//! accumulator per lane: lane `i` starts at zero and adds `dy · x` over the rows in order
//! and within each row over the columns `x0 + i` for `x0 = 0, lanes, 2·lanes, …`, each step
//! as above. The lanes are then added as a tree, `(l0 + l1) + (l2 + l3)` (and for eight
//! lanes that sum plus the same of `l4..l8`), and the image's sum is added to the running
//! total, image after image.
//! [`conv3x3_direct_weight_grad_twin`](super::conv::conv3x3_direct_weight_grad_twin) sums in
//! the same order.

/// One padded image and where its outputs go: what a kernel needs for one task.
pub(super) struct Job {
    /// The image, `cin` planes of `(h + 2) × (w + 2)` with a border of zeros.
    pub(super) pad: *const f32,
    /// The weights in blocks of four output channels, each weight stored `rep` times (4 at
    /// SSE4.2, so a vector load gives it broadcast; 1 elsewhere):
    /// `[cout / 4][cin][9][4][rep]`.
    pub(super) weights: *const f32,
    /// The bias, `cout` values, added after each sum; none for the input gradient.
    pub(super) bias: Option<*const f32>,
    /// The outputs, `cout` planes of `h × w`.
    pub(super) out: *mut f32,
    /// Whether each sum is added to the value already in `out` instead of replacing it.
    pub(super) add: bool,
    /// Input channels.
    pub(super) cin: usize,
    /// Output channels, a multiple of 4.
    pub(super) cout: usize,
    /// Output (and unpadded input) height.
    pub(super) h: usize,
    /// Output (and unpadded input) width, a multiple of the kernel's lane count.
    pub(super) w: usize,
}

/// Writes the four-output-channel blocks of `job` with a kernel `$name` (and its row helper
/// `$rows`) of `$lanes` lanes under target feature `$feat`, with `$zero`, `$load`, `$store`,
/// `$splat`, `$add`, `$madd(acc, x, w)`, which returns `acc + x · w`, and `$weight(p)`,
/// which reads the weight at `p` (stored `$rep` times) as a broadcast vector.
macro_rules! direct_kernel {
    ($name:ident, $rows:ident, $feat:literal, $lanes:literal, $rep:literal, $zero:expr,
     $load:path, $store:path, $splat:path, $add:path, $madd:path, $weight:path) => {
        /// Writes every output of `job`, `UR` vectors of a row at a time while they fit and
        /// then one.
        ///
        /// # Safety
        ///
        /// `job`'s pointers must hold what [`Job`] describes, `job.w` must be a multiple of
        /// the lane count and `job.cout` of 4, the outputs must not overlap the inputs, and
        /// the CPU must support the target feature.
        #[target_feature(enable = $feat)]
        pub(in crate::kernels) unsafe fn $name(job: &Job) {
            for ob in 0..job.cout / 4 {
                for y in 0..job.h {
                    let mut x0 = 0;
                    while x0 + 4 * $lanes <= job.w {
                        // SAFETY: as this function's contract; columns x0..x0 + 4 lanes lie
                        // inside the row.
                        unsafe { $rows::<4>(job, ob, y, x0) };
                        x0 += 4 * $lanes;
                    }
                    while x0 < job.w {
                        // SAFETY: as above, one vector.
                        unsafe { $rows::<1>(job, ob, y, x0) };
                        x0 += $lanes;
                    }
                }
            }
        }

        /// Writes outputs `x0..x0 + UR · lanes` of row `y` of output channels
        /// `4·ob..4·ob + 4`.
        ///
        /// # Safety
        ///
        /// As the kernel's, with `x0 + UR · lanes <= job.w`.
        #[inline]
        #[target_feature(enable = $feat)]
        unsafe fn $rows<const UR: usize>(job: &Job, ob: usize, y: usize, x0: usize) {
            let wp = job.w + 2;
            let plane = (job.h + 2) * wp;
            let mut acc = [[$zero; UR]; 4];
            // SAFETY: every read is inside the padded image (rows y..y + 3, columns
            // x0..x0 + UR · lanes + 2 of each plane) or the weights' block `ob`; every
            // write is an output of rows `y` of channels `4·ob..4·ob + 4`, columns
            // `x0..x0 + UR · lanes`, all inside `out` by the contract.
            unsafe {
                let wb = job.weights.add(ob * job.cin * 36 * $rep);
                for c in 0..job.cin {
                    let base = job.pad.add(c * plane + y * wp + x0);
                    let wc = wb.add(c * 36 * $rep);
                    for ky in 0..3 {
                        let row = base.add(ky * wp);
                        for kx in 0..3 {
                            let mut v = [$zero; UR];
                            for (u, v) in v.iter_mut().enumerate() {
                                *v = $load(row.add(kx + $lanes * u));
                            }
                            let wk = wc.add((ky * 3 + kx) * 4 * $rep);
                            for (o, acc) in acc.iter_mut().enumerate() {
                                let wv = $weight(wk.add(o * $rep));
                                for (a, &x) in acc.iter_mut().zip(&v) {
                                    *a = $madd(*a, x, wv);
                                }
                            }
                        }
                    }
                }
                for (o, acc) in acc.iter().enumerate() {
                    let ch = ob * 4 + o;
                    let dst = job.out.add((ch * job.h + y) * job.w + x0);
                    for (u, &a) in acc.iter().enumerate() {
                        let at = dst.add($lanes * u);
                        let mut r = a;
                        if let Some(b) = job.bias {
                            r = $add(r, $splat(*b.add(ch)));
                        }
                        if job.add {
                            r = $add($load(at), r);
                        }
                        $store(at, r);
                    }
                }
            }
        }
    };
}

/// One padded input plane and the output gradient it meets: what a weight-gradient kernel
/// needs for one input channel of one image and a block of output channels.
pub(super) struct WeightJob {
    /// Input channel `c` of the image, a plane of `(h + 2) × (w + 2)` with a border of zeros.
    pub(super) pad: *const f32,
    /// The image's output gradient, `cout` planes of `h × w`.
    pub(super) dy: *const f32,
    /// The block's running totals, `[o1 − o0][cin][9]`; each image's sums are added to them.
    pub(super) out: *mut f32,
    /// The input channel.
    pub(super) c: usize,
    /// Input channels.
    pub(super) cin: usize,
    /// The block's first output channel.
    pub(super) o0: usize,
    /// One past the block's last output channel.
    pub(super) o1: usize,
    /// Output (and unpadded input) height.
    pub(super) h: usize,
    /// Output (and unpadded input) width, a multiple of the kernel's lane count.
    pub(super) w: usize,
}

/// The sum of `lanes` as a tree: each half summed the same way, then the two halves added;
/// one value is itself. Written out for the kernels' four and eight lanes, so that it
/// inlines into their loops.
#[inline(always)]
pub(super) fn tree_sum(lanes: &[f32]) -> f32 {
    match *lanes {
        [a, b, c, d] => (a + b) + (c + d),
        [a, b, c, d, e, f, g, h] => ((a + b) + (c + d)) + ((e + f) + (g + h)),
        _ => tree_sum_any(lanes),
    }
}

/// [`tree_sum`] for any number of values.
fn tree_sum_any(lanes: &[f32]) -> f32 {
    match lanes.len() {
        0 => 0.0,
        1 => lanes[0],
        n => tree_sum_any(&lanes[..n / 2]) + tree_sum_any(&lanes[n / 2..]),
    }
}

/// Adds one image's weight-gradient sums for `job` with a kernel `$name` of `$lanes` lanes
/// under target feature `$feat`, with `$zero`, `$load`, `$store` and `$madd(acc, x, w)`,
/// which returns `acc + x · w`. NEON has its own kernel, which takes three output channels
/// at a time.
#[cfg(target_arch = "x86_64")]
macro_rules! weight_grad_kernel {
    ($name:ident, $feat:literal, $lanes:literal, $zero:expr, $load:path, $store:path,
     $madd:path) => {
        /// For each output channel of `job`'s block, adds `Σ dy · x` over the image for
        /// each of the nine taps to the block's totals.
        ///
        /// # Safety
        ///
        /// `job`'s pointers must hold what [`WeightJob`] describes, `job.w` must be a
        /// multiple of the lane count, and the CPU must support the target feature.
        #[target_feature(enable = $feat)]
        pub(in crate::kernels) unsafe fn $name(job: &WeightJob) {
            let wp = job.w + 2;
            for o in job.o0..job.o1 {
                let mut acc = [$zero; 9];
                // SAFETY: every read is inside the padded plane (rows y..y + 3, columns
                // x0..x0 + lanes + 2) or plane `o` of `dy`; every write is one of the
                // block's totals for output channel `o` and input channel `c`.
                unsafe {
                    let d = job.dy.add(o * job.h * job.w);
                    for y in 0..job.h {
                        let mut x0 = 0;
                        while x0 + $lanes <= job.w {
                            let dv = $load(d.add(y * job.w + x0));
                            for ky in 0..3 {
                                let row = job.pad.add((y + ky) * wp + x0);
                                for kx in 0..3 {
                                    let t = ky * 3 + kx;
                                    acc[t] = $madd(acc[t], dv, $load(row.add(kx)));
                                }
                            }
                            x0 += $lanes;
                        }
                    }
                    let out = job.out.add(((o - job.o0) * job.cin + job.c) * 9);
                    let mut lanes = [0.0f32; $lanes];
                    for (t, &a) in acc.iter().enumerate() {
                        $store(lanes.as_mut_ptr(), a);
                        *out.add(t) += tree_sum(&lanes);
                    }
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
pub(super) use x86::{direct_avx2, direct_sse, weight_grad_avx2, weight_grad_sse};

/// The x86_64 kernels.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{Job, WeightJob, tree_sum};
    use std::arch::x86_64::{
        __m128, __m256, _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps,
        _mm_storeu_ps, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };

    /// `acc + x · w`, rounded after the multiply and after the add.
    #[inline]
    #[target_feature(enable = "sse4.2")]
    fn madd_sse(acc: __m128, x: __m128, w: __m128) -> __m128 {
        _mm_add_ps(acc, _mm_mul_ps(x, w))
    }

    /// `acc + x · w`, rounded once.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn madd_avx2(acc: __m256, x: __m256, w: __m256) -> __m256 {
        _mm256_fmadd_ps(x, w, acc)
    }

    /// The weight at `p` in all eight lanes, one broadcast load.
    ///
    /// # Safety
    ///
    /// `p` must point to a readable f32, and the CPU must support AVX2.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn weight_avx2(p: *const f32) -> __m256 {
        // SAFETY: the caller's contract.
        _mm256_set1_ps(unsafe { *p })
    }

    direct_kernel!(
        direct_sse,
        rows_sse,
        "sse4.2",
        4,
        4,
        _mm_setzero_ps(),
        _mm_loadu_ps,
        _mm_storeu_ps,
        _mm_set1_ps,
        _mm_add_ps,
        madd_sse,
        _mm_loadu_ps
    );

    weight_grad_kernel!(
        weight_grad_sse,
        "sse4.2",
        4,
        _mm_setzero_ps(),
        _mm_loadu_ps,
        _mm_storeu_ps,
        madd_sse
    );

    weight_grad_kernel!(
        weight_grad_avx2,
        "avx2,fma",
        8,
        _mm256_setzero_ps(),
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        madd_avx2
    );

    direct_kernel!(
        direct_avx2,
        rows_avx2,
        "avx2,fma",
        8,
        1,
        _mm256_setzero_ps(),
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_set1_ps,
        _mm256_add_ps,
        madd_avx2,
        weight_avx2
    );
}

#[cfg(target_arch = "aarch64")]
pub(super) use arm::{direct_neon, weight_grad_neon};

/// The aarch64 kernel.
#[cfg(target_arch = "aarch64")]
mod arm {
    use super::{Job, WeightJob, tree_sum};
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1q_dup_f32, vld1q_f32, vst1q_f32,
    };

    /// `acc + x · w`, rounded once.
    #[inline]
    #[target_feature(enable = "neon")]
    fn madd_neon(acc: float32x4_t, x: float32x4_t, w: float32x4_t) -> float32x4_t {
        vfmaq_f32(acc, x, w)
    }

    direct_kernel!(
        direct_neon,
        rows_neon,
        "neon",
        4,
        1,
        vdupq_n_f32(0.0),
        vld1q_f32,
        vst1q_f32,
        vdupq_n_f32,
        vaddq_f32,
        madd_neon,
        vld1q_dup_f32
    );

    /// For each output channel of `job`'s block, adds `Σ dy · x` over the image for each of
    /// the nine taps to the block's totals, three output channels at a time (27
    /// accumulators, so each padded-input load feeds three fused multiply-adds), then two or
    /// one. Each sum's order is the one the module describes.
    ///
    /// # Safety
    ///
    /// `job`'s pointers must hold what [`WeightJob`] describes, `job.w` must be a multiple
    /// of 4, and the CPU must support NEON.
    #[target_feature(enable = "neon")]
    pub(in crate::kernels) unsafe fn weight_grad_neon(job: &WeightJob) {
        let mut o = job.o0;
        while o < job.o1 {
            // SAFETY: as this function's contract; the match keeps o + N <= job.o1.
            unsafe {
                match job.o1 - o {
                    1 => weight_grad_neon_block::<1>(job, o),
                    2 => weight_grad_neon_block::<2>(job, o),
                    _ => weight_grad_neon_block::<3>(job, o),
                }
            }
            o += (job.o1 - o).min(3);
        }
    }

    /// [`weight_grad_neon`] for output channels `o..o + N`.
    ///
    /// # Safety
    ///
    /// As [`weight_grad_neon`], with `o + N <= job.o1`.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn weight_grad_neon_block<const N: usize>(job: &WeightJob, o: usize) {
        let (wp, hw) = (job.w + 2, job.h * job.w);
        let mut acc = [[vdupq_n_f32(0.0); 9]; N];
        // SAFETY: every read is inside the padded plane (rows y..y + 3, columns x0..x0 + 6)
        // or planes `o..o + N` of `dy`; every write is one of the block's totals for those
        // output channels and input channel `c`.
        unsafe {
            let d = job.dy.add(o * hw);
            for y in 0..job.h {
                let mut x0 = 0;
                while x0 + 4 <= job.w {
                    let mut dv = [vdupq_n_f32(0.0); N];
                    for (i, v) in dv.iter_mut().enumerate() {
                        *v = vld1q_f32(d.add(i * hw + y * job.w + x0));
                    }
                    for ky in 0..3 {
                        let row = job.pad.add((y + ky) * wp + x0);
                        for kx in 0..3 {
                            let (t, xv) = (ky * 3 + kx, vld1q_f32(row.add(kx)));
                            for i in 0..N {
                                acc[i][t] = madd_neon(acc[i][t], dv[i], xv);
                            }
                        }
                    }
                    x0 += 4;
                }
            }
            let mut lanes = [0.0f32; 4];
            for (i, sums) in acc.iter().enumerate() {
                let out = job.out.add(((o + i - job.o0) * job.cin + job.c) * 9);
                for (t, &a) in sums.iter().enumerate() {
                    vst1q_f32(lanes.as_mut_ptr(), a);
                    *out.add(t) += tree_sum(&lanes);
                }
            }
        }
    }
}
