//! The SIMD kernels of the direct 3 × 3 convolution ([`conv3x3_direct`](super::conv::conv3x3_direct)):
//! SSE4.2 (four lanes, multiply then add), AVX2 (eight lanes, fused multiply-add) and NEON
//! (four lanes, fused multiply-add), each over one zero-padded image.
//!
//! Every output is a sum that starts at zero and adds `x · w` over the input channels in
//! order, and within each channel over the kernel's rows and then its columns; at SSE4.2
//! each step is a multiply and then an add, at AVX2 and NEON one fused multiply-add. The
//! scalar twin ([`conv3x3_direct_twin`](super::conv::conv3x3_direct_twin)) sums in the same
//! order, so each kernel matches it bit for bit.

/// One padded image and where its outputs go: what a kernel needs for one task.
pub(super) struct Job {
    /// The image, `cin` planes of `(h + 2) × (w + 2)` with a border of zeros.
    pub(super) pad: *const f32,
    /// The weights in blocks of four output channels: `[cout / 4][cin][9][4]`.
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
/// `$splat`, `$add` and `$madd(acc, x, w)`, which returns `acc + x · w`.
macro_rules! direct_kernel {
    ($name:ident, $rows:ident, $feat:literal, $lanes:literal, $zero:expr, $load:path,
     $store:path, $splat:path, $add:path, $madd:path) => {
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
                let wb = job.weights.add(ob * job.cin * 36);
                for c in 0..job.cin {
                    let base = job.pad.add(c * plane + y * wp + x0);
                    let wc = wb.add(c * 36);
                    for ky in 0..3 {
                        let row = base.add(ky * wp);
                        for kx in 0..3 {
                            let mut v = [$zero; UR];
                            for (u, v) in v.iter_mut().enumerate() {
                                *v = $load(row.add(kx + $lanes * u));
                            }
                            let wk = wc.add((ky * 3 + kx) * 4);
                            for (o, acc) in acc.iter_mut().enumerate() {
                                let wv = $splat(*wk.add(o));
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

#[cfg(target_arch = "x86_64")]
pub(super) use x86::{direct_avx2, direct_sse};

/// The x86_64 kernels.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::Job;
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

    direct_kernel!(
        direct_sse,
        rows_sse,
        "sse4.2",
        4,
        _mm_setzero_ps(),
        _mm_loadu_ps,
        _mm_storeu_ps,
        _mm_set1_ps,
        _mm_add_ps,
        madd_sse
    );

    direct_kernel!(
        direct_avx2,
        rows_avx2,
        "avx2,fma",
        8,
        _mm256_setzero_ps(),
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_set1_ps,
        _mm256_add_ps,
        madd_avx2
    );
}

#[cfg(target_arch = "aarch64")]
pub(super) use arm::direct_neon;

/// The aarch64 kernel.
#[cfg(target_arch = "aarch64")]
mod arm {
    use super::Job;
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32,
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
        vdupq_n_f32(0.0),
        vld1q_f32,
        vst1q_f32,
        vdupq_n_f32,
        vaddq_f32,
        madd_neon
    );
}
