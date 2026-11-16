//! The health values a training step reads back, on the GPU, as the CPU health checks
//! compute them:
//!
//! - [`SumSquares`]: each parameter gradient's sum of squares, which the gradient-norm
//!   checks read.
//! - [`FiniteFlag`]: the step's finite flag, set when the loss or any gradient value is NaN or
//!   infinite, which the optimiser and the running statistics read to skip their writes.
//! - [`DeadRelu`]: one "ever non-zero" bit per ReLU neuron (per channel after a convolution),
//!   OR-ed over the batch, which the dead-ReLU check reads.
//!
//! NaN and infinity are tested by their bits, never by comparisons, which a GPU compiler may
//! fold away. Every count runs in the per-channel sums' fixed tree, so the results are the
//! same on every run.

use super::ChannelDims;
use super::bindings::{Kind, bind, check_distinct, check_len, layout, upload_uniform, word_bytes};
use super::reduce::{ChannelSumsCall, SumsTo, TermSums, partials_pass};
use crate::{BufferPool, Gpu, GpuError, PipelineCache, PooledBuffer, Shader, Step, StepEncoder};

/// Threads per workgroup of the pack kernel, passed as the `WORKGROUP` override.
const WORKGROUP: u32 = 256;

/// Bytes of the fold's and the pack's uniforms: four words each.
const UNIFORM_BYTES: u64 = 16;

/// `g²` summed per tensor.
const SQUARES: Shader = Shader {
    name: "health sum of squares",
    source: partials_pass!(
        "@group(0) @binding(0) var<storage, read> g: array<f32>;",
        "fn term(i: u32, c: u32) -> f32 { return g[i] * g[i]; }"
    ),
};

/// 1 for each value whose exponent bits are all set (NaN or ±∞), summed per tensor.
const NON_FINITE: Shader = Shader {
    name: "health non-finite count",
    source: partials_pass!(
        "@group(0) @binding(0) var<storage, read> g: array<u32>;",
        "
        fn term(i: u32, c: u32) -> f32 {
            return select(0.0, 1.0, (g[i] & 0x7f800000u) == 0x7f800000u);
        }
        "
    ),
};

/// 1 for each value whose bits other than the sign are not all 0 (anything but ±0, NaN and
/// subnormals included), summed per channel.
const NON_ZERO: Shader = Shader {
    name: "health non-zero count",
    source: partials_pass!(
        "@group(0) @binding(0) var<storage, read> y: array<u32>;",
        "
        fn term(i: u32, c: u32) -> f32 {
            return select(0.0, 1.0, (y[i] & 0x7fffffffu) != 0u);
        }
        "
    ),
};

/// The finite flag from the loss's bits and each tensor's count of non-finite values, in one
/// thread: 0 when every one is finite, 1 otherwise.
const FOLD: Shader = Shader {
    name: "health finite flag",
    source: "
        struct Fold { tensors: u32, loss_at: u32, pad0: u32, pad1: u32 }

        @group(0) @binding(0) var<storage, read_write> flag: array<u32>;
        @group(0) @binding(1) var<uniform> f: Fold;
        @group(0) @binding(2) var<storage, read> counts: array<f32>;
        @group(0) @binding(3) var<storage, read> loss: array<u32>;

        @compute @workgroup_size(1)
        fn main() {
            var bad = (loss[f.loss_at] & 0x7f800000u) == 0x7f800000u;
            for (var k = 0u; k < f.tensors; k = k + 1u) {
                bad = bad || counts[k] > 0.0;
            }
            flag[0] = select(0u, 1u, bad);
        }
    ",
};

/// The dead-ReLU bits from each channel's count of non-zero values, one thread per word:
/// bit `c % 32` of word `c / 32` is set when channel `c` had one, OR-ed into the old word
/// when accumulating.
const PACK: Shader = Shader {
    name: "health dead relu bits",
    source: "
        override WORKGROUP: u32;

        struct Pack { channels: u32, offset: u32, accumulate: u32, pad: u32 }

        @group(0) @binding(0) var<storage, read_write> bits: array<u32>;
        @group(0) @binding(1) var<uniform> p: Pack;
        @group(0) @binding(2) var<storage, read> counts: array<f32>;

        @compute @workgroup_size(WORKGROUP)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let w = id.x;
            if (w >= (p.channels + 31u) / 32u) { return; }
            var word = 0u;
            for (var b = 0u; b < 32u; b = b + 1u) {
                let c = w * 32u + b;
                if (c < p.channels && counts[c] > 0.0) { word = word | (1u << b); }
            }
            if (p.accumulate != 0u) { word = word | bits[p.offset + w]; }
            bits[p.offset + w] = word;
        }
    ",
};

/// One tensor of `len` values starting `start` values into its buffer, as the shape the
/// per-channel sums take: one channel of one block.
fn flat(what: &str, start: u32, len: u32) -> ChannelDims {
    assert!(len > 0, "{what}: a tensor must hold at least one value");
    assert!(
        u64::from(start) + u64::from(len) <= 1 << 31,
        "{what}: values {start}..{} are past the 2³¹ the kernels index",
        u64::from(start) + u64::from(len)
    );
    ChannelDims::new(1, 1, len)
}

/// The sum of squares of each parameter gradient, on the GPU: one `f32` word per tensor,
/// `Σ g²` over its values in the per-channel sums' fixed tree. The gradient-norm checks read
/// it, as they read the CPU's `GradStats`.
///
/// Against the exact sum of the same `f32` values, a tensor of `L` values is within
/// `(2(L + 2) + 1) · 2⁻²⁴ · Σ g² + (3L − 1) · 2⁻¹²⁶`, the absolute term for every square and
/// every addition flushed. On integers whose sums stay below 2²⁴ it is exact. The CPU sums in
/// `f64`; in `f32` the sum overflows to infinity once it passes `f32::MAX` (a gradient value
/// near 1.8 · 10¹⁹), and the squares of values below about 10⁻¹⁹ flush to 0, so a gradient
/// that small sums to exactly 0. The finite flag tests the values themselves, so an
/// overflowed sum leaves it clear.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_gpu::kernels::SumSquares;
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let sums = SumSquares::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
///     let mut pool = BufferPool::new(&gpu);
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     // Two gradients in one flat buffer, [3, 4] and [1, 2, 2]; their sums go to words 1 and 2.
///     let (grads, out) = (pool.take(20, usage), pool.take(12, usage));
///     let mut steps = StepEncoder::new(&gpu, StepSizes { upload_bytes: 128, readback_bytes: 16, ..StepSizes::default() });
///     let first = sums.call(&mut steps, &mut pool, (&grads, 0, 2), &out, 1);
///     let second = sums.call(&mut steps, &mut pool, (&grads, 2, 3), &out, 2);
///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     let mut step = steps.begin();
///     step.upload(grads.buffer(), 0, &bytes(&[3.0, 4.0, 1.0, 2.0, 2.0]));
///     sums.record(&mut step, &first);
///     sums.record(&mut step, &second);
///     step.read_back(out.buffer(), 4, 8);
///     assert_eq!(step.finish().bytes(), &bytes(&[25.0, 9.0])[..]);
///     first.release(&mut pool);
///     second.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct SumSquares {
    /// The two passes of the sum of `g²`.
    sums: TermSums,
}

impl SumSquares {
    /// Builds the passes on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a pass does not build on this GPU.
    ///
    /// # Examples
    ///
    /// See [`SumSquares`].
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<SumSquares, GpuError> {
        Ok(SumSquares {
            sums: TermSums::new(gpu, cache, SQUARES, 0)?,
        })
    }

    /// Prepares the sum of squares of the gradient `(buffer, start, len)`, its `len` values
    /// from value `start` of `buffer`, into word `offset` of `out` (one word, overwritten).
    /// Made once, when a model is compiled.
    ///
    /// # Panics
    ///
    /// If the tensor is empty, more than 268,431,360 values long (65,535 chunks), past the
    /// 2³¹ values the kernels index, or past the end of its buffer, if `out` is shorter than
    /// `offset + 1` words, or if `out` is the gradient's buffer.
    ///
    /// # Examples
    ///
    /// See [`SumSquares`].
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        (buffer, start, len): (&PooledBuffer, u32, u32),
        out: &PooledBuffer,
        offset: u32,
    ) -> SumSquaresCall {
        let what = "sum of squares";
        let to = SumsTo {
            offset,
            ..SumsTo::new(out)
        };
        let dims = flat(what, start, len);
        SumSquaresCall(
            self.sums
                .call(steps, pool, what, dims, (buffer, start), &[], to),
        )
    }

    /// Adds the sum `call` to `step`.
    ///
    /// # Examples
    ///
    /// See [`SumSquares`].
    pub fn record(&self, step: &mut Step<'_>, call: &SumSquaresCall) {
        self.sums.record(step, &call.0);
    }
}

/// One prepared sum of squares, from [`SumSquares::call`].
///
/// # Examples
///
/// See [`SumSquares`].
#[derive(Debug)]
pub struct SumSquaresCall(ChannelSumsCall);

impl SumSquaresCall {
    /// Gives this call's buffers back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`SumSquares`].
    pub fn release(self, pool: &mut BufferPool) {
        self.0.release(pool);
    }
}

/// The step's finite flag, on the GPU: a `u32` at word 0 of its own buffer, 0 when the loss
/// and every value of every gradient are finite and 1 when any is NaN or infinite, the rule
/// of the CPU's `OnNonFinite::Skip`. The optimiser and the running statistics read it at
/// word 0 and skip their writes when it is set.
///
/// The values are tested by their exponent bits, so the flag is exact: a subnormal or the
/// largest finite value is finite, and a gradient large enough to overflow its sum of
/// squares does not set it. Each gradient's non-finite values are counted in the fixed tree
/// of the per-channel sums (a count is exact while below 2²⁴, and never 0 once a value is
/// counted), then one thread folds the counts and the loss's bits into the flag.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_gpu::kernels::FiniteFlag;
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let check = FiniteFlag::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
///     let mut pool = BufferPool::new(&gpu);
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     let (loss, grads, flag) = (pool.take(4, usage), pool.take(12, usage), pool.take(4, usage));
///     let mut steps = StepEncoder::new(&gpu, StepSizes { upload_bytes: 128, readback_bytes: 8, ..StepSizes::default() });
///     let call = check.call(&mut steps, &mut pool, (&loss, 0), &[(&grads, 0, 3)], &flag);
///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     let mut step = steps.begin();
///     step.upload(loss.buffer(), 0, &bytes(&[0.5]));
///     step.upload(grads.buffer(), 0, &bytes(&[1.0, f32::INFINITY, 2.0]));
///     check.record(&mut step, &call);
///     step.read_back(flag.buffer(), 0, 4);
///     assert_eq!(step.finish().bytes(), &1u32.to_le_bytes()[..]);
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct FiniteFlag {
    /// The two passes of the count of non-finite values.
    counts: TermSums,
    /// The fold's layout: the flag, the uniform, the counts and the loss.
    fold_group: wgpu::BindGroupLayout,
    /// The fold.
    fold: wgpu::ComputePipeline,
}

impl FiniteFlag {
    /// Builds the passes and the fold on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// See [`FiniteFlag`].
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<FiniteFlag, GpuError> {
        use Kind::{Read, Uniform, Write};
        let (fold_group, fold_layout) = layout(
            gpu,
            "health finite flag",
            &[Write, Uniform(UNIFORM_BYTES), Read, Read],
        );
        Ok(FiniteFlag {
            counts: TermSums::new(gpu, cache, NON_FINITE, 0)?,
            fold: cache.get_with_layout(FOLD, "main", &[], &fold_layout)?,
            fold_group,
        })
    }

    /// Prepares the finite flag of the loss, the `f32` at word `loss.1` of `loss.0`, and of
    /// the `grads`, each `(buffer, start, len)` as [`SumSquares::call`] takes it, into word 0
    /// of `flag` (one `u32`, overwritten). Made once, when a model is compiled.
    ///
    /// # Panics
    ///
    /// As [`SumSquares::call`] for each gradient; if the loss's word is past its buffer, or
    /// if `flag` is shorter than a word or is the loss's buffer or a gradient's: wgpu refuses
    /// a dispatch that writes a buffer it reads.
    ///
    /// # Examples
    ///
    /// See [`FiniteFlag`].
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        loss: (&PooledBuffer, u32),
        grads: &[(&PooledBuffer, u32, u32)],
        flag: &PooledBuffer,
    ) -> FiniteFlagCall {
        let what = "finite flag";
        check_len(what, "loss", loss.0, u64::from(loss.1) + 1);
        check_len(what, "flag", flag, 1);
        check_distinct(what, ("flag", flag), &[("loss", loss.0)]);
        for (k, (buffer, _, _)) in grads.iter().enumerate() {
            assert!(
                buffer.buffer() != flag.buffer(),
                "{what}: the flag and gradient {k} must be different buffers"
            );
        }
        let shapes: Vec<ChannelDims> = grads
            .iter()
            .enumerate()
            .map(|(k, &(buffer, start, len))| {
                let dims = flat(what, start, len);
                let values = u64::from(start) + u64::from(len);
                check_len(what, &format!("gradient {k}"), buffer, values);
                dims
            })
            .collect();
        let counts = pool.take(4 * grads.len().max(1) as u64, wgpu::BufferUsages::STORAGE);
        let tensors: Vec<ChannelSumsCall> = grads
            .iter()
            .zip(shapes)
            .enumerate()
            .map(|(k, (&(buffer, start, _), dims))| {
                let to = SumsTo {
                    offset: k as u32,
                    ..SumsTo::new(&counts)
                };
                self.counts
                    .call(steps, pool, what, dims, (buffer, start), &[], to)
            })
            .collect();
        let uniform = upload_uniform(steps, pool, &word_bytes([grads.len() as u32, loss.1, 0, 0]));
        let group = bind(steps, &self.fold_group, &[flag, &uniform, &counts, loss.0]);
        FiniteFlagCall {
            tensors,
            counts,
            uniform,
            group,
        }
    }

    /// Adds the flag `call` to `step`: each gradient's count, then the fold. Record it after
    /// the kernels that write the loss and the gradients, and before those that read the
    /// flag.
    ///
    /// # Examples
    ///
    /// See [`FiniteFlag`].
    pub fn record(&self, step: &mut Step<'_>, call: &FiniteFlagCall) {
        for tensor in &call.tensors {
            self.counts.record(step, tensor);
        }
        step.dispatch(&self.fold, &call.group, 1);
    }
}

/// One prepared finite flag, from [`FiniteFlag::call`].
///
/// # Examples
///
/// See [`FiniteFlag`].
#[derive(Debug)]
pub struct FiniteFlagCall {
    /// Each gradient's count of non-finite values.
    tensors: Vec<ChannelSumsCall>,
    /// The counts, one per gradient.
    counts: PooledBuffer,
    /// The fold's uniform: the number of gradients and the loss's word.
    uniform: PooledBuffer,
    /// The fold's bind group.
    group: wgpu::BindGroup,
}

impl FiniteFlagCall {
    /// Gives this call's buffers back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`FiniteFlag`].
    pub fn release(self, pool: &mut BufferPool) {
        for tensor in self.tensors {
            tensor.release(pool);
        }
        pool.give(self.counts);
        pool.give(self.uniform);
    }
}

/// The dead-ReLU check's bits, on the GPU: for a ReLU's output laid out
/// `[outer][channels][inner]` ([`ChannelDims`]), one bit per channel, set when any of its
/// values is not ±0. A dense layer's output is `ChannelDims::rows(batch, width)`, a bit per
/// neuron. A convolution's is `(batch, channels, h · w)`, a bit per channel, as the CPU
/// counts a position of a convolution's channel dead only when its whole channel is.
///
/// The test is by bits, as the CPU's `v != 0.0`: −0 counts as 0, and NaN and subnormals as
/// non-zero. The bits are exact. Bit `c % 32` of word `c / 32` is channel `c`'s, so a layer of
/// `C` channels takes [`DeadRelu::words`]`(C)` words. Accumulating ORs them into the old
/// words, so a buffer cleared at the start of an epoch ends it holding every channel that was
/// ever non-zero.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_gpu::kernels::{ChannelDims, DeadRelu};
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let dead = DeadRelu::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
///     let mut pool = BufferPool::new(&gpu);
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     // Two rows of three neurons: neuron 1 is 0 in both, so its bit stays clear.
///     let dims = ChannelDims::rows(2, 3);
///     let (y, bits) = (pool.take(24, usage), pool.take(4, usage));
///     let mut steps = StepEncoder::new(&gpu, StepSizes { upload_bytes: 128, readback_bytes: 8, ..StepSizes::default() });
///     let call = dead.call(&mut steps, &mut pool, dims, &y, &bits, 0, false);
///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     let mut step = steps.begin();
///     step.upload(y.buffer(), 0, &bytes(&[0.5, 0.0, 0.0, 0.0, -0.0, 2.0]));
///     dead.record(&mut step, &call);
///     step.read_back(bits.buffer(), 0, 4);
///     assert_eq!(step.finish().bytes(), &0b101u32.to_le_bytes()[..]);
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct DeadRelu {
    /// The two passes of the count of non-zero values per channel.
    counts: TermSums,
    /// The pack's layout: the bits, the uniform and the counts.
    pack_group: wgpu::BindGroupLayout,
    /// The pack.
    pack: wgpu::ComputePipeline,
}

impl DeadRelu {
    /// Builds the passes and the pack on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// See [`DeadRelu`].
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<DeadRelu, GpuError> {
        use Kind::{Read, Uniform, Write};
        let (pack_group, pack_layout) = layout(
            gpu,
            "health dead relu bits",
            &[Write, Uniform(UNIFORM_BYTES), Read],
        );
        let workgroup = [("WORKGROUP", f64::from(WORKGROUP))];
        Ok(DeadRelu {
            counts: TermSums::new(gpu, cache, NON_ZERO, 0)?,
            pack: cache.get_with_layout(PACK, "main", &workgroup, &pack_layout)?,
            pack_group,
        })
    }

    /// The words the bits of `channels` channels take: one per 32.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::DeadRelu;
    /// assert_eq!([1, 32, 33].map(DeadRelu::words), [1, 1, 2]);
    /// ```
    pub fn words(channels: u32) -> u32 {
        channels.div_ceil(32)
    }

    /// Prepares the bits of the ReLU output `y` (shaped `dims`) into words `offset` onwards
    /// of `out`, [`DeadRelu::words`]`(channels)` of them, overwritten or (`accumulate`) OR-ed
    /// into. Made once, when a model is compiled.
    ///
    /// # Panics
    ///
    /// As `ChannelSums::call` for the shape; if `out` is shorter than its words or is `y`'s
    /// buffer.
    ///
    /// # Examples
    ///
    /// See [`DeadRelu`].
    #[allow(clippy::too_many_arguments)]
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        dims: ChannelDims,
        y: &PooledBuffer,
        out: &PooledBuffer,
        offset: u32,
        accumulate: bool,
    ) -> DeadReluCall {
        let what = "dead relu";
        let words = DeadRelu::words(dims.channels);
        check_len(what, "output", out, u64::from(offset) + u64::from(words));
        check_distinct(what, ("output", out), &[("ReLU output", y)]);
        dims.check(what);
        check_len(what, "ReLU output", y, dims.len());
        let counts = pool.take(4 * u64::from(dims.channels), wgpu::BufferUsages::STORAGE);
        let sums = self
            .counts
            .call(steps, pool, what, dims, (y, 0), &[], SumsTo::new(&counts));
        let uniform = upload_uniform(
            steps,
            pool,
            &word_bytes([dims.channels, offset, u32::from(accumulate), 0]),
        );
        let group = bind(steps, &self.pack_group, &[out, &uniform, &counts]);
        DeadReluCall {
            words,
            sums,
            counts,
            uniform,
            group,
        }
    }

    /// Adds the bits `call` to `step`: each channel's count, then one thread per word.
    ///
    /// # Examples
    ///
    /// See [`DeadRelu`].
    pub fn record(&self, step: &mut Step<'_>, call: &DeadReluCall) {
        self.counts.record(step, &call.sums);
        step.dispatch(&self.pack, &call.group, call.words.div_ceil(WORKGROUP));
    }
}

/// One prepared set of dead-ReLU bits, from [`DeadRelu::call`].
///
/// # Examples
///
/// See [`DeadRelu`].
#[derive(Debug)]
pub struct DeadReluCall {
    /// The words written.
    words: u32,
    /// The count of non-zero values per channel.
    sums: ChannelSumsCall,
    /// The counts, one per channel.
    counts: PooledBuffer,
    /// The pack's uniform: the channels, the offset and the accumulate flag.
    uniform: PooledBuffer,
    /// The pack's bind group.
    group: wgpu::BindGroup,
}

impl DeadReluCall {
    /// Gives this call's buffers back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`DeadRelu`].
    pub fn release(self, pool: &mut BufferPool) {
        self.sums.release(pool);
        pool.give(self.counts);
        pool.give(self.uniform);
    }
}
