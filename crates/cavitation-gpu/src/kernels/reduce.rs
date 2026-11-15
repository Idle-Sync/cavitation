//! Per-channel kernels on the GPU over tensors laid out `[outer][channels][inner]`: a Linear
//! layer's `M × N` output is `outer = M`, `inner = 1`, a convolution's NCHW output is
//! `outer = batch`, `inner = h · w`. [`BiasAdd`] adds a bias per channel, with the CPU's bits
//! on normal values;
//! [`ChannelSums`] sums each channel over the other two axes in a fixed tree, for a bias's
//! gradient and batch norm's statistics, within the bound of a sum in any order. Both only
//! add, so neither assumes anything about fused multiply-adds.

use crate::{
    BufferPool, Gpu, GpuError, MAX_WORKGROUPS_PER_DIMENSION, PipelineCache, PooledBuffer, Shader,
    Step, StepEncoder,
};

/// Threads per workgroup in every kernel here, passed to them as the `WORKGROUP` override.
/// The sums' workgroup tree halves it, so it must be a power of two.
const WORKGROUP: u32 = 256;

/// Values of one channel each lane of [`ChannelSums`]' first pass adds, one workgroup's width
/// apart, passed as the `PER_LANE` override: a chunk is [`WORKGROUP`] · `PER_LANE` values.
const PER_LANE: u32 = 16;

/// The values one workgroup of [`ChannelSums`]' first pass sums into one partial sum.
const CHUNK: u64 = (WORKGROUP * PER_LANE) as u64;

/// The most values a tensor may hold: the kernels index them with `u32`s, with room for a
/// workgroup's lanes past the end.
const MAX_VALUES: u64 = 1 << 31;

/// Bytes of the `Dims` uniform: `outer`, `channels`, `inner`, the chunks per channel, whether
/// the sums add to the output, where they go in it, what they are divided by, and where the
/// summed values start in the input.
const DIMS_BYTES: u64 = 32;

/// The parts every kernel here shares: the `WORKGROUP` override and the `Dims` uniform.
macro_rules! dims {
    () => {
        "
        override WORKGROUP: u32;

        struct Dims {
            outer: u32,
            channels: u32,
            inner: u32,
            chunks: u32,
            accumulate: u32,
            offset: u32,
            divisor: f32,
            start: u32,
        }
        "
    };
}
pub(crate) use dims;

/// Bias add, one thread per value of `out`, numbered over `dispatch_split`'s grid:
/// `y[(b·C + c)·P + p] += bias[c]`.
const BIAS_ADD: Shader = Shader {
    name: "bias add",
    source: concat!(
        dims!(),
        "
        @group(0) @binding(0) var<storage, read> bias: array<f32>;
        @group(0) @binding(1) var<storage, read_write> y: array<f32>;
        @group(0) @binding(2) var<uniform> dims: Dims;

        @compute @workgroup_size(WORKGROUP)
        fn main(
            @builtin(workgroup_id) group: vec3<u32>,
            @builtin(num_workgroups) groups: vec3<u32>,
            @builtin(local_invocation_index) lane: u32,
        ) {
            let i = (group.x + group.y * groups.x) * WORKGROUP + lane;
            if (i >= dims.outer * dims.channels * dims.inner) { return; }
            y[i] = y[i] + bias[(i / dims.inner) % dims.channels];
        }
        "
    ),
};

/// The first pass of a per-channel sum, around a kernel's own bindings and its
/// `fn term(i: u32, c: u32) -> f32`, the value summed for value `i` of channel `c`:
/// workgroup `(chunk, c)` sums the terms of values `chunk · CHUNK` onwards of channel `c`,
/// taken in the order `j = b·P + p`, the tensor starting `dims.start` values into its inputs.
/// Each lane adds every `WORKGROUP`-th term of the chunk in order, then the lanes' sums meet
/// in a fixed halving tree. The partial sums are binding 1 and the `Dims` uniform binding 2;
/// the term's inputs take binding 0 and from 3 on.
macro_rules! partials_pass {
    ($bindings:literal, $term:literal) => {
        concat!(
            $crate::kernels::reduce::dims!(),
            "
            override PER_LANE: u32;

            @group(0) @binding(1) var<storage, read_write> partials: array<f32>;
            @group(0) @binding(2) var<uniform> dims: Dims;
            ",
            $bindings,
            $term,
            "
            var<workgroup> lanes: array<f32, WORKGROUP>;

            @compute @workgroup_size(WORKGROUP)
            fn main(
                @builtin(workgroup_id) group: vec3<u32>,
                @builtin(local_invocation_index) lane: u32,
            ) {
                let chunk = group.x;
                let c = group.y;
                let len = dims.outer * dims.inner;
                let first = chunk * WORKGROUP * PER_LANE + lane;
                var acc = 0.0;
                for (var k = 0u; k < PER_LANE; k = k + 1u) {
                    let j = first + k * WORKGROUP;
                    if (j < len) {
                        let b = j / dims.inner;
                        let i = (b * dims.channels + c) * dims.inner + j % dims.inner;
                        acc = acc + term(dims.start + i, c);
                    }
                }
                lanes[lane] = acc;
                workgroupBarrier();
                for (var width = WORKGROUP / 2u; width > 0u; width = width / 2u) {
                    if (lane < width) { lanes[lane] = lanes[lane] + lanes[lane + width]; }
                    workgroupBarrier();
                }
                if (lane == 0u) { partials[c * dims.chunks + chunk] = lanes[0]; }
            }
            "
        )
    };
}
pub(crate) use partials_pass;

/// Channel sums, first pass: the terms are the values themselves.
pub(crate) const CHANNEL_PARTIALS: Shader = Shader {
    name: "channel sums partials",
    source: partials_pass!(
        "@group(0) @binding(0) var<storage, read> x: array<f32>;",
        "fn term(i: u32, c: u32) -> f32 { return x[i]; }"
    ),
};

/// Channel sums, second pass: one thread per channel (at most 65,535, so one grid row) adds
/// its chunks' partial sums in chunk order, divides by the divisor when there is one (not
/// 0), then, when accumulating, adds the output's old value; channel `c` goes to
/// `sums[offset + c]`.
const CHANNEL_MERGE: Shader = Shader {
    name: "channel sums merge",
    source: concat!(
        dims!(),
        "
        @group(0) @binding(0) var<storage, read> partials: array<f32>;
        @group(0) @binding(1) var<storage, read_write> sums: array<f32>;
        @group(0) @binding(2) var<uniform> dims: Dims;

        @compute @workgroup_size(WORKGROUP)
        fn main(
            @builtin(workgroup_id) group: vec3<u32>,
            @builtin(local_invocation_index) lane: u32,
        ) {
            let c = group.x * WORKGROUP + lane;
            if (c >= dims.channels) { return; }
            let base = c * dims.chunks;
            var acc = partials[base];
            for (var k = 1u; k < dims.chunks; k = k + 1u) {
                acc = acc + partials[base + k];
            }
            if (dims.divisor != 0.0) { acc = acc / dims.divisor; }
            let at = dims.offset + c;
            if (dims.accumulate != 0u) { acc = acc + sums[at]; }
            sums[at] = acc;
        }
        "
    ),
};

/// The shape of a tensor the per-channel kernels work on: `outer` blocks of `channels`
/// channels of `inner` values each, `[outer][channels][inner]` in row-major order.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::kernels::ChannelDims;
///
/// let conv = ChannelDims::new(128, 32, 32 * 32); // NCHW: batch, channels, h · w
/// assert_eq!((conv.len(), conv.summed()), (128 * 32 * 1024, 128 * 1024));
/// let linear = ChannelDims::rows(128, 10); // an M × N matrix: one channel per column
/// assert_eq!(linear, ChannelDims::new(128, 10, 1));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelDims {
    /// The blocks of channels: the batch, or a matrix's rows.
    pub outer: u32,
    /// The channels: a convolution's channels, or a matrix's columns.
    pub channels: u32,
    /// The values of one channel in one block: an image plane's `h · w`, or 1 for a matrix.
    pub inner: u32,
}

impl ChannelDims {
    /// `outer` blocks of `channels` channels of `inner` values.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::ChannelDims;
    /// assert_eq!(ChannelDims::new(2, 3, 4).inner, 4);
    /// ```
    pub fn new(outer: u32, channels: u32, inner: u32) -> ChannelDims {
        ChannelDims {
            outer,
            channels,
            inner,
        }
    }

    /// A row-major `m × n` matrix, each column a channel: `new(m, n, 1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::ChannelDims;
    /// assert_eq!(ChannelDims::rows(5, 7).summed(), 5);
    /// ```
    pub fn rows(m: u32, n: u32) -> ChannelDims {
        ChannelDims::new(m, n, 1)
    }

    /// The values of the whole tensor: `outer · channels · inner`, or `u64::MAX` for a shape
    /// too large to count in a `u64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::ChannelDims;
    /// assert_eq!(ChannelDims::new(2, 3, 4).len(), 24);
    /// ```
    pub fn len(&self) -> u64 {
        u64::try_from(self.len_wide()).unwrap_or(u64::MAX)
    }

    /// The values of the whole tensor, without overflow.
    fn len_wide(&self) -> u128 {
        u128::from(self.outer) * u128::from(self.channels) * u128::from(self.inner)
    }

    /// Whether the tensor holds no values.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::ChannelDims;
    /// assert!(ChannelDims::new(0, 3, 4).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The values summed into each channel's sum: `outer · inner`, the length `L` the
    /// sums' bound counts.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::ChannelDims;
    /// assert_eq!(ChannelDims::new(2, 3, 4).summed(), 8);
    /// ```
    pub fn summed(&self) -> u64 {
        u64::from(self.outer) * u64::from(self.inner)
    }

    /// Panics, naming `what` and the dims, unless every dimension is above 0 and the tensor
    /// fits the kernels' `u32` indices.
    pub(crate) fn check(&self, what: &str) {
        assert!(
            !self.is_empty(),
            "{what}: every dimension must be above 0, got {self:?}"
        );
        assert!(
            self.len_wide() <= u128::from(MAX_VALUES),
            "{what}: {self:?} holds {} values, more than the {MAX_VALUES} the kernels take",
            self.len_wide()
        );
    }

    /// The `Dims` uniform's bytes, with `chunks` chunks per channel, the input's `start` and
    /// where the sums go, `to`.
    fn uniform(&self, chunks: u32, start: u32, to: &SumsTo<'_>) -> [u8; DIMS_BYTES as usize] {
        let mut bytes = [0u8; DIMS_BYTES as usize];
        for (slot, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip([
            self.outer,
            self.channels,
            self.inner,
            chunks,
            u32::from(to.accumulate),
            to.offset,
            to.divisor.unwrap_or(0.0).to_bits(),
            start,
        ]) {
            slot.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// A bind group layout entry for a compute storage buffer at `binding`.
fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// The layout every kernel here takes: a read-only storage buffer, a read-write one, and the
/// `Dims` uniform, at bindings 0, 1 and 2.
fn layout(gpu: &Gpu, label: &str) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
    layout_with(gpu, label, 0)
}

/// [`layout`] with `extra` more read-only storage buffers, at bindings 3 onwards.
fn layout_with(
    gpu: &Gpu,
    label: &str,
    extra: u32,
) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
    let device = gpu.device();
    let mut entries = vec![
        storage(0, true),
        storage(1, false),
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(DIMS_BYTES),
            },
            count: None,
        },
    ];
    entries.extend((0..extra).map(|k| storage(3 + k, true)));
    let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    });
    let pipeline = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&group)],
        immediate_size: 0,
    });
    (group, pipeline)
}

/// A `Dims` uniform from `pool` holding `bytes`, uploaded in a step of its own.
fn upload_dims(
    steps: &mut StepEncoder,
    pool: &mut BufferPool,
    bytes: &[u8; DIMS_BYTES as usize],
) -> PooledBuffer {
    let params = pool.take(
        DIMS_BYTES,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let mut step = steps.begin();
    step.upload(params.buffer(), 0, bytes);
    let _ = step.finish();
    params
}

/// A counted bind group on `layout` with `input`, `output` and the uniform `params` at
/// bindings 0, 1 and 2.
fn bind(
    steps: &mut StepEncoder,
    layout: &wgpu::BindGroupLayout,
    input: &PooledBuffer,
    output: &PooledBuffer,
    params: &PooledBuffer,
) -> wgpu::BindGroup {
    bind_with(steps, layout, input, output, params, &[])
}

/// [`bind`] with `extra` read-only buffers at bindings 3 onwards.
fn bind_with(
    steps: &mut StepEncoder,
    layout: &wgpu::BindGroupLayout,
    input: &PooledBuffer,
    output: &PooledBuffer,
    params: &PooledBuffer,
    extra: &[&PooledBuffer],
) -> wgpu::BindGroup {
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: input.binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: output.binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: params.binding(),
        },
    ];
    entries.extend(extra.iter().enumerate().map(|(k, b)| wgpu::BindGroupEntry {
        binding: 3 + k as u32,
        resource: b.binding(),
    }));
    steps.bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &entries,
    })
}

/// Panics, naming `what`, unless `buffer` holds at least `values` `f32`s.
fn check_len(what: &str, buffer: &PooledBuffer, values: u64) {
    assert!(
        buffer.len() >= 4 * values,
        "{what}: needs {values} f32s ({} bytes), but its buffer has {} bytes",
        4 * values,
        buffer.len()
    );
}

/// Adds a bias per channel to a tensor in place, on the GPU: `out[b][c][p] += bias[c]` for
/// a tensor laid out `[outer][channels][inner]` ([`ChannelDims`]). Each value takes one
/// correctly rounded addition, as the CPU's broadcast `add` does, so on normal values and
/// sums the result is the CPU's, bit for bit; a subnormal is flushed.
///
/// [`call`](BiasAdd::call) prepares one bias add on given buffers, made once when a model is
/// compiled, and [`record`](BiasAdd::record) adds it to a step.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_gpu::kernels::{BiasAdd, ChannelDims};
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let bias_add = BiasAdd::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
///     let mut pool = BufferPool::new(&gpu);
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     let (bias, out) = (pool.take(8, usage), pool.take(16, usage));
///     let mut steps = StepEncoder::new(&gpu, StepSizes { upload_bytes: 64, readback_bytes: 16, ..StepSizes::default() });
///     let call = bias_add.call(&mut steps, &mut pool, ChannelDims::rows(2, 2), &bias, &out);
///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     let mut step = steps.begin();
///     step.upload(bias.buffer(), 0, &bytes(&[10.0, 20.0]));
///     step.upload(out.buffer(), 0, &bytes(&[1.0, 2.0, 3.0, 4.0]));
///     bias_add.record(&mut step, &call);
///     step.read_back(out.buffer(), 0, 16);
///     assert_eq!(step.finish().bytes(), &bytes(&[11.0, 22.0, 13.0, 24.0])[..]);
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct BiasAdd {
    /// The layout of group 0: the bias, the tensor and the `Dims` uniform.
    group: wgpu::BindGroupLayout,
    /// The kernel.
    pipeline: wgpu::ComputePipeline,
}

impl BiasAdd {
    /// Builds the kernel on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if the kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::BiasAdd;
    /// use cavitation_gpu::{Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut cache = PipelineCache::new(&gpu);
    ///     BiasAdd::new(&gpu, &mut cache).unwrap();
    ///     assert_eq!(cache.pipelines_built(), 1);
    /// }
    /// ```
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<BiasAdd, GpuError> {
        let (group, layout) = layout(gpu, "bias add");
        let constants = [("WORKGROUP", f64::from(WORKGROUP))];
        let pipeline = cache.get_with_layout(BIAS_ADD, "main", &constants, &layout)?;
        Ok(BiasAdd { group, pipeline })
    }

    /// Prepares one bias add of `bias` (one value per channel) into `out` (shaped `dims`):
    /// its `Dims` uniform, taken from `pool` and uploaded in a step of its own, and its bind
    /// group, made through `steps` so it is counted. Made once, when a model is compiled.
    ///
    /// # Panics
    ///
    /// If a dimension is 0, the tensor holds more than 2³¹ values, a buffer is shorter than
    /// its values, or `bias` and `out` are the same buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::{BiasAdd, ChannelDims};
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let bias_add = BiasAdd::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let (bias, out) = (pool.take(4 * 32, wgpu::BufferUsages::STORAGE), pool.take(4 * 32 * 64, wgpu::BufferUsages::STORAGE));
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let call = bias_add.call(&mut steps, &mut pool, ChannelDims::new(1, 32, 64), &bias, &out);
    ///     assert_eq!(call.dims(), ChannelDims::new(1, 32, 64));
    ///     call.release(&mut pool);
    /// }
    /// ```
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        dims: ChannelDims,
        bias: &PooledBuffer,
        out: &PooledBuffer,
    ) -> BiasAddCall {
        dims.check("bias add");
        check_len("bias add: the bias", bias, u64::from(dims.channels));
        check_len("bias add: the tensor", out, dims.len());
        assert!(
            bias.buffer() != out.buffer(),
            "bias add: the bias and the tensor it is added to must be different buffers"
        );
        let params = upload_dims(steps, pool, &dims.uniform(0, 0, &SumsTo::new(out)));
        let group = bind(steps, &self.group, bias, out, &params);
        BiasAddCall {
            dims,
            params,
            group,
        }
    }

    /// Adds the bias add `call` to `step`: one thread per value, over two grid dimensions
    /// when one is not enough.
    ///
    /// # Examples
    ///
    /// See [`BiasAdd`].
    pub fn record(&self, step: &mut Step<'_>, call: &BiasAddCall) {
        let workgroups = call.dims.len().div_ceil(u64::from(WORKGROUP)) as u32;
        step.dispatch_split(&self.pipeline, &call.group, workgroups);
    }
}

/// One prepared bias add: its shape, its `Dims` uniform and its bind group, from
/// [`BiasAdd::call`].
///
/// # Examples
///
/// See [`BiasAdd`].
#[derive(Debug)]
pub struct BiasAddCall {
    /// The tensor's shape.
    dims: ChannelDims,
    /// The `Dims` uniform, from the pool.
    params: PooledBuffer,
    /// The bind group of the bias, the tensor and the uniform.
    group: wgpu::BindGroup,
}

impl BiasAddCall {
    /// The shape this bias add was prepared for.
    ///
    /// # Examples
    ///
    /// See [`BiasAdd::call`].
    pub fn dims(&self) -> ChannelDims {
        self.dims
    }

    /// Gives this call's uniform buffer back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`BiasAdd`].
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.params);
    }
}

/// Sums each channel of a tensor laid out `[outer][channels][inner]` ([`ChannelDims`]) over
/// its `outer · inner` values, on the GPU: a bias's gradient, or the sums batch norm's
/// statistics start from. Each call either overwrites its output or adds the sums to it.
///
/// The sum is a fixed tree, the same on every run: a first pass sums each chunk of a
/// channel's values (in the order `b·inner + p`) in one workgroup, each lane in order and
/// then the lanes in a halving tree, and a second pass adds a channel's chunks in order. The
/// tree depends on the shape alone. It is not the CPU's order, so against the exact sum of
/// the same `f32` values each channel's sum of its `L = outer · inner` values is within
/// `2(L + 2) · 2⁻²⁴ · Σ|x| + (2L − 1) · 2⁻¹²⁶`. The tree passes each value through at most
/// 15 + 8 + (chunks − 1) additions, fewer than `L`, so the relative term holds as for any
/// order; the absolute term allows every input and every addition to be flushed. On integers
/// whose partial sums stay below 2²⁴ it is the CPU's sum bit for bit. Accumulating adds the
/// output's old value `o` once more, after the tree: count it as one more value, `L + 1`
/// terms with `|o|` in the sum of magnitudes.
///
/// With `inner = 1` (a matrix's columns), neighbouring lanes read values `channels` apart,
/// which costs memory bandwidth on wide matrices.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_gpu::kernels::{ChannelDims, ChannelSums};
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let sums = ChannelSums::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
///     let mut pool = BufferPool::new(&gpu);
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     let (x, out) = (pool.take(24, usage), pool.take(8, usage));
///     let mut steps = StepEncoder::new(&gpu, StepSizes { upload_bytes: 64, readback_bytes: 8, ..StepSizes::default() });
///     let call = sums.call(&mut steps, &mut pool, ChannelDims::rows(3, 2), &x, &out, false);
///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     let mut step = steps.begin();
///     step.upload(x.buffer(), 0, &bytes(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]));
///     sums.record(&mut step, &call);
///     step.read_back(out.buffer(), 0, 8);
///     assert_eq!(step.finish().bytes(), &bytes(&[6.0, 60.0])[..]);
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct ChannelSums {
    /// The two passes, summing the values themselves.
    sums: TermSums,
}

impl ChannelSums {
    /// Builds the two passes on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a pass does not build on this GPU.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::ChannelSums;
    /// use cavitation_gpu::{Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut cache = PipelineCache::new(&gpu);
    ///     ChannelSums::new(&gpu, &mut cache).unwrap();
    ///     assert_eq!(cache.pipelines_built(), 2);
    /// }
    /// ```
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<ChannelSums, GpuError> {
        Ok(ChannelSums {
            sums: TermSums::new(gpu, cache, CHANNEL_PARTIALS, 0)?,
        })
    }

    /// Prepares one sum of each channel of `x` (shaped `dims`) into `out` (one value per
    /// channel), overwriting it, or adding the sums to it when `accumulate` is set (for a
    /// gradient with several sources): its `Dims` uniform and its buffer of partial sums (one per
    /// chunk of 4,096 values of each channel), taken from `pool`, the uniform uploaded in a
    /// step of its own, and the two passes' bind groups, made through `steps` so they are
    /// counted. Made once, when a model is compiled.
    ///
    /// # Panics
    ///
    /// If a dimension is 0, the tensor holds more than 2³¹ values, there are more than
    /// 65,535 channels or more than 65,535 chunks of a channel, a buffer is shorter than its
    /// values, or `x` and `out` are the same buffer: wgpu would allow it (the passes are
    /// separate dispatches), but a tensor summed into itself is refused as a mistake.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::{ChannelDims, ChannelSums};
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let sums = ChannelSums::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let dims = ChannelDims::new(128, 32, 1024);
    ///     let x = pool.take(4 * dims.len(), wgpu::BufferUsages::STORAGE);
    ///     let out = pool.take(4 * 32, wgpu::BufferUsages::STORAGE);
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let call = sums.call(&mut steps, &mut pool, dims, &x, &out, false);
    ///     assert_eq!(call.chunks(), 32); // 131,072 values per channel
    ///     call.release(&mut pool);
    /// }
    /// ```
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        dims: ChannelDims,
        x: &PooledBuffer,
        out: &PooledBuffer,
        accumulate: bool,
    ) -> ChannelSumsCall {
        let to = SumsTo {
            accumulate,
            ..SumsTo::new(out)
        };
        self.sums
            .call(steps, pool, "channel sums", dims, (x, 0), &[], to)
    }

    /// Adds the sums `call` to `step`: the first pass's grid of chunks by channels, then the
    /// second pass's one thread per channel.
    ///
    /// # Examples
    ///
    /// See [`ChannelSums`].
    pub fn record(&self, step: &mut Step<'_>, call: &ChannelSumsCall) {
        self.sums.record(step, call);
    }
}

/// Where a per-channel sum goes: channel `c` to `out[offset + c]`, divided by `divisor` if
/// given, then overwriting the old value or (`accumulate`) added to it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SumsTo<'a> {
    /// The output buffer.
    pub(crate) out: &'a PooledBuffer,
    /// Where channel 0 goes, in values.
    pub(crate) offset: u32,
    /// Whether the sums add to the old values.
    pub(crate) accumulate: bool,
    /// What each sum is divided by, as an `f32` the host rounded, for a mean.
    pub(crate) divisor: Option<f32>,
}

impl<'a> SumsTo<'a> {
    /// The plain case: sums overwriting `out` from its start.
    pub(crate) fn new(out: &'a PooledBuffer) -> SumsTo<'a> {
        SumsTo {
            out,
            offset: 0,
            accumulate: false,
            divisor: None,
        }
    }
}

/// The two passes of a per-channel sum of a kernel's own term (built with
/// `partials_pass!`), with `extra` read-only inputs besides the one at binding 0: the
/// machinery behind [`ChannelSums`], and the sums of other kernels' terms, such as batch
/// norm's centred squares.
#[derive(Debug)]
pub(crate) struct TermSums {
    /// The first pass's layout: the first input, the partial sums, the `Dims` uniform and the
    /// extra inputs.
    group: wgpu::BindGroupLayout,
    /// The second pass's layout: the partial sums, the output and the `Dims` uniform.
    merge_group: wgpu::BindGroupLayout,
    /// The extra inputs the term reads.
    extra: u32,
    /// The first pass: one partial sum per chunk of each channel.
    partials: wgpu::ComputePipeline,
    /// The second pass: each channel's partial sums in order.
    merge: wgpu::ComputePipeline,
}

impl TermSums {
    /// Builds the passes of `shader`, a first pass from `partials_pass!` reading `extra`
    /// inputs past binding 0, and the shared second pass, on `gpu` through `cache`.
    pub(crate) fn new(
        gpu: &Gpu,
        cache: &mut PipelineCache,
        shader: Shader,
        extra: u32,
    ) -> Result<TermSums, GpuError> {
        let (merge_group, merge_layout) = layout(gpu, "channel sums merge");
        let (group, layout) = layout_with(gpu, shader.name, extra);
        let workgroup = ("WORKGROUP", f64::from(WORKGROUP));
        let partials = cache.get_with_layout(
            shader,
            "main",
            &[workgroup, ("PER_LANE", f64::from(PER_LANE))],
            &layout,
        )?;
        let merge = cache.get_with_layout(CHANNEL_MERGE, "main", &[workgroup], &merge_layout)?;
        Ok(TermSums {
            group,
            merge_group,
            extra,
            partials,
            merge,
        })
    }

    /// Prepares one sum, named `what` in panics, of each channel's terms of a tensor shaped
    /// `dims` into `to`: `first` at binding 0, the tensor starting `start` values into it,
    /// and `extra` inputs (each with the values it must hold) from binding 3, as
    /// [`ChannelSums::call`] does.
    ///
    /// # Panics
    ///
    /// As [`ChannelSums::call`], for every input (the first holding `start` values more),
    /// and if `extra` does not hold the inputs the term reads or the output is too short
    /// for its offset.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        what: &str,
        dims: ChannelDims,
        (first, start): (&PooledBuffer, u32),
        extra: &[(&PooledBuffer, u64)],
        to: SumsTo<'_>,
    ) -> ChannelSumsCall {
        let out = to.out;
        dims.check(what);
        let chunks = dims.summed().div_ceil(CHUNK);
        assert!(
            u64::from(dims.channels) <= u64::from(MAX_WORKGROUPS_PER_DIMENSION)
                && chunks <= u64::from(MAX_WORKGROUPS_PER_DIMENSION),
            "{what}: {dims:?} needs a grid of {chunks} chunks by {} channels, more than \
             {MAX_WORKGROUPS_PER_DIMENSION} in a dimension",
            dims.channels
        );
        assert!(
            extra.len() == self.extra as usize,
            "{what}: the term reads {} more inputs, got {}",
            self.extra,
            extra.len()
        );
        check_len(
            &format!("{what}: the tensor"),
            first,
            u64::from(start) + dims.len(),
        );
        for (k, (buffer, len)) in extra.iter().enumerate() {
            check_len(&format!("{what}: input {}", k + 1), buffer, *len);
        }
        check_len(
            &format!("{what}: the output"),
            out,
            u64::from(to.offset) + u64::from(dims.channels),
        );
        assert!(
            first.buffer() != out.buffer(),
            "{what}: the tensor and its sums must be different buffers"
        );
        for (k, (buffer, _)) in extra.iter().enumerate() {
            assert!(
                buffer.buffer() != out.buffer(),
                "{what}: input {} and the sums must be different buffers",
                k + 1
            );
        }
        let chunks = chunks as u32;
        let params = upload_dims(steps, pool, &dims.uniform(chunks, start, &to));
        let partials = pool.take(
            4 * u64::from(chunks) * u64::from(dims.channels),
            wgpu::BufferUsages::STORAGE,
        );
        let extras: Vec<&PooledBuffer> = extra.iter().map(|e| e.0).collect();
        let partials_group = bind_with(steps, &self.group, first, &partials, &params, &extras);
        let merge_group = bind(steps, &self.merge_group, &partials, out, &params);
        ChannelSumsCall {
            dims,
            chunks,
            params,
            partials,
            partials_group,
            merge_group,
        }
    }

    /// Adds the sums `call` to `step`: the first pass's grid of chunks by channels, then the
    /// second pass's one thread per channel.
    pub(crate) fn record(&self, step: &mut Step<'_>, call: &ChannelSumsCall) {
        step.dispatch_grid(
            &self.partials,
            &call.partials_group,
            [call.chunks, call.dims.channels, 1],
        );
        step.dispatch(
            &self.merge,
            &call.merge_group,
            call.dims.channels.div_ceil(WORKGROUP),
        );
    }
}

/// One prepared sum: its shape, its chunks per channel, its `Dims` uniform, its partial sums
/// and the two passes' bind groups, from [`ChannelSums::call`] (or another kernel's sum of
/// its own terms).
///
/// # Examples
///
/// See [`ChannelSums`].
#[derive(Debug)]
pub struct ChannelSumsCall {
    /// The tensor's shape.
    dims: ChannelDims,
    /// Chunks of [`CHUNK`] values per channel.
    chunks: u32,
    /// The `Dims` uniform, from the pool.
    params: PooledBuffer,
    /// One partial sum per chunk of each channel, channel by channel.
    partials: PooledBuffer,
    /// The first pass's bind group: the tensor, the partial sums and the uniform.
    partials_group: wgpu::BindGroup,
    /// The second pass's bind group: the partial sums, the output and the uniform.
    merge_group: wgpu::BindGroup,
}

impl ChannelSumsCall {
    /// The shape these sums were prepared for.
    ///
    /// # Examples
    ///
    /// See [`ChannelSums::call`].
    pub fn dims(&self) -> ChannelDims {
        self.dims
    }

    /// Chunks of 4,096 values each channel's sum is split into: one partial sum each.
    ///
    /// # Examples
    ///
    /// See [`ChannelSums::call`].
    pub fn chunks(&self) -> u32 {
        self.chunks
    }

    /// Gives this call's uniform and partial sums back to `pool`, once it is no longer
    /// recorded.
    ///
    /// # Examples
    ///
    /// See [`ChannelSums`].
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.params);
        pool.give(self.partials);
    }
}
