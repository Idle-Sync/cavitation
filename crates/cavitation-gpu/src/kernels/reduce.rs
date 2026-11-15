//! Per-channel kernels on the GPU over tensors laid out `[outer][channels][inner]`: a Linear
//! layer's `M × N` output is `outer = M`, `inner = 1`, a convolution's NCHW output is
//! `outer = batch`, `inner = h · w`. [`BiasAdd`] adds a bias per channel, exactly;
//! [`ChannelSums`] sums each channel over the other two axes in a fixed tree, for a bias's
//! gradient and batch norm's statistics, within the bound of a sum in any order. Both only
//! add, so neither assumes anything about fused multiply-adds.

use crate::{BufferPool, Gpu, GpuError, PipelineCache, PooledBuffer, Shader, Step, StepEncoder};

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

/// The most workgroups a grid dimension takes.
const MAX_GRID: u64 = 65_535;

/// Bytes of the `Dims` uniform: `outer`, `channels`, `inner` and the chunks per channel.
const DIMS_BYTES: u64 = 16;

/// The parts every kernel here shares: the `Dims` uniform's layout.
macro_rules! dims {
    () => {
        "
        override WORKGROUP: u32;

        struct Dims { outer: u32, channels: u32, inner: u32, chunks: u32 }
        "
    };
}

/// Bias add, one thread per value of `out`, numbered over `dispatch_split`'s grid:
/// `out[(b·C + c)·P + p] += bias[c]`.
const BIAS_ADD: Shader = Shader {
    name: "bias add",
    source: concat!(
        dims!(),
        "
        @group(0) @binding(0) var<storage, read> bias: array<f32>;
        @group(0) @binding(1) var<storage, read_write> out: array<f32>;
        @group(0) @binding(2) var<uniform> dims: Dims;

        @compute @workgroup_size(WORKGROUP)
        fn main(
            @builtin(workgroup_id) group: vec3<u32>,
            @builtin(num_workgroups) groups: vec3<u32>,
            @builtin(local_invocation_index) lane: u32,
        ) {
            let i = (group.x + group.y * groups.x) * WORKGROUP + lane;
            if (i >= dims.outer * dims.channels * dims.inner) { return; }
            out[i] = out[i] + bias[(i / dims.inner) % dims.channels];
        }
        "
    ),
};

/// Channel sums, first pass: workgroup `(chunk, c)` sums values `chunk · CHUNK` onwards of
/// channel `c`, taken in the order `j = b·P + p`. Each lane adds every `WORKGROUP`-th value
/// of the chunk in order, then the lanes' sums meet in a fixed halving tree.
const CHANNEL_PARTIALS: Shader = Shader {
    name: "channel sums partials",
    source: concat!(
        dims!(),
        "
        override PER_LANE: u32;

        @group(0) @binding(0) var<storage, read> x: array<f32>;
        @group(0) @binding(1) var<storage, read_write> partials: array<f32>;
        @group(0) @binding(2) var<uniform> dims: Dims;

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
                    acc = acc + x[(b * dims.channels + c) * dims.inner + j % dims.inner];
                }
            }
            lanes[lane] = acc;
            workgroupBarrier();
            for (var half = WORKGROUP / 2u; half > 0u; half = half / 2u) {
                if (lane < half) { lanes[lane] = lanes[lane] + lanes[lane + half]; }
                workgroupBarrier();
            }
            if (lane == 0u) { partials[c * dims.chunks + chunk] = lanes[0]; }
        }
        "
    ),
};

/// Channel sums, second pass: one thread per channel, numbered over `dispatch_split`'s grid,
/// adds its chunks' partial sums in chunk order.
const CHANNEL_MERGE: Shader = Shader {
    name: "channel sums merge",
    source: concat!(
        dims!(),
        "
        @group(0) @binding(0) var<storage, read> partials: array<f32>;
        @group(0) @binding(1) var<storage, read_write> out: array<f32>;
        @group(0) @binding(2) var<uniform> dims: Dims;

        @compute @workgroup_size(WORKGROUP)
        fn main(
            @builtin(workgroup_id) group: vec3<u32>,
            @builtin(num_workgroups) groups: vec3<u32>,
            @builtin(local_invocation_index) lane: u32,
        ) {
            let c = (group.x + group.y * groups.x) * WORKGROUP + lane;
            if (c >= dims.channels) { return; }
            let base = c * dims.chunks;
            var acc = partials[base];
            for (var k = 1u; k < dims.chunks; k = k + 1u) {
                acc = acc + partials[base + k];
            }
            out[c] = acc;
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
    /// ```
    /// use cavitation_gpu::kernels::ChannelDims;
    /// assert_eq!(ChannelDims::new(2, 3, 4).len(), 24);
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
    /// ```
    /// use cavitation_gpu::kernels::ChannelDims;
    /// assert_eq!(ChannelDims::rows(5, 7).summed(), 5);
    /// ```
    pub fn rows(m: u32, n: u32) -> ChannelDims {
        ChannelDims::new(m, n, 1)
    }

    /// The values of the whole tensor: `outer · channels · inner`.
    ///
    /// ```
    /// use cavitation_gpu::kernels::ChannelDims;
    /// assert_eq!(ChannelDims::new(2, 3, 4).len(), 24);
    /// ```
    pub fn len(&self) -> u64 {
        u64::from(self.outer) * u64::from(self.channels) * u64::from(self.inner)
    }

    /// Whether the tensor holds no values.
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
    /// ```
    /// use cavitation_gpu::kernels::ChannelDims;
    /// assert_eq!(ChannelDims::new(2, 3, 4).summed(), 8);
    /// ```
    pub fn summed(&self) -> u64 {
        u64::from(self.outer) * u64::from(self.inner)
    }

    /// Panics, naming `what` and the dims, unless every dimension is above 0 and the tensor
    /// fits the kernels' `u32` indices.
    fn check(&self, what: &str) {
        assert!(
            !self.is_empty(),
            "{what}: every dimension must be above 0, got {self:?}"
        );
        assert!(
            self.len() <= MAX_VALUES,
            "{what}: {self:?} holds {} values, more than the {MAX_VALUES} the kernels take",
            self.len()
        );
    }

    /// The `Dims` uniform's bytes, with `chunks` chunks per channel.
    fn uniform(&self, chunks: u32) -> [u8; DIMS_BYTES as usize] {
        let mut bytes = [0u8; DIMS_BYTES as usize];
        for (slot, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip([
            self.outer,
            self.channels,
            self.inner,
            chunks,
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
    let device = gpu.device();
    let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
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
        ],
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
    steps.bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &[
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
        ],
    })
}

/// Panics, naming `what`, unless `buffer` holds at least `values` `f32`s.
fn check_len(what: &str, buffer: &PooledBuffer, values: u64) {
    assert!(
        buffer.len() >= 4 * values,
        "{what} needs {values} f32s ({} bytes), but its buffer has {} bytes",
        4 * values,
        buffer.len()
    );
}

/// Adds a bias per channel to a tensor in place, on the GPU: `out[b][c][p] += bias[c]` for
/// a tensor laid out `[outer][channels][inner]` ([`ChannelDims`]). Each value takes one
/// correctly rounded addition, as the CPU's broadcast `add` does, so on normal values the
/// result is the CPU's, bit for bit.
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
        check_len("the bias", bias, u64::from(dims.channels));
        check_len("the tensor", out, dims.len());
        assert!(
            bias.buffer() != out.buffer(),
            "bias add: the bias and the tensor it is added to must be different buffers"
        );
        let params = upload_dims(steps, pool, &dims.uniform(0));
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
/// statistics start from.
///
/// The sum is a fixed tree, the same on every run: a first pass sums each chunk of a
/// channel's values (in the order `b·inner + p`) in one workgroup, each lane in order and
/// then the lanes in a halving tree, and a second pass adds a channel's chunks in order. The
/// tree depends on the shape alone. It is not the CPU's order, so against the exact sum of
/// the same `f32` values each channel's sum of `L = outer · inner` values is within
/// `2(L + 2) · 2⁻²⁴ · Σ|x| + 2⁻¹²⁶`, as any order of the additions is; on small integers,
/// whose sums are exact in any order, it is the CPU's sum bit for bit.
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
///     let call = sums.call(&mut steps, &mut pool, ChannelDims::rows(3, 2), &x, &out);
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
    /// The layout of group 0 of both passes: the input, the output and the `Dims` uniform.
    group: wgpu::BindGroupLayout,
    /// The first pass: one partial sum per chunk of each channel.
    partials: wgpu::ComputePipeline,
    /// The second pass: each channel's partial sums in order.
    merge: wgpu::ComputePipeline,
}

impl ChannelSums {
    /// Builds the two passes on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a pass does not build on this GPU.
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
        let (group, layout) = layout(gpu, "channel sums");
        let workgroup = ("WORKGROUP", f64::from(WORKGROUP));
        let partials = cache.get_with_layout(
            CHANNEL_PARTIALS,
            "main",
            &[workgroup, ("PER_LANE", f64::from(PER_LANE))],
            &layout,
        )?;
        let merge = cache.get_with_layout(CHANNEL_MERGE, "main", &[workgroup], &layout)?;
        Ok(ChannelSums {
            group,
            partials,
            merge,
        })
    }

    /// Prepares one sum of each channel of `x` (shaped `dims`) into `out` (one value per
    /// channel, overwritten): its `Dims` uniform and its buffer of partial sums (one per
    /// chunk of 4,096 values of each channel), taken from `pool`, the uniform uploaded in a
    /// step of its own, and the two passes' bind groups, made through `steps` so they are
    /// counted. Made once, when a model is compiled.
    ///
    /// # Panics
    ///
    /// If a dimension is 0, the tensor holds more than 2³¹ values, there are more than
    /// 65,535 channels or more than 65,535 chunks of a channel, a buffer is shorter than its
    /// values, or `x` and `out` are the same buffer.
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
    ///     let call = sums.call(&mut steps, &mut pool, dims, &x, &out);
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
    ) -> ChannelSumsCall {
        dims.check("channel sums");
        let chunks = dims.summed().div_ceil(CHUNK);
        assert!(
            u64::from(dims.channels) <= MAX_GRID && chunks <= MAX_GRID,
            "channel sums: {dims:?} needs a grid of {chunks} chunks by {} channels, more than \
             {MAX_GRID} in a dimension",
            dims.channels
        );
        check_len("the tensor", x, dims.len());
        check_len("the sums", out, u64::from(dims.channels));
        assert!(
            x.buffer() != out.buffer(),
            "channel sums: the tensor and its sums must be different buffers"
        );
        let chunks = chunks as u32;
        let params = upload_dims(steps, pool, &dims.uniform(chunks));
        let partials = pool.take(
            4 * u64::from(chunks) * u64::from(dims.channels),
            wgpu::BufferUsages::STORAGE,
        );
        let partials_group = bind(steps, &self.group, x, &partials, &params);
        let merge_group = bind(steps, &self.group, &partials, out, &params);
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
    ///
    /// # Examples
    ///
    /// See [`ChannelSums`].
    pub fn record(&self, step: &mut Step<'_>, call: &ChannelSumsCall) {
        step.dispatch_grid(
            &self.partials,
            &call.partials_group,
            [call.chunks, call.dims.channels, 1],
        );
        step.dispatch_split(
            &self.merge,
            &call.merge_group,
            call.dims.channels.div_ceil(WORKGROUP),
        );
    }
}

/// One prepared sum: its shape, its chunks per channel, its `Dims` uniform, its partial sums
/// and the two passes' bind groups, from [`ChannelSums::call`].
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
