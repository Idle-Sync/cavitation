//! The tiled GEMM on the GPU: `C = A·B + β·C` in f32, with either operand transposed, for
//! any sizes. Its CPU twin is the f64 product of the same f32 inputs; every output is within
//! `2(k + 3) · u · (Σ|a·b| + |β·c|)` of it, and with small-integer inputs it is exact.

use crate::{BufferPool, Gpu, GpuError, PipelineCache, PooledBuffer, Shader, Step, StepEncoder};

/// The GEMM shader: 64 × 64 tiles of `C` per workgroup of 16 × 16 threads, 4 × 4 outputs per
/// thread, k in steps of 16 through 8.25 KiB of workgroup memory (padded against bank
/// conflicts); the transposes are `override`
/// constants.
const GEMM: Shader = Shader {
    name: "gemm",
    source: include_str!("gemm.wgsl"),
};

/// The side of the square tile of `C` one workgroup computes.
const TILE: u32 = 64;

/// The bytes of the shader's `Dims` uniform: `m`, `n`, `k` and `beta`.
const DIMS_BYTES: u64 = 16;

/// The shape of one GEMM, `C = A·B + β·C`: `A` is `m × k` (stored `k × m` when
/// `trans_a`), `B` is `k × n` (stored `n × k` when `trans_b`), and `C` is `m × n`, all
/// row-major f32.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::kernels::GemmDims;
///
/// let dims = GemmDims { trans_b: true, ..GemmDims::new(128, 32, 288) };
/// assert_eq!((dims.a_len(), dims.b_len(), dims.c_len()), (128 * 288, 32 * 288, 128 * 32));
/// assert_eq!(dims.flops(), 2.0 * 128.0 * 32.0 * 288.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GemmDims {
    /// Rows of `A` and `C`.
    pub m: u32,
    /// Columns of `B` and `C`.
    pub n: u32,
    /// Columns of `A` and rows of `B`: the length of each dot product.
    pub k: u32,
    /// Whether `A` is stored transposed, `k × m`.
    pub trans_a: bool,
    /// Whether `B` is stored transposed, `n × k`.
    pub trans_b: bool,
    /// The factor on `C`'s old value: 0 overwrites `C` without reading it, so whatever it
    /// held (NaN included) does not matter.
    pub beta: f32,
}

impl GemmDims {
    /// `C = A·B` for an `m × k` `A` and a `k × n` `B`, neither transposed, overwriting `C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::GemmDims;
    ///
    /// let dims = GemmDims::new(4, 5, 6);
    /// assert!(!dims.trans_a && !dims.trans_b && dims.beta == 0.0);
    /// ```
    pub fn new(m: u32, n: u32, k: u32) -> GemmDims {
        GemmDims {
            m,
            n,
            k,
            trans_a: false,
            trans_b: false,
            beta: 0.0,
        }
    }

    /// The number of f32s in `A`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::GemmDims;
    ///
    /// assert_eq!(GemmDims::new(4, 5, 6).a_len(), 24);
    /// ```
    pub fn a_len(&self) -> u64 {
        u64::from(self.m) * u64::from(self.k)
    }

    /// The number of f32s in `B`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::GemmDims;
    ///
    /// assert_eq!(GemmDims::new(4, 5, 6).b_len(), 30);
    /// ```
    pub fn b_len(&self) -> u64 {
        u64::from(self.k) * u64::from(self.n)
    }

    /// The number of f32s in `C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::GemmDims;
    ///
    /// assert_eq!(GemmDims::new(4, 5, 6).c_len(), 20);
    /// ```
    pub fn c_len(&self) -> u64 {
        u64::from(self.m) * u64::from(self.n)
    }

    /// The floating-point operations of the product, `2 · m · n · k`, for GFLOPS figures.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::GemmDims;
    ///
    /// assert_eq!(GemmDims::new(1024, 1024, 1024).flops(), 2.0 * 1024f64.powi(3));
    /// ```
    pub fn flops(&self) -> f64 {
        2.0 * f64::from(self.m) * f64::from(self.n) * f64::from(self.k)
    }

    /// The shader's `Dims` uniform for these dimensions.
    fn uniform(&self) -> [u8; DIMS_BYTES as usize] {
        let mut bytes = [0u8; DIMS_BYTES as usize];
        bytes[0..4].copy_from_slice(&self.m.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.n.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.k.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.beta.to_le_bytes());
        bytes
    }
}

/// The tiled GEMM: its four pipelines (one per pair of transposes) and the bind group layout
/// they share, built once. [`call`](Gemm::call) prepares one GEMM on given buffers, made once
/// when a model is compiled, and [`record`](Gemm::record) adds it to a step.
///
/// Each output adds its `k` products in increasing `k` with `fma`, which a backend may round once
/// or twice, so it is within `2(k + 3) · u · (Σ|a·b| + |β·c|)` of the exact product, `u` =
/// 2^-24, to first order in `u` (the bound the tests check). With inputs whose products and partial sums are
/// integers below 2^24, the result is exact. The same inputs give the same bits on every
/// run on one GPU and backend.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::kernels::{Gemm, GemmDims};
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let mut cache = PipelineCache::new(&gpu);
///     let mut pool = BufferPool::new(&gpu);
///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
///     let gemm = Gemm::new(&gpu, &mut cache).unwrap();
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
///     // [1 2; 3 4] · [5 6; 7 8] = [19 22; 43 50]
///     let (a, b, c) = (pool.take(16, usage), pool.take(16, usage), pool.take(16, usage));
///     let bytes = |v: [f32; 4]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     let call = gemm.call(&mut steps, &mut pool, GemmDims::new(2, 2, 2), &a, &b, &c);
///     let mut step = steps.begin();
///     step.upload(a.buffer(), 0, &bytes([1.0, 2.0, 3.0, 4.0]));
///     step.upload(b.buffer(), 0, &bytes([5.0, 6.0, 7.0, 8.0]));
///     gemm.record(&mut step, &call);
///     step.read_back(c.buffer(), 0, 16);
///     assert_eq!(step.finish().bytes(), &bytes([19.0, 22.0, 43.0, 50.0])[..]);
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct Gemm {
    /// The layout of group 0: `A`, `B` (read-only storage), `C` (read-write storage) and
    /// the `Dims` uniform.
    group: wgpu::BindGroupLayout,
    /// The pipelines, indexed by `2 · trans_a + trans_b`.
    pipelines: [wgpu::ComputePipeline; 4],
}

impl Gemm {
    /// Builds the GEMM's bind group layout and its four pipelines on `gpu`, through `cache`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::Gemm;
    /// use cavitation_gpu::{Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut cache = PipelineCache::new(&gpu);
    ///     Gemm::new(&gpu, &mut cache).unwrap();
    ///     assert_eq!(cache.pipelines_built(), 4);
    /// }
    /// ```
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<Gemm, GpuError> {
        let device = gpu.device();
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemm"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gemm"),
            bind_group_layouts: &[Some(&group)],
            immediate_size: 0,
        });
        let build = |cache: &mut PipelineCache, trans_a: bool, trans_b: bool| {
            let flag = |on: bool| if on { 1.0 } else { 0.0 };
            cache.get_with_layout(
                GEMM,
                "main",
                &[("TRANS_A", flag(trans_a)), ("TRANS_B", flag(trans_b))],
                &layout,
            )
        };
        let pipelines = [
            build(cache, false, false)?,
            build(cache, false, true)?,
            build(cache, true, false)?,
            build(cache, true, true)?,
        ];
        Ok(Gemm { group, pipelines })
    }

    /// Prepares one GEMM of shape `dims` on `a`, `b` and `c`: its `Dims` uniform, taken
    /// from `pool` and uploaded in a step of its own, and its bind group, made through
    /// `steps` so it is counted. Made once, when a model is compiled, and recorded with
    /// [`record`](Gemm::record) as often as needed.
    ///
    /// # Panics
    ///
    /// If a buffer is shorter than its matrix, or if a dimension is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::{Gemm, GemmDims};
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut cache = PipelineCache::new(&gpu);
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let gemm = Gemm::new(&gpu, &mut cache).unwrap();
    ///     let dims = GemmDims::new(8, 8, 8);
    ///     let take = |pool: &mut BufferPool| pool.take(4 * 64, wgpu::BufferUsages::STORAGE);
    ///     let (a, b, c) = (take(&mut pool), take(&mut pool), take(&mut pool));
    ///     let call = gemm.call(&mut steps, &mut pool, dims, &a, &b, &c);
    ///     assert_eq!(call.dims(), dims);
    ///     call.release(&mut pool);
    /// }
    /// ```
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        dims: GemmDims,
        a: &PooledBuffer,
        b: &PooledBuffer,
        c: &PooledBuffer,
    ) -> GemmCall {
        assert!(
            dims.m > 0 && dims.n > 0 && dims.k > 0,
            "a GEMM needs every dimension above 0, not {dims:?}"
        );
        for (name, buffer, len) in [
            ("A", a, dims.a_len()),
            ("B", b, dims.b_len()),
            ("C", c, dims.c_len()),
        ] {
            assert!(
                buffer.len() >= 4 * len,
                "{name} needs {len} f32s ({} bytes) for {dims:?}, but its buffer has {} bytes",
                4 * len,
                buffer.len()
            );
        }
        let params = pool.take(
            DIMS_BYTES,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let mut step = steps.begin();
        step.upload(params.buffer(), 0, &dims.uniform());
        let _ = step.finish();
        let group = steps.bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemm"),
            layout: &self.group,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.binding(),
                },
            ],
        });
        GemmCall {
            dims,
            params,
            group,
        }
    }

    /// Adds the GEMM `call` to `step`: one dispatch of `ceil(n / 64) × ceil(m / 64)`
    /// workgroups.
    ///
    /// # Panics
    ///
    /// If `m` or `n` needs more than 65,535 tiles of 64 (more than 4,194,240).
    ///
    /// # Examples
    ///
    /// See [`Gemm`].
    pub fn record(&self, step: &mut Step<'_>, call: &GemmCall) {
        let d = call.dims;
        let pipeline = &self.pipelines[2 * usize::from(d.trans_a) + usize::from(d.trans_b)];
        step.dispatch_grid(
            pipeline,
            &call.group,
            [d.n.div_ceil(TILE), d.m.div_ceil(TILE), 1],
        );
    }
}

/// One prepared GEMM: its shape, its `Dims` uniform and its bind group, from
/// [`Gemm::call`].
///
/// # Examples
///
/// See [`Gemm`].
#[derive(Debug)]
pub struct GemmCall {
    /// The shape.
    dims: GemmDims,
    /// The `Dims` uniform, from the pool.
    params: PooledBuffer,
    /// The bind group of `A`, `B`, `C` and the uniform.
    group: wgpu::BindGroup,
}

impl GemmCall {
    /// The shape this GEMM was prepared for.
    ///
    /// # Examples
    ///
    /// See [`Gemm::call`].
    pub fn dims(&self) -> GemmDims {
        self.dims
    }

    /// Gives this GEMM's uniform buffer back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`Gemm::call`].
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.params);
    }
}
