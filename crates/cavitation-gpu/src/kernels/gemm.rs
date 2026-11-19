//! The tiled GEMM on the GPU: `C = A·B + β·C` in f32, with either operand transposed, for
//! any sizes. Its CPU twin is the f64 product of the same f32 inputs; every output is within
//! `2(k + 3) · u · (Σ|a·b| + |β·c|)` of it, and with small-integer inputs it is exact.

use crate::{
    Backend, BufferPool, Gpu, GpuError, PipelineCache, PooledBuffer, Shader, Step, StepEncoder,
};

/// The GEMM shader: 64 × 64 tiles of `C` per workgroup of 16 × 16 threads, 4 × 4 outputs per
/// thread, k in steps of 16 through 8.25 KiB of workgroup memory (padded against bank
/// conflicts); the transposes are `override` constants. Its head, default loaders, default
/// store and core are separate files, so a convolution can supply its own loaders and store.
const GEMM: Shader = Shader {
    name: "gemm",
    source: concat!(
        include_str!("gemm_head.wgsl"),
        include_str!("gemm_load.wgsl"),
        include_str!("gemm_store.wgsl"),
        include_str!("gemm_core.wgsl")
    ),
};

/// The GEMM on the wide tile, for a `C` of at most [`WIDE_ROWS`] rows: 32 × 128 tiles of `C`
/// per workgroup of 16 × 16 threads, 2 × 8 outputs per thread, with the same loaders, store
/// and order of sums as [`GEMM`], so each output has the same bits from either.
const GEMM_WIDE: Shader = Shader {
    name: "gemm_wide",
    source: concat!(
        include_str!("gemm_head.wgsl"),
        include_str!("gemm_load.wgsl"),
        include_str!("gemm_store.wgsl"),
        include_str!("gemm_core_wide.wgsl")
    ),
};

/// The side of the square tile of `C` one workgroup computes.
pub(super) const TILE: u32 = 64;

/// The rows of the wide tile: a `C` of at most this many rows is computed on it, where the
/// square tile's 64 rows would be half empty or worse.
pub(super) const WIDE_ROWS: u32 = 32;

/// The columns of the wide tile.
pub(super) const WIDE_COLS: u32 = 128;

/// The fewest workgroups a dispatch on the wide tile must have for the wide tile to be used.
/// Below it the square tile is faster although half of each of its tiles is empty: it has
/// twice the workgroups, and so few do not fill the GPU.
///
/// Measured on an NVIDIA GTX 1660 Super (Vulkan, `cavitation-bench gemm-gpu --shapes`), the
/// wide tile's time over the square tile's, by the wide tile's workgroups: 1.45 at 8, 1.38
/// at 24, 0.93 to 1.11 at 32, 0.72 to 0.73 at 64, and 0.64 to 0.71 from 128 up.
pub(super) const WIDE_MIN_WORKGROUPS: u64 = 64;

/// Whether a product whose `C` is `m × n` runs on the wide tile, and its grid of workgroups
/// for `batch` products: tiles across `n`, tiles down `m`, products. The wide tile is used
/// when all three hold:
/// - `m` is at most [`WIDE_ROWS`], so one row of wide tiles holds every row of `C`;
/// - `n` is above [`TILE`], so the wide tile takes fewer workgroups than the square one (a
///   `C` that fits one square tile stays on it, where each thread loads less of `B`);
/// - the dispatch has at least [`WIDE_MIN_WORKGROUPS`] wide tiles.
pub(super) fn grid(m: u32, n: u32, batch: u32) -> (bool, [u32; 3]) {
    let wide_tiles = u64::from(n.div_ceil(WIDE_COLS)) * u64::from(batch);
    if m <= WIDE_ROWS && n > TILE && wide_tiles >= WIDE_MIN_WORKGROUPS {
        (true, [n.div_ceil(WIDE_COLS), 1, batch])
    } else {
        (false, square_grid(m, n, batch))
    }
}

/// Whether the wide tile's pipelines are built on the cache's GPU: on Vulkan, and on DX12
/// with DXC named as its shader compiler (`WGPU_DX12_COMPILER=dxc`).
///
/// - **DX12 without DXC named:** FXC, which DX12 may fall back to under the `auto` setting,
///   takes about 27 s over each wide pipeline (measured on an NVIDIA GTX 1660 Super: a
///   `cifar-cnn` trainer builds in 244 s with them against 134 s without, to save 2.6 ms a
///   step), where DXC takes about 1 s.
/// - **Metal:** on an Apple M4 the wide tile is slower than the square tile on every product
///   the rule gives it, 1.13× to 1.79× (`gemm-gpu --shapes`, from 8 products of
///   32 × 1024 × 288 to 128, and 64 of 10 × 256 × 128), and the `cifar-cnn` epoch took
///   22.75 s against 22.47 s.
pub(super) fn wide_allowed(cache: &PipelineCache) -> bool {
    let info = cache.gpu().info();
    match info.backend {
        Backend::Metal => false,
        _ => match &info.dx12_compiler {
            None => true,
            Some(setting) => setting.starts_with("dxc"),
        },
    }
}

/// The square tile's grid of workgroups for `batch` products whose `C` is `m × n`: any
/// product can run on it.
pub(super) fn square_grid(m: u32, n: u32, batch: u32) -> [u32; 3] {
    [n.div_ceil(TILE), m.div_ceil(TILE), batch]
}

/// The pipeline of `shader` (the GEMM on one of its tiles) for one pair of transposes.
fn pipeline(
    cache: &mut PipelineCache,
    layout: &wgpu::PipelineLayout,
    shader: Shader,
    trans_a: bool,
    trans_b: bool,
) -> Result<wgpu::ComputePipeline, GpuError> {
    let flag = |on: bool| if on { 1.0 } else { 0.0 };
    cache.get_with_layout(
        shader,
        "main",
        &[("TRANS_A", flag(trans_a)), ("TRANS_B", flag(trans_b))],
        layout,
    )
}

/// The bytes of the shader's `Dims` uniform: sixteen 4-byte fields.
pub(super) const DIMS_BYTES: u64 = 64;

/// The shape and layout of one GEMM, or of a batch of them, `C = A·B + β·C`.
///
/// `A` is `m × k` (stored `k × m` when `trans_a`), `B` is `k × n` (stored `n × k` when
/// `trans_b`), and `C` is `m × n`, all f32, each stored row by row. The leading dimensions
/// `lda`, `ldb` and `ldc` are the distances between stored rows, in elements, 0 meaning
/// packed (the stored row's length). The offsets place each matrix's first element in its
/// buffer. With `batch` above 1, product `z` adds `z · a_batch`, `z · b_batch` and
/// `z · c_batch` elements to the offsets; a batch stride of 0 shares the matrix (one weight
/// matrix for every image of a convolution).
///
/// # Examples
///
/// ```
/// use cavitation_gpu::kernels::GemmDims;
///
/// let dims = GemmDims { trans_b: true, ..GemmDims::new(128, 32, 288) };
/// assert_eq!((dims.a_len(), dims.b_len(), dims.c_len()), (128 * 288, 32 * 288, 128 * 32));
/// assert_eq!(dims.flops(), 2.0 * 128.0 * 32.0 * 288.0);
/// // A 10 × 10 block at row 3, column 5 of a 64 × 64 matrix, times a packed 10 × 7.
/// let block = GemmDims { lda: 64, a_offset: 3 * 64 + 5, ..GemmDims::new(10, 7, 10) };
/// assert_eq!(block.a_extent(), 3 * 64 + 5 + 9 * 64 + 10);
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
    /// The distance between `A`'s stored rows, in elements; 0 for packed.
    pub lda: u32,
    /// The distance between `B`'s stored rows, in elements; 0 for packed.
    pub ldb: u32,
    /// The distance between `C`'s rows, in elements; 0 for packed (`n`), and never below `n`.
    pub ldc: u32,
    /// The number of products, along the grid's z axis; 1 for one.
    pub batch: u32,
    /// The element of `A`'s buffer where product 0's `A` begins.
    pub a_offset: u32,
    /// The element of `B`'s buffer where product 0's `B` begins.
    pub b_offset: u32,
    /// The element of `C`'s buffer where product 0's `C` begins.
    pub c_offset: u32,
    /// The elements between consecutive products' `A`; 0 shares one `A`.
    pub a_batch: u32,
    /// The elements between consecutive products' `B`; 0 shares one `B`.
    pub b_batch: u32,
    /// The elements between consecutive products' `C`. With `batch` above 1 it must keep the
    /// products' `C`s apart, at least `(m − 1) · ldc + n`: products run in parallel, so
    /// two writing one element would race.
    pub c_batch: u32,
}

impl GemmDims {
    /// `C = A·B` for an `m × k` `A` and a `k × n` `B`, neither transposed, all packed from
    /// offset 0, one product, overwriting `C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::GemmDims;
    ///
    /// let dims = GemmDims::new(4, 5, 6);
    /// assert!(!dims.trans_a && !dims.trans_b && dims.beta == 0.0 && dims.batch == 1);
    /// ```
    pub fn new(m: u32, n: u32, k: u32) -> GemmDims {
        GemmDims {
            m,
            n,
            k,
            trans_a: false,
            trans_b: false,
            beta: 0.0,
            lda: 0,
            ldb: 0,
            ldc: 0,
            batch: 1,
            a_offset: 0,
            b_offset: 0,
            c_offset: 0,
            a_batch: 0,
            b_batch: 0,
            c_batch: 0,
        }
    }

    /// The number of f32s in one packed `A`, `m · k`.
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

    /// The number of f32s in one packed `B`, `k · n`.
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

    /// The number of f32s in one packed `C`, `m · n`.
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

    /// The floating-point operations of the products, `2 · m · n · k · batch`, for GFLOPS
    /// figures.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::GemmDims;
    ///
    /// assert_eq!(GemmDims::new(1024, 1024, 1024).flops(), 2.0 * 1024f64.powi(3));
    /// ```
    pub fn flops(&self) -> f64 {
        2.0 * f64::from(self.m)
            * f64::from(self.n)
            * f64::from(self.k)
            * f64::from(self.batch.max(1))
    }

    /// `A`'s leading dimension in use: `lda`, or the stored row's length when it is 0.
    fn lda_used(&self) -> u32 {
        if self.lda != 0 {
            self.lda
        } else if self.trans_a {
            self.m
        } else {
            self.k
        }
    }

    /// `B`'s leading dimension in use: `ldb`, or the stored row's length when it is 0.
    fn ldb_used(&self) -> u32 {
        if self.ldb != 0 {
            self.ldb
        } else if self.trans_b {
            self.k
        } else {
            self.n
        }
    }

    /// `C`'s leading dimension in use: `ldc`, or `n` when it is 0.
    fn ldc_used(&self) -> u32 {
        if self.ldc != 0 { self.ldc } else { self.n }
    }

    /// The f32s `A`'s buffer must hold: one past the last element any product reads.
    ///
    /// # Examples
    ///
    /// See [`GemmDims`].
    pub fn a_extent(&self) -> u64 {
        let (rows, cols) = if self.trans_a {
            (self.k, self.m)
        } else {
            (self.m, self.k)
        };
        extent(
            self.a_offset,
            self.a_batch,
            self.batch,
            rows,
            cols,
            self.lda_used(),
        )
    }

    /// The f32s `B`'s buffer must hold: one past the last element any product reads.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::GemmDims;
    ///
    /// assert_eq!(GemmDims::new(4, 5, 6).b_extent(), 30);
    /// ```
    pub fn b_extent(&self) -> u64 {
        let (rows, cols) = if self.trans_b {
            (self.n, self.k)
        } else {
            (self.k, self.n)
        };
        extent(
            self.b_offset,
            self.b_batch,
            self.batch,
            rows,
            cols,
            self.ldb_used(),
        )
    }

    /// The f32s `C`'s buffer must hold: one past the last element any product writes.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::GemmDims;
    ///
    /// let three = GemmDims { batch: 3, c_batch: 20, ..GemmDims::new(4, 5, 6) };
    /// assert_eq!(three.c_extent(), 60);
    /// ```
    pub fn c_extent(&self) -> u64 {
        extent(
            self.c_offset,
            self.c_batch,
            self.batch,
            self.m,
            self.n,
            self.ldc_used(),
        )
    }

    /// The shader's `Dims` uniform for these dimensions, leading dimensions resolved.
    pub(super) fn uniform(&self) -> [u8; DIMS_BYTES as usize] {
        let words = [
            self.m,
            self.n,
            self.k,
            self.beta.to_bits(),
            self.lda_used(),
            self.ldb_used(),
            self.ldc_used(),
            self.batch.max(1),
            self.a_offset,
            self.b_offset,
            self.c_offset,
            0,
            self.a_batch,
            self.b_batch,
            self.c_batch,
            0,
        ];
        let mut bytes = [0u8; DIMS_BYTES as usize];
        let (chunks, _) = bytes.as_chunks_mut::<4>();
        for (chunk, word) in chunks.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// One past the last element of `batch` matrices of `rows` stored rows of `cols` elements,
/// `ld` apart, the first at `offset` and each next one `stride` further.
fn extent(offset: u32, stride: u32, batch: u32, rows: u32, cols: u32, ld: u32) -> u64 {
    let last = u64::from(batch.max(1) - 1) * u64::from(stride);
    u64::from(offset) + last + u64::from(rows.saturating_sub(1)) * u64::from(ld) + u64::from(cols)
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
    /// The square tile's pipelines, indexed by `2 · trans_a + trans_b`.
    pipelines: [wgpu::ComputePipeline; 4],
    /// The wide tile's pipelines, indexed the same way, each built by
    /// [`ensure_wide`](Gemm::ensure_wide) for the first product that needs it.
    wide: [Option<wgpu::ComputePipeline>; 4],
    /// The pipeline layout every pipeline is built with.
    layout: wgpu::PipelineLayout,
}

impl Gemm {
    /// Builds the GEMM's bind group layout and the square tile's four pipelines on `gpu`,
    /// through `cache`. The wide tile's pipelines are built later, by
    /// [`ensure_wide`](Gemm::ensure_wide), for the products that use them: a backend's
    /// shader compiler may take seconds over each (FXC about 26 s on an NVIDIA GTX 1660
    /// Super), which a model with no such product should not pay.
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
        let pipelines = [
            pipeline(cache, &layout, GEMM, false, false)?,
            pipeline(cache, &layout, GEMM, false, true)?,
            pipeline(cache, &layout, GEMM, true, false)?,
            pipeline(cache, &layout, GEMM, true, true)?,
        ];
        Ok(Gemm {
            group,
            pipelines,
            wide: [None, None, None, None],
            layout,
        })
    }

    /// Builds the wide tile's pipeline for `call`, through `cache`, if `call` is a product
    /// the wide tile computes (at most 32 rows, more than 64 columns and at least 64 wide
    /// tiles in its dispatch) and the pipeline for its transposes is not built yet. Returns
    /// whether `call` runs on the wide tile from now on.
    ///
    /// On Metal it builds nothing and returns `false`: there the wide tile is slower. On DX12
    /// it does the same unless DXC is named as the shader compiler
    /// (`WGPU_DX12_COMPILER=dxc`): FXC takes about 27 s over each wide pipeline.
    ///
    /// [`record`](Gemm::record) uses the wide tile only for a product this has been called
    /// for; without it the product runs on the square tile, with the same bits. Call it
    /// once for each prepared call, when a model is compiled.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if the wide kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::{Gemm, GemmDims};
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let (mut cache, mut pool) = (PipelineCache::new(&gpu), BufferPool::new(&gpu));
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let mut gemm = Gemm::new(&gpu, &mut cache).unwrap();
    ///     let take = |pool: &mut BufferPool, n: u64| pool.take(4 * n, wgpu::BufferUsages::STORAGE);
    ///     // 64 products of 8 × 128 × 4, sharing A and B: 64 wide tiles.
    ///     let dims = GemmDims { batch: 64, c_batch: 8 * 128, ..GemmDims::new(8, 128, 4) };
    ///     let (a, b, c) = (take(&mut pool, 32), take(&mut pool, 512), take(&mut pool, 64 * 1024));
    ///     let call = gemm.call(&mut steps, &mut pool, dims, &a, &b, &c);
    ///     // True except on DX12 without DXC named as its compiler.
    ///     let wide = gemm.ensure_wide(&mut cache, &call).unwrap();
    ///     assert_eq!(cache.pipelines_built(), 4 + usize::from(wide));
    ///     // One product of the same shape is one wide tile: it stays on the square tile.
    ///     let one = gemm.call(&mut steps, &mut pool, GemmDims::new(8, 128, 4), &a, &b, &c);
    ///     assert!(!gemm.ensure_wide(&mut cache, &one).unwrap());
    /// }
    /// ```
    pub fn ensure_wide(
        &mut self,
        cache: &mut PipelineCache,
        call: &GemmCall,
    ) -> Result<bool, GpuError> {
        let d = call.dims;
        if !wide_allowed(cache) || !grid(d.m, d.n, d.batch.max(1)).0 {
            return Ok(false);
        }
        let at = 2 * usize::from(d.trans_a) + usize::from(d.trans_b);
        if self.wide[at].is_none() {
            self.wide[at] = Some(pipeline(
                cache,
                &self.layout,
                GEMM_WIDE,
                d.trans_a,
                d.trans_b,
            )?);
        }
        Ok(true)
    }

    /// Prepares one GEMM of shape `dims` on `a`, `b` and `c`: its `Dims` uniform, taken
    /// from `pool` and uploaded in a step of its own, and its bind group, made through
    /// `steps` so it is counted. Made once, when a model is compiled, and recorded with
    /// [`record`](Gemm::record) as often as needed.
    ///
    /// # Panics
    ///
    /// If a buffer is shorter than its matrix (offsets, leading dimensions and batch
    /// included), if a dimension is 0, if `C`'s rows or a batch's `C`s overlap (`ldc` below
    /// `n`, or `c_batch` below `(m − 1) · ldc + n` with `batch` above 1), or if `C`'s buffer is
    /// `A`'s or `B`'s: wgpu does not let one buffer be read and written in one dispatch. A
    /// leading dimension of `A` or `B` below its stored row's length is allowed: rows that
    /// share elements are only read.
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
        let ldc = dims.ldc_used();
        assert!(
            ldc >= dims.n,
            "C's rows overlap: ldc {ldc} is below n for {dims:?}"
        );
        let c_size = u64::from(dims.m - 1) * u64::from(ldc) + u64::from(dims.n);
        assert!(
            dims.batch <= 1 || u64::from(dims.c_batch) >= c_size,
            "the batch's Cs overlap: c_batch must be at least {c_size} for {dims:?}"
        );
        for (name, buffer, len) in [
            ("A", a, dims.a_extent()),
            ("B", b, dims.b_extent()),
            ("C", c, dims.c_extent()),
        ] {
            assert!(
                buffer.len() >= 4 * len,
                "{name} needs {len} f32s ({} bytes) for {dims:?}, but its buffer has {} bytes",
                4 * len,
                buffer.len()
            );
        }
        assert!(
            c.buffer() != a.buffer() && c.buffer() != b.buffer(),
            "a GEMM cannot write C into a buffer it reads A or B from"
        );
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

    /// Adds the GEMM `call` to `step`: one dispatch of `ceil(n / 64) × ceil(m / 64) × batch`
    /// workgroups, or of `ceil(n / 128) × 1 × batch` on the wide tile when `m` is at most 32,
    /// `n` is above 64, that grid has at least 64 workgroups (measured: below that the
    /// square tile is faster) and [`ensure_wide`](Gemm::ensure_wide) has built the wide
    /// pipeline for the call. Each output is the same bits on either tile.
    ///
    /// # Panics
    ///
    /// If the grid has more than 65,535 workgroups along a side: on the square tile `m` or
    /// `n` above 4,194,240 (65,535 tiles of 64), on the wide tile `n` above 8,388,480
    /// (65,535 tiles of 128), or `batch` above 65,535. Split such products into several
    /// calls.
    ///
    /// # Examples
    ///
    /// See [`Gemm`].
    pub fn record(&self, step: &mut Step<'_>, call: &GemmCall) {
        let d = call.dims;
        let at = 2 * usize::from(d.trans_a) + usize::from(d.trans_b);
        let batch = d.batch.max(1);
        match (grid(d.m, d.n, batch), &self.wide[at]) {
            ((true, wide_grid), Some(wide)) => step.dispatch_grid(wide, &call.group, wide_grid),
            _ => step.dispatch_grid(
                &self.pipelines[at],
                &call.group,
                square_grid(d.m, d.n, batch),
            ),
        }
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

#[cfg(test)]
mod tests {
    //! The rule that chooses the tile; the products themselves are checked on a GPU in
    //! `tests/gemm.rs`.

    use super::grid;

    /// The wide tile is chosen for at most 32 rows, more than 64 columns and at least 64 wide
    /// tiles; every other product keeps the square tile, with its grid.
    #[test]
    fn the_wide_tile_is_chosen_when_it_pays() {
        // A 32-channel convolution over a batch: 8 wide tiles per image, 128 images.
        assert_eq!(grid(32, 1024, 128), (true, [8, 1, 128]));
        assert_eq!(grid(32, 1024, 8), (true, [8, 1, 8]));
        assert_eq!(grid(1, 65, 64), (true, [1, 1, 64]));
        // One wide tile short of 64.
        assert_eq!(grid(32, 1024, 7), (false, [16, 1, 7]));
        // One product of few rows, as the last layer's weight gradient is.
        assert_eq!(grid(10, 256, 1), (false, [4, 1, 1]));
        // A C that fits one square tile.
        assert_eq!(grid(32, 64, 4096), (false, [1, 1, 4096]));
        assert_eq!(grid(32, 27, 128), (false, [1, 1, 128]));
        // More than 32 rows.
        assert_eq!(grid(33, 1024, 128), (false, [16, 1, 128]));
        assert_eq!(grid(200, 1000, 1), (false, [16, 4, 1]));
    }
}
