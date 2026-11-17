//! The 2-D convolution on the GPU, as implicit GEMMs through the tiled GEMM's core, with
//! stride, padding, dilation and an optional bias.
//!
//! - [`ConvForward`]: the forward pass. Its CPU twin is
//!   `cavitation_core::kernels::conv::conv2d_reference`. Every output is within
//!   `2(k + 3) · u · (Σ|w·x| + |bias|)` of the f64 convolution of the same f32 inputs,
//!   `k` = in·kh·kw, and exact on small-integer inputs.
//! - [`ConvInputGrad`]: the input gradient. Its CPU twin is `conv2d_backward_reference`'s
//!   `dx`.
//! - [`ConvWeightGrad`]: the weight gradient. Its CPU twin is `conv2d_backward_reference`'s
//!   `dw`.

use super::bindings::{Kind, layout};
use super::gemm::{DIMS_BYTES, GemmDims, grid, square_grid, wide_allowed};
use crate::{
    BufferPool, Gpu, GpuError, MAX_WORKGROUPS_PER_DIMENSION, PipelineCache, PooledBuffer, Shader,
    Step, StepEncoder,
};

/// The convolution shader: the GEMM's head and core around the convolution's loaders, which
/// read the input images' windows as the tiles are filled, and its store, which adds the bias.
const CONV_FORWARD: Shader = Shader {
    name: "conv_forward",
    source: concat!(
        include_str!("gemm_head.wgsl"),
        include_str!("conv_forward.wgsl"),
        include_str!("gemm_core.wgsl")
    ),
};

/// [`CONV_FORWARD`] on the GEMM's wide tile, for at most 32 output channels.
const CONV_FORWARD_WIDE: Shader = Shader {
    name: "conv_forward_wide",
    source: concat!(
        include_str!("gemm_head.wgsl"),
        include_str!("conv_forward.wgsl"),
        include_str!("gemm_core_wide.wgsl")
    ),
};

/// The input-gradient shader: the GEMM's head, the gradient's loaders, which gather each input
/// position's output gradients as the tiles are filled, the GEMM's own store and its core.
const CONV_INPUT_GRAD: Shader = Shader {
    name: "conv_input_grad",
    source: concat!(
        include_str!("gemm_head.wgsl"),
        include_str!("conv_input_grad.wgsl"),
        include_str!("gemm_store.wgsl"),
        include_str!("gemm_core.wgsl")
    ),
};

/// [`CONV_INPUT_GRAD`] on the GEMM's wide tile, for at most 32 input channels.
const CONV_INPUT_GRAD_WIDE: Shader = Shader {
    name: "conv_input_grad_wide",
    source: concat!(
        include_str!("gemm_head.wgsl"),
        include_str!("conv_input_grad.wgsl"),
        include_str!("gemm_store.wgsl"),
        include_str!("gemm_core_wide.wgsl")
    ),
};

/// The weight-gradient shader: the GEMM's head, the gradient's loaders, which read the output
/// gradients and the input images' windows of one block of positions as the tiles are
/// filled, the GEMM's own store, which writes the block's partial gradient, and its core.
const CONV_WEIGHT_GRAD: Shader = Shader {
    name: "conv_weight_grad",
    source: concat!(
        include_str!("gemm_head.wgsl"),
        include_str!("conv_weight_grad.wgsl"),
        include_str!("gemm_store.wgsl"),
        include_str!("gemm_core.wgsl")
    ),
};

/// [`CONV_WEIGHT_GRAD`] on the GEMM's wide tile, for at most 32 output channels.
const CONV_WEIGHT_GRAD_WIDE: Shader = Shader {
    name: "conv_weight_grad_wide",
    source: concat!(
        include_str!("gemm_head.wgsl"),
        include_str!("conv_weight_grad.wgsl"),
        include_str!("gemm_store.wgsl"),
        include_str!("gemm_core_wide.wgsl")
    ),
};

/// Threads per workgroup of the weight gradient's merge, passed as its `WORKGROUP` override.
const MERGE_WORKGROUP: u32 = 256;

/// The weight gradient's second pass: one thread per weight adds its blocks' partial
/// gradients in block order and overwrites the weight gradient with the sum. It reads the
/// first eight words of the GEMM's `Dims` uniform: `m · n` weights, `batch` blocks.
const CONV_WEIGHT_MERGE: Shader = Shader {
    name: "conv_weight_merge",
    source: "
        override WORKGROUP: u32;

        struct Sizes {
            m: u32,
            n: u32,
            k: u32,
            beta: f32,
            lda: u32,
            ldb: u32,
            ldc: u32,
            blocks: u32,
        }

        @group(0) @binding(0) var<storage, read> partials: array<f32>;
        @group(0) @binding(1) var<storage, read_write> dw: array<f32>;
        @group(0) @binding(2) var<uniform> sizes: Sizes;

        @compute @workgroup_size(WORKGROUP)
        fn main(
            @builtin(workgroup_id) group: vec3<u32>,
            @builtin(num_workgroups) groups: vec3<u32>,
            @builtin(local_invocation_index) lane: u32,
        ) {
            let e = (group.x + group.y * groups.x) * WORKGROUP + lane;
            let len = sizes.m * sizes.n;
            if (e >= len) { return; }
            var acc = partials[e];
            for (var z = 1u; z < sizes.blocks; z = z + 1u) {
                acc = acc + partials[z * len + e];
            }
            dw[e] = acc;
        }
    ",
};

/// The positions (image, output row, output column) each block of the weight gradient sums
/// into one partial gradient, counted across images; a multiple of the GEMM's 16-deep step.
pub const WEIGHT_GRAD_BLOCK: u32 = 1024;

/// The bytes of each shader's convolution uniform (`Conv`, `ConvGrad` or `ConvWeightGrad`):
/// sixteen 4-byte fields.
const CONV_BYTES: u64 = 64;

/// The shape of one convolution of a batch of images, in NCHW order: `batch` images of
/// `in_channels` planes of `input` (height, width), by `out_channels` kernels of
/// `in_channels` planes of `kernel` (height, width), with `stride`, zero `padding` on each
/// side and `dilation` between taps, as `cavitation_core`'s `Conv2dGeom` holds them.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::kernels::ConvDims;
///
/// // CIFAR's first convolution: 3 → 32 channels, 3 × 3, padding 1, on 128 images.
/// let dims = ConvDims { padding: (1, 1), ..ConvDims::new(128, 3, 32, (32, 32), (3, 3)) };
/// assert_eq!(dims.out_size(), Some((32, 32)));
/// assert_eq!((dims.x_len(), dims.weight_len(), dims.out_len()), (128 * 3 * 1024, 32 * 27, 128 * 32 * 1024));
/// // A 5 × 5 kernel does not fit a 4 × 4 image.
/// assert_eq!(ConvDims::new(1, 1, 6, (4, 4), (5, 5)).out_size(), None);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvDims {
    /// The number of images.
    pub batch: u32,
    /// Channels of each input image.
    pub in_channels: u32,
    /// Channels of each output image: the number of kernels.
    pub out_channels: u32,
    /// Height and width of each input image.
    pub input: (u32, u32),
    /// Kernel height and width.
    pub kernel: (u32, u32),
    /// Step between windows, down and across.
    pub stride: (u32, u32),
    /// Zeros added on each side, top and bottom, left and right.
    pub padding: (u32, u32),
    /// Spacing between the kernel's taps, down and across.
    pub dilation: (u32, u32),
}

/// The output size along one axis, or `None` if no window fits or the stride or dilation is 0.
fn axis_out(size: u32, kernel: u32, stride: u32, padding: u32, dilation: u32) -> Option<u32> {
    if stride == 0 || dilation == 0 || kernel == 0 {
        return None;
    }
    let span = u64::from(dilation) * u64::from(kernel - 1) + 1;
    let padded = u64::from(size) + 2 * u64::from(padding);
    // The shader's padded coordinates are 32-bit.
    if padded > u64::from(u32::MAX) {
        return None;
    }
    let room = padded.checked_sub(span)?;
    u32::try_from(room / u64::from(stride) + 1).ok()
}

impl ConvDims {
    /// A convolution with stride 1, no padding and no dilation.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::ConvDims;
    ///
    /// let dims = ConvDims::new(64, 1, 6, (28, 28), (5, 5));
    /// assert_eq!((dims.stride, dims.padding, dims.dilation), ((1, 1), (0, 0), (1, 1)));
    /// assert_eq!(dims.out_size(), Some((24, 24)));
    /// ```
    pub fn new(
        batch: u32,
        in_channels: u32,
        out_channels: u32,
        input: (u32, u32),
        kernel: (u32, u32),
    ) -> ConvDims {
        ConvDims {
            batch,
            in_channels,
            out_channels,
            input,
            kernel,
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
        }
    }

    /// The output height and width, or `None` if no window fits, or a stride, dilation or
    /// kernel size is 0.
    ///
    /// # Examples
    ///
    /// See [`ConvDims`].
    pub fn out_size(&self) -> Option<(u32, u32)> {
        Some((
            axis_out(
                self.input.0,
                self.kernel.0,
                self.stride.0,
                self.padding.0,
                self.dilation.0,
            )?,
            axis_out(
                self.input.1,
                self.kernel.1,
                self.stride.1,
                self.padding.1,
                self.dilation.1,
            )?,
        ))
    }

    /// The f32s of the input images, `batch · in · h · w`.
    ///
    /// # Examples
    ///
    /// See [`ConvDims`].
    pub fn x_len(&self) -> u64 {
        u64::from(self.batch)
            * u64::from(self.in_channels)
            * u64::from(self.input.0)
            * u64::from(self.input.1)
    }

    /// The f32s of the weight, `out · in · kh · kw`.
    ///
    /// # Examples
    ///
    /// See [`ConvDims`].
    pub fn weight_len(&self) -> u64 {
        u64::from(self.out_channels) * self.taps()
    }

    /// The f32s of the output images, `batch · out · oh · ow`, or 0 if no window fits.
    ///
    /// # Examples
    ///
    /// See [`ConvDims`].
    pub fn out_len(&self) -> u64 {
        self.out_size().map_or(0, |(oh, ow)| {
            u64::from(self.batch) * u64::from(self.out_channels) * u64::from(oh) * u64::from(ow)
        })
    }

    /// The multiplications and additions, `2 · batch · out · oh · ow · in · kh · kw`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::ConvDims;
    ///
    /// assert_eq!(ConvDims::new(1, 2, 3, (4, 4), (3, 3)).flops(), 2.0 * 3.0 * 4.0 * 18.0);
    /// ```
    pub fn flops(&self) -> f64 {
        2.0 * self.out_len() as f64 * self.taps() as f64
    }

    /// The length of each dot product, `in · kh · kw`.
    fn taps(&self) -> u64 {
        u64::from(self.in_channels) * u64::from(self.kernel.0) * u64::from(self.kernel.1)
    }

    /// The batched GEMM this convolution runs: per image, `C` (out × oh·ow) = `W` (out ×
    /// in·kh·kw) times the image's windows. Call it only once `out_size` is `Some`.
    fn gemm(&self, (oh, ow): (u32, u32)) -> GemmDims {
        let n = oh * ow;
        GemmDims {
            batch: self.batch,
            c_batch: self.out_channels * n,
            ..GemmDims::new(self.out_channels, n, self.taps() as u32)
        }
    }

    /// The shader's `Conv` uniform.
    fn uniform(&self, ow: u32, has_bias: bool) -> [u8; CONV_BYTES as usize] {
        let (h, w) = self.input;
        let words = [
            self.in_channels,
            h,
            w,
            self.kernel.1,
            self.stride.0,
            self.stride.1,
            self.padding.0,
            self.padding.1,
            self.dilation.0,
            self.dilation.1,
            ow,
            self.kernel.0 * self.kernel.1,
            self.in_channels * h * w,
            u32::from(has_bias),
            0,
            0,
        ];
        words_to_bytes(words)
    }

    /// The batched GEMM of the input gradient: per image, `dX` (in × h·w) = `W'` (in ×
    /// out·kh·kw) times the image's gathered output gradients.
    fn grad_gemm(&self) -> GemmDims {
        let n = self.input.0 * self.input.1;
        GemmDims {
            batch: self.batch,
            c_batch: self.in_channels * n,
            ..GemmDims::new(
                self.in_channels,
                n,
                self.out_channels * self.kernel.0 * self.kernel.1,
            )
        }
    }

    /// The input-gradient shader's `ConvGrad` uniform.
    fn grad_uniform(&self, (oh, ow): (u32, u32)) -> [u8; CONV_BYTES as usize] {
        let (h, w) = self.input;
        words_to_bytes([
            self.in_channels,
            h,
            w,
            self.kernel.1,
            self.stride.0,
            self.stride.1,
            self.padding.0,
            self.padding.1,
            self.dilation.0,
            self.dilation.1,
            oh,
            ow,
            self.kernel.0 * self.kernel.1,
            self.out_channels * oh * ow,
            0,
            0,
        ])
    }

    /// The batched GEMM of the weight gradient: per block `z` of [`WEIGHT_GRAD_BLOCK`]
    /// positions, its partial `dW` (out × in·kh·kw) = `dY` (out × the block's positions) times
    /// the block's windows; `batch` is the number of blocks. Call it only once `out_size` is
    /// `Some`.
    fn weight_grad_gemm(&self, (oh, ow): (u32, u32)) -> GemmDims {
        let positions = self.positions((oh, ow));
        let block = positions.min(u64::from(WEIGHT_GRAD_BLOCK)) as u32;
        GemmDims {
            batch: positions.div_ceil(u64::from(block)) as u32,
            c_batch: self.weight_len() as u32,
            ..GemmDims::new(self.out_channels, self.taps() as u32, block)
        }
    }

    /// The positions the weight gradient sums over, `batch · oh · ow`.
    fn positions(&self, (oh, ow): (u32, u32)) -> u64 {
        u64::from(self.batch) * u64::from(oh) * u64::from(ow)
    }

    /// The weight-gradient shader's `ConvWeightGrad` uniform.
    fn weight_grad_uniform(&self, (oh, ow): (u32, u32)) -> [u8; CONV_BYTES as usize] {
        let (h, w) = self.input;
        words_to_bytes([
            self.in_channels,
            h,
            w,
            self.kernel.1,
            self.stride.0,
            self.stride.1,
            self.padding.0,
            self.padding.1,
            self.dilation.0,
            self.dilation.1,
            ow,
            oh * ow,
            self.kernel.0 * self.kernel.1,
            self.out_channels,
            self.positions((oh, ow)) as u32,
            0,
        ])
    }

    /// The output size, after checking that the shape is one the kernels take.
    ///
    /// # Panics
    ///
    /// If no window fits, a count, size, stride or dilation is 0, or `batch` is above
    /// 65,535.
    fn checked_out_size(&self) -> (u32, u32) {
        let size = self
            .out_size()
            .filter(|_| self.batch > 0 && self.in_channels > 0 && self.out_channels > 0);
        let Some(size) = size else {
            panic!("a convolution needs every count above 0 and a window that fits, not {self:?}");
        };
        assert!(
            self.batch <= 65_535,
            "a convolution takes at most 65,535 images per call, not {}",
            self.batch
        );
        size
    }

    /// Panics unless every `(name, buffer, len)` holds its tensor of `len` f32s within the
    /// shader's 32-bit indices, and `written` is none of the buffers read.
    fn check_buffers(
        &self,
        tensors: &[(&str, &PooledBuffer, u64)],
        written: &PooledBuffer,
        read: &[&PooledBuffer],
    ) {
        for &(name, buffer, len) in tensors {
            assert!(
                len <= u64::from(u32::MAX),
                "{name} holds {len} values for {self:?}, more than the shader's 32-bit indices reach"
            );
            assert!(
                buffer.len() >= 4 * len,
                "{name} needs {len} f32s ({} bytes) for {self:?}, but its buffer has {} bytes",
                4 * len,
                buffer.len()
            );
        }
        assert!(
            read.iter().all(|r| r.buffer() != written.buffer()),
            "a convolution cannot write its output into a buffer it reads"
        );
    }
}

/// The little-endian bytes of a convolution uniform's sixteen words.
fn words_to_bytes(words: [u32; 16]) -> [u8; CONV_BYTES as usize] {
    let mut bytes = [0u8; CONV_BYTES as usize];
    let (chunks, _) = bytes.as_chunks_mut::<4>();
    for (chunk, word) in chunks.iter_mut().zip(words) {
        *chunk = word.to_le_bytes();
    }
    bytes
}

/// A bind group layout and its pipeline layout for a convolution shader: `A` and `B`
/// (read-only storage), `C` (read-write storage), the GEMM's `Dims` and the convolution
/// uniform, then `extra` more read-only storage buffers.
fn conv_layout(
    gpu: &Gpu,
    label: &str,
    extra: u32,
) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
    let buffer = |binding, ty, size: u64| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(size),
        },
        count: None,
    };
    let read = wgpu::BufferBindingType::Storage { read_only: true };
    let uniform = wgpu::BufferBindingType::Uniform;
    let mut entries = vec![
        buffer(0, read, 0),
        buffer(1, read, 0),
        buffer(2, wgpu::BufferBindingType::Storage { read_only: false }, 0),
        buffer(3, uniform, DIMS_BYTES),
        buffer(4, uniform, CONV_BYTES),
    ];
    entries.extend((0..extra).map(|k| buffer(5 + k, read, 0)));
    let device = gpu.device();
    let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&group)],
        immediate_size: 0,
    });
    (group, layout)
}

/// The two uniforms of one convolution call, from `pool`, uploaded in a step of their own.
fn upload_params(
    steps: &mut StepEncoder,
    pool: &mut BufferPool,
    gemm: &GemmDims,
    conv: &[u8; CONV_BYTES as usize],
) -> [PooledBuffer; 2] {
    let usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
    let (gemm_params, conv_params) = (pool.take(DIMS_BYTES, usage), pool.take(CONV_BYTES, usage));
    let mut step = steps.begin();
    step.upload(gemm_params.buffer(), 0, &gemm.uniform());
    step.upload(conv_params.buffer(), 0, conv);
    let _ = step.finish();
    [gemm_params, conv_params]
}

/// A counted bind group on `layout` with `buffers` at bindings 0, 1, ….
fn bind(
    steps: &mut StepEncoder,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    buffers: &[&PooledBuffer],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
        .iter()
        .enumerate()
        .map(|(k, b)| wgpu::BindGroupEntry {
            binding: k as u32,
            resource: b.binding(),
        })
        .collect();
    steps.bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

/// The convolution forward pass: one pipeline and its bind group layout, built once.
/// [`call`](ConvForward::call) prepares one convolution on given buffers, made once when a
/// model is compiled, and [`record`](ConvForward::record) adds it to a step.
///
/// Each output adds its `in · kh · kw` products in the CPU reference's order (channel, then
/// kernel row, then column) with `fma`, a tap in the padding adding 0, then adds the bias
/// with one rounding. So it is within `2(k + 3) · u · (Σ|w·x| + |bias|)` of the exact
/// convolution, `u` = 2^-24, to first order in `u`, and exact on inputs whose products and
/// partial sums are integers below 2^24. The same inputs give the same bits on every run on
/// one GPU and backend. One difference from the CPU reference, which skips the taps in the
/// padding: a NaN or infinite weight meets the padding's 0 there, so its outputs are NaN.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::kernels::{ConvDims, ConvForward};
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let mut cache = PipelineCache::new(&gpu);
///     let mut pool = BufferPool::new(&gpu);
///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
///     let conv = ConvForward::new(&gpu, &mut cache).unwrap();
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     // One 1 × 3 image [1 2 3] by the kernel [2] plus bias 0.5: [2.5 4.5 6.5].
///     let dims = ConvDims::new(1, 1, 1, (1, 3), (1, 1));
///     let (x, w, b, y) = (pool.take(12, usage), pool.take(4, usage), pool.take(4, usage), pool.take(12, usage));
///     let call = conv.call(&mut steps, &mut pool, dims, &x, &w, Some(&b), &y);
///     let mut step = steps.begin();
///     step.upload(x.buffer(), 0, &bytes(&[1.0, 2.0, 3.0]));
///     step.upload(w.buffer(), 0, &bytes(&[2.0]));
///     step.upload(b.buffer(), 0, &bytes(&[0.5]));
///     conv.record(&mut step, &call);
///     step.read_back(y.buffer(), 0, 12);
///     assert_eq!(step.finish().bytes(), &bytes(&[2.5, 4.5, 6.5])[..]);
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct ConvForward {
    /// The layout of group 0: the weight, the images (read-only storage), the output
    /// (read-write storage), the GEMM's `Dims` and the `Conv` uniform, and the bias
    /// (read-only storage).
    group: wgpu::BindGroupLayout,
    /// The pipeline on the GEMM's square tile.
    pipeline: wgpu::ComputePipeline,
    /// The pipeline on the GEMM's wide tile, built by `ensure_wide` for the first call that
    /// needs it.
    wide: Option<wgpu::ComputePipeline>,
    /// The pipeline layout both are built with.
    layout: wgpu::PipelineLayout,
}

impl ConvForward {
    /// Builds the convolution's bind group layout and pipeline on `gpu`, through `cache`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::ConvForward;
    /// use cavitation_gpu::{Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut cache = PipelineCache::new(&gpu);
    ///     ConvForward::new(&gpu, &mut cache).unwrap();
    ///     assert_eq!(cache.pipelines_built(), 1);
    /// }
    /// ```
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<ConvForward, GpuError> {
        let (group, layout) = conv_layout(gpu, "conv_forward", 1);
        let pipeline = cache.get_with_layout(CONV_FORWARD, "main", &[], &layout)?;
        Ok(ConvForward {
            group,
            pipeline,
            wide: None,
            layout,
        })
    }

    /// Builds this kernel's pipeline on the GEMM's wide tile for `call`, through `cache`, if
    /// `call`'s product is one the wide tile computes ([`Gemm::ensure_wide`](super::Gemm::ensure_wide)
    /// gives the rule) and it is not built yet. Returns whether `call` runs on the wide tile
    /// from now on. `record` uses the wide tile only for a call this has been called for;
    /// without it the call runs on the square tile, with the same bits.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if the wide kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// See [`Gemm::ensure_wide`](super::Gemm::ensure_wide).
    pub fn ensure_wide(
        &mut self,
        cache: &mut PipelineCache,
        call: &ConvCall,
    ) -> Result<bool, GpuError> {
        let d = call.gemm;
        if !wide_allowed(cache) || !grid(d.m, d.n, d.batch).0 {
            return Ok(false);
        }
        if self.wide.is_none() {
            self.wide =
                Some(cache.get_with_layout(CONV_FORWARD_WIDE, "main", &[], &self.layout)?);
        }
        Ok(true)
    }

    /// Prepares one convolution of shape `dims`: the input images `x`, the weight
    /// (`[out][in][kh][kw]`), the bias (`[out]`) if any, and the output images `out`. Its two
    /// uniforms are taken from `pool` and uploaded in a step of their own, and its bind
    /// group is made through `steps`, so it is counted. Made once, when a model is compiled,
    /// and recorded with [`record`](ConvForward::record) as often as needed.
    ///
    /// # Panics
    ///
    /// - if no window fits, or a count, size, stride or dilation is 0;
    /// - if a buffer is shorter than its tensor;
    /// - if a tensor holds 2^32 values or more (the shader's indices are 32-bit), or `batch`
    ///   is above 65,535;
    /// - if `out` is also `x`, the weight or the bias: wgpu does not let one buffer be read
    ///   and written in one dispatch.
    ///
    /// # Examples
    ///
    /// See [`ConvForward`].
    #[allow(clippy::too_many_arguments)]
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        dims: ConvDims,
        x: &PooledBuffer,
        weight: &PooledBuffer,
        bias: Option<&PooledBuffer>,
        out: &PooledBuffer,
    ) -> ConvCall {
        let (oh, ow) = dims.checked_out_size();
        let mut tensors = vec![
            ("the input", x, dims.x_len()),
            ("the weight", weight, dims.weight_len()),
            ("the output", out, dims.out_len()),
        ];
        let mut read = vec![x, weight];
        if let Some(b) = bias {
            tensors.push(("the bias", b, u64::from(dims.out_channels)));
            read.push(b);
        }
        dims.check_buffers(&tensors, out, &read);
        let gemm = dims.gemm((oh, ow));
        let params = upload_params(steps, pool, &gemm, &dims.uniform(ow, bias.is_some()));
        // Without a bias the shader never reads binding 5, so the weight stands in for it.
        let group = bind(
            steps,
            &self.group,
            "conv_forward",
            &[
                weight,
                x,
                out,
                &params[0],
                &params[1],
                bias.unwrap_or(weight),
            ],
        );
        ConvCall {
            dims,
            gemm,
            params,
            group,
        }
    }

    /// Adds the convolution `call` to `step`: one dispatch of `ceil(oh·ow / 64) ×
    /// ceil(out / 64) × batch` workgroups.
    ///
    /// # Panics
    ///
    /// If `oh · ow` or `out` needs more than 65,535 of the GEMM's tiles (64 a side, or 128
    /// positions across on the wide tile, which [`Gemm::record`](super::Gemm::record)
    /// describes and which at most 32 output channels may use): split such a convolution
    /// into several calls.
    ///
    /// # Examples
    ///
    /// See [`ConvForward`].
    pub fn record(&self, step: &mut Step<'_>, call: &ConvCall) {
        let d = call.gemm;
        match (grid(d.m, d.n, d.batch), &self.wide) {
            ((true, wide_grid), Some(wide)) => step.dispatch_grid(wide, &call.group, wide_grid),
            _ => step.dispatch_grid(&self.pipeline, &call.group, square_grid(d.m, d.n, d.batch)),
        }
    }
}

/// One prepared convolution pass: its shape, its two uniforms and its bind group, from
/// [`ConvForward::call`] or [`ConvInputGrad::call`]. It records only through the kernel that
/// made it: the two kernels' bind group layouts differ.
///
/// # Examples
///
/// See [`ConvForward`].
#[derive(Debug)]
pub struct ConvCall {
    /// The shape.
    dims: ConvDims,
    /// The batched GEMM it runs.
    gemm: GemmDims,
    /// The GEMM's `Dims` uniform and the `Conv` or `ConvGrad` uniform, from the pool.
    params: [PooledBuffer; 2],
    /// The bind group.
    group: wgpu::BindGroup,
}

impl ConvCall {
    /// The shape this convolution was prepared for.
    ///
    /// # Examples
    ///
    /// See [`ConvForward`].
    pub fn dims(&self) -> ConvDims {
        self.dims
    }

    /// Gives this convolution's uniform buffers back to `pool`, once it is no longer
    /// recorded.
    ///
    /// # Examples
    ///
    /// See [`ConvForward`].
    pub fn release(self, pool: &mut BufferPool) {
        let [a, b] = self.params;
        pool.give(a);
        pool.give(b);
    }
}

/// The convolution's input gradient: one pipeline and its bind group layout, built once.
/// [`call`](ConvInputGrad::call) prepares one gradient on given buffers, made once when a
/// model is compiled, and [`record`](ConvInputGrad::record) adds it to a step.
///
/// The input gradient `dx = Σ w · dy` is gathered: each input value sums, over the output
/// channels and the kernel's taps in the CPU reference's order (`o`, then the kernel row,
/// then the column), the output gradient of the one window whose tap lands on it, or 0 where
/// none does. It uses `fma` and writes every input gradient, overwriting it, with no
/// atomics. So it is within `2(k + 3) · u · Σ|w·dy|` of the exact gradient, `k` = out·kh·kw
/// and `u` = 2^-24, to first order in `u`, and exact on inputs whose products and partial sums
/// are integers below 2^24. The same inputs give the same bits on every run on one GPU and
/// backend. As in the forward pass, a NaN or infinite weight meets 0 at a tap no window
/// covers, where the CPU reference skips the tap, so its input gradients are NaN.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::kernels::{ConvDims, ConvInputGrad};
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let mut cache = PipelineCache::new(&gpu);
///     let mut pool = BufferPool::new(&gpu);
///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
///     let grad = ConvInputGrad::new(&gpu, &mut cache).unwrap();
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     // A 1 × 3 image by the kernel [2 1]: the outputs are 2·x0 + x1 and 2·x1 + x2, so an
///     // output gradient of [1 1] gives the input gradient [2 3 1].
///     let dims = ConvDims::new(1, 1, 1, (1, 3), (1, 2));
///     let (w, dy, dx) = (pool.take(8, usage), pool.take(8, usage), pool.take(12, usage));
///     let call = grad.call(&mut steps, &mut pool, dims, &w, &dy, &dx);
///     let mut step = steps.begin();
///     step.upload(w.buffer(), 0, &bytes(&[2.0, 1.0]));
///     step.upload(dy.buffer(), 0, &bytes(&[1.0, 1.0]));
///     grad.record(&mut step, &call);
///     step.read_back(dx.buffer(), 0, 12);
///     assert_eq!(step.finish().bytes(), &bytes(&[2.0, 3.0, 1.0])[..]);
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct ConvInputGrad {
    /// The layout of group 0: the weight and the output gradients (read-only storage), the
    /// input gradients (read-write storage), the GEMM's `Dims` and the `ConvGrad` uniform.
    group: wgpu::BindGroupLayout,
    /// The pipeline on the GEMM's square tile.
    pipeline: wgpu::ComputePipeline,
    /// The pipeline on the GEMM's wide tile, built by `ensure_wide` for the first call that
    /// needs it.
    wide: Option<wgpu::ComputePipeline>,
    /// The pipeline layout both are built with.
    layout: wgpu::PipelineLayout,
}

impl ConvInputGrad {
    /// Builds the input gradient's bind group layout and pipeline on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if the kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// See [`ConvInputGrad`].
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<ConvInputGrad, GpuError> {
        let (group, layout) = conv_layout(gpu, "conv_input_grad", 0);
        let pipeline = cache.get_with_layout(CONV_INPUT_GRAD, "main", &[], &layout)?;
        Ok(ConvInputGrad {
            group,
            pipeline,
            wide: None,
            layout,
        })
    }

    /// Builds this kernel's pipeline on the GEMM's wide tile for `call`, through `cache`, if
    /// `call`'s product is one the wide tile computes ([`Gemm::ensure_wide`](super::Gemm::ensure_wide)
    /// gives the rule) and it is not built yet. Returns whether `call` runs on the wide tile
    /// from now on. `record` uses the wide tile only for a call this has been called for;
    /// without it the call runs on the square tile, with the same bits.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if the wide kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// See [`Gemm::ensure_wide`](super::Gemm::ensure_wide).
    pub fn ensure_wide(
        &mut self,
        cache: &mut PipelineCache,
        call: &ConvCall,
    ) -> Result<bool, GpuError> {
        let d = call.gemm;
        if !wide_allowed(cache) || !grid(d.m, d.n, d.batch).0 {
            return Ok(false);
        }
        if self.wide.is_none() {
            self.wide =
                Some(cache.get_with_layout(CONV_INPUT_GRAD_WIDE, "main", &[], &self.layout)?);
        }
        Ok(true)
    }

    /// Prepares the input gradient of one convolution of shape `dims`: the weight
    /// (`[out][in][kh][kw]`), the output gradients `dy` (`[batch][out][oh][ow]`) and the
    /// input gradients `dx` (`[batch][in][h][w]`, overwritten). Its two uniforms are taken
    /// from `pool` and uploaded in a step of their own, and its bind group is made through
    /// `steps`, so it is counted. Made once, when a model is compiled.
    ///
    /// # Panics
    ///
    /// As [`ConvForward::call`]: if no window fits, or a count, size, stride or dilation is
    /// 0 (an input of size 0 included, which padding alone would let a window fit), a buffer
    /// is shorter than its tensor, a tensor holds 2^32 values or more, `batch` is above
    /// 65,535, or `dx` is the weight or `dy`.
    ///
    /// # Examples
    ///
    /// See [`ConvInputGrad`].
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        dims: ConvDims,
        weight: &PooledBuffer,
        dy: &PooledBuffer,
        dx: &PooledBuffer,
    ) -> ConvCall {
        let size = dims.checked_out_size();
        assert!(
            dims.input.0 > 0 && dims.input.1 > 0,
            "a convolution's input gradient needs an input of at least 1 × 1, not {dims:?}"
        );
        dims.check_buffers(
            &[
                ("the weight", weight, dims.weight_len()),
                ("the output gradient", dy, dims.out_len()),
                ("the input gradient", dx, dims.x_len()),
            ],
            dx,
            &[weight, dy],
        );
        let gemm = dims.grad_gemm();
        let params = upload_params(steps, pool, &gemm, &dims.grad_uniform(size));
        let group = bind(
            steps,
            &self.group,
            "conv_input_grad",
            &[weight, dy, dx, &params[0], &params[1]],
        );
        ConvCall {
            dims,
            gemm,
            params,
            group,
        }
    }

    /// Adds the input gradient `call` to `step`: one dispatch of `ceil(h·w / 64) ×
    /// ceil(in / 64) × batch` workgroups.
    ///
    /// # Panics
    ///
    /// If `h · w` or `in` needs more than 65,535 of the GEMM's tiles (64 a side, or 128
    /// positions across on the wide tile, which at most 32 input channels may use): split
    /// such a convolution into
    /// several calls.
    ///
    /// # Examples
    ///
    /// See [`ConvInputGrad`].
    pub fn record(&self, step: &mut Step<'_>, call: &ConvCall) {
        let d = call.gemm;
        match (grid(d.m, d.n, d.batch), &self.wide) {
            ((true, wide_grid), Some(wide)) => step.dispatch_grid(wide, &call.group, wide_grid),
            _ => step.dispatch_grid(&self.pipeline, &call.group, square_grid(d.m, d.n, d.batch)),
        }
    }
}

/// The convolution's weight gradient: two pipelines and their bind group layouts, built once.
/// [`call`](ConvWeightGrad::call) prepares one gradient on given buffers, made once when a
/// model is compiled, and [`record`](ConvWeightGrad::record) adds it to a step.
///
/// The weight gradient `dw = Σ dy · x` sums over every image's output positions, `batch ·
/// oh · ow` of them, in the CPU reference's order (image, then output row, then column), cut
/// into fixed blocks of [`WEIGHT_GRAD_BLOCK`] consecutive positions. One pass sums each
/// block with `fma` into a partial gradient of its own, a tap in the padding adding 0; a
/// second adds the partials in block order and overwrites the weight gradient. Nothing needs
/// atomics, and the blocks depend only on the shape, so the same inputs give the same bits on
/// every run on one GPU and backend. With `L` the block's length and `B` the blocks, each
/// weight's gradient is within `2(L + B + 2) · u · Σ|dy·x|` of the exact gradient, `u` =
/// 2^-24, to first order in `u`, and exact on inputs whose products and partial sums are
/// integers below 2^24. A NaN or infinite output gradient meets 0 at a tap in the padding,
/// where the CPU reference skips the tap, so its weight gradients are NaN.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::kernels::{ConvDims, ConvWeightGrad};
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let mut cache = PipelineCache::new(&gpu);
///     let mut pool = BufferPool::new(&gpu);
///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
///     let grad = ConvWeightGrad::new(&gpu, &mut cache).unwrap();
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
///     let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     // A 1 × 3 image [1 2 3] by a 1 × 2 kernel: the outputs are w0·x0 + w1·x1 and
///     // w0·x1 + w1·x2, so an output gradient of [1 1] gives the weight gradient [3 5].
///     let dims = ConvDims::new(1, 1, 1, (1, 3), (1, 2));
///     let (x, dy, dw) = (pool.take(12, usage), pool.take(8, usage), pool.take(8, usage));
///     let call = grad.call(&mut steps, &mut pool, dims, &x, &dy, &dw);
///     assert_eq!((call.blocks(), call.block_len()), (1, 2));
///     let mut step = steps.begin();
///     step.upload(x.buffer(), 0, &bytes(&[1.0, 2.0, 3.0]));
///     step.upload(dy.buffer(), 0, &bytes(&[1.0, 1.0]));
///     grad.record(&mut step, &call);
///     step.read_back(dw.buffer(), 0, 8);
///     assert_eq!(step.finish().bytes(), &bytes(&[3.0, 5.0])[..]);
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct ConvWeightGrad {
    /// The layout of the first pass's group 0: the output gradients and the input images
    /// (read-only storage), the partial gradients (read-write storage), the GEMM's `Dims` and
    /// the `ConvWeightGrad` uniform.
    group: wgpu::BindGroupLayout,
    /// The first pass: each block's partial gradient.
    pipeline: wgpu::ComputePipeline,
    /// The first pass on the GEMM's wide tile, built by `ensure_wide` for the first call
    /// that needs it.
    wide: Option<wgpu::ComputePipeline>,
    /// The pipeline layout both first passes are built with.
    layout: wgpu::PipelineLayout,
    /// The layout of the merge's group 0: the partial gradients (read-only storage), the
    /// weight gradient (read-write storage) and the GEMM's `Dims`.
    merge_group: wgpu::BindGroupLayout,
    /// The second pass: the partials added in block order.
    merge: wgpu::ComputePipeline,
}

impl ConvWeightGrad {
    /// Builds the weight gradient's two bind group layouts and pipelines on `gpu`, through
    /// `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::ConvWeightGrad;
    /// use cavitation_gpu::{Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut cache = PipelineCache::new(&gpu);
    ///     ConvWeightGrad::new(&gpu, &mut cache).unwrap();
    ///     assert_eq!(cache.pipelines_built(), 2);
    /// }
    /// ```
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<ConvWeightGrad, GpuError> {
        let (group, pipeline_layout) = conv_layout(gpu, "conv_weight_grad", 0);
        // Consecutive threads fill the B tile along k, consecutive output positions, so
        // their reads of an image row are adjacent.
        let pipeline = cache.get_with_layout(
            CONV_WEIGHT_GRAD,
            "main",
            &[("TRANS_B", 1.0)],
            &pipeline_layout,
        )?;

        let (merge_group, merge_layout) = layout(
            gpu,
            "conv_weight_merge",
            &[Kind::Read, Kind::Write, Kind::Uniform(DIMS_BYTES)],
        );
        let merge = cache.get_with_layout(
            CONV_WEIGHT_MERGE,
            "main",
            &[("WORKGROUP", f64::from(MERGE_WORKGROUP))],
            &merge_layout,
        )?;
        Ok(ConvWeightGrad {
            group,
            pipeline,
            wide: None,
            layout: pipeline_layout,
            merge_group,
            merge,
        })
    }

    /// Builds the first pass's pipeline on the GEMM's wide tile for `call`, through `cache`,
    /// if `call`'s product is one the wide tile computes ([`Gemm::ensure_wide`](super::Gemm::ensure_wide)
    /// gives the rule) and it is not built yet. Returns whether `call` runs on the wide tile
    /// from now on. `record` uses the wide tile only for a call this has been called for;
    /// without it the call runs on the square tile, with the same bits.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if the wide kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// See [`Gemm::ensure_wide`](super::Gemm::ensure_wide).
    pub fn ensure_wide(
        &mut self,
        cache: &mut PipelineCache,
        call: &ConvWeightGradCall,
    ) -> Result<bool, GpuError> {
        let d = call.gemm;
        if !wide_allowed(cache) || !grid(d.m, d.n, d.batch).0 {
            return Ok(false);
        }
        if self.wide.is_none() {
            self.wide = Some(cache.get_with_layout(
                CONV_WEIGHT_GRAD_WIDE,
                "main",
                &[("TRANS_B", 1.0)],
                &self.layout,
            )?);
        }
        Ok(true)
    }

    /// Prepares the weight gradient of one convolution of shape `dims`: the input images `x`
    /// (`[batch][in][h][w]`), the output gradients `dy` (`[batch][out][oh][ow]`) and the
    /// weight gradient `dw` (`[out][in][kh][kw]`, overwritten). Its two uniforms and its
    /// buffer of partial gradients, one weight gradient per block, are taken from `pool`, the
    /// uniforms uploaded in a step of their own, and its bind groups are made through
    /// `steps`, so they are counted. Made once, when a model is compiled.
    ///
    /// # Panics
    ///
    /// As [`ConvForward::call`]: if no window fits, or a count, size, stride or dilation is
    /// 0, a buffer is shorter than its tensor, a tensor holds 2^32 values or more, `batch` is
    /// above 65,535, or `dw` is `x` or `dy`. Also if the partial gradients would hold 2^32
    /// values or more, or the blocks are more than 65,535.
    ///
    /// # Examples
    ///
    /// See [`ConvWeightGrad`].
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        dims: ConvDims,
        x: &PooledBuffer,
        dy: &PooledBuffer,
        dw: &PooledBuffer,
    ) -> ConvWeightGradCall {
        let size = dims.checked_out_size();
        dims.check_buffers(
            &[
                ("the input", x, dims.x_len()),
                ("the output gradient", dy, dims.out_len()),
                ("the weight gradient", dw, dims.weight_len()),
            ],
            dw,
            &[x, dy],
        );
        // A block's positions are counted in 32 bits, the last block's past the end too.
        let positions = dims.positions(size);
        assert!(
            positions + u64::from(WEIGHT_GRAD_BLOCK) <= u64::from(u32::MAX),
            "{dims:?} has {positions} output positions, more than the shader's 32-bit indices reach"
        );
        let gemm = dims.weight_grad_gemm(size);
        let partial_len = u64::from(gemm.batch) * dims.weight_len();
        assert!(
            partial_len <= u64::from(u32::MAX),
            "{dims:?} needs {partial_len} partial gradients, more than the shader's 32-bit indices reach"
        );
        assert!(
            gemm.batch <= MAX_WORKGROUPS_PER_DIMENSION,
            "{dims:?} needs {} blocks of {WEIGHT_GRAD_BLOCK} positions, more than {MAX_WORKGROUPS_PER_DIMENSION}",
            gemm.batch
        );
        let partials = pool.take(4 * partial_len, wgpu::BufferUsages::STORAGE);
        let params = upload_params(steps, pool, &gemm, &dims.weight_grad_uniform(size));
        let group = bind(
            steps,
            &self.group,
            "conv_weight_grad",
            &[dy, x, &partials, &params[0], &params[1]],
        );
        let merge_group = bind(
            steps,
            &self.merge_group,
            "conv_weight_merge",
            &[&partials, dw, &params[0]],
        );
        ConvWeightGradCall {
            dims,
            gemm,
            params,
            partials,
            group,
            merge_group,
        }
    }

    /// Adds the weight gradient `call` to `step`: one dispatch of `ceil(in·kh·kw / 64) ×
    /// ceil(out / 64) × blocks` workgroups for the partials, then one thread per weight to
    /// add them.
    ///
    /// # Panics
    ///
    /// If `in · kh · kw` or `out` needs more than 65,535 of the GEMM's tiles (64 a side, or
    /// 128 weights across on the wide tile, which at most 32 output channels may use).
    ///
    /// # Examples
    ///
    /// See [`ConvWeightGrad`].
    pub fn record(&self, step: &mut Step<'_>, call: &ConvWeightGradCall) {
        let d = call.gemm;
        match (grid(d.m, d.n, d.batch), &self.wide) {
            ((true, wide_grid), Some(wide)) => step.dispatch_grid(wide, &call.group, wide_grid),
            _ => step.dispatch_grid(&self.pipeline, &call.group, square_grid(d.m, d.n, d.batch)),
        }
        step.dispatch_split(
            &self.merge,
            &call.merge_group,
            (d.m * d.n).div_ceil(MERGE_WORKGROUP),
        );
    }
}

/// One prepared weight gradient: its shape, its two uniforms, its partial gradients and its
/// two passes' bind groups, from [`ConvWeightGrad::call`].
///
/// # Examples
///
/// See [`ConvWeightGrad`].
#[derive(Debug)]
pub struct ConvWeightGradCall {
    /// The shape.
    dims: ConvDims,
    /// The batched GEMM of the first pass, one product per block.
    gemm: GemmDims,
    /// The GEMM's `Dims` uniform and the `ConvWeightGrad` uniform, from the pool.
    params: [PooledBuffer; 2],
    /// One partial weight gradient per block, from the pool.
    partials: PooledBuffer,
    /// The first pass's bind group.
    group: wgpu::BindGroup,
    /// The merge's bind group.
    merge_group: wgpu::BindGroup,
}

impl ConvWeightGradCall {
    /// The shape this gradient was prepared for.
    ///
    /// # Examples
    ///
    /// See [`ConvWeightGrad`].
    pub fn dims(&self) -> ConvDims {
        self.dims
    }

    /// The blocks the positions are cut into, `B` in [`ConvWeightGrad`]'s bound.
    ///
    /// # Examples
    ///
    /// See [`ConvWeightGrad`].
    pub fn blocks(&self) -> u32 {
        self.gemm.batch
    }

    /// The positions in each block but the last, which may have fewer: [`WEIGHT_GRAD_BLOCK`],
    /// or every position if there are fewer. `L` in [`ConvWeightGrad`]'s bound.
    ///
    /// # Examples
    ///
    /// See [`ConvWeightGrad`].
    pub fn block_len(&self) -> u32 {
        self.gemm.k
    }

    /// Gives this gradient's uniforms and partial gradients back to `pool`, once it is no
    /// longer recorded.
    ///
    /// # Examples
    ///
    /// See [`ConvWeightGrad`].
    pub fn release(self, pool: &mut BufferPool) {
        let [a, b] = self.params;
        pool.give(a);
        pool.give(b);
        pool.give(self.partials);
    }
}
