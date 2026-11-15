//! The 2-D convolution forward pass on the GPU, as an implicit GEMM through the tiled GEMM's
//! core, with stride, padding, dilation and an optional bias. Its CPU twin is
//! `cavitation_core::kernels::conv::conv2d_reference`: every output is within
//! `2(k + 3) · u · (Σ|w·x| + |bias|)` of the f64 convolution of the same f32 inputs, `k` =
//! in·kh·kw, and with small-integer inputs it is exact.

use super::gemm::{DIMS_BYTES, GemmDims, TILE};
use crate::{BufferPool, Gpu, GpuError, PipelineCache, PooledBuffer, Shader, Step, StepEncoder};

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

/// The bytes of the shader's `Conv` uniform: sixteen 4-byte fields.
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
        let mut bytes = [0u8; CONV_BYTES as usize];
        let (chunks, _) = bytes.as_chunks_mut::<4>();
        for (chunk, word) in chunks.iter_mut().zip(words) {
            *chunk = word.to_le_bytes();
        }
        bytes
    }
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
    /// The pipeline.
    pipeline: wgpu::ComputePipeline,
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
        let device = gpu.device();
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
        let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("conv_forward"),
            entries: &[
                buffer(0, read, 0),
                buffer(1, read, 0),
                buffer(2, wgpu::BufferBindingType::Storage { read_only: false }, 0),
                buffer(3, uniform, DIMS_BYTES),
                buffer(4, uniform, CONV_BYTES),
                buffer(5, read, 0),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("conv_forward"),
            bind_group_layouts: &[Some(&group)],
            immediate_size: 0,
        });
        let pipeline = cache.get_with_layout(CONV_FORWARD, "main", &[], &layout)?;
        Ok(ConvForward { group, pipeline })
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
        let size = dims
            .out_size()
            .filter(|_| dims.batch > 0 && dims.in_channels > 0 && dims.out_channels > 0);
        let Some((oh, ow)) = size else {
            panic!("a convolution needs every count above 0 and a window that fits, not {dims:?}");
        };
        assert!(
            dims.batch <= 65_535,
            "a convolution takes at most 65,535 images per call, not {}",
            dims.batch
        );
        let mut tensors = vec![
            ("the input", x, dims.x_len()),
            ("the weight", weight, dims.weight_len()),
            ("the output", out, dims.out_len()),
        ];
        if let Some(b) = bias {
            tensors.push(("the bias", b, u64::from(dims.out_channels)));
        }
        for (name, buffer, len) in tensors {
            assert!(
                len <= u64::from(u32::MAX),
                "{name} holds {len} values for {dims:?}, more than the shader's 32-bit indices reach"
            );
            assert!(
                buffer.len() >= 4 * len,
                "{name} needs {len} f32s ({} bytes) for {dims:?}, but its buffer has {} bytes",
                4 * len,
                buffer.len()
            );
        }
        let output = out.buffer();
        assert!(
            output != x.buffer()
                && output != weight.buffer()
                && bias.is_none_or(|b| output != b.buffer()),
            "a convolution cannot write its output into a buffer it reads"
        );
        let gemm = dims.gemm((oh, ow));
        let usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let (gemm_params, conv_params) =
            (pool.take(DIMS_BYTES, usage), pool.take(CONV_BYTES, usage));
        let mut step = steps.begin();
        step.upload(gemm_params.buffer(), 0, &gemm.uniform());
        step.upload(conv_params.buffer(), 0, &dims.uniform(ow, bias.is_some()));
        let _ = step.finish();
        fn entry(binding: u32, buffer: &PooledBuffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.binding(),
            }
        }
        // Without a bias the shader never reads binding 5, so the weight stands in for it.
        let group = steps.bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv_forward"),
            layout: &self.group,
            entries: &[
                entry(0, weight),
                entry(1, x),
                entry(2, out),
                entry(3, &gemm_params),
                entry(4, &conv_params),
                entry(5, bias.unwrap_or(weight)),
            ],
        });
        ConvCall {
            dims,
            gemm,
            params: [gemm_params, conv_params],
            group,
        }
    }

    /// Adds the convolution `call` to `step`: one dispatch of `ceil(oh·ow / 64) ×
    /// ceil(out / 64) × batch` workgroups.
    ///
    /// # Panics
    ///
    /// If `oh · ow` or `out` needs more than 65,535 tiles of 64: split such a convolution
    /// into several calls.
    ///
    /// # Examples
    ///
    /// See [`ConvForward`].
    pub fn record(&self, step: &mut Step<'_>, call: &ConvCall) {
        let d = call.gemm;
        step.dispatch_grid(
            &self.pipeline,
            &call.group,
            [d.n.div_ceil(TILE), d.m.div_ceil(TILE), d.batch],
        );
    }
}

/// One prepared convolution: its shape, its two uniforms and its bind group, from
/// [`ConvForward::call`].
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
    /// The GEMM's `Dims` uniform and the `Conv` uniform, from the pool.
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
