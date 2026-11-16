//! The element-wise GPU kernels: add, multiply and scale; ReLU and leaky ReLU, tanh, the
//! sigmoid and GELU (both forms), forward and backward; and dropout and neuron masks with a
//! bit-packed mask.
//! Their CPU twins are `cavitation-core`'s: the one-rounding operations match it bit for bit
//! on normal values, and tanh, the sigmoid and GELU are the CPU kernels' formulas ported
//! operation for operation, within the bounds their tests state.

use std::num::NonZeroU64;

use crate::{BufferPool, Gpu, GpuError, PipelineCache, PooledBuffer, Shader, Step, StepEncoder};

/// The element-wise shader, one entry point per operation.
const ELEMENTWISE: Shader = Shader {
    name: "elementwise",
    source: include_str!("elementwise.wgsl"),
};

/// The dropout shader, whose mask binding holds `u32` words.
const DROPOUT: Shader = Shader {
    name: "dropout",
    source: include_str!("dropout.wgsl"),
};

/// Threads per workgroup in both shaders.
const WORKGROUP: u32 = 256;

/// The bytes of the shaders' `Params` uniform: the length, `alpha`, `scale` and padding.
const PARAMS_BYTES: u64 = 16;

/// One element-wise operation. `a` is its first input, `b` its second (the gradient flowing
/// back, for a backward pass; the mask, for dropout), and the result goes to `out`.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::kernels::Elementwise;
///
/// assert!(Elementwise::Add.takes_b());
/// assert!(!Elementwise::Relu.takes_b());
/// assert!(Elementwise::Dropout.takes_b()); // the mask
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Elementwise {
    /// `a + b`.
    Add,
    /// `a · b`.
    Mul,
    /// `alpha · a`.
    Scale,
    /// `max(a, 0)`: 0 at or below 0 (so −0 gives +0), NaN for NaN.
    Relu,
    /// `b · relu'(a)`: `b` above 0, 0 at or below, NaN for a NaN `a`.
    ReluBackward,
    /// `a` above 0, `alpha · a` at or below, NaN for NaN.
    LeakyRelu,
    /// `b · (1 above 0, alpha at or below)`, NaN for a NaN `a`.
    LeakyReluBackward,
    /// `tanh(a)`.
    Tanh,
    /// `b · (1 − a²)`, with `a` the forward output.
    TanhBackward,
    /// `1 / (1 + e^-a)`.
    Sigmoid,
    /// `b · a · (1 − a)`, with `a` the forward output.
    SigmoidBackward,
    /// GELU's tanh form, `0.5 · a · (1 + tanh(√(2/π) · (a + 0.044715 · a³)))`.
    GeluTanh,
    /// `b` times the derivative of GELU's tanh form at the forward input `a`.
    GeluTanhBackward,
    /// GELU's erf form, `0.5 · a · (1 + erf(a / √2))`.
    GeluErf,
    /// `b` times `Φ(a) + a · φ(a)`, the derivative of GELU's erf form at the input `a`.
    GeluErfBackward,
    /// `a · m`, `m` = `scale` where the mask in `b` has the element's bit set and 0 where it is
    /// clear (so a dropped −3 is −0 and a dropped NaN stays NaN):
    /// dropout forward, and its backward pass on the gradient with the same mask.
    Dropout,
    /// `a` where the mask in `b` has the element's bit set and `+0` where it is clear, selected
    /// rather than multiplied (so a cleared −3 or NaN is `+0`), as `Tape::keep_where` computes
    /// it: a neuron mask forward, and its backward pass on the gradient with the same mask.
    KeepWhere,
}

impl Elementwise {
    /// Every operation, in declaration order.
    pub const ALL: [Elementwise; 17] = [
        Elementwise::Add,
        Elementwise::Mul,
        Elementwise::Scale,
        Elementwise::Relu,
        Elementwise::ReluBackward,
        Elementwise::LeakyRelu,
        Elementwise::LeakyReluBackward,
        Elementwise::Tanh,
        Elementwise::TanhBackward,
        Elementwise::Sigmoid,
        Elementwise::SigmoidBackward,
        Elementwise::GeluTanh,
        Elementwise::GeluTanhBackward,
        Elementwise::GeluErf,
        Elementwise::GeluErfBackward,
        Elementwise::Dropout,
        Elementwise::KeepWhere,
    ];

    /// Whether the operation reads `b`: binary operations, backward passes, dropout and
    /// [`KeepWhere`](Elementwise::KeepWhere) (the mask).
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::Elementwise;
    ///
    /// assert!(Elementwise::TanhBackward.takes_b());
    /// assert!(!Elementwise::GeluErf.takes_b());
    /// ```
    pub fn takes_b(self) -> bool {
        !matches!(
            self,
            Elementwise::Scale
                | Elementwise::Relu
                | Elementwise::LeakyRelu
                | Elementwise::Tanh
                | Elementwise::Sigmoid
                | Elementwise::GeluTanh
                | Elementwise::GeluErf
        )
    }

    /// The shader and entry point that compute it.
    fn entry(self) -> (Shader, &'static str) {
        let e = match self {
            Elementwise::Add => "add",
            Elementwise::Mul => "mul",
            Elementwise::Scale => "scale",
            Elementwise::Relu => "relu",
            Elementwise::ReluBackward => "relu_backward",
            Elementwise::LeakyRelu => "leaky_relu",
            Elementwise::LeakyReluBackward => "leaky_relu_backward",
            Elementwise::Tanh => "tanh_forward",
            Elementwise::TanhBackward => "tanh_backward",
            Elementwise::Sigmoid => "sigmoid_forward",
            Elementwise::SigmoidBackward => "sigmoid_backward",
            Elementwise::GeluTanh => "gelu_tanh_forward",
            Elementwise::GeluTanhBackward => "gelu_tanh_backward",
            Elementwise::GeluErf => "gelu_erf_forward",
            Elementwise::GeluErfBackward => "gelu_erf_backward",
            Elementwise::Dropout => return (DROPOUT, "dropout"),
            Elementwise::KeepWhere => return (DROPOUT, "keep_where"),
        };
        (ELEMENTWISE, e)
    }
}

/// The element-wise kernels: one pipeline per [`Elementwise`] operation, all on one bind
/// group layout, built once. [`call`](ElementwiseKernels::call) prepares one operation on
/// given buffers, once, and [`record`](ElementwiseKernels::record) adds it to a step.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::kernels::{Elementwise, ElementwiseKernels};
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let mut cache = PipelineCache::new(&gpu);
///     let mut pool = BufferPool::new(&gpu);
///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
///     let kernels = ElementwiseKernels::new(&gpu, &mut cache).unwrap();
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
///     let (x, y) = (pool.take(16, usage), pool.take(16, usage));
///     let relu = kernels.call(&mut steps, &mut pool, Elementwise::Relu, 4, 0.0, 0.0, &x, None, &y);
///     let bytes = |v: [f32; 4]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
///     let mut step = steps.begin();
///     step.upload(x.buffer(), 0, &bytes([-1.0, 2.0, -0.0, 3.5]));
///     kernels.record(&mut step, &relu);
///     step.read_back(y.buffer(), 0, 16);
///     assert_eq!(step.finish().bytes(), &bytes([0.0, 2.0, 0.0, 3.5])[..]);
///     relu.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct ElementwiseKernels {
    /// The layout of group 0: `a`, `b` (read-only storage), `out` (read-write storage) and
    /// the `Params` uniform.
    group: wgpu::BindGroupLayout,
    /// The pipelines, in the order of [`Elementwise::ALL`].
    pipelines: Vec<wgpu::ComputePipeline>,
}

impl ElementwiseKernels {
    /// Builds the bind group layout and every operation's pipeline on `gpu`, through
    /// `cache`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::{Elementwise, ElementwiseKernels};
    /// use cavitation_gpu::{Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut cache = PipelineCache::new(&gpu);
    ///     ElementwiseKernels::new(&gpu, &mut cache).unwrap();
    ///     assert_eq!(cache.pipelines_built(), Elementwise::ALL.len());
    /// }
    /// ```
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<ElementwiseKernels, GpuError> {
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
            label: Some("elementwise"),
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
                        min_binding_size: NonZeroU64::new(PARAMS_BYTES),
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("elementwise"),
            bind_group_layouts: &[Some(&group)],
            immediate_size: 0,
        });
        let pipelines = Elementwise::ALL
            .iter()
            .map(|op| {
                let (shader, entry) = op.entry();
                cache.get_with_layout(shader, entry, &[], &layout)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ElementwiseKernels { group, pipelines })
    }

    /// Prepares `op` over `n` elements of `a` (and `b`, when [`Elementwise::takes_b`])
    /// into `out`, with `alpha` (for [`Elementwise::Scale`] and the leaky ReLUs) and `scale`
    /// (for [`Elementwise::Dropout`], `1 / f32(1 − p)` as the CPU rounds it): its uniform, taken from
    /// `pool` and uploaded in a step of its own, and its bind group. Made once, when a model
    /// is compiled. An operation that does not read `b` binds `a` in its place. `out` must be
    /// a different buffer from `a` and `b`: wgpu does not let one buffer be read and written in
    /// one dispatch.
    ///
    /// # Panics
    ///
    /// If `n` is 0; if `out` is `a` or `b`; if `a` or `out` holds fewer than `n` f32s, or `b` fewer than `n` (or, as a
    /// mask, fewer than `ceil(n / 32)` words); or if `op` takes `b` and none is given.
    ///
    /// # Examples
    ///
    /// See [`ElementwiseKernels`].
    #[allow(clippy::too_many_arguments)]
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        op: Elementwise,
        n: u32,
        alpha: f32,
        scale: f32,
        a: &PooledBuffer,
        b: Option<&PooledBuffer>,
        out: &PooledBuffer,
    ) -> ElementwiseCall {
        assert!(
            n > 0,
            "an element-wise operation needs at least one element"
        );
        let bytes = 4 * u64::from(n);
        assert!(
            a.len() >= bytes && out.len() >= bytes,
            "{op:?} over {n} elements needs {bytes} bytes in a and out, not {} and {}",
            a.len(),
            out.len()
        );
        let b = if op.takes_b() {
            let b = b.unwrap_or_else(|| panic!("{op:?} reads b, and none was given"));
            let need = if matches!(op, Elementwise::Dropout | Elementwise::KeepWhere) {
                4 * u64::from(n.div_ceil(32))
            } else {
                bytes
            };
            assert!(
                b.len() >= need,
                "{op:?} over {n} elements needs {need} bytes in b, not {}",
                b.len()
            );
            b
        } else {
            a
        };
        assert!(
            out.buffer() != a.buffer() && out.buffer() != b.buffer(),
            "{op:?} cannot write to a buffer it reads: give out its own buffer"
        );
        let pipeline = Elementwise::ALL
            .iter()
            .position(|&o| o == op)
            .expect("every operation has a pipeline");
        let params = pool.take(
            PARAMS_BYTES,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let mut uniform = [0u8; PARAMS_BYTES as usize];
        uniform[0..4].copy_from_slice(&n.to_le_bytes());
        uniform[4..8].copy_from_slice(&alpha.to_le_bytes());
        uniform[8..12].copy_from_slice(&scale.to_le_bytes());
        let mut step = steps.begin();
        step.upload(params.buffer(), 0, &uniform);
        let _ = step.finish();
        let group = steps.bind_group(&wgpu::BindGroupDescriptor {
            label: Some("elementwise"),
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
                    resource: out.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.binding(),
                },
            ],
        });
        ElementwiseCall {
            op,
            pipeline,
            n,
            params,
            group,
        }
    }

    /// Adds the prepared operation `call` to `step`: `ceil(n / 256)` workgroups, split over
    /// two dimensions when there are more than one dimension holds.
    ///
    /// # Examples
    ///
    /// See [`ElementwiseKernels`].
    pub fn record(&self, step: &mut Step<'_>, call: &ElementwiseCall) {
        step.dispatch_split(
            &self.pipelines[call.pipeline],
            &call.group,
            call.n.div_ceil(WORKGROUP),
        );
    }
}

/// One prepared element-wise operation: which, over how many elements, its uniform and its
/// bind group, from [`ElementwiseKernels::call`].
///
/// # Examples
///
/// See [`ElementwiseKernels`].
#[derive(Debug)]
pub struct ElementwiseCall {
    /// The operation.
    op: Elementwise,
    /// Its pipeline's index in [`ElementwiseKernels`]'s list.
    pipeline: usize,
    /// The number of elements.
    n: u32,
    /// The `Params` uniform, from the pool.
    params: PooledBuffer,
    /// The bind group.
    group: wgpu::BindGroup,
}

impl ElementwiseCall {
    /// The operation this call was prepared for.
    ///
    /// # Examples
    ///
    /// See [`ElementwiseKernels`].
    pub fn op(&self) -> Elementwise {
        self.op
    }

    /// Gives the call's uniform buffer back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`ElementwiseKernels`].
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.params);
    }
}
