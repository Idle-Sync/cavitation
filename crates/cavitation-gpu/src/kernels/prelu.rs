//! PReLU on the GPU, as `cavitation_core`'s `Tape::prelu` computes it: `x` above 0 and
//! `slope · x` elsewhere, with one slope per column of a `[rows][channels]` batch or one
//! slope shared by every value; the input's gradient, one thread per value; and the slopes'
//! gradient, `Σ g·x` over each channel's values not above 0, in the fixed tree of
//! [`ChannelSums`](super::ChannelSums).

use super::ChannelDims;
use super::bindings::{Kind, bind, check_distinct, check_len, layout, upload_uniform, word_bytes};
use super::reduce::{ChannelSumsCall, SumsTo, TermSums, partials_pass};
use crate::{BufferPool, Gpu, GpuError, PipelineCache, PooledBuffer, Shader, Step, StepEncoder};

/// The kernels' name in panic messages.
const NAME: &str = "prelu";

/// Threads per workgroup, passed as the `WORKGROUP` override.
const WORKGROUP: u32 = 256;

/// Bytes of the element-wise kernels' uniform: the values, the channels and two words of
/// padding.
const UNIFORM_BYTES: u64 = 16;

/// The element-wise kernels, one thread per value, numbered over a split grid. Bindings: the
/// output, the uniform, `x` (the forward input), the slopes and `g` (the gradient flowing
/// back; bound to `x` and unused forward). Value `i` takes slope `i % channels`. NaN is
/// tested by its bits, never with a comparison, and passes through as the leaky ReLU's
/// kernel passes it.
const ELEMENTWISE: Shader = Shader {
    name: "prelu",
    source: "
        override WORKGROUP: u32;

        struct Dims { n: u32, channels: u32, pad0: u32, pad1: u32 }

        @group(0) @binding(0) var<storage, read_write> out: array<f32>;
        @group(0) @binding(1) var<uniform> d: Dims;
        @group(0) @binding(2) var<storage, read> x: array<f32>;
        @group(0) @binding(3) var<storage, read> slope: array<f32>;
        @group(0) @binding(4) var<storage, read> g: array<f32>;

        // Whether v is NaN, from its bits.
        fn is_nan(v: f32) -> bool {
            return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u;
        }

        // The value this thread computes, numbered over a split grid.
        fn element(w: vec3<u32>, n: vec3<u32>, l: u32) -> u32 {
            return (w.x + w.y * n.x) * WORKGROUP + l;
        }

        // x above 0, slope · x at or below; NaN stays NaN.
        @compute @workgroup_size(WORKGROUP)
        fn forward(
            @builtin(workgroup_id) w: vec3<u32>,
            @builtin(num_workgroups) n: vec3<u32>,
            @builtin(local_invocation_index) l: u32,
        ) {
            let i = element(w, n, l);
            if (i >= d.n) { return; }
            let v = x[i];
            let s = slope[i % d.channels];
            out[i] = select(select(s * v, v, v > 0.0), v, is_nan(v));
        }

        // g · (1 above 0, the slope at or below, NaN for a NaN x).
        @compute @workgroup_size(WORKGROUP)
        fn input_grad(
            @builtin(workgroup_id) w: vec3<u32>,
            @builtin(num_workgroups) n: vec3<u32>,
            @builtin(local_invocation_index) l: u32,
        ) {
            let i = element(w, n, l);
            if (i >= d.n) { return; }
            let v = x[i];
            let s = slope[i % d.channels];
            out[i] = g[i] * select(select(s, 1.0, v > 0.0), v, is_nan(v));
        }
    ",
};

/// The slopes' gradient, first pass: the term of value `i` is `g·x` where `x` is not above 0
/// (a NaN `x` included, which carries into the sum) and 0 elsewhere.
const SLOPE_PARTIALS: Shader = Shader {
    name: "prelu slope partials",
    source: partials_pass!(
        "
        @group(0) @binding(0) var<storage, read> x: array<f32>;
        @group(0) @binding(3) var<storage, read> g: array<f32>;
        ",
        "
        fn term(i: u32, c: u32) -> f32 {
            let v = x[i];
            let above = v > 0.0 && (bitcast<u32>(v) & 0x7fffffffu) <= 0x7f800000u;
            return select(g[i] * v, 0.0, above);
        }
        "
    ),
};

/// One element-wise PReLU operation of [`PreluKernels`].
///
/// # Examples
///
/// ```
/// use cavitation_gpu::kernels::Prelu;
///
/// assert!(!Prelu::Forward.takes_g());
/// assert!(Prelu::InputGrad.takes_g());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Prelu {
    /// `x` above 0, `slope · x` elsewhere.
    Forward,
    /// `g · (1 above 0, slope elsewhere)`, the input's gradient.
    InputGrad,
}

impl Prelu {
    /// Every operation, in declaration order.
    ///
    /// ```
    /// use cavitation_gpu::kernels::Prelu;
    ///
    /// assert_eq!(Prelu::ALL.len(), 2);
    /// ```
    pub const ALL: [Prelu; 2] = [Prelu::Forward, Prelu::InputGrad];

    /// Whether the operation reads the gradient `g`: the input's gradient.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::Prelu;
    ///
    /// assert!(Prelu::InputGrad.takes_g());
    /// ```
    pub fn takes_g(self) -> bool {
        matches!(self, Prelu::InputGrad)
    }

    /// The entry point that computes it.
    fn entry(self) -> &'static str {
        match self {
            Prelu::Forward => "forward",
            Prelu::InputGrad => "input_grad",
        }
    }
}

/// PReLU on the GPU, forward and backward, for a `[rows][channels]` batch whose value `i`
/// takes slope `i % channels`: one slope per column (`channels` the width), or one shared
/// slope (`channels = 1`, `rows` every value). [`call`](PreluKernels::call) prepares the
/// forward pass or the input's gradient on given buffers, once, and
/// [`record`](PreluKernels::record) adds it to a step;
/// [`slope_grad_call`](PreluKernels::slope_grad_call) and
/// [`record_slope_grad`](PreluKernels::record_slope_grad) do the same for the slopes'
/// gradient.
///
/// - **Forward** and **input gradient:** one multiplication per value, as the tape's, so on
///   normal values and products they are the CPU's bits; a subnormal is flushed. NaN stays
///   NaN, and the derivative is the slope at exactly 0, as on the CPU.
/// - **Slopes' gradient:** each channel's `Σ g·x` over its values not above 0, each term one
///   correctly rounded product (a NaN `x` makes the sum NaN, as on the CPU), summed in
///   [`ChannelSums`](super::ChannelSums)' fixed tree rather than the CPU's order: within
///   `2(L + 2)·2⁻²⁴·Σ|g·x|` of the exact sum of the same `f32` terms, for `L = rows` terms per
///   channel, plus `3L · 2⁻¹²⁶` for the terms and partial sums a GPU may flush. On terms whose
///   partial sums are exact it is the CPU's sum bit for bit.
///
/// The tree depends on the shape alone, so two runs give the same bits. `tests/prelu.rs`
/// checks each.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_gpu::kernels::{Prelu, PreluKernels};
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let kernels = PreluKernels::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
///     let mut pool = BufferPool::new(&gpu);
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     let (x, slope, y) = (pool.take(16, usage), pool.take(8, usage), pool.take(16, usage));
///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
///     let call =
///         kernels.call(&mut steps, &mut pool, Prelu::Forward, (2, 2), &x, &slope, None, &y);
///     let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
///     let mut step = steps.begin();
///     step.upload(x.buffer(), 0, &bytes(&[-4.0, 2.0, 3.0, -8.0]));
///     step.upload(slope.buffer(), 0, &bytes(&[0.25, 0.5]));
///     kernels.record(&mut step, &call);
///     step.read_back(y.buffer(), 0, 16);
///     assert_eq!(step.finish().bytes(), &bytes(&[-1.0, 2.0, 3.0, -4.0])[..]);
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct PreluKernels {
    /// The layout of the element-wise kernels' group 0: the output, the uniform, `x`, the
    /// slopes and `g`.
    group: wgpu::BindGroupLayout,
    /// The element-wise pipelines, in the order of [`Prelu::ALL`].
    pipelines: Vec<wgpu::ComputePipeline>,
    /// The slopes' gradient: the two passes of a per-channel sum of `g·x`.
    slope: TermSums,
}

impl PreluKernels {
    /// Builds every pipeline on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// See [`PreluKernels`].
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<PreluKernels, GpuError> {
        use Kind::{Read, Uniform, Write};
        let (group, pipeline_layout) = layout(
            gpu,
            NAME,
            &[Write, Uniform(UNIFORM_BYTES), Read, Read, Read],
        );
        let workgroup = [("WORKGROUP", f64::from(WORKGROUP))];
        let pipelines = Prelu::ALL
            .iter()
            .map(|op| cache.get_with_layout(ELEMENTWISE, op.entry(), &workgroup, &pipeline_layout))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PreluKernels {
            group,
            pipelines,
            slope: TermSums::new(gpu, cache, SLOPE_PARTIALS, 1)?,
        })
    }

    /// Prepares `op` over `rows × channels` values of `x`, with `channels` slopes (and `g`,
    /// when [`Prelu::takes_g`]), into `out`, overwritten: its uniform, taken from `pool` and
    /// uploaded in a step of its own, and its bind group. Made once, when a model is
    /// compiled. The forward pass binds `x` in place of `g`.
    ///
    /// # Panics
    ///
    /// If `rows` or `channels` is 0 or `rows × channels` does not fit a `u32`, a buffer is
    /// shorter than its values, `op` reads `g` and none is given, or `out` is one of the
    /// inputs.
    ///
    /// # Examples
    ///
    /// See [`PreluKernels`].
    #[allow(clippy::too_many_arguments)]
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        op: Prelu,
        (rows, channels): (u32, u32),
        x: &PooledBuffer,
        slope: &PooledBuffer,
        g: Option<&PooledBuffer>,
        out: &PooledBuffer,
    ) -> PreluCall {
        let n = check_shape(rows, channels);
        check_len(NAME, "input", x, u64::from(n));
        check_len(NAME, "slopes", slope, u64::from(channels));
        check_len(NAME, "output", out, u64::from(n));
        let g = if op.takes_g() {
            let g = g.unwrap_or_else(|| panic!("{NAME}: {op:?} reads g, and none was given"));
            check_len(NAME, "gradient", g, u64::from(n));
            g
        } else {
            x
        };
        check_distinct(
            NAME,
            ("output", out),
            &[("input", x), ("slopes", slope), ("gradient", g)],
        );
        let pipeline = Prelu::ALL
            .iter()
            .position(|&o| o == op)
            .expect("every operation has a pipeline");
        let uniform = upload_uniform(steps, pool, &word_bytes([n, channels, 0, 0]));
        let group = bind(steps, &self.group, &[out, &uniform, x, slope, g]);
        PreluCall {
            op,
            pipeline,
            n,
            uniform,
            group,
        }
    }

    /// Adds the prepared operation `call` to `step`: one thread per value, over a grid split
    /// across two dimensions when one does not hold it.
    ///
    /// # Examples
    ///
    /// See [`PreluKernels`].
    pub fn record(&self, step: &mut Step<'_>, call: &PreluCall) {
        step.dispatch_split(
            &self.pipelines[call.pipeline],
            &call.group,
            call.n.div_ceil(WORKGROUP),
        );
    }

    /// Prepares the slopes' gradient for `rows × channels` values of the forward input `x`
    /// and the gradient `g` into `out` (`channels` values), overwritten: as
    /// [`ChannelSums::call`](super::ChannelSums::call) prepares a sum.
    ///
    /// # Panics
    ///
    /// As [`call`](PreluKernels::call), and if there are more than 65,535 channels or more
    /// than 65,535 chunks of 4,096 rows, or `out` is `x` or `g`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::PreluKernels;
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let kernels = PreluKernels::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
    ///         | wgpu::BufferUsages::COPY_SRC;
    ///     let (x, g, ds) = (pool.take(16, usage), pool.take(16, usage), pool.take(8, usage));
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let call = kernels.slope_grad_call(&mut steps, &mut pool, (2, 2), &x, &g, &ds);
    ///     let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    ///     let mut step = steps.begin();
    ///     step.upload(x.buffer(), 0, &bytes(&[-4.0, 2.0, -3.0, -8.0]));
    ///     step.upload(g.buffer(), 0, &bytes(&[1.0, 5.0, 2.0, 0.5]));
    ///     kernels.record_slope_grad(&mut step, &call);
    ///     step.read_back(ds.buffer(), 0, 8);
    ///     // Column 0: 1·(−4) + 2·(−3); column 1: 0.5·(−8), the 2 being above 0.
    ///     assert_eq!(step.finish().bytes(), &bytes(&[-10.0, -4.0])[..]);
    ///     call.release(&mut pool);
    /// }
    /// ```
    pub fn slope_grad_call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        (rows, channels): (u32, u32),
        x: &PooledBuffer,
        g: &PooledBuffer,
        out: &PooledBuffer,
    ) -> ChannelSumsCall {
        let n = check_shape(rows, channels);
        self.slope.call(
            steps,
            pool,
            "prelu slope gradient",
            ChannelDims::rows(rows, channels),
            (x, 0),
            &[(g, u64::from(n))],
            SumsTo::new(out),
        )
    }

    /// Adds the slopes' gradient `call` to `step`: the sums' two passes.
    ///
    /// # Examples
    ///
    /// See [`slope_grad_call`](PreluKernels::slope_grad_call).
    pub fn record_slope_grad(&self, step: &mut Step<'_>, call: &ChannelSumsCall) {
        self.slope.record(step, call);
    }
}

/// `rows × channels`, after panicking unless both are above 0 and the product fits a `u32`.
fn check_shape(rows: u32, channels: u32) -> u32 {
    assert!(
        rows > 0 && channels > 0,
        "{NAME}: the rows and the channels must be above 0, got {rows} × {channels}"
    );
    rows.checked_mul(channels).unwrap_or_else(|| {
        panic!("{NAME}: {rows} rows of {channels} are more than the kernels take")
    })
}

/// One prepared element-wise PReLU operation, from [`PreluKernels::call`].
///
/// # Examples
///
/// See [`PreluKernels`].
#[derive(Debug)]
pub struct PreluCall {
    /// The operation.
    op: Prelu,
    /// Its pipeline's index in [`PreluKernels`]'s list.
    pipeline: usize,
    /// The values.
    n: u32,
    /// The uniform, from the pool.
    uniform: PooledBuffer,
    /// The bind group.
    group: wgpu::BindGroup,
}

impl PreluCall {
    /// The operation this call was prepared for.
    ///
    /// # Examples
    ///
    /// See [`PreluKernels`].
    pub fn op(&self) -> Prelu {
        self.op
    }

    /// Gives the call's uniform back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`PreluKernels`].
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.uniform);
    }
}
