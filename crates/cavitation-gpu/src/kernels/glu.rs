//! Gated linear units on the GPU, as `cavitation_core`'s `Tape::glu` computes them: each row
//! of a `[rows][width]` batch split into a first half `a` and a second half `b`, the output
//! `a · gate(b)` of width `width / 2`, and the backward pass writing both halves of the input's
//! gradient, `g · gate(b)` and `g · a · gate′(b)`. The gates (the sigmoid, GELU in both forms,
//! ReLU and SiLU) are the element-wise kernels' own functions, from `activation.wgsl`.

use cavitation_core::{Gate, GeluApprox};

use super::bindings::{Kind, bind, check_distinct, check_len, layout, upload_uniform, word_bytes};
use crate::{BufferPool, Gpu, GpuError, PipelineCache, PooledBuffer, Shader, Step, StepEncoder};

/// The kernels' name in panic messages.
const NAME: &str = "glu";

/// Threads per workgroup, as the shader's `WORKGROUP` constant has it.
const WORKGROUP: u32 = 256;

/// Bytes of the kernels' uniform: the pairs, half the width, a word of padding and the word
/// `held()` reads, always 0.
const UNIFORM_BYTES: u64 = 16;

/// The GLU kernels, one thread per pair `(a, b)` numbered over a split grid, the gate chosen
/// by the `GATE` override (0 the sigmoid, 1 GELU's tanh form, 2 its erf form, 3 ReLU, 4 SiLU),
/// with the scalar functions of `activation.wgsl` appended. Bindings: the output, the
/// uniform, `x` (the forward input) and `g` (the gradient flowing back; bound to `x` and
/// unused forward). The gate's value and derivative are the element-wise kernels' formulas,
/// operation for operation; NaN is tested by its bits.
const GLU: Shader = Shader {
    name: "glu",
    source: concat!(
        "
        override GATE: u32;

        struct Params { pairs: u32, half: u32, pad: u32, zero: u32 }

        @group(0) @binding(0) var<storage, read_write> out: array<f32>;
        @group(0) @binding(1) var<uniform> params: Params;
        @group(0) @binding(2) var<storage, read> x: array<f32>;
        @group(0) @binding(3) var<storage, read> g: array<f32>;

        const WORKGROUP: u32 = 256u;

        // The pair this thread computes, numbered over a split grid.
        fn element(w: vec3<u32>, n: vec3<u32>, l: u32) -> u32 {
            return (w.x + w.y * n.x) * WORKGROUP + l;
        }

        // The index in x of pair i's a; its b is half further on.
        fn first_half(i: u32) -> u32 {
            let r = i / params.half;
            return i + r * params.half;
        }

        // The gate's value at v, as the element-wise kernels compute it.
        fn gate_value(v: f32) -> f32 {
            switch GATE {
                case 0u: { return sigmoid_one(v); }
                case 1u: {
                    let t = tanh_one(SQRT_2_OVER_PI * (v + GELU_CUBIC * v * v * v));
                    return 0.5 * v * (1.0 + t);
                }
                case 2u: { return 0.5 * v * (1.0 + erf_one(v * FRAC_1_SQRT_2)); }
                case 3u: { return select(select(v, 0.0, v <= 0.0), v, is_nan(v)); }
                default: { return v * sigmoid_one(v); }
            }
        }

        // The gate's derivative at v, as the element-wise kernels' backward passes take it.
        fn gate_derivative(v: f32) -> f32 {
            switch GATE {
                case 0u: {
                    let s = sigmoid_one(v);
                    return s * (1.0 - s);
                }
                case 1u: {
                    let v2 = v * v;
                    let t = tanh_one(SQRT_2_OVER_PI * (v + GELU_CUBIC * v2 * v));
                    // Where tanh saturates the second term is dropped, as the CPU does.
                    let first = 0.5 * (1.0 + t);
                    let flat = 1.0 - t * t;
                    if (flat == 0.0) { return first; }
                    let du = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_CUBIC * v2);
                    return first + 0.5 * v * flat * du;
                }
                case 2u: {
                    let cdf = 0.5 * (1.0 + erf_one(v * FRAC_1_SQRT_2));
                    let pdf = FRAC_1_SQRT_2PI * exp(-(0.5 * v * v));
                    // Where the density underflows the x · φ term is dropped, as the CPU does.
                    if (pdf == 0.0) { return cdf; }
                    return cdf + v * pdf;
                }
                case 3u: { return select(select(0.0, 1.0, v > 0.0), v, is_nan(v)); }
                default: {
                    let s = sigmoid_one(v);
                    return s * (1.0 + v * (1.0 - s));
                }
            }
        }

        // a · gate(b).
        @compute @workgroup_size(256)
        fn forward(
            @builtin(workgroup_id) w: vec3<u32>,
            @builtin(num_workgroups) n: vec3<u32>,
            @builtin(local_invocation_index) l: u32,
        ) {
            let i = element(w, n, l);
            if (i >= params.pairs) { return; }
            let k = first_half(i);
            out[i] = x[k] * gate_value(x[k + params.half]);
        }

        // g · gate(b) into a's place and (g · a) · gate'(b) into b's.
        @compute @workgroup_size(256)
        fn backward(
            @builtin(workgroup_id) w: vec3<u32>,
            @builtin(num_workgroups) n: vec3<u32>,
            @builtin(local_invocation_index) l: u32,
        ) {
            let i = element(w, n, l);
            if (i >= params.pairs) { return; }
            let k = first_half(i);
            let a = x[k];
            let b = x[k + params.half];
            let gi = g[i];
            out[k] = gi * gate_value(b);
            out[k + params.half] = (gi * a) * gate_derivative(b);
        }
        ",
        include_str!("activation.wgsl")
    ),
};

/// One GLU pass of [`GluKernels`].
///
/// # Examples
///
/// ```
/// use cavitation_gpu::kernels::Glu;
///
/// assert!(!Glu::Forward.takes_g());
/// assert!(Glu::Backward.takes_g());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Glu {
    /// `a · gate(b)`, half the input's width.
    Forward,
    /// The input's gradient, both halves: `g · gate(b)` and `g · a · gate′(b)`.
    Backward,
}

impl Glu {
    /// Every pass, in declaration order.
    ///
    /// ```
    /// use cavitation_gpu::kernels::Glu;
    ///
    /// assert_eq!(Glu::ALL.len(), 2);
    /// ```
    pub const ALL: [Glu; 2] = [Glu::Forward, Glu::Backward];

    /// Whether the pass reads the gradient `g`: the backward pass.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::Glu;
    ///
    /// assert!(Glu::Backward.takes_g());
    /// ```
    pub fn takes_g(self) -> bool {
        matches!(self, Glu::Backward)
    }

    /// The entry point that computes it.
    fn entry(self) -> &'static str {
        match self {
            Glu::Forward => "forward",
            Glu::Backward => "backward",
        }
    }
}

/// Every gate, in the order of the shader's `GATE` values.
const GATES: [Gate; 5] = [
    Gate::Sigmoid,
    Gate::Gelu(GeluApprox::Tanh),
    Gate::Gelu(GeluApprox::Exact),
    Gate::Relu,
    Gate::Silu,
];

/// The shader's `GATE` value for `gate`.
fn gate_index(gate: Gate) -> usize {
    match gate {
        Gate::Sigmoid => 0,
        Gate::Gelu(GeluApprox::Tanh) => 1,
        Gate::Gelu(GeluApprox::Exact) => 2,
        Gate::Relu => 3,
        Gate::Silu => 4,
    }
}

/// Gated linear units on the GPU, forward and backward, for a `[rows][width]` batch with an
/// even `width`, with any [`Gate`]: [`call`](GluKernels::call) prepares a pass on given
/// buffers, once, and [`record`](GluKernels::record) adds it to a step. One thread takes a
/// pair `(a, b)`, the backward pass writing both of its gradients, so every value of the
/// input's gradient is written.
///
/// - **ReLU gate:** one multiplication forward, `g · relu(b)` and `(g · a) · relu′(b)`
///   backward, as the tape's, so on normal values and products the output and the gradient
///   are the CPU's bits; NaN in either half gives NaN where the tape has it, and the
///   derivative at exactly 0 is 0.
/// - **Sigmoid, GELU (both forms) and SiLU gates:** the gate's value and derivative are the
///   element-wise kernels' (`Elementwise::Sigmoid`, `GeluTanh`, `GeluErf`, `Silu` and their
///   backward passes), which the CPU's gradient takes with std's `exp` and `tanh` instead,
///   so they are within bounds of the `f64` tape stated and derived in `tests/glu.rs`: the
///   gate's own bound from `tests/elementwise.rs` carried by `|a|`, `|g|` or `|g · a|`, plus
///   one rounding for each product. On ±0 and ±∞ they give the CPU's bits, and NaN for NaN.
/// - **Flushed subnormals:** on a GPU that flushes subnormals a subnormal result is flushed,
///   and a subnormal input may be read as 0. A positive subnormal `b` read as 0 is not above
///   0, so through a ReLU gate its output is `a · 0` instead of `a · b` and its gradient
///   `g · a · 0` instead of `g · a`, an error of the size of the gradient itself; through the
///   other gates a subnormal input changes the gate by about its own size, carried by `a` or
///   `g`. CUDA's kernels keep subnormals and have none of this.
///
/// Each thread's operations depend on its pair alone, so two runs give the same bits.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_core::Gate;
/// use cavitation_gpu::kernels::{Glu, GluKernels};
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let kernels = GluKernels::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
///     let mut pool = BufferPool::new(&gpu);
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     let (x, y) = (pool.take(16, usage), pool.take(8, usage));
///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
///     let call =
///         kernels.call(&mut steps, &mut pool, Glu::Forward, Gate::Relu, (1, 4), &x, None, &y);
///     let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
///     let mut step = steps.begin();
///     step.upload(x.buffer(), 0, &bytes(&[2.0, 3.0, -1.0, 4.0]));
///     kernels.record(&mut step, &call);
///     step.read_back(y.buffer(), 0, 8);
///     // 2 · relu(−1) and 3 · relu(4).
///     assert_eq!(step.finish().bytes(), &bytes(&[0.0, 12.0])[..]);
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct GluKernels {
    /// The layout of group 0: the output, the uniform, `x` and `g`.
    group: wgpu::BindGroupLayout,
    /// The pipelines, two per gate in the order of the shader's `GATE` values, each gate's
    /// in the order of [`Glu::ALL`].
    pipelines: Vec<wgpu::ComputePipeline>,
}

impl GluKernels {
    /// Builds every pass's pipeline for every gate on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// See [`GluKernels`].
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<GluKernels, GpuError> {
        use Kind::{Read, Uniform, Write};
        let (group, pipeline_layout) =
            layout(gpu, NAME, &[Write, Uniform(UNIFORM_BYTES), Read, Read]);
        let mut pipelines = Vec::with_capacity(GATES.len() * Glu::ALL.len());
        for gate in GATES {
            let constants = [("GATE", gate_index(gate) as f64)];
            for op in Glu::ALL {
                pipelines.push(cache.get_with_layout(
                    GLU,
                    op.entry(),
                    &constants,
                    &pipeline_layout,
                )?);
            }
        }
        Ok(GluKernels { group, pipelines })
    }

    /// Prepares `op` with `gate` over `rows` rows of `width` values of `x` (and `g`, `rows`
    /// rows of `width / 2`, when [`Glu::takes_g`]) into `out` (`width / 2` values a row
    /// forward, `width` backward), overwritten: its uniform, taken from `pool` and uploaded
    /// in a step of its own, and its bind group. Made once, when a model is compiled. The
    /// forward pass binds `x` in place of `g`.
    ///
    /// # Panics
    ///
    /// If `rows` or `width` is 0, `width` is odd, `rows × width` does not fit a `u32`, a
    /// buffer is shorter than its values, `op` reads `g` and none is given, or `out` is one
    /// of the inputs.
    ///
    /// # Examples
    ///
    /// See [`GluKernels`].
    #[allow(clippy::too_many_arguments)]
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        op: Glu,
        gate: Gate,
        (rows, width): (u32, u32),
        x: &PooledBuffer,
        g: Option<&PooledBuffer>,
        out: &PooledBuffer,
    ) -> GluCall {
        let n = check_shape(rows, width);
        let pairs = n / 2;
        check_len(NAME, "input", x, u64::from(n));
        let written = if op.takes_g() { n } else { pairs };
        check_len(NAME, "output", out, u64::from(written));
        let g = if op.takes_g() {
            let g = g.unwrap_or_else(|| panic!("{NAME}: {op:?} reads g, and none was given"));
            check_len(NAME, "gradient", g, u64::from(pairs));
            g
        } else {
            x
        };
        check_distinct(NAME, ("output", out), &[("input", x), ("gradient", g)]);
        let pipeline = gate_index(gate) * Glu::ALL.len()
            + Glu::ALL
                .iter()
                .position(|&o| o == op)
                .expect("every pass has a pipeline");
        let uniform = upload_uniform(steps, pool, &word_bytes([pairs, width / 2, 0, 0]));
        let group = bind(steps, &self.group, &[out, &uniform, x, g]);
        GluCall {
            op,
            gate,
            pipeline,
            pairs,
            uniform,
            group,
        }
    }

    /// Adds the prepared pass `call` to `step`: one thread per pair, over a grid split
    /// across two dimensions when one does not hold it.
    ///
    /// # Examples
    ///
    /// See [`GluKernels`].
    pub fn record(&self, step: &mut Step<'_>, call: &GluCall) {
        step.dispatch_split(
            &self.pipelines[call.pipeline],
            &call.group,
            call.pairs.div_ceil(WORKGROUP),
        );
    }
}

/// `rows × width`, after panicking unless both are above 0, the width is even and the product
/// fits a `u32`.
fn check_shape(rows: u32, width: u32) -> u32 {
    assert!(
        rows > 0 && width > 0,
        "{NAME}: the rows and the width must be above 0, got {rows} × {width}"
    );
    assert!(
        width.is_multiple_of(2),
        "{NAME}: the width must be even, got {width}"
    );
    rows.checked_mul(width)
        .unwrap_or_else(|| panic!("{NAME}: {rows} rows of {width} are more than the kernels take"))
}

/// One prepared GLU pass, from [`GluKernels::call`].
///
/// # Examples
///
/// See [`GluKernels`].
#[derive(Debug)]
pub struct GluCall {
    /// The pass.
    op: Glu,
    /// The gate.
    gate: Gate,
    /// Its pipeline's index in [`GluKernels`]'s list.
    pipeline: usize,
    /// The pairs, one thread each.
    pairs: u32,
    /// The uniform, from the pool.
    uniform: PooledBuffer,
    /// The bind group.
    group: wgpu::BindGroup,
}

impl GluCall {
    /// The pass this call was prepared for.
    ///
    /// # Examples
    ///
    /// See [`GluKernels`].
    pub fn op(&self) -> Glu {
        self.op
    }

    /// The gate this call was prepared for.
    ///
    /// # Examples
    ///
    /// See [`GluKernels`].
    pub fn gate(&self) -> Gate {
        self.gate
    }

    /// Gives the call's uniform back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`GluKernels`].
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.uniform);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `GATES` lists every gate once, at its `GATE` value.
    #[test]
    fn gates_are_at_their_index() {
        for (i, &gate) in GATES.iter().enumerate() {
            assert_eq!(gate_index(gate), i, "{gate:?}");
        }
    }
}
