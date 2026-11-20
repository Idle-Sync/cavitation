//! Softmax and log-softmax over each row of a `[rows][width]` batch, forward and backward, as
//! `cavitation_core`'s `Tape::softmax` and `Tape::log_softmax` compute them over the last
//! axis: one thread per row, which takes the row's values in the CPU's order.

use super::bindings::{Kind, bind, check_distinct, check_len, layout, upload_uniform, word_bytes};
use crate::{BufferPool, Gpu, GpuError, PipelineCache, PooledBuffer, Shader, Step, StepEncoder};

/// Threads per workgroup, passed as the `WORKGROUP` override.
const WORKGROUP: u32 = 256;

/// Bytes of the kernels' uniform: the rows, the width and two words of padding.
const UNIFORM_BYTES: u64 = 16;

/// The most rows a call takes: one thread each, in one dimension of at most 65,535
/// workgroups of 256.
///
/// ```
/// assert_eq!(cavitation_gpu::kernels::MAX_SOFTMAX_ROWS, 65_535 * 256);
/// ```
pub const MAX_SOFTMAX_ROWS: u32 = crate::MAX_WORKGROUPS_PER_DIMENSION * WORKGROUP;

/// The four kernels, one entry point each. Bindings: the output (written as bits), the
/// uniform, `a` (the forward input, or the forward output for a backward pass) and `b` (the
/// gradient flowing back; bound to `a` and unused forward).
///
/// A forward row with a NaN or `+∞`, or with `−∞` everywhere, is written as NaN bits, since
/// WGSL may fold NaN arithmetic away; a `−∞` elsewhere is tested by its bits and takes the
/// CPU's values without infinite arithmetic: a term of 0, a softmax of 0, a log-softmax of
/// `−∞`, and `e^y = 0` in the log-softmax's backward pass. The logarithm of the log-softmax is
/// `ln(1 + e)` with `e` the sum without the first maximum's term: for `e ≤ 1` it is
/// `2·atanh(e / (2 + e))` from its series to `y¹⁵`, as the cross-entropy's kernel takes it,
/// which keeps `e`'s low bits and avoids WGSL's `log` near 1.
const ROWS: Shader = Shader {
    name: "row softmax",
    source: "
        override WORKGROUP: u32;

        struct Rows { rows: u32, width: u32, pad0: u32, pad1: u32 }

        @group(0) @binding(0) var<storage, read_write> out: array<u32>;
        @group(0) @binding(1) var<uniform> r: Rows;
        @group(0) @binding(2) var<storage, read> a: array<f32>;
        @group(0) @binding(3) var<storage, read> b: array<f32>;

        const NAN_BITS: u32 = 0x7fc00000u;
        const NEG_INF_BITS: u32 = 0xff800000u;

        // Whether x is -inf, from its bits.
        fn is_neg_inf(x: f32) -> bool {
            return bitcast<u32>(x) == NEG_INF_BITS;
        }

        // Writes NaN over the row at base and returns true where the CPU gives a NaN row: a
        // value in it is NaN or +inf, or every value is -inf.
        fn bad_row(base: u32) -> bool {
            var bad = false;
            var finite = false;
            for (var j = 0u; j < r.width; j = j + 1u) {
                let z = a[base + j];
                if (is_neg_inf(z)) { continue; }
                bad = bad || (bitcast<u32>(z) & 0x7f800000u) == 0x7f800000u;
                finite = true;
            }
            bad = bad || !finite;
            if (bad) {
                for (var j = 0u; j < r.width; j = j + 1u) { out[base + j] = NAN_BITS; }
            }
            return bad;
        }

        // The row's maximum over its values other than -inf (a row bad_row passed has one).
        fn row_max(base: u32) -> f32 {
            var m = 0.0;
            var seen = false;
            for (var j = 0u; j < r.width; j = j + 1u) {
                let z = a[base + j];
                if (is_neg_inf(z)) { continue; }
                if (seen) { m = max(m, z); } else { m = z; seen = true; }
            }
            return m;
        }

        // e^(z - m), and 0 for a z of -inf, as the CPU's exp(-inf) gives.
        fn exp_from(z: f32, m: f32) -> f32 {
            if (is_neg_inf(z)) { return 0.0; }
            return exp(z - m);
        }

        // ln(1 + e) for e >= 0.
        fn ln_1p(e: f32) -> f32 {
            if (e > 1.0) { return log(1.0 + e); }
            let y = e / (2.0 + e);
            let q = y * y;
            var t = 1.0 / 15.0;
            t = 1.0 / 13.0 + q * t;
            t = 1.0 / 11.0 + q * t;
            t = 1.0 / 9.0 + q * t;
            t = 1.0 / 7.0 + q * t;
            t = 1.0 / 5.0 + q * t;
            t = 1.0 / 3.0 + q * t;
            t = 1.0 + q * t;
            return 2.0 * y * t;
        }

        // e^(z - max) / sum of e^(z - max), the terms kept in out and summed in order.
        @compute @workgroup_size(WORKGROUP)
        fn softmax(@builtin(global_invocation_id) id: vec3<u32>) {
            if (id.x >= r.rows) { return; }
            let base = id.x * r.width;
            if (bad_row(base)) { return; }
            let m = row_max(base);
            var s = 0.0;
            for (var j = 0u; j < r.width; j = j + 1u) {
                let e = exp_from(a[base + j], m);
                out[base + j] = bitcast<u32>(e);
                s = s + e;
            }
            for (var j = 0u; j < r.width; j = j + 1u) {
                out[base + j] = bitcast<u32>(bitcast<f32>(out[base + j]) / s);
            }
        }

        // z - max - ln(1 + e), e the sum of e^(z - max) without the first maximum's term.
        @compute @workgroup_size(WORKGROUP)
        fn log_softmax(@builtin(global_invocation_id) id: vec3<u32>) {
            if (id.x >= r.rows) { return; }
            let base = id.x * r.width;
            if (bad_row(base)) { return; }
            let m = row_max(base);
            var k = 0u;
            for (var j = 0u; j < r.width; j = j + 1u) {
                if (a[base + j] == m) { k = j; break; }
            }
            var e = 0.0;
            for (var j = 0u; j < r.width; j = j + 1u) {
                if (j != k) { e = e + exp_from(a[base + j], m); }
            }
            let lse = ln_1p(e);
            for (var j = 0u; j < r.width; j = j + 1u) {
                let z = a[base + j];
                // -inf - max - lse is -inf, written as bits.
                out[base + j] = select(bitcast<u32>(z - m - lse), NEG_INF_BITS, is_neg_inf(z));
            }
        }

        // y * (g - sum of y * g), with y the forward output in a and g in b.
        @compute @workgroup_size(WORKGROUP)
        fn softmax_backward(@builtin(global_invocation_id) id: vec3<u32>) {
            if (id.x >= r.rows) { return; }
            let base = id.x * r.width;
            var dot = 0.0;
            for (var j = 0u; j < r.width; j = j + 1u) { dot = dot + a[base + j] * b[base + j]; }
            for (var j = 0u; j < r.width; j = j + 1u) {
                out[base + j] = bitcast<u32>(a[base + j] * (b[base + j] - dot));
            }
        }

        // g - e^y * sum of g, with y the forward output in a and g in b; e^y is 0 for a y
        // of -inf.
        @compute @workgroup_size(WORKGROUP)
        fn log_softmax_backward(@builtin(global_invocation_id) id: vec3<u32>) {
            if (id.x >= r.rows) { return; }
            let base = id.x * r.width;
            var total = 0.0;
            for (var j = 0u; j < r.width; j = j + 1u) { total = total + b[base + j]; }
            for (var j = 0u; j < r.width; j = j + 1u) {
                let p = exp_from(a[base + j], 0.0);
                out[base + j] = bitcast<u32>(b[base + j] - p * total);
            }
        }
    ",
};

/// One row operation of [`RowSoftmaxKernels`]: `a` is its first input, `b` the gradient
/// flowing back for a backward pass, and the result goes to `out`.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::kernels::RowSoftmax;
///
/// assert!(!RowSoftmax::Softmax.takes_b());
/// assert!(RowSoftmax::LogSoftmaxBackward.takes_b());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RowSoftmax {
    /// `e^(a − max) / Σ e^(a − max)` over each row.
    Softmax,
    /// `a · (b − Σ a·b)` over each row, with `a` the softmax's output.
    SoftmaxBackward,
    /// `a − max − ln Σ e^(a − max)` over each row.
    LogSoftmax,
    /// `b − e^a · Σ b` over each row, with `a` the log-softmax's output.
    LogSoftmaxBackward,
}

impl RowSoftmax {
    /// Every operation, in declaration order.
    ///
    /// ```
    /// use cavitation_gpu::kernels::RowSoftmax;
    ///
    /// assert_eq!(RowSoftmax::ALL.iter().filter(|op| op.takes_b()).count(), 2);
    /// ```
    pub const ALL: [RowSoftmax; 4] = [
        RowSoftmax::Softmax,
        RowSoftmax::SoftmaxBackward,
        RowSoftmax::LogSoftmax,
        RowSoftmax::LogSoftmaxBackward,
    ];

    /// Whether the operation reads `b`: the backward passes.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::kernels::RowSoftmax;
    ///
    /// assert!(RowSoftmax::SoftmaxBackward.takes_b());
    /// assert!(!RowSoftmax::LogSoftmax.takes_b());
    /// ```
    pub fn takes_b(self) -> bool {
        matches!(
            self,
            RowSoftmax::SoftmaxBackward | RowSoftmax::LogSoftmaxBackward
        )
    }

    /// The entry point that computes it.
    fn entry(self) -> &'static str {
        match self {
            RowSoftmax::Softmax => "softmax",
            RowSoftmax::SoftmaxBackward => "softmax_backward",
            RowSoftmax::LogSoftmax => "log_softmax",
            RowSoftmax::LogSoftmaxBackward => "log_softmax_backward",
        }
    }
}

/// Softmax and log-softmax over each row, forward and backward, on the GPU: one pipeline per
/// [`RowSoftmax`] operation, built once. [`call`](RowSoftmaxKernels::call) prepares one
/// operation on given buffers, once, and [`record`](RowSoftmaxKernels::record) adds it to a
/// step.
///
/// Each row runs in one thread, which reads its values in order, as the CPU does, so two
/// runs give the same bits. The forward passes are `Tape::softmax` and `Tape::log_softmax`
/// ported operation for operation: the row's maximum subtracted first, the softmax's terms
/// summed in order, and the log-softmax's `ln Σ e^(z − max)` taken as `ln(1 + e)` with `e`
/// the sum without the first maximum's term. A row with a NaN or `+∞`, or with `−∞`
/// everywhere, gives a NaN row, and a `−∞` elsewhere a softmax of 0 and a log-softmax of
/// `−∞`, as on the CPU. The backward passes are the tape's: `y · (g − Σ y·g)` for the
/// softmax, and `g − e^y · Σ g` for the log-softmax, from the forward output `y` and the
/// gradient `g`.
///
/// The bounds are against the exact result of the same `f32` inputs. WGSL gives `exp`
/// within `3 + 2|x|` ulp, `log` within 3 ulp off `[0.5, 2]` and division within 2.5 ulp, and
/// every multiply-add may round twice. For a row of `W` values with `x_j = z_j − max`:
///
/// - **Softmax:** with `s = Σ e^(x_j)` and `Δs = Σ (6 + 5|x_j|)·u·e^(x_j) + 2(W + 2)·u·s`,
///   output `j` is within `y_j · ((6 + 5|x_j|)·u + Δs / s + 5u)`.
/// - **Log-softmax:** with `e = Σ_{j≠k} e^(x_j)`, `S = 1 + e`, `Δe = Σ_{j≠k} (6 + 5|x_j|)·u
///   ·e^(x_j) + 2(W + 1)·u·e`, and `Δ = Δe + 10u·ln S` if `e ≤ 1`, else `(Δe + u·S)/S +
///   6u·ln S`, output `j` is within `2u·(|z_j| + |max| + ln S) + Δ`: its two subtractions in
///   either order, as a compiler may reassociate them.
/// - **Softmax backward:** with `d = Σ y·g` and `Δd = 2(W + 2)·u·Σ|y·g|`, output `j` is
///   within `|y_j|·(Δd + 2u·|g_j − d|)`.
/// - **Log-softmax backward:** with `p_j = e^(y_j)`, `t = Σ g` and `Δt = 2(W + 2)·u·Σ|g|`,
///   output `j` is within `p_j·Δt + (7 + 4|y_j|)·u·p_j·|t| + u·|g_j − p_j·t|`.
///
/// Each bound grows by a factor `1 + 16u` for its own rounding and adds `2⁻¹²⁶` for every
/// value that may be flushed. `tests/softmax.rs` checks each, and the softmax's backward pass
/// bit for bit on small integers, whose sums and products are exact.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_gpu::kernels::{RowSoftmax, RowSoftmaxKernels};
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let kernels = RowSoftmaxKernels::new(&gpu, &mut PipelineCache::new(&gpu)).unwrap();
///     let mut pool = BufferPool::new(&gpu);
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     let (x, y) = (pool.take(16, usage), pool.take(16, usage));
///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
///     let call =
///         kernels.call(&mut steps, &mut pool, RowSoftmax::Softmax, (2, 2), &x, None, &y);
///     let bytes = |v: [f32; 4]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
///     let mut step = steps.begin();
///     step.upload(x.buffer(), 0, &bytes([1.0, 1.0, 1e4, 0.0]));
///     kernels.record(&mut step, &call);
///     step.read_back(y.buffer(), 0, 16);
///     let output = step.finish();
///     let (words, _) = output.bytes().as_chunks::<4>();
///     let y: Vec<f32> = words.iter().map(|w| f32::from_le_bytes(*w)).collect();
///     let want = [0.5, 0.5, 1.0, 0.0];
///     assert!(y.iter().zip(want).all(|(y, w)| (y - w).abs() < 1e-6), "{y:?}");
///     call.release(&mut pool);
/// }
/// ```
#[derive(Debug)]
pub struct RowSoftmaxKernels {
    /// The layout of group 0: the output, the uniform, `a` and `b`.
    group: wgpu::BindGroupLayout,
    /// The pipelines, in the order of [`RowSoftmax::ALL`].
    pipelines: Vec<wgpu::ComputePipeline>,
}

impl RowSoftmaxKernels {
    /// Builds every operation's pipeline on `gpu`, through `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a kernel does not build on this GPU.
    ///
    /// # Examples
    ///
    /// See [`RowSoftmaxKernels`].
    pub fn new(gpu: &Gpu, cache: &mut PipelineCache) -> Result<RowSoftmaxKernels, GpuError> {
        use Kind::{Read, Uniform, Write};
        let (group, pipeline_layout) = layout(
            gpu,
            "row softmax",
            &[Write, Uniform(UNIFORM_BYTES), Read, Read],
        );
        let workgroup = [("WORKGROUP", f64::from(WORKGROUP))];
        let pipelines = RowSoftmax::ALL
            .iter()
            .map(|op| cache.get_with_layout(ROWS, op.entry(), &workgroup, &pipeline_layout))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(RowSoftmaxKernels { group, pipelines })
    }

    /// Prepares `op` over `rows` rows of `width` values of `a` (and `b`, when
    /// [`RowSoftmax::takes_b`]) into `out`, overwritten: its uniform, taken from `pool` and
    /// uploaded in a step of its own, and its bind group. Made once, when a model is
    /// compiled. An operation that does not read `b` binds `a` in its place.
    ///
    /// # Panics
    ///
    /// If `rows` or `width` is 0, `rows` is above [`MAX_SOFTMAX_ROWS`] or `rows × width`
    /// does not fit a `u32`, a buffer holds fewer than `rows × width` values, `op` takes `b`
    /// and none is given, or `out` is `a` or `b`.
    ///
    /// # Examples
    ///
    /// See [`RowSoftmaxKernels`].
    #[allow(clippy::too_many_arguments)]
    pub fn call(
        &self,
        steps: &mut StepEncoder,
        pool: &mut BufferPool,
        op: RowSoftmax,
        (rows, width): (u32, u32),
        a: &PooledBuffer,
        b: Option<&PooledBuffer>,
        out: &PooledBuffer,
    ) -> RowSoftmaxCall {
        let what = "row softmax";
        assert!(
            rows > 0 && width > 0,
            "{what}: the rows and the width must be above 0, got {rows} × {width}"
        );
        assert!(
            rows <= MAX_SOFTMAX_ROWS && rows.checked_mul(width).is_some(),
            "{what}: {rows} rows of {width} are more than the kernels take"
        );
        let values = u64::from(rows) * u64::from(width);
        check_len(what, "input", a, values);
        check_len(what, "output", out, values);
        let b = if op.takes_b() {
            let b = b.unwrap_or_else(|| panic!("{what}: {op:?} reads b, and none was given"));
            check_len(what, "gradient", b, values);
            b
        } else {
            a
        };
        check_distinct(what, ("output", out), &[("input", a), ("gradient", b)]);
        let pipeline = RowSoftmax::ALL
            .iter()
            .position(|&o| o == op)
            .expect("every operation has a pipeline");
        let uniform = upload_uniform(steps, pool, &word_bytes([rows, width, 0, 0]));
        let group = bind(steps, &self.group, &[out, &uniform, a, b]);
        RowSoftmaxCall {
            op,
            pipeline,
            rows,
            uniform,
            group,
        }
    }

    /// Adds the prepared operation `call` to `step`: one thread per row.
    ///
    /// # Examples
    ///
    /// See [`RowSoftmaxKernels`].
    pub fn record(&self, step: &mut Step<'_>, call: &RowSoftmaxCall) {
        step.dispatch(
            &self.pipelines[call.pipeline],
            &call.group,
            call.rows.div_ceil(WORKGROUP),
        );
    }
}

/// One prepared row operation, from [`RowSoftmaxKernels::call`].
///
/// # Examples
///
/// See [`RowSoftmaxKernels`].
#[derive(Debug)]
pub struct RowSoftmaxCall {
    /// The operation.
    op: RowSoftmax,
    /// Its pipeline's index in [`RowSoftmaxKernels`]'s list.
    pipeline: usize,
    /// The rows.
    rows: u32,
    /// The uniform, from the pool.
    uniform: PooledBuffer,
    /// The bind group.
    group: wgpu::BindGroup,
}

impl RowSoftmaxCall {
    /// The operation this call was prepared for.
    ///
    /// # Examples
    ///
    /// See [`RowSoftmaxKernels`].
    pub fn op(&self) -> RowSoftmax {
        self.op
    }

    /// Gives the call's uniform back to `pool`, once it is no longer recorded.
    ///
    /// # Examples
    ///
    /// See [`RowSoftmaxKernels`].
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.uniform);
    }
}
