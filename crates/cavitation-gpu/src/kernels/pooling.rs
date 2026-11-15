//! 2-D max and average pooling on the GPU, forward and backward, as the CPU kernels in
//! `cavitation_core::kernels::pooling` compute them: max pooling bit for bit on normal
//! values (the same winners, ties and NaNs), average pooling within its rounding bound.
//!
//! Each kernel is specialised for one geometry and batch through `override` constants, and
//! numbers its threads over a two-dimensional grid, so any batch fits. The backward passes
//! are gathers, one thread per input pixel adding its outputs' gradients in the CPU's order,
//! so they need no atomics and give the same bits on every run.

use cavitation_core::Pool2dGeom;

use crate::{GpuError, PipelineCache, PooledBuffer, Shader, Step, StepEncoder, wgpu};

/// Threads per workgroup: every pooling kernel's `@workgroup_size`, passed to it as the
/// `WORKGROUP` override.
const WORKGROUP: u32 = 256;

/// The largest thread count a pooling kernel takes: its thread index, workgroup index times
/// [`WORKGROUP`] plus the lane, must not wrap in `u32` on the split grid's last row.
const MAX_THREADS: usize = 1 << 31;

/// The entry point every pooling kernel shares: numbers its thread `t` over the grid
/// `Step::dispatch_split` makes, returns past `TOTAL` threads, and runs the kernel's
/// `body(t)`.
macro_rules! entry {
    () => {
        "
        override WORKGROUP: u32;
        override TOTAL: u32;

        @compute @workgroup_size(WORKGROUP)
        fn main(
            @builtin(workgroup_id) group: vec3<u32>,
            @builtin(num_workgroups) groups: vec3<u32>,
            @builtin(local_invocation_index) lane: u32,
        ) {
            let t = (group.x + group.y * groups.x) * WORKGROUP + lane;
            if (t < TOTAL) { body(t); }
        }
        "
    };
}

/// The max-pooling geometry, as the `override` constants both max kernels declare.
macro_rules! max_geometry {
    () => {
        "
        override H: u32;
        override W: u32;
        override OH: u32;
        override OW: u32;
        override KH: u32;
        override KW: u32;
        override SH: u32;
        override SW: u32;
        override PH: u32;
        override PW: u32;
        override DH: u32;
        override DW: u32;
        "
    };
}

/// Max pooling forward: one thread per output scans its window's in-image taps in row-major
/// order, keeping the first tap and then each later one that is greater or NaN, unless the
/// value kept so far is NaN and the new one is not. Values are compared as integers that
/// order the f32 bits as the floats are ordered (the two zeros equal), so fast math and
/// subnormal flushing cannot change a winner, and the winner's bits are copied unchanged.
const MAX_FORWARD: Shader = Shader {
    name: "max_pool2d forward",
    source: concat!(
        entry!(),
        max_geometry!(),
        "
        @group(0) @binding(0) var<storage, read> x: array<u32>;
        @group(0) @binding(1) var<storage, read_write> out: array<u32>;
        @group(0) @binding(2) var<storage, read_write> winners: array<u32>;

        fn is_nan(bits: u32) -> bool {
            return (bits & 0x7fffffffu) > 0x7f800000u;
        }

        // The f32 bits as an unsigned key in the floats' order: -inf lowest, +inf highest.
        fn order_key(bits: u32) -> u32 {
            return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
        }

        // Whether the float with bits `v` is greater than the one with bits `best`, neither NaN.
        fn greater(v: u32, best: u32) -> bool {
            let both_zero = ((v | best) & 0x7fffffffu) == 0u;
            return !both_zero && order_key(v) > order_key(best);
        }

        fn body(i: u32) {
            let ox = i % OW;
            let oy = (i / OW) % OH;
            let base = (i / (OW * OH)) * H * W;
            var best = 0u;
            var at = 0xffffffffu;
            for (var a = 0u; a < KH; a = a + 1u) {
                let py = oy * SH + a * DH;
                if (py < PH || py - PH >= H) { continue; }
                let y = py - PH;
                for (var b = 0u; b < KW; b = b + 1u) {
                    let px = ox * SW + b * DW;
                    if (px < PW || px - PW >= W) { continue; }
                    let pos = y * W + px - PW;
                    let v = x[base + pos];
                    var take = at == 0xffffffffu || is_nan(v);
                    if (!take && !is_nan(best)) { take = greater(v, best); }
                    if (take) {
                        best = v;
                        at = pos;
                    }
                }
            }
            out[i] = best;
            winners[i] = at;
        }
        "
    ),
};

/// Max pooling backward, one thread per input pixel: adds to the pixel's gradient the
/// gradient of each output that pixel won, those outputs taken in plane order, as the CPU
/// adds them.
const MAX_BACKWARD: Shader = Shader {
    name: "max_pool2d backward",
    source: concat!(
        entry!(),
        max_geometry!(),
        "
        @group(0) @binding(0) var<storage, read> grad_out: array<f32>;
        @group(0) @binding(1) var<storage, read> winners: array<u32>;
        @group(0) @binding(2) var<storage, read_write> grad_in: array<f32>;

        // The first output along an axis whose window can reach padded position `p`.
        fn first_output(p: u32, reach: u32, stride: u32) -> u32 {
            return select(0u, (p - reach + stride - 1u) / stride, p > reach);
        }

        fn body(j: u32) {
            let col = j % W;
            let row = (j / W) % H;
            let plane = j / (W * H);
            let pos = row * W + col;
            let py = row + PH;
            let px = col + PW;
            let y0 = first_output(py, (KH - 1u) * DH, SH);
            let y1 = min(py / SH, OH - 1u);
            let x0 = first_output(px, (KW - 1u) * DW, SW);
            let x1 = min(px / SW, OW - 1u);
            var acc = grad_in[j];
            for (var oy = y0; oy <= y1; oy = oy + 1u) {
                for (var ox = x0; ox <= x1; ox = ox + 1u) {
                    let o = (plane * OH + oy) * OW + ox;
                    if (winners[o] == pos) { acc = acc + grad_out[o]; }
                }
            }
            grad_in[j] = acc;
        }
        "
    ),
};

/// The average-pooling geometry as `override` constants, and the window and divisor shared
/// by the forward and backward kernels. Positions are in padded coordinates (the image's
/// row `r` is `r + P`), so every index is unsigned: along one axis, output `o`'s window
/// starts at `o · S`, ends at the padded image's edge or `K` later, and covers the image
/// from `lo` to `hi`, as `avg_window` in the CPU kernel computes it.
macro_rules! avg_common {
    () => {
        "
        override H: u32;
        override W: u32;
        override OH: u32;
        override OW: u32;
        override KH: u32;
        override KW: u32;
        override SH: u32;
        override SW: u32;
        override PH: u32;
        override PW: u32;
        override COUNT_PAD: bool;
        override DIVISOR: u32;

        // Along one axis: the window's length counting the padding, and its first and
        // one-past-last in-image positions (padded coordinates).
        fn window(o: u32, k: u32, s: u32, p: u32, size: u32) -> vec3<u32> {
            let start = o * s;
            let end = min(start + k, size + 2u * p);
            let lo = max(start, p);
            return vec3<u32>(end - start, lo, max(min(end, size + p), lo));
        }

        fn divisor(r: vec3<u32>, c: vec3<u32>) -> f32 {
            if (DIVISOR != 0u) { return f32(DIVISOR); }
            if (COUNT_PAD) { return f32(r.x * c.x); }
            return f32((r.z - r.y) * (c.z - c.y));
        }
        "
    };
}

/// Average pooling forward, one thread per output: the sum of its window's in-image pixels
/// in row-major order, divided by its divisor.
const AVG_FORWARD: Shader = Shader {
    name: "avg_pool2d forward",
    source: concat!(
        entry!(),
        avg_common!(),
        "
        @group(0) @binding(0) var<storage, read> x: array<f32>;
        @group(0) @binding(1) var<storage, read_write> out: array<f32>;

        fn body(i: u32) {
            let ox = i % OW;
            let oy = (i / OW) % OH;
            let base = (i / (OW * OH)) * H * W;
            let r = window(oy, KH, SH, PH, H);
            let c = window(ox, KW, SW, PW, W);
            var acc = 0.0;
            for (var y = r.y; y < r.z; y = y + 1u) {
                for (var xx = c.y; xx < c.z; xx = xx + 1u) {
                    acc = acc + x[base + (y - PH) * W + xx - PW];
                }
            }
            out[i] = acc / divisor(r, c);
        }
        "
    ),
};

/// Average pooling backward, one thread per input pixel: adds to the pixel's gradient each
/// covering output's gradient divided by that output's divisor, the outputs in plane order,
/// as the CPU adds them.
const AVG_BACKWARD: Shader = Shader {
    name: "avg_pool2d backward",
    source: concat!(
        entry!(),
        avg_common!(),
        "
        @group(0) @binding(0) var<storage, read> grad_out: array<f32>;
        @group(0) @binding(1) var<storage, read_write> grad_in: array<f32>;

        // The first output along an axis whose window can reach padded position `p`.
        fn first_output(p: u32, reach: u32, stride: u32) -> u32 {
            return select(0u, (p - reach + stride - 1u) / stride, p > reach);
        }

        fn body(j: u32) {
            let col = j % W;
            let row = (j / W) % H;
            let plane = j / (W * H);
            let py = row + PH;
            let px = col + PW;
            let y0 = first_output(py, KH - 1u, SH);
            let y1 = min(py / SH, OH - 1u);
            let x0 = first_output(px, KW - 1u, SW);
            let x1 = min(px / SW, OW - 1u);
            var acc = grad_in[j];
            for (var oy = y0; oy <= y1; oy = oy + 1u) {
                let r = window(oy, KH, SH, PH, H);
                for (var ox = x0; ox <= x1; ox = ox + 1u) {
                    let c = window(ox, KW, SW, PW, W);
                    acc = acc + grad_out[(plane * OH + oy) * OW + ox] / divisor(r, c);
                }
            }
            grad_in[j] = acc;
        }
        "
    ),
};

/// Panics, naming `what`, unless `buffer` holds exactly `values` 4-byte values.
fn check_buffer(what: &str, buffer: &PooledBuffer, values: usize) {
    assert!(
        buffer.len() == 4 * values as u64,
        "pool2d: {what} holds {} bytes, expected {} ({values} values)",
        buffer.len(),
        4 * values
    );
}

/// The thread count `threads` as the `TOTAL` override, after checking that it fits.
fn total(threads: usize) -> u32 {
    assert!(
        threads <= MAX_THREADS,
        "pool2d: {threads} values is more than the {MAX_THREADS} a pooling kernel takes"
    );
    threads as u32
}

/// Dispatches `pipeline` with one thread per value of `threads` values.
fn dispatch(
    step: &mut Step<'_>,
    pipeline: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
    threads: u32,
) {
    step.dispatch_split(pipeline, group, threads.div_ceil(WORKGROUP));
}

/// A bind group for `pipeline`'s group 0 with `buffers` at bindings 0, 1, ….
fn bind(
    steps: &mut StepEncoder,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&PooledBuffer],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.binding(),
        })
        .collect();
    steps.bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

/// Panics unless every index the kernels form along each axis of `geom` fits in `limit`:
/// the padded size plus a window and a stride, which bounds every padded position and
/// every intermediate of the kernels' window arithmetic.
fn check_axes(what: &str, geom: &Pool2dGeom, limit: usize) {
    let (dh, dw) = geom.dilation();
    let axes = [
        (
            geom.input().0,
            geom.kernel().0,
            geom.stride().0,
            geom.padding().0,
            dh,
        ),
        (
            geom.input().1,
            geom.kernel().1,
            geom.stride().1,
            geom.padding().1,
            dw,
        ),
    ];
    let fits = axes
        .iter()
        .all(|&(size, kernel, stride, padding, dilation)| {
            let span = dilation
                .checked_mul(kernel)
                .and_then(|s| s.checked_add(stride));
            let padded = padding.checked_mul(2).and_then(|p| p.checked_add(size));
            span.zip(padded)
                .and_then(|(s, p)| s.checked_add(p))
                .is_some_and(|reach| reach <= limit)
        });
    assert!(
        fits,
        "{what}: the geometry's indices do not fit the kernel's {limit}: {geom:?}"
    );
}

/// The workgroup size, the thread count and the geometry's sizes and window as `override`
/// constants, under the names the kernels declare.
fn geometry_constants(geom: &Pool2dGeom, threads: u32) -> Vec<(&'static str, f64)> {
    let (h, w) = geom.input();
    let (oh, ow) = geom.out_size();
    let (kh, kw) = geom.kernel();
    let (sh, sw) = geom.stride();
    let (ph, pw) = geom.padding();
    let as_f64 = |v: usize| v as f64;
    vec![
        ("WORKGROUP", f64::from(WORKGROUP)),
        ("TOTAL", f64::from(threads)),
        ("H", as_f64(h)),
        ("W", as_f64(w)),
        ("OH", as_f64(oh)),
        ("OW", as_f64(ow)),
        ("KH", as_f64(kh)),
        ("KW", as_f64(kw)),
        ("SH", as_f64(sh)),
        ("SW", as_f64(sw)),
        ("PH", as_f64(ph)),
        ("PW", as_f64(pw)),
    ]
}

/// Max pooling on the GPU for one geometry and batch size, as
/// `cavitation_core::kernels::pooling::max_pool2d_reference` computes it, bit for bit: each
/// output is its window's largest value, the taps read in row-major order, a later tap
/// winning when it is greater or NaN (so the last NaN of a window wins), padding never
/// winning, and the winner's bits copied unchanged (signed zeros, subnormals and NaN
/// payloads included). The comparisons are on integers, so this holds on every backend.
///
/// The backward pass adds each output's gradient to its winner's input gradient, one thread
/// per input adding its outputs in the CPU's order, so on normal values it gives
/// `max_pool2d_backward`'s bits wherever the backend keeps that order: measured on Metal,
/// with overlapping windows' sums that round. A subnormal sum is flushed to zero, as the
/// GPU flushes it.
///
/// The images are flat NCHW `f32` rows, the batch's planes one after another. Winners are
/// `u32`s, each the winner's position in its plane (`row · width + column`), as the CPU's
/// `usize` winners are.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_core::Pool2dGeom;
/// use cavitation_gpu::kernels::MaxPool2d;
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let geom = Pool2dGeom::new(1, (2, 2), (2, 2)).unwrap();
///     let pool2d = MaxPool2d::new(&mut PipelineCache::new(&gpu), &geom, 1).unwrap();
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     let mut buffers = BufferPool::new(&gpu);
///     let (x, out, winners) = (buffers.take(16, usage), buffers.take(4, usage), buffers.take(4, usage));
///     let sizes = StepSizes { upload_bytes: 16, readback_bytes: 16, ..StepSizes::default() };
///     let mut steps = StepEncoder::new(&gpu, sizes);
///     let group = pool2d.forward_bind_group(&mut steps, &x, &out, &winners);
///     let mut step = steps.begin();
///     let pixels: Vec<u8> = [1.0f32, 4.0, 3.0, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
///     step.upload(x.buffer(), 0, &pixels);
///     pool2d.forward(&mut step, &group);
///     let at_out = step.read_back(out.buffer(), 0, 4);
///     let at_winner = step.read_back(winners.buffer(), 0, 4);
///     let bytes = step.finish().bytes();
///     assert_eq!(f32::from_le_bytes(bytes[at_out..][..4].try_into().unwrap()), 4.0);
///     assert_eq!(u32::from_le_bytes(bytes[at_winner..][..4].try_into().unwrap()), 1);
/// }
/// ```
#[derive(Debug)]
pub struct MaxPool2d {
    /// The pooling's geometry.
    geom: Pool2dGeom,
    /// Images per batch.
    batch: usize,
    /// The forward kernel, specialised for the geometry and batch.
    forward: wgpu::ComputePipeline,
    /// The backward kernel, specialised for the geometry and batch.
    backward: wgpu::ComputePipeline,
    /// Outputs in a batch: one forward thread each.
    outputs: u32,
    /// Input pixels in a batch: one backward thread each.
    inputs: u32,
}

impl MaxPool2d {
    /// The forward and backward kernels for `batch` images pooled by `geom`, built (or
    /// found) in `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a kernel does not build on this GPU.
    ///
    /// # Panics
    ///
    /// If `batch` is 0, a batch holds more than 2³¹ inputs or outputs, or an axis's padded
    /// size plus its dilated window and stride does not fit in a `u32`.
    ///
    /// ```
    /// use cavitation_core::Pool2dGeom;
    /// use cavitation_gpu::kernels::MaxPool2d;
    /// use cavitation_gpu::{Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let geom = Pool2dGeom::new(32, (32, 32), (2, 2)).unwrap();
    ///     let pool2d = MaxPool2d::new(&mut PipelineCache::new(&gpu), &geom, 128).unwrap();
    ///     assert_eq!(pool2d.batch(), 128);
    /// }
    /// ```
    pub fn new(
        cache: &mut PipelineCache,
        geom: &Pool2dGeom,
        batch: usize,
    ) -> Result<MaxPool2d, GpuError> {
        assert!(
            batch > 0,
            "max_pool2d: the batch must hold at least one image"
        );
        check_axes("max_pool2d", geom, u32::MAX as usize);
        let outputs = total(batch * geom.out_width());
        let inputs = total(batch * geom.in_width());
        let (dh, dw) = geom.dilation();
        let with_dilation = |threads| {
            let mut constants = geometry_constants(geom, threads);
            constants.extend([("DH", dh as f64), ("DW", dw as f64)]);
            constants
        };
        Ok(MaxPool2d {
            geom: *geom,
            batch,
            forward: cache.get(MAX_FORWARD, "main", &with_dilation(outputs))?,
            backward: cache.get(MAX_BACKWARD, "main", &with_dilation(inputs))?,
            outputs,
            inputs,
        })
    }

    /// The pooling's geometry.
    ///
    /// ```
    /// use cavitation_core::Pool2dGeom;
    /// use cavitation_gpu::kernels::MaxPool2d;
    /// use cavitation_gpu::{Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let geom = Pool2dGeom::new(3, (8, 8), (2, 2)).unwrap();
    ///     let pool2d = MaxPool2d::new(&mut PipelineCache::new(&gpu), &geom, 2).unwrap();
    ///     assert_eq!(pool2d.geom(), &geom);
    /// }
    /// ```
    pub fn geom(&self) -> &Pool2dGeom {
        &self.geom
    }

    /// Images per batch.
    ///
    /// ```
    /// use cavitation_core::Pool2dGeom;
    /// use cavitation_gpu::kernels::MaxPool2d;
    /// use cavitation_gpu::{Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let geom = Pool2dGeom::new(3, (8, 8), (2, 2)).unwrap();
    ///     assert_eq!(MaxPool2d::new(&mut PipelineCache::new(&gpu), &geom, 2).unwrap().batch(), 2);
    /// }
    /// ```
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// The forward pass's bind group, made once: the batch's images `x` (`f32`), its
    /// outputs `out` (`f32`) and their `winners` (`u32`).
    ///
    /// # Panics
    ///
    /// If a buffer does not hold exactly the batch's values, naming it.
    ///
    /// ```
    /// use cavitation_core::Pool2dGeom;
    /// use cavitation_gpu::kernels::MaxPool2d;
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let geom = Pool2dGeom::new(1, (4, 4), (2, 2)).unwrap();
    ///     let pool2d = MaxPool2d::new(&mut PipelineCache::new(&gpu), &geom, 1).unwrap();
    ///     let mut buffers = BufferPool::new(&gpu);
    ///     let usage = wgpu::BufferUsages::STORAGE;
    ///     let (x, out, winners) = (buffers.take(64, usage), buffers.take(16, usage), buffers.take(16, usage));
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let _group = pool2d.forward_bind_group(&mut steps, &x, &out, &winners);
    /// }
    /// ```
    pub fn forward_bind_group(
        &self,
        steps: &mut StepEncoder,
        x: &PooledBuffer,
        out: &PooledBuffer,
        winners: &PooledBuffer,
    ) -> wgpu::BindGroup {
        check_buffer("the input", x, self.inputs as usize);
        check_buffer("the output", out, self.outputs as usize);
        check_buffer("the winners", winners, self.outputs as usize);
        bind(steps, &self.forward, &[x, out, winners])
    }

    /// Records the forward pass in `step`, with a group from
    /// [`forward_bind_group`](Self::forward_bind_group).
    ///
    /// ```
    /// use cavitation_core::Pool2dGeom;
    /// use cavitation_gpu::kernels::MaxPool2d;
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let geom = Pool2dGeom::new(1, (4, 4), (2, 2)).unwrap();
    ///     let pool2d = MaxPool2d::new(&mut PipelineCache::new(&gpu), &geom, 1).unwrap();
    ///     let mut buffers = BufferPool::new(&gpu);
    ///     let usage = wgpu::BufferUsages::STORAGE;
    ///     let (x, out, winners) = (buffers.take(64, usage), buffers.take(16, usage), buffers.take(16, usage));
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let group = pool2d.forward_bind_group(&mut steps, &x, &out, &winners);
    ///     let mut step = steps.begin();
    ///     pool2d.forward(&mut step, &group);
    ///     let _ = step.finish();
    /// }
    /// ```
    pub fn forward(&self, step: &mut Step<'_>, group: &wgpu::BindGroup) {
        dispatch(step, &self.forward, group, self.outputs);
    }

    /// The backward pass's bind group, made once: the output gradient `dy` (`f32`), the
    /// forward pass's `winners` (`u32`) and the input gradient `dx` (`f32`), which the pass
    /// adds to.
    ///
    /// # Panics
    ///
    /// If a buffer does not hold exactly the batch's values, naming it.
    ///
    /// ```
    /// use cavitation_core::Pool2dGeom;
    /// use cavitation_gpu::kernels::MaxPool2d;
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let geom = Pool2dGeom::new(1, (4, 4), (2, 2)).unwrap();
    ///     let pool2d = MaxPool2d::new(&mut PipelineCache::new(&gpu), &geom, 1).unwrap();
    ///     let mut buffers = BufferPool::new(&gpu);
    ///     let usage = wgpu::BufferUsages::STORAGE;
    ///     let (dy, winners, dx) = (buffers.take(16, usage), buffers.take(16, usage), buffers.take(64, usage));
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let _group = pool2d.backward_bind_group(&mut steps, &dy, &winners, &dx);
    /// }
    /// ```
    pub fn backward_bind_group(
        &self,
        steps: &mut StepEncoder,
        dy: &PooledBuffer,
        winners: &PooledBuffer,
        dx: &PooledBuffer,
    ) -> wgpu::BindGroup {
        check_buffer("the output gradient", dy, self.outputs as usize);
        check_buffer("the winners", winners, self.outputs as usize);
        check_buffer("the input gradient", dx, self.inputs as usize);
        bind(steps, &self.backward, &[dy, winners, dx])
    }

    /// Records the backward pass in `step`, with a group from
    /// [`backward_bind_group`](Self::backward_bind_group): adds each output gradient to its
    /// winner's input gradient.
    ///
    /// ```
    /// use cavitation_core::Pool2dGeom;
    /// use cavitation_gpu::kernels::MaxPool2d;
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let geom = Pool2dGeom::new(1, (4, 4), (2, 2)).unwrap();
    ///     let pool2d = MaxPool2d::new(&mut PipelineCache::new(&gpu), &geom, 1).unwrap();
    ///     let mut buffers = BufferPool::new(&gpu);
    ///     let usage = wgpu::BufferUsages::STORAGE;
    ///     let (dy, winners, dx) = (buffers.take(16, usage), buffers.take(16, usage), buffers.take(64, usage));
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let group = pool2d.backward_bind_group(&mut steps, &dy, &winners, &dx);
    ///     let mut step = steps.begin();
    ///     pool2d.backward(&mut step, &group);
    ///     let _ = step.finish();
    /// }
    /// ```
    pub fn backward(&self, step: &mut Step<'_>, group: &wgpu::BindGroup) {
        dispatch(step, &self.backward, group, self.inputs);
    }
}

/// Average pooling on the GPU for one geometry, batch size and divisor rule, as
/// `cavitation_core::kernels::pooling::avg_pool2d` and `avg_pool2d_backward` compute it:
/// each output is its window's in-image sum divided by `divisor_override` if given, else by
/// the window's size counting the padding (`count_include_pad`), else by the pixels summed.
///
/// The sums may be added in any order (some backends reassociate them) and the division is
/// WGSL's, accurate to 2.5 ulp, so the results agree with the exact values of the f32 inputs
/// within a rounding bound, not bit for bit. On normal inputs, an output summing at most
/// `n` pixels with divisor `d` is within `(2(n + 2) + 6) · 2⁻²⁴ · Σ|x| / d + 2⁻¹²⁶`, and an
/// input gradient that at most `m` outputs cover within
/// `(2(m + 3) + 5) · 2⁻²⁴ · (|dx| + Σ|dy / d|) + 2⁻¹²⁶`, the absolute term covering a result
/// the GPU flushes to zero. Subnormal inputs are outside the bound: the GPU may read them as
/// zero.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_core::Pool2dGeom;
/// use cavitation_gpu::kernels::AvgPool2d;
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let geom = Pool2dGeom::new(1, (2, 2), (2, 2)).unwrap();
///     let pool2d = AvgPool2d::new(&mut PipelineCache::new(&gpu), &geom, 1, true, None).unwrap();
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     let mut buffers = BufferPool::new(&gpu);
///     let (x, out) = (buffers.take(16, usage), buffers.take(4, usage));
///     let sizes = StepSizes { upload_bytes: 16, readback_bytes: 4, ..StepSizes::default() };
///     let mut steps = StepEncoder::new(&gpu, sizes);
///     let group = pool2d.forward_bind_group(&mut steps, &x, &out);
///     let mut step = steps.begin();
///     let pixels: Vec<u8> = [1.0f32, 4.0, 3.0, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
///     step.upload(x.buffer(), 0, &pixels);
///     pool2d.forward(&mut step, &group);
///     step.read_back(out.buffer(), 0, 4);
///     let bytes = step.finish().bytes();
///     assert_eq!(f32::from_le_bytes(bytes.try_into().unwrap()), 2.5);
/// }
/// ```
#[derive(Debug)]
pub struct AvgPool2d {
    /// The pooling's geometry.
    geom: Pool2dGeom,
    /// Images per batch.
    batch: usize,
    /// The forward kernel, specialised for the geometry, batch and divisor rule.
    forward: wgpu::ComputePipeline,
    /// The backward kernel, specialised for the geometry, batch and divisor rule.
    backward: wgpu::ComputePipeline,
    /// Outputs in a batch: one forward thread each.
    outputs: u32,
    /// Input pixels in a batch: one backward thread each.
    inputs: u32,
}

impl AvgPool2d {
    /// The forward and backward kernels for `batch` images pooled by `geom`, dividing by
    /// `divisor_override` if given, else by the window counting the padding when
    /// `count_include_pad`, else by the pixels summed; built (or found) in `cache`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if a kernel does not build on this GPU.
    ///
    /// # Panics
    ///
    /// As the CPU's `avg_pool2d`, if the geometry is dilated or `divisor_override` is 0;
    /// also if `batch` is 0, a batch holds more than 2³¹ inputs or outputs, the window's size
    /// or the divisor override does not fit in a `u32`, or an axis's padded size plus its
    /// window and stride does not.
    ///
    /// ```
    /// use cavitation_core::Pool2dGeom;
    /// use cavitation_gpu::kernels::AvgPool2d;
    /// use cavitation_gpu::{Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let geom = Pool2dGeom::new(16, (8, 8), (2, 2)).unwrap();
    ///     let pool2d = AvgPool2d::new(&mut PipelineCache::new(&gpu), &geom, 64, true, None).unwrap();
    ///     assert_eq!(pool2d.batch(), 64);
    /// }
    /// ```
    pub fn new(
        cache: &mut PipelineCache,
        geom: &Pool2dGeom,
        batch: usize,
        count_include_pad: bool,
        divisor_override: Option<usize>,
    ) -> Result<AvgPool2d, GpuError> {
        assert!(
            batch > 0,
            "avg_pool2d: the batch must hold at least one image"
        );
        assert!(
            geom.dilation() == (1, 1),
            "avg_pool2d: average pooling has no dilation, got {:?}",
            geom.dilation()
        );
        assert!(
            divisor_override != Some(0),
            "avg_pool2d: the divisor override must not be 0"
        );
        check_axes("avg_pool2d", geom, u32::MAX as usize);
        let (kh, kw) = geom.kernel();
        assert!(
            kh.checked_mul(kw).is_some_and(|n| n <= u32::MAX as usize)
                && divisor_override.unwrap_or(0) <= u32::MAX as usize,
            "avg_pool2d: the window or divisor does not fit the kernel's u32s: {geom:?}, \
             divisor {divisor_override:?}"
        );
        let outputs = total(batch * geom.out_width());
        let inputs = total(batch * geom.in_width());
        let with_divisor = |threads| {
            let mut constants = geometry_constants(geom, threads);
            constants.extend([
                ("COUNT_PAD", f64::from(u8::from(count_include_pad))),
                ("DIVISOR", divisor_override.unwrap_or(0) as f64),
            ]);
            constants
        };
        Ok(AvgPool2d {
            geom: *geom,
            batch,
            forward: cache.get(AVG_FORWARD, "main", &with_divisor(outputs))?,
            backward: cache.get(AVG_BACKWARD, "main", &with_divisor(inputs))?,
            outputs,
            inputs,
        })
    }

    /// The pooling's geometry.
    ///
    /// ```
    /// use cavitation_core::Pool2dGeom;
    /// use cavitation_gpu::kernels::AvgPool2d;
    /// use cavitation_gpu::{Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let geom = Pool2dGeom::new(3, (8, 8), (2, 2)).unwrap();
    ///     let pool2d = AvgPool2d::new(&mut PipelineCache::new(&gpu), &geom, 2, true, None).unwrap();
    ///     assert_eq!(pool2d.geom(), &geom);
    /// }
    /// ```
    pub fn geom(&self) -> &Pool2dGeom {
        &self.geom
    }

    /// Images per batch.
    ///
    /// ```
    /// use cavitation_core::Pool2dGeom;
    /// use cavitation_gpu::kernels::AvgPool2d;
    /// use cavitation_gpu::{Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let geom = Pool2dGeom::new(3, (8, 8), (2, 2)).unwrap();
    ///     let pool2d = AvgPool2d::new(&mut PipelineCache::new(&gpu), &geom, 2, true, None).unwrap();
    ///     assert_eq!(pool2d.batch(), 2);
    /// }
    /// ```
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// The forward pass's bind group, made once: the batch's images `x` and its outputs
    /// `out`, both `f32`.
    ///
    /// # Panics
    ///
    /// If a buffer does not hold exactly the batch's values, naming it.
    ///
    /// ```
    /// use cavitation_core::Pool2dGeom;
    /// use cavitation_gpu::kernels::AvgPool2d;
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let geom = Pool2dGeom::new(1, (4, 4), (2, 2)).unwrap();
    ///     let pool2d = AvgPool2d::new(&mut PipelineCache::new(&gpu), &geom, 1, true, None).unwrap();
    ///     let mut buffers = BufferPool::new(&gpu);
    ///     let (x, out) = (buffers.take(64, wgpu::BufferUsages::STORAGE), buffers.take(16, wgpu::BufferUsages::STORAGE));
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let _group = pool2d.forward_bind_group(&mut steps, &x, &out);
    /// }
    /// ```
    pub fn forward_bind_group(
        &self,
        steps: &mut StepEncoder,
        x: &PooledBuffer,
        out: &PooledBuffer,
    ) -> wgpu::BindGroup {
        check_buffer("the input", x, self.inputs as usize);
        check_buffer("the output", out, self.outputs as usize);
        bind(steps, &self.forward, &[x, out])
    }

    /// Records the forward pass in `step`, with a group from
    /// [`forward_bind_group`](Self::forward_bind_group).
    ///
    /// ```
    /// use cavitation_core::Pool2dGeom;
    /// use cavitation_gpu::kernels::AvgPool2d;
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let geom = Pool2dGeom::new(1, (4, 4), (2, 2)).unwrap();
    ///     let pool2d = AvgPool2d::new(&mut PipelineCache::new(&gpu), &geom, 1, true, None).unwrap();
    ///     let mut buffers = BufferPool::new(&gpu);
    ///     let (x, out) = (buffers.take(64, wgpu::BufferUsages::STORAGE), buffers.take(16, wgpu::BufferUsages::STORAGE));
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let group = pool2d.forward_bind_group(&mut steps, &x, &out);
    ///     let mut step = steps.begin();
    ///     pool2d.forward(&mut step, &group);
    ///     let _ = step.finish();
    /// }
    /// ```
    pub fn forward(&self, step: &mut Step<'_>, group: &wgpu::BindGroup) {
        dispatch(step, &self.forward, group, self.outputs);
    }

    /// The backward pass's bind group, made once: the output gradient `dy` and the input
    /// gradient `dx`, which the pass adds to, both `f32`.
    ///
    /// # Panics
    ///
    /// If a buffer does not hold exactly the batch's values, naming it.
    ///
    /// ```
    /// use cavitation_core::Pool2dGeom;
    /// use cavitation_gpu::kernels::AvgPool2d;
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let geom = Pool2dGeom::new(1, (4, 4), (2, 2)).unwrap();
    ///     let pool2d = AvgPool2d::new(&mut PipelineCache::new(&gpu), &geom, 1, true, None).unwrap();
    ///     let mut buffers = BufferPool::new(&gpu);
    ///     let (dy, dx) = (buffers.take(16, wgpu::BufferUsages::STORAGE), buffers.take(64, wgpu::BufferUsages::STORAGE));
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let _group = pool2d.backward_bind_group(&mut steps, &dy, &dx);
    /// }
    /// ```
    pub fn backward_bind_group(
        &self,
        steps: &mut StepEncoder,
        dy: &PooledBuffer,
        dx: &PooledBuffer,
    ) -> wgpu::BindGroup {
        check_buffer("the output gradient", dy, self.outputs as usize);
        check_buffer("the input gradient", dx, self.inputs as usize);
        bind(steps, &self.backward, &[dy, dx])
    }

    /// Records the backward pass in `step`, with a group from
    /// [`backward_bind_group`](Self::backward_bind_group): adds each output gradient,
    /// divided by its divisor, to every in-image pixel of its window.
    ///
    /// ```
    /// use cavitation_core::Pool2dGeom;
    /// use cavitation_gpu::kernels::AvgPool2d;
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let geom = Pool2dGeom::new(1, (4, 4), (2, 2)).unwrap();
    ///     let pool2d = AvgPool2d::new(&mut PipelineCache::new(&gpu), &geom, 1, true, None).unwrap();
    ///     let mut buffers = BufferPool::new(&gpu);
    ///     let (dy, dx) = (buffers.take(16, wgpu::BufferUsages::STORAGE), buffers.take(64, wgpu::BufferUsages::STORAGE));
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let group = pool2d.backward_bind_group(&mut steps, &dy, &dx);
    ///     let mut step = steps.begin();
    ///     pool2d.backward(&mut step, &group);
    ///     let _ = step.finish();
    /// }
    /// ```
    pub fn backward(&self, step: &mut Step<'_>, group: &wgpu::BindGroup) {
        dispatch(step, &self.backward, group, self.inputs);
    }
}
