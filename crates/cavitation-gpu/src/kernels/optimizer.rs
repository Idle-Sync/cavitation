//! SGD (with momentum and weight decay), Adam and AdamW on the GPU: one dispatch updates
//! every parameter of a model, held in flat buffers, with the arithmetic of the CPU
//! optimisers in `cavitation_nn::optim` done in `f32`.
//!
//! The sizes and the per-parameter constants (bias corrections, which parameters update)
//! are in buffers, not `override` constants, so one pipeline serves every model with the
//! same rule. The host keeps each parameter's update count and commits it only once a
//! step's readback shows the update ran.

use super::bind_buffers;
use crate::{
    BufferPool, Gpu, GpuError, PipelineCache, PooledBuffer, Shader, Step, StepEncoder, wgpu,
};

/// Threads per workgroup, and parameter elements per workgroup: every optimiser kernel's
/// `@workgroup_size`, passed to it as the `WORKGROUP` override.
const WORKGROUP: u32 = 256;

/// The most elements a model's parameters may hold together: the kernels index them with
/// `u32`s, with room for a workgroup's last lanes past the end.
const MAX_ELEMENTS: usize = 1 << 31;

/// Bytes of the shared constants' uniform buffer: [`SHARED_WORDS`] `f32`s.
const SHARED_BYTES: u64 = 4 * SHARED_WORDS as u64;

/// `f32`s in the shared constants: nine, padded to whole 16-byte rows, the unit HLSL lays a
/// constant buffer out in.
const SHARED_WORDS: usize = 12;

/// Bytes of one parameter's constants in the per-parameter buffer.
const INFO_BYTES: usize = 16;

/// The parts every optimiser kernel shares: the bindings of the values, gradients, the
/// workgroup table, the per-parameter constants, the shared constants and the finite flag;
/// the moment flush; and the entry point, which does nothing when the flag is set, maps
/// its workgroup to one parameter's run of elements, and runs the kernel's `update(i, info)`
/// on each element of a parameter that updates this step.
macro_rules! optimizer_common {
    () => {
        "
        override WORKGROUP: u32;

        struct Shared {
            lr: f32,
            momentum: f32,
            weight_decay: f32,
            beta1: f32,
            beta2: f32,
            one_minus_beta1: f32,
            one_minus_beta2: f32,
            eps: f32,
            shrink: f32,
            pad0: f32,
            pad1: f32,
            pad2: f32,
        }

        // One parameter: its end in the flat buffers, bit 0 set when it updates this step
        // and bit 1 on its first update, and its Adam bias corrections for this step.
        struct Info {
            end: u32,
            flags: u32,
            step_size: f32,
            inv_sqrt_correction2: f32,
        }

        @group(0) @binding(0) var<storage, read_write> values: array<f32>;
        @group(0) @binding(1) var<storage, read> grads: array<f32>;
        @group(0) @binding(2) var<storage, read> table: array<vec2<u32>>;
        @group(0) @binding(3) var<storage, read> infos: array<Info>;
        @group(0) @binding(4) var<uniform> consts: Shared;
        @group(0) @binding(5) var<storage, read> finite: array<u32>;

        // A moment stored for the next step: 0 below the smallest normal f32, as the CPU
        // optimisers store their state.
        fn store_state(x: f32) -> f32 {
            return select(x, 0.0, (bitcast<u32>(x) & 0x7f800000u) == 0u);
        }

        @compute @workgroup_size(WORKGROUP)
        fn main(
            @builtin(workgroup_id) group: vec3<u32>,
            @builtin(num_workgroups) groups: vec3<u32>,
            @builtin(local_invocation_index) lane: u32,
        ) {
            if (finite[0] != 0u) { return; }
            let w = group.x + group.y * groups.x;
            if (w >= arrayLength(&table)) { return; }
            let slot = table[w];
            let info = infos[slot.x];
            let i = slot.y + lane;
            if ((info.flags & 1u) == 0u || i >= info.end) { return; }
            update(i, info);
        }
        "
    };
}

/// SGD without momentum: `p ← p − lr·(g + wd·p)`, the decay term left out when `wd` is 0, as
/// the CPU leaves it out.
const SGD: Shader = Shader {
    name: "optimizer sgd",
    source: concat!(
        optimizer_common!(),
        "
        fn update(i: u32, info: Info) {
            let p = values[i];
            var step = grads[i];
            if (consts.weight_decay != 0.0) { step = fma(consts.weight_decay, p, step); }
            values[i] = fma(-consts.lr, step, p);
        }
        "
    ),
};

/// SGD with momentum: `b ← μ·b + (g + wd·p)`, `b` set to the step itself on a parameter's
/// first update, then `p ← p − lr·b`.
const SGD_MOMENTUM: Shader = Shader {
    name: "optimizer sgd momentum",
    source: concat!(
        optimizer_common!(),
        "
        @group(0) @binding(6) var<storage, read_write> buffer: array<f32>;

        fn update(i: u32, info: Info) {
            let p = values[i];
            var step = grads[i];
            if (consts.weight_decay != 0.0) { step = fma(consts.weight_decay, p, step); }
            if ((info.flags & 2u) == 0u) { step = fma(consts.momentum, buffer[i], step); }
            buffer[i] = store_state(step);
            values[i] = fma(-consts.lr, step, p);
        }
        "
    ),
};

/// Adam, or AdamW when `DECOUPLED`: AdamW first shrinks `p` by `1 − lr·wd`, Adam adds
/// `wd·p` to the gradient; then the moments `m ← β₁m + (1 − β₁)g` and
/// `v ← β₂v + (1 − β₂)g²` are stored and `p ← p − step_size · m / (√v · c₂ + eps)`, with the
/// bias corrections `step_size = lr / (1 − β₁ᵗ)` and `c₂ = 1 / √(1 − β₂ᵗ)` the parameter's
/// own.
const ADAM: Shader = Shader {
    name: "optimizer adam",
    source: concat!(
        optimizer_common!(),
        "
        override DECOUPLED: bool;

        @group(0) @binding(6) var<storage, read_write> m: array<f32>;
        @group(0) @binding(7) var<storage, read_write> v: array<f32>;

        fn update(i: u32, info: Info) {
            var p = values[i];
            var g = grads[i];
            if (DECOUPLED) {
                p = p * consts.shrink;
            } else if (consts.weight_decay != 0.0) {
                g = fma(consts.weight_decay, p, g);
            }
            let mi = fma(consts.beta1, m[i], consts.one_minus_beta1 * g);
            let vi = fma(consts.beta2, v[i], consts.one_minus_beta2 * (g * g));
            m[i] = store_state(mi);
            v[i] = store_state(vi);
            let denominator = fma(sqrt(vi), info.inv_sqrt_correction2, consts.eps);
            values[i] = fma(-info.step_size, mi / denominator, p);
        }
        "
    ),
};

/// An update rule [`OptimizerKernel`] runs, with its settings: the GPU forms of the CPU's
/// `Sgd`, `Adam` and `AdamW`. RMSprop and Adagrad have no GPU form.
///
/// Every setting is explicit. The CPU's `OptimizerConfig` leaves Adam's and AdamW's betas
/// and eps at their defaults, (0.9, 0.999) and 1e-8, so a rule built from it takes those
/// values, as [`adam`](Self::adam) and [`adam_w`](Self::adam_w) give them.
///
/// ```
/// use cavitation_gpu::kernels::OptimizerRule;
/// let adam = OptimizerRule::adam(1e-3, 0.0);
/// assert_eq!(adam, OptimizerRule::Adam { lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0 });
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizerRule {
    /// SGD: `b ← μ·b + (g + wd·p)` (`b` the step itself on a parameter's first update, no
    /// buffer at all when `momentum` is 0) and `p ← p − lr·b`.
    Sgd {
        /// The learning rate.
        lr: f64,
        /// The momentum factor; 0 for none.
        momentum: f64,
        /// The weight decay added to the gradient; 0 for none.
        weight_decay: f64,
    },
    /// Adam, as PyTorch's `torch.optim.Adam`, weight decay added to the gradient.
    Adam {
        /// The learning rate.
        lr: f64,
        /// The first moment's decay rate.
        beta1: f64,
        /// The second moment's decay rate.
        beta2: f64,
        /// Added to the corrected second moment's square root.
        eps: f64,
        /// The weight decay added to the gradient; 0 for none.
        weight_decay: f64,
    },
    /// AdamW: Adam with the parameter shrunk by `1 − lr·wd` before the update.
    AdamW {
        /// The learning rate.
        lr: f64,
        /// The first moment's decay rate.
        beta1: f64,
        /// The second moment's decay rate.
        beta2: f64,
        /// Added to the corrected second moment's square root.
        eps: f64,
        /// The decoupled weight decay; 0 for none.
        weight_decay: f64,
    },
}

impl OptimizerRule {
    /// Adam with learning rate `lr` and weight decay `weight_decay`, and the CPU's (and
    /// PyTorch's) defaults: betas (0.9, 0.999) and eps 1e-8.
    ///
    /// ```
    /// use cavitation_gpu::kernels::OptimizerRule;
    /// assert!(matches!(OptimizerRule::adam(1e-3, 0.0), OptimizerRule::Adam { beta1: 0.9, .. }));
    /// ```
    pub fn adam(lr: f64, weight_decay: f64) -> OptimizerRule {
        OptimizerRule::Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
        }
    }

    /// AdamW with learning rate `lr` and decoupled weight decay `weight_decay`, betas
    /// (0.9, 0.999) and eps 1e-8.
    ///
    /// ```
    /// use cavitation_gpu::kernels::OptimizerRule;
    /// assert!(matches!(OptimizerRule::adam_w(1e-3, 1e-2), OptimizerRule::AdamW { eps: 1e-8, .. }));
    /// ```
    pub fn adam_w(lr: f64, weight_decay: f64) -> OptimizerRule {
        OptimizerRule::AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
        }
    }

    /// The learning rate.
    ///
    /// ```
    /// use cavitation_gpu::kernels::OptimizerRule;
    /// assert_eq!(OptimizerRule::Sgd { lr: 0.1, momentum: 0.9, weight_decay: 0.0 }.learning_rate(), 0.1);
    /// ```
    pub fn learning_rate(&self) -> f64 {
        match *self {
            OptimizerRule::Sgd { lr, .. }
            | OptimizerRule::Adam { lr, .. }
            | OptimizerRule::AdamW { lr, .. } => lr,
        }
    }

    /// How many state buffers the rule keeps, each as long as the parameters: SGD's momentum
    /// buffer (none without momentum), or Adam's two moments.
    ///
    /// ```
    /// use cavitation_gpu::kernels::OptimizerRule;
    /// assert_eq!(OptimizerRule::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 }.state_buffers(), 0);
    /// assert_eq!(OptimizerRule::adam(1e-3, 0.0).state_buffers(), 2);
    /// ```
    pub fn state_buffers(&self) -> usize {
        match *self {
            OptimizerRule::Sgd { momentum, .. } => usize::from(momentum != 0.0),
            OptimizerRule::Adam { .. } | OptimizerRule::AdamW { .. } => 2,
        }
    }

    /// The settings as `[lr, momentum, weight decay, β₁, β₂, eps]`, 0 for those the rule
    /// does not have.
    fn settings(&self) -> [f64; 6] {
        match *self {
            OptimizerRule::Sgd {
                lr,
                momentum,
                weight_decay,
            } => [lr, momentum, weight_decay, 0.0, 0.0, 0.0],
            OptimizerRule::Adam {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            }
            | OptimizerRule::AdamW {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            } => [lr, 0.0, weight_decay, beta1, beta2, eps],
        }
    }

    /// Panics, naming the rule, unless every setting is finite once rounded to the `f32` the
    /// kernel receives, and the betas are in `[0, 1)`, as the CPU's Adam requires.
    fn check(&self) {
        let [.., beta1, beta2, _] = self.settings();
        assert!(
            (0.0..1.0).contains(&beta1) && (0.0..1.0).contains(&beta2),
            "optimizer: Adam betas must be in [0, 1), got ({beta1}, {beta2})"
        );
        assert!(
            self.settings().iter().all(|&v| (v as f32).is_finite()),
            "optimizer: every setting must be finite as an f32, got {self:?}"
        );
    }
}

/// SGD, momentum, Adam or AdamW on the GPU for one model's parameters: a single dispatch
/// updates them all, with each parameter's own update count for Adam's bias corrections,
/// and does nothing when the step's finite flag is set.
///
/// The parameters live one after another in flat `f32` buffers: the values, the gradients
/// and the rule's [state buffers](OptimizerRule::state_buffers), each as long as all the
/// parameters together, in the order of `lens`.
///
/// The arithmetic is the CPU optimisers' in `f32`, each step from the `f32`-rounded
/// settings and per-step constants (computed on the host in `f64`). Against the exact result
/// of those constants and the step's starting state, with `u = 2⁻²⁴`, the bounds below hold
/// whether each `fma` rounds once or twice (WGSL allows either; DX12's FXC splits Adam's,
/// as measured on the NVIDIA GTX 1660 Super), except plain SGD's single rounding:
///
/// - **SGD without weight decay or momentum:** the value `p' = p − lr·g` is one `fma`,
///   within `u·|p'|`, one rounding, where the backend fuses it (measured on Metal, Vulkan
///   and both DX12 compilers); within `u·|p'| + u·(1 + u)·lr·|g|` where it does not.
/// - **SGD with either:** for the step `s = g + wd·p`, the buffer `b = μ·b_old + s` (`b = s`
///   on a parameter's first update or without momentum) and
///   `e_b = u·(|b| + μ·|b_old| + |s| + |wd·p|)`, the momentum buffer within `e_b` and the
///   value within `u·(|p'| + lr·|b|) + lr·e_b`.
/// - **Adam and AdamW:** for the gradient `g` after Adam's decay, the new moments `m` and
///   `v`, the denominator `d = √v · c₂ + eps`, AdamW's shrunk value `p_s = p·(1 − lr·wd)` (0
///   for Adam) and Adam's decay term `w = |wd·p|` (0 for AdamW), the first moment within
///   `e_m = u·(|m| + β₁·|m_old| + 2(1 − β₁)|g| + (1 − β₁)·w)`, the second within
///   `e_v = u·(6v + 2(1 − β₂)|g|·w)`, and the value within
///   `u·(|p'| + |p_s|) + step_size·(|m/d|·(17u + e_v/(2v)) + e_m/d)`. These allow for WGSL's
///   square root (from `inverseSqrt`, 2 ulp, then a division) and its division (2.5 ulp).
///
/// Every bound also carries `2⁻¹²⁶`, for a value or state the GPU flushes to zero.
///
/// A step runs in three parts: [`record`](Self::record) uploads the step's constants and
/// dispatches, the caller finishes the step and reads its finite flag back, and
/// [`commit`](Self::commit) counts the update for each parameter it ran on, or for none when
/// the flag stopped it.
///
/// # Examples
///
/// Without a GPU the example does nothing:
///
/// ```
/// use cavitation_gpu::kernels::{OptimizerKernel, OptimizerRule};
/// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
///
/// if let Some(gpu) = Gpu::new() {
///     let mut pool = BufferPool::new(&gpu);
///     let rule = OptimizerRule::Sgd { lr: 0.5, momentum: 0.0, weight_decay: 0.0 };
///     let mut sgd = OptimizerKernel::new(&gpu, &mut PipelineCache::new(&gpu), &mut pool, rule, &[2]).unwrap();
///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
///         | wgpu::BufferUsages::COPY_SRC;
///     let (values, grads, flag) = (pool.take(8, usage), pool.take(8, usage), pool.take(4, usage));
///     let sizes = StepSizes { upload_bytes: 16 + sgd.upload_bytes(), readback_bytes: 8, ..StepSizes::default() };
///     let mut steps = StepEncoder::new(&gpu, sizes);
///     let group = sgd.bind_group(&mut steps, &values, &grads, &[], &flag);
///     let mut step = steps.begin();
///     let bytes = |v: [f32; 2]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
///     step.upload(values.buffer(), 0, &bytes([1.0, 2.0]));
///     step.upload(grads.buffer(), 0, &bytes([1.0, -1.0]));
///     step.clear(flag.buffer(), 0, 4);
///     sgd.record(&mut step, &group, &[true]);
///     let at = step.read_back(values.buffer(), 0, 8);
///     let out = step.finish().bytes();
///     assert_eq!(f32::from_le_bytes(out[at..][..4].try_into().unwrap()), 0.5);
///     sgd.commit(true);
///     assert_eq!(sgd.steps(), &[1]);
/// }
/// ```
#[derive(Debug)]
pub struct OptimizerKernel {
    /// The rule and its settings, the learning rate the current one.
    rule: OptimizerRule,
    /// Each parameter's length.
    lens: Vec<usize>,
    /// The elements of all parameters together.
    elements: usize,
    /// The rule's kernel.
    pipeline: wgpu::ComputePipeline,
    /// For each workgroup, its parameter and the first element it updates, written once.
    table: PooledBuffer,
    /// Each parameter's constants for the step being recorded.
    infos: PooledBuffer,
    /// The per-parameter constants' bytes, rebuilt in place every step.
    info_bytes: Vec<u8>,
    /// The shared constants, a uniform buffer.
    shared: PooledBuffer,
    /// Workgroups per dispatch: one per [`WORKGROUP`] elements of each parameter.
    workgroups: u32,
    /// Each parameter's committed update count.
    steps: Vec<u32>,
    /// Which parameters the recorded step updates, until it is committed.
    pending: Vec<bool>,
    /// Whether a recorded step awaits [`commit`](Self::commit).
    recorded: bool,
}

impl OptimizerKernel {
    /// The kernel for `rule` over parameters of `lens` elements each on `gpu`, built (or
    /// found) in `cache`, with its own small buffers (the workgroup table and the constants)
    /// from `pool`, which [`release`](Self::release) hands back. The workgroup table is
    /// written here, through `gpu`'s queue, so it is in place for the first step. Every
    /// parameter starts with no updates.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shader`] if the kernel does not build on this GPU.
    ///
    /// # Panics
    ///
    /// If `lens` is empty or holds a 0, the parameters hold more than 2³¹ elements together,
    /// a setting is not finite as an `f32`, or Adam's betas are outside `[0, 1)`.
    ///
    /// ```
    /// use cavitation_gpu::kernels::{OptimizerKernel, OptimizerRule};
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let rule = OptimizerRule::adam(1e-3, 0.0);
    ///     let adam = OptimizerKernel::new(&gpu, &mut PipelineCache::new(&gpu), &mut pool, rule, &[784 * 128, 128]).unwrap();
    ///     assert_eq!(adam.steps(), &[0, 0]);
    /// }
    /// ```
    pub fn new(
        gpu: &Gpu,
        cache: &mut PipelineCache,
        pool: &mut BufferPool,
        rule: OptimizerRule,
        lens: &[usize],
    ) -> Result<OptimizerKernel, GpuError> {
        rule.check();
        assert!(
            !lens.is_empty() && lens.iter().all(|&n| n > 0),
            "optimizer: every parameter must hold at least one element, got lengths {lens:?}"
        );
        let elements = lens.iter().sum::<usize>();
        assert!(
            elements <= MAX_ELEMENTS,
            "optimizer: {elements} elements is more than the {MAX_ELEMENTS} the kernel takes"
        );
        let workgroup = WORKGROUP as usize;
        let mut table_bytes = Vec::new();
        let mut start = 0;
        for (k, &n) in lens.iter().enumerate() {
            for first in (start..start + n).step_by(workgroup) {
                table_bytes.extend((k as u32).to_le_bytes());
                table_bytes.extend((first as u32).to_le_bytes());
            }
            start += n;
        }
        let workgroups = (table_bytes.len() / 8) as u32;
        let constants = [("WORKGROUP", f64::from(WORKGROUP))];
        let pipeline = match rule {
            OptimizerRule::Sgd { momentum: 0.0, .. } => cache.get(SGD, "main", &constants)?,
            OptimizerRule::Sgd { .. } => cache.get(SGD_MOMENTUM, "main", &constants)?,
            OptimizerRule::Adam { .. } | OptimizerRule::AdamW { .. } => {
                let decoupled = f64::from(u8::from(matches!(rule, OptimizerRule::AdamW { .. })));
                cache.get(ADAM, "main", &[constants[0], ("DECOUPLED", decoupled)])?
            }
        };
        let read = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let table = pool.take(table_bytes.len() as u64, read);
        gpu.queue().write_buffer(table.buffer(), 0, &table_bytes);
        Ok(OptimizerKernel {
            rule,
            lens: lens.to_vec(),
            elements,
            pipeline,
            table,
            infos: pool.take((INFO_BYTES * lens.len()) as u64, read),
            info_bytes: vec![0; INFO_BYTES * lens.len()],
            shared: pool.take(
                SHARED_BYTES,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            workgroups,
            steps: vec![0; lens.len()],
            pending: vec![false; lens.len()],
            recorded: false,
        })
    }

    /// The rule, with the current learning rate.
    ///
    /// ```
    /// use cavitation_gpu::kernels::{OptimizerKernel, OptimizerRule};
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let rule = OptimizerRule::adam_w(1e-3, 1e-2);
    ///     let kernel = OptimizerKernel::new(&gpu, &mut PipelineCache::new(&gpu), &mut BufferPool::new(&gpu), rule, &[4]).unwrap();
    ///     assert_eq!(kernel.rule(), rule);
    /// }
    /// ```
    pub fn rule(&self) -> OptimizerRule {
        self.rule
    }

    /// Sets the learning rate the next recorded step uses, as a schedule does between steps.
    ///
    /// # Panics
    ///
    /// If `lr` is not finite.
    ///
    /// ```
    /// use cavitation_gpu::kernels::{OptimizerKernel, OptimizerRule};
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let rule = OptimizerRule::adam(1e-3, 0.0);
    ///     let mut kernel = OptimizerKernel::new(&gpu, &mut PipelineCache::new(&gpu), &mut BufferPool::new(&gpu), rule, &[4]).unwrap();
    ///     kernel.set_learning_rate(5e-4);
    ///     assert_eq!(kernel.rule().learning_rate(), 5e-4);
    /// }
    /// ```
    pub fn set_learning_rate(&mut self, lr: f64) {
        assert!(
            lr.is_finite(),
            "optimizer: the learning rate must be finite, got {lr}"
        );
        match &mut self.rule {
            OptimizerRule::Sgd { lr: rate, .. }
            | OptimizerRule::Adam { lr: rate, .. }
            | OptimizerRule::AdamW { lr: rate, .. } => *rate = lr,
        }
    }

    /// Each parameter's committed update count, in the order of the lengths it was made
    /// with.
    ///
    /// ```
    /// use cavitation_gpu::kernels::{OptimizerKernel, OptimizerRule};
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let rule = OptimizerRule::adam(1e-3, 0.0);
    ///     let kernel = OptimizerKernel::new(&gpu, &mut PipelineCache::new(&gpu), &mut BufferPool::new(&gpu), rule, &[4, 2]).unwrap();
    ///     assert_eq!(kernel.steps(), &[0, 0]);
    /// }
    /// ```
    pub fn steps(&self) -> &[u32] {
        &self.steps
    }

    /// The bytes a step that records this kernel uploads: the per-parameter and shared
    /// constants. A step encoder's [`upload_bytes`](crate::StepSizes::upload_bytes) must
    /// leave room for them.
    ///
    /// ```
    /// use cavitation_gpu::kernels::{OptimizerKernel, OptimizerRule};
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let rule = OptimizerRule::adam(1e-3, 0.0);
    ///     let kernel = OptimizerKernel::new(&gpu, &mut PipelineCache::new(&gpu), &mut BufferPool::new(&gpu), rule, &[256]).unwrap();
    ///     assert_eq!(kernel.upload_bytes(), 16 + 48);
    /// }
    /// ```
    pub fn upload_bytes(&self) -> u64 {
        self.info_bytes.len() as u64 + SHARED_BYTES
    }

    /// The bind group, made once: the parameters' `values` and `grads`, the rule's `state`
    /// buffers (SGD's momentum buffer, or Adam's first and second moments; none for SGD
    /// without momentum), all `f32` and as long as the parameters together, and the step's
    /// `finite` flag, a `u32` at offset 0 that is 0 when the step's gradients are all
    /// finite and any other value to skip the update.
    ///
    /// # Panics
    ///
    /// If a buffer does not hold exactly the parameters' elements, or `state` holds the
    /// wrong number of buffers, naming the buffer.
    ///
    /// ```
    /// use cavitation_gpu::kernels::{OptimizerKernel, OptimizerRule};
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let rule = OptimizerRule::adam(1e-3, 0.0);
    ///     let adam = OptimizerKernel::new(&gpu, &mut PipelineCache::new(&gpu), &mut pool, rule, &[3, 5]).unwrap();
    ///     let usage = wgpu::BufferUsages::STORAGE;
    ///     let [values, grads, m, v] = [0; 4].map(|_| pool.take(32, usage));
    ///     let flag = pool.take(4, usage);
    ///     let mut steps = StepEncoder::new(&gpu, StepSizes::default());
    ///     let _group = adam.bind_group(&mut steps, &values, &grads, &[&m, &v], &flag);
    /// }
    /// ```
    pub fn bind_group(
        &self,
        steps: &mut StepEncoder,
        values: &PooledBuffer,
        grads: &PooledBuffer,
        state: &[&PooledBuffer],
        finite: &PooledBuffer,
    ) -> wgpu::BindGroup {
        let bytes = 4 * self.elements as u64;
        let check = |what: &str, buffer: &PooledBuffer| {
            assert!(
                buffer.len() == bytes,
                "optimizer: {what} holds {} bytes, expected {bytes} ({} values)",
                buffer.len(),
                self.elements
            );
        };
        check("the values", values);
        check("the gradients", grads);
        assert!(
            state.len() == self.rule.state_buffers(),
            "optimizer: {:?} keeps {} state buffers, got {}",
            self.rule,
            self.rule.state_buffers(),
            state.len()
        );
        for s in state {
            check("a state buffer", s);
        }
        let mut buffers = vec![
            values,
            grads,
            &self.table,
            &self.infos,
            &self.shared,
            finite,
        ];
        buffers.extend(state);
        bind_buffers(steps, &self.pipeline, &buffers)
    }

    /// Records one update of the parameters marked in `active` (those with a gradient this
    /// step; the others keep their values, state and update counts) in `step`, with a group
    /// from [`bind_group`](Self::bind_group): uploads the step's constants and dispatches. The update is skipped on the GPU when the
    /// step's finite flag is set; once the step is finished and its flag read back,
    /// [`commit`](Self::commit) says whether it ran.
    ///
    /// # Panics
    ///
    /// If `active` does not hold one entry per parameter, or the last recorded step has not
    /// been committed.
    ///
    /// ```
    /// use cavitation_gpu::kernels::{OptimizerKernel, OptimizerRule};
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let rule = OptimizerRule::adam(1e-3, 0.0);
    ///     let mut adam = OptimizerKernel::new(&gpu, &mut PipelineCache::new(&gpu), &mut pool, rule, &[3, 5]).unwrap();
    ///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    ///     let [values, grads, m, v] = [0; 4].map(|_| pool.take(32, usage));
    ///     let flag = pool.take(4, usage);
    ///     let sizes = StepSizes { upload_bytes: adam.upload_bytes(), ..StepSizes::default() };
    ///     let mut steps = StepEncoder::new(&gpu, sizes);
    ///     let group = adam.bind_group(&mut steps, &values, &grads, &[&m, &v], &flag);
    ///     let mut step = steps.begin();
    ///     step.clear(flag.buffer(), 0, 4);
    ///     adam.record(&mut step, &group, &[true, false]);
    ///     let _ = step.finish();
    ///     adam.commit(true);
    ///     assert_eq!(adam.steps(), &[1, 0]);
    /// }
    /// ```
    pub fn record(&mut self, step: &mut Step<'_>, group: &wgpu::BindGroup, active: &[bool]) {
        assert!(
            active.len() == self.lens.len(),
            "optimizer: {} active flags for {} parameters",
            active.len(),
            self.lens.len()
        );
        assert!(
            !self.recorded,
            "optimizer: the last recorded step was not committed"
        );
        let [lr, _, _, beta1, beta2, _] = self.rule.settings();
        let mut end = 0;
        for (k, info) in self
            .info_bytes
            .as_chunks_mut::<INFO_BYTES>()
            .0
            .iter_mut()
            .enumerate()
        {
            end += self.lens[k] as u32;
            let t = self.steps[k] + 1;
            let flags = u32::from(active[k]) | (u32::from(self.steps[k] == 0) << 1);
            let step_size = lr / (1.0 - beta1.powi(t as i32));
            let inv_sqrt_correction2 = 1.0 / (1.0 - beta2.powi(t as i32)).sqrt();
            let words = [
                end,
                flags,
                (step_size as f32).to_bits(),
                (inv_sqrt_correction2 as f32).to_bits(),
            ];
            for (bytes, word) in info.as_chunks_mut::<4>().0.iter_mut().zip(words) {
                bytes.copy_from_slice(&word.to_le_bytes());
            }
        }
        step.upload(self.infos.buffer(), 0, &self.info_bytes);
        let mut shared = [0u8; SHARED_BYTES as usize];
        for (bytes, value) in shared
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(shared_constants(&self.rule))
        {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        step.upload(self.shared.buffer(), 0, &shared);
        step.dispatch_split(&self.pipeline, group, self.workgroups);
        self.pending.copy_from_slice(active);
        self.recorded = true;
    }

    /// Counts the recorded step's update for each parameter it was recorded for when `ran`
    /// (the step's finite flag was clear), and for none when the flag stopped it, so a
    /// skipped step leaves every count, like every value and moment, as it was.
    ///
    /// # Panics
    ///
    /// If no step was recorded since the last commit.
    ///
    /// ```
    /// use cavitation_gpu::kernels::{OptimizerKernel, OptimizerRule};
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache, StepEncoder, StepSizes, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let rule = OptimizerRule::adam(1e-3, 0.0);
    ///     let mut adam = OptimizerKernel::new(&gpu, &mut PipelineCache::new(&gpu), &mut pool, rule, &[4]).unwrap();
    ///     let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    ///     let [values, grads, m, v] = [0; 4].map(|_| pool.take(16, usage));
    ///     let flag = pool.take(4, usage);
    ///     let sizes = StepSizes { upload_bytes: adam.upload_bytes() + 4, ..StepSizes::default() };
    ///     let mut steps = StepEncoder::new(&gpu, sizes);
    ///     let group = adam.bind_group(&mut steps, &values, &grads, &[&m, &v], &flag);
    ///     let mut step = steps.begin();
    ///     step.upload(flag.buffer(), 0, &1u32.to_le_bytes()); // a non-finite gradient
    ///     adam.record(&mut step, &group, &[true]);
    ///     let _ = step.finish();
    ///     adam.commit(false);
    ///     assert_eq!(adam.steps(), &[0]);
    /// }
    /// ```
    pub fn commit(&mut self, ran: bool) {
        assert!(self.recorded, "optimizer: no step was recorded to commit");
        if ran {
            for (count, &active) in self.steps.iter_mut().zip(&self.pending) {
                *count += u32::from(active);
            }
        }
        self.recorded = false;
    }

    /// Hands the kernel's own buffers (the workgroup table and the constants) back to
    /// `pool`, the pool it was made with, for reuse.
    ///
    /// ```
    /// use cavitation_gpu::kernels::{OptimizerKernel, OptimizerRule};
    /// use cavitation_gpu::{BufferPool, Gpu, PipelineCache};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let mut pool = BufferPool::new(&gpu);
    ///     let rule = OptimizerRule::adam(1e-3, 0.0);
    ///     let kernel = OptimizerKernel::new(&gpu, &mut PipelineCache::new(&gpu), &mut pool, rule, &[4]).unwrap();
    ///     kernel.release(&mut pool);
    ///     assert_eq!(pool.outstanding(), 0);
    /// }
    /// ```
    pub fn release(self, pool: &mut BufferPool) {
        pool.give(self.table);
        pool.give(self.infos);
        pool.give(self.shared);
    }
}

/// The shared constants of `rule` as the kernels' `Shared` uniform lays them out, each
/// rounded to `f32` from its `f64`: the learning rate, momentum, weight decay, betas, one
/// minus each beta, eps, AdamW's shrink factor `1 − lr·wd`, and padding.
fn shared_constants(rule: &OptimizerRule) -> [f32; SHARED_WORDS] {
    let [lr, momentum, wd, beta1, beta2, eps] = rule.settings();
    let values = [
        lr,
        momentum,
        wd,
        beta1,
        beta2,
        1.0 - beta1,
        1.0 - beta2,
        eps,
        1.0 - lr * wd,
        0.0,
        0.0,
        0.0,
    ];
    values.map(|v| v as f32)
}
