//! SGD (with momentum and weight decay), Adam and AdamW as this crate's own CUDA kernels: one
//! launch updates every parameter of a model, held in flat buffers, with the arithmetic of the
//! CPU optimisers in `cavitation_nn::optim` done in `f32`.
//!
//! The host keeps each parameter's update count and commits it only once the caller knows the
//! step's finite flag, so a step the flag skipped counts for nothing.

use std::fmt;

use crate::buffer::{DeviceBuffer, Staging};
use crate::device::Cuda;
use crate::error::CudaError;
use crate::module::{Arg, Module};
use crate::sys::{CuFunction, DevicePtr};

/// The kernels' source, a port of `cavitation-gpu`'s `kernels/optimizer.rs`.
const SOURCE: &str = include_str!("optimizer.cu");

/// Threads per block, and parameter elements per block.
const BLOCK: u32 = 256;

/// The most elements a model's parameters may hold together: the kernels index them with
/// `u32`s, with room for a block's last threads past the end.
const MAX_ELEMENTS: usize = 1 << 31;

/// The most blocks a launch takes: the grid's first dimension.
const MAX_BLOCKS: usize = (1 << 31) - 1;

/// Words of the shared constants at the start of the constants buffer (`optimizer.cu`'s
/// `SHARED`).
const SHARED_WORDS: usize = 9;

/// Words of one parameter's constants after the shared ones.
const INFO_WORDS: usize = 4;

/// An update rule [`OptimizerKernel`] runs, with its settings: the card's forms of the CPU's
/// `Sgd`, `Adam` and `AdamW`. RMSprop and Adagrad have no form here.
///
/// Every setting is explicit. The CPU's `OptimizerConfig` leaves Adam's and AdamW's betas
/// and eps at their defaults, (0.9, 0.999) and 1e-8, so a rule built from it takes those
/// values, as [`adam`](Self::adam) and [`adam_w`](Self::adam_w) give them.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::OptimizerRule;
///
/// let adam = OptimizerRule::adam(1e-3, 0.0);
/// assert_eq!(
///     adam,
///     OptimizerRule::Adam { lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0 }
/// );
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
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::OptimizerRule;
    ///
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
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::OptimizerRule;
    ///
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
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::OptimizerRule;
    ///
    /// let sgd = OptimizerRule::Sgd { lr: 0.1, momentum: 0.9, weight_decay: 0.0 };
    /// assert_eq!(sgd.learning_rate(), 0.1);
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
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::OptimizerRule;
    ///
    /// let sgd = OptimizerRule::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 };
    /// assert_eq!(sgd.state_buffers(), 0);
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

    /// Refuses the rule unless every setting is finite once rounded to the `f32` the kernel
    /// receives, and the betas are in `[0, 1)`, as the CPU's Adam requires, both as given and
    /// as the kernel holds them: a beta from `1 − 2⁻²⁵` up rounds to 1 in `f32`, which would
    /// make its bias correction infinite.
    fn check(&self) -> Result<(), CudaError> {
        let [.., beta1, beta2, _] = self.settings();
        let in_range = |b: f64| (0.0..1.0).contains(&b) && (0.0..1.0).contains(&as_stored(b));
        if !(in_range(beta1) && in_range(beta2)) {
            return Err(CudaError::Invalid(format!(
                "optimizer: Adam betas must be in [0, 1), also once rounded to f32, got \
                 ({beta1}, {beta2})"
            )));
        }
        if !self.settings().iter().all(|&v| (v as f32).is_finite()) {
            return Err(CudaError::Invalid(format!(
                "optimizer: every setting must be finite as an f32, got {self:?}"
            )));
        }
        Ok(())
    }
}

/// `beta` as the kernels hold it: rounded to `f32`, back in `f64`.
///
/// One minus each beta and the bias corrections are taken from this value, not from the
/// `f64` setting, so a moment's two weights add up to 1 and its corrections match the decay
/// the kernel applies (`cavitation-gpu`'s `as_stored`, #326).
fn as_stored(beta: f64) -> f64 {
    f64::from(beta as f32)
}

/// The shared constants of `rule` as the kernels lay them out, each rounded to `f32` from its
/// `f64`: the learning rate, momentum, weight decay, betas, one minus each beta (from the
/// beta as stored, [`as_stored`]), eps and AdamW's shrink factor `1 − lr·wd`.
fn shared_constants(rule: &OptimizerRule) -> [f32; SHARED_WORDS] {
    let [lr, momentum, wd, beta1, beta2, eps] = rule.settings();
    let (beta1, beta2) = (as_stored(beta1), as_stored(beta2));
    [
        lr,
        momentum,
        wd,
        beta1,
        beta2,
        1.0 - beta1,
        1.0 - beta2,
        eps,
        1.0 - lr * wd,
    ]
    .map(|v| v as f32)
}

/// The buffers the first update ran on, which every later update must use: the values and
/// the state buffers, by their addresses on the card.
#[derive(Clone, Debug, PartialEq)]
struct Bound {
    /// The values.
    values: DevicePtr,
    /// The state buffers.
    state: Vec<DevicePtr>,
}

/// SGD, momentum, Adam or AdamW on the card for one model's parameters: a single launch
/// updates them all, with each parameter's own update count for Adam's bias corrections,
/// and does nothing when the step's finite flag is set.
///
/// The parameters live one after another in flat `f32` buffers: the values, the gradients
/// and the rule's [state buffers](OptimizerRule::state_buffers), each holding all the
/// parameters together, in the order of `lens`.
///
/// The kernels are `cavitation-gpu`'s WGSL ones ported operation for operation. Each step
/// takes the `f32`-rounded settings and per-step constants (computed on the host in `f64`,
/// one minus each beta and the bias corrections from the beta as stored in `f32`). NVRTC
/// rounds each `fmaf` once, divides and takes square roots correctly rounded, fuses nothing
/// else and keeps subnormals, so every value and state word is the bits of the kernel's own
/// `f32` expression evaluated on the host. Against the exact result of those constants and
/// the step's starting state, with `u = 2⁻²⁴`, the WGSL kernel's bounds hold:
///
/// - **SGD without weight decay or momentum:** the value `p' = p − lr·g` is one `fmaf`,
///   within `u·|p'|`, one rounding.
/// - **SGD with either:** for the step `s = g + wd·p`, the buffer `b = μ·b_old + s` (`b = s`
///   on a parameter's first update or without momentum) and
///   `e_b = u·(|b| + μ·|b_old| + |s| + |wd·p|)`, the momentum buffer within `e_b` and the
///   value within `u·(|p'| + lr·|b|) + lr·e_b`.
/// - **Adam and AdamW:** for the gradient `g` after Adam's decay, the new moments `m` and
///   `v`, the denominator `d = √v · c₂ + eps`, AdamW's shrunk value `p_s = p·(1 − lr·wd)` (0
///   for Adam) and Adam's decay term `w = |wd·p|` (0 for AdamW), the first moment within
///   `e_m = u·(|m| + β₁·|m_old| + 2(1 − β₁)|g| + (1 − β₁)·w)`, the second within
///   `e_v = u·(6v + 2(1 − β₂)|g|·w)`, and the value within
///   `u·(|p'| + |p_s|) + step_size·(|m/d|·(17u + e_v/(2v)) + e_m/d)`.
///
/// Every bound also carries `2⁻¹²⁶` absolute: a moment or momentum buffer below the smallest
/// normal `f32` is stored as 0, as the CPU stores it, and a subnormal result's rounding is
/// absolute rather than relative. A NaN or infinite gradient with the flag clear gives NaN or ±∞ where the
/// expression does.
///
/// A step runs in two parts: [`update`](Self::update) uploads the step's constants and
/// launches, and the launch copies the finite flag it read to a word of the kernel's own;
/// [`commit`](Self::commit) reads that word back and counts the update for each parameter it
/// ran on, or for none when the flag stopped it, so the counts follow what the card did. The
/// first update fixes the values and state buffers; a later update on others is refused.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{Cuda, OptimizerKernel, OptimizerRule};
///
/// if let Some(cuda) = Cuda::new() {
///     let rule = OptimizerRule::Sgd { lr: 0.5, momentum: 0.0, weight_decay: 0.0 };
///     let mut sgd = OptimizerKernel::new(&cuda, rule, &[2]).unwrap();
///     let (mut values, mut grads) = (cuda.buffer(8).unwrap(), cuda.buffer(8).unwrap());
///     let finite = cuda.buffer(4).unwrap(); // zeroed: every gradient finite
///     values.write(&[1.0f32, 2.0]).unwrap();
///     grads.write(&[1.0f32, -1.0]).unwrap();
///     sgd.update(&mut values, &grads, &mut [], &finite, &[true]).unwrap();
///     assert!(sgd.commit().unwrap()); // the step ran
///     let mut out = [0.0f32; 2];
///     values.read(&mut out).unwrap();
///     assert_eq!(out, [0.5, 2.5]);
///     assert_eq!(sgd.steps(), &[1]);
/// }
/// ```
pub struct OptimizerKernel<'c> {
    /// The rule and its settings, the learning rate the current one.
    rule: OptimizerRule,
    /// Each parameter's length.
    lens: Vec<usize>,
    /// The end of the last parameter in the flat buffers: the elements every buffer holds.
    elements: usize,
    /// Each parameter's first element in the flat buffers.
    starts: Vec<usize>,
    /// The compiled kernels.
    module: Module<'c>,
    /// The rule's kernel.
    function: CuFunction,
    /// For each block, its parameter and the first element it updates, written once.
    table: DeviceBuffer<'c>,
    /// The shared and per-parameter constants for the step being launched.
    constants: DeviceBuffer<'c>,
    /// The finite flag the last launch read, copied by the kernel for [`commit`](Self::commit).
    seen: DeviceBuffer<'c>,
    /// Pinned host memory the constants are copied from without the host waiting; rewritten
    /// only after [`commit`](Self::commit)'s read shows the last copy done.
    staging: Staging<'c>,
    /// Whether a copy from the staging memory may still be running: set by each upload,
    /// cleared by the synchronous read in [`commit`](Self::commit).
    in_flight: bool,
    /// The CUDA it runs on, to wait for a copy no commit waited for.
    cuda: &'c Cuda,
    /// The constants' words, rebuilt in place every step.
    words: Vec<u32>,
    /// Blocks per launch: one per [`BLOCK`] elements of each parameter.
    blocks: u32,
    /// Each parameter's committed update count.
    steps: Vec<u32>,
    /// Which parameters the launched step updates, until it is committed.
    pending: Vec<bool>,
    /// Whether a launched step awaits [`commit`](Self::commit).
    recorded: bool,
    /// The buffers of the first update, once there was one.
    bound: Option<Bound>,
}

impl fmt::Debug for OptimizerKernel<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OptimizerKernel")
            .field("rule", &self.rule)
            .field("lens", &self.lens)
            .field("steps", &self.steps)
            .finish_non_exhaustive()
    }
}

/// Panics unless every buffer holds at least `values` four-byte values, naming the first
/// that does not.
fn check_buffers(buffers: &[(&DeviceBuffer<'_>, &str, usize)]) {
    for &(buffer, name, values) in buffers {
        let needs = values.saturating_mul(4);
        assert!(
            buffer.bytes() >= needs,
            "optimizer: the {name} buffer holds {} bytes, and {values} values need {needs}",
            buffer.bytes()
        );
    }
}

impl<'c> OptimizerKernel<'c> {
    /// Compiles the kernel for `rule` over parameters of `lens` elements each on `cuda`'s
    /// card, and writes the block table. Every parameter starts with no updates.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if `lens` is empty or holds a 0, the parameters hold more than
    /// 2³¹ elements together, a setting is not finite as an `f32`, or Adam's betas are
    /// outside `[0, 1)` as given or once rounded to `f32`; [`CudaError::Compile`] or
    /// [`CudaError::Call`] if NVRTC or the driver refuses the kernel or the memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::{Cuda, OptimizerKernel, OptimizerRule};
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     let adam = OptimizerKernel::new(&cuda, OptimizerRule::adam(1e-3, 0.0), &[784 * 128, 128]);
    ///     assert_eq!(adam.unwrap().steps(), &[0, 0]);
    /// }
    /// ```
    pub fn new(
        cuda: &'c Cuda,
        rule: OptimizerRule,
        lens: &[usize],
    ) -> Result<OptimizerKernel<'c>, CudaError> {
        let mut starts = Vec::with_capacity(lens.len());
        let mut at = 0usize;
        for &n in lens {
            starts.push(at);
            at = at.saturating_add(n);
        }
        Self::with_starts(cuda, rule, lens, &starts)
    }

    /// [`new`](Self::new) for parameters of `lens` elements at `starts` in the flat buffers
    /// (in order, not overlapping); the elements between them are never read or written.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new), and [`CudaError::Invalid`] if `starts` does not hold one start
    /// per parameter or a parameter starts inside the one before it.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::{Cuda, OptimizerKernel, OptimizerRule};
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     let sgd = OptimizerRule::Sgd { lr: 0.5, momentum: 0.0, weight_decay: 0.0 };
    ///     // Two one-element parameters at 0 and 2; element 1 is left alone.
    ///     let mut opt = OptimizerKernel::with_starts(&cuda, sgd, &[1, 1], &[0, 2]).unwrap();
    ///     let mut values = cuda.buffer(12).unwrap();
    ///     let mut grads = cuda.buffer(12).unwrap();
    ///     let finite = cuda.buffer(4).unwrap();
    ///     values.write(&[1.0f32, 7.0, 2.0]).unwrap();
    ///     grads.write(&[2.0f32, 9.0, 2.0]).unwrap();
    ///     opt.update(&mut values, &grads, &mut [], &finite, &[true, true]).unwrap();
    ///     assert!(opt.commit().unwrap());
    ///     let mut out = [0.0f32; 3];
    ///     values.read(&mut out).unwrap();
    ///     assert_eq!(out, [0.0, 7.0, 1.0]);
    /// }
    /// ```
    pub fn with_starts(
        cuda: &'c Cuda,
        rule: OptimizerRule,
        lens: &[usize],
        starts: &[usize],
    ) -> Result<OptimizerKernel<'c>, CudaError> {
        rule.check()?;
        if lens.is_empty() || lens.contains(&0) {
            return Err(CudaError::Invalid(format!(
                "optimizer: every parameter must hold at least one element, got lengths {lens:?}"
            )));
        }
        if starts.len() != lens.len() {
            return Err(CudaError::Invalid(format!(
                "optimizer: {} starts for {} parameters",
                starts.len(),
                lens.len()
            )));
        }
        let mut elements = 0usize;
        for (k, (&at, &n)) in starts.iter().zip(lens).enumerate() {
            if at < elements {
                return Err(CudaError::Invalid(format!(
                    "optimizer: parameter {k} starts at {at}, inside the parameter before it, \
                     which ends at {elements}"
                )));
            }
            elements = match at.checked_add(n) {
                Some(end) if end <= MAX_ELEMENTS => end,
                _ => {
                    return Err(CudaError::Invalid(format!(
                        "optimizer: the parameters hold more than the {MAX_ELEMENTS} elements \
                         the kernel takes"
                    )));
                }
            };
        }
        let block = BLOCK as usize;
        let blocks: usize = lens.iter().map(|n| n.div_ceil(block)).sum();
        if blocks > MAX_BLOCKS {
            return Err(CudaError::Invalid(format!(
                "optimizer: {blocks} blocks are more than the {MAX_BLOCKS} a launch takes"
            )));
        }
        let mut table = Vec::with_capacity(2 * blocks);
        for (k, (&start, &n)) in starts.iter().zip(lens).enumerate() {
            for first in (start..start + n).step_by(block) {
                table.extend([k as u32, first as u32]);
            }
        }
        let module = cuda.compile("optimizer.cu", SOURCE)?;
        let function = module.function(match rule {
            OptimizerRule::Sgd { momentum: 0.0, .. } => "sgd",
            OptimizerRule::Sgd { .. } => "sgd_momentum",
            OptimizerRule::Adam { .. } | OptimizerRule::AdamW { .. } => "adam",
        })?;
        let mut table_buffer = cuda.buffer(4 * table.len())?;
        table_buffer.write(&table)?;
        let words = SHARED_WORDS + INFO_WORDS * lens.len();
        Ok(OptimizerKernel {
            rule,
            lens: lens.to_vec(),
            elements,
            starts: starts.to_vec(),
            module,
            function,
            table: table_buffer,
            constants: cuda.buffer(4 * words)?,
            staging: Staging::new(cuda, words)?,
            in_flight: false,
            cuda,
            seen: cuda.buffer(4)?,
            words: vec![0; words],
            blocks: blocks as u32,
            steps: vec![0; lens.len()],
            pending: vec![false; lens.len()],
            recorded: false,
            bound: None,
        })
    }

    /// The rule, with the current learning rate.
    pub fn rule(&self) -> OptimizerRule {
        self.rule
    }

    /// Each parameter's length, in order.
    pub fn lens(&self) -> &[usize] {
        &self.lens
    }

    /// Sets the learning rate the next update uses, as a schedule does between steps.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if `lr` is not finite as an `f32`; the rule is then unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::{Cuda, OptimizerKernel, OptimizerRule};
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     let mut adam = OptimizerKernel::new(&cuda, OptimizerRule::adam(1e-3, 0.0), &[4]).unwrap();
    ///     adam.set_learning_rate(5e-4).unwrap();
    ///     assert_eq!(adam.rule().learning_rate(), 5e-4);
    ///     assert!(adam.set_learning_rate(f64::INFINITY).is_err());
    /// }
    /// ```
    pub fn set_learning_rate(&mut self, lr: f64) -> Result<(), CudaError> {
        if !(lr as f32).is_finite() {
            return Err(CudaError::Invalid(format!(
                "optimizer: the learning rate must be finite as an f32, got {lr}"
            )));
        }
        match &mut self.rule {
            OptimizerRule::Sgd { lr: rate, .. }
            | OptimizerRule::Adam { lr: rate, .. }
            | OptimizerRule::AdamW { lr: rate, .. } => *rate = lr,
        }
        Ok(())
    }

    /// Each parameter's committed update count, in the order of the lengths it was made with.
    pub fn steps(&self) -> &[u32] {
        &self.steps
    }

    /// Updates the parameters marked in `active` (those with a gradient this step; the others
    /// keep their values, state and update counts): uploads the step's constants and launches.
    /// `values` and `grads` hold the parameters' `f32`s, `state` the rule's state buffers
    /// (SGD's momentum buffer, or Adam's first and second moments; none for SGD without
    /// momentum), each as long, and `finite` the step's finite flag, a `u32` at its start
    /// that is 0 when the step's gradients are all finite and any other value to skip the
    /// update. [`commit`](Self::commit) then counts the step from the flag the launch read.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if `active` does not hold one entry per parameter, `state`
    /// holds the wrong number of buffers, the last update was not committed, `values` or a
    /// state buffer is not the one the first update ran on, a parameter's update count would
    /// pass `u32::MAX`, or a constant derived from the settings is not finite as an `f32`
    /// (such as Adam's `lr / (1 − β₁ᵗ)` or AdamW's `1 − lr·wd`); [`CudaError::Call`] if the
    /// upload or the launch is refused.
    ///
    /// # Panics
    ///
    /// If a buffer is shorter than its values, naming it.
    pub fn update(
        &mut self,
        values: &mut DeviceBuffer<'_>,
        grads: &DeviceBuffer<'_>,
        state: &mut [&mut DeviceBuffer<'_>],
        finite: &DeviceBuffer<'_>,
        active: &[bool],
    ) -> Result<(), CudaError> {
        let n = self.elements;
        check_buffers(&[(values, "values", n), (grads, "gradients", n)]);
        for s in state.iter() {
            check_buffers(&[(s, "state", n)]);
        }
        check_buffers(&[(finite, "finite flag", 1)]);
        if active.len() != self.lens.len() {
            return Err(CudaError::Invalid(format!(
                "optimizer: {} active flags for {} parameters",
                active.len(),
                self.lens.len()
            )));
        }
        if state.len() != self.rule.state_buffers() {
            return Err(CudaError::Invalid(format!(
                "optimizer: {:?} keeps {} state buffers, got {}",
                self.rule,
                self.rule.state_buffers(),
                state.len()
            )));
        }
        if self.recorded {
            return Err(CudaError::Invalid(
                "optimizer: the last update was not committed".into(),
            ));
        }
        let bound = Bound {
            values: values.pointer,
            state: state.iter().map(|s| s.pointer).collect(),
        };
        if self.bound.as_ref().is_some_and(|b| *b != bound) {
            return Err(CudaError::Invalid(
                "optimizer: an update must use the values and state buffers the first one used"
                    .into(),
            ));
        }
        self.fill_constants(active)?;
        // An upload whose launch then failed was never committed: wait for its copy.
        if self.in_flight {
            self.cuda.synchronize()?;
            self.in_flight = false;
        }
        // SAFETY: no copy from the staging memory is running: `commit` reads `seen`
        // synchronously after the work queued before it, which includes the last copy, and
        // clears `in_flight`; an upload no commit followed was waited for just above.
        unsafe { self.staging.upload(&self.words, &mut self.constants)? };
        self.in_flight = true;
        let mut args = vec![
            Arg::Buffer(values),
            Arg::Buffer(grads),
            Arg::Buffer(&self.table),
            Arg::Buffer(&self.constants),
            Arg::Buffer(finite),
            Arg::U32(self.blocks),
            Arg::Buffer(&self.seen),
        ];
        args.extend(state.iter().map(|s| Arg::Buffer(s)));
        if state.len() == 2 {
            args.push(Arg::U32(u32::from(matches!(
                self.rule,
                OptimizerRule::AdamW { .. }
            ))));
        }
        // SAFETY: the handle was looked up in this module when it was compiled.
        let kernel = unsafe { self.module.kernel_from(self.function) };
        // SAFETY: every kernel takes (float *values, const float *grads, const u32 *table,
        // const float *constants, const u32 *finite, u32 groups, u32 *seen), then
        // sgd_momentum its buffer and adam (float *m, float *v, u32 decoupled), which `args` holds for the
        // rule's kernel (the state's count checked above). Block w < groups reads table
        // words 2w and 2w + 1, which the table holds, and the constants of the parameter
        // they name, below the constants' length; it touches elements below the parameter's
        // end, at most the elements every buffer holds (checked above), reads one flag, and
        // its first thread writes the one word of `seen`, the kernel's own.
        // values and the state buffers, borrowed mutably, are distinct from each other and
        // from the inputs.
        unsafe { kernel.launch([self.blocks, 1, 1], [BLOCK, 1, 1], &args)? };
        self.pending.copy_from_slice(active);
        self.recorded = true;
        self.bound = Some(bound);
        Ok(())
    }

    /// Writes the step's constants into the constants' words: the shared ones, then each
    /// parameter's end, flags and Adam bias corrections for its next update; refuses
    /// constants that are not finite as `f32`s and counts past `u32::MAX`.
    fn fill_constants(&mut self, active: &[bool]) -> Result<(), CudaError> {
        let [lr, _, _, beta1, beta2, _] = self.rule.settings();
        let (beta1, beta2) = (as_stored(beta1), as_stored(beta2));
        let shared = shared_constants(&self.rule);
        if !shared.iter().all(|v| v.is_finite()) {
            return Err(CudaError::Invalid(format!(
                "optimizer: a constant of {:?} is not finite as an f32: {shared:?}",
                self.rule
            )));
        }
        for (word, value) in self.words.iter_mut().zip(shared) {
            *word = value.to_bits();
        }
        let (infos, _) = self.words[SHARED_WORDS..].as_chunks_mut::<INFO_WORDS>();
        for (k, info) in infos.iter_mut().enumerate() {
            // Below 2³¹: checked when the kernel was made.
            let end = (self.starts[k] + self.lens[k]) as u32;
            let t = self.steps[k].checked_add(1).ok_or_else(|| {
                CudaError::Invalid(format!(
                    "optimizer: parameter {k} has {} updates, the most a count holds",
                    u32::MAX
                ))
            })?;
            let flags = u32::from(active[k]) | (u32::from(self.steps[k] == 0) << 1);
            // powi takes an i32; past i32::MAX the powers are 0 to the f64's precision.
            let t = i32::try_from(t).unwrap_or(i32::MAX);
            let step_size = (lr / (1.0 - beta1.powi(t))) as f32;
            let inv_sqrt_correction2 = (1.0 / (1.0 - beta2.powi(t)).sqrt()) as f32;
            if !(step_size.is_finite() && inv_sqrt_correction2.is_finite()) {
                return Err(CudaError::Invalid(format!(
                    "optimizer: parameter {k}'s bias-corrected step size {step_size} or second \
                     correction {inv_sqrt_correction2} is not finite as an f32"
                )));
            }
            info.copy_from_slice(&[
                end,
                flags,
                step_size.to_bits(),
                inv_sqrt_correction2.to_bits(),
            ]);
        }
        Ok(())
    }

    /// Reads back the finite flag the last launch read, waiting for the launch, and counts
    /// the step's update for each parameter it was launched for when the flag was 0, and for
    /// none when the flag stopped it, so a skipped step leaves every count, like every value
    /// and moment, as it was. Returns whether the step ran.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if no update was launched since the last commit;
    /// [`CudaError::Call`] if the flag cannot be read, which is also how a launch that
    /// failed after it returned is reported (the step then stays to be committed).
    pub fn commit(&mut self) -> Result<bool, CudaError> {
        if !self.recorded {
            return Err(CudaError::Invalid(
                "optimizer: no update was launched to commit".into(),
            ));
        }
        let mut flag = [0u32];
        self.seen.read(&mut flag)?;
        // The read waited for all the card's earlier work, the constants' copy included.
        self.in_flight = false;
        let ran = flag[0] == 0;
        if ran {
            for (count, &active) in self.steps.iter_mut().zip(&self.pending) {
                *count += u32::from(active);
            }
        }
        self.recorded = false;
        Ok(ran)
    }
}

#[cfg(test)]
mod tests {
    use super::{OptimizerRule, shared_constants};
    use crate::error::CudaError;

    /// A beta below 1 that rounds to 1 in `f32` is refused, as 1 itself is.
    #[test]
    fn a_beta_that_rounds_to_one_is_refused() {
        for beta2 in [0.999_999_99, 1.0] {
            let rule = OptimizerRule::Adam {
                lr: 1e-3,
                beta1: 0.9,
                beta2,
                eps: 1e-8,
                weight_decay: 0.0,
            };
            let Err(CudaError::Invalid(text)) = rule.check() else {
                panic!("β₂ = {beta2} accepted");
            };
            assert!(text.contains("also once rounded to f32"), "{text}");
        }
    }

    /// For each beta, the weight on the new term over one minus the weight on the old, both
    /// as the kernels hold them, is 1, so a moment of a constant input settles on that input.
    #[test]
    fn a_moment_of_a_constant_input_settles_on_it() {
        for (beta1, beta2) in [(0.9, 0.999), (0.8, 0.99), (0.95, 0.9999), (0.5, 0.999_99)] {
            let rule = OptimizerRule::Adam {
                lr: 1e-3,
                beta1,
                beta2,
                eps: 1e-8,
                weight_decay: 0.0,
            };
            let c = shared_constants(&rule).map(f64::from);
            for (beta, one_minus) in [(c[3], c[5]), (c[4], c[6])] {
                let settles = one_minus / (1.0 - beta);
                assert!(
                    (settles - 1.0).abs() <= (2.0f64).powi(-20),
                    "betas ({beta1}, {beta2}): a moment settles at {settles} of its input"
                );
            }
        }
    }
}
