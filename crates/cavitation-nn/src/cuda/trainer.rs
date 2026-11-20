//! `CudaTrainer`: training a [`CudaModel`] on an NVIDIA card one batch at a time. A step
//! uploads the batch and its dropout masks, queues the training forward pass, the loss, the
//! backward pass, the finite flag and the health values, the optimiser's update and batch
//! norm's running statistics on the card, and reads the loss, the flag and the health values
//! back in one copy.

use std::time::{Duration, Instant};

use cavitation_core::{Error, ParamId, ParamStore, dropout_mask};
use cavitation_cuda::{
    ChannelDims, Cuda, CudaError, DeadRelu, FiniteFlag, GluOp, MeanSquaredError,
    NegativeLogLikelihood, OptimizerKernel, OptimizerRule, PreluOp, Regression, RegressionLoss,
    RowSoftmaxOp, RunningUpdate, SoftmaxCrossEntropy, SumSquares,
};

use super::model::{CudaModel, ran, refused};
use super::pass::{
    DropoutMask, Forward, Op, Parts, Store, backward_op, glu_gate, prelu_shape, run,
};
use crate::health::channel_size;
use crate::plan::{Act, Kind, Running};
use crate::{
    BatchView, GradStats, Health, HealthMonitor, Loss, OnNonFinite, OptimizerConfig, Sequential,
    StepInputs, TargetsView,
};

/// What a failed call to the card names.
const STEP: &str = "CudaTrainer::step";

/// The alignment, in `f32`s, of each parameter in the flat buffers: 256 bytes, the alignment
/// of an allocation of its own.
const ALIGN: usize = 64;

/// The loss of a training pass, set up for its rows.
enum LossKernel<'c> {
    /// Softmax cross-entropy on class labels.
    CrossEntropy(SoftmaxCrossEntropy<'c>),
    /// Mean squared error on target values.
    Mse(MeanSquaredError<'c>),
    /// The negative log-likelihood on class labels.
    Nll(NegativeLogLikelihood<'c>),
    /// L1, Huber or smooth L1 on target values.
    Regression(RegressionLoss<'c>),
}

/// The regression kernel's form of `loss`, with its threshold as the `f32` the tape takes,
/// or `None` for a loss that is not L1, Huber or smooth L1.
fn regression(loss: Loss) -> Option<Regression> {
    match loss {
        Loss::L1 => Some(Regression::L1),
        Loss::Huber { delta } => Some(Regression::Huber {
            delta: delta as f32,
        }),
        Loss::SmoothL1 { beta } => Some(Regression::SmoothL1 { beta: beta as f32 }),
        _ => None,
    }
}

/// One ReLU layer's dead-neuron bits in a pass: one per channel of `spatial` values, or one
/// per neuron, as the CPU health checks group them.
struct ReluBits<'c> {
    /// Its first word in the bits.
    word: usize,
    /// The kernel, set up for the pass's rows.
    kernel: DeadRelu<'c>,
    /// The ReLU's output, an index into the store.
    y: usize,
}

/// A training step for one number of rows: the forward pass in training mode, the loss, the
/// backward pass and the dead-ReLU bits, with the buffers of its own.
struct TrainPass<'c> {
    /// The forward pass, whose kernels the backward pass shares.
    forward: Forward<'c>,
    /// The loss.
    loss: LossKernel<'c>,
    /// The targets: class labels as `u32`, or target values.
    targets: usize,
    /// The gradient of the output.
    out_grad: usize,
    /// The backward calls, last layer first.
    backward: Vec<Op>,
    /// Where each layer's backward calls end in `backward`, last layer first.
    backward_ends: Vec<usize>,
    /// Where a `Linear`'s or a convolution's input gradient begins in `backward`, last layer
    /// first; `None` for a layer whose backward calls are one part.
    backward_inputs: Vec<Option<usize>>,
    /// The gradients the pooling backward passes add to, which every step clears first.
    cleared: Vec<usize>,
    /// Each ReLU layer's bits.
    relus: Vec<ReluBits<'c>>,
    /// Each batch norm's kernel and statistics, by layer.
    norms: Vec<(usize, usize, usize)>,
    /// The next masks' bits, one list per dropout layer, for a pass of fewer rows than the
    /// trainer's batch; a full pass's are the trainer's.
    next: Vec<Vec<u32>>,
}

/// The optimiser on the card: the parameters and gradients laid end to end, each on a
/// 256-byte boundary, which the layers read and write in place; its state buffers; and where
/// each parameter sits.
struct Flat<'c> {
    /// The update.
    kernel: OptimizerKernel<'c>,
    /// Every parameter's values, end to end; each parameter's own buffer is its part.
    values: usize,
    /// Every parameter's gradient, end to end; each gradient's buffer is its part.
    grads: usize,
    /// The state buffers: SGD's momentum, or Adam's first and second moments.
    state: Vec<usize>,
    /// Each parameter's first element in the flat buffers.
    offsets: Vec<usize>,
    /// Whether each parameter updates: every one has a gradient each step.
    active: Vec<bool>,
}

/// The health kernels and the buffers a step reads back.
struct Checks<'c> {
    /// The finite flag over the loss and every gradient.
    finite: FiniteFlag<'c>,
    /// Each gradient's sum of squares.
    squares: SumSquares<'c>,
    /// The loss, one `f32`.
    loss: usize,
    /// The finite flag, one `u32`.
    flag: usize,
    /// The sums of squares, one per parameter.
    sums: usize,
    /// The bits of every ReLU layer, overwritten each step.
    bits: usize,
    /// The words of `bits`.
    words: usize,
    /// The loss, the flag, the sums, the counts and the bits, gathered for one readback.
    gathered: usize,
    /// The gathered words, read back.
    host: Vec<u32>,
}

/// The optimiser's state, as [`CudaTrainer::optimizer_state`] reads it back.
///
/// # Examples
///
/// See [`CudaTrainer::optimizer_state`].
#[derive(Clone, Debug, PartialEq)]
pub struct CudaOptimizerState {
    /// The parameters, in the model's order.
    pub ids: Vec<ParamId>,
    /// Each parameter's update count, as the CPU optimisers count it.
    pub steps: Vec<u32>,
    /// The state buffers, each with one entry per parameter of its values: for SGD with
    /// momentum the momentum buffer; for Adam and AdamW the first and second moments, in
    /// that order; none for plain SGD.
    pub buffers: Vec<Vec<Vec<f32>>>,
}

/// One training step's time by part, from [`CudaTrainer::last_profile`]: what
/// [`Trainer::last_profile`](crate::Trainer::last_profile) gives on the CPU. A profiled step
/// waits for the card after each part and reads the host's clock, so its parts describe a
/// step that runs them one after another, with the card idle while the host queues the next.
///
/// The struct is `#[non_exhaustive]`, so a later field is not a breaking change: outside
/// this crate a profile comes from [`CudaTrainer::last_profile`] and is read through its
/// public fields.
///
/// # Examples
///
/// See [`CudaTrainer::profile_next_step`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct CudaStepProfile {
    /// Each layer's forward pass, in layer order.
    pub forward: Vec<Duration>,
    /// The loss and its gradient.
    pub loss: Duration,
    /// Each layer's backward pass, in layer order.
    pub backward: Vec<Duration>,
    /// The part of each [`backward`](Self::backward) time that is the layer's input gradient,
    /// for a `Linear` or a `Conv2d` after the first layer, in layer order; `None` for every
    /// other layer.
    pub backward_input: Vec<Option<Duration>>,
    /// The health values: the finite flag, the sums of squares and the ReLU bits.
    pub health: Duration,
    /// The optimiser's update.
    pub optimizer: Duration,
    /// Batch norm's running statistics.
    pub running: Duration,
    /// The readback of the loss, the flag and the health values.
    pub readback: Duration,
    /// The whole step on the host's clock, from the call to its return: the batch's checks
    /// and uploads, a first short batch's set-up, the drawing of the next step's dropout
    /// masks and the health checks included.
    pub total: Duration,
}

/// Trains a [`Sequential`] on an NVIDIA card through CUDA, one batch at a time:
/// [`step`](CudaTrainer::step) runs the forward pass in training mode, the loss, the backward
/// pass and the optimiser's update on the card and reads back the loss and the health values,
/// and [`sync_to`](CudaTrainer::sync_to) copies the parameters and batch-norm buffers back into
/// a [`ParamStore`].
///
/// A step does what [`Trainer::step`](crate::Trainer::step) does on the CPU, as
/// [`GpuTrainer`](crate::GpuTrainer) does on wgpu: the same dropout masks (from
/// [`dropout_mask`] with the trainer's seed and the step's number), the same order of work,
/// and an epoch's shorter last batch through a step set up for its rows. The model's layers
/// must be ones [`CudaModel::compile`] takes; the losses are [`Loss::CrossEntropy`],
/// [`Loss::Nll`], [`Loss::Mse`], [`Loss::L1`], [`Loss::Huber`] and [`Loss::SmoothL1`], and the
/// optimisers SGD (with or without momentum), Adam and AdamW. A parameter used by two layers
/// is refused.
///
/// The convolution's passes run through cuDNN and `Linear`'s through cuBLAS, which publish no
/// error bounds; their distance from the CPU is measured in `cavitation-cuda`'s tests and in
/// this crate's, which hold each step's gradients within a tolerance of the `f64` tape's.
/// cuDNN runs deterministic algorithms only, chosen by its heuristics, so two runs give the
/// same bits. With [`Cuda::set_cudnn_benchmark`] on (or `CAVITATION_CUDNN_BENCHMARK=1`), the
/// algorithms are timed once per shape and the fastest kept: runs within one process still
/// give the same bits, but another process or card may choose others and differ in the last
/// bits.
///
/// Every step runs the CPU trainer's [`Health`] checks (the default unless
/// [`set_health`](CudaTrainer::set_health) changes it) on values the step reads back with its
/// loss, and sends the CPU's warnings and returns the CPU's errors, at the same steps and
/// naming the same parameters. As on wgpu, a step whose loss or gradient is NaN or infinite is
/// never applied, so under [`OnNonFinite::Stop`] the parameters, the optimiser's state and the
/// running statistics stay as they were, and [`OnNonFinite::Warn`] is refused.
///
/// The trainer borrows the [`Cuda`] it was made on, which threads may share. The trainer itself
/// is neither `Send` nor `Sync`: it holds its kernels' handles and its buffers, so it is used
/// on the thread that made it.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_cuda::Cuda;
/// use cavitation_nn::{
///     Batch, Batcher, CudaTrainer, Dataset, Linear, Loss, OptimizerConfig, Relu, Sequential,
///     Targets,
/// };
///
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 8, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 8, 2, &mut rng));
/// let data = Dataset {
///     x: Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], &[4, 2]).unwrap(),
///     targets: Targets::Classes(vec![1, 1, 0, 0]),
/// };
/// if let Some(cuda) = Cuda::new() {
///     let sgd = OptimizerConfig::Sgd { lr: 0.5, momentum: 0.0, weight_decay: 0.0 };
///     let mut trainer = CudaTrainer::new(&net, &params, &sgd, Loss::CrossEntropy, 4, &cuda).unwrap();
///     let mut batch = Batch::new(&data, 4);
///     let view = Batcher::new(4, 4, None).gather(&data, 0..4, &mut batch);
///     let first = trainer.step(&net, view).unwrap();
///     let mut last = first;
///     for _ in 0..50 {
///         let view = Batcher::new(4, 4, None).gather(&data, 0..4, &mut batch);
///         last = trainer.step(&net, view).unwrap();
///     }
///     assert!(last < first);
///     trainer.sync_to(&mut params);
/// }
/// ```
pub struct CudaTrainer<'c, 's> {
    /// The model: its parameters and buffers, and its evaluation pass.
    model: CudaModel<'c>,
    /// The step for a full batch.
    pass: TrainPass<'c>,
    /// The steps for batches of fewer rows than [`batch_size`](CudaTrainer::batch_size) (an
    /// epoch's last), with their rows, each built the first time a batch of its rows comes.
    short: Vec<(usize, TrainPass<'c>)>,
    /// Each parameter's gradient, in the model's parameter order, shared by every pass.
    param_grads: Vec<usize>,
    /// The optimiser.
    flat: Flat<'c>,
    /// The health kernels and the readback.
    checks: Checks<'c>,
    /// Each batch norm that keeps running statistics: its layer, its statistics and its
    /// update's rates and batch count, shared by every pass.
    running: Vec<(usize, Running, RunningUpdate)>,
    /// Each ReLU layer's place in the bits: the layer, its first word and its channels.
    relu_words: Vec<(usize, usize, usize)>,
    /// The ReLU layers and their values per channel, for building a pass.
    relus: Vec<(usize, usize)>,
    /// The loss.
    loss: Loss,
    /// The steps taken.
    taken: u64,
    /// The seed of the dropout masks.
    dropout_seed: u64,
    /// The bits of the next full step's masks, one list per dropout layer, drawn ahead.
    next: Vec<Vec<u32>>,
    /// The step number `next` was drawn for.
    next_for: u64,
    /// The labels of a batch as `u32`s, reused from step to step.
    labels: Vec<u32>,
    /// The health checks run on every step.
    health: HealthMonitor<'s>,
    /// The model's parameter names, for the health checks' messages.
    names: Vec<(String, ParamId)>,
    /// The model's parameters (an index into its parameter order) in the order of the
    /// store, the order the CPU optimisers fill their gradient statistics in.
    store_order: Vec<(ParamId, usize)>,
    /// The gradient statistics of the last step.
    stats: GradStats,
    /// The last step's health values.
    read: HealthRead,
    /// The steps left out under [`OnNonFinite::Skip`].
    skipped: u64,
    /// Whether the next step is profiled.
    profile_asked: bool,
    /// The last profiled step's profile, when it is the last step.
    profile: Option<CudaStepProfile>,
}

/// One step's health values, read back with the loss and the flag.
#[derive(Debug, Default)]
struct HealthRead {
    /// Each parameter gradient's sum of squares, in the model's order.
    squares: Vec<f32>,
    /// Each parameter gradient's count of NaN and infinite values, in the model's order.
    counts: Vec<f32>,
    /// The ReLU layers' bits.
    bits: Vec<u32>,
}

impl std::fmt::Debug for CudaTrainer<'_, '_> {
    /// Writes the model, the steps taken and skipped and the health settings.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTrainer")
            .field("model", &self.model)
            .field("loss", &self.loss)
            .field("steps", &self.taken)
            .field("skipped", &self.skipped)
            .field("health", self.health.health())
            .finish_non_exhaustive()
    }
}

/// The clock of a profiled step: each lap waits for the card and reads the host's clock.
struct Clock<'c> {
    /// The card, when the step is profiled; `None` reads no clock.
    cuda: Option<&'c Cuda>,
    /// The end of the last lap.
    last: Instant,
}

impl Clock<'_> {
    /// The time since the last lap, after the card's work so far; zero when not profiling.
    fn lap(&mut self) -> Duration {
        let Some(cuda) = self.cuda else {
            return Duration::ZERO;
        };
        ran(STEP, cuda.synchronize());
        let now = Instant::now();
        let lap = now - self.last;
        self.last = now;
        lap
    }
}

impl<'c, 's> CudaTrainer<'c, 's> {
    /// A trainer for `model`, with its values in `params`, that minimises `loss` with
    /// `optimizer` on batches of `batch_size` rows on `cuda`'s card.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for `RmsProp` and `Adagrad`, for binary cross-entropy (with
    /// or without logits) and KL divergence, for a parameter used by two layers, for an
    /// optimiser setting that is not finite as an `f32`, and as [`CudaModel::compile`]
    /// returns; [`Error::InvalidData`] as `compile` returns, for a Huber `delta` not above 0
    /// or a smooth L1 `beta` below 0 (NaN either way, as `f32`s; the CPU's tape panics on
    /// them), if the model has no parameters or one that holds no values, and if a batch
    /// norm would normalise a training batch over one value per channel.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_cuda::Cuda;
    /// use cavitation_nn::{CudaTrainer, Linear, Loss, OptimizerConfig, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
    /// if let Some(cuda) = Cuda::new() {
    ///     let adam = OptimizerConfig::Adam { lr: 1e-3, weight_decay: 0.0 };
    ///     assert!(CudaTrainer::new(&net, &params, &adam, Loss::Mse, 16, &cuda).is_ok());
    ///     let rms = OptimizerConfig::RmsProp { lr: 1e-3, alpha: 0.99, momentum: 0.0, weight_decay: 0.0, centered: false };
    ///     assert!(CudaTrainer::new(&net, &params, &rms, Loss::Mse, 16, &cuda).is_err());
    ///     let huber = Loss::Huber { delta: 0.5 };
    ///     assert!(CudaTrainer::new(&net, &params, &adam, huber, 16, &cuda).is_ok());
    ///     let flat = Loss::Huber { delta: 0.0 };
    ///     assert!(CudaTrainer::new(&net, &params, &adam, flat, 16, &cuda).is_err());
    /// }
    /// ```
    pub fn new(
        model: &Sequential,
        params: &ParamStore,
        optimizer: &OptimizerConfig,
        loss: Loss,
        batch_size: usize,
        cuda: &'c Cuda,
    ) -> Result<CudaTrainer<'c, 's>, Error> {
        const WHO: &str = "CudaTrainer";
        let rule = match *optimizer {
            OptimizerConfig::Sgd {
                lr,
                momentum,
                weight_decay,
            } => OptimizerRule::Sgd {
                lr,
                momentum,
                weight_decay,
            },
            OptimizerConfig::Adam { lr, weight_decay } => OptimizerRule::adam(lr, weight_decay),
            OptimizerConfig::AdamW { lr, weight_decay } => OptimizerRule::adam_w(lr, weight_decay),
            other => {
                return Err(Error::Unsupported(format!(
                    "{WHO}: CUDA has no {other:?}; it runs SGD, Adam and AdamW"
                )));
            }
        };
        if !matches!(loss, Loss::CrossEntropy | Loss::Nll | Loss::Mse) && regression(loss).is_none()
        {
            return Err(Error::Unsupported(format!(
                "{WHO}: CUDA has no Loss::{loss:?}; it runs CrossEntropy, Nll, Mse, L1, Huber and \
                 SmoothL1"
            )));
        }
        loss.check_threshold(WHO)?;
        let mut m = CudaModel::compile(model, params, batch_size, cuda)?;
        if m.params.is_empty() {
            return Err(Error::InvalidData(format!(
                "{WHO}: the model has no parameters to train"
            )));
        }
        check_untied(&m, model)?;
        if let Some(t) = m.params.iter().find(|t| t.len == 0) {
            return Err(Error::InvalidData(format!(
                "{WHO}: parameter {} holds no values",
                t.id.position()
            )));
        }
        check_rows(&m, model, batch_size, WHO)?;
        let refuse = |e| refused(WHO, e);
        // The ReLU layers whose outputs the dead-neuron check reads, with their values per
        // channel as the CPU check groups them.
        let relus: Vec<(usize, usize)> = (0..m.layers.len())
            .filter(|&i| matches!(m.layers[i].kind, Kind::Activation(Act::Relu | Act::Relu6)))
            .map(|i| (i, channel_size(model, i, m.layers[i].outputs).unwrap_or(1)))
            .collect();
        let mut relu_words = Vec::with_capacity(relus.len());
        let mut words = 0;
        for &(i, spatial) in &relus {
            let channels = m.layers[i].outputs / spatial;
            relu_words.push((i, words, channels));
            words += DeadRelu::words(u32_of(channels)) as usize;
        }
        let lens: Vec<usize> = m.params.iter().map(|t| t.len).collect();
        // Each parameter starts on a 256-byte boundary in the flat buffers, as a buffer of
        // its own would, so cuDNN and cuBLAS see the alignment they were timed on.
        let mut offsets = Vec::with_capacity(lens.len());
        let mut elements = 0usize;
        for &len in &lens {
            offsets.push(elements);
            elements = (elements + len).next_multiple_of(ALIGN);
        }
        let kernel = OptimizerKernel::with_starts(cuda, rule, &lens, &offsets).map_err(refuse)?;
        let s = &mut m.store;
        let mut new = |n: usize| s.values(n).map_err(refuse);
        let values = new(elements)?;
        let grads = new(elements)?;
        let state = (0..rule.state_buffers())
            .map(|_| new(elements))
            .collect::<Result<Vec<_>, _>>()?;
        let n = lens.len();
        let checks = Checks {
            finite: FiniteFlag::with_starts(cuda, &lens, &offsets).map_err(refuse)?,
            squares: SumSquares::with_starts(cuda, &lens, &offsets).map_err(refuse)?,
            loss: new(1)?,
            flag: new(1)?,
            sums: new(n)?,
            bits: new(words.max(1))?,
            words,
            gathered: new(2 + 2 * n + words)?,
            host: vec![0; 2 + 2 * n + words],
        };
        // The flat values start as the parameters, and the state at 0. Then each parameter's
        // buffer becomes its place in the flat values, and each gradient its place in the
        // flat gradients, so the optimiser and the health values read and write them where
        // the layers do, with no copies.
        let mut param_grads = Vec::with_capacity(n);
        for (t, &off) in m.params.iter().zip(&offsets) {
            m.store
                .copy((values, off), (t.at, 0), t.len)
                .map_err(refuse)?;
            // SAFETY: a parameter's part of the flat values and its gradient's part of the
            // flat gradients are only ever given to a call with buffers of other memory: the
            // layers' activations and gradients, the other flat buffer, the optimiser's
            // state and the health values' outputs. The flat buffers themselves go to the
            // optimiser and the health kernels, which take no parameter or gradient part.
            unsafe {
                m.store.make_part(t.at, (values, off), t.len);
                param_grads.push(m.store.part((grads, off), t.len));
            }
        }
        let flat = Flat {
            kernel,
            values,
            grads,
            state,
            offsets,
            active: vec![true; n],
        };
        let pass =
            TrainPass::build(&mut m, loss, &relus, batch_size, &param_grads).map_err(refuse)?;
        let running = (m.layers.iter().enumerate())
            .filter_map(|(i, layer)| match layer.kind {
                Kind::BatchNorm2d(norm) => norm.running.map(|r| {
                    let count = r.count.map(|c| m.counts[c].1);
                    (i, r, RunningUpdate::new(r.momentum, count))
                }),
                _ => None,
            })
            .collect();
        let next = (pass.forward.dropouts.iter())
            .map(|d| vec![0u32; d.len.div_ceil(32)])
            .collect();
        let names = model.parameter_names();
        let mut store_order: Vec<(ParamId, usize)> = (m.params.iter().enumerate())
            .map(|(k, t)| (t.id, k))
            .collect();
        store_order.sort_by_key(|(id, _)| id.position());
        let mut trainer = CudaTrainer {
            model: m,
            pass,
            short: Vec::new(),
            param_grads,
            flat,
            checks,
            running,
            relu_words,
            relus,
            loss,
            taken: 0,
            dropout_seed: 0,
            next,
            next_for: 0,
            labels: Vec::with_capacity(batch_size),
            health: HealthMonitor::new(Health::default()).with_loss(loss),
            names,
            store_order,
            stats: GradStats::new(),
            read: HealthRead::default(),
            skipped: 0,
            profile_asked: false,
            profile: None,
        };
        trainer.draw_next(1);
        Ok(trainer)
    }

    /// Trains on one batch, as [`Trainer::step`](crate::Trainer::step) does: the forward
    /// pass in training mode, the loss (its mean over the batch), the backward pass and the
    /// optimiser's update, then batch norm's running statistics. Returns the loss.
    ///
    /// The batch holds [`batch_size`](Self::batch_size) rows, or fewer, as an epoch's last
    /// batch does: a batch of fewer rows runs a training step set up for that many rows the
    /// first time one comes and kept, one per row count. It shares the parameters, their
    /// gradients, the optimiser's state and batch norm's running statistics and batch counts
    /// with the full step, and has buffers of its own for every value and mask, sized for
    /// its rows.
    ///
    /// The update and the running statistics are skipped on the card when the loss or a
    /// gradient is NaN or infinite; the step is counted either way, and under
    /// [`OnNonFinite::Skip`] also in [`skipped_steps`](Self::skipped_steps), with its loss
    /// returned. The health checks run on every step (see [`CudaTrainer`]).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if the batch does not hold 1 to [`batch_size`](Self::batch_size)
    /// rows of the model's input width, if its targets do not suit the loss or the model's
    /// output, or if a batch norm would normalise its rows over one value per channel;
    /// [`Error::Unsupported`] if the card or NVIDIA's libraries refuse the step for a batch
    /// of fewer rows. Under [`OnNonFinite::Stop`], [`Error::NonFinite`] as the CPU trainer
    /// returns it, naming the step and `"loss"` (with the first NaN or infinite input or
    /// target value, or that the inputs were finite) or the first parameter in the store
    /// whose gradient was not finite; the parameters, the optimiser's state and the running
    /// statistics are then as they were.
    ///
    /// # Panics
    ///
    /// If the card reports an error while the step runs.
    ///
    /// # Examples
    ///
    /// See [`CudaTrainer`].
    pub fn step(&mut self, model: &Sequential, batch: BatchView<'_, f32>) -> Result<f32, Error> {
        self.step_rows(model, batch, None).map(|(loss, _)| loss)
    }

    /// [`step`](Self::step) for a batch whose rows are rows `rows` of a data set, which a
    /// non-finite loss's error then names; also returns whether the step trained (`false`
    /// when [`OnNonFinite::Skip`] left it out). [`fit`](crate::fit) steps through this.
    pub(crate) fn step_rows(
        &mut self,
        model: &Sequential,
        batch: BatchView<'_, f32>,
        rows: Option<&[usize]>,
    ) -> Result<(f32, bool), Error> {
        let started = Instant::now();
        debug_assert_eq!(model.len(), self.model.layers.len());
        let (full, input, output) = (
            self.model.batch,
            self.model.input_width,
            self.model.output_width,
        );
        let b = batch.x.shape().dims().first().copied().unwrap_or(0);
        if batch.x.shape().dims() != [b, input] || b == 0 || b > full {
            return Err(Error::InvalidData(format!(
                "{STEP}: the batch is {:?}, but the trainer takes [rows, {input}] with 1 to \
                 {full} rows",
                batch.x.shape().dims()
            )));
        }
        let targets = match (self.loss, batch.targets) {
            (loss, TargetsView::Classes(labels)) if loss.needs_classes() && labels.len() == b => {
                if let Some((row, &label)) = labels.iter().enumerate().find(|(_, l)| **l >= output)
                {
                    return Err(Error::InvalidData(format!(
                        "{STEP}: the label of row {row} is {label}, but the model has {output} \
                         outputs"
                    )));
                }
                self.labels.clear();
                self.labels.extend(labels.iter().map(|&l| l as u32));
                None
            }
            (loss, TargetsView::Values(t))
                if !loss.needs_classes() && t.shape().dims() == [b, output] =>
            {
                Some(t.data())
            }
            (loss, _) => {
                return Err(Error::InvalidData(format!(
                    "{STEP}: the targets do not suit Loss::{loss:?} on {b} rows of {output} \
                     outputs"
                )));
            }
        };
        let short = if b < full {
            Some(self.prepare_short(model, b)?)
        } else {
            None
        };
        let step_number = self.taken + 1;
        let profiling = std::mem::take(&mut self.profile_asked);
        if short.is_none() && self.next_for != step_number {
            self.draw_next(step_number);
        }
        let cuda = self.model.cuda;
        let s = &self.model.store;
        let shared = &self.model.shared;
        let pass = match short {
            Some(k) => &mut self.short[k].1,
            None => &mut self.pass,
        };
        if short.is_some() {
            for (mask, words) in pass.forward.dropouts.iter().zip(pass.next.iter_mut()) {
                draw(self.dropout_seed, step_number, mask, words);
            }
        }
        let masks: &[Vec<u32>] = if short.is_some() {
            &pass.next
        } else {
            &self.next
        };
        // The batch, the targets and the masks go up.
        ran(STEP, s.get_mut(pass.forward.input).write(batch.x.data()));
        ran(
            STEP,
            match targets {
                Some(values) => s.get_mut(pass.targets).write(values),
                None => s.get_mut(pass.targets).write(&self.labels),
            },
        );
        for (mask, words) in pass.forward.dropouts.iter().zip(masks) {
            ran(STEP, s.get_mut(mask.bits).write(words));
        }
        for &c in &pass.cleared {
            ran(STEP, s.get_mut(c).zero());
        }
        let mut clock = Clock {
            cuda: profiling.then_some(cuda),
            last: Instant::now(),
        };
        let mut profile = profiling.then(|| CudaStepProfile {
            forward: Vec::with_capacity(model.len()),
            loss: Duration::ZERO,
            backward: vec![Duration::ZERO; model.len()],
            backward_input: vec![None; model.len()],
            health: Duration::ZERO,
            optimizer: Duration::ZERO,
            running: Duration::ZERO,
            readback: Duration::ZERO,
            total: Duration::ZERO,
        });
        clock.lap();
        // The forward pass, layer by layer when profiled.
        let f = &mut pass.forward;
        let mut start = 0;
        for &end in &f.ends {
            ran(STEP, run(&f.ops[start..end], &mut f.kernels, shared, s));
            start = end;
            if let Some(p) = profile.as_mut() {
                p.forward.push(clock.lap());
            }
        }
        {
            let mut loss_out = s.get_mut(self.checks.loss);
            let (y, t) = (s.get(f.output), s.get(pass.targets));
            let mut grad = s.get_mut(pass.out_grad);
            ran(
                STEP,
                match &pass.loss {
                    LossKernel::CrossEntropy(ce) => ce.run(&y, &t, &mut grad, (&mut loss_out, 0)),
                    LossKernel::Mse(mse) => mse.run(&y, &t, &mut grad, (&mut loss_out, 0)),
                    LossKernel::Nll(nll) => nll.run(&y, &t, &mut grad, (&mut loss_out, 0)),
                    LossKernel::Regression(r) => r.run(&y, &t, &mut grad, (&mut loss_out, 0)),
                },
            );
        }
        if let Some(p) = profile.as_mut() {
            p.loss = clock.lap();
        }
        // The backward pass, last layer first; a `Linear`'s or a convolution's input
        // gradient timed apart when profiled.
        let mut start = 0;
        let layers = model.len();
        for (r, (&end, &input_at)) in pass
            .backward_ends
            .iter()
            .zip(&pass.backward_inputs)
            .enumerate()
        {
            let layer = layers - 1 - r;
            let split = input_at.filter(|_| profiling).unwrap_or(end);
            ran(
                STEP,
                run(&pass.backward[start..split], &mut f.kernels, shared, s),
            );
            let own = clock.lap();
            ran(
                STEP,
                run(&pass.backward[split..end], &mut f.kernels, shared, s),
            );
            let input_time = clock.lap();
            if let Some(p) = profile.as_mut() {
                p.backward[layer] = own + input_time;
                p.backward_input[layer] = input_at.map(|_| input_time);
            }
            start = end;
        }
        // The next full step's masks are drawn while the card runs this one.
        if short.is_none() {
            let (seed, next) = (self.dropout_seed, &mut self.next);
            for (mask, words) in f.dropouts.iter().zip(next.iter_mut()) {
                draw(seed, step_number + 1, mask, words);
            }
            self.next_for = step_number + 1;
            // A profiled step counts the drawing in its total alone.
            clock.lap();
        }
        // The finite flag and the health values, from the gradients in the flat buffer.
        let flat = &mut self.flat;
        let checks = &mut self.checks;
        ran(
            STEP,
            checks.finite.run(
                (&s.get(checks.loss), 0),
                &s.get(flat.grads),
                &mut s.get_mut(checks.flag),
            ),
        );
        ran(
            STEP,
            checks
                .squares
                .run(&s.get(flat.grads), (&mut s.get_mut(checks.sums), 0)),
        );
        let dead_relu = self.health.health().dead_relu;
        if dead_relu {
            for relu in &pass.relus {
                ran(
                    STEP,
                    relu.kernel.run(
                        &s.get(relu.y),
                        (&mut s.get_mut(checks.bits), u32_of(relu.word)),
                        false,
                    ),
                );
            }
        }
        if let Some(p) = profile.as_mut() {
            p.health = clock.lap();
        }
        // The update, skipped on the card when the flag is set, in place in the parameters.
        {
            let mut values = s.get_mut(flat.values);
            let grads = s.get(flat.grads);
            let flag = s.get(checks.flag);
            let updated = match flat.state.as_slice() {
                [] => flat
                    .kernel
                    .update(&mut values, &grads, &mut [], &flag, &flat.active),
                &[a] => flat.kernel.update(
                    &mut values,
                    &grads,
                    &mut [&mut *s.get_mut(a)],
                    &flag,
                    &flat.active,
                ),
                &[a, b] => {
                    let (mut m1, mut m2) = (s.get_mut(a), s.get_mut(b));
                    flat.kernel.update(
                        &mut values,
                        &grads,
                        &mut [&mut *m1, &mut *m2],
                        &flag,
                        &flat.active,
                    )
                }
                _ => unreachable!("an optimiser keeps at most two state buffers"),
            };
            ran(STEP, updated);
        }
        if let Some(p) = profile.as_mut() {
            p.optimizer = clock.lap();
        }
        // Batch norm's running statistics, skipped on the card when the flag is set.
        for (layer, r, update) in &self.running {
            let &(_, k, stats) = pass
                .norms
                .iter()
                .find(|(l, _, _)| l == layer)
                .expect("every batch norm has its kernel in each pass");
            let (mean, var) = (self.model.buffers[r.mean].at, self.model.buffers[r.var].at);
            ran(
                STEP,
                f.kernels.norms[k].update_running(
                    update,
                    &s.get(stats),
                    (&mut s.get_mut(mean), &mut s.get_mut(var)),
                    &s.get(checks.flag),
                ),
            );
        }
        if let Some(p) = profile.as_mut() {
            p.running = clock.lap();
        }
        // One readback: the loss, the flag, the sums, the counts and the bits.
        let n = self.param_grads.len();
        let words = if dead_relu { checks.words } else { 0 };
        let g = checks.gathered;
        ran(STEP, s.copy((g, 0), (checks.loss, 0), 1));
        ran(STEP, s.copy((g, 1), (checks.flag, 0), 1));
        ran(STEP, s.copy((g, 2), (checks.sums, 0), n));
        ran(
            STEP,
            s.get_mut(g)
                .copy_from(4 * (2 + n), checks.finite.counts(), 0, 4 * n),
        );
        if words > 0 {
            ran(STEP, s.copy((g, 2 + 2 * n), (checks.bits, 0), words));
        }
        let host = &mut checks.host[..2 + 2 * n + words];
        ran(STEP, s.get(g).read(host));
        let value = f32::from_bits(host[0]);
        let flag = host[1];
        let read = &mut self.read;
        read.squares.clear();
        read.squares
            .extend(host[2..2 + n].iter().map(|&w| f32::from_bits(w)));
        read.counts.clear();
        read.counts
            .extend(host[2 + n..2 + 2 * n].iter().map(|&w| f32::from_bits(w)));
        read.bits.clear();
        read.bits.extend_from_slice(&host[2 + 2 * n..]);
        let ran_update = ran(STEP, flat.kernel.commit());
        assert_eq!(
            ran_update,
            flag == 0,
            "{STEP}: the optimiser read another finite flag than the one read back"
        );
        for (_, _, update) in &mut self.running {
            update.commit(ran_update);
        }
        if let Some(p) = profile.as_mut() {
            p.readback = clock.lap();
        }
        self.profile = profile;
        self.taken = step_number;
        let inputs = StepInputs {
            x: batch.x,
            targets: batch.targets,
            rows,
        };
        let checked = self.check(model, step_number, value, inputs);
        // The card applied the update exactly when the flag was clear; the checks must agree:
        // a set flag is a stopped step (an error) or a skipped one (no update).
        assert!(
            match &checked {
                Ok(applied) => *applied == ran_update,
                Err(_) => !ran_update,
            },
            "{STEP}: the health checks disagree with the card's finite flag ({flag}) at step \
             {step_number}"
        );
        // As on the CPU: a step left out has no profile, and the next step is asked for one.
        let left_out = matches!(checked, Ok(false));
        if let Some(p) = self.profile.as_mut() {
            p.total = started.elapsed();
        }
        if left_out && profiling {
            self.profile = None;
            self.profile_asked = true;
        }
        match checked {
            Ok(applied) => {
                self.skipped += u64::from(!applied);
                Ok((value, applied))
            }
            Err(e) => Err(e),
        }
    }

    /// Asks the next step for a [`CudaStepProfile`], read afterwards with
    /// [`last_profile`](Self::last_profile), as [`Trainer::profile_next_step`](crate::Trainer::profile_next_step)
    /// does on the CPU. Only that step is profiled: it waits for the card after each layer's
    /// forward and backward pass (a `Linear`'s or a convolution's input gradient apart from
    /// its parameter gradients), the loss, the health values, the update, the running
    /// statistics and the readback, and times each on the host's clock. A step left out under
    /// [`OnNonFinite::Skip`] has no profile, and the next step is asked instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, TensorView};
    /// use cavitation_cuda::Cuda;
    /// use cavitation_nn::{
    ///     BatchView, CudaTrainer, Health, Linear, Loss, OptimizerConfig, Relu, Sequential,
    ///     TargetsView,
    /// };
    ///
    /// let mut params = ParamStore::new();
    /// let mut rng = Rng::seed(0);
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 4, &mut rng))
    ///     .push(Relu)
    ///     .push(Linear::new(&mut params, 4, 1, &mut rng));
    /// if let Some(cuda) = Cuda::new() {
    ///     let sgd = OptimizerConfig::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 };
    ///     let mut trainer = CudaTrainer::new(&net, &params, &sgd, Loss::Mse, 2, &cuda).unwrap();
    ///     trainer.set_health(Health::quiet()).unwrap();
    ///     let (x, t) = ([1.0f32, 2.0, 3.0, 4.0], [1.0f32, 2.0]);
    ///     let batch = BatchView {
    ///         x: TensorView::from_slice(&x, &[2, 2]).unwrap(),
    ///         targets: TargetsView::Values(TensorView::from_slice(&t, &[2, 1]).unwrap()),
    ///     };
    ///     trainer.profile_next_step();
    ///     trainer.step(&net, batch).unwrap();
    ///     let profile = trainer.last_profile().unwrap();
    ///     assert_eq!((profile.forward.len(), profile.backward.len()), (3, 3));
    ///     // The last `Linear`'s input gradient is timed apart; the first has none.
    ///     let input: Vec<bool> = profile.backward_input.iter().map(|t| t.is_some()).collect();
    ///     assert_eq!(input, [false, false, true]);
    ///     trainer.step(&net, batch).unwrap();
    ///     assert!(trainer.last_profile().is_none()); // not asked
    /// }
    /// ```
    pub fn profile_next_step(&mut self) {
        self.profile_asked = true;
    }

    /// The profile of the last step, if that step was asked for one with
    /// [`profile_next_step`](Self::profile_next_step); `None` after any step that was not, and
    /// before the first step.
    ///
    /// # Examples
    ///
    /// See [`profile_next_step`](Self::profile_next_step).
    pub fn last_profile(&self) -> Option<&CudaStepProfile> {
        self.profile.as_ref()
    }

    /// The health checks of step `step` with loss `value`, on the values read back: the
    /// loss, the dead-ReLU bits, the gradients' finiteness (under [`OnNonFinite::Skip`]) and
    /// their norms, in the CPU trainer's order. `Ok(true)` when the step trained, `Ok(false)`
    /// when it was left out.
    fn check(
        &mut self,
        model: &Sequential,
        step: u64,
        value: f32,
        inputs: StepInputs<'_>,
    ) -> Result<bool, Error> {
        self.health.check_loss(step, value, inputs)?;
        if self.health.health().dead_relu && !self.health.left_out(value) {
            let (layers, relus, bits) = (&self.model.layers, &self.relu_words, &self.read.bits);
            self.health.track_relus(
                model,
                layers.len(),
                |i| layers[i].outputs,
                |i, spatial, alive| {
                    let &(_, word, _) = relus
                        .iter()
                        .find(|r| r.0 == i)
                        .expect("every ReLU layer has its bits");
                    for (c, values) in alive.chunks_mut(spatial).enumerate() {
                        if bits[word + c / 32] >> (c % 32) & 1 == 1 {
                            values.fill(true);
                        }
                    }
                },
            );
        }
        let (counts, order) = (&self.read.counts, &self.store_order);
        let index = |id: ParamId| {
            order
                .iter()
                .find(|(i, _)| *i == id)
                .map(|&(_, k)| k)
                .expect("every named parameter is the model's")
        };
        if !self
            .health
            .before_update_with(step, value, &self.names, |id| counts[index(id)] == 0.0)
        {
            return Ok(false);
        }
        self.stats.reset(order.len());
        for &(id, k) in order {
            let finite = self.read.counts[k] == 0.0;
            let squares = self.read.squares[k];
            // The card sums in `f32`, the CPU in `f64`: a finite gradient whose sum overflows
            // `f32` is reported with its norm at least √f32::MAX, as exploding.
            let sum = if finite && !squares.is_finite() {
                f64::from(f32::MAX)
            } else {
                f64::from(squares)
            };
            self.stats.push(id, sum, finite);
        }
        self.health.after_update(step, &self.stats, &self.names)?;
        Ok(true)
    }

    /// Replaces the health checks run on every step with `health`, starting afresh, as
    /// [`Trainer::set_health`](crate::Trainer::set_health) does.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for [`OnNonFinite::Warn`]: the card never applies a step whose
    /// loss or gradient is NaN or infinite, so it cannot train on as `Warn` does;
    /// [`OnNonFinite::Skip`] leaves such a step out, and [`OnNonFinite::Stop`] stops at it.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_cuda::Cuda;
    /// use cavitation_nn::{CudaTrainer, Health, Linear, Loss, OnNonFinite, OptimizerConfig, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
    /// if let Some(cuda) = Cuda::new() {
    ///     let sgd = OptimizerConfig::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 };
    ///     let mut trainer = CudaTrainer::new(&net, &params, &sgd, Loss::Mse, 4, &cuda).unwrap();
    ///     let skip = Health { on_non_finite: OnNonFinite::Skip, ..Health::quiet() };
    ///     assert!(trainer.set_health(skip).is_ok());
    ///     let warn = Health { on_non_finite: OnNonFinite::Warn, ..Health::quiet() };
    ///     assert!(trainer.set_health(warn).is_err());
    /// }
    /// ```
    pub fn set_health(&mut self, health: Health<'s>) -> Result<(), Error> {
        Self::check_health(&health)?;
        self.health = HealthMonitor::new(health).with_loss(self.loss);
        Ok(())
    }

    /// Whether [`set_health`](Self::set_health) takes `health`, without a trainer to set it
    /// on.
    ///
    /// # Errors
    ///
    /// As [`set_health`](Self::set_health) returns them.
    pub(crate) fn check_health(health: &Health<'_>) -> Result<(), Error> {
        if health.on_non_finite == OnNonFinite::Warn {
            return Err(Error::Unsupported(
                "CudaTrainer: OnNonFinite::Warn is not run on CUDA, which never applies a step \
                 whose loss or gradient is NaN or infinite; use OnNonFinite::Skip to train past \
                 such a step"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Ends epoch `epoch` of the health checks, as [`Trainer::end_epoch`](crate::Trainer::end_epoch)
    /// does: the dead-neuron report, and the check that not every step of the epoch was left
    /// out under [`OnNonFinite::Skip`].
    ///
    /// # Errors
    ///
    /// [`Error::NonFinite`] if every step of the epoch was left out under `Skip`.
    ///
    /// # Examples
    ///
    /// See [`skipped_steps`](Self::skipped_steps).
    pub fn end_epoch(&mut self, epoch: usize) -> Result<(), Error> {
        self.health.end_epoch(epoch)
    }

    /// The steps left out under [`OnNonFinite::Skip`], as
    /// [`Trainer::skipped_steps`](crate::Trainer::skipped_steps) counts them; each is counted
    /// in [`steps`](Self::steps) too.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, TensorView};
    /// use cavitation_cuda::Cuda;
    /// use cavitation_nn::{
    ///     BatchView, CudaTrainer, Health, Linear, Loss, OnNonFinite, OptimizerConfig, Sequential,
    ///     TargetsView,
    /// };
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// if let Some(cuda) = Cuda::new() {
    ///     let sgd = OptimizerConfig::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 };
    ///     let mut trainer = CudaTrainer::new(&net, &params, &sgd, Loss::Mse, 2, &cuda).unwrap();
    ///     let skip = Health { on_non_finite: OnNonFinite::Skip, ..Health::quiet() };
    ///     trainer.set_health(skip).unwrap();
    ///     static GOOD: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    ///     static BAD: [f32; 4] = [1.0, f32::NAN, 3.0, 4.0];
    ///     static T: [f32; 2] = [1.0, 2.0];
    ///     let batch = |x: &'static [f32]| BatchView {
    ///         x: TensorView::from_slice(x, &[2, 2]).unwrap(),
    ///         targets: TargetsView::Values(TensorView::from_slice(&T, &[2, 1]).unwrap()),
    ///     };
    ///     assert!(trainer.step(&net, batch(&GOOD)).unwrap().is_finite());
    ///     assert!(trainer.step(&net, batch(&BAD)).unwrap().is_nan()); // left out
    ///     assert_eq!((trainer.steps(), trainer.skipped_steps()), (2, 1));
    ///     assert!(trainer.end_epoch(1).is_ok()); // one step of the epoch trained
    /// }
    /// ```
    pub fn skipped_steps(&self) -> u64 {
        self.skipped
    }

    /// Copies the parameters, batch norm's running statistics and batch counts into `params`
    /// (the store the trainer was made from, or one laid out like it).
    ///
    /// # Panics
    ///
    /// If an id of the model is not in `params`, or holds another number of values there, or
    /// the card reports an error.
    ///
    /// # Examples
    ///
    /// See [`CudaTrainer`].
    pub fn sync_to(&mut self, params: &mut ParamStore) {
        for (_, r, update) in &self.running {
            if let (Some(count), Some(c)) = (update.count(), r.count) {
                self.model.counts[c].1 = count;
            }
        }
        self.model.download_to(params);
    }

    /// The optimiser's state: each parameter's update count and the moments or momentum, read
    /// back from the card.
    ///
    /// # Panics
    ///
    /// If the card reports an error.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_cuda::Cuda;
    /// use cavitation_nn::{Batch, Batcher, CudaTrainer, Dataset, Linear, Loss, OptimizerConfig, Sequential, Targets};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let data = Dataset { x: Tensor::full(&[2, 2], 1.0), targets: Targets::Values(Tensor::full(&[2, 1], 3.0)) };
    /// if let Some(cuda) = Cuda::new() {
    ///     let adam = OptimizerConfig::Adam { lr: 1e-2, weight_decay: 0.0 };
    ///     let mut trainer = CudaTrainer::new(&net, &params, &adam, Loss::Mse, 2, &cuda).unwrap();
    ///     let mut batch = Batch::new(&data, 2);
    ///     trainer.step(&net, Batcher::new(2, 2, None).gather(&data, 0..2, &mut batch)).unwrap();
    ///     let state = trainer.optimizer_state();
    ///     assert_eq!(state.steps, [1, 1]);
    ///     assert_eq!(state.buffers.len(), 2); // Adam's two moments
    ///     assert_eq!(state.buffers[0][0].len(), 2); // the weight's first moments
    /// }
    /// ```
    pub fn optimizer_state(&mut self) -> CudaOptimizerState {
        let flat = &self.flat;
        let lens: Vec<usize> = self.model.params.iter().map(|t| t.len).collect();
        // The end of the last parameter: each starts on its boundary in the flat buffers.
        let total = (flat.offsets.last().zip(lens.last())).map_or(0, |(&off, &n)| off + n);
        let buffers = (flat.state.iter())
            .map(|&at| {
                let mut all = vec![0.0f32; total];
                ran(
                    "CudaTrainer::optimizer_state",
                    self.model.store.get(at).read(&mut all),
                );
                (flat.offsets.iter().zip(&lens))
                    .map(|(&off, &n)| all[off..off + n].to_vec())
                    .collect()
            })
            .collect();
        CudaOptimizerState {
            ids: self.model.params.iter().map(|t| t.id).collect(),
            steps: flat.kernel.steps().to_vec(),
            buffers,
        }
    }

    /// The model the trainer trains, whose evaluation pass shares the trained parameters:
    /// [`CudaModel::evaluate`] on it measures the parameters as they are now, without a
    /// [`sync_to`](Self::sync_to).
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_cuda::Cuda;
    /// use cavitation_nn::{CudaTrainer, Linear, Loss, OptimizerConfig, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
    /// if let Some(cuda) = Cuda::new() {
    ///     let sgd = OptimizerConfig::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 };
    ///     let mut trainer = CudaTrainer::new(&net, &params, &sgd, Loss::Mse, 8, &cuda).unwrap();
    ///     assert_eq!(trainer.model().batch(), 8);
    /// }
    /// ```
    pub fn model(&mut self) -> &mut CudaModel<'c> {
        &mut self.model
    }

    /// The rows a full step takes; a step takes up to this many.
    ///
    /// # Examples
    ///
    /// See [`CudaTrainer::model`].
    pub fn batch_size(&self) -> usize {
        self.model.batch
    }

    /// The number of steps taken so far, those whose update was skipped included.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_cuda::Cuda;
    /// use cavitation_nn::{CudaTrainer, Linear, Loss, OptimizerConfig, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
    /// if let Some(cuda) = Cuda::new() {
    ///     let sgd = OptimizerConfig::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 };
    ///     assert_eq!(CudaTrainer::new(&net, &params, &sgd, Loss::Mse, 8, &cuda).unwrap().steps(), 0);
    /// }
    /// ```
    pub fn steps(&self) -> u64 {
        self.taken
    }

    /// Sets the seed of the dropout masks, as [`Trainer::set_dropout_seed`](crate::Trainer::set_dropout_seed)
    /// does: step `n` drops the values a training tape seeded `seed` drops at step `n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_cuda::Cuda;
    /// use cavitation_nn::{CudaTrainer, Linear, Loss, OptimizerConfig, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
    /// if let Some(cuda) = Cuda::new() {
    ///     let sgd = OptimizerConfig::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 };
    ///     let mut trainer = CudaTrainer::new(&net, &params, &sgd, Loss::Mse, 8, &cuda).unwrap();
    ///     trainer.set_dropout_seed(42);
    ///     assert_eq!(trainer.dropout_seed(), 42);
    /// }
    /// ```
    pub fn set_dropout_seed(&mut self, seed: u64) {
        self.dropout_seed = seed;
        self.next_for = 0;
    }

    /// The seed of the dropout masks.
    ///
    /// # Examples
    ///
    /// See [`set_dropout_seed`](Self::set_dropout_seed).
    pub fn dropout_seed(&self) -> u64 {
        self.dropout_seed
    }

    /// Sets the learning rate for the later steps.
    ///
    /// # Panics
    ///
    /// If `lr` is not finite as an `f32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_cuda::Cuda;
    /// use cavitation_nn::{CudaTrainer, Linear, Loss, OptimizerConfig, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
    /// if let Some(cuda) = Cuda::new() {
    ///     let sgd = OptimizerConfig::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 };
    ///     let mut trainer = CudaTrainer::new(&net, &params, &sgd, Loss::Mse, 8, &cuda).unwrap();
    ///     trainer.set_learning_rate(0.05);
    ///     assert_eq!(trainer.learning_rate(), 0.05);
    /// }
    /// ```
    pub fn set_learning_rate(&mut self, lr: f64) {
        if let Err(e) = self.flat.kernel.set_learning_rate(lr) {
            panic!("CudaTrainer::set_learning_rate: {e}");
        }
    }

    /// The learning rate the next step trains at.
    ///
    /// # Examples
    ///
    /// See [`set_learning_rate`](Self::set_learning_rate).
    pub fn learning_rate(&self) -> f64 {
        self.flat.kernel.rule().learning_rate()
    }

    /// Draws step `step`'s masks into `next`.
    fn draw_next(&mut self, step: u64) {
        for (mask, words) in self.pass.forward.dropouts.iter().zip(self.next.iter_mut()) {
            draw(self.dropout_seed, step, mask, words);
        }
        self.next_for = step;
    }

    /// The index in `short` of the step for a batch of `rows` rows, fewer than the trainer's
    /// batch size: built the first time a batch of that many rows comes, and kept.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if a batch norm would normalise `rows` rows over one value per
    /// channel; [`Error::Unsupported`] if the card or NVIDIA's libraries refuse the step.
    fn prepare_short(&mut self, model: &Sequential, rows: usize) -> Result<usize, Error> {
        if let Some(k) = self.short.iter().position(|(r, _)| *r == rows) {
            return Ok(k);
        }
        check_rows(&self.model, model, rows, STEP)?;
        let mut pass = TrainPass::build(
            &mut self.model,
            self.loss,
            &self.relus,
            rows,
            &self.param_grads,
        )
        .map_err(|e| refused("CudaTrainer", e))?;
        pass.next = (pass.forward.dropouts.iter())
            .map(|d| vec![0u32; d.len.div_ceil(32)])
            .collect();
        self.short.push((rows, pass));
        Ok(self.short.len() - 1)
    }

    /// Reads back each max pooling's winners from the last step, in the model's order: for
    /// the tests.
    #[cfg(test)]
    pub(crate) fn last_winners(&self) -> Vec<Vec<u32>> {
        (self.pass.forward.saved.iter().zip(&self.model.layers))
            .filter(|(_, l)| matches!(l.kind, Kind::MaxPool2d(_)))
            .map(|(saved, _)| {
                let b = self
                    .model
                    .store
                    .get(saved.scratch.expect("max pooling's winners"));
                let mut words = vec![0u32; b.bytes() / 4];
                b.read(&mut words).unwrap();
                words
            })
            .collect()
    }

    /// Reads back each max pooling's input from the last full step, by layer: for the tests.
    #[cfg(test)]
    pub(crate) fn last_pool_inputs(&self) -> Vec<(usize, Vec<f32>)> {
        (self
            .pass
            .forward
            .saved
            .iter()
            .zip(&self.model.layers)
            .enumerate())
        .filter(|(_, (_, l))| matches!(l.kind, Kind::MaxPool2d(_)))
        .map(|(i, (saved, l))| {
            let mut values = vec![0.0f32; self.model.batch * l.inputs];
            self.model.store.get(saved.input).read(&mut values).unwrap();
            (i, values)
        })
        .collect()
    }

    /// Reads back each ReLU's input from the last full step, by layer: for the tests.
    #[cfg(test)]
    pub(crate) fn last_relu_inputs(&self) -> Vec<(usize, Vec<f32>)> {
        (self
            .pass
            .forward
            .saved
            .iter()
            .zip(&self.model.layers)
            .enumerate())
        .filter(|(_, (_, l))| matches!(l.kind, Kind::Activation(Act::Relu)))
        .map(|(i, (saved, l))| {
            let mut values = vec![0.0f32; self.model.batch * l.inputs];
            self.model.store.get(saved.input).read(&mut values).unwrap();
            (i, values)
        })
        .collect()
    }

    /// Reads back each parameter's gradient from the last step, in the model's order: for
    /// the tests.
    #[cfg(test)]
    pub(crate) fn last_gradients(&self) -> Vec<(ParamId, Vec<f32>)> {
        (self.model.params.iter().zip(&self.param_grads))
            .map(|(t, &g)| {
                let mut out = vec![0.0f32; t.len];
                self.model.store.get(g).read(&mut out).unwrap();
                (t.id, out)
            })
            .collect()
    }
}

/// `n` as a `u32`, for a size the kernels were set up with.
fn u32_of(n: usize) -> u32 {
    u32::try_from(n).expect("the kernels were set up with this size as a u32")
}

/// Refuses a step of `rows` rows when a batch norm would normalise them over one value per
/// channel, naming `who`.
fn check_rows(m: &CudaModel<'_>, model: &Sequential, rows: usize, who: &str) -> Result<(), Error> {
    for (i, layer) in m.layers.iter().enumerate() {
        if let Kind::BatchNorm2d(norm) = layer.kind
            && rows * norm.spatial < 2
        {
            return Err(Error::InvalidData(format!(
                "{who}: layer {i} ({}) normalises a training batch with its own statistics, \
                 which need more than one value per channel; a batch of {rows} gives {}",
                model.layer(i).name(),
                rows * norm.spatial
            )));
        }
    }
    Ok(())
}

/// The mask `mask` of step `step` under `seed`, packed into `words` (low bit first).
fn draw(seed: u64, step: u64, mask: &DropoutMask, words: &mut [u32]) {
    words.fill(0);
    dropout_mask(seed, step, mask.stream, mask.p, mask.len, |i, kept| {
        if kept {
            words[i / 32] |= 1 << (i % 32);
        }
    });
}

/// Refuses a model whose layers share a parameter: the kernels overwrite each gradient, so a
/// second use would replace the first's contribution.
fn check_untied(m: &CudaModel<'_>, model: &Sequential) -> Result<(), Error> {
    let mut uses = vec![0usize; m.params.len()];
    for layer in &m.layers {
        let ids: Vec<usize> = match layer.kind {
            Kind::Linear { weight, bias, .. } | Kind::Conv2d { weight, bias, .. } => {
                std::iter::once(weight).chain(bias).collect()
            }
            Kind::BatchNorm2d(n) => n.weight.into_iter().chain(n.bias).collect(),
            Kind::Prelu { weight, .. } => vec![weight],
            _ => Vec::new(),
        };
        for k in ids {
            uses[k] += 1;
        }
    }
    if let Some(k) = uses.iter().position(|&n| n > 1) {
        let id = m.params[k].id;
        let name = (model.parameter_names().into_iter())
            .find(|(_, i)| *i == id)
            .map_or_else(|| format!("parameter {}", id.position()), |(n, _)| n);
        return Err(Error::Unsupported(format!(
            "CudaTrainer: {name} is used by more than one layer, which CUDA does not train yet"
        )));
    }
    Ok(())
}

impl<'c> TrainPass<'c> {
    /// Sets up the training step of `m` for batches of `rows` rows (at most the model's
    /// batch): the forward pass in training mode, the loss, the backward pass into
    /// `param_grads` (each parameter's gradient, in the model's order) and the dead-ReLU
    /// bits of `relus` (each ReLU layer with its values per channel).
    fn build(
        m: &mut CudaModel<'c>,
        loss: Loss,
        relus: &[(usize, usize)],
        rows: usize,
        param_grads: &[usize],
    ) -> Result<TrainPass<'c>, CudaError> {
        let cuda = m.cuda;
        let params: Vec<usize> = m.params.iter().map(|t| t.at).collect();
        let buffers: Vec<usize> = m.buffers.iter().map(|t| t.at).collect();
        let masks: Vec<usize> = m.masks.iter().map(|&(_, at)| at).collect();
        let parts = Parts {
            layers: &m.layers,
            params: &params,
            buffers: &buffers,
            masks: &masks,
            input_width: m.input_width,
        };
        let forward = Forward::build(&mut m.store, &parts, rows, true)?;
        let s: &mut Store<'c> = &mut m.store;
        let out_width = m.output_width;
        let targets = s.values(rows * out_width)?;
        let out_grad = s.values(rows * out_width)?;
        let loss = match loss {
            Loss::CrossEntropy => LossKernel::CrossEntropy(SoftmaxCrossEntropy::new(
                cuda,
                (u32_of(rows), u32_of(out_width)),
            )?),
            Loss::Nll => LossKernel::Nll(NegativeLogLikelihood::new(
                cuda,
                (u32_of(rows), u32_of(out_width)),
            )?),
            _ => match regression(loss) {
                Some(kind) => LossKernel::Regression(RegressionLoss::new(
                    cuda,
                    u32_of(rows * out_width),
                    kind,
                )?),
                None => LossKernel::Mse(MeanSquaredError::new(cuda, u32_of(rows * out_width))?),
            },
        };

        // The backward pass, last layer first. A parameter's gradient is written once, by its
        // layer, into its own buffer.
        let mut g = out_grad;
        let mut backward = Vec::new();
        let mut backward_ends = Vec::with_capacity(m.layers.len());
        let mut backward_inputs = Vec::with_capacity(m.layers.len());
        let mut cleared = Vec::new();
        for (i, layer) in m.layers.iter().enumerate().rev() {
            let saved = forward.saved[i];
            let (n_out, n_in) = (rows * layer.outputs, rows * layer.inputs);
            if let Some(mi) = layer.mask {
                let pre = s.values(n_out)?;
                backward.push(Op::KeepWhere {
                    n: n_out,
                    x: g,
                    mask: masks[mi],
                    y: pre,
                });
                g = pre;
            }
            let first = i == 0;
            let x = saved.input;
            let kernel = || saved.kernel.expect("the layer saves its kernel");
            let mut input_at = None;
            g = match layer.kind {
                Kind::Linear { weight, bias, .. } => {
                    let k = kernel();
                    backward.push(Op::LinearWeightGrad {
                        k,
                        x,
                        dy: g,
                        dw: param_grads[weight],
                    });
                    if let Some(b) = bias {
                        backward.push(Op::LinearBiasGrad {
                            k,
                            dy: g,
                            db: param_grads[b],
                        });
                    }
                    if first {
                        g
                    } else {
                        let dx = s.values(n_in)?;
                        input_at = Some(backward.len());
                        backward.push(Op::LinearInputGrad {
                            k,
                            w: params[weight],
                            dy: g,
                            dx,
                        });
                        dx
                    }
                }
                Kind::Conv2d { weight, bias, .. } => {
                    let k = kernel();
                    backward.push(Op::ConvWeightGrad {
                        k,
                        x,
                        dy: g,
                        dw: param_grads[weight],
                    });
                    if let Some(b) = bias {
                        backward.push(Op::ConvBiasGrad {
                            k,
                            dy: g,
                            db: param_grads[b],
                        });
                    }
                    if first {
                        g
                    } else {
                        let dx = s.values(n_in)?;
                        input_at = Some(backward.len());
                        backward.push(Op::ConvInputGrad {
                            k,
                            w: params[weight],
                            dy: g,
                            dx,
                        });
                        dx
                    }
                }
                Kind::Activation(act) => {
                    let dx = s.values(n_in)?;
                    let (op, from_output) = backward_op(act);
                    backward.push(Op::Elementwise {
                        op,
                        settings: act.settings(),
                        n: n_in,
                        a: if from_output { saved.pre } else { x },
                        b: Some(g),
                        y: dx,
                    });
                    dx
                }
                Kind::Softmax { log } => {
                    let dx = s.values(n_in)?;
                    backward.push(Op::Softmax {
                        op: if log {
                            RowSoftmaxOp::LogSoftmaxBackward
                        } else {
                            RowSoftmaxOp::SoftmaxBackward
                        },
                        rows,
                        width: layer.outputs,
                        a: saved.pre,
                        b: Some(g),
                        y: dx,
                    });
                    dx
                }
                Kind::Prelu { channels, weight } => {
                    let (rows, channels) = prelu_shape(rows, layer.inputs, channels);
                    let partials =
                        s.values(cavitation_cuda::Prelu::partials_len(rows, channels))?;
                    backward.push(Op::PreluSlopeGrad {
                        rows,
                        channels,
                        x,
                        dy: g,
                        partials,
                        ds: param_grads[weight],
                    });
                    let dx = s.values(n_in)?;
                    backward.push(Op::Prelu {
                        op: PreluOp::InputGrad,
                        rows,
                        channels,
                        x,
                        slope: params[weight],
                        dy: Some(g),
                        y: dx,
                    });
                    dx
                }
                Kind::Glu(gate) => {
                    let dx = s.values(n_in)?;
                    backward.push(Op::Glu {
                        op: GluOp::Backward,
                        gate: glu_gate(gate),
                        rows,
                        width: layer.inputs,
                        x,
                        dy: Some(g),
                        y: dx,
                    });
                    dx
                }
                Kind::MaxPool2d(_) => {
                    let dx = s.values(n_in)?;
                    backward.push(Op::MaxPoolBackward {
                        k: kernel(),
                        dy: g,
                        winners: saved.scratch.expect("max pooling saves its winners"),
                        dx,
                    });
                    // The pooling backward passes add to their input gradient.
                    cleared.push(dx);
                    dx
                }
                Kind::AvgPool2d { .. } => {
                    let dx = s.values(n_in)?;
                    backward.push(Op::AvgPoolBackward {
                        k: kernel(),
                        dy: g,
                        dx,
                    });
                    cleared.push(dx);
                    dx
                }
                Kind::AdaptiveAvgPool2d { .. } => {
                    let dx = s.values(n_in)?;
                    backward.push(Op::AdaptivePoolBackward {
                        k: kernel(),
                        dy: g,
                        dx,
                    });
                    cleared.push(dx);
                    dx
                }
                Kind::BatchNorm2d(norm) => {
                    let dx = s.values(n_in)?;
                    let affine = match (norm.weight, norm.bias) {
                        (Some(w), Some(b)) => Some((params[w], param_grads[w], param_grads[b])),
                        _ => None,
                    };
                    backward.push(Op::NormBackward {
                        k: kernel(),
                        x,
                        stats: saved.scratch.expect("batch norm saves its statistics"),
                        dy: g,
                        dx,
                        affine,
                    });
                    dx
                }
                Kind::Dropout(p) => {
                    let dx = s.values(n_in)?;
                    let mask = forward.dropouts[saved.dropout.expect("dropout saves its mask")];
                    backward.push(Op::Dropout {
                        n: n_in,
                        x: g,
                        mask: mask.bits,
                        y: dx,
                        scale: cavitation_cuda::Dropout::scale(p)?,
                    });
                    dx
                }
                Kind::Identity => g,
            };
            backward_ends.push(backward.len());
            backward_inputs.push(input_at);
        }
        let mut word = 0;
        let mut relu_bits = Vec::with_capacity(relus.len());
        for &(i, spatial) in relus {
            let channels = m.layers[i].outputs / spatial;
            let dims = ChannelDims::new(u32_of(rows), u32_of(channels), u32_of(spatial));
            relu_bits.push(ReluBits {
                word,
                kernel: DeadRelu::new(cuda, dims)?,
                y: forward.saved[i].pre,
            });
            word += DeadRelu::words(u32_of(channels)) as usize;
        }
        let norms = (m.layers.iter().enumerate())
            .filter(|(_, l)| matches!(l.kind, Kind::BatchNorm2d(_)))
            .map(|(i, _)| {
                let saved = forward.saved[i];
                (
                    i,
                    saved.kernel.expect("batch norm saves its kernel"),
                    saved.scratch.expect("batch norm saves its statistics"),
                )
            })
            .collect();
        Ok(TrainPass {
            forward,
            loss,
            targets,
            out_grad,
            backward,
            backward_ends,
            backward_inputs,
            cleared,
            relus: relu_bits,
            norms,
            next: Vec::new(),
        })
    }
}
