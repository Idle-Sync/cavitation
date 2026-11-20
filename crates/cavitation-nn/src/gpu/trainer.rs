//! `GpuTrainer`: training a [`GpuModel`] on the GPU one batch at a time. A step is one
//! command encoder and one readback: the batch and its dropout masks go up, the training
//! forward pass, the loss, the backward pass, the finite flag, the optimiser's update and
//! batch norm's running statistics run, and the loss and the flag come back.

use std::thread;
use std::time::{Duration, Instant};

use cavitation_core::{Error, ParamId, ParamStore, dropout_mask};
use cavitation_gpu::kernels::{
    AvgPool2d, BatchNormBackward, BatchNormForward, BatchStats, BiasAddCall, ChannelDims,
    ChannelSums, ChannelSumsCall, ConvCall, ConvDims, ConvInputGrad, ConvWeightGrad,
    ConvWeightGradCall, CrossEntropyCall, DeadRelu, DeadReluCall, Elementwise, ElementwiseCall,
    FiniteFlag, FiniteFlagCall, GemmCall, GemmDims, MaxPool2d, MeanSquaredError, MseCall,
    OptimizerKernel, OptimizerRule, Prelu, PreluCall, RowSoftmax, RowSoftmaxCall, RunningUpdate,
    SoftmaxCrossEntropy, SumSquares, SumSquaresCall,
};
use cavitation_gpu::{
    BufferPool, Gpu, GpuError, PipelineCache, PooledBuffer, Step, StepEncoder, StepOutput,
    StepSizes, TimedKind, Timing, wgpu,
};

use super::forward::{Kernels, prelu_shape};
use super::model::{Act, GpuModel, Kind};
use crate::health::channel_size;
use crate::{
    BatchView, GradStats, Health, HealthMonitor, Loss, OnNonFinite, OptimizerConfig, Sequential,
    StepInputs, TargetsView,
};

/// The usage of every training buffer: bound as storage, written by copies and uploads, read
/// back.
fn usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
}

/// `n` as a `u32`, for a size [`GpuModel::compile`] has checked fits a binding.
fn u32_of(n: usize) -> u32 {
    u32::try_from(n).expect("GpuModel::compile checks every size against the GPU's bindings")
}

/// The kernels training adds to the model's.
#[derive(Debug)]
struct TrainKernels {
    /// The convolution's input gradient.
    conv_input: ConvInputGrad,
    /// The convolution's weight gradient.
    conv_weight: ConvWeightGrad,
    /// Per-channel sums: the bias gradients.
    sums: ChannelSums,
    /// Softmax cross-entropy and its gradient.
    cross_entropy: SoftmaxCrossEntropy,
    /// Mean squared error and its gradient.
    mse: MeanSquaredError,
    /// The finite flag over the loss and every gradient.
    finite: FiniteFlag,
    /// Each parameter gradient's sum of squares, for the gradient-norm checks.
    squares: SumSquares,
    /// Each ReLU neuron's (or channel's) bit, set when it was non-zero in the step, for the
    /// dead-ReLU check, which ORs the steps' bits over the epoch.
    dead_relu: DeadRelu,
}

/// A pooling layer's kernel, built for the batch.
#[derive(Debug)]
enum Pool {
    /// Max pooling.
    Max(MaxPool2d),
    /// Average pooling.
    Avg(AvgPool2d),
}

/// One prepared call of a training step, forward or backward.
#[derive(Debug)]
enum Op {
    /// A GEMM.
    Gemm(GemmCall),
    /// A bias added per column.
    Bias(BiasAddCall),
    /// A convolution forward pass.
    Conv(ConvCall),
    /// A convolution's input gradient.
    ConvInput(ConvCall),
    /// A convolution's weight gradient.
    ConvWeight(ConvWeightGradCall),
    /// Per-channel sums.
    Sums(ChannelSumsCall),
    /// An element-wise operation.
    Elementwise(ElementwiseCall),
    /// A softmax or log-softmax over each row, forward or backward.
    Softmax(RowSoftmaxCall),
    /// PReLU forward, or its input gradient.
    Prelu(PreluCall),
    /// A PReLU's slope gradient.
    PreluSlope(ChannelSumsCall),
    /// Pooling forward: the kernel's index in [`TrainPass::pools`] and the bind group.
    PoolForward(usize, wgpu::BindGroup),
    /// Pooling backward: the kernel's index and the bind group.
    PoolBackward(usize, wgpu::BindGroup),
    /// Batch norm's forward pass, boxed for its size.
    NormForward(Box<BatchNormForward>),
    /// Batch norm's backward pass.
    NormBackward(Box<BatchNormBackward>),
}

/// The loss of a training step.
#[derive(Debug)]
enum LossCall {
    /// Softmax cross-entropy on class labels.
    CrossEntropy(CrossEntropyCall),
    /// Mean squared error on target values.
    Mse(MseCall),
}

/// A dropout layer's mask for a step: the layer's stream, drop probability and the packed
/// bits on the GPU.
#[derive(Debug)]
struct DropoutMask {
    /// The tape's random stream for the layer: its rank among the model's dropout layers.
    stream: u64,
    /// The drop probability.
    p: f64,
    /// The values the mask covers: the batch's rows times the layer's width.
    len: usize,
    /// The bits on the GPU, one per value.
    bits: PooledBuffer,
}

/// The training step's calls and buffers.
#[derive(Debug)]
struct TrainPass {
    /// The forward calls, in order.
    forward: Vec<Op>,
    /// Where each layer's forward calls end in `forward`, in layer order.
    forward_ends: Vec<usize>,
    /// The backward calls, in order.
    backward: Vec<Op>,
    /// Where each layer's backward calls end in `backward`, last layer first.
    backward_ends: Vec<usize>,
    /// Where a `Linear`'s or a `Conv2d`'s input gradient begins in `backward`, after its
    /// parameter gradients, last layer first; `None` for a layer whose backward calls are
    /// one part: every other kind, and a first layer, which has no input gradient.
    backward_inputs: Vec<Option<usize>>,
    /// The loss.
    loss: LossCall,
    /// The finite flag over the loss and the parameter gradients.
    finite: FiniteFlagCall,
    /// Each batch norm's running-statistics update, with its layer.
    running: Vec<(usize, RunningUpdate)>,
    /// The pooling kernels.
    pools: Vec<Pool>,
    /// The values: the input first, then each output a forward call writes.
    values: Vec<PooledBuffer>,
    /// The gradients of the values.
    grads: Vec<PooledBuffer>,
    /// Each parameter's gradient, in the model's parameter order.
    param_grads: Vec<PooledBuffer>,
    /// The targets: class labels as `u32`, or target values.
    targets: PooledBuffer,
    /// The loss, one `f32`.
    loss_value: PooledBuffer,
    /// The finite flag, one `u32`, 0 when the loss and every gradient are finite.
    flag: PooledBuffer,
    /// Max pooling's winners and batch norm's statistics, held as long as their calls.
    #[allow(dead_code)] // Read only through the bind groups.
    scratch: Vec<PooledBuffer>,
    /// The dropout masks, in the model's order.
    masks: Vec<DropoutMask>,
    /// The gradients the pooling backward passes add to (an index into `grads` each), which
    /// every step clears first.
    cleared: Vec<usize>,
    /// Each max pooling's winners, an index into the scratch, in the model's order.
    #[allow(dead_code)] // The tests read the winners.
    winners: Vec<usize>,
    /// Each parameter gradient's sum of squares, into word `k` of `square_sums`.
    squares: Vec<SumSquaresCall>,
    /// The sums of squares, one `f32` per parameter in the model's order.
    square_sums: PooledBuffer,
    /// Each ReLU layer's bits.
    relus: Vec<ReluBits>,
    /// The bits of every ReLU layer, overwritten each step.
    relu_bits: PooledBuffer,
}

/// One ReLU layer's dead-neuron bits: one per channel of `spatial` values, or one per neuron
/// (`spatial` 1), as the CPU health checks group them.
#[derive(Debug)]
struct ReluBits {
    /// The layer.
    layer: usize,
    /// Its first word in the bits.
    word: usize,
    /// Its channels: its width over `spatial`.
    channels: usize,
    /// The bits' call.
    call: DeadReluCall,
}

/// The optimiser's state on the GPU: the parameters and gradients laid end to end, its state
/// buffers, and where each parameter sits.
#[derive(Debug)]
struct Flat {
    /// The update.
    kernel: OptimizerKernel,
    /// Its bind group.
    group: wgpu::BindGroup,
    /// Every parameter's values, end to end.
    values: PooledBuffer,
    /// Every parameter's gradient, end to end.
    grads: PooledBuffer,
    /// The state buffers: SGD's momentum, or Adam's first and second moments.
    state: Vec<PooledBuffer>,
    /// Each parameter's first element in the flat buffers.
    offsets: Vec<usize>,
    /// Whether each parameter updates: every one has a gradient each step.
    active: Vec<bool>,
}

/// The optimiser's state, as [`GpuTrainer::optimizer_state`] reads it back.
///
/// # Examples
///
/// See [`GpuTrainer::optimizer_state`].
#[derive(Clone, Debug, PartialEq)]
pub struct GpuOptimizerState {
    /// The parameters, in the model's order.
    pub ids: Vec<ParamId>,
    /// Each parameter's update count, as the CPU optimisers count it.
    pub steps: Vec<u32>,
    /// The state buffers, each with one entry per parameter of its values: for SGD with
    /// momentum the momentum buffer; for Adam and AdamW the first and second moments, in
    /// that order; none for plain SGD.
    pub buffers: Vec<Vec<Vec<f32>>>,
}

/// Trains a [`Sequential`] on the GPU, one batch at a time: [`step`](GpuTrainer::step) runs
/// the forward pass in training mode, the loss, the backward pass and the optimiser's update
/// in one command encoder with one readback (the loss and the finite flag), and
/// [`sync_to`](GpuTrainer::sync_to) copies the parameters and batch-norm buffers back into a
/// [`ParamStore`].
///
/// A step does what [`Trainer::step`](crate::Trainer::step) does on the CPU, with the same
/// dropout masks (from [`dropout_mask`] with the trainer's seed and the step's number) and
/// the same order of work. The model's layers must be ones [`GpuModel::compile`] takes; the
/// losses are [`Loss::CrossEntropy`] and [`Loss::Mse`], and the optimisers SGD (with or
/// without momentum), Adam and AdamW. A parameter used by two layers is refused.
///
/// While a full step runs on the GPU, the next full step's dropout masks are drawn on the host;
/// a batch of fewer rows draws its own before it runs.
///
/// Every step runs the CPU trainer's [`Health`] checks (the default unless
/// [`set_health`](GpuTrainer::set_health) changes it) on values the step reads back with its
/// loss: the finite flag, each gradient's count of NaN and infinite values and its sum of
/// squares, and each ReLU neuron's bit, set when it was non-zero in the step (one per channel
/// after a convolution).
/// It sends the CPU's warnings and returns the CPU's errors, at the same steps and naming the
/// same parameters. One difference is deliberate: a step whose loss or gradient is NaN or
/// infinite is never applied on the GPU, so under [`OnNonFinite::Stop`] the parameters, the
/// optimiser's state and the running statistics stay as they were, where the CPU trainer has
/// already applied a non-finite gradient that came with a finite loss.
/// [`OnNonFinite::Warn`], which trains on through such a step, is refused.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_gpu::Gpu;
/// use cavitation_nn::{
///     Batch, Batcher, Dataset, GpuTrainer, Linear, Loss, OptimizerConfig, Relu, Sequential,
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
/// if let Some(gpu) = Gpu::new() {
///     let sgd = OptimizerConfig::Sgd { lr: 0.5, momentum: 0.0, weight_decay: 0.0 };
///     let mut trainer = GpuTrainer::new(&net, &params, &sgd, Loss::CrossEntropy, 4, &gpu).unwrap();
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
pub struct GpuTrainer<'s> {
    /// The model: its parameters and buffers, and its evaluation pass.
    model: GpuModel,
    /// The training kernels.
    #[allow(dead_code)] // Read only through the prepared calls.
    kernels: TrainKernels,
    /// The step's calls and buffers.
    pass: TrainPass,
    /// The optimiser.
    flat: Flat,
    /// The encoder the steps are recorded in.
    steps: StepEncoder,
    /// The loss.
    loss: Loss,
    /// The steps taken.
    taken: u64,
    /// The seed of the dropout masks.
    dropout_seed: u64,
    /// The bits of the next step's masks, one list per dropout layer, drawn ahead.
    next: Vec<Vec<u32>>,
    /// The step number `next` was drawn for.
    next_for: u64,
    /// The bytes of the batch's upload, reused from step to step.
    upload: Vec<u8>,
    /// The health checks run on every step.
    health: HealthMonitor<'s>,
    /// The model's parameter names, for the health checks' messages.
    names: Vec<(String, ParamId)>,
    /// The model's parameters (an index into its parameter order) in the order of the
    /// store, the order the CPU optimisers fill their gradient statistics in.
    store_order: Vec<(ParamId, usize)>,
    /// The gradient statistics of the last step, from the sums of squares and the counts.
    stats: GradStats,
    /// The last step's health values: each gradient's sum of squares and count of NaN and
    /// infinite values, and the ReLU bits.
    read: HealthRead,
    /// The steps left out under [`OnNonFinite::Skip`].
    skipped: u64,
    /// The ReLU layers and their values per channel, for building a pass.
    relus: Vec<(usize, usize)>,
    /// Whether the next step is profiled.
    profile_asked: bool,
    /// The last profiled step's profile, when it is the last step.
    profile: Option<GpuStepProfile>,
    /// The steps for batches of fewer rows than [`batch_size`](GpuTrainer::batch_size) (an
    /// epoch's last), one per row count, each built the first time a batch of its rows comes.
    short: Vec<ShortPass>,
}

/// The training step for a batch of fewer rows than the trainer's batch size.
#[derive(Debug)]
struct ShortPass {
    /// Its rows.
    rows: usize,
    /// Its calls and buffers.
    pass: TrainPass,
    /// The optimiser's bind group with this pass's finite flag.
    group: wgpu::BindGroup,
    /// Its dropout masks' bits, one list per dropout layer.
    next: Vec<Vec<u32>>,
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

/// One training step's GPU time by part, from [`GpuTrainer::last_profile`]: what
/// [`Trainer::last_profile`](crate::Trainer::last_profile) gives on the CPU, timed on the GPU.
/// A part the device could not time is `None`; [`timing`](GpuStepProfile::timing) says which
/// are timed and names what is missing. A profiled step runs each part in compute passes of
/// its own, so its parts describe a step structured for profiling.
///
/// The struct is `#[non_exhaustive]`, so a later field is not a breaking change: outside
/// this crate a profile comes from [`GpuTrainer::last_profile`] and is read through its
/// public fields.
///
/// # Examples
///
/// See [`GpuTrainer::profile_next_step`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct GpuStepProfile {
    /// Each layer's forward passes, in layer order.
    pub forward: Vec<Option<Duration>>,
    /// The loss and its gradient.
    pub loss: Option<Duration>,
    /// Each layer's backward passes, in layer order.
    pub backward: Vec<Option<Duration>>,
    /// The part of each [`backward`](Self::backward) time that is the layer's input gradient,
    /// for a `Linear` or a `Conv2d`, in layer order; the rest of the layer's time is its
    /// parameter gradients (the weight's and the bias's, after the layer's mask if it has
    /// one). The two parts run in compute passes of their own. `None` for a layer whose
    /// backward pass is one part: every other kind of layer, and a first layer, which
    /// computes no input gradient; and, as for every part, where the device could not time
    /// it. A layer with a `backward` time and no `backward_input` is one that is not split.
    pub backward_input: Vec<Option<Duration>>,
    /// The health values: the finite flag, the sums of squares and the ReLU bits.
    pub health: Option<Duration>,
    /// The optimiser's update.
    pub optimizer: Option<Duration>,
    /// The copies of the gradients into the optimiser's flat buffer and of the values back
    /// out, with the update's uniform uploads, when the device times copies
    /// ([`Timing::PassesAndCopies`]).
    pub copies: Option<Duration>,
    /// Batch norm's running statistics, with their uniform uploads where the device times
    /// copies.
    pub running: Option<Duration>,
    /// The readback of the loss, the flag and the health values, when the device times
    /// copies.
    pub readback: Option<Duration>,
    /// The whole step on the host's clock, from the call to its return: the batch's checks
    /// and packing, a first short batch's compile, the uploads and the health checks
    /// included.
    pub total: Duration,
    /// What the device times, and by name what it cannot.
    pub timing: Timing,
}

/// The labels of a profiled step's spans, in the order recorded: each layer's forward pass,
/// the loss, each layer's backward pass (last layer first; a `Linear`'s or a `Conv2d`'s
/// input gradient is a span of its own, [`BACKWARD_INPUT`], after the layer's
/// [`BACKWARD`]), the health values, the update, the running statistics and the readback.
const FORWARD: &str = "forward";
/// See [`FORWARD`].
const LOSS: &str = "loss";
/// See [`FORWARD`].
const BACKWARD: &str = "backward";
/// See [`FORWARD`].
const BACKWARD_INPUT: &str = "backward, input gradient";
/// See [`FORWARD`].
const HEALTH: &str = "health";
/// See [`FORWARD`].
const OPTIMIZER: &str = "optimizer";
/// See [`FORWARD`].
const RUNNING: &str = "running statistics";
/// See [`FORWARD`].
const READBACK: &str = "readback";

/// The timed passes a profiled step of `layers` layers, `norms` of them batch norms, `splits`
/// of them with an input gradient timed apart and `convs` of them convolutions, needs (see
/// `GpuTrainer::new`), or 0 if that many passes' timestamps do not fit wgpu's queries on
/// `gpu`: two per pass, and with encoder timestamps as many again for the copy runs. Where
/// the GPU uses MPS, a convolution's three passes are Metal work, after which the span's
/// next commands open a timed pass of their own: one more pass each.
fn profiled_passes(layers: usize, norms: usize, splits: usize, convs: usize, gpu: &Gpu) -> u32 {
    let metal = if gpu.uses_mps() { 3 * convs } else { 0 };
    let passes = (2 * layers + 6 + norms + splits + metal).max(64);
    let per_pass = if gpu.info().encoder_timestamps { 4 } else { 2 };
    if passes * per_pass <= wgpu::QUERY_SET_MAX_QUERIES as usize {
        passes as u32
    } else {
        0
    }
}

/// The profile of a step from its `output`, as [`FORWARD`] orders the spans. `split` says,
/// last layer first, which layers' input gradients are spans of their own
/// (`TrainPass::backward_inputs`).
fn profile_of(output: &StepOutput<'_>, split: &[Option<usize>], timing: Timing) -> GpuStepProfile {
    let time = |ns: Option<f64>| ns.map(|ns| Duration::from_secs_f64(ns * 1e-9));
    let spans = output.spans();
    let layers = split.len();
    let labels: Vec<&str> = spans.iter().map(|s| s.0).collect();
    let mut want = vec![FORWARD; layers];
    want.push(LOSS);
    // Each layer's backward span and, where it has one, its input gradient's.
    let mut at = vec![(0, None); layers];
    for (r, s) in split.iter().enumerate() {
        let layer = layers - 1 - r;
        at[layer].0 = want.len();
        want.push(BACKWARD);
        if s.is_some() {
            at[layer].1 = Some(want.len());
            want.push(BACKWARD_INPUT);
        }
    }
    let after = want.len();
    want.extend([HEALTH, OPTIMIZER, RUNNING, READBACK]);
    assert_eq!(
        labels, want,
        "a profiled step records its spans in this order"
    );
    let span = |k: usize| time(spans.get(k).and_then(|s| s.1));
    let kind = |k: usize, kind: TimedKind| {
        let mut entries = output
            .timed()
            .iter()
            .filter(|t| t.span == k && t.kind == kind);
        let first = entries.next()?;
        let rest: Option<f64> = entries.map(|t| t.nanoseconds).sum();
        time(first.nanoseconds.zip(rest).map(|(a, b)| a + b))
    };
    let copies = |k: usize| match timing {
        Timing::PassesAndCopies => kind(k, TimedKind::Copies),
        _ => None,
    };
    GpuStepProfile {
        forward: (0..layers).map(span).collect(),
        loss: span(layers),
        backward: (at.iter())
            .map(|&(own, input)| match input {
                Some(k) => span(own).zip(span(k)).map(|(a, b)| a + b),
                None => span(own),
            })
            .collect(),
        backward_input: at.iter().map(|&(_, input)| input.and_then(span)).collect(),
        health: span(after),
        optimizer: kind(after + 1, TimedKind::Pass),
        copies: copies(after + 1),
        running: span(after + 2),
        readback: copies(after + 3),
        total: Duration::ZERO,
        timing,
    }
}

impl std::fmt::Debug for GpuTrainer<'_> {
    /// Writes the model, the steps taken and skipped and the health settings.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTrainer")
            .field("model", &self.model)
            .field("loss", &self.loss)
            .field("steps", &self.taken)
            .field("skipped", &self.skipped)
            .field("health", self.health.health())
            .finish_non_exhaustive()
    }
}

impl<'s> GpuTrainer<'s> {
    /// A trainer for `model`, with its values in `params`, that minimises `loss` with
    /// `optimizer` on batches of `batch_size` rows on `gpu`.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for `RmsProp` and `Adagrad`, for a loss other than
    /// cross-entropy and MSE, for a parameter used by two layers, and as
    /// [`GpuModel::compile`] returns; [`Error::InvalidData`] as `compile` returns, if the
    /// model has no parameters or one that holds no values, and if a batch norm would
    /// normalise a training batch over one value per channel.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_gpu::Gpu;
    /// use cavitation_nn::{GpuTrainer, Linear, Loss, OptimizerConfig, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
    /// if let Some(gpu) = Gpu::new() {
    ///     let adam = OptimizerConfig::Adam { lr: 1e-3, weight_decay: 0.0 };
    ///     assert!(GpuTrainer::new(&net, &params, &adam, Loss::Mse, 16, &gpu).is_ok());
    ///     let rms = OptimizerConfig::RmsProp { lr: 1e-3, alpha: 0.99, momentum: 0.0, weight_decay: 0.0, centered: false };
    ///     assert!(GpuTrainer::new(&net, &params, &rms, Loss::Mse, 16, &gpu).is_err());
    /// }
    /// ```
    pub fn new(
        model: &Sequential,
        params: &ParamStore,
        optimizer: &OptimizerConfig,
        loss: Loss,
        batch_size: usize,
        gpu: &Gpu,
    ) -> Result<GpuTrainer<'s>, Error> {
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
                    "GpuTrainer: the GPU has no {other:?}; it runs SGD, Adam and AdamW"
                )));
            }
        };
        if !matches!(loss, Loss::CrossEntropy | Loss::Mse) {
            return Err(Error::Unsupported(format!(
                "GpuTrainer: the GPU has no Loss::{loss:?}; it runs CrossEntropy and Mse"
            )));
        }
        let mut gpu_model = GpuModel::compile(model, params, batch_size, gpu)?;
        if gpu_model.params.is_empty() {
            return Err(Error::InvalidData(
                "GpuTrainer: the model has no parameters to train".to_string(),
            ));
        }
        check_untied(&gpu_model, model)?;
        if let Some(t) = gpu_model.params.iter().find(|t| t.len == 0) {
            return Err(Error::InvalidData(format!(
                "GpuTrainer: parameter {} holds no values",
                t.id.position()
            )));
        }
        for (i, layer) in gpu_model.layers.iter().enumerate() {
            if let Kind::BatchNorm2d(norm) = layer.kind
                && batch_size * norm.spatial < 2
            {
                return Err(Error::InvalidData(format!(
                    "GpuTrainer: layer {i} ({}) normalises a training batch with its own \
                     statistics, which need more than one value per channel; a batch of \
                     {batch_size} gives {}",
                    model.layer(i).name(),
                    batch_size * norm.spatial
                )));
            }
        }
        let shader = |e: GpuError| Error::Unsupported(format!("GpuTrainer: {e}"));
        let m = &mut gpu_model;
        let mut kernels = TrainKernels {
            conv_input: ConvInputGrad::new(gpu, &mut m.cache).map_err(shader)?,
            conv_weight: ConvWeightGrad::new(gpu, &mut m.cache).map_err(shader)?,
            sums: ChannelSums::new(gpu, &mut m.cache).map_err(shader)?,
            cross_entropy: SoftmaxCrossEntropy::new(gpu, &mut m.cache).map_err(shader)?,
            mse: MeanSquaredError::new(gpu, &mut m.cache).map_err(shader)?,
            finite: FiniteFlag::new(gpu, &mut m.cache).map_err(shader)?,
            squares: SumSquares::new(gpu, &mut m.cache).map_err(shader)?,
            dead_relu: DeadRelu::new(gpu, &mut m.cache).map_err(shader)?,
        };
        // The ReLU layers whose outputs the dead-neuron check reads, with their values per
        // channel as the CPU check groups them.
        let relus: Vec<(usize, usize)> = (0..m.layers.len())
            .filter(|&i| matches!(m.layers[i].kind, Kind::Activation(Act::Relu | Act::Relu6)))
            .map(|i| (i, channel_size(model, i, m.layers[i].outputs).unwrap_or(1)))
            .collect();
        let relu_words: usize = relus
            .iter()
            .map(|&(i, spatial)| (m.layers[i].outputs / spatial).div_ceil(32))
            .sum();
        let lens: Vec<usize> = m.params.iter().map(|t| t.len).collect();
        let kernel =
            OptimizerKernel::new(gpu, &mut m.cache, &mut m.pool, rule, &lens).map_err(shader)?;

        // The step's encoder: the batch, the targets and the masks go up each step, with the
        // optimiser's and the running updates' constants; the loss and the flag come back,
        // and the whole of one flat buffer for `optimizer_state`.
        let batch = m.batch;
        let elements: usize = lens.iter().sum();
        let mask_words: usize = (0..m.layers.len())
            .filter(|&i| matches!(m.layers[i].kind, Kind::Dropout(_)))
            .map(|i| (batch * m.layers[i].outputs).div_ceil(32))
            .sum();
        let round = |n: usize| (n.div_ceil(8) * 8) as u64;
        let target_bytes = 4 * batch * m.output_width;
        let norms = m
            .layers
            .iter()
            .filter(|l| matches!(l.kind, Kind::BatchNorm2d(_)))
            .count();
        // The layers whose input gradient is a span of its own: the first layer has none.
        let splits = m
            .layers
            .iter()
            .skip(1)
            .filter(|l| matches!(l.kind, Kind::Linear { .. } | Kind::Conv2d { .. }))
            .count();
        let convs = m
            .layers
            .iter()
            .filter(|l| matches!(l.kind, Kind::Conv2d { .. }))
            .count();
        let sizes = StepSizes {
            upload_bytes: round(4 * batch * m.input_width)
                + round(target_bytes)
                + round(4 * mask_words)
                + kernel.upload_bytes()
                + 64 * norms as u64
                + 4096,
            // A profiled step's spans: two per layer, one more for each `Linear` and `Conv2d`
            // after the first layer (its input gradient) and five more. Each opens a pass, the
            // update two (its copies end the first), and each batch norm's running update one
            // more (its upload ends the pass before it). A model too deep for wgpu's timestamp
            // queries is not timed.
            spans: (u32_of(2 * m.layers.len() + splits) + 5).max(16),
            timed_passes: profiled_passes(m.layers.len(), norms, splits, convs, gpu),
            // The loss, the flag and the health values each step, or one flat buffer.
            readback_bytes: round(4 * elements)
                .max(16 + 2 * round(4 * lens.len()) + round(4 * relu_words)),
        };
        let mut steps = StepEncoder::new(gpu, sizes);
        let pass = TrainPass::build(m, &mut kernels, &mut steps, loss, &relus, batch_size)
            .map_err(shader)?;

        let flat_values = m.pool.take(4 * elements as u64, usage());
        let flat_grads = m.pool.take(4 * elements as u64, usage());
        let state: Vec<PooledBuffer> = (0..rule.state_buffers())
            .map(|_| m.pool.take(4 * elements as u64, usage()))
            .collect();
        let mut offsets = Vec::with_capacity(lens.len());
        let mut at = 0;
        for &n in &lens {
            offsets.push(at);
            at += n;
        }
        let state_refs: Vec<&PooledBuffer> = state.iter().collect();
        let group = kernel.bind_group(
            &mut steps,
            &flat_values,
            &flat_grads,
            &state_refs,
            &pass.flag,
        );
        // The flat values start as the parameters, and the state at 0.
        let mut step = steps.begin();
        for (t, &off) in m.params.iter().zip(&offsets) {
            step.copy(
                t.buffer.buffer(),
                0,
                flat_values.buffer(),
                4 * off as u64,
                4 * t.len as u64,
            );
        }
        for s in &state {
            step.clear(s.buffer(), 0, 4 * elements as u64);
        }
        let _ = step.finish();
        let flat = Flat {
            kernel,
            group,
            values: flat_values,
            grads: flat_grads,
            state,
            offsets,
            active: vec![true; lens.len()],
        };
        let next = pass
            .masks
            .iter()
            .map(|d| vec![0u32; d.len.div_ceil(32)])
            .collect();
        let names = model.parameter_names();
        let mut store_order: Vec<(ParamId, usize)> = gpu_model
            .params
            .iter()
            .enumerate()
            .map(|(k, t)| (t.id, k))
            .collect();
        store_order.sort_by_key(|(id, _)| id.position());
        let mut trainer = GpuTrainer {
            model: gpu_model,
            kernels,
            pass,
            flat,
            steps,
            loss,
            taken: 0,
            dropout_seed: 0,
            next,
            next_for: 0,
            upload: Vec::new(),
            health: HealthMonitor::new(Health::default()).with_loss(loss),
            names,
            store_order,
            stats: GradStats::new(),
            read: HealthRead::default(),
            skipped: 0,
            relus,
            short: Vec::new(),
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
    /// batch does: a batch of fewer rows runs a training step compiled for that many rows the
    /// first time one comes and kept, one per row count (one for a run whose every epoch ends
    /// with the same rows). It shares the parameters, the optimiser's state and batch norm's
    /// running statistics and batch counts with the full step, and has buffers of its own for
    /// every value, gradient and mask, sized for its rows.
    ///
    /// The update and the running statistics are skipped on the GPU when the loss or a
    /// gradient is NaN or infinite; the step is counted either way, and under
    /// [`OnNonFinite::Skip`] also in [`skipped_steps`](Self::skipped_steps), with its loss
    /// returned. The health checks run on every step (see [`GpuTrainer`]).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if the batch does not hold 1 to [`batch_size`](Self::batch_size)
    /// rows of the model's input width, if its targets do not suit the loss or the model's
    /// output, or if a batch norm would normalise its rows over one value per channel;
    /// [`Error::Unsupported`] if a step for a batch of fewer rows does not build on this GPU.
    /// Under [`OnNonFinite::Stop`], [`Error::NonFinite`] as the CPU trainer returns it, naming
    /// the step and `"loss"` (with the first NaN or infinite input or target value, or that
    /// the inputs were finite) or the first parameter in the store whose gradient was not
    /// finite; the parameters, the optimiser's state and the running statistics are then as
    /// they were.
    ///
    /// # Examples
    ///
    /// See [`GpuTrainer`].
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
                "GpuTrainer::step: the batch is {:?}, but the trainer takes [rows, {input}] \
                 with 1 to {full} rows",
                batch.x.shape().dims()
            )));
        }
        let short = if b < full {
            Some(self.prepare_short(model, b)?)
        } else {
            None
        };
        self.upload.clear();
        self.upload
            .extend(batch.x.data().iter().flat_map(|v| v.to_le_bytes()));
        let x_bytes = self.upload.len();
        match (self.loss, batch.targets) {
            (Loss::CrossEntropy, TargetsView::Classes(labels)) if labels.len() == b => {
                if let Some((row, &label)) = labels.iter().enumerate().find(|(_, l)| **l >= output)
                {
                    return Err(Error::InvalidData(format!(
                        "GpuTrainer::step: the label of row {row} is {label}, but the model has \
                         {output} outputs"
                    )));
                }
                self.upload
                    .extend(labels.iter().flat_map(|&l| (l as u32).to_le_bytes()));
            }
            (Loss::Mse, TargetsView::Values(t)) if t.shape().dims() == [b, output] => {
                self.upload
                    .extend(t.data().iter().flat_map(|v| v.to_le_bytes()));
            }
            (loss, _) => {
                return Err(Error::InvalidData(format!(
                    "GpuTrainer::step: the targets do not suit Loss::{loss:?} on {b} rows of \
                     {output} outputs"
                )));
            }
        }
        let step_number = self.taken + 1;
        let profiling = std::mem::take(&mut self.profile_asked);
        self.steps.set_profiling(profiling);
        let timing = self.steps.timing();
        let target_end = self.upload.len();
        if let Some(s) = short.map(|k| &mut self.short[k]) {
            // The short step's masks are drawn now, for its rows; the next full step's are
            // drawn when it comes.
            for (mask, words) in s.pass.masks.iter().zip(s.next.iter_mut()) {
                draw(self.dropout_seed, step_number, mask, words);
            }
            for words in &s.next {
                self.upload
                    .extend(words.iter().flat_map(|w| w.to_le_bytes()));
            }
        } else {
            if self.next_for != step_number {
                self.draw_next(step_number);
            }
            for words in &self.next {
                self.upload
                    .extend(words.iter().flat_map(|w| w.to_le_bytes()));
            }
        }
        let (pass, short_group) = match short.map(|k| &mut self.short[k]) {
            Some(s) => (&mut s.pass, Some(&s.group)),
            None => (&mut self.pass, None),
        };
        let mut step = self.steps.begin();
        step.upload(pass.values[0].buffer(), 0, &self.upload[..x_bytes]);
        step.upload(pass.targets.buffer(), 0, &self.upload[x_bytes..target_end]);
        let mut at = target_end;
        for mask in &pass.masks {
            let len = 4 * mask.len.div_ceil(32);
            step.upload(mask.bits.buffer(), 0, &self.upload[at..at + len]);
            at += len;
        }
        step.clear(pass.flag.buffer(), 0, 4);
        for &g in &pass.cleared {
            let b = &pass.grads[g];
            step.clear(b.buffer(), 0, b.len());
        }
        let kernels = &self.model.kernels;
        if profiling {
            let mut start = 0;
            for &end in &pass.forward_ends {
                step.span(FORWARD);
                record(
                    &pass.forward[start..end],
                    kernels,
                    &self.kernels,
                    &pass.pools,
                    &mut step,
                );
                start = end;
            }
            step.span(LOSS);
        } else {
            record(
                &pass.forward,
                kernels,
                &self.kernels,
                &pass.pools,
                &mut step,
            );
        }
        match &pass.loss {
            LossCall::CrossEntropy(call) => self.kernels.cross_entropy.record(&mut step, call),
            LossCall::Mse(call) => self.kernels.mse.record(&mut step, call),
        }
        if profiling {
            let mut start = 0;
            for (&end, &input) in pass.backward_ends.iter().zip(&pass.backward_inputs) {
                step.span(BACKWARD);
                if let Some(from) = input {
                    record(
                        &pass.backward[start..from],
                        kernels,
                        &self.kernels,
                        &pass.pools,
                        &mut step,
                    );
                    step.span(BACKWARD_INPUT);
                    start = from;
                }
                record(
                    &pass.backward[start..end],
                    kernels,
                    &self.kernels,
                    &pass.pools,
                    &mut step,
                );
                start = end;
            }
            step.span(HEALTH);
        } else {
            record(
                &pass.backward,
                kernels,
                &self.kernels,
                &pass.pools,
                &mut step,
            );
        }
        self.kernels.finite.record(&mut step, &pass.finite);
        for call in &pass.squares {
            self.kernels.squares.record(&mut step, call);
        }
        let dead_relu = self.health.health().dead_relu;
        if dead_relu {
            for relu in &pass.relus {
                self.kernels.dead_relu.record(&mut step, &relu.call);
            }
        }
        if profiling {
            step.span(OPTIMIZER);
        }
        let flat = &mut self.flat;
        for (g, &off) in pass.param_grads.iter().zip(&flat.offsets) {
            step.copy(g.buffer(), 0, flat.grads.buffer(), 4 * off as u64, g.len());
        }
        let group = short_group.unwrap_or(&flat.group);
        flat.kernel.record(&mut step, group, &flat.active);
        for (t, &off) in self.model.params.iter().zip(&flat.offsets) {
            step.copy(
                flat.values.buffer(),
                4 * off as u64,
                t.buffer.buffer(),
                0,
                4 * t.len as u64,
            );
        }
        if profiling {
            step.span(RUNNING);
        }
        for (_, update) in &pass.running {
            kernels.batch_norm.record_running(&mut step, update);
        }
        if profiling {
            step.span(READBACK);
        }
        let loss_at = step.read_back(pass.loss_value.buffer(), 0, 4);
        let flag_at = step.read_back(pass.flag.buffer(), 0, 4);
        let n = pass.squares.len() as u64;
        let squares_at = step.read_back(pass.square_sums.buffer(), 0, 4 * n);
        let counts_at = step.read_back(pass.finite.counts().buffer(), 0, 4 * n);
        let words = pass.relus.last().map_or(0, |r| {
            r.word + cavitation_gpu::kernels::DeadRelu::words(u32_of(r.channels)) as usize
        });
        let bits_at = (dead_relu && words > 0)
            .then(|| step.read_back(pass.relu_bits.buffer(), 0, 4 * words as u64));
        // The next full step's masks are drawn while this one runs.
        let (seed, masks, next) = (self.dropout_seed, &pass.masks, &mut self.next);
        let read = &mut self.read;
        let profile = &mut self.profile;
        let split = &pass.backward_inputs;
        let (value, flag) = thread::scope(|scope| {
            let drawing = (short.is_none() && !masks.is_empty()).then(|| {
                scope.spawn(move || {
                    for (mask, words) in masks.iter().zip(next.iter_mut()) {
                        draw(seed, step_number + 1, mask, words);
                    }
                })
            });
            let output = step.finish();
            *profile = profiling.then(|| profile_of(&output, split, timing));
            let bytes = output.bytes();
            let word = |at: usize| [bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]];
            let value = f32::from_le_bytes(word(loss_at));
            let flag = u32::from_le_bytes(word(flag_at));
            let floats = |at: usize, n: usize, out: &mut Vec<f32>| {
                out.clear();
                out.extend((0..n).map(|k| f32::from_le_bytes(word(at + 4 * k))));
            };
            floats(squares_at, n as usize, &mut read.squares);
            floats(counts_at, n as usize, &mut read.counts);
            read.bits.clear();
            if let Some(at) = bits_at {
                read.bits
                    .extend((0..words).map(|k| u32::from_le_bytes(word(at + 4 * k))));
            }
            if let Some(drawing) = drawing {
                drawing
                    .join()
                    .expect("drawing the dropout masks does not panic");
            }
            (value, flag)
        });
        if short.is_none() {
            self.next_for = step_number + 1;
        }
        let ran = flag == 0;
        self.flat.kernel.commit(ran);
        for (_, update) in &mut pass.running {
            update.commit(ran);
        }
        // The other passes' updates of the same layers take up the batch counts.
        if let Some(k) = short {
            take_counts(&self.short[k].pass.running, &mut self.pass.running);
        }
        for s in &mut self.short {
            take_counts(&self.pass.running, &mut s.pass.running);
        }
        self.taken = step_number;
        let inputs = StepInputs {
            x: batch.x,
            targets: batch.targets,
            rows,
        };
        let checked = self.check(model, step_number, value, inputs);
        // The GPU applied the update exactly when the flag was clear; the checks must agree:
        // a set flag is a stopped step (an error) or a skipped one (no update).
        assert!(
            match &checked {
                Ok(applied) => *applied == ran,
                Err(_) => !ran,
            },
            "GpuTrainer::step: the health checks disagree with the GPU's finite flag ({flag}) at \
             step {step_number}"
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

    /// Asks the next step for a [`GpuStepProfile`], read afterwards with
    /// [`last_profile`](Self::last_profile), as [`Trainer::profile_next_step`](crate::Trainer::profile_next_step)
    /// does on the CPU. Only that step is profiled; it runs each layer's forward and backward
    /// passes, the loss, the health values, the update, the running statistics and the
    /// readback in compute passes of their own, timed on the GPU, a `Linear`'s or a `Conv2d`'s
    /// input gradient apart from its parameter gradients. A step left out under
    /// [`OnNonFinite::Skip`] has no profile, and the next step is asked instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, TensorView};
    /// use cavitation_gpu::{Gpu, Timing};
    /// use cavitation_nn::{
    ///     BatchView, GpuTrainer, Health, Linear, Loss, OptimizerConfig, Relu, Sequential,
    ///     TargetsView,
    /// };
    ///
    /// let mut params = ParamStore::new();
    /// let mut rng = Rng::seed(0);
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 4, &mut rng))
    ///     .push(Relu)
    ///     .push(Linear::new(&mut params, 4, 1, &mut rng));
    /// if let Some(gpu) = Gpu::new() {
    ///     let sgd = OptimizerConfig::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 };
    ///     let mut trainer = GpuTrainer::new(&net, &params, &sgd, Loss::Mse, 2, &gpu).unwrap();
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
    ///     if !matches!(profile.timing, Timing::Untimed { .. }) {
    ///         assert!(profile.forward.iter().all(|t| t.is_some()));
    ///         // The last `Linear`'s input gradient is timed apart; the first has none.
    ///         let input: Vec<bool> = profile.backward_input.iter().map(|t| t.is_some()).collect();
    ///         assert_eq!(input, [false, false, true]);
    ///     }
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
    pub fn last_profile(&self) -> Option<&GpuStepProfile> {
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
            let (layers, relus, bits) = (&self.model.layers, &self.pass.relus, &self.read.bits);
            self.health.track_relus(
                model,
                layers.len(),
                |i| layers[i].outputs,
                |i, spatial, alive| {
                    let relu = relus
                        .iter()
                        .find(|r| r.layer == i)
                        .expect("every ReLU layer has its bits");
                    for (c, values) in alive.chunks_mut(spatial).enumerate() {
                        let w = bits[relu.word + c / 32];
                        if w >> (c % 32) & 1 == 1 {
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
            // The GPU sums in `f32`, the CPU in `f64`: a finite gradient whose sum overflows
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
    /// [`Error::Unsupported`] for [`OnNonFinite::Warn`]: the GPU never applies a step whose
    /// loss or gradient is NaN or infinite, so it cannot train on as `Warn` does;
    /// [`OnNonFinite::Skip`] leaves such a step out, and [`OnNonFinite::Stop`] stops at it.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_gpu::Gpu;
    /// use cavitation_nn::{GpuTrainer, Health, Linear, Loss, OnNonFinite, OptimizerConfig, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
    /// if let Some(gpu) = Gpu::new() {
    ///     let sgd = OptimizerConfig::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 };
    ///     let mut trainer = GpuTrainer::new(&net, &params, &sgd, Loss::Mse, 4, &gpu).unwrap();
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
                "GpuTrainer: OnNonFinite::Warn is not run on the GPU, which never applies a step \
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
    /// use cavitation_gpu::Gpu;
    /// use cavitation_nn::{
    ///     BatchView, GpuTrainer, Health, Linear, Loss, OnNonFinite, OptimizerConfig, Sequential,
    ///     TargetsView,
    /// };
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// if let Some(gpu) = Gpu::new() {
    ///     let sgd = OptimizerConfig::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 };
    ///     let mut trainer = GpuTrainer::new(&net, &params, &sgd, Loss::Mse, 2, &gpu).unwrap();
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
    /// (the store the trainer was made from, or one laid out like it), in one step with one
    /// readback.
    ///
    /// # Panics
    ///
    /// If an id of the model is not in `params`, or holds another number of values there.
    ///
    /// # Examples
    ///
    /// See [`GpuTrainer`].
    pub fn sync_to(&mut self, params: &mut ParamStore) {
        for &(layer, ref update) in &self.pass.running {
            if let (Some(count), Kind::BatchNorm2d(norm)) =
                (update.count(), self.model.layers[layer].kind)
                && let Some(c) = norm.running.and_then(|r| r.count)
            {
                self.model.counts[c].1 = count;
            }
        }
        self.model.download_to(params);
    }

    /// The optimiser's state: each parameter's update count and the moments or momentum, read
    /// back from the GPU one state buffer per step.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_gpu::Gpu;
    /// use cavitation_nn::{Batch, Batcher, Dataset, GpuTrainer, Linear, Loss, OptimizerConfig, Sequential, Targets};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let data = Dataset { x: Tensor::full(&[2, 2], 1.0), targets: Targets::Values(Tensor::full(&[2, 1], 3.0)) };
    /// if let Some(gpu) = Gpu::new() {
    ///     let adam = OptimizerConfig::Adam { lr: 1e-2, weight_decay: 0.0 };
    ///     let mut trainer = GpuTrainer::new(&net, &params, &adam, Loss::Mse, 2, &gpu).unwrap();
    ///     let mut batch = Batch::new(&data, 2);
    ///     trainer.step(&net, Batcher::new(2, 2, None).gather(&data, 0..2, &mut batch)).unwrap();
    ///     let state = trainer.optimizer_state();
    ///     assert_eq!(state.steps, [1, 1]);
    ///     assert_eq!(state.buffers.len(), 2); // Adam's two moments
    ///     assert_eq!(state.buffers[0][0].len(), 2); // the weight's first moments
    /// }
    /// ```
    pub fn optimizer_state(&mut self) -> GpuOptimizerState {
        let flat = &self.flat;
        let lens: Vec<usize> = self.model.params.iter().map(|t| t.len).collect();
        let mut buffers = Vec::with_capacity(flat.state.len());
        for s in &flat.state {
            let mut step = self.steps.begin();
            let at = step.read_back(s.buffer(), 0, s.len());
            let output = step.finish();
            let (words, _) = output.bytes()[at..at + s.len() as usize].as_chunks::<4>();
            let all: Vec<f32> = words.iter().map(|b| f32::from_le_bytes(*b)).collect();
            buffers.push(
                (flat.offsets.iter().zip(&lens))
                    .map(|(&off, &n)| all[off..off + n].to_vec())
                    .collect(),
            );
        }
        GpuOptimizerState {
            ids: self.model.params.iter().map(|t| t.id).collect(),
            steps: flat.kernel.steps().to_vec(),
            buffers,
        }
    }

    /// The model the trainer trains, whose evaluation pass shares the trained parameters:
    /// [`GpuModel::evaluate`] on it measures the parameters as they are now, without a
    /// [`sync_to`](Self::sync_to).
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_gpu::Gpu;
    /// use cavitation_nn::{GpuTrainer, Linear, Loss, OptimizerConfig, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
    /// if let Some(gpu) = Gpu::new() {
    ///     let sgd = OptimizerConfig::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 };
    ///     let mut trainer = GpuTrainer::new(&net, &params, &sgd, Loss::Mse, 8, &gpu).unwrap();
    ///     assert_eq!(trainer.model().batch(), 8);
    /// }
    /// ```
    pub fn model(&mut self) -> &mut GpuModel {
        &mut self.model
    }

    /// The rows a full step takes; a step takes up to this many.
    ///
    /// # Examples
    ///
    /// See [`GpuTrainer::model`].
    pub fn batch_size(&self) -> usize {
        self.model.batch
    }

    /// The number of steps taken so far, those whose update was skipped included.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_gpu::Gpu;
    /// use cavitation_nn::{GpuTrainer, Linear, Loss, OptimizerConfig, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
    /// if let Some(gpu) = Gpu::new() {
    ///     let sgd = OptimizerConfig::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 };
    ///     assert_eq!(GpuTrainer::new(&net, &params, &sgd, Loss::Mse, 8, &gpu).unwrap().steps(), 0);
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
    /// use cavitation_gpu::Gpu;
    /// use cavitation_nn::{GpuTrainer, Linear, Loss, OptimizerConfig, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
    /// if let Some(gpu) = Gpu::new() {
    ///     let sgd = OptimizerConfig::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 };
    ///     let mut trainer = GpuTrainer::new(&net, &params, &sgd, Loss::Mse, 8, &gpu).unwrap();
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
    /// If `lr` is not finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_gpu::Gpu;
    /// use cavitation_nn::{GpuTrainer, Linear, Loss, OptimizerConfig, Sequential};
    ///
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
    /// if let Some(gpu) = Gpu::new() {
    ///     let sgd = OptimizerConfig::Sgd { lr: 0.1, momentum: 0.0, weight_decay: 0.0 };
    ///     let mut trainer = GpuTrainer::new(&net, &params, &sgd, Loss::Mse, 8, &gpu).unwrap();
    ///     trainer.set_learning_rate(0.05);
    ///     assert_eq!(trainer.learning_rate(), 0.05);
    /// }
    /// ```
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.flat.kernel.set_learning_rate(lr);
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
        for (mask, words) in self.pass.masks.iter().zip(self.next.iter_mut()) {
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
    /// channel; [`Error::Unsupported`] if a kernel does not build.
    fn prepare_short(&mut self, model: &Sequential, rows: usize) -> Result<usize, Error> {
        if let Some(k) = self.short.iter().position(|s| s.rows == rows) {
            return Ok(k);
        }
        for (i, layer) in self.model.layers.iter().enumerate() {
            if let Kind::BatchNorm2d(norm) = layer.kind
                && rows * norm.spatial < 2
            {
                return Err(Error::InvalidData(format!(
                    "GpuTrainer::step: layer {i} ({}) normalises a training batch with its own \
                     statistics, which need more than one value per channel; a batch of \
                     {rows} gives {}",
                    model.layer(i).name(),
                    rows * norm.spatial
                )));
            }
        }
        let shader = |e: GpuError| Error::Unsupported(format!("GpuTrainer: {e}"));
        let mut pass = TrainPass::build(
            &mut self.model,
            &mut self.kernels,
            &mut self.steps,
            self.loss,
            &self.relus,
            rows,
        )
        .map_err(shader)?;
        take_counts(&self.pass.running, &mut pass.running);
        let flat = &self.flat;
        let state: Vec<&PooledBuffer> = flat.state.iter().collect();
        let group = flat.kernel.bind_group(
            &mut self.steps,
            &flat.values,
            &flat.grads,
            &state,
            &pass.flag,
        );
        let next = pass
            .masks
            .iter()
            .map(|d| vec![0u32; d.len.div_ceil(32)])
            .collect();
        self.short.push(ShortPass {
            rows,
            pass,
            group,
            next,
        });
        Ok(self.short.len() - 1)
    }

    /// The wall-clock seconds of `n` steps that only copy the gradients into the flat buffer
    /// and the values back out, as a training step does, and of `n` empty steps: for
    /// measuring the copies' share of a step.
    #[cfg(test)]
    pub(crate) fn time_copies(&mut self, n: usize) -> (f64, f64) {
        let start = std::time::Instant::now();
        for _ in 0..n {
            let mut step = self.steps.begin();
            for (g, &off) in self.pass.param_grads.iter().zip(&self.flat.offsets) {
                step.copy(
                    g.buffer(),
                    0,
                    self.flat.grads.buffer(),
                    4 * off as u64,
                    g.len(),
                );
            }
            for (t, &off) in self.model.params.iter().zip(&self.flat.offsets) {
                step.copy(
                    self.flat.values.buffer(),
                    4 * off as u64,
                    t.buffer.buffer(),
                    0,
                    4 * t.len as u64,
                );
            }
            let _ = step.finish();
        }
        let copies = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        for _ in 0..n {
            let _ = self.steps.begin().finish();
        }
        (copies, start.elapsed().as_secs_f64())
    }

    /// Reads back each max pooling's winners from the last step, in the model's order: for
    /// the tests.
    #[cfg(test)]
    pub(crate) fn last_winners(&mut self) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        for &w in &self.pass.winners {
            let b = &self.pass.scratch[w];
            let mut step = self.steps.begin();
            let at = step.read_back(b.buffer(), 0, b.len());
            let output = step.finish();
            let (words, _) = output.bytes()[at..at + b.len() as usize].as_chunks::<4>();
            out.push(words.iter().map(|w| u32::from_le_bytes(*w)).collect());
        }
        out
    }

    /// Reads back each parameter's gradient from the last step, in the model's order: for
    /// the tests.
    #[cfg(test)]
    pub(crate) fn last_gradients(&mut self) -> Vec<(ParamId, Vec<f32>)> {
        let mut out = Vec::new();
        for (t, g) in self.model.params.iter().zip(&self.pass.param_grads) {
            let mut step = self.steps.begin();
            let at = step.read_back(g.buffer(), 0, g.len());
            let output = step.finish();
            let (words, _) = output.bytes()[at..at + g.len() as usize].as_chunks::<4>();
            out.push((t.id, words.iter().map(|b| f32::from_le_bytes(*b)).collect()));
        }
        out
    }

    /// The GPU objects made so far, for the tests: the pool's buffers, the pipelines, and the
    /// bind groups of the model's step encoder and of the trainer's.
    #[cfg(test)]
    pub(crate) fn made(&self) -> [usize; 4] {
        [
            self.model.pool.created(),
            self.model.cache.pipelines_built(),
            self.model.steps.bind_groups_made(),
            self.steps.bind_groups_made(),
        ]
    }
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

/// Sets each running update in `to` to the batch count of the one for the same layer in
/// `from`: the two training passes update the same statistics.
fn take_counts(from: &[(usize, RunningUpdate)], to: &mut [(usize, RunningUpdate)]) {
    for ((_, a), (_, b)) in from.iter().zip(to.iter_mut()) {
        if let Some(count) = a.count() {
            b.set_count(count);
        }
    }
}

/// Refuses a model whose layers share a parameter: the kernels overwrite each gradient, so a
/// second use would replace the first's contribution.
fn check_untied(gpu: &GpuModel, model: &Sequential) -> Result<(), Error> {
    let mut uses = vec![0usize; gpu.params.len()];
    for layer in &gpu.layers {
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
        let id = gpu.params[k].id;
        let name = (model.parameter_names().into_iter())
            .find(|(_, i)| *i == id)
            .map_or_else(|| format!("parameter {}", id.position()), |(n, _)| n);
        return Err(Error::Unsupported(format!(
            "GpuTrainer: {name} is used by more than one layer, which the GPU does not train yet"
        )));
    }
    Ok(())
}

/// Records `ops` in `step`.
fn record(ops: &[Op], model: &Kernels, train: &TrainKernels, pools: &[Pool], step: &mut Step<'_>) {
    for op in ops {
        match op {
            Op::Gemm(call) => model.gemm.record(step, call),
            Op::Bias(call) => model.bias.record(step, call),
            Op::Conv(call) => model.conv.record(step, call),
            Op::ConvInput(call) => train.conv_input.record(step, call),
            Op::ConvWeight(call) => train.conv_weight.record(step, call),
            Op::Sums(call) => train.sums.record(step, call),
            Op::Elementwise(call) => model.elementwise.record(step, call),
            Op::Softmax(call) => model.softmax.record(step, call),
            Op::Prelu(call) => model.prelu.record(step, call),
            Op::PreluSlope(call) => model.prelu.record_slope_grad(step, call),
            Op::PoolForward(k, group) => match &pools[*k] {
                Pool::Max(p) => p.forward(step, group),
                Pool::Avg(p) => p.forward(step, group),
            },
            Op::PoolBackward(k, group) => match &pools[*k] {
                Pool::Max(p) => p.backward(step, group),
                Pool::Avg(p) => p.backward(step, group),
            },
            Op::NormForward(call) => model.batch_norm.record_forward(step, call),
            Op::NormBackward(call) => model.batch_norm.record_backward(step, call),
        }
    }
}

/// What the backward pass needs of one layer's forward pass.
#[derive(Debug)]
struct Saved {
    /// The layer's input, an index into the values.
    input: usize,
    /// Its output before its mask.
    pre: usize,
    /// Its pooling kernel's index and, for max pooling, its winners' index in the scratch.
    pool: Option<(usize, Option<usize>)>,
    /// Its batch norm's forward call (an index into the forward calls) and statistics (an
    /// index into the scratch).
    norm: Option<(usize, usize)>,
    /// Its dropout mask's index.
    mask: Option<usize>,
}

impl TrainPass {
    /// Prepares the training step of `m`: the forward pass in training mode, the loss, the
    /// backward pass, the finite flag, the health values and the running-statistics updates,
    /// for batches of `rows` rows (at most the model's batch). `relus` lists each ReLU layer
    /// with its values per channel.
    fn build(
        m: &mut GpuModel,
        train: &mut TrainKernels,
        steps: &mut StepEncoder,
        loss: Loss,
        relus: &[(usize, usize)],
        rows: usize,
    ) -> Result<TrainPass, GpuError> {
        let batch = rows;
        let k = &m.kernels;
        let cache: &mut PipelineCache = &mut m.cache;
        let pool: &mut BufferPool = &mut m.pool;
        let mut values = vec![pool.take(4 * (batch * m.input_width) as u64, usage())];
        let mut forward = Vec::new();
        let mut forward_ends = Vec::with_capacity(m.layers.len());
        let mut scratch: Vec<PooledBuffer> = Vec::new();
        let mut pools = Vec::new();
        let mut masks = Vec::new();
        let mut saved = Vec::with_capacity(m.layers.len());
        let mut x = 0;
        for layer in &m.layers {
            let n = batch * layer.outputs;
            let mut s = Saved {
                input: x,
                pre: x,
                pool: None,
                norm: None,
                mask: None,
            };
            let take = |values: &mut Vec<PooledBuffer>, pool: &mut BufferPool| {
                values.push(pool.take(4 * n as u64, usage()));
                values.len() - 1
            };
            let out = match layer.kind {
                Kind::Linear {
                    inputs,
                    outputs,
                    weight,
                    bias,
                } => {
                    let out = take(&mut values, pool);
                    let dims = GemmDims {
                        trans_b: true,
                        ..GemmDims::new(u32_of(batch), u32_of(outputs), u32_of(inputs))
                    };
                    let w = &m.params[weight].buffer;
                    forward.push(Op::Gemm(k.gemm.call(
                        steps,
                        pool,
                        dims,
                        &values[x],
                        w,
                        &values[out],
                    )));
                    if let Some(b) = bias {
                        let dims = ChannelDims::rows(u32_of(batch), u32_of(outputs));
                        forward.push(Op::Bias(k.bias.call(
                            steps,
                            pool,
                            dims,
                            &m.params[b].buffer,
                            &values[out],
                        )));
                    }
                    out
                }
                Kind::Conv2d { geom, weight, bias } => {
                    let out = take(&mut values, pool);
                    let dims = conv_dims(batch, &geom);
                    let w = &m.params[weight].buffer;
                    let b = bias.map(|b| &m.params[b].buffer);
                    forward.push(Op::Conv(k.conv.call(
                        steps,
                        pool,
                        dims,
                        &values[x],
                        w,
                        b,
                        &values[out],
                    )));
                    out
                }
                Kind::Activation(act) => {
                    let out = take(&mut values, pool);
                    let (alpha, scale) = act.settings();
                    forward.push(Op::Elementwise(k.elementwise.call(
                        steps,
                        pool,
                        forward_op(act),
                        u32_of(n),
                        alpha,
                        scale,
                        &values[x],
                        None,
                        &values[out],
                    )));
                    out
                }
                Kind::Softmax { log } => {
                    let out = take(&mut values, pool);
                    let op = if log {
                        RowSoftmax::LogSoftmax
                    } else {
                        RowSoftmax::Softmax
                    };
                    let shape = (u32_of(batch), u32_of(layer.outputs));
                    forward.push(Op::Softmax(k.softmax.call(
                        steps,
                        pool,
                        op,
                        shape,
                        &values[x],
                        None,
                        &values[out],
                    )));
                    out
                }
                Kind::Prelu { channels, weight } => {
                    let out = take(&mut values, pool);
                    forward.push(Op::Prelu(k.prelu.call(
                        steps,
                        pool,
                        Prelu::Forward,
                        prelu_shape(batch, layer.inputs, channels),
                        &values[x],
                        &m.params[weight].buffer,
                        None,
                        &values[out],
                    )));
                    out
                }
                Kind::MaxPool2d(geom) => {
                    let out = take(&mut values, pool);
                    let kernel = MaxPool2d::new(cache, &geom, batch)?;
                    scratch.push(pool.take(4 * n as u64, usage()));
                    let w = scratch.len() - 1;
                    let group =
                        kernel.forward_bind_group(steps, &values[x], &values[out], &scratch[w]);
                    pools.push(Pool::Max(kernel));
                    forward.push(Op::PoolForward(pools.len() - 1, group));
                    s.pool = Some((pools.len() - 1, Some(w)));
                    out
                }
                Kind::AvgPool2d {
                    geom,
                    count_include_pad,
                    divisor_override,
                } => {
                    let out = take(&mut values, pool);
                    let kernel =
                        AvgPool2d::new(cache, &geom, batch, count_include_pad, divisor_override)?;
                    let group = kernel.forward_bind_group(steps, &values[x], &values[out]);
                    pools.push(Pool::Avg(kernel));
                    forward.push(Op::PoolForward(pools.len() - 1, group));
                    s.pool = Some((pools.len() - 1, None));
                    out
                }
                Kind::BatchNorm2d(norm) => {
                    let out = take(&mut values, pool);
                    scratch
                        .push(pool.take(4 * 3 * norm.channels as u64, wgpu::BufferUsages::STORAGE));
                    let st = scratch.len() - 1;
                    let affine = match (norm.weight, norm.bias) {
                        (Some(w), Some(b)) => Some((&m.params[w].buffer, &m.params[b].buffer)),
                        _ => None,
                    };
                    let dims = ChannelDims::new(
                        u32_of(batch),
                        u32_of(norm.channels),
                        u32_of(norm.spatial),
                    );
                    let call = k.batch_norm.forward_call(
                        steps,
                        pool,
                        dims,
                        norm.eps,
                        BatchStats::Batch,
                        &values[x],
                        affine,
                        &values[out],
                        &scratch[st],
                    );
                    forward.push(Op::NormForward(Box::new(call)));
                    s.norm = Some((forward.len() - 1, st));
                    out
                }
                Kind::Dropout(p) => {
                    let out = take(&mut values, pool);
                    let words = n.div_ceil(32);
                    masks.push(DropoutMask {
                        stream: masks.len() as u64,
                        p,
                        len: n,
                        bits: pool.take(4 * words as u64, usage()),
                    });
                    let scale = 1.0f32 / (1.0 - p) as f32;
                    let mask = &masks.last().expect("just pushed").bits;
                    forward.push(Op::Elementwise(k.elementwise.call(
                        steps,
                        pool,
                        Elementwise::Dropout,
                        u32_of(n),
                        0.0,
                        scale,
                        &values[x],
                        Some(mask),
                        &values[out],
                    )));
                    s.mask = Some(masks.len() - 1);
                    out
                }
                Kind::Identity => x,
            };
            s.pre = out;
            x = match layer.mask {
                Some(mi) => {
                    let masked = take(&mut values, pool);
                    forward.push(Op::Elementwise(k.elementwise.call(
                        steps,
                        pool,
                        Elementwise::KeepWhere,
                        u32_of(n),
                        0.0,
                        0.0,
                        &values[out],
                        Some(&m.masks[mi].bits),
                        &values[masked],
                    )));
                    masked
                }
                None => out,
            };
            saved.push(s);
            forward_ends.push(forward.len());
        }

        // The loss and the gradient of the output.
        let mut grads: Vec<PooledBuffer> = Vec::new();
        let new_grad = |grads: &mut Vec<PooledBuffer>, pool: &mut BufferPool, n: usize| {
            grads.push(pool.take(4 * n as u64, usage()));
            grads.len() - 1
        };
        let out_width = m.output_width;
        let targets = pool.take(4 * (batch * out_width) as u64, usage());
        let loss_value = pool.take(4, usage());
        let flag = pool.take(4, usage());
        let mut g = new_grad(&mut grads, pool, batch * out_width);
        let loss = match loss {
            Loss::CrossEntropy => LossCall::CrossEntropy(train.cross_entropy.call(
                steps,
                pool,
                (u32_of(batch), u32_of(out_width)),
                &values[x],
                &targets,
                &grads[g],
                (&loss_value, 0),
            )),
            _ => LossCall::Mse(train.mse.call(
                steps,
                pool,
                u32_of(batch * out_width),
                &values[x],
                &targets,
                &grads[g],
                (&loss_value, 0),
            )),
        };

        // The backward pass, last layer first. A parameter's gradient is written once, by
        // its layer, into its own buffer.
        let param_grads: Vec<PooledBuffer> = (m.params.iter())
            .map(|t| pool.take(4 * t.len.max(1) as u64, usage()))
            .collect();
        let mut backward = Vec::new();
        let mut backward_ends = Vec::with_capacity(m.layers.len());
        let mut backward_inputs = Vec::with_capacity(m.layers.len());
        let mut cleared = Vec::new();
        for (i, layer) in m.layers.iter().enumerate().rev() {
            let s = &saved[i];
            let n_out = batch * layer.outputs;
            let n_in = batch * layer.inputs;
            if let Some(mi) = layer.mask {
                let pre = new_grad(&mut grads, pool, n_out);
                backward.push(Op::Elementwise(k.elementwise.call(
                    steps,
                    pool,
                    Elementwise::KeepWhere,
                    u32_of(n_out),
                    0.0,
                    0.0,
                    &grads[g],
                    Some(&m.masks[mi].bits),
                    &grads[pre],
                )));
                g = pre;
            }
            let first = i == 0;
            let x = &values[s.input];
            let mut input_at = None;
            g = match layer.kind {
                Kind::Linear {
                    inputs,
                    outputs,
                    weight,
                    bias,
                } => {
                    let dw = GemmDims {
                        trans_a: true,
                        ..GemmDims::new(u32_of(outputs), u32_of(inputs), u32_of(batch))
                    };
                    backward.push(Op::Gemm(k.gemm.call(
                        steps,
                        pool,
                        dw,
                        &grads[g],
                        x,
                        &param_grads[weight],
                    )));
                    if let Some(b) = bias {
                        let dims = ChannelDims::rows(u32_of(batch), u32_of(outputs));
                        backward.push(Op::Sums(train.sums.call(
                            steps,
                            pool,
                            dims,
                            &grads[g],
                            &param_grads[b],
                            false,
                        )));
                    }
                    if first {
                        g
                    } else {
                        let dx = new_grad(&mut grads, pool, n_in);
                        let dims = GemmDims::new(u32_of(batch), u32_of(inputs), u32_of(outputs));
                        input_at = Some(backward.len());
                        backward.push(Op::Gemm(k.gemm.call(
                            steps,
                            pool,
                            dims,
                            &grads[g],
                            &m.params[weight].buffer,
                            &grads[dx],
                        )));
                        dx
                    }
                }
                Kind::Conv2d { geom, weight, bias } => {
                    let dims = conv_dims(batch, &geom);
                    backward.push(Op::ConvWeight(train.conv_weight.call(
                        steps,
                        pool,
                        dims,
                        x,
                        &grads[g],
                        &param_grads[weight],
                    )));
                    if let Some(b) = bias {
                        let (oh, ow) = geom.out_size();
                        let cd = ChannelDims::new(
                            u32_of(batch),
                            u32_of(geom.out_channels()),
                            u32_of(oh * ow),
                        );
                        backward.push(Op::Sums(train.sums.call(
                            steps,
                            pool,
                            cd,
                            &grads[g],
                            &param_grads[b],
                            false,
                        )));
                    }
                    if first {
                        g
                    } else {
                        let dx = new_grad(&mut grads, pool, n_in);
                        input_at = Some(backward.len());
                        backward.push(Op::ConvInput(train.conv_input.call(
                            steps,
                            pool,
                            dims,
                            &m.params[weight].buffer,
                            &grads[g],
                            &grads[dx],
                        )));
                        dx
                    }
                }
                Kind::Activation(act) => {
                    let dx = new_grad(&mut grads, pool, n_in);
                    let (op, from_output) = backward_op(act);
                    let (alpha, scale) = act.settings();
                    let a = if from_output { &values[s.pre] } else { x };
                    backward.push(Op::Elementwise(k.elementwise.call(
                        steps,
                        pool,
                        op,
                        u32_of(n_in),
                        alpha,
                        scale,
                        a,
                        Some(&grads[g]),
                        &grads[dx],
                    )));
                    dx
                }
                Kind::Softmax { log } => {
                    let dx = new_grad(&mut grads, pool, n_in);
                    let op = if log {
                        RowSoftmax::LogSoftmaxBackward
                    } else {
                        RowSoftmax::SoftmaxBackward
                    };
                    let shape = (u32_of(batch), u32_of(layer.outputs));
                    backward.push(Op::Softmax(k.softmax.call(
                        steps,
                        pool,
                        op,
                        shape,
                        &values[s.pre],
                        Some(&grads[g]),
                        &grads[dx],
                    )));
                    dx
                }
                Kind::Prelu { channels, weight } => {
                    let shape = prelu_shape(batch, layer.inputs, channels);
                    backward.push(Op::PreluSlope(k.prelu.slope_grad_call(
                        steps,
                        pool,
                        shape,
                        x,
                        &grads[g],
                        &param_grads[weight],
                    )));
                    let dx = new_grad(&mut grads, pool, n_in);
                    backward.push(Op::Prelu(k.prelu.call(
                        steps,
                        pool,
                        Prelu::InputGrad,
                        shape,
                        x,
                        &m.params[weight].buffer,
                        Some(&grads[g]),
                        &grads[dx],
                    )));
                    dx
                }
                Kind::MaxPool2d(_) | Kind::AvgPool2d { .. } => {
                    let dx = new_grad(&mut grads, pool, n_in);
                    let (kp, winners) = s.pool.expect("a pooling layer saves its kernel");
                    let group = match (&pools[kp], winners) {
                        (Pool::Max(p), Some(w)) => {
                            p.backward_bind_group(steps, &grads[g], &scratch[w], &grads[dx])
                        }
                        (Pool::Avg(p), _) => p.backward_bind_group(steps, &grads[g], &grads[dx]),
                        (Pool::Max(_), None) => unreachable!("max pooling saves its winners"),
                    };
                    backward.push(Op::PoolBackward(kp, group));
                    // The pooling backward passes add to their input gradient.
                    cleared.push(dx);
                    dx
                }
                Kind::BatchNorm2d(norm) => {
                    let dx = new_grad(&mut grads, pool, n_in);
                    let (f, st) = s.norm.expect("a batch norm saves its forward call");
                    let Op::NormForward(fwd) = &forward[f] else {
                        unreachable!("the saved index is a batch norm's forward call")
                    };
                    let affine = match (norm.weight, norm.bias) {
                        (Some(w), Some(b)) => {
                            Some((&m.params[w].buffer, &param_grads[w], &param_grads[b]))
                        }
                        _ => None,
                    };
                    let call = k.batch_norm.backward_call(
                        steps,
                        pool,
                        fwd,
                        x,
                        &scratch[st],
                        &grads[g],
                        &grads[dx],
                        affine,
                    );
                    backward.push(Op::NormBackward(Box::new(call)));
                    dx
                }
                Kind::Dropout(p) => {
                    let dx = new_grad(&mut grads, pool, n_in);
                    let scale = 1.0f32 / (1.0 - p) as f32;
                    let mask = &masks[s.mask.expect("a dropout layer saves its mask")].bits;
                    backward.push(Op::Elementwise(k.elementwise.call(
                        steps,
                        pool,
                        Elementwise::Dropout,
                        u32_of(n_in),
                        0.0,
                        scale,
                        &grads[g],
                        Some(mask),
                        &grads[dx],
                    )));
                    dx
                }
                Kind::Identity => g,
            };
            backward_ends.push(backward.len());
            backward_inputs.push(input_at);
        }

        // The finite flag over the loss and every parameter's gradient, then the running
        // statistics' updates, which it gates.
        let grads_list: Vec<(&PooledBuffer, u32, u32)> = (param_grads.iter().zip(&m.params))
            .map(|(b, t)| (b, 0, u32_of(t.len)))
            .collect();
        let finite = train
            .finite
            .call(steps, pool, (&loss_value, 0), &grads_list, &flag);
        // The health values: each gradient's sum of squares, and each ReLU layer's bits.
        let square_sums = pool.take(4 * grads_list.len().max(1) as u64, usage());
        let squares = grads_list
            .iter()
            .enumerate()
            .map(|(k, &g)| train.squares.call(steps, pool, g, &square_sums, u32_of(k)))
            .collect();
        let words: usize = relus
            .iter()
            .map(|&(i, spatial)| DeadRelu::words(u32_of(m.layers[i].outputs / spatial)) as usize)
            .sum();
        let relu_bits = pool.take(4 * words.max(1) as u64, usage());
        let mut word = 0;
        let relus = relus
            .iter()
            .map(|&(i, spatial)| {
                let channels = m.layers[i].outputs / spatial;
                let dims = ChannelDims::new(u32_of(batch), u32_of(channels), u32_of(spatial));
                let y = &values[saved[i].pre];
                let call =
                    train
                        .dead_relu
                        .call(steps, pool, dims, y, &relu_bits, u32_of(word), false);
                let bits = ReluBits {
                    layer: i,
                    word,
                    channels,
                    call,
                };
                word += DeadRelu::words(u32_of(channels)) as usize;
                bits
            })
            .collect();
        let mut running = Vec::new();
        for (i, layer) in m.layers.iter().enumerate() {
            if let (Kind::BatchNorm2d(norm), Some((f, st))) = (layer.kind, saved[i].norm)
                && let Some(r) = norm.running
            {
                let Op::NormForward(fwd) = &forward[f] else {
                    unreachable!("the saved index is a batch norm's forward call")
                };
                let count = r.count.map(|c| m.counts[c].1);
                let update = k.batch_norm.running_call(
                    steps,
                    pool,
                    fwd,
                    &scratch[st],
                    (&m.buffers[r.mean].buffer, &m.buffers[r.var].buffer),
                    &flag,
                    r.momentum,
                    count,
                );
                running.push((i, update));
            }
        }
        // The GEMM's wide tile, for the calls of this pass that use it: its pipelines are
        // built only for a model, and a batch, that has such a call.
        for op in forward.iter().chain(&backward) {
            match op {
                Op::Gemm(call) => {
                    m.kernels.gemm.ensure_wide(&mut m.cache, call)?;
                }
                Op::Conv(call) => {
                    m.kernels.conv.ensure_wide(&mut m.cache, call)?;
                }
                Op::ConvInput(call) => {
                    train.conv_input.ensure_wide(&mut m.cache, call)?;
                }
                Op::ConvWeight(call) => {
                    train.conv_weight.ensure_wide(&mut m.cache, call)?;
                }
                _ => {}
            }
        }
        Ok(TrainPass {
            forward,
            forward_ends,
            backward,
            backward_ends,
            backward_inputs,
            loss,
            finite,
            running,
            pools,
            values,
            grads,
            param_grads,
            targets,
            loss_value,
            flag,
            winners: saved
                .iter()
                .filter_map(|s| s.pool.and_then(|(_, w)| w))
                .collect(),
            cleared,
            scratch,
            masks,
            squares,
            square_sums,
            relus,
            relu_bits,
        })
    }
}

/// The convolution of `geom` over `batch` images, as the kernels take it.
fn conv_dims(batch: usize, geom: &cavitation_core::Conv2dGeom) -> ConvDims {
    let pair = |(a, b): (usize, usize)| (u32_of(a), u32_of(b));
    ConvDims {
        stride: pair(geom.stride()),
        padding: pair(geom.padding()),
        dilation: pair(geom.dilation()),
        ..ConvDims::new(
            u32_of(batch),
            u32_of(geom.in_channels()),
            u32_of(geom.out_channels()),
            pair(geom.input()),
            pair(geom.kernel()),
        )
    }
}

/// An activation's forward operation.
pub(super) fn forward_op(act: Act) -> Elementwise {
    match act {
        Act::Relu => Elementwise::Relu,
        Act::LeakyRelu(_) => Elementwise::LeakyRelu,
        Act::Tanh => Elementwise::Tanh,
        Act::Sigmoid => Elementwise::Sigmoid,
        Act::GeluTanh => Elementwise::GeluTanh,
        Act::GeluErf => Elementwise::GeluErf,
        Act::Elu(_) => Elementwise::Elu,
        Act::Selu => Elementwise::Selu,
        Act::Celu(_) => Elementwise::Celu,
        Act::Silu => Elementwise::Silu,
        Act::Mish => Elementwise::Mish,
        Act::Softplus { .. } => Elementwise::Softplus,
        Act::Softsign => Elementwise::Softsign,
        Act::Hardtanh { .. } | Act::Relu6 => Elementwise::Hardtanh,
        Act::Hardsigmoid => Elementwise::Hardsigmoid,
        Act::Hardswish => Elementwise::Hardswish,
        Act::LogSigmoid => Elementwise::LogSigmoid,
    }
}

/// An activation's backward operation, and whether it reads the forward output (tanh and the
/// sigmoid) rather than the input.
fn backward_op(act: Act) -> (Elementwise, bool) {
    match act {
        Act::Relu => (Elementwise::ReluBackward, false),
        Act::LeakyRelu(_) => (Elementwise::LeakyReluBackward, false),
        Act::Tanh => (Elementwise::TanhBackward, true),
        Act::Sigmoid => (Elementwise::SigmoidBackward, true),
        Act::GeluTanh => (Elementwise::GeluTanhBackward, false),
        Act::GeluErf => (Elementwise::GeluErfBackward, false),
        Act::Elu(_) => (Elementwise::EluBackward, false),
        Act::Selu => (Elementwise::SeluBackward, false),
        Act::Celu(_) => (Elementwise::CeluBackward, false),
        Act::Silu => (Elementwise::SiluBackward, false),
        Act::Mish => (Elementwise::MishBackward, false),
        Act::Softplus { .. } => (Elementwise::SoftplusBackward, false),
        Act::Softsign => (Elementwise::SoftsignBackward, false),
        Act::Hardtanh { .. } | Act::Relu6 => (Elementwise::HardtanhBackward, false),
        Act::Hardsigmoid => (Elementwise::HardsigmoidBackward, false),
        Act::Hardswish => (Elementwise::HardswishBackward, false),
        Act::LogSigmoid => (Elementwise::LogSigmoidBackward, false),
    }
}
