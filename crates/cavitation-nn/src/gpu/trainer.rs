//! `GpuTrainer`: training a [`GpuModel`] on the GPU one batch at a time. A step is one
//! command encoder and one readback: the batch and its dropout masks go up, the training
//! forward pass, the loss, the backward pass, the finite flag, the optimiser's update and
//! batch norm's running statistics run, and the loss and the flag come back.

use std::thread;

use cavitation_core::{Error, ParamId, ParamStore, dropout_mask};
use cavitation_gpu::kernels::{
    AvgPool2d, BatchNormBackward, BatchNormForward, BatchStats, BiasAddCall, ChannelDims,
    ChannelSums, ChannelSumsCall, ConvCall, ConvDims, ConvInputGrad, ConvWeightGrad,
    ConvWeightGradCall, CrossEntropyCall, Elementwise, ElementwiseCall, FiniteFlag, FiniteFlagCall,
    GemmCall, GemmDims, MaxPool2d, MeanSquaredError, MseCall, OptimizerKernel, OptimizerRule,
    RunningUpdate, SoftmaxCrossEntropy,
};
use cavitation_gpu::{
    BufferPool, Gpu, GpuError, PipelineCache, PooledBuffer, Step, StepEncoder, StepSizes, wgpu,
};

use super::forward::Kernels;
use super::model::{Act, GpuModel, Kind};
use crate::{BatchView, Loss, OptimizerConfig, Sequential, TargetsView};

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
    /// The backward calls, in order.
    backward: Vec<Op>,
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
    /// The gradients of the values, with their parameters' gradients.
    #[allow(dead_code)] // Read only through the bind groups.
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
    /// Each max pooling's winners, an index into the scratch, in the model's order.
    #[allow(dead_code)] // The tests read the winners.
    winners: Vec<usize>,
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
/// While a step runs on the GPU, the next step's dropout masks are drawn on the host.
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
#[derive(Debug)]
pub struct GpuTrainer {
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
}

impl GpuTrainer {
    /// A trainer for `model`, with its values in `params`, that minimises `loss` with
    /// `optimizer` on batches of `batch_size` rows on `gpu`.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for `RmsProp` and `Adagrad`, for a loss other than
    /// cross-entropy and MSE, for a parameter used by two layers, and as
    /// [`GpuModel::compile`] returns; [`Error::InvalidData`] as `compile` returns, and if the
    /// model has no parameters.
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
    ) -> Result<GpuTrainer, Error> {
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
        let kernels = TrainKernels {
            conv_input: ConvInputGrad::new(gpu, &mut m.cache).map_err(shader)?,
            conv_weight: ConvWeightGrad::new(gpu, &mut m.cache).map_err(shader)?,
            sums: ChannelSums::new(gpu, &mut m.cache).map_err(shader)?,
            cross_entropy: SoftmaxCrossEntropy::new(gpu, &mut m.cache).map_err(shader)?,
            mse: MeanSquaredError::new(gpu, &mut m.cache).map_err(shader)?,
            finite: FiniteFlag::new(gpu, &mut m.cache).map_err(shader)?,
        };
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
        let sizes = StepSizes {
            upload_bytes: round(4 * batch * m.input_width)
                + round(target_bytes)
                + round(4 * mask_words)
                + kernel.upload_bytes()
                + 64 * norms as u64
                + 4096,
            readback_bytes: round(4 * elements).max(16),
            ..StepSizes::default()
        };
        let mut steps = StepEncoder::new(gpu, sizes);
        let pass = TrainPass::build(m, &kernels, &mut steps, loss).map_err(shader)?;

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
        };
        trainer.draw_next(1);
        Ok(trainer)
    }

    /// Trains on one batch, as [`Trainer::step`](crate::Trainer::step) does: the forward
    /// pass in training mode, the loss (its mean over the batch), the backward pass and the
    /// optimiser's update, then batch norm's running statistics. Returns the loss.
    ///
    /// The update and the running statistics are skipped on the GPU when the loss or a
    /// gradient is NaN or infinite; the step is counted either way.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if the batch does not hold [`batch_size`](Self::batch_size) rows
    /// of the model's input width, or its targets do not suit the loss or the model's output;
    /// [`Error::NonFinite`], naming the step, when the loss or a gradient was not finite (the
    /// parameters, the optimiser's state and the running statistics are then as they were).
    ///
    /// # Examples
    ///
    /// See [`GpuTrainer`].
    pub fn step(&mut self, model: &Sequential, batch: BatchView<'_, f32>) -> Result<f32, Error> {
        debug_assert_eq!(model.len(), self.model.layers.len());
        let (b, input, output) = (
            self.model.batch,
            self.model.input_width,
            self.model.output_width,
        );
        if batch.x.shape().dims() != [b, input] {
            return Err(Error::InvalidData(format!(
                "GpuTrainer::step: the batch is {:?}, but the trainer takes [{b}, {input}]",
                batch.x.shape().dims()
            )));
        }
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
        if self.next_for != step_number {
            self.draw_next(step_number);
        }
        let target_end = self.upload.len();
        for words in &self.next {
            self.upload
                .extend(words.iter().flat_map(|w| w.to_le_bytes()));
        }
        let mut step = self.steps.begin();
        step.upload(self.pass.values[0].buffer(), 0, &self.upload[..x_bytes]);
        step.upload(
            self.pass.targets.buffer(),
            0,
            &self.upload[x_bytes..target_end],
        );
        let mut at = target_end;
        for mask in &self.pass.masks {
            let len = 4 * mask.len.div_ceil(32);
            step.upload(mask.bits.buffer(), 0, &self.upload[at..at + len]);
            at += len;
        }
        step.clear(self.pass.flag.buffer(), 0, 4);
        let kernels = &self.model.kernels;
        record(
            &self.pass.forward,
            kernels,
            &self.kernels,
            &self.pass.pools,
            &mut step,
        );
        match &self.pass.loss {
            LossCall::CrossEntropy(call) => self.kernels.cross_entropy.record(&mut step, call),
            LossCall::Mse(call) => self.kernels.mse.record(&mut step, call),
        }
        record(
            &self.pass.backward,
            kernels,
            &self.kernels,
            &self.pass.pools,
            &mut step,
        );
        self.kernels.finite.record(&mut step, &self.pass.finite);
        let flat = &mut self.flat;
        for (g, &off) in self.pass.param_grads.iter().zip(&flat.offsets) {
            step.copy(g.buffer(), 0, flat.grads.buffer(), 4 * off as u64, g.len());
        }
        flat.kernel.record(&mut step, &flat.group, &flat.active);
        for (t, &off) in self.model.params.iter().zip(&flat.offsets) {
            step.copy(
                flat.values.buffer(),
                4 * off as u64,
                t.buffer.buffer(),
                0,
                4 * t.len as u64,
            );
        }
        for (_, update) in &self.pass.running {
            kernels.batch_norm.record_running(&mut step, update);
        }
        let loss_at = step.read_back(self.pass.loss_value.buffer(), 0, 4);
        let flag_at = step.read_back(self.pass.flag.buffer(), 0, 4);
        // The next step's masks are drawn while this one runs.
        let (seed, masks, next) = (self.dropout_seed, &self.pass.masks, &mut self.next);
        let (value, flag) = thread::scope(|scope| {
            let drawing = (!masks.is_empty()).then(|| {
                scope.spawn(move || {
                    for (mask, words) in masks.iter().zip(next.iter_mut()) {
                        draw(seed, step_number + 1, mask, words);
                    }
                })
            });
            let output = step.finish();
            let bytes = output.bytes();
            let word = |at: usize| [bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]];
            let value = f32::from_le_bytes(word(loss_at));
            let flag = u32::from_le_bytes(word(flag_at));
            if let Some(drawing) = drawing {
                drawing
                    .join()
                    .expect("drawing the dropout masks does not panic");
            }
            (value, flag)
        });
        self.next_for = step_number + 1;
        let ran = flag == 0;
        self.flat.kernel.commit(ran);
        for (_, update) in &mut self.pass.running {
            update.commit(ran);
        }
        self.taken = step_number;
        if ran {
            Ok(value)
        } else {
            Err(Error::NonFinite {
                step: step_number,
                location: "the loss or a gradient".to_string(),
                detail: None,
            })
        }
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

    /// The rows each step takes.
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
fn check_untied(gpu: &GpuModel, model: &Sequential) -> Result<(), Error> {
    let mut uses = vec![0usize; gpu.params.len()];
    for layer in &gpu.layers {
        let ids: Vec<usize> = match layer.kind {
            Kind::Linear { weight, bias, .. } | Kind::Conv2d { weight, bias, .. } => {
                std::iter::once(weight).chain(bias).collect()
            }
            Kind::BatchNorm2d(n) => n.weight.into_iter().chain(n.bias).collect(),
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
    /// backward pass, the finite flag and the running-statistics updates.
    fn build(
        m: &mut GpuModel,
        train: &TrainKernels,
        steps: &mut StepEncoder,
        loss: Loss,
    ) -> Result<TrainPass, GpuError> {
        let batch = m.batch;
        let k = &m.kernels;
        let cache: &mut PipelineCache = &mut m.cache;
        let pool: &mut BufferPool = &mut m.pool;
        let mut values = vec![pool.take(4 * (batch * m.input_width) as u64, usage())];
        let mut forward = Vec::new();
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
                    let (op, alpha) = forward_op(act);
                    forward.push(Op::Elementwise(k.elementwise.call(
                        steps,
                        pool,
                        op,
                        u32_of(n),
                        alpha,
                        0.0,
                        &values[x],
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
                    let (op, alpha, from_output) = backward_op(act);
                    let a = if from_output { &values[s.pre] } else { x };
                    backward.push(Op::Elementwise(k.elementwise.call(
                        steps,
                        pool,
                        op,
                        u32_of(n_in),
                        alpha,
                        0.0,
                        a,
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
        }

        // The finite flag over the loss and every parameter's gradient, then the running
        // statistics' updates, which it gates.
        let grads_list: Vec<(&PooledBuffer, u32, u32)> = (param_grads.iter().zip(&m.params))
            .map(|(b, t)| (b, 0, u32_of(t.len)))
            .collect();
        let finite = train
            .finite
            .call(steps, pool, (&loss_value, 0), &grads_list, &flag);
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
        Ok(TrainPass {
            forward,
            backward,
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
            scratch,
            masks,
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

/// An activation's forward operation and its `alpha`.
fn forward_op(act: Act) -> (Elementwise, f32) {
    match act {
        Act::Relu => (Elementwise::Relu, 0.0),
        Act::LeakyRelu(slope) => (Elementwise::LeakyRelu, slope as f32),
        Act::Tanh => (Elementwise::Tanh, 0.0),
        Act::Sigmoid => (Elementwise::Sigmoid, 0.0),
        Act::GeluTanh => (Elementwise::GeluTanh, 0.0),
        Act::GeluErf => (Elementwise::GeluErf, 0.0),
    }
}

/// An activation's backward operation, its `alpha`, and whether it reads the forward output
/// (tanh and the sigmoid) rather than the input.
fn backward_op(act: Act) -> (Elementwise, f32, bool) {
    match act {
        Act::Relu => (Elementwise::ReluBackward, 0.0, false),
        Act::LeakyRelu(slope) => (Elementwise::LeakyReluBackward, slope as f32, false),
        Act::Tanh => (Elementwise::TanhBackward, 0.0, true),
        Act::Sigmoid => (Elementwise::SigmoidBackward, 0.0, true),
        Act::GeluTanh => (Elementwise::GeluTanhBackward, 0.0, false),
        Act::GeluErf => (Elementwise::GeluErfBackward, 0.0, false),
    }
}
