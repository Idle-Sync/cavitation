//! The training step ([`Trainer`]), the losses it minimises, [`fit`] (with and without
//! hooks) and [`accuracy`].

use std::fmt;
use std::time::Instant;

use cavitation_core::{Error, Graph, ParamId, ParamStore, Reduction, Tape, TensorView, Var};

use super::{Batch, BatchView, Batcher, Dataset, Targets, TargetsView};
use crate::{
    Adam, AdamW, Health, HealthMonitor, Hooks, InferenceModel, Optimizer, Sequential, Sgd, Trace,
};

/// The loss [`Trainer`] and [`fit`] minimise, always averaged over the batch.
///
/// Every variant uses [`Reduction::Mean`] except [`KlDiv`](Loss::KlDiv), which uses
/// [`Reduction::BatchMean`] (PyTorch's recommended reduction for it). The class losses
/// ([`CrossEntropy`](Loss::CrossEntropy) and [`Nll`](Loss::Nll)) need
/// [`Targets::Classes`]; the others need [`Targets::Values`] shaped like the network's
/// output. A loss not listed here goes through the manual loop shown on [`Trainer`].
///
/// # Examples
///
/// ```
/// use cavitation_nn::Loss;
/// let loss = Loss::Huber { delta: 1.0 };
/// assert_ne!(loss, Loss::Mse);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Loss {
    /// Softmax cross-entropy on raw scores (logits), as [`Tape::softmax_cross_entropy`].
    CrossEntropy,
    /// Negative log-likelihood of log-probabilities (for example a
    /// [`LogSoftmax`](crate::LogSoftmax) output), as [`Tape::nll`].
    Nll,
    /// Mean squared error, as [`Tape::mse`].
    Mse,
    /// Mean absolute error, as [`Tape::l1`].
    L1,
    /// The Huber loss with threshold `delta`, as [`Tape::huber`].
    Huber {
        /// Where the loss turns from quadratic to linear.
        delta: f64,
    },
    /// The smooth L1 loss with threshold `beta`, as [`Tape::smooth_l1`].
    SmoothL1 {
        /// Where the loss turns from quadratic to linear.
        beta: f64,
    },
    /// Binary cross-entropy of probabilities (for example a [`Sigmoid`](crate::Sigmoid)
    /// output), as [`Tape::binary_cross_entropy`].
    BinaryCrossEntropy,
    /// Binary cross-entropy on raw scores, as [`Tape::binary_cross_entropy_with_logits`].
    BinaryCrossEntropyWithLogits,
    /// Kullback-Leibler divergence of a target distribution from log-probabilities, as
    /// [`Tape::kl_div`], divided by the batch size.
    KlDiv,
}

impl Loss {
    /// Whether the loss takes class labels rather than target values.
    fn needs_classes(self) -> bool {
        matches!(self, Loss::CrossEntropy | Loss::Nll)
    }

    /// Panics, naming the loss and the kind of targets, if `targets` do not suit the loss.
    fn check_targets(self, targets: TargetsView<'_, f32>) {
        match (self.needs_classes(), targets) {
            (true, TargetsView::Values(_)) => panic!(
                "Trainer::step: Loss::{self:?} needs class targets, but the batch has value targets"
            ),
            (false, TargetsView::Classes(_)) => panic!(
                "Trainer::step: Loss::{self:?} needs value targets, but the batch has class targets"
            ),
            _ => {}
        }
    }

    /// Records this loss of `out` against `targets` on `tape` and returns the loss variable.
    ///
    /// # Panics
    ///
    /// Panics as [`check_targets`](Loss::check_targets) does, and as the tape operation
    /// does on shapes it rejects.
    fn record(self, tape: &mut Tape<'_, f32>, out: Var, targets: TargetsView<'_, f32>) -> Var {
        self.check_targets(targets);
        let mean = Reduction::Mean;
        match targets {
            TargetsView::Classes(labels) => match self {
                Loss::Nll => tape.nll(out, labels, mean),
                _ => tape.softmax_cross_entropy(out, labels, mean),
            },
            TargetsView::Values(values) => {
                let t = tape.input(values);
                match self {
                    Loss::Mse => tape.mse(out, t, mean),
                    Loss::L1 => tape.l1(out, t, mean),
                    Loss::Huber { delta } => tape.huber(out, t, delta as f32, mean),
                    Loss::SmoothL1 { beta } => tape.smooth_l1(out, t, beta as f32, mean),
                    Loss::BinaryCrossEntropy => tape.binary_cross_entropy(out, t, mean),
                    Loss::BinaryCrossEntropyWithLogits => {
                        tape.binary_cross_entropy_with_logits(out, t, mean)
                    }
                    Loss::KlDiv => tape.kl_div(out, t, Reduction::BatchMean),
                    Loss::CrossEntropy | Loss::Nll => {
                        unreachable!("check_targets rejects value targets for class losses")
                    }
                }
            }
        }
    }
}

/// One training step at a time: the forward pass, the loss, the backward pass and the
/// optimiser step, reusing the same buffers every step.
///
/// A trainer owns everything a step needs besides the model and its parameters: the
/// [`Graph`] the steps record on, the [`Trace`] of each layer's input and output, the
/// optimiser (which keeps the gradient statistics of its last step, see
/// [`Optimizer::last_stats`]), the [`HealthMonitor`] that checks every step (the default
/// [`Health`] unless [`set_health`](Self::set_health) changes it) and the batch buffers
/// [`fit`] gathers into. After a warm-up epoch, including the smaller last batch, a healthy
/// step allocates nothing. Call [`end_epoch`](Self::end_epoch) after each epoch.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::{Batch, Batcher, Dataset, Linear, Loss, Sequential, Sgd, Targets, Trainer};
/// let data = Dataset {
///     x: Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], &[4, 1]).unwrap(),
///     targets: Targets::Values(Tensor::from_vec(vec![1.0f32, 3.0, 5.0, 7.0], &[4, 1]).unwrap()),
/// };
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 1, 1, &mut Rng::seed(0)));
/// let mut trainer = Trainer::new(&net, Sgd::new(0.05), Loss::Mse, 4);
/// let batcher = Batcher::new(4, 4, None);
/// let mut batch = Batch::new(&data, 4);
/// let first = trainer.step(&net, &mut params, batcher.gather(&data, 0..4, &mut batch));
/// let mut last = first;
/// for _ in 0..200 {
///     last = trainer.step(&net, &mut params, batcher.gather(&data, 0..4, &mut batch));
/// }
/// assert!(last < first * 0.01);
/// ```
///
/// The same step written out by hand, the way to train with a loss [`Loss`] does not list:
/// open a tape on the parameter values, add the batch as an input, run the model (traced,
/// for the health checks), record the loss, run `backward` into the gradients, run the
/// [`HealthMonitor`]'s tape checks, drop the tape, step the optimiser, run the monitor's
/// gradient checks on the optimiser's statistics and mark the gradients stale; at the end
/// of each epoch, end the monitor's epoch. The tape borrows `params.values` while
/// `backward` writes `params.grads`; borrowing the two fields separately is what lets this
/// compile.
///
/// ```
/// use cavitation_core::{Error, Graph, ParamStore, Reduction, Rng, Tensor};
/// use cavitation_nn::{Health, HealthMonitor, Linear, Optimizer, Sequential, Sgd, Trace};
/// let x = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], &[4, 1]).unwrap();
/// let y = Tensor::from_vec(vec![1.0f32, 3.0, 5.0, 7.0], &[4, 1]).unwrap();
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 1, 1, &mut Rng::seed(0)));
/// let names = net.parameter_names();
/// let mut optimizer = Sgd::new(0.05);
/// let mut monitor = HealthMonitor::new(Health::default());
/// let mut graph = Graph::new();
/// let mut trace = Trace::new();
/// let mut train = || -> Result<(), Error> {
///     for epoch in 1..=200 {
///         let step = epoch as u64; // one step per epoch here
///         let mut tape = graph.tape(&params.values);
///         let input = tape.input(x.view());
///         let target = tape.input(y.view());
///         let out = net.forward_traced(&mut tape, input, &mut trace);
///         let loss = tape.l1(out, target, Reduction::Sum);
///         let value = tape.value(loss).data()[0];
///         tape.backward(loss, &mut params.grads);
///         monitor.after_backward(step, value, &tape, &trace, &net)?;
///         drop(tape);
///         optimizer.step(&mut params);
///         monitor.after_update(step, optimizer.last_stats(), &names)?;
///         params.zero_grad();
///         monitor.end_epoch(epoch)?;
///     }
///     Ok(())
/// };
/// train().unwrap();
/// ```
pub struct Trainer {
    /// The storage every step records on.
    graph: Graph<f32>,
    /// Each layer's input and output in the last step.
    trace: Trace,
    /// The optimiser, with its state and the gradient statistics of its last step.
    optimizer: Box<dyn Optimizer<f32>>,
    /// The loss each step minimises.
    loss: Loss,
    /// The batch size `fit` gathers.
    batch_size: usize,
    /// The buffers `fit` gathers each batch into, made on its first epoch.
    batch: Option<Batch<f32>>,
    /// The number of steps taken.
    steps: u64,
    /// The health checks run on every step.
    health: HealthMonitor,
    /// The model's parameter names, for the health checks' messages.
    names: Vec<(String, ParamId)>,
    /// The first error a public step met under [`OnNonFinite::Stop`](crate::OnNonFinite)
    /// since the last [`end_epoch`](Trainer::end_epoch).
    stopped: Option<Error>,
}

impl Trainer {
    /// A trainer for `model` that minimises `loss` with `optimizer`, gathering batches of
    /// `batch` rows when it runs whole epochs for [`fit`].
    ///
    /// # Panics
    ///
    /// Panics if `batch` is 0.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Adam, Linear, Loss, Sequential, Trainer};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 4, 3, &mut Rng::seed(0)));
    /// let trainer = Trainer::new(&net, Adam::new(1e-3), Loss::CrossEntropy, 32);
    /// assert_eq!(trainer.steps(), 0);
    /// ```
    pub fn new(
        model: &Sequential,
        optimizer: impl Optimizer<f32> + 'static,
        loss: Loss,
        batch: usize,
    ) -> Trainer {
        Trainer::with_boxed(model, Box::new(optimizer), loss, batch)
    }

    /// [`Trainer::new`] for an optimiser that is already boxed.
    fn with_boxed(
        model: &Sequential,
        optimizer: Box<dyn Optimizer<f32>>,
        loss: Loss,
        batch: usize,
    ) -> Trainer {
        assert!(batch > 0, "Trainer::new: the batch size must be at least 1");
        Trainer {
            graph: Graph::new(),
            trace: Trace::with_capacity(model.len()),
            optimizer,
            loss,
            batch_size: batch,
            batch: None,
            steps: 0,
            health: HealthMonitor::new(Health::default()),
            names: model.parameter_names(),
            stopped: None,
        }
    }

    /// Replaces the health checks run on every step with `health`, starting afresh (nothing
    /// of the current epoch is carried over).
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Health, Linear, Loss, OnNonFinite, Sequential, Sgd, Trainer};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 8);
    /// trainer.set_health(Health { on_non_finite: OnNonFinite::Warn, ..Health::default() });
    /// assert!(trainer.end_epoch(1).is_ok());
    /// ```
    pub fn set_health(&mut self, health: Health) {
        self.health = HealthMonitor::new(health);
    }

    /// Ends epoch `epoch` of the health checks ([`HealthMonitor::end_epoch`]: the dead-neuron
    /// report, and a new epoch for the once-per-epoch warnings). Call it after the last step
    /// of each epoch; [`fit`] does.
    ///
    /// # Errors
    ///
    /// Returns the first [`Error::NonFinite`] that [`step`](Self::step) or
    /// [`step_with_hooks`](Self::step_with_hooks) met under
    /// [`OnNonFinite::Stop`](crate::OnNonFinite) since the last `end_epoch`, once.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_nn::{Batch, Batcher, Dataset, Linear, Loss, Sequential, Sgd, Targets, Trainer};
    /// let data = Dataset {
    ///     x: Tensor::full(&[2, 1], f32::NAN),
    ///     targets: Targets::Values(Tensor::zeros(&[2, 1])),
    /// };
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 1, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 2);
    /// let mut batch = Batch::new(&data, 2);
    /// let loss = trainer.step(&net, &mut params, Batcher::new(2, 2, None).gather(&data, 0..2, &mut batch));
    /// assert!(loss.is_nan());
    /// let err = trainer.end_epoch(1).unwrap_err();
    /// assert_eq!(err.to_string(), "non-finite value at step 1 in loss");
    /// ```
    pub fn end_epoch(&mut self, epoch: usize) -> Result<(), Error> {
        let stopped = self.stopped.take();
        self.health.end_epoch(epoch)?;
        stopped.map_or(Ok(()), Err)
    }

    /// Trains on one batch: the forward pass of `model` on `batch.x`, the loss against
    /// `batch.targets`, the backward pass into `params.grads`, the optimiser step and
    /// [`ParamStore::zero_grad`]. Returns the batch's loss (its mean over the batch).
    ///
    /// The gradients must be stale when it starts, as the previous step leaves them.
    ///
    /// Every step runs the health checks (see [`HealthMonitor`]). Under
    /// [`OnNonFinite::Stop`](crate::OnNonFinite) (the default), a NaN or infinite loss skips
    /// the optimiser step, so the parameters stay as they were, and the step returns the
    /// loss as it is; a NaN or infinite gradient is found after the update. Either way the
    /// first such error is kept and returned by [`end_epoch`](Self::end_epoch). [`fit`]
    /// stops at it instead.
    ///
    /// # Panics
    ///
    /// Panics, naming the loss and the kind of targets, if the targets do not suit the loss
    /// (see [`Loss`]), before any work is done. Also panics as [`Sequential::forward`] and
    /// the loss's tape operation do when the shapes do not fit, and as
    /// [`Optimizer::step`] does if the parameters change in number or size between steps.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_nn::{Batch, Batcher, Dataset, Linear, Loss, Sequential, Sgd, Targets, Trainer};
    /// let data = Dataset { x: Tensor::zeros(&[3, 2]), targets: Targets::Classes(vec![0, 1, 1]) };
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 2, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::CrossEntropy, 3);
    /// let mut batch = Batch::new(&data, 3);
    /// let loss = trainer.step(&net, &mut params, Batcher::new(3, 3, None).gather(&data, 0..3, &mut batch));
    /// // All-zero inputs and a zero bias give equal scores: the loss is ln 2.
    /// assert!((loss - 2f32.ln()).abs() < 1e-6);
    /// ```
    pub fn step(
        &mut self,
        model: &Sequential,
        params: &mut ParamStore,
        batch: BatchView<'_, f32>,
    ) -> f32 {
        self.step_with_hooks(model, params, batch, &mut Hooks::new())
    }

    /// Like [`step`](Self::step), and runs `hooks`: the forward hooks right after the forward
    /// pass, the backward hooks right after the backward pass, both before the optimiser
    /// updates the parameters (so every event shows the values and gradients of this step).
    /// With no hooks registered it does exactly what `step` does.
    ///
    /// # Panics
    ///
    /// Panics as [`step`](Self::step) does, and as [`Hooks::run_forward`] does when a hook is
    /// registered on a layer `model` does not have (before the forward pass, so nothing runs
    /// and no parameter changes).
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_nn::{
    ///     Batch, Batcher, Dataset, Hooks, Linear, Loss, Sequential, Sgd, Targets, Tanh, Trainer,
    /// };
    /// let data = Dataset { x: Tensor::full(&[2, 2], 0.5f32), targets: Targets::Classes(vec![0, 1]) };
    /// let mut params = ParamStore::new();
    /// let mut rng = Rng::seed(0);
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 3, &mut rng))
    ///     .push(Tanh)
    ///     .push(Linear::new(&mut params, 3, 2, &mut rng));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::CrossEntropy, 2);
    /// let mut batch = Batch::new(&data, 2);
    /// let mut widths = Vec::new();
    /// let mut hooks = Hooks::new();
    /// hooks.on_forward(1, |e| widths.push(e.output.shape().dims().to_vec()));
    /// let view = Batcher::new(2, 2, None).gather(&data, 0..2, &mut batch);
    /// trainer.step_with_hooks(&net, &mut params, view, &mut hooks);
    /// drop(hooks);
    /// assert_eq!(widths, [vec![2, 3]]);
    /// ```
    pub fn step_with_hooks(
        &mut self,
        model: &Sequential,
        params: &mut ParamStore,
        batch: BatchView<'_, f32>,
        hooks: &mut Hooks<'_>,
    ) -> f32 {
        let (value, checked) = self.checked_step(model, params, batch, hooks);
        if let Err(err) = checked
            && self.stopped.is_none()
        {
            self.stopped = Some(err);
        }
        value
    }

    /// The step behind [`step_with_hooks`](Self::step_with_hooks): returns the batch's loss
    /// and the health checks' verdict. When the tape checks fail the optimiser step is
    /// skipped; the gradients are marked stale and the step is counted either way.
    fn checked_step(
        &mut self,
        model: &Sequential,
        params: &mut ParamStore,
        batch: BatchView<'_, f32>,
        hooks: &mut Hooks<'_>,
    ) -> (f32, Result<(), Error>) {
        self.loss.check_targets(batch.targets);
        hooks.check_layers(model);
        let step = self.steps + 1;
        let (value, mut checked) = {
            let mut tape = self.graph.tape(&params.values);
            let x = tape.input(batch.x);
            let out = model.forward_traced(&mut tape, x, &mut self.trace);
            hooks.run_forward(&tape, &self.trace, model);
            let loss = self.loss.record(&mut tape, out, batch.targets);
            let value = tape.value(loss).data()[0];
            tape.backward(loss, &mut params.grads);
            hooks.run_backward(&tape, &self.trace, model);
            let checked = self
                .health
                .after_backward(step, value, &tape, &self.trace, model);
            (value, checked)
        };
        if checked.is_ok() {
            self.optimizer.step(params);
            checked = self
                .health
                .after_update(step, self.optimizer.last_stats(), &self.names);
        }
        params.zero_grad();
        self.steps = step;
        (value, checked)
    }

    /// The number of steps taken so far.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, Loss, Sequential, Sgd, Trainer};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// assert_eq!(Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 8).steps(), 0);
    /// ```
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Runs one epoch over `data` in the batches of `batcher` (reshuffled first), running
    /// `hooks` and the health checks on every step, and returns the mean loss over the
    /// epoch's rows: each batch's loss weighted by its row count.
    ///
    /// # Errors
    ///
    /// Returns the first error of the health checks, at the step that met it.
    fn run_epoch(
        &mut self,
        model: &Sequential,
        params: &mut ParamStore,
        data: &Dataset,
        batcher: &mut Batcher,
        hooks: &mut Hooks<'_>,
    ) -> Result<f64, Error> {
        let mut batch = self
            .batch
            .take()
            .unwrap_or_else(|| Batch::new(data, self.batch_size));
        batcher.epoch();
        let mut total = 0.0f64;
        let mut checked = Ok(());
        for range in batcher.batches() {
            let rows = range.len() as f64;
            let view = batcher.gather(data, range, &mut batch);
            let (loss, step_checked) = self.checked_step(model, params, view, hooks);
            total += f64::from(loss) * rows;
            if step_checked.is_err() {
                checked = step_checked;
                break;
            }
        }
        self.batch = Some(batch);
        checked.map(|()| total / data.len() as f64)
    }
}

/// The optimiser [`fit`] builds, with its settings.
///
/// # Examples
///
/// ```
/// use cavitation_nn::OptimizerConfig;
/// let opt = OptimizerConfig::Adam { lr: 1e-3, weight_decay: 0.0 }.build();
/// assert_eq!(opt.learning_rate(), 1e-3);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizerConfig {
    /// [`Sgd`] with momentum and weight decay (0 turns either off).
    Sgd {
        /// The learning rate.
        lr: f64,
        /// The momentum factor.
        momentum: f64,
        /// The weight decay added to the gradient.
        weight_decay: f64,
    },
    /// [`Adam`], with weight decay added to the gradient.
    Adam {
        /// The learning rate.
        lr: f64,
        /// The weight decay added to the gradient.
        weight_decay: f64,
    },
    /// [`AdamW`], with decoupled weight decay.
    AdamW {
        /// The learning rate.
        lr: f64,
        /// The decoupled weight decay.
        weight_decay: f64,
    },
}

impl OptimizerConfig {
    /// Builds the optimiser, with the default betas and epsilon for Adam and AdamW.
    ///
    /// ```
    /// use cavitation_nn::OptimizerConfig;
    /// let sgd = OptimizerConfig::Sgd { lr: 0.1, momentum: 0.9, weight_decay: 0.0 }.build();
    /// assert_eq!(sgd.learning_rate(), 0.1);
    /// ```
    pub fn build(&self) -> Box<dyn Optimizer<f32>> {
        match *self {
            OptimizerConfig::Sgd {
                lr,
                momentum,
                weight_decay,
            } => Box::new(
                Sgd::new(lr)
                    .with_momentum(momentum)
                    .with_weight_decay(weight_decay),
            ),
            OptimizerConfig::Adam { lr, weight_decay } => {
                Box::new(Adam::new(lr).with_weight_decay(weight_decay))
            }
            OptimizerConfig::AdamW { lr, weight_decay } => {
                Box::new(AdamW::new(lr).with_weight_decay(weight_decay))
            }
        }
    }
}

/// What one epoch of [`fit`] did.
///
/// # Examples
///
/// ```
/// let report = cavitation_nn::EpochReport { epoch: 1, mean_loss: 0.5, seconds: 0.01 };
/// assert_eq!(report.epoch, 1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpochReport {
    /// The epoch's number, counting from 1.
    pub epoch: usize,
    /// The mean loss over the epoch's rows: each batch's loss weighted by its row count.
    pub mean_loss: f64,
    /// The epoch's wall-clock time in seconds.
    pub seconds: f64,
}

/// Every epoch's report from one [`fit`].
///
/// # Examples
///
/// ```
/// use cavitation_nn::{EpochReport, History};
/// let history = History { epochs: vec![EpochReport { epoch: 1, mean_loss: 0.5, seconds: 0.01 }] };
/// assert_eq!(history.epochs.len(), 1);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct History {
    /// The reports, in epoch order.
    pub epochs: Vec<EpochReport>,
}

/// How [`fit`] trains. Taken by value, because it holds the `on_epoch` callback.
///
/// The default is 10 epochs in batches of 32, Adam at learning rate 1e-3 without weight
/// decay, cross-entropy, shuffled with seed 0, the default [`Health`] checks, and no
/// callback.
///
/// # Examples
///
/// ```
/// use cavitation_nn::{FitConfig, Loss};
/// let config = FitConfig { epochs: 3, loss: Loss::Mse, ..FitConfig::default() };
/// assert_eq!((config.epochs, config.batch), (3, 32));
/// ```
pub struct FitConfig {
    /// How many passes over the data.
    pub epochs: usize,
    /// The rows per step; the last batch of an epoch may be smaller.
    pub batch: usize,
    /// The optimiser to build.
    pub optimizer: OptimizerConfig,
    /// The loss to minimise.
    pub loss: Loss,
    /// The seed that shuffles the rows each epoch; `None` keeps the data's order.
    pub shuffle_seed: Option<u64>,
    /// The health checks run on every step and at the end of every epoch.
    pub health: Health,
    /// Called with each epoch's report as soon as the epoch ends.
    // The boxed callback is the plainest type for this field; an alias would only hide it.
    #[allow(clippy::type_complexity)]
    pub on_epoch: Option<Box<dyn FnMut(&EpochReport) + 'static>>,
}

impl Default for FitConfig {
    /// 10 epochs, batch 32, Adam at 1e-3, cross-entropy, shuffle seed 0, the default health
    /// checks, no callback.
    fn default() -> FitConfig {
        FitConfig {
            epochs: 10,
            batch: 32,
            optimizer: OptimizerConfig::Adam {
                lr: 1e-3,
                weight_decay: 0.0,
            },
            loss: Loss::CrossEntropy,
            shuffle_seed: Some(0),
            health: Health::default(),
            on_epoch: None,
        }
    }
}

impl fmt::Debug for FitConfig {
    /// Writes every setting; the callback shows only whether there is one.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FitConfig")
            .field("epochs", &self.epochs)
            .field("batch", &self.batch)
            .field("optimizer", &self.optimizer)
            .field("loss", &self.loss)
            .field("shuffle_seed", &self.shuffle_seed)
            .field("health", &self.health)
            .field("on_epoch", &self.on_epoch.as_ref().map(|_| "FnMut"))
            .finish()
    }
}

/// Checks that `data` can be trained on: `x` a matrix with at least one row, and one target
/// per row.
fn check_dataset(data: &Dataset) -> Result<(), Error> {
    let &[rows, _] = data.x.shape().dims() else {
        return Err(Error::Format(format!(
            "the inputs must have shape [rows, features], found {:?}",
            data.x.shape().dims()
        )));
    };
    if rows == 0 {
        return Err(Error::Format("the data set has no rows".to_string()));
    }
    match &data.targets {
        Targets::Classes(labels) if labels.len() != rows => Err(Error::ShapeMismatch {
            expected: vec![rows],
            found: vec![labels.len()],
        }),
        Targets::Values(t) => match t.shape().dims() {
            &[r, _] if r == rows => Ok(()),
            &[_, outputs] => Err(Error::ShapeMismatch {
                expected: vec![rows, outputs],
                found: t.shape().dims().to_vec(),
            }),
            dims => Err(Error::Format(format!(
                "value targets must have shape [rows, outputs], found {dims:?}"
            ))),
        },
        Targets::Classes(_) => Ok(()),
    }
}

/// Trains `model` on `train` as `config` says and returns each epoch's report.
///
/// Each epoch reshuffles the rows (with `config.shuffle_seed`), takes one [`Trainer::step`]
/// per batch, ends the epoch of the health checks ([`Trainer::end_epoch`]) and then calls
/// `config.on_epoch`. The same seeds give bit-identical parameters on every run on the same
/// machine and kernel level, for any thread count.
///
/// # Errors
///
/// Returns, before training, [`Error::Format`] if `train.x` is not a matrix or has no rows
/// (or value targets are not a matrix), and [`Error::ShapeMismatch`] if the targets do not
/// have one entry per row. During training, with `config.health.on_non_finite` set to
/// [`OnNonFinite::Stop`](crate::OnNonFinite) (the default), returns [`Error::NonFinite`] at
/// the first step whose loss or gradient is NaN or infinite, naming the step and `"loss"` or
/// the parameter (for example `"0.weight"`). A non-finite loss stops before the optimiser
/// step; a non-finite gradient with a finite loss is found after the update applied it.
///
/// # Panics
///
/// Panics if `config.batch` is 0, and as [`Trainer::step`] does: when the targets do not
/// suit the loss or the shapes do not fit the model.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::{Dataset, FitConfig, Linear, Loss, OptimizerConfig, Sequential, Targets, fit};
/// let data = Dataset {
///     x: Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], &[4, 1]).unwrap(),
///     targets: Targets::Values(Tensor::from_vec(vec![1.0f32, 3.0, 5.0, 7.0], &[4, 1]).unwrap()),
/// };
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 1, 1, &mut Rng::seed(0)));
/// let config = FitConfig {
///     epochs: 100,
///     batch: 2,
///     optimizer: OptimizerConfig::Sgd { lr: 0.05, momentum: 0.0, weight_decay: 0.0 },
///     loss: Loss::Mse,
///     ..FitConfig::default()
/// };
/// let history = fit(&net, &mut params, &data, config).unwrap();
/// assert!(history.epochs[99].mean_loss < history.epochs[0].mean_loss);
/// ```
pub fn fit(
    model: &Sequential,
    params: &mut ParamStore,
    train: &Dataset,
    config: FitConfig,
) -> Result<History, Error> {
    fit_with_hooks(model, params, train, config, &mut Hooks::new())
}

/// Like [`fit`], and runs `hooks` on every step as [`Trainer::step_with_hooks`] does. Hooks
/// only read, so the same seeds train to the same parameters as `fit`.
///
/// # Errors
///
/// Returns the same errors as [`fit`].
///
/// # Panics
///
/// Panics as [`fit`] does, and on the first step if a hook is registered on a layer `model`
/// does not have.
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::{Dataset, FitConfig, Hooks, Linear, Loss, Sequential, Targets, fit_with_hooks};
/// let data = Dataset {
///     x: Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], &[4, 1]).unwrap(),
///     targets: Targets::Values(Tensor::from_vec(vec![1.0f32, 3.0, 5.0, 7.0], &[4, 1]).unwrap()),
/// };
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 1, 1, &mut Rng::seed(0)));
/// let mut steps = 0;
/// let mut hooks = Hooks::new();
/// hooks.on_backward(0, |_| steps += 1);
/// let config = FitConfig { epochs: 5, batch: 2, loss: Loss::Mse, ..FitConfig::default() };
/// fit_with_hooks(&net, &mut params, &data, config, &mut hooks).unwrap();
/// drop(hooks);
/// assert_eq!(steps, 10);
/// ```
pub fn fit_with_hooks(
    model: &Sequential,
    params: &mut ParamStore,
    train: &Dataset,
    mut config: FitConfig,
    hooks: &mut Hooks<'_>,
) -> Result<History, Error> {
    check_dataset(train)?;
    let mut trainer =
        Trainer::with_boxed(model, config.optimizer.build(), config.loss, config.batch);
    trainer.set_health(config.health);
    let mut batcher = Batcher::new(train.len(), config.batch, config.shuffle_seed);
    let mut history = History {
        epochs: Vec::with_capacity(config.epochs),
    };
    params.zero_grad();
    for epoch in 1..=config.epochs {
        let start = Instant::now();
        let mean_loss = trainer.run_epoch(model, params, train, &mut batcher, hooks)?;
        trainer.end_epoch(epoch)?;
        let report = EpochReport {
            epoch,
            mean_loss,
            seconds: start.elapsed().as_secs_f64(),
        };
        if let Some(on_epoch) = config.on_epoch.as_mut() {
            on_epoch(&report);
        }
        history.epochs.push(report);
    }
    Ok(history)
}

/// The share of rows of `data` whose largest output of `model` is at the row's class
/// label, running `batch` rows at a time. A row whose outputs are all NaN counts as wrong;
/// among equal largest outputs the first counts.
///
/// The model is compiled once into an [`InferenceModel`] and each batch runs through
/// [`InferenceModel::infer_batch`]. A model that cannot be compiled (a layer of your own
/// without an inference form) runs through the training graph's forward pass instead.
///
/// # Panics
///
/// Panics if `data` has value targets rather than class labels, if it has no rows, if the
/// labels are not one per row, or if `batch` is 0; and as [`InferenceModel::compile`] and
/// [`InferenceModel::infer_batch`] (or [`Sequential::forward`]) do when the shapes do not
/// fit.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::{Dataset, Linear, Sequential, Targets, accuracy};
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 2, 2, &mut Rng::seed(0)));
/// let data = Dataset { x: Tensor::zeros(&[4, 2]), targets: Targets::Classes(vec![0, 1, 0, 1]) };
/// // Zero inputs and a zero bias give equal outputs, so every row predicts class 0.
/// assert_eq!(accuracy(&net, &params, &data, 3), 0.5);
/// ```
pub fn accuracy(model: &Sequential, params: &ParamStore, data: &Dataset, batch: usize) -> f64 {
    let Targets::Classes(labels) = &data.targets else {
        panic!("accuracy: the data has value targets; accuracy needs class labels");
    };
    let (rows, features) = (data.len(), data.features());
    assert!(rows > 0, "accuracy: the data has no rows");
    assert!(batch > 0, "accuracy: the batch size must be at least 1");
    assert_eq!(
        labels.len(),
        rows,
        "accuracy: the data has {rows} rows but {} labels",
        labels.len()
    );
    let mut compiled = InferenceModel::compile(model, params).ok();
    let mut graph = Graph::new();
    let mut correct = 0usize;
    for start in (0..rows).step_by(batch) {
        let end = (start + batch).min(rows);
        let x = TensorView::from_slice(
            &data.x.data()[start * features..end * features],
            &[end - start, features],
        )
        .expect("a run of whole rows of x has the batch's shape");
        let mut count = |scores: &[f32]| {
            let classes = scores.len() / (end - start);
            for (row, &label) in scores.chunks(classes).zip(&labels[start..end]) {
                if argmax(row) == Some(label) {
                    correct += 1;
                }
            }
        };
        match compiled.as_mut() {
            Some(inference) => count(inference.infer_batch(x).data()),
            None => {
                let mut tape = graph.tape(&params.values);
                let input = tape.input(x);
                let out = model.forward(&mut tape, input);
                count(tape.value(out).data());
            }
        }
    }
    correct as f64 / rows as f64
}

/// The index of the first largest value of `row`, ignoring NaN; `None` if every value is
/// NaN.
fn argmax(row: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in row.iter().enumerate() {
        if !v.is_nan() && best.is_none_or(|(_, b)| v > b) {
            best = Some((i, v));
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    //! Tests for the training step, `fit`, the losses and `accuracy`.

    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::train::data::tests::classes_dataset;
    use crate::{Layer, Linear, Sigmoid, Tanh};
    use cavitation_core::{ParamId, Rng, Tensor};

    /// `n` rows of `features` uniform values in `[-1, 1)` from `Rng::seed(seed)`, labelled
    /// by which of `classes` equal slices of the features has the largest sum.
    fn synthetic_classes(n: usize, features: usize, classes: usize, seed: u64) -> Dataset<f32> {
        let mut rng = Rng::seed(seed);
        let x: Vec<f32> = (0..n * features)
            .map(|_| rng.range(-1.0, 1.0) as f32)
            .collect();
        let width = features / classes;
        let labels = (0..n)
            .map(|i| {
                let row = &x[i * features..(i + 1) * features];
                let sums: Vec<f32> = (0..classes)
                    .map(|c| row[c * width..(c + 1) * width].iter().sum())
                    .collect();
                (0..classes)
                    .max_by(|&a, &b| sums[a].total_cmp(&sums[b]))
                    .unwrap()
            })
            .collect();
        Dataset {
            x: Tensor::from_vec(x, &[n, features]).unwrap(),
            targets: Targets::Classes(labels),
        }
    }

    /// A `features` → 16 → `classes` tanh network from `Rng::seed(seed)`.
    fn small_classifier(
        params: &mut ParamStore,
        features: usize,
        classes: usize,
        seed: u64,
    ) -> Sequential {
        let mut rng = Rng::seed(seed);
        Sequential::new()
            .push(Linear::new(params, features, 16, &mut rng))
            .push(Tanh)
            .push(Linear::new(params, 16, classes, &mut rng))
    }

    /// The bits of every parameter value of `net`, in order.
    fn parameter_bits(net: &Sequential, params: &ParamStore) -> Vec<u32> {
        net.parameter_names()
            .into_iter()
            .flat_map(|(_, id)| {
                params
                    .value(id)
                    .data()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// The network's outputs for every row of `x`, recorded on a tape.
    fn outputs(net: &Sequential, params: &ParamStore, x: &Tensor<f32>) -> Vec<f32> {
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let xi = tape.input(x.view());
        let y = net.forward(&mut tape, xi);
        tape.value(y).data().to_vec()
    }

    /// XOR with a 2 → 8 tanh → 1 sigmoid network, mean squared error, SGD at learning rate
    /// 0.5, the full batch of 4 for 2,000 epochs: every output ends within 0.1 of its target.
    #[test]
    fn xor_learns() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(1);
        let net = Sequential::new()
            .push(Linear::new(&mut params, 2, 8, &mut rng))
            .push(Tanh)
            .push(Linear::new(&mut params, 8, 1, &mut rng))
            .push(Sigmoid);
        let x = Tensor::from_vec(vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]).unwrap();
        let y = Tensor::from_vec(vec![0.0f32, 1.0, 1.0, 0.0], &[4, 1]).unwrap();
        let data = Dataset {
            x: x.clone(),
            targets: Targets::Values(y.clone()),
        };
        let config = FitConfig {
            epochs: 2000,
            batch: 4,
            optimizer: OptimizerConfig::Sgd {
                lr: 0.5,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::Mse,
            shuffle_seed: None,
            health: Health::default(),
            on_epoch: None,
        };
        let history = fit(&net, &mut params, &data, config).unwrap();
        assert_eq!(history.epochs.len(), 2000);
        let out = outputs(&net, &params, &x);
        println!(
            "xor outputs {out:?}, final mean loss {}",
            history.epochs[1999].mean_loss
        );
        for (o, t) in out.iter().zip(y.data()) {
            assert!(
                (o - t).abs() < 0.1,
                "xor output {o} for target {t}: {out:?}"
            );
        }
    }

    /// Two runs of 2 epochs from the same seeds give bit-identical parameters and the same
    /// reported losses.
    #[test]
    fn fit_is_deterministic() {
        let data = synthetic_classes(200, 8, 4, 3);
        let run = || {
            let mut params = ParamStore::new();
            let net = small_classifier(&mut params, 8, 4, 5);
            let config = FitConfig {
                epochs: 2,
                batch: 16,
                shuffle_seed: Some(11),
                ..FitConfig::default()
            };
            let history = fit(&net, &mut params, &data, config).unwrap();
            let losses: Vec<f64> = history.epochs.iter().map(|e| e.mean_loss).collect();
            (parameter_bits(&net, &params), losses)
        };
        let (a, la) = run();
        let (b, lb) = run();
        assert_eq!(a, b);
        assert_eq!(la, lb);
        let mut params = ParamStore::new();
        let untrained = small_classifier(&mut params, 8, 4, 5);
        assert_ne!(a, parameter_bits(&untrained, &params));
    }

    /// With 300 rows and batch 128 an epoch takes 3 steps, the last of 44 rows, and every
    /// step changes the parameters.
    #[test]
    fn last_partial_batch_trains() {
        let data = synthetic_classes(300, 8, 4, 2);
        let mut params = ParamStore::new();
        let net = small_classifier(&mut params, 8, 4, 1);
        let mut trainer = Trainer::new(&net, Adam::new(0.01), Loss::CrossEntropy, 128);
        let mut batcher = Batcher::new(300, 128, Some(4));
        let mut batch = Batch::new(&data, 128);
        batcher.epoch();
        let ranges: Vec<_> = batcher.batches().collect();
        assert_eq!(
            ranges.iter().map(|r| r.len()).collect::<Vec<_>>(),
            [128, 128, 44]
        );
        for range in ranges {
            let before = parameter_bits(&net, &params);
            let loss = trainer.step(&net, &mut params, batcher.gather(&data, range, &mut batch));
            assert!(loss.is_finite() && loss > 0.0);
            assert_ne!(parameter_bits(&net, &params), before);
        }
        assert_eq!(trainer.steps(), 3);
        let history = fit(
            &net,
            &mut params,
            &data,
            FitConfig {
                epochs: 1,
                batch: 128,
                ..FitConfig::default()
            },
        )
        .unwrap();
        assert!(history.epochs[0].mean_loss.is_finite());
    }

    /// Batches of one row train: a linear regression on 20 rows with SGD takes 20 steps an
    /// epoch, and its mean loss falls every epoch.
    #[test]
    fn batch_of_one_trains() {
        let mut rng = Rng::seed(9);
        let x: Vec<f32> = (0..40).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        let y: Vec<f32> = x.chunks(2).map(|r| 2.0 * r[0] - r[1] + 0.5).collect();
        let data = Dataset {
            x: Tensor::from_vec(x, &[20, 2]).unwrap(),
            targets: Targets::Values(Tensor::from_vec(y, &[20, 1]).unwrap()),
        };
        let mut params = ParamStore::new();
        let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(3)));
        let mut trainer = Trainer::new(&net, Sgd::new(0.05), Loss::Mse, 1);
        let mut batcher = Batcher::new(20, 1, Some(1));
        let mut batch = Batch::new(&data, 1);
        batcher.epoch();
        for range in batcher.batches() {
            let view = batcher.gather(&data, range, &mut batch);
            assert_eq!(view.x.shape().dims(), &[1, 2]);
            assert!(trainer.step(&net, &mut params, view).is_finite());
        }
        assert_eq!(trainer.steps(), 20);
        let config = FitConfig {
            epochs: 5,
            batch: 1,
            optimizer: OptimizerConfig::Sgd {
                lr: 0.05,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::Mse,
            ..FitConfig::default()
        };
        let losses: Vec<f64> = fit(&net, &mut params, &data, config)
            .unwrap()
            .epochs
            .iter()
            .map(|e| e.mean_loss)
            .collect();
        assert!(losses.windows(2).all(|w| w[1] < w[0]), "{losses:?}");
    }

    /// Class labels with mean squared error panic, naming the loss and the kind of targets.
    #[test]
    #[should_panic(expected = "Loss::Mse needs value targets, but the batch has class targets")]
    fn mse_needs_value_targets() {
        let data = classes_dataset(4);
        let mut params = ParamStore::new();
        let net = Sequential::new().push(Linear::new(&mut params, 3, 1, &mut Rng::seed(0)));
        let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 4);
        let batcher = Batcher::new(4, 4, None);
        let mut batch = Batch::new(&data, 4);
        let view = batcher.gather(&data, 0..4, &mut batch);
        trainer.step(&net, &mut params, view);
    }

    /// Value targets with cross-entropy panic, naming the loss and the kind of targets.
    #[test]
    #[should_panic(
        expected = "Loss::CrossEntropy needs class targets, but the batch has value targets"
    )]
    fn cross_entropy_needs_class_targets() {
        let data = Dataset {
            x: Tensor::zeros(&[2, 3]),
            targets: Targets::Values(Tensor::zeros(&[2, 2])),
        };
        let mut params = ParamStore::new();
        let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
        let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::CrossEntropy, 2);
        let batcher = Batcher::new(2, 2, None);
        let mut batch = Batch::new(&data, 2);
        let view = batcher.gather(&data, 0..2, &mut batch);
        trainer.step(&net, &mut params, view);
    }

    /// A layer that passes its input through and counts its forward passes.
    struct CountForwards {
        /// The number of forward passes so far.
        count: Arc<AtomicUsize>,
    }

    impl Layer<f32> for CountForwards {
        fn name(&self) -> &str {
            "count_forwards"
        }
        fn output_width(&self, input_width: usize) -> usize {
            input_width
        }
        fn forward(&self, _tape: &mut Tape<'_, f32>, x: Var) -> Var {
            self.count.fetch_add(1, Ordering::SeqCst);
            x
        }
        fn parameters(&self) -> Vec<(&'static str, ParamId)> {
            Vec::new()
        }
    }

    /// A hook on a layer the model does not have panics, naming the index and the range,
    /// before the forward pass runs and before any parameter changes.
    #[test]
    fn bad_hook_index_panics_before_forward() {
        let data = classes_dataset(4);
        let mut params = ParamStore::new();
        let forwards = Arc::new(AtomicUsize::new(0));
        let net = Sequential::new()
            .push(CountForwards {
                count: Arc::clone(&forwards),
            })
            .push(Linear::new(&mut params, 3, 4, &mut Rng::seed(0)));
        let before = parameter_bits(&net, &params);
        let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::CrossEntropy, 4);
        let batcher = Batcher::new(4, 4, None);
        let mut batch = Batch::new(&data, 4);
        let view = batcher.gather(&data, 0..4, &mut batch);
        let mut hooks = Hooks::new();
        hooks.on_forward(7, |_| {});
        let result = catch_unwind(AssertUnwindSafe(|| {
            trainer.step_with_hooks(&net, &mut params, view, &mut hooks)
        }));
        let message = result.expect_err("a hook on layer 7 of 2 panics");
        let message = message
            .downcast_ref::<String>()
            .expect("the panic message is a String");
        assert!(
            message.contains("layer 7, but the model has 2 layers (valid indices 0..2)"),
            "{message}"
        );
        assert_eq!(forwards.load(Ordering::SeqCst), 0, "the forward pass ran");
        assert_eq!(parameter_bits(&net, &params), before);
        assert_eq!(trainer.steps(), 0);
    }

    /// `fit` returns an error, before training, when the targets do not have one entry per
    /// row or the data set is empty.
    #[test]
    fn fit_rejects_inconsistent_data() {
        let mut params = ParamStore::new();
        let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
        let before = parameter_bits(&net, &params);
        let short = Dataset {
            x: Tensor::zeros(&[4, 3]),
            targets: Targets::Classes(vec![0, 1, 0]),
        };
        let err = fit(&net, &mut params, &short, FitConfig::default()).unwrap_err();
        assert_eq!(err.to_string(), "shape mismatch: expected [4], found [3]");
        let values = Dataset {
            x: Tensor::zeros(&[4, 3]),
            targets: Targets::Values(Tensor::zeros(&[5, 2])),
        };
        let err = fit(&net, &mut params, &values, FitConfig::default()).unwrap_err();
        assert_eq!(
            err.to_string(),
            "shape mismatch: expected [4, 2], found [5, 2]"
        );
        let empty = Dataset {
            x: Tensor::zeros(&[0, 3]),
            targets: Targets::Classes(vec![]),
        };
        assert!(fit(&net, &mut params, &empty, FitConfig::default()).is_err());
        assert_eq!(parameter_bits(&net, &params), before);
    }

    /// `on_epoch` sees each epoch's report, numbered from 1, the same reports `fit` returns.
    #[test]
    fn fit_reports_every_epoch() {
        use std::sync::{Arc, Mutex};
        let data = synthetic_classes(50, 8, 4, 1);
        let mut params = ParamStore::new();
        let net = small_classifier(&mut params, 8, 4, 2);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let config = FitConfig {
            epochs: 3,
            on_epoch: Some(Box::new(move |r: &EpochReport| {
                sink.lock().unwrap().push(*r)
            })),
            ..FitConfig::default()
        };
        let history = fit(&net, &mut params, &data, config).unwrap();
        assert_eq!(*seen.lock().unwrap(), history.epochs);
        let numbers: Vec<usize> = history.epochs.iter().map(|e| e.epoch).collect();
        assert_eq!(numbers, [1, 2, 3]);
        assert!(
            history
                .epochs
                .iter()
                .all(|e| e.seconds >= 0.0 && e.mean_loss > 0.0)
        );
    }

    /// The defaults: 10 epochs of 32 rows, Adam at 1e-3, cross-entropy, shuffled with seed 0.
    #[test]
    fn fit_config_defaults() {
        let config = FitConfig::default();
        assert_eq!((config.epochs, config.batch), (10, 32));
        assert_eq!(
            config.optimizer,
            OptimizerConfig::Adam {
                lr: 1e-3,
                weight_decay: 0.0
            }
        );
        assert_eq!(config.loss, Loss::CrossEntropy);
        assert_eq!(config.shuffle_seed, Some(0));
        assert!(config.on_epoch.is_none());
    }

    /// Every loss trains a step on matching targets and returns the loss the tape computes
    /// for the same batch.
    #[test]
    fn every_loss_steps() {
        let x = Tensor::from_vec(vec![0.1f32, -0.4, 0.3, 0.8, -0.2, 0.5], &[2, 3]).unwrap();
        let classes = Dataset {
            x: x.clone(),
            targets: Targets::Classes(vec![1, 0]),
        };
        let probs = Dataset {
            x: x.clone(),
            targets: Targets::Values(
                Tensor::from_vec(vec![0.25f32, 0.75, 1.0, 0.0], &[2, 2]).unwrap(),
            ),
        };
        let losses = [
            (Loss::CrossEntropy, &classes),
            (Loss::Nll, &classes),
            (Loss::Mse, &probs),
            (Loss::L1, &probs),
            (Loss::Huber { delta: 0.5 }, &probs),
            (Loss::SmoothL1 { beta: 0.5 }, &probs),
            (Loss::BinaryCrossEntropy, &probs),
            (Loss::BinaryCrossEntropyWithLogits, &probs),
            (Loss::KlDiv, &probs),
        ];
        for (loss, data) in losses {
            let mut params = ParamStore::new();
            let mut net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
            net = match loss {
                Loss::Nll | Loss::KlDiv => net.push(crate::LogSoftmax),
                Loss::BinaryCrossEntropy => net.push(Sigmoid),
                _ => net,
            };
            let mut trainer = Trainer::new(&net, Sgd::new(0.1), loss, 2);
            let batcher = Batcher::new(2, 2, None);
            let mut batch = Batch::new(data, 2);
            let expected = {
                let mut graph = Graph::new();
                let mut tape = graph.tape(&params.values);
                let xi = tape.input(x.view());
                let out = net.forward(&mut tape, xi);
                let view = batcher.gather(data, 0..2, &mut batch);
                let l = loss.record(&mut tape, out, view.targets);
                tape.value(l).data()[0]
            };
            let before = parameter_bits(&net, &params);
            let got = trainer.step(&net, &mut params, batcher.gather(data, 0..2, &mut batch));
            assert_eq!(got.to_bits(), expected.to_bits(), "{loss:?}");
            assert!(got.is_finite() && got > 0.0, "{loss:?}: {got}");
            assert_ne!(parameter_bits(&net, &params), before, "{loss:?}");
        }
    }

    /// Accuracy counts the rows whose largest output is at the label; a row with NaN
    /// everywhere counts as wrong.
    #[test]
    fn accuracy_counts_argmax_hits() {
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 2, 2, &mut Rng::seed(0));
        // Identity weights and no bias: the outputs are the inputs.
        params
            .value_mut(layer.weight())
            .data_mut()
            .copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        let net = Sequential::new().push(layer);
        let data = Dataset {
            x: Tensor::from_vec(
                vec![
                    1.0f32,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    1.0,
                    f32::NAN,
                    f32::NAN,
                    0.0,
                    3.0,
                ],
                &[5, 2],
            )
            .unwrap(),
            targets: Targets::Classes(vec![0, 1, 1, 0, 1]),
        };
        for batch in [1, 2, 5, 8] {
            assert_eq!(accuracy(&net, &params, &data, batch), 0.6, "batch {batch}");
        }
    }

    /// Accuracy needs class labels.
    #[test]
    #[should_panic(expected = "accuracy: the data has value targets")]
    fn accuracy_needs_class_targets() {
        let mut params = ParamStore::new();
        let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
        let data = Dataset {
            x: Tensor::zeros(&[3, 2]),
            targets: Targets::Values(Tensor::zeros(&[3, 1])),
        };
        accuracy(&net, &params, &data, 2);
    }

    /// Every optimiser configuration builds the optimiser it names, with its learning rate.
    #[test]
    fn optimizer_config_builds() {
        let configs = [
            OptimizerConfig::Sgd {
                lr: 0.1,
                momentum: 0.9,
                weight_decay: 1e-4,
            },
            OptimizerConfig::Adam {
                lr: 0.01,
                weight_decay: 0.0,
            },
            OptimizerConfig::AdamW {
                lr: 0.001,
                weight_decay: 0.01,
            },
        ];
        let lrs: Vec<f64> = configs.iter().map(|c| c.build().learning_rate()).collect();
        assert_eq!(lrs, [0.1, 0.01, 0.001]);
    }
}
