//! The training step ([`Trainer`]) with its optional per-layer [`StepProfile`], the losses
//! it minimises, [`fit`] (with and without hooks), [`fit_controlled`] with a [`StepControl`]
//! between steps, and [`accuracy`].

use std::fmt;
use std::time::{Duration, Instant};

use cavitation_core::{Error, Graph, ParamId, ParamStore, Reduction, Tape, TensorView, Var};

use super::{
    Batch, BatchView, Batcher, Dataset, Evaluation, Evaluator, Flow, Targets, TargetsView,
};
use crate::{
    Adagrad, Adam, AdamW, Health, HealthMonitor, Hooks, InferenceModel, NeuronMask, Optimizer,
    RmsProp, Sequential, Sgd, StepInputs, Trace,
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
    /// Whether this loss cannot be negative on valid inputs: true for
    /// [`CrossEntropy`](Loss::CrossEntropy), [`Nll`](Loss::Nll), [`Mse`](Loss::Mse),
    /// [`L1`](Loss::L1), [`Huber`](Loss::Huber), [`SmoothL1`](Loss::SmoothL1),
    /// [`BinaryCrossEntropy`](Loss::BinaryCrossEntropy) and
    /// [`BinaryCrossEntropyWithLogits`](Loss::BinaryCrossEntropyWithLogits); false for
    /// [`KlDiv`](Loss::KlDiv), which is negative when its input is not a log-probability
    /// vector. [`Trainer`] sends [`Warning::NegativeLoss`](crate::Warning::NegativeLoss) when
    /// a loss for which this is true comes out below 0.
    ///
    /// ```
    /// use cavitation_nn::Loss;
    /// assert!(Loss::Nll.non_negative());
    /// assert!(!Loss::KlDiv.non_negative());
    /// ```
    pub fn non_negative(&self) -> bool {
        !matches!(self, Loss::KlDiv)
    }

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

    /// Records this loss of `out` against `targets` on `tape` and returns the loss variable,
    /// exactly as [`Trainer`] records it in each step (with [`Reduction::Mean`], or
    /// [`Reduction::BatchMean`] for [`KlDiv`](Loss::KlDiv)). Value targets are recorded as
    /// an input first. Use it to measure the training loss on a tape of your own, such as
    /// one row run forward and backward beside training.
    ///
    /// # Panics
    ///
    /// Panics if the kind of `targets` is not the kind the loss needs (class labels for
    /// [`CrossEntropy`](Loss::CrossEntropy) and [`Nll`](Loss::Nll), values for the others;
    /// the message names both), and as the tape operation does on shapes it rejects.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// use cavitation_nn::{Loss, TargetsView};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let out = tape.input(Tensor::from_vec(vec![1.0f32, 3.0], &[2, 1]).unwrap().view());
    /// let target = Tensor::from_vec(vec![0.0f32, 1.0], &[2, 1]).unwrap();
    /// let loss = Loss::Mse.record(&mut tape, out, TargetsView::Values(target.view()));
    /// // ((1 - 0)² + (3 - 1)²) / 2
    /// assert_eq!(tape.value(loss).data(), &[2.5]);
    /// ```
    pub fn record(self, tape: &mut Tape<'_, f32>, out: Var, targets: TargetsView<'_, f32>) -> Var {
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
/// step allocates nothing. Each step returns its loss, or the health checks' error at the
/// step that met it; call [`end_epoch`](Self::end_epoch) after each epoch.
///
/// # Examples
///
/// ```
/// use cavitation_core::{Error, ParamStore, Rng, Tensor};
/// use cavitation_nn::{
///     Batch, Batcher, Dataset, Linear, Loss, Sequential, Sgd, Targets, Trainer,
/// };
/// let data = Dataset {
///     x: Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], &[4, 1]).unwrap(),
///     targets: Targets::Values(Tensor::from_vec(vec![1.0f32, 3.0, 5.0, 7.0], &[4, 1]).unwrap()),
/// };
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 1, 1, &mut Rng::seed(0)));
/// let mut trainer = Trainer::new(&net, Sgd::new(0.05), Loss::Mse, 4);
/// let batcher = Batcher::new(4, 4, None);
/// let mut batch = Batch::new(&data, 4);
/// let mut train = || -> Result<(f32, f32), Error> {
///     let first = trainer.step(&net, &mut params, batcher.gather(&data, 0..4, &mut batch))?;
///     let mut last = first;
///     for epoch in 1..=200 {
///         last = trainer.step(&net, &mut params, batcher.gather(&data, 0..4, &mut batch))?;
///         trainer.end_epoch(epoch)?;
///     }
///     Ok((first, last))
/// };
/// let (first, last) = train().unwrap();
/// assert!(last < first * 0.01);
/// ```
///
/// The same step written out by hand, the way to train with a loss [`Loss`] does not list:
/// open a tape on the parameter values, add the batch as an input, run the model (traced,
/// for the health checks), record the loss, run `backward` into the gradients, run the
/// [`HealthMonitor`]'s tape checks, drop the tape, ask the monitor whether to apply the
/// update (under [`OnNonFinite::Skip`](crate::OnNonFinite::Skip) it checks the gradients
/// first), step the optimiser, run the monitor's gradient checks on the optimiser's
/// statistics and mark the gradients stale; at the end
/// of each epoch, end the monitor's epoch. The tape borrows `params.values` while
/// `backward` writes `params.grads`; borrowing the two fields separately is what lets this
/// compile.
///
/// ```
/// use cavitation_core::{Error, Graph, ParamStore, Reduction, Rng, Tensor};
/// use cavitation_nn::{
///     Health, HealthMonitor, Linear, Optimizer, Sequential, Sgd, StepInputs, TargetsView, Trace,
/// };
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
///         let inputs = StepInputs {
///             x: x.view(),
///             targets: TargetsView::Values(y.view()),
///             rows: None,
///         };
///         monitor.after_backward(step, value, &tape, &trace, &net, inputs)?;
///         drop(tape);
///         if monitor.before_update(step, value, &params, &names) {
///             optimizer.step(&mut params);
///             monitor.after_update(step, optimizer.last_stats(), &names)?;
///         }
///         params.zero_grad();
///         monitor.end_epoch(epoch)?;
///     }
///     Ok(())
/// };
/// train().unwrap();
/// ```
pub struct Trainer<'s> {
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
    /// The number of those steps whose update was left out under
    /// [`OnNonFinite::Skip`](crate::OnNonFinite::Skip).
    skipped: u64,
    /// The health checks run on every step.
    health: HealthMonitor<'s>,
    /// The model's parameter names, for the health checks' messages.
    names: Vec<(String, ParamId)>,
    /// The step profile's state: asked for, last result and reused buffers.
    profiler: Profiler,
    /// Whether [`override_learning_rate`](Self::override_learning_rate) has stopped the
    /// learning-rate schedule for the rest of the run.
    schedule_stopped: bool,
    /// The run's learning-rate schedule, which [`run_epoch`](Self::run_epoch) applies;
    /// `None` outside [`fit`] and [`fit_controlled`].
    rates: Option<StepRates>,
}

impl<'s> Trainer<'s> {
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
    ) -> Trainer<'s> {
        Trainer::with_boxed(model, Box::new(optimizer), loss, batch)
    }

    /// [`Trainer::new`] for an optimiser that is already boxed.
    fn with_boxed(
        model: &Sequential,
        optimizer: Box<dyn Optimizer<f32>>,
        loss: Loss,
        batch: usize,
    ) -> Trainer<'s> {
        assert!(batch > 0, "Trainer::new: the batch size must be at least 1");
        Trainer {
            graph: Graph::new(),
            trace: Trace::with_capacity(model.len()),
            optimizer,
            loss,
            batch_size: batch,
            batch: None,
            steps: 0,
            skipped: 0,
            health: HealthMonitor::new(Health::default()).with_loss(loss),
            names: model.parameter_names(),
            profiler: Profiler::default(),
            schedule_stopped: false,
            rates: None,
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
    pub fn set_health(&mut self, health: Health<'s>) {
        self.health = HealthMonitor::new(health).with_loss(self.loss);
    }

    /// Ends epoch `epoch` of the health checks ([`HealthMonitor::end_epoch`]: the dead-neuron
    /// report, and a new epoch for the once-per-epoch warnings). Call it after the last step
    /// of each epoch; [`fit`] does.
    ///
    /// # Errors
    ///
    /// None of the end-of-epoch checks stops training, so this returns `Ok`; it returns a
    /// `Result` so that a loop can use `?` on it and on [`step`](Self::step) alike.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, Loss, Sequential, Sgd, Trainer};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 1, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 2);
    /// assert!(trainer.end_epoch(1).is_ok());
    /// ```
    pub fn end_epoch(&mut self, epoch: usize) -> Result<(), Error> {
        self.health.end_epoch(epoch)
    }

    /// Trains on one batch: the forward pass of `model` on `batch.x`, the loss against
    /// `batch.targets`, the backward pass into `params.grads`, the optimiser step and
    /// [`ParamStore::zero_grad`]. Returns the batch's loss (its mean over the batch).
    ///
    /// The gradients must be stale when it starts, as the previous step leaves them.
    ///
    /// Every step runs the health checks (see [`HealthMonitor`]).
    ///
    /// # Errors
    ///
    /// Under [`OnNonFinite::Stop`](crate::OnNonFinite) (the default), returns [`Error::NonFinite`]
    /// when the loss or a gradient is NaN or infinite, naming the step and `"loss"` or the
    /// parameter, and sends the matching [`Warning::NonFinite`] to the health sink. For a
    /// non-finite loss the error's `detail` names the first NaN or infinite value in the batch's
    /// inputs or value targets (with no dataset row, as `step` does not know one), or says that the
    /// inputs were finite and the values overflowed. A NaN or infinite loss skips the optimiser
    /// step, so the parameters stay as they were; a NaN or infinite gradient with a finite loss is
    /// found after the update has applied it, so the parameters then hold the NaN. The step is
    /// counted and the gradients are marked stale either way. [`fit`] stops at this error. Under
    /// [`OnNonFinite::Warn`](crate::OnNonFinite) the step returns its loss, NaN or not, and the
    /// update is applied. Under [`OnNonFinite::Skip`](crate::OnNonFinite) the step returns its
    /// loss too, but a step whose loss or gradient is NaN or infinite leaves the parameters and
    /// the optimiser's state as they were; it is counted in [`steps`](Self::steps) and in
    /// [`skipped_steps`](Self::skipped_steps). Its hooks have run, and a profile asked for
    /// with [`profile_next_step`](Self::profile_next_step) carries over to the next step that
    /// trains.
    ///
    /// [`Warning::NonFinite`]: crate::Warning::NonFinite
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
    /// use cavitation_nn::{
    ///     Batch, Batcher, Dataset, Linear, Loss, Sequential, Sgd, Targets, Trainer,
    /// };
    /// let data = Dataset { x: Tensor::zeros(&[3, 2]), targets: Targets::Classes(vec![0, 1, 1]) };
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 2, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::CrossEntropy, 3);
    /// let mut batch = Batch::new(&data, 3);
    /// let view = Batcher::new(3, 3, None).gather(&data, 0..3, &mut batch);
    /// let loss = trainer.step(&net, &mut params, view).unwrap();
    /// // All-zero inputs and a zero bias give equal scores: the loss is ln 2.
    /// assert!((loss - 2f32.ln()).abs() < 1e-6);
    /// ```
    pub fn step(
        &mut self,
        model: &Sequential,
        params: &mut ParamStore,
        batch: BatchView<'_, f32>,
    ) -> Result<f32, Error> {
        self.step_with_hooks(model, params, batch, &mut Hooks::new())
    }

    /// Like [`step`](Self::step), and runs `hooks`: the forward hooks right after the forward
    /// pass, the backward hooks right after the backward pass, both before the optimiser
    /// updates the parameters (so every event shows the values and gradients of this step).
    /// With no hooks registered it does exactly what `step` does.
    ///
    /// # Errors
    ///
    /// Returns the errors [`step`](Self::step) returns, after the hooks have run.
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
    /// let data = Dataset {
    ///     x: Tensor::full(&[2, 2], 0.5f32),
    ///     targets: Targets::Classes(vec![0, 1]),
    /// };
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
    /// trainer.step_with_hooks(&net, &mut params, view, &mut hooks).unwrap();
    /// drop(hooks);
    /// assert_eq!(widths, [vec![2, 3]]);
    /// ```
    pub fn step_with_hooks(
        &mut self,
        model: &Sequential,
        params: &mut ParamStore,
        batch: BatchView<'_, f32>,
        hooks: &mut Hooks<'_>,
    ) -> Result<f32, Error> {
        let (value, checked) = self.checked_step(model, params, batch, None, hooks);
        checked.map(|_| value)
    }

    /// The step behind [`step_with_hooks`](Self::step_with_hooks): returns the batch's loss
    /// and the health checks' verdict, `Ok(true)` when the update was applied and `Ok(false)`
    /// when [`OnNonFinite::Skip`](crate::OnNonFinite::Skip) left it out. When the tape checks
    /// fail, or the update is left out, the optimiser step does not run; the gradients are
    /// marked stale and the step is counted either way.
    ///
    /// A step that was asked for a profile ([`profile_next_step`](Self::profile_next_step))
    /// times its parts into the profile's buffers; any other step reads no clock and only
    /// marks the last profile as absent.
    fn checked_step(
        &mut self,
        model: &Sequential,
        params: &mut ParamStore,
        batch: BatchView<'_, f32>,
        rows: Option<&[usize]>,
        hooks: &mut Hooks<'_>,
    ) -> (f32, Result<bool, Error>) {
        self.loss.check_targets(batch.targets);
        hooks.check_layers(model);
        let step = self.steps + 1;
        let profiling = std::mem::take(&mut self.profiler.asked);
        self.profiler.valid = false;
        let start = profiling.then(Instant::now);
        let (value, mut checked) = {
            let mut tape = self.graph.tape(&params.values);
            let x = tape.input(batch.x);
            self.trace.set_timing(profiling);
            let out = model.forward_traced(&mut tape, x, &mut self.trace);
            self.trace.set_timing(false);
            hooks.run_forward(&tape, &self.trace, model);
            let loss_start = profiling.then(Instant::now);
            let loss = self.loss.record(&mut tape, out, batch.targets);
            let value = tape.value(loss).data()[0];
            if let Some(loss_start) = loss_start {
                let profile = &mut self.profiler.profile;
                profile.loss = loss_start.elapsed();
                profile.forward.clear();
                profile
                    .forward
                    .extend((0..self.trace.len()).map(|i| self.trace.forward_time(i)));
                let marks = &mut self.profiler.marks;
                marks.clear();
                marks.extend((0..self.trace.len()).map(|i| self.trace.output(i)));
                let times = &mut self.profiler.times;
                times.clear();
                times.resize(marks.len() + 1, Duration::ZERO);
                tape.backward_timed(loss, &mut params.grads, marks, times);
                profile.backward.clear();
                profile.backward.extend_from_slice(&times[..marks.len()]);
                profile.backward_loss = times[marks.len()];
            } else {
                tape.backward(loss, &mut params.grads);
            }
            hooks.run_backward(&tape, &self.trace, model);
            let inputs = StepInputs {
                x: batch.x,
                targets: batch.targets,
                rows,
            };
            let checked =
                self.health
                    .after_backward(step, value, &tape, &self.trace, model, inputs);
            (value, checked)
        };
        let apply = checked.is_ok() && self.health.before_update(step, value, params, &self.names);
        if apply {
            let update_start = profiling.then(Instant::now);
            self.optimizer.step(params);
            if let Some(update_start) = update_start {
                self.profiler.profile.optimizer = update_start.elapsed();
            }
            checked = self
                .health
                .after_update(step, self.optimizer.last_stats(), &self.names);
        } else if profiling {
            self.profiler.profile.optimizer = Duration::ZERO;
        }
        let skipped = checked.is_ok() && !apply;
        if skipped {
            self.skipped += 1;
            // The step that was asked for a profile did not train: ask the next one.
            self.profiler.asked |= profiling;
        }
        params.zero_grad();
        self.steps = step;
        if let Some(start) = start
            && !skipped
        {
            self.profiler.profile.total = start.elapsed();
            self.profiler.valid = true;
        }
        (value, checked.map(|()| apply))
    }

    /// Asks the next step to record a [`StepProfile`]: each layer's forward and backward
    /// time, the loss, the optimiser step and the whole step, read afterwards with
    /// [`last_profile`](Self::last_profile). Only that one step is profiled; a step that was
    /// not asked reads no clock and does no extra work.
    ///
    /// The profile's buffers are made by the first profiled step and reused by every later
    /// one, so after one profiled step a profiled step allocates no more than any other.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_nn::{
    ///     Batch, Batcher, Dataset, Linear, Loss, Relu, Sequential, Sgd, Targets, Trainer,
    /// };
    /// let targets = Targets::Classes(vec![0, 1, 0, 1]);
    /// let data = Dataset { x: Tensor::full(&[4, 2], 0.5f32), targets };
    /// let mut params = ParamStore::new();
    /// let mut rng = Rng::seed(0);
    /// let net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 8, &mut rng))
    ///     .push(Relu)
    ///     .push(Linear::new(&mut params, 8, 2, &mut rng));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::CrossEntropy, 4);
    /// let mut batch = Batch::new(&data, 4);
    /// let batcher = Batcher::new(4, 4, None);
    /// trainer.profile_next_step();
    /// trainer.step(&net, &mut params, batcher.gather(&data, 0..4, &mut batch)).unwrap();
    /// let profile = trainer.last_profile().unwrap();
    /// assert_eq!((profile.forward.len(), profile.backward.len()), (3, 3));
    /// // The next step was not asked, so it has no profile.
    /// trainer.step(&net, &mut params, batcher.gather(&data, 0..4, &mut batch)).unwrap();
    /// assert!(trainer.last_profile().is_none());
    /// ```
    pub fn profile_next_step(&mut self) {
        self.profiler.asked = true;
    }

    /// The profile of the last step, if that step was asked for one with
    /// [`profile_next_step`](Self::profile_next_step); `None` after any step that was not,
    /// and before the first step.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, Loss, Sequential, Sgd, Trainer};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 8);
    /// trainer.profile_next_step();
    /// // Asking is not enough: the profile comes from the step that runs next.
    /// assert!(trainer.last_profile().is_none());
    /// ```
    pub fn last_profile(&self) -> Option<&StepProfile> {
        self.profiler.valid.then_some(&self.profiler.profile)
    }

    /// Sets the optimiser's learning rate to `lr` for every later step, and stops
    /// [`fit_controlled`]'s learning-rate schedule ([`FitConfig::schedule`]) for the rest of
    /// the run, so that no later step sets the rate again. Called between the steps of an
    /// epoch, it applies from the next step on. Outside a run (a loop of
    /// [`step`](Self::step) calls) it only sets the rate.
    ///
    /// Only the optimiser's rate changes: the rest of its state carries on. [`Adam`] and
    /// [`AdamW`] keep their moments and their per-parameter step counts (so the bias
    /// correction does not restart), and [`Sgd`] keeps its momentum buffers, which the new
    /// rate then scales. An optimiser of your own that changes its rate inside
    /// [`Optimizer::step`] may move it again on the next step;
    /// [`learning_rate`](Self::learning_rate) always reports the optimiser's own value.
    ///
    /// # Panics
    ///
    /// Panics if `lr` is negative, NaN or infinite: a rate like that is a programming error,
    /// as the other panics of this crate are. A caller that takes the rate from outside the
    /// program (a command line, a file, a network request) checks that it is finite and at
    /// least 0 first, and reports a bad one to whoever sent it.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Adam, Linear, Loss, Sequential, Trainer};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Adam::new(1e-3), Loss::Mse, 8);
    /// trainer.override_learning_rate(5e-4);
    /// assert_eq!(trainer.learning_rate(), 5e-4);
    /// ```
    pub fn override_learning_rate(&mut self, lr: f64) {
        assert!(
            lr.is_finite() && lr >= 0.0,
            "Trainer::override_learning_rate: the learning rate must be finite and at least 0, \
             got {lr}"
        );
        self.optimizer.set_learning_rate(lr);
        self.schedule_stopped = true;
    }

    /// The optimiser's current learning rate: the rate the next step trains at.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, Loss, Sequential, Sgd, Trainer};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// assert_eq!(Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 8).learning_rate(), 0.1);
    /// ```
    pub fn learning_rate(&self) -> f64 {
        self.optimizer.learning_rate()
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

    /// How many of the steps taken so far left their update out under
    /// [`OnNonFinite::Skip`](crate::OnNonFinite::Skip), because their loss or a gradient was
    /// NaN or infinite. Always 0 under the other policies.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_nn::{
    ///     Batch, Batcher, Dataset, Health, Linear, Loss, OnNonFinite, Sequential, Sgd, Targets,
    ///     Trainer,
    /// };
    /// let x = Tensor::from_vec(vec![f32::NAN, 1.0], &[2, 1]).unwrap();
    /// let data = Dataset { x, targets: Targets::Values(Tensor::zeros(&[2, 1])) };
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 1, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 2);
    /// trainer.set_health(Health { on_non_finite: OnNonFinite::Skip, ..Health::quiet() });
    /// let mut batch = Batch::new(&data, 2);
    /// let view = Batcher::new(2, 2, None).gather(&data, 0..2, &mut batch);
    /// assert!(trainer.step(&net, &mut params, view).unwrap().is_nan());
    /// assert_eq!((trainer.steps(), trainer.skipped_steps()), (1, 1));
    /// ```
    pub fn skipped_steps(&self) -> u64 {
        self.skipped
    }

    /// Runs epoch `epoch` over `data` in the batches of `batcher` (reshuffled first), running
    /// `hooks` and the health checks on every step and, for a controlled run, the control
    /// after every step. Returns the mean loss over the rows the epoch trained on (each
    /// batch's loss weighted by its row count; a step left out under
    /// [`OnNonFinite::Skip`](crate::OnNonFinite::Skip) trained on none), how many rows that
    /// was, whether every batch ran, and whether the control stopped the run. The control
    /// does not run after a step that was left out.
    ///
    /// # Errors
    ///
    /// Returns the first error of the health checks, at the step that met it.
    fn run_epoch(
        &mut self,
        run: &mut Run<'_, '_>,
        params: &mut ParamStore,
        data: &Dataset,
        batcher: &mut Batcher,
        hooks: &mut Hooks<'_>,
        epoch: usize,
    ) -> Result<EpochEnd, Error> {
        let mut batch = self
            .batch
            .take()
            .unwrap_or_else(|| Batch::new(data, self.batch_size));
        batcher.epoch();
        let mut total = 0.0f64;
        let mut trained = 0usize;
        let mut seen = 0usize;
        let mut checked = Ok(());
        let mut stopped = false;
        for (position, range) in batcher.batches().enumerate() {
            // The per-epoch schedules set the rate at an epoch's first step only, so an
            // optimiser of your own that moves its rate inside a step keeps its move until
            // the next epoch, as before the per-step schedules.
            if let Some(rates) = self.rates
                && !self.schedule_stopped
                && (position == 0 || rates.per_step())
            {
                let step = (epoch - 1) * rates.steps_per_epoch + position;
                self.optimizer.set_learning_rate(rates.at(step));
            }
            // Only a control reads the step's time, so only a controlled run reads the clock.
            let start = matches!(run, Run::Controlled(..)).then(Instant::now);
            let rows = range.len();
            let view = batcher.gather(data, range.clone(), &mut batch);
            let rows_of_batch = &batcher.order()[range];
            let (loss, step_checked) =
                self.checked_step(run.model(), params, view, Some(rows_of_batch), hooks);
            seen += rows;
            match step_checked {
                Err(err) => {
                    checked = Err(err);
                    break;
                }
                Ok(false) => continue,
                Ok(true) => {}
            }
            total += f64::from(loss) * rows as f64;
            trained += rows;
            if let (Run::Controlled(model, control), Some(start)) = (&mut *run, start) {
                let info = StepInfo {
                    step: self.steps,
                    epoch,
                    loss,
                    rows,
                    seconds: start.elapsed().as_secs_f64(),
                };
                let mut handle = ModelHandle::new(model);
                let mut trainer = TrainerHandle::new(self);
                if control.after_step(&info, &mut handle, params, &mut trainer) == Flow::Stop {
                    stopped = true;
                    break;
                }
            }
        }
        self.batch = Some(batch);
        checked.map(|()| EpochEnd {
            mean_loss: total / trained as f64,
            rows: trained,
            whole: seen == data.len(),
            stopped,
        })
    }
}

/// The run's learning-rate schedule as [`Trainer::run_epoch`] applies it before each step.
#[derive(Clone, Copy, Debug)]
struct StepRates {
    /// The schedule.
    schedule: LrSchedule,
    /// The optimiser's configured rate, the schedule's peak.
    initial: f64,
    /// The step positions of one epoch: the batches of the data, the last one smaller.
    steps_per_epoch: usize,
    /// The run's epochs.
    epochs: usize,
}

impl StepRates {
    /// The rate of step position `step` of the run, counting from 0.
    fn at(&self, step: usize) -> f64 {
        self.schedule
            .learning_rate(self.initial, step, self.steps_per_epoch, self.epochs)
    }

    /// Whether the schedule changes the rate within an epoch.
    fn per_step(&self) -> bool {
        matches!(
            self.schedule,
            LrSchedule::Warmup { .. } | LrSchedule::WarmupCosine { .. }
        )
    }
}

/// How an epoch of [`Trainer::run_epoch`] ended without an error.
#[derive(Debug)]
struct EpochEnd {
    /// The mean loss over the rows the epoch trained on; NaN when it trained on none.
    mean_loss: f64,
    /// The rows the epoch trained on: all of the data's, less the rows of steps left out
    /// under [`OnNonFinite::Skip`](crate::OnNonFinite::Skip) and of the steps after a
    /// control stopped the run.
    rows: usize,
    /// Whether every batch of the epoch ran, trained or left out: false only when the
    /// control stopped the run before the epoch's last step.
    whole: bool,
    /// Whether the control stopped the run during the epoch.
    stopped: bool,
}

/// The model a run trains, and the control between its steps, if it has one.
enum Run<'m, 'c> {
    /// [`fit`] and [`fit_with_hooks`]: the model is only read and nothing runs between steps.
    Plain(&'m Sequential),
    /// [`fit_controlled`]: the control runs after each step and each epoch and may change
    /// the model's neuron masks through a [`ModelHandle`].
    Controlled(&'m mut Sequential, &'c mut dyn StepControl),
}

impl Run<'_, '_> {
    /// The model, to read.
    fn model(&self) -> &Sequential {
        match self {
            Run::Plain(model) => model,
            Run::Controlled(model, _) => model,
        }
    }
}

/// The step profile's state inside a [`Trainer`]: whether the next step should be profiled,
/// whether the last one was, the profile and the reused buffers of the timed backward pass.
#[derive(Default)]
struct Profiler {
    /// Whether the next step records a profile.
    asked: bool,
    /// Whether `profile` holds the last step's profile.
    valid: bool,
    /// The last profiled step's profile; its vectors are reused.
    profile: StepProfile,
    /// Each layer's output on the profiled step's tape, the marks of the timed backward pass.
    marks: Vec<Var>,
    /// The timed backward pass's times: one per layer, then the loss's.
    times: Vec<Duration>,
}

/// Where the time of one training step went, taken by [`Trainer`] on a step asked for with
/// [`Trainer::profile_next_step`] and read with [`Trainer::last_profile`].
///
/// Every entry is wall-clock time read from [`Instant`], so it includes whatever else the
/// machine did meanwhile; the parts never add up to more than [`total`](Self::total).
///
/// What each entry includes:
/// - `forward[i]`: layer `i`'s forward pass, and its neuron mask when it has one (the mask's
///   condition is recorded as an input and applied with a select, both counted in the layer).
/// - `loss`: recording the loss on the network's output (the targets, when they are values,
///   become an input there) and reading its value.
/// - `backward[i]`: the backward pass through the nodes layer `i` recorded: its operations,
///   its parameters and its mask's nodes. `backward[0]` also covers the batch's input node,
///   recorded just before layer 0. Parameters and inputs take no work in the backward pass
///   beyond being skipped, so they add little.
/// - `backward_loss`: the backward pass through the loss's nodes, from the loss down to the
///   network's output.
/// - `optimizer`: the optimiser's update (0 when the health checks stopped the step before
///   it).
/// - `total`: the whole step, from before the batch's input is recorded to after the
///   gradients are marked stale, including the hooks and the health checks, which no other
///   entry counts.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use cavitation_nn::StepProfile;
/// let profile = StepProfile::default();
/// assert!(profile.forward.is_empty() && profile.total == Duration::ZERO);
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepProfile {
    /// Each layer's forward time, in layer order.
    pub forward: Vec<Duration>,
    /// The time to record the loss and read its value.
    pub loss: Duration,
    /// Each layer's backward time, in layer order.
    pub backward: Vec<Duration>,
    /// The backward time of the loss's nodes.
    pub backward_loss: Duration,
    /// The optimiser's update.
    pub optimizer: Duration,
    /// The whole step.
    pub total: Duration,
}

/// What a [`StepControl`] may change in the model during [`fit_controlled`]: the neuron
/// masks, and nothing else. The trainer has sized its buffers for the model's layers, so the
/// layers themselves stay as they are; a mask changes which neurons pass their output, not
/// any shape.
///
/// A change made between two steps applies from the next step on, and stays on the model
/// after the run.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Rng};
/// use cavitation_nn::{Linear, ModelHandle, Relu, Sequential};
/// let mut params = ParamStore::new();
/// let mut net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
///     .push(Relu);
/// let mut handle = ModelHandle::new(&mut net);
/// handle.set_neuron(0, 1, false);
/// assert_eq!(handle.model().neuron_mask(0).unwrap().off().collect::<Vec<_>>(), [1]);
/// ```
pub struct ModelHandle<'a> {
    /// The model whose masks the handle changes.
    model: &'a mut Sequential,
}

impl<'a> ModelHandle<'a> {
    /// A handle on `model`, for testing a [`StepControl`] outside a run; [`fit_controlled`]
    /// makes its own.
    ///
    /// ```
    /// use cavitation_nn::{ModelHandle, Sequential};
    /// let mut net = Sequential::new();
    /// assert!(ModelHandle::new(&mut net).model().is_empty());
    /// ```
    pub fn new(model: &'a mut Sequential) -> ModelHandle<'a> {
        ModelHandle { model }
    }

    /// The model, to read: its layers, masks and parameter names.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, ModelHandle, Sequential};
    /// let mut params = ParamStore::new();
    /// let mut net = Sequential::new().push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)));
    /// assert_eq!(ModelHandle::new(&mut net).model().len(), 1);
    /// ```
    pub fn model(&self) -> &Sequential {
        self.model
    }

    /// Switches neuron `neuron` of the `Linear` at layer `linear` on or off, as
    /// [`Sequential::set_neuron`] does: the mask goes on
    /// [`neuron_layer`](Sequential::neuron_layer), the last layer of the element-wise run
    /// after the `Linear`.
    ///
    /// # Panics
    ///
    /// Panics as [`Sequential::set_neuron`] does.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, ModelHandle, Sequential, Tanh};
    /// let mut params = ParamStore::new();
    /// let mut net = Sequential::new()
    ///     .push(Linear::new(&mut params, 2, 4, &mut Rng::seed(0)))
    ///     .push(Tanh);
    /// ModelHandle::new(&mut net).set_neuron(0, 3, false);
    /// assert!(net.mask(1).is_some());
    /// ```
    pub fn set_neuron(&mut self, linear: usize, neuron: usize, on: bool) {
        self.model.set_neuron(linear, neuron, on);
    }

    /// Puts `mask` on the neurons of the `Linear` at layer `linear`, as
    /// [`Sequential::set_neuron_mask`] does.
    ///
    /// # Panics
    ///
    /// Panics as [`Sequential::set_neuron_mask`] does.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, ModelHandle, NeuronMask, Sequential};
    /// let mut params = ParamStore::new();
    /// let mut net = Sequential::new().push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)));
    /// ModelHandle::new(&mut net).set_neuron_mask(0, NeuronMask::from_off(3, &[0, 2]));
    /// assert_eq!(net.neuron_mask(0).unwrap().off().collect::<Vec<_>>(), [0, 2]);
    /// ```
    pub fn set_neuron_mask(&mut self, linear: usize, mask: NeuronMask) {
        self.model.set_neuron_mask(linear, mask);
    }

    /// Removes the mask from the neurons of the `Linear` at layer `linear`, as
    /// [`Sequential::clear_neuron_mask`] does.
    ///
    /// # Panics
    ///
    /// Panics as [`Sequential::clear_neuron_mask`] does.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, ModelHandle, Sequential};
    /// let mut params = ParamStore::new();
    /// let mut net = Sequential::new().push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)));
    /// net.set_neuron(0, 1, false);
    /// ModelHandle::new(&mut net).clear_neuron_mask(0);
    /// assert!(net.neuron_mask(0).is_none());
    /// ```
    pub fn clear_neuron_mask(&mut self, linear: usize) {
        self.model.clear_neuron_mask(linear);
    }
}

/// What one step of [`fit_controlled`] did, as its [`StepControl`] sees it.
///
/// # Examples
///
/// ```
/// let info = cavitation_nn::StepInfo { step: 1, epoch: 1, loss: 0.7, rows: 32, seconds: 0.002 };
/// assert_eq!(info.rows as f64 / info.seconds, 16_000.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepInfo {
    /// The step's number in the run, counting from 1 (the trainer's
    /// [`steps`](Trainer::steps) after it).
    pub step: u64,
    /// The epoch the step belongs to, counting from 1.
    pub epoch: usize,
    /// The batch's loss (its mean over the batch's rows).
    pub loss: f32,
    /// The batch's row count; the last batch of an epoch may be smaller.
    pub rows: usize,
    /// The step's wall-clock time in seconds, from before its batch is gathered to after
    /// the training step: the forward and backward passes, the hooks, the health checks and
    /// the optimiser's update. The control's own time (this `after_step` call and earlier
    /// ones) is not included, so `rows / seconds` is the training throughput even while a
    /// control pauses the run.
    pub seconds: f64,
}

/// What a [`StepControl`] may do with the [`Trainer`] of a [`fit_controlled`] run: ask for
/// the next step's profile, read the last one, override the learning rate and read it, and
/// read the step count. Nothing else: a control cannot take steps, end epochs or replace the
/// health checks, so it cannot change the optimiser's state, the step count or the run's
/// epoch accounting.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Rng};
/// use cavitation_nn::{Linear, Loss, Sequential, Sgd, Trainer, TrainerHandle};
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
/// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 4);
/// let mut handle = TrainerHandle::new(&mut trainer);
/// handle.override_learning_rate(0.05);
/// assert_eq!((handle.learning_rate(), handle.steps()), (0.05, 0));
/// ```
pub struct TrainerHandle<'a> {
    /// The trainer the handle acts on, behind the calls a handle allows, so the handle's
    /// type does not carry the trainer's own lifetime.
    trainer: &'a mut (dyn HandleTarget + 'a),
}

/// The calls a [`TrainerHandle`] passes on to its [`Trainer`].
trait HandleTarget {
    /// As [`Trainer::profile_next_step`].
    fn profile_next_step(&mut self);
    /// As [`Trainer::last_profile`].
    fn last_profile(&self) -> Option<&StepProfile>;
    /// As [`Trainer::override_learning_rate`].
    fn override_learning_rate(&mut self, lr: f64);
    /// As [`Trainer::learning_rate`].
    fn learning_rate(&self) -> f64;
    /// As [`Trainer::steps`].
    fn steps(&self) -> u64;
    /// As [`Trainer::skipped_steps`].
    fn skipped_steps(&self) -> u64;
}

impl HandleTarget for Trainer<'_> {
    fn profile_next_step(&mut self) {
        Trainer::profile_next_step(self);
    }
    fn last_profile(&self) -> Option<&StepProfile> {
        Trainer::last_profile(self)
    }
    fn override_learning_rate(&mut self, lr: f64) {
        Trainer::override_learning_rate(self, lr);
    }
    fn learning_rate(&self) -> f64 {
        Trainer::learning_rate(self)
    }
    fn steps(&self) -> u64 {
        Trainer::steps(self)
    }
    fn skipped_steps(&self) -> u64 {
        Trainer::skipped_steps(self)
    }
}

impl<'a> TrainerHandle<'a> {
    /// A handle on `trainer`, for testing a [`StepControl`] outside a run;
    /// [`fit_controlled`] makes its own.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, Loss, Sequential, Sgd, Trainer, TrainerHandle};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 4);
    /// assert!(TrainerHandle::new(&mut trainer).last_profile().is_none());
    /// ```
    pub fn new(trainer: &'a mut Trainer<'_>) -> TrainerHandle<'a> {
        TrainerHandle { trainer }
    }

    /// Asks the next step for a [`StepProfile`], as [`Trainer::profile_next_step`] does.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, Loss, Sequential, Sgd, Trainer, TrainerHandle};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 4);
    /// TrainerHandle::new(&mut trainer).profile_next_step();
    /// // The profile comes from the step that runs next.
    /// assert!(trainer.last_profile().is_none());
    /// ```
    pub fn profile_next_step(&mut self) {
        self.trainer.profile_next_step();
    }

    /// The last step's profile, if it was asked for one, as [`Trainer::last_profile`] gives.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, Loss, Sequential, Sgd, Trainer, TrainerHandle};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 4);
    /// assert!(TrainerHandle::new(&mut trainer).last_profile().is_none());
    /// ```
    pub fn last_profile(&self) -> Option<&StepProfile> {
        self.trainer.last_profile()
    }

    /// Sets the learning rate for every later step and stops the run's schedule, as
    /// [`Trainer::override_learning_rate`] does.
    ///
    /// # Panics
    ///
    /// Panics if `lr` is negative, NaN or infinite. A control that takes the rate from
    /// outside the program checks it first.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Adam, Linear, Loss, Sequential, Trainer, TrainerHandle};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Adam::new(1e-3), Loss::Mse, 4);
    /// TrainerHandle::new(&mut trainer).override_learning_rate(5e-4);
    /// assert_eq!(trainer.learning_rate(), 5e-4);
    /// ```
    pub fn override_learning_rate(&mut self, lr: f64) {
        self.trainer.override_learning_rate(lr);
    }

    /// The learning rate the next step trains at, as [`Trainer::learning_rate`] gives.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, Loss, Sequential, Sgd, Trainer, TrainerHandle};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 4);
    /// assert_eq!(TrainerHandle::new(&mut trainer).learning_rate(), 0.1);
    /// ```
    pub fn learning_rate(&self) -> f64 {
        self.trainer.learning_rate()
    }

    /// The number of steps taken so far in the run, as [`Trainer::steps`] gives.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, Loss, Sequential, Sgd, Trainer, TrainerHandle};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 4);
    /// assert_eq!(TrainerHandle::new(&mut trainer).steps(), 0);
    /// ```
    pub fn steps(&self) -> u64 {
        self.trainer.steps()
    }

    /// How many of those steps were left out under
    /// [`OnNonFinite::Skip`](crate::OnNonFinite::Skip), as [`Trainer::skipped_steps`] gives.
    /// The control does not run after a skipped step, so a jump in [`StepInfo::step`] is the
    /// other sign of one.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Linear, Loss, Sequential, Sgd, Trainer, TrainerHandle};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 4);
    /// assert_eq!(TrainerHandle::new(&mut trainer).skipped_steps(), 0);
    /// ```
    pub fn skipped_steps(&self) -> u64 {
        self.trainer.skipped_steps()
    }
}

/// What runs between the steps of [`fit_controlled`]: after each step and after each epoch,
/// with a [`ModelHandle`] to change the model's neuron masks, the parameters to read, and a
/// [`TrainerHandle`] to ask for the next step's profile, read the last one, or override the
/// learning rate. Both methods do nothing by default, so a control implements only what it
/// needs. Returning [`Flow::Stop`] ends the run.
///
/// A control that only reads, or only asks for profiles, trains to the same parameters as
/// no control at all, bit for bit.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::{
///     Dataset, FitConfig, Flow, Health, Hooks, Linear, Loss, ModelHandle, Sequential,
///     StepControl, StepInfo, Targets, TrainerHandle, fit_controlled,
/// };
/// /// Stops once the loss is below a threshold.
/// struct StopBelow(f32);
/// impl StepControl for StopBelow {
///     fn after_step(
///         &mut self,
///         info: &StepInfo,
///         _model: &mut ModelHandle<'_>,
///         _params: &ParamStore,
///         _trainer: &mut TrainerHandle<'_>,
///     ) -> Flow {
///         if info.loss < self.0 { Flow::Stop } else { Flow::Continue }
///     }
/// }
/// let x = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], &[4, 1]).unwrap();
/// let y = Tensor::from_vec(vec![1.0f32, 3.0, 5.0, 7.0], &[4, 1]).unwrap();
/// let data = Dataset { x, targets: Targets::Values(y) };
/// let mut params = ParamStore::new();
/// let mut net = Sequential::new().push(Linear::new(&mut params, 1, 1, &mut Rng::seed(0)));
/// let config = FitConfig {
///     epochs: 10_000,
///     batch: 4,
///     loss: Loss::Mse,
///     health: Health::quiet(),
///     ..FitConfig::default()
/// };
/// let mut control = StopBelow(0.01);
/// let history =
///     fit_controlled(&mut net, &mut params, &data, config, &mut Hooks::new(), &mut control)
///         .unwrap();
/// assert!(history.epochs.len() < 10_000);
/// ```
pub trait StepControl {
    /// Runs after each step that the health checks passed, before the next step; not after
    /// a step left out under [`OnNonFinite::Skip`](crate::OnNonFinite::Skip). Changes to
    /// the masks through `model` apply from the next step, and so does a learning rate
    /// overridden through `trainer`.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{
    ///     Flow, Linear, Loss, ModelHandle, Sequential, Sgd, StepControl, StepInfo, Trainer,
    ///     TrainerHandle,
    /// };
    /// struct Nothing;
    /// impl StepControl for Nothing {}
    /// let mut params = ParamStore::new();
    /// let mut net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 4);
    /// let info = StepInfo { step: 1, epoch: 1, loss: 0.5, rows: 4, seconds: 0.001 };
    /// let mut model = ModelHandle::new(&mut net);
    /// let mut handle = TrainerHandle::new(&mut trainer);
    /// let flow = Nothing.after_step(&info, &mut model, &params, &mut handle);
    /// assert_eq!(flow, Flow::Continue);
    /// ```
    fn after_step(
        &mut self,
        info: &StepInfo,
        model: &mut ModelHandle<'_>,
        params: &ParamStore,
        trainer: &mut TrainerHandle<'_>,
    ) -> Flow {
        let _ = (info, model, params, trainer);
        Flow::Continue
    }

    /// Runs after each whole epoch, after [`FitConfig::on_epoch`], with the epoch's report.
    /// A learning rate overridden here applies from the next step, the first of the next
    /// epoch, and the schedule no longer sets it.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{
    ///     EpochReport, Flow, Linear, Loss, ModelHandle, Sequential, Sgd, StepControl, Trainer,
    ///     TrainerHandle,
    /// };
    /// struct Nothing;
    /// impl StepControl for Nothing {}
    /// let mut params = ParamStore::new();
    /// let mut net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 4);
    /// let report =
    ///     EpochReport { epoch: 1, mean_loss: 0.5, rows: 4, seconds: 0.01, validation: None };
    /// let mut model = ModelHandle::new(&mut net);
    /// let mut handle = TrainerHandle::new(&mut trainer);
    /// let flow = Nothing.after_epoch(&report, &mut model, &params, &mut handle);
    /// assert_eq!(flow, Flow::Continue);
    /// ```
    fn after_epoch(
        &mut self,
        report: &EpochReport,
        model: &mut ModelHandle<'_>,
        params: &ParamStore,
        trainer: &mut TrainerHandle<'_>,
    ) -> Flow {
        let _ = (report, model, params, trainer);
        Flow::Continue
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
    /// [`RmsProp`], with momentum and weight decay (0 turns either off), and eps 1e-8.
    RmsProp {
        /// The learning rate.
        lr: f64,
        /// The decay rate of the squared-gradient average (PyTorch's default is 0.99).
        alpha: f64,
        /// The momentum factor.
        momentum: f64,
        /// The weight decay added to the gradient.
        weight_decay: f64,
    },
    /// [`Adagrad`], with sums starting at 0 and eps 1e-10.
    Adagrad {
        /// The learning rate.
        lr: f64,
        /// The rate's decay per update.
        lr_decay: f64,
        /// The weight decay added to the gradient.
        weight_decay: f64,
    },
}

impl OptimizerConfig {
    /// Builds the optimiser, with the default betas and epsilon for Adam and AdamW and the
    /// default epsilons for RMSprop and Adagrad.
    ///
    /// # Panics
    ///
    /// Panics as [`RmsProp::with_alpha`] does for an `alpha` outside `[0, 1]`.
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
            OptimizerConfig::RmsProp {
                lr,
                alpha,
                momentum,
                weight_decay,
            } => Box::new(
                RmsProp::new(lr)
                    .with_alpha(alpha)
                    .with_momentum(momentum)
                    .with_weight_decay(weight_decay),
            ),
            OptimizerConfig::Adagrad {
                lr,
                lr_decay,
                weight_decay,
            } => Box::new(
                Adagrad::new(lr)
                    .with_lr_decay(lr_decay)
                    .with_weight_decay(weight_decay),
            ),
        }
    }
}

/// How [`fit`] changes the learning rate from step to step. The rate the optimiser is
/// configured with is the schedule's peak: the rate of the first epoch, or the rate the
/// warm-up climbs to.
///
/// [`Constant`](Self::Constant) and [`Cosine`](Self::Cosine) set the rate before the first
/// step of each epoch; [`Warmup`](Self::Warmup) and [`WarmupCosine`](Self::WarmupCosine)
/// set it before every step. `fit` takes the rate from
/// [`learning_rate`](Self::learning_rate), counting every step position of the run (a step
/// left out under [`OnNonFinite::Skip`](crate::OnNonFinite::Skip) still counts), until a
/// [`StepControl`] overrides the rate.
///
/// # Examples
///
/// ```
/// use cavitation_nn::{FitConfig, LrSchedule};
/// let config = FitConfig {
///     schedule: LrSchedule::WarmupCosine { warmup_steps: 100, min_lr: 0.0 },
///     ..FitConfig::default()
/// };
/// assert_ne!(config.schedule, LrSchedule::Constant);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum LrSchedule {
    /// Every step trains at the configured rate.
    #[default]
    Constant,
    /// Cosine annealing from the configured rate down towards `min_lr`, set once at the
    /// start of each epoch, as PyTorch's `CosineAnnealingLR(optimizer, T_max=epochs,
    /// eta_min=min_lr)` stepped once per epoch: epoch `e` (counting from 0) of `epochs`
    /// trains at `min_lr + (lr − min_lr)·(1 + cos(π·e/epochs))/2`.
    Cosine {
        /// The rate the schedule approaches; the last epoch trains a little above it.
        min_lr: f64,
    },
    /// A linear warm-up over the first `steps` steps, then the configured rate: step `s`
    /// (counting from 0) of the warm-up trains at `lr·(s + 1)/steps`, so the first step
    /// trains at `lr/steps` and step `steps − 1` at `lr`. `steps` of 0 is
    /// [`Constant`](Self::Constant).
    Warmup {
        /// How many steps the warm-up takes.
        steps: usize,
    },
    /// The linear warm-up of [`Warmup`](Self::Warmup) over `warmup_steps` steps, then cosine
    /// annealing by step from the configured rate towards `min_lr` over the rest of the
    /// run: step `s ≥ warmup_steps` of `total` trains at
    /// `min_lr + (lr − min_lr)·(1 + cos(π·t/n))/2` with `t = s − warmup_steps` and
    /// `n = total − warmup_steps`. With `warmup_steps` of 0 it is cosine annealing by step,
    /// as PyTorch's `CosineAnnealingLR(optimizer, T_max=total, eta_min=min_lr)` stepped after
    /// every batch.
    WarmupCosine {
        /// How many steps the warm-up takes.
        warmup_steps: usize,
        /// The rate the schedule approaches; the last step trains a little above it.
        min_lr: f64,
    },
}

impl LrSchedule {
    /// The learning rate of step `step` (counting from 0, the number of step positions
    /// already run) of a run of `epochs` epochs of `steps_per_epoch` steps each, whose peak
    /// rate is `initial`. For the per-epoch schedules the step's epoch is
    /// `step / steps_per_epoch`.
    ///
    /// # Panics
    ///
    /// Panics if `epochs` or `steps_per_epoch` is 0.
    ///
    /// ```
    /// use cavitation_nn::LrSchedule;
    /// let cosine = LrSchedule::Cosine { min_lr: 0.0 };
    /// // Epoch 2 of 4, at its first and last step of 10.
    /// assert_eq!(cosine.learning_rate(1e-3, 20, 10, 4), 5e-4);
    /// assert_eq!(cosine.learning_rate(1e-3, 29, 10, 4), 5e-4);
    /// let warm = LrSchedule::Warmup { steps: 4 };
    /// assert_eq!(warm.learning_rate(1e-3, 0, 10, 4), 2.5e-4);
    /// assert_eq!(warm.learning_rate(1e-3, 3, 10, 4), 1e-3);
    /// let by_step = LrSchedule::WarmupCosine { warmup_steps: 0, min_lr: 0.0 };
    /// assert_eq!(by_step.learning_rate(1e-3, 20, 10, 4), 5e-4);
    /// assert_eq!(LrSchedule::Constant.learning_rate(1e-3, 3, 10, 4), 1e-3);
    /// ```
    pub fn learning_rate(
        &self,
        initial: f64,
        step: usize,
        steps_per_epoch: usize,
        epochs: usize,
    ) -> f64 {
        assert!(epochs > 0, "LrSchedule: a run needs at least one epoch");
        assert!(
            steps_per_epoch > 0,
            "LrSchedule: an epoch needs at least one step"
        );
        let cosine = |min_lr: f64, done: usize, total: usize| {
            // PyTorch's order of operations: π·t first, then the division.
            let angle = std::f64::consts::PI * done as f64 / total as f64;
            min_lr + (initial - min_lr) * (1.0 + angle.cos()) / 2.0
        };
        let warmup = |steps: usize| initial * (step + 1) as f64 / steps as f64;
        match *self {
            LrSchedule::Constant => initial,
            LrSchedule::Cosine { min_lr } => cosine(min_lr, step / steps_per_epoch, epochs),
            LrSchedule::Warmup { steps } if step < steps => warmup(steps),
            LrSchedule::Warmup { .. } => initial,
            LrSchedule::WarmupCosine { warmup_steps, .. } if step < warmup_steps => {
                warmup(warmup_steps)
            }
            LrSchedule::WarmupCosine {
                warmup_steps,
                min_lr,
            } => cosine(
                min_lr,
                step - warmup_steps,
                (steps_per_epoch * epochs)
                    .saturating_sub(warmup_steps)
                    .max(1),
            ),
        }
    }
}

/// What one epoch of [`fit`] did.
///
/// # Examples
///
/// ```
/// use cavitation_nn::EpochReport;
/// let report =
///     EpochReport { epoch: 1, mean_loss: 0.5, rows: 100, seconds: 0.01, validation: None };
/// assert_eq!(report.epoch, 1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpochReport {
    /// The epoch's number, counting from 1.
    pub epoch: usize,
    /// The mean loss over the epoch's rows: each batch's loss weighted by its row count. A
    /// step left out under [`OnNonFinite::Skip`](crate::OnNonFinite::Skip) does not count;
    /// an epoch in which every step was left out gets no report, as it ends the run with
    /// [`Error::NonFinite`].
    ///
    /// A PyTorch loop that averages the batch means gives a slightly different figure when the
    /// last batch is short, because it counts that batch as much as a full one. This loop
    /// gives the same figure as `mean_loss`:
    ///
    /// ```python
    /// total, rows = 0.0, 0
    /// for x, y in loader:
    ///     loss = criterion(model(x), y)  # mean over the batch's rows
    ///     total += loss.item() * len(x)
    ///     rows += len(x)
    ///     # ... backward and step ...
    /// mean_loss = total / rows
    /// ```
    pub mean_loss: f64,
    /// The rows the epoch trained on: every row of the data, less the rows of each step left
    /// out under [`OnNonFinite::Skip`](crate::OnNonFinite::Skip), and for the last epoch of a
    /// [`fit_controlled`] run that its control stopped before the epoch's last step, only the
    /// rows of the steps that ran (`mean_loss` is then their mean).
    pub rows: usize,
    /// The epoch's wall-clock time in seconds, from before its first batch to after its
    /// end-of-epoch health checks: every step with its hooks and health checks and, under
    /// [`fit_controlled`], every [`StepControl::after_step`] call, including any time a
    /// control spends pausing the run. It does not include [`FitConfig::on_epoch`] or
    /// [`StepControl::after_epoch`]. Summing [`StepInfo::seconds`] over the epoch gives the
    /// training time without the control's. It does not include the validation either.
    pub seconds: f64,
    /// The evaluation of [`FitConfig::validation`] after the epoch's updates, with the run's
    /// loss, at the run's batch size: what [`evaluate`](crate::evaluate) measures on the
    /// model and parameters as the epoch left them, neuron masks included. `None` when the
    /// run has no validation set, and for the last epoch of a [`fit_controlled`] run that its
    /// control stopped before the epoch's last step.
    pub validation: Option<Evaluation>,
}

/// Every epoch's report from one [`fit`].
///
/// # Examples
///
/// ```
/// use cavitation_nn::{EpochReport, History};
/// let report =
///     EpochReport { epoch: 1, mean_loss: 0.5, rows: 100, seconds: 0.01, validation: None };
/// let history = History { epochs: vec![report] };
/// assert_eq!(history.epochs.len(), 1);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct History {
    /// The reports, in epoch order.
    pub epochs: Vec<EpochReport>,
}

/// How [`fit`] trains. Taken by value, because it holds the `on_epoch` callback.
///
/// The default is 10 epochs in batches of 32, Adam at a constant learning rate of 1e-3
/// without weight decay, cross-entropy, shuffled with seed 0, the default [`Health`] checks,
/// no validation set and no callback.
///
/// # Examples
///
/// ```
/// use cavitation_nn::{FitConfig, Loss};
/// let config = FitConfig { epochs: 3, loss: Loss::Mse, ..FitConfig::default() };
/// assert_eq!((config.epochs, config.batch), (3, 32));
/// ```
pub struct FitConfig<'s> {
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
    /// How the learning rate changes from step to step or epoch to epoch, with the
    /// optimiser's rate as its peak.
    pub schedule: LrSchedule,
    /// The health checks run on every step and at the end of every epoch.
    pub health: Health<'s>,
    /// Called as soon as each epoch ends with its report, the model and the parameters as
    /// they are after the epoch, so a held-out metric such as [`accuracy`] can be measured
    /// every epoch.
    // The boxed callback is the plainest type for this field; an alias would only hide it.
    #[allow(clippy::type_complexity)]
    pub on_epoch: Option<Box<dyn FnMut(&EpochReport, &Sequential, &ParamStore) + Send + 's>>,
    /// A held-out set measured after every whole epoch, recorded in
    /// [`EpochReport::validation`] before `on_epoch` runs: the loss with
    /// [`loss`](Self::loss), and the accuracy (class targets) or R² (value targets), `batch`
    /// rows at a time. One [`Evaluator`] is made before training and kept for the run; after
    /// each epoch it copies in the new parameter values and the model's neuron masks (which
    /// a [`StepControl`] may have changed), in place, so from the second epoch on the
    /// validation allocates nothing for a model that compiles to an [`InferenceModel`]. A
    /// model with a layer that has no inference form is measured on the tape and copies its
    /// parameter values each epoch.
    pub validation: Option<Dataset>,
}

impl Default for FitConfig<'_> {
    /// 10 epochs, batch 32, Adam at a constant 1e-3, cross-entropy, shuffle seed 0, the
    /// default health checks, no callback, no validation set.
    fn default() -> Self {
        FitConfig {
            epochs: 10,
            batch: 32,
            optimizer: OptimizerConfig::Adam {
                lr: 1e-3,
                weight_decay: 0.0,
            },
            loss: Loss::CrossEntropy,
            shuffle_seed: Some(0),
            schedule: LrSchedule::Constant,
            health: Health::default(),
            on_epoch: None,
            validation: None,
        }
    }
}

impl fmt::Debug for FitConfig<'_> {
    /// Writes every setting; the callback shows only whether there is one, and the
    /// validation set only its rows and features.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FitConfig")
            .field("epochs", &self.epochs)
            .field("batch", &self.batch)
            .field("optimizer", &self.optimizer)
            .field("loss", &self.loss)
            .field("shuffle_seed", &self.shuffle_seed)
            .field("schedule", &self.schedule)
            .field("health", &self.health)
            .field("on_epoch", &self.on_epoch.as_ref().map(|_| "FnMut"))
            .field(
                "validation",
                &self.validation.as_ref().map(|v| (v.len(), v.features())),
            )
            .finish()
    }
}

/// Checks that `data` can be trained on: `x` a matrix with at least one row, and one target
/// per row.
pub(crate) fn check_dataset(data: &Dataset) -> Result<(), Error> {
    let &[rows, _] = data.x.shape().dims() else {
        return Err(Error::InvalidData(format!(
            "the inputs must have shape [rows, features], found {:?}",
            data.x.shape().dims()
        )));
    };
    if rows == 0 {
        return Err(Error::InvalidData("the data set has no rows".to_string()));
    }
    match &data.targets {
        Targets::Classes(labels) if labels.len() != rows => Err(Error::InvalidData(format!(
            "there are {rows} rows but {} labels",
            labels.len()
        ))),
        Targets::Values(t) => match t.shape().dims() {
            &[r, _] if r == rows => Ok(()),
            &[_, _] => Err(Error::InvalidData(format!(
                "the value targets are {:?}, but the inputs are {:?}",
                t.shape().dims(),
                data.x.shape().dims()
            ))),
            dims => Err(Error::InvalidData(format!(
                "the value targets must have shape [rows, outputs], found {dims:?}"
            ))),
        },
        Targets::Classes(_) => Ok(()),
    }
}

/// Checks, before any training, that `model` and `loss` suit `data` (already passed by
/// [`check_dataset`]): the kind of targets is the kind `loss` needs, binary cross-entropy
/// targets are in `[0, 1]` (neither needs the output width, so both run even for a model
/// whose layers disagree), the inputs are as wide as each layer that names an input width
/// expects, every class label is below the
/// model's output width, and value targets are as wide as the model's output.
pub(super) fn check_model_and_loss(
    model: &Sequential,
    loss: Loss,
    data: &Dataset,
) -> Result<(), Error> {
    let kind = |classes: bool| if classes { "class" } else { "value" };
    let has_classes = matches!(data.targets, Targets::Classes(_));
    if loss.needs_classes() != has_classes {
        return Err(Error::InvalidData(format!(
            "Loss::{loss:?} needs {} targets, but the data has {} targets",
            kind(loss.needs_classes()),
            kind(has_classes)
        )));
    }
    if matches!(
        loss,
        Loss::BinaryCrossEntropy | Loss::BinaryCrossEntropyWithLogits
    ) && let Targets::Values(t) = &data.targets
        && let &[_, columns] = t.shape().dims()
        && let Some(i) = t.data().iter().position(|v| !(0.0..=1.0).contains(v))
    {
        let value = t.data()[i];
        return Err(Error::InvalidData(format!(
            "row {}, column {}: the target is {value}, but Loss::{loss:?} needs targets in [0, 1]",
            i / columns,
            i % columns
        )));
    }
    check_widths(model, data)
}

/// Checks, before any work, that `data` (already passed by [`check_dataset`]) suits the
/// widths of `model`: the inputs are as wide as each layer that names an input width
/// expects, every class label is below the model's output width, and value targets are as
/// wide as the model's output. Two layers that disagree skip the checks that need the output
/// width.
pub(crate) fn check_widths(model: &Sequential, data: &Dataset) -> Result<(), Error> {
    let (rows, mut width) = (data.len(), data.features());
    // Whether an earlier layer named an input width, so that `width` is checked against it.
    let mut named = false;
    for i in 0..model.len() {
        let layer = model.layer(i);
        if let Some(expected) = layer.input_width() {
            if expected != width {
                if !named {
                    let reach = if width == data.features() {
                        String::new()
                    } else {
                        format!(" and reach layer {i} with width {width}")
                    };
                    return Err(Error::InvalidData(format!(
                        "the inputs are [{rows}, {}]{reach}, but layer {i} ({}) takes {expected}",
                        data.features(),
                        layer.name()
                    )));
                }
                // Two layers disagree: a model error, which the forward pass reports with
                // both layers named. The output width is unknown, so the checks that need it
                // (the labels and the value-target width) are skipped.
                return Ok(());
            }
            named = true;
        }
        width = layer.output_width(width);
    }
    match &data.targets {
        Targets::Classes(labels) => match labels.iter().position(|&l| l >= width) {
            Some(row) => Err(Error::InvalidData(format!(
                "the label of row {row} is {}, but the model has {width} outputs (classes 0..{width})",
                labels[row]
            ))),
            None => Ok(()),
        },
        Targets::Values(t) => match t.shape().dims() {
            dims @ &[_, outputs] if outputs != width => Err(Error::InvalidData(format!(
                "the value targets are {dims:?}, but the model outputs [{rows}, {width}]"
            ))),
            _ => Ok(()),
        },
    }
}

/// Checks that every input and every value target is finite, naming the first one that is
/// not by tensor, row, column and value. Allocates only to build the error.
fn check_finite(data: &Dataset) -> Result<(), Error> {
    let scan = |values: &[f32], columns: usize, what: &str| match values
        .iter()
        .position(|v| !v.is_finite())
    {
        Some(i) => Err(Error::InvalidData(format!(
            "row {}, column {} of {what} is {}; training needs finite values",
            i / columns,
            i % columns,
            values[i]
        ))),
        None => Ok(()),
    };
    scan(data.x.data(), data.features().max(1), "the inputs")?;
    if let Targets::Values(t) = &data.targets {
        let columns = t.shape().dims().last().copied().unwrap_or(1).max(1);
        scan(t.data(), columns, "the value targets")?;
    }
    Ok(())
}

/// Trains `model` on `train` as `config` says and returns each epoch's report.
///
/// The optimiser's learning rate follows `config.schedule`, set before each step or each
/// epoch as [`LrSchedule`] says. Each epoch reshuffles the rows (with
/// `config.shuffle_seed`), takes one [`Trainer::step`] per batch, ends the epoch of the health
/// checks ([`Trainer::end_epoch`]), measures `config.validation` when there is one
/// ([`EpochReport::validation`]) and then calls `config.on_epoch`. The validation only reads the
/// parameters, so it does not change what the run trains to. The same seeds give
/// bit-identical parameters on every run on the same machine and kernel level, for any thread
/// count.
///
/// # Errors
///
/// Returns, before training, [`Error::InvalidData`] if `train.x` is not a matrix or has no rows, if
/// value targets are not a matrix, if the kind of targets is not the kind `config.loss` needs (see
/// [`Loss`]), if a class label is not below the model's output width, or, for
/// [`Loss::BinaryCrossEntropy`] and [`Loss::BinaryCrossEntropyWithLogits`], if a target is NaN,
/// infinite or outside `[0, 1]` (the message names its row and column), if the targets do not have
/// one entry per row, if value targets are not as wide as the model's output, if the inputs are not
/// as wide as the first layer that names an input width expects (layers that do not, such as
/// activations, may come before it; the message then gives the width that reaches the layer), or if
/// any input or value target is NaN or infinite (the message names the row, the column and the
/// value). Each message names the tensors with their shapes, such as "the inputs are [8, 5]".
/// `config.validation` goes through the same checks, before training, and a message about it
/// starts with `validation: `. Two layers that disagree on a width are a model error, not a data
/// error: the checks that need the output width are skipped and the first forward pass panics
/// naming both layers. During training,
/// with `config.health.on_non_finite` set to [`OnNonFinite::Stop`](crate::OnNonFinite) (the
/// default), returns [`Error::NonFinite`] at the first step whose loss or gradient is NaN or
/// infinite, naming the step and `"loss"` or the parameter (for example `"0.weight"`), after
/// sending the matching warning to the health sink. A non-finite loss stops before the optimiser
/// step; a non-finite gradient with a finite loss is found after the update applied it. With
/// [`OnNonFinite::Skip`](crate::OnNonFinite), such a step is left out instead and training
/// continues (see [`EpochReport::rows`]), but an epoch in which every step was left out ends
/// the run with [`Error::NonFinite`] naming the epoch, since the parameters did not change.
///
/// # Panics
///
/// Panics if `config.batch` is 0, and as [`Trainer::step`] does when the layers' shapes do
/// not fit one another.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::{
///     Dataset, FitConfig, Linear, Loss, OptimizerConfig, Sequential, Targets, fit,
/// };
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
/// use cavitation_nn::{
///     Dataset, FitConfig, Hooks, Linear, Loss, Sequential, Targets, fit_with_hooks,
/// };
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
    config: FitConfig,
    hooks: &mut Hooks<'_>,
) -> Result<History, Error> {
    run_fit(Run::Plain(model), params, train, config, hooks)
}

/// Like [`fit_with_hooks`], and runs `control` between steps: [`StepControl::after_step`]
/// after every step and [`StepControl::after_epoch`] after every epoch (after
/// `config.on_epoch`). Through a [`ModelHandle`] the control can switch neurons and change or
/// clear neuron masks, which apply from the next step and stay on `model` after the run;
/// through a [`TrainerHandle`] it can ask for the next step's [`StepProfile`] or override the
/// learning rate, which stops `config.schedule` for the rest of the run. With a control that
/// changes nothing, the same seeds train to the same parameters as [`fit`], bit for bit.
///
/// Each step reads the clock twice for [`StepInfo::seconds`], which leaves out the control's
/// own time; [`fit`] and [`fit_with_hooks`] run the same loop without a control and read no
/// clock per step. [`EpochReport::seconds`] is the epoch's wall-clock time and includes the
/// `after_step` calls, so a control that pauses the run lengthens it.
///
/// A control that returns [`Flow::Stop`] ends the run there and gets the history so far:
/// - From [`after_epoch`](StepControl::after_epoch): the history ends with that epoch.
/// - From [`after_step`](StepControl::after_step) on an epoch's last step: the epoch is
///   whole, so its end-of-epoch health checks and `config.on_epoch` run as usual and the
///   history ends with its report; `after_epoch` does not run for it, since the control has
///   already stopped the run.
/// - From `after_step` on any other step: the history ends with a report for the partial
///   epoch, whose [`rows`](EpochReport::rows) is below the data's row count and whose
///   `mean_loss` is the mean over those rows (each batch's loss weighted by its row count);
///   its `seconds` runs up to the stop. Its end-of-epoch health checks, `config.on_epoch` and
///   `after_epoch` do not run, as the epoch did not end.
///
/// # Errors
///
/// Returns the same errors as [`fit`]. A step that meets a health-check error ends the run
/// with that error before the control sees it.
///
/// # Panics
///
/// Panics as [`fit_with_hooks`] does, and as the [`ModelHandle`] and [`TrainerHandle`]
/// methods do when the control addresses a layer or neuron the model does not have or
/// overrides the learning rate with one that is negative or not finite.
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::{
///     Dataset, FitConfig, Flow, Hooks, Linear, Loss, ModelHandle, Sequential, StepControl,
///     StepInfo, Tanh, Targets, TrainerHandle, fit_controlled,
/// };
/// /// Switches neuron 0 of the first `Linear` off after step 2.
/// struct SwitchOff;
/// impl StepControl for SwitchOff {
///     fn after_step(
///         &mut self,
///         info: &StepInfo,
///         model: &mut ModelHandle<'_>,
///         _params: &ParamStore,
///         _trainer: &mut TrainerHandle<'_>,
///     ) -> Flow {
///         if info.step == 2 {
///             model.set_neuron(0, 0, false);
///         }
///         Flow::Continue
///     }
/// }
/// let labels = vec![0, 1, 0, 1, 0, 1, 0, 1];
/// let data = Dataset { x: Tensor::full(&[8, 2], 0.5f32), targets: Targets::Classes(labels) };
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let mut net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 4, &mut rng))
///     .push(Tanh)
///     .push(Linear::new(&mut params, 4, 2, &mut rng));
/// let config = FitConfig { epochs: 2, batch: 4, ..FitConfig::default() };
/// let mut hooks = Hooks::new();
/// let history =
///     fit_controlled(&mut net, &mut params, &data, config, &mut hooks, &mut SwitchOff).unwrap();
/// assert_eq!(history.epochs.len(), 2);
/// assert_eq!(net.neuron_mask(0).unwrap().off().collect::<Vec<_>>(), [0]);
/// ```
pub fn fit_controlled(
    model: &mut Sequential,
    params: &mut ParamStore,
    train: &Dataset,
    config: FitConfig,
    hooks: &mut Hooks<'_>,
    control: &mut dyn StepControl,
) -> Result<History, Error> {
    run_fit(
        Run::Controlled(model, control),
        params,
        train,
        config,
        hooks,
    )
}

/// The loop behind [`fit`], [`fit_with_hooks`] and [`fit_controlled`]: checks the data, then
/// trains epoch by epoch, running the control between steps and epochs when `run` has one.
fn run_fit(
    mut run: Run<'_, '_>,
    params: &mut ParamStore,
    train: &Dataset,
    mut config: FitConfig,
    hooks: &mut Hooks<'_>,
) -> Result<History, Error> {
    check_dataset(train)?;
    check_model_and_loss(run.model(), config.loss, train)?;
    check_finite(train)?;
    let validation_set = config.validation.take();
    if let Some(held_out) = &validation_set {
        let named = |e: Error| match e {
            Error::InvalidData(msg) => Error::InvalidData(format!("validation: {msg}")),
            other => other,
        };
        check_dataset(held_out).map_err(named)?;
        check_model_and_loss(run.model(), config.loss, held_out).map_err(named)?;
        check_finite(held_out).map_err(named)?;
    }
    let mut trainer = Trainer::with_boxed(
        run.model(),
        config.optimizer.build(),
        config.loss,
        config.batch,
    );
    // Made after the trainer, which panics on a batch size of 0 before this could return
    // its error for it.
    let mut evaluator = (validation_set.is_some())
        .then(|| Evaluator::new(run.model(), params, config.loss, config.batch))
        .transpose()?;
    trainer.set_health(config.health);
    let mut batcher = Batcher::new(train.len(), config.batch, config.shuffle_seed);
    let mut history = History {
        epochs: Vec::with_capacity(config.epochs.min(1024)),
    };
    params.zero_grad();
    trainer.rates = Some(StepRates {
        schedule: config.schedule,
        initial: trainer.optimizer.learning_rate(),
        steps_per_epoch: batcher.batches().count(),
        epochs: config.epochs,
    });
    for epoch in 1..=config.epochs {
        let start = Instant::now();
        let end = trainer.run_epoch(&mut run, params, train, &mut batcher, hooks, epoch)?;
        let whole = end.whole;
        if whole {
            trainer.end_epoch(epoch)?;
        }
        let seconds = start.elapsed().as_secs_f64();
        let validation = match (&mut evaluator, &validation_set) {
            (Some(evaluator), Some(held_out)) if whole => {
                evaluator.refresh(run.model(), params)?;
                Some(evaluator.evaluate(held_out)?)
            }
            _ => None,
        };
        let report = EpochReport {
            epoch,
            mean_loss: end.mean_loss,
            rows: end.rows,
            seconds,
            validation,
        };
        if whole && let Some(on_epoch) = config.on_epoch.as_mut() {
            on_epoch(&report, run.model(), params);
        }
        history.epochs.push(report);
        if end.stopped {
            break;
        }
        if let Run::Controlled(model, control) = &mut run {
            let mut handle = ModelHandle::new(model);
            let mut trainer = TrainerHandle::new(&mut trainer);
            if control.after_epoch(&report, &mut handle, params, &mut trainer) == Flow::Stop {
                break;
            }
        }
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
/// [`evaluate`](crate::evaluate) measures the same accuracy together with the loss, and
/// returns [`Error::InvalidData`](cavitation_core::Error::InvalidData) for unsuitable data
/// where this function panics; an [`Evaluator`](crate::Evaluator) measures again without
/// compiling again.
///
/// # Panics
///
/// Panics if `data` has value targets rather than class labels, if it has no rows, if the
/// labels are not one per row, if a label is not below the model's output width, or if
/// `batch` is 0; and as [`InferenceModel::compile`] and
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
            for (i, (row, &label)) in scores.chunks(classes).zip(&labels[start..end]).enumerate() {
                assert!(
                    label < classes,
                    "accuracy: the label of row {} is {label}, but the model has {classes} outputs (classes 0..{classes})",
                    start + i
                );
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
pub(super) fn argmax(row: &[f32]) -> Option<usize> {
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
    use crate::{Glu, Layer, Linear, Sigmoid, Tanh, Warning};
    use cavitation_core::{Gate, ParamId, Rng, Tensor};

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
            schedule: LrSchedule::Constant,
            health: Health::default(),
            on_epoch: None,
            validation: None,
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
            let loss = trainer
                .step(&net, &mut params, batcher.gather(&data, range, &mut batch))
                .unwrap();
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
            assert!(trainer.step(&net, &mut params, view).unwrap().is_finite());
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
        let _ = trainer.step(&net, &mut params, view);
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
        let _ = trainer.step(&net, &mut params, view);
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
            let _ = trainer.step_with_hooks(&net, &mut params, view, &mut hooks);
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
        assert_eq!(
            err.to_string(),
            "invalid data: there are 4 rows but 3 labels"
        );
        let values = Dataset {
            x: Tensor::zeros(&[4, 3]),
            targets: Targets::Values(Tensor::zeros(&[5, 2])),
        };
        let err = fit(&net, &mut params, &values, FitConfig::default()).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid data: the value targets are [5, 2], but the inputs are [4, 3]"
        );
        let empty = Dataset {
            x: Tensor::zeros(&[0, 3]),
            targets: Targets::Classes(vec![]),
        };
        assert!(fit(&net, &mut params, &empty, FitConfig::default()).is_err());
        assert_eq!(parameter_bits(&net, &params), before);
    }

    /// `fit` returns an error, before training and without panicking, when the targets do
    /// not suit the loss, a class label is not below the model's output width, value targets
    /// are not as wide as the model's output, or the inputs are not as wide as the first
    /// layer expects.
    #[test]
    fn fit_rejects_data_that_does_not_suit_the_loss_or_the_model() {
        let mut params = ParamStore::new();
        let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
        let before = parameter_bits(&net, &params);
        let mse = || FitConfig {
            loss: Loss::Mse,
            ..FitConfig::default()
        };
        let cases: [(Dataset, FitConfig, &str); 5] = [
            (
                Dataset {
                    x: Tensor::zeros(&[4, 3]),
                    targets: Targets::Values(Tensor::zeros(&[4, 2])),
                },
                FitConfig::default(),
                "invalid data: Loss::CrossEntropy needs class targets, but the data has value targets",
            ),
            (
                classes_dataset(4),
                mse(),
                "invalid data: Loss::Mse needs value targets, but the data has class targets",
            ),
            (
                Dataset {
                    x: Tensor::zeros(&[4, 3]),
                    targets: Targets::Classes(vec![0, 1, 2, 1]),
                },
                FitConfig::default(),
                "invalid data: the label of row 2 is 2, but the model has 2 outputs (classes 0..2)",
            ),
            (
                Dataset {
                    x: Tensor::zeros(&[4, 3]),
                    targets: Targets::Values(Tensor::zeros(&[4, 1])),
                },
                mse(),
                "invalid data: the value targets are [4, 1], but the model outputs [4, 2]",
            ),
            (
                Dataset {
                    x: Tensor::zeros(&[4, 5]),
                    targets: Targets::Classes(vec![0; 4]),
                },
                FitConfig::default(),
                "invalid data: the inputs are [4, 5], but layer 0 (linear) takes 3",
            ),
        ];
        for (data, config, message) in cases {
            let result = catch_unwind(AssertUnwindSafe(|| fit(&net, &mut params, &data, config)));
            let err = result
                .expect("fit returns the error instead of panicking")
                .unwrap_err();
            assert_eq!(err.to_string(), message);
        }
        assert_eq!(parameter_bits(&net, &params), before);
    }

    /// A one-column value data set with the given targets and zero inputs.
    fn unit_targets(targets: Vec<f32>) -> Dataset<f32> {
        let rows = targets.len();
        Dataset {
            x: Tensor::zeros(&[rows, 1]),
            targets: Targets::Values(Tensor::from_vec(targets, &[rows, 1]).unwrap()),
        }
    }

    /// A one-input, one-output network.
    fn one_to_one(params: &mut ParamStore) -> Sequential {
        Sequential::new().push(Linear::new(params, 1, 1, &mut Rng::seed(0)))
    }

    /// A one-input, one-output network that ends in a sigmoid, so its output is a probability.
    fn one_to_one_probability(params: &mut ParamStore) -> Sequential {
        one_to_one(params).push(Sigmoid)
    }

    /// `fit` config for `loss` with two epochs.
    fn config_for(loss: Loss) -> FitConfig<'static> {
        FitConfig {
            epochs: 2,
            batch: 2,
            loss,
            ..FitConfig::default()
        }
    }

    /// A binary cross-entropy target outside `[0, 1]` is a data error naming its row, column
    /// and value, for both binary losses.
    #[test]
    fn bce_rejects_target_outside_unit_interval() {
        for loss in [Loss::BinaryCrossEntropy, Loss::BinaryCrossEntropyWithLogits] {
            let mut params = ParamStore::new();
            let net = one_to_one_probability(&mut params);
            let data = unit_targets(vec![0.0, 3.0]);
            let err = fit(&net, &mut params, &data, config_for(loss)).unwrap_err();
            let Error::InvalidData(message) = &err else {
                panic!("expected InvalidData, found {err:?}");
            };
            assert!(message.contains("row 1, column 0"), "{message}");
            assert!(message.contains('3'), "{message}");
        }
    }

    /// A NaN binary cross-entropy target is a data error, for both binary losses.
    #[test]
    fn bce_rejects_nan_target() {
        for loss in [Loss::BinaryCrossEntropy, Loss::BinaryCrossEntropyWithLogits] {
            let mut params = ParamStore::new();
            let net = one_to_one_probability(&mut params);
            let data = unit_targets(vec![f32::NAN, 1.0]);
            let err = fit(&net, &mut params, &data, config_for(loss)).unwrap_err();
            assert!(matches!(err, Error::InvalidData(_)), "{err:?}");
        }
    }

    /// Targets of exactly 0 and 1 are valid binary cross-entropy targets.
    #[test]
    fn bce_accepts_zero_and_one() {
        for loss in [Loss::BinaryCrossEntropy, Loss::BinaryCrossEntropyWithLogits] {
            let mut params = ParamStore::new();
            let net = one_to_one_probability(&mut params);
            let data = unit_targets(vec![0.0, 1.0]);
            let config = FitConfig {
                optimizer: OptimizerConfig::Sgd {
                    lr: 0.01,
                    momentum: 0.0,
                    weight_decay: 0.0,
                },
                ..config_for(loss)
            };
            fit(&net, &mut params, &data, config).unwrap();
        }
    }

    /// Two layers that disagree on a width are a model error: `fit` does not return a data
    /// error, and the forward pass panics naming both layers.
    #[test]
    #[should_panic(expected = "layer 1 (")]
    fn disagreeing_layers_panic_in_forward() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(0);
        let net = Sequential::new()
            .push(Linear::new(&mut params, 3, 4, &mut rng))
            .push(Linear::new(&mut params, 5, 2, &mut rng));
        let data = Dataset {
            x: Tensor::zeros(&[4, 3]),
            targets: Targets::Values(Tensor::zeros(&[4, 2])),
        };
        let config = FitConfig {
            loss: Loss::Mse,
            ..FitConfig::default()
        };
        let _ = fit(&net, &mut params, &data, config);
    }

    /// Binary cross-entropy targets outside `[0, 1]` are a data error even when the model's
    /// layers disagree on a width, because that check does not need the output width.
    #[test]
    fn bce_target_check_runs_for_disagreeing_layers() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(0);
        let net = Sequential::new()
            .push(Linear::new(&mut params, 3, 4, &mut rng))
            .push(Linear::new(&mut params, 5, 1, &mut rng));
        let data = Dataset {
            x: Tensor::zeros(&[2, 3]),
            targets: Targets::Values(Tensor::from_vec(vec![0.0, 3.0], &[2, 1]).unwrap()),
        };
        let err = fit(
            &net,
            &mut params,
            &data,
            config_for(Loss::BinaryCrossEntropy),
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "{err:?}");
    }

    /// Inputs of the wrong width are a data error even when an activation comes before the
    /// first layer that names a width.
    #[test]
    fn activation_first_wrong_width() {
        let mut params = ParamStore::new();
        let net =
            Sequential::new()
                .push(Tanh)
                .push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
        let data = Dataset {
            x: Tensor::zeros(&[4, 4]),
            targets: Targets::Classes(vec![0; 4]),
        };
        let err = fit(&net, &mut params, &data, FitConfig::default()).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid data: the inputs are [4, 4], but layer 1 (linear) takes 3"
        );
    }

    /// Runs `fit` on `data` with a one-layer `Linear(inputs -> outputs)` model and returns
    /// the message of the error it must give.
    fn fit_error(inputs: usize, outputs: usize, data: &Dataset, loss: Loss) -> String {
        let mut params = ParamStore::new();
        let net =
            Sequential::new().push(Linear::new(&mut params, inputs, outputs, &mut Rng::seed(0)));
        let err = fit(&net, &mut params, data, config_for(loss)).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "{err:?}");
        err.to_string()
    }

    /// Value targets of the wrong width name both tensors.
    #[test]
    fn value_target_width_names_tensors() {
        let data = Dataset {
            x: Tensor::zeros(&[1437, 4]),
            targets: Targets::Values(Tensor::zeros(&[1437, 1])),
        };
        assert_eq!(
            fit_error(4, 10, &data, Loss::Mse),
            "invalid data: the value targets are [1437, 1], but the model outputs [1437, 10]"
        );
    }

    /// Value targets with the wrong row count name both tensors; a wrong label count names
    /// the rows and the labels.
    #[test]
    fn label_count_mismatch_names_tensors() {
        let data = Dataset {
            x: Tensor::zeros(&[100, 3]),
            targets: Targets::Classes(vec![0; 99]),
        };
        let message = fit_error(3, 2, &data, Loss::CrossEntropy);
        assert!(
            message.contains("there are 100 rows but 99 labels"),
            "{message}"
        );
        let data = Dataset {
            x: Tensor::zeros(&[4, 3]),
            targets: Targets::Values(Tensor::zeros(&[3, 2])),
        };
        let message = fit_error(3, 2, &data, Loss::Mse);
        assert!(
            message.contains("the value targets are [3, 2]")
                && message.contains("the inputs are [4, 3]"),
            "{message}"
        );
    }

    /// Inputs of the wrong width name the inputs and the width the layer takes.
    #[test]
    fn input_width_names_tensors() {
        let data = Dataset {
            x: Tensor::zeros(&[8, 5]),
            targets: Targets::Classes(vec![0; 8]),
        };
        let message = fit_error(3, 4, &data, Loss::CrossEntropy);
        assert!(message.contains("the inputs are [8, 5]"), "{message}");
        assert!(message.contains("takes 3"), "{message}");
    }

    /// A NaN input is rejected before training, naming its row, column and value.
    #[test]
    fn non_finite_input_rejected() {
        let mut x = Tensor::zeros(&[1500, 40]);
        x.data_mut()[1325 * 40 + 36] = f32::NAN;
        let data = Dataset {
            x,
            targets: Targets::Classes(vec![0; 1500]),
        };
        let message = fit_error(40, 2, &data, Loss::CrossEntropy);
        for part in ["row 1325", "column 36", "NaN", "the inputs"] {
            assert!(message.contains(part), "{part}: {message}");
        }
    }

    /// An infinite value target is rejected before training, naming its row and column.
    #[test]
    fn infinite_value_target_rejected() {
        let mut t = Tensor::zeros(&[4, 2]);
        t.data_mut()[2 * 2] = f32::INFINITY;
        let data = Dataset {
            x: Tensor::zeros(&[4, 3]),
            targets: Targets::Values(t),
        };
        let message = fit_error(3, 2, &data, Loss::Mse);
        for part in ["row 2", "column 0", "inf", "the value targets"] {
            assert!(message.contains(part), "{part}: {message}");
        }
    }

    /// Layers that change the width without naming an input width come before the first
    /// layer that does: the message gives the width that reaches that layer.
    #[test]
    fn width_after_glu_names_reached_width() {
        let mut params = ParamStore::new();
        let net = Sequential::new()
            .push(Glu {
                gate: Gate::Sigmoid,
            })
            .push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
        let data = Dataset {
            x: Tensor::zeros(&[4, 8]),
            targets: Targets::Classes(vec![0; 4]),
        };
        let err = fit(&net, &mut params, &data, FitConfig::default()).unwrap_err();
        let message = err.to_string();
        assert!(matches!(err, Error::InvalidData(_)), "{err:?}");
        for part in [
            "the inputs are [4, 8]",
            "reach layer 1 with width 4",
            "takes 3",
        ] {
            assert!(message.contains(part), "{part}: {message}");
        }
    }

    /// `fit` with an epoch count of `usize::MAX` stops at the first non-finite loss and does
    /// not reserve room for every epoch's report.
    #[test]
    fn fit_huge_epochs_does_not_reserve() {
        let mut params = ParamStore::new();
        let net = one_to_one(&mut params);
        let data = Dataset {
            x: Tensor::from_vec(vec![1.0f32, 2.0], &[2, 1]).unwrap(),
            targets: Targets::Values(Tensor::from_vec(vec![5.0f32, 9.0], &[2, 1]).unwrap()),
        };
        let config = FitConfig {
            epochs: usize::MAX,
            batch: 2,
            optimizer: OptimizerConfig::Sgd {
                lr: 1e20,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::Mse,
            ..FitConfig::default()
        };
        let err = fit(&net, &mut params, &data, config).unwrap_err();
        assert!(matches!(err, Error::NonFinite { .. }), "{err:?}");
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
            on_epoch: Some(Box::new(
                move |r: &EpochReport, _: &Sequential, _: &ParamStore| {
                    sink.lock().unwrap().push(*r)
                },
            )),
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
        assert_eq!(config.schedule, LrSchedule::Constant);
        assert!(config.on_epoch.is_none());
    }

    /// The cosine rates for 4 epochs equal PyTorch's `CosineAnnealingLR(T_max=4)` stepped
    /// once per epoch (values printed by PyTorch 2.15.0), with and without a floor; the
    /// constant schedule returns the initial rate for every epoch.
    #[test]
    fn lr_schedule_rates() {
        let cosine = LrSchedule::Cosine { min_lr: 0.0 };
        let rates: Vec<f64> = (0..4)
            .map(|e| cosine.learning_rate(1e-3, e, 1, 4))
            .collect();
        assert_eq!(
            rates,
            [1e-3, 8.535533905932737e-4, 5e-4, 1.4644660940672628e-4]
        );
        let floored = LrSchedule::Cosine { min_lr: 0.01 };
        let rates: Vec<f64> = (0..4)
            .map(|e| floored.learning_rate(0.1, e, 1, 4))
            .collect();
        assert_eq!(
            rates,
            [
                0.1,
                0.08681980515339464,
                0.05500000000000001,
                0.023180194846605363
            ]
        );
        for e in 0..4 {
            assert_eq!(LrSchedule::Constant.learning_rate(0.1, e, 1, 4), 0.1);
        }
        assert_eq!(LrSchedule::default(), LrSchedule::Constant);
    }

    /// The warm-up climbs linearly to the peak and ends on it; the cosine by step then falls
    /// from the peak over the remaining steps; a cosine by step with no warm-up is the
    /// per-epoch cosine at one step per epoch; and `Warmup { steps: 0 }` is constant.
    #[test]
    fn step_schedule_rates() {
        let warm = LrSchedule::Warmup { steps: 4 };
        let rates: Vec<f64> = (0..6).map(|s| warm.learning_rate(0.1, s, 3, 2)).collect();
        // `lr·(s + 1)/steps` in that order of operations, so step 2 is 0.1·3/4.
        assert_eq!(
            rates,
            [0.1 / 4.0, 0.1 * 2.0 / 4.0, 0.1 * 3.0 / 4.0, 0.1, 0.1, 0.1]
        );
        let both = LrSchedule::WarmupCosine {
            warmup_steps: 2,
            min_lr: 0.0,
        };
        let rates: Vec<f64> = (0..6).map(|s| both.learning_rate(0.1, s, 3, 2)).collect();
        // Steps 2..6 are 0..4 of a cosine over 4 steps.
        let cosine = |t: f64| 0.1 * (1.0 + (std::f64::consts::PI * t / 4.0).cos()) / 2.0;
        assert_eq!(
            rates,
            [
                0.05,
                0.1,
                cosine(0.0),
                cosine(1.0),
                cosine(2.0),
                cosine(3.0)
            ]
        );
        let by_step = LrSchedule::WarmupCosine {
            warmup_steps: 0,
            min_lr: 0.01,
        };
        let by_epoch = LrSchedule::Cosine { min_lr: 0.01 };
        for e in 0..4 {
            assert_eq!(
                by_step.learning_rate(0.1, e, 1, 4),
                by_epoch.learning_rate(0.1, e, 1, 4)
            );
        }
        assert_eq!(
            LrSchedule::Warmup { steps: 0 }.learning_rate(0.1, 0, 3, 2),
            0.1
        );
    }

    /// `fit` under a warm-up then a cosine by step trains to the parameters and the epoch
    /// losses of a manual loop that sets each step's rate from the same schedule and calls
    /// `Trainer::step`, bit for bit; and its rates really change within an epoch.
    #[test]
    fn step_schedule_fit_matches_a_manual_loop() {
        let data = classes_dataset(10);
        let schedule = LrSchedule::WarmupCosine {
            warmup_steps: 2,
            min_lr: 1e-4,
        };
        let (epochs, batch, seed) = (3, 4, 5);
        let build = |params: &mut ParamStore| {
            let mut rng = Rng::seed(1);
            Sequential::new()
                .push(Linear::new(params, 3, 6, &mut rng))
                .push(Tanh)
                .push(Linear::new(params, 6, 10, &mut rng))
        };
        let mut fitted = ParamStore::new();
        let net = build(&mut fitted);
        let config = FitConfig {
            epochs,
            batch,
            shuffle_seed: Some(seed),
            schedule,
            loss: Loss::CrossEntropy,
            optimizer: OptimizerConfig::Adam {
                lr: 0.05,
                weight_decay: 0.0,
            },
            health: Health::quiet(),
            ..FitConfig::default()
        };
        let history = fit(&net, &mut fitted, &data, config).unwrap();

        let mut manual = ParamStore::new();
        let twin = build(&mut manual);
        let mut trainer = Trainer::new(&twin, Adam::new(0.05), Loss::CrossEntropy, batch);
        trainer.set_health(Health::quiet());
        let mut batcher = Batcher::new(data.len(), batch, Some(seed));
        let mut buffer = Batch::new(&data, batch);
        let steps_per_epoch = batcher.batches().count();
        assert_eq!(steps_per_epoch, 3);
        let mut step = 0;
        let mut rates = Vec::new();
        let mut losses = Vec::new();
        for epoch in 1..=epochs {
            batcher.epoch();
            let (mut total, mut rows) = (0.0f64, 0usize);
            for range in batcher.batches() {
                let lr = schedule.learning_rate(0.05, step, steps_per_epoch, epochs);
                trainer.override_learning_rate(lr);
                rates.push(lr);
                step += 1;
                let n = range.len();
                let view = batcher.gather(&data, range, &mut buffer);
                let loss = trainer.step(&twin, &mut manual, view).unwrap();
                total += f64::from(loss) * n as f64;
                rows += n;
            }
            trainer.end_epoch(epoch).unwrap();
            losses.push((total / rows as f64).to_bits());
        }
        assert_eq!(
            parameter_bits(&net, &fitted),
            parameter_bits(&twin, &manual)
        );
        let fitted_losses: Vec<u64> = history
            .epochs
            .iter()
            .map(|r| r.mean_loss.to_bits())
            .collect();
        assert_eq!(fitted_losses, losses);
        assert_eq!(rates.len(), 9);
        assert!(rates[0] < rates[1] && rates[1] == 0.05 && rates[3] > rates[4]);
    }

    /// The learning rate each epoch of `fit` trains at under `schedule`, starting from SGD at
    /// 0.1 over 4 epochs, read back from the parameters in `on_epoch`.
    ///
    /// The network is one `Linear` from a zero input to a target of 1 under mean squared
    /// error, one row per epoch, so the output is the bias `b` and one step multiplies the
    /// distance `1 − b` by `1 − 2·lr`: `lr = (1 − d_new / d_old) / 2`.
    fn rates_fit_trains_at(schedule: LrSchedule) -> Vec<f64> {
        use std::sync::{Arc, Mutex};
        let data = Dataset {
            x: Tensor::zeros(&[1, 1]),
            targets: Targets::Values(Tensor::full(&[1, 1], 1.0f32)),
        };
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 1, 1, &mut Rng::seed(0));
        let bias = layer.bias().unwrap();
        let net = Sequential::new().push(layer);
        let distances = Arc::new(Mutex::new(vec![1.0f64]));
        let sink = Arc::clone(&distances);
        let config = FitConfig {
            epochs: 4,
            batch: 1,
            optimizer: OptimizerConfig::Sgd {
                lr: 0.1,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::Mse,
            schedule,
            // The weight's gradient is exactly 0 (the input is 0); its warning is expected.
            health: Health {
                sink: Some(Box::new(|_: &crate::Warning| {})),
                ..Health::default()
            },
            on_epoch: Some(Box::new(
                move |_: &EpochReport, _: &Sequential, params: &ParamStore| {
                    let b = f64::from(params.value(bias).data()[0]);
                    sink.lock().unwrap().push(1.0 - b);
                },
            )),
            ..FitConfig::default()
        };
        fit(&net, &mut params, &data, config).unwrap();
        let d = distances.lock().unwrap();
        d.windows(2).map(|w| (1.0 - w[1] / w[0]) / 2.0).collect()
    }

    /// `fit` sets each epoch's rate from the cosine schedule before the epoch trains, and
    /// `on_epoch` sees the parameters after the epoch.
    #[test]
    fn fit_follows_the_cosine_schedule() {
        let rates = rates_fit_trains_at(LrSchedule::Cosine { min_lr: 0.01 });
        let expected = [
            0.1,
            0.08681980515339464,
            0.05500000000000001,
            0.023180194846605363,
        ];
        assert_eq!(rates.len(), 4);
        for (e, (got, want)) in rates.iter().zip(expected).enumerate() {
            // The bias is f32, so the recovered rate carries its rounding.
            assert!((got - want).abs() < 1e-6, "epoch {e}: {got} vs {want}");
        }
    }

    /// With the constant schedule every epoch of `fit` trains at the configured rate.
    #[test]
    fn fit_constant_schedule_keeps_the_rate() {
        let rates = rates_fit_trains_at(LrSchedule::Constant);
        assert_eq!(rates.len(), 4);
        for (e, got) in rates.iter().enumerate() {
            assert!((got - 0.1).abs() < 1e-6, "epoch {e}: {got}");
        }
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
                // Each loss's core operation and reduction, as its documentation states,
                // written out here rather than through `Loss::record`, so a wrong mapping
                // there fails this test.
                let mean = Reduction::Mean;
                let l = match (&loss, &data.targets) {
                    (Loss::CrossEntropy, Targets::Classes(labels)) => {
                        tape.softmax_cross_entropy(out, labels, mean)
                    }
                    (Loss::Nll, Targets::Classes(labels)) => tape.nll(out, labels, mean),
                    (_, Targets::Values(values)) => {
                        let t = tape.input(values.view());
                        match loss {
                            Loss::Mse => tape.mse(out, t, mean),
                            Loss::L1 => tape.l1(out, t, mean),
                            Loss::Huber { delta } => tape.huber(out, t, delta as f32, mean),
                            Loss::SmoothL1 { beta } => tape.smooth_l1(out, t, beta as f32, mean),
                            Loss::BinaryCrossEntropy => tape.binary_cross_entropy(out, t, mean),
                            Loss::BinaryCrossEntropyWithLogits => {
                                tape.binary_cross_entropy_with_logits(out, t, mean)
                            }
                            Loss::KlDiv => tape.kl_div(out, t, Reduction::BatchMean),
                            other => panic!("{other:?} takes class targets"),
                        }
                    }
                    (other, Targets::Classes(_)) => panic!("{other:?} takes value targets"),
                };
                tape.value(l).data()[0]
            };
            let before = parameter_bits(&net, &params);
            let got = trainer
                .step(&net, &mut params, batcher.gather(data, 0..2, &mut batch))
                .unwrap();
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

    /// A label equal to the output width is out of range, and `accuracy` panics naming the
    /// row, the label and the width.
    #[test]
    #[should_panic(expected = "the label of row 1 is 3, but the model has 3 outputs")]
    fn accuracy_label_equal_to_width_panics() {
        let mut params = ParamStore::new();
        let net = Sequential::new().push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)));
        let data = Dataset {
            x: Tensor::zeros(&[2, 2]),
            targets: Targets::Classes(vec![0, 3]),
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
            OptimizerConfig::RmsProp {
                lr: 0.002,
                alpha: 0.99,
                momentum: 0.9,
                weight_decay: 0.0,
            },
            OptimizerConfig::Adagrad {
                lr: 0.05,
                lr_decay: 0.0,
                weight_decay: 1e-4,
            },
        ];
        let lrs: Vec<f64> = configs.iter().map(|c| c.build().learning_rate()).collect();
        assert_eq!(lrs, [0.1, 0.01, 0.001, 0.002, 0.05]);
    }

    /// Every loss that cannot be negative on valid inputs says so, and `KlDiv`, which can be
    /// negative, does not.
    #[test]
    fn non_negative_losses_listed() {
        let listed = [
            Loss::CrossEntropy,
            Loss::Nll,
            Loss::Mse,
            Loss::L1,
            Loss::Huber { delta: 1.0 },
            Loss::SmoothL1 { beta: 1.0 },
            Loss::BinaryCrossEntropy,
            Loss::BinaryCrossEntropyWithLogits,
        ];
        assert!(listed.iter().all(|l| l.non_negative()), "{listed:?}");
        assert!(!Loss::KlDiv.non_negative());
    }

    /// An epoch over a shuffled batcher that reaches a NaN names the dataset row the batcher
    /// put at the bad position, not just the position in the batch.
    #[test]
    fn non_finite_input_names_the_dataset_row() {
        let mut data = classes_dataset(12);
        data.x.data_mut()[5 * 3 + 1] = f32::NAN;
        let mut params = ParamStore::new();
        let net = Sequential::new().push(Linear::new(&mut params, 3, 10, &mut Rng::seed(0)));
        let mut trainer = Trainer::new(&net, Sgd::new(0.01), Loss::CrossEntropy, 4);
        trainer.set_health(Health::quiet());
        let mut batcher = Batcher::new(12, 4, Some(9));
        let mut shuffled = batcher.clone();
        shuffled.epoch();
        let at = shuffled.order().iter().position(|&r| r == 5).unwrap();
        let err = trainer
            .run_epoch(
                &mut Run::Plain(&net),
                &mut params,
                &data,
                &mut batcher,
                &mut Hooks::new(),
                1,
            )
            .unwrap_err();
        let text = err.to_string();
        let expected = format!(
            "the batch's input held NaN at batch position {}, feature 1 (row 5 of the dataset)",
            at % 4
        );
        assert!(text.ends_with(&expected), "{text}");
        assert_eq!(trainer.steps(), at as u64 / 4 + 1);
    }

    /// The 64-row, 4-feature, 2-class set and the 4 → 16 → tanh → 2 network the control
    /// tests train: batch 16 gives 4 steps per epoch.
    fn control_setup() -> (Dataset<f32>, ParamStore, Sequential) {
        let data = synthetic_classes(64, 4, 2, 21);
        let mut params = ParamStore::new();
        let net = small_classifier(&mut params, 4, 2, 22);
        (data, params, net)
    }

    /// The configuration of the control tests: `epochs` epochs at batch 16, Adam, shuffled,
    /// with quiet health checks.
    fn control_config(epochs: usize) -> FitConfig<'static> {
        FitConfig {
            epochs,
            batch: 16,
            shuffle_seed: Some(4),
            health: Health::quiet(),
            ..FitConfig::default()
        }
    }

    // A configuration, a trainer and their health checks can move to another thread.
    const _: () = {
        /// Requires `T: Send`.
        const fn send<T: Send>() {}
        send::<FitConfig<'static>>();
        send::<Trainer<'static>>();
        send::<Health<'static>>();
    };

    /// A `FitConfig` whose sink and epoch callback record where they run is built on this
    /// thread, moved into `std::thread::spawn` with the data, model and parameters, and
    /// trains there: both callbacks ran on the worker, not on this thread (the epoch
    /// callback once per epoch, the sink on the steps, with a bound every gradient passes),
    /// and the run gives the parameters of the same run on this thread, bit for bit.
    #[test]
    fn a_fit_config_moves_to_another_thread() {
        type Seen = Arc<std::sync::Mutex<Vec<std::thread::ThreadId>>>;
        let make = |epochs: Seen, warnings: Seen| FitConfig {
            health: Health {
                grad_norm_high: 0.0,
                sink: Some(Box::new(move |_: &Warning| {
                    warnings.lock().unwrap().push(std::thread::current().id());
                })),
                ..Health::default()
            },
            on_epoch: Some(Box::new(
                move |_: &EpochReport, _: &Sequential, _: &ParamStore| {
                    epochs.lock().unwrap().push(std::thread::current().id());
                },
            )),
            ..control_config(3)
        };
        let (data, mut here, net_here) = control_setup();
        fit(
            &net_here,
            &mut here,
            &data,
            make(Seen::default(), Seen::default()),
        )
        .unwrap();
        let (epochs_seen, warned_on) = (Seen::default(), Seen::default());
        let config = make(Arc::clone(&epochs_seen), Arc::clone(&warned_on));
        let (data, params, net) = control_setup();
        let there = std::thread::spawn(move || {
            let mut params = params;
            fit(&net, &mut params, &data, config).unwrap();
            parameter_bits(&net, &params)
        })
        .join()
        .unwrap();
        let me = std::thread::current().id();
        let (epochs, warned) = (epochs_seen.lock().unwrap(), warned_on.lock().unwrap());
        assert_eq!(epochs.len(), 3);
        assert!(!warned.is_empty());
        assert!(epochs.iter().chain(warned.iter()).all(|&id| id != me));
        assert_eq!(there, parameter_bits(&net_here, &here));
    }

    /// The sink and the epoch callback can borrow local state: they push onto lists owned
    /// by the test, here from a scoped thread, and the lists are read once the run is over.
    #[test]
    fn callbacks_borrow_local_state() {
        let (data, mut params, net) = control_setup();
        let mut warnings: Vec<String> = Vec::new();
        let mut losses: Vec<f64> = Vec::new();
        std::thread::scope(|s| {
            s.spawn(|| {
                let config = FitConfig {
                    // A bound every gradient passes, so every step sends a warning.
                    health: Health {
                        grad_norm_high: 0.0,
                        sink: Some(Box::new(|w: &Warning| warnings.push(w.to_string()))),
                        ..Health::default()
                    },
                    on_epoch: Some(Box::new(
                        |r: &EpochReport, _: &Sequential, _: &ParamStore| losses.push(r.mean_loss),
                    )),
                    ..control_config(2)
                };
                fit(&net, &mut params, &data, config).unwrap();
            });
        });
        assert_eq!(losses.len(), 2);
        assert!(!warnings.is_empty());
        assert!(
            warnings.iter().all(|w| w.contains("exploding")),
            "{warnings:?}"
        );
    }

    /// A control that changes nothing.
    struct NoOp;

    impl StepControl for NoOp {}

    /// `last_profile` is `None` after every step that did not ask for a profile and `Some`,
    /// with one forward and one backward entry per layer, after the step that did; its parts
    /// never add up to more than its total.
    #[test]
    fn profile_only_when_asked() {
        /// Asks for a profile after steps 3 and 9, and records what `last_profile` holds
        /// after each step.
        struct AskTwice {
            /// Each step's number and, when profiled, its entry counts.
            seen: Vec<(u64, Option<(usize, usize)>)>,
        }

        impl StepControl for AskTwice {
            fn after_step(
                &mut self,
                info: &StepInfo,
                _model: &mut ModelHandle<'_>,
                _params: &ParamStore,
                trainer: &mut TrainerHandle<'_>,
            ) -> Flow {
                let profile = trainer.last_profile();
                if let Some(p) = profile {
                    let parts = p.forward.iter().sum::<Duration>()
                        + p.loss
                        + p.backward.iter().sum::<Duration>()
                        + p.backward_loss
                        + p.optimizer;
                    assert!(parts <= p.total, "step {}: {p:?}", info.step);
                    assert!(p.total > Duration::ZERO, "step {}: {p:?}", info.step);
                }
                self.seen.push((
                    info.step,
                    profile.map(|p| (p.forward.len(), p.backward.len())),
                ));
                if info.step == 3 || info.step == 9 {
                    trainer.profile_next_step();
                }
                Flow::Continue
            }
        }

        let (data, mut params, mut net) = control_setup();
        let mut control = AskTwice { seen: Vec::new() };
        fit_controlled(
            &mut net,
            &mut params,
            &data,
            control_config(3),
            &mut Hooks::new(),
            &mut control,
        )
        .unwrap();
        assert_eq!(control.seen.len(), 12);
        for &(step, profile) in &control.seen {
            let expected = (step == 4 || step == 10).then_some((3, 3));
            assert_eq!(profile, expected, "step {step}");
        }

        // The same through `Trainer::step` directly.
        let mut trainer = Trainer::new(&net, Sgd::new(0.01), Loss::CrossEntropy, 16);
        let mut batch = Batch::new(&data, 16);
        let batcher = Batcher::new(64, 16, None);
        assert!(trainer.last_profile().is_none());
        trainer.profile_next_step();
        trainer
            .step(&net, &mut params, batcher.gather(&data, 0..16, &mut batch))
            .unwrap();
        let profile = trainer.last_profile().unwrap();
        assert_eq!((profile.forward.len(), profile.backward.len()), (3, 3));
        trainer
            .step(&net, &mut params, batcher.gather(&data, 16..32, &mut batch))
            .unwrap();
        assert!(trainer.last_profile().is_none());
    }

    /// A control that switches neuron 3 of the `Linear` at layer 0 off after step 5: the
    /// hooks on layer 1 (the tanh, which holds that `Linear`'s neurons) see column 3 nonzero
    /// on steps 1 to 5 and exactly 0 on every step from 6 on.
    #[test]
    fn control_can_switch_a_neuron_between_steps() {
        /// Switches neuron 3 of `Linear` 0 off after step 5.
        struct SwitchOff;

        impl StepControl for SwitchOff {
            fn after_step(
                &mut self,
                info: &StepInfo,
                model: &mut ModelHandle<'_>,
                _params: &ParamStore,
                _trainer: &mut TrainerHandle<'_>,
            ) -> Flow {
                if info.step == 5 {
                    model.set_neuron(0, 3, false);
                    assert_eq!(model.model().neuron_layer(0), 1);
                }
                Flow::Continue
            }
        }

        let (data, mut params, mut net) = control_setup();
        let mut zero = Vec::new();
        {
            let mut hooks = Hooks::new();
            hooks.on_forward(1, |e| {
                zero.push(e.output.data().chunks(16).all(|row| row[3] == 0.0));
            });
            fit_controlled(
                &mut net,
                &mut params,
                &data,
                control_config(3),
                &mut hooks,
                &mut SwitchOff,
            )
            .unwrap();
        }
        assert_eq!(zero.len(), 12);
        assert_eq!(&zero[..5], &[false; 5]);
        assert_eq!(&zero[5..], &[true; 7]);
        let off: Vec<usize> = net.neuron_mask(0).unwrap().off().collect();
        assert_eq!(off, [3]);
    }

    /// Under a cosine schedule, a control that overrides the learning rate to 0.05 after
    /// epoch 1 makes every later epoch train at 0.05, read back from the parameters as
    /// `rates_fit_trains_at` does and from `Trainer::learning_rate` after each step.
    #[test]
    fn override_replaces_schedule() {
        /// Overrides the rate after epoch 1; records the bias's distance from its target
        /// after each epoch and the trainer's rate after each step.
        struct Override {
            /// The bias parameter.
            bias: ParamId,
            /// `1 − b` before training and after each epoch.
            distances: Vec<f64>,
            /// `Trainer::learning_rate` after each step.
            rates: Vec<f64>,
        }

        impl StepControl for Override {
            fn after_step(
                &mut self,
                _info: &StepInfo,
                _model: &mut ModelHandle<'_>,
                _params: &ParamStore,
                trainer: &mut TrainerHandle<'_>,
            ) -> Flow {
                self.rates.push(trainer.learning_rate());
                Flow::Continue
            }

            fn after_epoch(
                &mut self,
                report: &EpochReport,
                _model: &mut ModelHandle<'_>,
                params: &ParamStore,
                trainer: &mut TrainerHandle<'_>,
            ) -> Flow {
                let b = f64::from(params.value(self.bias).data()[0]);
                self.distances.push(1.0 - b);
                if report.epoch == 1 {
                    trainer.override_learning_rate(0.05);
                    assert_eq!(trainer.learning_rate(), 0.05);
                }
                Flow::Continue
            }
        }

        let data = Dataset {
            x: Tensor::zeros(&[1, 1]),
            targets: Targets::Values(Tensor::full(&[1, 1], 1.0f32)),
        };
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 1, 1, &mut Rng::seed(0));
        let bias = layer.bias().unwrap();
        let mut net = Sequential::new().push(layer);
        let config = FitConfig {
            epochs: 4,
            batch: 1,
            optimizer: OptimizerConfig::Sgd {
                lr: 0.1,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::Mse,
            schedule: LrSchedule::Cosine { min_lr: 0.01 },
            health: Health::quiet(),
            ..FitConfig::default()
        };
        let mut control = Override {
            bias,
            distances: vec![1.0],
            rates: Vec::new(),
        };
        fit_controlled(
            &mut net,
            &mut params,
            &data,
            config,
            &mut Hooks::new(),
            &mut control,
        )
        .unwrap();
        assert_eq!(control.rates, [0.1, 0.05, 0.05, 0.05]);
        let rates: Vec<f64> = control
            .distances
            .windows(2)
            .map(|w| (1.0 - w[1] / w[0]) / 2.0)
            .collect();
        assert_eq!(rates.len(), 4);
        for (e, (got, want)) in rates.iter().zip([0.1, 0.05, 0.05, 0.05]).enumerate() {
            // The bias is f32, so the recovered rate carries its rounding.
            assert!((got - want).abs() < 1e-6, "epoch {e}: {got} vs {want}");
        }
    }

    /// `Flow::Stop` after step 7 (epoch 2, step 3 of 4) ends training there: 7 steps ran,
    /// and the history holds epoch 1 and the partial epoch 2, whose mean loss is over the
    /// rows of its 3 steps. `Flow::Stop` from `after_epoch` ends the run after that epoch.
    #[test]
    fn control_stop_ends_training() {
        /// Stops after step `at_step` or after epoch `at_epoch`; records each step's loss
        /// and rows and the epochs it saw end.
        struct StopAt {
            /// The step after which to stop.
            at_step: u64,
            /// The epoch after which to stop.
            at_epoch: usize,
            /// Each step's number, the trainer's count, its loss and its rows.
            steps: Vec<(u64, u64, f32, usize)>,
            /// The epochs `after_epoch` saw.
            epochs: Vec<usize>,
        }

        impl StepControl for StopAt {
            fn after_step(
                &mut self,
                info: &StepInfo,
                _model: &mut ModelHandle<'_>,
                _params: &ParamStore,
                trainer: &mut TrainerHandle<'_>,
            ) -> Flow {
                assert!(info.seconds >= 0.0);
                self.steps
                    .push((info.step, trainer.steps(), info.loss, info.rows));
                if info.step == self.at_step {
                    Flow::Stop
                } else {
                    Flow::Continue
                }
            }

            fn after_epoch(
                &mut self,
                report: &EpochReport,
                _model: &mut ModelHandle<'_>,
                _params: &ParamStore,
                _trainer: &mut TrainerHandle<'_>,
            ) -> Flow {
                self.epochs.push(report.epoch);
                if report.epoch == self.at_epoch {
                    Flow::Stop
                } else {
                    Flow::Continue
                }
            }
        }

        let (data, mut params, mut net) = control_setup();
        let mut backward = 0;
        let mut control = StopAt {
            at_step: 7,
            at_epoch: 0,
            steps: Vec::new(),
            epochs: Vec::new(),
        };
        let history = {
            let mut hooks = Hooks::new();
            hooks.on_backward(0, |_| backward += 1);
            fit_controlled(
                &mut net,
                &mut params,
                &data,
                control_config(5),
                &mut hooks,
                &mut control,
            )
            .unwrap()
        };
        assert_eq!(backward, 7);
        let numbers: Vec<(u64, u64)> = control.steps.iter().map(|s| (s.0, s.1)).collect();
        assert_eq!(numbers, (1..=7).map(|s| (s, s)).collect::<Vec<_>>());
        assert_eq!(control.epochs, [1]);
        let epochs: Vec<usize> = history.epochs.iter().map(|r| r.epoch).collect();
        assert_eq!(epochs, [1, 2]);
        let rows: Vec<usize> = history.epochs.iter().map(|r| r.rows).collect();
        assert_eq!(rows, [64, 48]);
        let partial = &control.steps[4..];
        let rows: usize = partial.iter().map(|s| s.3).sum();
        assert_eq!(rows, 48);
        let total: f64 = partial.iter().map(|s| f64::from(s.2) * s.3 as f64).sum();
        assert_eq!(history.epochs[1].mean_loss, total / rows as f64);

        let (data, mut params, mut net) = control_setup();
        let mut control = StopAt {
            at_step: 0,
            at_epoch: 2,
            steps: Vec::new(),
            epochs: Vec::new(),
        };
        let history = fit_controlled(
            &mut net,
            &mut params,
            &data,
            control_config(5),
            &mut Hooks::new(),
            &mut control,
        )
        .unwrap();
        assert_eq!(control.steps.len(), 8);
        assert_eq!(control.epochs, [1, 2]);
        assert_eq!(history.epochs.len(), 2);
    }

    /// A stop on an epoch's last step leaves that epoch whole: its end-of-epoch health
    /// checks (a dead-neuron report here) and `on_epoch` run and its report counts every
    /// row, but `after_epoch` does not run. A stop one step earlier leaves a partial epoch
    /// with neither.
    #[test]
    fn stop_on_last_step_ends_the_epoch() {
        use std::sync::Mutex;

        /// Stops after step `at`; records the epochs `after_epoch` saw.
        struct StopAt {
            /// The step after which to stop.
            at: u64,
            /// The epochs `after_epoch` saw.
            epochs: Vec<usize>,
        }

        impl StepControl for StopAt {
            fn after_step(
                &mut self,
                info: &StepInfo,
                _model: &mut ModelHandle<'_>,
                _params: &ParamStore,
                _trainer: &mut TrainerHandle<'_>,
            ) -> Flow {
                if info.step == self.at {
                    Flow::Stop
                } else {
                    Flow::Continue
                }
            }

            fn after_epoch(
                &mut self,
                report: &EpochReport,
                _model: &mut ModelHandle<'_>,
                _params: &ParamStore,
                _trainer: &mut TrainerHandle<'_>,
            ) -> Flow {
                self.epochs.push(report.epoch);
                Flow::Continue
            }
        }

        /// Runs 5 epochs of 4 steps on a network whose ReLU neurons are all dead, stopping
        /// after step `at`. Returns the history, the epochs `after_epoch` saw, the epochs
        /// `on_epoch` saw and the epochs of the dead-neuron reports.
        fn run(at: u64) -> (History, Vec<usize>, Vec<usize>, Vec<usize>) {
            let data = synthetic_classes(64, 4, 2, 21);
            let mut params = ParamStore::new();
            let mut rng = Rng::seed(22);
            let first = Linear::new(&mut params, 4, 16, &mut rng);
            params
                .value_mut(first.bias().unwrap())
                .data_mut()
                .fill(-100.0);
            let mut net = Sequential::new()
                .push(first)
                .push(crate::Relu)
                .push(Linear::new(&mut params, 16, 2, &mut rng));
            let dead = Arc::new(Mutex::new(Vec::new()));
            let on_epoch = Arc::new(Mutex::new(Vec::new()));
            let (dead_sink, on_epoch_sink) = (Arc::clone(&dead), Arc::clone(&on_epoch));
            let config = FitConfig {
                health: Health {
                    sink: Some(Box::new(move |w: &crate::Warning| {
                        if w.kind() == crate::WarningKind::DeadNeurons {
                            dead_sink.lock().unwrap().push(w.epoch().unwrap());
                        }
                    })),
                    ..Health::default()
                },
                on_epoch: Some(Box::new(
                    move |r: &EpochReport, _: &Sequential, _: &ParamStore| {
                        on_epoch_sink.lock().unwrap().push(r.epoch);
                    },
                )),
                ..control_config(5)
            };
            let mut control = StopAt {
                at,
                epochs: Vec::new(),
            };
            let history = fit_controlled(
                &mut net,
                &mut params,
                &data,
                config,
                &mut Hooks::new(),
                &mut control,
            )
            .unwrap();
            let dead = dead.lock().unwrap().clone();
            let on_epoch = on_epoch.lock().unwrap().clone();
            (history, control.epochs, on_epoch, dead)
        }

        let (history, after_epoch, on_epoch, dead) = run(8);
        let rows: Vec<usize> = history.epochs.iter().map(|r| r.rows).collect();
        assert_eq!(rows, [64, 64]);
        assert_eq!(after_epoch, [1]);
        assert_eq!(on_epoch, [1, 2]);
        assert_eq!(dead, [1, 2]);

        let (history, after_epoch, on_epoch, dead) = run(7);
        let rows: Vec<usize> = history.epochs.iter().map(|r| r.rows).collect();
        assert_eq!(rows, [64, 48]);
        assert_eq!(after_epoch, [1]);
        assert_eq!(on_epoch, [1]);
        assert_eq!(dead, [1]);
    }

    /// A control that asks for a profile on every step and reads everything its trainer
    /// handle offers trains to the same parameters and losses as a control that does
    /// nothing, bit for bit.
    #[test]
    fn profiling_control_trains_like_no_control() {
        /// Asks for every step's profile and reads the profile, the rate and the step count.
        struct Watch {
            /// A running sum of what it read, so the reads are used.
            seen: f64,
        }

        impl StepControl for Watch {
            fn after_step(
                &mut self,
                info: &StepInfo,
                _model: &mut ModelHandle<'_>,
                _params: &ParamStore,
                trainer: &mut TrainerHandle<'_>,
            ) -> Flow {
                assert_eq!(trainer.steps(), info.step);
                if let Some(p) = trainer.last_profile() {
                    self.seen += p.total.as_secs_f64();
                }
                self.seen += trainer.learning_rate();
                trainer.profile_next_step();
                Flow::Continue
            }

            fn after_epoch(
                &mut self,
                _report: &EpochReport,
                _model: &mut ModelHandle<'_>,
                _params: &ParamStore,
                trainer: &mut TrainerHandle<'_>,
            ) -> Flow {
                trainer.profile_next_step();
                Flow::Continue
            }
        }

        let config = || FitConfig {
            schedule: LrSchedule::Cosine { min_lr: 1e-4 },
            ..control_config(3)
        };
        let (data, mut quiet_params, mut quiet_net) = control_setup();
        let quiet = fit_controlled(
            &mut quiet_net,
            &mut quiet_params,
            &data,
            config(),
            &mut Hooks::new(),
            &mut NoOp,
        )
        .unwrap();
        let (_, mut params, mut net) = control_setup();
        let mut watch = Watch { seen: 0.0 };
        let watched = fit_controlled(
            &mut net,
            &mut params,
            &data,
            config(),
            &mut Hooks::new(),
            &mut watch,
        )
        .unwrap();
        assert!(watch.seen > 0.0);
        assert_eq!(
            parameter_bits(&quiet_net, &quiet_params),
            parameter_bits(&net, &params)
        );
        let losses = |h: &History| {
            h.epochs
                .iter()
                .map(|r| r.mean_loss.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(losses(&quiet), losses(&watched));
    }

    /// Each profile entry belongs to its layer: on 16 → 16 → relu → 512 → relu → 512 → relu
    /// → 2, the 512 → 512 `Linear` (layer 4) has the largest forward and backward entries,
    /// at least 4 times layer 0's (16 → 16), taking each entry's least value over 5
    /// profiled steps to shed scheduling noise.
    #[test]
    fn profile_entries_follow_the_layers() {
        let data = synthetic_classes(64, 16, 2, 31);
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(32);
        let net = Sequential::new()
            .push(Linear::new(&mut params, 16, 16, &mut rng))
            .push(crate::Relu)
            .push(Linear::new(&mut params, 16, 512, &mut rng))
            .push(crate::Relu)
            .push(Linear::new(&mut params, 512, 512, &mut rng))
            .push(crate::Relu)
            .push(Linear::new(&mut params, 512, 2, &mut rng));
        let mut trainer = Trainer::new(&net, Sgd::new(0.01), Loss::CrossEntropy, 64);
        trainer.set_health(Health::quiet());
        let mut batch = Batch::new(&data, 64);
        let batcher = Batcher::new(64, 64, None);
        let mut forward = vec![Duration::MAX; net.len()];
        let mut backward = vec![Duration::MAX; net.len()];
        for step in 0..7 {
            // Two warm-up steps make the buffers first.
            if step >= 2 {
                trainer.profile_next_step();
            }
            let view = batcher.gather(&data, 0..64, &mut batch);
            trainer.step(&net, &mut params, view).unwrap();
            if let Some(p) = trainer.last_profile() {
                for (least, &t) in forward.iter_mut().zip(&p.forward) {
                    *least = (*least).min(t);
                }
                for (least, &t) in backward.iter_mut().zip(&p.backward) {
                    *least = (*least).min(t);
                }
            }
        }
        let largest = |times: &[Duration]| (0..times.len()).max_by_key(|&i| times[i]).unwrap();
        assert_eq!(largest(&forward), 4, "forward {forward:?}");
        assert_eq!(largest(&backward), 4, "backward {backward:?}");
        assert!(forward[4] >= forward[0] * 4, "forward {forward:?}");
        assert!(backward[4] >= backward[0] * 4, "backward {backward:?}");
    }

    /// An override from `after_step` in the middle of an epoch applies from the next step
    /// and holds through the next epoch, which the cosine schedule would otherwise set:
    /// overriding to 0.05 after step 2 of 4 makes steps 3 to 8 train at 0.05, read from the
    /// trainer and back from the parameters.
    #[test]
    fn override_from_after_step_applies_next_step() {
        /// Overrides the rate after step 2; records the rate each step trained at and the
        /// bias's distance from its target after each step.
        struct Override {
            /// The bias parameter.
            bias: ParamId,
            /// `1 − b` before training and after each step.
            distances: Vec<f64>,
            /// The rate each step trained at.
            rates: Vec<f64>,
        }

        impl StepControl for Override {
            fn after_step(
                &mut self,
                info: &StepInfo,
                _model: &mut ModelHandle<'_>,
                params: &ParamStore,
                trainer: &mut TrainerHandle<'_>,
            ) -> Flow {
                self.rates.push(trainer.learning_rate());
                let b = f64::from(params.value(self.bias).data()[0]);
                self.distances.push(1.0 - b);
                if info.step == 2 {
                    trainer.override_learning_rate(0.05);
                }
                Flow::Continue
            }
        }

        let data = Dataset {
            x: Tensor::zeros(&[4, 1]),
            targets: Targets::Values(Tensor::full(&[4, 1], 1.0f32)),
        };
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 1, 1, &mut Rng::seed(0));
        let bias = layer.bias().unwrap();
        let mut net = Sequential::new().push(layer);
        let config = FitConfig {
            epochs: 2,
            batch: 1,
            optimizer: OptimizerConfig::Sgd {
                lr: 0.1,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::Mse,
            schedule: LrSchedule::Cosine { min_lr: 0.01 },
            health: Health::quiet(),
            ..FitConfig::default()
        };
        let mut control = Override {
            bias,
            distances: vec![1.0],
            rates: Vec::new(),
        };
        fit_controlled(
            &mut net,
            &mut params,
            &data,
            config,
            &mut Hooks::new(),
            &mut control,
        )
        .unwrap();
        let expected = [0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05];
        assert_eq!(control.rates, expected);
        let rates: Vec<f64> = control
            .distances
            .windows(2)
            .map(|w| (1.0 - w[1] / w[0]) / 2.0)
            .collect();
        assert_eq!(rates.len(), 8);
        for (step, (got, want)) in rates.iter().zip(expected).enumerate() {
            // The bias is f32, so the recovered rate carries its rounding.
            assert!(
                (got - want).abs() < 1e-5,
                "step {}: {got} vs {want}",
                step + 1
            );
        }
    }

    /// `fit_controlled` with a control that changes nothing trains to bit-identical
    /// parameters and the same epoch losses as `fit`, under a cosine schedule.
    #[test]
    fn fit_controlled_matches_fit() {
        let config = || FitConfig {
            schedule: LrSchedule::Cosine { min_lr: 1e-4 },
            ..control_config(3)
        };
        let (data, mut params, net) = control_setup();
        let plain = fit(&net, &mut params, &data, config()).unwrap();
        let (_, mut controlled_params, mut controlled_net) = control_setup();
        let controlled = fit_controlled(
            &mut controlled_net,
            &mut controlled_params,
            &data,
            config(),
            &mut Hooks::new(),
            &mut NoOp,
        )
        .unwrap();
        assert_eq!(
            parameter_bits(&net, &params),
            parameter_bits(&controlled_net, &controlled_params)
        );
        let losses = |h: &History| {
            h.epochs
                .iter()
                .map(|r| r.mean_loss.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(losses(&plain), losses(&controlled));
    }

    /// The held-out set of the validation tests: 40 rows of the control tests' problem, from
    /// another seed.
    fn validation_set() -> Dataset<f32> {
        synthetic_classes(40, 4, 2, 23)
    }

    /// The bits of an evaluation's loss, accuracy and R², to compare two evaluations exactly.
    fn evaluation_bits(e: &Evaluation) -> (u64, Option<u64>, Option<u64>) {
        (
            e.loss.to_bits(),
            e.accuracy.map(f64::to_bits),
            e.r2.map(f64::to_bits),
        )
    }

    /// `fit` with a validation set records one validation loss and accuracy per epoch, and
    /// each is bit for bit what `evaluate` measures (the same loss, at the run's batch size)
    /// on the parameters as that epoch left them, read in `on_epoch`. The figures change as
    /// the run trains, so each epoch reads its own parameters.
    #[test]
    fn validation_matches_evaluate_each_epoch() {
        let (data, mut params, net) = control_setup();
        let validation = validation_set();
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (record, held_out) = (Arc::clone(&seen), validation.clone());
        let config = FitConfig {
            validation: Some(validation),
            on_epoch: Some(Box::new(move |report, model, params| {
                let manual =
                    crate::evaluate(model, params, &held_out, Loss::CrossEntropy, 16).unwrap();
                record.lock().unwrap().push((report.validation, manual));
            })),
            ..control_config(5)
        };
        let history = fit(&net, &mut params, &data, config).unwrap();
        assert_eq!(history.epochs.len(), 5);
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 5);
        for (report, (in_callback, manual)) in history.epochs.iter().zip(seen.iter()) {
            let recorded = report.validation.expect("every epoch has a validation");
            assert!(recorded.accuracy.is_some() && recorded.r2.is_none());
            assert_eq!(evaluation_bits(&recorded), evaluation_bits(manual));
            assert_eq!(
                in_callback.map(|e| evaluation_bits(&e)),
                Some(evaluation_bits(manual))
            );
        }
        let first = history.epochs[0].validation.unwrap().loss;
        let last = history.epochs[4].validation.unwrap().loss;
        assert_ne!(first.to_bits(), last.to_bits());
    }

    /// With value targets the validation has a loss and R² and no accuracy, bit for bit
    /// `evaluate`'s on the trained parameters after the last epoch.
    #[test]
    fn validation_with_value_targets() {
        let mut rng = Rng::seed(31);
        let rows = |n: usize, rng: &mut Rng| {
            let x: Vec<f32> = (0..2 * n).map(|_| rng.range(-1.0, 1.0) as f32).collect();
            let y: Vec<f32> = x.chunks(2).map(|r| 2.0 * r[0] - r[1] + 0.5).collect();
            Dataset {
                x: Tensor::from_vec(x, &[n, 2]).unwrap(),
                targets: Targets::Values(Tensor::from_vec(y, &[n, 1]).unwrap()),
            }
        };
        let (data, validation) = (rows(48, &mut rng), rows(20, &mut rng));
        let mut params = ParamStore::new();
        let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(3)));
        let config = FitConfig {
            epochs: 3,
            batch: 8,
            loss: Loss::Mse,
            validation: Some(validation.clone()),
            ..FitConfig::default()
        };
        let history = fit(&net, &mut params, &data, config).unwrap();
        let last = history.epochs[2].validation.unwrap();
        assert!(last.accuracy.is_none() && last.r2.is_some());
        let manual = crate::evaluate(&net, &params, &validation, Loss::Mse, 8).unwrap();
        assert_eq!(evaluation_bits(&last), evaluation_bits(&manual));
    }

    /// The validation measures the run's loss, not the default cross-entropy: with
    /// `Loss::Nll` on class targets, after a `LogSoftmax` and on raw scores, each epoch's
    /// validation is bit for bit `evaluate` with `Loss::Nll`. On raw scores cross-entropy
    /// gives a different loss, so an evaluator built with the wrong loss fails here. (After a
    /// `LogSoftmax` cross-entropy equals NLL in exact arithmetic, as log-softmax of
    /// log-probabilities gives them back, which is why the raw-score case is there.)
    #[test]
    fn validation_uses_the_runs_class_loss() {
        let data = synthetic_classes(64, 4, 2, 21);
        let validation = validation_set();
        for log_softmax in [true, false] {
            let mut params = ParamStore::new();
            let mut net = small_classifier(&mut params, 4, 2, 22);
            if log_softmax {
                net = net.push(crate::LogSoftmax);
            }
            let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
            let (record, held_out) = (Arc::clone(&seen), validation.clone());
            let config = FitConfig {
                loss: Loss::Nll,
                validation: Some(validation.clone()),
                on_epoch: Some(Box::new(move |report, model, params| {
                    let nll = crate::evaluate(model, params, &held_out, Loss::Nll, 16).unwrap();
                    let ce =
                        crate::evaluate(model, params, &held_out, Loss::CrossEntropy, 16).unwrap();
                    record.lock().unwrap().push((report.validation, nll, ce));
                })),
                ..control_config(3)
            };
            let history = fit(&net, &mut params, &data, config).unwrap();
            assert_eq!(history.epochs.len(), 3);
            for (recorded, nll, ce) in seen.lock().unwrap().iter() {
                let recorded = recorded.expect("every epoch has a validation");
                assert!(recorded.accuracy.is_some());
                assert_eq!(evaluation_bits(&recorded), evaluation_bits(nll));
                if !log_softmax {
                    assert!(
                        (recorded.loss - ce.loss).abs() > 1e-3,
                        "{recorded:?} {ce:?}"
                    );
                }
            }
        }
    }

    /// Without a validation set (the default) no epoch has a validation.
    #[test]
    fn no_validation_by_default() {
        let (data, mut params, net) = control_setup();
        let history = fit(&net, &mut params, &data, control_config(2)).unwrap();
        assert!(history.epochs.iter().all(|r| r.validation.is_none()));
    }

    /// A validation set that does not suit the model or the loss is `InvalidData` before
    /// training, naming `validation`: rows of the wrong width, targets of the wrong kind, a
    /// label the model has no output for, a NaN input. The parameters stay as they were.
    #[test]
    fn unsuitable_validation_is_invalid_data() {
        let (data, mut params, net) = control_setup();
        let before = parameter_bits(&net, &params);
        let cases = [
            Dataset {
                x: Tensor::zeros(&[5, 3]),
                targets: Targets::Classes(vec![0; 5]),
            },
            Dataset {
                x: Tensor::zeros(&[5, 4]),
                targets: Targets::Values(Tensor::zeros(&[5, 2])),
            },
            Dataset {
                x: Tensor::zeros(&[5, 4]),
                targets: Targets::Classes(vec![0, 1, 2, 0, 1]),
            },
            Dataset {
                x: Tensor::from_vec(vec![f32::NAN; 20], &[5, 4]).unwrap(),
                targets: Targets::Classes(vec![0; 5]),
            },
            Dataset {
                x: Tensor::zeros(&[0, 4]),
                targets: Targets::Classes(Vec::new()),
            },
        ];
        for validation in cases {
            let config = FitConfig {
                validation: Some(validation),
                ..control_config(2)
            };
            match fit(&net, &mut params, &data, config) {
                Err(Error::InvalidData(msg)) => assert!(msg.starts_with("validation: "), "{msg}"),
                other => panic!("expected InvalidData, got {other:?}"),
            }
            assert_eq!(parameter_bits(&net, &params), before);
        }
    }

    /// Under `fit_controlled` the validation follows the masks the control changes: it
    /// switches neurons of the first `Linear` off and on between steps (adding the mask at
    /// step 2, changing its values at steps 6 and 10, clearing it at step 14 and adding it
    /// again at step 18), and every epoch's validation is bit for bit `evaluate`'s on the
    /// model as the epoch left it, masks included, read in `after_epoch`.
    #[test]
    fn validation_follows_masks_under_control() {
        /// Changes the first `Linear`'s neurons and compares each epoch's validation with
        /// `evaluate`.
        struct Masking {
            /// The held-out set.
            validation: Dataset<f32>,
            /// Each epoch's recorded and manual evaluations, and whether a mask was on.
            seen: Vec<(Evaluation, Evaluation, bool)>,
        }

        impl StepControl for Masking {
            fn after_step(
                &mut self,
                info: &StepInfo,
                model: &mut ModelHandle<'_>,
                _params: &ParamStore,
                _trainer: &mut TrainerHandle<'_>,
            ) -> Flow {
                match info.step {
                    2 | 18 => model.set_neuron(0, 3, false),
                    6 => model.set_neuron(0, 5, false),
                    10 => model.set_neuron(0, 3, true),
                    14 => model.clear_neuron_mask(0),
                    _ => {}
                }
                Flow::Continue
            }

            fn after_epoch(
                &mut self,
                report: &EpochReport,
                model: &mut ModelHandle<'_>,
                params: &ParamStore,
                _trainer: &mut TrainerHandle<'_>,
            ) -> Flow {
                let net = model.model();
                let manual =
                    crate::evaluate(net, params, &self.validation, Loss::CrossEntropy, 16).unwrap();
                let masked = net.neuron_mask(0).is_some();
                self.seen.push((report.validation.unwrap(), manual, masked));
                Flow::Continue
            }
        }

        let (data, mut params, mut net) = control_setup();
        let validation = validation_set();
        let mut control = Masking {
            validation: validation.clone(),
            seen: Vec::new(),
        };
        let config = FitConfig {
            validation: Some(validation.clone()),
            ..control_config(5)
        };
        fit_controlled(
            &mut net,
            &mut params,
            &data,
            config,
            &mut Hooks::new(),
            &mut control,
        )
        .unwrap();
        assert_eq!(control.seen.len(), 5);
        let masked: Vec<bool> = control.seen.iter().map(|s| s.2).collect();
        assert_eq!(masked, [true, true, true, false, true]);
        for (epoch, (recorded, manual, _)) in control.seen.iter().enumerate() {
            assert_eq!(
                evaluation_bits(recorded),
                evaluation_bits(manual),
                "epoch {}",
                epoch + 1
            );
        }
        // The mask matters: the model without it measures differently.
        let mut unmasked = net.clone();
        unmasked.clear_neuron_mask(0);
        let plain =
            crate::evaluate(&unmasked, &params, &validation, Loss::CrossEntropy, 16).unwrap();
        assert_ne!(evaluation_bits(&plain), evaluation_bits(&control.seen[4].0));
    }

    /// A control that stops the run part-way through an epoch leaves that epoch's report
    /// without a validation (the epoch did not end); the whole epochs before it have one.
    #[test]
    fn partial_epoch_has_no_validation() {
        /// Stops after step 6, the second step of epoch 2.
        struct StopAtSix;

        impl StepControl for StopAtSix {
            fn after_step(
                &mut self,
                info: &StepInfo,
                _model: &mut ModelHandle<'_>,
                _params: &ParamStore,
                _trainer: &mut TrainerHandle<'_>,
            ) -> Flow {
                if info.step == 6 {
                    Flow::Stop
                } else {
                    Flow::Continue
                }
            }
        }

        let (data, mut params, mut net) = control_setup();
        let config = FitConfig {
            validation: Some(validation_set()),
            ..control_config(3)
        };
        let history = fit_controlled(
            &mut net,
            &mut params,
            &data,
            config,
            &mut Hooks::new(),
            &mut StopAtSix,
        )
        .unwrap();
        assert_eq!(history.epochs.len(), 2);
        assert!(history.epochs[0].validation.is_some());
        assert_eq!(history.epochs[1].rows, 32);
        assert!(history.epochs[1].validation.is_none());
    }
}
