//! Training under the inspector: [`fit`], the [`StepControl`] the inspector runs between
//! steps, and the run state it keeps (the probe, the held-out evaluation, the pause, the
//! sampling clock and the events waiting to be published).
//!
//! Everything here runs on the training thread, and none of it waits for the server: the
//! client count is read without a lock, snapshots and events are published with `try_lock`
//! (an event the lock refused waits in `pending` for the next step), and commands are read
//! with `try_lock`. Nothing is captured while no client is connected.

use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};

use cavitation_core::{Error, ParamStore};
use cavitation_nn::{
    Dataset, EpochReport, Evaluation, Evaluator, FitConfig, Flow, History, Hooks, Loss,
    ModelHandle, Sequential, StepControl, StepInfo, Targets, TrainerHandle, fit_controlled,
};

use crate::commands::{Command, MAX_LEARNING_RATE};
use crate::probe::Probe;
use crate::server::{Inspector, InspectorConfig};
use crate::sse::{Event, Hub, Snapshot};
use crate::wire::{self, EpochEvaluation, ModelInfo, RunState, Stats};

/// How often a paused run looks for commands and for its clients leaving.
const PAUSE_POLL: Duration = Duration::from_millis(5);

/// A time no run reaches: a century after `now`, for a sampling period too long to add to
/// the clock.
fn far_future(now: Instant) -> Instant {
    now + Duration::from_secs(100 * 365 * 24 * 3600)
}

/// `n` followed by the singular or the plural noun: "1 level", "2 levels".
pub(crate) fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// An event that could not be published yet (the hub's lock was busy), kept to retry at the
/// next step.
#[derive(Debug)]
enum Pending {
    /// The `model` event.
    Model,
    /// A `status` event.
    Status(RunState, String),
    /// An `epoch` event, with its evaluation (measures, rows, seconds) when one ran; the
    /// misclassified rows are those of the latest evaluation.
    Epoch(EpochReport, Option<(Evaluation, usize, f64)>),
}

/// The training side's state: set up by [`fit`] for each run, read and changed between
/// steps by the [`StepControl`] methods.
#[derive(Debug)]
pub(crate) struct Training {
    /// The loss of the run, which the probe and the evaluation measure.
    loss: Loss,
    /// The epochs the run trains.
    epochs: usize,
    /// The rows per evaluation batch.
    batch: usize,
    /// The work of one training sample in the `Linear` layers.
    flops: u64,
    /// The number of parameter values.
    parameters: usize,
    /// The width of an input row.
    input_width: usize,
    /// Each layer's output width.
    pub(crate) widths: Vec<usize>,
    /// The layer index of every `Linear`, whose neurons the evaluator can switch.
    linears: Vec<usize>,
    /// The first training rows, the probe rows when the configuration names none.
    pub(crate) own_probe: Option<Dataset>,
    /// The first `eval_rows` probe rows, made when there are more probe rows than that.
    eval_set: Option<Dataset>,
    /// The time between captures.
    period: Duration,
    /// The probe, its buffers kept from run to run.
    probe: Probe,
    /// The probe row shown.
    row: usize,
    /// The neuron shown: (layer, neuron).
    neuron: Option<(usize, usize)>,
    /// Whether the page has paused the run.
    paused: bool,
    /// When the next capture is due; `None` before the first.
    next_sample: Option<Instant>,
    /// Whether a command asks for a capture at the next chance.
    capture_now: bool,
    /// Rows trained since the last `stats` snapshot.
    rows_since: usize,
    /// Training-step seconds since the last `stats` snapshot.
    seconds_since: f64,
    /// The last throughput measured, kept for snapshots taken while paused.
    samples_per_second: f64,
    /// The last step's number.
    pub(crate) step: u64,
    /// The epoch in progress.
    epoch: usize,
    /// The last step's batch loss.
    last_loss: f32,
    /// The held-out evaluator, made at the first evaluation.
    evaluator: Option<Evaluator>,
    /// The rows the last evaluation classified wrong.
    misclassified: Vec<usize>,
    /// When the last evaluation ended and what it cost.
    last_eval: Option<(Instant, Duration)>,
    /// Events waiting to be published, oldest first.
    pending: VecDeque<Pending>,
    /// How many times a snapshot has been captured.
    pub(crate) captures: u64,
    /// Whether the probe's last failure has been reported, so it is said once.
    probe_failed: bool,
}

impl Training {
    /// The state before any run.
    pub(crate) fn new() -> Training {
        Training {
            loss: Loss::CrossEntropy,
            epochs: 0,
            batch: 256,
            flops: 0,
            parameters: 0,
            input_width: 0,
            widths: Vec::new(),
            linears: Vec::new(),
            own_probe: None,
            eval_set: None,
            period: Duration::from_millis(250),
            probe: Probe::new(),
            row: 0,
            neuron: None,
            paused: false,
            next_sample: None,
            capture_now: false,
            rows_since: 0,
            seconds_since: 0.0,
            samples_per_second: f64::NAN,
            step: 0,
            epoch: 0,
            last_loss: f32::NAN,
            evaluator: None,
            misclassified: Vec::new(),
            last_eval: None,
            pending: VecDeque::with_capacity(16),
            captures: 0,
            probe_failed: false,
        }
    }

    /// The state a status message is sent with.
    fn state(&self) -> RunState {
        if self.paused {
            RunState::Paused
        } else {
            RunState::Training
        }
    }

    /// Queues a `status` event in the current state.
    fn say(&mut self, message: String) {
        let state = self.state();
        self.pending.push_back(Pending::Status(state, message));
    }

    /// The rows the page can probe: the configuration's, or the first training rows.
    fn probe_set<'a>(config: &'a InspectorConfig, own: &'a Option<Dataset>) -> Option<&'a Dataset> {
        config.probe.as_ref().or(own.as_ref())
    }

    /// Publishes the waiting events in order, stopping at the first the hub refuses; with
    /// `blocking`, waits for the hub's lock (only once training is over).
    fn flush(&mut self, config: &InspectorConfig, hub: &Hub, model: &Sequential, blocking: bool) {
        while let Some(event) = self.pending.front() {
            if hub.stopping().load(Ordering::Acquire) {
                self.pending.clear();
                return;
            }
            let published = match event {
                Pending::Model => {
                    let probe = Training::probe_set(config, &self.own_probe);
                    let probe_rows = probe.map_or(0, Dataset::len);
                    let info = ModelInfo {
                        title: &config.title,
                        input_width: self.input_width,
                        input_image: config.input_image,
                        class_names: config.class_names.as_deref(),
                        probe_rows,
                        probe_is_training: config.probe.is_none(),
                        eval_rows: config.eval_rows.min(probe_rows),
                        parameters: self.parameters,
                        flops_per_sample: self.flops,
                    };
                    hub.publish_event(Event::Model, |w| wire::model(w, model, &info), blocking)
                }
                Pending::Status(state, message) => hub.publish_event(
                    Event::Status,
                    |w| wire::status(w, *state, message),
                    blocking,
                ),
                Pending::Epoch(report, evaluation) => {
                    let evaluation =
                        evaluation.map(|(evaluation, rows, seconds)| EpochEvaluation {
                            evaluation,
                            rows,
                            misclassified: &self.misclassified,
                            seconds,
                        });
                    hub.publish_event(
                        Event::Epoch,
                        |w| wire::epoch(w, report, self.epochs, evaluation.as_ref()),
                        blocking,
                    )
                }
            };
            if !published {
                if blocking {
                    // A blocking publish fails only once the hub is closed.
                    self.pending.clear();
                }
                return;
            }
            self.pending.pop_front();
        }
    }

    /// Captures a snapshot: the `stats`, then the probe row and the chosen neuron. Counted
    /// in `captures`. The caller checks that a client is connected. Returns whether every
    /// publish went through; when the hub's lock refused one, `capture_now` stays set, so
    /// the snapshot is taken again at the next chance.
    fn capture(
        &mut self,
        config: &InspectorConfig,
        hub: &Hub,
        model: &Sequential,
        params: &ParamStore,
        learning_rate: f64,
    ) -> bool {
        self.captures += 1;
        if self.rows_since > 0 && self.seconds_since > 0.0 {
            self.samples_per_second = self.rows_since as f64 / self.seconds_since;
            self.rows_since = 0;
            self.seconds_since = 0.0;
        }
        let stats = Stats {
            step: self.step,
            epoch: self.epoch,
            epochs: self.epochs,
            samples_per_second: self.samples_per_second,
            gflops: self.samples_per_second * self.flops as f64 / 1e9,
            loss: self.last_loss,
            learning_rate,
            paused: self.paused,
        };
        let mut published = hub.publish_snapshot(Snapshot::Stats, |w| wire::stats(w, &stats));
        let Training {
            probe,
            own_probe,
            row,
            neuron,
            loss,
            step,
            ..
        } = self;
        if let Some(data) = Training::probe_set(config, own_probe) {
            match probe.run(model, params, data, *row, *loss) {
                Ok(()) => {
                    self.probe_failed = false;
                    published &= hub.publish_snapshot(Snapshot::Probe, |w| {
                        wire::probe(w, *step, probe, model);
                    });
                    if let Some((layer, n)) = *neuron
                        && probe.neuron(model, params, layer, n).is_ok()
                    {
                        published &= hub.publish_snapshot(Snapshot::Neuron, |w| {
                            wire::neuron(w, *step, probe, probe.last_neuron());
                        });
                    }
                }
                Err(e) => {
                    if !self.probe_failed {
                        self.probe_failed = true;
                        self.say(format!("the probe row cannot be shown: {e}"));
                    }
                }
            }
        }
        self.capture_now = !published;
        published
    }

    /// Checks a page's request to switch neuron `neuron` of the `linear`-th `Linear`
    /// (counting only `Linear`s), and returns the `Linear`'s layer index, or why it cannot
    /// be switched.
    pub(crate) fn mask_target(
        model: &Sequential,
        linear: usize,
        neuron: usize,
    ) -> Result<usize, String> {
        let Some((layer, lin)) = model.linears().nth(linear) else {
            return Err(format!(
                "cannot switch a neuron of Linear layer {linear}: the model has {}",
                plural(model.linears().count(), "Linear layer", "Linear layers")
            ));
        };
        let width = lin.outputs();
        if neuron >= width {
            return Err(format!(
                "cannot switch neuron {neuron} of Linear layer {linear} (layer {layer}): it has \
                 {width} neurons"
            ));
        }
        let at = model.neuron_layer(layer);
        if let Some(other) = (layer..at).find(|&l| model.mask(l).is_some()) {
            return Err(format!(
                "cannot switch neurons of Linear layer {linear}: layer {other} carries a mask \
                 set by layer index"
            ));
        }
        if model.linears().all(|(i, _)| i <= layer) {
            return Err(format!(
                "cannot switch neurons of Linear layer {linear}: it is the output layer, and \
                 switching an output off would change what the network answers"
            ));
        }
        if let Some(mask) = model.mask(at)
            && mask.width() != width
        {
            return Err(format!(
                "cannot switch neurons of Linear layer {linear}: the mask on layer {at} has \
                 width {}, not {width}",
                mask.width()
            ));
        }
        Ok(layer)
    }

    /// Carries out one command from the page. Every index and value is checked first; a
    /// command that cannot be carried out is answered with a `status` message.
    fn command(
        &mut self,
        command: Command,
        config: &InspectorConfig,
        model: &mut ModelHandle<'_>,
        trainer: &mut TrainerHandle<'_>,
    ) {
        match command {
            Command::SetMask { linear, neuron, on } => {
                match Training::mask_target(model.model(), linear, neuron) {
                    Ok(layer) => {
                        model.set_neuron(layer, neuron, on);
                        if let (Some(evaluator), Some(mask)) =
                            (self.evaluator.as_mut(), model.model().neuron_mask(layer))
                        {
                            // Kept in step with the model; `refresh` before each
                            // evaluation carries the masks over too.
                            let _ = evaluator.set_neuron_mask(layer, mask);
                        }
                        self.capture_now = true;
                    }
                    Err(message) => self.say(message),
                }
            }
            Command::SetLearningRate(rate) => {
                if rate.is_finite() && (0.0..=MAX_LEARNING_RATE).contains(&rate) {
                    trainer.override_learning_rate(rate);
                    self.capture_now = true;
                } else {
                    self.say(format!(
                        "the learning rate must be at least 0 and at most \
                         {MAX_LEARNING_RATE}, not {rate}"
                    ));
                }
            }
            Command::Pause => {
                if !self.paused {
                    self.paused = true;
                    self.capture_now = true;
                    self.say("paused from the page".to_string());
                }
            }
            Command::Resume => {
                if self.paused {
                    self.paused = false;
                    self.say("resumed from the page".to_string());
                }
            }
            Command::SelectProbe(index) => {
                let rows = Training::probe_set(config, &self.own_probe).map_or(0, Dataset::len);
                if index < rows {
                    self.row = index;
                    self.probe_failed = false;
                    self.capture_now = true;
                } else {
                    self.say(format!(
                        "there is no probe row {index}: there {} {}",
                        if rows == 1 { "is" } else { "are" },
                        plural(rows, "probe row", "probe rows")
                    ));
                }
            }
            Command::SelectNeuron { layer, neuron } => match self.widths.get(layer) {
                Some(&width) if neuron < width => {
                    self.neuron = Some((layer, neuron));
                    self.capture_now = true;
                }
                Some(&width) => self.say(format!(
                    "there is no neuron {neuron} in layer {layer}: it has width {width}"
                )),
                None => self.say(format!(
                    "there is no layer {layer}: the model has {}",
                    plural(self.widths.len(), "layer", "layers")
                )),
            },
            Command::Ablate { .. } | Command::Shrink | Command::Cancel | Command::Finish => {
                self.say(
                    "ablation, shrinking, cancel and finish work on a trained network in \
                     explore mode, not during training"
                        .to_string(),
                );
            }
        }
    }

    /// Reads and carries out every command waiting now.
    fn serve(
        &mut self,
        inspector: (&InspectorConfig, &Hub, &crate::commands::CommandQueue),
        model: &mut ModelHandle<'_>,
        trainer: &mut TrainerHandle<'_>,
    ) {
        let (config, _, commands) = inspector;
        while let Some(command) = commands.try_pop() {
            self.command(command, config, model, trainer);
        }
    }

    /// Holds a paused run between two steps until the page resumes it, no client has been
    /// connected for `idle_resume`, or the server shuts down. Commands are served meanwhile,
    /// and a snapshot is captured whenever one changes what the page shows.
    fn hold(
        &mut self,
        inspector: (&InspectorConfig, &Hub, &crate::commands::CommandQueue),
        model: &mut ModelHandle<'_>,
        params: &ParamStore,
        trainer: &mut TrainerHandle<'_>,
    ) {
        let (config, hub, _) = inspector;
        let mut idle_since: Option<Instant> = None;
        loop {
            if hub.stopping().load(Ordering::Acquire) {
                self.paused = false;
                return;
            }
            self.serve(inspector, model, trainer);
            self.flush(config, hub, model.model(), false);
            if !self.paused {
                return;
            }
            let now = Instant::now();
            if hub.clients() == 0 {
                let since = *idle_since.get_or_insert(now);
                if now.duration_since(since) >= config.idle_resume {
                    self.paused = false;
                    self.say(format!(
                        "resumed: no page was connected for {:.1} s",
                        config.idle_resume.as_secs_f64()
                    ));
                    self.flush(config, hub, model.model(), false);
                    return;
                }
            } else {
                idle_since = None;
                if self.capture_now {
                    // A refused publish leaves `capture_now` set for the next poll.
                    self.capture(config, hub, model.model(), params, trainer.learning_rate());
                }
            }
            thread::sleep(PAUSE_POLL);
        }
    }

    /// Runs the held-out evaluation on the first `eval_rows` probe rows: its measures, the
    /// rows measured and its seconds, or `None` (with a `status` message on an error) when
    /// there is nothing to measure.
    fn evaluate(
        &mut self,
        config: &InspectorConfig,
        model: &Sequential,
        params: &ParamStore,
    ) -> Option<(Evaluation, usize, f64)> {
        let start = Instant::now();
        let probe = Training::probe_set(config, &self.own_probe)?;
        let rows = config.eval_rows.min(probe.len());
        if rows == 0 {
            return None;
        }
        if rows < probe.len() && self.eval_set.as_ref().is_none_or(|s| s.len() != rows) {
            self.eval_set = Some(probe.subset(0..rows));
        }
        let data = if rows < probe.len() {
            self.eval_set.as_ref()?
        } else {
            probe
        };
        let ready = match self.evaluator.as_mut() {
            Some(evaluator) => evaluator.refresh(model, params),
            None => {
                Evaluator::with_neuron_masks(model, params, self.loss, self.batch, &self.linears)
                    .map(|e| self.evaluator = Some(e))
            }
        };
        let evaluator = self.evaluator.as_mut()?;
        let measured = ready.and_then(|()| evaluator.evaluate(data));
        let evaluation = match measured {
            Ok(e) => e,
            Err(e) => {
                self.say(format!("the evaluation could not run: {e}"));
                return None;
            }
        };
        if matches!(data.targets, Targets::Classes(_)) {
            if evaluator
                .misclassified(data, &mut self.misclassified)
                .is_err()
            {
                self.misclassified.clear();
            }
        } else {
            self.misclassified.clear();
        }
        let cost = start.elapsed();
        self.last_eval = Some((Instant::now(), cost));
        Some((evaluation, rows, cost.as_secs_f64()))
    }
}

/// Checks the configuration's probe set against the training data and the loss.
pub(crate) fn check_probe(
    probe: &Dataset,
    train: Option<&Dataset>,
    loss: Loss,
) -> Result<(), Error> {
    let bad = |why: String| Err(Error::InvalidData(format!("inspector probe set: {why}")));
    let dims = probe.x.shape().dims();
    if dims.len() != 2 || dims[0] == 0 {
        return bad(format!(
            "x must be a matrix with at least one row, not {dims:?}"
        ));
    }
    let targets = match &probe.targets {
        Targets::Classes(labels) => labels.len(),
        Targets::Values(t) => t.shape().dims().first().copied().unwrap_or(0),
    };
    if targets != dims[0] {
        return bad(format!("{} rows but {targets} targets", dims[0]));
    }
    if let Targets::Values(t) = &probe.targets
        && t.shape().dims().len() != 2
    {
        return bad(format!(
            "value targets must be a matrix, not {:?}",
            t.shape().dims()
        ));
    }
    if let Some(train) = train
        && train.features() != dims[1]
    {
        return bad(format!(
            "rows of {} features, but the training rows have {}",
            dims[1],
            train.features()
        ));
    }
    let classes = matches!(probe.targets, Targets::Classes(_));
    if matches!(loss, Loss::CrossEntropy | Loss::Nll) != classes {
        return bad(format!(
            "{} targets, which Loss::{loss:?} does not take",
            if classes { "class" } else { "value" }
        ));
    }
    Ok(())
}

/// Whether `data` has the shape training needs (a matrix with rows, one target per row), so
/// its first rows can be copied; `fit` reports the problem when it does not.
pub(crate) fn trainable(data: &Dataset) -> bool {
    let dims = data.x.shape().dims();
    let targets = match &data.targets {
        Targets::Classes(labels) => labels.len(),
        Targets::Values(t) => match t.shape().dims() {
            &[rows, _] => rows,
            _ => usize::MAX,
        },
    };
    dims.len() == 2 && dims[0] > 0 && targets == dims[0]
}

impl Inspector {
    /// Prepares a run of `model` on `train`: the probe rows, the counts the page shows, and
    /// the `model` and `status` events.
    fn begin(
        &mut self,
        model: &Sequential,
        params: &ParamStore,
        train: &Dataset,
        fit: &FitConfig,
    ) -> Result<(), Error> {
        let Inspector {
            config,
            shared,
            training: t,
            ..
        } = self;
        let valid = trainable(train);
        if let Some(probe) = &config.probe {
            check_probe(probe, valid.then_some(train), fit.loss)?;
        }
        // A fresh state, probe included: the probe's gradient set belongs to one parameter
        // store, and this run's store may be another.
        let captures = t.captures;
        *t = Training::new();
        t.captures = captures;
        t.loss = fit.loss;
        t.epochs = fit.epochs;
        t.batch = fit.batch.max(1);
        // `start` checked that the period fits a `Duration`.
        t.period = Duration::try_from_secs_f64(1.0 / config.snapshot_hz).unwrap_or(Duration::MAX);
        if valid {
            t.input_width = train.features();
            if config.probe.is_none() {
                let rows = config.eval_rows.max(1).min(train.len());
                t.own_probe = Some(train.subset(0..rows));
            }
        }
        let mut width = t.input_width;
        t.widths = (0..model.len())
            .map(|i| {
                width = model.layer(i).output_width(width);
                width
            })
            .collect();
        t.linears = model.linears().map(|(i, _)| i).collect();
        t.neuron = t.linears.first().map(|&l| (model.neuron_layer(l), 0));
        t.flops = wire::flops_per_sample(model);
        t.parameters = wire::parameter_count(model, params);
        t.pending.push_back(Pending::Model);
        t.say(format!(
            "training started: {}",
            plural(fit.epochs, "epoch", "epochs")
        ));
        // A new run: a page that connects later must not see an earlier explore mode's
        // shrink result. A refused lock leaves the old one, which the page drops on `model`.
        shared
            .hub
            .publish_snapshot(Snapshot::Network, |w| wire::network(w, None));
        t.flush(config, &shared.hub, model, false);
        Ok(())
    }

    /// Ends a run: says how it ended and publishes every waiting event, waiting for the
    /// hub's lock now that training is over.
    fn end(&mut self, model: &Sequential, result: &Result<History, Error>) {
        let Inspector {
            config,
            shared,
            training: t,
            ..
        } = self;
        t.paused = false;
        let message = match result {
            Ok(history) => format!(
                "training finished after {}",
                plural(history.epochs.len(), "epoch", "epochs")
            ),
            Err(e) => format!("training stopped: {e}"),
        };
        t.pending
            .push_back(Pending::Status(RunState::Finished, message));
        t.flush(config, &shared.hub, model, true);
    }
}

impl StepControl for Inspector {
    /// Between two steps: publishes the profile of a profiled step and any event still
    /// waiting, carries out the page's commands, holds the run while it is paused, and on a
    /// sampled step (at most `snapshot_hz` a second, checked with one clock read, and only
    /// while a client is connected) captures a snapshot and asks for the next step's
    /// profile.
    ///
    /// Run it through [`fit`], which prepares the probe rows and the page's description of
    /// the model; passed to [`fit_controlled`] directly it streams the statistics and the
    /// profiles, without probe or evaluation.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_inspect::{Inspector, InspectorConfig};
    /// use cavitation_nn::{
    ///     Flow, Linear, Loss, ModelHandle, Sequential, Sgd, StepControl, StepInfo, Trainer,
    ///     TrainerHandle,
    /// };
    /// let config = InspectorConfig { address: "127.0.0.1:0".parse().unwrap(), ..Default::default() };
    /// let mut inspector = Inspector::start(config)?;
    /// let mut params = ParamStore::new();
    /// let mut net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 4);
    /// let info = StepInfo { step: 1, epoch: 1, loss: 0.5, rows: 4, seconds: 0.001 };
    /// let flow = inspector.after_step(
    ///     &info,
    ///     &mut ModelHandle::new(&mut net),
    ///     &params,
    ///     &mut TrainerHandle::new(&mut trainer),
    /// );
    /// assert_eq!(flow, Flow::Continue);
    /// assert_eq!(inspector.snapshots_taken(), 0); // no client is connected
    /// # Ok::<(), cavitation_core::Error>(())
    /// ```
    fn after_step(
        &mut self,
        info: &StepInfo,
        model: &mut ModelHandle<'_>,
        params: &ParamStore,
        trainer: &mut TrainerHandle<'_>,
    ) -> Flow {
        let Inspector {
            config,
            shared,
            training: t,
            ..
        } = self;
        let hub = &shared.hub;
        let inspector = (&*config, hub, &shared.commands);
        t.step = info.step;
        t.epoch = info.epoch;
        t.last_loss = info.loss;
        t.rows_since += info.rows;
        t.seconds_since += info.seconds;
        if hub.clients() > 0
            && let Some(profile) = trainer.last_profile()
        {
            hub.publish_snapshot(Snapshot::Profile, |w| {
                wire::profile(w, info.step, model.model(), profile);
            });
        }
        t.flush(config, hub, model.model(), false);
        t.serve(inspector, model, trainer);
        if t.paused {
            t.hold(inspector, model, params, trainer);
        }
        // Once the hub has closed nothing can be published, though its client list stays.
        if hub.clients() > 0 && !hub.stopping().load(Ordering::Acquire) {
            let now = Instant::now();
            if (t.capture_now || t.next_sample.is_none_or(|due| now >= due))
                && t.capture(config, hub, model.model(), params, trainer.learning_rate())
            {
                // A capture the hub's lock refused is taken again at the next step.
                // A period too long to add to the clock means no further sample this run.
                t.next_sample = Some(now.checked_add(t.period).unwrap_or_else(|| far_future(now)));
                trainer.profile_next_step();
            }
        }
        t.flush(config, hub, model.model(), false);
        Flow::Continue
    }

    /// After an epoch: carries out waiting commands, runs the held-out evaluation when it
    /// is due, and publishes the `epoch` event.
    ///
    /// The evaluation runs only while a client is connected, and only when the time since
    /// the last one ended is at least 100 times that one's own cost, so it takes at most
    /// about 1% of the run; the last epoch always has one while a client is connected.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_inspect::{Inspector, InspectorConfig};
    /// use cavitation_nn::{
    ///     EpochReport, Flow, Linear, Loss, ModelHandle, Sequential, Sgd, StepControl, Trainer,
    ///     TrainerHandle,
    /// };
    /// let config = InspectorConfig { address: "127.0.0.1:0".parse().unwrap(), ..Default::default() };
    /// let mut inspector = Inspector::start(config)?;
    /// let mut params = ParamStore::new();
    /// let mut net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 4);
    /// let report = EpochReport { epoch: 1, mean_loss: 0.5, rows: 4, seconds: 0.01 };
    /// let flow = inspector.after_epoch(
    ///     &report,
    ///     &mut ModelHandle::new(&mut net),
    ///     &params,
    ///     &mut TrainerHandle::new(&mut trainer),
    /// );
    /// assert_eq!(flow, Flow::Continue);
    /// # Ok::<(), cavitation_core::Error>(())
    /// ```
    fn after_epoch(
        &mut self,
        report: &EpochReport,
        model: &mut ModelHandle<'_>,
        params: &ParamStore,
        trainer: &mut TrainerHandle<'_>,
    ) -> Flow {
        let Inspector {
            config,
            shared,
            training: t,
            ..
        } = self;
        let hub = &shared.hub;
        t.flush(config, hub, model.model(), false);
        t.serve((&*config, hub, &shared.commands), model, trainer);
        let last = report.epoch >= t.epochs;
        let due = t
            .last_eval
            .is_none_or(|(end, cost)| end.elapsed() >= cost.saturating_mul(100));
        let evaluation = if hub.clients() > 0 && (last || due) {
            t.evaluate(config, model.model(), params)
        } else {
            None
        };
        t.pending.push_back(Pending::Epoch(*report, evaluation));
        t.flush(config, hub, model.model(), false);
        Flow::Continue
    }
}

/// Trains `model` as [`cavitation_nn::fit`] does, under the inspector: the page shows the
/// run live, and can switch neurons, change the learning rate, pause and resume, and pick
/// the probe row and the neuron to show. This is [`fit_controlled`] with no hooks and the
/// inspector as its control.
///
/// Before training it checks the configuration's probe set, copies the first `eval_rows`
/// training rows (at least one) to probe when there is none, and publishes the `model` and
/// `status` events; between steps the inspector runs as [`StepControl::after_step`] says;
/// after training it publishes a final `status`. With no client connected nothing is
/// captured, and a run with fixed seeds and no commands gives the parameters of
/// [`cavitation_nn::fit`], bit for bit, whether a client is reading or not. Masks switched
/// from the page stay on `model` afterwards.
///
/// # Errors
///
/// Returns [`Error::InvalidData`] if the configuration's probe set is not a matrix of
/// rows as wide as the training rows with one target per row of the kind the loss takes,
/// and the errors of [`fit_controlled`].
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_inspect::{Inspector, InspectorConfig};
/// use cavitation_nn::{Dataset, FitConfig, Linear, Loss, Relu, Sequential, Targets};
/// let config = InspectorConfig { address: "127.0.0.1:0".parse().unwrap(), ..Default::default() };
/// let mut inspector = Inspector::start(config)?;
/// println!("open {}", inspector.url());
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let mut net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 8, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 8, 2, &mut rng));
/// let data = Dataset {
///     x: Tensor::from_vec(vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]).unwrap(),
///     targets: Targets::Classes(vec![0, 1, 1, 0]),
/// };
/// let config = FitConfig { epochs: 5, batch: 4, loss: Loss::CrossEntropy, ..FitConfig::default() };
/// let history = cavitation_inspect::fit(&mut net, &mut params, &data, config, &mut inspector)?;
/// assert_eq!(history.epochs.len(), 5);
/// # Ok::<(), cavitation_core::Error>(())
/// ```
pub fn fit(
    model: &mut Sequential,
    params: &mut ParamStore,
    train: &Dataset,
    config: FitConfig,
    inspector: &mut Inspector,
) -> Result<History, Error> {
    inspector.begin(model, params, train, &config)?;
    let result = fit_controlled(model, params, train, config, &mut Hooks::new(), inspector);
    inspector.end(model, &result);
    result
}
