//! Explore mode: [`Inspector::explore`] serves a trained network to the page, which can
//! switch its neurons on and off, rank a hidden layer's neurons by what switching each off
//! costs, and shrink the network with the shrink tool, until the page says it is done.
//!
//! Everything runs on the caller's thread, which has nothing else to do meanwhile, so
//! events are published with the hub's blocking `publish_event`. A ranking or a shrink run
//! can take a long time: while one runs, a second thread answers the page's commands at
//! once, so `Cancel` and `Finish` are acknowledged straight away and take effect after the
//! neuron or the level in progress, and a command that cannot run beside it is answered
//! with a `status` message.

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use cavitation_core::{Error, ParamStore};
use cavitation_nn::shrink::{Ablation, ShrinkConfig, ShrinkReport, ShrinkStep};
use cavitation_nn::{Dataset, Flow, Loss, Sequential};

use crate::commands::Command;
use crate::probe::Probe;
use crate::server::{Inspector, InspectorConfig, Shared};
use crate::snapshot::{Training, check_probe, plural, trainable};
use crate::sse::{Event, Hub, Snapshot};
use crate::wire::{self, ModelInfo, RunState};

/// How often explore mode looks for commands and for its clients leaving.
const POLL: Duration = Duration::from_millis(5);

/// The rows measured at a time in a ranking.
const RANK_BATCH: usize = 256;

/// How many of the cheapest neurons a ranking's `status` message names.
const RANK_NAMED: usize = 10;

/// The identifier of the next ranking or shrink run, unique in the process, so the page
/// can tell one run's events from another's (the `sweep` and `run` fields).
static NEXT_RUN: AtomicU64 = AtomicU64::new(1);

/// What explore mode works with, beside the inspector's own configuration.
///
/// The page's neuron switches and the ranking need only the network. The shrink lab needs
/// the training rows to retrain on and a [`ShrinkConfig`] for each run: `shrink` builds a
/// fresh one for each `Shrink` command, and it must name a
/// [`validation`](ShrinkConfig::validation) set, which has no default. Without either, the
/// page's `Shrink` is answered with a `status` message saying what is missing.
///
/// [`Default`] is cross-entropy with neither training rows nor a factory; set the fields
/// you need and fill the rest with `..ExploreConfig::default()`, so a field added in a
/// later release does not break your code. The struct is not `#[non_exhaustive]`, which
/// would forbid that update syntax outside this crate.
///
/// A shrink run reads the page's `Cancel` at once but can stop only between levels, so a
/// cancel takes effect when the level in progress, its retraining included, ends: with
/// many training rows or retraining epochs per level that can take a while.
///
/// # Examples
///
/// ```
/// use cavitation_core::Tensor;
/// use cavitation_inspect::ExploreConfig;
/// use cavitation_nn::shrink::{ShrinkConfig, Validation};
/// use cavitation_nn::{Dataset, Loss, Targets};
/// let held_out = Dataset { x: Tensor::zeros(&[300, 4]), targets: Targets::Classes(vec![0; 300]) };
/// let config = ExploreConfig {
///     train: Some(Dataset {
///         x: Tensor::zeros(&[100, 4]),
///         targets: Targets::Classes(vec![0; 100]),
///     }),
///     shrink: Some(Box::new(move || ShrinkConfig {
///         validation: Some(Validation::Given(held_out.clone())),
///         ..ShrinkConfig::default()
///     })),
///     ..ExploreConfig::default()
/// };
/// assert_eq!(config.loss, Loss::CrossEntropy);
/// assert!(format!("{config:?}").contains("CrossEntropy"));
/// ```
pub struct ExploreConfig<'s> {
    /// The loss the probe and the ranking measure with: the one the network was trained
    /// with.
    pub loss: Loss,
    /// The training rows, which the shrink lab retrains on; its first `eval_rows` rows are
    /// also the probe rows when the inspector's configuration names none. `None` turns the
    /// shrink lab off.
    pub train: Option<Dataset>,
    /// Builds a fresh shrink configuration for each `Shrink` command; its `on_step` is
    /// replaced by one that streams each level to the page and stops on `Cancel`. `None`
    /// turns the shrink lab off. The factory and what it builds are `Send` and may borrow
    /// for `'s`, so an `ExploreConfig` can be made on another thread than the one that
    /// explores.
    // The boxed factory is the plainest type for this field; an alias would only hide it.
    #[allow(clippy::type_complexity)]
    pub shrink: Option<Box<dyn FnMut() -> ShrinkConfig<'s> + Send + 's>>,
}

impl Default for ExploreConfig<'_> {
    /// Cross-entropy, no training rows and no shrink factory: neuron switches and rankings
    /// work, and the shrink lab answers that it needs `train` and `shrink`.
    fn default() -> Self {
        ExploreConfig {
            loss: Loss::CrossEntropy,
            train: None,
            shrink: None,
        }
    }
}

impl fmt::Debug for ExploreConfig<'_> {
    /// Writes the loss, the training rows' shape and whether there is a shrink factory.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExploreConfig")
            .field("loss", &self.loss)
            .field(
                "train",
                &self.train.as_ref().map(|t| t.x.shape().dims().to_vec()),
            )
            .field("shrink", &self.shrink.as_ref().map(|_| "FnMut"))
            .finish()
    }
}

/// What to do after a command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Next {
    /// Go on serving the page.
    Stay,
    /// The page sent `Finish`: return.
    Finish,
}

/// Publishes a `status` event in state `explore`, waiting for the hub's lock.
fn say(hub: &Hub, message: &str) {
    hub.publish_event(
        Event::Status,
        |w| wire::status(w, RunState::Explore, message),
        true,
    );
}

/// What the thread that answers commands during a run has been asked.
#[derive(Debug, Default)]
struct Watch {
    /// The page asked to stop the run (`Cancel` or `Finish`).
    stop: AtomicBool,
    /// The page asked to end explore mode once the run has stopped (`Finish`).
    finish: AtomicBool,
    /// The run has returned, so the thread ends.
    done: AtomicBool,
}

/// Sets its flag when dropped, so the command thread ends even if the run panics.
struct Done<'a>(&'a AtomicBool);

impl Drop for Done<'_> {
    /// Raises the flag.
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// Runs `run` on this thread while a second thread answers the page's commands every
/// [`POLL`]: `Cancel` and `Finish` raise [`Watch::stop`] (and `Finish` [`Watch::finish`])
/// and are acknowledged at once with `cancelling`, and anything else is answered with a
/// `status` saying that `what` is in progress. `run` reads the flags at its own stopping
/// points. The commands already queued are answered on this thread before `run` starts,
/// so a cancel sent before the run began takes effect at its first stopping point, however
/// late the second thread is scheduled.
fn watched<R>(
    shared: &Shared,
    what: &str,
    cancelling: &str,
    run: impl FnOnce(&Arc<Watch>) -> R,
) -> R {
    let watch = Arc::new(Watch::default());
    // Answers every command queued now.
    let answer = |flags: &Watch| {
        let hub = &shared.hub;
        while let Some(command) = shared.commands.try_pop() {
            match command {
                Command::Cancel | Command::Finish => {
                    if command == Command::Finish {
                        flags.finish.store(true, Ordering::Release);
                    }
                    if !flags.stop.swap(true, Ordering::AcqRel) {
                        say(hub, cancelling);
                    }
                }
                _ => say(
                    hub,
                    &format!("{what} is in progress: Cancel ends it, or wait for it to end"),
                ),
            }
        }
    };
    answer(&watch);
    thread::scope(|s| {
        let flags = Arc::clone(&watch);
        s.spawn(move || {
            let hub = &shared.hub;
            while !flags.done.load(Ordering::Acquire) && !hub.stopping().load(Ordering::Acquire) {
                answer(&flags);
                thread::sleep(POLL);
            }
        });
        let _done = Done(&watch.done);
        run(&watch)
    })
}

/// The width of the model's input rows when its first layer is a `Linear`, which fixes it.
fn input_width(model: &Sequential) -> Option<usize> {
    model
        .linears()
        .next()
        .filter(|&(i, _)| i == 0)
        .map(|(_, l)| l.inputs())
}

/// Explore mode's state between commands: the probe and what it shows.
struct Explore<'a> {
    /// The loss the probe and the ranking measure with.
    loss: Loss,
    /// The rows the page can probe, if any.
    probe_set: Option<&'a Dataset>,
    /// The rows a ranking measures on: the first `eval_rows` probe rows.
    eval: Option<Cow<'a, Dataset>>,
    /// Whether those rows are training rows (no probe set was configured).
    eval_is_training: bool,
    /// Each layer's output width.
    widths: Vec<usize>,
    /// The probe; replaced after a shrink run, which can replace the parameter store.
    probe: Probe,
    /// The probe row shown.
    row: usize,
    /// The neuron shown: (layer, neuron).
    neuron: Option<(usize, usize)>,
    /// The training step the network was left at, shown with each probe.
    step: u64,
    /// Whether the page needs a fresh probe and neuron.
    capture_now: bool,
    /// Whether the probe's last failure has been reported, so it is said once.
    probe_failed: bool,
    /// The level the last shrink run kept (its checkpoint dropped), for the `network`
    /// snapshot.
    shrunk: Option<ShrinkStep>,
    /// Whether the `network` snapshot needs publishing.
    network_due: bool,
}

impl Explore<'_> {
    /// Runs the probe on the chosen row and publishes it with the chosen neuron. A publish
    /// the hub's lock refused leaves `capture_now` set, so it is tried again at the next
    /// poll.
    fn capture(&mut self, hub: &Hub, model: &Sequential, params: &ParamStore) {
        let Some(data) = self.probe_set else {
            self.capture_now = false;
            return;
        };
        if let Err(e) = self.probe.run(model, params, data, self.row, self.loss) {
            if !self.probe_failed {
                self.probe_failed = true;
                say(hub, &format!("the probe row cannot be shown: {e}"));
            }
            self.capture_now = false;
            return;
        }
        self.probe_failed = false;
        let (probe, step) = (&mut self.probe, self.step);
        let mut published =
            hub.publish_snapshot(Snapshot::Probe, |w| wire::probe(w, step, probe, model));
        if let Some((layer, n)) = self.neuron
            && probe.neuron(model, params, layer, n).is_ok()
        {
            published &= hub.publish_snapshot(Snapshot::Neuron, |w| {
                wire::neuron(w, step, probe, probe.last_neuron());
            });
        }
        self.capture_now = !published;
    }

    /// Publishes the `network` snapshot; a publish the hub's lock refused is tried again at
    /// the next poll.
    fn publish_network(&mut self, hub: &Hub) {
        let shrunk = self.shrunk.as_ref();
        self.network_due = !hub.publish_snapshot(Snapshot::Network, |w| wire::network(w, shrunk));
    }

    /// Carries out one command from the page; every index is checked first, and a command
    /// that cannot be carried out is answered with a `status` message.
    fn command(
        &mut self,
        command: Command,
        hub: &Hub,
        shared: &Arc<Shared>,
        model: &mut Sequential,
        params: &mut ParamStore,
        config: &mut ExploreConfig<'_>,
    ) -> Next {
        match command {
            Command::SetMask { linear, neuron, on } => {
                match Training::mask_target(model, linear, neuron) {
                    Ok(layer) => {
                        model.set_neuron(layer, neuron, on);
                        self.capture_now = true;
                    }
                    Err(message) => say(hub, &message),
                }
            }
            Command::SelectProbe(index) => {
                let rows = self.probe_set.map_or(0, Dataset::len);
                if index < rows {
                    self.row = index;
                    self.probe_failed = false;
                    self.capture_now = true;
                } else {
                    say(
                        hub,
                        &format!(
                            "there is no probe row {index}: there {} {}",
                            if rows == 1 { "is" } else { "are" },
                            plural(rows, "probe row", "probe rows")
                        ),
                    );
                }
            }
            Command::SelectNeuron { layer, neuron } => match self.widths.get(layer) {
                Some(&width) if neuron < width => {
                    self.neuron = Some((layer, neuron));
                    self.capture_now = true;
                }
                Some(&width) => say(
                    hub,
                    &format!("there is no neuron {neuron} in layer {layer}: it has width {width}"),
                ),
                None => say(
                    hub,
                    &format!(
                        "there is no layer {layer}: the model has {}",
                        plural(self.widths.len(), "layer", "layers")
                    ),
                ),
            },
            Command::SetLearningRate(_) | Command::Pause | Command::Resume => say(
                hub,
                "pausing, resuming and the learning rate apply during training; explore mode \
                 works on the trained network",
            ),
            Command::Cancel => say(hub, "nothing is running to cancel"),
            Command::Finish => return Next::Finish,
            Command::Ablate { linear } => {
                return self.rank(linear, shared, model, params);
            }
            Command::Shrink => return self.shrink(hub, shared, model, params, config),
        }
        Next::Stay
    }

    /// Ranks the neurons of the `linear`-th `Linear` (counting only `Linear`s): one
    /// `ablation` event per neuron as it is measured, all with one new `sweep` number, then
    /// a `status` message with the cheapest neurons. A `Cancel` stops it after the neuron in
    /// progress.
    fn rank(
        &mut self,
        linear: usize,
        shared: &Shared,
        model: &Sequential,
        params: &ParamStore,
    ) -> Next {
        let hub = &shared.hub;
        let Some(data) = self.eval.as_deref() else {
            say(
                hub,
                "cannot rank neurons: there are no rows to measure on; give the inspector \
                 probe rows (InspectorConfig::probe) or ExploreConfig::train",
            );
            return Next::Stay;
        };
        let Some((layer, _)) = model.linears().nth(linear) else {
            say(
                hub,
                &format!(
                    "cannot rank the neurons of Linear layer {linear}: the model has {}",
                    plural(model.linears().count(), "Linear layer", "Linear layers")
                ),
            );
            return Next::Stay;
        };
        let sweep = NEXT_RUN.fetch_add(1, Ordering::Relaxed);
        let mut measured = 0usize;
        let (result, stopped, finish) = watched(
            shared,
            "a neuron ranking",
            "cancelling: the ranking stops after the neuron in progress",
            |watch| {
                let result = cavitation_nn::shrink::ablate_controlled(
                    model,
                    params,
                    data,
                    &[layer],
                    self.loss,
                    RANK_BATCH,
                    &mut |a: &Ablation| {
                        hub.publish_event(Event::Ablation, |w| wire::ablation(w, a, sweep), true);
                        measured += 1;
                        if watch.stop.load(Ordering::Acquire)
                            || hub.stopping().load(Ordering::Acquire)
                        {
                            Flow::Stop
                        } else {
                            Flow::Continue
                        }
                    },
                );
                let stopped = watch.stop.load(Ordering::Acquire);
                (result, stopped, watch.finish.load(Ordering::Acquire))
            },
        );
        match result {
            Ok(ranked) => {
                let mut message = if stopped {
                    format!(
                        "ranking cancelled after {} of Linear layer {linear} (layer {layer})",
                        plural(measured, "neuron", "neurons")
                    )
                } else {
                    format!(
                        "ranked the {} of Linear layer {linear} (layer {layer}) by what \
                         switching each off costs",
                        plural(ranked.len(), "neuron", "neurons")
                    )
                };
                if self.eval_is_training {
                    message.push_str(&format!(
                        ", measured on the first {}, which the network was trained on",
                        plural(data.len(), "training row", "training rows")
                    ));
                }
                if !ranked.is_empty() {
                    message.push_str("; cheapest first:");
                    for (k, a) in ranked.iter().rev().take(RANK_NAMED).enumerate() {
                        let sep = if k == 0 { " " } else { ", " };
                        message.push_str(&format!("{sep}neuron {} ({:.2})", a.neuron, a.drop));
                    }
                }
                say(hub, &message);
            }
            Err(e) => say(
                hub,
                &format!("cannot rank the neurons of Linear layer {linear}: {e}"),
            ),
        }
        if finish { Next::Finish } else { Next::Stay }
    }

    /// Runs the shrink tool with a fresh configuration from the factory: one `shrink`
    /// event per level, `Cancel` ends the run after the level in progress, and the model
    /// and its parameters are left at the kept level. Then a `status` message says what was
    /// kept, with the tool's warnings, and the probe is rebuilt and published.
    fn shrink(
        &mut self,
        hub: &Hub,
        shared: &Arc<Shared>,
        model: &mut Sequential,
        params: &mut ParamStore,
        config: &mut ExploreConfig<'_>,
    ) -> Next {
        let (train, factory) = match (config.train.as_ref(), config.shrink.as_mut()) {
            (Some(train), Some(factory)) => (train, factory),
            (None, None) => {
                say(
                    hub,
                    "cannot shrink: the shrink lab needs the training rows to retrain on \
                     (ExploreConfig::train) and a shrink configuration (ExploreConfig::shrink), \
                     and explore mode was started with neither",
                );
                return Next::Stay;
            }
            (None, Some(_)) => {
                say(
                    hub,
                    "cannot shrink: the shrink lab needs the training rows to retrain on, and \
                     ExploreConfig::train is None",
                );
                return Next::Stay;
            }
            (Some(_), None) => {
                say(
                    hub,
                    "cannot shrink: the shrink lab needs a shrink configuration, and \
                     ExploreConfig::shrink is None",
                );
                return Next::Stay;
            }
        };
        let mut shrink_config = factory();
        let run = NEXT_RUN.fetch_add(1, Ordering::Relaxed);
        say(
            hub,
            "shrinking: each level switches more neurons off, retrains and measures the \
             network; Cancel takes effect when the level in progress ends, and keeps the best \
             level so far",
        );
        let levels = Arc::new(AtomicU64::new(0));
        let (result, stopped, finish) = watched(
            shared,
            "a shrink run",
            "cancelling: the shrink run stops when the level in progress ends, its retraining \
             included, and keeps the best level so far",
            |watch| {
                let (shared, watch_step, levels) =
                    (Arc::clone(shared), Arc::clone(watch), Arc::clone(&levels));
                shrink_config.on_step = Some(Box::new(move |step: &ShrinkStep| {
                    let hub = &shared.hub;
                    hub.publish_event(Event::Shrink, |w| wire::shrink(w, step, run), true);
                    levels.fetch_add(1, Ordering::Relaxed);
                    if watch_step.stop.load(Ordering::Acquire)
                        || hub.stopping().load(Ordering::Acquire)
                    {
                        Flow::Stop
                    } else {
                        Flow::Continue
                    }
                }));
                let result = cavitation_nn::shrink::shrink(model, params, train, shrink_config);
                let stopped = watch.stop.load(Ordering::Acquire);
                (result, stopped, watch.finish.load(Ordering::Acquire))
            },
        );
        // The run may have replaced the parameter store, which the probe's gradients belong
        // to.
        self.probe = Probe::new();
        self.capture_now = true;
        let levels = usize::try_from(levels.load(Ordering::Relaxed)).unwrap_or(usize::MAX);
        let message = match result {
            Ok(mut report) => {
                let message = summary(&report, stopped);
                if let Some(i) = report.chosen {
                    let mut step = report.steps.swap_remove(i);
                    step.checkpoint = None;
                    self.shrunk = Some(step);
                    self.network_due = true;
                }
                message
            }
            Err(e) if levels == 0 => format!("cannot shrink: {e}"),
            Err(e) => format!(
                "shrink stopped after {}: {e}; the network is as it was before the run",
                plural(levels, "level", "levels")
            ),
        };
        say(hub, &message);
        if finish { Next::Finish } else { Next::Stay }
    }
}

/// The `status` message at the end of a shrink run: what was kept, and the tool's warnings
/// in its own words.
fn summary(report: &ShrinkReport, cancelled: bool) -> String {
    let verb = if cancelled {
        "shrink cancelled"
    } else {
        "shrink finished"
    };
    let n = plural(report.steps.len(), "level", "levels");
    let mut message = match report.chosen.map(|i| &report.steps[i]) {
        Some(step) => {
            // One count per shrunk layer, in order: "58" or "40 + 12".
            let kept = step
                .kept
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(" + ");
            let measure = match (step.validation.accuracy, step.validation.r2) {
                (Some(a), _) => format!("validation accuracy {:.2}%", a * 100.0),
                (None, Some(r2)) => format!("validation R² {r2:.4}"),
                (None, None) => format!("validation loss {:.4}", step.validation.loss),
            };
            format!(
                "{verb} after {n}: kept the level that removes {:.0}% of each layer, \
                 with {} neurons on and {} parameters, {measure}",
                f64::from(step.removed_bp) / 100.0,
                kept,
                step.parameters
            )
        }
        None => {
            format!("{verb} after {n}: no level met the stop rule, so the network is as it was")
        }
    };
    for warning in &report.warnings {
        message.push_str("; warning: ");
        message.push_str(&warning.to_string());
    }
    message
}

impl Inspector {
    /// Serves a trained network to the page in explore mode, until the page sends `Finish`
    /// or no client has been connected for the configuration's `idle_resume`.
    ///
    /// The page can then:
    ///
    /// - switch a neuron of a hidden `Linear` on or off, or choose the probe row and the
    ///   neuron to show: the probe and the neuron are measured again and published at once;
    /// - rank a hidden `Linear`'s neurons (`Ablate`): each neuron is switched off in turn
    ///   and the network measured on the first `eval_rows` probe rows, with one `ablation`
    ///   event per neuron as it is measured and then a `status` message naming the
    ///   cheapest; the model is not changed;
    /// - shrink the network (`Shrink`): [`shrink`](cavitation_nn::shrink::shrink) runs with
    ///   a fresh configuration from [`ExploreConfig::shrink`] on
    ///   [`ExploreConfig::train`], with one `shrink` event per level, and leaves `model`
    ///   and `params` at the level it keeps, which the page's view of switched-off neurons
    ///   follows; it can run again on a later click, on the network the last run left.
    ///
    /// While a ranking or a shrink run is going, a second thread answers the page's
    /// commands: `Cancel` is acknowledged at once with a `status` and ends the run after the
    /// neuron or the level in progress (a shrink level includes its retraining, so with
    /// large data that can take a while; a cancelled shrink run keeps the best level so
    /// far), `Finish` does that and then ends explore mode, and any other command is
    /// answered with a `status` message saying a run is in progress. After a shrink run
    /// keeps a level, the `network` snapshot gives its size. A command that cannot be
    /// carried out (an index out of range, a shrink without training rows or factory, a
    /// network the shrink tool refuses) is answered with a `status` message in the
    /// library's words, never a panic. Masks switched from the page stay on `model`
    /// afterwards. A ranking measured on training rows (no probe set configured) says so.
    ///
    /// The probe rows are the configuration's [`probe`](InspectorConfig::probe) set, or
    /// else the first `eval_rows` rows of [`ExploreConfig::train`], or else those of the
    /// last [`fit`](crate::fit) under this inspector. The page gets the `model` event when
    /// this network is not the one the last `fit` showed, then a `status` in state
    /// `explore`, and a `status` in state `finished` when explore mode ends. Everything
    /// runs on the calling thread.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidData`] if the configuration's probe set is not a matrix of
    /// rows of the model's input width (and the training rows' width) with one target per
    /// row of the kind `loss` takes.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    /// use cavitation_core::{ParamStore, Rng, Tensor};
    /// use cavitation_inspect::{ExploreConfig, Inspector, InspectorConfig};
    /// use cavitation_nn::{Dataset, Linear, Relu, Sequential, Targets};
    /// let mut inspector = Inspector::start(InspectorConfig {
    ///     address: "127.0.0.1:0".parse().unwrap(),
    ///     idle_resume: Duration::from_millis(50),
    ///     ..InspectorConfig::default()
    /// })?;
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
    /// println!("open {}", inspector.url());
    /// let config = ExploreConfig { train: Some(data), ..ExploreConfig::default() };
    /// // With no page connected it returns after `idle_resume`.
    /// inspector.explore(&mut net, &mut params, config)?;
    /// # Ok::<(), cavitation_core::Error>(())
    /// ```
    pub fn explore(
        &mut self,
        model: &mut Sequential,
        params: &mut ParamStore,
        mut config: ExploreConfig<'_>,
    ) -> Result<(), Error> {
        let Inspector {
            config: settings,
            shared,
            training,
            ..
        } = self;
        let settings: &InspectorConfig = settings;
        let width = input_width(model);
        let train = config.train.as_ref().filter(|t| trainable(t));
        if let Some(probe) = &settings.probe {
            check_probe(probe, train, config.loss)?;
            if let Some(width) = width
                && probe.features() != width
            {
                return Err(Error::InvalidData(format!(
                    "inspector probe set: rows of {} features, but the model takes {width}",
                    probe.features()
                )));
            }
        }
        let fits = |d: &&Dataset| width.is_none_or(|w| d.features() == w);
        let own = train.map(|t| t.subset(0..settings.eval_rows.max(1).min(t.len())));
        let probe_set = settings
            .probe
            .as_ref()
            .or(own.as_ref().filter(fits))
            .or(training.own_probe.as_ref().filter(fits));
        let eval = probe_set.and_then(|p| {
            let rows = settings.eval_rows.min(p.len());
            match rows {
                0 => None,
                r if r < p.len() => Some(Cow::Owned(p.subset(0..r))),
                _ => Some(Cow::Borrowed(p)),
            }
        });
        let input = probe_set.map(Dataset::features).or(width).unwrap_or(0);
        let mut w = input;
        let widths: Vec<usize> = (0..model.len())
            .map(|i| {
                w = model.layer(i).output_width(w);
                w
            })
            .collect();
        let hub = &shared.hub;
        if training.widths != widths {
            let probe_rows = probe_set.map_or(0, Dataset::len);
            let info = ModelInfo {
                title: &settings.title,
                input_width: input,
                input_image: settings.input_image,
                class_names: settings.class_names.as_deref(),
                probe_rows,
                probe_is_training: settings.probe.is_none(),
                eval_rows: settings.eval_rows.min(probe_rows),
                parameters: wire::parameter_count(model, params),
                flops_per_sample: wire::flops_per_sample(model),
            };
            hub.publish_event(Event::Model, |w| wire::model(w, model, &info), true);
        }
        let first = model
            .linears()
            .next()
            .map(|(l, _)| (model.neuron_layer(l), 0));
        let mut explore = Explore {
            loss: config.loss,
            probe_set,
            eval,
            eval_is_training: settings.probe.is_none(),
            widths,
            probe: Probe::new(),
            row: 0,
            neuron: first,
            step: training.step,
            capture_now: true,
            probe_failed: false,
            shrunk: None,
            network_due: true,
        };
        say(
            hub,
            "explore mode: switch neurons, rank a layer's neurons and shrink the network from \
             the page; Finish ends it",
        );
        let mut idle_since: Option<Instant> = None;
        loop {
            if hub.stopping().load(Ordering::Acquire) {
                return Ok(());
            }
            while let Some(command) = shared.commands.try_pop() {
                if explore.command(command, hub, shared, model, params, &mut config) == Next::Finish
                {
                    hub.publish_event(
                        Event::Status,
                        |w| {
                            wire::status(
                                w,
                                RunState::Finished,
                                "explore mode finished from the page",
                            )
                        },
                        true,
                    );
                    return Ok(());
                }
            }
            let now = Instant::now();
            if hub.clients() == 0 {
                let since = *idle_since.get_or_insert(now);
                if now.duration_since(since) >= settings.idle_resume {
                    let message = format!(
                        "explore mode ended: no page was connected for {:.1} s",
                        settings.idle_resume.as_secs_f64()
                    );
                    hub.publish_event(
                        Event::Status,
                        |w| wire::status(w, RunState::Finished, &message),
                        true,
                    );
                    return Ok(());
                }
            } else {
                idle_since = None;
                if explore.capture_now {
                    explore.capture(hub, model, params);
                }
            }
            if explore.network_due {
                explore.publish_network(hub);
            }
            thread::sleep(POLL);
        }
    }
}
