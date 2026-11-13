//! The JSON of every event the inspector sends, field by field, and the writers that make
//! it.
//!
//! Each event arrives on the stream as `event: <name>` and one `data:` line holding a JSON
//! object. Numbers that are NaN or infinite are written as `null`, so a field documented
//! as a number may also be `null` when the network produced such a value. Arrays of layer
//! values are in neuron order.
//!
//! | Event | Kind | When |
//! |---|---|---|
//! | `model` | event | once per connection: the network's description |
//! | `stats` | snapshot | on each sampled step, and when the run pauses |
//! | `epoch` | event | at the end of each epoch |
//! | `probe` | snapshot | on each sampled step: the probed row through every layer |
//! | `neuron` | snapshot | with each `probe`: the chosen neuron |
//! | `profile` | snapshot | on the step after each sampled step: where its time went |
//! | `status` | event | when the run's state changes, or a command cannot be carried out |
//! | `ablation` | event | in explore mode, one per neuron switched off |
//! | `shrink` | event | in explore mode, one per level of the shrink tool |
//!
//! Snapshots keep only their latest value: a client that falls behind skips to the newest.
//! Events are delivered in order, none skipped.
//!
//! # `model`
//!
//! - `title` (string): the page's title.
//! - `input_width` (integer): the width of an input row.
//! - `input_image` (`[height, width]` or `null`): the shape to draw an input row as.
//! - `class_names` (array of strings or `null`): the output classes' names.
//! - `probe_rows` (integer): how many rows the page can probe.
//! - `probe_is_training` (boolean): whether the probe rows are training rows (no separate
//!   probe set was given), so the page labels them as such.
//! - `eval_rows` (integer): how many of the probe rows the held-out evaluation uses.
//! - `parameters` (integer): the number of parameter values.
//! - `flops_per_sample` (integer): the floating-point operations one training sample takes
//!   in the `Linear` layers: for each `Linear` of `in` inputs and `out` outputs,
//!   `2·in·out` forward, `2·in·out` for the weight gradient and `2·in·out` for the input
//!   gradient, except the first `Linear`'s input gradient, which training does not need.
//! - `layers` (array), one object per layer in order:
//!   - `index` (integer): the layer's index in the model.
//!   - `kind` (string): the layer's name, such as `linear`, `relu` or `softmax`.
//!   - `width` (integer): the width of its output.
//!   - `linear` (boolean): whether it is a `Linear`.
//!   - `inputs` (integer or `null`): a `Linear`'s input width.
//!   - `activation` (string or `null`): the element-wise activation it applies.
//!   - `masked` (array of integers): the neurons its mask switches off when the event was
//!     made; the `probe` snapshot carries the current list.
//!
//! # `stats`
//!
//! - `step` (integer): the steps taken so far.
//! - `epoch` (integer): the epoch in progress, counting from 1.
//! - `epochs` (integer): the epochs the run will train.
//! - `samples_per_second` (number): the training throughput since the previous `stats`:
//!   rows trained divided by the time the training steps took (gathering each batch, the
//!   forward and backward passes, the health checks and the update). The inspector's own
//!   time between steps, and any pause, are left out, so this is the speed training runs
//!   at, not the wall-clock rate.
//! - `gflops` (number): `samples_per_second · flops_per_sample / 10⁹`, the work done in the
//!   `Linear` layers, in billions of floating-point operations per second.
//! - `loss` (number): the last step's batch loss.
//! - `learning_rate` (number): the rate the next step trains at.
//! - `paused` (boolean): whether the run is paused.
//!
//! # `epoch`
//!
//! - `epoch` (integer), `epochs` (integer): the epoch that ended, and the run's total.
//! - `mean_loss` (number): the mean training loss over the epoch's rows.
//! - `rows` (integer): the rows the epoch trained on.
//! - `seconds` (number): the epoch's wall-clock time, pauses included.
//! - `evaluation` (object or `null`): the held-out evaluation on the first `eval_rows` probe
//!   rows, or `null` when it was skipped (no client was connected, or the time since the
//!   last evaluation was under 100 times that evaluation's own cost; the last epoch always
//!   has one while a client is connected):
//!   - `rows` (integer): the rows measured.
//!   - `loss` (number): their mean loss, as training defines it.
//!   - `accuracy` (number or `null`): the share classified right, for class targets.
//!   - `r2` (number or `null`): the coefficient of determination, for value targets.
//!   - `misclassified` (array of integers): up to 100 probe-row indices classified wrong,
//!     in order, for the page's "show a mistake".
//!   - `seconds` (number): the evaluation's own time.
//!
//! # `probe`
//!
//! - `step` (integer): the step after which the row was probed.
//! - `row` (integer): the probe row's index.
//! - `label` (integer or `null`): its class label, for class targets.
//! - `target` (array of numbers or `null`): its value targets, for value targets.
//! - `predicted` (integer or `null`): the first largest output, for class targets.
//! - `loss` (number): the row's loss.
//! - `input` (array of numbers): the row's input.
//! - `layers` (array), one object per layer in order:
//!   - `index` (integer): the layer's index.
//!   - `a` (array of numbers): its output (after its mask, if it has one).
//!   - `z` (array of numbers or `null`): its input, for an element-wise layer.
//!   - `df` (array of numbers or `null`): its local derivative f′(z), when it gives one.
//!   - `grad_a` (array of numbers or `null`): `∂L/∂a`.
//!   - `grad_z` (array of numbers or `null`): `∂L/∂z`, for an element-wise layer.
//!   - `masked` (array of integers): the neurons its mask switches off now.
//!
//! # `neuron`
//!
//! - `step` (integer), `row` (integer): as in `probe`.
//! - `layer` (integer), `neuron` (integer): the neuron chosen.
//! - `linear` (integer or `null`): the layer index of the `Linear` it comes from.
//! - `on` (boolean): whether it is switched on.
//! - `a`, `z`, `df`, `grad_a`, `grad_z` (number or `null`): its values in the chosen layer,
//!   as in `probe`.
//! - `pre_activation` (number or `null`): `bias + Σ weight·x`, before any mask.
//! - `bias` (number or `null`).
//! - `weights` (array of numbers): its incoming weights, one per input of its `Linear`.
//! - `outgoing_norm` (number or `null`): the Euclidean norm of its weights in the next
//!   `Linear`.
//! - `top_inputs` (array), up to 10, largest `|weight·x|` first, each with `input`
//!   (integer), `x`, `weight` and `product` (numbers).
//!
//! # `profile`
//!
//! - `step` (integer): the profiled step.
//! - `layers` (array), one per layer: `index` (integer), `kind` (string), `forward_us` and
//!   `backward_us` (numbers, microseconds).
//! - `loss_us`, `backward_loss_us`, `optimizer_us`, `total_us` (numbers, microseconds):
//!   recording the loss, its backward pass, the optimiser's update and the whole step,
//!   as [`StepProfile`] defines them.
//!
//! # `status`
//!
//! - `state` (string): `training`, `paused`, `explore` or `finished`.
//! - `message` (string): what happened, in plain words.
//!
//! # `ablation`
//!
//! - `linear` (integer), `neuron` (integer): the neuron switched off.
//! - `drop` (number): what switching it off costs, as [`Ablation::drop`] defines it.
//! - `loss` (number), `accuracy` (number or `null`), `r2` (number or `null`): the
//!   network's measures with it off.
//!
//! # `shrink`
//!
//! - `removed_bp` (integer): the share of each layer's width removed, in basis points.
//! - `kept` (array of integers): the neurons left on in each shrunk `Linear`.
//! - `parameters` (integer): the parameters the network has at this level.
//! - `met` (boolean): whether the level meets the stop rule.
//! - `loss` (number), `accuracy` (number or `null`), `r2` (number or `null`): the
//!   retrained network's measures on the validation set.

use cavitation_core::ParamStore;
use cavitation_nn::shrink::{Ablation, ShrinkStep};
use cavitation_nn::{EpochReport, Evaluation, Sequential, StepProfile};

use crate::json::JsonWriter;
use crate::probe::{NeuronValues, Probe};

/// The state a run is in, as the `status` event names it.
///
/// # Examples
///
/// ```
/// assert_eq!(cavitation_inspect::wire::RunState::Paused.name(), "paused");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunState {
    /// Training is running.
    Training,
    /// Training is paused from the page.
    Paused,
    /// A trained network is being explored.
    Explore,
    /// The run has ended.
    Finished,
}

impl RunState {
    /// The name the `status` event carries.
    ///
    /// ```
    /// assert_eq!(cavitation_inspect::wire::RunState::Finished.name(), "finished");
    /// ```
    pub fn name(self) -> &'static str {
        match self {
            RunState::Training => "training",
            RunState::Paused => "paused",
            RunState::Explore => "explore",
            RunState::Finished => "finished",
        }
    }
}

/// What the `model` event says beside the layers.
///
/// # Examples
///
/// ```
/// use cavitation_inspect::wire::ModelInfo;
/// let info = ModelInfo {
///     title: "digits",
///     input_width: 64,
///     input_image: Some((8, 8)),
///     class_names: None,
///     probe_rows: 360,
///     probe_is_training: false,
///     eval_rows: 360,
///     parameters: 50_890,
///     flops_per_sample: 250_000,
/// };
/// assert_eq!(info.input_image, Some((8, 8)));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ModelInfo<'a> {
    /// The page's title.
    pub title: &'a str,
    /// The width of an input row.
    pub input_width: usize,
    /// `(height, width)` when an input row is an image.
    pub input_image: Option<(usize, usize)>,
    /// The output classes' names.
    pub class_names: Option<&'a [String]>,
    /// How many rows the page can probe.
    pub probe_rows: usize,
    /// Whether the probe rows are training rows.
    pub probe_is_training: bool,
    /// How many probe rows the evaluation uses.
    pub eval_rows: usize,
    /// The number of parameter values.
    pub parameters: usize,
    /// The work of one training sample, as [`flops_per_sample`] counts it.
    pub flops_per_sample: u64,
}

/// What the `stats` snapshot says.
///
/// # Examples
///
/// ```
/// use cavitation_inspect::wire::Stats;
/// let s = Stats { step: 10, epoch: 1, epochs: 5, samples_per_second: 1e4, gflops: 2.5,
///                 loss: 0.7, learning_rate: 1e-3, paused: false };
/// assert_eq!(s.step, 10);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stats {
    /// The steps taken so far.
    pub step: u64,
    /// The epoch in progress, counting from 1.
    pub epoch: usize,
    /// The epochs the run will train.
    pub epochs: usize,
    /// Rows trained per second of training-step time since the previous snapshot.
    pub samples_per_second: f64,
    /// Billions of floating-point operations per second in the `Linear` layers.
    pub gflops: f64,
    /// The last step's batch loss.
    pub loss: f32,
    /// The rate the next step trains at.
    pub learning_rate: f64,
    /// Whether the run is paused.
    pub paused: bool,
}

/// The floating-point operations one training sample takes in `model`'s `Linear` layers:
/// for each `Linear` of `in` inputs and `out` outputs, `2·in·out` forward, `2·in·out` for
/// the weight gradient and `2·in·out` for the input gradient, except the first `Linear`'s
/// input gradient (nothing before it needs one).
///
/// ```
/// use cavitation_core::{ParamStore, Rng};
/// use cavitation_nn::{Linear, Relu, Sequential};
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 64, 32, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 32, 10, &mut rng));
/// assert_eq!(cavitation_inspect::wire::flops_per_sample(&net), 4 * 64 * 32 + 6 * 32 * 10);
/// ```
pub fn flops_per_sample(model: &Sequential) -> u64 {
    model
        .linears()
        .enumerate()
        .map(|(k, (_, linear))| {
            let product = 2 * (linear.inputs() as u64) * (linear.outputs() as u64);
            if k == 0 { 2 * product } else { 3 * product }
        })
        .sum()
}

/// The number of parameter values of `model` in `params`.
///
/// ```
/// use cavitation_core::{ParamStore, Rng};
/// use cavitation_nn::{Linear, Sequential};
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)));
/// assert_eq!(cavitation_inspect::wire::parameter_count(&net, &params), 8);
/// ```
pub fn parameter_count(model: &Sequential, params: &ParamStore) -> usize {
    (0..model.len())
        .flat_map(|i| model.layer(i).parameters())
        .map(|(_, id)| params.value(id).data().len())
        .sum()
}

/// Writes the indices of the neurons layer `i`'s mask switches off, as an array.
fn masked(w: &mut JsonWriter, model: &Sequential, i: usize) {
    w.begin_array();
    if let Some(mask) = model.mask(i) {
        for j in mask.off() {
            w.uint(j as u64);
        }
    }
    w.end_array();
}

/// Writes `values` as an array, or `null`.
fn values_or_null(w: &mut JsonWriter, values: Option<&[f32]>) {
    match values {
        Some(v) => w.f32s(v),
        None => w.null(),
    };
}

/// Writes an optional number.
fn number_or_null(w: &mut JsonWriter, x: Option<f64>) {
    match x {
        Some(x) => w.number(x),
        None => w.null(),
    };
}

/// Writes an optional integer.
fn uint_or_null(w: &mut JsonWriter, x: Option<usize>) {
    match x {
        Some(x) => w.uint(x as u64),
        None => w.null(),
    };
}

/// Writes the `model` event for `model`, whose input rows are `info.input_width` wide.
///
/// ```
/// use cavitation_core::{ParamStore, Rng};
/// use cavitation_inspect::json::JsonWriter;
/// use cavitation_inspect::wire::{self, ModelInfo};
/// use cavitation_nn::{Linear, Relu, Sequential};
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0))).push(Relu);
/// let info = ModelInfo {
///     title: "t", input_width: 2, input_image: None, class_names: None, probe_rows: 4,
///     probe_is_training: true, eval_rows: 4, parameters: 9, flops_per_sample: 24,
/// };
/// let mut w = JsonWriter::new();
/// wire::model(&mut w, &net, &info);
/// assert!(w.as_str().contains(r#""kind":"relu","width":3,"linear":false"#));
/// ```
pub fn model(w: &mut JsonWriter, model: &Sequential, info: &ModelInfo<'_>) {
    w.begin_object()
        .key("title")
        .string(info.title)
        .key("input_width")
        .uint(info.input_width as u64)
        .key("input_image");
    match info.input_image {
        Some((h, wd)) => {
            w.begin_array().uint(h as u64).uint(wd as u64).end_array();
        }
        None => {
            w.null();
        }
    }
    w.key("class_names");
    match info.class_names {
        Some(names) => {
            w.begin_array();
            for name in names {
                w.string(name);
            }
            w.end_array();
        }
        None => {
            w.null();
        }
    }
    w.key("probe_rows")
        .uint(info.probe_rows as u64)
        .key("probe_is_training")
        .bool(info.probe_is_training)
        .key("eval_rows")
        .uint(info.eval_rows as u64)
        .key("parameters")
        .uint(info.parameters as u64)
        .key("flops_per_sample")
        .uint(info.flops_per_sample)
        .key("layers")
        .begin_array();
    let mut width = info.input_width;
    for i in 0..model.len() {
        let layer = model.layer(i);
        width = layer.output_width(width);
        let linear = model.linear(i);
        w.begin_object()
            .key("index")
            .uint(i as u64)
            .key("kind")
            .string(layer.name())
            .key("width")
            .uint(width as u64)
            .key("linear")
            .bool(linear.is_some())
            .key("inputs");
        uint_or_null(w, linear.map(|l| l.inputs()));
        w.key("activation");
        match layer.activation() {
            Some(act) => w.string(act.name()),
            None => w.null(),
        };
        w.key("masked");
        masked(w, model, i);
        w.end_object();
    }
    w.end_array().end_object();
}

/// Writes the `stats` snapshot.
///
/// ```
/// use cavitation_inspect::json::JsonWriter;
/// use cavitation_inspect::wire::{self, Stats};
/// let s = Stats { step: 3, epoch: 1, epochs: 2, samples_per_second: f64::NAN, gflops: 0.0,
///                 loss: 0.5, learning_rate: 0.1, paused: true };
/// let mut w = JsonWriter::new();
/// wire::stats(&mut w, &s);
/// assert!(w.as_str().contains(r#""samples_per_second":null"#));
/// ```
pub fn stats(w: &mut JsonWriter, s: &Stats) {
    w.begin_object()
        .key("step")
        .uint(s.step)
        .key("epoch")
        .uint(s.epoch as u64)
        .key("epochs")
        .uint(s.epochs as u64)
        .key("samples_per_second")
        .number(s.samples_per_second)
        .key("gflops")
        .number(s.gflops)
        .key("loss")
        .f32(s.loss)
        .key("learning_rate")
        .number(s.learning_rate)
        .key("paused")
        .bool(s.paused)
        .end_object();
}

/// A held-out evaluation as the `epoch` event reports it.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_inspect::wire::EpochEvaluation;
/// use cavitation_nn::{Dataset, Linear, Loss, Sequential, Targets, evaluate};
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 2, 2, &mut Rng::seed(0)));
/// let data = Dataset { x: Tensor::zeros(&[2, 2]), targets: Targets::Classes(vec![0, 1]) };
/// let evaluation = evaluate(&net, &params, &data, Loss::CrossEntropy, 2).unwrap();
/// let e = EpochEvaluation { evaluation, rows: 2, misclassified: &[1], seconds: 0.001 };
/// assert_eq!(e.rows, 2);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct EpochEvaluation<'a> {
    /// The measures.
    pub evaluation: Evaluation,
    /// The rows measured.
    pub rows: usize,
    /// Probe-row indices classified wrong; the event carries the first 100.
    pub misclassified: &'a [usize],
    /// The evaluation's own time.
    pub seconds: f64,
}

/// Writes the `epoch` event for `report`, of a run of `epochs` epochs.
///
/// ```
/// use cavitation_inspect::json::JsonWriter;
/// use cavitation_inspect::wire;
/// use cavitation_nn::EpochReport;
/// let report = EpochReport { epoch: 1, mean_loss: 0.5, rows: 64, seconds: 0.01 };
/// let mut w = JsonWriter::new();
/// wire::epoch(&mut w, &report, 3, None);
/// assert!(w.as_str().ends_with(r#""evaluation":null}"#));
/// ```
pub fn epoch(
    w: &mut JsonWriter,
    report: &EpochReport,
    epochs: usize,
    evaluation: Option<&EpochEvaluation<'_>>,
) {
    w.begin_object()
        .key("epoch")
        .uint(report.epoch as u64)
        .key("epochs")
        .uint(epochs as u64)
        .key("mean_loss")
        .number(report.mean_loss)
        .key("rows")
        .uint(report.rows as u64)
        .key("seconds")
        .number(report.seconds)
        .key("evaluation");
    match evaluation {
        None => {
            w.null();
        }
        Some(e) => {
            w.begin_object()
                .key("rows")
                .uint(e.rows as u64)
                .key("loss")
                .number(e.evaluation.loss)
                .key("accuracy");
            number_or_null(w, e.evaluation.accuracy);
            w.key("r2");
            number_or_null(w, e.evaluation.r2);
            w.key("misclassified").begin_array();
            for &i in e.misclassified.iter().take(100) {
                w.uint(i as u64);
            }
            w.end_array().key("seconds").number(e.seconds).end_object();
        }
    }
    w.end_object();
}

/// Writes the `probe` snapshot from `probe`'s last run of `model`, after step `step`.
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_inspect::json::JsonWriter;
/// use cavitation_inspect::{probe::Probe, wire};
/// use cavitation_nn::{Dataset, Linear, Loss, Sequential, Targets};
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 2, 2, &mut Rng::seed(0)));
/// let data = Dataset { x: Tensor::zeros(&[1, 2]), targets: Targets::Classes(vec![1]) };
/// let mut probe = Probe::new();
/// probe.run(&net, &params, &data, 0, Loss::CrossEntropy)?;
/// let mut w = JsonWriter::new();
/// wire::probe(&mut w, 7, &probe, &net);
/// assert!(w.as_str().starts_with(r#"{"step":7,"row":0,"label":1,"target":null"#));
/// # Ok::<(), cavitation_core::Error>(())
/// ```
pub fn probe(w: &mut JsonWriter, step: u64, probe: &Probe, model: &Sequential) {
    w.begin_object()
        .key("step")
        .uint(step)
        .key("row")
        .uint(probe.row() as u64)
        .key("label");
    uint_or_null(w, probe.label());
    w.key("target");
    values_or_null(w, probe.label().is_none().then_some(probe.target()));
    w.key("predicted");
    uint_or_null(w, probe.predicted());
    w.key("loss")
        .f32(probe.loss())
        .key("input")
        .f32s(probe.input())
        .key("layers")
        .begin_array();
    for (i, layer) in probe.layers().iter().enumerate() {
        w.begin_object()
            .key("index")
            .uint(i as u64)
            .key("a")
            .f32s(layer.a())
            .key("z");
        values_or_null(w, layer.z());
        w.key("df");
        values_or_null(w, layer.df());
        w.key("grad_a");
        values_or_null(w, layer.grad_a());
        w.key("grad_z");
        values_or_null(w, layer.grad_z());
        w.key("masked");
        if i < model.len() {
            masked(w, model, i);
        } else {
            w.begin_array().end_array();
        }
        w.end_object();
    }
    w.end_array().end_object();
}

/// Writes the `neuron` snapshot for `neuron`, filled by [`Probe::neuron`] from `probe`'s
/// last run, after step `step`.
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_inspect::json::JsonWriter;
/// use cavitation_inspect::{probe::Probe, wire};
/// use cavitation_nn::{Dataset, Linear, Loss, Relu, Sequential, Targets};
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 2, 2, &mut Rng::seed(0))).push(Relu);
/// let data = Dataset { x: Tensor::zeros(&[1, 2]), targets: Targets::Classes(vec![1]) };
/// let mut probe = Probe::new();
/// probe.run(&net, &params, &data, 0, Loss::CrossEntropy)?;
/// probe.neuron(&net, &params, 1, 0)?;
/// let mut w = JsonWriter::new();
/// wire::neuron(&mut w, 1, &probe, probe.last_neuron());
/// assert!(w.as_str().contains(r#""layer":1,"neuron":0,"linear":0,"on":true"#));
/// # Ok::<(), cavitation_core::Error>(())
/// ```
pub fn neuron(w: &mut JsonWriter, step: u64, probe: &Probe, neuron: &NeuronValues) {
    let j = neuron.neuron();
    let values = probe.layers().get(neuron.layer());
    let at = |v: Option<&[f32]>| v.and_then(|v| v.get(j)).map(|&x| f64::from(x));
    w.begin_object()
        .key("step")
        .uint(step)
        .key("row")
        .uint(probe.row() as u64)
        .key("layer")
        .uint(neuron.layer() as u64)
        .key("neuron")
        .uint(j as u64)
        .key("linear");
    uint_or_null(w, neuron.linear());
    w.key("on").bool(neuron.on());
    for (key, value) in [
        ("a", values.map(|l| l.a())),
        ("z", values.and_then(|l| l.z())),
        ("df", values.and_then(|l| l.df())),
        ("grad_a", values.and_then(|l| l.grad_a())),
        ("grad_z", values.and_then(|l| l.grad_z())),
    ] {
        w.key(key);
        number_or_null(w, at(value));
    }
    w.key("pre_activation");
    number_or_null(w, neuron.pre_activation().map(f64::from));
    w.key("bias");
    number_or_null(w, neuron.bias().map(f64::from));
    w.key("weights").f32s(neuron.weights()).key("outgoing_norm");
    number_or_null(w, neuron.outgoing_norm().map(f64::from));
    w.key("top_inputs").begin_array();
    for c in neuron.top_inputs() {
        w.begin_object()
            .key("input")
            .uint(c.input as u64)
            .key("x")
            .f32(c.x)
            .key("weight")
            .f32(c.weight)
            .key("product")
            .f32(c.product)
            .end_object();
    }
    w.end_array().end_object();
}

/// Microseconds in `d`, as a number.
fn micros(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// Writes the `profile` snapshot of step `step` of `model`.
///
/// ```
/// use std::time::Duration;
/// use cavitation_inspect::json::JsonWriter;
/// use cavitation_inspect::wire;
/// use cavitation_nn::{Relu, Sequential, StepProfile};
/// let net = Sequential::new().push(Relu);
/// let profile = StepProfile {
///     forward: vec![Duration::from_micros(3)],
///     backward: vec![Duration::from_micros(5)],
///     total: Duration::from_micros(20),
///     ..StepProfile::default()
/// };
/// let mut w = JsonWriter::new();
/// wire::profile(&mut w, 4, &net, &profile);
/// assert!(w.as_str().contains(r#""forward_us":3.0,"backward_us":5.0"#));
/// ```
pub fn profile(w: &mut JsonWriter, step: u64, model: &Sequential, profile: &StepProfile) {
    w.begin_object()
        .key("step")
        .uint(step)
        .key("layers")
        .begin_array();
    for (i, (f, b)) in profile.forward.iter().zip(&profile.backward).enumerate() {
        w.begin_object().key("index").uint(i as u64).key("kind");
        if i < model.len() {
            w.string(model.layer(i).name());
        } else {
            w.null();
        }
        w.key("forward_us")
            .number(micros(*f))
            .key("backward_us")
            .number(micros(*b))
            .end_object();
    }
    w.end_array()
        .key("loss_us")
        .number(micros(profile.loss))
        .key("backward_loss_us")
        .number(micros(profile.backward_loss))
        .key("optimizer_us")
        .number(micros(profile.optimizer))
        .key("total_us")
        .number(micros(profile.total))
        .end_object();
}

/// Writes the `status` event.
///
/// ```
/// use cavitation_inspect::json::JsonWriter;
/// use cavitation_inspect::wire::{self, RunState};
/// let mut w = JsonWriter::new();
/// wire::status(&mut w, RunState::Training, "training started");
/// assert_eq!(w.as_str(), r#"{"state":"training","message":"training started"}"#);
/// ```
pub fn status(w: &mut JsonWriter, state: RunState, message: &str) {
    w.begin_object()
        .key("state")
        .string(state.name())
        .key("message")
        .string(message)
        .end_object();
}

/// Writes the measures of an evaluation: `loss`, `accuracy` and `r2`.
fn measures(w: &mut JsonWriter, e: &Evaluation) {
    w.key("loss").number(e.loss).key("accuracy");
    number_or_null(w, e.accuracy);
    w.key("r2");
    number_or_null(w, e.r2);
}

/// Writes the `ablation` event for one neuron's result.
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_inspect::json::JsonWriter;
/// use cavitation_inspect::wire;
/// use cavitation_nn::shrink::ablate;
/// use cavitation_nn::{Dataset, Linear, Loss, Relu, Sequential, Targets};
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 3, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 3, 2, &mut rng));
/// let data = Dataset { x: Tensor::zeros(&[4, 2]), targets: Targets::Classes(vec![0; 4]) };
/// let all = ablate(&net, &params, &data, &[0], Loss::CrossEntropy, 4).unwrap();
/// let mut w = JsonWriter::new();
/// wire::ablation(&mut w, &all[0]);
/// assert!(w.as_str().starts_with(r#"{"linear":0,"neuron":"#));
/// ```
pub fn ablation(w: &mut JsonWriter, a: &Ablation) {
    w.begin_object()
        .key("linear")
        .uint(a.linear as u64)
        .key("neuron")
        .uint(a.neuron as u64)
        .key("drop")
        .number(a.drop);
    measures(w, &a.evaluation);
    w.end_object();
}

/// Writes the `shrink` event for one level of the shrink tool.
///
/// ```
/// use cavitation_inspect::json::JsonWriter;
/// use cavitation_inspect::wire;
/// use cavitation_nn::shrink::ShrinkStep;
/// fn write(step: &ShrinkStep) -> String {
///     let mut w = JsonWriter::new();
///     wire::shrink(&mut w, step);
///     w.as_str().to_string()
/// }
/// # let _ = write;
/// ```
pub fn shrink(w: &mut JsonWriter, step: &ShrinkStep) {
    w.begin_object()
        .key("removed_bp")
        .uint(u64::from(step.removed_bp))
        .key("kept")
        .begin_array();
    for &k in &step.kept {
        w.uint(k as u64);
    }
    w.end_array()
        .key("parameters")
        .uint(step.parameters as u64)
        .key("met")
        .bool(step.met);
    measures(w, &step.validation);
    w.end_object();
}
