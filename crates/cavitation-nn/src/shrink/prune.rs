//! The shrink tool: [`shrink`] removes the neurons of a trained network's hidden `Linear`
//! layers and the output channels of its `Conv2d` layers a step at a time, retrains after
//! each step, and keeps the smallest network that still meets a criterion measured on a
//! validation set.

use std::borrow::Cow;
use std::fmt;

use cavitation_core::{Error, ParamStore, Rng};

use super::ablate::{check_hidden, points_lost};
use super::activity::position_means;
use super::compact::{foreign_parameters, has_image_layers, linear_after_run, unvouched_layer};
use super::compact_image::{foreign_image_parameters, unvouched_channel_layer};
use super::{activity, channel_activity, channels_on, compact, neurons_on};
use crate::train::check_dataset;
use crate::{
    Dataset, Evaluation, FitConfig, Flow, Health, Loss, LrSchedule, NeuronMask, OptimizerConfig,
    Sequential, Targets, evaluate, fit,
};

/// The rows run at a time when measuring activity and evaluating on the validation set.
const EVAL_BATCH: usize = 256;

/// The whole network in basis points: 10,000 bp is 100%.
const FULL_BP: u32 = 10_000;

/// How [`shrink`] orders the units of a layer for removal (the neurons of a `Linear`, the
/// output channels of a `Conv2d`): each unit gets a score, and the lowest scores go first
/// (ties by index, lowest first; a NaN score, whatever its sign bit, sorts above every
/// number, so it goes last). Every ranking is computed again at each step, on the network
/// as the previous step's retraining left it.
///
/// ```
/// use cavitation_nn::shrink::{RankContext, Ranking, ShrinkConfig};
/// // Score each unit by its mean |activation| times its index (an example only).
/// let custom = Ranking::Custom(Box::new(|ctx: &RankContext<'_>| {
///     ctx.on.iter().map(|&j| ctx.mean_abs[j] * j as f64).collect()
/// }));
/// let config = ShrinkConfig { ranking: custom, ..ShrinkConfig::default() };
/// assert!(matches!(config.ranking, Ranking::Custom(_)));
/// assert!(matches!(ShrinkConfig::default().ranking, Ranking::OutNorm));
/// ```
#[non_exhaustive]
pub enum Ranking<'s> {
    /// The Euclidean norm of the unit's outgoing weights. For a neuron, its column in the
    /// weight of the `Linear` that follows its layer's run; for a channel, its input
    /// channel's weights (every output channel and kernel tap) in the next `Conv2d`, or its
    /// block of columns in the `Linear` after a `Flatten`. A unit the next layer barely uses
    /// goes first.
    OutNorm,
    /// The unit's mean absolute output on the training rows ([`RankContext::mean_abs`]),
    /// over every position for a channel.
    Activity,
    /// A random order from this seed: the same seed removes the same units on every run.
    Random(u64),
    /// Your own scores: the function gets a [`RankContext`] for one layer and returns one
    /// score for each entry of [`RankContext::on`], in that order. It is called only for
    /// a layer that loses units at the step. It is `Send` and may borrow for `'s`.
    // The boxed callback is the plainest type for this variant; an alias would only hide it.
    #[allow(clippy::type_complexity)]
    Custom(Box<dyn FnMut(&RankContext<'_>) -> Vec<f64> + Send + 's>),
}

impl fmt::Debug for Ranking<'_> {
    /// Writes the variant; a custom function shows only as `Custom(FnMut)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ranking::OutNorm => f.write_str("OutNorm"),
            Ranking::Activity => f.write_str("Activity"),
            Ranking::Random(seed) => f.debug_tuple("Random").field(seed).finish(),
            Ranking::Custom(_) => f.write_str("Custom(FnMut)"),
        }
    }
}

/// What a [`Ranking::Custom`] function sees for one layer at one step: the network as it
/// is (after the previous step's retraining, before this step removes anything) and the
/// activity of the layer's units (neurons of a `Linear`, output channels of a `Conv2d`) on
/// the training rows, read where each unit's output is final, as
/// [`activity`](super::activity()) and [`channel_activity`](super::channel_activity())
/// measure it.
///
/// It is `#[non_exhaustive]`, so a later release can add a field without breaking your
/// code: read its fields in your ranking function.
///
/// ```
/// use cavitation_nn::shrink::{RankContext, Ranking};
/// // For a Linear: rank by firing share, and break ties by the outgoing weights' sum of
/// // squares.
/// fn by_firing(ctx: &RankContext<'_>) -> Vec<f64> {
///     let next = ctx.model.linear(ctx.model.neuron_layer(ctx.layer) + 1).unwrap();
///     let w = ctx.params.value(next.weight()).data();
///     let width = next.inputs();
///     ctx.on
///         .iter()
///         .map(|&j| {
///             let norm: f64 = w.iter().skip(j).step_by(width).map(|&v| f64::from(v * v)).sum();
///             ctx.firing[j] + 1e-6 * norm
///         })
///         .collect()
/// }
/// let ranking = Ranking::Custom(Box::new(by_firing));
/// assert!(matches!(ranking, Ranking::Custom(_)));
/// ```
#[non_exhaustive]
pub struct RankContext<'a> {
    /// The index of the layer whose units are ranked: a `Linear` (its neurons) or a
    /// `Conv2d` (its output channels).
    pub layer: usize,
    /// The units of that layer that are on, in increasing order: the ones to score.
    pub on: &'a [usize],
    /// The network, with its masks as they are.
    pub model: &'a Sequential,
    /// The network's parameters.
    pub params: &'a ParamStore,
    /// Each unit's mean output on the training rows (over every position, for a channel).
    /// One value per unit of the layer; the units that are off read 0.
    pub mean: &'a [f64],
    /// Each unit's mean absolute output.
    pub mean_abs: &'a [f64],
    /// The share of each unit's outputs above 0: for a ReLU network, how often it fires.
    pub firing: &'a [f64],
    /// Each unit's largest output.
    pub max: &'a [f64],
}

/// When a step of [`shrink`] is good enough, against the unpruned network's measures on
/// the validation set (the baseline).
///
/// ```
/// use cavitation_nn::shrink::Criterion;
/// // Lose at most half an accuracy point.
/// let strict = Criterion::AccuracyDrop { points: 0.5 };
/// // For value targets: a validation loss at most 5% above the baseline's.
/// let loose = Criterion::LossIncrease { relative: 0.05 };
/// assert_ne!(strict, loose);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Criterion {
    /// The validation accuracy is at least the baseline's minus `points` accuracy points.
    /// The drop is counted in whole rows, as [`Ablation::drop`](super::Ablation::drop)
    /// is, so one row lost in 100 is exactly 1.0 and meets `points: 1.0`. Class targets
    /// only.
    AccuracyDrop {
        /// The accuracy points the network may lose; 1.0 is one percentage point.
        points: f64,
    },
    /// The validation loss is at most the baseline's plus `relative` times its magnitude,
    /// `loss ≤ baseline + relative·|baseline|` (for a positive baseline,
    /// `baseline·(1 + relative)`). A NaN loss never meets it.
    LossIncrease {
        /// The relative rise allowed; 0.05 is 5%.
        relative: f64,
    },
}

impl Criterion {
    /// Whether `now` meets the criterion against `base`, both measured on `rows` rows.
    fn met(&self, base: &Evaluation, now: &Evaluation, rows: usize) -> bool {
        match *self {
            Criterion::AccuracyDrop { points } => match (base.accuracy, now.accuracy) {
                (Some(before), Some(after)) => points_lost(before, after, rows) <= points,
                _ => false,
            },
            Criterion::LossIncrease { relative } => {
                // The documented inequality itself; a zero baseline allows no rise, and the
                // product is left out so that an infinite `relative` does not give NaN.
                let limit = if base.loss == 0.0 {
                    base.loss
                } else {
                    base.loss + relative * base.loss.abs()
                };
                !now.loss.is_nan() && now.loss <= limit
            }
        }
    }
}

/// The rows [`shrink`] measures each step on. There is no default: the stop rule is only
/// as good as the rows it is measured on, and they must be rows the model has not been
/// trained on.
///
/// ```
/// use cavitation_core::Tensor;
/// use cavitation_nn::shrink::Validation;
/// use cavitation_nn::{Dataset, Targets};
/// let held_out = Dataset { x: Tensor::zeros(&[300, 4]), targets: Targets::Classes(vec![0; 300]) };
/// let given = Validation::Given(held_out);
/// // Or: keep the last 20% of the training rows for validation and retrain on the rest.
/// let carved = Validation::Carve { share: 0.2 };
/// assert_ne!(given, carved);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Validation {
    /// These rows, which the model must not have seen in training.
    Given(Dataset),
    /// The last `ceil(share · n)` of the `n` training rows, held out of retraining (which
    /// runs on the rest). The model must not have seen those rows: carve the same rows off
    /// before the first training, or the baseline and every step are measured on rows the
    /// network learned, and the stop rule cannot tell what pruning cost. `share` is above
    /// 0 and below 1, and both parts keep at least one row. The share is read as the
    /// decimal you wrote, so `0.07` of 100 rows holds out exactly 7.
    Carve {
        /// The share of the training rows to hold out, above 0 and below 1.
        share: f64,
    },
}

/// How [`shrink`] retrains the network after each step: a [`fit`] run with a fresh
/// optimiser at a constant learning rate, and health checks that stop on a NaN or infinite
/// loss or gradient (the error is returned) but print nothing.
///
/// The default is 3 epochs in batches of 32, Adam at 1e-3 without weight decay,
/// cross-entropy, and shuffle seed 7.
///
/// ```
/// use cavitation_nn::Loss;
/// use cavitation_nn::shrink::RetrainConfig;
/// let mse = RetrainConfig { loss: Loss::Mse, ..RetrainConfig::default() };
/// assert_eq!((mse.epochs, mse.batch, mse.shuffle_seed), (3, 32, Some(7)));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetrainConfig {
    /// The epochs of retraining after each step; 0 measures each step without retraining.
    pub epochs: usize,
    /// The rows per training step, at least 1.
    pub batch: usize,
    /// The optimiser, built afresh for each step.
    pub optimizer: OptimizerConfig,
    /// The loss to train with, and to measure on the validation set.
    pub loss: Loss,
    /// The seed that shuffles the rows each epoch; `None` keeps the data's order.
    pub shuffle_seed: Option<u64>,
}

impl Default for RetrainConfig {
    /// 3 epochs, batch 32, Adam at 1e-3, cross-entropy, shuffle seed 7.
    fn default() -> RetrainConfig {
        RetrainConfig {
            epochs: 3,
            batch: 32,
            optimizer: OptimizerConfig::Adam {
                lr: 1e-3,
                weight_decay: 0.0,
            },
            loss: Loss::CrossEntropy,
            shuffle_seed: Some(7),
        }
    }
}

/// How [`shrink`] runs. Taken by value, because it can hold callbacks. Levels are in whole
/// basis points of each layer's width (its neurons, or its output channels): 1 bp is
/// 0.01%, 10,000 bp the whole layer.
///
/// The callbacks ([`on_step`](Self::on_step) and a [`Ranking::Custom`] function) are
/// `Send` and may borrow for the lifetime `'s`, as [`FitConfig`](crate::FitConfig)'s are: an
/// `on_step` can push each step onto a local `Vec` that the caller reads after the run, and
/// a configuration can move to another thread (with owned callbacks, or borrowing ones
/// inside [`std::thread::scope`]).
///
/// The default shrinks every hidden `Linear` and every `Conv2d` whose channels reach a
/// `Conv2d` or a `Linear`, removes first the units whose output is 0 on every training row
/// (for a ReLU, those that never fire) and then the lowest out-norms, in steps of 1,000 bp up to 7,000 and of 500 bp from there to
/// 9,500, retrains as [`RetrainConfig::default`], keeps the latest step that loses at most
/// 1 accuracy point (class targets), and stops after 2 misses in a row. It does not fold
/// removed units' mean outputs into the next layer's bias. It has no validation set: set
/// [`validation`](Self::validation).
///
/// ```
/// use cavitation_nn::shrink::{Ranking, ShrinkConfig};
/// let config = ShrinkConfig::default();
/// assert!(config.layers.is_empty() && config.dead_first && !config.fold_means);
/// assert!(matches!(config.ranking, Ranking::OutNorm));
/// assert_eq!(
///     (config.step_bp, config.fine_from_bp, config.fine_step_bp, config.max_removed_bp),
///     (1000, 7000, 500, 9500)
/// );
/// assert_eq!(config.patience, Some(2));
/// assert!(config.criterion.is_none() && config.validation.is_none());
/// assert!(!config.keep_checkpoints && config.on_step.is_none());
/// ```
pub struct ShrinkConfig<'s> {
    /// The layers to shrink, by layer index: a `Linear` loses neurons, a `Conv2d` output
    /// channels ([`Sequential::set_channel_mask`]). Empty (the default) shrinks every hidden
    /// `Linear` (each one with another `Linear` after it) and every `Conv2d` whose channels
    /// reach a `Conv2d` or a `Linear` ([`Sequential::channel_layer`] names the end of their
    /// run). A layer listed twice is shrunk once.
    pub layers: Vec<usize>,
    /// The order in which each layer's units are removed.
    pub ranking: Ranking<'s>,
    /// Whether units whose output is 0 on every training row (for a ReLU, units that never
    /// fire) go before the ranking's order; removing them changes nothing. On by default.
    pub dead_first: bool,
    /// Whether each removed unit's mean output on the training rows is folded into the
    /// bias of the layer that receives it, so that the next layer still gets the unit's
    /// average contribution:
    ///
    /// - a neuron into the next `Linear`, `b[i] += W[i, j] · mean_j`;
    /// - a channel into the next `Conv2d`, `b[o] += mean_c · Σ_taps W[o, c, taps]`, with
    ///   `mean_c` its mean over positions ([`ChannelActivity::mean`](super::ChannelActivity::mean));
    ///   that conv must have no padding, where a border output would read zeros in place of
    ///   the channel;
    /// - a channel into the `Linear` after a `Flatten`, each position with its own mean,
    ///   `b[i] += Σ_p W[i, c·plane + p] · mean_{c,p}`.
    ///
    /// The means are measured at each step, before retraining, on the network as it stands
    /// once the step has removed the units of the listed layers before this one (and folded
    /// theirs), read where the unit's output is final
    /// ([`LayerActivity::mean`](super::LayerActivity::mean) for a neuron). For a unit whose
    /// output is the same on every row (and, for a channel, at every position) the fold is
    /// exact: the network's outputs do not change, within rounding. The fold changes the
    /// parameters, so it is in the step's checkpoint and in the network the run keeps. It
    /// costs one extra pass over the training rows per step for each listed layer after the
    /// first that loses units at that step, and for each `Conv2d` whose channels reach a
    /// `Linear`, to measure its means anew. The fold needs a bias to fold into, so with it
    /// on, [`shrink`] refuses a listed layer whose receiving layer has none, and a listed
    /// `Conv2d` whose channels reach a `Conv2d` with padding. Off by default.
    pub fold_means: bool,
    /// The step between levels, in basis points, up to `fine_from_bp`. Default 1,000.
    pub step_bp: u32,
    /// The level from which the steps are `fine_step_bp`. Default 7,000.
    pub fine_from_bp: u32,
    /// The step between levels from `fine_from_bp` on. Default 500.
    pub fine_step_bp: u32,
    /// The last level: the most of each layer the run removes, at most 10,000. Default
    /// 9,500.
    pub max_removed_bp: u32,
    /// The run ends after this many steps in a row miss the criterion; `None` runs every
    /// level. Default `Some(2)`.
    pub patience: Option<usize>,
    /// When a step is good enough. `None` (the default) is
    /// [`Criterion::AccuracyDrop`] of 1.0 point for class targets, and an error for value
    /// targets, which have no natural threshold.
    pub criterion: Option<Criterion>,
    /// The rows each step is measured on. `None` (the default) is an error.
    pub validation: Option<Validation>,
    /// How each step retrains.
    pub retrain: RetrainConfig,
    /// Whether every step keeps its masks and parameters in
    /// [`ShrinkStep::checkpoint`] (memory: one copy of the parameters per step). Off by
    /// default; the chosen step is kept either way, in the model.
    pub keep_checkpoints: bool,
    /// Called after each step with its record; [`Flow::Stop`] ends the run there, keeping
    /// the best step so far.
    // The boxed callback is the plainest type for this field; an alias would only hide it.
    #[allow(clippy::type_complexity)]
    pub on_step: Option<Box<dyn FnMut(&ShrinkStep) -> Flow + Send + 's>>,
}

impl Default for ShrinkConfig<'_> {
    /// Every hidden `Linear` and `Conv2d`, units whose output is 0 on every training row
    /// first and then out-norm, levels 1,000 to 7,000
    /// bp by 1,000 and on to 9,500 by 500, patience 2, the default criterion, no validation
    /// set, the default retraining, no checkpoints, no callback.
    fn default() -> Self {
        ShrinkConfig {
            layers: Vec::new(),
            ranking: Ranking::OutNorm,
            dead_first: true,
            fold_means: false,
            step_bp: 1000,
            fine_from_bp: 7000,
            fine_step_bp: 500,
            max_removed_bp: 9500,
            patience: Some(2),
            criterion: None,
            validation: None,
            retrain: RetrainConfig::default(),
            keep_checkpoints: false,
            on_step: None,
        }
    }
}

impl fmt::Debug for ShrinkConfig<'_> {
    /// Writes every setting; the callback shows only whether there is one.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShrinkConfig")
            .field("layers", &self.layers)
            .field("ranking", &self.ranking)
            .field("dead_first", &self.dead_first)
            .field("fold_means", &self.fold_means)
            .field("step_bp", &self.step_bp)
            .field("fine_from_bp", &self.fine_from_bp)
            .field("fine_step_bp", &self.fine_step_bp)
            .field("max_removed_bp", &self.max_removed_bp)
            .field("patience", &self.patience)
            .field("criterion", &self.criterion)
            .field("validation", &self.validation)
            .field("retrain", &self.retrain)
            .field("keep_checkpoints", &self.keep_checkpoints)
            .field("on_step", &self.on_step.as_ref().map(|_| "FnMut"))
            .finish()
    }
}

/// One step of a [`shrink`] run: the level, what was kept, and how the retrained network
/// measured on the validation set.
///
/// It is `#[non_exhaustive]`, so a later release can add a field without breaking your
/// code: read its fields, in [`ShrinkReport::steps`] or in an `on_step` callback.
///
/// ```
/// use cavitation_nn::shrink::{ShrinkConfig, ShrinkStep};
/// use cavitation_nn::Flow;
/// let config = ShrinkConfig {
///     on_step: Some(Box::new(|step: &ShrinkStep| {
///         let (bp, kept, parameters) = (step.removed_bp, &step.kept, step.parameters);
///         println!("{bp} bp: kept {kept:?}, {parameters} parameters");
///         Flow::Continue
///     })),
///     ..ShrinkConfig::default()
/// };
/// assert!(config.on_step.is_some());
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ShrinkStep {
    /// The level: the share of each layer's width removed, in basis points.
    pub removed_bp: u32,
    /// The units left on in each shrunk layer (neurons of a `Linear`, output channels of a
    /// `Conv2d`), in the order of [`ShrinkConfig::layers`] (or, when that is empty, of the
    /// layers it stands for, in order of index).
    pub kept: Vec<usize>,
    /// The number of parameters the network has once its off units are removed for good
    /// with [`compact`](super::compact()). When the report has a
    /// [`ShrinkWarning::NotCompactable`], `compact` refuses the network, which runs only
    /// masked: the count then leaves out the off neurons of its `Linear` layers and nothing
    /// else.
    pub parameters: usize,
    /// The retrained network's measures on the validation set.
    pub validation: Evaluation,
    /// Whether the step meets the criterion.
    pub met: bool,
    /// With [`ShrinkConfig::keep_checkpoints`], the step's network: each shrunk layer's
    /// mask by layer index (a neuron mask for a `Linear`, for
    /// [`Sequential::set_neuron_mask`]; a channel mask for a `Conv2d`, for
    /// [`Sequential::set_channel_mask`]) and the parameters after retraining.
    pub checkpoint: Option<(Vec<(usize, NeuronMask)>, ParamStore)>,
}

/// Something about a [`shrink`] run worth knowing, in [`ShrinkReport::warnings`]. Its
/// `Display` says what it means and what to do.
///
/// It is `#[non_exhaustive]`, so a later release can add a warning without breaking your
/// code: match it with a `_` arm.
///
/// ```
/// use cavitation_nn::shrink::ShrinkWarning;
/// let w = ShrinkWarning::SmallValidation { rows: 150, row_points: 100.0 / 150.0, threshold: 1.0 };
/// assert!(w.to_string().contains("150 rows"));
/// ```
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ShrinkWarning {
    /// One validation row is worth more than half the allowed accuracy drop
    /// (`100 / rows > points / 2`), so the stop rule turns on one or two rows, which noise
    /// alone can flip.
    SmallValidation {
        /// The validation rows.
        rows: usize,
        /// The accuracy points one row is worth, `100 / rows`.
        row_points: f64,
        /// The allowed drop, [`Criterion::AccuracyDrop`]'s `points`.
        threshold: f64,
    },
    /// Switching units off works on this network, so the run and its measures are sound,
    /// but [`compact`](super::compact()) cannot rebuild the result as a smaller dense
    /// network: use the chosen network masked. Given at the start of the run.
    NotCompactable {
        /// The layer `compact` refuses.
        layer: usize,
        /// Why, in a sentence.
        reason: String,
    },
}

impl fmt::Display for ShrinkWarning {
    /// A sentence saying what was found and what to do about it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ShrinkWarning::NotCompactable { layer, ref reason } => write!(
                f,
                "layer {layer}: {reason}, so compact cannot rebuild the shrunk network; it \
                 runs masked, with every off unit's work still done"
            ),
            ShrinkWarning::SmallValidation {
                rows,
                row_points,
                threshold,
            } => {
                write!(
                    f,
                    "the validation set has {rows} rows, so one row is {row_points:.3} accuracy \
                     points, more than half the allowed drop of {threshold} points: the stop \
                     rule turns on one or two rows"
                )?;
                if threshold > 0.0 {
                    let enough = (200.0 / threshold).ceil();
                    write!(f, "; use at least {enough} validation rows")?;
                }
                Ok(())
            }
        }
    }
}

/// What a [`shrink`] run found: the unpruned network's measures, every step, the step
/// kept, and any warnings.
///
/// It is `#[non_exhaustive]`, so a later release can add a field without breaking your
/// code: read its fields, and get one from [`shrink`].
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::shrink::{ShrinkConfig, ShrinkReport, Validation, shrink};
/// use cavitation_nn::{Dataset, Linear, Relu, Sequential, Targets};
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let mut net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 6, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 6, 2, &mut rng));
/// let data = Dataset { x: Tensor::zeros(&[8, 2]), targets: Targets::Classes(vec![0; 8]) };
/// let config = ShrinkConfig {
///     validation: Some(Validation::Given(data.clone())),
///     max_removed_bp: 2000,
///     ..ShrinkConfig::default()
/// };
/// let report: ShrinkReport = shrink(&mut net, &mut params, &data, config).unwrap();
/// assert_eq!(report.steps.len(), 2);
/// if let Some(i) = report.chosen {
///     println!("kept {:?} at {} bp", report.steps[i].kept, report.steps[i].removed_bp);
/// }
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ShrinkReport {
    /// The network as it was given, measured on the validation set.
    pub baseline: Evaluation,
    /// Every step run, in order of level.
    pub steps: Vec<ShrinkStep>,
    /// The index in `steps` of the step kept: the latest one that met the criterion, or
    /// `None` if none did.
    pub chosen: Option<usize>,
    /// What the run found worth a warning.
    pub warnings: Vec<ShrinkWarning>,
}

/// Shrinks the hidden `Linear` layers and the `Conv2d` layers of `model`: removes neurons
/// and output channels (the units) a step at a time, by switching them off with neuron
/// and channel masks, retrains after each step, measures the network on a validation set,
/// and keeps the latest step whose measures still meet the criterion.
///
/// The levels are `step_bp, 2·step_bp, …` below `fine_from_bp`, then `fine_from_bp` and
/// `fine_step_bp` increments below `max_removed_bp`, and `max_removed_bp` last (the
/// default: 1,000 to 7,000 by 1,000, then 7,500 to 9,500 by 500). At each level each
/// listed layer of width `w` (neurons, or output channels) keeps
/// `max(1, ceil(w · (10,000 − level) / 10,000))` units (units already off count as
/// removed). Each step:
///
/// 1. measures every listed layer's activity on the training rows
///    ([`activity()`](super::activity()) for a `Linear`,
///    [`channel_activity()`](super::channel_activity()) for a `Conv2d`) and ranks its units
///    that are on, on the network as the previous step left it;
/// 2. switches off, in each layer, the units it must lose: with
///    [`dead_first`](ShrinkConfig::dead_first), first those whose output is 0 on every
///    training row, then the lowest-ranked. With [`fold_means`](ShrinkConfig::fold_means),
///    the layers are taken in order of index, and before a layer's units go their mean
///    outputs, measured on the network with the earlier layers already shrunk, are folded
///    into the bias of the layer that receives them;
/// 3. retrains the network on the training rows as [`ShrinkConfig::retrain`] says, with a
///    fresh optimiser (masked units pass no gradient, so they stay off);
/// 4. measures it on the validation set, records the step, and calls `on_step`
///    ([`Flow::Stop`] ends the run).
///
/// The run also ends after [`patience`](ShrinkConfig::patience) misses in a row, or after
/// the last level. The step kept is the latest one that met the criterion, not the one
/// before the first miss: a later step can recover what an earlier one lost. On return
/// `model` and `params` hold the kept step's network: its masks on the listed layers and
/// its retrained parameters. If no step met the criterion they are as they were given.
/// Call [`compact`](super::compact()) on the result for the smaller network that runs
/// faster. When `compact` would refuse the network (a layer of your own with parameters,
/// or an element-wise layer of your own after a shrunk layer), the run still works and the
/// report starts with a [`ShrinkWarning::NotCompactable`]: the result can be used masked,
/// but not compacted.
///
/// What the run finds is a size and a network of that size. A network of the same size
/// trained from scratch may match the pruned one or fall short of it, depending on the
/// problem and the training: train one, for as many epochs in all, with
/// [`train_found_size`](super::train_found_size()), and compare before you settle on the
/// pruned network. The guide's chapter on shrinking does this for the digits network.
///
/// Units the model already has off stay off. The masks the run sets are neuron masks, by
/// `Linear` ([`Sequential::set_neuron_mask`]), and channel masks, by `Conv2d`
/// ([`Sequential::set_channel_mask`]); masks on other layers are left as they are. The
/// baseline and every step are measured with the loss in [`RetrainConfig::loss`], 256 rows
/// at a time; a warning is added when one validation row is worth more than half of an
/// [`AccuracyDrop`](Criterion::AccuracyDrop)'s points.
///
/// # Errors
///
/// Returns [`Error::InvalidData`], before changing anything, if
/// [`ShrinkConfig::validation`] is `None`; if the training data or a given validation set
/// is not a matrix of inputs with at least one row and one target per row (the message
/// says which); if a [`Validation::Carve`] share is not above 0
/// and below 1 or leaves either part without rows; if the model has nothing to shrink (no
/// `Linear` with another `Linear` after it and no `Conv2d` whose channels reach a `Conv2d`
/// or a `Linear`), or a listed layer is neither a hidden `Linear` nor a `Conv2d`; if
/// [`ShrinkConfig::criterion`] is `None` for value targets, or an `AccuracyDrop` is given
/// for value targets, or the criterion's number is NaN; if `step_bp` or `fine_step_bp` is
/// 0, `max_removed_bp` is 0 or above 10,000, `patience` is `Some(0)` or the retraining
/// batch is 0; if a listed layer's mask does not have its width; if
/// [`fold_means`](ShrinkConfig::fold_means) is on and the layer that receives a listed
/// layer's units has no bias to fold into, or is a `Conv2d` with padding after a listed
/// `Conv2d`; and as [`evaluate`] does for the
/// validation set. Returns [`Error::Unsupported`] if a listed layer, or a layer between it
/// and the layer that holds its units, carries a mask set with [`Sequential::set_mask`]
/// (clear it and mask by `Linear` or by `Conv2d` instead), if the mask at the end of a
/// listed `Conv2d`'s run is not a whole-channel mask, and if the layer after a listed
/// layer's run is not a `Linear` (for a `Linear`: a `Softmax` or a `Glu`, say) or neither a
/// `Conv2d` nor a `Linear` (for a `Conv2d`), where a switched-off unit still has an effect,
/// so that masking it does not remove it. During the run, returns the errors of
/// [`activity()`](super::activity()), [`channel_activity()`](super::channel_activity())
/// and [`fit`] (an [`Error::NonFinite`] from a NaN or infinite loss or gradient among
/// them), and [`Error::InvalidData`] if a [`Ranking::Custom`] function returns the wrong
/// number of scores; `model` and `params` are then put back as they were given.
///
/// # Panics
///
/// Panics as [`fit`] does when the layers' widths do not fit one another.
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::shrink::{ShrinkConfig, Validation, shrink};
/// use cavitation_nn::{Dataset, FitConfig, Linear, Relu, Sequential, Targets, fit};
/// // Two classes split by the sign of the first input.
/// let mut rng = Rng::seed(1);
/// let x: Vec<f32> = (0..400).map(|_| rng.range(-1.0, 1.0) as f32).collect();
/// let labels = x.chunks(2).map(|r| usize::from(r[0] > 0.0)).collect();
/// let data = Dataset {
///     x: Tensor::from_vec(x, &[200, 2]).unwrap(),
///     targets: Targets::Classes(labels),
/// };
/// let (train, held_out) = data.split_at(100);
/// let mut params = ParamStore::new();
/// let mut net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 16, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 16, 2, &mut rng));
/// fit(&net, &mut params, &train, FitConfig { epochs: 30, ..FitConfig::default() }).unwrap();
///
/// let config = ShrinkConfig {
///     validation: Some(Validation::Given(held_out)),
///     max_removed_bp: 5000,
///     ..ShrinkConfig::default()
/// };
/// let report = shrink(&mut net, &mut params, &train, config).unwrap();
/// // 100 validation rows make one row a whole point: more than half the allowed drop.
/// assert_eq!(report.warnings.len(), 1);
/// for step in &report.steps {
///     println!("{} bp: kept {:?}, met {}", step.removed_bp, step.kept, step.met);
/// }
/// match report.chosen {
///     Some(i) => assert_eq!(net.neuron_mask(0).unwrap().count_on(), report.steps[i].kept[0]),
///     None => assert!(net.neuron_mask(0).is_none()),
/// }
/// ```
pub fn shrink(
    model: &mut Sequential,
    params: &mut ParamStore,
    train: &Dataset,
    config: ShrinkConfig<'_>,
) -> Result<ShrinkReport, Error> {
    let loss = config.retrain.loss;
    let mut measure = |model: &Sequential, params: &ParamStore, data: &Dataset| {
        evaluate(model, params, data, loss, EVAL_BATCH)
    };
    run(model, params, train, config, &mut measure)
}

/// The measurement [`run`] makes on the validation set: [`evaluate`] in [`shrink`], a
/// script in the tests.
type Measure<'a> = dyn FnMut(&Sequential, &ParamStore, &Dataset) -> Result<Evaluation, Error> + 'a;

/// [`shrink`] with the validation measurement given by `measure`, so that a test can script
/// what each step measures.
pub(crate) fn run(
    model: &mut Sequential,
    params: &mut ParamStore,
    train: &Dataset,
    mut config: ShrinkConfig<'_>,
    measure: &mut Measure<'_>,
) -> Result<ShrinkReport, Error> {
    let (train, validation) = split(train, config.validation.take())?;
    let targets = targets(model, &config.layers)?;
    check_masks(model, &targets)?;
    let mut warnings = check_layout(model, params, &targets)?;
    if config.fold_means {
        check_fold_biases(model, &targets)?;
    }
    let criterion = criterion(config.criterion, &train.targets)?;
    check_numbers(&config)?;
    let levels = levels(
        config.step_bp,
        config.fine_from_bp,
        config.fine_step_bp,
        config.max_removed_bp,
    );
    let rows = validation.len();
    if let Criterion::AccuracyDrop { points } = criterion {
        let row_points = 100.0 / rows as f64;
        if row_points > points / 2.0 {
            warnings.push(ShrinkWarning::SmallValidation {
                rows,
                row_points,
                threshold: points,
            });
        }
    }
    let baseline = measure(model, params, &validation)?;
    // The mask on the layer that holds each target's units, as given.
    let original_masks: Vec<Option<NeuronMask>> = targets
        .iter()
        .map(|t| model.mask(t.end(model)).cloned())
        .collect();
    let original_params = params.clone();
    let mut run = Run {
        model,
        params,
        train: &train,
        validation: &validation,
        targets: &targets,
        criterion,
        baseline,
        config: &mut config,
        measure,
    };
    let outcome = run.steps(&levels);
    let (model, params) = (run.model, run.params);
    let restore = |model: &mut Sequential, params: &mut ParamStore| {
        for (target, mask) in targets.iter().zip(&original_masks) {
            let end = target.end(model);
            match mask {
                Some(mask) => model.set_mask(end, mask.clone()),
                None => model.clear_mask(end),
            }
        }
        *params = original_params.clone();
    };
    let (steps, best) = match outcome {
        Ok(found) => found,
        Err(e) => {
            restore(model, params);
            return Err(e);
        }
    };
    let chosen = match best {
        Some(Best { step, masks, store }) => {
            for (target, (_, mask)) in targets.iter().zip(masks) {
                target.set(model, mask);
            }
            *params = store;
            Some(step)
        }
        None => {
            restore(model, params);
            None
        }
    };
    Ok(ShrinkReport {
        baseline,
        steps,
        chosen,
        warnings,
    })
}

/// What a layer [`shrink`] shrinks loses: the neurons of a `Linear` or the output channels
/// of a `Conv2d`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unit {
    /// The neurons of a `Linear`, masked with `set_neuron_mask`.
    Neurons,
    /// The output channels of a `Conv2d`, masked with `set_channel_mask`.
    Channels,
}

/// A layer [`shrink`] shrinks: its index and what it loses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Target {
    /// The layer's index.
    layer: usize,
    /// What it loses.
    unit: Unit,
}

impl Target {
    /// The layer's units: a `Linear`'s outputs, a `Conv2d`'s output channels.
    fn width(&self, model: &Sequential) -> usize {
        match self.unit {
            Unit::Neurons => model.linear(self.layer).expect("a Linear").outputs(),
            Unit::Channels => model.channel_shape(self.layer).0,
        }
    }

    /// The layer that holds the units as the next layer receives them, where their mask
    /// goes: [`Sequential::neuron_layer`] or [`Sequential::channel_layer`].
    fn end(&self, model: &Sequential) -> usize {
        match self.unit {
            Unit::Neurons => model.neuron_layer(self.layer),
            Unit::Channels => model.channel_layer(self.layer),
        }
    }

    /// Which units are on.
    fn on(&self, model: &Sequential) -> Vec<bool> {
        match self.unit {
            Unit::Neurons => neurons_on(model, self.layer),
            Unit::Channels => channels_on(model, self.layer),
        }
    }

    /// Puts `mask`, one bit per unit, on the layer.
    fn set(&self, model: &mut Sequential, mask: NeuronMask) {
        match self.unit {
            Unit::Neurons => model.set_neuron_mask(self.layer, mask),
            Unit::Channels => model.set_channel_mask(self.layer, mask),
        }
    }

    /// The word for one unit, for messages.
    fn kind(&self) -> &'static str {
        match self.unit {
            Unit::Neurons => "linear",
            Unit::Channels => "conv",
        }
    }
}

/// One layer's activity on the training rows, per unit, as [`RankContext`] shows it.
struct UnitStats {
    /// The mean output.
    mean: Vec<f64>,
    /// The mean absolute output.
    mean_abs: Vec<f64>,
    /// The share of outputs above 0.
    firing: Vec<f64>,
    /// The largest output.
    max: Vec<f64>,
}

/// The activity of each of `targets` on `data`, in their order: one pass for the `Linear`
/// targets and one for the `Conv2d` targets (each left out when there are none).
fn unit_stats(
    model: &Sequential,
    params: &ParamStore,
    data: &Dataset,
    targets: &[Target],
) -> Result<Vec<UnitStats>, Error> {
    let of = |unit: Unit| -> Vec<usize> {
        targets
            .iter()
            .filter(|t| t.unit == unit)
            .map(|t| t.layer)
            .collect()
    };
    let (linears, convs) = (of(Unit::Neurons), of(Unit::Channels));
    let mut neurons = if linears.is_empty() {
        Vec::new()
    } else {
        activity(model, params, data, &linears, EVAL_BATCH)?
    }
    .into_iter();
    let mut channels = if convs.is_empty() {
        Vec::new()
    } else {
        channel_activity(model, params, data, &convs, EVAL_BATCH)?
    }
    .into_iter();
    Ok(targets
        .iter()
        .map(|t| match t.unit {
            Unit::Neurons => {
                let a = neurons.next().expect("one activity per Linear target");
                UnitStats {
                    mean: a.mean,
                    mean_abs: a.mean_abs,
                    firing: a.firing,
                    max: a.max,
                }
            }
            Unit::Channels => {
                let a = channels.next().expect("one activity per Conv2d target");
                UnitStats {
                    mean: a.mean,
                    mean_abs: a.mean_abs,
                    firing: a.firing,
                    max: a.max,
                }
            }
        })
        .collect())
}

/// The latest step that met the criterion: its index, masks and parameters.
struct Best {
    /// The step's index in the report.
    step: usize,
    /// Each shrunk layer's mask, by layer index, in the order of the targets.
    masks: Vec<(usize, NeuronMask)>,
    /// The parameters after the step's retraining.
    store: ParamStore,
}

/// The state of one [`run`]: the network being shrunk and everything each step reads.
struct Run<'r, 'm, 'c> {
    /// The network, masked step by step.
    model: &'r mut Sequential,
    /// Its parameters, retrained step by step.
    params: &'r mut ParamStore,
    /// The rows to measure activity and retrain on.
    train: &'r Dataset,
    /// The rows to measure each step on.
    validation: &'r Dataset,
    /// The layers to shrink.
    targets: &'r [Target],
    /// When a step is good enough.
    criterion: Criterion,
    /// The unpruned network's measures on the validation set.
    baseline: Evaluation,
    /// The run's settings (the ranking and the callback are used mutably).
    config: &'r mut ShrinkConfig<'c>,
    /// The validation measurement.
    measure: &'r mut Measure<'m>,
}

impl Run<'_, '_, '_> {
    /// Runs the steps at `levels` until the last, `Flow::Stop` or the patience runs out,
    /// and returns them with the latest step that met the criterion.
    fn steps(&mut self, levels: &[u32]) -> Result<(Vec<ShrinkStep>, Option<Best>), Error> {
        let widths: Vec<usize> = self.targets.iter().map(|t| t.width(self.model)).collect();
        let mut rng = Rng::seed(match self.config.ranking {
            Ranking::Random(seed) => seed,
            _ => 0,
        });
        let mut steps: Vec<ShrinkStep> = Vec::with_capacity(levels.len());
        let mut best = None;
        let mut misses = 0;
        for &level in levels {
            let stats = unit_stats(self.model, self.params, self.train, self.targets)?;
            let mut masks = Vec::with_capacity(self.targets.len());
            for (k, &target) in self.targets.iter().enumerate() {
                let keep = kept_at(widths[k], level);
                let mask = self.mask_for(target, keep, &stats[k], &mut rng)?;
                masks.push((target.layer, mask));
            }
            if self.config.fold_means {
                self.fold_and_mask(&masks, &stats)?;
            } else {
                for (target, (_, mask)) in self.targets.iter().zip(&masks) {
                    target.set(self.model, mask.clone());
                }
            }
            self.retrain()?;
            let validation = (self.measure)(self.model, self.params, self.validation)?;
            let met = self
                .criterion
                .met(&self.baseline, &validation, self.validation.len());
            if met {
                best = Some(Best {
                    step: steps.len(),
                    masks: masks.clone(),
                    store: self.params.clone(),
                });
                misses = 0;
            } else {
                misses += 1;
            }
            let checkpoint = self
                .config
                .keep_checkpoints
                .then(|| (masks.clone(), self.params.clone()));
            steps.push(ShrinkStep {
                removed_bp: level,
                kept: masks.iter().map(|(_, m)| m.count_on()).collect(),
                parameters: parameter_count(self.model, self.params),
                validation,
                met,
                checkpoint,
            });
            if let Some(on_step) = self.config.on_step.as_mut()
                && on_step(steps.last().expect("just pushed")) == Flow::Stop
            {
                break;
            }
            if self.config.patience.is_some_and(|p| misses >= p) {
                break;
            }
        }
        Ok((steps, best))
    }

    /// Sets the step's `masks`, one listed layer at a time in order of layer index, and
    /// before each folds the mean output of every unit the mask newly switches off into
    /// the bias of the layer that receives it, summed in `f64` and rounded once:
    ///
    /// - a neuron into the next `Linear`, `b[i] += Σⱼ W[i, j] · mean_j`;
    /// - a channel into the next `Conv2d` (which has no padding), `b[o] += Σ_c mean_c ·
    ///   Σ_taps W[o, c, taps]`, with `mean_c` its mean over positions and rows;
    /// - a channel into the `Linear` after a `Flatten`, `b[i] += Σ_c Σ_p W[i, c·plane + p] ·
    ///   mean_{c,p}`, with `mean_{c,p}` the mean of each of its positions over the rows,
    ///   measured for this fold.
    ///
    /// The means of the first layer that loses units are `stats`, measured before the step
    /// changed anything; those of a later layer are measured again, on the network with the
    /// earlier layers' masks and folds in place. A mean of exactly 0 (a dead unit) adds
    /// nothing, even against an infinite weight.
    fn fold_and_mask(
        &mut self,
        masks: &[(usize, NeuronMask)],
        stats: &[UnitStats],
    ) -> Result<(), Error> {
        let mut order: Vec<usize> = (0..masks.len()).collect();
        order.sort_by_key(|&k| masks[k].0);
        let mut changed = false;
        for k in order {
            let target = self.targets[k];
            let mask = &masks[k].1;
            let before = target.on(self.model);
            let removed: Vec<usize> = (0..before.len())
                .filter(|&j| before[j] && !mask.is_on(j))
                .collect();
            if !removed.is_empty() {
                let increments = self.fold_increments(target, &removed, &stats[k], changed)?;
                let bias = receiver_bias(self.model, target)
                    .expect("check_fold_biases found a bias after each run");
                let b = self.params.value_mut(bias).data_mut();
                for (b, increment) in b.iter_mut().zip(increments) {
                    *b = (f64::from(*b) + increment) as f32;
                }
                changed = true;
            }
            target.set(self.model, mask.clone());
        }
        Ok(())
    }

    /// What [`fold_and_mask`](Self::fold_and_mask) adds to each entry of the receiving
    /// layer's bias for the units `removed` of `target`: their means (from `stats`, or
    /// measured again when `remeasure`) times their outgoing weights.
    fn fold_increments(
        &self,
        target: Target,
        removed: &[usize],
        stats: &UnitStats,
        remeasure: bool,
    ) -> Result<Vec<f64>, Error> {
        let next_at = target.end(self.model) + 1;
        let train = self.train;
        if target.unit == Unit::Channels && self.model.linear(next_at).is_some() {
            // A channel into a `Linear`: each of its positions has its own weights, so its
            // own mean.
            let means = position_means(self.model, self.params, train, target.layer, EVAL_BATCH)?;
            let plane = self.model.channel_shape(target.layer).1;
            let next = self.model.linear(next_at).expect("just checked");
            let w = self.params.value(next.weight()).data();
            return Ok(w
                .chunks_exact(next.inputs())
                .map(|row| {
                    removed
                        .iter()
                        .flat_map(|&c| c * plane..(c + 1) * plane)
                        .filter(|&j| means[j] != 0.0)
                        .map(|j| f64::from(row[j]) * means[j])
                        .sum()
                })
                .collect());
        }
        let measured;
        let mean: &[f64] = if remeasure {
            measured = match target.unit {
                Unit::Neurons => {
                    activity(self.model, self.params, train, &[target.layer], EVAL_BATCH)?
                        .remove(0)
                        .mean
                }
                Unit::Channels => {
                    channel_activity(self.model, self.params, train, &[target.layer], EVAL_BATCH)?
                        .remove(0)
                        .mean
                }
            };
            &measured
        } else {
            &stats.mean
        };
        if let Some(conv) = self.model.layer(next_at).as_conv2d() {
            // A channel into a `Conv2d` without padding: every output reads every tap of the
            // channel, so the channel's mean times the sum of its taps.
            let g = conv.geom();
            let (kh, kw) = g.kernel();
            let (taps, ins) = (kh * kw, g.in_channels());
            let w = self.params.value(conv.weight()).data();
            return Ok((0..g.out_channels())
                .map(|o| {
                    removed
                        .iter()
                        .filter(|&&c| mean[c] != 0.0)
                        .map(|&c| {
                            let taps_sum: f64 = w[(o * ins + c) * taps..][..taps]
                                .iter()
                                .map(|&v| f64::from(v))
                                .sum();
                            taps_sum * mean[c]
                        })
                        .sum()
                })
                .collect());
        }
        let next = linear_after_run(self.model, target.layer)
            .expect("check_layout found a Linear after each run");
        let w = self.params.value(next.weight()).data();
        Ok(w.chunks_exact(next.inputs())
            .map(|row| {
                removed
                    .iter()
                    .filter(|&&j| mean[j] != 0.0)
                    .map(|&j| f64::from(row[j]) * mean[j])
                    .sum()
            })
            .collect())
    }

    /// The mask of `target` that keeps `keep` of its units (or every unit on, if no more
    /// than `keep` are on): its on units ranked, dead ones first with `dead_first`, and the
    /// first ones in that order switched off.
    fn mask_for(
        &mut self,
        target: Target,
        keep: usize,
        stats: &UnitStats,
        rng: &mut Rng,
    ) -> Result<NeuronMask, Error> {
        let flags = target.on(self.model);
        let on: Vec<usize> = (0..flags.len()).filter(|&j| flags[j]).collect();
        let mut mask = NeuronMask::from_on(flags.len(), &on);
        if on.len() <= keep {
            return Ok(mask);
        }
        let scores = self.scores(target, &on, stats, rng)?;
        let dead_first = self.config.dead_first;
        let dead = |i: usize| dead_first && stats.mean_abs[on[i]] == 0.0;
        let mut order: Vec<usize> = (0..on.len()).collect();
        order.sort_by(|&a, &b| {
            dead(b)
                .cmp(&dead(a))
                .then(scores[a].is_nan().cmp(&scores[b].is_nan()))
                .then(scores[a].total_cmp(&scores[b]))
                .then(on[a].cmp(&on[b]))
        });
        for &i in &order[..on.len() - keep] {
            mask.set(on[i], false);
        }
        Ok(mask)
    }

    /// The ranking's score of each unit in `on` of `target`.
    fn scores(
        &mut self,
        target: Target,
        on: &[usize],
        stats: &UnitStats,
        rng: &mut Rng,
    ) -> Result<Vec<f64>, Error> {
        let scores = match &mut self.config.ranking {
            Ranking::OutNorm => out_norms(self.model, self.params, target, on),
            Ranking::Activity => on.iter().map(|&j| stats.mean_abs[j]).collect(),
            Ranking::Random(_) => on.iter().map(|_| rng.uniform_f64()).collect(),
            Ranking::Custom(rank) => {
                let scores = rank(&RankContext {
                    layer: target.layer,
                    on,
                    model: self.model,
                    params: self.params,
                    mean: &stats.mean,
                    mean_abs: &stats.mean_abs,
                    firing: &stats.firing,
                    max: &stats.max,
                });
                if scores.len() != on.len() {
                    return Err(Error::InvalidData(format!(
                        "shrink: the custom ranking returned {} scores for the {} layer {}, \
                         which has {} units on; return one score per entry of RankContext::on",
                        scores.len(),
                        target.kind(),
                        target.layer,
                        on.len()
                    )));
                }
                scores
            }
        };
        Ok(scores)
    }

    /// Retrains the network on the training rows as the config says, with a fresh
    /// optimiser and quiet health checks that still stop on a non-finite value.
    fn retrain(&mut self) -> Result<(), Error> {
        let r = self.config.retrain;
        let config = FitConfig {
            epochs: r.epochs,
            batch: r.batch,
            optimizer: r.optimizer,
            loss: r.loss,
            shuffle_seed: r.shuffle_seed,
            // A network with dropout draws its retraining masks from the shuffle seed.
            dropout_seed: r.shuffle_seed.unwrap_or(0),
            schedule: LrSchedule::Constant,
            health: Health::quiet(),
            on_epoch: None,
            validation: None,
            early_stopping: None,
            device: crate::Device::Cpu,
        };
        fit(self.model, self.params, self.train, config).map(|_| ())
    }
}

/// The Euclidean norm of the outgoing weights of each unit in `on` of `target`: a neuron's
/// column in the next `Linear`; a channel's input channel in the next `Conv2d` (every
/// output channel and tap), or its block of columns in the next `Linear`. Summed in `f64`.
/// `check_layout` found the receiving layer.
fn out_norms(model: &Sequential, params: &ParamStore, target: Target, on: &[usize]) -> Vec<f64> {
    let norm = |values: &mut dyn Iterator<Item = f32>| -> f64 {
        values
            .map(|v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            .sqrt()
    };
    let next_at = target.end(model) + 1;
    if let Some(conv) = model.layer(next_at).as_conv2d() {
        let g = conv.geom();
        let (kh, kw) = g.kernel();
        let (taps, ins) = (kh * kw, g.in_channels());
        let w = params.value(conv.weight()).data();
        return on
            .iter()
            .map(|&c| {
                norm(
                    &mut (0..g.out_channels())
                        .flat_map(|o| w[(o * ins + c) * taps..][..taps].iter().copied()),
                )
            })
            .collect();
    }
    let next = model
        .linear(next_at)
        .expect("check_layout found a Conv2d or a Linear after each run");
    let (w, inputs) = (params.value(next.weight()).data(), next.inputs());
    let plane = match target.unit {
        Unit::Neurons => 1,
        Unit::Channels => model.channel_shape(target.layer).1,
    };
    on.iter()
        .map(|&j| {
            norm(
                &mut w
                    .chunks_exact(inputs)
                    .flat_map(|row| row[j * plane..(j + 1) * plane].iter().copied()),
            )
        })
        .collect()
}

/// The rows to retrain on and the validation set, from `train` and the config's
/// validation.
fn split(
    train: &Dataset,
    validation: Option<Validation>,
) -> Result<(Cow<'_, Dataset>, Dataset), Error> {
    let carve_rule = "the model must not have seen those rows: carve them off before the \
                      first training, or give rows it has not seen with Validation::Given";
    let Some(validation) = validation else {
        return Err(Error::InvalidData(format!(
            "shrink: ShrinkConfig::validation is None, and the stop rule needs a validation \
             set the model must not have seen: give one with Validation::Given(rows), or \
             hold out the last rows of the training data with Validation::Carve {{ share }} \
             ({carve_rule})"
        )));
    };
    let prefixed = |what: &str, e: Error| match e {
        Error::InvalidData(msg) => Error::InvalidData(format!("shrink: {what}: {msg}")),
        other => other,
    };
    check_dataset(train).map_err(|e| prefixed("the training data", e))?;
    match validation {
        Validation::Given(rows) => {
            check_dataset(&rows).map_err(|e| prefixed("the validation set", e))?;
            Ok((Cow::Borrowed(train), rows))
        }
        Validation::Carve { share } => {
            let n = train.len();
            if !(share > 0.0 && share < 1.0) {
                return Err(Error::InvalidData(format!(
                    "shrink: Validation::Carve {{ share: {share} }} must hold out a share above \
                     0 and below 1 of the training rows; {carve_rule}"
                )));
            }
            let held = carved_rows(share, n);
            if held >= n {
                return Err(Error::InvalidData(format!(
                    "shrink: Validation::Carve {{ share: {share} }} holds out {held} of the {n} \
                     training rows and leaves none to retrain on; {carve_rule}"
                )));
            }
            let (rest, held_out) = train.split_at(n - held);
            Ok((Cow::Owned(rest), held_out))
        }
    }
}

/// The rows `Validation::Carve { share }` holds out of `n`: `ceil(share · n)`, with `share`
/// read as the shortest decimal that prints as it (`0.07` is 7/100, not the binary value
/// just above it), so that `0.07` of 100 rows is 7 rows. The arithmetic is in integers;
/// a share whose decimal has more than 30 digits after the point, or a product too large
/// for them, falls back to the floating-point product.
fn carved_rows(share: f64, n: usize) -> usize {
    let text = share.to_string();
    let digits = text.split_once('.').map_or("", |(_, frac)| frac);
    let exact = (digits.len() <= 30)
        .then(|| digits.parse::<u128>().ok())
        .flatten()
        .and_then(|num| {
            let den = 10u128.pow(digits.len() as u32);
            let whole = text.split_once('.').map_or(text.as_str(), |(w, _)| w);
            let whole = whole.parse::<u128>().ok()?;
            let scaled = whole.checked_mul(den)?.checked_add(num)?;
            Some(scaled.checked_mul(n as u128)?.div_ceil(den))
        });
    match exact {
        Some(rows) => usize::try_from(rows).unwrap_or(usize::MAX),
        None => (share * n as f64).ceil() as usize,
    }
}

/// Whether the channels of the `Conv2d` at layer `conv` reach a `Conv2d` or a `Linear`,
/// the layers compaction can remove them from.
fn channels_received(model: &Sequential, conv: usize) -> bool {
    let next = model.channel_layer(conv) + 1;
    next < model.len() && (model.layer(next).as_conv2d().is_some() || model.linear(next).is_some())
}

/// The layers to shrink: `listed` without repeats, each checked to be a hidden `Linear` or
/// a `Conv2d`, or, when `listed` is empty, every hidden `Linear` of `model` and every
/// `Conv2d` whose channels reach a `Conv2d` or a `Linear`, in order of index.
fn targets(model: &Sequential, listed: &[usize]) -> Result<Vec<Target>, Error> {
    let conv = |layer| Target {
        layer,
        unit: Unit::Channels,
    };
    let linear = |layer| Target {
        layer,
        unit: Unit::Neurons,
    };
    if listed.is_empty() {
        let all: Vec<Target> = (0..model.len())
            .filter_map(|i| {
                if model.linear(i).is_some() {
                    (i + 1..model.len())
                        .any(|l| model.linear(l).is_some())
                        .then(|| linear(i))
                } else if model.layer(i).as_conv2d().is_some() {
                    channels_received(model, i).then(|| conv(i))
                } else {
                    None
                }
            })
            .collect();
        if all.is_empty() {
            return Err(Error::InvalidData(
                "shrink: the model has no hidden layer to shrink: a hidden layer is a Linear \
                 with another Linear after it (the last Linear's neurons are the model's \
                 outputs), or a Conv2d whose channels reach a Conv2d or a Linear"
                    .to_string(),
            ));
        }
        return Ok(all);
    }
    let mut chosen: Vec<Target> = Vec::with_capacity(listed.len());
    for &layer in listed {
        let target = if layer < model.len() && model.layer(layer).as_conv2d().is_some() {
            conv(layer)
        } else if layer < model.len() && model.linear(layer).is_some() {
            check_hidden(model, layer, "shrink")?;
            linear(layer)
        } else {
            return Err(Error::InvalidData(format!(
                "shrink: layer {layer} is neither a Linear nor a Conv2d layer"
            )));
        };
        if !chosen.contains(&target) {
            chosen.push(target);
        }
    }
    Ok(chosen)
}

/// Refuses a mask the run cannot work with on a listed layer's run: one set with
/// `set_mask` on the layer or a layer before the one that holds its units
/// (`Unsupported`), a mask there that is not a whole-channel mask for a `Conv2d`
/// (`Unsupported`), or a mask of the wrong width (`InvalidData`).
fn check_masks(model: &Sequential, targets: &[Target]) -> Result<(), Error> {
    for &target in targets {
        let (layer, end) = (target.layer, target.end(model));
        let setter = match target.unit {
            Unit::Neurons => "set_neuron_mask",
            Unit::Channels => "set_channel_mask",
        };
        for l in layer..end {
            if model.mask(l).is_some() {
                return Err(Error::Unsupported(format!(
                    "shrink: layer {l} ({}) carries a mask set with set_mask, inside the run of \
                     the {} layer {layer}, whose units the tool masks at layer {end}; clear it \
                     with clear_mask({l}) and mask those units with {setter}({layer}, mask) \
                     instead",
                    model.layer(l).name(),
                    target.kind()
                )));
            }
        }
        let Some(mask) = model.mask(end) else {
            continue;
        };
        match target.unit {
            Unit::Neurons => {
                let width = target.width(model);
                if mask.width() != width {
                    return Err(Error::InvalidData(format!(
                        "shrink: the mask on layer {end} has width {}, but the linear layer \
                         {layer} outputs width {width}",
                        mask.width()
                    )));
                }
            }
            Unit::Channels => {
                let (channels, plane) = model.channel_shape(layer);
                if mask.width() != channels * plane {
                    return Err(Error::InvalidData(format!(
                        "shrink: the mask on layer {end} has width {}, but the conv layer \
                         {layer}'s {channels} channels are {} values there",
                        mask.width(),
                        channels * plane
                    )));
                }
                let whole = model
                    .channel_mask(layer)
                    .is_some_and(|m| model.channel_values(layer, &m) == *mask);
                if !whole {
                    return Err(Error::Unsupported(format!(
                        "shrink: the mask on layer {end} ({}) switches off part of a channel of \
                         the conv layer {layer}, which the tool masks by whole channels; clear \
                         it with clear_mask({end}) and mask whole channels with \
                         set_channel_mask({layer}, mask) instead",
                        model.layer(end).name()
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Checks, before any training, that switching off a unit of each listed layer removes
/// it: a `Linear`'s neurons must reach the next `Linear` through element-wise layers only,
/// a `Conv2d`'s channels a `Conv2d` or a `Linear` through its run (`Unsupported`
/// otherwise, naming the layer they reach). Returns a [`ShrinkWarning::NotCompactable`]
/// for each layer [`compact`](super::compact()) would refuse in the result: a layer of
/// your own with parameters anywhere, or a layer in a listed run that compaction cannot
/// follow (for a `Linear`, an element-wise layer that is neither an activation nor a
/// `Prelu`; for a `Conv2d`, a layer other than batch norm, a pooling, an activation, a
/// one-slope `Prelu`, `Flatten`, `Dropout` or `Identity`). These are the layout checks
/// `compact` makes.
fn check_layout(
    model: &Sequential,
    params: &ParamStore,
    targets: &[Target],
) -> Result<Vec<ShrinkWarning>, Error> {
    let mut warnings = Vec::new();
    for &target in targets {
        let layer = target.layer;
        let next = target.end(model) + 1;
        let reached = || {
            if next < model.len() {
                format!("layer {next} ({})", model.layer(next).name())
            } else {
                "the model's output".to_string()
            }
        };
        match target.unit {
            Unit::Neurons => {
                if linear_after_run(model, layer).is_none() {
                    return Err(Error::Unsupported(format!(
                        "shrink: the neurons of the linear layer {layer} reach {}, which is \
                         not a Linear, so switching one off does not remove it (a softmax \
                         still counts a 0 input, a GLU still passes half of a value whose gate \
                         is 0); shrink only Linear layers whose run of element-wise layers \
                         leads to the next Linear, listed in ShrinkConfig::layers",
                        reached()
                    )));
                }
                if let Some(l) = unvouched_layer(model, layer) {
                    warnings.push(ShrinkWarning::NotCompactable {
                        layer: l,
                        reason: format!(
                            "the element-wise layer {} after the linear layer {layer} is \
                             neither an activation (a built-in or an ActivationLayer) nor a \
                             Prelu, and compact cannot tell whether it treats every position \
                             alike",
                            model.layer(l).name()
                        ),
                    });
                }
            }
            Unit::Channels => {
                if !channels_received(model, layer) {
                    return Err(Error::Unsupported(format!(
                        "shrink: the channels of the conv layer {layer} reach {}, which is \
                         neither a Conv2d nor a Linear, so switching one off does not remove \
                         it; shrink only Conv2d layers whose channels reach a Conv2d or a \
                         Linear, listed in ShrinkConfig::layers",
                        reached()
                    )));
                }
                if let Some(l) = unvouched_channel_layer(model, params, layer) {
                    warnings.push(ShrinkWarning::NotCompactable {
                        layer: l,
                        reason: format!(
                            "the layer {} is in the run of the conv layer {layer}, and compact \
                             cannot tell whether it treats every channel alike",
                            model.layer(l).name()
                        ),
                    });
                }
            }
        }
    }
    let foreign = if has_image_layers(model) {
        foreign_image_parameters(model)
    } else {
        foreign_parameters(model)
    };
    if let Some(layer) = foreign {
        // One warning per layer: a layer of your own can be both in a run and have
        // parameters, and its parameters are the reason that decides.
        warnings.retain(|warning| {
            !matches!(warning, ShrinkWarning::NotCompactable { layer: l, .. } if *l == layer)
        });
        let rebuilt = if has_image_layers(model) {
            "a Conv2d, a BatchNorm2d, a Linear nor a Prelu"
        } else {
            "a Linear nor a Prelu"
        };
        warnings.push(ShrinkWarning::NotCompactable {
            layer,
            reason: format!(
                "the layer {} has parameters and is neither {rebuilt}, and compact cannot move \
                 them to a new store",
                model.layer(layer).name()
            ),
        });
    }
    Ok(warnings)
}

/// The bias of the layer that receives the units of `target`: the next `Linear` after a
/// `Linear`'s run, or the `Conv2d` or `Linear` after a `Conv2d`'s run; `None` if it has
/// none (or the layer is neither, which `check_layout` refused).
fn receiver_bias(model: &Sequential, target: Target) -> Option<cavitation_core::ParamId> {
    let next = target.end(model) + 1;
    if next >= model.len() {
        return None;
    }
    match model.layer(next).as_conv2d() {
        Some(conv) => conv.bias(),
        None => model.linear(next).and_then(|l| l.bias()),
    }
}

/// Refuses, with `InvalidData`, a listed layer whose receiving layer has no bias, which
/// [`ShrinkConfig::fold_means`] would have to fold the removed units' means into, and a
/// listed `Conv2d` whose channels reach a `Conv2d` with padding, where the border outputs
/// read zeros for some taps and no bias can stand for the channel's mean. Runs after
/// `check_layout`, which found the receiving layer of each listed run.
fn check_fold_biases(model: &Sequential, targets: &[Target]) -> Result<(), Error> {
    for &target in targets {
        let (layer, kind) = (target.layer, target.kind());
        let next_at = target.end(model) + 1;
        if let Some(conv) = model.layer(next_at).as_conv2d()
            && conv.geom().padding() != (0, 0)
        {
            return Err(Error::InvalidData(format!(
                "shrink: fold_means would fold the means of the conv layer {layer}'s removed \
                 channels into the bias of the conv layer {next_at}, whose padding {:?} makes \
                 its border outputs read zeros where the channel was, so no bias stands for the \
                 channel's mean; turn fold_means off, or leave layer {layer} out of \
                 ShrinkConfig::layers",
                conv.geom().padding()
            )));
        }
        if receiver_bias(model, target).is_none() {
            let receiver = if model.layer(next_at).as_conv2d().is_some() {
                "conv"
            } else {
                "linear"
            };
            return Err(Error::InvalidData(format!(
                "shrink: fold_means folds the means of the {kind} layer {layer}'s removed units \
                 into the bias of the {receiver} layer {next_at}, which has no bias; turn \
                 fold_means off, or give that layer a bias"
            )));
        }
    }
    Ok(())
}

/// The criterion to use: the one given (its number checked), or `AccuracyDrop` of 1 point
/// for class targets; an error for value targets with none, or with an accuracy criterion.
fn criterion(given: Option<Criterion>, targets: &Targets) -> Result<Criterion, Error> {
    let classes = matches!(targets, Targets::Classes(_));
    match given {
        None if classes => Ok(Criterion::AccuracyDrop { points: 1.0 }),
        None => Err(Error::InvalidData(
            "shrink: ShrinkConfig::criterion is None, and value targets have no default: \
             set it, for example to Criterion::LossIncrease { relative: 0.05 } for a \
             validation loss at most 5% above the unpruned network's"
                .to_string(),
        )),
        Some(Criterion::AccuracyDrop { .. }) if !classes => Err(Error::InvalidData(
            "shrink: Criterion::AccuracyDrop needs class targets, but the data has value \
             targets; use Criterion::LossIncrease"
                .to_string(),
        )),
        Some(Criterion::AccuracyDrop { points: v } | Criterion::LossIncrease { relative: v })
            if v.is_nan() =>
        {
            Err(Error::InvalidData(
                "shrink: the criterion's threshold is NaN".to_string(),
            ))
        }
        Some(c) => Ok(c),
    }
}

/// Checks the levels, the patience and the retraining batch.
fn check_numbers(config: &ShrinkConfig<'_>) -> Result<(), Error> {
    let bad = |what: String| Err(Error::InvalidData(format!("shrink: {what}")));
    if config.step_bp == 0 || config.fine_step_bp == 0 {
        return bad(format!(
            "step_bp ({}) and fine_step_bp ({}) must be at least 1",
            config.step_bp, config.fine_step_bp
        ));
    }
    if config.max_removed_bp == 0 || config.max_removed_bp > FULL_BP {
        return bad(format!(
            "max_removed_bp is {}, but must be from 1 to 10,000 (the whole layer)",
            config.max_removed_bp
        ));
    }
    if config.patience == Some(0) {
        return bad("patience is Some(0); use Some(1) to stop at the first miss".to_string());
    }
    if config.retrain.batch == 0 {
        return bad("the retraining batch size is 0".to_string());
    }
    Ok(())
}

/// The levels of a run, in basis points: `step` increments below `fine_from` (and below
/// `max`), then `fine_from` and `fine_step` increments below `max`, then `max`. Each level
/// is above the one before.
fn levels(step: u32, fine_from: u32, fine_step: u32, max: u32) -> Vec<u32> {
    let mut levels = Vec::new();
    let mut level = step;
    while level < fine_from.min(max) {
        levels.push(level);
        level = level.saturating_add(step);
    }
    let mut level = fine_from;
    while level < max {
        if level > 0 && levels.last().is_none_or(|&last| level > last) {
            levels.push(level);
        }
        level = level.saturating_add(fine_step);
    }
    levels.push(max);
    levels
}

/// The neurons a layer of `width` keeps at `level` basis points removed:
/// `max(1, ceil(width · (10,000 − level) / 10,000))`, in integers.
fn kept_at(width: usize, level: u32) -> usize {
    let full = u128::from(FULL_BP);
    let left = width as u128 * (full - u128::from(level.min(FULL_BP)));
    (left.div_ceil(full) as usize).max(1)
}

/// The parameters `model` would have with its off neurons removed, as
/// [`compact`](super::compact()) removes them: each `Linear` keeps the rows of its on
/// neurons and the columns of the on neurons of the `Linear` before it, a `Prelu` with one
/// slope per neuron keeps the slopes of the on ones, and every other parameter counts in
/// full. For a model with image layers that `compact` accepts, the count is that of the
/// network `compact` builds, which also leaves out the off channels; when it refuses, the
/// count above.
fn parameter_count(model: &Sequential, params: &ParamStore) -> usize {
    if has_image_layers(model)
        && let Ok((small, store)) = compact(model, params)
    {
        return (0..small.len())
            .flat_map(|i| small.layer(i).parameters())
            .map(|(_, id)| store.value(id).data().len())
            .sum();
    }
    // For each layer in the run of a `Linear`, that `Linear`'s width and on count.
    let mut runs: Vec<Option<(usize, usize)>> = vec![None; model.len()];
    for (linear, layer) in model.linears() {
        let on = neurons_on(model, linear).iter().filter(|&&on| on).count();
        for run in &mut runs[linear..=model.neuron_layer(linear)] {
            *run = Some((layer.outputs(), on));
        }
    }
    let mut total = 0;
    for i in 0..model.len() {
        let layer = model.layer(i);
        if let Some(linear) = layer.as_linear() {
            let rows = runs[i].map_or(linear.outputs(), |(_, on)| on);
            let cols = match i.checked_sub(1).and_then(|p| runs[p]) {
                Some((width, on)) if width == linear.inputs() => on,
                _ => linear.inputs(),
            };
            total += rows * cols + rows;
        } else if let Some(prelu) = layer.as_prelu() {
            let slopes = params.value(prelu.weight()).data().len();
            total += match runs[i] {
                Some((width, on)) if slopes == width && slopes > 1 => on,
                _ => slopes,
            };
        } else {
            for (_, id) in layer.parameters() {
                total += params.value(id).data().len();
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    //! Unit tests of the levels, the kept counts and the criterion's boundary, and of the
    //! stop rule on a scripted validation measurement.

    use super::*;
    use crate::{Linear, Relu, Targets};
    use cavitation_core::Tensor;

    /// An evaluation with the given loss and accuracy.
    fn eval(loss: f64, accuracy: Option<f64>) -> Evaluation {
        Evaluation {
            loss,
            accuracy,
            r2: None,
        }
    }

    /// The default schedule is 1,000 to 7,000 by 1,000 and 7,500 to 9,500 by 500; other
    /// settings keep each level above the last and end at `max`.
    #[test]
    fn levels_of_the_schedule() {
        assert_eq!(
            levels(1000, 7000, 500, 9500),
            [
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 7500, 8000, 8500, 9000, 9500
            ]
        );
        assert_eq!(
            levels(3000, 7000, 500, 8000),
            [3000, 6000, 7000, 7500, 8000]
        );
        assert_eq!(levels(1000, 7000, 500, 2500), [1000, 2000, 2500]);
        assert_eq!(levels(1000, 0, 2000, 5000), [2000, 4000, 5000]);
        assert_eq!(levels(1000, 7000, 500, 9990), {
            let mut l = levels(1000, 7000, 500, 9500);
            l.push(9990);
            l
        });
        assert_eq!(levels(u32::MAX, 7000, 500, 7200), [7000, 7200]);
    }

    /// The kept counts round up and never reach 0.
    #[test]
    fn kept_rounds_up_to_at_least_one() {
        assert_eq!(kept_at(256, 1000), 231);
        assert_eq!(kept_at(256, 7500), 64);
        assert_eq!(kept_at(8, 9990), 1);
        assert_eq!(kept_at(8, 10_000), 1);
        assert_eq!(kept_at(10, 0), 10);
    }

    /// One row lost in 100 is exactly 1 point and meets `points: 1.0`; two rows do not. A
    /// NaN loss never meets a loss criterion, even an infinite one.
    #[test]
    fn accuracy_drop_boundary() {
        let one = Criterion::AccuracyDrop { points: 1.0 };
        let base = eval(0.1, Some(0.97));
        assert!(one.met(&base, &eval(0.2, Some(0.96)), 100));
        assert!(!one.met(&base, &eval(0.2, Some(0.95)), 100));
        assert!(one.met(&base, &eval(0.2, Some(0.98)), 100));
        let loss = Criterion::LossIncrease { relative: 0.5 };
        assert!(loss.met(&base, &eval(0.15, None), 100));
        assert!(!loss.met(&base, &eval(0.150_001, None), 100));
        let any = Criterion::LossIncrease {
            relative: f64::INFINITY,
        };
        assert!(!any.met(&base, &eval(f64::NAN, None), 100));
        // A zero baseline allows no rise, but a loss at or below 0 meets it, for any
        // `relative`.
        let zero = eval(0.0, None);
        for c in [loss, any] {
            assert!(c.met(&zero, &eval(-0.1, None), 100));
            assert!(c.met(&zero, &eval(0.0, None), 100));
            assert!(!c.met(&zero, &eval(0.1, None), 100));
        }
    }

    /// `Carve` reads the share as its decimal: 0.07 of 100 rows is 7, though
    /// `0.07 * 100.0` is `7.000000000000001`; other shares round up.
    #[test]
    fn carved_rows_are_exact() {
        assert_eq!(0.07 * 100.0, 7.000000000000001);
        assert_eq!(carved_rows(0.07, 100), 7);
        assert_eq!(carved_rows(0.255, 100), 26);
        assert_eq!(carved_rows(0.2, 48), 10);
        assert_eq!(carved_rows(0.5, 3), 2);
        assert_eq!(carved_rows(0.1, 1_000_000), 100_000);
        assert_eq!(carved_rows(1e-40, 10), 1);
    }

    /// With validation accuracies scripted as met, met, miss and `Flow::Stop` at the miss,
    /// the run ends there and keeps step 2: the model holds step 2's masks and store, not
    /// those of the step it stopped at.
    #[test]
    fn stop_at_a_miss_keeps_the_earlier_best() {
        let mut params = ParamStore::new();
        let (mut net, train) = small(&mut params);
        let validation = Dataset {
            x: Tensor::zeros(&[100, 4]),
            targets: Targets::Classes(vec![0; 100]),
        };
        let mut script = vec![0.90, 0.90, 0.90, 0.50].into_iter();
        let mut measure = |_: &Sequential, _: &ParamStore, _: &Dataset| {
            Ok(eval(
                1.0,
                Some(script.next().expect("no more than 3 steps")),
            ))
        };
        let config = ShrinkConfig {
            validation: Some(Validation::Given(validation)),
            keep_checkpoints: true,
            retrain: RetrainConfig {
                epochs: 1,
                ..RetrainConfig::default()
            },
            on_step: Some(Box::new(
                |step: &ShrinkStep| {
                    if step.met { Flow::Continue } else { Flow::Stop }
                },
            )),
            ..ShrinkConfig::default()
        };
        let report = run(&mut net, &mut params, &train, config, &mut measure).unwrap();
        assert_eq!(report.steps.len(), 3);
        assert_eq!(report.chosen, Some(1));
        let (masks, store) = report.steps[1].checkpoint.as_ref().unwrap();
        let (last_masks, last_store) = report.steps[2].checkpoint.as_ref().unwrap();
        assert_ne!(masks, last_masks);
        assert_eq!(net.neuron_mask(0), Some(&masks[0].1));
        let w = net.linear(0).unwrap().weight();
        assert_eq!(params.value(w).data(), store.value(w).data());
        assert_ne!(params.value(w).data(), last_store.value(w).data());
    }

    /// A 4 → 8 → ReLU → 3 network and 30 rows of seeded class data.
    fn small(params: &mut ParamStore) -> (Sequential, Dataset) {
        let mut rng = Rng::seed(4);
        let net = Sequential::new()
            .push(Linear::new(params, 4, 8, &mut rng))
            .push(Relu)
            .push(Linear::new(params, 8, 3, &mut rng));
        let x: Vec<f32> = (0..120).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        let data = Dataset {
            x: Tensor::from_vec(x, &[30, 4]).unwrap(),
            targets: Targets::Classes((0..30).map(|r| r % 3).collect()),
        };
        (net, data)
    }

    /// With validation accuracies scripted as met, miss, met, miss, miss (100 rows, 1
    /// point allowed), a custom-ranked run keeps step 3 (the latest met, not the one before
    /// the first miss) and ends after step 5, its second miss in a row; the model is left
    /// with step 3's masks.
    #[test]
    fn best_so_far_not_first_miss() {
        let mut params = ParamStore::new();
        let (mut net, train) = small(&mut params);
        let validation = Dataset {
            x: Tensor::zeros(&[100, 4]),
            targets: Targets::Classes(vec![0; 100]),
        };
        let mut script = vec![0.90, 0.90, 0.50, 0.89, 0.50, 0.50].into_iter();
        let mut measure = |_: &Sequential, _: &ParamStore, data: &Dataset| {
            assert_eq!(data.len(), 100);
            Ok(eval(
                1.0,
                Some(script.next().expect("no more than 5 steps")),
            ))
        };
        let mut ranked = 0;
        let config = ShrinkConfig {
            validation: Some(Validation::Given(validation)),
            ranking: Ranking::Custom(Box::new(|ctx: &RankContext<'_>| {
                ranked += 1;
                ctx.on.iter().map(|&j| -(j as f64)).collect()
            })),
            dead_first: false,
            keep_checkpoints: true,
            retrain: RetrainConfig {
                epochs: 1,
                ..RetrainConfig::default()
            },
            ..ShrinkConfig::default()
        };
        let report = run(&mut net, &mut params, &train, config, &mut measure).unwrap();
        // At 1,000 bp all 8 neurons stay (ceil(7.2) = 8), so nothing is ranked; steps 2
        // to 5 each rank once.
        assert_eq!(ranked, 4);
        let met: Vec<bool> = report.steps.iter().map(|s| s.met).collect();
        assert_eq!(met, [true, false, true, false, false]);
        assert_eq!(report.chosen, Some(2));
        let (masks, store) = report.steps[2].checkpoint.as_ref().unwrap();
        assert_eq!(net.neuron_mask(0), Some(&masks[0].1));
        let w = net.linear(0).unwrap().weight();
        assert_eq!(params.value(w).data(), store.value(w).data());
        // The custom scores put the highest indices first: at 3,000 bp, 8 → 6 kept, and
        // neurons 6 and 7 are off.
        assert_eq!(masks[0].1.off().collect::<Vec<_>>(), [6, 7]);
    }

    /// A custom ranking that returns the wrong number of scores is `InvalidData`, and the
    /// model and parameters are put back as they were.
    #[test]
    fn custom_ranking_of_the_wrong_length_is_refused() {
        let mut params = ParamStore::new();
        let (mut net, train) = small(&mut params);
        let before = params.clone();
        let config = ShrinkConfig {
            validation: Some(Validation::Given(train.clone())),
            ranking: Ranking::Custom(Box::new(|_: &RankContext<'_>| vec![0.0])),
            ..ShrinkConfig::default()
        };
        let err = shrink(&mut net, &mut params, &train, config).unwrap_err();
        assert!(err.to_string().contains("RankContext::on"), "{err}");
        assert!(net.neuron_mask(0).is_none());
        let w = net.linear(0).unwrap().weight();
        assert_eq!(params.value(w).data(), before.value(w).data());
    }

    /// The config's `Debug` shows a custom ranking and a callback without their contents.
    #[test]
    fn config_debug_names_the_callbacks() {
        let config = ShrinkConfig {
            ranking: Ranking::Custom(Box::new(|_: &RankContext<'_>| Vec::new())),
            on_step: Some(Box::new(|_: &ShrinkStep| Flow::Continue)),
            ..ShrinkConfig::default()
        };
        let text = format!("{config:?}");
        assert!(text.contains("Custom(FnMut)") && text.contains("on_step: Some(\"FnMut\")"));
    }
}
