//! The shrink tool: [`shrink`] removes the neurons of a trained network's hidden `Linear`
//! layers a step at a time, retrains after each step, and keeps the smallest network that
//! still meets a criterion measured on a validation set.

use std::borrow::Cow;
use std::fmt;

use cavitation_core::{Error, ParamStore, Rng};

use super::ablate::{check_hidden, points_lost, relative_increase};
use super::{LayerActivity, activity, neurons_on};
use crate::train::check_dataset;
use crate::{
    Dataset, Evaluation, FitConfig, Flow, Health, Linear, Loss, LrSchedule, NeuronMask,
    OptimizerConfig, Sequential, Targets, evaluate, fit,
};

/// The rows run at a time when measuring activity and evaluating on the validation set.
const EVAL_BATCH: usize = 256;

/// The whole network in basis points: 10,000 bp is 100%.
const FULL_BP: u32 = 10_000;

/// How [`shrink`] orders the neurons of a layer for removal: each neuron gets a score, and
/// the lowest scores go first (ties by neuron index, lowest first; a NaN score sorts above
/// every number, so it goes last). Every ranking is computed again at each step, on the
/// network as the previous step's retraining left it.
///
/// ```
/// use cavitation_nn::shrink::{RankContext, Ranking, ShrinkConfig};
/// // Score each neuron by its mean |activation| times its index (an example only).
/// let custom = Ranking::Custom(Box::new(|ctx: &RankContext<'_>| {
///     ctx.on.iter().map(|&j| ctx.activity.mean_abs[j] * j as f64).collect()
/// }));
/// let config = ShrinkConfig { ranking: custom, ..ShrinkConfig::default() };
/// assert!(matches!(config.ranking, Ranking::Custom(_)));
/// assert!(matches!(ShrinkConfig::default().ranking, Ranking::OutNorm));
/// ```
pub enum Ranking {
    /// The Euclidean norm of the neuron's outgoing weights: its column in the weight of the
    /// `Linear` that follows its layer. A neuron whose output the next layer barely uses
    /// goes first. This needs a `Linear` right after the neurons' run of element-wise
    /// layers.
    OutNorm,
    /// The neuron's mean absolute output on the training rows
    /// ([`LayerActivity::mean_abs`]).
    Activity,
    /// A random order from this seed: the same seed removes the same neurons on every run.
    Random(u64),
    /// Your own scores: the function gets a [`RankContext`] for one layer and returns one
    /// score for each entry of [`RankContext::on`], in that order. It is called only for
    /// a layer that loses neurons at the step.
    // The boxed callback is the plainest type for this variant; an alias would only hide it.
    #[allow(clippy::type_complexity)]
    Custom(Box<dyn FnMut(&RankContext<'_>) -> Vec<f64>>),
}

impl fmt::Debug for Ranking {
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
/// is (after the previous step's retraining, before this step removes anything) and its
/// activity on the training rows.
///
/// It is `#[non_exhaustive]`, so a later release can add a field without breaking your
/// code: read its fields in your ranking function.
///
/// ```
/// use cavitation_nn::shrink::{RankContext, Ranking};
/// // Rank by firing share, and break ties by the outgoing weights' sum of squares.
/// fn by_firing(ctx: &RankContext<'_>) -> Vec<f64> {
///     let next = ctx.model.linear(ctx.model.neuron_layer(ctx.linear) + 1).unwrap();
///     let w = ctx.params.value(next.weight()).data();
///     let width = next.inputs();
///     ctx.on
///         .iter()
///         .map(|&j| {
///             let norm: f64 = w.iter().skip(j).step_by(width).map(|&v| f64::from(v * v)).sum();
///             ctx.activity.firing[j] + 1e-6 * norm
///         })
///         .collect()
/// }
/// let ranking = Ranking::Custom(Box::new(by_firing));
/// assert!(matches!(ranking, Ranking::Custom(_)));
/// ```
#[non_exhaustive]
pub struct RankContext<'a> {
    /// The index of the `Linear` layer whose neurons are ranked.
    pub linear: usize,
    /// The neurons of that layer that are on, in increasing order: the ones to score.
    pub on: &'a [usize],
    /// The network, with its masks as they are.
    pub model: &'a Sequential,
    /// The network's parameters.
    pub params: &'a ParamStore,
    /// The layer's activity on the training rows, one value per neuron of the layer (the
    /// neurons that are off read 0).
    pub activity: &'a LayerActivity,
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
                !now.loss.is_nan() && relative_increase(base.loss, now.loss) <= relative
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
pub enum Validation {
    /// These rows, which the model must not have seen in training.
    Given(Dataset),
    /// The last `ceil(share · n)` of the `n` training rows, held out of retraining (which
    /// runs on the rest). The model must not have seen those rows: carve the same rows off
    /// before the first training, or the baseline and every step are measured on rows the
    /// network learned, and the stop rule cannot tell what pruning cost. `share` is above
    /// 0 and below 1, and both parts keep at least one row.
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
/// basis points of each layer's width: 1 bp is 0.01%, 10,000 bp the whole layer.
///
/// The default shrinks every hidden `Linear`, removes neurons that never fire first and
/// then the lowest out-norms, in steps of 1,000 bp up to 7,000 and of 500 bp from there to
/// 9,500, retrains as [`RetrainConfig::default`], keeps the latest step that loses at most
/// 1 accuracy point (class targets), and stops after 2 misses in a row. It has no
/// validation set: set [`validation`](Self::validation).
///
/// ```
/// use cavitation_nn::shrink::{Ranking, ShrinkConfig};
/// let config = ShrinkConfig::default();
/// assert!(config.linears.is_empty() && config.dead_first);
/// assert!(matches!(config.ranking, Ranking::OutNorm));
/// assert_eq!(
///     (config.step_bp, config.fine_from_bp, config.fine_step_bp, config.max_removed_bp),
///     (1000, 7000, 500, 9500)
/// );
/// assert_eq!(config.patience, Some(2));
/// assert!(config.criterion.is_none() && config.validation.is_none());
/// assert!(!config.keep_checkpoints && config.on_step.is_none());
/// ```
pub struct ShrinkConfig {
    /// The `Linear` layers to shrink, by layer index; empty (the default) shrinks every
    /// hidden `Linear` (each one with another `Linear` after it). A layer listed twice is
    /// shrunk once.
    pub linears: Vec<usize>,
    /// The order in which each layer's neurons are removed.
    pub ranking: Ranking,
    /// Whether neurons whose output is 0 on every training row (for a ReLU, neurons that
    /// never fire) go before the ranking's order; removing them changes nothing. On by
    /// default.
    pub dead_first: bool,
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
    pub on_step: Option<Box<dyn FnMut(&ShrinkStep) -> Flow>>,
}

impl Default for ShrinkConfig {
    /// Every hidden `Linear`, dead neurons first and then out-norm, levels 1,000 to 7,000
    /// bp by 1,000 and on to 9,500 by 500, patience 2, the default criterion, no validation
    /// set, the default retraining, no checkpoints, no callback.
    fn default() -> ShrinkConfig {
        ShrinkConfig {
            linears: Vec::new(),
            ranking: Ranking::OutNorm,
            dead_first: true,
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

impl fmt::Debug for ShrinkConfig {
    /// Writes every setting; the callback shows only whether there is one.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShrinkConfig")
            .field("linears", &self.linears)
            .field("ranking", &self.ranking)
            .field("dead_first", &self.dead_first)
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
///         println!("{} bp: kept {:?}, {} parameters", step.removed_bp, step.kept, step.parameters);
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
    /// The neurons left on in each shrunk `Linear`, in the order of
    /// [`ShrinkConfig::linears`] (or of the model's hidden `Linear`s when that is empty).
    pub kept: Vec<usize>,
    /// The number of parameters the network has once its off neurons are removed for good
    /// with [`compact`](super::compact()).
    pub parameters: usize,
    /// The retrained network's measures on the validation set.
    pub validation: Evaluation,
    /// Whether the step meets the criterion.
    pub met: bool,
    /// With [`ShrinkConfig::keep_checkpoints`], the step's network: each shrunk `Linear`'s
    /// neuron mask (by layer index) and the parameters after retraining.
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
#[derive(Clone, Copy, Debug, PartialEq)]
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
}

impl fmt::Display for ShrinkWarning {
    /// A sentence saying what was found and how many rows would be enough.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
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

/// Shrinks the hidden `Linear` layers of `model`: removes neurons a step at a time, by
/// switching them off with neuron masks, retrains after each step, measures the network
/// on a validation set, and keeps the latest step whose measures still meet the criterion.
///
/// The levels are `step_bp, 2·step_bp, …` below `fine_from_bp`, then `fine_from_bp` and
/// `fine_step_bp` increments below `max_removed_bp`, and `max_removed_bp` last (the
/// default: 1,000 to 7,000 by 1,000, then 7,500 to 9,500 by 500). At each level each
/// listed `Linear` of width `w` keeps `max(1, ceil(w · (10,000 − level) / 10,000))` neurons
/// (neurons already off count as removed). Each step:
///
/// 1. measures every listed layer's activity on the training rows
///    ([`activity()`](super::activity())) and ranks its neurons that are on, on the
///    network as the previous step left it;
/// 2. switches off, in each layer, the neurons it must lose: with
///    [`dead_first`](ShrinkConfig::dead_first), first those whose output is 0 on every
///    training row, then the lowest-ranked;
/// 3. retrains the network on the training rows as [`ShrinkConfig::retrain`] says, with a
///    fresh optimiser (masked neurons pass no gradient, so they stay off);
/// 4. measures it on the validation set, records the step, and calls `on_step`
///    ([`Flow::Stop`] ends the run).
///
/// The run also ends after [`patience`](ShrinkConfig::patience) misses in a row, or after
/// the last level. The step kept is the latest one that met the criterion, not the one
/// before the first miss: a later step can recover what an earlier one lost. On return
/// `model` and `params` hold the kept step's network: its neuron masks on the listed
/// `Linear`s and its retrained parameters. If no step met the criterion they are as they
/// were given. Call [`compact`](super::compact()) on the result for the smaller dense
/// network that runs faster.
///
/// What pruning mostly buys is the size. On a 64 → 256 → 128 → 10 network on the digits
/// data, networks pruned and retrained this way were about as accurate as networks of the
/// same size trained from scratch at 75–90% removed, and worse at 95%. Treat the size the
/// run finds as the answer, and compare with a network of that size trained from scratch.
///
/// Neurons the model already has off in a neuron mask stay off. The masks the run sets
/// are neuron masks, by `Linear` ([`Sequential::set_neuron_mask`]); masks on other layers
/// are left as they are. The baseline and every step are measured with the loss in
/// [`RetrainConfig::loss`], 256 rows at a time; a warning is added when one validation row
/// is worth more than half of an [`AccuracyDrop`](Criterion::AccuracyDrop)'s points.
///
/// # Errors
///
/// Returns [`Error::InvalidData`], before changing anything, if
/// [`ShrinkConfig::validation`] is `None`; if the training data or a given validation set
/// is not a matrix of inputs with at least one row and one target per row (the message
/// says which); if a [`Validation::Carve`] share is not above 0
/// and below 1 or leaves either part without rows; if the model has no hidden `Linear`
/// (no `Linear` with another `Linear` after it) or a listed layer is not one; if
/// [`ShrinkConfig::criterion`] is `None` for value targets, or an `AccuracyDrop` is given
/// for value targets, or the criterion's number is NaN; if `step_bp` or `fine_step_bp` is
/// 0, `max_removed_bp` is 0 or above 10,000, `patience` is `Some(0)` or the retraining
/// batch is 0; if a listed layer's neuron mask does not have its width; and as
/// [`evaluate`] does for the validation set. Returns [`Error::Unsupported`] if a listed
/// `Linear`, or a layer between it and the layer that holds its neurons, carries a mask
/// set with [`Sequential::set_mask`] (clear it and mask by `Linear` instead), and, for
/// [`Ranking::OutNorm`], if the layer after a listed `Linear`'s neurons is not a `Linear`
/// that takes them. During the run, returns the errors of [`activity()`](super::activity())
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
/// let data = Dataset { x: Tensor::from_vec(x, &[200, 2]).unwrap(), targets: Targets::Classes(labels) };
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
    config: ShrinkConfig,
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
    mut config: ShrinkConfig,
    measure: &mut Measure<'_>,
) -> Result<ShrinkReport, Error> {
    let (train, validation) = split(train, config.validation.take())?;
    let linears = hidden_linears(model, &config.linears)?;
    check_masks(model, &linears)?;
    if matches!(config.ranking, Ranking::OutNorm) {
        for &linear in &linears {
            out_linear(model, linear)?;
        }
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
    let mut warnings = Vec::new();
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
    let original_masks: Vec<Option<NeuronMask>> = linears
        .iter()
        .map(|&l| model.neuron_mask(l).cloned())
        .collect();
    let original_params = params.clone();
    let mut run = Run {
        model,
        params,
        train: &train,
        validation: &validation,
        linears: &linears,
        criterion,
        baseline,
        config: &mut config,
        measure,
    };
    let outcome = run.steps(&levels);
    let (model, params) = (run.model, run.params);
    let restore = |model: &mut Sequential, params: &mut ParamStore| {
        for (&linear, mask) in linears.iter().zip(&original_masks) {
            match mask {
                Some(mask) => model.set_neuron_mask(linear, mask.clone()),
                None => model.clear_neuron_mask(linear),
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
            for (linear, mask) in masks {
                model.set_neuron_mask(linear, mask);
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

/// The latest step that met the criterion: its index, masks and parameters.
struct Best {
    /// The step's index in the report.
    step: usize,
    /// Each shrunk `Linear`'s neuron mask.
    masks: Vec<(usize, NeuronMask)>,
    /// The parameters after the step's retraining.
    store: ParamStore,
}

/// The state of one [`run`]: the network being shrunk and everything each step reads.
struct Run<'r, 'm> {
    /// The network, masked step by step.
    model: &'r mut Sequential,
    /// Its parameters, retrained step by step.
    params: &'r mut ParamStore,
    /// The rows to measure activity and retrain on.
    train: &'r Dataset,
    /// The rows to measure each step on.
    validation: &'r Dataset,
    /// The `Linear` layers to shrink.
    linears: &'r [usize],
    /// When a step is good enough.
    criterion: Criterion,
    /// The unpruned network's measures on the validation set.
    baseline: Evaluation,
    /// The run's settings (the ranking and the callback are used mutably).
    config: &'r mut ShrinkConfig,
    /// The validation measurement.
    measure: &'r mut Measure<'m>,
}

impl Run<'_, '_> {
    /// Runs the steps at `levels` until the last, `Flow::Stop` or the patience runs out,
    /// and returns them with the latest step that met the criterion.
    fn steps(&mut self, levels: &[u32]) -> Result<(Vec<ShrinkStep>, Option<Best>), Error> {
        let widths: Vec<usize> = self
            .linears
            .iter()
            .map(|&l| self.model.linear(l).expect("a hidden Linear").outputs())
            .collect();
        let mut rng = Rng::seed(match self.config.ranking {
            Ranking::Random(seed) => seed,
            _ => 0,
        });
        let mut steps: Vec<ShrinkStep> = Vec::with_capacity(levels.len());
        let mut best = None;
        let mut misses = 0;
        for &level in levels {
            let stats = activity(
                self.model,
                self.params,
                self.train,
                self.linears,
                EVAL_BATCH,
            )?;
            let mut masks = Vec::with_capacity(self.linears.len());
            for (k, &linear) in self.linears.iter().enumerate() {
                let keep = kept_at(widths[k], level);
                masks.push((linear, self.mask_for(linear, keep, &stats[k], &mut rng)?));
            }
            for (linear, mask) in &masks {
                self.model.set_neuron_mask(*linear, mask.clone());
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

    /// The neuron mask of the `Linear` at `linear` that keeps `keep` of its neurons (or
    /// every neuron on, if no more than `keep` are on): its on neurons ranked, dead ones
    /// first with `dead_first`, and the first ones in that order switched off.
    fn mask_for(
        &mut self,
        linear: usize,
        keep: usize,
        stats: &LayerActivity,
        rng: &mut Rng,
    ) -> Result<NeuronMask, Error> {
        let flags = neurons_on(self.model, linear);
        let on: Vec<usize> = (0..flags.len()).filter(|&j| flags[j]).collect();
        let mut mask = NeuronMask::from_on(flags.len(), &on);
        if on.len() <= keep {
            return Ok(mask);
        }
        let scores = self.scores(linear, &on, stats, rng)?;
        let dead_first = self.config.dead_first;
        let dead = |i: usize| dead_first && stats.mean_abs[on[i]] == 0.0;
        let mut order: Vec<usize> = (0..on.len()).collect();
        order.sort_by(|&a, &b| {
            dead(b)
                .cmp(&dead(a))
                .then(scores[a].total_cmp(&scores[b]))
                .then(on[a].cmp(&on[b]))
        });
        for &i in &order[..on.len() - keep] {
            mask.set(on[i], false);
        }
        Ok(mask)
    }

    /// The ranking's score of each neuron in `on` of the `Linear` at `linear`.
    fn scores(
        &mut self,
        linear: usize,
        on: &[usize],
        stats: &LayerActivity,
        rng: &mut Rng,
    ) -> Result<Vec<f64>, Error> {
        let scores = match &mut self.config.ranking {
            Ranking::OutNorm => {
                let next = out_linear(self.model, linear)?;
                let (w, inputs) = (self.params.value(next.weight()).data(), next.inputs());
                on.iter()
                    .map(|&j| {
                        let sum: f64 = w
                            .iter()
                            .skip(j)
                            .step_by(inputs)
                            .map(|&v| f64::from(v) * f64::from(v))
                            .sum();
                        sum.sqrt()
                    })
                    .collect()
            }
            Ranking::Activity => on.iter().map(|&j| stats.mean_abs[j]).collect(),
            Ranking::Random(_) => on.iter().map(|_| rng.uniform_f64()).collect(),
            Ranking::Custom(rank) => {
                let scores = rank(&RankContext {
                    linear,
                    on,
                    model: self.model,
                    params: self.params,
                    activity: stats,
                });
                if scores.len() != on.len() {
                    return Err(Error::InvalidData(format!(
                        "shrink: the custom ranking returned {} scores for the linear layer \
                         {linear}, which has {} neurons on; return one score per entry of \
                         RankContext::on",
                        scores.len(),
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
            schedule: LrSchedule::Constant,
            health: Health::quiet(),
            on_epoch: None,
        };
        fit(self.model, self.params, self.train, config).map(|_| ())
    }
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
            let held = (share * n as f64).ceil() as usize;
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

/// The `Linear` layers to shrink: `listed` without repeats, each checked to be a hidden
/// `Linear`, or every hidden `Linear` of `model` when `listed` is empty.
fn hidden_linears(model: &Sequential, listed: &[usize]) -> Result<Vec<usize>, Error> {
    if listed.is_empty() {
        let all: Vec<usize> = model
            .linears()
            .map(|(i, _)| i)
            .filter(|&i| (i + 1..model.len()).any(|l| model.linear(l).is_some()))
            .collect();
        if all.is_empty() {
            return Err(Error::InvalidData(
                "shrink: the model has no hidden layer to shrink: a hidden layer is a Linear \
                 with another Linear after it, and the last Linear's neurons are the model's \
                 outputs"
                    .to_string(),
            ));
        }
        return Ok(all);
    }
    let mut chosen = Vec::with_capacity(listed.len());
    for &linear in listed {
        check_hidden(model, linear, "shrink")?;
        if !chosen.contains(&linear) {
            chosen.push(linear);
        }
    }
    Ok(chosen)
}

/// Refuses a mask the run cannot work with on a listed `Linear`'s run: one set with
/// `set_mask` on the `Linear` or a layer before the one that holds its neurons
/// (`Unsupported`), or a neuron mask of the wrong width (`InvalidData`).
fn check_masks(model: &Sequential, linears: &[usize]) -> Result<(), Error> {
    for &linear in linears {
        let end = model.neuron_layer(linear);
        for layer in linear..end {
            if model.mask(layer).is_some() {
                return Err(Error::Unsupported(format!(
                    "shrink: layer {layer} ({}) carries a mask set with set_mask, inside the \
                     run of the linear layer {linear}, whose neurons the tool masks at layer \
                     {end}; clear it with clear_mask({layer}) and mask those neurons with \
                     set_neuron_mask({linear}, mask) instead",
                    model.layer(layer).name()
                )));
            }
        }
        let width = model.linear(linear).expect("a hidden Linear").outputs();
        if let Some(mask) = model.mask(end)
            && mask.width() != width
        {
            return Err(Error::InvalidData(format!(
                "shrink: the mask on layer {end} has width {}, but the linear layer {linear} \
                 outputs width {width}",
                mask.width()
            )));
        }
    }
    Ok(())
}

/// The `Linear` right after the neurons of the `Linear` at `linear`, which takes them,
/// for [`Ranking::OutNorm`]; `Unsupported` if there is none.
fn out_linear(model: &Sequential, linear: usize) -> Result<&Linear, Error> {
    let next = model.neuron_layer(linear) + 1;
    let width = model.linear(linear).expect("a hidden Linear").outputs();
    match model.linear(next) {
        Some(out) if out.inputs() == width => Ok(out),
        _ => Err(Error::Unsupported(format!(
            "shrink: Ranking::OutNorm reads each neuron's column in the weight of the Linear \
             that takes it, but layer {next} ({}), after the neurons of the linear layer \
             {linear}, is not a Linear that takes {width} inputs; choose Ranking::Activity, \
             Ranking::Random or Ranking::Custom",
            model.layer(next).name()
        ))),
    }
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
fn check_numbers(config: &ShrinkConfig) -> Result<(), Error> {
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
/// full.
fn parameter_count(model: &Sequential, params: &ParamStore) -> usize {
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
    use crate::{Relu, Targets};
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
        let ranked = std::rc::Rc::new(std::cell::Cell::new(0));
        let count = ranked.clone();
        let config = ShrinkConfig {
            validation: Some(Validation::Given(validation)),
            ranking: Ranking::Custom(Box::new(move |ctx: &RankContext<'_>| {
                count.set(count.get() + 1);
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
        assert_eq!(ranked.get(), 4);
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
