//! Health checks during training: non-finite values, exploding and vanishing gradient norms,
//! clamped binary cross-entropy logarithms, impossible negative losses and dead ReLU neurons.

use std::fmt;

use cavitation_core::{Error, ParamId, ParamStore, Tape, TensorView};

use crate::activation::last_dim;
use crate::{ActivationKind, GradStats, Loss, Sequential, TargetsView, Trace};

/// What the health checks do when the loss or a gradient is NaN or infinite.
///
/// The enum is `#[non_exhaustive]`: a `match` on it outside this crate needs a wildcard
/// arm, so that a later policy is not a breaking change.
///
/// # Examples
///
/// ```
/// use cavitation_nn::{Health, OnNonFinite};
/// assert_eq!(Health::default().on_non_finite, OnNonFinite::Stop);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OnNonFinite {
    /// Stop training with [`Error::NonFinite`], naming the step and where the value was,
    /// after sending the matching [`Warning::NonFinite`] to the sink. A NaN or infinite loss
    /// stops before the optimiser runs, so the parameters keep the last good step's values.
    /// A NaN or infinite gradient with a finite loss is found in the optimiser's own
    /// statistics, after the update has applied it: the parameters (and the optimiser's
    /// momentum or moments) then already hold the NaN when the error comes back. Finding it
    /// sooner would mean reading every gradient once more on every step; to keep the last
    /// good values on any bad step, use [`Skip`](OnNonFinite::Skip), which pays that read.
    Stop,
    /// Send a [`Warning::NonFinite`] to the sink and keep training. The update is applied
    /// even so: after a NaN or infinite loss the optimiser steps on that loss's gradients,
    /// and a NaN or infinite gradient is written into its parameter (and into the
    /// optimiser's momentum or moments), so from the next step the NaN reaches every
    /// parameter. Use [`Skip`](OnNonFinite::Skip) to train past a bad step.
    Warn,
    /// Send a [`Warning::NonFinite`] to the sink, then leave the step out: the parameters
    /// and the optimiser's state (its momentum or moments and Adam's step counts) stay as
    /// they were, as if the step had not run, and training continues.
    ///
    /// The step is found bad when its loss is NaN or infinite, or when any gradient of the
    /// model's parameters is. To find the gradient before the update applies it, each step
    /// reads every gradient once more before the optimiser runs; [`Stop`](OnNonFinite::Stop)
    /// and [`Warn`](OnNonFinite::Warn) do not. The warning is sent at most once per epoch
    /// for each location, as under `Warn`. Forward and backward [`Hooks`](crate::Hooks) run
    /// on every step, a step that is then left out included, and see its NaN or infinite
    /// values.
    ///
    /// A skipped step still counts as a step: [`Trainer::steps`](crate::Trainer::steps)
    /// and the step numbers in warnings include it, and
    /// [`Trainer::skipped_steps`](crate::Trainer::skipped_steps) counts it. It does not
    /// count as training: [`fit`](crate::fit) leaves its rows out of
    /// [`EpochReport::rows`](crate::EpochReport::rows) and its loss out of
    /// [`EpochReport::mean_loss`](crate::EpochReport::mean_loss), and
    /// [`StepControl::after_step`](crate::StepControl::after_step) does not run after it.
    /// `fit`'s learning-rate schedule is set once per epoch, so a skipped step does not move
    /// it. A skipped step does not set the loss the vanishing-gradient check compares
    /// against ([`Health::grad_norm_low`]). With the same seeds and the other batches
    /// unchanged and in the same order (for example `shuffle_seed: None`), a run that skips a
    /// step ends with the parameters, bit for bit, of the same run without that step's batch.
    ///
    /// An epoch in which every step was left out ends the run:
    /// [`HealthMonitor::end_epoch`] returns [`Error::NonFinite`] saying so, since the
    /// parameters did not change and every later epoch would be left out too.
    ///
    /// A hand-written loop asks [`HealthMonitor::before_update`] whether to apply each
    /// update.
    Skip,
}

/// The first non-finite value found in a step's batch, named so that a NaN or an infinity
/// in the loss can be traced to the data. Two bad values are equal when every field is
/// equal and the values have the same bits. `value` is compared by its bits because NaN is
/// not equal to itself: with a bit comparison two warnings about the same NaN compare
/// equal, in tests and in sinks that collect warnings.
///
/// # Examples
///
/// ```
/// use cavitation_nn::BadInput;
/// let bad = BadInput { position: 3, row: Some(17), feature: 2, value: f32::NAN, target: false };
/// assert_eq!(
///     bad.to_string(),
///     "the batch's input held NaN at batch position 3, feature 2 (row 17 of the dataset)"
/// );
/// ```
#[derive(Clone, Copy, Debug)]
pub struct BadInput {
    /// The position in the batch, counting from 0.
    pub position: usize,
    /// The row of the dataset that sat at that position, when the caller passed the batch's
    /// rows (a hand-written loop that sets `StepInputs::rows`); `None` when it gave none.
    /// [`fit`](crate::fit) rejects non-finite data before training, with an error that names
    /// the row, so it never reaches this check.
    pub row: Option<usize>,
    /// The feature (the position within the row) for an input; the output index for a target.
    pub feature: usize,
    /// The value: NaN, infinity or negative infinity.
    pub value: f32,
    /// Whether the value is in the targets rather than in the inputs.
    pub target: bool,
}

impl PartialEq for BadInput {
    /// Equal when every field is equal, comparing `value` by its bits.
    ///
    /// ```
    /// use cavitation_nn::BadInput;
    /// let a = BadInput { position: 0, row: None, feature: 1, value: f32::NAN, target: false };
    /// assert_eq!(a, a);
    /// ```
    fn eq(&self, other: &BadInput) -> bool {
        self.position == other.position
            && self.row == other.row
            && self.feature == other.feature
            && self.value.to_bits() == other.value.to_bits()
            && self.target == other.target
    }
}

impl fmt::Display for BadInput {
    /// Writes where the bad value sat, for example `the batch's input held NaN at batch
    /// position 3, feature 2 (row 17 of the dataset)`.
    ///
    /// ```
    /// use cavitation_nn::BadInput;
    /// let bad = BadInput {
    ///     position: 0,
    ///     row: None,
    ///     feature: 4,
    ///     value: f32::INFINITY,
    ///     target: true,
    /// };
    /// assert_eq!(
    ///     bad.to_string(),
    ///     "the batch's targets held infinity at batch position 0, output 4"
    /// );
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = if self.value.is_nan() {
            "NaN"
        } else if self.value > 0.0 {
            "infinity"
        } else {
            "-infinity"
        };
        let (what, part) = if self.target {
            ("targets", "output")
        } else {
            ("input", "feature")
        };
        write!(
            f,
            "the batch's {what} held {value} at batch position {}, {part} {}",
            self.position, self.feature
        )?;
        match self.row {
            Some(row) => write!(f, " (row {row} of the dataset)"),
            None => Ok(()),
        }
    }
}

/// What a step fed the network, borrowed, for [`HealthMonitor::after_backward`] to look
/// through when the loss turns out not to be finite. Nothing is read unless it does.
///
/// # Examples
///
/// ```
/// use cavitation_core::Tensor;
/// use cavitation_nn::{StepInputs, TargetsView};
/// let x = Tensor::from_vec(vec![0.5f32, 1.0], &[2, 1]).unwrap();
/// let inputs = StepInputs { x: x.view(), targets: TargetsView::Classes(&[0, 1]), rows: None };
/// assert_eq!(inputs.x.shape().dims(), &[2, 1]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct StepInputs<'a> {
    /// The batch's inputs, `[rows, features]`.
    pub x: TensorView<'a, f32>,
    /// The batch's targets.
    pub targets: TargetsView<'a, f32>,
    /// The dataset row at each batch position (for example a slice of
    /// [`Batcher::order`](crate::Batcher::order)), or `None` when the batch did not come from
    /// a dataset.
    pub rows: Option<&'a [usize]>,
}

impl StepInputs<'_> {
    /// The first non-finite value in the inputs, else in the value targets; `None` when
    /// every value is finite.
    fn first_non_finite(&self) -> Option<BadInput> {
        let find = |data: &[f32], per_position: usize, target: bool| {
            let at = data.iter().position(|v| !v.is_finite())?;
            let per_position = per_position.max(1);
            let position = at / per_position;
            Some(BadInput {
                position,
                row: self.rows.and_then(|rows| rows.get(position).copied()),
                feature: at % per_position,
                value: data[at],
                target,
            })
        };
        let per = |view: &TensorView<'_, f32>| {
            view.data().len() / view.shape().dims().first().copied().unwrap_or(1).max(1)
        };
        find(self.x.data(), per(&self.x), false).or_else(|| match &self.targets {
            TargetsView::Values(v) => find(v.data(), per(v), true),
            TargetsView::Classes(_) => None,
        })
    }
}

/// The kind of a [`Warning`], without its fields, for counting or filtering warnings.
///
/// # Examples
///
/// ```
/// use cavitation_nn::{Warning, WarningKind};
/// let w = Warning::ClampedLogs { step: 4, count: 1 };
/// assert_eq!(w.kind(), WarningKind::ClampedLogs);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WarningKind {
    /// [`Warning::ClampedLogs`].
    ClampedLogs,
    /// [`Warning::NonFinite`].
    NonFinite,
    /// [`Warning::Exploding`].
    Exploding,
    /// [`Warning::Vanishing`].
    Vanishing,
    /// [`Warning::DeadNeurons`].
    DeadNeurons,
    /// [`Warning::NegativeLoss`].
    NegativeLoss,
}

/// Something the health checks noticed. Every warning names the step or epoch it belongs
/// to; its [`Display`](fmt::Display) form is one plain-English line that ends with what to
/// try. [`kind`](Warning::kind), [`step`](Warning::step), [`epoch`](Warning::epoch),
/// [`location`](Warning::location) and [`layer`](Warning::layer) read the parts without
/// matching on the variant.
///
/// # Examples
///
/// ```
/// use cavitation_nn::Warning;
/// let w = Warning::ClampedLogs { step: 4, count: 1 };
/// assert!(w.to_string().starts_with("step 4:"));
/// ```
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Warning {
    /// Binary cross-entropy met probabilities of exactly 0 or 1 and clamped their
    /// logarithms to -100 (see [`Tape::clamped_logs`]). Sent at most once per epoch.
    ClampedLogs {
        /// The step, counting from 1.
        step: u64,
        /// How many logarithms that step clamped.
        count: usize,
    },
    /// The loss or a gradient held a NaN or an infinity. With [`OnNonFinite::Warn`] and
    /// [`OnNonFinite::Skip`] it is sent at most once per epoch for each location; with
    /// [`OnNonFinite::Stop`] it is sent once, just before the step returns
    /// [`Error::NonFinite`] for the same location.
    NonFinite {
        /// The step, counting from 1.
        step: u64,
        /// `"loss"`, or the parameter's name, for example `"0.weight"`.
        location: String,
        /// For a non-finite loss: the first non-finite value in the batch's inputs or
        /// targets. `None` when every input was finite (the values overflowed inside the
        /// network) and for a gradient.
        input: Option<BadInput>,
    },
    /// A parameter's gradient norm was above [`Health::grad_norm_high`]. Sent at most once
    /// per epoch for each parameter.
    Exploding {
        /// The step, counting from 1.
        step: u64,
        /// The parameter's name, for example `"0.weight"`.
        layer: String,
        /// The Euclidean norm of the parameter's gradient.
        norm: f64,
    },
    /// A parameter's gradient norm was below [`Health::grad_norm_low`]. Sent at most once
    /// per epoch for each parameter.
    Vanishing {
        /// The step, counting from 1.
        step: u64,
        /// The parameter's name, for example `"0.weight"`.
        layer: String,
        /// The Euclidean norm of the parameter's gradient.
        norm: f64,
    },
    /// Neurons of a ReLU or ReLU6 layer that output exactly 0 for every sample of the epoch.
    /// Neurons turned off by a [`NeuronMask`](crate::NeuronMask), on the activation layer or
    /// on the layer before it, are not counted. Sent to the sink every epoch; without a sink,
    /// printed when the layer's share of dead neurons passes 25%, 50% or 90% for the first
    /// time. "Dead" is measured per epoch and on the data the epoch ran over: a neuron that is
    /// silent on every training row of one epoch can fire in a later epoch, once the layers
    /// before it change, or on rows it was not counted on.
    DeadNeurons {
        /// The epoch, as passed to [`HealthMonitor::end_epoch`].
        epoch: usize,
        /// The layer's index in the [`Sequential`].
        layer: usize,
        /// The layer's output width: how many neurons it has.
        width: usize,
        /// The dead neurons' positions on the layer's output, in increasing order.
        neurons: Vec<usize>,
    },
    /// The loss was below 0 although [`Loss::non_negative`] says the trainer's loss cannot
    /// be negative on valid inputs. Sent at most once per epoch, only by a monitor that
    /// knows the loss ([`HealthMonitor::with_loss`]).
    ///
    /// It is sent at the first step whose loss is below 0, which can be a few steps after
    /// the mistake starts: with [`Loss::Nll`] on raw scores, for example, the loss is
    /// negative only once the target class's score is positive.
    NegativeLoss {
        /// The step, counting from 1.
        step: u64,
        /// The loss.
        loss: f32,
    },
}

impl Warning {
    /// Which kind of warning this is.
    ///
    /// ```
    /// use cavitation_nn::{Warning, WarningKind};
    /// let w = Warning::Exploding { step: 3, layer: "0.weight".to_string(), norm: 1e4 };
    /// assert_eq!(w.kind(), WarningKind::Exploding);
    /// ```
    pub fn kind(&self) -> WarningKind {
        match self {
            Warning::ClampedLogs { .. } => WarningKind::ClampedLogs,
            Warning::NonFinite { .. } => WarningKind::NonFinite,
            Warning::Exploding { .. } => WarningKind::Exploding,
            Warning::Vanishing { .. } => WarningKind::Vanishing,
            Warning::DeadNeurons { .. } => WarningKind::DeadNeurons,
            Warning::NegativeLoss { .. } => WarningKind::NegativeLoss,
        }
    }

    /// The step the warning belongs to, counting from 1; `None` for
    /// [`Warning::DeadNeurons`], which belongs to an epoch.
    ///
    /// ```
    /// use cavitation_nn::Warning;
    /// let w = Warning::ClampedLogs { step: 4, count: 1 };
    /// assert_eq!(w.step(), Some(4));
    /// ```
    pub fn step(&self) -> Option<u64> {
        match self {
            Warning::ClampedLogs { step, .. }
            | Warning::NonFinite { step, .. }
            | Warning::Exploding { step, .. }
            | Warning::Vanishing { step, .. }
            | Warning::NegativeLoss { step, .. } => Some(*step),
            Warning::DeadNeurons { .. } => None,
        }
    }

    /// The epoch the warning belongs to; `Some` only for [`Warning::DeadNeurons`].
    ///
    /// ```
    /// use cavitation_nn::Warning;
    /// let w = Warning::DeadNeurons { epoch: 2, layer: 1, width: 8, neurons: vec![0] };
    /// assert_eq!(w.epoch(), Some(2));
    /// ```
    pub fn epoch(&self) -> Option<usize> {
        match self {
            Warning::DeadNeurons { epoch, .. } => Some(*epoch),
            _ => None,
        }
    }

    /// Where the warning happened, as a parameter's name (for example `"0.weight"`) or
    /// `"loss"`; `None` for [`Warning::ClampedLogs`], [`Warning::DeadNeurons`] and
    /// [`Warning::NegativeLoss`].
    ///
    /// ```
    /// use cavitation_nn::Warning;
    /// let w = Warning::NonFinite { step: 1, location: "loss".to_string(), input: None };
    /// assert_eq!(w.location(), Some("loss"));
    /// ```
    pub fn location(&self) -> Option<&str> {
        match self {
            Warning::NonFinite { location, .. } => Some(location),
            Warning::Exploding { layer, .. } | Warning::Vanishing { layer, .. } => Some(layer),
            Warning::ClampedLogs { .. }
            | Warning::DeadNeurons { .. }
            | Warning::NegativeLoss { .. } => None,
        }
    }

    /// The layer's index in the [`Sequential`]; `Some` only for [`Warning::DeadNeurons`].
    ///
    /// ```
    /// use cavitation_nn::Warning;
    /// let w = Warning::DeadNeurons { epoch: 2, layer: 1, width: 8, neurons: vec![0] };
    /// assert_eq!(w.layer(), Some(1));
    /// ```
    pub fn layer(&self) -> Option<usize> {
        match self {
            Warning::DeadNeurons { layer, .. } => Some(*layer),
            _ => None,
        }
    }
}

/// The share of dead neurons that a printed report must reach to count as worse, as
/// percentages of the layer.
const DEAD_THRESHOLDS: [usize; 3] = [25, 50, 90];

/// How many of [`DEAD_THRESHOLDS`] `dead` neurons of `width` have reached.
fn dead_level(dead: usize, width: usize) -> usize {
    DEAD_THRESHOLDS
        .iter()
        .filter(|&&percent| dead * 100 >= width * percent)
        .count()
}

/// How many neuron indices a dead-neuron line lists before its ellipsis.
const LISTED_NEURONS: usize = 10;

impl fmt::Display for Warning {
    /// Writes the warning as one plain-English line, starting with its step or epoch and
    /// ending with what to try.
    ///
    /// ```
    /// use cavitation_nn::Warning;
    /// let w = Warning::NonFinite { step: 1, location: "loss".to_string(), input: None };
    /// assert_eq!(w.to_string(), "step 1: a NaN or infinite value in loss");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Warning::ClampedLogs { step, count } => write!(
                f,
                "step {step}: binary cross-entropy met {count} probabilities of exactly 0 or 1 \
                 and clamped their logarithms to -100; the clamped rows pass no gradient, so use \
                 Loss::BinaryCrossEntropyWithLogits on raw scores instead"
            ),
            Warning::NonFinite {
                step,
                location,
                input,
            } => {
                write!(f, "step {step}: a NaN or infinite value in {location}")?;
                match input {
                    Some(bad) => write!(f, "; {bad}"),
                    None => Ok(()),
                }
            }
            Warning::NegativeLoss { step, loss } => write!(
                f,
                "step {step}: the loss is {loss:.4e}, below 0, but this loss cannot be negative on \
                 valid inputs; the usual causes are Loss::Nll on scores that are not \
                 log-probabilities (end the model with LogSoftmax, or use Loss::CrossEntropy) and \
                 binary cross-entropy targets outside [0, 1]"
            ),
            Warning::Exploding { step, layer, norm } => write!(
                f,
                "step {step}: the gradient norm of {layer} is {norm:.3e}, above the exploding limit \
                 (Health::grad_norm_high, 1e3 by default); lower the learning rate, or raise the \
                 limit if this size is normal for the model"
            ),
            Warning::Vanishing { step, layer, norm } => {
                write!(
                    f,
                    "step {step}: the gradient norm of {layer} is {norm:.3e}, below the vanishing \
                     limit (Health::grad_norm_low, 1e-7 by default; a gradient this small with a \
                     loss that has not fallen 10,000-fold is a vanishing gradient); "
                )?;
                if *norm == 0.0 {
                    write!(
                        f,
                        "a norm of exactly 0 means no gradient reached it, which points at dead \
                         ReLUs or a saturated output"
                    )
                } else {
                    write!(
                        f,
                        "a small norm that is not 0 points at a deep network or saturating \
                         activations such as sigmoid and tanh"
                    )
                }
            }
            Warning::DeadNeurons {
                epoch,
                layer,
                width,
                neurons,
            } => {
                let dead = neurons.len();
                let percent = (dead * 200 + width) / (2 * (*width).max(1));
                write!(
                    f,
                    "epoch {epoch}: {dead} of {width} ({percent}%) neurons of layer {layer} output 0 \
                     for every sample (dead ReLU): ["
                )?;
                for (i, n) in neurons.iter().take(LISTED_NEURONS).enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{n}")?;
                }
                if dead > LISTED_NEURONS {
                    f.write_str(", …")?;
                }
                f.write_str(
                    "]; try a lower learning rate, a positive bias initialisation or a leaky \
                     activation such as LeakyRelu",
                )
            }
        }
    }
}

/// Which health checks run during training, and where their warnings go.
///
/// The default stops on a non-finite loss or gradient, warns about gradient norms above
/// 1e3 or below 1e-7, reports dead ReLU neurons, and prints the warnings to standard error
/// as `cavitation: warning: <the warning>`. Printing is bounded so a long bad run stays
/// readable and a healthy run prints nothing:
///
/// - a layer's dead-neuron report is printed at the end of an epoch in which its share of
///   dead neurons passes 25%, 50% or 90% for the first time (one line, however many it
///   passes), so a layer at 3% prints nothing;
/// - an exploding or vanishing gradient, or a non-finite value under
///   [`OnNonFinite::Warn`] or [`OnNonFinite::Skip`], is printed once for each parameter,
///   and each later epoch adds one line per kind naming the parameters that repeated.
/// - a negative loss that cannot be negative on valid inputs ([`Warning::NegativeLoss`]) is
///   printed in full once, and each later epoch in which it happens again adds one line.
///
/// A [`sink`](Health::sink) receives every warning of every epoch, and [`Health::quiet`]
/// drops them all.
///
/// # Examples
///
/// ```
/// use cavitation_nn::{Health, OnNonFinite, Warning};
/// let health = Health {
///     on_non_finite: OnNonFinite::Warn,
///     dead_relu: false,
///     sink: Some(Box::new(|w: &Warning| println!("training: {w}"))),
///     ..Health::default()
/// };
/// assert_eq!(health.grad_norm_high, 1e3);
/// ```
pub struct Health {
    /// What to do when the loss or a gradient is NaN or infinite.
    pub on_non_finite: OnNonFinite,
    /// A gradient norm above this sends [`Warning::Exploding`].
    pub grad_norm_high: f64,
    /// A gradient norm below this sends [`Warning::Vanishing`]. A gradient this small with a
    /// loss that has not fallen is a vanishing gradient; once the loss has fallen 10,000-fold
    /// from its first value (or to 0), the warning is not sent, since that is convergence.
    /// A first loss that is not above 0 never counts, so the warning is always sent then.
    pub grad_norm_low: f64,
    /// Whether to look for dead ReLU and ReLU6 neurons (one pass over those layers' outputs
    /// each step).
    pub dead_relu: bool,
    /// Receives every warning of every epoch; `None` prints them to standard error, prefixed
    /// `cavitation: warning:`, with the bounded printing described on [`Health`].
    // The boxed callback is the plainest type for this field; an alias would only hide it.
    #[allow(clippy::type_complexity)]
    pub sink: Option<Box<dyn FnMut(&Warning)>>,
}

impl Health {
    /// Every check on and `Stop` on non-finite values, but no output: the warnings go to a
    /// sink that drops them. A step that meets a NaN or infinity still returns
    /// [`Error::NonFinite`].
    ///
    /// ```
    /// use cavitation_nn::{Health, OnNonFinite};
    /// let health = Health::quiet();
    /// assert_eq!(health.on_non_finite, OnNonFinite::Stop);
    /// assert!(health.dead_relu && health.sink.is_some());
    /// ```
    pub fn quiet() -> Health {
        Health {
            sink: Some(Box::new(|_: &Warning| {})),
            ..Health::default()
        }
    }
}

impl Default for Health {
    /// Stop on non-finite values, norm bounds 1e3 and 1e-7, dead ReLU checks on, warnings
    /// printed to standard error.
    fn default() -> Health {
        Health {
            on_non_finite: OnNonFinite::Stop,
            grad_norm_high: 1e3,
            grad_norm_low: 1e-7,
            dead_relu: true,
            sink: None,
        }
    }
}

impl fmt::Debug for Health {
    /// Writes every setting; the sink shows only whether there is one.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Health")
            .field("on_non_finite", &self.on_non_finite)
            .field("grad_norm_high", &self.grad_norm_high)
            .field("grad_norm_low", &self.grad_norm_low)
            .field("dead_relu", &self.dead_relu)
            .field("sink", &self.sink.as_ref().map(|_| "FnMut"))
            .finish()
    }
}

/// Bit in a parameter's warned flags: [`Warning::NonFinite`] was sent this epoch.
const WARNED_NON_FINITE: u8 = 1;
/// Bit in a parameter's warned flags: [`Warning::Exploding`] was sent this epoch.
const WARNED_EXPLODING: u8 = 2;
/// Bit in a parameter's warned flags: [`Warning::Vanishing`] was sent this epoch.
const WARNED_VANISHING: u8 = 4;

/// The error detail for a non-finite loss whose inputs and targets were all finite.
const OVERFLOW_DETAIL: &str = "the inputs were finite, so the values overflowed inside the \
     network; try a lower learning rate or smaller inputs";

/// A run whose latest loss is at most this fraction of its first loss has converged, so a
/// small gradient is not a vanishing one.
const CONVERGED_LOSS_RATIO: f64 = 1e-4;

/// Where one ReLU or ReLU6 layer's neurons sit in the monitor's `alive` buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReluLayer {
    /// The layer's index in the model.
    layer: usize,
    /// The position of its first neuron in `alive`.
    offset: usize,
    /// Its output width.
    width: usize,
}

/// Runs the [`Health`] checks during training. [`Trainer`](crate::Trainer) and
/// [`fit`](crate::fit) run one for you; a hand-written loop calls it directly, three places
/// per epoch:
///
/// 1. [`after_backward`](Self::after_backward) after `tape.backward`, while the tape still
///    lives: the loss's finiteness, binary cross-entropy's clamped logarithms
///    ([`Tape::clamped_logs`]) and the ReLU outputs for the dead-neuron check. Run the
///    forward pass with [`Sequential::forward_traced`] so the monitor can find each layer's
///    output.
/// 2. [`after_update`](Self::after_update) after the tape is dropped and
///    [`Optimizer::step`](crate::Optimizer::step) has run: each gradient's finiteness and
///    norm, from the optimiser's [`GradStats`] (read with
///    [`last_stats`](crate::Optimizer::last_stats)), so the gradients are not read twice.
///    A gradient that is not finite while the loss is finite is therefore found after the
///    update has applied it, except under [`OnNonFinite::Skip`], where
///    [`before_update`](Self::before_update), called after the tape is dropped and before
///    the optimiser step, finds it first and says whether to apply the update at all.
/// 3. [`end_epoch`](Self::end_epoch) once each epoch ends: the dead-neuron report, the
///    check that not every step of the epoch was left out under [`OnNonFinite::Skip`], and
///    the start of a new epoch for the once-per-epoch warnings.
///
/// After its first epoch the monitor allocates nothing while training is healthy; a
/// warning that names a parameter or lists neurons allocates its text when it is sent.
///
/// # Examples
///
/// ```
/// use cavitation_core::{Error, Graph, ParamStore, Reduction, Rng, Tensor};
/// use cavitation_nn::{
///     Health, HealthMonitor, Linear, Optimizer, Sequential, Sgd, StepInputs, TargetsView, Trace,
/// };
/// let x = Tensor::from_vec(vec![0.0f32, 1.0, f32::NAN, 3.0], &[4, 1]).unwrap();
/// let y = Tensor::from_vec(vec![1.0f32, 3.0, 5.0, 7.0], &[4, 1]).unwrap();
/// let mut params = ParamStore::new();
/// let net = Sequential::new().push(Linear::new(&mut params, 1, 1, &mut Rng::seed(0)));
/// let names = net.parameter_names();
/// let mut monitor = HealthMonitor::new(Health::default());
/// let (mut graph, mut trace, mut optimizer) = (Graph::new(), Trace::new(), Sgd::new(0.05));
/// let mut train = || -> Result<(), Error> {
///     for step in 1..=10 {
///         let mut tape = graph.tape(&params.values);
///         let input = tape.input(x.view());
///         let target = tape.input(y.view());
///         let out = net.forward_traced(&mut tape, input, &mut trace);
///         let loss = tape.mse(out, target, Reduction::Mean);
///         let value = tape.value(loss).data()[0];
///         tape.backward(loss, &mut params.grads);
///         let inputs = StepInputs {
///             x: x.view(),
///             targets: TargetsView::Values(y.view()),
///             rows: None,
///         };
///         let checked = monitor.after_backward(step, value, &tape, &trace, &net, inputs);
///         drop(tape);
///         // Update only if the loss was finite (and, under `Skip`, the gradients too), and
///         // clear the gradients before returning the error, so a stopped step leaves nothing
///         // behind for the next one to add to.
///         let checked = checked.and_then(|()| {
///             if monitor.before_update(step, value, &params, &names) {
///                 optimizer.step(&mut params);
///                 monitor.after_update(step, optimizer.last_stats(), &names)?;
///             }
///             Ok(())
///         });
///         params.zero_grad();
///         checked?;
///     }
///     monitor.end_epoch(1)
/// };
/// let err = train().unwrap_err();
/// assert_eq!(
///     err.to_string(),
///     "non-finite value at step 1 in loss: the batch's input held NaN at batch position 2, \
///      feature 0"
/// );
/// ```
pub struct HealthMonitor {
    /// The checks and the sink.
    health: Health,
    /// Whether the non-finite loss warning was sent this epoch.
    loss_warned: bool,
    /// Whether the clamped-logarithm warning was sent this epoch.
    clamped_warned: bool,
    /// Whether the loss cannot be negative on valid inputs, so a negative one is reported.
    non_negative: bool,
    /// Whether the negative-loss warning was sent this epoch.
    negative_warned: bool,
    /// For each parameter, in the order of the names passed to `after_update`, which
    /// warnings were sent this epoch (the `WARNED_*` bits).
    warned: Vec<u8>,
    /// The ReLU and ReLU6 layers seen in the last step.
    relu_layers: Vec<ReluLayer>,
    /// For every neuron of those layers, whether it output anything but 0 this epoch (or is
    /// masked off, and so never reported).
    alive: Vec<bool>,
    /// Whether `alive` holds at least one step of this epoch.
    tracked: bool,
    /// For each layer whose dead-neuron report was printed to standard error (with no
    /// sink), how many of [`DEAD_THRESHOLDS`] the printed share had reached.
    dead_printed: Vec<(usize, usize)>,
    /// The kind and parameter of each norm or non-finite warning printed to standard error
    /// (with no sink), so a later one is summarised instead.
    printed: Vec<(WarningKind, String)>,
    /// This epoch's norm and non-finite warnings that were already printed once: the kind,
    /// the parameter and its norm. Summarised and cleared by `end_epoch`.
    repeats: Vec<(WarningKind, String, f64)>,
    /// The first finite loss seen by `after_backward`, kept for the monitor's whole life.
    first_loss: Option<f64>,
    /// The latest finite loss seen by `after_backward`.
    latest_loss: Option<f64>,
    /// Under [`OnNonFinite::Skip`], this step's finite loss, held by `after_backward` until
    /// `after_update` shows the step trained.
    pending_loss: Option<f64>,
    /// Under [`OnNonFinite::Skip`], the last step of this epoch that `before_update` kept
    /// out, if any.
    skipped_this_epoch: Option<u64>,
    /// Under [`OnNonFinite::Skip`], whether `before_update` let any step of this epoch
    /// train.
    trained_this_epoch: bool,
}

impl HealthMonitor {
    /// A monitor running the checks of `health`.
    ///
    /// ```
    /// use cavitation_nn::{Health, HealthMonitor};
    /// let mut monitor = HealthMonitor::new(Health::default());
    /// assert!(monitor.end_epoch(1).is_ok());
    /// ```
    pub fn new(health: Health) -> HealthMonitor {
        HealthMonitor {
            health,
            loss_warned: false,
            clamped_warned: false,
            non_negative: false,
            negative_warned: false,
            warned: Vec::new(),
            relu_layers: Vec::new(),
            alive: Vec::new(),
            tracked: false,
            dead_printed: Vec::new(),
            printed: Vec::new(),
            repeats: Vec::new(),
            first_loss: None,
            latest_loss: None,
            pending_loss: None,
            skipped_this_epoch: None,
            trained_this_epoch: false,
        }
    }

    /// Makes the monitor report a loss below 0 with [`Warning::NegativeLoss`] (at most once
    /// per epoch) when `loss` cannot be negative on valid inputs ([`Loss::non_negative`]).
    /// [`Trainer`](crate::Trainer) does this for its own loss; a hand-written loop opts in
    /// here.
    ///
    /// ```
    /// use cavitation_nn::{Health, HealthMonitor, Loss};
    /// let mut monitor = HealthMonitor::new(Health::default()).with_loss(Loss::Nll);
    /// assert!(monitor.end_epoch(1).is_ok());
    /// ```
    pub fn with_loss(mut self, loss: Loss) -> HealthMonitor {
        self.non_negative = loss.non_negative();
        self
    }

    /// Records `loss` as the latest finite loss, and as the first if there was none.
    fn record_loss(&mut self, loss: f64) {
        self.first_loss.get_or_insert(loss);
        self.latest_loss = Some(loss);
    }

    /// Whether the loss has fallen to at most [`CONVERGED_LOSS_RATIO`] of its first value.
    /// False when no finite loss was recorded, and false when the first loss is not above 0
    /// (a loss such as `kl_div` on inputs that are not log-probabilities can start negative,
    /// and a fraction of a negative loss is not a sign of convergence).
    fn converged(&self) -> bool {
        match (self.first_loss, self.latest_loss) {
            (Some(first), Some(latest)) if first > 0.0 => latest <= first * CONVERGED_LOSS_RATIO,
            _ => false,
        }
    }

    /// Sends `warning` to the sink, or, with none, prints it to standard error by the rules
    /// on [`Health`]: a dead-neuron report when its share passes a threshold not passed
    /// before; the first norm or non-finite warning of each parameter, with later ones kept
    /// for the summary of `end_epoch`; anything else always.
    fn emit(&mut self, warning: Warning) {
        if let Some(sink) = self.health.sink.as_mut() {
            sink(&warning);
            return;
        }
        match warning {
            Warning::DeadNeurons {
                layer,
                width,
                ref neurons,
                ..
            } => {
                let level = dead_level(neurons.len(), width);
                let at = match self.dead_printed.iter().position(|(l, _)| *l == layer) {
                    Some(at) => at,
                    None => {
                        self.dead_printed.push((layer, 0));
                        self.dead_printed.len() - 1
                    }
                };
                if level > self.dead_printed[at].1 {
                    self.dead_printed[at].1 = level;
                    eprintln!("cavitation: warning: {warning}");
                }
            }
            Warning::NonFinite { .. }
            | Warning::Exploding { .. }
            | Warning::Vanishing { .. }
            | Warning::NegativeLoss { .. } => {
                let kind = warning.kind();
                let name = warning.location().unwrap_or_default();
                let norm = match warning {
                    Warning::Exploding { norm, .. } | Warning::Vanishing { norm, .. } => norm,
                    Warning::NegativeLoss { loss, .. } => f64::from(loss),
                    _ => 0.0,
                };
                if self.printed.iter().any(|(k, n)| *k == kind && n == name) {
                    self.repeats.push((kind, name.to_string(), norm));
                } else {
                    eprintln!("cavitation: warning: {warning}");
                    self.printed.push((kind, name.to_string()));
                }
            }
            Warning::ClampedLogs { .. } => {
                eprintln!("cavitation: warning: {warning}")
            }
        }
    }

    /// Prints one line per kind for the parameters whose norm or non-finite warning repeated
    /// in epoch `epoch` after being printed once, and one line if the negative loss repeated,
    /// then forgets them.
    fn print_repeats(&mut self, epoch: usize) {
        let kinds = [
            (WarningKind::NonFinite, "locations", "non-finite"),
            (WarningKind::Exploding, "parameters", "exploding"),
            (WarningKind::Vanishing, "parameters", "vanishing"),
        ];
        for (kind, noun, what) in kinds {
            let of_kind = || self.repeats.iter().filter(|(k, _, _)| *k == kind);
            let count = of_kind().count();
            if count == 0 {
                continue;
            }
            let mut line =
                format!("cavitation: warning: epoch {epoch}: {count} {noun} {what} again: ");
            for (i, (_, name, norm)) in of_kind().take(LISTED_NEURONS).enumerate() {
                if i > 0 {
                    line.push_str(", ");
                }
                line.push_str(name);
                if kind != WarningKind::NonFinite {
                    line.push_str(&format!(" {norm:.3e}"));
                }
            }
            if count > LISTED_NEURONS {
                line.push_str(", …");
            }
            eprintln!("{line}");
        }
        if let Some((_, _, first)) = self
            .repeats
            .iter()
            .find(|(k, _, _)| *k == WarningKind::NegativeLoss)
        {
            eprintln!(
                "cavitation: warning: epoch {epoch}: the loss was negative again \
                 (first negative loss this epoch: {first:.3e})"
            );
        }
        self.repeats.clear();
    }

    /// Sends [`Warning::NonFinite`] for `location` at `step` to the sink and returns the
    /// matching [`Error::NonFinite`] with `detail`, for [`OnNonFinite::Stop`].
    fn stop(
        &mut self,
        step: u64,
        location: String,
        input: Option<BadInput>,
        detail: Option<String>,
    ) -> Error {
        self.emit(Warning::NonFinite {
            step,
            location: location.clone(),
            input,
        });
        Error::NonFinite {
            step,
            location,
            detail,
        }
    }

    /// The checks that need the tape, run after the backward pass of step `step` (counting
    /// from 1) and before the tape is dropped: `loss` must be finite, binary cross-entropy
    /// clamps send [`Warning::ClampedLogs`] (at most once per epoch), and, with
    /// [`Health::dead_relu`], each ReLU and ReLU6 layer's output in `trace` is checked for
    /// neurons that are not 0. `trace` must come from this step's
    /// [`Sequential::forward_traced`] of `model` on `tape`. A monitor made with
    /// [`with_loss`](HealthMonitor::with_loss) for a loss that cannot be negative sends
    /// [`Warning::NegativeLoss`] (at most once per epoch) when `loss` is below 0.
    ///
    /// `inputs` is what the step fed the network. It is read only when `loss` is NaN or
    /// infinite, to name the first non-finite input or target value (or to say that the
    /// inputs were finite, so the values overflowed inside the network); a healthy step
    /// reads and allocates nothing for it.
    ///
    /// # Errors
    ///
    /// With [`OnNonFinite::Stop`], returns [`Error::NonFinite`] with location `"loss"` if
    /// `loss` is NaN or infinite, after sending [`Warning::NonFinite`] to the sink; the error's
    /// detail names the input or says the inputs were finite. With [`OnNonFinite::Warn`] and
    /// [`OnNonFinite::Skip`] it only sends the warning (at most once per epoch) and returns
    /// `Ok`; under `Skip` the step's ReLU outputs are then not counted for the dead-neuron
    /// check, as the step is left out.
    ///
    /// # Panics
    ///
    /// May panic if `trace` was not recorded on `tape`.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// use cavitation_nn::{
    ///     Health, HealthMonitor, Relu, Sequential, StepInputs, TargetsView, Trace,
    /// };
    /// let params = ParamStore::new();
    /// let net = Sequential::new().push(Relu);
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&params.values);
    /// let data = Tensor::full(&[2, 2], 1.0f32);
    /// let x = tape.input(data.view());
    /// let mut trace = Trace::new();
    /// net.forward_traced(&mut tape, x, &mut trace);
    /// let inputs = StepInputs {
    ///     x: data.view(),
    ///     targets: TargetsView::Classes(&[0, 1]),
    ///     rows: None,
    /// };
    /// let mut monitor = HealthMonitor::new(Health::default());
    /// assert!(monitor.after_backward(1, 4.0, &tape, &trace, &net, inputs).is_ok());
    /// let err = monitor.after_backward(2, f32::NAN, &tape, &trace, &net, inputs).unwrap_err();
    /// // Every input was finite, so the detail says the values overflowed.
    /// assert!(
    ///     err.to_string()
    ///         .starts_with("non-finite value at step 2 in loss: the inputs were finite")
    /// );
    /// ```
    pub fn after_backward(
        &mut self,
        step: u64,
        loss: f32,
        tape: &Tape<'_, f32>,
        trace: &Trace,
        model: &Sequential,
        inputs: StepInputs<'_>,
    ) -> Result<(), Error> {
        let clamped = tape.clamped_logs();
        if clamped > 0 && !self.clamped_warned {
            self.clamped_warned = true;
            self.emit(Warning::ClampedLogs {
                step,
                count: clamped,
            });
        }
        if loss.is_finite() {
            let loss = f64::from(loss);
            if self.health.on_non_finite == OnNonFinite::Skip {
                // Recorded by `after_update`, once the step is known to have trained.
                self.pending_loss = Some(loss);
            } else {
                self.record_loss(loss);
            }
            if self.non_negative && loss < 0.0 && !self.negative_warned {
                self.negative_warned = true;
                self.emit(Warning::NegativeLoss {
                    step,
                    loss: loss as f32,
                });
            }
        } else {
            match self.health.on_non_finite {
                OnNonFinite::Stop => {
                    let input = inputs.first_non_finite();
                    let detail = match input {
                        Some(bad) => bad.to_string(),
                        None => OVERFLOW_DETAIL.to_string(),
                    };
                    return Err(self.stop(step, "loss".to_string(), input, Some(detail)));
                }
                OnNonFinite::Warn | OnNonFinite::Skip if !self.loss_warned => {
                    self.loss_warned = true;
                    let input = inputs.first_non_finite();
                    self.emit(Warning::NonFinite {
                        step,
                        location: "loss".to_string(),
                        input,
                    });
                }
                OnNonFinite::Warn | OnNonFinite::Skip => {}
            }
        }
        let left_out = !loss.is_finite() && self.health.on_non_finite == OnNonFinite::Skip;
        if self.health.dead_relu && !left_out {
            self.track_relu_outputs(tape, trace, model);
        }
        Ok(())
    }

    /// Marks every neuron of each ReLU and ReLU6 layer that output anything but 0 in this
    /// step, and every neuron masked off, as alive. Rebuilds the layer table when the
    /// layers or widths differ from the last step's; otherwise allocates nothing.
    fn track_relu_outputs(&mut self, tape: &Tape<'_, f32>, trace: &Trace, model: &Sequential) {
        let mut k = 0;
        let mut offset = 0;
        for i in 0..trace.len().min(model.len()) {
            let is_relu = model
                .layer(i)
                .activation()
                .and_then(|a| a.builtin())
                .is_some_and(|kind| matches!(kind, ActivationKind::Relu | ActivationKind::Relu6));
            if !is_relu {
                continue;
            }
            let out = tape.value(trace.output(i));
            let width = last_dim(*out.shape());
            let entry = ReluLayer {
                layer: i,
                offset,
                width,
            };
            if self.relu_layers.get(k) != Some(&entry) {
                self.relu_layers.truncate(k);
                self.relu_layers.push(entry);
                self.alive.truncate(offset);
                self.alive.resize(offset + width, false);
            }
            let alive = &mut self.alive[offset..offset + width];
            if width > 0 {
                for row in out.data().chunks_exact(width) {
                    for (a, &v) in alive.iter_mut().zip(row) {
                        *a |= v != 0.0;
                    }
                }
            }
            // A neuron masked on the layer's own output, on the layer before it (the
            // `Linear` that feeds the ReLU), or on an element-wise layer in the run right
            // after it (where `Sequential::neuron_layer` puts the mask) is off on purpose,
            // not dead.
            let first = i.saturating_sub(1);
            let mut last = i;
            while last + 1 < model.len() && model.layer(last + 1).is_elementwise() {
                last += 1;
            }
            for p in first..=last {
                let Some(mask) = model.mask(p) else { continue };
                if mask.width() == width {
                    for (j, a) in alive.iter_mut().enumerate() {
                        *a |= !mask.is_on(j);
                    }
                }
            }
            k += 1;
            offset += width;
        }
        self.relu_layers.truncate(k);
        self.alive.truncate(offset);
        self.tracked = true;
    }

    /// Whether to apply step `step`'s update, asked after the tape is dropped and before
    /// [`Optimizer::step`](crate::Optimizer::step), with the step's `loss` and the gradients
    /// in `params`. Under [`OnNonFinite::Stop`] and [`OnNonFinite::Warn`] it returns `true`
    /// and reads nothing (under `Stop`, [`after_backward`](Self::after_backward) has already
    /// returned the error for a non-finite loss). Under [`OnNonFinite::Skip`] it returns
    /// `false` when `loss` is NaN or infinite (`after_backward` has sent that warning) or
    /// when the gradient of a parameter in `names` holds a NaN or an infinity, sending
    /// [`Warning::NonFinite`] for each such parameter (at most once per epoch each); it reads
    /// every gradient of `names` once to find out, and allocates nothing but the warnings.
    ///
    /// When it returns `false`, leave the optimiser and [`after_update`](Self::after_update)
    /// out for this step and clear the gradients, so the parameters and the optimiser's
    /// state stay as they were.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Reduction, Rng, Tensor};
    /// use cavitation_nn::{
    ///     Health, HealthMonitor, Linear, OnNonFinite, Optimizer, Sequential, Sgd, StepInputs,
    ///     TargetsView, Trace,
    /// };
    /// let x = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], &[4, 1]).unwrap();
    /// let y = Tensor::from_vec(vec![1.0f32, 3.0, 5.0, 7.0], &[4, 1]).unwrap();
    /// let mut bad = x.clone();
    /// bad.data_mut()[2] = f32::NAN;
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 1, 1, &mut Rng::seed(0)));
    /// let names = net.parameter_names();
    /// let health = Health {
    ///     on_non_finite: OnNonFinite::Skip,
    ///     ..Health::quiet()
    /// };
    /// let mut monitor = HealthMonitor::new(health);
    /// let (mut graph, mut trace, mut optimizer) = (Graph::new(), Trace::new(), Sgd::new(0.05));
    /// let mut skipped = 0;
    /// for step in 1..=10 {
    ///     // Every third batch holds a NaN.
    ///     let input = if step % 3 == 0 { &bad } else { &x };
    ///     let mut tape = graph.tape(&params.values);
    ///     let xv = tape.input(input.view());
    ///     let target = tape.input(y.view());
    ///     let out = net.forward_traced(&mut tape, xv, &mut trace);
    ///     let loss = tape.mse(out, target, Reduction::Mean);
    ///     let value = tape.value(loss).data()[0];
    ///     tape.backward(loss, &mut params.grads);
    ///     let inputs = StepInputs {
    ///         x: input.view(),
    ///         targets: TargetsView::Values(y.view()),
    ///         rows: None,
    ///     };
    ///     monitor.after_backward(step, value, &tape, &trace, &net, inputs).unwrap();
    ///     drop(tape);
    ///     if monitor.before_update(step, value, &params, &names) {
    ///         optimizer.step(&mut params);
    ///         monitor.after_update(step, optimizer.last_stats(), &names).unwrap();
    ///     } else {
    ///         skipped += 1;
    ///     }
    ///     params.zero_grad();
    /// }
    /// assert_eq!(skipped, 3);
    /// assert!(names.iter().all(|(_, id)| params.value(*id).data()[0].is_finite()));
    /// ```
    pub fn before_update(
        &mut self,
        step: u64,
        loss: f32,
        params: &ParamStore,
        names: &[(String, ParamId)],
    ) -> bool {
        if self.health.on_non_finite != OnNonFinite::Skip {
            return true;
        }
        let apply = loss.is_finite() && self.gradients_finite(step, params, names);
        if apply {
            self.trained_this_epoch = true;
        } else {
            self.pending_loss = None;
            self.skipped_this_epoch = Some(step);
        }
        apply
    }

    /// Whether every gradient of `names` in `params` is finite, sending
    /// [`Warning::NonFinite`] for each parameter whose gradient is not (at most once per
    /// epoch each).
    fn gradients_finite(
        &mut self,
        step: u64,
        params: &ParamStore,
        names: &[(String, ParamId)],
    ) -> bool {
        self.warned.resize(names.len(), 0);
        let mut apply = true;
        for (k, (name, id)) in names.iter().enumerate() {
            // `&` rather than a short-circuiting `all`, so the loop has no branch per value.
            let finite = params
                .grad(*id)
                .data()
                .iter()
                .fold(true, |finite, g| finite & g.is_finite());
            if finite {
                continue;
            }
            apply = false;
            if self.warned[k] & WARNED_NON_FINITE != 0 {
                continue;
            }
            self.warned[k] |= WARNED_NON_FINITE;
            self.emit(Warning::NonFinite {
                step,
                location: name.clone(),
                input: None,
            });
        }
        apply
    }

    /// The checks on the gradients, run after [`Optimizer::step`](crate::Optimizer::step)
    /// of step `step` with that optimiser's [`last_stats`](crate::Optimizer::last_stats).
    /// `names` gives each parameter's name, as [`Sequential::parameter_names`] returns them
    /// (compute it once, before training). Each named parameter's gradient must be finite,
    /// and its norm between [`Health::grad_norm_low`] and [`Health::grad_norm_high`]: one
    /// outside sends [`Warning::Vanishing`] or [`Warning::Exploding`], at most once per
    /// epoch for each parameter. No [`Warning::Vanishing`] is sent once the latest loss given
    /// to [`after_backward`](HealthMonitor::after_backward) is at most 1e-4 of the first
    /// (the run has converged); without a recorded loss the check always runs.
    ///
    /// The update has already applied the gradients when this runs, so a non-finite
    /// gradient has reached the parameters by then, unless
    /// [`before_update`](Self::before_update) kept the update out under
    /// [`OnNonFinite::Skip`]; do not call this for a step whose update was kept out. Under
    /// `Skip`, this is also where the step's loss is recorded for the vanishing-gradient
    /// check, so a step that was kept out never sets it.
    ///
    /// # Errors
    ///
    /// With [`OnNonFinite::Stop`], returns [`Error::NonFinite`] naming the first parameter
    /// (in the order of the store) whose gradient held a NaN or an infinity, after sending
    /// [`Warning::NonFinite`] to the sink. With [`OnNonFinite::Warn`] (and
    /// [`OnNonFinite::Skip`], where [`before_update`](Self::before_update) has already kept
    /// such a step out) it sends [`Warning::NonFinite`] for each such parameter instead (at
    /// most once per epoch each) and returns `Ok`.
    ///
    /// # Panics
    ///
    /// Panics if a parameter in `names` was not in the optimiser's last step.
    ///
    /// ```
    /// use std::cell::Cell;
    /// use std::rc::Rc;
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Health, HealthMonitor, Linear, Optimizer, Sequential, Sgd, Warning};
    /// let mut params = ParamStore::new();
    /// let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
    /// let mut optimizer = Sgd::new(0.1);
    /// // No backward pass: nothing moves, and every gradient counts as 0 in the statistics.
    /// optimizer.step(&mut params);
    /// // The sink is `'static`, so it shares the count through an `Rc`.
    /// let vanishing = Rc::new(Cell::new(0));
    /// let count = Rc::clone(&vanishing);
    /// let mut monitor = HealthMonitor::new(Health {
    ///     sink: Some(Box::new(move |w: &Warning| {
    ///         if let Warning::Vanishing { .. } = w {
    ///             count.set(count.get() + 1);
    ///         }
    ///     })),
    ///     ..Health::default()
    /// });
    /// monitor.after_update(1, optimizer.last_stats(), &net.parameter_names()).unwrap();
    /// assert_eq!(vanishing.get(), 2);
    /// ```
    pub fn after_update(
        &mut self,
        step: u64,
        stats: &GradStats,
        names: &[(String, ParamId)],
    ) -> Result<(), Error> {
        if self.health.on_non_finite == OnNonFinite::Stop
            && let Some(id) = stats.first_non_finite()
        {
            let location = match names.iter().find(|(_, i)| *i == id) {
                Some((name, _)) => name.clone(),
                None => format!("{id:?}"),
            };
            return Err(self.stop(step, location, None, None));
        }
        if let Some(loss) = self.pending_loss.take() {
            self.record_loss(loss);
        }
        self.warned.resize(names.len(), 0);
        let (high, low) = (self.health.grad_norm_high, self.health.grad_norm_low);
        let converged = self.converged();
        for (k, (name, id)) in names.iter().enumerate() {
            // A sum of squares of finite `f32` values is finite in `f64`, so a norm that is
            // not finite means the gradient held a NaN or an infinity.
            let norm = stats.norm(*id);
            let bit = if !norm.is_finite() {
                WARNED_NON_FINITE
            } else if norm > high {
                WARNED_EXPLODING
            } else if norm < low {
                if converged {
                    continue;
                }
                WARNED_VANISHING
            } else {
                continue;
            };
            if self.warned[k] & bit != 0 {
                continue;
            }
            self.warned[k] |= bit;
            let name = name.clone();
            let warning = match bit {
                WARNED_NON_FINITE => Warning::NonFinite {
                    step,
                    location: name,
                    input: None,
                },
                WARNED_EXPLODING => Warning::Exploding {
                    step,
                    layer: name,
                    norm,
                },
                _ => Warning::Vanishing {
                    step,
                    layer: name,
                    norm,
                },
            };
            self.emit(warning);
        }
        Ok(())
    }

    /// Ends epoch `epoch`: with [`Health::dead_relu`], sends [`Warning::DeadNeurons`] for
    /// each ReLU or ReLU6 layer with neurons that output exactly 0 for every sample since the
    /// last `end_epoch` (masked-off neurons excepted), prints the summary lines of repeated
    /// norm warnings when there is no sink, then starts a new epoch, so each once-per-epoch
    /// warning can be sent again.
    ///
    /// # Errors
    ///
    /// Under [`OnNonFinite::Skip`], returns [`Error::NonFinite`] when
    /// [`before_update`](Self::before_update) kept every step of the epoch out (at least one
    /// step ran, and none trained): the parameters did not change, so every later epoch
    /// would be kept out too. The error names the epoch's last step and says so. The other
    /// end-of-epoch checks never stop training.
    ///
    /// ```
    /// use cavitation_nn::{Health, HealthMonitor};
    /// let mut monitor = HealthMonitor::new(Health::default());
    /// // No step ran this epoch, so there is nothing to report.
    /// assert!(monitor.end_epoch(1).is_ok());
    /// ```
    pub fn end_epoch(&mut self, epoch: usize) -> Result<(), Error> {
        if self.tracked {
            for k in 0..self.relu_layers.len() {
                let ReluLayer {
                    layer,
                    offset,
                    width,
                } = self.relu_layers[k];
                let alive = &self.alive[offset..offset + width];
                if alive.iter().all(|&a| a) {
                    continue;
                }
                let neurons = (0..width).filter(|&j| !alive[j]).collect();
                self.emit(Warning::DeadNeurons {
                    epoch,
                    layer,
                    width,
                    neurons,
                });
            }
        }
        self.print_repeats(epoch);
        let all_skipped = match self.skipped_this_epoch.take() {
            Some(step) if !std::mem::take(&mut self.trained_this_epoch) => Some(step),
            _ => None,
        };
        self.trained_this_epoch = false;
        self.alive.fill(false);
        self.tracked = false;
        self.loss_warned = false;
        self.clamped_warned = false;
        self.negative_warned = false;
        self.warned.fill(0);
        match all_skipped {
            Some(step) => Err(Error::NonFinite {
                step,
                location: format!("epoch {epoch}"),
                detail: Some(format!(
                    "every step of epoch {epoch} had a NaN or infinite loss or gradient and was \
                     left out (OnNonFinite::Skip), so the parameters did not change and every \
                     later epoch would be left out too; try a lower learning rate or check the \
                     data"
                )),
            }),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the health checks, through `fit`, `Trainer` and the manual loop.

    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::{
        Batch, Batcher, Dataset, FitConfig, Linear, Loss, NeuronMask, Optimizer, OptimizerConfig,
        Relu, Sequential, Sgd, Sigmoid, Tanh, Targets, Trainer, fit,
    };
    use cavitation_core::{Error, Graph, ParamStore, Reduction, Rng, Tensor, TensorView};

    /// The list a collecting sink writes to.
    type Seen = Rc<RefCell<Vec<Warning>>>;

    /// A warning sink, as [`Health::sink`] holds it.
    type Sink = Box<dyn FnMut(&Warning)>;

    /// A sink that keeps a copy of every warning it receives, and the list it keeps them in.
    fn collecting_sink() -> (Seen, Sink) {
        let seen = Rc::new(RefCell::new(Vec::new()));
        let sink = Rc::clone(&seen);
        (
            seen,
            Box::new(move |w: &Warning| sink.borrow_mut().push(w.clone())),
        )
    }

    /// The bad value of `nan_dataset` as `Trainer::step` reports it: the first feature of the
    /// first row, with no dataset row given.
    fn nan_at_origin() -> BadInput {
        BadInput {
            position: 0,
            row: None,
            feature: 0,
            value: f32::NAN,
            target: false,
        }
    }

    /// Inputs for a step whose data the test does not care about: `x` and no targets.
    fn no_labels(x: &Tensor<f32>) -> StepInputs<'_> {
        StepInputs {
            x: x.view(),
            targets: TargetsView::Classes(&[]),
            rows: None,
        }
    }

    /// The default checks with `on_non_finite` and the given sink.
    fn health(on_non_finite: OnNonFinite, sink: Sink) -> Health {
        Health {
            on_non_finite,
            sink: Some(sink),
            ..Health::default()
        }
    }

    /// 64 rows of 4 features in `[-1, 1)` from `Rng::seed(4)` with class labels 0 to 2;
    /// the first feature of row 0 is NaN.
    fn nan_dataset() -> Dataset {
        let mut rng = Rng::seed(4);
        let mut x: Vec<f32> = (0..64 * 4).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        x[0] = f32::NAN;
        Dataset {
            x: Tensor::from_vec(x, &[64, 4]).unwrap(),
            targets: Targets::Classes((0..64).map(|i| i % 3).collect()),
        }
    }

    /// A 4 → 8 tanh → 3 classifier from `Rng::seed(6)`.
    fn classifier(params: &mut ParamStore) -> Sequential {
        let mut rng = Rng::seed(6);
        Sequential::new()
            .push(Linear::new(params, 4, 8, &mut rng))
            .push(Tanh)
            .push(Linear::new(params, 8, 3, &mut rng))
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

    /// Whether `err` is `Error::NonFinite` at `step` in `location`.
    fn is_non_finite(err: &Error, step: u64, location: &str) -> bool {
        matches!(err, Error::NonFinite { step: s, location: l, .. } if *s == step && l == location)
    }

    /// `fit` rejects a NaN input before training, so nothing is trained or changed.
    #[test]
    fn fit_rejects_nan_input() {
        let data = nan_dataset();
        let mut params = ParamStore::new();
        let net = classifier(&mut params);
        let before = parameter_bits(&net, &params);
        let err = fit(&net, &mut params, &data, FitConfig::default()).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "{err:?}");
        assert!(err.to_string().contains("row 0, column 0"), "{err}");
        assert_eq!(parameter_bits(&net, &params), before);
    }

    /// A NaN in the first input row (fed through `Trainer::step`, which does not check the
    /// data) stops training at step 1 with the loss named as the location, before the
    /// optimiser applies anything.
    #[test]
    fn nan_in_input_stops_training() {
        let data = nan_dataset();
        let mut params = ParamStore::new();
        let net = classifier(&mut params);
        let before = parameter_bits(&net, &params);
        let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::CrossEntropy, 32);
        let batcher = Batcher::new(64, 32, None);
        let mut batch = Batch::new(&data, 32);
        let err = trainer
            .step(&net, &mut params, batcher.gather(&data, 0..32, &mut batch))
            .unwrap_err();
        assert!(is_non_finite(&err, 1, "loss"), "{err:?}");
        assert_eq!(
            err.to_string(),
            "non-finite value at step 1 in loss: the batch's input held NaN at batch position 0, \
             feature 0"
        );
        assert_eq!(parameter_bits(&net, &params), before);
    }

    /// With `OnNonFinite::Warn` the sink receives the non-finite loss (once per epoch) and
    /// the non-finite gradients, and a manual loop of steps runs every epoch (`Trainer::step`
    /// does not check the data, as `fit` does).
    #[test]
    fn warn_mode_continues() {
        let data = nan_dataset();
        let mut params = ParamStore::new();
        let net = classifier(&mut params);
        let (seen, sink) = collecting_sink();
        let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::CrossEntropy, 32);
        trainer.set_health(health(OnNonFinite::Warn, sink));
        let batcher = Batcher::new(64, 32, None);
        let mut batch = Batch::new(&data, 32);
        for epoch in 1..=2 {
            for range in [0..32, 32..64] {
                trainer
                    .step(&net, &mut params, batcher.gather(&data, range, &mut batch))
                    .unwrap();
            }
            trainer.end_epoch(epoch).unwrap();
        }
        assert_eq!(trainer.steps(), 4);
        let seen = seen.borrow();
        assert_eq!(
            seen[0],
            Warning::NonFinite {
                step: 1,
                location: "loss".to_string(),
                input: Some(nan_at_origin())
            }
        );
        let loss_steps: Vec<u64> = seen
            .iter()
            .filter_map(|w| match w {
                Warning::NonFinite { step, location, .. } if location == "loss" => Some(*step),
                _ => None,
            })
            .collect();
        assert_eq!(loss_steps, [1, 3]);
        assert!(seen.contains(&Warning::NonFinite {
            step: 1,
            location: "0.weight".to_string(),
            input: None
        }));
    }

    /// One `Linear` 2 → 1 fitting a target of 1e4 from inputs [1, 1] with a learning rate of
    /// 1e-12, so nothing moves: the step-1 gradient norm of `0.weight` is far above 1e3, and
    /// the sink receives exactly one `Exploding` warning for it per epoch.
    #[test]
    fn exploding_gradient_warns_once_per_epoch() {
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 2, 1, &mut Rng::seed(0));
        let (w, b) = (layer.weight(), layer.bias().unwrap());
        let net = Sequential::new().push(layer);
        let y = f64::from(params.value(w).data().iter().sum::<f32>() + params.value(b).data()[0]);
        // Mean squared error over a batch of 2 rows: each weight's gradient is 2 (y − 1e4).
        let expected_norm = 2.0 * 2f64.sqrt() * (y - 1e4).abs();
        let data = Dataset {
            x: Tensor::full(&[4, 2], 1.0f32),
            targets: Targets::Values(Tensor::full(&[4, 1], 1e4f32)),
        };
        let (seen, sink) = collecting_sink();
        let config = FitConfig {
            epochs: 2,
            batch: 2,
            optimizer: OptimizerConfig::Sgd {
                lr: 1e-12,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::Mse,
            shuffle_seed: None,
            health: health(OnNonFinite::Warn, sink),
            ..FitConfig::default()
        };
        fit(&net, &mut params, &data, config).unwrap();
        let exploding: Vec<(u64, f64)> = seen
            .borrow()
            .iter()
            .filter_map(|w| match w {
                Warning::Exploding { step, layer, norm } if layer == "0.weight" => {
                    Some((*step, *norm))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            exploding.iter().map(|e| e.0).collect::<Vec<_>>(),
            [1, 3],
            "{exploding:?}"
        );
        let (_, norm) = exploding[0];
        assert!(norm > 1e3, "{norm}");
        assert!(
            (norm - expected_norm).abs() <= 1e-5 * expected_norm,
            "{norm} vs {expected_norm}"
        );
    }

    /// A 2 → 4 → ReLU → 1 network whose first bias is −100 (so the ReLU outputs 0 for every
    /// sample), with `masked` neurons of layer `mask_layer` (0, the `Linear`, or 1, the ReLU)
    /// turned off, and 8 rows to train on.
    fn dead_network(
        params: &mut ParamStore,
        masked: &[usize],
        mask_layer: usize,
    ) -> (Sequential, Dataset) {
        let mut rng = Rng::seed(3);
        let first = Linear::new(params, 2, 4, &mut rng);
        params
            .value_mut(first.bias().unwrap())
            .data_mut()
            .fill(-100.0);
        let mut net = Sequential::new()
            .push(first)
            .push(Relu)
            .push(Linear::new(params, 4, 1, &mut rng));
        if !masked.is_empty() {
            let mut mask = NeuronMask::all_on(4);
            for &n in masked {
                mask.set(n, false);
            }
            net.set_mask(mask_layer, mask);
        }
        let x: Vec<f32> = (0..16).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        let data = Dataset {
            x: Tensor::from_vec(x, &[8, 2]).unwrap(),
            targets: Targets::Values(Tensor::full(&[8, 1], 0.5f32)),
        };
        (net, data)
    }

    /// The `DeadNeurons` warnings in `seen`.
    fn dead_reports(seen: &Seen) -> Vec<Warning> {
        seen.borrow()
            .iter()
            .filter(|w| matches!(w, Warning::DeadNeurons { .. }))
            .cloned()
            .collect()
    }

    /// A ReLU layer whose input is always about −100 has every neuron reported dead at the
    /// epoch's end; a neuron turned off by a mask is not reported.
    #[test]
    fn dead_relu_reported() {
        for (masked, dead) in [(vec![], vec![0, 1, 2, 3]), (vec![2], vec![0, 1, 3])] {
            let mut params = ParamStore::new();
            let (net, data) = dead_network(&mut params, &masked, 1);
            let (seen, sink) = collecting_sink();
            let config = FitConfig {
                epochs: 1,
                batch: 4,
                optimizer: OptimizerConfig::Sgd {
                    lr: 0.1,
                    momentum: 0.0,
                    weight_decay: 0.0,
                },
                loss: Loss::Mse,
                health: health(OnNonFinite::Stop, sink),
                ..FitConfig::default()
            };
            fit(&net, &mut params, &data, config).unwrap();
            assert_eq!(
                dead_reports(&seen),
                [Warning::DeadNeurons {
                    epoch: 1,
                    layer: 1,
                    width: 4,
                    neurons: dead
                }],
                "masked {masked:?}"
            );
        }
    }

    /// On 2 → 4 → ReLU → Sigmoid → 1 with the first bias at −100 (every ReLU neuron dead),
    /// neurons 1 and 2 masked through `set_neuron_mask` sit on the `Sigmoid`, at the end of
    /// the run after the ReLU: they are off on purpose and not reported, while the unmasked
    /// dead neurons 0 and 3 still are.
    #[test]
    fn mask_after_the_relu_run_is_not_dead() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(3);
        let first = Linear::new(&mut params, 2, 4, &mut rng);
        params
            .value_mut(first.bias().unwrap())
            .data_mut()
            .fill(-100.0);
        let mut net = Sequential::new()
            .push(first)
            .push(Relu)
            .push(Sigmoid)
            .push(Linear::new(&mut params, 4, 1, &mut rng));
        net.set_neuron_mask(0, NeuronMask::from_off(4, &[1, 2]));
        assert!(net.mask(2).is_some() && net.mask(1).is_none());
        let x: Vec<f32> = (0..16).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        let data = Dataset {
            x: Tensor::from_vec(x, &[8, 2]).unwrap(),
            targets: Targets::Values(Tensor::full(&[8, 1], 0.5f32)),
        };
        let (seen, sink) = collecting_sink();
        let config = FitConfig {
            epochs: 1,
            batch: 4,
            optimizer: OptimizerConfig::Sgd {
                lr: 0.1,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::Mse,
            health: health(OnNonFinite::Stop, sink),
            ..FitConfig::default()
        };
        fit(&net, &mut params, &data, config).unwrap();
        assert_eq!(
            dead_reports(&seen),
            [Warning::DeadNeurons {
                epoch: 1,
                layer: 1,
                width: 4,
                neurons: vec![0, 3]
            }]
        );
    }

    /// `epochs` epochs of `fit` on [`dead_network`] with `health`, using SGD and mean squared
    /// error; the network stays dead every epoch.
    fn fit_dead_network(health: Health, epochs: usize) {
        let mut params = ParamStore::new();
        let (net, data) = dead_network(&mut params, &[], 1);
        let config = FitConfig {
            epochs,
            batch: 4,
            optimizer: OptimizerConfig::Sgd {
                lr: 0.1,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::Mse,
            health,
            ..FitConfig::default()
        };
        fit(&net, &mut params, &data, config).unwrap();
    }

    /// A sink receives the dead-neuron report of every epoch, not only the first.
    #[test]
    fn dead_relu_sink_receives_every_epoch() {
        let (seen, sink) = collecting_sink();
        fit_dead_network(health(OnNonFinite::Stop, sink), 3);
        let epochs: Vec<usize> = dead_reports(&seen)
            .iter()
            .map(|w| match w {
                Warning::DeadNeurons { epoch, .. } => *epoch,
                _ => unreachable!("dead_reports keeps only DeadNeurons"),
            })
            .collect();
        assert_eq!(epochs, [1, 2, 3]);
    }

    /// Without a sink, a layer that is dead from the first epoch on is printed once, with its
    /// count and share, not again every epoch: the test binary re-runs itself as a child that
    /// trains the dead network for 50 epochs, and its standard error has one such line.
    #[test]
    fn fully_dead_layer_prints_once() {
        let out = crate::test_child::run_alone("health::tests::fully_dead_printing_child", &[]);
        crate::test_child::assert_passed(&out);
        let stderr = String::from_utf8_lossy(&out.stderr);
        let dead: Vec<&str> = stderr
            .lines()
            .filter(|l| l.contains("(dead ReLU)"))
            .collect();
        assert_eq!(dead.len(), 1, "stderr:\n{stderr}");
        assert!(
            dead[0].starts_with("cavitation: warning: epoch 1: 4 of 4 (100%) neurons of layer 1"),
            "{}",
            dead[0]
        );
    }

    /// Helper for `fully_dead_layer_prints_once`: trains the dead network for 50 epochs with
    /// the default checks, which print to standard error. Ignored by default; it is run in a
    /// child process.
    #[test]
    #[ignore = "run in a child process by fully_dead_layer_prints_once"]
    fn fully_dead_printing_child() {
        fit_dead_network(Health::default(), 50);
    }

    /// A one-layer ReLU network of 20 neurons, trained by a manual loop of one step per epoch
    /// with the default checks, whose biases are lowered between epochs so that 1, 3, 6, 6 and
    /// 19 neurons (5%, 15%, 30%, 30%, 95%) are dead in epochs 1 to 5.
    fn thresholds_loop() {
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 2, 20, &mut Rng::seed(0));
        let (w, b) = (layer.weight(), layer.bias().unwrap());
        params.value_mut(w).data_mut().fill(1.0);
        let net = Sequential::new().push(layer).push(Relu);
        let mut monitor = HealthMonitor::new(Health::default());
        let (mut graph, mut trace) = (Graph::new(), crate::Trace::new());
        let x = Tensor::full(&[4, 2], 1.0f32);
        for (epoch, dead) in [1usize, 3, 6, 6, 19].into_iter().enumerate() {
            let bias = params.value_mut(b).data_mut();
            bias.fill(0.0);
            bias[..dead].fill(-100.0);
            let mut tape = graph.tape(&params.values);
            let input = tape.input(x.view());
            net.forward_traced(&mut tape, input, &mut trace);
            monitor
                .after_backward(epoch as u64 + 1, 1.0, &tape, &trace, &net, no_labels(&x))
                .unwrap();
            drop(tape);
            monitor.end_epoch(epoch + 1).unwrap();
        }
    }

    /// Without a sink, a layer's dead report is printed in the epoch its share passes a
    /// threshold (25%, 50%, 90%) it had not passed: with 5%, 15%, 30%, 30% and 95% dead in
    /// epochs 1 to 5, standard error has two dead lines, for epochs 3 and 5.
    #[test]
    fn dead_report_printed_at_thresholds() {
        let out = crate::test_child::run_alone("health::tests::thresholds_child", &[]);
        crate::test_child::assert_passed(&out);
        let stderr = String::from_utf8_lossy(&out.stderr);
        let dead: Vec<&str> = stderr
            .lines()
            .filter(|l| l.contains("(dead ReLU)"))
            .collect();
        assert_eq!(dead.len(), 2, "stderr:\n{stderr}");
        assert!(dead[0].contains("epoch 3: 6 of 20 (30%)"), "{}", dead[0]);
        assert!(dead[1].contains("epoch 5: 19 of 20 (95%)"), "{}", dead[1]);
    }

    /// Helper for `dead_report_printed_at_thresholds`. Ignored by default; it is run in a
    /// child process.
    #[test]
    #[ignore = "run in a child process by dead_report_printed_at_thresholds"]
    fn thresholds_child() {
        thresholds_loop();
    }

    /// `epochs` epochs of `fit` on one `Linear` 2 → 1 fitting a target of 1e4 with a learning
    /// rate of 1e-12, so nothing moves and both parameters' gradient norms (about 2.8e4 and
    /// 2e4) are above the exploding limit at every step.
    fn fit_exploding(health: Health, epochs: usize) {
        let mut params = ParamStore::new();
        let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
        let data = Dataset {
            x: Tensor::full(&[4, 2], 1.0f32),
            targets: Targets::Values(Tensor::full(&[4, 1], 1e4f32)),
        };
        let config = FitConfig {
            epochs,
            batch: 2,
            optimizer: OptimizerConfig::Sgd {
                lr: 1e-12,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::Mse,
            shuffle_seed: None,
            health,
            ..FitConfig::default()
        };
        fit(&net, &mut params, &data, config).unwrap();
    }

    /// Without a sink, an exploding parameter is printed once for the run, then each later
    /// epoch adds one summary line for the kind: three epochs give two per-parameter lines
    /// (`0.weight` and `0.bias`), then one summary line each for epochs 2 and 3.
    #[test]
    fn norm_warnings_print_once_then_summarise() {
        let out = crate::test_child::run_alone("health::tests::norm_printing_child", &[]);
        crate::test_child::assert_passed(&out);
        let stderr = String::from_utf8_lossy(&out.stderr);
        let lines: Vec<&str> = stderr
            .lines()
            .filter(|l| l.starts_with("cavitation: warning:"))
            .collect();
        assert_eq!(lines.len(), 4, "stderr:\n{stderr}");
        assert!(lines[0].contains("step 1:") && lines[0].contains("0.weight"));
        assert!(lines[1].contains("step 1:") && lines[1].contains("0.bias"));
        for (line, epoch) in lines[2..].iter().zip([2, 3]) {
            assert!(
                line.starts_with(&format!(
                    "cavitation: warning: epoch {epoch}: 2 parameters exploding again: 0.weight "
                )) && line.contains(", 0.bias "),
                "{line}"
            );
        }
    }

    /// Helper for `norm_warnings_print_once_then_summarise`. Ignored by default; it is run in
    /// a child process.
    #[test]
    #[ignore = "run in a child process by norm_warnings_print_once_then_summarise"]
    fn norm_printing_child() {
        fit_exploding(Health::default(), 3);
    }

    /// A sink still receives every warning of every epoch: three epochs of the exploding run
    /// give one `Exploding` for `0.weight` in each, at steps 1, 3 and 5.
    #[test]
    fn sink_receives_every_warning() {
        let (seen, sink) = collecting_sink();
        fit_exploding(health(OnNonFinite::Stop, sink), 3);
        let steps: Vec<u64> = seen
            .borrow()
            .iter()
            .filter_map(|w| match w {
                Warning::Exploding { step, layer, .. } if layer == "0.weight" => Some(*step),
                _ => None,
            })
            .collect();
        assert_eq!(steps, [1, 3, 5]);
    }

    /// Two parameters above the limit send two warnings, one for each, in the same epoch.
    #[test]
    fn exploding_limit_is_per_parameter() {
        let (seen, sink) = collecting_sink();
        fit_exploding(health(OnNonFinite::Stop, sink), 1);
        let names: Vec<String> = seen
            .borrow()
            .iter()
            .filter_map(|w| match w {
                Warning::Exploding { layer, .. } => Some(layer.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(names, ["0.weight", "0.bias"]);
    }

    /// `Health::quiet` keeps every check and `Stop`: a NaN loss still returns the error. It
    /// drops the warnings: a child that trains the dead network with it prints nothing.
    #[test]
    fn quiet_drops_warnings_but_stops() {
        let health = Health::quiet();
        assert_eq!(health.on_non_finite, OnNonFinite::Stop);
        assert!(health.dead_relu && health.sink.is_some());
        let data = nan_dataset();
        let mut params = ParamStore::new();
        let net = classifier(&mut params);
        let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::CrossEntropy, 32);
        trainer.set_health(Health::quiet());
        let batcher = Batcher::new(64, 32, None);
        let mut batch = Batch::new(&data, 32);
        let err = trainer
            .step(&net, &mut params, batcher.gather(&data, 0..32, &mut batch))
            .unwrap_err();
        assert!(is_non_finite(&err, 1, "loss"), "{err:?}");

        let out = crate::test_child::run_alone("health::tests::quiet_printing_child", &[]);
        crate::test_child::assert_passed(&out);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!stderr.contains("cavitation: warning"), "{stderr}");
    }

    /// Helper for `quiet_drops_warnings_but_stops`: trains the dead network with
    /// `Health::quiet()`. Ignored by default; it is run in a child process.
    #[test]
    #[ignore = "run in a child process by quiet_drops_warnings_but_stops"]
    fn quiet_printing_child() {
        fit_dead_network(Health::quiet(), 3);
    }

    /// A neuron masked on the `Linear` before a ReLU layer is not reported dead, as one masked
    /// on the ReLU layer itself is not.
    #[test]
    fn mask_on_linear_not_reported_dead() {
        let mut params = ParamStore::new();
        let (net, data) = dead_network(&mut params, &[2], 0);
        let (seen, sink) = collecting_sink();
        let config = FitConfig {
            epochs: 1,
            batch: 4,
            optimizer: OptimizerConfig::Sgd {
                lr: 0.1,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::Mse,
            health: health(OnNonFinite::Stop, sink),
            ..FitConfig::default()
        };
        fit(&net, &mut params, &data, config).unwrap();
        assert_eq!(
            dead_reports(&seen),
            [Warning::DeadNeurons {
                epoch: 1,
                layer: 1,
                width: 4,
                neurons: vec![0, 1, 3]
            }]
        );
    }

    /// An activation `f(x) = x` whose derivative is NaN above 0.
    struct NanGrad;

    impl crate::Activation<f32> for NanGrad {
        /// Returns `"nan_grad"`.
        fn name(&self) -> &str {
            "nan_grad"
        }
        /// `x`.
        fn f(&self, x: f32) -> f32 {
            x
        }
        /// NaN above 0, else 1.
        fn df(&self, x: f32, _y: f32) -> f32 {
            if x > 0.0 { f32::NAN } else { 1.0 }
        }
    }

    /// A finite loss with a NaN gradient reaching the first layer: `fit` stops with
    /// `Error::NonFinite` naming `0.weight`.
    #[test]
    fn fit_stops_on_non_finite_gradient() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(2);
        let first = Linear::new(&mut params, 2, 3, &mut rng);
        params.value_mut(first.weight()).data_mut().fill(1.0);
        let net = Sequential::new()
            .push(first)
            .push(crate::ActivationLayer::new(NanGrad))
            .push(Linear::new(&mut params, 3, 1, &mut rng));
        let data = Dataset {
            x: Tensor::full(&[8, 2], 1.0f32),
            targets: Targets::Values(Tensor::full(&[8, 1], 0.5f32)),
        };
        let config = FitConfig {
            epochs: 1,
            batch: 4,
            loss: Loss::Mse,
            ..FitConfig::default()
        };
        let err = fit(&net, &mut params, &data, config).unwrap_err();
        assert!(is_non_finite(&err, 1, "0.weight"), "{err:?}");
    }

    /// Every variant's accessors: `kind`, `step`, `epoch`, `location` and `layer`.
    #[test]
    fn warning_accessors() {
        let exploding = Warning::Exploding {
            step: 3,
            layer: "0.weight".to_string(),
            norm: 1e4,
        };
        assert_eq!(exploding.kind(), WarningKind::Exploding);
        assert_eq!(exploding.step(), Some(3));
        assert_eq!(exploding.epoch(), None);
        assert_eq!(exploding.location(), Some("0.weight"));
        assert_eq!(exploding.layer(), None);
        let dead = Warning::DeadNeurons {
            epoch: 2,
            layer: 1,
            width: 8,
            neurons: vec![0],
        };
        assert_eq!(dead.kind(), WarningKind::DeadNeurons);
        assert_eq!(dead.step(), None);
        assert_eq!(dead.epoch(), Some(2));
        assert_eq!(dead.location(), None);
        assert_eq!(dead.layer(), Some(1));
        let clamped = Warning::ClampedLogs { step: 4, count: 1 };
        assert_eq!(clamped.kind(), WarningKind::ClampedLogs);
        assert_eq!(
            (
                clamped.step(),
                clamped.epoch(),
                clamped.location(),
                clamped.layer()
            ),
            (Some(4), None, None, None)
        );
        let non_finite = Warning::NonFinite {
            step: 1,
            location: "loss".to_string(),
            input: None,
        };
        assert_eq!(non_finite.kind(), WarningKind::NonFinite);
        assert_eq!(
            (
                non_finite.step(),
                non_finite.epoch(),
                non_finite.location(),
                non_finite.layer()
            ),
            (Some(1), None, Some("loss"), None)
        );
        let vanishing = Warning::Vanishing {
            step: 9,
            layer: "1.bias".to_string(),
            norm: 0.0,
        };
        assert_eq!(vanishing.kind(), WarningKind::Vanishing);
        assert_eq!(
            (
                vanishing.step(),
                vanishing.epoch(),
                vanishing.location(),
                vanishing.layer()
            ),
            (Some(9), None, Some("1.bias"), None)
        );
    }

    /// A dead report of 247 of 256 neurons shows the count and share, the first ten indices
    /// and an ellipsis, and the remedies.
    #[test]
    fn dead_line_shows_share_and_ten_indices() {
        let line = Warning::DeadNeurons {
            epoch: 2,
            layer: 1,
            width: 256,
            neurons: (0..247).collect(),
        }
        .to_string();
        assert!(line.contains("247 of 256 (96%)"), "{line}");
        assert!(line.contains("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, …]"), "{line}");
        assert!(!line.contains("10,"), "{line}");
        for phrase in ["learning rate", "bias", "leaky"] {
            assert!(line.contains(phrase), "{phrase}: {line}");
        }
    }

    /// Every warning's text carries its hint; a norm of exactly 0 gets its own explanation.
    #[test]
    fn hints_in_messages() {
        let exploding = Warning::Exploding {
            step: 2,
            layer: "0.weight".to_string(),
            norm: 2.5e4,
        }
        .to_string();
        assert!(exploding.contains("Health::grad_norm_high"), "{exploding}");
        assert!(exploding.contains("lower the learning rate"), "{exploding}");
        let vanishing = |norm| {
            Warning::Vanishing {
                step: 2,
                layer: "2.bias".to_string(),
                norm,
            }
            .to_string()
        };
        let zero = vanishing(0.0);
        assert!(zero.contains("Health::grad_norm_low"), "{zero}");
        assert!(zero.contains("no gradient reached"), "{zero}");
        let small = vanishing(1e-9);
        assert!(small.contains("Health::grad_norm_low"), "{small}");
        assert!(!small.contains("no gradient reached"), "{small}");
        let clamped = Warning::ClampedLogs { step: 3, count: 2 }.to_string();
        assert!(clamped.contains("pass no gradient"), "{clamped}");
        assert!(
            clamped.contains("BinaryCrossEntropyWithLogits"),
            "{clamped}"
        );
    }

    /// Runs the manual loop over the first `steps` batches of 32 rows of `data` with a
    /// monitor, stopping at the first error the monitor returns. Returns the steps finished.
    fn manual_epoch(
        net: &Sequential,
        params: &mut ParamStore,
        data: &Dataset,
        monitor: &mut HealthMonitor,
        steps: u64,
    ) -> Result<u64, Error> {
        let Targets::Classes(labels) = &data.targets else {
            unreachable!("the set has class labels")
        };
        let names = net.parameter_names();
        let mut optimizer = Sgd::new(0.1);
        let mut graph = Graph::new();
        let mut trace = crate::Trace::new();
        let features = data.features();
        for step in 1..=steps {
            let rows = (step as usize - 1) * 32..step as usize * 32;
            let x = TensorView::from_slice(
                &data.x.data()[rows.start * features..rows.end * features],
                &[32, features],
            )
            .unwrap();
            let mut tape = graph.tape(&params.values);
            let xi = tape.input(x);
            let out = net.forward_traced(&mut tape, xi, &mut trace);
            let loss = tape.softmax_cross_entropy(out, &labels[rows.clone()], Reduction::Mean);
            let value = tape.value(loss).data()[0];
            tape.backward(loss, &mut params.grads);
            let inputs = StepInputs {
                x,
                targets: TargetsView::Classes(&labels[rows]),
                rows: None,
            };
            monitor.after_backward(step, value, &tape, &trace, net, inputs)?;
            drop(tape);
            optimizer.step(params);
            monitor.after_update(step, optimizer.last_stats(), &names)?;
            params.zero_grad();
        }
        monitor.end_epoch(1)?;
        Ok(steps)
    }

    /// In the manual loop the monitor's `after_backward` returns the non-finite loss of step 1
    /// as an error; without the NaN the same loop finishes.
    #[test]
    fn manual_loop_monitor_stops_on_nan() {
        let data = nan_dataset();
        let mut params = ParamStore::new();
        let net = classifier(&mut params);
        let before = parameter_bits(&net, &params);
        let mut monitor = HealthMonitor::new(Health::default());
        let err = manual_epoch(&net, &mut params, &data, &mut monitor, 2).unwrap_err();
        assert!(is_non_finite(&err, 1, "loss"), "{err:?}");
        assert_eq!(parameter_bits(&net, &params), before);

        // The stopped loop left the NaN gradients of step 1 in place, so start afresh.
        let mut params = ParamStore::new();
        let net = classifier(&mut params);
        let mut clean = nan_dataset();
        clean.x.data_mut()[0] = 0.25;
        let mut monitor = HealthMonitor::new(Health::default());
        assert_eq!(
            manual_epoch(&net, &mut params, &clean, &mut monitor, 2).unwrap(),
            2
        );
    }

    /// A non-finite gradient with a finite loss is reported by `after_update`, naming the
    /// parameter, both as the error and to the sink.
    #[test]
    fn non_finite_gradient_names_the_parameter() {
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 2, 1, &mut Rng::seed(0));
        let bias = layer.bias().unwrap();
        let net = Sequential::new().push(layer);
        params.grads.accumulate(bias).0.data_mut()[0] = f32::INFINITY;
        let mut optimizer = Sgd::new(0.1);
        optimizer.step(&mut params);
        let (seen, sink) = collecting_sink();
        let mut monitor = HealthMonitor::new(health(OnNonFinite::Stop, sink));
        let err = monitor
            .after_update(5, optimizer.last_stats(), &net.parameter_names())
            .unwrap_err();
        assert!(is_non_finite(&err, 5, "0.bias"), "{err:?}");
        assert_eq!(
            *seen.borrow(),
            [Warning::NonFinite {
                step: 5,
                location: "0.bias".to_string(),
                input: None
            }]
        );
    }

    /// Binary cross-entropy with one prediction of exactly 0 against a target of 1 gives the
    /// sink one `ClampedLogs` warning per epoch (two steps each), and training continues.
    #[test]
    fn clamped_bce_warns() {
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 1, 1, &mut Rng::seed(0));
        params.value_mut(layer.weight()).data_mut()[0] = 0.5;
        let net = Sequential::new().push(layer);
        // Predictions 0, 0.5, 0, 0.5: the zero against a target of 1 is clamped, and every
        // gradient is 0, so the predictions stay where they are.
        let data = Dataset {
            x: Tensor::from_vec(vec![0.0f32, 1.0, 0.0, 1.0], &[4, 1]).unwrap(),
            targets: Targets::Values(
                Tensor::from_vec(vec![1.0f32, 0.5, 1.0, 0.5], &[4, 1]).unwrap(),
            ),
        };
        let (seen, sink) = collecting_sink();
        let config = FitConfig {
            epochs: 2,
            batch: 2,
            optimizer: OptimizerConfig::Sgd {
                lr: 0.1,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::BinaryCrossEntropy,
            shuffle_seed: None,
            health: health(OnNonFinite::Stop, sink),
            ..FitConfig::default()
        };
        let history = fit(&net, &mut params, &data, config).unwrap();
        assert_eq!(history.epochs.len(), 2);
        let clamped: Vec<Warning> = seen
            .borrow()
            .iter()
            .filter(|w| matches!(w, Warning::ClampedLogs { .. }))
            .cloned()
            .collect();
        assert_eq!(
            clamped,
            [
                Warning::ClampedLogs { step: 1, count: 1 },
                Warning::ClampedLogs { step: 3, count: 1 }
            ]
        );
    }

    /// Under `OnNonFinite::Stop`, `Trainer::step` returns the error at the step that met
    /// it, skips the update, counts the step and sends the warning to the sink; nothing is
    /// held back for `Trainer::end_epoch`.
    #[test]
    fn trainer_step_returns_the_error_at_the_step() {
        let data = nan_dataset();
        let mut params = ParamStore::new();
        let net = classifier(&mut params);
        let before = parameter_bits(&net, &params);
        let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::CrossEntropy, 32);
        let (seen, sink) = collecting_sink();
        trainer.set_health(health(OnNonFinite::Stop, sink));
        let batcher = Batcher::new(64, 32, None);
        let mut batch = Batch::new(&data, 32);
        let err = trainer
            .step(&net, &mut params, batcher.gather(&data, 0..32, &mut batch))
            .unwrap_err();
        assert!(is_non_finite(&err, 1, "loss"), "{err:?}");
        assert_eq!(parameter_bits(&net, &params), before);
        assert_eq!(trainer.steps(), 1);
        assert_eq!(
            *seen.borrow(),
            [Warning::NonFinite {
                step: 1,
                location: "loss".to_string(),
                input: Some(nan_at_origin())
            }]
        );
        assert!(trainer.end_epoch(1).is_ok());
        // The second batch has no NaN, so the next step trains.
        let loss = trainer
            .step(&net, &mut params, batcher.gather(&data, 32..64, &mut batch))
            .unwrap();
        assert!(loss.is_finite());
        assert_ne!(parameter_bits(&net, &params), before);
    }

    /// A `Linear` 1 → 1 fitting `y = 2x + 1` on 8 points evenly spaced in `[0, 1]` (so the
    /// weight and the bias are coupled; full batch, SGD with a learning rate of 0.5, mean
    /// squared error, 400 epochs) reaches a mean loss below 1e-8, and its gradients fall below
    /// the vanishing limit only because it has converged: the sink receives no `Vanishing`
    /// over the whole run.
    #[test]
    fn converged_regression_sends_no_vanishing() {
        let mut params = ParamStore::new();
        let net = Sequential::new().push(Linear::new(&mut params, 1, 1, &mut Rng::seed(0)));
        let xs: Vec<f32> = (0..8).map(|i| i as f32 / 7.0).collect();
        let ys: Vec<f32> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
        let data = Dataset {
            x: Tensor::from_vec(xs, &[8, 1]).unwrap(),
            targets: Targets::Values(Tensor::from_vec(ys, &[8, 1]).unwrap()),
        };
        let (seen, sink) = collecting_sink();
        let config = FitConfig {
            epochs: 400,
            batch: 8,
            optimizer: OptimizerConfig::Sgd {
                lr: 0.5,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::Mse,
            shuffle_seed: None,
            health: health(OnNonFinite::Stop, sink),
            ..FitConfig::default()
        };
        let history = fit(&net, &mut params, &data, config).unwrap();
        let last = history.epochs.last().unwrap().mean_loss;
        assert!(last < 1e-8, "{last}");
        let vanishing: Vec<Warning> = seen
            .borrow()
            .iter()
            .filter(|w| matches!(w, Warning::Vanishing { .. }))
            .cloned()
            .collect();
        assert!(
            vanishing.is_empty(),
            "{} warnings, first {:?}",
            vanishing.len(),
            vanishing.first()
        );
    }

    /// A gradient of 0 with a loss that has only fallen from 1.0 to 0.9 is a vanishing
    /// gradient: `after_update` still sends `Vanishing` for both parameters of a `Linear`
    /// 2 → 1.
    #[test]
    fn vanishing_still_fires_when_loss_has_not_fallen() {
        let mut params = ParamStore::new();
        let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
        let names = net.parameter_names();
        let mut graph = Graph::new();
        let trace = crate::Trace::new();
        let mut optimizer = Sgd::new(0.1);
        optimizer.step(&mut params);
        let (seen, sink) = collecting_sink();
        let mut monitor = HealthMonitor::new(health(OnNonFinite::Stop, sink));
        let tape = graph.tape(&params.values);
        let x = Tensor::full(&[1, 1], 0.0f32);
        for (step, loss) in [(1, 1.0), (2, 0.9)] {
            monitor
                .after_backward(step, loss, &tape, &trace, &net, no_labels(&x))
                .unwrap();
        }
        monitor
            .after_update(2, optimizer.last_stats(), &names)
            .unwrap();
        let seen = seen.borrow();
        let vanishing = seen
            .iter()
            .filter(|w| matches!(w, Warning::Vanishing { .. }))
            .count();
        assert_eq!(vanishing, 2, "{seen:?}");
    }

    /// A loss that falls from 1 to exactly 0 counts as converged: zero gradients then send no
    /// `Vanishing`, with no division by the loss.
    #[test]
    fn zero_loss_counts_as_converged() {
        let mut params = ParamStore::new();
        let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
        let names = net.parameter_names();
        let mut graph = Graph::new();
        let trace = crate::Trace::new();
        let mut optimizer = Sgd::new(0.1);
        optimizer.step(&mut params);
        let (seen, sink) = collecting_sink();
        let mut monitor = HealthMonitor::new(health(OnNonFinite::Stop, sink));
        let tape = graph.tape(&params.values);
        let x = Tensor::full(&[1, 1], 0.0f32);
        for (step, loss) in [(1, 1.0), (2, 0.0)] {
            monitor
                .after_backward(step, loss, &tape, &trace, &net, no_labels(&x))
                .unwrap();
        }
        monitor
            .after_update(2, optimizer.last_stats(), &names)
            .unwrap();
        assert!(seen.borrow().is_empty(), "{:?}", seen.borrow());
    }

    /// A negative first loss (possible with `kl_div` on inputs that are not log-probabilities)
    /// never counts as converged, so zero gradients still send `Vanishing`.
    #[test]
    fn negative_first_loss_never_counts_as_converged() {
        let mut params = ParamStore::new();
        let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
        let names = net.parameter_names();
        let mut graph = Graph::new();
        let trace = crate::Trace::new();
        let mut optimizer = Sgd::new(0.1);
        optimizer.step(&mut params);
        let (seen, sink) = collecting_sink();
        let mut monitor = HealthMonitor::new(health(OnNonFinite::Stop, sink));
        let tape = graph.tape(&params.values);
        let x = Tensor::full(&[1, 1], 0.0f32);
        for (step, loss) in [(1, -1.0), (2, -1.0)] {
            monitor
                .after_backward(step, loss, &tape, &trace, &net, no_labels(&x))
                .unwrap();
        }
        monitor
            .after_update(2, optimizer.last_stats(), &names)
            .unwrap();
        assert!(
            seen.borrow()
                .iter()
                .any(|w| matches!(w, Warning::Vanishing { .. })),
            "{:?}",
            seen.borrow()
        );
    }

    /// Each warning reads as one plain-English line.
    #[test]
    fn warnings_display() {
        let lines = [
            Warning::ClampedLogs { step: 3, count: 2 }.to_string(),
            Warning::NonFinite {
                step: 1,
                location: "loss".to_string(),
                input: None,
            }
            .to_string(),
            Warning::Exploding {
                step: 2,
                layer: "0.weight".to_string(),
                norm: 2.5e4,
            }
            .to_string(),
            Warning::Vanishing {
                step: 2,
                layer: "2.bias".to_string(),
                norm: 0.0,
            }
            .to_string(),
            Warning::DeadNeurons {
                epoch: 1,
                layer: 1,
                width: 4,
                neurons: vec![0, 3],
            }
            .to_string(),
        ];
        assert_eq!(
            lines,
            [
                "step 3: binary cross-entropy met 2 probabilities of exactly 0 or 1 and clamped their \
                 logarithms to -100; the clamped rows pass no gradient, so use \
                 Loss::BinaryCrossEntropyWithLogits on raw scores instead",
                "step 1: a NaN or infinite value in loss",
                "step 2: the gradient norm of 0.weight is 2.500e4, above the exploding limit \
                 (Health::grad_norm_high, 1e3 by default); lower the learning rate, or raise the \
                 limit if this size is normal for the model",
                "step 2: the gradient norm of 2.bias is 0.000e0, below the vanishing limit \
                 (Health::grad_norm_low, 1e-7 by default; a gradient this small with a loss that \
                 has not fallen 10,000-fold is a vanishing gradient); a norm of exactly 0 means no \
                 gradient reached it, which points at dead ReLUs or a saturated output",
                "epoch 1: 2 of 4 (50%) neurons of layer 1 output 0 for every sample (dead ReLU): \
                 [0, 3]; try a lower learning rate, a positive bias initialisation or a leaky \
                 activation such as LeakyRelu",
            ]
        );
    }

    /// `Trainer::step` on 8 rows with a NaN at batch position 3, feature 2 stops with an error
    /// whose detail names the position and the feature, and the sink gets the same bad value.
    #[test]
    fn non_finite_loss_names_the_input() {
        let mut rng = Rng::seed(4);
        let mut x: Vec<f32> = (0..8 * 4).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        x[3 * 4 + 2] = f32::NAN;
        let data = Dataset {
            x: Tensor::from_vec(x, &[8, 4]).unwrap(),
            targets: Targets::Classes((0..8).map(|i| i % 3).collect()),
        };
        let mut params = ParamStore::new();
        let net = classifier(&mut params);
        let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::CrossEntropy, 8);
        let (seen, sink) = collecting_sink();
        trainer.set_health(health(OnNonFinite::Stop, sink));
        let mut batch = Batch::new(&data, 8);
        let view = Batcher::new(8, 8, None).gather(&data, 0..8, &mut batch);
        let err = trainer.step(&net, &mut params, view).unwrap_err();
        let Error::NonFinite {
            step,
            location,
            detail: Some(detail),
        } = &err
        else {
            panic!("expected a non-finite error with a detail, got {err:?}");
        };
        assert_eq!((*step, location.as_str()), (1, "loss"));
        assert!(detail.contains("position 3"), "{detail}");
        assert!(detail.contains("feature 2"), "{detail}");
        let bad = BadInput {
            position: 3,
            row: None,
            feature: 2,
            value: f32::NAN,
            target: false,
        };
        assert_eq!(
            *seen.borrow(),
            [Warning::NonFinite {
                step: 1,
                location: "loss".to_string(),
                input: Some(bad)
            }]
        );
    }

    /// An infinite value target is named as a target, after the inputs (all finite) are
    /// checked; the scan reads only a non-finite loss's batch.
    #[test]
    fn non_finite_target_is_named_as_a_target() {
        let mut params = ParamStore::new();
        let net = Sequential::new().push(Linear::new(&mut params, 2, 1, &mut Rng::seed(0)));
        let mut targets = vec![0.5f32; 4];
        targets[1] = f32::INFINITY;
        let data = Dataset {
            x: Tensor::full(&[4, 2], 1.0f32),
            targets: Targets::Values(Tensor::from_vec(targets, &[4, 1]).unwrap()),
        };
        let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 4);
        trainer.set_health(Health::quiet());
        let mut batch = Batch::new(&data, 4);
        let view = Batcher::new(4, 4, None).gather(&data, 0..4, &mut batch);
        let err = trainer.step(&net, &mut params, view).unwrap_err();
        let text = err.to_string();
        assert!(
            text.ends_with("the batch's targets held infinity at batch position 1, output 0"),
            "{text}"
        );
    }

    /// The 64 → 256 → 128 → 10 ReLU network on finite synthetic data (values up to 1000) with
    /// SGD at a learning rate of 10 blows up; the error says the inputs were finite and
    /// suggests a lower learning rate rather than naming an input.
    #[test]
    fn overflow_says_inputs_finite() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(1);
        let net = Sequential::new()
            .push(Linear::new(&mut params, 64, 256, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, 256, 128, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, 128, 10, &mut rng));
        let x: Vec<f32> = (0..128 * 64)
            .map(|_| rng.range(0.0, 1000.0) as f32)
            .collect();
        let data = Dataset {
            x: Tensor::from_vec(x, &[128, 64]).unwrap(),
            targets: Targets::Classes((0..128).map(|i| i % 10).collect()),
        };
        let config = FitConfig {
            epochs: 30,
            batch: 32,
            optimizer: OptimizerConfig::Sgd {
                lr: 10.0,
                momentum: 0.0,
                weight_decay: 0.0,
            },
            loss: Loss::CrossEntropy,
            health: Health::quiet(),
            ..FitConfig::default()
        };
        let err = fit(&net, &mut params, &data, config).unwrap_err();
        let Error::NonFinite {
            location,
            detail: Some(detail),
            ..
        } = &err
        else {
            panic!("expected a non-finite error with a detail, got {err:?}");
        };
        assert_eq!(location, "loss", "{err}");
        assert!(detail.contains("inputs were finite"), "{detail}");
        assert!(detail.contains("learning rate"), "{detail}");
    }

    /// A `Linear` 2 → 2 with zero parameters under `Loss::Nll` (raw scores, not
    /// log-probabilities): the loss is 0 at step 1 and falls below 0 from step 2, so the sink
    /// receives `NegativeLoss` at step 2 and, after `end_epoch`, at the first step of the next
    /// epoch, and no other step sends one.
    #[test]
    fn negative_loss_warns() {
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 2, 2, &mut Rng::seed(0));
        for id in [layer.weight(), layer.bias().unwrap()] {
            params.value_mut(id).data_mut().fill(0.0);
        }
        let net = Sequential::new().push(layer);
        let data = Dataset {
            x: Tensor::full(&[4, 2], 1.0f32),
            targets: Targets::Classes(vec![0; 4]),
        };
        let mut trainer = Trainer::new(&net, Sgd::new(0.5), Loss::Nll, 4);
        let (seen, sink) = collecting_sink();
        trainer.set_health(health(OnNonFinite::Stop, sink));
        let batcher = Batcher::new(4, 4, None);
        let mut batch = Batch::new(&data, 4);
        let mut losses = Vec::new();
        for epoch in 1..=2 {
            for _ in 0..3 {
                let view = batcher.gather(&data, 0..4, &mut batch);
                losses.push(trainer.step(&net, &mut params, view).unwrap());
            }
            trainer.end_epoch(epoch).unwrap();
        }
        assert_eq!(losses[0], 0.0);
        assert!(losses[1..].iter().all(|&l| l < 0.0), "{losses:?}");
        let warnings = seen.borrow();
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert!(
            matches!(warnings[0], Warning::NegativeLoss { step: 2, loss } if loss == losses[1])
        );
        assert!(
            matches!(warnings[1], Warning::NegativeLoss { step: 4, loss } if loss == losses[3])
        );
        assert_eq!(warnings[0].kind(), WarningKind::NegativeLoss);
        assert_eq!(warnings[0].step(), Some(2));
    }

    /// `Loss::KlDiv` can be negative on valid use of the library's own conventions, and a
    /// monitor built without a loss does not know the loss: neither sends `NegativeLoss`.
    #[test]
    fn negative_loss_is_silent_when_not_impossible() {
        let params = ParamStore::new();
        let net = Sequential::new();
        let mut graph = Graph::new();
        let tape = graph.tape(&params.values);
        let trace = crate::Trace::new();
        let x = Tensor::full(&[1, 1], 0.0f32);
        for monitor in [
            HealthMonitor::new(Health::quiet()).with_loss(Loss::KlDiv),
            HealthMonitor::new(Health::quiet()),
        ] {
            let mut monitor = monitor;
            let (seen, sink) = collecting_sink();
            monitor.health.sink = Some(sink);
            monitor
                .after_backward(1, -1.0, &tape, &trace, &net, no_labels(&x))
                .unwrap();
            assert!(seen.borrow().is_empty(), "{:?}", seen.borrow());
        }
    }

    /// Without a sink, a negative loss prints its full line once for the run, then one summary
    /// line in each later epoch where it happens again: three epochs of three steps give one
    /// full line (step 2) and summaries for epochs 2 and 3.
    #[test]
    fn negative_loss_prints_once_then_summarises() {
        let out = crate::test_child::run_alone("health::tests::negative_loss_printing_child", &[]);
        crate::test_child::assert_passed(&out);
        let stderr = String::from_utf8_lossy(&out.stderr);
        let lines: Vec<&str> = stderr
            .lines()
            .filter(|l| l.starts_with("cavitation: warning:"))
            .collect();
        assert_eq!(lines.len(), 3, "stderr:\n{stderr}");
        assert!(lines[0].contains("step 2:") && lines[0].contains("LogSoftmax"));
        for (line, epoch) in lines[1..].iter().zip([2, 3]) {
            assert!(
                line.starts_with(&format!(
                    "cavitation: warning: epoch {epoch}: the loss was negative again"
                )),
                "{line}"
            );
        }
    }

    /// Helper for `negative_loss_prints_once_then_summarises`: trains the zero-parameter `Nll`
    /// model for three epochs of three steps with the default checks, which print to standard
    /// error. Ignored by default; it is run in a child process.
    #[test]
    #[ignore = "run in a child process by negative_loss_prints_once_then_summarises"]
    fn negative_loss_printing_child() {
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 2, 2, &mut Rng::seed(0));
        for id in [layer.weight(), layer.bias().unwrap()] {
            params.value_mut(id).data_mut().fill(0.0);
        }
        let net = Sequential::new().push(layer);
        let data = Dataset {
            x: Tensor::full(&[4, 2], 1.0f32),
            targets: Targets::Classes(vec![0; 4]),
        };
        let mut trainer = Trainer::new(&net, Sgd::new(0.5), Loss::Nll, 4);
        let batcher = Batcher::new(4, 4, None);
        let mut batch = Batch::new(&data, 4);
        for epoch in 1..=3 {
            for _ in 0..3 {
                let view = batcher.gather(&data, 0..4, &mut batch);
                trainer.step(&net, &mut params, view).unwrap();
            }
            trainer.end_epoch(epoch).unwrap();
        }
    }

    /// The accessors and the text of `NegativeLoss`, and of a non-finite loss that names its
    /// input.
    #[test]
    fn negative_loss_and_input_text() {
        let w = Warning::NegativeLoss {
            step: 7,
            loss: -0.5,
        };
        assert_eq!(
            (w.kind(), w.step(), w.epoch(), w.location(), w.layer()),
            (WarningKind::NegativeLoss, Some(7), None, None, None)
        );
        let text = w.to_string();
        assert!(
            text.starts_with("step 7: the loss is -5.0000e-1, below 0"),
            "{text}"
        );
        assert!(
            text.contains("Loss::Nll") && text.contains("LogSoftmax"),
            "{text}"
        );
        assert!(
            text.contains("binary cross-entropy targets outside [0, 1]"),
            "{text}"
        );
        let named = Warning::NonFinite {
            step: 1,
            location: "loss".to_string(),
            input: Some(BadInput {
                position: 3,
                row: Some(17),
                feature: 2,
                value: f32::NEG_INFINITY,
                target: false,
            }),
        };
        assert_eq!(
            named.to_string(),
            "step 1: a NaN or infinite value in loss; the batch's input held -infinity at batch \
             position 3, feature 2 (row 17 of the dataset)"
        );
    }

    /// 96 rows of 4 features in `[-1, 1)` from `Rng::seed(9)` with class labels 0 to 2; the
    /// third feature of row 40 (in the second batch of 32) is NaN.
    fn one_nan_row() -> Dataset {
        let mut rng = Rng::seed(9);
        let mut x: Vec<f32> = (0..96 * 4).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        x[40 * 4 + 2] = f32::NAN;
        Dataset {
            x: Tensor::from_vec(x, &[96, 4]).unwrap(),
            targets: Targets::Classes((0..96).map(|i| i % 3).collect()),
        }
    }

    /// The optimisers the `Skip` comparisons run: Adam, whose bias correction counts its
    /// steps, and SGD with momentum, whose buffer carries every update into the next.
    fn skip_optimizers() -> [Box<dyn Optimizer<f32>>; 2] {
        [
            Box::new(crate::Adam::new(0.01)),
            Box::new(Sgd::new(0.1).with_momentum(0.9)),
        ]
    }

    /// Trains `classifier` with `Trainer::step` for 3 epochs over the batches `ranges` of
    /// `data` (in that order, unshuffled) under `health`, ending each epoch, and returns
    /// the parameter bits and the trainer.
    fn train_ranges(
        optimizer: Box<dyn Optimizer<f32>>,
        health: Health,
        data: &Dataset,
        ranges: &[std::ops::Range<usize>],
    ) -> (Vec<u32>, Trainer) {
        let mut params = ParamStore::new();
        let net = classifier(&mut params);
        let mut trainer = Trainer::new(&net, optimizer, Loss::CrossEntropy, 32);
        trainer.set_health(health);
        let batcher = Batcher::new(data.len(), 32, None);
        let mut batch = Batch::new(data, 32);
        for epoch in 1..=3 {
            for range in ranges {
                let view = batcher.gather(data, range.clone(), &mut batch);
                trainer.step(&net, &mut params, view).unwrap();
            }
            trainer.end_epoch(epoch).unwrap();
        }
        (parameter_bits(&net, &params), trainer)
    }

    /// Under `OnNonFinite::Skip`, a run over one NaN row ends with finite parameters equal,
    /// bit for bit, to the same run without that row's batch, with Adam and with SGD with
    /// momentum: the skipped step changes neither the parameters nor the optimiser's state
    /// (Adam's step count included). The sink gets the non-finite loss once per epoch, the
    /// step returns its NaN loss as `Ok`, and the trainer counts the step and the skip.
    #[test]
    fn skip_matches_training_without_the_bad_step() {
        let data = one_nan_row();
        for (with, without) in skip_optimizers().into_iter().zip(skip_optimizers()) {
            let (seen, sink) = collecting_sink();
            let all = [0..32, 32..64, 64..96];
            let (skipped, trainer) =
                train_ranges(with, health(OnNonFinite::Skip, sink), &data, &all);
            let (_, quiet) = collecting_sink();
            let (clean, _) = train_ranges(
                without,
                health(OnNonFinite::Stop, quiet),
                &data,
                &[0..32, 64..96],
            );
            assert_eq!(skipped, clean);
            assert!(skipped.iter().all(|&b| f32::from_bits(b).is_finite()));
            assert_eq!((trainer.steps(), trainer.skipped_steps()), (9, 3));
            let loss_steps: Vec<u64> = seen
                .borrow()
                .iter()
                .filter_map(|w| match w {
                    Warning::NonFinite { step, location, .. } if location == "loss" => Some(*step),
                    _ => None,
                })
                .collect();
            assert_eq!(loss_steps, [2, 5, 8]);
            assert!(
                seen.borrow()
                    .iter()
                    .all(|w| w.kind() != WarningKind::NonFinite || w.location() == Some("loss")),
                "{:?}",
                seen.borrow()
            );
        }
    }

    /// A skipped step returns `Ok` with its loss, NaN here, and leaves the parameters as they
    /// were.
    #[test]
    fn skipped_step_returns_its_loss() {
        let data = one_nan_row();
        let mut params = ParamStore::new();
        let net = classifier(&mut params);
        let before = parameter_bits(&net, &params);
        let (_, sink) = collecting_sink();
        let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::CrossEntropy, 32);
        trainer.set_health(health(OnNonFinite::Skip, sink));
        let batcher = Batcher::new(96, 32, None);
        let mut batch = Batch::new(&data, 32);
        let view = batcher.gather(&data, 32..64, &mut batch);
        let loss = trainer.step(&net, &mut params, view).unwrap();
        assert!(loss.is_nan());
        assert_eq!(parameter_bits(&net, &params), before);
        assert_eq!((trainer.steps(), trainer.skipped_steps()), (1, 1));
    }

    /// An activation `f(x) = x` whose derivative is NaN where `|x|` is above 50, else 1.
    struct NanGradBeyond;

    impl crate::Activation<f32> for NanGradBeyond {
        /// Returns `"nan_grad_beyond"`.
        fn name(&self) -> &str {
            "nan_grad_beyond"
        }
        /// `x`.
        fn f(&self, x: f32) -> f32 {
            x
        }
        /// NaN where `|x|` is above 50, else 1.
        fn df(&self, x: f32, _y: f32) -> f32 {
            if x.abs() > 50.0 { f32::NAN } else { 1.0 }
        }
    }

    /// A 2 → 3 → `NanGradBeyond` → 1 network from `Rng::seed(5)`.
    fn nan_grad_beyond_net(params: &mut ParamStore) -> Sequential {
        let mut rng = Rng::seed(5);
        Sequential::new()
            .push(Linear::new(params, 2, 3, &mut rng))
            .push(crate::ActivationLayer::new(NanGradBeyond))
            .push(Linear::new(params, 3, 1, &mut rng))
    }

    /// `rows` rows of 2 features in `[-1, 1)` from `Rng::seed(8)` with value targets; with
    /// `bad`, rows 4 to 7 (the second batch of 4) have inputs of 1e3 instead, so their loss
    /// is finite but their gradient is NaN.
    fn finite_loss_nan_grad_data(bad: bool) -> Dataset {
        let mut rng = Rng::seed(8);
        let mut x: Vec<f32> = (0..12 * 2).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        let mut y: Vec<f32> = (0..12).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        if bad {
            x[8..16].fill(1e3);
        } else {
            x.drain(8..16);
            y.drain(4..8);
        }
        let rows = y.len();
        Dataset {
            x: Tensor::from_vec(x, &[rows, 2]).unwrap(),
            targets: Targets::Values(Tensor::from_vec(y, &[rows, 1]).unwrap()),
        }
    }

    /// The `fit` configuration of the `Skip` comparison: 3 unshuffled epochs of Adam in
    /// batches of 4, mean squared error, `health`.
    fn skip_fit_config(health: Health) -> FitConfig {
        FitConfig {
            epochs: 3,
            batch: 4,
            optimizer: OptimizerConfig::Adam {
                lr: 0.01,
                weight_decay: 0.0,
            },
            loss: Loss::Mse,
            shuffle_seed: None,
            health,
            ..FitConfig::default()
        }
    }

    /// Under `Skip`, `fit` over a batch whose loss is finite but whose gradient is NaN skips
    /// that step each epoch and ends with the parameters, and the epoch reports, of `fit` on
    /// the data without that batch, bit for bit. Each skipped step's rows are left out of
    /// `rows` and `mean_loss`, and the sink names the parameter once per epoch.
    #[test]
    fn fit_skip_matches_fit_without_the_bad_batch() {
        let mut params = ParamStore::new();
        let net = nan_grad_beyond_net(&mut params);
        let (seen, sink) = collecting_sink();
        let bad = finite_loss_nan_grad_data(true);
        let config = skip_fit_config(health(OnNonFinite::Skip, sink));
        let skipped = fit(&net, &mut params, &bad, config).unwrap();

        let mut clean_params = ParamStore::new();
        let clean_net = nan_grad_beyond_net(&mut clean_params);
        let clean = finite_loss_nan_grad_data(false);
        let (_, quiet) = collecting_sink();
        let config = skip_fit_config(health(OnNonFinite::Stop, quiet));
        let reference = fit(&clean_net, &mut clean_params, &clean, config).unwrap();

        let bits = parameter_bits(&net, &params);
        assert_eq!(bits, parameter_bits(&clean_net, &clean_params));
        assert!(bits.iter().all(|&b| f32::from_bits(b).is_finite()));
        for (a, b) in skipped.epochs.iter().zip(&reference.epochs) {
            assert_eq!((a.epoch, a.rows), (b.epoch, b.rows));
            assert_eq!(a.rows, 8);
            assert_eq!(a.mean_loss.to_bits(), b.mean_loss.to_bits());
        }
        let non_finite: Vec<(u64, String)> = seen
            .borrow()
            .iter()
            .filter_map(|w| match w {
                Warning::NonFinite { step, location, .. } => Some((*step, location.clone())),
                _ => None,
            })
            .collect();
        let at = |step| (step, "0.weight".to_string());
        assert!(non_finite.contains(&at(2)), "{non_finite:?}");
        assert!(non_finite.contains(&at(5)), "{non_finite:?}");
        assert!(non_finite.contains(&at(8)), "{non_finite:?}");
        assert!(non_finite.iter().all(|(s, _)| [2, 5, 8].contains(s)));
    }

    /// Under `Skip`, `fit_controlled` runs the control after each step that trained, not
    /// after the skipped one, and the step numbers it sees count the skipped step.
    #[test]
    fn control_does_not_run_after_a_skipped_step() {
        /// Records the step number of every `after_step` call.
        struct Steps(Vec<u64>);
        impl crate::StepControl for Steps {
            /// Records the step's number and continues.
            fn after_step(
                &mut self,
                info: &crate::StepInfo,
                _model: &mut crate::ModelHandle<'_>,
                _params: &ParamStore,
                _trainer: &mut crate::TrainerHandle<'_>,
            ) -> crate::Flow {
                self.0.push(info.step);
                crate::Flow::Continue
            }
        }
        let mut params = ParamStore::new();
        let mut net = nan_grad_beyond_net(&mut params);
        let (_, sink) = collecting_sink();
        let mut control = Steps(Vec::new());
        let history = crate::fit_controlled(
            &mut net,
            &mut params,
            &finite_loss_nan_grad_data(true),
            skip_fit_config(health(OnNonFinite::Skip, sink)),
            &mut crate::Hooks::new(),
            &mut control,
        )
        .unwrap();
        assert_eq!(control.0, [1, 3, 4, 6, 7, 9]);
        assert!(history.epochs.iter().all(|e| e.rows == 8));
    }

    /// Under `Skip`, an epoch whose every step is skipped ends the run with
    /// `Error::NonFinite` naming the epoch and its last step, and the parameters stay as they
    /// were.
    #[test]
    fn fit_skip_every_step_ends_the_run() {
        let mut params = ParamStore::new();
        let net = nan_grad_beyond_net(&mut params);
        let before = parameter_bits(&net, &params);
        let data = Dataset {
            x: Tensor::full(&[8, 2], 1e3f32),
            targets: Targets::Values(Tensor::full(&[8, 1], 0.5f32)),
        };
        let (_, sink) = collecting_sink();
        let err = fit(
            &net,
            &mut params,
            &data,
            skip_fit_config(health(OnNonFinite::Skip, sink)),
        )
        .unwrap_err();
        assert!(is_non_finite(&err, 2, "epoch 1"), "{err:?}");
        assert!(err.to_string().contains("every step of epoch 1"), "{err}");
        assert_eq!(parameter_bits(&net, &params), before);
    }

    /// An epoch with at least one trained step does not end the run, however many steps
    /// were skipped; an epoch with no step at all does not either.
    #[test]
    fn end_epoch_errs_only_when_nothing_trained() {
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 2, 1, &mut Rng::seed(0));
        let names = Sequential::new().push(layer).parameter_names();
        let (_, sink) = collecting_sink();
        let mut monitor = HealthMonitor::new(health(OnNonFinite::Skip, sink));
        monitor.end_epoch(1).unwrap();
        assert!(!monitor.before_update(1, f32::NAN, &params, &names));
        assert!(monitor.before_update(2, 0.5, &params, &names));
        monitor.end_epoch(2).unwrap();
        assert!(!monitor.before_update(3, f32::NAN, &params, &names));
        let err = monitor.end_epoch(3).unwrap_err();
        assert!(is_non_finite(&err, 3, "epoch 3"), "{err:?}");
        // The next epoch starts afresh.
        assert!(monitor.before_update(4, 0.5, &params, &names));
        monitor.end_epoch(4).unwrap();
    }

    /// Under `Skip`, a skipped step whose loss is finite does not set the loss the vanishing
    /// check compares against: after a skipped first step with a loss of 1e6, a zero gradient
    /// at a loss of 1 is still reported as vanishing.
    #[test]
    fn skipped_step_sets_no_loss_baseline() {
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 2, 1, &mut Rng::seed(0));
        let bias = layer.bias().unwrap();
        let net = Sequential::new().push(layer);
        let names = net.parameter_names();
        let (seen, sink) = collecting_sink();
        let mut monitor = HealthMonitor::new(health(OnNonFinite::Skip, sink));
        let mut graph = Graph::new();
        let trace = crate::Trace::new();
        let x = Tensor::full(&[1, 1], 0.0f32);
        {
            let tape = graph.tape(&params.values);
            monitor
                .after_backward(1, 1e6, &tape, &trace, &net, no_labels(&x))
                .unwrap();
        }
        params.grads.accumulate(bias).0.data_mut()[0] = f32::NAN;
        assert!(!monitor.before_update(1, 1e6, &params, &names));
        params.zero_grad();
        {
            let tape = graph.tape(&params.values);
            monitor
                .after_backward(2, 1.0, &tape, &trace, &net, no_labels(&x))
                .unwrap();
        }
        assert!(monitor.before_update(2, 1.0, &params, &names));
        let mut optimizer = Sgd::new(0.1);
        optimizer.step(&mut params);
        monitor
            .after_update(2, optimizer.last_stats(), &names)
            .unwrap();
        let vanishing = seen
            .borrow()
            .iter()
            .filter(|w| matches!(w, Warning::Vanishing { .. }))
            .count();
        assert_eq!(vanishing, 2, "{:?}", seen.borrow());
    }

    /// A profile asked for before a skipped step carries over to the next step that trains.
    #[test]
    fn profile_request_survives_a_skipped_step() {
        let data = one_nan_row();
        let mut params = ParamStore::new();
        let net = classifier(&mut params);
        let (_, sink) = collecting_sink();
        let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::CrossEntropy, 32);
        trainer.set_health(health(OnNonFinite::Skip, sink));
        let batcher = Batcher::new(96, 32, None);
        let mut batch = Batch::new(&data, 32);
        trainer.profile_next_step();
        trainer
            .step(&net, &mut params, batcher.gather(&data, 32..64, &mut batch))
            .unwrap();
        assert_eq!(trainer.skipped_steps(), 1);
        assert!(trainer.last_profile().is_none());
        trainer
            .step(&net, &mut params, batcher.gather(&data, 0..32, &mut batch))
            .unwrap();
        assert!(trainer.last_profile().is_some());
        trainer
            .step(&net, &mut params, batcher.gather(&data, 64..96, &mut batch))
            .unwrap();
        assert!(trainer.last_profile().is_none());
    }

    /// `before_update` reads the gradients only under `Skip`: there it refuses a non-finite
    /// loss without a second warning, and refuses a NaN gradient, naming each non-finite
    /// parameter once per epoch; under `Stop` and `Warn` it always allows the update.
    #[test]
    fn before_update_checks_only_under_skip() {
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 2, 1, &mut Rng::seed(0));
        let (weight, bias) = (layer.weight(), layer.bias().unwrap());
        let names = Sequential::new().push(layer).parameter_names();
        params.grads.accumulate(weight).0.data_mut().fill(0.5);
        params.grads.accumulate(bias).0.data_mut()[0] = f32::NAN;
        for policy in [OnNonFinite::Stop, OnNonFinite::Warn] {
            let (seen, sink) = collecting_sink();
            let mut monitor = HealthMonitor::new(health(policy, sink));
            assert!(monitor.before_update(1, f32::NAN, &params, &names));
            assert!(monitor.before_update(1, 0.5, &params, &names));
            assert!(seen.borrow().is_empty());
        }
        let (seen, sink) = collecting_sink();
        let mut monitor = HealthMonitor::new(health(OnNonFinite::Skip, sink));
        assert!(!monitor.before_update(1, f32::NAN, &params, &names));
        assert!(seen.borrow().is_empty());
        assert!(!monitor.before_update(2, 0.5, &params, &names));
        assert!(!monitor.before_update(3, 0.5, &params, &names));
        // Every step of epoch 1 was skipped.
        assert!(monitor.end_epoch(1).is_err());
        assert!(!monitor.before_update(4, 0.5, &params, &names));
        let bias_warning = |step| Warning::NonFinite {
            step,
            location: "0.bias".to_string(),
            input: None,
        };
        assert_eq!(*seen.borrow(), [bias_warning(2), bias_warning(4)]);
        params.zero_grad();
        assert!(monitor.before_update(5, 0.5, &params, &names));
    }

    /// Under `Stop`, a step with a finite loss and a NaN gradient applies the update, then
    /// returns `Error::NonFinite` naming the parameter: the parameter already holds the NaN,
    /// as the docs of `OnNonFinite::Stop` say.
    #[test]
    fn stop_applies_a_nan_gradient_then_stops() {
        let data = finite_loss_nan_grad_data(true);
        let batcher = Batcher::new(data.len(), 4, None);
        let mut batch = Batch::new(&data, 4);
        let mut params = ParamStore::new();
        let net = nan_grad_beyond_net(&mut params);
        let mut trainer = Trainer::new(&net, crate::Adam::new(0.01), Loss::Mse, 4);
        trainer.set_health(health(OnNonFinite::Stop, collecting_sink().1));
        let err = trainer
            .step(&net, &mut params, batcher.gather(&data, 4..8, &mut batch))
            .unwrap_err();
        assert!(is_non_finite(&err, 1, "0.weight"), "{err:?}");
        let (_, weight) = net.parameter_names()[0].clone();
        assert!(params.value(weight).data().iter().any(|v| v.is_nan()));
    }

    /// Under `Skip`, `after_backward` sends the non-finite loss warning once per epoch and
    /// returns `Ok`.
    #[test]
    fn after_backward_under_skip_warns_and_continues() {
        let params = ParamStore::new();
        let net = Sequential::new().push(Relu);
        let mut graph = Graph::new();
        let tape = graph.tape(&params.values);
        let trace = Trace::new();
        let x = Tensor::full(&[1, 1], 1.0f32);
        let (seen, sink) = collecting_sink();
        let mut monitor = HealthMonitor::new(health(OnNonFinite::Skip, sink));
        for step in 1..=2 {
            monitor
                .after_backward(step, f32::NAN, &tape, &trace, &net, no_labels(&x))
                .unwrap();
        }
        assert_eq!(
            *seen.borrow(),
            [Warning::NonFinite {
                step: 1,
                location: "loss".to_string(),
                input: None
            }]
        );
    }
}
