//! Watching one measure from epoch to epoch: [`Monitor`] names the measure, [`Threshold`]
//! says how much better counts as better for [`LrSchedule::ReduceOnPlateau`], and
//! [`EarlyStopping`] is [`FitConfig::early_stopping`]'s rule. The per-run state [`fit`]
//! keeps for each rule, `Plateau` and `Watch`, is here too.
//!
//! [`LrSchedule::ReduceOnPlateau`]: crate::LrSchedule::ReduceOnPlateau
//! [`FitConfig::early_stopping`]: crate::FitConfig::early_stopping
//! [`fit`]: crate::fit

use super::fit::EpochReport;

/// The measure an epoch-by-epoch rule watches, read from each whole epoch's
/// [`EpochReport`].
///
/// # Examples
///
/// ```
/// use cavitation_nn::{EpochReport, Monitor};
/// let report = EpochReport::new(1, 0.25, 100, 0.5);
/// assert_eq!(Monitor::TrainingLoss.value(&report), Some(0.25));
/// // No validation set, so nothing to read.
/// assert_eq!(Monitor::ValidationLoss.value(&report), None);
/// assert!(Monitor::ValidationScore.higher_is_better());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Monitor {
    /// The epoch's mean training loss, [`EpochReport::mean_loss`]. Lower is better.
    TrainingLoss,
    /// The validation set's loss, [`Evaluation::loss`](crate::Evaluation::loss). Lower is
    /// better. Needs [`FitConfig::validation`](crate::FitConfig::validation).
    ValidationLoss,
    /// The validation set's accuracy for class targets, or its R² for value targets.
    /// Higher is better. Needs [`FitConfig::validation`](crate::FitConfig::validation).
    ValidationScore,
}

impl Monitor {
    /// Whether a larger value is an improvement: true for
    /// [`ValidationScore`](Self::ValidationScore), false for the losses.
    ///
    /// ```
    /// use cavitation_nn::Monitor;
    /// assert!(!Monitor::ValidationLoss.higher_is_better());
    /// ```
    pub fn higher_is_better(self) -> bool {
        matches!(self, Monitor::ValidationScore)
    }

    /// The measure in `report`, or `None` when the report has none (a validation measure
    /// on a run without a validation set).
    ///
    /// ```
    /// use cavitation_nn::{EpochReport, Monitor};
    /// let report = EpochReport::new(2, 0.5, 10, 0.1);
    /// assert_eq!(Monitor::ValidationScore.value(&report), None);
    /// ```
    pub fn value(self, report: &EpochReport) -> Option<f64> {
        match self {
            Monitor::TrainingLoss => Some(report.mean_loss),
            Monitor::ValidationLoss => report.validation.map(|v| v.loss),
            Monitor::ValidationScore => report.validation.and_then(|v| v.accuracy.or(v.r2)),
        }
    }

    /// Whether the measure comes from the validation set.
    pub(crate) fn needs_validation(self) -> bool {
        !matches!(self, Monitor::TrainingLoss)
    }
}

/// How much better a measure has to be than the best so far to count as better, for
/// [`LrSchedule::ReduceOnPlateau`](crate::LrSchedule::ReduceOnPlateau): PyTorch's
/// `threshold` with its `threshold_mode`. The default is `Relative(1e-4)`, PyTorch's.
///
/// # Examples
///
/// ```
/// use cavitation_nn::Threshold;
/// assert_eq!(Threshold::default(), Threshold::Relative(1e-4));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Threshold {
    /// Better by this share of the best value: a loss must fall below `best · (1 − r)`, a
    /// score rise above `best · (1 + r)`. PyTorch's `threshold_mode='rel'`.
    Relative(f64),
    /// Better by this amount: a loss must fall below `best − a`, a score rise above
    /// `best + a`. PyTorch's `threshold_mode='abs'`.
    Absolute(f64),
}

impl Default for Threshold {
    /// `Relative(1e-4)`, PyTorch's default.
    fn default() -> Self {
        Threshold::Relative(1e-4)
    }
}

impl Threshold {
    /// The share or amount, which must be finite and not negative.
    pub(crate) fn amount(self) -> f64 {
        match self {
            Threshold::Relative(r) => r,
            Threshold::Absolute(a) => a,
        }
    }
}

/// When [`fit`](crate::fit) ends a run early: after `patience` whole epochs in a row in which
/// `monitor` did not improve on its best value by more than `min_delta`. With `restore_best`
/// (the default), the run then ends with the parameter values of the best epoch, buffers
/// such as batch norm's running statistics included; this also holds when the run reaches
/// its last epoch without stopping early. [`History::best_epoch`](crate::History::best_epoch)
/// names that epoch.
///
/// A value improves on the best when it is below `best − min_delta` (a loss) or above
/// `best + min_delta` (a score), as Keras's `EarlyStopping` counts it. The first epoch's
/// value is always an improvement unless it is NaN; a NaN value never is. Unlike Keras,
/// the best weights are put back whenever the run ends, not only when it stops early.
/// The neuron masks a [`StepControl`](crate::StepControl) sets and the optimiser's state
/// are not put back.
///
/// The struct is `#[non_exhaustive]`: make one with [`new`](Self::new) and set the fields.
///
/// # Examples
///
/// Stop once the validation accuracy has not improved for 3 epochs:
///
/// ```
/// use cavitation_nn::{EarlyStopping, Monitor};
/// let mut stop = EarlyStopping::new(Monitor::ValidationScore, 3);
/// stop.min_delta = 1e-3;
/// assert!(stop.restore_best);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct EarlyStopping {
    /// The measure watched.
    pub monitor: Monitor,
    /// How many whole epochs in a row without an improvement end the run; at least 1.
    pub patience: usize,
    /// The smallest change that counts as an improvement; finite and not negative.
    pub min_delta: f64,
    /// Whether the run ends with the best epoch's parameter values.
    pub restore_best: bool,
}

impl EarlyStopping {
    /// Watches `monitor` with `patience`, a `min_delta` of 0 and `restore_best` on.
    ///
    /// ```
    /// use cavitation_nn::{EarlyStopping, Monitor};
    /// let stop = EarlyStopping::new(Monitor::ValidationLoss, 5);
    /// assert_eq!((stop.patience, stop.min_delta), (5, 0.0));
    /// ```
    pub fn new(monitor: Monitor, patience: usize) -> EarlyStopping {
        EarlyStopping {
            monitor,
            patience,
            min_delta: 0.0,
            restore_best: true,
        }
    }
}

/// The value a rule starts from: the worst possible, so any finite first value is better.
fn worst(higher_is_better: bool) -> f64 {
    if higher_is_better {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    }
}

/// The state of [`LrSchedule::ReduceOnPlateau`](crate::LrSchedule::ReduceOnPlateau) over a
/// run, stepped once per whole epoch as PyTorch's `ReduceLROnPlateau.step(metric)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Plateau {
    /// The best value so far.
    best: f64,
    /// Epochs in a row without an improvement, outside the cooldown.
    bad: usize,
    /// Epochs of cooldown left after the last reduction.
    cooldown_left: usize,
}

/// The settings of a plateau rule, as in
/// [`LrSchedule::ReduceOnPlateau`](crate::LrSchedule::ReduceOnPlateau).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlateauRule {
    /// Whether a larger value is better.
    pub(crate) higher_is_better: bool,
    /// What each reduction multiplies the rate by.
    pub(crate) factor: f64,
    /// Bad epochs allowed before a reduction; the reduction comes on the one after.
    pub(crate) patience: usize,
    /// What counts as better.
    pub(crate) threshold: Threshold,
    /// Epochs after a reduction in which bad epochs are not counted.
    pub(crate) cooldown: usize,
    /// The lowest rate.
    pub(crate) min_lr: f64,
}

/// PyTorch's `eps`: a reduction smaller than this leaves the rate as it is.
const PLATEAU_EPS: f64 = 1e-8;

impl Plateau {
    /// A plateau rule that has seen no epoch.
    pub(crate) fn new(higher_is_better: bool) -> Plateau {
        Plateau {
            best: worst(higher_is_better),
            bad: 0,
            cooldown_left: 0,
        }
    }

    /// Takes one epoch's `value` and returns the rate for the next epoch, given the current
    /// `rate`, in PyTorch's order: count the epoch good or bad, then the cooldown, then
    /// reduce once the bad epochs pass the patience.
    pub(crate) fn step(&mut self, rule: &PlateauRule, value: f64, rate: f64) -> f64 {
        let better = match (rule.higher_is_better, rule.threshold) {
            (false, Threshold::Relative(r)) => value < self.best * (1.0 - r),
            (false, Threshold::Absolute(a)) => value < self.best - a,
            (true, Threshold::Relative(r)) => value > self.best * (r + 1.0),
            (true, Threshold::Absolute(a)) => value > self.best + a,
        };
        if better {
            self.best = value;
            self.bad = 0;
        } else {
            self.bad += 1;
        }
        if self.cooldown_left > 0 {
            self.cooldown_left -= 1;
            self.bad = 0;
        }
        let mut next = rate;
        if self.bad > rule.patience {
            let reduced = (rate * rule.factor).max(rule.min_lr);
            if rate - reduced > PLATEAU_EPS {
                next = reduced;
            }
            self.cooldown_left = rule.cooldown;
            self.bad = 0;
        }
        next
    }
}

/// What one epoch did to an [`EarlyStopping`] rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// The epoch is the best so far.
    Improved,
    /// Not better, and the patience is not used up.
    Waiting,
    /// Not better, and this was the last epoch the patience allows.
    Stop,
}

/// The state of an [`EarlyStopping`] rule over a run.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Watch {
    /// The best value so far.
    best: f64,
    /// The epoch that had it.
    pub(crate) best_epoch: Option<usize>,
    /// Epochs in a row without an improvement.
    wait: usize,
}

impl Watch {
    /// A rule that has seen no epoch.
    pub(crate) fn new(rule: &EarlyStopping) -> Watch {
        Watch {
            best: worst(rule.monitor.higher_is_better()),
            best_epoch: None,
            wait: 0,
        }
    }

    /// Takes epoch `epoch`'s `value`.
    pub(crate) fn step(&mut self, rule: &EarlyStopping, epoch: usize, value: f64) -> Verdict {
        let better = if rule.monitor.higher_is_better() {
            value - rule.min_delta > self.best
        } else {
            value + rule.min_delta < self.best
        };
        if better {
            self.best = value;
            self.best_epoch = Some(epoch);
            self.wait = 0;
            Verdict::Improved
        } else {
            self.wait += 1;
            if self.wait >= rule.patience {
                Verdict::Stop
            } else {
                Verdict::Waiting
            }
        }
    }
}

/// Tests for the two rules on their own: the plateau rule against PyTorch's rates, and the
/// early-stopping count.
#[cfg(test)]
mod tests {
    use super::*;

    /// The rates of a plateau rule starting at 0.1 over `values`.
    fn rates(rule: PlateauRule, values: &[f64]) -> Vec<f64> {
        let mut plateau = Plateau::new(rule.higher_is_better);
        let mut rate = 0.1;
        values
            .iter()
            .map(|&v| {
                rate = plateau.step(&rule, v, rate);
                rate
            })
            .collect()
    }

    /// A loss rule with PyTorch's defaults apart from `patience`.
    fn loss_rule(patience: usize) -> PlateauRule {
        PlateauRule {
            higher_is_better: false,
            factor: 0.1,
            patience,
            threshold: Threshold::default(),
            cooldown: 0,
            min_lr: 0.0,
        }
    }

    /// The loss sequence of `scripts/fixtures/reduce_on_plateau.py`.
    const LOSS: [f64; 16] = [
        1.0, 0.8, 0.8, 0.8, 0.79995, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.9, 0.9, 0.9, 0.9,
    ];

    /// The accuracy sequence of the same script.
    const ACCURACY: [f64; 14] = [
        0.5, 0.6, 0.6, 0.6, 0.6, 0.65, 0.65, 0.65, 0.65, 0.65, 0.7, 0.7, 0.7, 0.7,
    ];

    /// Each case of `scripts/fixtures/reduce_on_plateau.py` gives PyTorch 2.15's rates, bit
    /// for bit (the expected values are its output).
    #[test]
    fn reduce_on_plateau_matches_pytorch() {
        let (a, b, c, d) = (
            0.010000000000000002,
            0.0010000000000000002,
            0.00010000000000000003,
            1.0000000000000004e-05,
        );
        assert_eq!(
            rates(loss_rule(2), &LOSS),
            [0.1, 0.1, 0.1, 0.1, a, a, a, b, b, b, b, c, c, c, d, d]
        );
        let abs = PlateauRule {
            factor: 0.5,
            patience: 1,
            threshold: Threshold::Absolute(0.01),
            cooldown: 1,
            ..loss_rule(1)
        };
        assert_eq!(
            rates(abs, &LOSS),
            [
                0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025, 0.0125, 0.0125,
                0.0125, 0.00625, 0.00625, 0.00625
            ]
        );
        let floor = PlateauRule {
            min_lr: 1e-4,
            ..loss_rule(0)
        };
        let mut want = vec![0.1, 0.1, a, b];
        want.resize(16, c);
        assert_eq!(rates(floor, &LOSS), want);
        let score = PlateauRule {
            higher_is_better: true,
            factor: 0.5,
            ..loss_rule(1)
        };
        assert_eq!(
            rates(score, &ACCURACY),
            [
                0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025, 0.0125, 0.0125, 0.0125,
                0.00625, 0.00625
            ]
        );
        let score_abs = PlateauRule {
            patience: 0,
            threshold: Threshold::Absolute(0.02),
            cooldown: 2,
            ..score
        };
        assert_eq!(
            rates(score_abs, &ACCURACY),
            [
                0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025, 0.0125, 0.0125, 0.0125,
                0.00625, 0.00625
            ]
        );
    }

    /// The early-stopping count: an improvement by more than `min_delta` resets it, a NaN
    /// or a smaller change does not, and the `patience`-th bad epoch in a row stops.
    #[test]
    fn early_stopping_counts_bad_epochs() {
        let mut rule = EarlyStopping::new(Monitor::ValidationLoss, 2);
        rule.min_delta = 0.01;
        let mut watch = Watch::new(&rule);
        let verdicts: Vec<Verdict> = [1.0, 0.95, 0.945, f64::NAN, 0.9, 0.9, 0.9]
            .iter()
            .enumerate()
            .map(|(i, &v)| watch.step(&rule, i + 1, v))
            .collect();
        use Verdict::*;
        assert_eq!(
            verdicts,
            [Improved, Improved, Waiting, Stop, Improved, Waiting, Stop]
        );
        assert_eq!(watch.best_epoch, Some(5));
        let score = EarlyStopping::new(Monitor::ValidationScore, 1);
        let mut watch = Watch::new(&score);
        assert_eq!(watch.step(&score, 1, f64::NAN), Stop);
        assert_eq!(watch.best_epoch, None);
        assert_eq!(watch.step(&score, 2, 0.5), Improved);
        assert_eq!(watch.step(&score, 3, 0.5), Stop);
    }
}
