//! [`StepLog`]: a [`StepControl`] that keeps one [`StepRecord`] per training step of a
//! [`fit_controlled`](super::fit_controlled) run, for finding when a run settles or timing
//! its updates without writing a training loop.

use std::time::Duration;

use cavitation_core::ParamStore;

use super::Flow;
use super::fit::{ModelHandle, StepControl, StepInfo, TrainerHandle};

/// One training step as a [`StepLog`] saw it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct StepRecord {
    /// The step's number in the run, counting from 1.
    pub step: u64,
    /// The epoch the step belongs to, counting from 1.
    pub epoch: usize,
    /// The batch's loss (its mean over the batch's rows).
    pub loss: f32,
    /// The batch's row count.
    pub rows: usize,
    /// The step's wall-clock time in seconds, as [`StepInfo::seconds`] measures it.
    pub seconds: f64,
    /// The optimiser's update time, from the step's profile: `None` unless the log was made
    /// with [`StepLog::with_update_times`], and `None` for the first step the log records,
    /// which no earlier recorded step could ask a profile for.
    pub update: Option<Duration>,
}

/// A [`StepControl`] that records every training step of a
/// [`fit_controlled`](super::fit_controlled) run: its loss, rows and wall-clock time, and
/// with [`with_update_times`](Self::with_update_times) the optimiser's update time. It never
/// stops the run or changes the model.
///
/// A step that [`OnNonFinite::Skip`](crate::OnNonFinite::Skip) leaves out is not recorded,
/// since [`StepControl::after_step`] does not run for it. Each record is pushed onto a
/// `Vec`, which grows as the run goes on; [`reserve`](Self::reserve) sizes it once, so a
/// run of at most that many steps adds no allocation per step for the records (with update
/// times on, the first profiled step still allocates the profile's buffers, once). A run
/// through [`fit`](super::fit), without a log, is unchanged.
///
/// With update times on, every step after the first recorded one is profiled (each layer's
/// forward and backward pass is timed as well), which adds the cost of reading the clock
/// around each layer to every step.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::{
///     Dataset, FitConfig, Hooks, Linear, Sequential, StepLog, Targets, fit_controlled,
/// };
/// let data = Dataset {
///     x: Tensor::full(&[8, 2], 0.5f32),
///     targets: Targets::Classes(vec![0, 1, 0, 1, 0, 1, 0, 1]),
/// };
/// let mut params = ParamStore::new();
/// let mut net = Sequential::new().push(Linear::new(&mut params, 2, 2, &mut Rng::seed(0)));
/// let config = FitConfig { epochs: 3, batch: 4, ..FitConfig::default() };
/// let mut log = StepLog::with_update_times();
/// fit_controlled(&mut net, &mut params, &data, config, &mut Hooks::new(), &mut log).unwrap();
/// // Two steps per epoch for three epochs.
/// assert_eq!(log.records().len(), 6);
/// assert_eq!((log.records()[5].step, log.records()[5].epoch), (6, 3));
/// assert!(log.records()[0].update.is_none());
/// assert!(log.records()[1..].iter().all(|r| r.update.is_some()));
/// ```
#[derive(Clone, Debug, Default)]
pub struct StepLog {
    /// The steps recorded so far, in order.
    records: Vec<StepRecord>,
    /// Whether to ask for every next step's profile and record its update time.
    update_times: bool,
}

impl StepLog {
    /// A log of each step's loss, rows and wall-clock time.
    ///
    /// ```
    /// assert!(cavitation_nn::StepLog::new().records().is_empty());
    /// ```
    pub fn new() -> StepLog {
        StepLog::default()
    }

    /// A log that also records each step's update time, from a profile it asks for before
    /// every step after the first recorded one.
    ///
    /// ```
    /// assert!(cavitation_nn::StepLog::with_update_times().records().is_empty());
    /// ```
    pub fn with_update_times() -> StepLog {
        StepLog {
            update_times: true,
            ..StepLog::default()
        }
    }

    /// The log with room for `steps` records before its `Vec` grows.
    ///
    /// ```
    /// let log = cavitation_nn::StepLog::new().reserve(1000);
    /// assert!(log.records().is_empty());
    /// ```
    pub fn reserve(mut self, steps: usize) -> StepLog {
        self.records.reserve(steps);
        self
    }

    /// The steps recorded so far, in order.
    ///
    /// ```
    /// assert_eq!(cavitation_nn::StepLog::new().records(), &[]);
    /// ```
    pub fn records(&self) -> &[StepRecord] {
        &self.records
    }

    /// Takes the records out, leaving the log empty (with its setting kept).
    ///
    /// ```
    /// let mut log = cavitation_nn::StepLog::new();
    /// assert!(log.take().is_empty());
    /// ```
    pub fn take(&mut self) -> Vec<StepRecord> {
        std::mem::take(&mut self.records)
    }
}

impl StepControl for StepLog {
    /// Records the step; with update times on, reads the update time from the step's
    /// profile (asked for by the step before) and asks for the next step's.
    fn after_step(
        &mut self,
        info: &StepInfo,
        _model: &mut ModelHandle<'_>,
        _params: &ParamStore,
        trainer: &mut TrainerHandle<'_>,
    ) -> Flow {
        let update = if self.update_times {
            let update = trainer.last_profile().map(|p| p.optimizer);
            trainer.profile_next_step();
            update
        } else {
            None
        };
        self.records.push(StepRecord {
            step: info.step,
            epoch: info.epoch,
            loss: info.loss,
            rows: info.rows,
            seconds: info.seconds,
            update,
        });
        Flow::Continue
    }
}

/// Tests for the step log.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::{Dataset, FitConfig, History, Targets, fit_controlled};
    use crate::{Hooks, Linear, Relu, Sequential};
    use cavitation_core::{Rng, Tensor};

    /// A 4 → 8 → 2 network and 22 rows of class targets, trained for 3 epochs at batch 8 (3
    /// steps an epoch, the last of 6 rows) with `log`.
    fn run(log: &mut StepLog) -> History {
        let mut rng = Rng::seed(3);
        let mut params = ParamStore::new();
        let mut net = Sequential::new()
            .push(Linear::new(&mut params, 4, 8, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, 8, 2, &mut rng));
        let x: Vec<f32> = (0..22 * 4).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        let data = Dataset {
            x: Tensor::from_vec(x, &[22, 4]).unwrap(),
            targets: Targets::Classes((0..22).map(|i| i % 2).collect()),
        };
        let config = FitConfig {
            epochs: 3,
            batch: 8,
            shuffle_seed: Some(1),
            ..FitConfig::default()
        };
        fit_controlled(&mut net, &mut params, &data, config, &mut Hooks::new(), log).unwrap()
    }

    /// One record per step, numbered and placed in their epochs, with the batch sizes; the
    /// records' losses, weighted by rows, give each epoch's mean loss exactly as `History`
    /// reports it. Without update times no step is profiled.
    #[test]
    fn records_every_step() {
        let mut log = StepLog::new();
        let history = run(&mut log);
        let r = log.records();
        assert_eq!(r.len(), 9);
        for (i, rec) in r.iter().enumerate() {
            assert_eq!((rec.step, rec.epoch), (i as u64 + 1, i / 3 + 1));
            assert_eq!(rec.rows, if i % 3 == 2 { 6 } else { 8 });
            assert!(rec.seconds > 0.0 && rec.loss.is_finite());
            assert!(rec.update.is_none());
        }
        for (e, epoch) in history.epochs.iter().enumerate() {
            let steps = &r[e * 3..e * 3 + 3];
            let sum: f64 = steps
                .iter()
                .map(|s| f64::from(s.loss) * s.rows as f64)
                .sum();
            assert_eq!(sum / 22.0, epoch.mean_loss, "epoch {}", e + 1);
        }
    }

    /// With update times on, every step but the first has the optimiser time of its own
    /// profile, no longer than the step; and the log does not change the training: the
    /// same seeds give the same losses with and without it.
    #[test]
    fn update_times_from_the_profiles() {
        let mut timed = StepLog::with_update_times().reserve(9);
        let mut plain = StepLog::new();
        run(&mut timed);
        run(&mut plain);
        let r = timed.records();
        assert!(r[0].update.is_none());
        for rec in &r[1..] {
            let update = rec.update.expect("a profiled step");
            assert!(update.as_secs_f64() <= rec.seconds, "step {}", rec.step);
        }
        let losses = |log: &StepLog| -> Vec<u32> {
            log.records().iter().map(|r| r.loss.to_bits()).collect()
        };
        assert_eq!(losses(&timed), losses(&plain));
        assert_eq!(timed.take().len(), 9);
        assert!(timed.records().is_empty());
    }
}
