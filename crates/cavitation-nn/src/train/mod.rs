//! Training: datasets, targets and batches (in `data`), and the training step
//! ([`Trainer`]), [`fit`] and [`accuracy`] (in `fit`), held-out measurement
//! ([`Evaluator`] and [`evaluate`], in `evaluate`), [`Standardizer`], which standardises
//! inputs or targets and folds that into a network (in `standardize`), and [`Flow`], which a progress
//! callback returns to go on or stop.

mod data;
mod evaluate;
mod fit;
mod standardize;

pub use data::{Batch, BatchView, Batcher, Dataset, Targets, TargetsView};
pub use evaluate::{Evaluation, Evaluator, evaluate};
pub use fit::{
    EpochReport, FitConfig, History, Loss, LrSchedule, OptimizerConfig, Trainer, accuracy, fit,
    fit_with_hooks,
};
pub(crate) use fit::{check_dataset, check_widths};
pub use standardize::Standardizer;

/// What a progress callback returns: go on with the run, or stop it there. The
/// [`shrink`](crate::shrink::shrink()) tool's `on_step` callback returns one after each step.
///
/// ```
/// use cavitation_nn::Flow;
/// // Go on while fewer than 3 steps have run.
/// let on_step = |done: usize| if done < 3 { Flow::Continue } else { Flow::Stop };
/// let mut done = 0;
/// while on_step(done) == Flow::Continue {
///     done += 1;
/// }
/// assert_eq!(done, 3);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Flow {
    /// Go on with the run.
    #[default]
    Continue,
    /// End the run after the current step, keeping what it has found so far.
    Stop,
}
