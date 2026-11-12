//! Training: datasets, targets and batches (in `data`), and the training step
//! ([`Trainer`]), [`fit`] and [`accuracy`] (in `fit`), and held-out measurement
//! ([`Evaluator`] and [`evaluate`], in `evaluate`).

mod data;
mod evaluate;
mod fit;

pub use data::{Batch, BatchView, Batcher, Dataset, Targets, TargetsView};
pub use evaluate::{Evaluation, Evaluator, evaluate};
pub(crate) use fit::check_widths;
pub use fit::{
    EpochReport, FitConfig, History, Loss, LrSchedule, OptimizerConfig, Trainer, accuracy, fit,
    fit_with_hooks,
};
