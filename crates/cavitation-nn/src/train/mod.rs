//! Training: datasets, targets and batches (in `data`), and the training step
//! ([`Trainer`]), [`fit`] and [`accuracy`] (in `fit`).

mod data;
mod fit;

pub use data::{Batch, BatchView, Batcher, Dataset, Targets, TargetsView};
pub use fit::{
    EpochReport, FitConfig, History, Loss, LrSchedule, OptimizerConfig, Trainer, accuracy, fit,
    fit_with_hooks,
};
