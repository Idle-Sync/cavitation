//! [`fit`](crate::fit) on CUDA, for [`Device::Cuda`](crate::Device) and for a
//! [`Device::Auto`](crate::Device) that picks CUDA: the trainer built for the run, with every
//! reason CUDA cannot train it found before any training, and the epoch loop that trains
//! through that [`CudaTrainer`] and measures the validation set with its
//! [`CudaModel`](super::CudaModel), copying the parameters back when the host reads them.

use std::time::Instant;

use cavitation_core::{Error, ParamStore};
use cavitation_cuda::Cuda;

use super::CudaTrainer;
use crate::train::{StepRates, Watching};
use crate::{Batch, Batcher, Dataset, DeviceKind, EpochReport, FitConfig, History, Sequential};

/// The [`CudaTrainer`] for [`fit`](crate::fit)'s run of `model` on `cuda` under `config`,
/// from the values in `params`, or the first reason CUDA cannot train the run: every reason
/// known before the first step is found here, so a run under [`Device::Auto`](crate::Device)
/// can still train elsewhere. An error of a step, or of the step for an epoch's short last
/// batch, set up when that batch comes, ends the run.
///
/// # Errors
///
/// As [`CudaTrainer::new`] returns them; [`Error::InvalidData`] (starting `validation: `) if
/// a batch-norm layer that keeps no running statistics would see a short last batch of
/// `validation`; and [`Error::Unsupported`] for a `config.health` that
/// [`CudaTrainer::set_health`] refuses.
///
/// # Panics
///
/// If `config.batch` is 0.
pub(crate) fn trainer_for<'c, 's>(
    model: &Sequential,
    params: &ParamStore,
    config: &FitConfig<'_>,
    validation: Option<&Dataset>,
    cuda: &'c Cuda,
) -> Result<CudaTrainer<'c, 's>, Error> {
    assert!(
        config.batch > 0,
        "Trainer::new: the batch size must be at least 1"
    );
    let mut trainer = CudaTrainer::new(
        model,
        params,
        &config.optimizer,
        config.loss,
        config.batch,
        cuda,
    )?;
    if let Some(held_out) = validation {
        trainer.model().check_rows(held_out).map_err(|e| match e {
            Error::InvalidData(msg) => Error::InvalidData(format!("validation: {msg}")),
            other => other,
        })?;
    }
    CudaTrainer::check_health(&config.health)?;
    Ok(trainer)
}

/// [`fit`](crate::fit)'s run through `trainer`, built by [`trainer_for`] for this `model`,
/// `params`, `config` and `validation_set`, after `fit`'s checks of the data and of
/// `watching`'s rules: trains `model` epoch by epoch, measures `validation_set` with the
/// trainer's model, and copies the parameters back into `params` before `config.on_epoch`,
/// when early stopping keeps a best epoch, before an error met during training is returned,
/// and at the end.
///
/// # Errors
///
/// As [`CudaTrainer::step`] and [`CudaTrainer::end_epoch`] return them, and the validation's
/// measurement.
pub(crate) fn fit_on_cuda<'s>(
    model: &Sequential,
    params: &mut ParamStore,
    train: &Dataset,
    mut config: FitConfig<'s>,
    validation_set: Option<Dataset>,
    mut watching: Watching,
    mut trainer: CudaTrainer<'_, 's>,
) -> Result<History, Error> {
    trainer.set_health(config.health)?;
    trainer.set_dropout_seed(config.dropout_seed);
    let mut batcher = Batcher::new(train.len(), config.batch, config.shuffle_seed);
    let mut batch = Batch::new(train, config.batch);
    let mut history = History {
        epochs: Vec::with_capacity(config.epochs.min(1024)),
        device: DeviceKind::Cuda,
        ..History::default()
    };
    params.zero_grad();
    let mut rates = StepRates {
        schedule: config.schedule,
        initial: trainer.learning_rate(),
        steps_per_epoch: batcher.batches().count(),
        epochs: config.epochs,
    };
    // Whether `params` holds what the card holds: true until a step trains, and again after
    // each copy back.
    let mut synced = true;
    for epoch in 1..=config.epochs {
        let start = Instant::now();
        batcher.epoch();
        let mut total = 0.0f64;
        let mut trained = 0usize;
        for (position, range) in batcher.batches().enumerate() {
            if position == 0 || rates.per_step() {
                let step = (epoch - 1) * rates.steps_per_epoch + position;
                trainer.set_learning_rate(rates.at(step));
            }
            let rows = range.len();
            let view = batcher.gather(train, range.clone(), &mut batch);
            synced = false;
            match trainer.step_rows(model, view, Some(&batcher.order()[range])) {
                Ok((loss, true)) => {
                    total += f64::from(loss) * rows as f64;
                    trained += rows;
                }
                Ok((_, false)) => {}
                Err(e) => {
                    trainer.sync_to(params);
                    return Err(e);
                }
            }
        }
        watching.trained();
        let ended = trainer.end_epoch(epoch);
        let seconds = start.elapsed().as_secs_f64();
        let measured = ended.and_then(|()| match &validation_set {
            Some(held_out) => trainer
                .model()
                .evaluate(model, held_out, config.loss)
                .map(Some),
            None => Ok(None),
        });
        let validation = match measured {
            Ok(validation) => validation,
            Err(e) => {
                trainer.sync_to(params);
                return Err(e);
            }
        };
        let report = EpochReport {
            epoch,
            mean_loss: total / trained as f64,
            rows: trained,
            seconds,
            validation,
        };
        if let Some(on_epoch) = config.on_epoch.as_mut() {
            if !synced {
                trainer.sync_to(params);
                synced = true;
            }
            on_epoch(&report, model, params);
        }
        history.epochs.push(report);
        rates.initial = watching.peak_rate(&report, rates.initial);
        let stop_early =
            watching.after_epoch(&report, config.epochs, &mut history, params, |params| {
                if !synced {
                    trainer.sync_to(params);
                    synced = true;
                }
            });
        if stop_early {
            history.stopped_early = true;
            break;
        }
    }
    // The best epoch's values replace every value, so nothing is copied back for them.
    if !watching.restore_best(params) && !synced {
        trainer.sync_to(params);
    }
    Ok(history)
}
