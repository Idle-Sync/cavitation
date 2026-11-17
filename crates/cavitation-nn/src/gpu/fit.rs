//! [`fit`](crate::fit) on the GPU, for [`Device::Gpu`](crate::Device): the epoch loop that
//! trains through a [`GpuTrainer`] and measures the validation set with its
//! [`GpuModel`](super::GpuModel), copying the parameters back when the host reads them.

use std::time::Instant;

use cavitation_core::{Error, ParamStore};
use cavitation_gpu::Gpu;

use super::GpuTrainer;
use crate::train::{StepRates, Watching};
use crate::{Batch, Batcher, Dataset, EpochReport, FitConfig, History, Sequential};

/// [`fit`](crate::fit)'s run on `gpu`, after its checks of the data and of `watching`'s
/// rules: trains `model` from the values in `params` epoch by epoch through a
/// [`GpuTrainer`], measures `validation` with the trainer's model, and copies the parameters
/// back into `params` before `config.on_epoch`, when early stopping keeps a best epoch,
/// before an error met during training is returned, and at the end.
///
/// # Errors
///
/// As [`GpuTrainer::new`], [`GpuTrainer::set_health`] and [`GpuTrainer::step`] return them,
/// and [`Error::InvalidData`] (starting `validation: `) if a batch-norm layer that keeps no
/// running statistics would see a short last batch of `validation`.
///
/// # Panics
///
/// If `config.batch` is 0.
pub(crate) fn fit_on_gpu(
    model: &Sequential,
    params: &mut ParamStore,
    train: &Dataset,
    mut config: FitConfig<'_>,
    validation_set: Option<Dataset>,
    mut watching: Watching,
    gpu: &Gpu,
) -> Result<History, Error> {
    assert!(
        config.batch > 0,
        "Trainer::new: the batch size must be at least 1"
    );
    let mut trainer = GpuTrainer::new(
        model,
        params,
        &config.optimizer,
        config.loss,
        config.batch,
        gpu,
    )?;
    if let Some(held_out) = &validation_set {
        trainer.model().check_rows(held_out).map_err(|e| match e {
            Error::InvalidData(msg) => Error::InvalidData(format!("validation: {msg}")),
            other => other,
        })?;
    }
    trainer.set_health(config.health)?;
    trainer.set_dropout_seed(config.dropout_seed);
    let mut batcher = Batcher::new(train.len(), config.batch, config.shuffle_seed);
    let mut batch = Batch::new(train, config.batch);
    let mut history = History {
        epochs: Vec::with_capacity(config.epochs.min(1024)),
        ..History::default()
    };
    params.zero_grad();
    let mut rates = StepRates {
        schedule: config.schedule,
        initial: trainer.learning_rate(),
        steps_per_epoch: batcher.batches().count(),
        epochs: config.epochs,
    };
    // Whether `params` holds what the GPU holds: true until a step trains, and again after
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
