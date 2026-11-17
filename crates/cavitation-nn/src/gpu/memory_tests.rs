//! After one epoch, a `GpuTrainer` makes no more GPU objects: no buffers from the model's
//! pool, no bind groups and no pipelines, on full, short, skipped and profiled steps, across
//! epochs. (Each step encoder's own buffers are made once, when it is built.) Host
//! allocations per step are checked by `tests/gpu_step_alloc.rs`, which needs a process of
//! its own.
//!
//! Without a GPU every test says so and checks nothing; `CAVITATION_GPU_REQUIRED=1` makes
//! that a failure.

use std::sync::atomic::{AtomicUsize, Ordering};

use cavitation_core::ParamStore;

use super::GpuTrainer;
use super::health_tests::{Data, for_each_gpu};
use super::short_batch_tests::cnn;
use crate::{Health, Loss, OnNonFinite, OptimizerConfig, Warning};

/// The batch with a NaN input, left out under [`OnNonFinite::Skip`] every epoch.
const BAD: usize = 2;

/// Five epochs of 22 rows at batch 6 (6, 6, 6 with a NaN and a short 4) on the CNN with
/// batch norm, max pooling and dropout, every health check on and the NaN batch left out
/// under `Skip`, which sends a warning each epoch. The first epoch builds everything, a
/// profiled step included. Every step after it must leave every count where the first epoch
/// left it, including a profiled step at each place in turn: the short step, and the skipped
/// step, whose profile passes to the step after it.
#[test]
fn nothing_is_made_after_the_first_epoch() {
    let mut params = ParamStore::new();
    let net = cnn(&mut params, 91);
    let mut batches: Vec<Data> = [6, 6, 6, 4]
        .iter()
        .enumerate()
        .map(|(s, &rows)| Data::new(300 + s as u64, rows, 2 * 64, 5))
        .collect();
    batches[BAD].x[5] = f32::NAN;
    let configs = [
        OptimizerConfig::Sgd {
            lr: 0.05,
            momentum: 0.9,
            weight_decay: 1e-4,
        },
        OptimizerConfig::Adam {
            lr: 1e-3,
            weight_decay: 0.0,
        },
    ];
    for config in &configs {
        for_each_gpu(|gpu| {
            let warnings = AtomicUsize::new(0);
            let mut t = GpuTrainer::new(&net, &params, config, Loss::CrossEntropy, 6, gpu).unwrap();
            t.set_health(Health {
                on_non_finite: OnNonFinite::Skip,
                sink: Some(Box::new(|_: &Warning| {
                    warnings.fetch_add(1, Ordering::Relaxed);
                })),
                ..Health::quiet()
            })
            .unwrap();
            t.profile_next_step();
            for d in &batches {
                t.step(&net, d.view()).unwrap();
            }
            t.end_epoch(1).unwrap();
            let warm = t.made();
            for epoch in 2..=5 {
                let warned = warnings.load(Ordering::Relaxed);
                let mut profiled = false;
                for (i, d) in batches.iter().enumerate() {
                    if i == epoch - 2 {
                        t.profile_next_step();
                        profiled = true;
                    }
                    t.step(&net, d.view()).unwrap();
                    if profiled && i != BAD {
                        assert!(t.last_profile().is_some(), "epoch {epoch}, step {}", i + 1);
                        profiled = false;
                    }
                    assert_eq!(
                        t.made(),
                        warm,
                        "{config:?}: epoch {epoch}, step {}: [pool buffers, pipelines, model \
                         bind groups, trainer bind groups]",
                        i + 1
                    );
                }
                t.end_epoch(epoch).unwrap();
                assert_eq!(t.made(), warm, "{config:?}: after epoch {epoch}");
                assert!(
                    warnings.load(Ordering::Relaxed) > warned,
                    "{config:?}: epoch {epoch} sent no warning"
                );
            }
            assert_eq!(t.skipped_steps(), 5, "{config:?}");
        });
    }
}
