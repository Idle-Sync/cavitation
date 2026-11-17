//! `GpuTrainer` on batches of fewer rows than its batch size, an epoch's last, against the
//! CPU `Trainer` from the same start: each step's loss within 2⁻¹⁰ of the CPU's, and the
//! parameters, the running statistics and the batch counts after the last step within 2⁻¹⁰
//! of each tensor's largest value plus 2⁻¹⁶, on a CNN with batch norm (keeping a cumulative
//! average, so the two compiled steps must share its batch count), max pooling and dropout
//! over two epochs, and on the MLP over 50,000 rows at batch 128 (the last batch 80); and the
//! batches a trainer refuses.
//!
//! Without a GPU every test says so and checks nothing; `CAVITATION_GPU_REQUIRED=1` makes
//! that a failure.

use cavitation_core::{Conv2dGeom, Error, ParamStore, Pool2dGeom, Rng, TensorView};

use super::GpuTrainer;
use super::forward_tests::mlp;
use super::health_tests::{Data, for_each_gpu};
use crate::{
    BatchNorm2d, BatchView, Conv2d, Dropout, Flatten, Health, Linear, Loss, MaxPool2d, OnNonFinite,
    OptimizerConfig, Relu, Sequential, TargetsView, Trainer,
};

/// A CNN on 2 × 8 × 8 images: a convolution, batch norm with a cumulative average, ReLU, max
/// pooling, dropout, flatten and a `Linear`.
pub(super) fn cnn(params: &mut ParamStore, seed: u64) -> Sequential {
    let mut rng = Rng::seed(seed);
    let g = Conv2dGeom::with_options(2, 4, (8, 8), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
    Sequential::new()
        .push(Conv2d::new(params, g, &mut rng))
        .push(BatchNorm2d::with_options(params, 4, (8, 8), 1e-5, None, true, true).unwrap())
        .push(Relu)
        .push(MaxPool2d::new(Pool2dGeom::new(4, (8, 8), (2, 2)).unwrap()))
        .push(Dropout::new(0.3).unwrap())
        .push(Flatten)
        .push(Linear::new(params, 64, 5, &mut rng))
}

/// Trains `net` from `params` on `epochs` of `batches` on the CPU and on each GPU, a batch of
/// `full` rows at most, and checks the GPU against the CPU as the module says.
fn track(
    net: &Sequential,
    params: &ParamStore,
    config: &OptimizerConfig,
    full: usize,
    batches: &[Data],
    epochs: usize,
    what: &str,
) {
    let mut cpu = params.clone();
    let mut trainer = Trainer::new(net, config.build(), Loss::CrossEntropy, full);
    trainer.set_health(Health::quiet());
    let mut cpu_losses = Vec::new();
    for e in 1..=epochs {
        for d in batches {
            cpu_losses.push(trainer.step(net, &mut cpu, d.view()).unwrap());
        }
        trainer.end_epoch(e).unwrap();
    }
    for_each_gpu(|gpu| {
        let mut g = GpuTrainer::new(net, params, config, Loss::CrossEntropy, full, gpu).unwrap();
        g.set_health(Health::quiet()).unwrap();
        let mut worst = 0.0f32;
        let mut k = 0;
        for e in 1..=epochs {
            for d in batches {
                let loss = g.step(net, d.view()).unwrap();
                let want = cpu_losses[k];
                let off = (loss - want).abs() / want.abs();
                assert!(
                    off <= 1.0 / 1024.0,
                    "{what}: step {}: loss {loss}, the CPU's {want}",
                    k + 1
                );
                worst = worst.max(off);
                k += 1;
            }
            g.end_epoch(e).unwrap();
        }
        assert_eq!(g.steps(), trainer.steps(), "{what}");
        println!("{what}: {k} steps, worst loss off by {worst:.3e} of the CPU's");
        let mut out = params.clone();
        g.sync_to(&mut out);
        for (name, id) in net.state_names() {
            let (a, b) = (out.value(id).data(), cpu.value(id).data());
            let scale = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let off = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(
                off <= scale / 1024.0 + 1.0 / 65536.0,
                "{what}: {name} off by {off:e} of {scale:e}"
            );
        }
    });
}

/// Two epochs of 22 rows at batch 6 (batches of 6, 6, 6 and 4) on the CNN with SGD and
/// momentum: the short step's batch norm, pooling, dropout masks and gradients, and the batch
/// count the two steps share.
#[test]
fn a_short_last_batch_tracks_the_cpu() {
    let mut params = ParamStore::new();
    let net = cnn(&mut params, 71);
    let batches: Vec<Data> = [6, 6, 6, 4]
        .iter()
        .enumerate()
        .map(|(s, &rows)| Data::new(100 + s as u64, rows, 2 * 64, 5))
        .collect();
    let sgd = OptimizerConfig::Sgd {
        lr: 0.05,
        momentum: 0.9,
        weight_decay: 1e-4,
    };
    track(&net, &params, &sgd, 6, &batches, 2, "cnn, 22 rows at 6");
    // Two short row counts in one run: two short steps, and three passes sharing the batch
    // count.
    let mixed: Vec<Data> = [6, 4, 6, 2]
        .iter()
        .enumerate()
        .map(|(s, &rows)| Data::new(120 + s as u64, rows, 2 * 64, 5))
        .collect();
    track(
        &net,
        &params,
        &sgd,
        6,
        &mixed,
        2,
        "cnn, batches of 6, 4, 6 and 2",
    );
}

/// A NaN in a short batch is left out under `Skip` and stops under `Stop`, through the short
/// step's own finite flag: the parameters, the optimiser's state, the running statistics and
/// the batch count stay as they were, and the next full step trains.
#[test]
fn a_nan_in_a_short_batch_changes_nothing() {
    let mut params = ParamStore::new();
    let net = cnn(&mut params, 75);
    let full = Data::new(130, 6, 2 * 64, 5);
    let mut bad = Data::new(131, 4, 2 * 64, 5);
    bad.x[5] = f32::NAN;
    let adam = OptimizerConfig::Adam {
        lr: 1e-2,
        weight_decay: 0.0,
    };
    for policy in [OnNonFinite::Skip, OnNonFinite::Stop] {
        for_each_gpu(|gpu| {
            let mut t = GpuTrainer::new(&net, &params, &adam, Loss::CrossEntropy, 6, gpu).unwrap();
            t.set_health(Health {
                on_non_finite: policy,
                ..Health::quiet()
            })
            .unwrap();
            t.step(&net, full.view()).unwrap();
            let mut before = params.clone();
            t.sync_to(&mut before);
            let state = t.optimizer_state();
            let r = t.step(&net, bad.view());
            match policy {
                OnNonFinite::Skip => assert!(r.unwrap().is_nan() && t.skipped_steps() == 1),
                _ => assert!(matches!(r, Err(Error::NonFinite { step: 2, .. })), "{r:?}"),
            }
            let mut after = params.clone();
            t.sync_to(&mut after);
            for (name, id) in net.state_names() {
                let bits = |p: &ParamStore| {
                    p.value(id)
                        .data()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<u32>>()
                };
                assert_eq!(bits(&before), bits(&after), "{policy:?}: {name}");
            }
            assert_eq!(t.optimizer_state(), state, "{policy:?}");
            assert!(t.step(&net, full.view()).unwrap().is_finite());
            assert_eq!(t.optimizer_state().steps, vec![2; state.steps.len()]);
        });
    }
}

/// Two epochs of 50,000 rows at batch 128 on the MLP with Adam: 390 full batches, then one of
/// 80, each epoch.
#[test]
fn fifty_thousand_rows_at_batch_128() {
    let mut params = ParamStore::new();
    let net = mlp(&mut params, 73);
    let batches: Vec<Data> = (0..391)
        .map(|s| Data::new(200 + s, if s < 390 { 128 } else { 80 }, 12, 4))
        .collect();
    assert_eq!(batches.iter().map(|d| d.rows).sum::<usize>(), 50_000);
    let adam = OptimizerConfig::Adam {
        lr: 1e-3,
        weight_decay: 0.0,
    };
    // Two epochs, so the short step's update is read by the next epoch's steps.
    track(
        &net,
        &params,
        &adam,
        128,
        &batches,
        2,
        "mlp, 50,000 rows at 128",
    );
}

/// A batch of no rows, of more rows than the batch size, and a short batch that would leave
/// batch norm one value per channel are refused, and the trainer then trains on.
#[test]
fn batches_a_trainer_refuses() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(79);
    let net = Sequential::new()
        .push(Linear::new(&mut params, 3, 4, &mut rng))
        .push(BatchNorm2d::new(&mut params, 4, (1, 1)))
        .push(Relu)
        .push(Linear::new(&mut params, 4, 2, &mut rng));
    let sgd = OptimizerConfig::Sgd {
        lr: 0.1,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    for_each_gpu(|gpu| {
        let mut t = GpuTrainer::new(&net, &params, &sgd, Loss::CrossEntropy, 4, gpu).unwrap();
        t.set_health(Health::quiet()).unwrap();
        let empty = BatchView {
            x: TensorView::from_slice(&[], &[0, 3]).unwrap(),
            targets: TargetsView::Classes(&[]),
        };
        let five = Data::new(305, 5, 3, 2);
        let one = Data::new(301, 1, 3, 2);
        for (batch, what) in [
            (empty, "0 rows"),
            (five.view(), "5 rows"),
            (one.view(), "1 row"),
        ] {
            let r = t.step(&net, batch);
            assert!(matches!(r, Err(Error::InvalidData(_))), "{what}: {r:?}");
        }
        assert!(t.step(&net, Data::new(310, 2, 3, 2).view()).is_ok());
        assert!(t.step(&net, Data::new(311, 4, 3, 2).view()).is_ok());
    });
}
