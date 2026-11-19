//! `GpuTrainer::profile_next_step`: a profile with every layer of a CNN present, forward and
//! backward, each part timed where the device times it and `None` with the reason named
//! where it does not, the parts no longer than the whole step, a `Linear`'s or a `Conv2d`'s
//! input gradient timed apart from its parameter gradients; a profile only for the step
//! that asked; and a step left out under `Skip` passing the request to the next.
//!
//! Without a GPU every test says so and checks nothing; `CAVITATION_GPU_REQUIRED=1` makes
//! that a failure.

use std::time::Duration;

use cavitation_core::{Conv2dGeom, ParamStore, Pool2dGeom, Rng};
use cavitation_gpu::Timing;

use super::GpuTrainer;
use super::health_tests::{Data, for_each_gpu};
use crate::{
    BatchNorm2d, Conv2d, Dropout, Flatten, Health, Linear, Loss, MaxPool2d, OnNonFinite,
    OptimizerConfig, Relu, Sequential,
};

/// A CNN of eight layers on 2 × 8 × 8 images.
fn cnn(params: &mut ParamStore) -> Sequential {
    let mut rng = Rng::seed(81);
    let g = Conv2dGeom::with_options(2, 4, (8, 8), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
    Sequential::new()
        .push(Conv2d::new(params, g, &mut rng))
        .push(BatchNorm2d::new(params, 4, (8, 8)))
        .push(Relu)
        .push(MaxPool2d::new(Pool2dGeom::new(4, (8, 8), (2, 2)).unwrap()))
        .push(Dropout::new(0.3).unwrap())
        .push(Flatten)
        .push(Linear::new(params, 64, 8, &mut rng))
        .push(Linear::new(params, 8, 5, &mut rng))
}

/// A profiled step of the CNN has every layer's forward and backward time, the loss, the
/// health values, the update and the running statistics, each `Some` unless the device times
/// nothing; the copies and the readback are timed exactly where the device times copies; and
/// the timed parts add up to no more than the step on the host's clock.
#[test]
fn a_profile_has_every_layer() {
    let mut params = ParamStore::new();
    let net = cnn(&mut params);
    let data = Data::new(400, 6, 2 * 64, 5);
    let sgd = OptimizerConfig::Sgd {
        lr: 0.05,
        momentum: 0.9,
        weight_decay: 0.0,
    };
    for_each_gpu(|gpu| {
        let mut t = GpuTrainer::new(&net, &params, &sgd, Loss::CrossEntropy, 6, gpu).unwrap();
        t.set_health(Health::quiet()).unwrap();
        assert!(t.last_profile().is_none());
        t.step(&net, data.view()).unwrap();
        t.profile_next_step();
        t.step(&net, data.view()).unwrap();
        let p = t
            .last_profile()
            .expect("the step was asked for a profile")
            .clone();
        println!("{:?}: {p:#?}", p.timing);
        assert_eq!((p.forward.len(), p.backward.len()), (net.len(), net.len()));
        let timed = !matches!(p.timing, Timing::Untimed { .. });
        let mut parts = p.forward.clone();
        parts.extend(&p.backward);
        parts.extend([p.loss, p.health, p.optimizer, p.running]);
        assert!(
            parts.iter().all(|t| t.is_some() == timed),
            "{:?}: {parts:?}",
            p.timing
        );
        let copies = matches!(p.timing, Timing::PassesAndCopies);
        assert_eq!(p.copies.is_some(), copies, "{:?}", p.timing);
        assert_eq!(p.readback.is_some(), copies, "{:?}", p.timing);
        parts.extend([p.copies, p.readback]);
        let sum: Duration = parts.iter().flatten().sum();
        assert!(sum <= p.total, "{sum:?} of parts against {:?}", p.total);
        // The input gradient is timed apart for the two `Linear` layers and for no other: the
        // convolution is the first layer and computes none. It is a part of the layer's
        // backward time, and the parameter gradients, a pass of their own, are the rest. Neither
        // part is asserted above 0: a short pass's two timestamps can read the same tick.
        assert_eq!(p.backward_input.len(), net.len());
        for (i, (input, whole)) in p.backward_input.iter().zip(&p.backward).enumerate() {
            let split = i == 6 || i == 7;
            assert_eq!(input.is_some(), split && timed, "layer {i}: {input:?}");
            if let (Some(input), Some(whole)) = (input, whole) {
                assert!(input <= whole, "layer {i}: {input:?} of {whole:?}");
            }
        }
        // Each time is its own layer's: `Flatten` (no work) is the shortest both ways, which a
        // profile with the layers in the wrong order would not show. No ordering between layers
        // that do work is asserted: one step's times are single samples, and on some backends
        // (Vulkan and DX12 on an NVIDIA GTX 1660 Super) batch norm's or a `Linear`'s backward
        // pass has come out longer than the convolution's (#315).
        if timed {
            let times = |v: &[Option<Duration>]| v.iter().map(|t| t.unwrap()).collect::<Vec<_>>();
            let (fwd, bwd) = (times(&p.forward), times(&p.backward));
            assert_eq!(bwd.iter().min(), Some(&bwd[5]), "{bwd:?}");
            assert_eq!(fwd.iter().min(), Some(&fwd[5]), "{fwd:?}");
        }
        // The next step was not asked.
        t.step(&net, data.view()).unwrap();
        assert!(t.last_profile().is_none());
    });
}

/// A step left out under `Skip` has no profile, and the next step, which trains, has one.
#[test]
fn a_skipped_step_passes_the_profile_on() {
    let mut params = ParamStore::new();
    let net = cnn(&mut params);
    let good = Data::new(410, 6, 2 * 64, 5);
    let mut bad = Data::new(411, 6, 2 * 64, 5);
    bad.x[0] = f32::NAN;
    let sgd = OptimizerConfig::Sgd {
        lr: 0.05,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    for_each_gpu(|gpu| {
        let mut t = GpuTrainer::new(&net, &params, &sgd, Loss::CrossEntropy, 6, gpu).unwrap();
        t.set_health(Health {
            on_non_finite: OnNonFinite::Skip,
            ..Health::quiet()
        })
        .unwrap();
        t.profile_next_step();
        assert!(t.step(&net, bad.view()).unwrap().is_nan());
        assert!(t.last_profile().is_none());
        t.step(&net, good.view()).unwrap();
        assert!(t.last_profile().is_some());
    });
}

/// A model with nine batch norms (each running update its own pass) is profiled: the timed
/// passes are sized for them.
#[test]
fn nine_batch_norms_are_profiled() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(83);
    let mut net = Sequential::new();
    for i in 0..9 {
        let g = Conv2dGeom::with_options(
            if i == 0 { 2 } else { 4 },
            4,
            (4, 4),
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
        )
        .unwrap();
        net = net
            .push(Conv2d::new(&mut params, g, &mut rng))
            .push(BatchNorm2d::new(&mut params, 4, (4, 4)))
            .push(Relu);
    }
    let net = net
        .push(Flatten)
        .push(Linear::new(&mut params, 64, 3, &mut rng));
    let data = Data::new(420, 4, 2 * 16, 3);
    let sgd = OptimizerConfig::Sgd {
        lr: 0.01,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    for_each_gpu(|gpu| {
        let mut t = GpuTrainer::new(&net, &params, &sgd, Loss::CrossEntropy, 4, gpu).unwrap();
        t.set_health(Health::quiet()).unwrap();
        t.profile_next_step();
        t.step(&net, data.view()).unwrap();
        let p = t.last_profile().unwrap();
        assert_eq!(p.forward.len(), net.len());
        if !matches!(p.timing, Timing::Untimed { .. }) {
            assert!(p.running.is_some());
        }
    });
}
