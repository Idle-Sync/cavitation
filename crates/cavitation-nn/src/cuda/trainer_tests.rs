//! `CudaTrainer` against the CPU: each parameter's gradient after one step against the `f64`
//! training tape's from the same weights, batch and dropout masks, on the test MLP (with a
//! neuron mask), a small CNN (convolution, batch norm, max pooling, dropout), a CNN through
//! average and adaptive average pooling, `mnist-cnn` and `cifar-cnn`; the parameters after one
//! SGD step within the update's bound; two runs giving the same bits; a step on non-finite
//! inputs leaving the parameters and the optimiser's state as they were; five steps tracking
//! the CPU `Trainer`; and a profiled step.
//!
//! The `f64` reference takes the card's discrete decisions: each ReLU keeps the values whose
//! input was above 0 on the card, and each max pooling takes the card's winners (as a sum over
//! a window in which only the winner is kept). A value near 0 or a near tie can go either way
//! in `f32`, and a decision the reference took the other way would send a gradient where the
//! card sends none, which no tolerance covers; with the card's decisions every layer is
//! asserted. The winners are checked first, exactly, against the first largest value of each
//! window of the card's own `f32` pooling input.
//!
//! The gradient check is the wgpu engine's tolerance, not a derived bound: each tensor's
//! worst difference from the `f64` gradient must be under 2⁻¹² of the tensor's largest `f64`
//! gradient plus 2⁻²⁰ of the network's largest, and the tests print the measured worst ratio
//! to that tolerance. cuDNN and cuBLAS publish no error bounds, so for their passes this is a
//! measured agreement, not a proof.
//!
//! Without CUDA every test says so and checks nothing; `CAVITATION_CUDA_REQUIRED=1` makes
//! that a failure.

use std::time::Duration;

use cavitation_core::{Gate, Graph, ParamStore, Pool2dGeom, Reduction, Tensor};

use super::CudaTrainer;
use super::test_util::{
    ADAPTIVE_OUTPUTS, Data, GLU_GATES, Twin, activation_mlp, cifar_cnn, glu_mlp, mlp, mnist_cnn,
    more_activations, off, pool_cnn, prelu_mlp, small_cnn, softmax_mlp, twin, with_cuda,
};
use crate::{Health, InferenceLayer, Loss, NeuronMask, OnNonFinite, OptimizerConfig, Trainer};

/// The unit roundoff of `f32`.
const U: f64 = 1.0 / (1u64 << 24) as f64;

/// What the card decided in a step: each ReLU's keep mask and each max pooling's winners as
/// a keep mask over its input, both one value per input value, by layer.
#[derive(Debug, Default)]
struct Decisions {
    /// Each ReLU layer and 1 where its input was above 0 on the card, 0 elsewhere.
    relus: Vec<(usize, Vec<f64>)>,
    /// Each max pooling layer, its shape, and 1 at each output's winner in its input.
    pools: Vec<(usize, Pool2dGeom, Vec<f64>)>,
}

/// Each parameter's gradient on the `f64` training tape (seed 0, step 1, as a trainer's
/// first step), with cross-entropy, in the `f64` store's parameter order, with each ReLU and
/// max pooling taking the card's `decisions`.
fn tape_gradients(t: &mut Twin, data: &Data, decisions: &Decisions) -> Vec<Vec<f64>> {
    let width = data.x.len() / data.rows;
    let x64: Vec<f64> = data.x.iter().map(|&v| f64::from(v)).collect();
    let input = Tensor::from_vec(x64, &[data.rows, width]).unwrap();
    let mut graph = Graph::<f64>::default();
    {
        let mut tape = graph.training_tape(&t.params64.values, 0, 1);
        let mut v = tape.input(input.view());
        for i in 0..t.net64.len() {
            let shape = [data.rows, tape.value(v).data().len() / data.rows];
            let keep_of = |values: &[f64]| Tensor::from_vec(values.to_vec(), &shape).unwrap();
            let mut y = if let Some((_, keep)) = decisions.relus.iter().find(|r| r.0 == i) {
                let keep = tape.input(keep_of(keep).view());
                tape.keep_where(v, keep)
            } else if let Some((_, geom, keep)) = decisions.pools.iter().find(|p| p.0 == i) {
                // Only the winner of each window is kept, so the window's sum is its value.
                let keep = tape.input(keep_of(keep).view());
                let kept = tape.keep_where(v, keep);
                tape.avg_pool2d(kept, *geom, true, Some(1))
            } else {
                t.net64.layer(i).forward(&mut tape, v)
            };
            if let Some(mask) = t.net64.mask(i) {
                let on: Vec<f64> = (0..mask.width())
                    .map(|j| f64::from(u8::from(mask.is_on(j))))
                    .collect();
                let keep = tape.input(Tensor::from_vec(on, &[mask.width()]).unwrap().view());
                y = tape.keep_where(y, keep);
            }
            v = y;
        }
        let loss = tape.softmax_cross_entropy(v, &data.labels, Reduction::Mean);
        tape.backward(loss, &mut t.params64.grads);
    }
    (t.net64.parameter_names().into_iter())
        .map(|(_, id)| t.params64.grad(id).data().to_vec())
        .collect()
}

/// The card's decisions in its last step, after checking each max pooling's winners exactly
/// against the first largest value of each window of the card's own `f32` input, the taps
/// in row-major order.
fn card_decisions(t: &Twin, trainer: &CudaTrainer<'_, '_>, rows: usize, what: &str) -> Decisions {
    let relus = (trainer.last_relu_inputs().into_iter())
        .map(|(i, x)| (i, x.iter().map(|&v| f64::from(u8::from(v > 0.0))).collect()))
        .collect();
    let mut pools = Vec::new();
    for ((layer, input), winners) in trainer
        .last_pool_inputs()
        .into_iter()
        .zip(trainer.last_winners())
    {
        let Some(InferenceLayer::MaxPool2d(g)) = t.net.layer(layer).inference(&t.params.values)
        else {
            unreachable!("only max pooling layers are listed")
        };
        let ((h, w), (kh, kw), (sh, sw)) = (g.input(), g.kernel(), g.stride());
        let (out_h, out_w) = g.out_size();
        assert_eq!((g.padding(), g.dilation()), ((0, 0), (1, 1)), "{what}");
        let mut keep = vec![0.0f64; input.len()];
        for img in 0..rows * g.channels() {
            let plane = &input[img * h * w..(img + 1) * h * w];
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut best = (f32::NEG_INFINITY, usize::MAX);
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let at = (oy * sh + ky) * w + ox * sw + kx;
                            if best.1 == usize::MAX || plane[at] > best.0 {
                                best = (plane[at], at);
                            }
                        }
                    }
                    let o = img * out_h * out_w + oy * out_w + ox;
                    assert_eq!(
                        winners[o] as usize, best.1,
                        "{what}: layer {layer}, output {o}: the card's winner"
                    );
                    keep[img * h * w + best.1] = 1.0;
                }
            }
        }
        pools.push((layer, g, keep));
    }
    Decisions { relus, pools }
}

/// One trainer step with a learning rate of 0, and each parameter's gradient on the card
/// against the `f64` tape's with the card's ReLU and max-pooling decisions, every layer
/// asserted.
fn check_gradients(mut t: Twin, data: &Data, what: &str) {
    let names: Vec<String> = (t.net.parameter_names().into_iter())
        .map(|(n, _)| n)
        .collect();
    with_cuda(|cuda| {
        let still = OptimizerConfig::Sgd {
            lr: 0.0,
            momentum: 0.0,
            weight_decay: 0.0,
        };
        let mut trainer = CudaTrainer::new(
            &t.net,
            &t.params,
            &still,
            Loss::CrossEntropy,
            data.rows,
            cuda,
        )
        .unwrap();
        trainer.step(&t.net, data.view()).unwrap();
        let got = trainer.last_gradients();
        let decisions = card_decisions(&t, &trainer, data.rows, what);
        drop(trainer);
        let want = tape_gradients(&mut t, data, &decisions);
        assert_eq!(
            (names.len(), want.len()),
            (got.len(), got.len()),
            "{what}: parameter counts"
        );
        let ids = t.net.parameter_names().into_iter().map(|(_, id)| id);
        assert!(
            got.iter().map(|(id, _)| *id).eq(ids),
            "{what}: parameter order"
        );
        let mut worst = (0.0f64, String::new());
        let largest = want.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max);
        for ((name, w), (_, g)) in names.iter().zip(&want).zip(&got) {
            let scale = w.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let tolerance = scale / 4096.0 + largest / 1_048_576.0;
            let off = (g.iter().zip(w))
                .map(|(&a, &b)| (f64::from(a) - b).abs())
                .fold(0.0, f64::max);
            let ratio = off / tolerance;
            println!(
                "{what} {name}: worst {ratio:.4} of the tolerance (largest gradient {scale:.3e})"
            );
            assert!(
                ratio <= 1.0,
                "{what}: {name}: off by {off:e} of a largest gradient {scale:e}"
            );
            if ratio > worst.0 {
                worst = (ratio, name.clone());
            }
        }
        println!(
            "{what}: worst {:.4} of the tolerance ({})",
            worst.0, worst.1
        );
    });
}

/// The test MLP (with a neuron mask) and a small CNN with batch norm, pooling and dropout.
#[test]
fn gradients_match_the_tape() {
    let mut t = twin(mlp::<f64>, mlp::<f32>, 21);
    let mask = NeuronMask::from_off(16, &[1, 5]);
    t.net.set_mask(1, mask.clone());
    t.net64.set_mask(1, mask);
    check_gradients(t, &Data::within(1, 24, 12, 4, (-2.0, 2.0)), "mlp");
    let t = twin(small_cnn::<f64>, small_cnn::<f32>, 22);
    check_gradients(t, &Data::new(2, 6, 2 * 64, 5), "small cnn");
}

/// Each activation beyond ReLU, tanh, the sigmoid and GELU, twice in an MLP: the card's
/// gradients against the `f64` tape's, within the tolerance above.
#[test]
fn every_activation_matches_the_tape() {
    for (i, kind) in more_activations().into_iter().enumerate() {
        let seed = 60 + i as u64;
        let t = twin(
            |p, s| activation_mlp(p, s, kind),
            |p, s| activation_mlp(p, s, kind),
            seed,
        );
        check_gradients(
            t,
            &Data::within(seed, 24, 12, 4, (-2.0, 2.0)),
            &format!("{kind:?}"),
        );
    }
}

/// Softmax and log-softmax after each `Linear` of an MLP: the card's gradients against the
/// `f64` tape's, within the tolerance above.
#[test]
fn softmax_layers_match_the_tape() {
    for (seed, log) in [(90, false), (91, true)] {
        let t = twin(
            |p, s| softmax_mlp(p, s, log),
            |p, s| softmax_mlp(p, s, log),
            seed,
        );
        let what = if log { "log-softmax" } else { "softmax" };
        check_gradients(t, &Data::within(seed, 24, 12, 4, (-2.0, 2.0)), what);
    }
}

/// A PReLU before each `Linear` of an MLP, the first layer among them, with one slope per
/// column and with one shared slope: the card's gradients, the slopes' among them, against
/// the `f64` tape's, within the tolerance above.
#[test]
fn prelu_layers_match_the_tape() {
    for (seed, shared) in [(94, false), (95, true)] {
        let t = twin(
            |p, s| prelu_mlp(p, s, shared),
            |p, s| prelu_mlp(p, s, shared),
            seed,
        );
        let what = if shared { "shared PReLU" } else { "PReLU" };
        check_gradients(t, &Data::within(seed, 24, 12, 4, (-2.0, 2.0)), what);
    }
}

/// A GLU of each gate after the first two `Linear`s of an MLP, and with the ReLU and SiLU
/// gates a GLU as the first layer too (the input 24 wide): the card's gradients against the
/// `f64` tape's, within the tolerance above.
#[test]
fn glu_layers_match_the_tape() {
    let leading = [(Gate::Relu, true), (Gate::Silu, true)];
    let cases = GLU_GATES.into_iter().map(|g| (g, false)).chain(leading);
    for (k, (gate, lead)) in cases.enumerate() {
        let seed = 120 + k as u64;
        let t = twin(
            |p, s| glu_mlp(p, s, gate, lead),
            |p, s| glu_mlp(p, s, gate, lead),
            seed,
        );
        let width = if lead { 24 } else { 12 };
        let what = format!("GLU with {gate:?}{}", if lead { ", leading" } else { "" });
        check_gradients(t, &Data::within(seed, 24, width, 4, (-2.0, 2.0)), &what);
    }
}

/// A CNN through average pooling and adaptive average pooling to each of
/// `ADAPTIVE_OUTPUTS` (overlapping windows, global pooling, an output larger than its input):
/// the card's gradients against the `f64` tape's, within the tolerance above.
#[test]
fn pool_layers_match_the_tape() {
    for (k, to) in ADAPTIVE_OUTPUTS.into_iter().enumerate() {
        let seed = 150 + k as u64;
        let t = twin(|p, s| pool_cnn(p, s, to), |p, s| pool_cnn(p, s, to), seed);
        let what = format!("adaptive pooling to {to:?}");
        check_gradients(t, &Data::new(seed, 6, 98, 5), &what);
    }
}

/// One `mnist-cnn` step and one `cifar-cnn` step.
#[test]
fn benchmark_networks_match_the_tape() {
    let t = twin(mnist_cnn::<f64>, mnist_cnn::<f32>, 23);
    check_gradients(
        t,
        &Data::within(3, 16, 28 * 28, 10, (0.0, 1.0)),
        "mnist-cnn",
    );
    let t = twin(cifar_cnn::<f64>, cifar_cnn::<f32>, 24);
    check_gradients(t, &Data::new(4, 8, 3 * 32 * 32, 10), "cifar-cnn");
}

/// The parameters after one SGD step are the old ones minus the learning rate times the
/// card's gradient, within the update's bound: `u·|p'| + u·(1 + u)·lr·|g|`.
#[test]
fn sgd_updates_within_its_bound() {
    let t = twin(small_cnn::<f64>, small_cnn::<f32>, 25);
    let data = Data::new(5, 6, 2 * 64, 5);
    let lr = 0.1;
    with_cuda(|cuda| {
        let sgd = OptimizerConfig::Sgd {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
        };
        let mut trainer =
            CudaTrainer::new(&t.net, &t.params, &sgd, Loss::CrossEntropy, 6, cuda).unwrap();
        trainer.step(&t.net, data.view()).unwrap();
        let grads = trainer.last_gradients();
        let mut after = t.params.clone();
        trainer.sync_to(&mut after);
        for (id, g) in grads {
            let (p0, p1) = (t.params.value(id).data(), after.value(id).data());
            for ((&a, &b), &g) in p0.iter().zip(p1).zip(&g) {
                let want = f64::from(a) - lr * f64::from(g);
                let bound = U * f64::from(b).abs() + U * (1.0 + U) * lr * f64::from(g).abs();
                assert!(
                    (f64::from(b) - want).abs() <= bound,
                    "{a} - {lr}·{g} gave {b}"
                );
            }
        }
    });
}

/// Two runs of three Adam steps from the same start give the same bits, in the losses, the
/// parameters, the running statistics, the batch counts and the optimiser's state; the batch
/// count went up once per step.
#[test]
fn two_runs_give_the_same_bits() {
    let t = twin(small_cnn::<f64>, small_cnn::<f32>, 26);
    let batches: Vec<Data> = (0..3).map(|s| Data::new(10 + s, 6, 2 * 64, 5)).collect();
    with_cuda(|cuda| {
        let run = || {
            let adam = OptimizerConfig::Adam {
                lr: 1e-2,
                weight_decay: 0.0,
            };
            let mut trainer =
                CudaTrainer::new(&t.net, &t.params, &adam, Loss::CrossEntropy, 6, cuda).unwrap();
            let losses: Vec<u32> = (batches.iter())
                .map(|d| trainer.step(&t.net, d.view()).unwrap().to_bits())
                .collect();
            let mut out = t.params.clone();
            trainer.sync_to(&mut out);
            let bits: Vec<Vec<u32>> = (t.net.state_names().into_iter())
                .map(|(_, id)| out.value(id).data().iter().map(|v| v.to_bits()).collect())
                .collect();
            (losses, bits, trainer.optimizer_state(), out)
        };
        let (a, b) = (run(), run());
        assert_eq!(a.0, b.0, "losses");
        assert_eq!(a.1, b.1, "parameters and buffers");
        assert_eq!(a.2, b.2, "optimiser state");
        assert_eq!(a.2.steps, vec![3; a.2.steps.len()]);
        let count = (t.net.layer(1).as_batch_norm2d())
            .and_then(|bn| bn.running())
            .and_then(|r| r.count)
            .unwrap();
        assert_eq!(a.3.value(count).data()[0], 3.0);
    });
}

/// A batch with a NaN input gives `Error::NonFinite`, and leaves the parameters, the running
/// statistics, the batch count and the optimiser's update counts as they were; the step is
/// counted, and the next good step trains.
#[test]
fn a_non_finite_step_changes_nothing() {
    let t = twin(small_cnn::<f64>, small_cnn::<f32>, 27);
    let good = Data::new(6, 6, 2 * 64, 5);
    let mut bad = Data::new(7, 6, 2 * 64, 5);
    bad.x[3] = f32::NAN;
    with_cuda(|cuda| {
        let adam = OptimizerConfig::Adam {
            lr: 1e-2,
            weight_decay: 0.0,
        };
        let mut trainer =
            CudaTrainer::new(&t.net, &t.params, &adam, Loss::CrossEntropy, 6, cuda).unwrap();
        trainer.step(&t.net, good.view()).unwrap();
        let mut before = t.params.clone();
        trainer.sync_to(&mut before);
        let state = trainer.optimizer_state();
        let err = trainer.step(&t.net, bad.view()).unwrap_err();
        assert!(
            err.to_string().contains("non-finite value at step 2"),
            "{err}"
        );
        assert_eq!(trainer.steps(), 2);
        let mut after = t.params.clone();
        trainer.sync_to(&mut after);
        for (name, id) in t.net.state_names() {
            let bits = |p: &ParamStore| {
                (p.value(id).data().iter())
                    .map(|v| v.to_bits())
                    .collect::<Vec<u32>>()
            };
            assert_eq!(bits(&before), bits(&after), "{name}");
        }
        assert_eq!(trainer.optimizer_state(), state);
        assert!(trainer.step(&t.net, good.view()).unwrap().is_finite());
        assert_eq!(trainer.optimizer_state().steps, vec![2; state.steps.len()]);
    });
}

/// Five steps of the CUDA trainer against five of the CPU's `Trainer` from the same start, on
/// the small CNN (batch norm, max pooling, dropout) with SGD and momentum, on the MLP with
/// Adam and on the MLP with AdamW: each step's loss within 2⁻¹⁰ of the CPU's, and the
/// parameters, running statistics and batch counts after the last within 2⁻¹⁰ of each
/// tensor's largest value plus 2⁻¹⁶.
#[test]
fn steps_track_the_cpu_trainer() {
    let cases = [
        (
            twin(small_cnn::<f64>, small_cnn::<f32>, 29),
            2 * 64,
            5,
            OptimizerConfig::Sgd {
                lr: 0.05,
                momentum: 0.9,
                weight_decay: 1e-4,
            },
            "small cnn, sgd",
        ),
        (
            twin(mlp::<f64>, mlp::<f32>, 30),
            12,
            4,
            OptimizerConfig::Adam {
                lr: 1e-2,
                weight_decay: 0.0,
            },
            "mlp, adam",
        ),
        (
            twin(mlp::<f64>, mlp::<f32>, 31),
            12,
            4,
            OptimizerConfig::AdamW {
                lr: 1e-2,
                weight_decay: 1e-2,
            },
            "mlp, adamw",
        ),
    ];
    for (t, width, classes, config, what) in cases {
        let batches: Vec<Data> = (0..5)
            .map(|s| Data::new(40 + s, 6, width, classes))
            .collect();
        let mut cpu = t.params.clone();
        let mut trainer = Trainer::new(&t.net, config.build(), Loss::CrossEntropy, 6);
        let cpu_losses: Vec<f32> = (batches.iter())
            .map(|d| trainer.step(&t.net, &mut cpu, d.view()).unwrap())
            .collect();
        with_cuda(|cuda| {
            let mut card =
                CudaTrainer::new(&t.net, &t.params, &config, Loss::CrossEntropy, 6, cuda).unwrap();
            for (s, d) in batches.iter().enumerate() {
                let loss = card.step(&t.net, d.view()).unwrap();
                let want = cpu_losses[s];
                assert!(
                    (loss - want).abs() <= want.abs() / 1024.0,
                    "{what}: step {}: loss {loss}, the CPU's {want}",
                    s + 1
                );
            }
            let mut out = t.params.clone();
            card.sync_to(&mut out);
            for (name, id) in t.net.state_names() {
                let (off, scale) = off(out.value(id).data(), cpu.value(id).data());
                assert!(
                    off <= scale / 1024.0 + 1.0 / 65536.0,
                    "{what}: {name} off by {off:e} of {scale:e}"
                );
                println!("{what} {name}: off by {off:.3e} of a largest value {scale:.3e}");
            }
        });
    }
}

/// A profiled step times every layer's forward and backward pass, with a `Linear`'s input
/// gradient apart after the first layer, its parts within its total; the next step, not
/// asked, has no profile; a step left out under `Skip` passes the request on.
#[test]
fn a_profile_has_every_layer() {
    let t = twin(small_cnn::<f64>, small_cnn::<f32>, 32);
    let data = Data::new(8, 6, 2 * 64, 5);
    let mut bad = Data::new(9, 6, 2 * 64, 5);
    bad.x[0] = f32::NAN;
    with_cuda(|cuda| {
        let sgd = OptimizerConfig::Sgd {
            lr: 0.01,
            momentum: 0.0,
            weight_decay: 0.0,
        };
        let mut trainer =
            CudaTrainer::new(&t.net, &t.params, &sgd, Loss::CrossEntropy, 6, cuda).unwrap();
        trainer
            .set_health(Health {
                on_non_finite: OnNonFinite::Skip,
                ..Health::quiet()
            })
            .unwrap();
        trainer.profile_next_step();
        trainer.step(&t.net, bad.view()).unwrap();
        assert!(trainer.last_profile().is_none(), "a step left out");
        trainer.step(&t.net, data.view()).unwrap();
        let p = trainer.last_profile().expect("asked again").clone();
        let layers = t.net.len();
        assert_eq!((p.forward.len(), p.backward.len()), (layers, layers));
        let split: Vec<bool> = p.backward_input.iter().map(Option::is_some).collect();
        // The convolution is layer 0, with no input gradient; the `Linear` is the last.
        let mut want = vec![false; layers];
        want[layers - 1] = true;
        assert_eq!(split, want);
        let parts: Duration = p.forward.iter().chain(&p.backward).sum::<Duration>()
            + p.loss
            + p.health
            + p.optimizer
            + p.running
            + p.readback;
        assert!(parts <= p.total, "{p:?}");
        trainer.step(&t.net, data.view()).unwrap();
        assert!(trainer.last_profile().is_none(), "not asked");
    });
}

/// The time of a `cifar-cnn` training step at batch 128 with Adam, as #311 timed it: the
/// median of 21 steps after 5 warm-up steps, each from the call to its return; then the
/// median of each part over 11 profiled steps. The health checks are the default ones
/// (`Health::quiet` only drops the warnings). A measurement, printed, not asserted. Run
/// with `--ignored`.
#[test]
#[ignore = "a measurement: run with --ignored"]
fn cifar_step_time() {
    use std::time::Instant;

    let t = twin(cifar_cnn::<f64>, cifar_cnn::<f32>, 28);
    let data = Data::new(8, 128, 3 * 32 * 32, 10);
    with_cuda(|cuda| {
        let adam = OptimizerConfig::Adam {
            lr: 1e-3,
            weight_decay: 0.0,
        };
        let built = Instant::now();
        let mut trainer =
            CudaTrainer::new(&t.net, &t.params, &adam, Loss::CrossEntropy, 128, cuda).unwrap();
        trainer.set_health(Health::quiet()).unwrap();
        println!("set-up {:.1} ms", built.elapsed().as_secs_f64() * 1e3);
        for _ in 0..5 {
            trainer.step(&t.net, data.view()).unwrap();
        }
        let mut times: Vec<f64> = (0..21)
            .map(|_| {
                let start = Instant::now();
                trainer.step(&t.net, data.view()).unwrap();
                start.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        times.sort_by(f64::total_cmp);
        println!(
            "cifar-cnn step at batch 128: median {:.3} ms (min {:.3}, max {:.3})",
            times[10], times[0], times[20]
        );
        let mut profiles = Vec::new();
        for _ in 0..11 {
            trainer.profile_next_step();
            trainer.step(&t.net, data.view()).unwrap();
            profiles.push(trainer.last_profile().unwrap().clone());
        }
        let median = |pick: &dyn Fn(&super::CudaStepProfile) -> Duration| {
            let mut v: Vec<f64> = (profiles.iter())
                .map(|p| pick(p).as_secs_f64() * 1e3)
                .collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        println!("medians of 11 profiled steps:");
        for i in 0..t.net.len() {
            let input = (profiles[0].backward_input[i].is_some())
                .then(|| median(&|p| p.backward_input[i].unwrap_or_default()))
                .map_or(String::new(), |v| format!(" (input gradient {v:.3} ms)"));
            println!(
                "  {i:>2} {:<12} forward {:.3} ms, backward {:.3} ms{input}",
                t.net.layer(i).name(),
                median(&|p| p.forward[i]),
                median(&|p| p.backward[i]),
            );
        }
        println!(
            "  loss {:.3} ms, health {:.3} ms, optimizer {:.3} ms, running {:.3} ms, readback \
             {:.3} ms, total {:.3} ms",
            median(&|p| p.loss),
            median(&|p| p.health),
            median(&|p| p.optimizer),
            median(&|p| p.running),
            median(&|p| p.readback),
            median(&|p| p.total)
        );
    });
}
