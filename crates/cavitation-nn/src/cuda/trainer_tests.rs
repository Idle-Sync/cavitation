//! `CudaTrainer` against the CPU: each parameter's gradient after one step against the `f64`
//! training tape's from the same weights, batch and dropout masks, on the test MLP (with a
//! neuron mask), a small CNN (convolution, batch norm, max pooling, dropout), `mnist-cnn` and
//! `cifar-cnn`, with the max-pooling winners that differ from the `f64` tape's listed; the
//! parameters after one SGD step within the update's bound; two runs giving the same bits; a
//! step on non-finite inputs leaving the parameters and the optimiser's state as they were;
//! five steps tracking the CPU `Trainer`; and a profiled step.
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

use cavitation_core::{Graph, ParamStore, Reduction, Tensor};

use super::CudaTrainer;
use super::test_util::{Data, Twin, cifar_cnn, mlp, mnist_cnn, off, small_cnn, twin, with_cuda};
use crate::{
    Health, InferenceLayer, Loss, NeuronMask, OnNonFinite, OptimizerConfig, Trace, Trainer,
};

/// The unit roundoff of `f32`.
const U: f64 = 1.0 / (1u64 << 24) as f64;

/// A layer and its input on the `f64` tape.
type Pooled = (usize, Vec<f64>);

/// Each parameter's gradient on the `f64` training tape (seed 0, step 1, as a trainer's
/// first step), with cross-entropy, in the `f64` store's parameter order, and each max
/// pooling's and each ReLU's input, by layer.
fn tape_gradients(t: &mut Twin, data: &Data) -> (Vec<Vec<f64>>, Vec<Pooled>, Vec<Pooled>) {
    let x64: Vec<f64> = data.x.iter().map(|&v| f64::from(v)).collect();
    let input = Tensor::from_vec(x64, &[data.rows, data.x.len() / data.rows]).unwrap();
    let mut graph = Graph::<f64>::default();
    let mut pooled = Vec::new();
    let mut relus = Vec::new();
    {
        let mut tape = graph.training_tape(&t.params64.values, 0, 1);
        let v = tape.input(input.view());
        let mut trace = Trace::new();
        let out = t.net64.forward_traced(&mut tape, v, &mut trace);
        for i in 0..t.net.len() {
            if matches!(
                t.net.layer(i).inference(&t.params.values),
                Some(InferenceLayer::MaxPool2d(_))
            ) {
                pooled.push((i, tape.value(trace.input(i)).data().to_vec()));
            }
            if t.net.layer(i).name() == "relu" {
                relus.push((i, tape.value(trace.input(i)).data().to_vec()));
            }
        }
        let loss = tape.softmax_cross_entropy(out, &data.labels, Reduction::Mean);
        tape.backward(loss, &mut t.params64.grads);
    }
    let grads = (t.net64.parameter_names().into_iter())
        .map(|(_, id)| t.params64.grad(id).data().to_vec())
        .collect();
    (grads, pooled, relus)
}

/// The max-pooling windows whose winner on the `f64` tape's input differs from the card's:
/// `(layer, output index)`, with the window's two largest `f64` values.
fn flipped_winners(
    t: &Twin,
    pooled: &[Pooled],
    card: &[Vec<u32>],
    rows: usize,
) -> Vec<(usize, usize, f64, f64)> {
    let mut flips = Vec::new();
    for ((layer, input), winners) in pooled.iter().zip(card) {
        let Some(InferenceLayer::MaxPool2d(g)) = t.net.layer(*layer).inference(&t.params.values)
        else {
            unreachable!("only max pooling layers are listed")
        };
        let (h, w) = g.input();
        let (kh, kw) = g.kernel();
        let (sh, sw) = g.stride();
        let out_w = (w - kw) / sw + 1;
        let out_h = (h - kh) / sh + 1;
        for img in 0..rows * g.channels() {
            let plane = &input[img * h * w..(img + 1) * h * w];
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut best = (f64::NEG_INFINITY, 0usize, f64::NEG_INFINITY);
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let at = (oy * sh + ky) * w + ox * sw + kx;
                            let v = plane[at];
                            if v > best.0 {
                                best = (v, at, best.0);
                            } else if v > best.2 {
                                best.2 = v;
                            }
                        }
                    }
                    let o = img * out_h * out_w + oy * out_w + ox;
                    if winners[o] as usize != best.1 {
                        flips.push((*layer, o, best.0, best.2));
                    }
                }
            }
        }
    }
    flips
}

/// The ReLU inputs whose sign on the card differs from the `f64` tape's, `(layer, index)`,
/// with both values: the ReLU passes a gradient on one side and none on the other.
fn flipped_relus(tape: &[Pooled], card: &[(usize, Vec<f32>)]) -> Vec<(usize, usize, f64, f32)> {
    let mut flips = Vec::new();
    for ((layer, want), (_, got)) in tape.iter().zip(card) {
        for (k, (&w, &g)) in want.iter().zip(got).enumerate() {
            if (w > 0.0) != (g > 0.0) {
                flips.push((*layer, k, w, g));
            }
        }
    }
    flips
}

/// One trainer step with a learning rate of 0, and each parameter's gradient on the card
/// against the `f64` tape's, with the flipped max-pooling winners and ReLU signs listed.
fn check_gradients(mut t: Twin, data: &Data, what: &str) {
    let (want, pooled, relus) = tape_gradients(&mut t, data);
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
        let winners = trainer.last_winners();
        let flips = flipped_winners(&t, &pooled, &winners, data.rows);
        println!(
            "{what}: {} max-pooling winners differ from the f64 tape's",
            flips.len()
        );
        for (layer, o, a, b) in flips.iter().take(20) {
            println!("  layer {layer}, output {o}: f64 window's top two {a:e}, {b:e}");
        }
        let signs = flipped_relus(&relus, &trainer.last_relu_inputs());
        println!(
            "{what}: {} ReLU inputs have another sign than on the f64 tape",
            signs.len()
        );
        for (layer, k, w, g) in signs.iter().take(20) {
            println!("  layer {layer}, value {k}: f64 {w:e}, card {g:e}");
        }
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
        // A flipped winner sends a gradient to another input, and a flipped ReLU passes a
        // gradient the tape stops or stops one it passes, which no tolerance covers, so only
        // the layers after the last flip are asserted.
        let after_flips = (flips.iter().map(|f| f.0))
            .chain(signs.iter().map(|f| f.0))
            .max()
            .map_or(0, |layer| layer + 1);
        let mut worst = (0.0f64, String::new());
        let largest = want.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max);
        for ((name, w), (_, g)) in names.iter().zip(&want).zip(&got) {
            let scale = w.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let tolerance = scale / 4096.0 + largest / 1_048_576.0;
            let off = (g.iter().zip(w))
                .map(|(&a, &b)| (f64::from(a) - b).abs())
                .fold(0.0, f64::max);
            let ratio = off / tolerance;
            let layer: usize = name.split('.').next().and_then(|l| l.parse().ok()).unwrap();
            if layer >= after_flips {
                assert!(
                    ratio <= 1.0,
                    "{what}: {name}: off by {off:e} of a largest gradient {scale:e}"
                );
            }
            println!(
                "{what} {name}: worst {ratio:.4} of the tolerance (largest gradient {scale:.3e})"
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
/// median of 21 steps after 5 warm-up steps, each from the call to its return; then one
/// profiled step's parts.
/// A measurement, printed, not asserted. Run with `--ignored`.
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
        trainer.profile_next_step();
        trainer.step(&t.net, data.view()).unwrap();
        let p = trainer.last_profile().unwrap();
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        for i in 0..t.net.len() {
            let input = p.backward_input[i].map_or(String::new(), |d| {
                format!(" (input gradient {:.3} ms)", ms(d))
            });
            println!(
                "  {i:>2} {:<12} forward {:.3} ms, backward {:.3} ms{input}",
                t.net.layer(i).name(),
                ms(p.forward[i]),
                ms(p.backward[i]),
            );
        }
        println!(
            "  loss {:.3} ms, health {:.3} ms, optimizer {:.3} ms, running {:.3} ms, readback \
             {:.3} ms, total {:.3} ms",
            ms(p.loss),
            ms(p.health),
            ms(p.optimizer),
            ms(p.running),
            ms(p.readback),
            ms(p.total)
        );
    });
}
