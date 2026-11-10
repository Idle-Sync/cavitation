//! Gradient checks of the operations, through the numerical check harness.

use std::sync::Arc;

use cavitation_core::{
    CustomOp, Gate, GeluApprox, GradcheckReport, Graph, ParamStore, Real, Reduction, Rng, Shape,
    Tape, Tensor, TensorView, TensorViewMut, Var, gradcheck, gradcheck_params,
};

/// Builds a one-dimensional `f64` tensor from a slice.
fn t(data: &[f64]) -> Tensor<f64> {
    Tensor::from_vec(data.to_vec(), &[data.len()]).unwrap()
}

/// Graph builder for gradient checks: the sum of the first two inputs.
fn add_build(tape: &mut Tape<'_, f64>, v: &[Var]) -> Var {
    tape.add(v[0], v[1])
}

/// `add` of two inputs has the right gradient for both.
#[test]
fn gradcheck_passes_for_add() {
    let report = gradcheck(add_build, &[t(&[0.5, -1.0, 2.0]), t(&[1.5, 0.25, -3.0])]);
    report.assert_passed();
    assert!(report.passed());
    assert!(report.failures.is_empty());
    assert!(report.worst_abs <= 1e-7);
}

/// One parameter and one input are both checked, and both are reported on failure.
#[test]
fn gradcheck_params_checks_parameters() {
    let mut params = ParamStore::<f64>::new_generic();
    let p = params.add(t(&[1.0, 2.0]));
    let x = t(&[3.0, -4.0]);
    let build = |tape: &mut Tape<'_, f64>, v: &[Var]| {
        let w = tape.param(p);
        tape.add(w, v[0])
    };
    let report = gradcheck_params(&mut params, &[x], build);
    report.assert_passed();
    // The parameter's value is restored exactly.
    assert_eq!(params.value(p).data(), &[1.0, 2.0]);

    // The input is not left in the store.
    assert_eq!(params.len(), 1);
}

/// Two runs give identical reports.
#[test]
fn weighting_is_deterministic() {
    let run = || -> GradcheckReport {
        gradcheck(add_build, &[t(&[0.1, 0.2, 0.3]), t(&[-0.4, 0.5, 0.6])])
    };
    assert_eq!(run(), run());
}

/// A deterministic tensor of the given shape with entries away from zero.
fn filled(dims: &[usize], seed: f64) -> Tensor<f64> {
    let n: usize = dims.iter().product();
    let data = (0..n)
        .map(|i| 0.3 + 0.17 * ((i as f64 * 1.7 + seed).sin() + 1.5))
        .collect();
    Tensor::from_vec(data, dims).unwrap()
}

/// Shape pairs for the broadcasting checks.
const BROADCAST_CASES: [(&[usize], &[usize]); 4] = [
    (&[3, 4], &[3, 4]),
    (&[3, 4], &[4]),
    (&[3, 1], &[1, 4]),
    (&[], &[5]),
];

/// `add`, `sub` and `mul` have the right gradient for both inputs under broadcasting.
#[test]
fn gradcheck_broadcast_binary_ops() {
    type Build = fn(&mut Tape<'_, f64>, &[Var]) -> Var;
    let builds: [(&str, Build); 3] = [
        ("add", |t, v| t.add(v[0], v[1])),
        ("sub", |t, v| t.sub(v[0], v[1])),
        ("mul", |t, v| t.mul(v[0], v[1])),
    ];
    for (name, build) in builds {
        for (sa, sb) in BROADCAST_CASES {
            for (da, db) in [(sa, sb), (sb, sa)] {
                let inputs = [filled(da, 0.3), filled(db, 1.1)];
                let report = gradcheck(build, &inputs);
                assert!(report.passed(), "{name} {da:?} {db:?}: {report:?}");
            }
        }
    }
}

/// `scale` has the right gradient.
#[test]
fn gradcheck_scale() {
    let report = gradcheck(|t, v| t.scale(v[0], -2.5), &[filled(&[2, 3], 0.7)]);
    report.assert_passed();
}

/// `matmul` has the right gradient for both inputs, including a 1 x 1 result.
#[test]
fn gradcheck_matmul() {
    for (da, db) in [(&[3, 4], &[4, 5]), (&[1, 7], &[7, 1])] {
        let inputs = [filled(da, 0.4), filled(db, 1.3)];
        let report = gradcheck(|t, v| t.matmul(v[0], v[1]), &inputs);
        assert!(report.passed(), "matmul {da:?} {db:?}: {report:?}");
    }
}

/// `matmul_nt` has the right gradient for both inputs.
#[test]
fn gradcheck_matmul_nt() {
    for (da, db) in [(&[2, 3], &[4, 3]), (&[1, 7], &[1, 7])] {
        let inputs = [filled(da, 0.6), filled(db, 1.9)];
        let report = gradcheck(|t, v| t.matmul_nt(v[0], v[1]), &inputs);
        assert!(report.passed(), "matmul_nt {da:?} {db:?}: {report:?}");
    }
}

/// A parameter used by two matmuls gets both contributions (overwrite, then add).
#[test]
fn gradcheck_matmul_shared_parameter() {
    let mut params = ParamStore::<f64>::new_generic();
    let w = params.add(filled(&[3, 3], 0.2));
    let build = |tape: &mut Tape<'_, f64>, v: &[Var]| {
        let wv = tape.param(w);
        let h = tape.matmul(v[0], wv);
        tape.matmul_nt(h, wv)
    };
    let report = gradcheck_params(&mut params, &[filled(&[2, 3], 0.9)], build);
    assert!(report.passed(), "{report:?}");
}

/// A `[4, 5]` input with entries drawn from `[-2, 2]` by a seeded generator, resampled until
/// every entry is farther than `1e-3` from zero (ReLU and Leaky ReLU have a kink there).
fn away_from_zero(seed: u64) -> Tensor<f64> {
    let mut rng = Rng::seed(seed);
    let data = (0..20)
        .map(|_| {
            loop {
                let x = rng.range(-2.0, 2.0);
                if x.abs() > 1e-3 {
                    break x;
                }
            }
        })
        .collect();
    Tensor::from_vec(data, &[4, 5]).unwrap()
}

/// `sum` and `mean` have the right gradient.
#[test]
fn gradcheck_full_reductions() {
    let x = [away_from_zero(11)];
    gradcheck(|t, v| t.sum(v[0]), &x).assert_passed();
    gradcheck(|t, v| t.mean(v[0]), &x).assert_passed();
}

/// `sum_axis` and `mean_axis` have the right gradient along each axis, with and without
/// `keep_dim`.
#[test]
fn gradcheck_axis_reductions() {
    let x = [away_from_zero(12)];
    for axis in 0..2 {
        for keep_dim in [false, true] {
            let report = gradcheck(|t, v| t.sum_axis(v[0], axis, keep_dim), &x);
            assert!(report.passed(), "sum_axis {axis} {keep_dim}: {report:?}");
            let report = gradcheck(|t, v| t.mean_axis(v[0], axis, keep_dim), &x);
            assert!(report.passed(), "mean_axis {axis} {keep_dim}: {report:?}");
        }
    }
}

/// A reduction over the middle axis of a rank-3 value has the right gradient.
#[test]
fn gradcheck_reduction_middle_axis() {
    let data = (0..24).map(|i| (i as f64 * 0.7).sin()).collect();
    let x = [Tensor::from_vec(data, &[2, 3, 4]).unwrap()];
    gradcheck(|t, v| t.sum_axis(v[0], 1, false), &x).assert_passed();
    gradcheck(|t, v| t.mean_axis(v[0], 1, true), &x).assert_passed();
}

/// ReLU and Leaky ReLU have the right gradient away from the kink.
#[test]
fn gradcheck_relu_family() {
    let x = [away_from_zero(13)];
    gradcheck(|t, v| t.relu(v[0]), &x).assert_passed();
    gradcheck(|t, v| t.leaky_relu(v[0], 0.01), &x).assert_passed();
    gradcheck(|t, v| t.leaky_relu(v[0], 0.2), &x).assert_passed();
}

/// Tanh, sigmoid and both forms of GELU have the right gradient.
#[test]
fn gradcheck_smooth_activations() {
    let mut rng = Rng::seed(14);
    let data = (0..20).map(|_| rng.range(-3.0, 3.0)).collect();
    let x = [Tensor::from_vec(data, &[4, 5]).unwrap()];
    gradcheck(|t, v| t.tanh(v[0]), &x).assert_passed();
    gradcheck(|t, v| t.sigmoid(v[0]), &x).assert_passed();
    gradcheck(|t, v| t.gelu(v[0], GeluApprox::Exact), &x).assert_passed();
    gradcheck(|t, v| t.gelu(v[0], GeluApprox::Tanh), &x).assert_passed();
}

/// A parameter used by two activations gets both contributions.
#[test]
fn gradcheck_activation_shared_parameter() {
    let mut params = ParamStore::<f64>::new_generic();
    let w = params.add(filled(&[4, 5], 0.2));
    let build = |tape: &mut Tape<'_, f64>, v: &[Var]| {
        let wv = tape.param(w);
        let a = tape.mul(v[0], wv);
        let b = tape.tanh(a);
        let c = tape.sigmoid(wv);
        let d = tape.add(b, c);
        tape.mean_axis(d, 0, false)
    };
    let report = gradcheck_params(&mut params, &[filled(&[4, 5], 0.9)], build);
    assert!(report.passed(), "{report:?}");
}

/// A `[4, 5]` input with entries drawn from `[lo, hi]` by a seeded generator, resampled
/// until every entry is farther than `1e-3` from each of the `kinks`.
fn away_from(kinks: &[f64], lo: f64, hi: f64, seed: u64) -> Tensor<f64> {
    let mut rng = Rng::seed(seed);
    let data = (0..20)
        .map(|_| {
            loop {
                let x = rng.range(lo, hi);
                if kinks.iter().all(|k| (x - k).abs() > 1e-3) {
                    break x;
                }
            }
        })
        .collect();
    Tensor::from_vec(data, &[4, 5]).unwrap()
}

/// ELU, SELU and CELU have the right gradient away from zero.
#[test]
fn gradcheck_exponential_linear_units() {
    let x = [away_from(&[0.0], -3.0, 3.0, 21)];
    gradcheck(|t, v| t.elu(v[0], 1.0), &x).assert_passed();
    gradcheck(|t, v| t.elu(v[0], 0.7), &x).assert_passed();
    gradcheck(|t, v| t.selu(v[0]), &x).assert_passed();
    gradcheck(|t, v| t.celu(v[0], 1.0), &x).assert_passed();
    gradcheck(|t, v| t.celu(v[0], 2.0), &x).assert_passed();
    gradcheck(|t, v| t.celu(v[0], 0.5), &x).assert_passed();
}

/// SiLU, Mish, softplus, softsign and log-sigmoid have the right gradient.
#[test]
fn gradcheck_smooth_extra_activations() {
    let x = [away_from(&[], -4.0, 4.0, 22)];
    gradcheck(|t, v| t.silu(v[0]), &x).assert_passed();
    gradcheck(|t, v| t.mish(v[0]), &x).assert_passed();
    gradcheck(|t, v| t.softplus(v[0], 1.0, 20.0), &x).assert_passed();
    gradcheck(|t, v| t.softplus(v[0], 2.5, 20.0), &x).assert_passed();
    gradcheck(|t, v| t.softsign(v[0]), &x).assert_passed();
    gradcheck(|t, v| t.log_sigmoid(v[0]), &x).assert_passed();
}

/// Softplus has the right gradient on both sides of its threshold.
#[test]
fn gradcheck_softplus_threshold() {
    let x = [away_from(&[20.0], 15.0, 25.0, 23)];
    gradcheck(|t, v| t.softplus(v[0], 1.0, 20.0), &x).assert_passed();
}

/// The clamp family has the right gradient away from its kinks.
#[test]
fn gradcheck_clamp_family() {
    let x = [away_from(&[-1.0, 1.0], -2.5, 2.5, 24)];
    gradcheck(|t, v| t.hardtanh(v[0], -1.0, 1.0), &x).assert_passed();
    let x = [away_from(&[0.0, 6.0], -2.0, 8.0, 25)];
    gradcheck(|t, v| t.relu6(v[0]), &x).assert_passed();
    let x = [away_from(&[-3.0, 3.0], -5.0, 5.0, 26)];
    gradcheck(|t, v| t.hardsigmoid(v[0]), &x).assert_passed();
    gradcheck(|t, v| t.hardswish(v[0]), &x).assert_passed();
}

/// PReLU has the right gradient for x and for the slope, with one slope and one per channel.
#[test]
fn gradcheck_prelu() {
    let x = away_from(&[0.0], -2.0, 2.0, 27);
    for dims in [&[1usize][..], &[5]] {
        let n: usize = dims.iter().product();
        let slope = Tensor::from_vec((0..n).map(|i| 0.1 + 0.2 * i as f64).collect(), dims).unwrap();
        let report = gradcheck(|t, v| t.prelu(v[0], v[1]), &[x.clone(), slope]);
        assert!(report.passed(), "prelu slope {dims:?}: {report:?}");
    }
}

/// Every gated linear unit has the right gradient on a `[3, 6]` input; for the ReLU gate the
/// second half of each row is resampled until every entry is farther than `1e-3` from 0.
#[test]
fn gradcheck_glu() {
    let mut rng = Rng::seed(35);
    let data: Vec<f64> = (0..18)
        .map(|i| {
            loop {
                let x = rng.range(-2.5, 2.5);
                if i % 6 < 3 || x.abs() > 1e-3 {
                    break x;
                }
            }
        })
        .collect();
    let x = [Tensor::from_vec(data, &[3, 6]).unwrap()];
    for gate in [
        Gate::Sigmoid,
        Gate::Gelu(GeluApprox::Exact),
        Gate::Gelu(GeluApprox::Tanh),
        Gate::Relu,
        Gate::Silu,
    ] {
        let report = gradcheck(|t, v| t.glu(v[0], gate), &x);
        assert!(report.passed(), "glu {gate:?}: {report:?}");
    }
}

/// A deterministic tensor of the given shape with entries in `[-2, 2]`.
fn spread(dims: &[usize], seed: u64) -> Tensor<f64> {
    let mut rng = Rng::seed(seed);
    let n: usize = dims.iter().product();
    Tensor::from_vec((0..n).map(|_| rng.range(-2.0, 2.0)).collect(), dims).unwrap()
}

/// Every reduction, for the loss checks.
const REDUCTIONS: [Reduction; 4] = [
    Reduction::None,
    Reduction::Sum,
    Reduction::Mean,
    Reduction::BatchMean,
];

/// Softmax and log-softmax have the right gradient on a `[3, 4]` input.
#[test]
fn gradcheck_softmax_and_log_softmax() {
    let x = [spread(&[3, 4], 10)];
    gradcheck(|t, v| t.softmax(v[0]), &x).assert_passed();
    gradcheck(|t, v| t.log_softmax(v[0]), &x).assert_passed();
}

/// Softmax cross-entropy has the right gradient under every reduction.
#[test]
fn gradcheck_softmax_cross_entropy() {
    let x = [spread(&[5, 3], 11)];
    let labels = [0usize, 2, 1, 1, 0];
    for r in REDUCTIONS {
        let report = gradcheck(|t, v| t.softmax_cross_entropy(v[0], &labels, r), &x);
        assert!(report.passed(), "cross-entropy {r:?}: {report:?}");
    }
}

/// MSE has the right gradient for the prediction and the target under every reduction.
#[test]
fn gradcheck_mse() {
    let x = [spread(&[2, 3], 12), spread(&[2, 3], 13)];
    for r in REDUCTIONS {
        let report = gradcheck(|t, v| t.mse(v[0], v[1], r), &x);
        assert!(report.passed(), "mse {r:?}: {report:?}");
    }
}

/// Gradients from several losses on the same logits add up (the second contribution takes
/// the add path), and a loss through a softmax layer is checked end to end.
#[test]
fn gradcheck_losses_sharing_an_input() {
    let x = [spread(&[4, 3], 14)];
    let labels = [2usize, 0, 1, 2];
    let report = gradcheck(
        |t, v| {
            let ce = t.softmax_cross_entropy(v[0], &labels, Reduction::Mean);
            let ls = t.log_softmax(v[0]);
            let total = t.sum(ls);
            let sm = t.softmax(v[0]);
            let target = t.softmax(ls);
            let err = t.mse(sm, target, Reduction::Sum);
            let a = t.add(ce, total);
            t.add(a, err)
        },
        &x,
    );
    assert!(report.passed(), "shared logits: {report:?}");
}

/// A tensor of the given shape with entries in `[0.05, 0.95]`, for probabilities.
fn probabilities(dims: &[usize], seed: u64) -> Tensor<f64> {
    let mut rng = Rng::seed(seed);
    let n: usize = dims.iter().product();
    Tensor::from_vec((0..n).map(|_| rng.range(0.05, 0.95)).collect(), dims).unwrap()
}

/// Draws a prediction and a target in `[-2, 2]`, resampling each target until the
/// difference is more than 1e-3 from 0 and from `knee`, so that no check sits on a kink.
fn away_from_kinks(dims: &[usize], knee: f64, seed: u64) -> [Tensor<f64>; 2] {
    let mut rng = Rng::seed(seed);
    let n: usize = dims.iter().product();
    let pred: Vec<f64> = (0..n).map(|_| rng.range(-2.0, 2.0)).collect();
    let target: Vec<f64> = pred
        .iter()
        .map(|&p| {
            loop {
                let t = rng.range(-2.0, 2.0);
                let d = (p - t).abs();
                if d > 1e-3 && (d - knee).abs() > 1e-3 {
                    break t;
                }
            }
        })
        .collect();
    [
        Tensor::from_vec(pred, dims).unwrap(),
        Tensor::from_vec(target, dims).unwrap(),
    ]
}

/// Binary cross-entropy has the right gradient for probabilities and targets under every
/// reduction.
#[test]
fn gradcheck_binary_cross_entropy() {
    let x = [probabilities(&[3, 4], 20), probabilities(&[3, 4], 21)];
    for r in REDUCTIONS {
        let report = gradcheck(|t, v| t.binary_cross_entropy(v[0], v[1], r), &x);
        assert!(report.passed(), "bce {r:?}: {report:?}");
    }
}

/// Binary cross-entropy with logits has the right gradient for logits and targets under every
/// reduction.
#[test]
fn gradcheck_binary_cross_entropy_with_logits() {
    let x = [spread(&[3, 4], 22), probabilities(&[3, 4], 23)];
    for r in REDUCTIONS {
        let report = gradcheck(|t, v| t.binary_cross_entropy_with_logits(v[0], v[1], r), &x);
        assert!(report.passed(), "bce with logits {r:?}: {report:?}");
    }
}

/// NLL has the right gradient directly and behind a log-softmax, under every reduction.
#[test]
fn gradcheck_nll() {
    let x = [spread(&[5, 3], 24)];
    let labels = [0usize, 2, 1, 1, 0];
    for r in REDUCTIONS {
        let direct = gradcheck(|t, v| t.nll(v[0], &labels, r), &x);
        assert!(direct.passed(), "nll {r:?}: {direct:?}");
        let through = gradcheck(
            |t, v| {
                let lp = t.log_softmax(v[0]);
                t.nll(lp, &labels, r)
            },
            &x,
        );
        assert!(through.passed(), "log_softmax + nll {r:?}: {through:?}");
    }
}

/// L1 has the right gradient for prediction and target under every reduction.
#[test]
fn gradcheck_l1() {
    let x = away_from_kinks(&[3, 4], 0.0, 25);
    for r in REDUCTIONS {
        let report = gradcheck(|t, v| t.l1(v[0], v[1], r), &x);
        assert!(report.passed(), "l1 {r:?}: {report:?}");
    }
}

/// Huber has the right gradient in both regimes, for several deltas and every reduction.
#[test]
fn gradcheck_huber() {
    for delta in [0.5, 1.0, 1.5] {
        let x = away_from_kinks(&[3, 4], delta, 26);
        for r in REDUCTIONS {
            let report = gradcheck(|t, v| t.huber(v[0], v[1], delta, r), &x);
            assert!(report.passed(), "huber {delta} {r:?}: {report:?}");
        }
    }
}

/// Smooth L1 has the right gradient in both regimes, for several betas and every reduction.
#[test]
fn gradcheck_smooth_l1() {
    for beta in [0.5, 1.0, 1.5] {
        let x = away_from_kinks(&[3, 4], beta, 27);
        for r in REDUCTIONS {
            let report = gradcheck(|t, v| t.smooth_l1(v[0], v[1], beta, r), &x);
            assert!(report.passed(), "smooth_l1 {beta} {r:?}: {report:?}");
        }
    }
}

/// KL divergence has the right gradient for log-probabilities and targets under every
/// reduction.
#[test]
fn gradcheck_kl_div() {
    let x = [spread(&[3, 4], 28), probabilities(&[3, 4], 29)];
    for r in REDUCTIONS {
        let report = gradcheck(|t, v| t.kl_div(v[0], v[1], r), &x);
        assert!(report.passed(), "kl_div {r:?}: {report:?}");
    }
}

/// A user-defined cube, `x³`, whose backward pass wrongly uses the derivative `2x`.
struct WrongCube;

impl<T: Real> CustomOp<T> for WrongCube {
    fn name(&self) -> &str {
        "wrong_cube"
    }

    fn output_shape(&self, inputs: &[Shape]) -> Shape {
        inputs[0]
    }

    fn forward(&self, inputs: &[TensorView<T>], mut output: TensorViewMut<T>) {
        for (o, &x) in output.data_mut().iter_mut().zip(inputs[0].data()) {
            *o = x * x * x;
        }
    }

    fn backward(
        &self,
        inputs: &[TensorView<T>],
        _output: TensorView<T>,
        grad_output: TensorView<T>,
        grad_inputs: &mut [TensorViewMut<T>],
    ) {
        let x = inputs[0].data();
        for (i, g) in grad_inputs[0].data_mut().iter_mut().enumerate() {
            *g = (x[i] + x[i]) * grad_output.data()[i];
        }
    }
}

/// The gradient check catches a user-defined operation with a wrong backward pass.
#[test]
fn custom_op_wrong_backward_is_caught() {
    let op: Arc<dyn CustomOp<f64> + Send + Sync> = Arc::new(WrongCube);
    let report = gradcheck(|t, v| t.custom(&op, &[v[0]]), &[t(&[0.5, -1.5, 2.0])]);
    assert!(!report.passed());
    assert_eq!(report.failures.len(), 3);
}

/// A user-defined two-input multiply, `a · b`.
struct CustomMul;

impl<T: Real> CustomOp<T> for CustomMul {
    fn name(&self) -> &str {
        "custom_mul"
    }

    fn output_shape(&self, inputs: &[Shape]) -> Shape {
        assert_eq!(inputs[0], inputs[1], "custom_mul: shapes differ");
        inputs[0]
    }

    fn forward(&self, inputs: &[TensorView<T>], mut output: TensorViewMut<T>) {
        let (a, b) = (inputs[0].data(), inputs[1].data());
        for (i, o) in output.data_mut().iter_mut().enumerate() {
            *o = a[i] * b[i];
        }
    }

    fn backward(
        &self,
        inputs: &[TensorView<T>],
        _output: TensorView<T>,
        grad_output: TensorView<T>,
        grad_inputs: &mut [TensorViewMut<T>],
    ) {
        let (a, b, g) = (inputs[0].data(), inputs[1].data(), grad_output.data());
        let [ga, gb] = grad_inputs else {
            panic!("custom_mul takes two inputs")
        };
        for i in 0..g.len() {
            ga.data_mut()[i] += b[i] * g[i];
            gb.data_mut()[i] += a[i] * g[i];
        }
    }
}

/// `x · x` written as a two-input custom multiply gets the gradient `2x`: each input slot
/// gets its own scratch buffer and both are added into `x`'s gradient.
#[test]
fn custom_op_same_var_twice() {
    let op: Arc<dyn CustomOp<f64> + Send + Sync> = Arc::new(CustomMul);
    let x = t(&[0.5, -1.5, 2.0]);
    gradcheck(
        |t, v| t.custom(&op, &[v[0], v[0]]),
        std::slice::from_ref(&x),
    )
    .assert_passed();

    let mut store = ParamStore::<f64>::new_generic();
    let p = store.add(x);
    let mut graph = Graph::<f64>::new_generic();
    let mut tape = graph.tape(&store.values);
    let v = tape.param(p);
    let sq = tape.custom(&op, &[v, v]);
    assert_eq!(tape.value(sq).data(), &[0.25, 2.25, 4.0]);
    let loss = tape.sum(sq);
    tape.backward(loss, &mut store.grads);
    drop(tape);
    assert_eq!(store.grad(p).data(), &[1.0, -3.0, 4.0]);
}
