//! Gradient checks of the operations, through the numerical check harness.

use cavitation_core::{
    Gate, GeluApprox, GradcheckReport, ParamStore, Rng, Tape, Tensor, Var, gradcheck,
    gradcheck_params,
};

fn t(data: &[f64]) -> Tensor<f64> {
    Tensor::from_vec(data.to_vec(), &[data.len()]).unwrap()
}

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
