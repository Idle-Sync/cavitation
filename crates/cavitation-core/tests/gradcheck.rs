//! Gradient checks of the operations, through the numerical check harness.

use cavitation_core::{
    GradcheckReport, ParamStore, Tape, Tensor, Var, gradcheck, gradcheck_params,
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
