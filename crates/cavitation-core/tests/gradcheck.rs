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
