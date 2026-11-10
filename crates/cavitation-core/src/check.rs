//! The numerical gradient check: compares the gradients the tape computes with central
//! finite differences, in `f64`.

use crate::{Graph, ParamId, ParamStore, Rng, Tape, Tensor, Var};

/// The finite-difference step.
const STEP: f64 = 1e-6;
/// A check passes when `|numeric - analytic| <= ABS_TOL + REL_TOL * max(|numeric|, |analytic|)`.
const ABS_TOL: f64 = 1e-7;
/// The relative part of the tolerance.
const REL_TOL: f64 = 1e-5;
/// The smallest magnitude used as the denominator of a relative error.
const REL_FLOOR: f64 = 1e-7;
/// How many failures `assert_passed` lists.
const LISTED_FAILURES: usize = 10;

/// One element whose analytic gradient disagrees with the numerical one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradcheckFailure {
    /// Which checked tensor: the parameters already in the store come first, in the order
    /// they were added, then the inputs, in the order given.
    pub input: usize,
    /// The element's position in that tensor's flat data.
    pub index: usize,
    /// The central-difference estimate.
    pub numeric: f64,
    /// The gradient the tape computed.
    pub analytic: f64,
}

/// The outcome of a gradient check.
///
/// # Examples
///
/// ```
/// use cavitation_core::{gradcheck, Tensor};
/// let x = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
/// let y = Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap();
/// let report = gradcheck(|tape, v| tape.add(v[0], v[1]), &[x, y]);
/// assert!(report.passed());
/// report.assert_passed();
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct GradcheckReport {
    /// The largest `|numeric - analytic|` over all checked elements (infinite if any
    /// gradient is infinite or not a number).
    pub worst_abs: f64,
    /// The largest `|numeric - analytic| / max(|numeric|, |analytic|, 1e-7)`.
    pub worst_rel: f64,
    /// Every element outside the tolerance.
    pub failures: Vec<GradcheckFailure>,
}

impl GradcheckReport {
    /// Whether every element agreed within the tolerance.
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    /// Panics if any element disagreed. The message lists the first 10 failures (checked
    /// tensor index, element index, numeric and analytic value) and the worst absolute and
    /// relative errors.
    ///
    /// ```
    /// use cavitation_core::{gradcheck, Tensor};
    /// let x = Tensor::from_vec(vec![0.5], &[1]).unwrap();
    /// gradcheck(|tape, v| tape.add(v[0], v[0]), &[x]).assert_passed();
    /// ```
    pub fn assert_passed(&self) {
        if self.passed() {
            return;
        }
        let mut message = format!(
            "gradient check failed for {} element(s); worst absolute error {:e}, worst relative error {:e}",
            self.failures.len(),
            self.worst_abs,
            self.worst_rel
        );
        for f in self.failures.iter().take(LISTED_FAILURES) {
            message.push_str(&format!(
                "\n  tensor {} element {}: numeric {:e}, analytic {:e}",
                f.input, f.index, f.numeric, f.analytic
            ));
        }
        if self.failures.len() > LISTED_FAILURES {
            message.push_str(&format!(
                "\n  ... and {} more",
                self.failures.len() - LISTED_FAILURES
            ));
        }
        panic!("{message}");
    }
}

/// Checks the gradients of a computation of `inputs` against central differences.
/// Shorthand for [`gradcheck_params`] with no parameters.
///
/// `build` records the computation on the tape from one variable per input and returns
/// its output, of any shape. The loss is the sum of the output's elements, each multiplied
/// by a fixed pseudo-random weight (the same weights every run) whose magnitude lies in
/// `[0.5, 1]` and whose sign is random, so every output element affects the check at full
/// strength.
///
/// # Panics
///
/// Panics if `build` returns outputs of different sizes on different calls.
///
/// ```
/// use cavitation_core::{gradcheck, Tensor};
/// let a = Tensor::from_vec(vec![1.0, -2.0, 0.5], &[3]).unwrap();
/// let b = Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap();
/// gradcheck(|tape, v| tape.add(v[0], v[1]), &[a, b]).assert_passed();
/// ```
pub fn gradcheck(
    build: impl Fn(&mut Tape<'_, f64>, &[Var]) -> Var,
    inputs: &[Tensor<f64>],
) -> GradcheckReport {
    gradcheck_params(&mut ParamStore::new_generic(), inputs, build)
}

/// Checks the gradients of every parameter in `params` and every input against central
/// differences (step `1e-6`; an element passes when
/// `|numeric - analytic| <= 1e-7 + 1e-5 * max(|numeric|, |analytic|)`).
///
/// The inputs are added to `params` as extra parameters so that they receive gradients;
/// they are removed again before the call returns, so the store is unchanged. `build` receives one variable per input, in order,
/// and reads the parameters it needs through their [`ParamId`]s. Parameter values are
/// perturbed in place and restored exactly. The store's gradients are zeroed before and
/// after the analytic pass.
///
/// Failures name checked tensors by position: the parameters that were in the store
/// before the call come first, then the inputs.
///
/// # Panics
///
/// Panics if `build` returns outputs of different sizes on different calls.
///
/// ```
/// use cavitation_core::{gradcheck_params, ParamStore, Tensor};
/// let mut params = ParamStore::<f64>::new_generic();
/// let bias = params.add(Tensor::from_vec(vec![0.5, -0.5], &[2]).unwrap());
/// let x = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
/// let report = gradcheck_params(&mut params, &[x], |tape, v| {
///     let b = tape.param(bias);
///     tape.add(v[0], b)
/// });
/// report.assert_passed();
/// ```
pub fn gradcheck_params(
    params: &mut ParamStore<f64>,
    inputs: &[Tensor<f64>],
    build: impl Fn(&mut Tape<'_, f64>, &[Var]) -> Var,
) -> GradcheckReport {
    let original_len = params.len();
    let mut checked: Vec<ParamId> = params.iter_mut().map(|(id, _, _)| id).collect();
    let input_ids: Vec<ParamId> = inputs.iter().map(|x| params.add(x.clone())).collect();
    checked.extend(&input_ids);

    let mut graph = Graph::<f64>::new_generic();

    // Analytic gradients of L = sum(w * out), with dL/dout = w.
    params.zero_grad();
    let weights = {
        let mut tape = graph.tape(&params.values);
        let vars: Vec<Var> = input_ids.iter().map(|&id| tape.param(id)).collect();
        let out = build(&mut tape, &vars);
        let mut rng = Rng::seed(0);
        let w: Vec<f64> = (0..tape.shape(out).len())
            .map(|_| {
                let magnitude = rng.range(0.5, 1.0);
                if rng.uniform_f64() < 0.5 {
                    -magnitude
                } else {
                    magnitude
                }
            })
            .collect();
        tape.backward_seeded(out, &w, &mut params.grads);
        w
    };
    let analytic: Vec<Vec<f64>> = checked
        .iter()
        .map(|&id| params.grad(id).data().to_vec())
        .collect();
    params.zero_grad();

    let loss = |params: &ParamStore<f64>, graph: &mut Graph<f64>| -> f64 {
        let mut tape = graph.tape(&params.values);
        let vars: Vec<Var> = input_ids.iter().map(|&id| tape.param(id)).collect();
        let out = build(&mut tape, &vars);
        let value = tape.value(out);
        assert!(
            value.data().len() == weights.len(),
            "gradcheck: build returned {} output elements, but earlier {}",
            value.data().len(),
            weights.len()
        );
        value.data().iter().zip(&weights).map(|(o, w)| o * w).sum()
    };

    let mut report = GradcheckReport {
        worst_abs: 0.0,
        worst_rel: 0.0,
        failures: Vec::new(),
    };
    for (k, &id) in checked.iter().enumerate() {
        for (index, &analytic) in analytic[k].iter().enumerate() {
            let original = params.value(id).data()[index];
            let (up, down) = (original + STEP, original - STEP);
            params.value_mut(id).data_mut()[index] = up;
            let loss_up = loss(params, &mut graph);
            params.value_mut(id).data_mut()[index] = down;
            let loss_down = loss(params, &mut graph);
            params.value_mut(id).data_mut()[index] = original;
            // The step actually taken, after rounding.
            let numeric = (loss_up - loss_down) / (up - down);

            let diff = (numeric - analytic).abs();
            let scale = numeric.abs().max(analytic.abs());
            let finite = numeric.is_finite() && analytic.is_finite();
            let (abs_err, rel_err) = if !finite {
                (f64::INFINITY, f64::INFINITY)
            } else {
                (diff, diff / scale.max(REL_FLOOR))
            };
            report.worst_abs = report.worst_abs.max(abs_err);
            report.worst_rel = report.worst_rel.max(rel_err);
            // A gradient that is not finite always fails.
            if !finite || diff > ABS_TOL + REL_TOL * scale {
                report.failures.push(GradcheckFailure {
                    input: k,
                    index,
                    numeric,
                    analytic,
                });
            }
        }
    }
    params.truncate(original_len);
    report
}

/// Tests for the gradient check itself.
#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a one-dimensional `f64` tensor from a slice.
    fn t(data: &[f64]) -> Tensor<f64> {
        Tensor::from_vec(data.to_vec(), &[data.len()]).unwrap()
    }

    /// An operation whose backward pass is twice the true gradient is caught, and
    /// `assert_passed` panics.
    #[test]
    fn gradcheck_catches_a_wrong_backward() {
        let report = gradcheck(|tape, v| tape.wrong_identity(v[0]), &[t(&[1.0, 2.0, 3.0])]);
        assert!(!report.passed());
        assert!(!report.failures.is_empty());
        let result = std::panic::catch_unwind(|| report.assert_passed());
        assert!(result.is_err());
    }

    /// The failures of a parameter and an input are reported under their own indices.
    #[test]
    fn failures_name_parameters_and_inputs() {
        let mut params = ParamStore::<f64>::new_generic();
        let p = params.add(t(&[1.0]));
        let report = gradcheck_params(&mut params, &[t(&[2.0])], |tape, v| {
            let w = tape.param(p);
            let w = tape.wrong_identity(w);
            let x = tape.wrong_identity(v[0]);
            tape.add(w, x)
        });
        let mut seen: Vec<usize> = report.failures.iter().map(|f| f.input).collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1]);
        assert_eq!(params.value(p).data(), &[1.0]);
    }

    /// A non-finite gradient fails instead of slipping through a comparison with NaN.
    #[test]
    fn nan_fails() {
        let report = gradcheck(
            |tape, v| tape.add(v[0], v[0]),
            &[Tensor::from_vec(vec![f64::NAN], &[1]).unwrap()],
        );
        assert!(!report.passed());
    }

    /// An infinite analytic gradient fails.
    #[test]
    fn infinite_analytic_gradient_fails() {
        let report = gradcheck(|tape, v| tape.infinite_grad(v[0]), &[t(&[1.0, 2.0])]);
        assert_eq!(report.failures.len(), 2);
        assert!(report.worst_abs.is_infinite());
    }

    /// A forward pass that overflows for a perturbed input gives a non-finite numeric
    /// gradient, which fails.
    #[test]
    fn infinite_numeric_gradient_fails() {
        let report = gradcheck(|tape, v| tape.blow_up(v[0]), &[t(&[1.0])]);
        assert!(!report.passed());
        assert!(!report.failures[0].numeric.is_finite());
    }

    /// The store keeps its length and values, and repeated calls report the same indices.
    #[test]
    fn gradcheck_params_leaves_store_unchanged() {
        let mut params = ParamStore::<f64>::new_generic();
        let p = params.add(t(&[1.0, 2.0]));
        let run = |params: &mut ParamStore<f64>| {
            gradcheck_params(params, &[t(&[3.0, 4.0])], |tape, v| {
                let w = tape.param(p);
                let w = tape.wrong_identity(w);
                let x = tape.wrong_identity(v[0]);
                tape.add(w, x)
            })
        };
        let first = run(&mut params);
        assert_eq!(params.len(), 1);
        assert_eq!(params.value(p).data(), &[1.0, 2.0]);
        let second = run(&mut params);
        assert_eq!(first, second);
        assert_eq!(params.len(), 1);
    }

    /// The panic message lists failures and the worst errors.
    #[test]
    #[should_panic(expected = "tensor 0 element 0")]
    fn assert_passed_lists_failures() {
        gradcheck(|tape, v| tape.wrong_identity(v[0]), &[t(&[1.0])]).assert_passed();
    }
}
