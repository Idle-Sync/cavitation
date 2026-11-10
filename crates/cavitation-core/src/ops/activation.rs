//! Activations: `relu`, `leaky_relu`, `tanh`, `sigmoid` and `gelu`.

use crate::graph::{BackwardCtx, Var};
use crate::ops::Op;
use crate::{Real, Tape};

/// Which form of GELU to compute (PyTorch's `approximate` argument).
///
/// # Examples
///
/// ```
/// use cavitation_core::{GeluApprox, Graph, ParamStore, Tensor};
/// let store = ParamStore::new();
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&store.values);
/// let x = Tensor::from_vec(vec![1.0f32], &[1]).unwrap();
/// let x = tape.input(x.view());
/// let exact = tape.gelu(x, GeluApprox::Exact);
/// let approx = tape.gelu(x, GeluApprox::Tanh);
/// assert!((tape.value(exact).data()[0] - tape.value(approx).data()[0]).abs() < 1e-3);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeluApprox {
    /// `0.5 · x · (1 + erf(x / √2))`, PyTorch's `approximate='none'`.
    Exact,
    /// `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`, PyTorch's `approximate='tanh'`.
    Tanh,
}

/// `√(2/π)`, the scale inside the tanh form of GELU.
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
/// The cubic coefficient inside the tanh form of GELU.
const GELU_CUBIC: f64 = 0.044715;
/// `1/√2`.
const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// `1/√(2π)`, the standard normal density at 0.
const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// `1 / (1 + e^-x)` for `x >= 0` and `e^x / (1 + e^x)` below, so the exponential never
/// overflows.
fn sigmoid_value<T: Real>(x: T) -> T {
    if x >= T::ZERO {
        T::ONE / (T::ONE + (-x).exp())
    } else {
        let e = x.exp();
        e / (T::ONE + e)
    }
}

/// GELU of `x` in the given form.
fn gelu_value<T: Real>(x: T, approx: GeluApprox) -> T {
    let half = T::from_f64(0.5);
    match approx {
        GeluApprox::Exact => half * x * (T::ONE + (x * T::from_f64(FRAC_1_SQRT_2)).erf()),
        GeluApprox::Tanh => {
            let u = T::from_f64(SQRT_2_OVER_PI) * (x + T::from_f64(GELU_CUBIC) * x * x * x);
            half * x * (T::ONE + u.tanh())
        }
    }
}

/// The derivative of GELU at `x` in the given form.
fn gelu_derivative<T: Real>(x: T, approx: GeluApprox) -> T {
    let half = T::from_f64(0.5);
    match approx {
        GeluApprox::Exact => {
            // Phi(x) + x * phi(x).
            let cdf = half * (T::ONE + (x * T::from_f64(FRAC_1_SQRT_2)).erf());
            let pdf = T::from_f64(FRAC_1_SQRT_2PI) * (-(half * x * x)).exp();
            cdf + x * pdf
        }
        GeluApprox::Tanh => {
            let x2 = x * x;
            let u = T::from_f64(SQRT_2_OVER_PI) * (x + T::from_f64(GELU_CUBIC) * x2 * x);
            let t = u.tanh();
            let du = T::from_f64(SQRT_2_OVER_PI) * (T::ONE + T::from_f64(3.0 * GELU_CUBIC) * x2);
            half * (T::ONE + t) + half * x * (T::ONE - t * t) * du
        }
    }
}

impl<T: Real> Tape<'_, T> {
    /// Records `max(x, 0)` for every element. The derivative at exactly 0 is 0, as in
    /// PyTorch.
    ///
    /// # Panics
    ///
    /// Panics if `x` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::from_vec(vec![-1.0f32, 2.0], &[2]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.relu(x);
    /// assert_eq!(tape.value(y).data(), &[0.0, 2.0]);
    /// ```
    pub fn relu(&mut self, x: Var) -> Var {
        self.map(Op::Relu(x), x, |v| if v > T::ZERO { v } else { T::ZERO })
    }

    /// Records Leaky ReLU: `x` where `x > 0`, `slope · x` elsewhere. PyTorch's default slope
    /// is 0.01; it is not assumed here, so pass it. The derivative at exactly 0 is `slope`.
    ///
    /// # Panics
    ///
    /// Panics if `x` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::from_vec(vec![-2.0f32, 3.0], &[2]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.leaky_relu(x, 0.5);
    /// assert_eq!(tape.value(y).data(), &[-1.0, 3.0]);
    /// ```
    pub fn leaky_relu(&mut self, x: Var, slope: T) -> Var {
        self.map(Op::LeakyRelu(x, slope.to_f64()), x, |v| {
            if v > T::ZERO { v } else { slope * v }
        })
    }

    /// Records the hyperbolic tangent of every element.
    ///
    /// # Panics
    ///
    /// Panics if `x` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::from_vec(vec![0.0f32], &[1]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.tanh(x);
    /// assert_eq!(tape.value(y).data(), &[0.0]);
    /// ```
    pub fn tanh(&mut self, x: Var) -> Var {
        self.map(Op::Tanh(x), x, Real::tanh)
    }

    /// Records the logistic sigmoid `1 / (1 + e^-x)` of every element. It is computed as
    /// `1 / (1 + e^-x)` for `x >= 0` and `e^x / (1 + e^x)` below, so it never overflows:
    /// large negative inputs give 0 and large positive inputs give 1.
    ///
    /// # Panics
    ///
    /// Panics if `x` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::from_vec(vec![0.0f32, 1000.0], &[2]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.sigmoid(x);
    /// assert_eq!(tape.value(y).data(), &[0.5, 1.0]);
    /// ```
    pub fn sigmoid(&mut self, x: Var) -> Var {
        self.map(Op::Sigmoid(x), x, sigmoid_value)
    }

    /// Records GELU of every element, in the exact form (with `erf`) or the tanh
    /// approximation; see [`GeluApprox`].
    ///
    /// # Panics
    ///
    /// Panics if `x` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{GeluApprox, Graph, ParamStore, Tensor};
    /// let store = ParamStore::<f64>::new_generic();
    /// let mut graph = Graph::<f64>::new_generic();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::from_vec(vec![1.0f64], &[1]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.gelu(x, GeluApprox::Exact);
    /// assert!((tape.value(y).data()[0] - 0.8413447460685429).abs() < 1e-15);
    /// ```
    pub fn gelu(&mut self, x: Var, approx: GeluApprox) -> Var {
        self.map(Op::Gelu(x, approx), x, |v| gelu_value(v, approx))
    }

    /// Records `op` with the value `f` applied to every element of `x`.
    fn map(&mut self, op: Op, x: Var, f: impl Fn(T) -> T) -> Var {
        let shape = self.shape(x);
        let mut out = self.take_buffer(shape.len());
        debug_assert_eq!(out.len(), self.data(x).len());
        for (o, &v) in out.iter_mut().zip(self.data(x)) {
            *o = f(v);
        }
        self.push(op, shape, out)
    }
}

/// Passes `out_grad[i] * derivative(i)` to `x`, where `derivative` gets the index of the
/// element. Every element of the gradient buffer is written.
fn backward_with<T: Real>(
    x: Var,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
    derivative: impl Fn(usize) -> T,
) {
    let Some(slot) = ctx.slot(x) else { return };
    slot.update_with(|buf, beta| {
        debug_assert_eq!(buf.len(), out_grad.len());
        if beta == T::ZERO {
            for (i, (d, &g)) in buf.iter_mut().zip(out_grad).enumerate() {
                *d = g * derivative(i);
            }
        } else {
            for (i, (d, &g)) in buf.iter_mut().zip(out_grad).enumerate() {
                *d += g * derivative(i);
            }
        }
    });
}

/// Backward of `relu`: the gradient where the input is positive, 0 elsewhere.
pub(super) fn relu_backward<T: Real>(x: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    let input = ctx.value(x);
    backward_with(x, out_grad, ctx, |i| {
        if input[i] > T::ZERO { T::ONE } else { T::ZERO }
    });
}

/// Backward of `leaky_relu`: the gradient where the input is positive, `slope` times it
/// elsewhere (including at 0).
pub(super) fn leaky_relu_backward<T: Real>(
    x: Var,
    slope: f64,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let input = ctx.value(x);
    let slope = T::from_f64(slope);
    backward_with(x, out_grad, ctx, |i| {
        if input[i] > T::ZERO { T::ONE } else { slope }
    });
}

/// Backward of `tanh`, from the output `y`: `1 - y²`.
pub(super) fn tanh_backward<T: Real>(
    x: Var,
    out: Var,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let y = ctx.value(out);
    backward_with(x, out_grad, ctx, |i| T::ONE - y[i] * y[i]);
}

/// Backward of `sigmoid`, from the output `y`: `y · (1 - y)`.
pub(super) fn sigmoid_backward<T: Real>(
    x: Var,
    out: Var,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let y = ctx.value(out);
    backward_with(x, out_grad, ctx, |i| y[i] * (T::ONE - y[i]));
}

/// Backward of `gelu` in the given form.
pub(super) fn gelu_backward<T: Real>(
    x: Var,
    approx: GeluApprox,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let input = ctx.value(x);
    backward_with(x, out_grad, ctx, |i| gelu_derivative(input[i], approx));
}

/// Tests for the activations.
#[cfg(test)]
mod tests {
    use crate::{GeluApprox, Graph, ParamStore, Tensor};

    /// ReLU's derivative at exactly zero is zero, and 1 above it.
    #[test]
    fn relu_derivative_at_zero_is_zero() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::from_vec(vec![-1.0f32, 0.0, 2.0], &[3]).unwrap());
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p);
        let r = tape.relu(v);
        let loss = tape.sum(r);
        assert_eq!(tape.value(r).data(), &[0.0, 0.0, 2.0]);
        tape.backward(loss, &mut store.grads);
        drop(tape);
        assert_eq!(store.grad(p).data(), &[0.0, 0.0, 1.0]);
    }

    /// Leaky ReLU's derivative at zero is the slope.
    #[test]
    fn leaky_relu_derivative_at_zero_is_slope() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::from_vec(vec![-2.0f32, 0.0, 3.0], &[3]).unwrap());
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p);
        let r = tape.leaky_relu(v, 0.5);
        let loss = tape.sum(r);
        assert_eq!(tape.value(r).data(), &[-1.0, 0.0, 3.0]);
        tape.backward(loss, &mut store.grads);
        drop(tape);
        assert_eq!(store.grad(p).data(), &[0.5, 0.5, 1.0]);
    }

    /// Sigmoid saturates to exactly 0 and 1 without NaN, and is 0.5 at 0.
    #[test]
    fn sigmoid_extremes() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = Tensor::from_vec(vec![-1000.0f32, 0.0, 1000.0, -1e30, 1e30], &[5]).unwrap();
        let x = tape.input(x.view());
        let y = tape.sigmoid(x);
        assert_eq!(tape.value(y).data(), &[0.0, 0.5, 1.0, 0.0, 1.0]);
    }

    /// Exact GELU at 1 is Phi(1) = 0.8413447460685429.
    #[test]
    fn gelu_exact_at_one() {
        let store = ParamStore::<f64>::new_generic();
        let mut graph = Graph::<f64>::new_generic();
        let mut tape = graph.tape(&store.values);
        let x = Tensor::from_vec(vec![1.0f64, 0.0], &[2]).unwrap();
        let x = tape.input(x.view());
        let y = tape.gelu(x, GeluApprox::Exact);
        let v = tape.value(y).data();
        assert!((v[0] - 0.8413447460685429).abs() < 1e-15, "{}", v[0]);
        assert_eq!(v[1], 0.0);
    }

    /// The tanh form is close to the exact form (PyTorch's two modes differ by under 1e-3).
    #[test]
    fn gelu_tanh_is_close_to_exact() {
        let store = ParamStore::<f64>::new_generic();
        let mut graph = Graph::<f64>::new_generic();
        let mut tape = graph.tape(&store.values);
        let x = Tensor::from_vec(vec![-3.0f64, -1.0, 0.5, 1.0, 2.5], &[5]).unwrap();
        let x = tape.input(x.view());
        let e = tape.gelu(x, GeluApprox::Exact);
        let t = tape.gelu(x, GeluApprox::Tanh);
        let (e, t) = (tape.value(e).data(), tape.value(t).data());
        for (a, b) in e.iter().zip(t) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
        assert!((t[3] - 0.8411919906082768).abs() < 1e-12, "{}", t[3]);
    }
}
