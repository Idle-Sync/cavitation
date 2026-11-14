//! Activations: `relu`, `leaky_relu`, `tanh`, `sigmoid`, `gelu`, and the ELU, SiLU, softplus,
//! clamp (hardtanh, hard sigmoid, hard swish) and PReLU families.

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
pub(crate) const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
/// The cubic coefficient inside the tanh form of GELU.
pub(crate) const GELU_CUBIC: f64 = 0.044715;
/// `1/√2`.
const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// `1/√(2π)`, the standard normal density at 0.
const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// `1 / (1 + e^-x)` for `x >= 0` and `e^x / (1 + e^x)` below, so the exponential never
/// overflows.
pub(crate) fn sigmoid_value<T: Real>(x: T) -> T {
    if x >= T::ZERO {
        T::ONE / (T::ONE + (-x).exp())
    } else {
        let e = x.exp();
        e / (T::ONE + e)
    }
}

/// `max(x, 0)`; NaN stays NaN.
pub(super) fn relu_value<T: Real>(x: T) -> T {
    if x <= T::ZERO { T::ZERO } else { x }
}

/// The derivative of ReLU: 1 above 0, 0 at and below it, NaN for NaN.
pub(super) fn relu_derivative<T: Real>(x: T) -> T {
    if x > T::ZERO {
        T::ONE
    } else if x <= T::ZERO {
        T::ZERO
    } else {
        nan()
    }
}

/// `x · σ(x)`.
pub(crate) fn silu_value<T: Real>(x: T) -> T {
    x * sigmoid_value(x)
}

/// The derivative of SiLU: `σ(x) · (1 + x · (1 - σ(x)))`.
pub(super) fn silu_derivative<T: Real>(x: T) -> T {
    let s = sigmoid_value(x);
    s * (T::ONE + x * (T::ONE - s))
}

/// SELU's scale, `1.0507009873554804934...`.
pub(crate) const SELU_SCALE: f64 = 1.050_700_987_355_480_5;
/// SELU's alpha, `1.6732632423543772848...`.
pub(crate) const SELU_ALPHA: f64 = 1.673_263_242_354_377_2;

/// NaN of the element type.
fn nan<T: Real>() -> T {
    T::from_f64(f64::NAN)
}

/// ELU: `x` above 0, `alpha · (eˣ - 1)` elsewhere, with `eˣ - 1` from `exp_m1` so tiny
/// negative inputs keep their precision and `-0.0` keeps its sign.
pub(crate) fn elu_value<T: Real>(x: T, alpha: T) -> T {
    if x > T::ZERO { x } else { alpha * x.exp_m1() }
}

/// CELU: `x` above 0, `alpha · (e^(x/alpha) - 1)` elsewhere, with the difference from
/// `exp_m1`.
pub(crate) fn celu_value<T: Real>(x: T, alpha: T) -> T {
    if x > T::ZERO {
        x
    } else {
        alpha * (x / alpha).exp_m1()
    }
}

/// Softplus `ln(1 + e^(beta·x)) / beta`, or `x` itself once `beta·x` passes `threshold`.
pub(crate) fn softplus_value<T: Real>(x: T, beta: T, threshold: T) -> T {
    let z = beta * x;
    if z > threshold {
        x
    } else {
        z.exp().ln_1p() / beta
    }
}

/// The derivative of [`softplus_value`].
fn softplus_derivative<T: Real>(x: T, beta: T, threshold: T) -> T {
    let z = beta * x;
    if z > threshold {
        T::ONE
    } else {
        sigmoid_value(z)
    }
}

/// Mish `x · tanh(softplus(x))`, with softplus at PyTorch's defaults (beta 1, threshold 20).
pub(crate) fn mish_value<T: Real>(x: T) -> T {
    x * softplus_value(x, T::ONE, T::from_f64(20.0)).tanh()
}

/// The derivative of Mish: `tanh(sp) + x · σ(x) · (1 - tanh²(sp))`.
fn mish_derivative<T: Real>(x: T) -> T {
    let t = softplus_value(x, T::ONE, T::from_f64(20.0)).tanh();
    t + x * sigmoid_value(x) * (T::ONE - t * t)
}

/// Clamp of `x` to `[lo, hi]`; NaN stays NaN because neither comparison holds for it.
fn clamp_value<T: Real>(x: T, lo: T, hi: T) -> T {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// The derivative of a clamp whose slope inside `(lo, hi)` is `inside`: `inside` strictly
/// between the bounds, 0 at and beyond them, NaN for NaN.
fn clamp_derivative<T: Real>(x: T, lo: T, hi: T, inside: T) -> T {
    if x > lo && x < hi {
        inside
    } else if x <= lo || x >= hi {
        T::ZERO
    } else {
        nan()
    }
}

/// Hard sigmoid: 0 up to -3, 1 from 3, `x/6 + 1/2` between (NaN stays NaN).
fn hardsigmoid_value<T: Real>(x: T) -> T {
    let three = T::from_f64(3.0);
    if x <= -three {
        T::ZERO
    } else if x >= three {
        T::ONE
    } else {
        x / T::from_f64(6.0) + T::from_f64(0.5)
    }
}

/// Hard swish: 0 up to -3, `x` from 3, `x · (x + 3) / 6` between (NaN stays NaN).
fn hardswish_value<T: Real>(x: T) -> T {
    let three = T::from_f64(3.0);
    if x <= -three {
        T::ZERO
    } else if x >= three {
        x
    } else {
        x * (x + three) / T::from_f64(6.0)
    }
}

/// The derivative of hard swish, with PyTorch's boundaries: 0 below -3, 1 above 3, and
/// `x/3 + 1/2` on `[-3, 3]` (NaN for NaN).
fn hardswish_derivative<T: Real>(x: T) -> T {
    let three = T::from_f64(3.0);
    if x < -three {
        T::ZERO
    } else if x > three {
        T::ONE
    } else if x >= -three && x <= three {
        x / three + T::from_f64(0.5)
    } else {
        nan()
    }
}

/// `log σ(x) = min(x, 0) - ln(1 + e^-|x|)`, which never overflows.
pub(crate) fn log_sigmoid_value<T: Real>(x: T) -> T {
    x.min(T::ZERO) - (-x.abs()).exp().ln_1p()
}

/// The slope index PReLU uses for element `i`: 0 for a single slope, the channel otherwise.
fn prelu_channel(i: usize, slopes: usize) -> usize {
    if slopes == 1 { 0 } else { i % slopes }
}

/// GELU of `x` in the given form.
pub(crate) fn gelu_value<T: Real>(x: T, approx: GeluApprox) -> T {
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
pub(super) fn gelu_derivative<T: Real>(x: T, approx: GeluApprox) -> T {
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
    /// PyTorch. NaN passes through, in the value and in the gradient.
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
        self.map(Op::Relu(x), x, relu_value)
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

    /// Records the hyperbolic tangent of every element. In f32 it runs the SIMD kernel of
    /// the current level, within 3 ulp of the correctly rounded result (see
    /// [`kernels::activation`](crate::kernels::activation)).
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
        self.map_slice(Op::Tanh(x), x, T::tanh_in_place)
    }

    /// Records the logistic sigmoid `1 / (1 + e^-x)` of every element. It is computed as
    /// `1 / (1 + e^-x)` for `x >= 0` and `e^x / (1 + e^x)` below, so it never overflows:
    /// large negative inputs give 0 and large positive inputs give 1. In f32 it runs the
    /// SIMD kernel of the current level, within 3 ulp of the correctly rounded result (see
    /// [`kernels::activation`](crate::kernels::activation)).
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
        self.map_slice(Op::Sigmoid(x), x, T::sigmoid_in_place)
    }

    /// Records GELU of every element, in the exact form (with `erf`) or the tanh
    /// approximation; see [`GeluApprox`]. In f32 the tanh form's value runs the tanh's SIMD
    /// kernel of the current level
    /// ([`kernels::activation::gelu_tanh_f32`](crate::kernels::activation::gelu_tanh_f32),
    /// with its bound); the derivative uses std's `tanh`.
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
        match approx {
            GeluApprox::Exact => self.map(Op::Gelu(x, approx), x, |v| gelu_value(v, approx)),
            GeluApprox::Tanh => self.map_slice(Op::Gelu(x, approx), x, T::gelu_tanh_in_place),
        }
    }

    /// Records ELU: `x` where `x > 0`, `alpha · (eˣ - 1)` elsewhere. PyTorch's default `alpha`
    /// is 1. The derivative is 1 for `x > 0` and `alpha · eˣ` elsewhere (so `alpha` at exactly 0).
    /// The backward pass recomputes `eˣ` from the input: reading it off the output as `y + alpha`
    /// would cancel most digits for large negative `x`.
    /// In f32 the value runs the SIMD kernel of the current level
    /// ([`kernels::activation::elu_f32`](crate::kernels::activation::elu_f32), with its bound);
    /// the derivative uses std's `exp`.
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
    /// let y = tape.elu(x, 1.0);
    /// assert_eq!(tape.value(y).data(), &[-0.63212055, 2.0]);
    /// ```
    pub fn elu(&mut self, x: Var, alpha: T) -> Var {
        self.map_slice(Op::Elu(x, alpha.to_f64()), x, |v| T::elu_in_place(v, alpha))
    }

    /// Records SELU: `scale · elu(x, alpha)` with `scale = 1.0507009873554805` and
    /// `alpha = 1.6732632423543772`, the constants PyTorch uses. The derivative is `scale` for
    /// `x > 0` and `scale · alpha · eˣ` elsewhere, computed from the input.
    /// In f32 the value runs the ELU's SIMD kernel of the current level
    /// ([`kernels::activation::selu_f32`](crate::kernels::activation::selu_f32)).
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
    /// let x = Tensor::from_vec(vec![0.0f32, 1.0], &[2]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.selu(x);
    /// assert_eq!(tape.value(y).data(), &[0.0, 1.050701]);
    /// ```
    pub fn selu(&mut self, x: Var) -> Var {
        self.map_slice(Op::Selu(x), x, T::selu_in_place)
    }

    /// Records CELU: `max(0, x) + min(0, alpha · (e^(x/alpha) - 1))`. PyTorch's default `alpha`
    /// is 1. The derivative is 1 for `x > 0` and `e^(x/alpha)` elsewhere.
    /// In f32 the value runs the ELU's SIMD kernel of the current level for a positive
    /// `alpha` ([`kernels::activation::celu_f32`](crate::kernels::activation::celu_f32)).
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is 0, or if `x` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::from_vec(vec![0.0f32, 2.0], &[2]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.celu(x, 1.0);
    /// assert_eq!(tape.value(y).data(), &[0.0, 2.0]);
    /// ```
    pub fn celu(&mut self, x: Var, alpha: T) -> Var {
        assert!(alpha != T::ZERO, "celu: alpha must not be 0");
        self.map_slice(Op::Celu(x, alpha.to_f64()), x, |v| {
            T::celu_in_place(v, alpha)
        })
    }

    /// Records SiLU (Swish): `x · σ(x)`. The derivative is `σ(x) · (1 + x · (1 - σ(x)))`.
    /// In f32 the value runs the sigmoid's SIMD kernel of the current level
    /// ([`kernels::activation::silu_f32`](crate::kernels::activation::silu_f32), within 7
    /// ulp); the derivative uses the scalar sigmoid.
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
    /// let x = Tensor::from_vec(vec![0.0f32, 2000.0], &[2]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.silu(x);
    /// assert_eq!(tape.value(y).data(), &[0.0, 2000.0]);
    /// ```
    pub fn silu(&mut self, x: Var) -> Var {
        self.map_slice(Op::Silu(x), x, T::silu_in_place)
    }

    /// Records Mish: `x · tanh(softplus(x))`, with softplus at PyTorch's defaults (`beta` 1,
    /// threshold 20). The derivative is `tanh(sp) + x · σ(x) · (1 - tanh²(sp))`.
    /// In f32 the value runs the softplus and tanh SIMD kernels of the current level
    /// ([`kernels::activation::mish_f32`](crate::kernels::activation::mish_f32), with its
    /// bound); the derivative uses the scalar forms.
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
    /// let x = Tensor::from_vec(vec![0.0f32, 100.0], &[2]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.mish(x);
    /// assert_eq!(tape.value(y).data(), &[0.0, 100.0]);
    /// ```
    pub fn mish(&mut self, x: Var) -> Var {
        self.map_slice(Op::Mish(x), x, T::mish_in_place)
    }

    /// Records softplus: `ln(1 + e^(beta·x)) / beta` where `beta · x <= threshold`, and `x` itself
    /// above, which is exact and avoids overflow. PyTorch's defaults are `beta = 1` and
    /// `threshold = 20`. The derivative is `σ(beta · x)`, and 1 above the threshold.
    /// In f32 the value runs the softplus SIMD kernel of the current level
    /// ([`kernels::activation::softplus_beta_f32`](crate::kernels::activation::softplus_beta_f32));
    /// the derivative uses the scalar sigmoid.
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
    /// let x = Tensor::from_vec(vec![30.0f32, 40.0], &[2]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.softplus(x, 1.0, 20.0);
    /// assert_eq!(tape.value(y).data(), &[30.0, 40.0]);
    /// ```
    pub fn softplus(&mut self, x: Var, beta: T, threshold: T) -> Var {
        let op = Op::Softplus(x, beta.to_f64(), threshold.to_f64());
        self.map_slice(op, x, |v| T::softplus_in_place(v, beta, threshold))
    }

    /// Records softsign: `x / (1 + |x|)`. The derivative is `1 / (1 + |x|)²`.
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
    /// let x = Tensor::from_vec(vec![3.0f32, -1.0], &[2]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.softsign(x);
    /// assert_eq!(tape.value(y).data(), &[0.75, -0.5]);
    /// ```
    pub fn softsign(&mut self, x: Var) -> Var {
        self.map(Op::Softsign(x), x, |v| v / (T::ONE + v.abs()))
    }

    /// Records hardtanh: `x` clamped to `[min, max]` (PyTorch's defaults are -1 and 1). The
    /// derivative is 1 strictly inside `(min, max)` and 0 at the bounds and beyond. NaN passes
    /// through, in the value and in the gradient.
    ///
    /// # Panics
    ///
    /// Panics if `min > max` (or either is NaN), or if `x` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::from_vec(vec![2.0f32, 0.5], &[2]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.hardtanh(x, -1.0, 1.0);
    /// assert_eq!(tape.value(y).data(), &[1.0, 0.5]);
    /// ```
    pub fn hardtanh(&mut self, x: Var, min: T, max: T) -> Var {
        assert!(min <= max, "hardtanh: min must not exceed max");
        self.map(Op::Hardtanh(x, min.to_f64(), max.to_f64()), x, |v| {
            clamp_value(v, min, max)
        })
    }

    /// Records ReLU6: `x` clamped to `[0, 6]`, the same as `hardtanh(x, 0, 6)`. The derivative is 0
    /// at exactly 0 and 6. NaN passes through.
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
    /// let x = Tensor::from_vec(vec![7.0f32, 3.0], &[2]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.relu6(x);
    /// assert_eq!(tape.value(y).data(), &[6.0, 3.0]);
    /// ```
    pub fn relu6(&mut self, x: Var) -> Var {
        self.hardtanh(x, T::ZERO, T::from_f64(6.0))
    }

    /// Records the hard sigmoid: 0 for `x <= -3`, 1 for `x >= 3`, `x/6 + 1/2` between. The
    /// derivative is 1/6 strictly inside `(-3, 3)` and 0 elsewhere. NaN passes through.
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
    /// let x = Tensor::from_vec(vec![-4.0f32, 0.0], &[2]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.hardsigmoid(x);
    /// assert_eq!(tape.value(y).data(), &[0.0, 0.5]);
    /// ```
    pub fn hardsigmoid(&mut self, x: Var) -> Var {
        self.map(Op::HardSigmoid(x), x, hardsigmoid_value)
    }

    /// Records the hard swish: 0 for `x <= -3`, `x` for `x >= 3`, `x · (x + 3) / 6` between. The
    /// derivative follows PyTorch's boundaries: 0 for `x < -3`, `x/3 + 1/2` for `-3 <= x <= 3`, 1
    /// for `x > 3`. NaN passes through.
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
    /// let x = Tensor::from_vec(vec![-4.0f32, 4.0], &[2]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.hardswish(x);
    /// assert_eq!(tape.value(y).data(), &[0.0, 4.0]);
    /// ```
    pub fn hardswish(&mut self, x: Var) -> Var {
        self.map(Op::HardSwish(x), x, hardswish_value)
    }

    /// Records log-sigmoid: `log σ(x)`, computed as `min(x, 0) - ln(1 + e^-|x|)` so it never
    /// overflows: `log_sigmoid(-1000)` is exactly -1000. The derivative is `σ(-x)`.
    /// In f32 the value runs the softplus SIMD kernel of the current level
    /// ([`kernels::activation::log_sigmoid_f32`](crate::kernels::activation::log_sigmoid_f32)).
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
    /// let x = Tensor::from_vec(vec![-1000.0f32, 1000.0], &[2]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.log_sigmoid(x);
    /// assert_eq!(tape.value(y).data(), &[-1000.0, 0.0]);
    /// ```
    pub fn log_sigmoid(&mut self, x: Var) -> Var {
        self.map_slice(Op::LogSigmoid(x), x, T::log_sigmoid_in_place)
    }

    /// Records PReLU: `x` where `x > 0`, `slope · x` elsewhere. `slope` has shape `[1]` (one
    /// slope for everything) or `[C]` with `C` the width of `x`'s last axis (one slope per
    /// channel). Gradients go to `x` and to `slope`; a slope's gradient is summed over every
    /// position of its channel. The derivative at exactly 0 is the slope.
    ///
    /// # Panics
    ///
    /// Panics, naming both shapes, if `slope` has any other shape, or if either variable
    /// was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::from_vec(vec![-2.0f32, 3.0, -4.0, 5.0], &[2, 2]).unwrap();
    /// let x = tape.input(x.view());
    /// let slope = Tensor::from_vec(vec![0.5f32, 0.25], &[2]).unwrap();
    /// let slope = tape.input(slope.view());
    /// let y = tape.prelu(x, slope);
    /// assert_eq!(tape.value(y).data(), &[-1.0, 3.0, -2.0, 5.0]);
    /// ```
    pub fn prelu(&mut self, x: Var, slope: Var) -> Var {
        let (sx, ss) = (self.shape(x), self.shape(slope));
        let width = sx.dims().last().copied().unwrap_or(1);
        let fits = ss.rank() == 1 && (ss.dims()[0] == 1 || ss.dims()[0] == width);
        assert!(
            fits,
            "prelu: slope shape {:?} does not fit x shape {:?}; expected [1] or [{width}]",
            ss.dims(),
            sx.dims()
        );
        let mut out = self.take_buffer(sx.len());
        let slopes = self.data(slope);
        for (i, (o, &v)) in out.iter_mut().zip(self.data(x)).enumerate() {
            *o = if v > T::ZERO {
                v
            } else {
                slopes[prelu_channel(i, slopes.len())] * v
            };
        }
        self.push(Op::Prelu(x, slope), sx, out)
    }

    /// Records `op` with the value `f` applied to every element of `x`.
    fn map(&mut self, op: Op<T>, x: Var, f: impl Fn(T) -> T) -> Var {
        let shape = self.shape(x);
        let mut out = self.take_buffer(shape.len());
        debug_assert_eq!(out.len(), self.data(x).len());
        for (o, &v) in out.iter_mut().zip(self.data(x)) {
            *o = f(v);
        }
        self.push(op, shape, out)
    }

    /// Records `op`, whose value is `x` with `f` applied to the whole buffer in place.
    fn map_slice(&mut self, op: Op<T>, x: Var, f: impl Fn(&mut [T])) -> Var {
        let shape = self.shape(x);
        let mut out = self.take_buffer(shape.len());
        out.copy_from_slice(self.data(x));
        f(&mut out);
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

/// Backward of `relu`: the gradient where the input is positive, 0 elsewhere, NaN where the
/// input is NaN.
pub(super) fn relu_backward<T: Real>(x: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    let input = ctx.value(x);
    backward_with(x, out_grad, ctx, |i| relu_derivative(input[i]));
}

/// Backward of `leaky_relu`: the gradient where the input is positive, `slope` times it
/// elsewhere (including at 0), NaN where the input is NaN.
pub(super) fn leaky_relu_backward<T: Real>(
    x: Var,
    slope: f64,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let input = ctx.value(x);
    let slope = T::from_f64(slope);
    backward_with(x, out_grad, ctx, |i| {
        if input[i] > T::ZERO {
            T::ONE
        } else if input[i] <= T::ZERO {
            slope
        } else {
            nan()
        }
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

/// Backward of `elu`: 1 above 0, `alpha · eˣ` elsewhere, from the input.
pub(super) fn elu_backward<T: Real>(
    x: Var,
    alpha: f64,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let input = ctx.value(x);
    let alpha = T::from_f64(alpha);
    backward_with(x, out_grad, ctx, |i| {
        if input[i] > T::ZERO {
            T::ONE
        } else {
            alpha * input[i].exp()
        }
    });
}

/// Backward of `selu`: `scale` above 0, `scale · alpha · eˣ` elsewhere, from the input.
pub(super) fn selu_backward<T: Real>(x: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    let input = ctx.value(x);
    let (scale, alpha) = (T::from_f64(SELU_SCALE), T::from_f64(SELU_ALPHA));
    backward_with(x, out_grad, ctx, |i| {
        if input[i] > T::ZERO {
            scale
        } else {
            scale * alpha * input[i].exp()
        }
    });
}

/// Backward of `celu`: 1 above 0, `e^(x/alpha)` elsewhere.
pub(super) fn celu_backward<T: Real>(
    x: Var,
    alpha: f64,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let input = ctx.value(x);
    let alpha = T::from_f64(alpha);
    backward_with(x, out_grad, ctx, |i| {
        if input[i] > T::ZERO {
            T::ONE
        } else {
            (input[i] / alpha).exp()
        }
    });
}

/// Backward of `silu`: `σ(x) · (1 + x · (1 - σ(x)))`.
pub(super) fn silu_backward<T: Real>(x: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    let input = ctx.value(x);
    backward_with(x, out_grad, ctx, |i| silu_derivative(input[i]));
}

/// Backward of `mish`.
pub(super) fn mish_backward<T: Real>(x: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    let input = ctx.value(x);
    backward_with(x, out_grad, ctx, |i| mish_derivative(input[i]));
}

/// Backward of `softplus`: `σ(beta · x)`, 1 above the threshold.
pub(super) fn softplus_backward<T: Real>(
    x: Var,
    beta: f64,
    threshold: f64,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let input = ctx.value(x);
    let (beta, threshold) = (T::from_f64(beta), T::from_f64(threshold));
    backward_with(x, out_grad, ctx, |i| {
        softplus_derivative(input[i], beta, threshold)
    });
}

/// Backward of `softsign`: `1 / (1 + |x|)²`.
pub(super) fn softsign_backward<T: Real>(x: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    let input = ctx.value(x);
    backward_with(x, out_grad, ctx, |i| {
        let d = T::ONE + input[i].abs();
        T::ONE / (d * d)
    });
}

/// Backward of `hardtanh`: 1 strictly inside the bounds, 0 elsewhere, NaN for NaN.
pub(super) fn hardtanh_backward<T: Real>(
    x: Var,
    lo: f64,
    hi: f64,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let input = ctx.value(x);
    let (lo, hi) = (T::from_f64(lo), T::from_f64(hi));
    backward_with(x, out_grad, ctx, |i| {
        clamp_derivative(input[i], lo, hi, T::ONE)
    });
}

/// Backward of `hardsigmoid`: 1/6 strictly inside `(-3, 3)`, 0 elsewhere, NaN for NaN.
pub(super) fn hardsigmoid_backward<T: Real>(x: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    let input = ctx.value(x);
    let three = T::from_f64(3.0);
    let sixth = T::ONE / T::from_f64(6.0);
    backward_with(x, out_grad, ctx, |i| {
        clamp_derivative(input[i], -three, three, sixth)
    });
}

/// Backward of `hardswish`.
pub(super) fn hardswish_backward<T: Real>(x: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    let input = ctx.value(x);
    backward_with(x, out_grad, ctx, |i| hardswish_derivative(input[i]));
}

/// Backward of `log_sigmoid`: `σ(-x)`.
pub(super) fn log_sigmoid_backward<T: Real>(x: Var, out_grad: &[T], ctx: &mut BackwardCtx<'_, T>) {
    let input = ctx.value(x);
    backward_with(x, out_grad, ctx, |i| sigmoid_value(-input[i]));
}

/// Backward of `prelu`: `x` gets the gradient where `x > 0` and `slope` times it elsewhere;
/// each slope gets the sum of `gradient · x` over the non-positive elements of its channel.
/// A NaN input gives a NaN input gradient and makes its channel's slope gradient NaN.
pub(super) fn prelu_backward<T: Real>(
    x: Var,
    slope: Var,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let input = ctx.value(x);
    let slopes = ctx.value(slope);
    backward_with(x, out_grad, ctx, |i| {
        if input[i] > T::ZERO {
            T::ONE
        } else if input[i] <= T::ZERO {
            slopes[prelu_channel(i, slopes.len())]
        } else {
            nan()
        }
    });
    if !ctx.takes_grad(slope) {
        return;
    }
    let mut sums = ctx.scratch(slopes.len());
    sums.fill(T::ZERO);
    for (i, (&g, &v)) in out_grad.iter().zip(input).enumerate() {
        if v > T::ZERO {
            continue;
        }
        // Not above 0 includes NaN, which `g * v` carries into the sum.
        sums[prelu_channel(i, slopes.len())] += g * v;
    }
    if let Some(slot) = ctx.slot(slope) {
        slot.add_from(&sums);
    }
    ctx.release(sums);
}

/// Tests for the activations.
#[cfg(test)]
mod tests {
    use super::{SELU_ALPHA, SELU_SCALE};
    use crate::{GeluApprox, Graph, ParamStore, Tape, Tensor, Var};

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

    /// In f32, where `tanh` and `sigmoid` run the SIMD kernels, values and gradients agree
    /// with the exact f64 functions and derivatives at the current kernel level (the process
    /// setting is not changed, since other tests run beside this one), over inputs on both
    /// sides of the switch between the kernel's two `tanh` formulas.
    #[test]
    fn f32_kernels_give_the_f64_values_and_gradients() {
        let level = crate::kernels::level();
        let xs: Vec<f32> = (-400..=400).map(|i| i as f32 * 0.0137).collect();
        for tanh in [true, false] {
            let mut store = ParamStore::new();
            let p = store.add(Tensor::from_vec(xs.clone(), &[xs.len()]).unwrap());
            let mut graph = Graph::new();
            let mut tape = graph.tape(&store.values);
            let v = tape.param(p);
            let y = if tanh { tape.tanh(v) } else { tape.sigmoid(v) };
            let loss = tape.sum(y);
            let values = tape.value(y).data().to_vec();
            tape.backward(loss, &mut store.grads);
            drop(tape);
            for ((&x, &got), &grad) in xs.iter().zip(&values).zip(store.grad(p).data()) {
                let x = x as f64;
                let (want, slope) = if tanh {
                    let t = x.tanh();
                    (t, 1.0 - t * t)
                } else {
                    let s = 1.0 / (1.0 + (-x).exp());
                    (s, s * (1.0 - s))
                };
                assert!((got as f64 - want).abs() <= 4e-7, "{level} value at {x}");
                assert!(
                    (grad as f64 - slope).abs() <= 1e-6,
                    "{level} gradient at {x}"
                );
            }
        }
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

    /// Records `build` on a one-element f64 parameter at `x`; returns the value and the
    /// derivative of the output with respect to `x`.
    fn value_and_derivative(x: f64, build: impl Fn(&mut Tape<'_, f64>, Var) -> Var) -> (f64, f64) {
        let mut store = ParamStore::<f64>::new_generic();
        let p = store.add(Tensor::from_vec(vec![x], &[1]).unwrap());
        let mut graph = Graph::<f64>::new_generic();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p);
        let y = build(&mut tape, v);
        let loss = tape.sum(y);
        let value = tape.value(y).data()[0];
        tape.backward(loss, &mut store.grads);
        drop(tape);
        (value, store.grad(p).data()[0])
    }

    /// Asserts `a` and `b` agree to 1e-14 (absolute).
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-14, "{a} vs {b}");
    }

    /// ELU at -1 is e^-1 - 1, with derivative e^-1; the slope above 0 is 1.
    #[test]
    fn elu_values() {
        let (v, d) = value_and_derivative(-1.0, |t, x| t.elu(x, 1.0));
        close(v, -0.6321205588285577);
        close(d, (-1.0f64).exp());
        assert_eq!(value_and_derivative(2.0, |t, x| t.elu(x, 1.0)), (2.0, 1.0));
        // At exactly 0 the derivative is alpha (PyTorch's convention).
        assert_eq!(value_and_derivative(0.0, |t, x| t.elu(x, 0.5)), (0.0, 0.5));
    }

    /// SELU at -1 and 1.
    #[test]
    fn selu_values() {
        close(
            value_and_derivative(-1.0, |t, x| t.selu(x)).0,
            -1.1113307378125625,
        );
        let (v, d) = value_and_derivative(1.0, |t, x| t.selu(x));
        close(v, 1.0507009873554805);
        close(d, 1.0507009873554805);
    }

    /// CELU at -1 with alpha 2.
    #[test]
    fn celu_values() {
        let (v, d) = value_and_derivative(-1.0, |t, x| t.celu(x, 2.0));
        close(v, -0.7869386805747332);
        close(d, (-0.5f64).exp());
    }

    /// Runs `build` on the f32 inputs `xs` and returns the outputs.
    fn f32_outputs(xs: &[f32], build: impl Fn(&mut Tape<'_, f32>, Var) -> Var) -> Vec<f32> {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = Tensor::from_vec(xs.to_vec(), &[xs.len()]).unwrap();
        let x = tape.input(x.view());
        let y = build(&mut tape, x);
        tape.value(y).data().to_vec()
    }

    /// ELU, SELU and CELU keep the sign of `-0.0`, keep a negative result for a subnormal
    /// negative input, and ELU matches `alpha * x` to one ulp for a tiny input.
    #[test]
    fn exp_family_keeps_signed_zero_and_subnormals() {
        let selu = f32_outputs(&[-0.0], |t, x| t.selu(x));
        assert!(selu[0] == 0.0 && selu[0].is_sign_negative(), "selu(-0.0)");
        let celu = f32_outputs(&[-0.0], |t, x| t.celu(x, 1.0));
        assert!(celu[0] == 0.0 && celu[0].is_sign_negative(), "celu(-0.0)");

        let tiny = -1e-40f32;
        for (name, y) in [
            ("elu", f32_outputs(&[tiny], |t, x| t.elu(x, 1.0))[0]),
            ("selu", f32_outputs(&[tiny], |t, x| t.selu(x))[0]),
            ("celu", f32_outputs(&[tiny], |t, x| t.celu(x, 1.0))[0]),
        ] {
            assert!(y < 0.0, "{name}(-1e-40) = {y}");
        }

        let x = -1e-10f32;
        let y = f32_outputs(&[x], |t, v| t.elu(v, 1.3))[0];
        let want = 1.3f32 * x;
        let ulp = (want.to_bits() as i64 - y.to_bits() as i64).abs();
        assert!(ulp <= 1, "elu(-1e-10, 1.3) = {y}, wanted {want}");
    }

    /// SiLU at 1.
    #[test]
    fn silu_values() {
        let (v, d) = value_and_derivative(1.0, |t, x| t.silu(x));
        close(v, 0.7310585786300049);
        close(d, 0.9276705118714869);
    }

    /// Mish at 1, and its saturation for large inputs.
    #[test]
    fn mish_values() {
        let (v, d) = value_and_derivative(1.0, |t, x| t.mish(x));
        close(v, 0.8650983882673103);
        close(d, 1.0490362200997922);
        let (v, d) = value_and_derivative(100.0, |t, x| t.mish(x));
        close(v, 100.0);
        close(d, 1.0);
    }

    /// Softplus below and above the threshold.
    #[test]
    fn softplus_values() {
        let (v, d) = value_and_derivative(1.0, |t, x| t.softplus(x, 1.0, 20.0));
        close(v, 1.3132616875182228);
        close(d, 0.7310585786300049);
        // Above the threshold the input passes through exactly, with derivative 1.
        assert_eq!(
            value_and_derivative(30.0, |t, x| t.softplus(x, 1.0, 20.0)),
            (30.0, 1.0)
        );
    }

    /// Softsign at 3.
    #[test]
    fn softsign_values() {
        let (v, d) = value_and_derivative(3.0, |t, x| t.softsign(x));
        assert_eq!(v, 0.75);
        close(d, 1.0 / 16.0);
    }

    /// Hardtanh clamps, with derivative 0 at the bounds and outside.
    #[test]
    fn hardtanh_values() {
        let f = |t: &mut Tape<'_, f64>, x| t.hardtanh(x, -1.0, 1.0);
        assert_eq!(value_and_derivative(2.0, f), (1.0, 0.0));
        assert_eq!(value_and_derivative(1.0, f), (1.0, 0.0));
        assert_eq!(value_and_derivative(0.5, f), (0.5, 1.0));
        assert_eq!(value_and_derivative(-3.0, f), (-1.0, 0.0));
    }

    /// ReLU6 clamps to [0, 6].
    #[test]
    fn relu6_values() {
        assert_eq!(value_and_derivative(7.0, |t, x| t.relu6(x)), (6.0, 0.0));
        assert_eq!(value_and_derivative(3.0, |t, x| t.relu6(x)), (3.0, 1.0));
        assert_eq!(value_and_derivative(-1.0, |t, x| t.relu6(x)), (0.0, 0.0));
    }

    /// Hard sigmoid in the middle and at the bounds.
    #[test]
    fn hardsigmoid_values() {
        let (v, d) = value_and_derivative(1.0, |t, x| t.hardsigmoid(x));
        close(v, 0.6666666666666666);
        close(d, 1.0 / 6.0);
        assert_eq!(
            value_and_derivative(3.0, |t, x| t.hardsigmoid(x)),
            (1.0, 0.0)
        );
        assert_eq!(
            value_and_derivative(-3.0, |t, x| t.hardsigmoid(x)),
            (0.0, 0.0)
        );
    }

    /// Hard swish, including PyTorch's derivative at the bounds.
    #[test]
    fn hardswish_values() {
        let (v, d) = value_and_derivative(1.0, |t, x| t.hardswish(x));
        close(v, 0.6666666666666666);
        close(d, 1.0 / 3.0 + 0.5);
        assert_eq!(
            value_and_derivative(-4.0, |t, x| t.hardswish(x)),
            (0.0, 0.0)
        );
        assert_eq!(value_and_derivative(4.0, |t, x| t.hardswish(x)), (4.0, 1.0));
        // At -3 and 3 the middle formula x/3 + 1/2 applies.
        assert_eq!(
            value_and_derivative(-3.0, |t, x| t.hardswish(x)),
            (0.0, -0.5)
        );
        assert_eq!(value_and_derivative(3.0, |t, x| t.hardswish(x)), (3.0, 1.5));
    }

    /// Log-sigmoid at 0, -1 and -1000 (finite, no overflow), and for large positive input.
    #[test]
    fn log_sigmoid_values() {
        let (v, d) = value_and_derivative(0.0, |t, x| t.log_sigmoid(x));
        close(v, -std::f64::consts::LN_2);
        close(d, 0.5);
        close(
            value_and_derivative(-1.0, |t, x| t.log_sigmoid(x)).0,
            -1.3132616875182228,
        );
        let (v, d) = value_and_derivative(-1000.0, |t, x| t.log_sigmoid(x));
        assert_eq!((v, d), (-1000.0, 1.0));
        assert_eq!(
            value_and_derivative(1000.0, |t, x| t.log_sigmoid(x)),
            (0.0, 0.0)
        );
    }

    /// PReLU with one slope and with one per channel, including the slope gradient.
    #[test]
    fn prelu_values_and_gradients() {
        let mut store = ParamStore::<f64>::new_generic();
        let s = store.add(Tensor::from_vec(vec![0.5, 0.25], &[2]).unwrap());
        let mut graph = Graph::<f64>::new_generic();
        let mut tape = graph.tape(&store.values);
        let x = Tensor::from_vec(vec![-2.0, 3.0, -4.0, -8.0], &[2, 2]).unwrap();
        let x = tape.input(x.view());
        let sv = tape.param(s);
        let y = tape.prelu(x, sv);
        assert_eq!(tape.value(y).data(), &[-1.0, 3.0, -2.0, -2.0]);
        let loss = tape.sum(y);
        tape.backward(loss, &mut store.grads);
        drop(tape);
        // Channel 0 sees -2 and -4; channel 1 sees only -8 (3 is positive).
        assert_eq!(store.grad(s).data(), &[-6.0, -8.0]);
    }

    /// A slope that is neither [1] nor [C] panics naming both shapes.
    #[test]
    #[should_panic(
        expected = "prelu: slope shape [2] does not fit x shape [2, 3]; expected [1] or [3]"
    )]
    fn prelu_bad_slope_shape_panics() {
        let store = ParamStore::<f64>::new_generic();
        let mut graph = Graph::<f64>::new_generic();
        let mut tape = graph.tape(&store.values);
        let x = Tensor::from_vec(vec![0.0; 6], &[2, 3]).unwrap();
        let x = tape.input(x.view());
        let s = Tensor::from_vec(vec![0.0; 2], &[2]).unwrap();
        let s = tape.input(s.view());
        tape.prelu(x, s);
    }

    /// ReLU keeps NaN in the output and in the gradient.
    #[test]
    fn relu_passes_nan_through() {
        let (v, d) = value_and_derivative(f64::NAN, |t, x| t.relu(x));
        assert!(v.is_nan() && d.is_nan(), "{v} {d}");
    }

    /// The clamp family keeps NaN in the output and in the gradient.
    #[test]
    fn hardtanh_passes_nan_through() {
        let (v, d) = value_and_derivative(f64::NAN, |t, x| t.hardtanh(x, -1.0, 1.0));
        assert!(v.is_nan() && d.is_nan(), "{v} {d}");
        let (v, d) = value_and_derivative(f64::NAN, |t, x| t.relu6(x));
        assert!(v.is_nan() && d.is_nan(), "{v} {d}");
        let (v, d) = value_and_derivative(f64::NAN, |t, x| t.hardsigmoid(x));
        assert!(v.is_nan() && d.is_nan(), "{v} {d}");
        let (v, d) = value_and_derivative(f64::NAN, |t, x| t.hardswish(x));
        assert!(v.is_nan() && d.is_nan(), "{v} {d}");
    }

    /// Whether two `f32` values differ by at most one unit in the last place (same sign).
    fn within_one_ulp(a: f32, b: f32) -> bool {
        a.to_bits().abs_diff(b.to_bits()) <= 1
    }

    /// The forward value of `build` at one `f32` input.
    fn forward_f32(x: f32, build: impl Fn(&mut Tape<'_, f32>, Var) -> Var) -> f32 {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let input = Tensor::from_vec(vec![x], &[1]).unwrap();
        let input = tape.input(input.view());
        let y = build(&mut tape, input);
        tape.value(y).data()[0]
    }

    /// ELU, SELU and CELU keep full precision for tiny negative inputs: `e^x - 1` computed
    /// as `exp_m1` is within 1 ulp of `x`, where `exp(x) - 1` would round to 0 or a
    /// multiple of 2^-24.
    #[test]
    fn elu_tiny_negative_keeps_precision() {
        let x = -1e-10f32;
        let elu = forward_f32(x, |t, v| t.elu(v, 1.0));
        assert!(within_one_ulp(elu, x), "elu {elu:e} vs {x:e}");
        let scale = SELU_SCALE as f32;
        let alpha = SELU_ALPHA as f32;
        let selu = forward_f32(x, |t, v| t.selu(v));
        let want = scale * alpha * x;
        assert!(within_one_ulp(selu, want), "selu {selu:e} vs {want:e}");
        let celu = forward_f32(x, |t, v| t.celu(v, 1.0));
        assert!(within_one_ulp(celu, x), "celu {celu:e} vs {x:e}");
    }

    /// `elu(-0.0)` keeps the sign of the zero.
    #[test]
    fn elu_negative_zero() {
        let y = forward_f32(-0.0, |t, v| t.elu(v, 1.0));
        assert_eq!(y, 0.0);
        assert!(y.is_sign_negative(), "sign bit lost: {y:?}");
    }

    /// Leaky ReLU's input gradient is NaN where the input is NaN, and the usual slope
    /// elsewhere.
    #[test]
    fn leaky_relu_nan_gradient() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::from_vec(vec![f32::NAN, 1.0, -1.0], &[3]).unwrap());
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p);
        let r = tape.leaky_relu(v, 0.5);
        let loss = tape.sum(r);
        tape.backward(loss, &mut store.grads);
        drop(tape);
        let g = store.grad(p).data();
        assert!(g[0].is_nan(), "{g:?}");
        assert_eq!(&g[1..], &[1.0, 0.5]);
    }

    /// PReLU's input gradient and its slope gradient are NaN where the input is NaN.
    #[test]
    fn prelu_nan_gradient() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::from_vec(vec![f32::NAN, 1.0, -1.0], &[3]).unwrap());
        let s = store.add(Tensor::from_vec(vec![0.25], &[1]).unwrap());
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p);
        let sv = tape.param(s);
        let r = tape.prelu(v, sv);
        let loss = tape.sum(r);
        tape.backward(loss, &mut store.grads);
        drop(tape);
        let g = store.grad(p).data();
        assert!(g[0].is_nan(), "{g:?}");
        assert_eq!(&g[1..], &[1.0, 0.25]);
        assert!(
            store.grad(s).data()[0].is_nan(),
            "{:?}",
            store.grad(s).data()
        );
    }
}
