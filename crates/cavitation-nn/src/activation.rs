//! The `Activation` trait, `ActivationKind`, the activation layers, and the layers that are
//! not element-wise functions of one value (`Softmax`, `LogSoftmax`, `Glu`, `Prelu`).

use std::fmt;
use std::sync::Arc;

use cavitation_core::{
    CustomOp, Gate, GeluApprox, ParamId, ParamStore, ParamValues, Real, Shape, Tape, Tensor,
    TensorView, TensorViewMut, Var,
};

use crate::{InferenceLayer, Layer};

/// A function applied to every value of a layer's input on its own: its value `f` and its
/// derivative `df`.
///
/// Wrap an implementation in an [`ActivationLayer`] to use it in a network. The layer
/// records it as one operation on the tape, built from `f` and `df`, so the network trains
/// and passes [`gradcheck`](cavitation_core::gradcheck) without any extra code.
///
/// The built-in activations ([`ActivationKind`]) also implement this trait, and report their
/// kind through [`builtin`](Self::builtin); a layer then uses the fused tape operation for
/// it instead of the generic one.
///
/// Implement it for every [`Real`] type so the same code runs in `f32` and in `f64`.
///
/// ```
/// use cavitation_core::Real;
/// use cavitation_nn::Activation;
///
/// /// `x³`.
/// struct Cube;
///
/// impl<T: Real> Activation<T> for Cube {
///     fn name(&self) -> &str {
///         "cube"
///     }
///     fn f(&self, x: T) -> T {
///         x * x * x
///     }
///     // The derivative at x, given y = f(x).
///     fn df(&self, x: T, _y: T) -> T {
///         T::from_f64(3.0) * x * x
///     }
/// }
///
/// assert_eq!(Activation::<f64>::f(&Cube, 2.0), 8.0);
/// assert_eq!(Activation::<f64>::df(&Cube, 2.0, 8.0), 12.0);
/// ```
pub trait Activation<T: Real>: Send + Sync {
    /// The activation's name, for example `"relu"`. Used as the layer's name.
    ///
    /// ```
    /// use cavitation_nn::{Activation, ActivationKind};
    /// assert_eq!(Activation::<f32>::name(&ActivationKind::Silu), "silu");
    /// ```
    fn name(&self) -> &str;

    /// The value at `x`.
    ///
    /// ```
    /// use cavitation_nn::{Activation, ActivationKind};
    /// assert_eq!(Activation::<f32>::f(&ActivationKind::Relu, -2.0), 0.0);
    /// ```
    fn f(&self, x: T) -> T;

    /// The derivative at `x`, given `y = f(x)` (some derivatives are cheaper from `y`).
    ///
    /// ```
    /// use cavitation_nn::{Activation, ActivationKind};
    /// assert_eq!(Activation::<f32>::df(&ActivationKind::Tanh, 0.0, 0.0), 1.0);
    /// ```
    fn df(&self, x: T, y: T) -> T;

    /// The built-in kind this activation is, if it is one. A built-in selects the fused tape
    /// operation, the inference kernel and the dead-neuron check. User-defined activations
    /// keep the default, `None`.
    ///
    /// ```
    /// use cavitation_nn::{Activation, ActivationKind};
    /// assert_eq!(Activation::<f32>::builtin(&ActivationKind::Mish), Some(ActivationKind::Mish));
    /// ```
    fn builtin(&self) -> Option<ActivationKind> {
        None
    }
}

/// The built-in activations, with their parameters.
///
/// Each follows PyTorch's definition and its conventions at the kinks (for example the
/// derivative of ReLU at 0 is 0).
///
/// ```
/// use cavitation_nn::{Activation, ActivationKind};
/// let kind = ActivationKind::LeakyRelu(0.1);
/// assert_eq!(Activation::<f64>::f(&kind, -2.0), -0.2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActivationKind {
    /// `max(x, 0)`.
    Relu,
    /// `x` above 0, `slope · x` elsewhere.
    LeakyRelu(f64),
    /// The hyperbolic tangent.
    Tanh,
    /// The logistic sigmoid `1 / (1 + e⁻ˣ)`.
    Sigmoid,
    /// GELU in the given form.
    Gelu(GeluApprox),
    /// `x` above 0, `alpha · (eˣ - 1)` elsewhere.
    Elu(f64),
    /// The scaled ELU of self-normalising networks.
    Selu,
    /// `x` above 0, `alpha · (e^(x/alpha) - 1)` elsewhere.
    Celu(f64),
    /// `x · σ(x)`, also called swish.
    Silu,
    /// `x · tanh(softplus(x))`.
    Mish,
    /// `ln(1 + e^(beta·x)) / beta`, and `x` itself once `beta · x` passes `threshold`.
    Softplus {
        /// The sharpness `beta`.
        beta: f64,
        /// The value of `beta · x` above which the output is `x`.
        threshold: f64,
    },
    /// `x / (1 + |x|)`.
    Softsign,
    /// `x` clamped to `[min, max]`.
    Hardtanh {
        /// The lower bound.
        min: f64,
        /// The upper bound.
        max: f64,
    },
    /// `min(max(x, 0), 6)`.
    Relu6,
    /// The piecewise-linear sigmoid: 0 up to -3, 1 from 3, `x/6 + 1/2` between.
    Hardsigmoid,
    /// `x · hardsigmoid(x)`.
    Hardswish,
    /// `ln σ(x)`.
    LogSigmoid,
}

/// `1 / (1 + e^-x)` without overflowing the exponential, one value at a time with std's
/// `exp`: the scalar sigmoid that the derivatives and the GLU gates use (the `Sigmoid` and
/// `Silu` layers run the SIMD kernels instead).
pub(crate) fn sigmoid<T: Real>(x: T) -> T {
    if x >= T::ZERO {
        T::ONE / (T::ONE + (-x).exp())
    } else {
        let e = x.exp();
        e / (T::ONE + e)
    }
}

/// The tanh form of GELU, one value at a time with std's `tanh`: the scalar form the GLU's
/// GELU gate uses, as the tape's GLU does (the `Gelu` layer runs the tanh kernel).
pub(crate) fn gelu_tanh_scalar<T: Real>(x: T) -> T {
    let c = T::from_f64;
    let u = c(SQRT_2_OVER_PI) * (x + c(GELU_CUBIC) * x * x * x);
    c(0.5) * x * (T::ONE + u.tanh())
}

/// NaN of the element type.
fn nan<T: Real>() -> T {
    T::from_f64(f64::NAN)
}

/// `1/√2`.
const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// `1/√(2π)`.
const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
/// `√(2/π)`, the scale inside the tanh form of GELU.
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
/// The cubic coefficient inside the tanh form of GELU.
const GELU_CUBIC: f64 = 0.044715;
/// SELU's scale.
const SELU_SCALE: f64 = 1.050_700_987_355_480_5;
/// SELU's alpha.
const SELU_ALPHA: f64 = 1.673_263_242_354_377_2;
/// PyTorch's default softplus threshold, which Mish uses.
const MISH_THRESHOLD: f64 = 20.0;

/// Softplus `ln(1 + e^(beta·x)) / beta`, or `x` once `beta·x` passes `threshold`.
fn softplus<T: Real>(x: T, beta: T, threshold: T) -> T {
    let z = beta * x;
    if z > threshold {
        x
    } else {
        z.exp().ln_1p() / beta
    }
}

/// The derivative of an upper-and-lower clamp whose slope inside `(lo, hi)` is `inside`:
/// 0 at and beyond the bounds, NaN for NaN.
fn clamp_derivative<T: Real>(x: T, lo: T, hi: T, inside: T) -> T {
    if x > lo && x < hi {
        inside
    } else if x <= lo || x >= hi {
        T::ZERO
    } else {
        nan()
    }
}

impl ActivationKind {
    /// The kind's name.
    fn kind_name(self) -> &'static str {
        match self {
            ActivationKind::Relu => "relu",
            ActivationKind::LeakyRelu(_) => "leaky_relu",
            ActivationKind::Tanh => "tanh",
            ActivationKind::Sigmoid => "sigmoid",
            ActivationKind::Gelu(_) => "gelu",
            ActivationKind::Elu(_) => "elu",
            ActivationKind::Selu => "selu",
            ActivationKind::Celu(_) => "celu",
            ActivationKind::Silu => "silu",
            ActivationKind::Mish => "mish",
            ActivationKind::Softplus { .. } => "softplus",
            ActivationKind::Softsign => "softsign",
            ActivationKind::Hardtanh { .. } => "hardtanh",
            ActivationKind::Relu6 => "relu6",
            ActivationKind::Hardsigmoid => "hardsigmoid",
            ActivationKind::Hardswish => "hardswish",
            ActivationKind::LogSigmoid => "log_sigmoid",
        }
    }

    /// The value at `x`.
    fn value<T: Real>(self, x: T) -> T {
        let c = T::from_f64;
        let half = c(0.5);
        let three = c(3.0);
        match self {
            ActivationKind::Relu => {
                if x <= T::ZERO {
                    T::ZERO
                } else {
                    x
                }
            }
            ActivationKind::LeakyRelu(slope) => {
                if x > T::ZERO {
                    x
                } else {
                    c(slope) * x
                }
            }
            // The kernels the tape and inference run, so the value here matches theirs bit
            // for bit.
            ActivationKind::Tanh => {
                let mut v = [x];
                T::tanh_in_place(&mut v);
                v[0]
            }
            ActivationKind::Sigmoid => {
                let mut v = [x];
                T::sigmoid_in_place(&mut v);
                v[0]
            }
            ActivationKind::Gelu(GeluApprox::Exact) => {
                half * x * (T::ONE + (x * c(FRAC_1_SQRT_2)).erf())
            }
            ActivationKind::Gelu(GeluApprox::Tanh) => {
                let mut v = [x];
                T::gelu_tanh_in_place(&mut v);
                v[0]
            }
            ActivationKind::Elu(alpha) => {
                if x > T::ZERO {
                    x
                } else {
                    c(alpha) * x.exp_m1()
                }
            }
            // `scale · (alpha · (eˣ - 1))`, grouped as the tape's SELU computes it.
            ActivationKind::Selu => {
                if x > T::ZERO {
                    c(SELU_SCALE) * x
                } else {
                    c(SELU_SCALE) * (c(SELU_ALPHA) * x.exp_m1())
                }
            }
            ActivationKind::Celu(alpha) => {
                if x > T::ZERO {
                    x
                } else {
                    c(alpha) * (x / c(alpha)).exp_m1()
                }
            }
            ActivationKind::Silu => {
                let mut v = [x];
                T::silu_in_place(&mut v);
                v[0]
            }
            ActivationKind::Mish => x * softplus(x, T::ONE, c(MISH_THRESHOLD)).tanh(),
            ActivationKind::Softplus { beta, threshold } => softplus(x, c(beta), c(threshold)),
            ActivationKind::Softsign => x / (T::ONE + x.abs()),
            ActivationKind::Hardtanh { min, max } => {
                let (lo, hi) = (c(min), c(max));
                if x < lo {
                    lo
                } else if x > hi {
                    hi
                } else {
                    x
                }
            }
            ActivationKind::Relu6 => {
                let six = c(6.0);
                if x < T::ZERO {
                    T::ZERO
                } else if x > six {
                    six
                } else {
                    x
                }
            }
            ActivationKind::Hardsigmoid => {
                if x <= -three {
                    T::ZERO
                } else if x >= three {
                    T::ONE
                } else {
                    x / c(6.0) + half
                }
            }
            ActivationKind::Hardswish => {
                if x <= -three {
                    T::ZERO
                } else if x >= three {
                    x
                } else {
                    x * (x + three) / c(6.0)
                }
            }
            ActivationKind::LogSigmoid => x.min(T::ZERO) - (-x.abs()).exp().ln_1p(),
        }
    }

    /// The derivative at `x`, given `y = value(x)`; the same expression the fused tape
    /// operation's backward pass uses.
    fn derivative<T: Real>(self, x: T, y: T) -> T {
        let c = T::from_f64;
        let half = c(0.5);
        let three = c(3.0);
        match self {
            ActivationKind::Relu => {
                if x > T::ZERO {
                    T::ONE
                } else if x <= T::ZERO {
                    T::ZERO
                } else {
                    nan()
                }
            }
            ActivationKind::LeakyRelu(slope) => {
                if x > T::ZERO {
                    T::ONE
                } else if x <= T::ZERO {
                    c(slope)
                } else {
                    nan()
                }
            }
            ActivationKind::Tanh => T::ONE - y * y,
            ActivationKind::Sigmoid => y * (T::ONE - y),
            ActivationKind::Gelu(GeluApprox::Exact) => {
                let cdf = half * (T::ONE + (x * c(FRAC_1_SQRT_2)).erf());
                let pdf = c(FRAC_1_SQRT_2PI) * (-(half * x * x)).exp();
                cdf + x * pdf
            }
            ActivationKind::Gelu(GeluApprox::Tanh) => {
                let x2 = x * x;
                let u = c(SQRT_2_OVER_PI) * (x + c(GELU_CUBIC) * x2 * x);
                let t = u.tanh();
                let du = c(SQRT_2_OVER_PI) * (T::ONE + c(3.0 * GELU_CUBIC) * x2);
                half * (T::ONE + t) + half * x * (T::ONE - t * t) * du
            }
            ActivationKind::Elu(alpha) => {
                if x > T::ZERO {
                    T::ONE
                } else {
                    c(alpha) * x.exp()
                }
            }
            ActivationKind::Selu => {
                if x > T::ZERO {
                    c(SELU_SCALE)
                } else {
                    c(SELU_SCALE) * c(SELU_ALPHA) * x.exp()
                }
            }
            ActivationKind::Celu(alpha) => {
                if x > T::ZERO {
                    T::ONE
                } else {
                    (x / c(alpha)).exp()
                }
            }
            ActivationKind::Silu => {
                let s = sigmoid(x);
                s * (T::ONE + x * (T::ONE - s))
            }
            ActivationKind::Mish => {
                let t = softplus(x, T::ONE, c(MISH_THRESHOLD)).tanh();
                t + x * sigmoid(x) * (T::ONE - t * t)
            }
            ActivationKind::Softplus { beta, threshold } => {
                let z = c(beta) * x;
                if z > c(threshold) { T::ONE } else { sigmoid(z) }
            }
            ActivationKind::Softsign => {
                let d = T::ONE + x.abs();
                T::ONE / (d * d)
            }
            ActivationKind::Hardtanh { min, max } => clamp_derivative(x, c(min), c(max), T::ONE),
            ActivationKind::Relu6 => clamp_derivative(x, T::ZERO, c(6.0), T::ONE),
            ActivationKind::Hardsigmoid => clamp_derivative(x, -three, three, T::ONE / c(6.0)),
            ActivationKind::Hardswish => {
                if x < -three {
                    T::ZERO
                } else if x > three {
                    T::ONE
                } else if x >= -three && x <= three {
                    x / three + half
                } else {
                    nan()
                }
            }
            ActivationKind::LogSigmoid => sigmoid(-x),
        }
    }

    /// Records the fused tape operation for this kind on `x`.
    fn record<T: Real>(self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        let c = T::from_f64;
        match self {
            ActivationKind::Relu => tape.relu(x),
            ActivationKind::LeakyRelu(slope) => tape.leaky_relu(x, c(slope)),
            ActivationKind::Tanh => tape.tanh(x),
            ActivationKind::Sigmoid => tape.sigmoid(x),
            ActivationKind::Gelu(approx) => tape.gelu(x, approx),
            ActivationKind::Elu(alpha) => tape.elu(x, c(alpha)),
            ActivationKind::Selu => tape.selu(x),
            ActivationKind::Celu(alpha) => tape.celu(x, c(alpha)),
            ActivationKind::Silu => tape.silu(x),
            ActivationKind::Mish => tape.mish(x),
            ActivationKind::Softplus { beta, threshold } => tape.softplus(x, c(beta), c(threshold)),
            ActivationKind::Softsign => tape.softsign(x),
            ActivationKind::Hardtanh { min, max } => tape.hardtanh(x, c(min), c(max)),
            ActivationKind::Relu6 => tape.relu6(x),
            ActivationKind::Hardsigmoid => tape.hardsigmoid(x),
            ActivationKind::Hardswish => tape.hardswish(x),
            ActivationKind::LogSigmoid => tape.log_sigmoid(x),
        }
    }
}

impl<T: Real> Activation<T> for ActivationKind {
    /// The kind's name, for example `"leaky_relu"`.
    fn name(&self) -> &str {
        self.kind_name()
    }

    /// The value at `x`.
    fn f(&self, x: T) -> T {
        self.value(x)
    }

    /// The derivative at `x`.
    fn df(&self, x: T, y: T) -> T {
        self.derivative(x, y)
    }

    /// Returns the kind itself.
    fn builtin(&self) -> Option<ActivationKind> {
        Some(*self)
    }
}

/// Runs a user-defined [`Activation`] as an element-wise custom operation.
struct ActivationOp<T: Real> {
    /// The activation whose `f` and `df` the operation calls.
    act: Arc<dyn Activation<T>>,
}

impl<T: Real> CustomOp<T> for ActivationOp<T> {
    /// The activation's name.
    fn name(&self) -> &str {
        self.act.name()
    }

    /// The output has the input's shape.
    fn output_shape(&self, inputs: &[Shape]) -> Shape {
        inputs[0]
    }

    /// Applies `f` to every element.
    fn forward(&self, inputs: &[TensorView<T>], mut output: TensorViewMut<T>) {
        for (y, &x) in output.data_mut().iter_mut().zip(inputs[0].data()) {
            *y = self.act.f(x);
        }
    }

    /// The input gradient is `df(x, y)` times the incoming gradient.
    fn backward(
        &self,
        inputs: &[TensorView<T>],
        output: TensorView<T>,
        grad_output: TensorView<T>,
        grad_inputs: &mut [TensorViewMut<T>],
    ) {
        let (x, y, g) = (inputs[0].data(), output.data(), grad_output.data());
        for (i, d) in grad_inputs[0].data_mut().iter_mut().enumerate() {
            *d = self.act.df(x[i], y[i]) * g[i];
        }
    }
}

/// A layer that applies an [`Activation`] to every value.
///
/// A built-in activation (one whose [`builtin`](Activation::builtin) is `Some`) records the
/// fused tape operation. Any other activation runs as a custom operation built once, in
/// [`new`](Self::new); each step clones its `Arc` and allocates nothing.
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Real, Tensor};
/// use cavitation_nn::{Activation, ActivationLayer, Layer};
///
/// /// `x / 2`, standing in for any function you write.
/// struct Halve;
/// impl<T: Real> Activation<T> for Halve {
///     fn name(&self) -> &str {
///         "halve"
///     }
///     fn f(&self, x: T) -> T {
///         x * T::from_f64(0.5)
///     }
///     fn df(&self, _x: T, _y: T) -> T {
///         T::from_f64(0.5)
///     }
/// }
///
/// let layer = ActivationLayer::<f32>::new(Halve);
/// let store = ParamStore::new();
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&store.values);
/// let x = tape.input(Tensor::from_vec(vec![2.0f32, -4.0], &[1, 2]).unwrap().view());
/// let y = layer.forward(&mut tape, x);
/// assert_eq!(tape.value(y).data(), &[1.0, -2.0]);
/// ```
pub struct ActivationLayer<T: Real> {
    /// The activation.
    act: Arc<dyn Activation<T>>,
    /// The activation's built-in kind, if any, read once.
    kind: Option<ActivationKind>,
    /// The custom operation for a non-built-in activation.
    op: Option<Arc<dyn CustomOp<T> + Send + Sync>>,
}

impl<T: Real> ActivationLayer<T> {
    /// Wraps `act`, building its custom operation once if it is not a built-in.
    ///
    /// ```
    /// use cavitation_nn::{ActivationKind, ActivationLayer, Layer};
    /// let layer = ActivationLayer::<f32>::new(ActivationKind::Tanh);
    /// assert_eq!(layer.name(), "tanh");
    /// ```
    pub fn new(act: impl Activation<T> + 'static) -> ActivationLayer<T> {
        ActivationLayer::from_arc(Arc::new(act))
    }

    /// Wraps a shared activation, building its custom operation once if it is not a
    /// built-in.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cavitation_nn::{Activation, ActivationKind, ActivationLayer, Layer};
    /// let act: Arc<dyn Activation<f32>> = Arc::new(ActivationKind::Silu);
    /// assert_eq!(ActivationLayer::from_arc(act).name(), "silu");
    /// ```
    pub fn from_arc(act: Arc<dyn Activation<T>>) -> ActivationLayer<T> {
        let kind = act.builtin();
        let op = kind.is_none().then(|| {
            Arc::new(ActivationOp {
                act: Arc::clone(&act),
            }) as Arc<dyn CustomOp<T> + Send + Sync>
        });
        ActivationLayer { act, kind, op }
    }
}

impl<T: Real> fmt::Debug for ActivationLayer<T> {
    /// Shows the activation's name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ActivationLayer({:?})", self.act.name())
    }
}

impl<T: Real> Layer<T> for ActivationLayer<T> {
    /// The activation's name.
    fn name(&self) -> &str {
        self.act.name()
    }

    /// The width is unchanged.
    fn output_width(&self, input_width: usize) -> usize {
        input_width
    }

    /// Records the fused operation for a built-in, the custom operation otherwise.
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        match (&self.kind, &self.op) {
            (Some(kind), _) => kind.record(tape, x),
            (None, Some(op)) => tape.custom(op, &[x]),
            (None, None) => unreachable!("a non-built-in activation always has its operation"),
        }
    }

    /// No parameters.
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        Vec::new()
    }

    /// The wrapped activation.
    fn activation(&self) -> Option<&dyn Activation<T>> {
        Some(&*self.act)
    }

    /// [`InferenceLayer::Activation`] for a built-in activation; `None` for one of your own.
    fn inference(&self, _params: &ParamValues<f32>) -> Option<InferenceLayer> {
        self.kind.map(InferenceLayer::Activation)
    }
}

/// Defines a built-in activation layer: its struct, and its `Activation` and `Layer` impls
/// through the matching [`ActivationKind`].
macro_rules! builtin_layer {
    (
        $(#[$meta:meta])*
        $name:ident, $ctor:expr, |$s:ident| $kind:expr
    ) => {
        $(#[$meta])*
        #[doc = concat!(
            "```\nuse cavitation_core::{Graph, ParamStore, Tensor};\n",
            "use cavitation_nn::{Layer, ", stringify!($name), "};\n",
            "let store = ParamStore::new();\nlet mut graph = Graph::new();\n",
            "let mut tape = graph.tape(&store.values);\n",
            "let x = tape.input(Tensor::full(&[2, 3], 0.5f32).view());\n",
            "let layer = ", stringify!($ctor), ";\n",
            "let y = layer.forward(&mut tape, x);\n",
            "assert_eq!(tape.shape(y).dims(), &[2, 3]);\n```"
        )]
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name;
        builtin_layer!(@impls $name, |$s| $kind);
    };
    (
        $(#[$meta:meta])*
        $name:ident { $($(#[$fmeta:meta])* $field:ident : $ty:ty),+ $(,)? },
        $ctor:expr, |$s:ident| $kind:expr
    ) => {
        $(#[$meta])*
        #[doc = concat!(
            "```\nuse cavitation_core::{Graph, ParamStore, Tensor};\n",
            "use cavitation_nn::{Layer, ", stringify!($name), "};\n",
            "let store = ParamStore::new();\nlet mut graph = Graph::new();\n",
            "let mut tape = graph.tape(&store.values);\n",
            "let x = tape.input(Tensor::full(&[2, 3], 0.5f32).view());\n",
            "let layer = ", stringify!($ctor), ";\n",
            "let y = layer.forward(&mut tape, x);\n",
            "assert_eq!(tape.shape(y).dims(), &[2, 3]);\n```"
        )]
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name {
            $($(#[$fmeta])* pub $field: $ty),+
        }
        builtin_layer!(@impls $name, |$s| $kind);
    };
    (@impls $name:ident, |$s:ident| $kind:expr) => {
        impl $name {
            /// The matching [`ActivationKind`].
            fn kind(&self) -> ActivationKind {
                let $s = self;
                $kind
            }
        }

        impl<T: Real> Activation<T> for $name {
            /// The activation's name.
            fn name(&self) -> &str {
                self.kind().kind_name()
            }

            /// The value at `x`.
            fn f(&self, x: T) -> T {
                self.kind().value(x)
            }

            /// The derivative at `x`.
            fn df(&self, x: T, y: T) -> T {
                self.kind().derivative(x, y)
            }

            /// The matching kind.
            fn builtin(&self) -> Option<ActivationKind> {
                Some(self.kind())
            }
        }

        impl<T: Real> Layer<T> for $name {
            /// The activation's name.
            fn name(&self) -> &str {
                self.kind().kind_name()
            }

            /// The width is unchanged.
            fn output_width(&self, input_width: usize) -> usize {
                input_width
            }

            /// Records the fused tape operation.
            fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
                self.kind().record(tape, x)
            }

            /// No parameters.
            fn parameters(&self) -> Vec<(&'static str, ParamId)> {
                Vec::new()
            }

            /// The layer itself.
            fn activation(&self) -> Option<&dyn Activation<T>> {
                Some(self)
            }

            /// [`InferenceLayer::Activation`] with the matching kind.
            fn inference(&self, _params: &ParamValues<f32>) -> Option<InferenceLayer> {
                Some(InferenceLayer::Activation(self.kind()))
            }
        }
    };
}

builtin_layer!(
    /// ReLU: `max(x, 0)`.
    Relu,
    Relu,
    |_s| ActivationKind::Relu
);
builtin_layer!(
    /// Leaky ReLU: `x` above 0, `slope · x` elsewhere.
    LeakyRelu {
        /// The slope below 0.
        slope: f64
    },
    LeakyRelu { slope: 0.01 },
    |s| ActivationKind::LeakyRelu(s.slope)
);

impl Default for LeakyRelu {
    /// PyTorch's `nn.LeakyReLU()` default: negative slope 0.01.
    ///
    /// ```
    /// use cavitation_nn::LeakyRelu;
    /// assert_eq!(LeakyRelu::default().slope, 0.01);
    /// ```
    fn default() -> LeakyRelu {
        LeakyRelu { slope: 0.01 }
    }
}
builtin_layer!(
    /// The hyperbolic tangent.
    Tanh,
    Tanh,
    |_s| ActivationKind::Tanh
);
builtin_layer!(
    /// The logistic sigmoid.
    Sigmoid,
    Sigmoid,
    |_s| ActivationKind::Sigmoid
);
builtin_layer!(
    /// GELU in the given form.
    Gelu {
        /// Exact (erf) or tanh form.
        approx: GeluApprox
    },
    Gelu {
        approx: cavitation_core::GeluApprox::Exact
    },
    |s| ActivationKind::Gelu(s.approx)
);

impl Default for Gelu {
    /// PyTorch's `nn.GELU()` default: the exact (erf) form, `approximate='none'`.
    ///
    /// ```
    /// use cavitation_nn::Gelu;
    /// assert_eq!(Gelu::default().approx, cavitation_core::GeluApprox::Exact);
    /// ```
    fn default() -> Gelu {
        Gelu {
            approx: cavitation_core::GeluApprox::Exact,
        }
    }
}
builtin_layer!(
    /// ELU: `x` above 0, `alpha · (eˣ - 1)` elsewhere.
    Elu {
        /// The scale below 0.
        alpha: f64
    },
    Elu { alpha: 1.0 },
    |s| ActivationKind::Elu(s.alpha)
);

impl Default for Elu {
    /// PyTorch's `nn.ELU()` default: `alpha` 1.0.
    ///
    /// ```
    /// use cavitation_nn::Elu;
    /// assert_eq!(Elu::default().alpha, 1.0);
    /// ```
    fn default() -> Elu {
        Elu { alpha: 1.0 }
    }
}
builtin_layer!(
    /// SELU, the scaled ELU of self-normalising networks.
    Selu,
    Selu,
    |_s| ActivationKind::Selu
);
builtin_layer!(
    /// CELU: `x` above 0, `alpha · (e^(x/alpha) - 1)` elsewhere.
    Celu {
        /// The scale below 0.
        alpha: f64
    },
    Celu { alpha: 1.0 },
    |s| ActivationKind::Celu(s.alpha)
);

impl Default for Celu {
    /// PyTorch's `nn.CELU()` default: `alpha` 1.0.
    ///
    /// ```
    /// use cavitation_nn::Celu;
    /// assert_eq!(Celu::default().alpha, 1.0);
    /// ```
    fn default() -> Celu {
        Celu { alpha: 1.0 }
    }
}
builtin_layer!(
    /// SiLU (swish): `x · σ(x)`.
    Silu,
    Silu,
    |_s| ActivationKind::Silu
);
builtin_layer!(
    /// Mish: `x · tanh(softplus(x))`.
    Mish,
    Mish,
    |_s| ActivationKind::Mish
);
builtin_layer!(
    /// Softplus: `ln(1 + e^(beta·x)) / beta`, and `x` once `beta · x` passes `threshold`.
    Softplus {
        /// The sharpness.
        beta: f64,
        /// The value of `beta · x` above which the output is `x`.
        threshold: f64
    },
    Softplus {
        beta: 1.0,
        threshold: 20.0
    },
    |s| ActivationKind::Softplus {
        beta: s.beta,
        threshold: s.threshold
    }
);

impl Default for Softplus {
    /// PyTorch's `nn.Softplus()` defaults: `beta` 1 and `threshold` 20.
    ///
    /// ```
    /// use cavitation_nn::Softplus;
    /// let softplus = Softplus::default();
    /// assert_eq!((softplus.beta, softplus.threshold), (1.0, 20.0));
    /// ```
    fn default() -> Softplus {
        Softplus {
            beta: 1.0,
            threshold: 20.0,
        }
    }
}
builtin_layer!(
    /// Softsign: `x / (1 + |x|)`.
    Softsign,
    Softsign,
    |_s| ActivationKind::Softsign
);
builtin_layer!(
    /// Hardtanh: `x` clamped to `[min, max]`.
    Hardtanh {
        /// The lower bound.
        min: f64,
        /// The upper bound.
        max: f64
    },
    Hardtanh {
        min: -1.0,
        max: 1.0
    },
    |s| ActivationKind::Hardtanh {
        min: s.min,
        max: s.max
    }
);

impl Default for Hardtanh {
    /// PyTorch's `nn.Hardtanh()` defaults: `min_val` -1 and `max_val` 1.
    ///
    /// ```
    /// use cavitation_nn::Hardtanh;
    /// let hardtanh = Hardtanh::default();
    /// assert_eq!((hardtanh.min, hardtanh.max), (-1.0, 1.0));
    /// ```
    fn default() -> Hardtanh {
        Hardtanh {
            min: -1.0,
            max: 1.0,
        }
    }
}
builtin_layer!(
    /// ReLU6: `min(max(x, 0), 6)`.
    Relu6,
    Relu6,
    |_s| ActivationKind::Relu6
);
builtin_layer!(
    /// Hard sigmoid: 0 up to -3, 1 from 3, `x/6 + 1/2` between.
    Hardsigmoid,
    Hardsigmoid,
    |_s| ActivationKind::Hardsigmoid
);
builtin_layer!(
    /// Hard swish: `x · hardsigmoid(x)`.
    Hardswish,
    Hardswish,
    |_s| ActivationKind::Hardswish
);
builtin_layer!(
    /// Log-sigmoid: `ln σ(x)`.
    LogSigmoid,
    LogSigmoid,
    |_s| ActivationKind::LogSigmoid
);

/// The last dimension of a shape, or 1 for a scalar.
pub(crate) fn last_dim(shape: Shape) -> usize {
    shape.dims().last().copied().unwrap_or(1)
}

/// Softmax over the last axis. Not an element-wise function, so it has no [`Activation`].
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Tensor};
/// use cavitation_nn::{Layer, Softmax};
/// let store = ParamStore::new();
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&store.values);
/// let x = tape.input(Tensor::from_vec(vec![0.0f32, 0.0], &[1, 2]).unwrap().view());
/// let y = Softmax.forward(&mut tape, x);
/// assert_eq!(tape.value(y).data(), &[0.5, 0.5]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Softmax;

impl<T: Real> Layer<T> for Softmax {
    /// Returns `"softmax"`.
    fn name(&self) -> &str {
        "softmax"
    }

    /// The width is unchanged.
    fn output_width(&self, input_width: usize) -> usize {
        input_width
    }

    /// Records the softmax over the last axis.
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        tape.softmax(x)
    }

    /// No parameters.
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        Vec::new()
    }
    /// [`InferenceLayer::Softmax`].
    fn inference(&self, _params: &ParamValues<f32>) -> Option<InferenceLayer> {
        Some(InferenceLayer::Softmax)
    }
}

/// Log-softmax over the last axis. Not an element-wise function, so it has no [`Activation`].
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Tensor};
/// use cavitation_nn::{Layer, LogSoftmax};
/// let store = ParamStore::new();
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&store.values);
/// let x = tape.input(Tensor::from_vec(vec![1e4f32, 0.0], &[1, 2]).unwrap().view());
/// let y = LogSoftmax.forward(&mut tape, x);
/// assert_eq!(tape.value(y).data(), &[0.0, -1e4]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogSoftmax;

impl<T: Real> Layer<T> for LogSoftmax {
    /// Returns `"log_softmax"`.
    fn name(&self) -> &str {
        "log_softmax"
    }

    /// The width is unchanged.
    fn output_width(&self, input_width: usize) -> usize {
        input_width
    }

    /// Records the log-softmax over the last axis.
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        tape.log_softmax(x)
    }

    /// No parameters.
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        Vec::new()
    }
    /// [`InferenceLayer::LogSoftmax`].
    fn inference(&self, _params: &ParamValues<f32>) -> Option<InferenceLayer> {
        Some(InferenceLayer::LogSoftmax)
    }
}

/// A gated linear unit: splits the last axis in two and returns `a ⊙ gate(b)`, halving the
/// width. With [`Gate::Sigmoid`] it is PyTorch's `nn.GLU`.
///
/// ```
/// use cavitation_core::{Gate, Graph, ParamStore, Tensor};
/// use cavitation_nn::{Glu, Layer};
/// let store = ParamStore::new();
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&store.values);
/// let x = tape.input(Tensor::from_vec(vec![2.0f32, 3.0, -1.0, 4.0], &[1, 4]).unwrap().view());
/// let y = Glu { gate: Gate::Relu }.forward(&mut tape, x);
/// assert_eq!(tape.value(y).data(), &[0.0, 12.0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Glu {
    /// The activation applied to the second half.
    pub gate: Gate,
}

impl<T: Real> Layer<T> for Glu {
    /// Returns `"glu"`.
    fn name(&self) -> &str {
        "glu"
    }

    /// Half the input width.
    fn output_width(&self, input_width: usize) -> usize {
        input_width / 2
    }

    /// Records the gated linear unit.
    ///
    /// # Panics
    ///
    /// Panics if the last axis of `x` is odd.
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        tape.glu(x, self.gate)
    }

    /// No parameters.
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        Vec::new()
    }
    /// [`InferenceLayer::Glu`] with the layer's gate.
    fn inference(&self, _params: &ParamValues<f32>) -> Option<InferenceLayer> {
        Some(InferenceLayer::Glu(self.gate))
    }
}

/// PReLU: `x` above 0, `slope · x` elsewhere, with a learnable slope per channel (or one
/// shared slope).
///
/// The slope is a parameter named `weight`, initialised to 0.25 as in PyTorch, so a PyTorch
/// `nn.PReLU` state dict loads into it.
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Tensor};
/// use cavitation_nn::{Layer, Prelu};
/// let mut params = ParamStore::new();
/// let layer = Prelu::new(&mut params, 2);
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&params.values);
/// let x = tape.input(Tensor::from_vec(vec![-4.0f32, 2.0], &[1, 2]).unwrap().view());
/// let y = layer.forward(&mut tape, x);
/// assert_eq!(tape.value(y).data(), &[-1.0, 2.0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prelu {
    /// The slopes, shape `[channels]`.
    weight: ParamId,
    /// The number of slopes.
    channels: usize,
}

impl Prelu {
    /// Creates a PReLU with `channels` slopes, all 0.25, added to `params`. With
    /// `channels = 1` one slope is shared by every value; otherwise `channels` must equal the
    /// width of the input's last axis.
    ///
    /// # Panics
    ///
    /// Panics if `channels` is 0.
    ///
    /// ```
    /// use cavitation_core::ParamStore;
    /// use cavitation_nn::Prelu;
    /// let mut params = ParamStore::new();
    /// let layer = Prelu::new(&mut params, 3);
    /// assert_eq!(params.value(layer.weight()).data(), &[0.25, 0.25, 0.25]);
    /// ```
    pub fn new<T: Real>(params: &mut ParamStore<T>, channels: usize) -> Prelu {
        assert!(channels > 0, "prelu: channels must be positive");
        let weight = params.add(Tensor::full(&[channels], T::from_f64(0.25)));
        Prelu { weight, channels }
    }

    /// The slopes' id in the store.
    ///
    /// ```
    /// use cavitation_core::ParamStore;
    /// use cavitation_nn::Prelu;
    /// let mut params = ParamStore::new();
    /// let layer = Prelu::new(&mut params, 1);
    /// assert_eq!(params.value(layer.weight()).shape().dims(), &[1]);
    /// ```
    pub fn weight(&self) -> ParamId {
        self.weight
    }
}

impl<T: Real> Layer<T> for Prelu {
    /// Returns `"prelu"`.
    fn name(&self) -> &str {
        "prelu"
    }

    /// The width is unchanged.
    fn output_width(&self, input_width: usize) -> usize {
        input_width
    }

    /// The channel count, when there is one slope per channel.
    fn input_width(&self) -> Option<usize> {
        (self.channels > 1).then_some(self.channels)
    }

    /// Records PReLU with the layer's slopes.
    ///
    /// # Panics
    ///
    /// Panics if the last axis of `x` is not `channels` (and `channels` is not 1).
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        let slope = tape.param(self.weight);
        tape.prelu(x, slope)
    }

    /// 1 where `x` is above 0, the value's slope where `x` is at or below 0, and NaN for a
    /// NaN input, as the tape's PReLU gradient uses.
    fn local_derivative(&self, tape: &Tape<'_, T>, x: Var, out: &mut [T]) -> bool {
        let slopes = tape.param_value(self.weight);
        let slopes = slopes.data();
        for (i, (d, &v)) in out.iter_mut().zip(tape.value(x).data()).enumerate() {
            *d = if v > T::ZERO {
                T::ONE
            } else if v <= T::ZERO {
                slopes[if slopes.len() == 1 {
                    0
                } else {
                    i % slopes.len()
                }]
            } else {
                nan()
            };
        }
        true
    }

    /// Returns `weight`.
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        vec![("weight", self.weight)]
    }

    /// `true`: each output depends only on the input at the same position.
    fn is_elementwise(&self) -> bool {
        true
    }

    /// The layer itself.
    fn prelu_form(&self, _: crate::layer::private::Token) -> Option<&Prelu> {
        Some(self)
    }

    /// `weight`, without allocating.
    fn parameter(&self, local: &str) -> Option<ParamId> {
        (local == "weight").then_some(self.weight)
    }

    /// [`InferenceLayer::Prelu`] with the slopes copied from `params`.
    ///
    /// # Panics
    ///
    /// Panics, naming the shape, if the slopes in `params` are not `[channels]`.
    fn inference(&self, params: &ParamValues<f32>) -> Option<InferenceLayer> {
        let w = params.get(self.weight);
        assert!(
            w.shape().dims() == [self.channels],
            "prelu: expected slopes [{}], got {:?}",
            self.channels,
            w.shape().dims()
        );
        Some(InferenceLayer::Prelu(w.data().to_vec()))
    }
}
