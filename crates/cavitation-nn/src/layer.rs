//! The `Layer` trait.

use cavitation_core::{ParamId, ParamValues, Real, Tape, Var};

use crate::{Activation, InferenceLayer, Linear, Prelu};

/// Keeps the layer downcasts closed: only this crate can name `Token`, so only this crate's
/// `Linear` and `Prelu` can override [`Layer::linear_form`] and [`Layer::prelu_form`].
pub(crate) mod private {
    /// The argument of the sealed [`Layer`](super::Layer) hooks; only this crate can make one.
    #[derive(Clone, Copy, Debug)]
    pub struct Token(pub(crate) ());
}

/// One stage of a network: it records its forward pass on a tape and names its parameters.
///
/// The trait is object safe, so a network can hold its layers as `Box<dyn Layer<T>>` or an `Arc<dyn
/// Layer<T>>`. Layers keep only [`ParamId`]s, never values: the values live in the
/// [`ParamStore`](cavitation_core::ParamStore).
///
/// ```
/// use cavitation_core::{ParamStore, Rng};
/// use cavitation_nn::{Layer, Linear};
/// let mut params = ParamStore::new();
/// let layers: Vec<Box<dyn Layer<f32>>> = vec![
///     Box::new(Linear::new(&mut params, 4, 3, &mut Rng::seed(0))),
///     Box::new(Linear::new(&mut params, 3, 2, &mut Rng::seed(1))),
/// ];
/// let width = layers.iter().fold(4, |w, l| l.output_width(w));
/// assert_eq!(width, 2);
/// ```
///
/// A layer of your own implements the trait for every [`Real`] type, so that the same code
/// trains in `f32` and runs through [`gradcheck_params`](cavitation_core::gradcheck_params)
/// in `f64`. This one scales `tanh(x)` by a learnable `alpha`:
///
/// ```
/// use cavitation_core::{ParamId, ParamStore, Real, Rng, Tape, Tensor, Var, gradcheck_params};
/// use cavitation_nn::{Layer, Linear, Sequential};
///
/// struct ScaledTanh {
///     alpha: ParamId,
/// }
///
/// impl<T: Real> Layer<T> for ScaledTanh {
///     fn name(&self) -> &str {
///         "scaled_tanh"
///     }
///     fn output_width(&self, input_width: usize) -> usize {
///         input_width
///     }
///     fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
///         let t = tape.tanh(x);
///         let alpha = tape.param(self.alpha);
///         tape.mul(t, alpha)
///     }
///     fn parameters(&self) -> Vec<(&'static str, ParamId)> {
///         vec![("alpha", self.alpha)]
///     }
/// }
///
/// let mut params = ParamStore::<f64>::new_generic();
/// let alpha = params.add(Tensor::from_vec(vec![1.5], &[1]).unwrap());
/// let net = Sequential::new_generic()
///     .push(Linear::new(&mut params, 3, 2, &mut Rng::seed(0)))
///     .push(ScaledTanh { alpha });
/// let x = Tensor::from_vec(vec![0.1, -0.4, 0.8], &[1, 3]).unwrap();
/// gradcheck_params(&mut params, &[x], |tape, v| net.forward(tape, v[0])).assert_passed();
/// ```
pub trait Layer<T: Real>: Send + Sync {
    /// The layer's kind, for example `"linear"`.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Layer, Linear};
    /// let layer = Linear::new(&mut ParamStore::new(), 2, 2, &mut Rng::seed(0));
    /// assert_eq!(Layer::<f32>::name(&layer), "linear");
    /// ```
    fn name(&self) -> &str;

    /// The width of the layer's output for an input of `input_width` columns.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Layer, Linear};
    /// let layer = Linear::new(&mut ParamStore::new(), 2, 5, &mut Rng::seed(0));
    /// assert_eq!(Layer::<f32>::output_width(&layer, 2), 5);
    /// ```
    fn output_width(&self, input_width: usize) -> usize;

    /// Records the layer's forward pass on `tape` for the input `x` and returns its output.
    ///
    /// # Panics
    ///
    /// Panics if the shape of `x` does not suit the layer; the message names the layer, the
    /// width it expects and the shape it got.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Rng, Tensor};
    /// use cavitation_nn::{Layer, Linear};
    /// let mut params = ParamStore::new();
    /// let layer = Linear::new(&mut params, 2, 3, &mut Rng::seed(0));
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&params.values);
    /// let x = tape.input(Tensor::full(&[4, 2], 1.0f32).view());
    /// let y = layer.forward(&mut tape, x);
    /// assert_eq!(tape.shape(y).dims(), &[4, 3]);
    /// ```
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var;

    /// The layer's parameters with their local names (for example `"weight"` and `"bias"`),
    /// in a fixed order. Meant for saving and listing, not for the training loop.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Layer, Linear};
    /// let layer = Linear::new(&mut ParamStore::new(), 2, 3, &mut Rng::seed(0));
    /// let names: Vec<_> = Layer::<f32>::parameters(&layer).iter().map(|p| p.0).collect();
    /// assert_eq!(names, ["weight", "bias"]);
    /// ```
    fn parameters(&self) -> Vec<(&'static str, ParamId)>;

    /// The parameter with local name `local` (for example `"weight"`), if the layer has one.
    /// The default searches [`parameters`](Self::parameters), which allocates; `Linear` and
    /// `Prelu` answer without allocating.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Layer, Linear};
    /// let layer = Linear::new(&mut ParamStore::new(), 2, 3, &mut Rng::seed(0));
    /// assert_eq!(Layer::<f32>::parameter(&layer, "bias"), layer.bias());
    /// assert_eq!(Layer::<f32>::parameter(&layer, "scale"), None);
    /// ```
    fn parameter(&self, local: &str) -> Option<ParamId> {
        self.parameters()
            .into_iter()
            .find(|(name, _)| *name == local)
            .map(|(_, id)| id)
    }

    /// Whether each output value depends only on the input value at the same position, so
    /// that a neuron mask on this layer's output switches off exactly the neuron that feeds
    /// it. [`Sequential::neuron_layer`](crate::Sequential::neuron_layer) puts a `Linear`'s
    /// neuron mask on the last layer of the run of element-wise layers after it. The
    /// default is whether the layer has an [`activation`](Self::activation); `Prelu`
    /// overrides it to `true`. A layer of your own that is element-wise but has no
    /// `activation()` should override this, so that neuron masks land after it.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Layer, Linear, Prelu, Relu};
    /// let mut params = ParamStore::new();
    /// assert!(Layer::<f32>::is_elementwise(&Relu));
    /// assert!(Layer::<f32>::is_elementwise(&Prelu::new(&mut params, 1)));
    /// let linear = Linear::new(&mut params, 2, 2, &mut Rng::seed(0));
    /// assert!(!Layer::<f32>::is_elementwise(&linear));
    /// ```
    fn is_elementwise(&self) -> bool {
        self.activation().is_some()
    }

    /// This layer as a [`Linear`], if it is one: the sealed hook behind `as_linear` on
    /// `dyn Layer`. Only this crate can name
    /// [`private::Token`], so only its own `Linear` can answer `Some`, and code that relies on
    /// the answer (compiling for inference, compaction) is never handed a layer that only
    /// claims to be a `Linear`.
    #[doc(hidden)]
    fn linear_form(&self, _: private::Token) -> Option<&Linear> {
        None
    }

    /// This layer as a [`Prelu`], if it is one: the sealed hook behind `as_prelu` on
    /// `dyn Layer`, closed to other crates as
    /// [`linear_form`](Self::linear_form) is.
    #[doc(hidden)]
    fn prelu_form(&self, _: private::Token) -> Option<&Prelu> {
        None
    }

    /// The width of input the layer requires, if it requires one. [`Sequential`](crate::Sequential)
    /// checks it against the width of the previous layer's output. The default is `None`
    /// (the layer accepts any width, or checks it itself).
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Layer, Linear, Relu};
    /// let linear = Linear::new(&mut ParamStore::new(), 4, 2, &mut Rng::seed(0));
    /// assert_eq!(Layer::<f32>::input_width(&linear), Some(4));
    /// assert_eq!(Layer::<f32>::input_width(&Relu), None);
    /// ```
    fn input_width(&self) -> Option<usize> {
        None
    }

    /// The element-wise [`Activation`] this layer applies, if it is one. Layers that are not
    /// an element-wise function of one value (`Linear`, `Softmax`, `Glu`, `Prelu`) return
    /// `None`, the default.
    ///
    /// ```
    /// use cavitation_nn::{Activation, ActivationKind, Layer, Softmax, Tanh};
    /// let act = Layer::<f32>::activation(&Tanh).unwrap();
    /// assert_eq!(act.builtin(), Some(ActivationKind::Tanh));
    /// assert!(Layer::<f32>::activation(&Softmax).is_none());
    /// ```
    fn activation(&self) -> Option<&dyn Activation<T>> {
        None
    }

    /// Writes the layer's local derivative at the input `x` into `out` and returns `true`,
    /// for a layer whose every output value depends only on the input value at the same
    /// position: `out[i]` is the derivative of output `i` with respect to input `i`.
    /// [`Hooks`](crate::Hooks) show it as [`LayerEvent::local_derivative`](crate::LayerEvent).
    ///
    /// The default computes `df(x, f(x))` with the layer's [`activation`](Self::activation)
    /// when it has one, and otherwise writes nothing and returns `false` (a layer such as
    /// `Linear` or `Softmax`, whose derivative is not element-wise). `out` holds as many
    /// values as `x`.
    ///
    /// # Panics
    ///
    /// May panic if `x` was not recorded on `tape`.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// use cavitation_nn::{Layer, Linear, Relu};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = tape.input(Tensor::from_vec(vec![-1.0f32, 2.0], &[1, 2]).unwrap().view());
    /// let mut out = [9.0f32; 2];
    /// assert!(Layer::<f32>::local_derivative(&Relu, &tape, x, &mut out));
    /// assert_eq!(out, [0.0, 1.0]);
    /// ```
    fn local_derivative(&self, tape: &Tape<'_, T>, x: Var, out: &mut [T]) -> bool {
        let Some(act) = self.activation() else {
            return false;
        };
        for (d, &v) in out.iter_mut().zip(tape.value(x).data()) {
            *d = act.df(v, act.f(v));
        }
        true
    }

    /// The layer's form for [`InferenceModel`](crate::InferenceModel), with its parameters
    /// read from `params` (the `f32` values of the store the layer was created with), or
    /// `None` if it has none. Every built-in layer has one; the default is `None`, so a
    /// model with a layer of your own cannot be compiled for inference unless that layer
    /// describes itself as one of the [`InferenceLayer`] forms.
    ///
    /// # Panics
    ///
    /// A layer may panic if its parameters in `params` do not have the shapes it created
    /// them with (`Linear` does).
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{InferenceLayer, Layer, Linear};
    /// let mut params = ParamStore::new();
    /// let layer = Linear::new(&mut params, 4, 2, &mut Rng::seed(0));
    /// match Layer::<f32>::inference(&layer, &params.values) {
    ///     Some(InferenceLayer::Linear { inputs, outputs, .. }) => {
    ///         assert_eq!((inputs, outputs), (4, 2))
    ///     }
    ///     _ => unreachable!(),
    /// }
    /// ```
    fn inference(&self, params: &ParamValues<f32>) -> Option<InferenceLayer> {
        let _ = params;
        None
    }
}

/// Adds `as_linear` and `as_prelu` to one form of `dyn Layer` (the bare trait object and
/// the ones with `Send`, `Sync` or both spelled out), each calling the sealed hooks.
macro_rules! downcasts {
    ($($bounds:tt)*) => {
        impl<'a, T: Real> dyn Layer<T> $($bounds)* + 'a {
            /// This layer as a [`Linear`], if it is one. Only the crate's own `Linear`
            /// answers `Some`: a layer of your own cannot claim to be one. It works on
            /// `dyn Layer<T>` and on the same object with `+ Send`, `+ Sync` or both.
            ///
            /// ```
            /// use cavitation_core::{ParamStore, Rng};
            /// use cavitation_nn::{Layer, Linear, Relu};
            /// let layer = Linear::new(&mut ParamStore::new(), 2, 3, &mut Rng::seed(0));
            /// let layers: [&dyn Layer<f32>; 2] = [&layer, &Relu];
            /// assert_eq!(layers[0].as_linear().unwrap().outputs(), 3);
            /// assert!(layers[1].as_linear().is_none());
            /// let boxed: Box<dyn Layer<f32> + Send> = Box::new(Relu);
            /// assert!(boxed.as_linear().is_none());
            /// ```
            pub fn as_linear(&self) -> Option<&Linear> {
                self.linear_form(private::Token(()))
            }

            /// This layer as a [`Prelu`], if it is one. Only the crate's own `Prelu`
            /// answers `Some`; [`shrink::compact`](crate::shrink::compact) uses it to find
            /// the slopes it rebuilds. It works on `dyn Layer<T>` and on the same object
            /// with `+ Send`, `+ Sync` or both.
            ///
            /// ```
            /// use cavitation_core::ParamStore;
            /// use cavitation_nn::{Layer, Prelu, Relu};
            /// let layer = Prelu::new(&mut ParamStore::new(), 3);
            /// let layers: [&dyn Layer<f32>; 2] = [&layer, &Relu];
            /// assert!(layers[0].as_prelu().is_some());
            /// assert!(layers[1].as_prelu().is_none());
            /// ```
            ///
            /// A layer of your own cannot answer it, so it cannot pass for a `Prelu`:
            ///
            /// ```compile_fail,E0407
            /// use cavitation_core::{ParamId, Real, Tape, Var};
            /// use cavitation_nn::{Layer, Prelu};
            /// struct Fake(Prelu);
            /// impl<T: Real> Layer<T> for Fake {
            ///     fn name(&self) -> &str { "fake" }
            ///     fn output_width(&self, width: usize) -> usize { width }
            ///     fn forward(&self, _tape: &mut Tape<'_, T>, x: Var) -> Var { x }
            ///     fn parameters(&self) -> Vec<(&'static str, ParamId)> { Vec::new() }
            ///     fn as_prelu(&self) -> Option<&Prelu> { Some(&self.0) }
            /// }
            /// ```
            pub fn as_prelu(&self) -> Option<&Prelu> {
                self.prelu_form(private::Token(()))
            }
        }
    };
}

downcasts!();
downcasts!(+ Send);
downcasts!(+ Sync);
downcasts!(+ Send + Sync);
