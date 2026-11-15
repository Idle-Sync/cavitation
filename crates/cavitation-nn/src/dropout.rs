//! The `Dropout`, `Flatten` and `Identity` layers: layers that change nothing at inference.

use cavitation_core::{Error, ParamId, ParamValues, Real, Tape, Var};

use crate::{InferenceLayer, Layer};

/// Inverted dropout, PyTorch's `nn.Dropout`: while training, each value is kept with
/// probability `1 - p` and scaled by `1 / (1 - p)`, the others set to 0; in evaluation (and at
/// inference) it passes its input through.
///
/// The masks are drawn on [training tapes](cavitation_core::Graph::training_tape) from the
/// run's dropout seed, the step and the order of the dropout layers on the tape
/// ([`Tape::next_stream`]), so the same run gives the same masks on every machine and for any
/// thread count. PyTorch's random numbers cannot be reproduced, so a run here and in PyTorch
/// drop different values.
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Tensor};
/// use cavitation_nn::{Dropout, Layer};
/// let store = ParamStore::new();
/// let dropout = Dropout::new(0.5).unwrap();
/// let mut graph = Graph::new();
/// let mut tape = graph.training_tape(&store.values, 0, 1);
/// let x = tape.input(Tensor::full(&[1, 8], 1.0f32).view());
/// let y = dropout.forward(&mut tape, x);
/// assert!(tape.value(y).data().iter().all(|&v| v == 0.0 || v == 2.0));
/// drop(tape);
/// let mut tape = graph.tape(&store.values);
/// let x = tape.input(Tensor::full(&[1, 8], 1.0f32).view());
/// assert_eq!(dropout.forward(&mut tape, x), x);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dropout {
    /// The probability of dropping each value, in `[0, 1)`.
    p: f64,
}

impl Dropout {
    /// A dropout layer that drops each value with probability `p`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] unless `p` is in `[0, 1)`. PyTorch also accepts `p = 1`, which
    /// sets every value and every gradient to 0; it is refused here as almost certainly a
    /// mistake.
    ///
    /// ```
    /// use cavitation_nn::Dropout;
    /// assert_eq!(Dropout::new(0.25).unwrap().p(), 0.25);
    /// assert!(Dropout::new(1.0).is_err());
    /// ```
    pub fn new(p: f64) -> Result<Dropout, Error> {
        if (0.0..1.0).contains(&p) {
            Ok(Dropout { p })
        } else {
            Err(Error::InvalidData(format!(
                "dropout: the drop probability must be in [0, 1), got {p}"
            )))
        }
    }

    /// The probability of dropping each value.
    ///
    /// ```
    /// use cavitation_nn::Dropout;
    /// assert_eq!(Dropout::new(0.1).unwrap().p(), 0.1);
    /// ```
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl<T: Real> Layer<T> for Dropout {
    /// Returns `"dropout"`.
    fn name(&self) -> &str {
        "dropout"
    }

    /// The width is unchanged.
    fn output_width(&self, input_width: usize) -> usize {
        input_width
    }

    /// Records dropout with the tape's next stream: a random mask on a training tape, the
    /// input itself on an evaluation tape.
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        let stream = tape.next_stream();
        tape.dropout(x, self.p, stream)
    }

    /// No parameters.
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        Vec::new()
    }

    /// Each output value depends only on the input value at the same position.
    fn is_elementwise(&self) -> bool {
        true
    }

    /// [`InferenceLayer::Identity`]: dropout does nothing at inference.
    fn inference(&self, _params: &ParamValues<f32>) -> Option<InferenceLayer> {
        Some(InferenceLayer::Identity)
    }

    /// The drop probability `p`.
    fn dropout_form(&self, _: crate::layer::private::Token) -> Option<f64> {
        Some(self.p)
    }
}

/// Defines a layer that passes its input through unchanged, named `$text`.
macro_rules! pass_through {
    ($(#[$doc:meta])* $name:ident, $text:literal) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct $name;

        impl<T: Real> Layer<T> for $name {
            #[doc = concat!("Returns `\"", $text, "\"`.")]
            fn name(&self) -> &str {
                $text
            }

            /// The width is unchanged.
            fn output_width(&self, input_width: usize) -> usize {
                input_width
            }

            /// Returns `x` itself, recording nothing.
            fn forward(&self, _tape: &mut Tape<'_, T>, x: Var) -> Var {
                x
            }

            /// No parameters.
            fn parameters(&self) -> Vec<(&'static str, ParamId)> {
                Vec::new()
            }

            /// Each output value is the input value at the same position.
            fn is_elementwise(&self) -> bool {
                true
            }

            /// [`InferenceLayer::Identity`].
            fn inference(&self, _params: &ParamValues<f32>) -> Option<InferenceLayer> {
                Some(InferenceLayer::Identity)
            }
        }
    };
}

pass_through!(
    /// PyTorch's `nn.Flatten`: rows are already flat here (an image is its channels one after
    /// another, each row by row, as `torch.flatten(x, 1)` lays it out), so it passes its input
    /// through. It exists so that a PyTorch `nn.Sequential` with an `nn.Flatten()` keeps its
    /// layer numbers, which its state dict's keys use.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// use cavitation_nn::{Flatten, Layer};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = tape.input(Tensor::full(&[2, 12], 1.0f32).view());
    /// assert_eq!(Flatten.forward(&mut tape, x), x);
    /// ```
    Flatten,
    "flatten"
);

pass_through!(
    /// PyTorch's `nn.Identity`: passes its input through. Useful as a placeholder that keeps
    /// a network's layer numbers, which its state dict's keys use.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// use cavitation_nn::{Identity, Layer};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = tape.input(Tensor::full(&[1, 3], 2.0f32).view());
    /// assert_eq!(Identity.forward(&mut tape, x), x);
    /// ```
    Identity,
    "identity"
);

/// Tests for the pass-through layers and dropout as a layer.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InferenceModel, Linear, Relu, Sequential};
    use cavitation_core::{Graph, ParamStore, Rng, Tensor};

    /// Probabilities outside [0, 1) and NaN are refused, naming the value.
    #[test]
    fn dropout_rejects_bad_probabilities() {
        for p in [-0.5, 1.0, 2.0, f64::NAN] {
            let e = Dropout::new(p).unwrap_err().to_string();
            assert!(e.contains("[0, 1)"), "{e}");
        }
    }

    /// Two dropout layers in one network draw different masks, and the same step of the same
    /// run draws the same masks again, whatever the hooks or masks recorded before them.
    #[test]
    fn dropout_layers_draw_their_own_masks() {
        let store = ParamStore::new();
        let d = Dropout::new(0.5).unwrap();
        let x = Tensor::full(&[1, 64], 1.0f32);
        let mut graph = Graph::new();
        let mut masks = |extra_nodes: usize| {
            let mut tape = graph.training_tape(&store.values, 9, 3);
            let mut v = tape.input(x.view());
            for _ in 0..extra_nodes {
                v = tape.relu(v);
            }
            let a = Layer::<f32>::forward(&d, &mut tape, v);
            let b = Layer::<f32>::forward(&d, &mut tape, v);
            (tape.value(a).data().to_vec(), tape.value(b).data().to_vec())
        };
        let (a, b) = masks(0);
        assert_ne!(a, b);
        assert_eq!(masks(3), (a, b));
    }

    /// `Dropout` answers `dropout_form` with its probability and each other built-in layer
    /// listed here with `None`, so a device executor finds the dropout layers among those
    /// that compile to `Identity` for inference, as `Flatten` and `Identity` also do.
    #[test]
    fn only_dropout_answers_dropout_form() {
        use crate::layer::private::Token;
        use crate::*;
        use cavitation_core::{Conv2dGeom, Gate, Pool2dGeom};
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(0);
        let pool = Pool2dGeom::new(2, (4, 4), (2, 2)).unwrap();
        let conv = Conv2dGeom::new(2, 2, (4, 4), (3, 3)).unwrap();
        let others: Vec<Box<dyn Layer<f32>>> = vec![
            Box::new(Linear::new(&mut params, 3, 2, &mut rng)),
            Box::new(Conv2d::new(&mut params, conv, &mut rng)),
            Box::new(BatchNorm2d::new(&mut params, 2, (4, 4))),
            Box::new(MaxPool2d::new(pool)),
            Box::new(AvgPool2d::new(pool)),
            Box::new(AdaptiveAvgPool2d::new(2, (4, 4), (2, 2)).unwrap()),
            Box::new(Prelu::new(&mut params, 2)),
            Box::new(Glu {
                gate: Gate::Sigmoid,
            }),
            Box::new(Softmax),
            Box::new(LogSoftmax),
            Box::new(ActivationLayer::new(Relu)),
            Box::new(Relu),
            Box::new(Relu6),
            Box::new(LeakyRelu::default()),
            Box::new(Elu::default()),
            Box::new(Celu::default()),
            Box::new(Selu),
            Box::new(Gelu::default()),
            Box::new(Silu),
            Box::new(Mish),
            Box::new(Softplus::default()),
            Box::new(Softsign),
            Box::new(Hardtanh::default()),
            Box::new(Hardsigmoid),
            Box::new(Hardswish),
            Box::new(LogSigmoid),
            Box::new(Tanh),
            Box::new(Sigmoid),
            Box::new(Flatten),
            Box::new(Identity),
        ];
        for layer in &others {
            assert_eq!(layer.dropout_form(Token(())), None, "{}", layer.name());
        }
        for p in [0.0, 0.2, 0.5] {
            let d: Box<dyn Layer<f32>> = Box::new(Dropout::new(p).unwrap());
            assert_eq!(d.dropout_form(Token(())), Some(p));
        }
    }

    /// Dropout, Flatten and Identity compile to steps that change nothing: a network with
    /// them gives the bits of the same network without them, one input and a batch, and the
    /// observed path reports each of their layers.
    #[test]
    fn pass_through_layers_change_nothing_at_inference() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(4);
        let a = Linear::new(&mut params, 6, 5, &mut rng);
        let b = Linear::new(&mut params, 5, 3, &mut rng);
        let plain = Sequential::new().push(a.clone()).push(Relu).push(b.clone());
        let with = Sequential::new()
            .push(Identity)
            .push(a)
            .push(Dropout::new(0.3).unwrap())
            .push(Relu)
            .push(Flatten)
            .push(b);
        let mut plain = InferenceModel::compile(&plain, &params).unwrap();
        let mut with = InferenceModel::compile(&with, &params).unwrap();
        let mut rng = Rng::seed(5);
        let rows: Vec<f32> = (0..24).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        let x = Tensor::from_vec(rows, &[4, 6]).unwrap();
        for row in x.data().chunks(6) {
            assert_eq!(plain.infer(row), with.infer(row));
        }
        assert_eq!(
            plain.infer_batch(x.view()).data(),
            with.infer_batch(x.view()).data()
        );
        let mut seen = Vec::new();
        with.infer_batch_observed(x.view(), &mut |layer, _| seen.push(layer));
        assert_eq!(seen, [0, 1, 2, 3, 4, 5]);
    }

    /// The pass-through layers keep a PyTorch `nn.Sequential`'s numbering: the `Linear`s
    /// after them are named by their own positions.
    #[test]
    fn pass_through_layers_keep_parameter_names() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(0);
        let net = Sequential::new()
            .push(Linear::new(&mut params, 4, 4, &mut rng))
            .push(Identity)
            .push(Relu)
            .push(Flatten)
            .push(Linear::new(&mut params, 4, 2, &mut rng));
        let names: Vec<String> = net.parameter_names().into_iter().map(|n| n.0).collect();
        assert_eq!(names, ["0.weight", "0.bias", "4.weight", "4.bias"]);
    }
}
