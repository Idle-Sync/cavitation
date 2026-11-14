//! `Linear`: a fully connected layer.

use cavitation_core::{Error, ParamId, ParamStore, ParamValues, Real, Rng, Tape, Tensor, Var};

use crate::{InferenceLayer, Init, Layer};

/// A fully connected layer: `y = x · Wᵀ + b`.
///
/// The weight has shape `[outputs, inputs]` (the same layout as PyTorch's `nn.Linear`, so
/// its `state_dict` loads by name) and the bias `[outputs]`. The layer holds only the
/// [`ParamId`]s of its weight and bias; the values live in the [`ParamStore`] it was created
/// from. Its local parameter names are `weight` and `bias`.
///
/// A layer made with [`without_bias`](Linear::without_bias) or
/// [`from_weight`](Linear::from_weight) has no bias, like `nn.Linear(inputs, outputs,
/// bias=False)`: it computes `y = x · Wᵀ`, its only parameter is `weight`, and
/// [`bias`](Linear::bias) returns `None`.
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Rng, Tensor};
/// use cavitation_nn::{Layer, Linear};
/// let mut params = ParamStore::new();
/// let layer = Linear::new(&mut params, 2, 3, &mut Rng::seed(0));
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&params.values);
/// let x = tape.input(Tensor::from_vec(vec![1.0f32, 2.0], &[1, 2]).unwrap().view());
/// let y = layer.forward(&mut tape, x);
/// assert_eq!(tape.shape(y).dims(), &[1, 3]);
/// ```
#[derive(Clone, Debug)]
pub struct Linear {
    /// The weight, `[outputs, inputs]`.
    weight: ParamId,
    /// The bias, `[outputs]`, or `None` for a layer without one.
    bias: Option<ParamId>,
    /// The number of input columns.
    inputs: usize,
    /// The number of output columns.
    outputs: usize,
}

impl Linear {
    /// Creates a layer with Xavier uniform weights and a zero bias, adding both to `params`.
    ///
    /// # Panics
    ///
    /// Panics if `inputs` or `outputs` is 0.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::Linear;
    /// let mut params = ParamStore::new();
    /// let layer = Linear::new(&mut params, 784, 128, &mut Rng::seed(0));
    /// assert_eq!(params.value(layer.weight()).shape().dims(), &[128, 784]);
    /// assert_eq!(params.len(), 2);
    /// ```
    pub fn new<T: Real>(
        params: &mut ParamStore<T>,
        inputs: usize,
        outputs: usize,
        rng: &mut Rng,
    ) -> Linear {
        Linear::with_init(
            params,
            inputs,
            outputs,
            Init::XavierUniform,
            Init::Zeros,
            rng,
        )
    }

    /// Creates a layer whose weight and bias are filled by the given initialisers (the weight
    /// first, then the bias, drawing from `rng` in that order).
    ///
    /// # Panics
    ///
    /// Panics if `inputs` or `outputs` is 0.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Init, Linear};
    /// let mut params = ParamStore::new();
    /// let layer = Linear::with_init(
    ///     &mut params,
    ///     4,
    ///     2,
    ///     Init::HeNormal,
    ///     Init::Constant(0.1),
    ///     &mut Rng::seed(0),
    /// );
    /// assert_eq!(params.value(layer.bias().unwrap()).data(), &[0.1, 0.1]);
    /// ```
    pub fn with_init<T: Real>(
        params: &mut ParamStore<T>,
        inputs: usize,
        outputs: usize,
        weight: Init,
        bias: Init,
        rng: &mut Rng,
    ) -> Linear {
        assert!(
            inputs > 0 && outputs > 0,
            "linear: inputs ({inputs}) and outputs ({outputs}) must be positive"
        );
        let mut w = Tensor::<T>::zeros_generic(&[outputs, inputs]);
        weight.fill(&mut w, inputs, outputs, rng);
        let mut b = Tensor::<T>::zeros_generic(&[outputs]);
        bias.fill(&mut b, inputs, outputs, rng);
        Linear {
            weight: params.add(w),
            bias: Some(params.add(b)),
            inputs,
            outputs,
        }
    }

    /// Creates a layer with no bias, `y = x · Wᵀ`, whose weight is filled by `weight` from
    /// `rng` (the same draws as [`with_init`](Linear::with_init) makes for the weight), and
    /// adds only the weight to `params`: the layer of PyTorch's `nn.Linear(inputs, outputs,
    /// bias=False)`.
    ///
    /// # Panics
    ///
    /// Panics if `inputs` or `outputs` is 0.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Init, Layer, Linear};
    /// let mut params = ParamStore::new();
    /// let layer = Linear::without_bias(&mut params, 4, 2, Init::XavierUniform, &mut Rng::seed(0));
    /// assert_eq!(layer.bias(), None);
    /// assert_eq!(params.len(), 1);
    /// assert_eq!(Layer::<f32>::parameters(&layer), [("weight", layer.weight())]);
    /// ```
    pub fn without_bias<T: Real>(
        params: &mut ParamStore<T>,
        inputs: usize,
        outputs: usize,
        weight: Init,
        rng: &mut Rng,
    ) -> Linear {
        assert!(
            inputs > 0 && outputs > 0,
            "linear: inputs ({inputs}) and outputs ({outputs}) must be positive"
        );
        let mut w = Tensor::<T>::zeros_generic(&[outputs, inputs]);
        weight.fill(&mut w, inputs, outputs, rng);
        Linear {
            weight: params.add(w),
            bias: None,
            inputs,
            outputs,
        }
    }

    /// Creates a layer from a weight of shape `[outputs, inputs]` and a bias of shape
    /// `[outputs]`, adding both to `params`: for weights trained elsewhere.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`], naming both shapes, if the weight is not a matrix with
    /// positive sides or the bias is not `[outputs]`. Nothing is added to `params` then.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::Linear;
    /// let mut params = ParamStore::new();
    /// let weight = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    /// let bias = Tensor::from_vec(vec![0.5f32, 0.0, -0.5], &[3]).unwrap();
    /// let layer = Linear::from_values(&mut params, weight, bias).unwrap();
    /// assert_eq!((layer.inputs(), layer.outputs()), (2, 3));
    /// assert_eq!(params.value(layer.bias().unwrap()).data(), &[0.5, 0.0, -0.5]);
    /// ```
    pub fn from_values<T: Real>(
        params: &mut ParamStore<T>,
        weight: Tensor<T>,
        bias: Tensor<T>,
    ) -> Result<Linear, Error> {
        Linear::from_parts(params, weight, Some(bias))
    }

    /// Creates a layer with no bias from a weight of shape `[outputs, inputs]`, adding only
    /// the weight to `params`: for the weights of an `nn.Linear(inputs, outputs, bias=False)`
    /// trained elsewhere.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`], naming the shape, if the weight is not a matrix with positive
    /// sides. Nothing is added to `params` then.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::Linear;
    /// let mut params = ParamStore::new();
    /// let weight = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    /// let layer = Linear::from_weight(&mut params, weight).unwrap();
    /// assert_eq!((layer.inputs(), layer.outputs()), (2, 3));
    /// assert_eq!(layer.bias(), None);
    /// ```
    pub fn from_weight<T: Real>(
        params: &mut ParamStore<T>,
        weight: Tensor<T>,
    ) -> Result<Linear, Error> {
        Linear::from_parts(params, weight, None)
    }

    /// Creates a layer from a weight of shape `[outputs, inputs]` and, if given, a bias of
    /// shape `[outputs]`, adding them to `params`; the shared body of
    /// [`from_values`](Linear::from_values) and [`from_weight`](Linear::from_weight).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`], naming the shapes, if the weight is not a matrix with positive
    /// sides or the bias is not `[outputs]`. Nothing is added to `params` then.
    pub(crate) fn from_parts<T: Real>(
        params: &mut ParamStore<T>,
        weight: Tensor<T>,
        bias: Option<Tensor<T>>,
    ) -> Result<Linear, Error> {
        let w = weight.shape().dims();
        let matrix = matches!(w, [o, i] if *o > 0 && *i > 0);
        match &bias {
            Some(bias) => {
                let b = bias.shape().dims();
                if !(matrix && b == [w[0]]) {
                    return Err(Error::InvalidData(format!(
                        "linear: expected weight [outputs, inputs] and bias [outputs], got weight {w:?} and bias {b:?}"
                    )));
                }
            }
            None if !matrix => {
                return Err(Error::InvalidData(format!(
                    "linear: expected weight [outputs, inputs], got weight {w:?}"
                )));
            }
            None => {}
        }
        let (outputs, inputs) = (w[0], w[1]);
        Ok(Linear {
            weight: params.add(weight),
            bias: bias.map(|b| params.add(b)),
            inputs,
            outputs,
        })
    }

    /// The weight's id in the store.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::Linear;
    /// let mut params = ParamStore::new();
    /// let layer = Linear::new(&mut params, 2, 3, &mut Rng::seed(0));
    /// assert_eq!(params.value(layer.weight()).shape().dims(), &[3, 2]);
    /// ```
    pub fn weight(&self) -> ParamId {
        self.weight
    }

    /// The bias's id in the store, or `None` for a layer without a bias (one made with
    /// [`without_bias`](Linear::without_bias) or [`from_weight`](Linear::from_weight)).
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::{Init, Linear};
    /// let mut params = ParamStore::new();
    /// let layer = Linear::new(&mut params, 2, 3, &mut Rng::seed(0));
    /// assert_eq!(params.value(layer.bias().unwrap()).shape().dims(), &[3]);
    /// let bare = Linear::without_bias(&mut params, 2, 3, Init::XavierUniform, &mut Rng::seed(0));
    /// assert_eq!(bare.bias(), None);
    /// ```
    pub fn bias(&self) -> Option<ParamId> {
        self.bias
    }

    /// The number of input columns the layer expects.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::Linear;
    /// let layer = Linear::new(&mut ParamStore::new(), 2, 3, &mut Rng::seed(0));
    /// assert_eq!(layer.inputs(), 2);
    /// ```
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    /// The number of output columns the layer produces.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::Linear;
    /// let layer = Linear::new(&mut ParamStore::new(), 2, 3, &mut Rng::seed(0));
    /// assert_eq!(layer.outputs(), 3);
    /// ```
    pub fn outputs(&self) -> usize {
        self.outputs
    }
}

impl<T: Real> Layer<T> for Linear {
    /// Returns `"linear"`.
    fn name(&self) -> &str {
        "linear"
    }

    /// Returns the layer's output width, whatever `input_width` is: a mismatch is reported by
    /// `forward`, where the shape is known.
    fn output_width(&self, _input_width: usize) -> usize {
        self.outputs
    }

    /// Records `x · Wᵀ + b` for `x` of shape `[batch, inputs]`, or `x · Wᵀ` without a bias.
    ///
    /// # Panics
    ///
    /// Panics if the last dimension of `x` is not `inputs`.
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        let shape = tape.shape(x);
        assert!(
            shape.dims().last() == Some(&self.inputs),
            "linear: expected inputs of width {}, got shape {:?}",
            self.inputs,
            shape.dims()
        );
        let w = tape.param(self.weight);
        let y = tape.matmul_nt(x, w);
        match self.bias {
            Some(bias) => {
                let b = tape.param(bias);
                tape.add(y, b)
            }
            None => y,
        }
    }

    /// `weight` or, if the layer has one, `bias`, without allocating.
    fn parameter(&self, local: &str) -> Option<ParamId> {
        match local {
            "weight" => Some(self.weight),
            "bias" => self.bias,
            _ => None,
        }
    }

    /// The layer itself.
    fn linear_form(&self, _: crate::layer::private::Token) -> Option<&Linear> {
        Some(self)
    }

    /// Returns the number of inputs.
    fn input_width(&self) -> Option<usize> {
        Some(self.inputs)
    }

    /// Returns `weight` and `bias`, or only `weight` for a layer without a bias.
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        match self.bias {
            Some(bias) => vec![("weight", self.weight), ("bias", bias)],
            None => vec![("weight", self.weight)],
        }
    }

    /// [`InferenceLayer::Linear`], with the weight copied transposed to
    /// `[inputs, outputs]` and the bias, if the layer has one, copied as it is.
    ///
    /// # Panics
    ///
    /// Panics, naming the shapes, if the weight in `params` is not `[outputs, inputs]` or
    /// the bias not `[outputs]`.
    fn inference(&self, params: &ParamValues<f32>) -> Option<InferenceLayer> {
        let w = params.get(self.weight);
        let (inputs, outputs) = (self.inputs, self.outputs);
        assert!(
            w.shape().dims() == [outputs, inputs],
            "linear: expected weight [{outputs}, {inputs}], got {:?}",
            w.shape().dims()
        );
        let bias = self.bias.map(|id| {
            let b = params.get(id);
            assert!(
                b.shape().dims() == [outputs],
                "linear: expected bias [{outputs}], got {:?}",
                b.shape().dims()
            );
            b.data().to_vec()
        });
        let w = w.data();
        let mut w_t = vec![0.0f32; inputs * outputs];
        for (i, row) in w_t.chunks_exact_mut(outputs).enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = w[j * inputs + i];
            }
        }
        Some(InferenceLayer::Linear {
            w_t,
            bias,
            inputs,
            outputs,
        })
    }
}

/// Tests of the layer.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Layer;
    use cavitation_core::{Graph, ParamStore, Rng, Tensor};

    /// Weight is `[outputs, inputs]`, bias is `[outputs]`.
    #[test]
    fn linear_shapes() {
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 2, 3, &mut Rng::seed(0));
        assert_eq!(params.value(layer.weight()).shape().dims(), &[3, 2]);
        let bias = layer.bias().unwrap();
        assert_eq!(params.value(bias).shape().dims(), &[3]);
        assert!(params.value(bias).data().iter().all(|&b| b == 0.0));
        assert_eq!((layer.inputs(), layer.outputs()), (2, 3));
        assert_eq!(Layer::<f32>::name(&layer), "linear");
        assert_eq!(Layer::<f32>::output_width(&layer, 2), 3);
        assert_eq!(
            Layer::<f32>::parameters(&layer),
            vec![("weight", layer.weight()), ("bias", bias)]
        );
    }

    /// `x · wᵀ + b` for known numbers.
    #[test]
    fn linear_forward_by_hand() {
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 2, 3, &mut Rng::seed(0));
        params
            .value_mut(layer.weight())
            .data_mut()
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        params
            .value_mut(layer.bias().unwrap())
            .data_mut()
            .copy_from_slice(&[0.5, 0.0, -0.5]);
        let x = Tensor::from_vec(vec![1.0f32, 1.0], &[1, 2]).unwrap();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let xv = tape.input(x.view());
        let y = layer.forward(&mut tape, xv);
        assert_eq!(tape.shape(y).dims(), &[1, 3]);
        assert_eq!(tape.value(y).data(), &[3.5, 7.0, 10.5]);
    }

    /// A wrong input width panics naming the layer, the expected width and the shape.
    #[test]
    #[should_panic(expected = "linear: expected inputs of width 2, got shape [1, 3]")]
    fn wrong_width_panics() {
        let mut params = ParamStore::new();
        let layer = Linear::new(&mut params, 2, 3, &mut Rng::seed(0));
        let x = Tensor::from_vec(vec![1.0f32; 3], &[1, 3]).unwrap();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let xv = tape.input(x.view());
        layer.forward(&mut tape, xv);
    }

    /// `from_values` takes `[outputs, inputs]` and `[outputs]`, and names both shapes
    /// otherwise.
    #[test]
    fn from_values_checks_shapes() {
        let mut params = ParamStore::new();
        let w = Tensor::from_vec(vec![1.0f32; 6], &[3, 2]).unwrap();
        let b = Tensor::from_vec(vec![0.0f32; 3], &[3]).unwrap();
        let layer = Linear::from_values(&mut params, w.clone(), b.clone()).unwrap();
        assert_eq!((layer.inputs(), layer.outputs()), (2, 3));
        assert_eq!(params.value(layer.weight()).data(), w.data());
        let short = Tensor::from_vec(vec![0.0f32; 2], &[2]).unwrap();
        let err = Linear::from_values(&mut params, w.clone(), short).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("[3, 2]") && msg.contains("[2]"), "{msg}");
        let flat = Tensor::from_vec(vec![1.0f32; 6], &[6]).unwrap();
        assert!(Linear::from_values(&mut params, flat, b).is_err());
        assert_eq!(params.len(), 2);
    }

    /// A layer built without a bias has only a weight: one entry in the store, one in
    /// `parameters`, no `bias` by name, and its forward pass is `x · Wᵀ` alone.
    #[test]
    fn without_bias_has_only_a_weight() {
        let mut params = ParamStore::new();
        let layer = Linear::without_bias(&mut params, 2, 3, Init::XavierUniform, &mut Rng::seed(0));
        assert_eq!(params.len(), 1);
        assert_eq!(layer.bias(), None);
        assert_eq!((layer.inputs(), layer.outputs()), (2, 3));
        assert_eq!(params.value(layer.weight()).shape().dims(), &[3, 2]);
        assert_eq!(
            Layer::<f32>::parameters(&layer),
            vec![("weight", layer.weight())]
        );
        assert_eq!(Layer::<f32>::parameter(&layer, "bias"), None);
        assert_eq!(
            Layer::<f32>::parameter(&layer, "weight"),
            Some(layer.weight())
        );
        params
            .value_mut(layer.weight())
            .data_mut()
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let x = Tensor::from_vec(vec![1.0f32, 1.0], &[1, 2]).unwrap();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let xv = tape.input(x.view());
        let y = layer.forward(&mut tape, xv);
        assert_eq!(tape.value(y).data(), &[3.0, 7.0, 11.0]);
    }

    /// `without_bias` draws the weight from `rng` exactly as `with_init` does, so the two
    /// weights agree for the same seed and initialiser.
    #[test]
    fn without_bias_draws_the_weight_as_with_init_does() {
        let mut a = ParamStore::new();
        let with = Linear::with_init(&mut a, 4, 3, Init::HeNormal, Init::Zeros, &mut Rng::seed(5));
        let mut b = ParamStore::new();
        let without = Linear::without_bias(&mut b, 4, 3, Init::HeNormal, &mut Rng::seed(5));
        assert_eq!(
            a.value(with.weight()).data(),
            b.value(without.weight()).data()
        );
    }

    /// `from_weight` takes `[outputs, inputs]`, adds only the weight, and names the shape
    /// otherwise.
    #[test]
    fn from_weight_checks_the_shape() {
        let mut params = ParamStore::new();
        let w = Tensor::from_vec(vec![1.0f32; 6], &[3, 2]).unwrap();
        let layer = Linear::from_weight(&mut params, w.clone()).unwrap();
        assert_eq!((layer.inputs(), layer.outputs()), (2, 3));
        assert_eq!(layer.bias(), None);
        assert_eq!(params.len(), 1);
        let flat = Tensor::from_vec(vec![1.0f32; 6], &[6]).unwrap();
        let err = Linear::from_weight(&mut params, flat).unwrap_err();
        assert!(err.to_string().contains("[6]"), "{err}");
        let empty = Tensor::from_vec(Vec::<f32>::new(), &[0, 2]).unwrap();
        assert!(Linear::from_weight(&mut params, empty).is_err());
        assert_eq!(params.len(), 1);
    }

    /// The inference form of a layer without a bias carries no bias.
    #[test]
    fn inference_form_without_bias() {
        let mut params = ParamStore::new();
        let w = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let layer = Linear::from_weight(&mut params, w).unwrap();
        match Layer::<f32>::inference(&layer, &params.values) {
            Some(InferenceLayer::Linear {
                w_t,
                bias,
                inputs,
                outputs,
            }) => {
                assert_eq!((inputs, outputs), (2, 3));
                assert_eq!(w_t, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
                assert_eq!(bias, None);
            }
            other => panic!("expected a Linear form, got {other:?}"),
        }
    }

    /// The layer can sit behind a trait object.
    #[test]
    fn layer_is_object_safe() {
        let mut params = ParamStore::new();
        let boxed: Box<dyn Layer<f32>> =
            Box::new(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)));
        assert_eq!(boxed.output_width(2), 3);
    }
}
