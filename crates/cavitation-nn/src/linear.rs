//! `Linear`: a fully connected layer.

use cavitation_core::{ParamId, ParamStore, ParamValues, Real, Rng, Tape, Tensor, Var};

use crate::{InferenceLayer, Init, Layer};

/// A fully connected layer: `y = x · Wᵀ + b`.
///
/// The weight has shape `[outputs, inputs]` (the same layout as PyTorch's `nn.Linear`, so
/// its `state_dict` loads by name) and the bias `[outputs]`. The layer holds only the
/// [`ParamId`]s of its weight and bias; the values live in the [`ParamStore`] it was created
/// from. Its local parameter names are `weight` and `bias`.
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
    /// The bias, `[outputs]`.
    bias: ParamId,
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
    /// let layer = Linear::with_init(&mut params, 4, 2, Init::HeNormal, Init::Constant(0.1), &mut Rng::seed(0));
    /// assert_eq!(params.value(layer.bias()).data(), &[0.1, 0.1]);
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
            bias: params.add(b),
            inputs,
            outputs,
        }
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

    /// The bias's id in the store.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Rng};
    /// use cavitation_nn::Linear;
    /// let mut params = ParamStore::new();
    /// let layer = Linear::new(&mut params, 2, 3, &mut Rng::seed(0));
    /// assert_eq!(params.value(layer.bias()).shape().dims(), &[3]);
    /// ```
    pub fn bias(&self) -> ParamId {
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

    /// Records `x · Wᵀ + b` for `x` of shape `[batch, inputs]`.
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
        let b = tape.param(self.bias);
        let y = tape.matmul_nt(x, w);
        tape.add(y, b)
    }

    /// Returns the number of inputs.
    fn input_width(&self) -> Option<usize> {
        Some(self.inputs)
    }

    /// Returns `weight` and `bias`.
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        vec![("weight", self.weight), ("bias", self.bias)]
    }

    /// [`InferenceLayer::Linear`], with the weight copied transposed to
    /// `[inputs, outputs]`.
    ///
    /// # Panics
    ///
    /// Panics, naming the shapes, if the weight in `params` is not `[outputs, inputs]` or
    /// the bias not `[outputs]`.
    fn inference(&self, params: &ParamValues<f32>) -> Option<InferenceLayer> {
        let (w, b) = (params.get(self.weight), params.get(self.bias));
        let (inputs, outputs) = (self.inputs, self.outputs);
        assert!(
            w.shape().dims() == [outputs, inputs] && b.shape().dims() == [outputs],
            "linear: expected weight [{outputs}, {inputs}] and bias [{outputs}], got {:?} and {:?}",
            w.shape().dims(),
            b.shape().dims()
        );
        let w = w.data();
        let mut w_t = vec![0.0f32; inputs * outputs];
        for (i, row) in w_t.chunks_exact_mut(outputs).enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = w[j * inputs + i];
            }
        }
        Some(InferenceLayer::Linear {
            w_t,
            bias: b.data().to_vec(),
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
        assert_eq!(params.value(layer.bias()).shape().dims(), &[3]);
        assert!(params.value(layer.bias()).data().iter().all(|&b| b == 0.0));
        assert_eq!((layer.inputs(), layer.outputs()), (2, 3));
        assert_eq!(Layer::<f32>::name(&layer), "linear");
        assert_eq!(Layer::<f32>::output_width(&layer, 2), 3);
        assert_eq!(
            Layer::<f32>::parameters(&layer),
            vec![("weight", layer.weight()), ("bias", layer.bias())]
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
            .value_mut(layer.bias())
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

    /// The layer can sit behind a trait object.
    #[test]
    fn layer_is_object_safe() {
        let mut params = ParamStore::new();
        let boxed: Box<dyn Layer<f32>> =
            Box::new(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)));
        assert_eq!(boxed.output_width(2), 3);
    }
}
