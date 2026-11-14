//! Gradient checks of the layers, in `f64`.

use std::sync::Arc;

use cavitation_core::{
    CustomOp, Gate, GeluApprox, ParamId, ParamStore, Real, Rng, Shape, Tape, Tensor, TensorView,
    TensorViewMut, Var, gradcheck_params,
};
use cavitation_nn::{
    Activation, ActivationLayer, Gelu, Glu, Init, Layer, Linear, LogSoftmax, NeuronMask, Prelu,
    Sequential, Sigmoid, Softmax, Tanh,
};

/// A `Linear` layer's forward pass has the right gradient for its input, weight and bias.
#[test]
fn gradcheck_linear() {
    let mut params = ParamStore::<f64>::new_generic();
    let mut rng = Rng::seed(11);
    // Non-zero biases, so that the bias gradient is exercised at a generic point.
    let layer = Linear::with_init(
        &mut params,
        3,
        5,
        Init::XavierNormal,
        Init::XavierUniform,
        &mut rng,
    );
    let data = (0..12).map(|i| ((i as f64) * 0.7).sin()).collect();
    let x = Tensor::from_vec(data, &[4, 3]).unwrap();
    let report = gradcheck_params(&mut params, &[x], |tape, v| layer.forward(tape, v[0]));
    report.assert_passed();
}

/// A `Linear` without a bias has the right gradient for its input and weight, alone and
/// inside a `Sequential` before a `Tanh` and a `Linear` that has a bias.
#[test]
fn gradcheck_linear_without_bias() {
    let mut params = ParamStore::<f64>::new_generic();
    let mut rng = Rng::seed(12);
    let layer = Linear::without_bias(&mut params, 3, 5, Init::XavierNormal, &mut rng);
    assert_eq!(params.len(), 1);
    let data = (0..12).map(|i| ((i as f64) * 0.7).sin()).collect();
    let x = Tensor::from_vec(data, &[4, 3]).unwrap();
    gradcheck_params(&mut params, std::slice::from_ref(&x), |tape, v| {
        layer.forward(tape, v[0])
    })
    .assert_passed();

    let mut params = ParamStore::<f64>::new_generic();
    let net = Sequential::new_generic()
        .push(Linear::without_bias(
            &mut params,
            3,
            4,
            Init::XavierNormal,
            &mut rng,
        ))
        .push(Tanh)
        .push(Linear::with_init(
            &mut params,
            4,
            2,
            Init::XavierNormal,
            Init::XavierUniform,
            &mut rng,
        ));
    let names: Vec<String> = net.parameter_names().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["0.weight", "2.weight", "2.bias"]);
    gradcheck_params(&mut params, &[x], |tape, v| net.forward(tape, v[0])).assert_passed();
}

/// A user-defined layer: `y = alpha · tanh(x)` with one learnable `alpha`.
struct ScaledTanh {
    /// The scale, shape `[1]`.
    alpha: ParamId,
}

impl<T: Real> Layer<T> for ScaledTanh {
    fn name(&self) -> &str {
        "scaled_tanh"
    }
    fn output_width(&self, input_width: usize) -> usize {
        input_width
    }
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        let t = tape.tanh(x);
        let a = tape.param(self.alpha);
        tape.mul(t, a)
    }
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        vec![("alpha", self.alpha)]
    }
}

/// A user layer with a parameter passes the gradient check inside a `Sequential`.
#[test]
fn user_layer_gradcheck() {
    let mut params = ParamStore::<f64>::new_generic();
    let mut rng = Rng::seed(3);
    let alpha = params.add(Tensor::from_vec(vec![1.5], &[1]).unwrap());
    let net = Sequential::new_generic()
        .push(Linear::new(&mut params, 3, 4, &mut rng))
        .push(ScaledTanh { alpha })
        .push(Linear::new(&mut params, 4, 2, &mut rng));
    assert_eq!(net.parameter_names()[2].0, "1.alpha");
    let x = Tensor::from_vec((0..6).map(|i| (i as f64 * 0.9).cos()).collect(), &[2, 3]).unwrap();
    gradcheck_params(&mut params, &[x], |tape, v| net.forward(tape, v[0])).assert_passed();
}

/// The Huber loss with `delta = 1`, summed, as a user-defined operation on (prediction, target).
struct Huber;

impl<T: Real> CustomOp<T> for Huber {
    fn name(&self) -> &str {
        "huber"
    }
    fn output_shape(&self, inputs: &[Shape]) -> Shape {
        assert_eq!(inputs[0], inputs[1], "huber: shapes differ");
        Shape::new(&[1])
    }
    fn forward(&self, inputs: &[TensorView<T>], mut output: TensorViewMut<T>) {
        let half = T::from_f64(0.5);
        let mut total = T::ZERO;
        for (&p, &t) in inputs[0].data().iter().zip(inputs[1].data()) {
            let d = (p - t).abs();
            total += if d <= T::ONE { half * d * d } else { d - half };
        }
        output.data_mut()[0] = total;
    }
    fn backward(
        &self,
        inputs: &[TensorView<T>],
        _output: TensorView<T>,
        grad_output: TensorView<T>,
        grad_inputs: &mut [TensorViewMut<T>],
    ) {
        let g = grad_output.data()[0];
        let (p, t) = (inputs[0].data(), inputs[1].data());
        let [gp, gt] = grad_inputs else {
            unreachable!()
        };
        for i in 0..p.len() {
            let d = (p[i] - t[i]).max(-T::ONE).min(T::ONE);
            gp.data_mut()[i] = g * d;
            gt.data_mut()[i] = -(g * d);
        }
    }
}

/// A user-written Huber loss passes the gradient check for both inputs, in the quadratic and
/// the linear region.
#[test]
fn user_loss_gradcheck() {
    let op: Arc<dyn CustomOp<f64> + Send + Sync> = Arc::new(Huber);
    let mut params = ParamStore::<f64>::new_generic();
    let pred = Tensor::from_vec(vec![0.3, -2.5, 1.1, 4.0, -0.2, 0.9], &[2, 3]).unwrap();
    let target = Tensor::from_vec(vec![0.1, 0.4, -0.6, 3.5, 0.5, -1.3], &[2, 3]).unwrap();
    gradcheck_params(&mut params, &[pred, target], |tape, v| {
        tape.custom(&op, &[v[0], v[1]])
    })
    .assert_passed();
}

/// A softplus written through `Activation` passes the gradient check inside a network.
#[test]
fn user_activation_gradcheck() {
    /// `ln(1 + eˣ)`.
    struct MySoftplus;
    impl<T: Real> Activation<T> for MySoftplus {
        fn name(&self) -> &str {
            "my_softplus"
        }
        fn f(&self, x: T) -> T {
            x.exp().ln_1p()
        }
        fn df(&self, x: T, _y: T) -> T {
            T::ONE / (T::ONE + (-x).exp())
        }
    }
    let mut params = ParamStore::<f64>::new_generic();
    let mut rng = Rng::seed(5);
    let net = Sequential::new_generic()
        .push(Linear::new(&mut params, 3, 4, &mut rng))
        .push(ActivationLayer::<f64>::new(MySoftplus))
        .push(Linear::new(&mut params, 4, 2, &mut rng));
    let x = Tensor::from_vec((0..6).map(|i| (i as f64 * 0.8).sin()).collect(), &[2, 3]).unwrap();
    gradcheck_params(&mut params, &[x], |tape, v| net.forward(tape, v[0])).assert_passed();
}

/// A 2 -> 4 -> 3 network with tanh passes the gradient check for every parameter and the input.
#[test]
fn gradcheck_three_layer_sequential() {
    let mut params = ParamStore::<f64>::new_generic();
    let mut rng = Rng::seed(7);
    let net = Sequential::new_generic()
        .push(Linear::with_init(
            &mut params,
            2,
            4,
            Init::XavierNormal,
            Init::XavierUniform,
            &mut rng,
        ))
        .push(Tanh)
        .push(Linear::with_init(
            &mut params,
            4,
            3,
            Init::XavierNormal,
            Init::XavierUniform,
            &mut rng,
        ));
    let x = Tensor::from_vec(vec![0.3, -0.7, 1.1, 0.2, -0.4, 0.9], &[3, 2]).unwrap();
    gradcheck_params(&mut params, &[x], |tape, v| net.forward(tape, v[0])).assert_passed();
}

/// A 3 -> 5 -> 2 tanh network with two hidden neurons and one output neuron masked off
/// passes the gradient check for every parameter and the input.
#[test]
fn gradcheck_masked_tanh_network() {
    let mut params = ParamStore::<f64>::new_generic();
    let mut rng = Rng::seed(13);
    let mut net = Sequential::new_generic()
        .push(Linear::with_init(
            &mut params,
            3,
            5,
            Init::XavierNormal,
            Init::XavierUniform,
            &mut rng,
        ))
        .push(Tanh)
        .push(Linear::with_init(
            &mut params,
            5,
            2,
            Init::XavierNormal,
            Init::XavierUniform,
            &mut rng,
        ))
        .push(Tanh);
    let mut hidden = NeuronMask::all_on(5);
    hidden.set(0, false);
    hidden.set(3, false);
    net.set_mask(1, hidden);
    let mut out = NeuronMask::all_on(2);
    out.set(1, false);
    net.set_mask(3, out);
    let x = Tensor::from_vec(vec![0.3, -0.7, 1.1, 0.2, -0.4, 0.9], &[2, 3]).unwrap();
    gradcheck_params(&mut params, &[x], |tape, v| net.forward(tape, v[0])).assert_passed();
}

/// Sigmoid, GELU, GLU, softmax and log-softmax layers compose and pass the gradient check.
#[test]
fn gradcheck_mixed_layers() {
    let mut params = ParamStore::<f64>::new_generic();
    let mut rng = Rng::seed(9);
    let net = Sequential::new_generic()
        .push(Linear::new(&mut params, 3, 6, &mut rng))
        .push(Glu {
            gate: Gate::Sigmoid,
        })
        .push(Linear::new(&mut params, 3, 4, &mut rng))
        .push(Gelu {
            approx: GeluApprox::Exact,
        })
        .push(Linear::new(&mut params, 4, 4, &mut rng))
        .push(Sigmoid)
        .push(Softmax);
    let x = Tensor::from_vec((0..6).map(|i| (i as f64 * 1.3).sin()).collect(), &[2, 3]).unwrap();
    gradcheck_params(&mut params, std::slice::from_ref(&x), |tape, v| {
        net.forward(tape, v[0])
    })
    .assert_passed();

    let log_net = Sequential::new_generic()
        .push(Linear::new(&mut params, 3, 4, &mut rng))
        .push(LogSoftmax);
    gradcheck_params(&mut params, &[x], |tape, v| log_net.forward(tape, v[0])).assert_passed();
}

/// A PReLU with inputs away from zero passes the gradient check for the input and the slope.
#[test]
fn gradcheck_prelu_layer() {
    let mut params = ParamStore::<f64>::new_generic();
    let layer = Prelu::new(&mut params, 3);
    let x = Tensor::from_vec(vec![-1.2, 0.8, -0.5, 2.0, -3.1, 0.6], &[2, 3]).unwrap();
    gradcheck_params(&mut params, &[x], |tape, v| layer.forward(tape, v[0])).assert_passed();
}
