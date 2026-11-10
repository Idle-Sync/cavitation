//! Tests of the activation layers, the non-element-wise layers and `Sequential`.

use std::sync::Arc;

use cavitation_core::{Gate, GeluApprox, Graph, ParamStore, Real, Rng, Tape, Tensor, Var};
use cavitation_nn::{
    Activation, ActivationKind, ActivationLayer, Celu, Elu, Gelu, Glu, Hardsigmoid, Hardswish,
    Hardtanh, Layer, LeakyRelu, Linear, LogSigmoid, LogSoftmax, Mish, Prelu, Relu, Relu6, Selu,
    Sequential, Sigmoid, Silu, Softmax, Softplus, Softsign, Tanh, Trace,
};

/// Every built-in activation with non-default parameters, so that a wrong parameter shows.
fn all_kinds() -> Vec<ActivationKind> {
    vec![
        ActivationKind::Relu,
        ActivationKind::LeakyRelu(0.1),
        ActivationKind::Tanh,
        ActivationKind::Sigmoid,
        ActivationKind::Gelu(GeluApprox::Exact),
        ActivationKind::Gelu(GeluApprox::Tanh),
        ActivationKind::Elu(1.3),
        ActivationKind::Selu,
        ActivationKind::Celu(0.7),
        ActivationKind::Silu,
        ActivationKind::Mish,
        ActivationKind::Softplus {
            beta: 2.0,
            threshold: 20.0,
        },
        ActivationKind::Softsign,
        ActivationKind::Hardtanh {
            min: -0.5,
            max: 0.8,
        },
        ActivationKind::Relu6,
        ActivationKind::Hardsigmoid,
        ActivationKind::Hardswish,
        ActivationKind::LogSigmoid,
    ]
}

/// The named layer struct for a kind.
fn named_layer<T: Real>(kind: ActivationKind) -> Box<dyn Layer<T>> {
    match kind {
        ActivationKind::Relu => Box::new(Relu),
        ActivationKind::LeakyRelu(slope) => Box::new(LeakyRelu { slope }),
        ActivationKind::Tanh => Box::new(Tanh),
        ActivationKind::Sigmoid => Box::new(Sigmoid),
        ActivationKind::Gelu(approx) => Box::new(Gelu { approx }),
        ActivationKind::Elu(alpha) => Box::new(Elu { alpha }),
        ActivationKind::Selu => Box::new(Selu),
        ActivationKind::Celu(alpha) => Box::new(Celu { alpha }),
        ActivationKind::Silu => Box::new(Silu),
        ActivationKind::Mish => Box::new(Mish),
        ActivationKind::Softplus { beta, threshold } => Box::new(Softplus { beta, threshold }),
        ActivationKind::Softsign => Box::new(Softsign),
        ActivationKind::Hardtanh { min, max } => Box::new(Hardtanh { min, max }),
        ActivationKind::Relu6 => Box::new(Relu6),
        ActivationKind::Hardsigmoid => Box::new(Hardsigmoid),
        ActivationKind::Hardswish => Box::new(Hardswish),
        ActivationKind::LogSigmoid => Box::new(LogSigmoid),
    }
}

/// The tape operation for a kind, called directly.
fn tape_op<T: Real>(tape: &mut Tape<'_, T>, kind: ActivationKind, x: Var) -> Var {
    let c = T::from_f64;
    match kind {
        ActivationKind::Relu => tape.relu(x),
        ActivationKind::LeakyRelu(s) => tape.leaky_relu(x, c(s)),
        ActivationKind::Tanh => tape.tanh(x),
        ActivationKind::Sigmoid => tape.sigmoid(x),
        ActivationKind::Gelu(a) => tape.gelu(x, a),
        ActivationKind::Elu(a) => tape.elu(x, c(a)),
        ActivationKind::Selu => tape.selu(x),
        ActivationKind::Celu(a) => tape.celu(x, c(a)),
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

/// A [4, 5] input spread over [-4.5, 4.5] that never lands on a kink.
fn input<T: Real>() -> Tensor<T> {
    let data = (0..20)
        .map(|i| T::from_f64(-4.5 + 9.0 * (i as f64 + 0.37) / 20.0))
        .collect();
    Tensor::from_vec(data, &[4, 5]).unwrap()
}

/// A built-in activation hidden behind a user-style implementation (no `builtin`), so that it
/// runs through the custom-operation path.
struct Hidden(ActivationKind);

impl<T: Real> Activation<T> for Hidden {
    fn name(&self) -> &str {
        "hidden"
    }
    fn f(&self, x: T) -> T {
        Activation::<T>::f(&self.0, x)
    }
    fn df(&self, x: T, y: T) -> T {
        Activation::<T>::df(&self.0, x, y)
    }
}

/// Each `ActivationKind` layer, built as a named struct, as an `ActivationLayer` and through
/// the custom-operation path, gives the tape operation's output and input gradient.
#[test]
fn every_builtin_activation_layer_matches_its_tape_op() {
    for kind in all_kinds() {
        let mut store = ParamStore::<f64>::new_generic();
        let p = store.add(input::<f64>());
        let mut graph = Graph::<f64>::new_generic();

        let mut run = |build: &dyn Fn(&mut Tape<'_, f64>, Var) -> Var| {
            store.zero_grad();
            let mut tape = graph.tape(&store.values);
            let v = tape.param(p);
            let y = build(&mut tape, v);
            let out = tape.value(y).data().to_vec();
            let loss = tape.sum(y);
            tape.backward(loss, &mut store.grads);
            drop(tape);
            (out, store.grad(p).data().to_vec())
        };
        let (want_y, want_g) = run(&|t, v| tape_op(t, kind, v));

        let named = named_layer::<f64>(kind);
        let (y, g) = run(&|t, v| named.forward(t, v));
        assert_eq!((&y, &g), (&want_y, &want_g), "named layer {kind:?}");
        assert_eq!(named.activation().and_then(|a| a.builtin()), Some(kind));

        let layer = ActivationLayer::<f64>::new(kind);
        let (y, g) = run(&|t, v| layer.forward(t, v));
        assert_eq!((&y, &g), (&want_y, &want_g), "ActivationLayer {kind:?}");

        // The custom-operation path agrees to rounding.
        let hidden = ActivationLayer::<f64>::new(Hidden(kind));
        assert_eq!(hidden.activation().unwrap().builtin(), None);
        let (y, g) = run(&|t, v| hidden.forward(t, v));
        for (a, b) in y.iter().zip(&want_y).chain(g.iter().zip(&want_g)) {
            assert!(
                (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
                "custom {kind:?}: {a} vs {b}"
            );
        }
    }
    // The f32 layers also run and agree with the f32 tape operation.
    for kind in all_kinds() {
        let layer = named_layer::<f32>(kind);
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = tape.input(input::<f32>().view());
        let a = layer.forward(&mut tape, x);
        let b = tape_op(&mut tape, kind, x);
        assert_eq!(tape.value(a).data(), tape.value(b).data(), "f32 {kind:?}");
    }
}

/// `f` and `df` of every built-in equal the value and the gradient the tape operation gives.
#[test]
fn builtin_f_and_df_match_the_tape_op() {
    for kind in all_kinds() {
        let mut store = ParamStore::<f64>::new_generic();
        let x = input::<f64>();
        let p = store.add(x.clone());
        let mut graph = Graph::<f64>::new_generic();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p);
        let y = tape_op(&mut tape, kind, v);
        let ys = tape.value(y).data().to_vec();
        let loss = tape.sum(y);
        tape.backward(loss, &mut store.grads);
        drop(tape);
        let g = store.grad(p).data();
        for (i, &xi) in x.data().iter().enumerate() {
            let f = Activation::<f64>::f(&kind, xi);
            let df = Activation::<f64>::df(&kind, xi, f);
            assert!(
                (f - ys[i]).abs() <= 1e-12 * (1.0 + ys[i].abs()),
                "{kind:?} f({xi})"
            );
            assert!(
                (df - g[i]).abs() <= 1e-12 * (1.0 + g[i].abs()),
                "{kind:?} df({xi})"
            );
        }
    }
}

/// Derivatives at the kinks follow the tape operations (and PyTorch).
#[test]
fn builtin_df_at_kinks() {
    let d = |k: ActivationKind, x: f64| Activation::<f64>::df(&k, x, Activation::<f64>::f(&k, x));
    assert_eq!(d(ActivationKind::Relu, 0.0), 0.0);
    assert_eq!(d(ActivationKind::LeakyRelu(0.1), 0.0), 0.1);
    assert_eq!(d(ActivationKind::Relu6, 6.0), 0.0);
    assert_eq!(
        d(
            ActivationKind::Hardtanh {
                min: -1.0,
                max: 1.0
            },
            1.0
        ),
        0.0
    );
    assert!(d(ActivationKind::Relu, f64::NAN).is_nan());
}

/// `Prelu` has one parameter, named `weight`, filled with 0.25, one slope per channel.
#[test]
fn prelu_parameter_named_weight() {
    let mut params = ParamStore::new();
    let layer = Prelu::new(&mut params, 3);
    let parameters = Layer::<f32>::parameters(&layer);
    assert_eq!(parameters.len(), 1);
    assert_eq!(parameters[0].0, "weight");
    assert_eq!(params.value(parameters[0].1).shape().dims(), &[3]);
    assert_eq!(params.value(parameters[0].1).data(), &[0.25, 0.25, 0.25]);
    assert_eq!(Layer::<f32>::name(&layer), "prelu");
    assert!(Layer::<f32>::activation(&layer).is_none());

    let mut graph = Graph::new();
    let mut tape = graph.tape(&params.values);
    let x = Tensor::from_vec(vec![-4.0f32, 2.0, -8.0, 1.0, -1.0, 3.0], &[2, 3]).unwrap();
    let x = tape.input(x.view());
    let y = layer.forward(&mut tape, x);
    assert_eq!(tape.value(y).data(), &[-1.0, 2.0, -2.0, 1.0, -0.25, 3.0]);

    drop(tape);
    let single = Prelu::new(&mut params, 1);
    assert_eq!(
        params
            .value(Layer::<f32>::parameters(&single)[0].1)
            .shape()
            .dims(),
        &[1]
    );
}

/// Softmax, log-softmax, GLU and PReLU are not element-wise, so they have no `Activation`.
#[test]
fn softmax_and_glu_layers_have_no_activation() {
    let mut params = ParamStore::new();
    let layers: Vec<Box<dyn Layer<f32>>> = vec![
        Box::new(Softmax),
        Box::new(LogSoftmax),
        Box::new(Glu {
            gate: Gate::Sigmoid,
        }),
        Box::new(Prelu::new(&mut params, 2)),
    ];
    for layer in &layers {
        assert!(layer.activation().is_none(), "{}", layer.name());
    }
    assert_eq!(layers[2].output_width(6), 3);
    assert_eq!(layers[0].output_width(6), 6);
    let linear: Box<dyn Layer<f32>> = Box::new(Linear::new(&mut params, 2, 2, &mut Rng::seed(0)));
    assert!(linear.activation().is_none());
    // And their outputs are right.
    let store = ParamStore::new();
    let mut graph = Graph::new();
    let mut tape = graph.tape(&store.values);
    let x = tape.input(
        Tensor::from_vec(vec![1.0f32, 1.0, 0.0, 0.0], &[1, 4])
            .unwrap()
            .view(),
    );
    let s = layers[0].forward(&mut tape, x);
    assert_eq!(tape.value(s).data().iter().sum::<f32>(), 1.0);
    let l = layers[1].forward(&mut tape, x);
    // log(e / (2e + 2)) = 1 - ln(2e + 2) for the first value of [1, 1, 0, 0].
    let want = 1.0 - (2.0 * 1.0f64.exp() + 2.0).ln();
    assert!((f64::from(tape.value(l).data()[0]) - want).abs() < 1e-6);
    let g = layers[2].forward(&mut tape, x);
    assert_eq!(tape.shape(g).dims(), &[1, 2]);
    assert!((tape.value(g).data()[0] - 0.5).abs() < 1e-6);
}

/// A user activation's metadata.
#[test]
fn activation_layer_reports_name_and_widths() {
    let layer = ActivationLayer::<f32>::new(ActivationKind::Silu);
    assert_eq!(layer.name(), "silu");
    assert_eq!(layer.output_width(7), 7);
    assert!(layer.parameters().is_empty());
    assert_eq!(
        ActivationLayer::<f32>::new(Hidden(ActivationKind::Silu)).name(),
        "hidden"
    );
    let shared: Arc<dyn Activation<f32>> = Arc::new(Hidden(ActivationKind::Tanh));
    assert_eq!(ActivationLayer::from_arc(shared).name(), "hidden");
}

/// A Linear, Relu, Linear stack names its parameters as PyTorch's `nn.Sequential` does.
#[test]
fn parameter_names_match_pytorch() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(0);
    let net = Sequential::new()
        .push(Linear::new(&mut params, 4, 3, &mut rng))
        .push(Relu)
        .push(Linear::new(&mut params, 3, 2, &mut rng));
    let names: Vec<String> = net.parameter_names().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["0.weight", "0.bias", "2.weight", "2.bias"]);
    assert_eq!(net.len(), 3);
    assert_eq!(net.layer(1).name(), "relu");
    let ids: Vec<_> = net
        .parameter_names()
        .into_iter()
        .map(|(_, id)| id)
        .collect();
    assert_eq!(params.value(ids[2]).shape().dims(), &[2, 3]);

    // A PReLU in the stack is named by its index, like nn.PReLU in nn.Sequential.
    let net = Sequential::new()
        .push(Relu)
        .push(Prelu::new(&mut params, 2));
    assert_eq!(net.parameter_names()[0].0, "1.weight");
}

/// The trace holds every layer's input and output, and keeps its capacity between steps.
#[test]
fn trace_records_every_layer() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(0);
    let net = Sequential::new()
        .push(Linear::new(&mut params, 4, 3, &mut rng))
        .push(Tanh)
        .push(Linear::new(&mut params, 3, 2, &mut rng));
    let mut graph = Graph::new();
    let mut trace = Trace::new();
    let x_data = Tensor::full(&[5, 4], 0.5f32);
    let mut capacity = 0;
    for step in 0..3 {
        let mut tape = graph.tape(&params.values);
        let x = tape.input(x_data.view());
        let y = net.forward_traced(&mut tape, x, &mut trace);
        assert_eq!(trace.len(), 3);
        assert_eq!(trace.input(0), x);
        assert_eq!(trace.output(2), y);
        assert_eq!(trace.input(1), trace.output(0));
        assert_eq!(trace.input(2), trace.output(1));
        assert_eq!(tape.shape(trace.output(0)).dims(), &[5, 3]);
        assert_eq!(tape.shape(trace.output(1)).dims(), &[5, 3]);
        assert_eq!(tape.shape(trace.output(2)).dims(), &[5, 2]);
        // Same result as the untraced pass.
        let plain = net.forward(&mut tape, x);
        assert_eq!(tape.value(plain).data(), tape.value(y).data());
        if step == 0 {
            capacity = trace.capacity();
            assert!(capacity >= 3);
        } else {
            assert_eq!(trace.capacity(), capacity);
        }
    }
    assert!(Trace::default().is_empty());
}

/// A width that does not chain panics naming the layer index, the layer and both widths.
#[test]
#[should_panic(
    expected = "sequential: layer 1 (linear) expects input width 5, but layer 0 (linear) outputs width 3"
)]
fn width_mismatch_names_the_layer() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(0);
    let net = Sequential::new()
        .push(Linear::new(&mut params, 2, 3, &mut rng))
        .push(Linear::new(&mut params, 5, 2, &mut rng));
    let mut graph = Graph::new();
    let mut tape = graph.tape(&params.values);
    let x = tape.input(Tensor::full(&[1, 2], 1.0f32).view());
    net.forward(&mut tape, x);
}

/// The first layer's check names the input.
#[test]
#[should_panic(
    expected = "sequential: layer 0 (linear) expects input width 2, but the input has width 3"
)]
fn first_layer_width_mismatch_names_the_input() {
    let mut params = ParamStore::new();
    let net = Sequential::new().push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)));
    let mut graph = Graph::new();
    let mut tape = graph.tape(&params.values);
    let x = tape.input(Tensor::full(&[1, 3], 1.0f32).view());
    net.forward(&mut tape, x);
}

/// An empty Sequential passes its input through.
#[test]
fn empty_sequential_is_identity() {
    let net = Sequential::new();
    assert!(net.is_empty());
    let store = ParamStore::new();
    let mut graph = Graph::new();
    let mut tape = graph.tape(&store.values);
    let x = tape.input(Tensor::full(&[1, 3], 1.0f32).view());
    assert_eq!(net.forward(&mut tape, x), x);
}

/// Asking for a layer past the end panics with the index and the length.
#[test]
#[should_panic(expected = "sequential: layer index 2 out of range for 1 layers")]
fn layer_out_of_range_panics() {
    Sequential::new().push(Relu).layer(2);
}
