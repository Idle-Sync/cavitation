//! The evaluation-mode forward pass against the CPU's `f64` tape on a masked MLP and the
//! `mnist-cnn` and `cifar-cnn` networks, two ways:
//!
//! - layer by layer: each layer's GPU output against the `f64` tape's layer applied to the
//!   GPU's own input for it, within that kernel's bound alone. This is the sharp check: a
//!   wrong formula, shape or parameter shows as a ratio far above 1.
//! - end to end: the network's GPU output against the `f64` tape run from the input, within a
//!   bound composed through the layers (each layer's error carried through it, plus its own).
//!   Carried through four convolution and batch-norm blocks, the worst case grows by orders of
//!   magnitude, so on the CNNs this bound is loose; the printed ratio and the bound's width
//!   say how loose.
//!
//! An MLP with a softmax or a log-softmax after each `Linear`, and one with a PReLU before each
//! `Linear`, are checked against the CPU's `InferenceModel` instead, within 2⁻¹⁰ of its
//! largest output.
//!
//! Without a GPU every test says so and checks nothing; `CAVITATION_GPU_REQUIRED=1` makes
//! that a failure.

use cavitation_core::{
    Conv2dGeom, Graph, ParamId, ParamStore, Pool2dGeom, Real, Rng, Tape, Tensor, Var,
};
use cavitation_gpu::{Backend, Gpu};

use super::GpuModel;
use crate::{
    ActivationKind, ActivationLayer, BatchNorm2d, Conv2d, Dropout, Flatten, InferenceLayer,
    InferenceModel, Init, Linear, LogSoftmax, MaxPool2d, NeuronMask, Prelu, Relu, Sequential,
    Softmax, Tanh, Trace,
};

/// The unit roundoff of `f32`.
const U: f64 = 1.0 / (1u64 << 24) as f64;

/// The smallest normal `f32`: the most a flushed subnormal can move a value, per rounding.
const FLOOR: f64 = 1.175_494_350_822_287_5e-38;

/// Runs `body` on each GPU this platform has, one per backend, and returns how many ran.
fn for_each_gpu(mut body: impl FnMut(&Gpu)) -> usize {
    let mut ran = 0;
    for &backend in Backend::on_this_platform() {
        if let Some(gpu) = Gpu::with_backend(backend) {
            println!("on {backend}: {}", gpu.info().name);
            body(&gpu);
            ran += 1;
        }
    }
    if ran == 0 {
        println!("no GPU: skipped");
    }
    ran
}

/// `n` values in `[lo, hi)` from `seed`.
fn uniform(seed: u64, n: usize, lo: f64, hi: f64) -> Vec<f32> {
    let mut rng = Rng::seed(seed);
    (0..n).map(|_| rng.range(lo, hi) as f32).collect()
}

/// The test MLP, 12 → 16 → 9 → 4 with tanh and ReLU, in either precision.
pub(super) fn mlp<T: Real>(params: &mut ParamStore<T>, seed: u64) -> Sequential<T> {
    let mut rng = Rng::seed(seed);
    Sequential::new_generic()
        .push(Linear::new(params, 12, 16, &mut rng))
        .push(Tanh)
        .push(Linear::new(params, 16, 9, &mut rng))
        .push(Relu)
        .push(Linear::new(params, 9, 4, &mut rng))
}

/// An MLP 12 → 16 → 9 → 4 with a softmax, or a log-softmax if `log`, after each `Linear`, in
/// either precision.
pub(super) fn softmax_mlp<T: Real>(
    params: &mut ParamStore<T>,
    seed: u64,
    log: bool,
) -> Sequential<T> {
    let mut rng = Rng::seed(seed);
    let rows = |net: Sequential<T>| {
        if log {
            net.push(LogSoftmax)
        } else {
            net.push(Softmax)
        }
    };
    let net = rows(Sequential::new_generic().push(Linear::new(params, 12, 16, &mut rng)));
    let net = rows(net.push(Linear::new(params, 16, 9, &mut rng)));
    rows(net.push(Linear::new(params, 9, 4, &mut rng)))
}

/// An MLP 12 → 16 → 9 → 4 with a PReLU before each `Linear`, one slope per column, or one
/// shared slope if `shared`, in either precision; the slopes are seeded in `[−0.5, 0.75)`.
pub(super) fn prelu_mlp<T: Real>(
    params: &mut ParamStore<T>,
    seed: u64,
    shared: bool,
) -> Sequential<T> {
    let mut rng = Rng::seed(seed);
    let mut prelu = |params: &mut ParamStore<T>, width: usize| {
        let layer = Prelu::new(params, if shared { 1 } else { width });
        for v in params.value_mut(layer.weight()).data_mut() {
            *v = T::from_f64(rng.range(-0.5, 0.75));
        }
        layer
    };
    let first = prelu(params, 12);
    let second = prelu(params, 16);
    let third = prelu(params, 9);
    let mut rng = Rng::seed(seed + 1);
    Sequential::new_generic()
        .push(first)
        .push(Linear::new(params, 12, 16, &mut rng))
        .push(second)
        .push(Linear::new(params, 16, 9, &mut rng))
        .push(third)
        .push(Linear::new(params, 9, 4, &mut rng))
}

/// The activations the GPU engines run besides ReLU, leaky ReLU, tanh, the sigmoid and GELU,
/// each with settings that take both of its branches on the test data: softplus with a low
/// threshold, hardtanh with bounds inside the values the first `Linear` gives.
pub(super) fn more_activations() -> [ActivationKind; 12] {
    [
        ActivationKind::Elu(0.75),
        ActivationKind::Selu,
        ActivationKind::Celu(1.5),
        ActivationKind::Silu,
        ActivationKind::Mish,
        ActivationKind::Softplus {
            beta: 1.5,
            threshold: 1.0,
        },
        ActivationKind::Softsign,
        ActivationKind::Hardtanh {
            min: -0.5,
            max: 0.75,
        },
        ActivationKind::Relu6,
        ActivationKind::Hardsigmoid,
        ActivationKind::Hardswish,
        ActivationKind::LogSigmoid,
    ]
}

/// An MLP 12 → 16 → 9 → 4 with `kind` after each of the first two `Linear`s, in either
/// precision.
pub(super) fn activation_mlp<T: Real>(
    params: &mut ParamStore<T>,
    seed: u64,
    kind: ActivationKind,
) -> Sequential<T> {
    let mut rng = Rng::seed(seed);
    Sequential::new_generic()
        .push(Linear::new(params, 12, 16, &mut rng))
        .push(ActivationLayer::<T>::new(kind))
        .push(Linear::new(params, 16, 9, &mut rng))
        .push(ActivationLayer::<T>::new(kind))
        .push(Linear::new(params, 9, 4, &mut rng))
}

/// `mnist-cnn`: LeNet-5's layout, as the benchmark builds it.
fn mnist_cnn<T: Real>(params: &mut ParamStore<T>, seed: u64) -> Sequential<T> {
    let mut rng = Rng::seed(seed);
    let g1 = Conv2dGeom::with_options(1, 6, (28, 28), (5, 5), (1, 1), (2, 2), (1, 1)).unwrap();
    let g2 = Conv2dGeom::new(6, 16, (14, 14), (5, 5)).unwrap();
    let pool = |c, s| MaxPool2d::new(Pool2dGeom::new(c, (s, s), (2, 2)).unwrap());
    Sequential::new_generic()
        .push(Conv2d::new(params, g1, &mut rng))
        .push(Relu)
        .push(pool(6, 28))
        .push(Conv2d::new(params, g2, &mut rng))
        .push(Relu)
        .push(pool(16, 10))
        .push(Flatten)
        .push(Linear::new(params, 400, 120, &mut rng))
        .push(Relu)
        .push(Linear::new(params, 120, 84, &mut rng))
        .push(Relu)
        .push(Linear::new(params, 84, 10, &mut rng))
}

/// `cifar-cnn`, as the benchmark builds it.
fn cifar_cnn<T: Real>(params: &mut ParamStore<T>, seed: u64) -> Sequential<T> {
    let mut rng = Rng::seed(seed);
    let mut net = Sequential::new_generic();
    for (i, o, s) in [(3, 32, 32), (32, 32, 32), (32, 64, 16), (64, 64, 16)] {
        let g = Conv2dGeom::with_options(i, o, (s, s), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
        net = net
            .push(Conv2d::with_init(
                params,
                g,
                Init::TorchDefault,
                None,
                &mut rng,
            ))
            .push(BatchNorm2d::new(params, o, (s, s)))
            .push(Relu);
        if i == o {
            net = net.push(MaxPool2d::new(Pool2dGeom::new(o, (s, s), (2, 2)).unwrap()));
        }
    }
    net.push(Dropout::new(0.25).unwrap())
        .push(Flatten)
        .push(Linear::new(params, 4096, 256, &mut rng))
        .push(Relu)
        .push(Dropout::new(0.5).unwrap())
        .push(Linear::new(params, 256, 10, &mut rng))
}

/// A network built twice from one seed, in `f32` and in `f64`, with the `f64` store holding
/// the `f32` store's values exactly.
pub(super) struct Twin {
    /// The network on `f32` values.
    pub(super) net: Sequential,
    /// Its values.
    pub(super) params: ParamStore,
    /// The same network on `f64` values.
    pub(super) net64: Sequential<f64>,
    /// The `f32` values, widened.
    pub(super) params64: ParamStore<f64>,
}

/// Builds a network in both precisions from `seed`. The biases, and the batch-norm layers'
/// running statistics and affine transform, get values away from their initial ones, so that
/// each does work; then the `f64` values are made the `f32` ones.
pub(super) fn twin(
    build64: impl Fn(&mut ParamStore<f64>, u64) -> Sequential<f64>,
    build32: impl Fn(&mut ParamStore, u64) -> Sequential,
    seed: u64,
) -> Twin {
    let mut params = ParamStore::new();
    let net = build32(&mut params, seed);
    let mut params64 = ParamStore::<f64>::new_generic();
    let net64 = build64(&mut params64, seed);
    let mut s = seed * 1000;
    for i in 0..net.len() {
        // `Linear` and `Conv2d` start with zero biases; give them values, so the bias add counts.
        let layer = net.layer(i);
        let bias = (layer.as_linear().and_then(|l| l.bias()))
            .or_else(|| layer.as_conv2d().and_then(|c| c.bias()));
        if let Some(b) = bias {
            s += 1;
            let n = params.value(b).data().len();
            params
                .value_mut(b)
                .data_mut()
                .copy_from_slice(&uniform(s, n, -0.5, 0.5));
        }
        if let Some(bn) = net.layer(i).as_batch_norm2d() {
            let c = bn.channels();
            let r = bn.running().unwrap();
            let mut set = |id, lo, hi, params: &mut ParamStore| {
                s += 1;
                params
                    .value_mut(id)
                    .data_mut()
                    .copy_from_slice(&uniform(s, c, lo, hi));
            };
            set(r.mean, -0.5, 0.5, &mut params);
            set(r.var, 0.5, 2.0, &mut params);
            set(bn.weight().unwrap(), 0.5, 1.5, &mut params);
            set(bn.bias().unwrap(), -0.5, 0.5, &mut params);
        }
    }
    for ((name, id), (name64, id64)) in net.state_names().into_iter().zip(net64.state_names()) {
        assert_eq!(name, name64);
        let v32 = params.value(id).data();
        for (a, &b) in params64.value_mut(id64).data_mut().iter_mut().zip(v32) {
            *a = f64::from(b);
        }
    }
    Twin {
        net,
        params,
        net64,
        params64,
    }
}

/// The `f32` values of `id` (an id of the `f32` store), widened.
fn wide(t: &Twin, id: ParamId) -> Vec<f64> {
    t.params
        .value(id)
        .data()
        .iter()
        .map(|&v| f64::from(v))
        .collect()
}

/// Runs `f` on a fresh `f64` evaluation tape over `store`, with `inputs` (data and shape), and
/// returns the output's values.
fn on_tape(
    store: &ParamStore<f64>,
    inputs: &[(&[f64], &[usize])],
    f: impl FnOnce(&mut Tape<'_, f64>, &[Var]) -> Var,
) -> Vec<f64> {
    let mut graph = Graph::<f64>::default();
    let mut tape = graph.tape(&store.values);
    let tensors: Vec<Tensor<f64>> = (inputs.iter())
        .map(|(d, s)| Tensor::from_vec(d.to_vec(), s).unwrap())
        .collect();
    let vars: Vec<Var> = tensors.iter().map(|t| tape.input(t.view())).collect();
    let y = f(&mut tape, &vars);
    tape.value(y).data().to_vec()
}

/// Each layer's output on the `f64` tape in evaluation mode, for `x` (`rows` rows).
fn reference(t: &Twin, x: &[f64], rows: usize) -> Vec<Vec<f64>> {
    let input = Tensor::from_vec(x.to_vec(), &[rows, x.len() / rows]).unwrap();
    let mut graph = Graph::<f64>::default();
    let mut tape = graph.tape(&t.params64.values);
    let v = tape.input(input.view());
    let mut trace = Trace::new();
    t.net64.forward_traced(&mut tape, v, &mut trace);
    (0..t.net64.len())
        .map(|i| tape.value(trace.output(i)).data().to_vec())
        .collect()
}

/// Layer `i` alone on the `f64` tape in evaluation mode, on `x`, with its mask.
fn one_layer(t: &Twin, i: usize, x: &[f64], rows: usize) -> Vec<f64> {
    let mut out = on_tape(&t.params64, &[(x, &[rows, x.len() / rows])], |tape, v| {
        t.net64.layer(i).forward(tape, v[0])
    });
    if let Some(mask) = t.net.mask(i) {
        let w = mask.width();
        for (j, y) in out.iter_mut().enumerate() {
            if !mask.is_on(j % w) {
                *y = 0.0;
            }
        }
    }
    out
}

/// The bound on layer `i`'s GPU output, for an input the GPU holds within `e` of `value`
/// (`rows` rows), whose exact output is `out`: the error `e` carried through the layer plus
/// the layer's own kernel bound. With `e` all 0 it is the kernel's bound alone.
///
/// - `Linear` and `Conv2d`: `Σ|w|·e` carried, plus the GEMM's `2(k + 3)·u·(Σ|w|·(|x| + e) +
///   |b|)`, and for `Linear` the bias add's rounding.
/// - ReLU: `e` carried (it is 1-Lipschitz and exact); tanh: `e` carried plus `8u` of its size.
/// - Max pooling: the window's largest `e`.
/// - Batch norm with running statistics: `|γ|·(inv + δinv)·e` carried, plus the kernel's
///   rounding of `inv` and of the normalisation.
/// - Dropout and flatten: `e` carried. A mask: 0 where it is off.
///
/// Every rounding also allows a flushed subnormal.
fn layer_bound(t: &Twin, i: usize, rows: usize, value: &[f64], e: &[f64], out: &[f64]) -> Vec<f64> {
    let layer = t.net.layer(i);
    let width_in = value.len() / rows;
    let none = ParamStore::<f64>::new_generic();
    let abs_x: Vec<f64> = value.iter().zip(e).map(|(v, e)| v.abs() + e).collect();
    let mut next: Vec<f64> = if let Some(lin) = layer.as_linear() {
        let (k, n) = (lin.inputs(), lin.outputs());
        let w: Vec<f64> = wide(t, lin.weight()).iter().map(|v| v.abs()).collect();
        let carried = on_tape(&none, &[(e, &[rows, k]), (&w, &[n, k])], |t, v| {
            t.matmul_nt(v[0], v[1])
        });
        let terms = on_tape(&none, &[(&abs_x, &[rows, k]), (&w, &[n, k])], |t, v| {
            t.matmul_nt(v[0], v[1])
        });
        (0..rows * n)
            .map(|j| {
                let kf = (k + 3) as f64;
                let gemm = carried[j] + 2.0 * kf * U * terms[j] + kf * FLOOR;
                // The bias add rounds once more.
                match lin.bias() {
                    Some(_) => gemm + U * (out[j].abs() + gemm) + FLOOR,
                    None => gemm,
                }
            })
            .collect()
    } else if let Some(conv) = layer.as_conv2d() {
        let g = conv.geom();
        let k = g.in_channels() * g.kernel().0 * g.kernel().1;
        let w: Vec<f64> = wide(t, conv.weight()).iter().map(|v| v.abs()).collect();
        let wshape = [
            g.out_channels(),
            g.in_channels(),
            g.kernel().0,
            g.kernel().1,
        ];
        let carried = on_tape(&none, &[(e, &[rows, width_in]), (&w, &wshape)], |t, v| {
            t.conv2d(v[0], v[1], None, g)
        });
        let terms = on_tape(
            &none,
            &[(&abs_x, &[rows, width_in]), (&w, &wshape)],
            |t, v| t.conv2d(v[0], v[1], None, g),
        );
        let plane = g.out_width() / g.out_channels();
        let bias = conv.bias().map(|b| wide(t, b));
        (0..carried.len())
            .map(|j| {
                let c = (j % g.out_width()) / plane;
                let b = bias.as_ref().map_or(0.0, |b| b[c].abs());
                let kf = (k + 3) as f64;
                carried[j] + 2.0 * kf * U * (terms[j] + b) + kf * FLOOR
            })
            .collect()
    } else if let Some(bn) = layer.as_batch_norm2d() {
        let c = bn.channels();
        let r = bn
            .running()
            .expect("the test networks keep running statistics");
        let (mean, var) = (wide(t, r.mean), wide(t, r.var));
        let (gamma, beta) = (wide(t, bn.weight().unwrap()), wide(t, bn.bias().unwrap()));
        let eps = bn.eps();
        let eps32 = f64::from(eps as f32);
        let spatial = width_in / c;
        (0..value.len())
            .map(|j| {
                let ch = (j % width_in) / spatial;
                let a = var[ch] + eps;
                let da = U * (var[ch] + eps32) + (eps32 - eps).abs() + FLOOR;
                let rr = da / a;
                let inv = 1.0 / a.sqrt();
                let e_inv = inv * (rr + 4.0 * U * (1.0 + rr)) + FLOOR;
                let g = gamma[ch].abs();
                let d = (value[j] - mean[ch]).abs() + e[j];
                let carried = g * (inv + e_inv) * e[j];
                let own = g * d * e_inv
                    + 4.0 * U * (out[j].abs() + g * d * (inv + e_inv) + beta[ch].abs());
                carried + own + 4.0 * FLOOR
            })
            .collect()
    } else {
        match layer.inference(&t.params.values) {
            Some(InferenceLayer::Activation(_)) if layer.name() == "relu" => e.to_vec(),
            Some(InferenceLayer::Activation(_)) if layer.name() == "tanh" => (0..e.len())
                .map(|j| e[j] + 8.0 * U * (out[j].abs() + e[j]) + FLOOR)
                .collect(),
            Some(InferenceLayer::MaxPool2d(geom)) => {
                on_tape(&none, &[(e, &[rows, width_in])], |t, v| {
                    t.max_pool2d(v[0], geom)
                })
            }
            Some(InferenceLayer::Identity) => e.to_vec(),
            other => panic!("no bound for layer {i} ({}): {other:?}", layer.name()),
        }
    };
    if let Some(mask) = t.net.mask(i) {
        let w = mask.width();
        for (j, b) in next.iter_mut().enumerate() {
            if !mask.is_on(j % w) {
                *b = 0.0;
            }
        }
    }
    next
}

/// Asserts `|got − want| ≤ bound` everywhere and returns the worst ratio of error to bound
/// with its index.
fn within(got: &[f32], want: &[f64], bound: &[f64], what: &str) -> (f64, usize) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let mut worst = (0.0f64, 0usize);
    for (j, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (f64::from(g) - w).abs();
        assert!(
            err <= bound[j],
            "{what}: value {j} is {g}, the f64 tape gives {w}: off by {err:e}, bound {:e}",
            bound[j]
        );
        let ratio = if bound[j] > 0.0 { err / bound[j] } else { 0.0 };
        if ratio > worst.0 {
            worst = (ratio, j);
        }
    }
    worst
}

/// The network's GPU forward pass on `x` (one full batch) against the `f64` tape, layer by
/// layer and end to end, on every GPU.
fn check_forward(t: &Twin, batch: usize, x: &[f32], what: &str) {
    let x64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let per_layer = reference(t, &x64, batch);
    let mut e = vec![0.0; x.len()];
    let mut value = x64.clone();
    for (i, out) in per_layer.iter().enumerate() {
        e = layer_bound(t, i, batch, &value, &e, out);
        value.clone_from(out);
    }
    let composed = e;
    for_each_gpu(|gpu| {
        let mut model = GpuModel::compile(&t.net, &t.params, batch, gpu).unwrap();
        let layers = model.infer_layers(x);
        let mut input = x64.clone();
        let mut sharpest = (0.0f64, 0usize);
        for (i, got) in layers.iter().enumerate() {
            let want = one_layer(t, i, &input, batch);
            let bound = layer_bound(t, i, batch, &input, &vec![0.0; input.len()], &want);
            let name = t.net.layer(i).name();
            let (ratio, _) = within(got, &want, &bound, &format!("{what} layer {i} ({name})"));
            println!("{what} layer {i} ({name}): worst {ratio:.3} of its own bound");
            // A masked-off value is +0 exactly, as a select gives it (a multiply would give −0
            // for a negative value and NaN for NaN).
            if let Some(mask) = t.net.mask(i) {
                for (j, v) in got.iter().enumerate() {
                    if !mask.is_on(j % mask.width()) {
                        assert_eq!(v.to_bits(), 0, "{what} layer {i}: masked value {j} is {v}");
                    }
                }
            }
            if ratio > sharpest.0 {
                sharpest = (ratio, i);
            }
            input = got.iter().map(|&v| f64::from(v)).collect();
        }
        let want = per_layer.last().unwrap();
        let got = layers.last().unwrap();
        let (ratio, j) = within(got, want, &composed, &format!("{what} end to end"));
        let scale = want.iter().map(|w| w.abs()).fold(0.0, f64::max);
        let widest = composed.iter().copied().fold(0.0, f64::max);
        let off = (got.iter().zip(want))
            .map(|(&g, &w)| (f64::from(g) - w).abs())
            .fold(0.0, f64::max);
        println!(
            "{what}: layers worst {:.3} (layer {}); end to end worst {ratio:.2e} of the composed bound \
             (output {j}), largest error {off:.2e}, widest bound {widest:.2e}, outputs up to {scale:.2e}",
            sharpest.0, sharpest.1
        );
        // The model's answer through infer_batch is the last layer's.
        let input =
            cavitation_core::TensorView::from_slice(x, &[batch, model.input_width()]).unwrap();
        let answer: Vec<u32> = model
            .infer_batch(input)
            .data()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let last: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
        assert_eq!(answer, last, "{what}: infer_batch");
    });
}

/// The MLP with a neuron mask on the first hidden layer (after its tanh) and a mask on the
/// ReLU's output.
#[test]
fn mlp_matches_the_tape() {
    let mut t = twin(mlp::<f64>, mlp::<f32>, 3);
    for (net_mask, layer) in [
        (NeuronMask::from_off(16, &[2, 7, 11]), 1),
        (NeuronMask::from_off(9, &[4]), 3),
    ] {
        t.net.set_mask(layer, net_mask.clone());
        t.net64.set_mask(layer, net_mask);
    }
    let batch = 37;
    check_forward(&t, batch, &uniform(11, batch * 12, -2.0, 2.0), "mlp");
}

/// `mnist-cnn` on 16 random images.
#[test]
fn mnist_cnn_matches_the_tape() {
    let t = twin(mnist_cnn::<f64>, mnist_cnn::<f32>, 5);
    let batch = 16;
    check_forward(
        &t,
        batch,
        &uniform(12, batch * 28 * 28, 0.0, 1.0),
        "mnist-cnn",
    );
}

/// `cifar-cnn` on 8 random images, its batch norms from running statistics.
#[test]
fn cifar_cnn_matches_the_tape() {
    let t = twin(cifar_cnn::<f64>, cifar_cnn::<f32>, 7);
    let batch = 8;
    check_forward(
        &t,
        batch,
        &uniform(13, batch * 3 * 32 * 32, -1.0, 1.0),
        "cifar-cnn",
    );
}

/// A short batch through `mnist-cnn` gives the full batch's rows bit for bit, while the rows it
/// leaves unused still hold NaN from the batch before: no layer reads another row.
#[test]
fn short_batches_ignore_stale_rows() {
    let t = twin(mnist_cnn::<f64>, mnist_cnn::<f32>, 8);
    let (batch, width) = (4, 28 * 28);
    let clean = uniform(14, batch * width, 0.0, 1.0);
    let mut stale = clean.clone();
    stale[2 * width..].fill(f32::NAN);
    for_each_gpu(|gpu| {
        let mut model = GpuModel::compile(&t.net, &t.params, batch, gpu).unwrap();
        fn view(x: &[f32], rows: usize) -> cavitation_core::TensorView<'_, f32> {
            cavitation_core::TensorView::from_slice(x, &[rows, x.len() / rows]).unwrap()
        }
        let full: Vec<u32> = model
            .infer_batch(view(&clean, batch))
            .data()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let poisoned = model.infer_batch(view(&stale, batch)).data().to_vec();
        assert!(
            poisoned[20..].iter().all(|v| v.is_nan()),
            "the stale rows' outputs are NaN"
        );
        let short: Vec<u32> = model
            .infer_batch(view(&clean[..2 * width], 2))
            .data()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(short, full[..20]);
    });
}

/// Softmax and log-softmax after each `Linear` of an MLP: the model's evaluation-mode outputs
/// against the CPU's `InferenceModel`, within 2⁻¹⁰ of the CPU's largest output.
#[test]
fn softmax_layers_infer_as_the_cpu() {
    for (seed, log) in [(94, false), (95, true)] {
        let what = if log { "log-softmax" } else { "softmax" };
        let mut params = ParamStore::new();
        let net = softmax_mlp::<f32>(&mut params, seed, log);
        let batch = 23;
        let input = Tensor::from_vec(uniform(seed, batch * 12, -2.0, 2.0), &[batch, 12]).unwrap();
        let mut cpu = InferenceModel::compile(&net, &params).unwrap();
        let want = cpu.infer_batch(input.view()).data().to_vec();
        for_each_gpu(|gpu| {
            let mut model = GpuModel::compile(&net, &params, batch, gpu).unwrap();
            let got = model.infer_batch(input.view()).data().to_vec();
            let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let off = (got.iter().zip(&want))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("{what}: off by {off:.3e} of a largest output {scale:.3e}");
            assert!(off <= scale / 1024.0, "{what}: off by {off:e} of {scale:e}");
        });
    }
}

/// A PReLU before each `Linear` of an MLP, with one slope per column and with one shared
/// slope: the model's evaluation-mode outputs against the CPU's `InferenceModel`, within 2⁻¹⁰
/// of the CPU's largest output.
#[test]
fn prelu_layers_infer_as_the_cpu() {
    for (seed, shared) in [(96, false), (97, true)] {
        let what = if shared { "shared PReLU" } else { "PReLU" };
        let mut params = ParamStore::new();
        let net = prelu_mlp::<f32>(&mut params, seed, shared);
        let batch = 23;
        let input = Tensor::from_vec(uniform(seed, batch * 12, -2.0, 2.0), &[batch, 12]).unwrap();
        let mut cpu = InferenceModel::compile(&net, &params).unwrap();
        let want = cpu.infer_batch(input.view()).data().to_vec();
        for_each_gpu(|gpu| {
            let mut model = GpuModel::compile(&net, &params, batch, gpu).unwrap();
            let got = model.infer_batch(input.view()).data().to_vec();
            let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let off = (got.iter().zip(&want))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("{what}: off by {off:.3e} of a largest output {scale:.3e}");
            assert!(off <= scale / 1024.0, "{what}: off by {off:e} of {scale:e}");
        });
    }
}
