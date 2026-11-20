//! What the CUDA engine's tests share: opening CUDA or saying why not, the test networks in
//! both precisions, and batches and data sets from seeds.
//!
//! Without CUDA every test says so and checks nothing; `CAVITATION_CUDA_REQUIRED=1` makes
//! that a failure, through [`Cuda::new`].

use cavitation_core::{Conv2dGeom, ParamStore, Pool2dGeom, Real, Rng, Tensor, TensorView};
use cavitation_cuda::Cuda;

use crate::{
    ActivationKind, ActivationLayer, BatchNorm2d, BatchView, Conv2d, Dataset, Dropout, Flatten,
    Init, Linear, MaxPool2d, Relu, Sequential, Tanh, Targets, TargetsView,
};

/// Runs `body` on CUDA if it opens, and returns whether it ran.
pub(super) fn with_cuda(body: impl FnOnce(&Cuda)) -> bool {
    match Cuda::new() {
        Some(cuda) => {
            println!("on CUDA: {}", cuda.info().name);
            body(&cuda);
            true
        }
        None => {
            println!("no CUDA: skipped");
            false
        }
    }
}

/// `n` values in `[lo, hi)` from `seed`.
pub(super) fn uniform(seed: u64, n: usize, lo: f64, hi: f64) -> Vec<f32> {
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

/// A small CNN on 2 × 8 × 8 images: a convolution with a bias, batch norm, ReLU, max pooling,
/// dropout, flatten and a `Linear`, in either precision.
pub(super) fn small_cnn<T: Real>(params: &mut ParamStore<T>, seed: u64) -> Sequential<T> {
    let mut rng = Rng::seed(seed);
    let g = Conv2dGeom::with_options(2, 4, (8, 8), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
    Sequential::new_generic()
        .push(Conv2d::new(params, g, &mut rng))
        .push(BatchNorm2d::new(params, 4, (8, 8)))
        .push(Relu)
        .push(MaxPool2d::new(Pool2dGeom::new(4, (8, 8), (2, 2)).unwrap()))
        .push(Dropout::new(0.3).unwrap())
        .push(Flatten)
        .push(Linear::new(params, 64, 5, &mut rng))
}

/// A CNN on 2 × 8 × 8 images with batch norm keeping a cumulative average (no momentum):
/// a convolution, batch norm, ReLU, max pooling, dropout, flatten and a `Linear` to 5
/// classes.
pub(super) fn cumulative_cnn(params: &mut ParamStore, seed: u64) -> Sequential {
    let mut rng = Rng::seed(seed);
    let g = Conv2dGeom::with_options(2, 4, (8, 8), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
    Sequential::new()
        .push(Conv2d::new(params, g, &mut rng))
        .push(BatchNorm2d::with_options(params, 4, (8, 8), 1e-5, None, true, true).unwrap())
        .push(Relu)
        .push(MaxPool2d::new(Pool2dGeom::new(4, (8, 8), (2, 2)).unwrap()))
        .push(Dropout::new(0.3).unwrap())
        .push(Flatten)
        .push(Linear::new(params, 64, 5, &mut rng))
}

/// `mnist-cnn`: LeNet-5's layout, as the benchmark builds it.
pub(super) fn mnist_cnn<T: Real>(params: &mut ParamStore<T>, seed: u64) -> Sequential<T> {
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
pub(super) fn cifar_cnn<T: Real>(params: &mut ParamStore<T>, seed: u64) -> Sequential<T> {
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

/// One batch: rows of inputs, and class labels or target values.
pub(super) struct Data {
    /// The inputs, row after row.
    pub(super) x: Vec<f32>,
    /// The rows.
    pub(super) rows: usize,
    /// Class labels, for cross-entropy.
    pub(super) labels: Vec<usize>,
    /// Target values, for MSE.
    pub(super) values: Option<Vec<f32>>,
}

impl Data {
    /// `rows` rows of `width` inputs in `[-1, 1)` and labels below `classes`, from `seed`.
    pub(super) fn new(seed: u64, rows: usize, width: usize, classes: usize) -> Data {
        Data::within(seed, rows, width, classes, (-1.0, 1.0))
    }

    /// `rows` rows of `width` inputs in `[lo, hi)` and labels below `classes`, from `seed`.
    pub(super) fn within(
        seed: u64,
        rows: usize,
        width: usize,
        classes: usize,
        (lo, hi): (f64, f64),
    ) -> Data {
        let mut rng = Rng::seed(seed + 1);
        Data {
            x: uniform(seed, rows * width, lo, hi),
            rows,
            labels: (0..rows)
                .map(|_| (rng.range(0.0, classes as f64) as usize).min(classes - 1))
                .collect(),
            values: None,
        }
    }

    /// The batch as a trainer takes it.
    pub(super) fn view(&self) -> BatchView<'_, f32> {
        let x = TensorView::from_slice(&self.x, &[self.rows, self.x.len() / self.rows]).unwrap();
        let targets = match &self.values {
            Some(v) => TargetsView::Values(
                TensorView::from_slice(v, &[self.rows, v.len() / self.rows]).unwrap(),
            ),
            None => TargetsView::Classes(&self.labels),
        };
        BatchView { x, targets }
    }
}

/// `rows` rows of `width` values in `[-1, 1)` from `seed`, with labels below `classes`.
pub(super) fn classes(seed: u64, rows: usize, width: usize, classes: usize) -> Dataset {
    let mut rng = Rng::seed(seed);
    let x: Vec<f32> = (0..rows * width)
        .map(|_| rng.range(-1.0, 1.0) as f32)
        .collect();
    let labels = (0..rows)
        .map(|_| rng.range(0.0, classes as f64) as usize)
        .collect();
    Dataset {
        x: Tensor::from_vec(x, &[rows, width]).unwrap(),
        targets: Targets::Classes(labels),
    }
}

/// Every parameter and buffer value of `net` in `params`, by its bits, in the model's order.
pub(super) fn bits(net: &Sequential, params: &ParamStore) -> Vec<u32> {
    net.state_names()
        .iter()
        .flat_map(|(_, id)| params.value(*id).data().iter().map(|v| v.to_bits()))
        .collect()
}

/// The largest difference between `got` and `want`, value by value, and `want`'s largest
/// magnitude: `(difference, magnitude)`.
pub(super) fn off(got: &[f32], want: &[f32]) -> (f32, f32) {
    let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let diff = (got.iter().zip(want))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    (diff, scale)
}
