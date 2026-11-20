//! `CudaModel` against the CPU: the plan's refusals under CUDA's name, without the wgpu
//! engine's own limits; every parameter, running statistic and mask read back bit for bit;
//! the evaluation-mode outputs of a network with every layer kind the card runs (two of them
//! masked), the masked MLP, `mnist-cnn` and `cifar-cnn` against the CPU's `InferenceModel`,
//! each within 2⁻¹⁰ of the CPU's largest output (a measured agreement: cuDNN and cuBLAS
//! publish no error bounds); short batches; and `evaluate` and `accuracy` against the CPU's.
//!
//! Without CUDA the tests that need it say so and check nothing;
//! `CAVITATION_CUDA_REQUIRED=1` makes that a failure.

use cavitation_core::{Conv2dGeom, Gate, GeluApprox, ParamId, ParamStore, Pool2dGeom, Rng, Tensor};

use super::CudaModel;
use super::test_util::{
    cifar_cnn, classes, mlp, mnist_cnn, off, prelu_mlp, twin, uniform, with_cuda,
};
use crate::plan::{Engine, Plan, pack_rows};
use crate::{
    AvgPool2d, BatchNorm2d, Celu, Conv2d, Dropout, Flatten, Gelu, Glu, Hardtanh, InferenceModel,
    Init, LeakyRelu, Linear, LogSoftmax, Loss, MaxPool2d, NeuronMask, Prelu, Relu, Sequential,
    Sigmoid, Softmax, Tanh, accuracy, evaluate,
};

/// A small CNN with every kind of layer the card runs, two of them masked: 1 × 6 × 6 images
/// through a convolution, batch norm, ReLU, max pooling, average pooling with padding,
/// dropout, flatten, two `Linear`s, each activation, and a softmax and a log-softmax.
fn every_kind(params: &mut ParamStore) -> Sequential {
    let mut rng = Rng::seed(7);
    let conv = Conv2dGeom::with_options(1, 3, (6, 6), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
    let mut net = Sequential::new()
        .push(Conv2d::new(params, conv, &mut rng))
        .push(BatchNorm2d::new(params, 3, (6, 6)))
        .push(Relu)
        .push(MaxPool2d::new(Pool2dGeom::new(3, (6, 6), (2, 2)).unwrap()))
        .push(AvgPool2d::new(
            Pool2dGeom::with_options(3, (3, 3), (2, 2), (1, 1), (1, 1), (1, 1), false).unwrap(),
        ))
        .push(Dropout::new(0.25).unwrap())
        .push(Flatten)
        .push(Linear::new(params, 3 * 4 * 4, 5, &mut rng))
        .push(LeakyRelu { slope: 0.125 })
        .push(Tanh)
        .push(Sigmoid)
        .push(Gelu::default())
        .push(Gelu {
            approx: GeluApprox::Tanh,
        })
        .push(Softmax)
        .push(LogSoftmax)
        .push(Linear::without_bias(
            params,
            5,
            2,
            Init::TorchDefault,
            &mut rng,
        ));
    net.set_mask(2, NeuronMask::from_off(3 * 6 * 6, &[0, 5, 107]));
    net.set_neuron_mask(7, NeuronMask::from_off(5, &[1, 3]));
    net
}

/// The plan names `CudaModel::compile` in its refusals, and the wgpu engine's limits (a
/// convolution's batch above 65,535, a softmax's above 16,776,960, a PReLU's slope gradient
/// over more than 268,431,360 values per slope, a batch norm's eps below 2⁻¹²⁶) do not apply;
/// a PReLU of more than 65,535 slopes, or over more than 2³¹ values per batch, is refused on
/// both.
#[test]
fn the_plan_refuses_under_cudas_name() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(0);
    let net = Sequential::new()
        .push(Linear::new(&mut params, 4, 3, &mut rng))
        .push(Glu {
            gate: Gate::Sigmoid,
        });
    let err = Plan::build_for(&net, &params, 8, Engine::CUDA).unwrap_err();
    assert_eq!(
        err.to_string(),
        "unsupported: CudaModel::compile: layer 1 (glu) is not a layer the GPU runs"
    );
    let err = Plan::build_for(&Sequential::new(), &params, 8, Engine::CUDA).unwrap_err();
    assert_eq!(
        err.to_string(),
        "unsupported: CudaModel::compile: the model has no layers"
    );
    let conv = Conv2dGeom::new(1, 1, (2, 2), (1, 1)).unwrap();
    let net = Sequential::new().push(Conv2d::new(&mut params, conv, &mut rng));
    assert!(Plan::build_for(&net, &params, 65_536, Engine::CUDA).is_ok());
    assert!(Plan::build_for(&net, &params, 65_536, Engine::WGPU).is_err());
    let mut store = ParamStore::new();
    let tiny_eps =
        BatchNorm2d::with_options(&mut store, 2, (2, 2), 1e-40, Some(0.1), true, true).unwrap();
    let net = Sequential::new().push(tiny_eps);
    assert!(Plan::build_for(&net, &store, 4, Engine::CUDA).is_ok());
    assert!(Plan::build_for(&net, &store, 4, Engine::WGPU).is_err());
    let norm =
        BatchNorm2d::with_options(&mut store, 2, (2, 2), 1e-5, Some(0.1), true, true).unwrap();
    for net in [
        Sequential::new().push(norm.clone()).push(Softmax),
        Sequential::new().push(norm).push(LogSoftmax),
    ] {
        assert!(Plan::build_for(&net, &store, 16_776_961, Engine::CUDA).is_ok());
        assert!(Plan::build_for(&net, &store, 16_776_961, Engine::WGPU).is_err());
    }
    let net = Sequential::new().push(Prelu::new(&mut store, 2));
    assert!(Plan::build_for(&net, &store, 268_431_361, Engine::CUDA).is_ok());
    // A shared slope over more than 2³¹ values per batch is refused when the model compiles.
    let mut rng = Rng::seed(1);
    let net = Sequential::new()
        .push(Linear::new(&mut store, 2, 4096, &mut rng))
        .push(Prelu::new(&mut store, 1));
    assert!(Plan::build_for(&net, &store, 524_288, Engine::CUDA).is_ok());
    let err = Plan::build_for(&net, &store, 524_289, Engine::CUDA).unwrap_err();
    assert_eq!(
        err.to_string(),
        "unsupported: CudaModel::compile: layer 1 (prelu) is a PReLU over 524289 rows of 4096 \
         values, more than the 2147483648 per batch the GPU takes"
    );
    assert!(Plan::build_for(&net, &store, 268_431_361, Engine::WGPU).is_err());
    let wide = Sequential::new().push(Prelu::new(&mut store, 65_536));
    let err = Plan::build_for(&wide, &store, 2, Engine::CUDA).unwrap_err();
    assert_eq!(
        err.to_string(),
        "unsupported: CudaModel::compile: layer 0 (prelu) is a PReLU with 65536 slopes; the GPU \
         takes at most 65535"
    );
}

/// Every parameter, running statistic and mask comes back from the card bit for bit, odd
/// values included, and the batch count as the host kept it.
#[test]
fn uploads_read_back_exactly() {
    let mut params = ParamStore::new();
    let net = every_kind(&mut params);
    let ids: Vec<ParamId> = net.state_names().into_iter().map(|(_, id)| id).collect();
    let odd = [
        -0.0,
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(1),
        f32::MAX,
        f32::NEG_INFINITY,
    ];
    for &id in &ids {
        for (v, &o) in params.value_mut(id).data_mut().iter_mut().zip(&odd) {
            *v = o;
        }
    }
    let count = (net.layer(1).as_batch_norm2d())
        .and_then(|bn| bn.running())
        .and_then(|r| r.count)
        .unwrap();
    params.value_mut(count).data_mut()[0] = 17.0;
    let want: Vec<Vec<u32>> = (ids.iter())
        .map(|&id| {
            params
                .value(id)
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
        .collect();
    with_cuda(|cuda| {
        let mut model = CudaModel::compile(&net, &params, 3, cuda).unwrap();
        let mut back = params.clone();
        for &id in &ids {
            back.value_mut(id).data_mut().fill(1.5);
        }
        model.download_to(&mut back);
        for (k, &id) in ids.iter().enumerate() {
            let got: Vec<u32> = back.value(id).data().iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, want[k], "parameter {}", id.position());
        }
        let masks = model.mask_words();
        let on2: Vec<bool> = (0..108).map(|j| ![0, 5, 107].contains(&j)).collect();
        let on12: Vec<bool> = (0..5).map(|j| ![1, 3].contains(&j)).collect();
        assert_eq!(masks, [(2, pack_rows(&on2, 3)), (12, pack_rows(&on12, 3))]);
    });
}

/// The card's evaluation-mode outputs of `net` on `rows` rows of `x` against the CPU's
/// `InferenceModel`, within 2⁻¹⁰ of the CPU's largest output; every layer's output on the
/// card is finite, and the last is the output.
fn check_outputs(net: &Sequential, params: &ParamStore, rows: usize, x: &[f32], what: &str) {
    let mut cpu = InferenceModel::compile(net, params).unwrap();
    let width = x.len() / rows;
    let input = Tensor::from_vec(x.to_vec(), &[rows, width]).unwrap();
    let want = cpu.infer_batch(input.view()).data().to_vec();
    with_cuda(|cuda| {
        let mut model = CudaModel::compile(net, params, rows, cuda).unwrap();
        let got = model.infer_batch(input.view()).data().to_vec();
        let (diff, scale) = off(&got, &want);
        println!("{what}: off by {diff:.3e} of a largest output {scale:.3e}");
        assert!(
            diff <= scale / 1024.0,
            "{what}: off by {diff:e} of {scale:e}"
        );
        let layers = model.infer_layers(x);
        assert_eq!(layers.len(), net.len());
        assert!(layers.iter().flatten().all(|v| v.is_finite()), "{what}");
        assert_eq!(
            layers.last().unwrap(),
            &got,
            "{what}: the last layer is the output"
        );
    });
}

/// A network with every layer kind, the masked MLP, `mnist-cnn` and `cifar-cnn`.
#[test]
fn outputs_match_the_cpu() {
    let mut params = ParamStore::new();
    let net = every_kind(&mut params);
    check_outputs(
        &net,
        &params,
        5,
        &uniform(1, 5 * 36, -2.0, 2.0),
        "every kind",
    );
    let mut t = twin(mlp::<f64>, mlp::<f32>, 2);
    t.net.set_mask(1, NeuronMask::from_off(16, &[1, 5]));
    let x = uniform(2, 7 * 12, -2.0, 2.0);
    check_outputs(&t.net, &t.params, 7, &x, "masked mlp");
    let t = twin(mnist_cnn::<f64>, mnist_cnn::<f32>, 3);
    let x = uniform(3, 9 * 784, 0.0, 1.0);
    check_outputs(&t.net, &t.params, 9, &x, "mnist-cnn");
    let t = twin(cifar_cnn::<f64>, cifar_cnn::<f32>, 4);
    let x = uniform(4, 6 * 3072, -1.0, 1.0);
    check_outputs(&t.net, &t.params, 6, &x, "cifar-cnn");
}

/// A batch of fewer rows than the model's runs as a full batch whose other rows are not
/// read: its outputs are the same rows' outputs of a full batch, bit for bit, whatever the
/// rows left from an earlier batch.
#[test]
fn short_batches_ignore_stale_rows() {
    let t = twin(mnist_cnn::<f64>, mnist_cnn::<f32>, 5);
    let x = uniform(5, 8 * 784, 0.0, 1.0);
    let other = uniform(6, 8 * 784, 0.0, 1.0);
    with_cuda(|cuda| {
        let mut model = CudaModel::compile(&t.net, &t.params, 8, cuda).unwrap();
        let full = Tensor::from_vec(x.clone(), &[8, 784]).unwrap();
        let want = model.infer_batch(full.view()).data()[..3 * 10].to_vec();
        let stale = Tensor::from_vec(other.clone(), &[8, 784]).unwrap();
        model.infer_batch(stale.view());
        let short = Tensor::from_vec(x[..3 * 784].to_vec(), &[3, 784]).unwrap();
        let got = model.infer_batch(short.view());
        assert_eq!(got.shape().dims(), &[3, 10]);
        let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(got.data()), bits(&want));
    });
}

/// `evaluate` and `accuracy` on the card measure what the CPU's do: the loss within 2⁻¹⁰ and
/// the same accuracy, on 50 rows at batch 16 (a last batch of 2); rows of another width are
/// refused.
#[test]
fn evaluate_and_accuracy_match_the_cpu() {
    let t = twin(mlp::<f64>, mlp::<f32>, 6);
    let data = classes(7, 50, 12, 4);
    let want = evaluate(&t.net, &t.params, &data, Loss::CrossEntropy, 16).unwrap();
    let cpu_accuracy = accuracy(&t.net, &t.params, &data, 16);
    with_cuda(|cuda| {
        let mut model = CudaModel::compile(&t.net, &t.params, 16, cuda).unwrap();
        let got = model.evaluate(&t.net, &data, Loss::CrossEntropy).unwrap();
        assert!(
            (got.loss - want.loss).abs() <= want.loss.abs() / 1024.0,
            "{got:?} against {want:?}"
        );
        assert_eq!(got.accuracy, want.accuracy);
        assert_eq!(model.accuracy(&data), cpu_accuracy);
        let wrong = classes(8, 4, 11, 4);
        assert!(model.evaluate(&t.net, &wrong, Loss::CrossEntropy).is_err());
    });
}

/// `CudaModel::compile` refuses, by index and name, a CELU whose `alpha` is not positive and
/// a hardtanh whose `min` is above its `max` or NaN (the CPU's tape panics on it), before it
/// touches the card.
#[test]
fn compile_refuses_activation_settings_the_kernels_do_not_take() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(3);
    let linear = Linear::new(&mut params, 4, 3, &mut rng);
    let cases: [(Sequential, &str); 3] = [
        (
            Sequential::new()
                .push(linear.clone())
                .push(Celu { alpha: -1.0 }),
            "layer 1 (celu) is a CELU with alpha -1; the GPU runs a positive alpha only",
        ),
        (
            Sequential::new().push(linear.clone()).push(Hardtanh {
                min: 1.0,
                max: -1.0,
            }),
            "layer 1 (hardtanh) is a hardtanh with min 1 and max -1; min must not exceed max",
        ),
        (
            Sequential::new().push(linear.clone()).push(Hardtanh {
                min: -1.0,
                max: f64::NAN,
            }),
            "layer 1 (hardtanh) is a hardtanh with min -1 and max NaN; min must not exceed max",
        ),
    ];
    with_cuda(|cuda| {
        for (net, why) in &cases {
            let Err(err) = CudaModel::compile(net, &params, 8, cuda) else {
                panic!("{why}: compiled");
            };
            assert_eq!(
                err.to_string(),
                format!("unsupported: CudaModel::compile: {why}")
            );
        }
    });
}

/// A PReLU before each `Linear` of an MLP, with one slope per column and with one shared
/// slope: the evaluation-mode outputs against the CPU's `InferenceModel`, and the slopes
/// back from the card bit for bit.
#[test]
fn prelu_layers_match_the_cpu() {
    for (seed, shared) in [(96, false), (97, true)] {
        let what = if shared { "shared PReLU" } else { "PReLU" };
        let mut params = ParamStore::new();
        let net = prelu_mlp::<f32>(&mut params, seed, shared);
        check_outputs(&net, &params, 23, &uniform(seed, 23 * 12, -2.0, 2.0), what);
        let slopes: Vec<ParamId> = (0..net.len())
            .filter_map(|i| net.layer(i).as_prelu().map(|p| p.weight()))
            .collect();
        assert_eq!(slopes.len(), 3);
        with_cuda(|cuda| {
            let mut model = CudaModel::compile(&net, &params, 4, cuda).unwrap();
            let mut back = params.clone();
            for &id in &slopes {
                back.value_mut(id).data_mut().fill(9.0);
            }
            model.download_to(&mut back);
            for &id in &slopes {
                assert_eq!(back.value(id).data(), params.value(id).data(), "{what}");
            }
        });
    }
}
