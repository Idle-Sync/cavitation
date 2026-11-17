//! `GpuModel`'s public answers in evaluation mode: `infer_batch` on short batches, and
//! `evaluate` and `accuracy` on whole data sets against the CPU's. (The forward pass's
//! layers against the CPU's `f64` tape are checked inside the crate, where each layer's
//! output can be read.) Without a GPU every test says so and checks nothing;
//! `CAVITATION_GPU_REQUIRED=1` makes that a failure.

#![cfg(feature = "gpu")]

use cavitation_core::{ParamStore, Real, Rng, Tensor, TensorView};
use cavitation_gpu::{Backend, Gpu};
use cavitation_nn::{
    BatchNorm2d, Dataset, GpuModel, Linear, Loss, Relu, Sequential, Tanh, Targets, accuracy,
    evaluate,
};

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

/// The test MLP, 12 → 16 → 9 → 4 with tanh and ReLU, built in either precision.
fn mlp<T: Real>(params: &mut ParamStore<T>, seed: u64) -> Sequential<T> {
    let mut rng = Rng::seed(seed);
    Sequential::new_generic()
        .push(Linear::new(params, 12, 16, &mut rng))
        .push(Tanh)
        .push(Linear::new(params, 16, 9, &mut rng))
        .push(Relu)
        .push(Linear::new(params, 9, 4, &mut rng))
}

/// The test MLP from `seed`, with its values.
fn build(seed: u64) -> (Sequential, ParamStore) {
    let mut params = ParamStore::new();
    let net = mlp(&mut params, seed);
    (net, params)
}

/// A short batch's rows are the full batch's, bit for bit: each row's outputs depend on that
/// row alone, whatever the rest of the batch holds.
#[test]
fn short_batches_give_the_full_batchs_rows() {
    let (net, params) = build(4);
    let x = uniform(21, 10 * 12, -2.0, 2.0);
    for_each_gpu(|gpu| {
        let mut model = GpuModel::compile(&net, &params, 10, gpu).unwrap();
        let full = model
            .infer_batch(TensorView::from_slice(&x, &[10, 12]).unwrap())
            .data()
            .to_vec();
        for rows in [1, 3, 9] {
            let start = 10 - rows;
            let part = TensorView::from_slice(&x[start * 12..], &[rows, 12]).unwrap();
            let got: Vec<u32> = model
                .infer_batch(part)
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let want: Vec<u32> = full[start * 4..].iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, want, "{rows} rows");
        }
    });
}

/// `evaluate` and `accuracy` on the GPU agree with the CPU's on a data set that ends in a
/// short batch: the same accuracy, and the loss and R² within rounding of the outputs.
#[test]
fn evaluate_and_accuracy_match_the_cpu() {
    let (net, params) = build(6);
    let rows = 203;
    let x = Tensor::from_vec(uniform(31, rows * 12, -2.0, 2.0), &[rows, 12]).unwrap();
    let mut rng = Rng::seed(32);
    let labels: Vec<usize> = (0..rows)
        .map(|_| (rng.range(0.0, 4.0) as usize).min(3))
        .collect();
    let classes = Dataset {
        x: x.clone(),
        targets: Targets::Classes(labels),
    };
    let values = Dataset {
        x,
        targets: Targets::Values(
            Tensor::from_vec(uniform(33, rows * 4, -1.0, 1.0), &[rows, 4]).unwrap(),
        ),
    };
    let cpu_ce = evaluate(&net, &params, &classes, Loss::CrossEntropy, 64).unwrap();
    let cpu_mse = evaluate(&net, &params, &values, Loss::Mse, 64).unwrap();
    let cpu_acc = accuracy(&net, &params, &classes, 64);
    for_each_gpu(|gpu| {
        let mut model = GpuModel::compile(&net, &params, 64, gpu).unwrap();
        let ce = model.evaluate(&net, &classes, Loss::CrossEntropy).unwrap();
        assert_eq!(ce.accuracy, cpu_ce.accuracy);
        assert!(
            (ce.loss - cpu_ce.loss).abs() <= 1e-5 * cpu_ce.loss.abs(),
            "{ce:?} {cpu_ce:?}"
        );
        assert_eq!(model.accuracy(&classes), cpu_acc);
        let mse = model.evaluate(&net, &values, Loss::Mse).unwrap();
        assert!(
            (mse.loss - cpu_mse.loss).abs() <= 1e-5 * cpu_mse.loss.abs(),
            "{mse:?} {cpu_mse:?}"
        );
        let (r2, cpu_r2) = (mse.r2.unwrap(), cpu_mse.r2.unwrap());
        assert!(
            (r2 - cpu_r2).abs() <= 1e-5 * (1.0 + cpu_r2.abs()),
            "{r2} {cpu_r2}"
        );
        // Data the model cannot take is refused as the CPU refuses it.
        let wrong = Dataset {
            x: Tensor::zeros(&[3, 5]),
            targets: Targets::Classes(vec![0; 3]),
        };
        assert!(model.evaluate(&net, &wrong, Loss::CrossEntropy).is_err());
        assert!(model.evaluate(&net, &values, Loss::CrossEntropy).is_err());
    });
}

/// A batch norm that keeps no running statistics normalises over the whole batch, so a short
/// batch is refused rather than answered from rows that are not the caller's.
#[test]
fn batch_statistics_need_whole_batches() {
    let mut params = ParamStore::new();
    let bn =
        BatchNorm2d::with_options(&mut params, 2, (2, 2), 1e-5, Some(0.1), true, false).unwrap();
    let net = Sequential::new().push(bn);
    for_each_gpu(|gpu| {
        let mut model = GpuModel::compile(&net, &params, 4, gpu).unwrap();
        let data = Dataset {
            x: Tensor::zeros(&[6, 8]),
            targets: Targets::Values(Tensor::zeros(&[6, 8])),
        };
        let err = model.evaluate(&net, &data, Loss::Mse).unwrap_err();
        assert!(err.to_string().contains("whole batches of 4"), "{err}");
        let x = Tensor::from_vec(uniform(41, 32, -1.0, 1.0), &[4, 8]).unwrap();
        assert_eq!(model.infer_batch(x.view()).shape().dims(), &[4, 8]);
    });
}

/// A model of 64 one-wide `Linear` layers: many kernel calls, each with its uniforms, on
/// tensors of a few bytes, against the CPU's answer.
#[test]
fn many_tiny_layers() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(9);
    let mut net = Sequential::new();
    for _ in 0..64 {
        let layer = Linear::new(&mut params, 1, 1, &mut rng);
        params.value_mut(layer.weight()).data_mut()[0] = 1.0 + rng.range(-0.01, 0.01) as f32;
        net = net.push(layer);
    }
    let x = Tensor::from_vec(vec![0.5f32, -2.0, 3.0], &[3, 1]).unwrap();
    let mut cpu = cavitation_nn::InferenceModel::compile(&net, &params).unwrap();
    let want = cpu.infer_batch(x.view()).data().to_vec();
    for_each_gpu(|gpu| {
        let mut model = GpuModel::compile(&net, &params, 3, gpu).unwrap();
        let got = model.infer_batch(x.view()).data().to_vec();
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (g - w).abs() <= 1e-5 * w.abs(),
                "{got:?} against the CPU's {want:?}"
            );
        }
    });
}
