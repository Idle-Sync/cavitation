//! Loads a CNN trained in PyTorch from its `state_dict` and checks that it gives PyTorch's
//! answers. The file is the CIFAR-10 network of the library's fixtures
//! (`crates/cavitation-nn/tests/fixtures/cnn/cifar-cnn.safetensors`): four convolutions
//! without a bias, each followed by a batch norm with its running statistics, two poolings,
//! dropout and two `Linear`s, saved from PyTorch in F16. The network is built here with
//! the same layout, the file is loaded into it, and its outputs on eight images are compared
//! with PyTorch's float64 outputs stored beside the file.
//!
//! The guide's chapter "Image models" walks through this file; the `ANCHOR` comments mark
//! the parts it includes.
//!
//! Run it with `cargo run --release --example load_torch_cnn -p cavitation`.

use std::path::Path;

use cavitation::prelude::*;

/// The PyTorch network, block by block. In PyTorch:
///
/// ```python
/// def block(i, o):
///     return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU()]
///
/// net = nn.Sequential(
///     *block(3, 32), *block(32, 32), nn.MaxPool2d(2),
///     *block(32, 64), *block(64, 64), nn.MaxPool2d(2),
///     nn.Dropout(0.25), nn.Flatten(), nn.Linear(4096, 256), nn.ReLU(), nn.Dropout(0.5),
///     nn.Linear(256, 10),
/// )
/// ```
fn cifar_network(params: &mut ParamStore) -> Result<Sequential, Error> {
    // ANCHOR: network
    let mut rng = Rng::seed(0); // the starting values are replaced by the file's
    let mut net = Sequential::new();
    for (i, o, size) in [(3, 32, 32), (32, 32, 32), (32, 64, 16), (64, 64, 16)] {
        let geom = Conv2dGeom::with_options(i, o, (size, size), (3, 3), (1, 1), (1, 1), (1, 1))?;
        net = net
            .push(Conv2d::with_init(
                params,
                geom,
                Init::TorchDefault,
                None,
                &mut rng,
            ))
            .push(BatchNorm2d::new(params, o, (size, size)))
            .push(Relu);
        if i == o {
            net = net.push(MaxPool2d::new(Pool2dGeom::new(o, (size, size), (2, 2))?));
        }
    }
    Ok(net
        .push(Dropout::new(0.25)?)
        .push(Flatten)
        .push(Linear::new(params, 4096, 256, &mut rng))
        .push(Relu)
        .push(Dropout::new(0.5)?)
        .push(Linear::new(params, 256, 10, &mut rng)))
    // ANCHOR_END: network
}

/// The values of tensor `name` in the safetensors `bytes`, as f64 (the file holds F32 or
/// F64).
fn read(bytes: &[u8], name: &str) -> Vec<f64> {
    let file = safetensors::SafeTensors::deserialize(bytes).expect("a valid file");
    let t = file.tensor(name).expect("the tensor is in the file");
    match t.dtype() {
        safetensors::Dtype::F32 => (t.data().as_chunks::<4>().0.iter())
            .map(|&b| f64::from(f32::from_le_bytes(b)))
            .collect(),
        safetensors::Dtype::F64 => (t.data().as_chunks::<8>().0.iter())
            .map(|&b| f64::from_le_bytes(b))
            .collect(),
        other => panic!("{name}: unexpected {other:?}"),
    }
}

/// Loads the state dict, prints what it holds, runs the eight images and compares.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../cavitation-nn/tests/fixtures/cnn");
    let file = dir.join("cifar-cnn.safetensors");

    // ANCHOR: inspect
    let tensors = inspect_file(&file)?;
    println!("{} tensors in the file, for example:", tensors.len());
    for t in tensors.iter().filter(|t| t.name.starts_with("1.")) {
        println!("  {:<22} {:<4} {:?}", t.name, t.dtype, t.shape);
    }
    // ANCHOR_END: inspect

    // ANCHOR: load
    let mut params = ParamStore::new();
    let net = cifar_network(&mut params)?;
    load(&file, &net, &mut params)?;
    // ANCHOR_END: load

    let io = std::fs::read(dir.join("cifar-cnn-io.safetensors"))?;
    let input: Vec<f32> = read(&io, "input").into_iter().map(|v| v as f32).collect();
    let torch = read(&io, "final_f64");

    // ANCHOR: run
    // `InferenceModel` runs dropout as the identity and batch norm with its running
    // statistics, as PyTorch's `model.eval()` does.
    let mut model = InferenceModel::compile(&net, &params)?;
    let images = TensorView::from_slice(&input, &[8, 3 * 32 * 32])?;
    let out = model.infer_batch(images);
    // ANCHOR_END: run

    let mut largest = 0.0f64;
    for (i, (ours, theirs)) in out.data().chunks(10).zip(torch.chunks(10)).enumerate() {
        let class = |v: &[f64]| (0..10).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap_or(0);
        let ours64: Vec<f64> = ours.iter().map(|&v| f64::from(v)).collect();
        for (a, b) in ours64.iter().zip(theirs) {
            largest = largest.max((a - b).abs());
        }
        println!(
            "image {i}: class {} here, {} in PyTorch",
            class(&ours64),
            class(theirs)
        );
    }
    println!("largest difference from PyTorch's float64 output: {largest:.1e}");
    Ok(())
}
