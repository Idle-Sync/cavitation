//! Helpers shared by the integration tests: the v0.3 CIFAR-10 benchmark network.

use cavitation_core::{Conv2dGeom, ParamStore, Pool2dGeom, Rng};
use cavitation_nn::{
    BatchNorm2d, Conv2d, Dropout, Flatten, Init, Linear, MaxPool2d, Relu, Sequential,
};

/// `Conv2d(i → o, 3 × 3, padding 1)` without a bias, `BatchNorm2d(o)` and ReLU on `size`
/// images, appended to `net`.
fn conv_block(
    net: Sequential,
    params: &mut ParamStore,
    rng: &mut Rng,
    (i, o): (usize, usize),
    size: (usize, usize),
) -> Sequential {
    let g = Conv2dGeom::with_options(i, o, size, (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
    net.push(Conv2d::with_init(params, g, Init::TorchDefault, None, rng))
        .push(BatchNorm2d::new(params, o, size))
        .push(Relu)
}

/// The v0.3 CIFAR-10 network on 3 × 32 × 32 images, with PyTorch's default initialisation
/// from `rng`: four 3 × 3 convolution blocks (32, 32, 64 and 64 channels, each with batch
/// norm and ReLU) with 2 × 2 max pooling after each pair, dropout 0.25, flatten,
/// `Linear(4096 → 256)`, ReLU, dropout 0.5 and `Linear(256 → 10)`.
pub fn cifar_cnn(params: &mut ParamStore, rng: &mut Rng) -> Sequential {
    let net = Sequential::new();
    let net = conv_block(net, params, rng, (3, 32), (32, 32));
    let net = conv_block(net, params, rng, (32, 32), (32, 32));
    let net = net.push(MaxPool2d::new(
        Pool2dGeom::new(32, (32, 32), (2, 2)).unwrap(),
    ));
    let net = conv_block(net, params, rng, (32, 64), (16, 16));
    let net = conv_block(net, params, rng, (64, 64), (16, 16));
    net.push(MaxPool2d::new(
        Pool2dGeom::new(64, (16, 16), (2, 2)).unwrap(),
    ))
    .push(Dropout::new(0.25).unwrap())
    .push(Flatten)
    .push(Linear::new(params, 4096, 256, rng))
    .push(Relu)
    .push(Dropout::new(0.5).unwrap())
    .push(Linear::new(params, 256, 10, rng))
}
