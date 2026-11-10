//! Layers, optimisers and training for the cavitation neural network library.

mod activation;
mod init;
mod layer;
mod linear;
mod sequential;

pub use activation::{
    Activation, ActivationKind, ActivationLayer, Celu, Elu, Gelu, Glu, Hardsigmoid, Hardswish,
    Hardtanh, LeakyRelu, LogSigmoid, LogSoftmax, Mish, Prelu, Relu, Relu6, Selu, Sigmoid, Silu,
    Softmax, Softplus, Softsign, Tanh,
};
pub use init::Init;
pub use layer::Layer;
pub use linear::Linear;
pub use sequential::{Sequential, Trace};
