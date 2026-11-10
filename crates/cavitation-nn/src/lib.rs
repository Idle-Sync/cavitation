//! Layers, optimisers and training for the cavitation neural network library.

mod activation;
mod health;
mod hooks;
mod inference;
mod init;
mod io;
mod layer;
mod linear;
mod mask;
mod optim;
mod sequential;
mod train;

pub use activation::{
    Activation, ActivationKind, ActivationLayer, Celu, Elu, Gelu, Glu, Hardsigmoid, Hardswish,
    Hardtanh, LeakyRelu, LogSigmoid, LogSoftmax, Mish, Prelu, Relu, Relu6, Selu, Sigmoid, Silu,
    Softmax, Softplus, Softsign, Tanh,
};
pub use health::{Health, HealthMonitor, OnNonFinite, Warning};
pub use hooks::{HookId, Hooks, LayerEvent};
pub use inference::{InferenceLayer, InferenceModel};
pub use init::Init;
pub use io::{load, load_lenient, save};
pub use layer::Layer;
pub use linear::Linear;
pub use mask::NeuronMask;
pub use optim::{Adam, AdamW, GradStats, Optimizer, Sgd};
pub use sequential::{Sequential, Trace};
pub use train::{
    Batch, BatchView, Batcher, Dataset, EpochReport, FitConfig, History, Loss, OptimizerConfig,
    Targets, TargetsView, Trainer, accuracy, fit, fit_with_hooks,
};
