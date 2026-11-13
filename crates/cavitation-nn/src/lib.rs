//! Layers, optimisers and training for the cavitation neural network library.
//!
//! This crate builds on `cavitation-core`'s tensors and tape; the `cavitation` crate
//! re-exports both with a prelude, and its documentation shows the three levels of use:
//! [`fit`], a manual loop, and your own parts.
//!
//! What is here:
//!
//! - Layers: the [`Layer`] trait, [`Linear`], a layer for every built-in activation,
//!   [`Activation`] and [`ActivationLayer`] for your own, and [`Sequential`] to chain them.
//! - Optimisers: [`Sgd`], [`Adam`] and [`AdamW`] behind the [`Optimizer`] trait.
//! - Training: [`Dataset`] and [`Batcher`], [`Trainer`] for one step at a time, and [`fit`]
//!   for whole runs, with the [`Health`] checks on every step; [`Evaluator`] and [`evaluate`]
//!   for the loss and accuracy or R² on held-out data, and [`Dataset::split_at`] to hold it
//!   out.
//! - Looking inside: [`Hooks`] (each layer's values and gradients), [`NeuronMask`] (turn
//!   neurons off) and [`Trace`].
//! - [`InferenceModel`], for answers without the training graph (with each layer's output
//!   on request, and masks changed per instance), and [`save`] and [`load`] for safetensors
//!   files named like a PyTorch `state_dict`.
//! - [`shrink`]: shrinking a trained network, with [`shrink::activity`], each neuron's mean,
//!   mean absolute value, firing share and maximum over a data set, and [`shrink::compact`],
//!   which removes switched-off neurons into smaller dense layers, and [`shrink::ablate`],
//!   which ranks neurons by what switching each one off costs.
//!
//! # Examples
//!
//! A 2 → 8 → 1 network learns `y = x₀ − x₁` with [`fit`], then answers through an
//! [`InferenceModel`]:
//!
//! ```
//! use cavitation_core::{ParamStore, Rng, Tensor};
//! use cavitation_nn::{
//!     Dataset, FitConfig, InferenceModel, Linear, Loss, OptimizerConfig, Sequential, Tanh,
//!     Targets, fit,
//! };
//! let mut rng = Rng::seed(0);
//! let x: Vec<f32> = (0..64).map(|_| rng.range(-1.0, 1.0) as f32).collect();
//! let y: Vec<f32> = x.chunks(2).map(|r| r[0] - r[1]).collect();
//! let data = Dataset {
//!     x: Tensor::from_vec(x, &[32, 2]).unwrap(),
//!     targets: Targets::Values(Tensor::from_vec(y, &[32, 1]).unwrap()),
//! };
//! let mut params = ParamStore::new();
//! let net = Sequential::new()
//!     .push(Linear::new(&mut params, 2, 8, &mut rng))
//!     .push(Tanh)
//!     .push(Linear::new(&mut params, 8, 1, &mut rng));
//! let config = FitConfig {
//!     epochs: 200,
//!     batch: 8,
//!     optimizer: OptimizerConfig::Adam { lr: 0.01, weight_decay: 0.0 },
//!     loss: Loss::Mse,
//!     ..FitConfig::default()
//! };
//! let history = fit(&net, &mut params, &data, config).unwrap();
//! assert!(history.epochs[199].mean_loss < 0.01 * history.epochs[0].mean_loss);
//! let mut model = InferenceModel::compile(&net, &params).unwrap();
//! let answer = model.infer(&[0.5, 0.25])[0];
//! assert!((answer - 0.25).abs() < 0.1, "{answer}");
//! ```

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
pub mod shrink;
#[cfg(test)]
mod test_child;
mod train;

pub use activation::{
    Activation, ActivationKind, ActivationLayer, Celu, Elu, Gelu, Glu, Hardsigmoid, Hardswish,
    Hardtanh, LeakyRelu, LogSigmoid, LogSoftmax, Mish, Prelu, Relu, Relu6, Selu, Sigmoid, Silu,
    Softmax, Softplus, Softsign, Tanh,
};
pub use health::{BadInput, Health, HealthMonitor, OnNonFinite, StepInputs, Warning, WarningKind};
pub use hooks::{HookId, Hooks, LayerEvent};
pub use inference::{InferenceLayer, InferenceModel};
pub use init::Init;
pub use io::{LoadOptions, load, load_lenient, load_with, save};
pub use layer::Layer;
pub use linear::Linear;
pub use mask::NeuronMask;
pub use optim::{Adam, AdamW, GradStats, Optimizer, Sgd};
pub use sequential::{Sequential, Trace};
pub use train::{
    Batch, BatchView, Batcher, Dataset, EpochReport, Evaluation, Evaluator, FitConfig, Flow,
    History, Loss, LrSchedule, OptimizerConfig, Targets, TargetsView, Trainer, accuracy, evaluate,
    fit, fit_with_hooks,
};
