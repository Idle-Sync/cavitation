//! A fast, glass-box neural network library for the CPU.
//!
//! This is the main crate of the cavitation workspace, the one to depend on. It
//! builds on the tensor and autograd core and the layers, optimisers and training
//! crates.
//!
//! The core and nn crates are re-exported by name (`cavitation::cavitation_core`,
//! `cavitation::cavitation_nn`), and [`prelude`] gathers the names most programs need.
//! With the `ml` feature, on by default, the classic machine learning crate is re-exported
//! too, as `cavitation::cavitation_ml`: least squares and regression in `f64`.
//!
//! # Three levels of use
//!
//! The library can be used at three levels, and each one is built from the one below it.
//!
//! **1. [`fit`](cavitation_nn::fit).** Give it a network, its parameters, a [`Dataset`]
//! and a [`FitConfig`]; it shuffles, batches, trains and checks the run's health. A
//! complete, small training run:
//!
//! ```
//! use cavitation::prelude::*;
//!
//! let (x, y) = (
//!     Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], &[4, 1]).unwrap(),
//!     Tensor::from_vec(vec![1.0f32, 3.0, 5.0, 7.0], &[4, 1]).unwrap(),
//! );
//! let data = Dataset { x, targets: Targets::Values(y) };
//! let mut params = ParamStore::new();
//! let net = Sequential::new().push(Linear::new(&mut params, 1, 1, &mut Rng::seed(0)));
//! let config = FitConfig {
//!     epochs: 200,
//!     batch: 4,
//!     optimizer: OptimizerConfig::Sgd { lr: 0.05, momentum: 0.0, weight_decay: 0.0 },
//!     loss: Loss::Mse,
//!     ..FitConfig::default()
//! };
//! let history = fit(&net, &mut params, &data, config).unwrap();
//! assert!(history.epochs[199].mean_loss < history.epochs[0].mean_loss);
//! println!("cavitation {}", cavitation::VERSION);
//! ```
//!
//! **2. A manual loop.** Each training step written out: open a [`Tape`] on the parameter
//! values, record the inputs, run the network forward, record the loss, run `backward`
//! into the gradients, drop the tape, step the optimiser and zero the gradients. Every
//! value on the tape can be read between these lines. This one trains a network on XOR:
//!
//! ```
//! use cavitation::prelude::*;
//!
//! let x = Tensor::from_vec(vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]).unwrap();
//! let y = Tensor::from_vec(vec![0.0f32, 1.0, 1.0, 0.0], &[4, 1]).unwrap();
//! let mut params = ParamStore::new();
//! let mut rng = Rng::seed(1);
//! let net = Sequential::new()
//!     .push(Linear::new(&mut params, 2, 8, &mut rng))
//!     .push(Tanh)
//!     .push(Linear::new(&mut params, 8, 1, &mut rng))
//!     .push(Sigmoid);
//! let mut optimizer = Sgd::new(0.5);
//! let mut graph = Graph::new();
//! let mut last = f32::INFINITY;
//! for _ in 0..1000 {
//!     // The tape reads `params.values`; `backward` writes `params.grads`.
//!     let mut tape = graph.tape(&params.values);
//!     let input = tape.input(x.view());
//!     let target = tape.input(y.view());
//!     let out = net.forward(&mut tape, input);
//!     let loss = tape.mse(out, target, Reduction::Mean);
//!     last = tape.value(loss).data()[0];
//!     tape.backward(loss, &mut params.grads);
//!     drop(tape); // releases `params.values` so the optimiser can change them
//!     optimizer.step(&mut params);
//!     params.zero_grad();
//! }
//! assert!(last < 0.01);
//! ```
//!
//! [`Trainer`]'s documentation shows the same loop with the health checks that `fit`
//! runs, and [`Hooks`] reads each layer's values and gradients from the tape.
//!
//! **3. Your own parts.** An [`Activation`] needs only its value and derivative; a
//! [`Layer`] records its forward pass from tape operations; a
//! [`CustomOp`](cavitation_core::CustomOp) is a new tape operation with its own forward
//! and backward. Each is generic over the number type, so
//! [`gradcheck`](cavitation_core::gradcheck) checks it in `f64`. A user-defined activation
//! in a network:
//!
//! ```
//! use cavitation::prelude::*;
//!
//! /// `x³`.
//! struct Cube;
//!
//! impl<T: Real> Activation<T> for Cube {
//!     fn name(&self) -> &str {
//!         "cube"
//!     }
//!     fn f(&self, x: T) -> T {
//!         x * x * x
//!     }
//!     // The derivative at x, given y = f(x).
//!     fn df(&self, x: T, _y: T) -> T {
//!         T::from_f64(3.0) * x * x
//!     }
//! }
//!
//! let mut params = ParamStore::new();
//! let net = Sequential::new()
//!     .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
//!     .push(ActivationLayer::new(Cube));
//! assert_eq!(net.layer(1).name(), "cube");
//! ```
//!
//! [`Dataset`]: cavitation_nn::Dataset
//! [`FitConfig`]: cavitation_nn::FitConfig
//! [`Tape`]: cavitation_core::Tape
//! [`Trainer`]: cavitation_nn::Trainer
//! [`Hooks`]: cavitation_nn::Hooks
//! [`Activation`]: cavitation_nn::Activation
//! [`Layer`]: cavitation_nn::Layer

pub use cavitation_core;
/// Classic machine learning in `f64`: least squares by Householder QR, and the regression
/// models built on it. Present with the `ml` feature, which is on by default.
///
/// ```
/// use cavitation::cavitation_core::Tensor;
/// use cavitation::cavitation_ml::linalg::lstsq;
/// let a = Tensor::from_vec(vec![1.0, 1.0], &[2, 1]).unwrap();
/// let b = Tensor::from_vec(vec![1.0, 3.0], &[2, 1]).unwrap();
/// assert!((lstsq(a.view(), b.view()).unwrap().data()[0] - 2.0).abs() < 1e-15);
/// ```
#[cfg(feature = "ml")]
pub use cavitation_ml;
pub use cavitation_nn;

/// The names most programs need, for `use cavitation::prelude::*;`: tensors and autograd,
/// layers and activations (with the settings their fields take), optimisers, the training
/// loop and its batches, evaluation, health checks, hooks, inference, saving and loading,
/// the random generator, the gradient checks, and the traits for writing your own parts.
///
/// ```
/// use cavitation::prelude::*;
/// let mut rng = Rng::seed(0);
/// let mut params = ParamStore::new();
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 4, &mut rng))
///     .push(Gelu { approx: GeluApprox::Tanh })
///     .push(Linear::new(&mut params, 4, 2, &mut rng))
///     .push(Glu { gate: Gate::Sigmoid });
/// assert_eq!(net.len(), 4);
///
/// // One `Trainer` step on a `Batch`, stopping on a non-finite value.
/// let data = Dataset { x: Tensor::zeros(&[2, 2]), targets: Targets::Values(Tensor::zeros(&[2, 1])) };
/// let mut trainer = Trainer::new(&net, Sgd::new(0.1), Loss::Mse, 2);
/// trainer.set_health(Health { on_non_finite: OnNonFinite::Stop, ..Health::default() });
/// let mut batch = Batch::new(&data, 2);
/// let view: BatchView<'_, f32> = Batcher::new(2, 2, None).gather(&data, 0..2, &mut batch);
/// assert!(trainer.step(&net, &mut params, view).unwrap().is_finite());
/// ```
pub mod prelude {
    pub use cavitation_core::{
        CustomOp, Error, Gate, GeluApprox, Graph, KeyErrors, ParamId, ParamStore, Real, Reduction,
        Rng, Tape, Tensor, TensorView, Var, gradcheck, gradcheck_params,
    };
    pub use cavitation_nn::{
        Activation, ActivationLayer, Adam, AdamW, BadInput, Batch, BatchView, Batcher, Celu,
        Dataset, Elu, EpochReport, Evaluation, Evaluator, FitConfig, Flow, Gelu, Glu, GradStats,
        Hardsigmoid, Hardswish, Hardtanh, Health, HealthMonitor, History, Hooks, InferenceModel,
        Init, Layer, LayerEvent, LeakyRelu, Linear, LoadOptions, LogSigmoid, LogSoftmax, Loss,
        LrSchedule, Mish, ModelHandle, NeuronMask, OnNonFinite, Optimizer, OptimizerConfig, Prelu,
        Relu, Relu6, Selu, Sequential, Sgd, Sigmoid, Silu, Softmax, Softplus, Softsign,
        Standardizer, StepControl, StepInfo, StepInputs, StepProfile, Tanh, Targets, TargetsView,
        Trace, Trainer, TrainerHandle, Warning, WarningKind, accuracy, evaluate, fit,
        fit_controlled, fit_with_hooks, load, load_lenient, load_with, save,
    };
}

/// The version of this crate, as set in `Cargo.toml`.
///
/// ```
/// assert_eq!(cavitation::VERSION, env!("CARGO_PKG_VERSION"));
/// ```
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The repository README, compiled only by `cargo test --doc` so that its Rust examples run
/// as tests and cannot drift from the library.
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
struct ReadmeDoctests;
