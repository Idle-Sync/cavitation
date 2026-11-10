//! A fast, glass-box neural network library for the CPU.
//!
//! This is the main crate of the cavitation workspace, the one to depend on. It
//! builds on the tensor and autograd core and the layers, optimisers and training
//! crates.
//!
//! Both crates are re-exported by name (`cavitation::cavitation_core`,
//! `cavitation::cavitation_nn`), and [`prelude`] gathers the names most programs need. A
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

pub use cavitation_core;
pub use cavitation_nn;

/// The names most programs need, for `use cavitation::prelude::*;`: tensors and autograd,
/// layers and activations, optimisers, the training loop, health checks, hooks, inference,
/// saving and loading, the random generator and the gradient checks.
///
/// ```
/// use cavitation::prelude::*;
/// let mut rng = Rng::seed(0);
/// let mut params = ParamStore::new();
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 4, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 4, 1, &mut rng))
///     .push(Sigmoid);
/// assert_eq!(net.len(), 4);
/// ```
pub mod prelude {
    pub use cavitation_core::{
        Error, Graph, ParamStore, Reduction, Rng, Tape, Tensor, Var, gradcheck, gradcheck_params,
    };
    pub use cavitation_nn::{
        Activation, Adam, AdamW, Batcher, Celu, Dataset, Elu, EpochReport, FitConfig, Gelu, Glu,
        Hardsigmoid, Hardswish, Hardtanh, Health, HealthMonitor, Hooks, InferenceModel, Init,
        Layer, LeakyRelu, Linear, LogSigmoid, LogSoftmax, Loss, Mish, NeuronMask, Optimizer,
        OptimizerConfig, Prelu, Relu, Relu6, Selu, Sequential, Sgd, Sigmoid, Silu, Softmax,
        Softplus, Softsign, Tanh, Targets, Trace, Trainer, Warning, accuracy, fit, load, save,
    };
}

/// The version of this crate, as set in `Cargo.toml`.
///
/// ```
/// assert_eq!(cavitation::VERSION, env!("CARGO_PKG_VERSION"));
/// ```
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
