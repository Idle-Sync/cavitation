//! Tensors, automatic differentiation and CPU kernels for the cavitation neural network library.

mod check;
mod error;
mod graph;
pub mod kernels;
mod ops;
mod real;
mod rng;
mod shape;
mod tensor;

pub use check::{GradcheckFailure, GradcheckReport, gradcheck, gradcheck_params};
pub use error::Error;
pub use graph::{Graph, ParamGrads, ParamId, ParamStore, ParamValues, Tape, Var};
pub use ops::{Gate, GeluApprox, Reduction};
pub use real::Real;
pub use rng::{Rng, SplitMix64};
pub use shape::Shape;
pub use tensor::{Tensor, TensorView, TensorViewMut};
