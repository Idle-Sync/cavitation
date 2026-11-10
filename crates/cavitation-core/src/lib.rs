//! Tensors, automatic differentiation and CPU kernels for the cavitation neural network library.

mod error;
pub mod kernels;
mod real;
mod rng;
mod shape;
mod tensor;

pub use error::Error;
pub use real::Real;
pub use rng::{Rng, SplitMix64};
pub use shape::Shape;
pub use tensor::{Tensor, TensorView, TensorViewMut};
