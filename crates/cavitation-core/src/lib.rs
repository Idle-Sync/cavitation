//! Tensors, automatic differentiation and CPU kernels for the cavitation neural network library.

mod error;
mod real;
mod shape;
mod tensor;

pub use error::Error;
pub use real::Real;
pub use shape::Shape;
pub use tensor::{Tensor, TensorView, TensorViewMut};
