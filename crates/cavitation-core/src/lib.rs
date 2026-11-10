//! Tensors, automatic differentiation and CPU kernels for the cavitation neural network library.
//!
//! This crate is the bottom layer. `cavitation-nn` builds layers, optimisers and training on
//! it, and the `cavitation` crate re-exports both with a prelude; depend on `cavitation`
//! unless you need only this layer. Its documentation shows the library's three levels of
//! use, from `fit` down to your own operations.
//!
//! What is here:
//!
//! - [`Tensor`], its views [`TensorView`] and [`TensorViewMut`], and [`Shape`]: row-major
//!   arrays of a [`Real`] number type (`f32` or `f64`).
//! - Reverse-mode automatic differentiation: a [`Graph`] holds the storage, a [`Tape`]
//!   records one pass of operations on [`Var`]s, and [`Tape::backward`] writes the
//!   gradients of the parameters in a [`ParamStore`]. Every operation passes the numerical
//!   gradient check and matches a PyTorch fixture.
//! - [`CustomOp`], for an operation of your own with its own forward and backward.
//! - [`gradcheck`] and [`gradcheck_params`], the numerical gradient check in `f64`.
//! - [`kernels`]: the matrix products and their reference twins, the kernel level, the
//!   thread count and the thread pool.
//! - [`Rng`], a seeded random generator, and [`Error`].
//!
//! # Examples
//!
//! The gradient of `sum(w · x)` with respect to `w` is `x`:
//!
//! ```
//! use cavitation_core::{Graph, ParamStore, Tensor, gradcheck};
//! let mut params = ParamStore::new();
//! let w = params.add(Tensor::from_vec(vec![0.5f32, -1.0, 2.0], &[3]).unwrap());
//! let x = Tensor::from_vec(vec![3.0f32, 4.0, 5.0], &[3]).unwrap();
//! let mut graph = Graph::new();
//! let mut tape = graph.tape(&params.values);
//! let (wv, xv) = (tape.param(w), tape.input(x.view()));
//! let product = tape.mul(wv, xv);
//! let loss = tape.sum(product);
//! assert_eq!(tape.value(loss).data(), &[7.5]);
//! tape.backward(loss, &mut params.grads);
//! drop(tape);
//! assert_eq!(params.grad(w).data(), &[3.0, 4.0, 5.0]);
//!
//! // The same computation passes the numerical check in f64.
//! let x64 = Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();
//! let w64 = Tensor::from_vec(vec![0.5, -1.0, 2.0], &[3]).unwrap();
//! gradcheck(&[w64, x64], |tape, v| {
//!     let p = tape.mul(v[0], v[1]);
//!     tape.sum(p)
//! })
//! .assert_passed();
//! ```

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
pub use ops::{CustomOp, Gate, GeluApprox, Reduction, log_softmax_row, softmax_row};
pub use real::Real;
pub use rng::{Rng, SplitMix64};
pub use shape::Shape;
pub use tensor::{Tensor, TensorView, TensorViewMut};
