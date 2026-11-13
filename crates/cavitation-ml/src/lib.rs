//! Classic machine learning for the cavitation library, computed in `f64`.
//!
//! Neural networks are not always the right tool. When a straight line, a plane or a
//! weighted sum explains the data, a classic model fits it exactly, in one step, and its
//! coefficients can be read directly. This crate holds those models and the numerical
//! methods under them.
//!
//! What is here:
//!
//! - [`linalg`]: least squares by Householder QR. [`linalg::lstsq`] finds the `x` that
//!   makes `a·x` as close as possible to `b`, and refuses a matrix whose columns are not
//!   independent, naming the column, instead of returning one answer out of many.
//!   [`linalg::Qr`] gives the factorisation itself, and [`linalg::qr_reference`] is the same
//!   solver written as plainly as possible, which the fast one is tested against.
//! - [`LinearRegression`]: ordinary least squares and ridge regression, matching
//!   scikit-learn's `LinearRegression` and `Ridge`. [`LinearRegression::fit`] solves exactly
//!   by QR; [`LinearRegression::fit_gradient`] trains a `Linear` layer to the same answer by
//!   gradient descent, as [`GradientConfig`] sets out. Both return a [`LinearModel`], which
//!   predicts, scores R² and converts to and from a one-layer `Sequential` that saves as a
//!   PyTorch `nn.Linear`.
//!
//! Everything here works in `f64`, on [`Tensor`](cavitation_core::Tensor)s and
//! [`TensorView`](cavitation_core::TensorView)s from `cavitation-core`, and returns
//! [`Error::InvalidData`](cavitation_core::Error::InvalidData) for data it cannot use. The
//! `cavitation` crate re-exports this one as `cavitation::cavitation_ml`.
//!
//! # Examples
//!
//! Fitting `y = c0 + c1·x` to four points that lie close to `y = 1 + 2x`:
//!
//! ```
//! use cavitation_core::Tensor;
//! use cavitation_ml::linalg::lstsq;
//! let xs = [0.0, 1.0, 2.0, 3.0];
//! let ys = [1.1, 2.9, 5.0, 7.1];
//! // One row per point: a column of ones for c0, then x for c1.
//! let a: Vec<f64> = xs.iter().flat_map(|&x| [1.0, x]).collect();
//! let a = Tensor::from_vec(a, &[4, 2]).unwrap();
//! let b = Tensor::from_vec(ys.to_vec(), &[4, 1]).unwrap();
//! let c = lstsq(a.view(), b.view()).unwrap();
//! assert!((c.data()[0] - 1.0).abs() < 0.1);
//! assert!((c.data()[1] - 2.0).abs() < 0.1);
//! ```

mod gradient;
pub mod linalg;
mod linear;

pub use gradient::GradientConfig;
pub use linear::{LinearModel, LinearRegression};
