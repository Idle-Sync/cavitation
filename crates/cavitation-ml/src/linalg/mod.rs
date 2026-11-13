//! Dense linear algebra in `f64`: least squares by Householder QR.
//!
//! [`Qr`] factors a tall matrix `a` (`[rows, cols]`, `rows ≥ cols`) as `Q·R` with
//! Householder reflections, and [`Qr::solve`] finds the `x` that minimises `‖a·x − b‖` for
//! each column of `b`. [`lstsq`] does both and refuses a matrix whose columns are not
//! independent. QR works on `a` itself rather than on `aᵀa`, so it loses only as many digits
//! as `a`'s condition number, where the normal equations lose twice that.
//!
//! [`qr_reference`] is the same solver written as plainly as possible: the shipped [`Qr`] is
//! tested against it.
//!
//! # Examples
//!
//! Fitting the line `y = c0 + c1·x` through (0, 1), (1, 2) and (2, 4):
//!
//! ```
//! use cavitation_core::Tensor;
//! use cavitation_ml::linalg::lstsq;
//! let a = Tensor::from_vec(vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0], &[3, 2]).unwrap();
//! let b = Tensor::from_vec(vec![1.0, 2.0, 4.0], &[3, 1]).unwrap();
//! let c = lstsq(a.view(), b.view()).unwrap();
//! assert!((c.data()[0] - 5.0 / 6.0).abs() < 1e-15);
//! assert!((c.data()[1] - 1.5).abs() < 1e-15);
//! ```

mod qr;
pub mod qr_reference;

pub use qr::{Qr, lstsq};

use cavitation_core::{Error, TensorView};

/// Checks that `a` can be factored: two-dimensional, at least as many rows as columns, every
/// value finite. Returns `(rows, cols)`.
fn matrix_dims(a: &TensorView<'_, f64>) -> Result<(usize, usize), Error> {
    let dims = a.shape().dims();
    let &[rows, cols] = dims else {
        return Err(Error::InvalidData(format!(
            "least squares needs a matrix of shape [rows, cols], found shape {dims:?}"
        )));
    };
    if cols > rows {
        return Err(Error::InvalidData(format!(
            "the matrix of shape [{rows}, {cols}] has more columns than rows, so its \
             least-squares solution is not unique"
        )));
    }
    if let Some(at) = a.data().iter().position(|v| !v.is_finite()) {
        return Err(Error::InvalidData(format!(
            "the matrix holds {} at row {}, column {}",
            a.data()[at],
            at / cols,
            at % cols
        )));
    }
    Ok((rows, cols))
}

/// Checks that `b` is a right-hand side for a matrix of `rows` rows: shape `[rows, k]`,
/// every value finite. Returns `k`.
fn rhs_columns(b: &TensorView<'_, f64>, rows: usize) -> Result<usize, Error> {
    let dims = b.shape().dims();
    let k = match *dims {
        [r, k] if r == rows => k,
        _ => {
            return Err(Error::InvalidData(format!(
                "the right-hand side must have shape [{rows}, k] to match the matrix's {rows} \
                 rows, found shape {dims:?}"
            )));
        }
    };
    if let Some(at) = b.data().iter().position(|v| !v.is_finite()) {
        return Err(Error::InvalidData(format!(
            "the right-hand side holds {} at row {}, column {}",
            b.data()[at],
            at / k,
            at % k
        )));
    }
    Ok(k)
}

/// Checks that no column norm overflowed: a column whose squared norm exceeds `f64::MAX`
/// cannot be factored without rescaling.
fn check_norms(norms: &[f64]) -> Result<(), Error> {
    match norms.iter().position(|n| !n.is_finite()) {
        Some(j) => Err(Error::InvalidData(format!(
            "the norm of column {j} overflows f64; rescale the column"
        ))),
        None => Ok(()),
    }
}

/// The first column `j` with `|R_jj| ≤ rows · ε · ‖a_j‖`. `|R_jj| / ‖a_j‖` is the sine of
/// the angle between column `j` and the span of the columns before it, so the test does not
/// depend on the columns' units. A zero column has norm 0 and `0 ≤ 0`, so it is dependent.
fn first_dependent(r_diagonal: &[f64], norms: &[f64], rows: usize) -> Option<usize> {
    let scale = rows as f64 * f64::EPSILON;
    r_diagonal
        .iter()
        .zip(norms)
        .position(|(r, n)| r.abs() <= scale * n)
}

/// The error [`lstsq`] returns for a dependent column `j`.
fn dependent_error(j: usize) -> Error {
    Error::InvalidData(format!(
        "column {j} is a linear combination of the columns before it, so the least-squares \
         solution is not unique"
    ))
}
