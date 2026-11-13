//! Dense linear algebra in `f64`: least squares by Householder QR.
//!
//! [`Qr`] factors a tall matrix `a` (`[rows, cols]`, `rows ≥ cols`) as `Q·R` with
//! Householder reflections, and [`Qr::solve`] finds the `x` that minimises `‖a·x − b‖` for
//! each column of `b`. [`lstsq`] does both. Both refuse a matrix whose columns are not
//! independent, naming the first column that depends on the ones before it. QR works on `a`
//! itself rather than on `aᵀa`, so it loses only as many digits as `a`'s condition number,
//! where the normal equations lose twice that. Norms and reflectors are scaled by each
//! column's size, so columns from about 1e-300 to 1e300 factor without overflow or underflow.
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
    check_tall(rows, cols)?;
    if let Some(at) = a.data().iter().position(|v| !v.is_finite()) {
        return Err(non_finite_error(a.data()[at], at / cols, at % cols));
    }
    Ok((rows, cols))
}

/// Checks that `data` holds a `[rows, cols]` matrix stored column-major that can be
/// factored: `rows · cols` values (else [`Error::ShapeMismatch`], as `Tensor::from_vec`
/// reports a wrong length), at least as many rows as columns, every value finite.
fn check_columns(data: &[f64], rows: usize, cols: usize) -> Result<(), Error> {
    if Some(data.len()) != rows.checked_mul(cols) {
        return Err(Error::ShapeMismatch {
            expected: vec![rows, cols],
            found: vec![data.len()],
        });
    }
    check_tall(rows, cols)?;
    if let Some(at) = data.iter().position(|v| !v.is_finite()) {
        return Err(non_finite_error(data[at], at % rows, at / rows));
    }
    Ok(())
}

/// Checks that a `[rows, cols]` matrix has at least as many rows as columns.
fn check_tall(rows: usize, cols: usize) -> Result<(), Error> {
    if cols > rows {
        return Err(Error::InvalidData(format!(
            "the matrix of shape [{rows}, {cols}] has more columns than rows, so its \
             least-squares solution is not unique"
        )));
    }
    Ok(())
}

/// The error for a NaN or infinite `value` at `row`, `col` of the matrix.
fn non_finite_error(value: f64, row: usize, col: usize) -> Error {
    Error::InvalidData(format!(
        "the matrix holds {value} at row {row}, column {col}"
    ))
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

/// The error for column `j` when its factorisation overflows f64. The norms and the
/// reflectors are scaled, so this happens only for a column whose norm is within a small
/// factor (up to about 4) of `f64::MAX`: its norm, an entry of `R`, or an intermediate of a
/// reflection applied to it does not fit in f64.
fn too_large_error(j: usize) -> Error {
    Error::InvalidData(format!(
        "column {j} holds values too close to the largest f64 to factor; rescale the column"
    ))
}

/// The power of two `2^e` with `2^e ≤ m < 2^(e+1)`, for `m > 0`; `f64::MIN_POSITIVE` for a
/// subnormal `m`. Dividing or multiplying by it, or by its reciprocal (also representable),
/// is exact, so scaling a column by it changes no digit, only the exponent: the scaled
/// algorithm rounds exactly as the unscaled one wherever the unscaled one does not overflow
/// or underflow.
fn power_of_two_below(m: f64) -> f64 {
    const EXPONENT: u64 = 0x7ff0_0000_0000_0000;
    f64::from_bits(m.max(f64::MIN_POSITIVE).to_bits() & EXPONENT)
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

/// The error [`lstsq`] and [`Qr::solve`] return for a dependent column `j`.
fn dependent_error(j: usize) -> Error {
    Error::InvalidData(format!(
        "column {j} is a linear combination of the columns before it, so the least-squares \
         solution is not unique"
    ))
}

#[cfg(test)]
mod tests {
    //! Unit tests for the shared helpers.

    use super::*;

    /// `power_of_two_below` gives the power of two at or below a positive value, and
    /// `f64::MIN_POSITIVE` for a subnormal one.
    #[test]
    fn power_of_two_below_values() {
        assert_eq!(power_of_two_below(1.0), 1.0);
        assert_eq!(power_of_two_below(3.0), 2.0);
        assert_eq!(power_of_two_below(0.75), 0.5);
        assert_eq!(power_of_two_below(f64::MAX), 2f64.powi(1023));
        assert_eq!(power_of_two_below(5e-324), f64::MIN_POSITIVE);
        assert!(1.0 / power_of_two_below(f64::MAX) > 0.0);
    }
}
