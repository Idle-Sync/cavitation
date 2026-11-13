//! The reference Householder QR: the textbook algorithm, row-major, one accumulator per sum,
//! columns in order. The shipped [`super::Qr`] is tested against it.
//!
//! It has the same API and the same sign convention as the shipped solver (`R_jj =
//! −sign(a_jj)·‖x‖`, `+` for 0), and scales its norms and reflectors the same way, so the
//! two agree element by element, the diagonal of `R` included.
//!
//! # Examples
//!
//! ```
//! use cavitation_core::Tensor;
//! use cavitation_ml::linalg::{lstsq, qr_reference};
//! let a = Tensor::from_vec(vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0], &[3, 2]).unwrap();
//! let b = Tensor::from_vec(vec![1.0, 2.0, 4.0], &[3, 1]).unwrap();
//! let shipped = lstsq(a.view(), b.view()).unwrap();
//! let reference = qr_reference::lstsq(a.view(), b.view()).unwrap();
//! for (s, r) in shipped.data().iter().zip(reference.data()) {
//!     assert!((s - r).abs() < 1e-15);
//! }
//! ```

use cavitation_core::{Error, Tensor, TensorView};

use super::{
    check_columns, dependent_error, first_dependent, matrix_dims, power_of_two_below, rhs_columns,
    too_large_error,
};

/// The reference Householder QR factorisation of a tall `f64` matrix, `a = Q·R`.
///
/// # Examples
///
/// ```
/// use cavitation_core::Tensor;
/// use cavitation_ml::linalg::qr_reference::Qr;
/// let a = Tensor::from_vec(vec![3.0, 0.0, 4.0, 1.0, 0.0, 2.0], &[3, 2]).unwrap();
/// let qr = Qr::new(a.view()).unwrap();
/// assert_eq!(qr.r_diagonal()[0], -5.0);
/// ```
#[derive(Debug, Clone)]
pub struct Qr {
    /// Number of rows of `a`.
    rows: usize,
    /// Number of columns of `a`.
    cols: usize,
    /// The factors, row-major `[rows, cols]`: `R` above the diagonal and reflector `j`'s
    /// vector, scaled by a power of two, in column `j` from row `j` down.
    factors: Vec<f64>,
    /// Each reflector's `β` in `I − β·v·vᵀ`; 0 for the identity.
    betas: Vec<f64>,
    /// The diagonal of `R`.
    r_diagonal: Vec<f64>,
    /// The Euclidean norm of each input column.
    norms: Vec<f64>,
}

impl Qr {
    /// Factors `a`, of shape `[rows, cols]` with `rows ≥ cols`.
    ///
    /// # Errors
    ///
    /// The same as the shipped [`super::Qr::new`]: [`Error::InvalidData`] if `a` is not
    /// two-dimensional, has more columns than rows or holds a NaN or infinite value, or if a
    /// column's norm or an entry of `R` does not fit in f64.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::linalg::qr_reference::Qr;
    /// let wide = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();
    /// assert!(Qr::new(wide.view()).is_err());
    /// ```
    pub fn new(a: TensorView<'_, f64>) -> Result<Qr, Error> {
        let (rows, cols) = matrix_dims(&a)?;
        Qr::factor(a.data().to_vec(), rows, cols)
    }

    /// Factors a `[rows, cols]` matrix given column-major, like the shipped
    /// [`super::Qr::from_columns`]; this version copies it row-major first.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if `data` does not hold `rows · cols` values, and for the same
    /// problems as [`Qr::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_ml::linalg::qr_reference::Qr;
    /// let qr = Qr::from_columns(vec![3.0, 4.0], 2, 1).unwrap();
    /// assert_eq!(qr.r_diagonal(), &[-5.0]);
    /// ```
    pub fn from_columns(data: Vec<f64>, rows: usize, cols: usize) -> Result<Qr, Error> {
        check_columns(&data, rows, cols)?;
        let mut f = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                f[i * cols + j] = data[j * rows + i];
            }
        }
        Qr::factor(f, rows, cols)
    }

    /// Factors the checked row-major matrix `f` in place.
    fn factor(mut f: Vec<f64>, rows: usize, cols: usize) -> Result<Qr, Error> {
        let mut norms = vec![0.0; cols];
        for (j, norm) in norms.iter_mut().enumerate() {
            *norm = column_norm(&f, cols, j, 0..rows);
        }
        let mut betas = vec![0.0; cols];
        let mut r_diagonal = vec![0.0; cols];
        for j in 0..cols {
            let size = column_norm(&f, cols, j, j..rows);
            if size == 0.0 {
                continue;
            }
            // Scale the column exactly by the power of two at or below its norm, as the
            // shipped version does: n is in [1, 2) and β = 2 / vᵀv = 1 / (n·(n + |x0|)).
            let unit = 1.0 / power_of_two_below(size);
            for i in j..rows {
                f[i * cols + j] *= unit;
            }
            let (n, x0) = (size * unit, f[j * cols + j]);
            let sign = if x0 >= 0.0 { 1.0 } else { -1.0 };
            f[j * cols + j] = x0 + sign * n;
            let beta = 1.0 / (n * (n + x0.abs()));
            betas[j] = beta;
            r_diagonal[j] = -sign * size;
            for l in j + 1..cols {
                let mut s = 0.0;
                for i in j..rows {
                    s += f[i * cols + j] * f[i * cols + l];
                }
                let s = beta * s;
                for i in j..rows {
                    f[i * cols + l] -= s * f[i * cols + j];
                }
            }
        }
        for j in 0..cols {
            let mut finite = norms[j].is_finite() && r_diagonal[j].is_finite();
            for i in 0..j {
                finite = finite && f[i * cols + j].is_finite();
            }
            if !finite {
                return Err(too_large_error(j));
            }
        }
        Ok(Qr {
            rows,
            cols,
            factors: f,
            betas,
            r_diagonal,
            norms,
        })
    }

    /// The least-squares solution `x` (`[cols, k]`) of `a·x ≈ b` for `b` of shape
    /// `[rows, k]`.
    ///
    /// # Errors
    ///
    /// The same as the shipped [`super::Qr::solve`]: [`Error::InvalidData`] for a dependent
    /// column (naming it), for a `b` not of shape `[rows, k]` or holding a NaN or infinite
    /// value, and for a solution that does not fit in f64.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::linalg::qr_reference::Qr;
    /// let a = Tensor::from_vec(vec![2.0], &[1, 1]).unwrap();
    /// let b = Tensor::from_vec(vec![6.0], &[1, 1]).unwrap();
    /// assert_eq!(Qr::new(a.view()).unwrap().solve(b.view()).unwrap().data(), &[3.0]);
    /// ```
    pub fn solve(&self, b: TensorView<'_, f64>) -> Result<Tensor<f64>, Error> {
        if let Some(j) = self.dependent_column() {
            return Err(dependent_error(j));
        }
        let (rows, cols) = (self.rows, self.cols);
        let k = rhs_columns(&b, rows)?;
        let f = &self.factors;
        let mut c = b.data().to_vec();
        for j in 0..cols {
            if self.betas[j] == 0.0 {
                continue;
            }
            for q in 0..k {
                let mut s = 0.0;
                for i in j..rows {
                    s += f[i * cols + j] * c[i * k + q];
                }
                let s = self.betas[j] * s;
                for i in j..rows {
                    c[i * k + q] -= s * f[i * cols + j];
                }
            }
        }
        let mut x = vec![0.0; cols * k];
        for q in 0..k {
            for i in (0..cols).rev() {
                let mut s = c[i * k + q];
                for l in i + 1..cols {
                    s -= f[i * cols + l] * x[l * k + q];
                }
                x[i * k + q] = s / self.r_diagonal[i];
            }
        }
        for value in &x {
            if !value.is_finite() {
                return Err(Error::InvalidData(
                    "the least-squares solution does not fit in f64; rescale the matrix or \
                     the right-hand side"
                        .to_string(),
                ));
            }
        }
        Tensor::from_vec(x, &[cols, k])
    }

    /// The diagonal of `R`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::linalg::qr_reference::Qr;
    /// let a = Tensor::from_vec(vec![-3.0, 4.0], &[2, 1]).unwrap();
    /// assert_eq!(Qr::new(a.view()).unwrap().r_diagonal(), &[5.0]);
    /// ```
    pub fn r_diagonal(&self) -> &[f64] {
        &self.r_diagonal
    }

    /// The Euclidean norm of each column of the input matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::linalg::qr_reference::Qr;
    /// let a = Tensor::from_vec(vec![3.0, 4.0], &[2, 1]).unwrap();
    /// assert_eq!(Qr::new(a.view()).unwrap().column_norms(), &[5.0]);
    /// ```
    pub fn column_norms(&self) -> &[f64] {
        &self.norms
    }

    /// The first column `j` with `|R_jj| ≤ rows · f64::EPSILON · ‖a_j‖`, the same rule as
    /// the shipped [`super::Qr::dependent_column`]; a zero column counts as dependent.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::linalg::qr_reference::Qr;
    /// let a = Tensor::from_vec(vec![1.0, 0.0, 2.0, 0.0], &[2, 2]).unwrap();
    /// assert_eq!(Qr::new(a.view()).unwrap().dependent_column(), Some(1));
    /// ```
    pub fn dependent_column(&self) -> Option<usize> {
        first_dependent(&self.r_diagonal, &self.norms, self.rows)
    }
}

/// The norm of column `j` of the row-major matrix `f` (`cols` wide) over `rows`, scaled by
/// the power of two at or below the largest absolute value in that range, an exact scaling,
/// so that no square overflows or underflows.
fn column_norm(f: &[f64], cols: usize, j: usize, rows: std::ops::Range<usize>) -> f64 {
    let mut m = 0.0f64;
    for i in rows.clone() {
        m = m.max(f[i * cols + j].abs());
    }
    if m == 0.0 {
        return 0.0;
    }
    let scale = power_of_two_below(m);
    let mut s = 0.0;
    for i in rows {
        let y = f[i * cols + j] / scale;
        s += y * y;
    }
    f64::sqrt(s) * scale
}

/// The reference least-squares solution: [`Qr::new`], [`Qr::dependent_column`] and
/// [`Qr::solve`] of this module, with the errors of the shipped [`super::lstsq`].
///
/// # Errors
///
/// [`Error::InvalidData`] for bad shapes or values, and for a column that is a linear
/// combination of the columns before it.
///
/// # Examples
///
/// ```
/// use cavitation_core::Tensor;
/// use cavitation_ml::linalg::qr_reference;
/// let a = Tensor::from_vec(vec![1.0, 1.0], &[2, 1]).unwrap();
/// let b = Tensor::from_vec(vec![1.0, 3.0], &[2, 1]).unwrap();
/// let x = qr_reference::lstsq(a.view(), b.view()).unwrap();
/// assert!((x.data()[0] - 2.0).abs() < 1e-15);
/// ```
pub fn lstsq(a: TensorView<'_, f64>, b: TensorView<'_, f64>) -> Result<Tensor<f64>, Error> {
    let qr = Qr::new(a)?;
    if let Some(j) = qr.dependent_column() {
        return Err(dependent_error(j));
    }
    qr.solve(b)
}
