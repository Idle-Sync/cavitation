//! The shipped Householder QR: column-major storage and sums in 4 partial sums, so the
//! compiler vectorises the inner loops. Tested against [`super::qr_reference`].

use cavitation_core::{Error, Tensor, TensorView};

use super::{
    check_columns, dependent_error, first_dependent, matrix_dims, power_of_two_below, rhs_columns,
    too_large_error,
};

/// The Householder QR factorisation of a tall `f64` matrix, `a = Q·R`, for solving least
/// squares.
///
/// Reflection `j` maps column `j`, from row `j` down, onto `R_jj·e_j`, with
/// `R_jj = −sign(a_jj)·‖x‖` (`+` sign for 0) so that forming the reflector never subtracts
/// nearly equal numbers. `Q` is kept as its reflectors and never formed.
///
/// Every norm and every reflector is computed on its column scaled by a power of two near
/// the column's size, so columns of any magnitude f64 can hold, from about 1e-300 to 1e300,
/// factor without overflow or underflow. The scaling is exact, so it changes no rounding.
///
/// # Examples
///
/// ```
/// use cavitation_core::Tensor;
/// use cavitation_ml::linalg::Qr;
/// let a = Tensor::from_vec(vec![3.0, 0.0, 4.0, 1.0, 0.0, 2.0], &[3, 2]).unwrap();
/// let qr = Qr::new(a.view()).unwrap();
/// assert_eq!(qr.column_norms()[0], 5.0);
/// assert_eq!(qr.r_diagonal()[0], -5.0);
/// assert_eq!(qr.dependent_column(), None);
/// ```
#[derive(Debug, Clone)]
pub struct Qr {
    /// Number of rows of `a`.
    rows: usize,
    /// Number of columns of `a`.
    cols: usize,
    /// The factors, column-major, `rows` values per column. Column `j` holds `R[0..j, j]` in
    /// rows `0..j` and reflector `j`'s vector `v` in rows `j..rows`.
    factors: Vec<f64>,
    /// Each reflector's `β`, so that it is `I − β·v·vᵀ`: between 1/8 and 1 when the
    /// column's norm from row `j` down is a normal f64, larger when it is subnormal, and 0
    /// for a column already zero from row `j` down, whose reflector is the identity.
    betas: Vec<f64>,
    /// The diagonal of `R`.
    r_diagonal: Vec<f64>,
    /// The Euclidean norm of each column of `a`, before any reflection.
    norms: Vec<f64>,
}

impl Qr {
    /// Factors `a`, of shape `[rows, cols]` with `rows ≥ cols`.
    ///
    /// The factorisation succeeds for dependent columns too; [`Qr::dependent_column`] finds
    /// them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidData`] if `a` is not two-dimensional, has more columns than
    /// rows or holds a NaN or infinite value, and, naming the column, if a column's norm is
    /// within a small factor (up to about 4) of `f64::MAX`, so that its norm, an entry of
    /// `R`, or an intermediate of a reflection applied to it does not fit in f64.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::linalg::Qr;
    /// let wide = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    /// let err = Qr::new(wide.view()).unwrap_err();
    /// assert!(err.to_string().contains("more columns than rows"));
    /// ```
    pub fn new(a: TensorView<'_, f64>) -> Result<Qr, Error> {
        let (rows, cols) = matrix_dims(&a)?;
        let mut columns = vec![0.0; rows * cols];
        for (i, row) in a.data().chunks_exact(cols.max(1)).enumerate().take(rows) {
            for (j, &value) in row.iter().enumerate() {
                columns[j * rows + i] = value;
            }
        }
        Qr::factor(columns, rows, cols)
    }

    /// Factors a `[rows, cols]` matrix given column-major, as `rows` values of column 0,
    /// then column 1, and so on, without copying it. For a matrix built column by column,
    /// such as one with weighted rows or added columns, this saves the transposing copy
    /// [`Qr::new`] makes; the factorisation is the same, bit for bit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ShapeMismatch`] (`expected` `[rows, cols]`, `found` the length) if
    /// `data` does not hold `rows · cols` values, as [`Tensor::from_vec`] does, and
    /// [`Error::InvalidData`] for the same problems as [`Qr::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::linalg::Qr;
    /// // The columns (1, 1, 1) and (0, 1, 2), the same matrix as [[1, 0], [1, 1], [1, 2]].
    /// let qr = Qr::from_columns(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2).unwrap();
    /// let b = Tensor::from_vec(vec![1.0, 2.0, 4.0], &[3, 1]).unwrap();
    /// let x = qr.solve(b.view()).unwrap();
    /// assert!((x.data()[1] - 1.5).abs() < 1e-15);
    /// ```
    pub fn from_columns(data: Vec<f64>, rows: usize, cols: usize) -> Result<Qr, Error> {
        check_columns(&data, rows, cols)?;
        Qr::factor(data, rows, cols)
    }

    /// Factors the checked column-major matrix `factors` in place.
    fn factor(mut factors: Vec<f64>, rows: usize, cols: usize) -> Result<Qr, Error> {
        let norms: Vec<f64> = (0..cols)
            .map(|j| norm(&factors[j * rows..(j + 1) * rows]))
            .collect();
        let mut betas = vec![0.0; cols];
        let mut r_diagonal = vec![0.0; cols];
        for j in 0..cols {
            let (done, later) = factors.split_at_mut((j + 1) * rows);
            let v = &mut done[j * rows + j..];
            let size = norm(v);
            if size == 0.0 {
                continue;
            }
            // Work on x·2^−e, with 2^e the power of two at or below ‖x‖: an exact scaling
            // that brings the norm n into [1, 2). Then v = x·2^−e − α·e0 with α = −sign·n,
            // and β = 2 / vᵀv = 1 / (n·(n + |x0|)) lies between 1/8 and 1. For a subnormal
            // ‖x‖ the scale stops at 2^−1022, so n < 1 and β is larger, still finite.
            let unit = 1.0 / power_of_two_below(size);
            for value in v.iter_mut() {
                *value *= unit;
            }
            let (n, x0) = (size * unit, v[0]);
            let sign = if x0 >= 0.0 { 1.0 } else { -1.0 };
            v[0] = x0 + sign * n;
            betas[j] = 1.0 / (n * (n + x0.abs()));
            r_diagonal[j] = -sign * size;
            for column in later.chunks_exact_mut(rows) {
                reflect(v, betas[j], &mut column[j..]);
            }
        }
        let bad = |values: &[f64]| values.iter().any(|v| !v.is_finite());
        if let Some(j) = (0..cols).find(|&j| {
            bad(&factors[j * rows..j * rows + j])
                || !r_diagonal[j].is_finite()
                || !norms[j].is_finite()
        }) {
            return Err(too_large_error(j));
        }
        Ok(Qr {
            rows,
            cols,
            factors,
            betas,
            r_diagonal,
            norms,
        })
    }

    /// The least-squares solution `x` (`[cols, k]`) of `a·x ≈ b` for `b` of shape
    /// `[rows, k]`: column `q` of `x` minimises `‖a·x_q − b_q‖`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidData`] if a column of `a` is a linear combination of the
    /// columns before it ([`Qr::dependent_column`]), naming the column; if `b` is not of
    /// shape `[rows, k]` or holds a NaN or infinite value; and if the solution, or a step
    /// on the way to it (`Qᵀb` or back substitution), overflows f64.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::linalg::Qr;
    /// // Two right-hand sides at once: the mean of each column of b.
    /// let a = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3, 1]).unwrap();
    /// let b = Tensor::from_vec(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0], &[3, 2]).unwrap();
    /// let x = Qr::new(a.view()).unwrap().solve(b.view()).unwrap();
    /// assert_eq!(x.shape().dims(), &[1, 2]);
    /// assert!((x.data()[0] - 2.0).abs() < 1e-15 && (x.data()[1] - 20.0).abs() < 1e-14);
    ///
    /// // A repeated column has no unique solution.
    /// let a = Tensor::from_vec(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], &[3, 2]).unwrap();
    /// let err = Qr::new(a.view()).unwrap().solve(b.view()).unwrap_err();
    /// assert!(err.to_string().contains("column 1 is a linear combination"));
    /// ```
    pub fn solve(&self, b: TensorView<'_, f64>) -> Result<Tensor<f64>, Error> {
        if let Some(j) = self.dependent_column() {
            return Err(dependent_error(j));
        }
        let (rows, cols) = (self.rows, self.cols);
        let k = rhs_columns(&b, rows)?;
        let data = b.data();
        let mut out = vec![0.0; cols * k];
        let mut c = vec![0.0; rows];
        for q in 0..k {
            for (i, value) in c.iter_mut().enumerate() {
                *value = data[i * k + q];
            }
            // c ← Qᵀ·b_q, one reflector at a time.
            for j in 0..cols {
                if self.betas[j] != 0.0 {
                    let v = &self.factors[j * rows + j..(j + 1) * rows];
                    reflect(v, self.betas[j], &mut c[j..]);
                }
            }
            // R·x = c[0..cols], by columns: once x_l is known, take its share out of the
            // rows above it, a contiguous update down column l of R.
            for l in (0..cols).rev() {
                let x = c[l] / self.r_diagonal[l];
                c[l] = x;
                axpy(-x, &self.factors[l * rows..l * rows + l], &mut c[..l]);
            }
            for (i, &x) in c[..cols].iter().enumerate() {
                out[i * k + q] = x;
            }
        }
        if out.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidData(
                "the least-squares solution overflows f64 while solving; rescale the matrix or \
                 the right-hand side"
                    .to_string(),
            ));
        }
        Tensor::from_vec(out, &[cols, k])
    }

    /// The diagonal of `R`, one value per column. Its signs follow the convention described
    /// on [`Qr`]; a column already zero from its diagonal down gives 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::linalg::Qr;
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
    /// use cavitation_ml::linalg::Qr;
    /// let a = Tensor::from_vec(vec![3.0, 1.0, 4.0, 0.0], &[2, 2]).unwrap();
    /// assert_eq!(Qr::new(a.view()).unwrap().column_norms(), &[5.0, 1.0]);
    /// ```
    pub fn column_norms(&self) -> &[f64] {
        &self.norms
    }

    /// The first column `j` that is, to working precision, a linear combination of the
    /// columns before it: `|R_jj| ≤ rows · f64::EPSILON · ‖a_j‖`, with `‖a_j‖` the column's
    /// norm in the input. `|R_jj| / ‖a_j‖` is the sine of the angle between column `j` and
    /// the columns before it, so the test does not depend on the columns' units or scale. A
    /// zero column (norm 0, and `0 ≤ 0`) counts as dependent. `None` when every column is
    /// independent.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_ml::linalg::Qr;
    /// // Column 1 is twice column 0.
    /// let a = Tensor::from_vec(vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], &[3, 2]).unwrap();
    /// assert_eq!(Qr::new(a.view()).unwrap().dependent_column(), Some(1));
    /// ```
    pub fn dependent_column(&self) -> Option<usize> {
        first_dependent(&self.r_diagonal, &self.norms, self.rows)
    }
}

/// The least-squares solution `x` (`[cols, k]`) of `a·x ≈ b`, for `a` of shape
/// `[rows, cols]` with `rows ≥ cols` and `b` of shape `[rows, k]`: [`Qr::new`], then
/// [`Qr::solve`], which refuses dependent columns ([`Qr::dependent_column`]).
///
/// # Errors
///
/// Returns [`Error::InvalidData`] for the shape and value errors of [`Qr::new`] and
/// [`Qr::solve`], and, naming the column, when a column is a linear combination of the
/// columns before it, so that the solution is not unique.
///
/// # Examples
///
/// ```
/// use cavitation_core::Tensor;
/// use cavitation_ml::linalg::lstsq;
/// let a = Tensor::from_vec(vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0], &[3, 2]).unwrap();
/// let b = Tensor::from_vec(vec![1.0, 2.0, 4.0], &[3, 1]).unwrap();
/// let x = lstsq(a.view(), b.view()).unwrap();
/// assert!((x.data()[1] - 1.5).abs() < 1e-15);
///
/// // Column 2 is column 0 plus column 1.
/// let a = Tensor::from_vec(vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 3.0], &[3, 3]).unwrap();
/// let err = lstsq(a.view(), b.view()).unwrap_err();
/// assert!(err.to_string().contains("column 2 is a linear combination"));
/// ```
pub fn lstsq(a: TensorView<'_, f64>, b: TensorView<'_, f64>) -> Result<Tensor<f64>, Error> {
    Qr::new(a)?.solve(b)
}

/// Applies the reflector `I − β·v·vᵀ` to `y`, which has `v`'s length.
fn reflect(v: &[f64], beta: f64, y: &mut [f64]) {
    let s = beta * dot(v, y);
    axpy(-s, v, y);
}

/// `xᵀy` over `x`'s length, in 4 partial sums over interleaved elements so that the
/// additions do not wait on each other and the compiler can vectorise them.
fn dot(x: &[f64], y: &[f64]) -> f64 {
    let y = &y[..x.len()];
    let (x4, x_tail) = x.as_chunks::<4>();
    let (y4, y_tail) = y.as_chunks::<4>();
    let mut s = [0.0f64; 4];
    for (a, b) in x4.iter().zip(y4) {
        s[0] += a[0] * b[0];
        s[1] += a[1] * b[1];
        s[2] += a[2] * b[2];
        s[3] += a[3] * b[3];
    }
    let mut tail = 0.0;
    for (a, b) in x_tail.iter().zip(y_tail) {
        tail += a * b;
    }
    ((s[0] + s[1]) + (s[2] + s[3])) + tail
}

/// The Euclidean norm of `x`, computed on `x` divided by the power of two at or below its
/// largest `|x_i|` (an exact scaling, as LAPACK's `dnrm2` scales), so that no square
/// overflows or underflows. The sum runs in 4 partial sums like [`dot`]. 0 for an empty or
/// zero `x`.
fn norm(x: &[f64]) -> f64 {
    let m = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if m == 0.0 {
        return 0.0;
    }
    let scale = power_of_two_below(m);
    let unit = 1.0 / scale;
    let (x4, tail) = x.as_chunks::<4>();
    let mut s = [0.0f64; 4];
    for a in x4 {
        let (a0, a1, a2, a3) = (a[0] * unit, a[1] * unit, a[2] * unit, a[3] * unit);
        s[0] += a0 * a0;
        s[1] += a1 * a1;
        s[2] += a2 * a2;
        s[3] += a3 * a3;
    }
    let mut rest = 0.0;
    for a in tail {
        let a = a * unit;
        rest += a * a;
    }
    (((s[0] + s[1]) + (s[2] + s[3])) + rest).sqrt() * scale
}

/// `y ← y + c·x` over `x`'s length.
fn axpy(c: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += c * xi;
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the shipped QR's building blocks and storage.

    use super::*;

    /// The 4-way dot product equals the plain sum on integer data, where every order of
    /// addition is exact, for every length up to a few chunks past the remainder cases.
    #[test]
    fn dot_matches_plain_sum() {
        for n in 0..13 {
            let x: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
            let y: Vec<f64> = (0..n).map(|i| 2.0 - i as f64).collect();
            let plain: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
            assert_eq!(dot(&x, &y), plain, "length {n}");
        }
    }

    /// `dot` reads only `x`'s length of a longer `y`.
    #[test]
    fn dot_uses_the_first_slices_length() {
        assert_eq!(dot(&[1.0, 2.0], &[3.0, 4.0, 100.0]), 11.0);
    }

    /// The scaled norm is exact on Pythagorean data of every length up to a few chunks,
    /// including the remainder cases, and far outside the range where squares fit in f64.
    #[test]
    fn norm_is_scaled() {
        assert_eq!(norm(&[]), 0.0);
        assert_eq!(norm(&[0.0, -0.0]), 0.0);
        for n in 0..9 {
            // n zeros, then 3 and 4 with the largest value 4: the norm is 5 exactly.
            let mut x = vec![0.0; n];
            x.extend([3.0, 4.0]);
            assert_eq!(norm(&x), 5.0, "length {}", n + 2);
        }
        for scale in [1e300, 1e-300] {
            let got = norm(&[3.0 * scale, -4.0 * scale]);
            assert!((got - 5.0 * scale).abs() <= 1e-15 * 5.0 * scale, "{got:e}");
        }
    }

    /// `axpy` adds `c·x` to `y` element by element.
    #[test]
    fn axpy_adds_a_multiple() {
        let mut y = [1.0, 1.0, 1.0, 1.0, 1.0];
        axpy(2.0, &[1.0, 2.0, 3.0, 4.0, 5.0], &mut y);
        assert_eq!(y, [3.0, 5.0, 7.0, 9.0, 11.0]);
    }

    /// A reflector built by `new` maps its column onto `R_jj·e_0`, and its `β` is in [1/8, 1].
    #[test]
    fn reflector_zeroes_below_the_diagonal() {
        let x = [3.0, 4.0, 0.0, 12.0];
        let a = Tensor::from_vec(x.to_vec(), &[4, 1]).unwrap();
        let qr = Qr::new(a.view()).unwrap();
        assert_eq!(qr.r_diagonal(), &[-13.0]);
        assert!((0.125..=1.0).contains(&qr.betas[0]));
        let mut y = x;
        reflect(&qr.factors, qr.betas[0], &mut y);
        assert!((y[0] + 13.0).abs() < 1e-14, "{y:?}");
        assert!(y[1..].iter().all(|v| v.abs() < 1e-14), "{y:?}");
    }

    /// The input is copied column-major, and the part above the diagonal holds `R`.
    #[test]
    fn storage_is_column_major() {
        // Column 0 is (1, 0, 0), already reduced: R_00 = -1 and the reflector maps e_0 to
        // -e_0, so R_01 = -a_01.
        let a = Tensor::from_vec(vec![1.0, 2.0, 0.0, 3.0, 0.0, 4.0], &[3, 2]).unwrap();
        let qr = Qr::new(a.view()).unwrap();
        assert_eq!(qr.r_diagonal()[0], -1.0);
        assert_eq!(qr.factors[3], -2.0);
        assert_eq!(qr.r_diagonal()[1].abs(), 5.0);
    }

    /// A matrix with no columns factors and solves to an empty result.
    #[test]
    fn no_columns() {
        let a = Tensor::from_vec(vec![], &[2, 0]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
        let x = lstsq(a.view(), b.view()).unwrap();
        assert_eq!(x.shape().dims(), &[0, 1]);
    }
}
