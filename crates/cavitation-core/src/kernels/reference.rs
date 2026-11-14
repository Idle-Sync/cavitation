//! Plain reference GEMM and matvec: the simple loops that fast kernels are tested against.
//!
//! They are generic over [`Real`], so they also give the `f64` products the
//! gradient checks use. The f32 activation twins, [`tanh_f32`] and [`sigmoid_f32`], are the
//! scalar form of the SIMD activation kernels.
//!
//! # Examples
//!
//! ```
//! use cavitation_core::kernels::{MatMut, MatRef, reference};
//! let a = [1.0f64, 2.0, 3.0];
//! let b = [4.0f64, 5.0, 6.0];
//! let mut c = [0.0f64; 1];
//! // A 1 x 3 row times a 3 x 1 column: the dot product.
//! reference::gemm(
//!     1.0,
//!     MatRef::new(&a, 1, 3, 3, 1).unwrap(),
//!     MatRef::new(&b, 3, 1, 1, 1).unwrap(),
//!     0.0,
//!     MatMut::new(&mut c, 1, 1, 1, 1).unwrap(),
//! );
//! assert_eq!(c, [32.0]);
//! ```

use super::{MatMut, MatRef};
use crate::Real;

/// General matrix multiply: `C = alpha * A * B + beta * C`.
///
/// `A` is `m` by `k`, `B` is `k` by `n`, `C` is `m` by `n`. Each element is computed as
///
/// ```text
/// // in order, in T, separate multiply and add
/// acc = 0; for p in 0..k { acc = acc + a[i][p] * b[p][j] }
/// c[i][j] = alpha * acc                      if beta == 0
/// c[i][j] = beta * c[i][j] + alpha * acc     otherwise
/// ```
///
/// When `beta == 0` the old contents of `C` are never read, so NaN or infinity already in
/// `C` does not leak into the result. With `k == 0` the sum is zero, so `C` becomes
/// `beta * C` (zeros if `beta == 0`, provided `alpha` is finite).
///
/// # Panics
///
/// Panics, naming the shapes, if `a.cols() != b.rows()` or if `C` is not
/// `a.rows()` by `b.cols()`.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::{reference, MatMut, MatRef};
/// let a = [1.0f32, 2.0, 3.0, 4.0];
/// let b = [5.0f32, 6.0, 7.0, 8.0];
/// let mut c = [0.0f32; 4];
/// reference::gemm(
///     1.0,
///     MatRef::new(&a, 2, 2, 2, 1).unwrap(),
///     MatRef::new(&b, 2, 2, 2, 1).unwrap(),
///     0.0,
///     MatMut::new(&mut c, 2, 2, 2, 1).unwrap(),
/// );
/// assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
/// ```
pub fn gemm<T: Real>(alpha: T, a: MatRef<T>, b: MatRef<T>, beta: T, mut c: MatMut<T>) {
    check_gemm_shapes(&a, &b, &c);
    for i in 0..a.rows() {
        for j in 0..b.cols() {
            let mut acc = T::ZERO;
            for p in 0..a.cols() {
                acc += a.get(i, p) * b.get(p, j);
            }
            let out = c.get_mut(i, j);
            *out = if beta == T::ZERO {
                alpha * acc
            } else {
                beta * *out + alpha * acc
            };
        }
    }
}

/// Panics, naming the shapes, unless `A · B` is defined and has `C`'s shape.
pub(super) fn check_gemm_shapes<T: Real>(a: &MatRef<T>, b: &MatRef<T>, c: &MatMut<T>) {
    assert!(
        a.cols() == b.rows(),
        "gemm: inner dimensions differ: A is {} x {}, B is {} x {}",
        a.rows(),
        a.cols(),
        b.rows(),
        b.cols()
    );
    assert!(
        c.rows() == a.rows() && c.cols() == b.cols(),
        "gemm: C is {} x {} but A ({} x {}) times B ({} x {}) is {} x {}",
        c.rows(),
        c.cols(),
        a.rows(),
        a.cols(),
        b.rows(),
        b.cols(),
        a.rows(),
        b.cols()
    );
}

/// One input vector times a weight matrix stored transposed:
/// `out[j] = bias[j] + sum_i x[i] * w_t[i][j]`.
///
/// `w_t` is `x.len()` by `out.len()`. The sum runs over `i` in order, accumulated in `T`
/// from zero; the bias, if present, is added to the finished sum.
///
/// # Panics
///
/// Panics, naming the lengths, if `x.len() != w_t.rows()`, `out.len() != w_t.cols()`, or
/// the bias length differs from `out.len()`.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::{reference, MatRef};
/// let w_t = [1.0f32, 2.0, 3.0, 4.0];
/// let mut out = [0.0f32; 2];
/// reference::matvec(
///     MatRef::new(&w_t, 2, 2, 2, 1).unwrap(),
///     &[1.0, 1.0],
///     Some(&[0.5, -0.5]),
///     &mut out,
/// );
/// assert_eq!(out, [4.5, 5.5]);
/// ```
pub fn matvec<T: Real>(w_t: MatRef<T>, x: &[T], bias: Option<&[T]>, out: &mut [T]) {
    check_matvec_shapes(&w_t, x, bias, out);
    for (j, o) in out.iter_mut().enumerate() {
        let mut acc = T::ZERO;
        for (i, &xi) in x.iter().enumerate() {
            acc += xi * w_t.get(i, j);
        }
        *o = match bias {
            Some(bias) => bias[j] + acc,
            None => acc,
        };
    }
}

/// Panics, naming the lengths, unless `x` has `w_t.rows()` values, `out` has `w_t.cols()`
/// and the bias, if present, as many as `out`.
pub(super) fn check_matvec_shapes<T: Real>(
    w_t: &MatRef<T>,
    x: &[T],
    bias: Option<&[T]>,
    out: &[T],
) {
    assert!(
        x.len() == w_t.rows(),
        "matvec: input has {} values but w_t is {} x {}",
        x.len(),
        w_t.rows(),
        w_t.cols()
    );
    assert!(
        out.len() == w_t.cols(),
        "matvec: output has {} values but w_t is {} x {}",
        out.len(),
        w_t.rows(),
        w_t.cols()
    );
    if let Some(bias) = bias {
        assert!(
            bias.len() == out.len(),
            "matvec: bias has {} values but the output has {}",
            bias.len(),
            out.len()
        );
    }
}

/// Replaces every value with its hyperbolic tangent, one element at a time, by the
/// algorithm of [`activation`](super::activation) with one rounding per operation: the
/// reference twin of the SIMD `tanh` kernels, within 2 ulp of the correctly rounded result
/// for every finite f32.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::reference;
/// let mut v = [0.0f32, 1.0];
/// reference::tanh_f32(&mut v);
/// assert_eq!(v[0], 0.0);
/// assert!((v[1] - 0.761_594_2).abs() <= f32::EPSILON);
/// ```
pub fn tanh_f32(values: &mut [f32]) {
    for v in values {
        *v = super::activation::tanh_one(*v);
    }
}

/// Replaces every value with its logistic sigmoid, one element at a time, by the algorithm
/// of [`activation`](super::activation) with one rounding per operation: the reference twin
/// of the SIMD sigmoid kernels, within 2 ulp of the correctly rounded result for every
/// finite f32.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::reference;
/// let mut v = [0.0f32, -1000.0];
/// reference::sigmoid_f32(&mut v);
/// assert_eq!(v, [0.5, 0.0]);
/// ```
pub fn sigmoid_f32(values: &mut [f32]) {
    for v in values {
        *v = super::activation::sigmoid_one(*v);
    }
}

/// Tests for the reference kernels.
#[cfg(test)]
mod tests {
    use super::*;

    /// Row-major view helper.
    fn rm<T: Real>(d: &[T], r: usize, c: usize) -> MatRef<'_, T> {
        MatRef::new(d, r, c, c, 1).unwrap()
    }

    /// Row-major mutable view helper.
    fn rm_mut<T: Real>(d: &mut [T], r: usize, c: usize) -> MatMut<'_, T> {
        MatMut::new(d, r, c, c, 1).unwrap()
    }

    /// A 2 x 2 product worked out by hand.
    #[test]
    fn gemm_2x2_by_hand() {
        let (a, b) = ([1.0f32, 2.0, 3.0, 4.0], [5.0f32, 6.0, 7.0, 8.0]);
        let mut c = [9.0f32; 4];
        gemm(1.0, rm(&a, 2, 2), rm(&b, 2, 2), 0.0, rm_mut(&mut c, 2, 2));
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }

    /// A transposed view through swapped strides: Aᵀ times B.
    #[test]
    fn gemm_transposed_views() {
        let (a, b) = ([1.0f32, 2.0, 3.0, 4.0], [5.0f32, 6.0, 7.0, 8.0]);
        let mut c = [0.0f32; 4];
        gemm(
            1.0,
            rm(&a, 2, 2).transpose(),
            rm(&b, 2, 2),
            0.0,
            rm_mut(&mut c, 2, 2),
        );
        assert_eq!(c, [26.0, 30.0, 38.0, 44.0]);
    }

    /// `alpha = 2`, `beta = -1`, `C` all ones.
    #[test]
    fn gemm_alpha_beta() {
        let (a, b) = ([1.0f32, 2.0, 3.0, 4.0], [5.0f32, 6.0, 7.0, 8.0]);
        let mut c = [1.0f32; 4];
        gemm(2.0, rm(&a, 2, 2), rm(&b, 2, 2), -1.0, rm_mut(&mut c, 2, 2));
        assert_eq!(c, [37.0, 43.0, 85.0, 99.0]);
    }

    /// With `beta = 0`, NaN and infinity already in `C` are overwritten.
    #[test]
    fn gemm_beta_zero_ignores_nan_in_c() {
        let (a, b) = ([1.0f32, 2.0, 3.0, 4.0], [5.0f32, 6.0, 7.0, 8.0]);
        let mut c = [f32::NAN, f32::INFINITY, f32::NAN, f32::NEG_INFINITY];
        gemm(1.0, rm(&a, 2, 2), rm(&b, 2, 2), 0.0, rm_mut(&mut c, 2, 2));
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }

    /// With `k = 0` the product is empty and `C` becomes `beta * C`.
    #[test]
    fn gemm_empty_k() {
        let mut c = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        gemm(
            1.0,
            MatRef::new(&[], 2, 0, 0, 1).unwrap(),
            MatRef::new(&[], 0, 3, 3, 1).unwrap(),
            2.0,
            rm_mut(&mut c, 2, 3),
        );
        assert_eq!(c, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }

    /// A 2 x 3 view with row stride 3 needs 6 values; 5 are rejected, for both view types.
    #[test]
    fn matref_rejects_out_of_bounds() {
        let mut d = [0.0f32; 5];
        assert!(MatRef::new(&d, 2, 3, 3, 1).is_none());
        assert!(MatMut::new(&mut d, 2, 3, 3, 1).is_none());
        assert!(MatRef::new(&d, 2, 3, usize::MAX, 1).is_none());
        assert!(MatRef::new(&d, 0, 3, 3, 1).is_some());
    }

    /// Overlapping strides are rejected by `MatMut`.
    #[test]
    fn matmut_rejects_overlapping_strides() {
        let mut d = [0.0f32; 4];
        assert!(MatMut::new(&mut d, 2, 2, 0, 0).is_none());
        assert!(MatMut::new(&mut d[..3], 2, 2, 1, 1).is_none());
    }

    /// Size-1 dimensions may have any stride.
    #[test]
    fn matmut_accepts_size_one_dims_with_any_stride() {
        let mut d = [0.0f32; 2];
        assert!(MatMut::new(&mut d, 2, 1, 1, 0).is_some());
    }

    /// A transposed layout is accepted.
    #[test]
    fn matmut_accepts_transposed_layout() {
        let mut d = [0.0f32; 6];
        assert!(MatMut::new(&mut d, 2, 3, 1, 2).is_some());
    }

    /// Empty views have no elements to alias, whatever their strides.
    #[test]
    fn matmut_accepts_empty_views() {
        assert!(MatMut::<f32>::new(&mut [], 5, 0, 0, 1).is_some());
        assert!(MatMut::<f32>::new(&mut [], 0, 96, 1, 0).is_some());
        assert!(MatMut::<f32>::new(&mut [], 0, 0, 0, 0).is_some());
    }

    /// Shape mismatches panic with a message naming both shapes.
    #[test]
    #[should_panic(expected = "A is 2 x 3, B is 2 x 2")]
    fn gemm_panics_on_inner_mismatch() {
        let (a, b, mut c) = ([0.0f32; 6], [0.0f32; 4], [0.0f32; 4]);
        gemm(1.0, rm(&a, 2, 3), rm(&b, 2, 2), 0.0, rm_mut(&mut c, 2, 2));
    }

    /// A wrong-sized `C` panics with a message naming the shapes.
    #[test]
    #[should_panic(expected = "C is 3 x 2")]
    fn gemm_panics_on_c_mismatch() {
        let (a, b, mut c) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 6]);
        gemm(1.0, rm(&a, 2, 2), rm(&b, 2, 2), 0.0, rm_mut(&mut c, 3, 2));
    }

    /// A matvec worked out by hand, with a bias.
    #[test]
    fn matvec_by_hand() {
        let w_t = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 2];
        matvec(rm(&w_t, 2, 2), &[1.0, 1.0], Some(&[0.5, -0.5]), &mut out);
        assert_eq!(out, [4.5, 5.5]);
        matvec(rm(&w_t, 2, 2), &[1.0, 1.0], None, &mut out);
        assert_eq!(out, [4.0, 6.0]);
    }

    /// A wrong input length panics.
    #[test]
    #[should_panic(expected = "input has 3 values")]
    fn matvec_panics_on_length_mismatch() {
        let w_t = [0.0f32; 4];
        let mut out = [0.0f32; 2];
        matvec(rm(&w_t, 2, 2), &[1.0, 1.0, 1.0], None, &mut out);
    }

    /// Integer inputs give identical results in f32 and f64.
    #[test]
    fn gemm_f64_matches_f32_on_small_ints() {
        let a32: Vec<f32> = (0..12).map(|i| (i % 7) as f32 - 3.0).collect();
        let b32: Vec<f32> = (0..20).map(|i| (i % 5) as f32 - 2.0).collect();
        let a64: Vec<f64> = a32.iter().map(|&x| x as f64).collect();
        let b64: Vec<f64> = b32.iter().map(|&x| x as f64).collect();
        let (mut c32, mut c64) = (vec![1.0f32; 15], vec![1.0f64; 15]);
        gemm(
            2.0,
            rm(&a32, 3, 4),
            rm(&b32, 4, 5),
            3.0,
            rm_mut(&mut c32, 3, 5),
        );
        gemm(
            2.0,
            rm(&a64, 3, 4),
            rm(&b64, 4, 5),
            3.0,
            rm_mut(&mut c64, 3, 5),
        );
        for (x, y) in c32.iter().zip(&c64) {
            assert_eq!(*x as f64, *y);
        }
    }
}
