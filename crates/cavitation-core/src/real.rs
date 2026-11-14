//! `Real`: the sealed number trait implemented by f32 and f64.

use std::fmt::{Debug, Display};

use crate::kernels::{MatMut, MatRef};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// Keeps `Real` closed: only this crate can name `Sealed`, so only f32 and f64 implement it.
mod private {
    /// Marker supertrait of `Real`.
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// The floating-point types the library computes in: `f32` (shipped) and `f64` (reference
/// kernels and gradient checks). The trait is sealed; no other type can implement it.
///
/// # Examples
///
/// ```
/// use cavitation_core::Real;
/// fn double<T: Real>(x: T) -> T { x + x }
/// assert_eq!(double(1.5f32), 3.0);
/// assert_eq!(double(1.5f64), 3.0);
/// ```
pub trait Real:
    Copy
    + PartialOrd
    + Debug
    + Display
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + private::Sealed
{
    /// Zero.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(f32::ZERO, 0.0);
    /// ```
    const ZERO: Self;
    /// One.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(f64::ONE, 1.0);
    /// ```
    const ONE: Self;
    /// The gap between 1 and the next larger value.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(<f32 as Real>::EPSILON, 1.1920929e-7);
    /// assert!(<f64 as Real>::EPSILON < <f32 as Real>::EPSILON.to_f64());
    /// ```
    const EPSILON: Self;
    /// The smallest positive normal value of the type: below it, values are subnormal and
    /// arithmetic on them can be many times slower.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(<f32 as Real>::MIN_POSITIVE, f32::MIN_POSITIVE);
    /// assert_eq!(<f64 as Real>::MIN_POSITIVE, f64::MIN_POSITIVE);
    /// ```
    const MIN_POSITIVE: Self;

    /// Converts from `f64`, rounding to the nearest value of this type.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(f32::from_f64(0.5), 0.5f32);
    /// ```
    fn from_f64(x: f64) -> Self;
    /// Converts to `f64` exactly.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(Real::to_f64(0.25f32), 0.25f64);
    /// ```
    fn to_f64(self) -> f64;
    /// The exponential `e^self`.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(Real::exp(0.0f64), 1.0);
    /// ```
    fn exp(self) -> Self;
    /// The natural logarithm.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(Real::ln(1.0f64), 0.0);
    /// ```
    fn ln(self) -> Self;
    /// `ln(1 + self)`, accurate for small values.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(Real::ln_1p(0.0f64), 0.0);
    /// ```
    fn ln_1p(self) -> Self;
    /// `e^self - 1`, accurate for small values: `(-1e-10f64).exp_m1()` is `-1e-10` to the last
    /// digit, where `exp(x) - 1` loses it. `-0.0` stays `-0.0`.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(Real::exp_m1(0.0f64), 0.0);
    /// assert_eq!(Real::exp_m1(-1e-10f32), -1e-10);
    /// ```
    fn exp_m1(self) -> Self;
    /// The hyperbolic tangent.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(Real::tanh(0.0f32), 0.0);
    /// ```
    fn tanh(self) -> Self;
    /// The error function, `2/√π` times the integral of `e^(-t²)` from 0 to `self`. Ported
    /// from FreeBSD's msun; accurate to about 1 ulp.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(Real::erf(0.0f64), 0.0);
    /// assert!((Real::erf(1.0f64) - 0.8427007929497149).abs() < 1e-15);
    /// ```
    fn erf(self) -> Self;
    /// The square root.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(Real::sqrt(9.0f32), 3.0);
    /// ```
    fn sqrt(self) -> Self;
    /// The absolute value.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(Real::abs(-2.0f32), 2.0);
    /// ```
    fn abs(self) -> Self;
    /// The larger of two values; if one is NaN, the other.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(Real::max(1.0f32, 2.0), 2.0);
    /// ```
    fn max(self, other: Self) -> Self;
    /// The smaller of two values; if one is NaN, the other.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(Real::min(1.0f32, 2.0), 1.0);
    /// ```
    fn min(self, other: Self) -> Self;
    /// The rectified linear unit, as the `Relu` layer computes it: 0 at or below 0 (so `-0`
    /// becomes `+0`), the value above 0, and NaN stays NaN.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(Real::relu(-2.0f32), 0.0);
    /// assert!(Real::relu(-0.0f64).is_sign_positive());
    /// assert_eq!(Real::relu(1.5f32), 1.5);
    /// assert!(Real::relu(f32::NAN).is_nan());
    /// ```
    fn relu(self) -> Self {
        if self <= Self::ZERO { Self::ZERO } else { self }
    }
    /// Raises to an integer power.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert_eq!(Real::powi(2.0f64, 3), 8.0);
    /// ```
    fn powi(self, n: i32) -> Self;
    /// Whether the value is neither NaN nor infinite.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// assert!(!Real::is_finite(f32::NAN));
    /// ```
    fn is_finite(self) -> bool;
    /// General matrix multiply, `C = alpha * A * B + beta * C`, with the kernel for this type:
    /// [`kernels::gemm::gemm_f32`](crate::kernels::gemm::gemm_f32) for f32 (the current kernel
    /// level and thread count), the reference loops for f64.
    /// Results match [`kernels::reference::gemm`](crate::kernels::reference::gemm) within the
    /// rounding of a different summation order; when `beta` is zero the old contents of `C`
    /// are never read.
    ///
    /// # Panics
    ///
    /// Panics, naming the shapes, if the inner dimensions differ or `C` has the wrong shape.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// use cavitation_core::kernels::{MatMut, MatRef};
    /// let a = [1.0f32, 2.0];
    /// let b = [3.0f32, 4.0];
    /// let mut c = [0.0f32];
    /// f32::gemm(
    ///     1.0,
    ///     MatRef::new(&a, 1, 2, 2, 1).unwrap(),
    ///     MatRef::new(&b, 2, 1, 1, 1).unwrap(),
    ///     0.0,
    ///     MatMut::new(&mut c, 1, 1, 1, 1).unwrap(),
    /// );
    /// assert_eq!(c, [11.0]);
    /// ```
    fn gemm(alpha: Self, a: MatRef<Self>, b: MatRef<Self>, beta: Self, c: MatMut<Self>);

    /// Replaces every value with its hyperbolic tangent: through
    /// [`kernels::activation::tanh_f32`](crate::kernels::activation::tanh_f32) for f32 (the
    /// current kernel level, within 3 ulp of the correctly rounded result), and
    /// [`Real::tanh`] on each value for f64.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// let mut v = [0.0f64, 1.0];
    /// f64::tanh_in_place(&mut v);
    /// assert_eq!(v, [0.0, Real::tanh(1.0f64)]);
    /// ```
    fn tanh_in_place(values: &mut [Self]);

    /// Replaces every value with its logistic sigmoid `1 / (1 + e^-x)`: through
    /// [`kernels::activation::sigmoid_f32`](crate::kernels::activation::sigmoid_f32) for f32
    /// (the current kernel level, within 3 ulp of the correctly rounded result), and for f64
    /// as `1 / (1 + e^-x)` for `x >= 0` and `e^x / (1 + e^x)` below, so it never overflows.
    ///
    /// ```
    /// use cavitation_core::Real;
    /// let mut v = [0.0f32, -1000.0, 1000.0];
    /// f32::sigmoid_in_place(&mut v);
    /// assert_eq!(v, [0.5, 0.0, 1.0]);
    /// ```
    fn sigmoid_in_place(values: &mut [Self]);
}

#[cfg(test)]
thread_local! {
    /// Test builds only: the number of [`Real::gemm`] calls made by this thread.
    pub(crate) static GEMM_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Implements `Real` for a primitive float type by forwarding to its inherent methods.
macro_rules! impl_real {
    ($t:ident, $erf:path, $gemm:path, $tanh:path, $sigmoid:path) => {
        impl Real for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const EPSILON: Self = $t::EPSILON;
            const MIN_POSITIVE: Self = $t::MIN_POSITIVE;
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn exp(self) -> Self {
                $t::exp(self)
            }
            fn ln(self) -> Self {
                $t::ln(self)
            }
            fn ln_1p(self) -> Self {
                $t::ln_1p(self)
            }
            fn exp_m1(self) -> Self {
                $t::exp_m1(self)
            }
            fn tanh(self) -> Self {
                $t::tanh(self)
            }
            fn erf(self) -> Self {
                $erf(self)
            }
            fn sqrt(self) -> Self {
                $t::sqrt(self)
            }
            fn abs(self) -> Self {
                $t::abs(self)
            }
            fn max(self, other: Self) -> Self {
                $t::max(self, other)
            }
            fn min(self, other: Self) -> Self {
                $t::min(self, other)
            }
            fn powi(self, n: i32) -> Self {
                $t::powi(self, n)
            }
            fn is_finite(self) -> bool {
                $t::is_finite(self)
            }
            fn gemm(alpha: Self, a: MatRef<Self>, b: MatRef<Self>, beta: Self, c: MatMut<Self>) {
                #[cfg(test)]
                GEMM_CALLS.with(|n| n.set(n.get() + 1));
                $gemm(alpha, a, b, beta, c);
            }
            fn tanh_in_place(values: &mut [Self]) {
                $tanh(values);
            }
            fn sigmoid_in_place(values: &mut [Self]) {
                $sigmoid(values);
            }
        }
    };
}

impl_real!(
    f32,
    crate::ops::erf_f32,
    crate::kernels::gemm::gemm_f32,
    crate::kernels::activation::tanh_f32,
    crate::kernels::activation::sigmoid_f32
);
impl_real!(
    f64,
    crate::ops::erf_f64,
    crate::kernels::reference::gemm,
    tanh_each,
    sigmoid_each
);

/// [`Real::tanh`] on each value: the f64 `tanh_in_place`.
fn tanh_each(values: &mut [f64]) {
    for v in values {
        *v = v.tanh();
    }
}

/// The overflow-free sigmoid on each value: the f64 `sigmoid_in_place`.
fn sigmoid_each(values: &mut [f64]) {
    for v in values {
        *v = crate::ops::sigmoid_value(*v);
    }
}

/// Tests for `Real`.
#[cfg(test)]
mod tests {
    use super::*;

    /// Converting through `f64` and back keeps the f32 value.
    #[test]
    fn real_roundtrip() {
        assert_eq!(f32::from_f64(0.1).to_f64(), 0.1f32 as f64);
    }
}
