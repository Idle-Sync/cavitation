//! `Tensor`, `TensorView` and `TensorViewMut`: dense row-major arrays.

use crate::{Error, Real, Shape};

/// A dense, row-major array that owns its data. `f32` unless another [`Real`] is named.
///
/// # Examples
///
/// ```
/// use cavitation_core::Tensor;
/// let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
/// assert_eq!(t.get(&[1, 0]), 3.0);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T: Real = f32> {
    /// The elements in row-major order.
    data: Vec<T>,
    /// The dimensions; `shape.len() == data.len()`.
    shape: Shape,
}

impl Tensor<f32> {
    /// A tensor of f32 zeros. Defined for `f32` only so that `Tensor::zeros(..)` infers its
    /// type; use [`Tensor::zeros_generic`] for other types.
    ///
    /// # Panics
    ///
    /// Panics if `shape` has more than [`Shape::MAX_RANK`] dimensions.
    ///
    /// ```
    /// let t = cavitation_core::Tensor::zeros(&[2, 3]);
    /// assert_eq!(t.data(), &[0.0; 6]);
    /// ```
    pub fn zeros(shape: &[usize]) -> Tensor<f32> {
        Tensor::zeros_generic(shape)
    }
}

impl<T: Real> Tensor<T> {
    /// A tensor of zeros of any [`Real`] type.
    ///
    /// # Panics
    ///
    /// Panics if `shape` has more than [`Shape::MAX_RANK`] dimensions.
    ///
    /// ```
    /// let t = cavitation_core::Tensor::<f64>::zeros_generic(&[2]);
    /// assert_eq!(t.data(), &[0.0f64, 0.0]);
    /// ```
    pub fn zeros_generic(shape: &[usize]) -> Tensor<T> {
        Tensor::full(shape, T::ZERO)
    }

    /// A tensor with every element equal to `value`.
    ///
    /// # Panics
    ///
    /// Panics if `shape` has more than [`Shape::MAX_RANK`] dimensions.
    ///
    /// ```
    /// let t = cavitation_core::Tensor::full(&[2], 7.0f32);
    /// assert_eq!(t.data(), &[7.0, 7.0]);
    /// ```
    pub fn full(shape: &[usize], value: T) -> Tensor<T> {
        let shape = Shape::new(shape);
        Tensor {
            data: vec![value; shape.len()],
            shape,
        }
    }

    /// Wraps `data` (row-major) as a tensor of the given shape.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ShapeMismatch`] if `data.len()` is not the product of `shape`; the
    /// error's `expected` is `shape` and its `found` is `[data.len()]`.
    ///
    /// # Panics
    ///
    /// Panics if `shape` has more than [`Shape::MAX_RANK`] dimensions.
    ///
    /// ```
    /// use cavitation_core::{Error, Tensor};
    /// assert!(Tensor::from_vec(vec![1.0f32; 4], &[2, 2]).is_ok());
    /// assert!(matches!(
    ///     Tensor::from_vec(vec![1.0f32; 3], &[2, 2]),
    ///     Err(Error::ShapeMismatch { .. })
    /// ));
    /// ```
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Tensor<T>, Error> {
        let s = Shape::new(shape);
        if data.len() != s.len() {
            return Err(Error::ShapeMismatch {
                expected: shape.to_vec(),
                found: vec![data.len()],
            });
        }
        Ok(Tensor { data, shape: s })
    }

    /// The shape.
    ///
    /// ```
    /// let t = cavitation_core::Tensor::zeros(&[2, 3]);
    /// assert_eq!(t.shape().dims(), &[2, 3]);
    /// ```
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The elements in row-major order.
    ///
    /// ```
    /// let t = cavitation_core::Tensor::full(&[2], 1.0f32);
    /// assert_eq!(t.data().len(), 2);
    /// ```
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// The elements in row-major order, mutably.
    ///
    /// ```
    /// let mut t = cavitation_core::Tensor::zeros(&[2]);
    /// t.data_mut()[1] = 5.0;
    /// assert_eq!(t.data(), &[0.0, 5.0]);
    /// ```
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// A borrowed read-only view of the whole tensor.
    ///
    /// ```
    /// let t = cavitation_core::Tensor::zeros(&[3]);
    /// assert_eq!(t.view().data().len(), 3);
    /// ```
    pub fn view(&self) -> TensorView<'_, T> {
        TensorView {
            data: &self.data,
            shape: self.shape,
        }
    }

    /// A borrowed mutable view of the whole tensor.
    ///
    /// ```
    /// let mut t = cavitation_core::Tensor::zeros(&[3]);
    /// t.view_mut().data_mut()[0] = 1.0;
    /// assert_eq!(t.data()[0], 1.0);
    /// ```
    pub fn view_mut(&mut self) -> TensorViewMut<'_, T> {
        TensorViewMut {
            data: &mut self.data,
            shape: self.shape,
        }
    }

    /// Changes the shape without touching the data.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ShapeMismatch`] (`expected` the new shape, `found` the current one) if
    /// the new shape holds a different number of elements; the tensor is left unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `shape` has more than [`Shape::MAX_RANK`] dimensions.
    ///
    /// ```
    /// let mut t = cavitation_core::Tensor::zeros(&[2, 3]);
    /// t.reshape(&[3, 2]).unwrap();
    /// assert_eq!(t.shape().dims(), &[3, 2]);
    /// assert!(t.reshape(&[4]).is_err());
    /// ```
    pub fn reshape(&mut self, shape: &[usize]) -> Result<(), Error> {
        let s = Shape::new(shape);
        if s.len() != self.data.len() {
            return Err(Error::ShapeMismatch {
                expected: shape.to_vec(),
                found: self.shape.dims().to_vec(),
            });
        }
        self.shape = s;
        Ok(())
    }

    /// The element at a multi-dimensional index.
    ///
    /// # Panics
    ///
    /// Panics if `index` has the wrong number of entries or any entry is out of range.
    ///
    /// ```
    /// let t = cavitation_core::Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], &[2, 2]).unwrap();
    /// assert_eq!(t.get(&[0, 1]), 1.0);
    /// ```
    pub fn get(&self, index: &[usize]) -> T {
        self.data[flat_index(&self.shape, index)]
    }
}

/// The row-major offset of `index` in `shape`; panics naming both if they do not fit.
fn flat_index(shape: &Shape, index: &[usize]) -> usize {
    assert!(
        index.len() == shape.rank(),
        "index {index:?} has {} entries but the shape is {:?}",
        index.len(),
        shape.dims()
    );
    let mut offset = 0;
    for (&i, &d) in index.iter().zip(shape.dims()) {
        assert!(
            i < d,
            "index {index:?} is out of range for shape {:?}",
            shape.dims()
        );
        offset = offset * d + i;
    }
    offset
}

/// A borrowed read-only slice with a shape.
///
/// # Examples
///
/// ```
/// let t = cavitation_core::Tensor::full(&[2, 2], 1.0f32);
/// let v = t.view();
/// assert_eq!(v.shape().len(), 4);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a, T: Real = f32> {
    /// The borrowed elements.
    data: &'a [T],
    /// Their shape.
    shape: Shape,
}

impl<'a, T: Real> TensorView<'a, T> {
    /// Wraps a slice that holds exactly `shape.len()` elements.
    pub(crate) fn from_parts(data: &'a [T], shape: Shape) -> TensorView<'a, T> {
        debug_assert_eq!(data.len(), shape.len(), "view length must match its shape");
        TensorView { data, shape }
    }

    /// The shape.
    ///
    /// ```
    /// let t = cavitation_core::Tensor::zeros(&[2]);
    /// assert_eq!(t.view().shape().dims(), &[2]);
    /// ```
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The elements.
    ///
    /// ```
    /// let t = cavitation_core::Tensor::zeros(&[2]);
    /// assert_eq!(t.view().data(), &[0.0, 0.0]);
    /// ```
    pub fn data(&self) -> &'a [T] {
        self.data
    }

    /// Copies the view into an owned tensor.
    ///
    /// ```
    /// let t = cavitation_core::Tensor::full(&[2], 3.0f32);
    /// assert_eq!(t.view().to_tensor(), t);
    /// ```
    pub fn to_tensor(&self) -> Tensor<T> {
        Tensor {
            data: self.data.to_vec(),
            shape: self.shape,
        }
    }
}

/// A borrowed mutable slice with a shape.
///
/// # Examples
///
/// ```
/// let mut t = cavitation_core::Tensor::zeros(&[2]);
/// let mut v = t.view_mut();
/// v.data_mut()[0] = 2.0;
/// assert_eq!(v.to_tensor().data(), &[2.0, 0.0]);
/// ```
#[derive(Debug)]
pub struct TensorViewMut<'a, T: Real = f32> {
    /// The borrowed elements.
    data: &'a mut [T],
    /// Their shape.
    shape: Shape,
}

impl<'a, T: Real> TensorViewMut<'a, T> {
    /// Wraps a slice that holds exactly `shape.len()` elements.
    pub(crate) fn from_parts(data: &'a mut [T], shape: Shape) -> TensorViewMut<'a, T> {
        debug_assert_eq!(data.len(), shape.len(), "view length must match its shape");
        TensorViewMut { data, shape }
    }
}

impl<T: Real> TensorViewMut<'_, T> {
    /// The shape.
    ///
    /// ```
    /// let mut t = cavitation_core::Tensor::zeros(&[2]);
    /// assert_eq!(t.view_mut().shape().dims(), &[2]);
    /// ```
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The elements.
    ///
    /// ```
    /// let mut t = cavitation_core::Tensor::zeros(&[2]);
    /// assert_eq!(t.view_mut().data(), &[0.0, 0.0]);
    /// ```
    pub fn data(&self) -> &[T] {
        self.data
    }

    /// The elements, mutably.
    ///
    /// ```
    /// let mut t = cavitation_core::Tensor::zeros(&[1]);
    /// t.view_mut().data_mut()[0] = 4.0;
    /// assert_eq!(t.data(), &[4.0]);
    /// ```
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    /// Copies the view into an owned tensor.
    ///
    /// ```
    /// let mut t = cavitation_core::Tensor::full(&[1], 3.0f32);
    /// assert_eq!(t.view_mut().to_tensor().data(), &[3.0]);
    /// ```
    pub fn to_tensor(&self) -> Tensor<T> {
        Tensor {
            data: self.data.to_vec(),
            shape: self.shape,
        }
    }
}

/// Tests for `Tensor`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    /// Six values cannot fill a [2, 4] tensor.
    #[test]
    fn from_vec_rejects_wrong_length() {
        let r = Tensor::from_vec(vec![0.0f32; 6], &[2, 4]);
        assert!(matches!(r, Err(Error::ShapeMismatch { .. })));
    }

    /// Reshaping changes the shape, not the data.
    #[test]
    fn reshape_keeps_data() {
        let mut t = Tensor::from_vec((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        t.reshape(&[3, 2]).unwrap();
        assert_eq!(t.shape().dims(), &[3, 2]);
        assert_eq!(t.data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    /// Element [1, 2] of a [2, 3] tensor is the sixth value.
    #[test]
    fn row_major_get() {
        let t = Tensor::from_vec((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        assert_eq!(t.get(&[1, 2]), 5.0);
    }
}
