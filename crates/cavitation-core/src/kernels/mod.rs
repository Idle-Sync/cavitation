//! CPU kernels: strided matrix views and the plain reference implementations.

pub mod reference;

use crate::Real;

/// Returns the number of elements a `rows` by `cols` view with the given strides needs
/// `data` to hold, or `None` if the arithmetic overflows. Empty views need none.
fn required_len(rows: usize, cols: usize, row_stride: usize, col_stride: usize) -> Option<usize> {
    if rows == 0 || cols == 0 {
        return Some(0);
    }
    (rows - 1)
        .checked_mul(row_stride)?
        .checked_add((cols - 1).checked_mul(col_stride)?)?
        .checked_add(1)
}

/// Whether no two distinct elements of a `rows` by `cols` view can share an offset.
///
/// Only dimensions with extent above 1 count. A single such dimension needs a stride of at
/// least 1. Two need, with the smaller stride `s` and its extent `e`, `s >= 1` and
/// `e * s <= ` the larger stride. This is sufficient, not necessary.
fn is_non_aliasing(rows: usize, cols: usize, row_stride: usize, col_stride: usize) -> bool {
    let mut dims: Vec<(usize, usize)> = Vec::with_capacity(2);
    if rows > 1 {
        dims.push((rows, row_stride));
    }
    if cols > 1 {
        dims.push((cols, col_stride));
    }
    match dims.as_slice() {
        [] => true,
        [(_, stride)] => *stride >= 1,
        [a, b] => {
            let (small, large) = if a.1 <= b.1 { (a, b) } else { (b, a) };
            small.1 >= 1 && small.0.checked_mul(small.1).is_some_and(|v| v <= large.1)
        }
        _ => unreachable!("a matrix has two dimensions"),
    }
}

/// A read-only matrix view over a slice, with an independent stride for rows and columns.
///
/// Element `(i, j)` is `data[i * row_stride + j * col_stride]`. A row-major `r` by `c`
/// matrix has strides `(c, 1)`; its transpose is the same data with the strides swapped.
/// The constructor checks that every element lies inside the slice, so a view built from
/// safe code can never index out of bounds.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::MatRef;
/// let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let m = MatRef::new(&data, 2, 3, 3, 1).unwrap();
/// assert_eq!(m.get(1, 2), 6.0);
/// assert_eq!(m.transpose().get(2, 1), 6.0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MatRef<'a, T> {
    /// The underlying storage.
    data: &'a [T],
    /// Number of rows.
    rows: usize,
    /// Number of columns.
    cols: usize,
    /// Distance in `data` between vertically adjacent elements.
    row_stride: usize,
    /// Distance in `data` between horizontally adjacent elements.
    col_stride: usize,
}

impl<'a, T: Real> MatRef<'a, T> {
    /// Builds a view, or returns `None` if the largest index
    /// `(rows - 1) * row_stride + (cols - 1) * col_stride` is not inside `data`.
    /// A view with zero rows or zero columns is always allowed.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::kernels::MatRef;
    /// let data = [0.0f32; 5];
    /// assert!(MatRef::new(&data, 2, 3, 3, 1).is_none()); // needs 6 values
    /// assert!(MatRef::new(&data, 0, 3, 3, 1).is_some());
    /// ```
    pub fn new(
        data: &'a [T],
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Option<Self> {
        let need = required_len(rows, cols, row_stride, col_stride)?;
        if need > data.len() {
            return None;
        }
        Some(Self {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
        })
    }

    /// The number of rows.
    ///
    /// ```
    /// use cavitation_core::kernels::MatRef;
    /// let m = MatRef::new(&[0.0f32; 6], 2, 3, 3, 1).unwrap();
    /// assert_eq!(m.rows(), 2);
    /// ```
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    ///
    /// ```
    /// use cavitation_core::kernels::MatRef;
    /// let m = MatRef::new(&[0.0f32; 6], 2, 3, 3, 1).unwrap();
    /// assert_eq!(m.cols(), 3);
    /// ```
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The distance in the slice between vertically adjacent elements.
    ///
    /// ```
    /// use cavitation_core::kernels::MatRef;
    /// let m = MatRef::new(&[0.0f32; 6], 2, 3, 3, 1).unwrap();
    /// assert_eq!(m.row_stride(), 3);
    /// ```
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// The distance in the slice between horizontally adjacent elements.
    ///
    /// ```
    /// use cavitation_core::kernels::MatRef;
    /// let m = MatRef::new(&[0.0f32; 6], 2, 3, 3, 1).unwrap();
    /// assert_eq!(m.col_stride(), 1);
    /// ```
    pub fn col_stride(&self) -> usize {
        self.col_stride
    }

    /// The element at row `i`, column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= rows` or `j >= cols`.
    ///
    /// ```
    /// use cavitation_core::kernels::MatRef;
    /// let m = MatRef::new(&[1.0f32, 2.0, 3.0, 4.0], 2, 2, 2, 1).unwrap();
    /// assert_eq!(m.get(1, 0), 3.0);
    /// ```
    pub fn get(&self, i: usize, j: usize) -> T {
        assert!(
            i < self.rows && j < self.cols,
            "MatRef::get: index ({i}, {j}) outside a {} x {} view",
            self.rows,
            self.cols
        );
        self.data[i * self.row_stride + j * self.col_stride]
    }

    /// The same data seen as the transpose: rows and columns swap, and so do the strides.
    ///
    /// ```
    /// use cavitation_core::kernels::MatRef;
    /// let m = MatRef::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, 3, 1).unwrap();
    /// let t = m.transpose();
    /// assert_eq!((t.rows(), t.cols()), (3, 2));
    /// assert_eq!(t.get(0, 1), 4.0);
    /// ```
    pub fn transpose(self) -> Self {
        Self {
            data: self.data,
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
        }
    }
}

/// A writable matrix view over a slice, with the same layout rules as [`MatRef`].
///
/// A `MatMut` never aliases: `new` returns `None` unless distinct elements map to distinct
/// positions in the slice, so writing different elements can never touch the same value.
/// Only dimensions longer than 1 count (a single row or column may have any stride). For
/// two such dimensions the rule is conservative: with the smaller stride `s` and its extent
/// `e`, it needs `s >= 1` and `e * s <=` the larger stride. A few non-aliasing layouts, such
/// as strides `(3, 2)` on a 2 x 2 view, are rejected too. A row-major, column-major or
/// transposed layout is always accepted. ([`MatRef`] allows overlap, since reading is safe.)
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::MatMut;
/// let mut data = [0.0f32; 4];
/// let mut m = MatMut::new(&mut data, 2, 2, 2, 1).unwrap();
/// *m.get_mut(1, 1) = 7.0;
/// assert_eq!(data, [0.0, 0.0, 0.0, 7.0]);
/// ```
#[derive(Debug)]
pub struct MatMut<'a, T> {
    /// The underlying storage.
    data: &'a mut [T],
    /// Number of rows.
    rows: usize,
    /// Number of columns.
    cols: usize,
    /// Distance in `data` between vertically adjacent elements.
    row_stride: usize,
    /// Distance in `data` between horizontally adjacent elements.
    col_stride: usize,
}

impl<'a, T: Real> MatMut<'a, T> {
    /// Builds a view, or returns `None` if the bounds rule of [`MatRef::new`] fails or two
    /// distinct elements could share a position (see the type's documentation).
    ///
    /// ```
    /// use cavitation_core::kernels::MatMut;
    /// let mut data = [0.0f32; 5];
    /// assert!(MatMut::new(&mut data, 2, 3, 3, 1).is_none()); // out of bounds
    /// assert!(MatMut::new(&mut data, 2, 2, 0, 0).is_none()); // aliasing
    /// ```
    pub fn new(
        data: &'a mut [T],
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Option<Self> {
        if !is_non_aliasing(rows, cols, row_stride, col_stride) {
            return None;
        }
        let need = required_len(rows, cols, row_stride, col_stride)?;
        if need > data.len() {
            return None;
        }
        Some(Self {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
        })
    }

    /// The number of rows.
    ///
    /// ```
    /// use cavitation_core::kernels::MatMut;
    /// let mut d = [0.0f32; 6];
    /// assert_eq!(MatMut::new(&mut d, 2, 3, 3, 1).unwrap().rows(), 2);
    /// ```
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    ///
    /// ```
    /// use cavitation_core::kernels::MatMut;
    /// let mut d = [0.0f32; 6];
    /// assert_eq!(MatMut::new(&mut d, 2, 3, 3, 1).unwrap().cols(), 3);
    /// ```
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The distance in the slice between vertically adjacent elements.
    ///
    /// ```
    /// use cavitation_core::kernels::MatMut;
    /// let mut d = [0.0f32; 6];
    /// assert_eq!(MatMut::new(&mut d, 2, 3, 3, 1).unwrap().row_stride(), 3);
    /// ```
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// The distance in the slice between horizontally adjacent elements.
    ///
    /// ```
    /// use cavitation_core::kernels::MatMut;
    /// let mut d = [0.0f32; 6];
    /// assert_eq!(MatMut::new(&mut d, 2, 3, 3, 1).unwrap().col_stride(), 1);
    /// ```
    pub fn col_stride(&self) -> usize {
        self.col_stride
    }

    /// The element at row `i`, column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= rows` or `j >= cols`.
    ///
    /// ```
    /// use cavitation_core::kernels::MatMut;
    /// let mut d = [1.0f32, 2.0];
    /// let m = MatMut::new(&mut d, 1, 2, 2, 1).unwrap();
    /// assert_eq!(m.get(0, 1), 2.0);
    /// ```
    pub fn get(&self, i: usize, j: usize) -> T {
        assert!(
            i < self.rows && j < self.cols,
            "MatMut::get: index ({i}, {j}) outside a {} x {} view",
            self.rows,
            self.cols
        );
        self.data[i * self.row_stride + j * self.col_stride]
    }

    /// A mutable reference to the element at row `i`, column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= rows` or `j >= cols`.
    ///
    /// ```
    /// use cavitation_core::kernels::MatMut;
    /// let mut d = [0.0f32; 2];
    /// let mut m = MatMut::new(&mut d, 1, 2, 2, 1).unwrap();
    /// *m.get_mut(0, 1) += 3.0;
    /// assert_eq!(m.get(0, 1), 3.0);
    /// ```
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        assert!(
            i < self.rows && j < self.cols,
            "MatMut::get_mut: index ({i}, {j}) outside a {} x {} view",
            self.rows,
            self.cols
        );
        &mut self.data[i * self.row_stride + j * self.col_stride]
    }
}
