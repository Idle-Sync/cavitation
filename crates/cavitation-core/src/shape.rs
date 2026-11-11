//! `Shape`: up to six dimensions stored inline, with NumPy broadcasting.

/// The most dimensions a [`Shape`] can hold.
const MAX_RANK: usize = 6;

/// The dimensions of a tensor: up to six, stored inline so that creating one never allocates.
///
/// # Examples
///
/// ```
/// use cavitation_core::Shape;
/// let s = Shape::new(&[2, 3]);
/// assert_eq!(s.dims(), &[2, 3]);
/// assert_eq!(s.len(), 6);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    /// The dimensions; entries from `rank` on are zero, so derived equality is correct.
    dims: [usize; MAX_RANK],
    /// How many entries of `dims` are in use.
    rank: usize,
}

impl Shape {
    /// The largest rank a shape can have.
    ///
    /// ```
    /// assert_eq!(cavitation_core::Shape::MAX_RANK, 6);
    /// ```
    pub const MAX_RANK: usize = MAX_RANK;

    /// Builds a shape from its dimensions. An empty slice is a scalar.
    ///
    /// # Panics
    ///
    /// Panics if `dims` has more than [`Shape::MAX_RANK`] entries, or if the product of
    /// the dimensions does not fit in `usize`.
    ///
    /// ```
    /// let s = cavitation_core::Shape::new(&[4, 5]);
    /// assert_eq!(s.rank(), 2);
    /// ```
    pub fn new(dims: &[usize]) -> Shape {
        assert!(
            dims.len() <= MAX_RANK,
            "Shape::new: rank {} exceeds the maximum rank {MAX_RANK} (dims {dims:?})",
            dims.len()
        );
        let mut d = [0; MAX_RANK];
        d[..dims.len()].copy_from_slice(dims);
        let shape = Shape {
            dims: d,
            rank: dims.len(),
        };
        // Checks the element count once, so an overflowing shape panics where it is made.
        shape.len();
        shape
    }

    /// The dimensions as a slice.
    ///
    /// ```
    /// assert_eq!(cavitation_core::Shape::new(&[7]).dims(), &[7]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        &self.dims[..self.rank]
    }

    /// The number of dimensions.
    ///
    /// ```
    /// assert_eq!(cavitation_core::Shape::new(&[]).rank(), 0);
    /// ```
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The number of elements: the product of the dimensions, 1 for a scalar.
    ///
    /// # Panics
    ///
    /// Panics if the product does not fit in `usize`. [`Shape::new`] rejects such shapes, but
    /// [`Shape::broadcast`] builds its result directly, so the check here is the guard for those.
    ///
    /// ```
    /// assert_eq!(cavitation_core::Shape::new(&[2, 3, 4]).len(), 24);
    /// ```
    pub fn len(&self) -> usize {
        self.dims()
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .unwrap_or_else(|| {
                panic!(
                    "shape {:?} has more elements than fit in usize",
                    self.dims()
                )
            })
    }

    /// Whether the shape holds no elements (some dimension is 0).
    ///
    /// ```
    /// assert!(cavitation_core::Shape::new(&[3, 0]).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The shape two operands broadcast to under NumPy rules, aligning dimensions from the
    /// right: each pair must be equal or contain a 1. Returns `None` when they do not fit.
    ///
    /// ```
    /// use cavitation_core::Shape;
    /// let r = Shape::broadcast(&Shape::new(&[3, 1]), &Shape::new(&[4]));
    /// assert_eq!(r, Some(Shape::new(&[3, 4])));
    /// assert_eq!(Shape::broadcast(&Shape::new(&[2, 3]), &Shape::new(&[2])), None);
    /// ```
    pub fn broadcast(a: &Shape, b: &Shape) -> Option<Shape> {
        let rank = a.rank.max(b.rank);
        let mut out = [0; MAX_RANK];
        for i in 0..rank {
            // Index from the right; a missing leading dimension counts as 1.
            let da = if i < a.rank {
                a.dims[a.rank - 1 - i]
            } else {
                1
            };
            let db = if i < b.rank {
                b.dims[b.rank - 1 - i]
            } else {
                1
            };
            out[rank - 1 - i] = match (da, db) {
                (x, y) if x == y => x,
                (1, y) => y,
                (x, 1) => x,
                _ => return None,
            };
        }
        Some(Shape { dims: out, rank })
    }
}

/// Tests for `Shape`.
#[cfg(test)]
mod tests {
    use super::*;

    /// A rank-0 shape holds one value.
    #[test]
    fn shape_len_of_scalar_is_one() {
        assert_eq!(Shape::new(&[]).len(), 1);
    }

    /// Seven dimensions are one too many.
    #[test]
    #[should_panic(expected = "rank")]
    fn shape_rank_7_panics() {
        let _ = Shape::new(&[1, 1, 1, 1, 1, 1, 1]);
    }

    /// NumPy rules, aligned from the right.
    #[test]
    fn broadcast_rules() {
        let b = |a: &[usize], c: &[usize]| Shape::broadcast(&Shape::new(a), &Shape::new(c));
        assert_eq!(b(&[3, 1], &[4]), Some(Shape::new(&[3, 4])));
        assert_eq!(b(&[2, 3], &[3]), Some(Shape::new(&[2, 3])));
        assert_eq!(b(&[2, 3], &[2]), None);
        assert_eq!(b(&[], &[5]), Some(Shape::new(&[5])));
    }

    /// A shape whose element count overflows `usize` panics where it is made.
    #[test]
    #[should_panic(expected = "more elements than fit in usize")]
    fn len_overflow_panics() {
        Shape::new(&[usize::MAX, 2]);
    }
}
