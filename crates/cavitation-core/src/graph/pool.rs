//! `Pool`: node buffers kept between steps and handed out again by length.

use crate::Real;

/// Spare buffers grouped by length, so a step that needs the same buffers as the last one
/// allocates nothing. Nothing is ever freed while the pool lives, and the bucket lists keep
/// their capacity.
#[derive(Debug)]
pub(crate) struct Pool<T> {
    /// One entry per length seen: the length and the spare buffers of exactly that length.
    buckets: Vec<(usize, Vec<Vec<T>>)>,
}

impl<T: Real> Pool<T> {
    /// An empty pool.
    pub(crate) fn new() -> Pool<T> {
        Pool {
            buckets: Vec::new(),
        }
    }

    /// A buffer of `len` elements. Its contents are unspecified (a reused buffer keeps the
    /// values of its last use); the caller overwrites every element. Allocates only when no
    /// spare buffer of this length exists.
    pub(crate) fn take(&mut self, len: usize) -> Vec<T> {
        self.buckets
            .iter_mut()
            .find(|(l, _)| *l == len)
            .and_then(|(_, spare)| spare.pop())
            .unwrap_or_else(|| vec![T::ZERO; len])
    }

    /// Returns a buffer to the pool for reuse. Buffers that own no memory are dropped.
    pub(crate) fn give(&mut self, buf: Vec<T>) {
        if buf.capacity() == 0 {
            return;
        }
        let len = buf.len();
        match self.buckets.iter_mut().find(|(l, _)| *l == len) {
            Some((_, spare)) => spare.push(buf),
            None => self.buckets.push((len, vec![buf])),
        }
    }
}

/// Tests for the pool.
#[cfg(test)]
mod tests {
    use super::*;

    /// A returned buffer comes back for the same length, not for another.
    #[test]
    fn reuses_by_length() {
        let mut pool = Pool::<f32>::new();
        let mut a = pool.take(3);
        a[0] = 9.0;
        let ptr = a.as_ptr();
        pool.give(a);
        let b = pool.take(4);
        assert_eq!(b.len(), 4);
        let c = pool.take(3);
        assert_eq!(c.as_ptr(), ptr);
        assert_eq!(c.len(), 3);
    }
}
