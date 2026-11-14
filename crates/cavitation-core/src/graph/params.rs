//! `ParamStore`: parameter values and their gradients, kept apart from any model.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::{Real, Shape, Tensor, TensorView, TensorViewMut};

/// The id the next [`ParamStore`] gets, so ids from one store are recognised in another.
static NEXT_STORE_ID: AtomicU32 = AtomicU32::new(0);

/// Names one parameter in the [`ParamStore`] that created it.
///
/// An id is only meaningful for the store that returned it (or a clone of that store);
/// using it with another store panics.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Tensor};
/// let mut store = ParamStore::new();
/// let a = store.add(Tensor::zeros(&[2]));
/// let b = store.add(Tensor::zeros(&[3]));
/// assert_ne!(a, b);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParamId {
    /// The parameter's position in its store.
    index: usize,
    /// The id of the store that created it.
    store: u32,
}

impl ParamId {
    /// The parameter's place in its store: 0 for the first one added, 1 for the next, and
    /// so on. Code that keeps one entry per parameter, such as an optimiser's state, indexes
    /// by it.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let a = store.add(Tensor::zeros(&[2]));
    /// let b = store.add(Tensor::zeros(&[3]));
    /// assert_eq!((a.position(), b.position()), (0, 1));
    /// ```
    pub fn position(&self) -> usize {
        self.index
    }
}

/// The position of `id` in the store with id `store`.
///
/// # Panics
///
/// Panics if `id` was created by another store.
fn index_in(store: u32, id: ParamId) -> usize {
    assert!(
        id.store == store,
        "ParamId belongs to another ParamStore (its store is {}, this store is {store})",
        id.store
    );
    id.index
}

/// Panics unless parameter `index`'s value and gradient have the same shape.
fn check_shapes(index: usize, value: &Shape, grad: &Shape) {
    assert!(
        value == grad,
        "parameter {index}: its value has shape {:?} but its gradient has shape {:?}; \
         change a parameter's elements, never its shape",
        value.dims(),
        grad.dims()
    );
}

/// The values of every parameter in a [`ParamStore`], indexed by [`ParamId`].
///
/// A [`Tape`](crate::Tape) borrows this half of the store while it records, so that
/// parameters enter the graph without being copied.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Tensor};
/// let mut store = ParamStore::new();
/// let p = store.add(Tensor::full(&[2], 1.0f32));
/// store.values.get_mut(p).data_mut()[0] = 3.0;
/// assert_eq!(store.values.get(p).data(), &[3.0, 1.0]);
/// ```
#[derive(Debug)]
pub struct ParamValues<T: Real = f32> {
    /// The id of the store these values belong to.
    store: u32,
    /// One tensor per parameter, in the order they were added.
    tensors: Vec<Tensor<T>>,
}

impl<T: Real> Clone for ParamValues<T> {
    /// A copy of every value, for the same store.
    fn clone(&self) -> Self {
        ParamValues {
            store: self.store,
            tensors: self.tensors.clone(),
        }
    }

    /// Makes these values a copy of `source`, reusing each tensor's buffer when it is large
    /// enough: copying the values of the same parameters again allocates nothing.
    fn clone_from(&mut self, source: &Self) {
        self.store = source.store;
        self.tensors.clone_from(&source.tensors);
    }
}

impl<T: Real> ParamValues<T> {
    /// The value of parameter `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` was created by another store.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[1], 2.0f32));
    /// assert_eq!(store.values.get(p).data(), &[2.0]);
    /// ```
    pub fn get(&self, id: ParamId) -> &Tensor<T> {
        &self.tensors[index_in(self.store, id)]
    }

    /// The value of parameter `id`, mutably. Change its elements, never its shape: the
    /// gradient buffer keeps the shape the parameter was added with.
    ///
    /// # Panics
    ///
    /// Panics if `id` was created by another store. A value whose shape was changed is
    /// caught later: [`ParamStore::iter_mut`] panics when the iterator reaches that
    /// parameter (for example during an optimiser step), and so does the backward pass.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::zeros(&[1]));
    /// store.values.get_mut(p).data_mut()[0] = 5.0;
    /// assert_eq!(store.values.get(p).data(), &[5.0]);
    /// ```
    pub fn get_mut(&mut self, id: ParamId) -> &mut Tensor<T> {
        &mut self.tensors[index_in(self.store, id)]
    }

    /// The id of the store these values belong to.
    pub(crate) fn store_id(&self) -> u32 {
        self.store
    }
}

/// The gradients of every parameter in a [`ParamStore`], indexed by [`ParamId`].
///
/// A store holds one set, [`ParamStore::grads`]. [`ParamGrads::like`] makes another for the
/// same store, so a second backward pass (an inspector's probe, say) can run without
/// touching the gradients training uses.
///
/// Each gradient is either current or stale. [`ParamStore::zero_grad`] marks every gradient
/// stale without writing to it; a stale gradient reads as zeros. The first contribution
/// after that overwrites the buffer and later ones add to it, which saves a zeroing pass
/// over every gradient each step.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Tensor};
/// let mut store = ParamStore::new();
/// let p = store.add(Tensor::zeros(&[2]));
/// assert!(store.grads.is_stale(p));
/// let (mut g, stale) = store.grads.accumulate(p);
/// assert!(stale);
/// g.data_mut().copy_from_slice(&[1.0, 2.0]);
/// assert!(!store.grads.is_stale(p));
/// assert_eq!(store.grad(p).data(), &[1.0, 2.0]);
/// ```
#[derive(Clone, Debug)]
pub struct ParamGrads<T: Real = f32> {
    /// The id of the store these gradients belong to.
    store: u32,
    /// The shape of each parameter's gradient: its value's shape when the parameter was
    /// added, or when the set was made with [`ParamGrads::like`].
    shapes: Vec<Shape>,
    /// One gradient buffer per parameter, as long as its shape once made. A set made with
    /// [`ParamGrads::like`] starts with every buffer empty (unallocated) and makes each the
    /// first time it is written; an unmade buffer is stale.
    bufs: Vec<Vec<T>>,
    /// Whether each gradient is stale: its buffer holds old data and it reads as zeros.
    stale: Vec<bool>,
    /// Zeros as long as the largest parameter; stale gradients are views into it.
    zeros: Vec<T>,
}

impl<T: Real> ParamGrads<T> {
    /// A second gradient set for the store whose values are `values`, with one stale
    /// gradient per parameter the store holds now. A backward pass into it writes here and
    /// leaves [`ParamStore::grads`] untouched, stale flags included.
    ///
    /// Each gradient buffer is allocated the first time something writes to it (a backward
    /// pass that reaches the parameter, or [`accumulate`](Self::accumulate)) and reused from
    /// then on. Making the set allocates its bookkeeping and one zero buffer as long as the
    /// largest parameter, nothing per parameter.
    ///
    /// Clear it with [`zero_grad`](Self::zero_grad) between passes, as a store is cleared;
    /// otherwise each pass adds to the last.
    ///
    /// A parameter added to the store afterwards has no gradient here: a backward pass that
    /// reaches it panics. Make a new set after adding parameters.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamGrads, ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[1], 3.0f32));
    /// let mut second = ParamGrads::like(&store.values);
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let v = tape.param(p);
    /// let loss = tape.mul(v, v);
    /// tape.backward(loss, &mut second);
    /// drop(tape);
    /// assert!(store.grads.is_stale(p)); // the store's own gradients are untouched
    /// assert!(!second.is_stale(p));
    /// assert_eq!(second.accumulate(p).0.data(), &[6.0]);
    /// ```
    pub fn like(values: &ParamValues<T>) -> ParamGrads<T> {
        let largest = values.tensors.iter().map(|t| t.data().len()).max();
        ParamGrads {
            store: values.store,
            shapes: values.tensors.iter().map(|t| *t.shape()).collect(),
            bufs: values.tensors.iter().map(|_| Vec::new()).collect(),
            stale: vec![true; values.tensors.len()],
            zeros: vec![T::ZERO; largest.unwrap_or(0)],
        }
    }

    /// Marks every gradient in this set stale, so each reads as zeros and its next
    /// contribution overwrites it. Writes nothing to the buffers and allocates nothing.
    /// [`ParamStore::zero_grad`] does this for the store's own set; call it directly on a
    /// set made with [`like`](Self::like).
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamGrads, ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[1], 3.0f32));
    /// let mut second = ParamGrads::like(&store.values);
    /// let mut graph = Graph::new();
    /// for _ in 0..2 {
    ///     second.zero_grad();
    ///     let mut tape = graph.tape(&store.values);
    ///     let v = tape.param(p);
    ///     let loss = tape.mul(v, v);
    ///     tape.backward(loss, &mut second);
    /// }
    /// // The second pass overwrote the first instead of adding to it.
    /// assert_eq!(second.accumulate(p).0.data(), &[6.0]);
    /// ```
    pub fn zero_grad(&mut self) {
        self.stale.fill(true);
    }

    /// The position of `id` in this set.
    ///
    /// # Panics
    ///
    /// Panics if `id` was created by another store, or was added to the store after this
    /// set was made with [`like`](Self::like).
    fn index(&self, id: ParamId) -> usize {
        let i = index_in(self.store, id);
        assert!(
            i < self.shapes.len(),
            "parameter {i} was added to the store after this gradient set was made \
             (it has {} parameters); make a new one with ParamGrads::like",
            self.shapes.len()
        );
        i
    }

    /// The gradient buffer at position `i`, allocated (as zeros) if this is its first use.
    fn made(&mut self, i: usize) -> &mut [T] {
        let len = self.shapes[i].len();
        let buf = &mut self.bufs[i];
        if buf.len() != len {
            *buf = vec![T::ZERO; len];
        }
        buf
    }

    /// Whether the gradient of `id` is stale: it reads as zeros, and the next contribution
    /// overwrites its buffer instead of adding to it.
    ///
    /// # Panics
    ///
    /// Panics if `id` was created by another store.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::zeros(&[1]));
    /// assert!(store.grads.is_stale(p));
    /// ```
    pub fn is_stale(&self, id: ParamId) -> bool {
        self.stale[self.index(id)]
    }

    /// The gradient buffer of `id` for writing, and whether it was stale. The writer must
    /// overwrite every element when it was stale and add to the buffer otherwise. The
    /// gradient is current from then on, even if the write never happens. In a set made with
    /// [`like`](Self::like), the first call for a parameter allocates its buffer.
    ///
    /// # Panics
    ///
    /// Panics if `id` was created by another store, or added to it after this set was made.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::zeros(&[1]));
    /// for _ in 0..2 {
    ///     let (mut g, stale) = store.grads.accumulate(p);
    ///     if stale { g.data_mut()[0] = 1.0 } else { g.data_mut()[0] += 1.0 }
    /// }
    /// assert_eq!(store.grad(p).data(), &[2.0]);
    /// ```
    pub fn accumulate(&mut self, id: ParamId) -> (TensorViewMut<'_, T>, bool) {
        let i = self.index(id);
        let shape = self.shapes[i];
        let stale = std::mem::replace(&mut self.stale[i], false);
        (TensorViewMut::from_parts(self.made(i), shape), stale)
    }

    /// The gradient buffer of `id` and its stale flag, left unchanged, for a writer that
    /// clears the flag only once its write has succeeded. In a set made with
    /// [`like`](Self::like), the first call for a parameter allocates its buffer.
    ///
    /// # Panics
    ///
    /// Panics if `value_shape`, the shape of the parameter's value, differs from the
    /// gradient buffer's shape, or if the parameter was added after this set was made.
    pub(crate) fn buffer_and_flag(
        &mut self,
        id: ParamId,
        value_shape: &Shape,
    ) -> (&mut [T], &mut bool) {
        let i = self.index(id);
        check_shapes(i, value_shape, &self.shapes[i]);
        self.made(i);
        (&mut self.bufs[i], &mut self.stale[i])
    }

    /// The id of the store these gradients belong to.
    pub(crate) fn store_id(&self) -> u32 {
        self.store
    }

    /// The gradient of `id`: a view of zeros when it is stale.
    fn view(&self, id: ParamId) -> TensorView<'_, T> {
        let i = self.index(id);
        let shape = self.shapes[i];
        if self.stale[i] {
            TensorView::from_parts(&self.zeros[..shape.len()], shape)
        } else {
            TensorView::from_parts(&self.bufs[i], shape)
        }
    }
}

/// Every trainable parameter: values and gradients, held apart from the model. Layers keep
/// [`ParamId`]s; a [`Tape`](crate::Tape) borrows [`values`](ParamStore::values) while it
/// records, and backward writes into [`grads`](ParamStore::grads).
///
/// # Examples
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Tensor};
/// let mut store = ParamStore::new();
/// let p = store.add(Tensor::full(&[1], 2.0f32));
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&store.values);
/// let v = tape.param(p);
/// let loss = tape.add(v, v);
/// tape.backward(loss, &mut store.grads);
/// drop(tape);
/// assert_eq!(store.grad(p).data(), &[2.0]);
/// ```
#[derive(Clone, Debug)]
pub struct ParamStore<T: Real = f32> {
    /// The parameter values.
    pub values: ParamValues<T>,
    /// The parameter gradients.
    pub grads: ParamGrads<T>,
}

impl ParamStore<f32> {
    /// An empty f32 store. Use [`ParamStore::new_generic`] for other types.
    ///
    /// ```
    /// let store = cavitation_core::ParamStore::new();
    /// assert_eq!(store.len(), 0);
    /// ```
    pub fn new() -> ParamStore<f32> {
        ParamStore::new_generic()
    }
}

impl<T: Real> Default for ParamStore<T> {
    fn default() -> Self {
        ParamStore::new_generic()
    }
}

impl<T: Real> ParamStore<T> {
    /// An empty store of any [`Real`] type.
    ///
    /// ```
    /// let store = cavitation_core::ParamStore::<f64>::new_generic();
    /// assert!(store.is_empty());
    /// ```
    pub fn new_generic() -> ParamStore<T> {
        let store = NEXT_STORE_ID.fetch_add(1, Ordering::Relaxed);
        ParamStore {
            values: ParamValues {
                store,
                tensors: Vec::new(),
            },
            grads: ParamGrads {
                store,
                shapes: Vec::new(),
                bufs: Vec::new(),
                stale: Vec::new(),
                zeros: Vec::new(),
            },
        }
    }

    /// Adds a parameter with the given initial value and returns its id. Its gradient starts
    /// stale (reads as zeros).
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[2], 0.5f32));
    /// assert_eq!(store.value(p).data(), &[0.5, 0.5]);
    /// assert_eq!(store.grad(p).data(), &[0.0, 0.0]);
    /// ```
    pub fn add(&mut self, value: Tensor<T>) -> ParamId {
        let id = ParamId {
            index: self.values.tensors.len(),
            store: self.values.store,
        };
        let len = value.data().len();
        if self.grads.zeros.len() < len {
            self.grads.zeros.resize(len, T::ZERO);
        }
        self.grads.shapes.push(*value.shape());
        self.grads.bufs.push(vec![T::ZERO; len]);
        self.grads.stale.push(true);
        self.values.tensors.push(value);
        id
    }

    /// Removes every parameter added after the first `len`, with its gradient. The shared
    /// zeros buffer keeps its size. Does nothing if the store holds `len` or fewer.
    pub(crate) fn truncate(&mut self, len: usize) {
        self.values.tensors.truncate(len);
        self.grads.shapes.truncate(len);
        self.grads.bufs.truncate(len);
        self.grads.stale.truncate(len);
    }

    /// The number of parameters.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// store.add(Tensor::zeros(&[4]));
    /// assert_eq!(store.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.values.tensors.len()
    }

    /// Whether the store holds no parameters.
    ///
    /// ```
    /// assert!(cavitation_core::ParamStore::new().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.values.tensors.is_empty()
    }

    /// Marks every gradient stale, so each reads as zeros and its next contribution
    /// overwrites it. Writes nothing to the gradient buffers.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::zeros(&[1]));
    /// store.grads.accumulate(p).0.data_mut()[0] = 3.0;
    /// store.zero_grad();
    /// assert_eq!(store.grad(p).data(), &[0.0]);
    /// ```
    pub fn zero_grad(&mut self) {
        self.grads.zero_grad();
    }

    /// The value of parameter `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` was created by another store.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[1], 1.5f32));
    /// assert_eq!(store.value(p).data(), &[1.5]);
    /// ```
    pub fn value(&self, id: ParamId) -> &Tensor<T> {
        self.values.get(id)
    }

    /// The value of parameter `id`, mutably. Change its elements, never its shape.
    ///
    /// # Panics
    ///
    /// Panics if `id` was created by another store. A value whose shape was changed is
    /// caught later: [`iter_mut`](Self::iter_mut) panics when the iterator reaches that
    /// parameter (for example during an optimiser step), and so does the backward pass.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::zeros(&[1]));
    /// store.value_mut(p).data_mut()[0] = 2.0;
    /// assert_eq!(store.value(p).data(), &[2.0]);
    /// ```
    pub fn value_mut(&mut self, id: ParamId) -> &mut Tensor<T> {
        self.values.get_mut(id)
    }

    /// The gradient of parameter `id`; zeros when it is stale.
    ///
    /// # Panics
    ///
    /// Panics if `id` was created by another store.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::zeros(&[3]));
    /// assert_eq!(store.grad(p).data(), &[0.0; 3]);
    /// ```
    pub fn grad(&self, id: ParamId) -> TensorView<'_, T> {
        self.grads.view(id)
    }

    /// Every parameter in the order it was added: its id, its value (mutable, for an
    /// optimiser step) and its gradient, or `None` when the gradient is stale (nothing has
    /// contributed to it since [`zero_grad`](Self::zero_grad)), as PyTorch's `grad` is
    /// `None` for a parameter the backward pass did not reach. The optimisers skip such a
    /// parameter.
    ///
    /// # Panics
    ///
    /// Panics, while the iterator is consumed, at a parameter whose value shape differs
    /// from its gradient's shape (its shape was changed through
    /// [`value_mut`](Self::value_mut)). The check runs as the iterator reaches each
    /// parameter, so a lazily consumed iterator panics only at the offending one.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[2], 1.0f32));
    /// let q = store.add(Tensor::full(&[1], 1.0f32));
    /// store.grads.accumulate(p).0.data_mut().fill(0.5);
    /// for (_, value, grad) in store.iter_mut() {
    ///     let Some(grad) = grad else { continue };
    ///     for (v, g) in value.data_mut().iter_mut().zip(grad.data()) {
    ///         *v -= 0.1 * *g;
    ///     }
    /// }
    /// assert_eq!(store.value(p).data(), &[0.95, 0.95]);
    /// assert_eq!(store.value(q).data(), &[1.0]);
    /// ```
    pub fn iter_mut(
        &mut self,
    ) -> impl Iterator<Item = (ParamId, &mut Tensor<T>, Option<TensorView<'_, T>>)> {
        let grads = &self.grads;
        let store = self.values.store;
        self.values
            .tensors
            .iter_mut()
            .enumerate()
            .map(move |(index, value)| {
                let id = ParamId { index, store };
                check_shapes(index, value.shape(), &grads.shapes[index]);
                let grad = (!grads.stale[index]).then(|| grads.view(id));
                (id, value, grad)
            })
    }
}

/// Tests for the parameter store.
#[cfg(test)]
mod tests {
    use super::*;

    /// A stale gradient of a small parameter is a zero view of its own length, even when a
    /// larger parameter grew the shared zero buffer.
    #[test]
    fn stale_views_have_their_own_shape() {
        let mut store = ParamStore::new();
        let small = store.add(Tensor::full(&[2], 1.0f32));
        let large = store.add(Tensor::full(&[2, 3], 1.0f32));
        assert_eq!(store.grad(small).shape().dims(), &[2]);
        assert_eq!(store.grad(small).data(), &[0.0; 2]);
        assert_eq!(store.grad(large).data(), &[0.0; 6]);
    }

    /// `iter_mut` visits every parameter, with its gradient when it has one this step and
    /// `None` when it is stale, so an optimiser can step the ones that have one.
    #[test]
    fn iter_mut_steps_every_parameter() {
        let mut store = ParamStore::new();
        let a = store.add(Tensor::full(&[1], 1.0f32));
        let b = store.add(Tensor::full(&[2], 2.0f32));
        store.grads.accumulate(a).0.data_mut()[0] = 0.5;
        let mut stale = Vec::new();
        for (id, value, grad) in store.iter_mut() {
            let Some(grad) = grad else {
                stale.push(id);
                continue;
            };
            for (v, g) in value.data_mut().iter_mut().zip(grad.data()) {
                *v -= *g;
            }
        }
        assert_eq!(stale, [b]);
        assert_eq!(store.value(a).data(), &[0.5]);
        assert_eq!(store.value(b).data(), &[2.0, 2.0]);
        store.zero_grad();
        assert!(store.iter_mut().all(|(_, _, g)| g.is_none()));
    }

    /// A `ParamId` from one store panics when used with another.
    #[test]
    #[should_panic(expected = "ParamId belongs to another ParamStore")]
    fn param_id_from_other_store_panics() {
        let mut s1 = ParamStore::new();
        let mut s2 = ParamStore::new();
        let p = s1.add(Tensor::full(&[1], 1.0f32));
        s2.add(Tensor::full(&[1], 2.0f32));
        s2.value(p);
    }

    /// A store with one `[2, 3]` parameter whose value was reshaped to `[3, 2]` through
    /// `value_mut`, and its id.
    fn reshaped_store() -> (ParamStore, ParamId) {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::full(&[2, 3], 1.0f32));
        store.value_mut(p).reshape(&[3, 2]).unwrap();
        (store, p)
    }

    /// An optimiser step over a parameter whose value changed shape panics.
    #[test]
    #[should_panic(expected = "its value has shape [3, 2] but its gradient has shape [2, 3]")]
    fn reshaped_param_panics_at_optimizer_step() {
        let (mut store, p) = reshaped_store();
        store.grads.accumulate(p).0.data_mut().fill(1.0);
        let _ = store.iter_mut().count();
    }

    /// The backward pass into a parameter whose value changed shape panics.
    #[test]
    #[should_panic(expected = "its value has shape [3, 2] but its gradient has shape [2, 3]")]
    fn reshaped_param_panics_at_backward() {
        let (mut store, p) = reshaped_store();
        let mut graph = crate::Graph::new();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p);
        let loss = tape.sum(v);
        tape.backward(loss, &mut store.grads);
    }

    /// A second set makes no gradient buffer until one is written, then reuses it, and
    /// reads as zeros until then.
    #[test]
    fn like_makes_buffers_on_first_use() {
        let mut store = ParamStore::new();
        let a = store.add(Tensor::full(&[2, 3], 1.0f32));
        let b = store.add(Tensor::full(&[4], 1.0f32));
        let mut second = ParamGrads::like(&store.values);
        assert!(second.bufs.iter().all(|buf| buf.capacity() == 0));
        assert_eq!(second.view(a).data(), &[0.0; 6]);
        assert_eq!(second.view(b).shape().dims(), &[4]);
        second.accumulate(a).0.data_mut().fill(2.0);
        let made = second.bufs[0].as_ptr();
        assert_eq!(second.bufs[1].capacity(), 0, "b is not written yet");
        let (mut g, stale) = second.accumulate(a);
        assert!(!stale);
        g.data_mut()[0] += 1.0;
        assert_eq!(second.bufs[0].as_ptr(), made, "the buffer is reused");
        assert_eq!(second.view(a).data(), &[3.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
        assert!(store.grads.is_stale(a));
    }

    /// A backward pass into a second set that reaches a parameter added after the set was
    /// made panics, saying what to do.
    #[test]
    #[should_panic(
        expected = "parameter 1 was added to the store after this gradient set was made"
    )]
    fn like_refuses_parameters_added_later() {
        let mut store = ParamStore::new();
        store.add(Tensor::full(&[1], 1.0f32));
        let mut second = ParamGrads::like(&store.values);
        let late = store.add(Tensor::full(&[1], 2.0f32));
        let mut graph = crate::Graph::new();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(late);
        let loss = tape.add(v, v);
        tape.backward(loss, &mut second);
    }

    /// Writing elements through `value_mut` is fine: the shape is unchanged.
    #[test]
    fn element_writes_through_value_mut_still_work() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::full(&[2, 3], 1.0f32));
        store.value_mut(p).data_mut()[0] = 4.0;
        store.grads.accumulate(p).0.data_mut().fill(1.0);
        assert_eq!(store.iter_mut().count(), 1);
        assert_eq!(store.value(p).data()[0], 4.0);
    }

    /// `clone_from` between the values of stores with the same parameters copies every value
    /// into the buffers the target already has.
    #[test]
    fn values_clone_from_reuses_the_buffers() {
        let mut store = ParamStore::new();
        let a = store.add(Tensor::full(&[3], 1.0f32));
        let b = store.add(Tensor::full(&[2, 2], 2.0f32));
        let mut copy = store.values.clone();
        let buffers = [copy.get(a).data().as_ptr(), copy.get(b).data().as_ptr()];
        store.values.get_mut(a).data_mut()[1] = 7.0;
        store.values.get_mut(b).data_mut()[3] = -1.0;
        copy.clone_from(&store.values);
        assert_eq!(copy.get(a).data(), &[1.0, 7.0, 1.0]);
        assert_eq!(copy.get(b).data(), &[2.0, 2.0, 2.0, -1.0]);
        assert_eq!(
            [copy.get(a).data().as_ptr(), copy.get(b).data().as_ptr()],
            buffers
        );
    }
}
