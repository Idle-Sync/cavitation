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
#[derive(Clone, Debug)]
pub struct ParamValues<T: Real = f32> {
    /// The id of the store these values belong to.
    store: u32,
    /// One tensor per parameter, in the order they were added.
    tensors: Vec<Tensor<T>>,
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
    /// Panics if `id` was created by another store.
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
    /// One gradient buffer per parameter, shaped like its value.
    tensors: Vec<Tensor<T>>,
    /// Whether each gradient is stale: its buffer holds old data and it reads as zeros.
    stale: Vec<bool>,
    /// Zeros as long as the largest parameter; stale gradients are views into it.
    zeros: Vec<T>,
}

impl<T: Real> ParamGrads<T> {
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
        self.stale[index_in(self.store, id)]
    }

    /// The gradient buffer of `id` for writing, and whether it was stale. The writer must
    /// overwrite every element when it was stale and add to the buffer otherwise. The
    /// gradient is current from then on, even if the write never happens.
    ///
    /// # Panics
    ///
    /// Panics if `id` was created by another store.
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
        let i = index_in(self.store, id);
        let stale = std::mem::replace(&mut self.stale[i], false);
        (self.tensors[i].view_mut(), stale)
    }

    /// The gradient buffer of `id` and its stale flag, left unchanged, for a writer that
    /// clears the flag only once its write has succeeded.
    pub(crate) fn buffer_and_flag(&mut self, id: ParamId) -> (&mut [T], &mut bool) {
        let i = index_in(self.store, id);
        (self.tensors[i].data_mut(), &mut self.stale[i])
    }

    /// The id of the store these gradients belong to.
    pub(crate) fn store_id(&self) -> u32 {
        self.store
    }

    /// The gradient of `id`: a view of zeros when it is stale.
    fn view(&self, id: ParamId) -> TensorView<'_, T> {
        let i = index_in(self.store, id);
        let t = &self.tensors[i];
        if self.stale[i] {
            let shape: Shape = *t.shape();
            TensorView::from_parts(&self.zeros[..shape.len()], shape)
        } else {
            t.view()
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
                tensors: Vec::new(),
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
        self.grads
            .tensors
            .push(Tensor::zeros_generic(value.shape().dims()));
        self.grads.stale.push(true);
        self.values.tensors.push(value);
        id
    }

    /// Removes every parameter added after the first `len`, with its gradient. The shared
    /// zeros buffer keeps its size. Does nothing if the store holds `len` or fewer.
    pub(crate) fn truncate(&mut self, len: usize) {
        self.values.tensors.truncate(len);
        self.grads.tensors.truncate(len);
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
        self.grads.stale.fill(true);
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
    /// Panics if `id` was created by another store.
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
    /// optimiser step) and its gradient (zeros when stale).
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[2], 1.0f32));
    /// store.grads.accumulate(p).0.data_mut().fill(0.5);
    /// for (_, value, grad) in store.iter_mut() {
    ///     for (v, g) in value.data_mut().iter_mut().zip(grad.data()) {
    ///         *v -= 0.1 * *g;
    ///     }
    /// }
    /// assert_eq!(store.value(p).data(), &[0.95, 0.95]);
    /// ```
    pub fn iter_mut(
        &mut self,
    ) -> impl Iterator<Item = (ParamId, &mut Tensor<T>, TensorView<'_, T>)> {
        let grads = &self.grads;
        let store = self.values.store;
        self.values
            .tensors
            .iter_mut()
            .enumerate()
            .map(move |(index, value)| {
                let id = ParamId { index, store };
                (id, value, grads.view(id))
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

    /// `iter_mut` visits every parameter with its gradient, so an optimiser can step.
    #[test]
    fn iter_mut_steps_every_parameter() {
        let mut store = ParamStore::new();
        let a = store.add(Tensor::full(&[1], 1.0f32));
        let b = store.add(Tensor::full(&[2], 2.0f32));
        store.grads.accumulate(a).0.data_mut()[0] = 0.5;
        for (_, value, grad) in store.iter_mut() {
            for (v, g) in value.data_mut().iter_mut().zip(grad.data()) {
                *v -= *g;
            }
        }
        assert_eq!(store.value(a).data(), &[0.5]);
        assert_eq!(store.value(b).data(), &[2.0, 2.0]);
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
}
