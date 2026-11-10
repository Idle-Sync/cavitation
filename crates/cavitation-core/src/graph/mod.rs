//! The autograd tape: `Graph`, `Tape`, `Var` and the backward pass.

mod params;
mod pool;

pub use params::{ParamGrads, ParamId, ParamStore, ParamValues};

use crate::ops::Op;
use crate::{Real, Shape, TensorView};
use pool::Pool;

/// A value recorded on a [`Tape`]: an index into that tape's nodes.
///
/// A `Var` is only meaningful on the tape that returned it; using it on another tape reads
/// a different node or panics.
///
/// # Examples
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Tensor};
/// let store = ParamStore::new();
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&store.values);
/// let x = Tensor::full(&[2], 1.0f32);
/// let v = tape.input(x.view());
/// let w = v; // `Var` is `Copy`
/// let s = tape.add(v, w);
/// assert_eq!(tape.value(s).data(), &[2.0, 2.0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Var(usize);

/// One recorded value: the operation that made it, its shape and its elements.
#[derive(Debug)]
pub(crate) struct Node<T> {
    /// The operation and the inputs it read.
    op: Op,
    /// The shape of the value.
    shape: Shape,
    /// The elements, taken from the pool; empty for a parameter, whose value stays in the
    /// [`ParamValues`] the tape borrows.
    value: Vec<T>,
}

/// Reads recorded values: node buffers, and parameter values for parameter nodes.
#[derive(Clone, Copy)]
pub(crate) struct Values<'a, T: Real> {
    /// The tape's nodes.
    nodes: &'a [Node<T>],
    /// The parameter values the tape borrows.
    params: &'a ParamValues<T>,
}

impl<'a, T: Real> Values<'a, T> {
    /// The elements of `v`.
    fn get(self, v: Var) -> &'a [T] {
        let node = &self.nodes[v.0];
        match node.op {
            Op::Param(id) => self.params.get(id).data(),
            _ => &node.value,
        }
    }
}

/// A gradient buffer to contribute to, and whether this is its first contribution in the
/// backward pass (overwrite) or a later one (add).
pub(crate) struct GradSlot<'s, T> {
    /// The gradient buffer, as long as the value it belongs to.
    pub(crate) buf: &'s mut [T],
    /// Whether to overwrite `buf` instead of adding to it.
    pub(crate) overwrite: bool,
}

impl<T: Real> GradSlot<'_, T> {
    /// Contributes `src`: copies it when overwriting, adds it otherwise.
    ///
    /// # Panics
    ///
    /// Panics if `src` and the buffer differ in length.
    pub(crate) fn add_from(self, src: &[T]) {
        if self.overwrite {
            self.buf.copy_from_slice(src);
        } else {
            assert_eq!(self.buf.len(), src.len(), "gradient length mismatch");
            for (d, &s) in self.buf.iter_mut().zip(src) {
                *d += s;
            }
        }
    }
}

/// What an operation's backward pass writes into: the gradients of the nodes recorded
/// before it, and the parameter gradients.
pub(crate) struct BackwardCtx<'a, T: Real> {
    /// Recorded values, for operations whose backward pass reads them.
    values: Values<'a, T>,
    /// Gradients of the nodes before the one being differentiated; `None` until the first
    /// contribution.
    grads: &'a mut [Option<Vec<T>>],
    /// Where new gradient buffers come from.
    pool: &'a mut Pool<T>,
    /// Parameter gradients, written under the overwrite-then-add rule.
    params: &'a mut ParamGrads<T>,
}

impl<T: Real> BackwardCtx<'_, T> {
    /// The gradient buffer of `v` to contribute to, or `None` for an input (inputs take no
    /// gradient). A node's first contribution gets a fresh pool buffer to overwrite; a
    /// parameter's slot follows its stale flag.
    pub(crate) fn slot(&mut self, v: Var) -> Option<GradSlot<'_, T>> {
        let node = &self.values.nodes[v.0];
        match node.op {
            Op::Input => None,
            Op::Param(id) => {
                let (buf, overwrite) = self.params.accumulate_slice(id);
                Some(GradSlot { buf, overwrite })
            }
            _ => {
                let entry = &mut self.grads[v.0];
                let overwrite = entry.is_none();
                let buf = entry.get_or_insert_with(|| self.pool.take(node.shape.len()));
                Some(GradSlot { buf, overwrite })
            }
        }
    }
}

/// Storage reused across training steps: the buffer pool and the node lists. Create one,
/// then record each step on a fresh [`Tape`] from [`Graph::tape`]; after the first steps,
/// recording the same computation again allocates nothing.
///
/// # Examples
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Tensor};
/// let mut store = ParamStore::new();
/// let p = store.add(Tensor::full(&[1], 1.0f32));
/// let mut graph = Graph::new();
/// for _ in 0..3 {
///     store.zero_grad();
///     let mut tape = graph.tape(&store.values);
///     let v = tape.param(p);
///     let loss = tape.add(v, v);
///     tape.backward(loss, &mut store.grads);
/// }
/// assert_eq!(store.grad(p).data(), &[2.0]);
/// ```
#[derive(Debug)]
pub struct Graph<T: Real = f32> {
    /// Spare buffers for node values and gradients.
    pool: Pool<T>,
    /// The nodes of the tape being recorded; empty between tapes, capacity kept.
    nodes: Vec<Node<T>>,
    /// The gradient of each node after a backward pass; empty between tapes, capacity kept.
    grads: Vec<Option<Vec<T>>>,
}

impl Graph<f32> {
    /// An empty f32 graph. Use [`Graph::new_generic`] for other types.
    ///
    /// ```
    /// let mut graph = cavitation_core::Graph::new();
    /// let store = cavitation_core::ParamStore::new();
    /// assert_eq!(graph.tape(&store.values).node_count(), 0);
    /// ```
    pub fn new() -> Graph<f32> {
        Graph::new_generic()
    }
}

impl<T: Real> Default for Graph<T> {
    fn default() -> Self {
        Graph::new_generic()
    }
}

impl<T: Real> Graph<T> {
    /// An empty graph of any [`Real`] type.
    ///
    /// ```
    /// let mut graph = cavitation_core::Graph::<f64>::new_generic();
    /// let store = cavitation_core::ParamStore::<f64>::new_generic();
    /// assert_eq!(graph.tape(&store.values).node_count(), 0);
    /// ```
    pub fn new_generic() -> Graph<T> {
        Graph {
            pool: Pool::new(),
            nodes: Vec::new(),
            grads: Vec::new(),
        }
    }

    /// Starts recording a step. The tape reads parameter values from `params` without
    /// copying them; dropping it returns every buffer it used to this graph.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[2], 3.0f32));
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let v = tape.param(p);
    /// assert_eq!(tape.value(v).data(), &[3.0, 3.0]);
    /// ```
    pub fn tape<'g>(&'g mut self, params: &'g ParamValues<T>) -> Tape<'g, T> {
        // A tape that was leaked instead of dropped leaves its nodes behind.
        self.reclaim();
        Tape {
            graph: self,
            params,
        }
    }

    /// Returns every node value and gradient buffer to the pool and empties the node lists,
    /// keeping their capacity.
    fn reclaim(&mut self) {
        self.reclaim_grads();
        for node in self.nodes.drain(..) {
            self.pool.give(node.value);
        }
    }

    /// Returns every gradient buffer to the pool and empties the gradient list.
    fn reclaim_grads(&mut self) {
        for buf in self.grads.drain(..).flatten() {
            self.pool.give(buf);
        }
    }
}

/// The record of one step: every value computed from inputs and parameters, and after
/// [`Tape::backward`] the gradient of the loss with respect to each of them.
///
/// Operations compute their value as they are recorded. Dropping the tape returns every
/// buffer to the [`Graph`] that made it.
///
/// # Examples
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Tensor};
/// let mut store = ParamStore::new();
/// let w = store.add(Tensor::full(&[1], 0.5f32));
/// let x = Tensor::full(&[1], 2.0f32);
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&store.values);
/// let xi = tape.input(x.view());
/// let wv = tape.param(w);
/// let loss = tape.add(xi, wv);
/// assert_eq!(tape.value(loss).data(), &[2.5]);
/// tape.backward(loss, &mut store.grads);
/// drop(tape);
/// assert_eq!(store.grad(w).data(), &[1.0]);
/// ```
pub struct Tape<'g, T: Real = f32> {
    /// The graph whose storage this tape records into.
    graph: &'g mut Graph<T>,
    /// The parameter values, read in place.
    params: &'g ParamValues<T>,
}

impl<T: Real> Tape<'_, T> {
    /// Records an input: copies `x` into a pooled buffer. Inputs take no gradient.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::full(&[3], 1.0f32);
    /// let v = tape.input(x.view());
    /// assert_eq!(tape.value(v).data(), &[1.0; 3]);
    /// ```
    pub fn input(&mut self, x: TensorView<'_, T>) -> Var {
        let mut buf = self.take_buffer(x.data().len());
        buf.copy_from_slice(x.data());
        self.push(Op::Input, *x.shape(), buf)
    }

    /// Records parameter `id`. Its value is read from the borrowed [`ParamValues`], not
    /// copied; its gradient goes to [`ParamGrads`] on [`Tape::backward`].
    ///
    /// # Panics
    ///
    /// Panics if `id` does not belong to the store whose values this tape borrows.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[2], 4.0f32));
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let v = tape.param(p);
    /// assert_eq!(tape.shape(v).dims(), &[2]);
    /// ```
    pub fn param(&mut self, id: ParamId) -> Var {
        let shape = *self.params.get(id).shape();
        self.push(Op::Param(id), shape, Vec::new())
    }

    /// The value of `v`.
    ///
    /// # Panics
    ///
    /// Panics if `v` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::full(&[1], 7.0f32);
    /// let v = tape.input(x.view());
    /// assert_eq!(tape.value(v).data(), &[7.0]);
    /// ```
    pub fn value(&self, v: Var) -> TensorView<'_, T> {
        TensorView::from_parts(self.values().get(v), self.graph.nodes[v.0].shape)
    }

    /// The gradient of the loss with respect to `v`, after [`Tape::backward`]. `None` for
    /// inputs, for parameters (their gradients are in [`ParamGrads`]), for values the loss
    /// does not depend on, and before any backward pass.
    ///
    /// # Panics
    ///
    /// Panics if `v` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[1], 1.0f32));
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let v = tape.param(p);
    /// let h = tape.add(v, v);
    /// let loss = tape.add(h, h);
    /// tape.backward(loss, &mut store.grads);
    /// assert_eq!(tape.grad(h).unwrap().data(), &[2.0]);
    /// assert!(tape.grad(v).is_none());
    /// ```
    pub fn grad(&self, v: Var) -> Option<TensorView<'_, T>> {
        let node = &self.graph.nodes[v.0];
        match node.op {
            Op::Input | Op::Param(_) => None,
            _ => {
                let buf = self.graph.grads.get(v.0)?.as_deref()?;
                Some(TensorView::from_parts(buf, node.shape))
            }
        }
    }

    /// The shape of `v`.
    ///
    /// # Panics
    ///
    /// Panics if `v` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::zeros(&[2, 3]);
    /// let v = tape.input(x.view());
    /// assert_eq!(tape.shape(v).dims(), &[2, 3]);
    /// ```
    pub fn shape(&self, v: Var) -> Shape {
        self.graph.nodes[v.0].shape
    }

    /// The number of nodes recorded so far (inputs, parameters and operation results).
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::zeros(&[1]);
    /// let v = tape.input(x.view());
    /// tape.add(v, v);
    /// assert_eq!(tape.node_count(), 2);
    /// ```
    pub fn node_count(&self) -> usize {
        self.graph.nodes.len()
    }

    /// Computes the gradient of `loss` with respect to every value it depends on. Seeds the
    /// loss's gradient with 1 and walks the nodes in reverse. Each intermediate gradient's
    /// first contribution overwrites its buffer and later ones add; parameter gradients go
    /// into `grads` under the same rule, following their stale flags, so repeated backward
    /// passes without [`ParamStore::zero_grad`] add up. Calling it again on the same tape
    /// recomputes the intermediate gradients.
    ///
    /// # Panics
    ///
    /// Panics if `loss` does not have exactly one element (the message names the loss and its
    /// shape), or if `loss` was not recorded on this tape.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let a = store.add(Tensor::full(&[1], 2.0f32));
    /// let b = store.add(Tensor::full(&[1], 3.0f32));
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let (va, vb) = (tape.param(a), tape.param(b));
    /// let loss = tape.add(va, vb);
    /// tape.backward(loss, &mut store.grads);
    /// drop(tape);
    /// assert_eq!(store.grad(a).data(), &[1.0]);
    /// assert_eq!(store.grad(b).data(), &[1.0]);
    /// ```
    pub fn backward(&mut self, loss: Var, grads: &mut ParamGrads<T>) {
        let shape = self.graph.nodes[loss.0].shape;
        assert!(
            shape.len() == 1,
            "backward: the loss must have one element, but its shape is {:?}",
            shape.dims()
        );
        self.graph.reclaim_grads();
        let Graph {
            pool,
            nodes,
            grads: node_grads,
        } = &mut *self.graph;
        node_grads.resize_with(nodes.len(), || None);
        let values = Values {
            nodes,
            params: self.params,
        };
        let mut seed_ctx = BackwardCtx {
            values,
            grads: node_grads,
            pool,
            params: grads,
        };
        if let Some(slot) = seed_ctx.slot(loss) {
            slot.add_from(&[T::ONE]);
        }
        for i in (0..=loss.0).rev() {
            let (before, rest) = node_grads.split_at_mut(i);
            let Some(out_grad) = rest[0].as_deref() else {
                continue;
            };
            let mut ctx = BackwardCtx {
                values,
                grads: before,
                pool,
                params: grads,
            };
            nodes[i].op.backward(out_grad, &mut ctx);
        }
    }

    /// Reads recorded values.
    fn values(&self) -> Values<'_, T> {
        Values {
            nodes: &self.graph.nodes,
            params: self.params,
        }
    }

    /// The elements of `v`, for an operation's forward pass.
    pub(crate) fn data(&self, v: Var) -> &[T] {
        self.values().get(v)
    }

    /// A pooled buffer of `len` elements with unspecified contents, for a new node's value.
    pub(crate) fn take_buffer(&mut self, len: usize) -> Vec<T> {
        self.graph.pool.take(len)
    }

    /// Appends a node and returns its variable.
    pub(crate) fn push(&mut self, op: Op, shape: Shape, value: Vec<T>) -> Var {
        debug_assert!(matches!(op, Op::Param(_)) || value.len() == shape.len());
        self.graph.nodes.push(Node { op, shape, value });
        Var(self.graph.nodes.len() - 1)
    }
}

impl<T: Real> Drop for Tape<'_, T> {
    /// Returns every buffer the tape used to the graph's pool.
    fn drop(&mut self) {
        self.graph.reclaim();
    }
}

/// Tests for the tape.
#[cfg(test)]
mod tests {
    use crate::{Graph, ParamStore, Tensor};

    /// Gradients add up over backward calls until `zero_grad`, which reads as zeros.
    #[test]
    fn grad_accumulates_across_backward_calls_until_zero_grad() {
        let mut store = ParamStore::new();
        let p1 = store.add(Tensor::from_vec(vec![2.0f32], &[1]).unwrap());
        let p2 = store.add(Tensor::from_vec(vec![5.0f32], &[1]).unwrap());
        let mut graph = Graph::new();
        for _ in 0..2 {
            let mut tape = graph.tape(&store.values);
            let (a, b) = (tape.param(p1), tape.param(p2));
            let loss = tape.add(a, b);
            tape.backward(loss, &mut store.grads);
        }
        assert_eq!(store.grad(p1).data(), &[2.0]);
        assert_eq!(store.grad(p2).data(), &[2.0]);
        store.zero_grad();
        assert_eq!(store.grad(p1).data(), &[0.0]);
        assert_eq!(store.grad(p2).data(), &[0.0]);
        assert!(store.grads.is_stale(p1));
    }

    /// Inputs need no gradient, so the tape keeps none for them.
    #[test]
    fn input_has_no_grad() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::from_vec(vec![1.0f32], &[1]).unwrap());
        let x = Tensor::from_vec(vec![4.0f32], &[1]).unwrap();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let xi = tape.input(x.view());
        let pv = tape.param(p);
        let h = tape.add(xi, pv);
        let loss = tape.add(h, h);
        tape.backward(loss, &mut store.grads);
        assert!(tape.grad(xi).is_none());
        assert!(tape.grad(pv).is_none());
        assert_eq!(tape.grad(h).unwrap().data(), &[2.0]);
        assert_eq!(tape.grad(loss).unwrap().data(), &[1.0]);
        drop(tape);
        assert_eq!(store.grad(p).data(), &[2.0]);
    }

    /// The loss must be a single value.
    #[test]
    #[should_panic(expected = "the loss must have one element, but its shape is [2]")]
    fn backward_on_non_scalar_panics() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::zeros(&[2]));
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p);
        let loss = tape.add(v, v);
        tape.backward(loss, &mut store.grads);
    }
}
