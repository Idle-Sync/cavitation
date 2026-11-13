//! The autograd tape: `Graph`, `Tape`, `Var` and the backward pass.

mod params;
mod pool;

pub use params::{ParamGrads, ParamId, ParamStore, ParamValues};

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use crate::ops::Op;
use crate::{Real, Shape, TensorView};
use pool::Pool;

/// The id the next [`Tape`] gets, so variables from one tape are recognised on another.
static NEXT_TAPE_ID: AtomicU32 = AtomicU32::new(0);

/// A value recorded on a [`Tape`]: an index into that tape's nodes.
///
/// A `Var` is only meaningful on the tape that returned it; using it on another tape,
/// including a later tape of the same [`Graph`], panics.
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
pub struct Var {
    /// The node's position on its tape.
    index: u32,
    /// The id of the tape that recorded it.
    tape: u32,
}

impl Var {
    /// The node position of `self` on the tape with id `tape`.
    ///
    /// # Panics
    ///
    /// Panics if `self` was recorded on another tape.
    fn index_on(self, tape: u32) -> usize {
        assert!(
            self.tape == tape,
            "Var belongs to another tape (its tape is {}, this tape is {tape})",
            self.tape
        );
        self.index as usize
    }

    /// Fills the unused input slots of a custom operation; never read.
    pub(crate) const UNUSED: Var = Var {
        index: u32::MAX,
        tape: u32::MAX,
    };
}

/// One recorded value: the operation that made it, its shape and its elements.
#[derive(Debug)]
pub(crate) struct Node<T: Real> {
    /// The operation and the inputs it read.
    op: Op<T>,
    /// The shape of the value.
    shape: Shape,
    /// The elements, taken from the pool; empty for a parameter, whose value stays in the
    /// [`ParamValues`] the tape borrows.
    value: Vec<T>,
}

/// Reads recorded values: node buffers, and parameter values for parameter nodes.
#[derive(Clone, Copy)]
pub(crate) struct Values<'a, T: Real> {
    /// The id of the tape the nodes belong to.
    tape: u32,
    /// The tape's nodes.
    nodes: &'a [Node<T>],
    /// The parameter values the tape borrows.
    params: &'a ParamValues<T>,
}

impl<'a, T: Real> Values<'a, T> {
    /// The elements of `v`.
    fn get(self, v: Var) -> &'a [T] {
        let node = &self.nodes[v.index_on(self.tape)];
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
    buf: &'s mut [T],
    /// Whether to overwrite `buf` instead of adding to it.
    overwrite: bool,
    /// A parameter's stale flag, cleared once the contribution has been written.
    stale: Option<&'s mut bool>,
}

impl<T: Real> GradSlot<'_, T> {
    /// Contributes `src`: copies it when overwriting, adds it otherwise. A parameter's
    /// gradient becomes current only after the write.
    ///
    /// # Panics
    ///
    /// Panics if `src` and the buffer differ in length, before writing anything.
    pub(crate) fn add_from(self, src: &[T]) {
        assert_eq!(
            self.buf.len(),
            src.len(),
            "gradient length {} does not match the value's length {}",
            src.len(),
            self.buf.len()
        );
        if self.overwrite {
            self.buf.copy_from_slice(src);
        } else {
            for (d, &s) in self.buf.iter_mut().zip(src) {
                *d += s;
            }
        }
        if let Some(stale) = self.stale {
            *stale = false;
        }
    }
}

impl<T: Real> GradSlot<'_, T> {
    /// Lets `f` write the contribution straight into the gradient buffer, as a GEMM does.
    /// `f` gets the buffer and the `beta` to use: zero when this is the buffer's first
    /// contribution this step (so `f` must write every element and never read the old
    /// contents), one when it adds to what is there. A parameter's gradient becomes current
    /// only after `f` returns.
    pub(crate) fn update_with(self, f: impl FnOnce(&mut [T], T)) {
        f(self.buf, if self.overwrite { T::ZERO } else { T::ONE });
        if let Some(stale) = self.stale {
            *stale = false;
        }
    }
}

impl<T: Real> GradSlot<'_, T> {
    /// Contributes `scale * src`: writes it when overwriting, adds it otherwise. A
    /// parameter's gradient becomes current only after the write.
    ///
    /// # Panics
    ///
    /// Panics if `src` and the buffer differ in length, before writing anything.
    pub(crate) fn add_scaled_from(self, src: &[T], scale: T) {
        assert_eq!(
            self.buf.len(),
            src.len(),
            "gradient length {} does not match the value's length {}",
            src.len(),
            self.buf.len()
        );
        if self.overwrite {
            for (d, &s) in self.buf.iter_mut().zip(src) {
                *d = scale * s;
            }
        } else {
            for (d, &s) in self.buf.iter_mut().zip(src) {
                *d += scale * s;
            }
        }
        if let Some(stale) = self.stale {
            *stale = false;
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
    /// The labels recorded by operations that take them (`Tape::push_labels`).
    labels: &'a [usize],
    /// Where new gradient buffers come from.
    pool: &'a mut Pool<T>,
    /// Parameter gradients, written under the overwrite-then-add rule.
    params: &'a mut ParamGrads<T>,
}

impl<'a, T: Real> BackwardCtx<'a, T> {
    /// The elements of `v`.
    pub(crate) fn value(&self, v: Var) -> &'a [T] {
        self.values.get(v)
    }

    /// The labels the tape recorded, all operations' together; an operation reads the range
    /// it kept the start of.
    pub(crate) fn labels(&self) -> &'a [usize] {
        self.labels
    }

    /// The shape of `v`.
    pub(crate) fn shape(&self, v: Var) -> Shape {
        self.values.nodes[v.index_on(self.values.tape)].shape
    }

    /// Whether `v` takes a gradient: everything but an input does.
    pub(crate) fn takes_grad(&self, v: Var) -> bool {
        !matches!(
            self.values.nodes[v.index_on(self.values.tape)].op,
            Op::Input
        )
    }

    /// A pooled scratch buffer of `len` elements with unspecified contents; give it back
    /// with [`release`](Self::release).
    pub(crate) fn scratch(&mut self, len: usize) -> Vec<T> {
        self.pool.take(len)
    }

    /// Returns a scratch buffer to the pool.
    pub(crate) fn release(&mut self, buf: Vec<T>) {
        self.pool.give(buf);
    }
}

impl<T: Real> BackwardCtx<'_, T> {
    /// The gradient buffer of `v` to contribute to, or `None` for an input (inputs take no
    /// gradient). A node's first contribution gets a fresh pool buffer to overwrite; a
    /// parameter's slot follows its stale flag.
    pub(crate) fn slot(&mut self, v: Var) -> Option<GradSlot<'_, T>> {
        let index = v.index_on(self.values.tape);
        let node = &self.values.nodes[index];
        match node.op {
            Op::Input => None,
            Op::Param(id) => {
                let (buf, stale) = self.params.buffer_and_flag(id, &node.shape);
                Some(GradSlot {
                    buf,
                    overwrite: *stale,
                    stale: Some(stale),
                })
            }
            _ => {
                let entry = &mut self.grads[index];
                let overwrite = entry.is_none();
                let buf = entry.get_or_insert_with(|| self.pool.take(node.shape.len()));
                Some(GradSlot {
                    buf,
                    overwrite,
                    stale: None,
                })
            }
        }
    }
}

/// Storage reused across training steps: the buffer pool and the node lists. Create one,
/// then record each step on a fresh [`Tape`] from [`Graph::tape`]; after the first steps,
/// recording the same computation again allocates nothing.
///
/// The `cavitation` crate's documentation shows a full training loop built on one graph:
/// open a tape, run the network forward, record the loss, run `backward`, drop the tape,
/// step the optimiser and zero the gradients.
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
    /// Class labels recorded by the tape's loss operations, for their backward passes; empty
    /// between tapes, capacity kept.
    labels: Vec<usize>,
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
            labels: Vec::new(),
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
            id: NEXT_TAPE_ID.fetch_add(1, Ordering::Relaxed),
            clamped_logs: 0,
        }
    }

    /// Returns every node value and gradient buffer to the pool and empties the node and
    /// label lists, keeping their capacity.
    fn reclaim(&mut self) {
        self.reclaim_grads();
        self.labels.clear();
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
/// A training step records one tape; the `cavitation` crate's documentation shows the whole
/// loop of steps (the manual training loop).
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
    /// This tape's id, carried by every [`Var`] it records.
    id: u32,
    /// How many logarithms `binary_cross_entropy` has clamped on this tape.
    clamped_logs: usize,
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
        TensorView::from_parts(self.values().get(v), self.shape(v))
    }

    /// The value of parameter `id`, read from the [`ParamValues`] this tape borrows, whether
    /// or not the tape has recorded it.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not belong to the store whose values this tape borrows.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[2], 0.25f32));
    /// let mut graph = Graph::new();
    /// let tape = graph.tape(&store.values);
    /// assert_eq!(tape.param_value(p).data(), &[0.25, 0.25]);
    /// ```
    pub fn param_value(&self, id: ParamId) -> TensorView<'_, T> {
        self.params.get(id).view()
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
        let index = v.index_on(self.id);
        let node = &self.graph.nodes[index];
        match node.op {
            Op::Input | Op::Param(_) => None,
            _ => {
                let buf = self.graph.grads.get(index)?.as_deref()?;
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
        self.graph.nodes[v.index_on(self.id)].shape
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
    /// shape), if `loss` was not recorded on this tape, or if `grads` belongs to another
    /// [`ParamStore`] than the values this tape reads.
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
        let shape = self.shape(loss);
        assert!(
            shape.len() == 1,
            "backward: the loss must have one element, but its shape is {:?}",
            shape.dims()
        );
        self.backward_seeded(loss, &[T::ONE], grads);
    }

    /// The same pass as [`backward`](Tape::backward), with the time it spends split at
    /// `marks`. The time spent on the nodes recorded after `marks[j - 1]` and up to and
    /// including `marks[j]` is added to `times[j]` (for `j = 0`, every node up to `marks[0]`),
    /// and the time spent on the nodes after the last mark to `times[marks.len()]`. With each
    /// layer's output as a mark, `times[j]` is layer `j`'s backward time and the last entry
    /// is the loss's.
    ///
    /// The gradients are bit for bit those `backward` gives: both run one node loop. The
    /// clock is read once before the walk and once at each mark boundary the walk crosses,
    /// never per node. Times are added, not written, so a caller can sum them over steps;
    /// set `times` to zero to start afresh. Nothing is allocated beyond what `backward`
    /// allocates.
    ///
    /// Marks must be in the order they were recorded; two equal marks are allowed and give
    /// an empty segment. A segment with no node at or before the loss (a mark recorded after
    /// the loss, say) adds nothing.
    ///
    /// # Panics
    ///
    /// Panics if `times.len()` is not `marks.len() + 1`, if a mark comes before the one
    /// ahead of it, if a mark or `loss` was not recorded on this tape, and under the same
    /// conditions as [`backward`](Tape::backward).
    ///
    /// ```
    /// use std::time::Duration;
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let mut store = ParamStore::new();
    /// let a = store.add(Tensor::full(&[1], 2.0f32));
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let va = tape.param(a);
    /// let h = tape.mul(va, va); // segment 0: everything up to `h`
    /// let loss = tape.add(h, h); // segment 1: after `h`
    /// let mut times = [Duration::ZERO; 2];
    /// tape.backward_timed(loss, &mut store.grads, &[h], &mut times);
    /// drop(tape);
    /// assert_eq!(store.grad(a).data(), &[8.0]);
    /// ```
    pub fn backward_timed(
        &mut self,
        loss: Var,
        grads: &mut ParamGrads<T>,
        marks: &[Var],
        times: &mut [Duration],
    ) {
        self.backward_timed_with(loss, grads, marks, times, Instant::now);
    }

    /// [`backward_timed`](Tape::backward_timed) reading the clock through `now`, so a test
    /// can count the reads; `backward_timed` passes `Instant::now`.
    ///
    /// # Panics
    ///
    /// Panics under the conditions `backward_timed` lists.
    fn backward_timed_with(
        &mut self,
        loss: Var,
        grads: &mut ParamGrads<T>,
        marks: &[Var],
        times: &mut [Duration],
        now: impl FnMut() -> Instant,
    ) {
        let shape = self.shape(loss);
        assert!(
            shape.len() == 1,
            "backward_timed: the loss must have one element, but its shape is {:?}",
            shape.dims()
        );
        assert!(
            times.len() == marks.len() + 1,
            "backward_timed: {} marks need {} times, but times has {}",
            marks.len(),
            marks.len() + 1,
            times.len()
        );
        let mut previous = 0;
        for (j, mark) in marks.iter().enumerate() {
            let index = mark.index_on(self.id);
            assert!(
                j == 0 || index >= previous,
                "backward_timed: mark {j} (node {index}) was recorded before mark {} (node \
                 {previous}); marks go in the order they were recorded",
                j - 1
            );
            previous = index;
        }
        self.backward_pass(loss, &[T::ONE], grads, Some((marks, times, now)));
    }

    /// Like [`backward`](Tape::backward) for an output of any size: seeds the gradient of
    /// `out` with `seed` (the gradient of the loss with respect to each element of `out`)
    /// and runs the same reverse pass.
    ///
    /// # Panics
    ///
    /// Panics if `seed` does not have as many elements as `out`, if `out` was not recorded
    /// on this tape, or if `grads` belongs to another [`ParamStore`] than the values this
    /// tape reads.
    pub(crate) fn backward_seeded(&mut self, out: Var, seed: &[T], grads: &mut ParamGrads<T>) {
        self.backward_pass(out, seed, grads, None::<(_, _, fn() -> Instant)>);
    }

    /// The reverse pass behind [`backward`](Tape::backward), `backward_seeded` and
    /// [`backward_timed`](Tape::backward_timed): seeds `out`'s gradient with `seed` and
    /// walks the nodes from `out` down to the first. With `timing`, the walk is split at the
    /// marks (already checked to be in recording order) as [`segments`] gives them, and
    /// each segment's time is added to its entry; the node loop is the same either way. The
    /// clock, `now`, is read once before the walk and once after each segment.
    ///
    /// # Panics
    ///
    /// Panics under the conditions [`backward_seeded`](Tape::backward_seeded) lists.
    fn backward_pass(
        &mut self,
        out: Var,
        seed: &[T],
        grads: &mut ParamGrads<T>,
        timing: Option<(&[Var], &mut [Duration], impl FnMut() -> Instant)>,
    ) {
        assert!(
            grads.store_id() == self.params.store_id(),
            "backward: the grads belong to another ParamStore than the values this tape reads \
             (grads of store {}, values of store {})",
            grads.store_id(),
            self.params.store_id()
        );
        let loss_index = out.index_on(self.id);
        let shape = self.graph.nodes[loss_index].shape;
        assert!(
            shape.len() == seed.len(),
            "backward: the seed has {} elements, but the output's shape is {:?}",
            seed.len(),
            shape.dims()
        );
        self.graph.reclaim_grads();
        let Graph {
            pool,
            nodes,
            grads: node_grads,
            labels,
        } = &mut *self.graph;
        let labels: &[usize] = labels;
        node_grads.resize_with(nodes.len(), || None);
        let values = Values {
            tape: self.id,
            nodes,
            params: self.params,
        };
        let mut seed_ctx = BackwardCtx {
            values,
            grads: node_grads,
            labels,
            pool,
            params: grads,
        };
        if let Some(slot) = seed_ctx.slot(out) {
            slot.add_from(seed);
        }
        let tape = self.id;
        // Differentiates the nodes `range`, last first.
        let mut walk = |range: std::ops::Range<usize>| {
            for i in range.rev() {
                let (before, rest) = node_grads.split_at_mut(i);
                let Some(out_grad) = rest[0].as_deref() else {
                    continue;
                };
                let mut ctx = BackwardCtx {
                    values,
                    grads: before,
                    labels,
                    pool,
                    params: grads,
                };
                let this = Var {
                    index: i as u32,
                    tape,
                };
                nodes[i].op.backward(this, out_grad, &mut ctx);
            }
        };
        let Some((marks, times, mut now)) = timing else {
            walk(0..loss_index + 1);
            return;
        };
        let mut clock = now();
        for (j, range) in segments(marks, tape, loss_index) {
            walk(range);
            let t = now();
            times[j] += t - clock;
            clock = t;
        }
    }

    /// Reads recorded values.
    fn values(&self) -> Values<'_, T> {
        Values {
            tape: self.id,
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

    /// Adds `n` to the count of clamped logarithms.
    pub(crate) fn add_clamped_logs(&mut self, n: usize) {
        self.clamped_logs += n;
    }

    /// The number of logarithms that `binary_cross_entropy` clamped at -100 while recording
    /// on this tape and that carry weight: `ln p` with a non-zero target, `ln(1 - p)` with a
    /// target other than 1 (see [`Tape::binary_cross_entropy`]). A saturated correct
    /// prediction is not counted. It starts at 0 on every new tape; the health checks read it
    /// and turn a non-zero count into a warning.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Reduction, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let p = tape.input(Tensor::from_vec(vec![0.0f32, 0.5], &[2]).unwrap().view());
    /// let t = tape.input(Tensor::from_vec(vec![1.0f32, 1.0], &[2]).unwrap().view());
    /// assert_eq!(tape.clamped_logs(), 0);
    /// tape.binary_cross_entropy(p, t, Reduction::Mean);
    /// assert_eq!(tape.clamped_logs(), 1);
    /// ```
    pub fn clamped_logs(&self) -> usize {
        self.clamped_logs
    }

    /// Copies `labels` into the graph's label buffer, which keeps its capacity from step to
    /// step, and returns where they start.
    pub(crate) fn push_labels(&mut self, labels: &[usize]) -> usize {
        let start = self.graph.labels.len();
        self.graph.labels.extend_from_slice(labels);
        start
    }

    /// Appends a node and returns its variable.
    pub(crate) fn push(&mut self, op: Op<T>, shape: Shape, value: Vec<T>) -> Var {
        debug_assert!(matches!(op, Op::Param(_)) || value.len() == shape.len());
        let index = u32::try_from(self.graph.nodes.len()).expect("a tape holds at most 2^32 nodes");
        self.graph.nodes.push(Node { op, shape, value });
        Var {
            index,
            tape: self.id,
        }
    }
}

/// The non-empty segments of a timed backward pass over a tape whose loss is node
/// `loss_index`, in the order the pass walks them (last first): each segment's entry in
/// `times` and its node range. Segment `j` is the nodes after `marks[j - 1]` up to and
/// including `marks[j]` (from node 0 for `j = 0`); segment `marks.len()` is the nodes after
/// the last mark up to the loss. Nodes after the loss belong to no segment, so a segment
/// that would hold only such nodes, or none (between equal marks), is left out. `marks`
/// must be in recording order, on the tape with id `tape`.
fn segments(
    marks: &[Var],
    tape: u32,
    loss_index: usize,
) -> impl Iterator<Item = (usize, std::ops::Range<usize>)> + '_ {
    // Each segment ends where the one above it began.
    let mut end = loss_index + 1;
    (0..=marks.len()).rev().filter_map(move |j| {
        let start = match j {
            0 => 0,
            _ => marks[j - 1].index_on(tape) + 1,
        };
        if start >= end {
            return None;
        }
        let range = start..end;
        end = start;
        Some((j, range))
    })
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
    use std::time::Duration;

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

    /// A `Var` kept from a dropped tape panics instead of reading the node now at its index.
    #[test]
    #[should_panic(expected = "Var belongs to another tape")]
    fn stale_var_from_dropped_tape_panics() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::full(&[1], 1.0f32));
        let q = store.add(Tensor::full(&[1], 3.0f32));
        let x = Tensor::full(&[1], 2.0f32);
        let mut graph = Graph::new();
        let old = {
            let mut tape = graph.tape(&store.values);
            let v = tape.param(p);
            tape.add(v, v)
        };
        let mut tape = graph.tape(&store.values);
        tape.input(x.view());
        tape.param(q);
        tape.value(old);
    }

    /// A `Var` from a tape of another graph panics when used.
    #[test]
    #[should_panic(expected = "Var belongs to another tape")]
    fn var_from_other_graph_panics() {
        let store = ParamStore::new();
        let x = Tensor::full(&[1], 2.0f32);
        let (mut g1, mut g2) = (Graph::new(), Graph::new());
        let mut t1 = g1.tape(&store.values);
        let mut t2 = g2.tape(&store.values);
        let a = t1.input(x.view());
        t2.input(x.view());
        t2.add(a, a);
    }

    /// Gradients of one store cannot receive a backward pass over another store's values.
    #[test]
    #[should_panic(expected = "grads belong to another ParamStore")]
    fn backward_with_other_stores_grads_panics() {
        let mut s1 = ParamStore::new();
        let mut s2 = ParamStore::new();
        let p = s1.add(Tensor::full(&[1], 1.0f32));
        s2.add(Tensor::full(&[1], 1.0f32));
        let mut graph = Graph::new();
        let mut tape = graph.tape(&s1.values);
        let v = tape.param(p);
        let loss = tape.add(v, v);
        tape.backward(loss, &mut s2.grads);
    }

    /// A store with one parameter of value 3, its id and a graph, for the timed-pass tests.
    fn square_store() -> (ParamStore, crate::ParamId, Graph) {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::full(&[1], 3.0f32));
        (store, p, Graph::new())
    }

    /// `times` must have one entry more than `marks`.
    #[test]
    #[should_panic(expected = "backward_timed: 1 marks need 2 times, but times has 1")]
    fn backward_timed_checks_times_length() {
        let (mut store, p, mut graph) = square_store();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p);
        let h = tape.mul(v, v);
        let loss = tape.add(h, h);
        tape.backward_timed(loss, &mut store.grads, &[h], &mut [Duration::ZERO]);
    }

    /// Marks out of recording order are refused, naming both.
    #[test]
    #[should_panic(expected = "mark 1 (node 1) was recorded before mark 0 (node 2)")]
    fn backward_timed_checks_mark_order() {
        let (mut store, p, mut graph) = square_store();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p);
        let h = tape.mul(v, v);
        let g = tape.add(h, h);
        let loss = tape.add(g, g);
        tape.backward_timed(loss, &mut store.grads, &[g, h], &mut [Duration::ZERO; 3]);
    }

    /// A mark from another tape panics like any foreign `Var`.
    #[test]
    #[should_panic(expected = "Var belongs to another tape")]
    fn backward_timed_checks_mark_tape() {
        let (mut store, p, mut graph) = square_store();
        let mut other = Graph::new();
        let foreign = other.tape(&store.values).param(p);
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p);
        let loss = tape.mul(v, v);
        tape.backward_timed(loss, &mut store.grads, &[foreign], &mut [Duration::ZERO; 2]);
    }

    /// Equal marks give an empty segment and a mark after the loss gives nothing to its
    /// segments; neither changes the gradients, and nothing is timed for a segment with no
    /// node.
    #[test]
    fn backward_timed_allows_equal_marks_and_marks_after_the_loss() {
        let (mut store, p, mut graph) = square_store();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p);
        let h = tape.mul(v, v);
        let loss = tape.add(h, h);
        let after = tape.add(loss, loss);
        let mut times = [Duration::ZERO; 4];
        tape.backward_timed(loss, &mut store.grads, &[h, h, after], &mut times);
        assert_eq!(tape.grad(h).unwrap().data(), &[2.0]);
        drop(tape);
        assert_eq!(store.grad(p).data(), &[12.0]);
        assert_eq!(
            times[1],
            Duration::ZERO,
            "the empty segment between equal marks"
        );
        assert_eq!(
            times[3],
            Duration::ZERO,
            "the segment after a mark past the loss"
        );
    }

    /// `Var`s at the given node positions of tape 0, as marks for [`super::segments`].
    fn at(indices: &[u32]) -> Vec<super::Var> {
        indices
            .iter()
            .map(|&index| super::Var { index, tape: 0 })
            .collect()
    }

    /// The segments of a timed pass, case by case: each mark's own node is in its segment,
    /// equal marks and marks at or past the loss leave segments out, and no marks give one
    /// segment of every node up to the loss.
    #[test]
    fn segments_cover_each_node_once() {
        let seg = |marks: &[u32], loss| super::segments(&at(marks), 0, loss).collect::<Vec<_>>();
        // Layers end at nodes 2, 5 and 7; the loss is node 9.
        assert_eq!(
            seg(&[2, 5, 7], 9),
            [(3, 8..10), (2, 6..8), (1, 3..6), (0, 0..3)]
        );
        // Equal marks: segment 1 holds nothing.
        assert_eq!(seg(&[2, 2, 7], 9), [(3, 8..10), (2, 3..8), (0, 0..3)]);
        // A mark at the loss: nothing after it.
        assert_eq!(seg(&[2, 9], 9), [(1, 3..10), (0, 0..3)]);
        // A mark after the loss: its segment is cut at the loss, the one after is empty.
        assert_eq!(seg(&[2, 12], 9), [(1, 3..10), (0, 0..3)]);
        // A first mark at node 0.
        assert_eq!(seg(&[0, 4], 6), [(2, 5..7), (1, 1..5), (0, 0..1)]);
        // No marks.
        assert_eq!(seg(&[], 4), [(0, 0..5)]);
    }

    /// The timed pass reads the clock once before the walk and once after each non-empty
    /// segment, never per node, and each segment gets the time between its two reads.
    #[test]
    fn backward_timed_reads_the_clock_at_boundaries_only() {
        let (mut store, p, mut graph) = square_store();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p); // node 0
        let a = tape.mul(v, v); // 1
        let b = tape.add(a, a); // 2
        let c = tape.mul(b, b); // 3
        let d = tape.add(c, c); // 4
        let loss = tape.sum(d); // 5
        let base = std::time::Instant::now();
        let mut reads = 0u32;
        // Each read is one millisecond after the last.
        let clock = || {
            reads += 1;
            base + Duration::from_millis(u64::from(reads))
        };
        let mut times = [Duration::ZERO; 5];
        // Segments: 4 = 5..6, 3 = 3..5, 2 = empty (equal marks), 1 = 2..3, 0 = 0..2.
        tape.backward_timed_with(loss, &mut store.grads, &[a, b, b, d], &mut times, clock);
        let ms = Duration::from_millis(1);
        assert_eq!(times, [ms, ms, Duration::ZERO, ms, ms]);
        assert_eq!(reads, 4 + 1);
    }

    /// The timed pass checks the loss's size as `backward` does.
    #[test]
    #[should_panic(expected = "backward_timed: the loss must have one element")]
    fn backward_timed_on_non_scalar_panics() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::zeros(&[2]));
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(p);
        tape.backward_timed(v, &mut store.grads, &[], &mut [Duration::ZERO]);
    }

    /// A gradient write that panics on a shape mismatch leaves the parameter stale.
    #[test]
    fn failed_gradient_write_leaves_param_stale() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::full(&[2], 1.0f32));
        // Shrinking the value breaks the documented rule on purpose: the gradient buffer
        // keeps two elements while the value has one.
        *store.value_mut(p) = Tensor::full(&[1], 1.0f32);
        let mut graph = Graph::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut tape = graph.tape(&store.values);
            let v = tape.param(p);
            let loss = tape.add(v, v);
            tape.backward(loss, &mut store.grads);
        }));
        let message = result.unwrap_err();
        let message = message.downcast_ref::<String>().unwrap();
        assert!(message.contains("its gradient has shape [2]"), "{message}");
        assert!(store.grads.is_stale(p));
    }
}
