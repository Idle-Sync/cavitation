//! User-defined operations: the `CustomOp` trait and `Tape::custom`.

use std::fmt;
use std::sync::Arc;

use crate::graph::{BackwardCtx, Var};
use crate::ops::Op;
use crate::{Real, Shape, Tape, TensorView, TensorViewMut};

/// The most inputs a custom operation can take.
pub(crate) const MAX_INPUTS: usize = 4;

/// An operation you define yourself: an activation, a loss, a layer, anything with a forward
/// pass and its derivative. Record it with [`Tape::custom`]; it then takes part in
/// [`Tape::backward`] like a built-in operation.
///
/// Implement it for every [`Real`] type (`impl<T: Real> CustomOp<T> for MyOp`) rather than
/// for `f32` alone: then the same code trains in `f32` and runs through [`gradcheck`] in
/// `f64`, which compares your `backward` with finite differences of your `forward`.
///
/// [`gradcheck`]: crate::gradcheck
///
/// # Examples
///
/// The square, `x²`, whose derivative is `2x`:
///
/// ```
/// use std::sync::Arc;
/// use cavitation_core::{CustomOp, Real, Shape, Tensor, TensorView, TensorViewMut, gradcheck};
///
/// struct Square;
///
/// impl<T: Real> CustomOp<T> for Square {
///     fn name(&self) -> &str {
///         "square"
///     }
///
///     // One input; the output has its shape.
///     fn output_shape(&self, inputs: &[Shape]) -> Shape {
///         inputs[0]
///     }
///
///     fn forward(&self, inputs: &[TensorView<T>], mut output: TensorViewMut<T>) {
///         let x = inputs[0].data();
///         for (y, &x) in output.data_mut().iter_mut().zip(x) {
///             *y = x * x;
///         }
///     }
///
///     // d(x²)/dx = 2x, times the gradient arriving from above.
///     fn backward(
///         &self,
///         inputs: &[TensorView<T>],
///         _output: TensorView<T>,
///         grad_output: TensorView<T>,
///         grad_inputs: &mut [TensorViewMut<T>],
///     ) {
///         let x = inputs[0].data();
///         let g = grad_output.data();
///         for (i, dx) in grad_inputs[0].data_mut().iter_mut().enumerate() {
///             *dx = (x[i] + x[i]) * g[i];
///         }
///     }
/// }
///
/// let op: Arc<dyn CustomOp<f64> + Send + Sync> = Arc::new(Square);
/// let x = Tensor::from_vec(vec![0.5, -1.0, 2.0], &[3]).unwrap();
/// gradcheck(&[x], |tape, v| tape.custom(&op, &[v[0]])).assert_passed();
/// ```
pub trait CustomOp<T: Real> {
    /// The operation's name, used in panic messages.
    fn name(&self) -> &str;

    /// The output's shape for inputs of these shapes. Panic here, naming the operation and
    /// the shapes, if the operation does not accept them.
    fn output_shape(&self, inputs: &[Shape]) -> Shape;

    /// Computes the output from the inputs. `output` has the shape
    /// [`output_shape`](Self::output_shape) returned and arrives filled with zeros.
    fn forward(&self, inputs: &[TensorView<T>], output: TensorViewMut<T>);

    /// Computes the gradient of the loss with respect to each input, given the inputs, the
    /// output `forward` computed, and `grad_output`, the gradient of the loss with respect
    /// to the output.
    ///
    /// `grad_inputs` holds one buffer per input, in the same order and of the same shape,
    /// filled with zeros. Write (or add) the gradient into each. Every input gets its own
    /// buffer, even when the same [`Var`] is passed twice; the tape adds each one into that
    /// input's gradient afterwards.
    fn backward(
        &self,
        inputs: &[TensorView<T>],
        output: TensorView<T>,
        grad_output: TensorView<T>,
        grad_inputs: &mut [TensorViewMut<T>],
    );
}

impl<T: Real> fmt::Debug for dyn CustomOp<T> + Send + Sync + '_ {
    /// Shows the operation's name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CustomOp({:?})", self.name())
    }
}

impl<T: Real> Tape<'_, T> {
    /// Records the user-defined operation `op` on `inputs` (at most 4) and runs its forward
    /// pass. The tape keeps a clone of the `Arc`, so a model holding its operations stays
    /// `Send + Sync`. After the first steps, recording and differentiating the same
    /// operation again allocates nothing.
    ///
    /// # Panics
    ///
    /// Panics if `inputs` has more than 4 entries or the output shape has more elements
    /// than fit in memory (both messages name the operation), if an input was not recorded
    /// on this tape, or if `op` itself panics (for example on shapes it rejects).
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cavitation_core::{CustomOp, Graph, ParamStore, Real, Shape, Tensor, TensorView, TensorViewMut};
    ///
    /// /// `x + 1`.
    /// struct AddOne;
    ///
    /// impl<T: Real> CustomOp<T> for AddOne {
    ///     fn name(&self) -> &str {
    ///         "add_one"
    ///     }
    ///     fn output_shape(&self, inputs: &[Shape]) -> Shape {
    ///         inputs[0]
    ///     }
    ///     fn forward(&self, inputs: &[TensorView<T>], mut output: TensorViewMut<T>) {
    ///         for (y, &x) in output.data_mut().iter_mut().zip(inputs[0].data()) {
    ///             *y = x + T::ONE;
    ///         }
    ///     }
    ///     fn backward(
    ///         &self,
    ///         _inputs: &[TensorView<T>],
    ///         _output: TensorView<T>,
    ///         grad_output: TensorView<T>,
    ///         grad_inputs: &mut [TensorViewMut<T>],
    ///     ) {
    ///         grad_inputs[0].data_mut().copy_from_slice(grad_output.data());
    ///     }
    /// }
    ///
    /// let op: Arc<dyn CustomOp<f32> + Send + Sync> = Arc::new(AddOne);
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[2], 1.5f32));
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let v = tape.param(p);
    /// let y = tape.custom(&op, &[v]);
    /// assert_eq!(tape.value(y).data(), &[2.5, 2.5]);
    /// let loss = tape.sum(y);
    /// tape.backward(loss, &mut store.grads);
    /// drop(tape);
    /// assert_eq!(store.grad(p).data(), &[1.0, 1.0]);
    /// ```
    pub fn custom(&mut self, op: &Arc<dyn CustomOp<T> + Send + Sync>, inputs: &[Var]) -> Var {
        let count = inputs.len();
        assert!(
            count <= MAX_INPUTS,
            "custom op {}: takes at most {MAX_INPUTS} inputs, but got {count}",
            op.name()
        );
        let mut shapes = [Shape::new(&[]); MAX_INPUTS];
        for (s, &v) in shapes.iter_mut().zip(inputs) {
            *s = self.shape(v);
        }
        let shape = op.output_shape(&shapes[..count]);
        let len = element_count::<T>(&shape).unwrap_or_else(|| {
            panic!(
                "custom op {}: output shape {:?} has more elements than fit in memory",
                op.name(),
                shape.dims()
            )
        });
        let mut out = self.take_buffer(len);
        out.fill(T::ZERO);
        let views: [TensorView<'_, T>; MAX_INPUTS] = std::array::from_fn(|i| match inputs.get(i) {
            Some(&v) => self.value(v),
            None => empty_view(),
        });
        op.forward(&views[..count], TensorViewMut::from_parts(&mut out, shape));
        let mut slots = [Var::UNUSED; MAX_INPUTS];
        slots[..count].copy_from_slice(inputs);
        // `count <= MAX_INPUTS`, checked above, so it fits in a u8.
        self.push(Op::Custom(Arc::clone(op), slots, count as u8), shape, out)
    }
}

/// The number of elements of `shape`, or `None` if a buffer of them would not fit in memory
/// (more than `isize::MAX` bytes).
fn element_count<T>(shape: &Shape) -> Option<usize> {
    let len = shape
        .dims()
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))?;
    let bytes = len.checked_mul(size_of::<T>())?;
    (bytes <= isize::MAX as usize).then_some(len)
}

/// A view of no elements, for the unused input slots.
fn empty_view<'a, T: Real>() -> TensorView<'a, T> {
    TensorView::from_parts(&[], Shape::new(&[0]))
}

/// Backward of a custom operation: hands `op` one zeroed pooled buffer per input slot, then
/// adds each slot's buffer into its input's gradient and returns the buffers to the pool.
pub(super) fn backward<T: Real>(
    op: &(dyn CustomOp<T> + Send + Sync),
    inputs: &[Var],
    out: Var,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    if !inputs.iter().any(|&v| ctx.takes_grad(v)) {
        return;
    }
    let count = inputs.len();
    let mut shapes = [Shape::new(&[0]); MAX_INPUTS];
    let mut bufs: [Vec<T>; MAX_INPUTS] = std::array::from_fn(|_| Vec::new());
    for i in 0..count {
        shapes[i] = ctx.shape(inputs[i]);
        let mut buf = ctx.scratch(shapes[i].len());
        buf.fill(T::ZERO);
        bufs[i] = buf;
    }
    {
        let views: [TensorView<'_, T>; MAX_INPUTS] = std::array::from_fn(|i| match inputs.get(i) {
            Some(&v) => TensorView::from_parts(ctx.value(v), shapes[i]),
            None => empty_view(),
        });
        let out_shape = ctx.shape(out);
        let output = TensorView::from_parts(ctx.value(out), out_shape);
        let grad_output = TensorView::from_parts(out_grad, out_shape);
        let [b0, b1, b2, b3] = &mut bufs;
        let mut grads = [
            TensorViewMut::from_parts(b0, shapes[0]),
            TensorViewMut::from_parts(b1, shapes[1]),
            TensorViewMut::from_parts(b2, shapes[2]),
            TensorViewMut::from_parts(b3, shapes[3]),
        ];
        op.backward(&views[..count], output, grad_output, &mut grads[..count]);
    }
    for (&v, buf) in inputs.iter().zip(&bufs) {
        if let Some(slot) = ctx.slot(v) {
            slot.add_from(buf);
        }
    }
    for buf in bufs {
        ctx.release(buf);
    }
}

/// Tests for the tape's checks on custom operations.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, ParamStore, Tensor};

    /// An operation that reports the output shape it was built with and computes nothing.
    struct Fixed(Shape);

    impl<T: Real> CustomOp<T> for Fixed {
        fn name(&self) -> &str {
            "fixed"
        }

        fn output_shape(&self, _inputs: &[Shape]) -> Shape {
            self.0
        }

        fn forward(&self, _inputs: &[TensorView<T>], _output: TensorViewMut<T>) {}

        fn backward(
            &self,
            _inputs: &[TensorView<T>],
            _output: TensorView<T>,
            _grad_output: TensorView<T>,
            _grad_inputs: &mut [TensorViewMut<T>],
        ) {
        }
    }

    /// Five inputs are one more than a custom operation takes.
    #[test]
    #[should_panic(expected = "custom op fixed: takes at most 4 inputs, but got 5")]
    fn more_than_four_inputs_panics() {
        let op: Arc<dyn CustomOp<f32> + Send + Sync> = Arc::new(Fixed(Shape::new(&[1])));
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = Tensor::full(&[1], 1.0f32);
        let v = tape.input(x.view());
        tape.custom(&op, &[v; 5]);
    }

    /// An output shape whose element count overflows is rejected before any buffer is
    /// taken.
    #[test]
    #[should_panic(expected = "custom op fixed: output shape [")]
    fn oversized_output_shape_panics() {
        let op: Arc<dyn CustomOp<f32> + Send + Sync> =
            Arc::new(Fixed(Shape::new(&[usize::MAX, 2])));
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        tape.custom(&op, &[]);
    }

    /// The recorded node shows the operation's name when debug-printed.
    #[test]
    fn debug_shows_the_name() {
        let op: Arc<dyn CustomOp<f32> + Send + Sync> = Arc::new(Fixed(Shape::new(&[1])));
        assert_eq!(format!("{op:?}"), "CustomOp(\"fixed\")");
    }
}
