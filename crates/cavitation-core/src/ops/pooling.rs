//! 2-D pooling on the tape: `max_pool2d`, `avg_pool2d` and `adaptive_avg_pool2d`, with their
//! backward passes, on images stored as flat NCHW rows.

use crate::graph::{BackwardCtx, Var};
use crate::kernels::pooling::{
    Pool2dGeom, WORDS, adaptive_avg_pool2d, adaptive_avg_pool2d_backward, avg_pool2d,
    avg_pool2d_backward, max_pool2d, max_pool2d_backward,
};
use crate::ops::Op;
use crate::{Real, Shape, Tape};

/// The batch size of a pooling input of shape `sx` holding rows of `width`, after checking
/// the shape.
///
/// # Panics
///
/// Panics, naming `op`, the width and the shape, unless `sx` is `[batch, width]`.
fn batch_of(op: &str, sx: &Shape, width: usize) -> usize {
    assert!(
        sx.rank() == 2 && sx.dims()[1] == width,
        "{op}: the input must be [batch, {width}], got {:?}",
        sx.dims()
    );
    sx.dims()[0]
}

impl<T: Real> Tape<'_, T> {
    /// Records 2-D max pooling of the images in `x`, as PyTorch's `max_pool2d` computes it on
    /// the CPU: each output is the largest value of its window, a tie going to the first in
    /// row-major order, a NaN winning (the last of several), padding never winning. The
    /// gradient goes to each window's winner alone.
    ///
    /// `x` is `[batch, channels · height · width]`, each row one image in NCHW order; the
    /// result is `[batch, channels · out_height · out_width]`.
    ///
    /// # Panics
    ///
    /// Panics if `x` is not `[batch, geom.in_width()]`, naming the operation and the shape.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let geom = Pool2dGeom::new(1, (2, 4), (2, 2)).unwrap();
    /// let x = Tensor::from_vec(vec![1.0f32, 5.0, 2.0, 0.0, 3.0, 4.0, 8.0, 7.0], &[1, 8]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.max_pool2d(x, geom);
    /// assert_eq!(tape.value(y).data(), &[5.0, 8.0]);
    /// ```
    pub fn max_pool2d(&mut self, x: Var, geom: Pool2dGeom) -> Var {
        let batch = batch_of("max_pool2d", &self.shape(x), geom.in_width());
        let outputs = batch * geom.out_width();
        let mut out = self.take_buffer(outputs);
        let words = self.push_labels(&geom.to_words());
        let (xv, _, winners) = self.data_and_label_space(x, outputs);
        max_pool2d(&geom, batch, xv, &mut out, Some(winners));
        let so = Shape::new(&[batch, geom.out_width()]);
        self.push(Op::MaxPool2d(x, words), so, out)
    }

    /// Records 2-D average pooling of the images in `x`, as PyTorch's `avg_pool2d` computes
    /// it: each output is the sum of its window's pixels inside the image divided by
    /// `divisor_override` if given, else by the window's size counting the padding when
    /// `count_include_pad` (PyTorch's default) or the pixels summed when not. The geometry
    /// must not be dilated.
    ///
    /// # Panics
    ///
    /// Panics if `x` is not `[batch, geom.in_width()]`, the geometry is dilated or
    /// `divisor_override` is 0, naming the operation.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let geom = Pool2dGeom::new(1, (2, 4), (2, 2)).unwrap();
    /// let x = Tensor::from_vec(vec![1.0f32, 5.0, 2.0, 0.0, 3.0, 4.0, 8.0, 6.0], &[1, 8]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.avg_pool2d(x, geom, true, None);
    /// assert_eq!(tape.value(y).data(), &[3.25, 4.0]);
    /// ```
    pub fn avg_pool2d(
        &mut self,
        x: Var,
        geom: Pool2dGeom,
        count_include_pad: bool,
        divisor_override: Option<usize>,
    ) -> Var {
        let batch = batch_of("avg_pool2d", &self.shape(x), geom.in_width());
        let mut out = self.take_buffer(batch * geom.out_width());
        avg_pool2d(
            &geom,
            batch,
            self.data(x),
            count_include_pad,
            divisor_override,
            &mut out,
        );
        let start = self.push_labels(&geom.to_words());
        // The divisor settings: whether the padding counts, and the override (0 for none,
        // which `avg_pool2d` has already refused as an override).
        self.push_labels(&[
            usize::from(count_include_pad),
            divisor_override.unwrap_or(0),
        ]);
        let so = Shape::new(&[batch, geom.out_width()]);
        self.push(Op::AvgPool2d(x, start), so, out)
    }

    /// Records 2-D adaptive average pooling of the images in `x` (`channels` planes of
    /// `input`, height then width) to `output`, as PyTorch's `adaptive_avg_pool2d` computes
    /// it: output `(i, j)` is the mean of rows `floor(i · h / oh) .. ceil((i + 1) · h / oh)`
    /// and the same across. Output `(1, 1)` is global average pooling.
    ///
    /// # Panics
    ///
    /// Panics if a size is 0 or `x` is not `[batch, channels · h · w]`, naming the operation.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 6.0, 0.0, 0.0, 0.0, 4.0], &[1, 8]).unwrap();
    /// let x = tape.input(x.view());
    /// let y = tape.adaptive_avg_pool2d(x, 2, (2, 2), (1, 1));
    /// assert_eq!(tape.value(y).data(), &[3.0, 1.0]);
    /// ```
    pub fn adaptive_avg_pool2d(
        &mut self,
        x: Var,
        channels: usize,
        input: (usize, usize),
        output: (usize, usize),
    ) -> Var {
        let width = channels * input.0 * input.1;
        let batch = batch_of("adaptive_avg_pool2d", &self.shape(x), width);
        let out_w = channels * output.0 * output.1;
        let mut out = self.take_buffer(batch * out_w);
        adaptive_avg_pool2d(batch, channels, input, output, self.data(x), &mut out);
        let start = self.push_labels(&[channels, input.0, input.1, output.0, output.1]);
        let so = Shape::new(&[batch, out_w]);
        self.push(Op::AdaptiveAvgPool2d(x, start), so, out)
    }
}

/// Backward of `max_pool2d`: each output gradient goes to its window's winner, recorded
/// after the geometry in the labels.
pub(super) fn max_backward<T: Real>(
    x: Var,
    start: usize,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let labels = ctx.labels();
    let geom = Pool2dGeom::from_words(&labels[start..start + WORDS]);
    let batch = ctx.shape(x).dims()[0];
    let winners = &labels[start + WORDS..][..batch * geom.out_width()];
    if let Some(slot) = ctx.slot(x) {
        slot.update_with(|buf, beta| {
            if beta == T::ZERO {
                buf.fill(T::ZERO);
            }
            max_pool2d_backward(&geom, batch, out_grad, winners, buf);
        });
    }
}

/// Backward of `avg_pool2d`: each output gradient, divided by its divisor, spread over its
/// window.
pub(super) fn avg_backward<T: Real>(
    x: Var,
    start: usize,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let labels = ctx.labels();
    let geom = Pool2dGeom::from_words(&labels[start..start + WORDS]);
    let (include, over) = (
        labels[start + WORDS] == 1,
        Some(labels[start + WORDS + 1]).filter(|&d| d > 0),
    );
    let batch = ctx.shape(x).dims()[0];
    if let Some(slot) = ctx.slot(x) {
        slot.update_with(|buf, beta| {
            if beta == T::ZERO {
                buf.fill(T::ZERO);
            }
            avg_pool2d_backward(&geom, batch, out_grad, include, over, buf);
        });
    }
}

/// Backward of `adaptive_avg_pool2d`: each output gradient, divided by its window's size,
/// spread over the window.
pub(super) fn adaptive_backward<T: Real>(
    x: Var,
    start: usize,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let l = &ctx.labels()[start..start + 5];
    let (channels, input, output) = (l[0], (l[1], l[2]), (l[3], l[4]));
    let batch = ctx.shape(x).dims()[0];
    if let Some(slot) = ctx.slot(x) {
        slot.update_with(|buf, beta| {
            if beta == T::ZERO {
                buf.fill(T::ZERO);
            }
            adaptive_avg_pool2d_backward(batch, channels, input, output, out_grad, buf);
        });
    }
}

/// Tests for the pooling operations on the tape: gradient checks, gradients reaching only
/// the winners, and accumulation across uses.
#[cfg(test)]
mod tests {
    use crate::kernels::pooling::Pool2dGeom;
    use crate::{Graph, ParamStore, Rng, Tensor, gradcheck_params};

    /// `n` distinct values in [-1, 1), so max pooling has no ties to make the numerical
    /// gradient ambiguous.
    fn distinct(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::seed(seed);
        let mut v: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * i as f64 / n as f64).collect();
        rng.shuffle(&mut v);
        v
    }

    /// The gradient check passes for each pooling, with stride, padding, dilation and
    /// `ceil_mode`, in f64.
    #[test]
    fn pooling_grad_checks() {
        let geoms = [
            Pool2dGeom::new(2, (4, 6), (2, 2)).unwrap(),
            Pool2dGeom::with_options(2, (7, 5), (3, 3), (2, 2), (1, 1), (1, 1), true).unwrap(),
            Pool2dGeom::with_options(1, (9, 9), (3, 3), (2, 1), (1, 0), (2, 2), false).unwrap(),
        ];
        for (k, g) in geoms.iter().enumerate() {
            let x =
                Tensor::from_vec(distinct(2 * g.in_width(), k as u64), &[2, g.in_width()]).unwrap();
            let geom = *g;
            let mut store = ParamStore::<f64>::new_generic();
            gradcheck_params(&mut store, std::slice::from_ref(&x), |tape, v| {
                let y = tape.max_pool2d(v[0], geom);
                let t = tape.tanh(y);
                tape.sum(t)
            })
            .assert_passed();
            if geom.dilation() == (1, 1) {
                for (include, over) in [(true, None), (false, None), (true, Some(5))] {
                    gradcheck_params(&mut store, std::slice::from_ref(&x), |tape, v| {
                        let y = tape.avg_pool2d(v[0], geom, include, over);
                        let t = tape.tanh(y);
                        tape.sum(t)
                    })
                    .assert_passed();
                }
            }
        }
        let x = Tensor::from_vec(distinct(2 * 3 * 7 * 5, 9), &[2, 3 * 7 * 5]).unwrap();
        let mut store = ParamStore::<f64>::new_generic();
        gradcheck_params(&mut store, &[x], |tape, v| {
            let y = tape.adaptive_avg_pool2d(v[0], 3, (7, 5), (3, 2));
            let t = tape.tanh(y);
            tape.sum(t)
        })
        .assert_passed();
    }

    /// Max pooling's gradient reaches the winners alone, and two uses of the same input add
    /// up: pooled twice and summed, each winner's gradient is 2.
    #[test]
    fn max_pool_gradient_goes_to_winners_and_accumulates() {
        let mut store = ParamStore::new();
        let x = store.add(
            Tensor::from_vec(vec![1.0f32, 5.0, 2.0, 0.0, 3.0, 4.0, 8.0, 7.0], &[1, 8]).unwrap(),
        );
        let geom = Pool2dGeom::new(1, (2, 4), (2, 2)).unwrap();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(x);
        let a = tape.max_pool2d(v, geom);
        let b = tape.max_pool2d(v, geom);
        let s = tape.add(a, b);
        let loss = tape.sum(s);
        tape.backward(loss, &mut store.grads);
        drop(tape);
        assert_eq!(
            store.grad(x).data(),
            &[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]
        );
    }

    /// A NaN on the tape wins its max-pooling window, the last of two NaNs takes the
    /// gradient, and average pooling spreads the NaN over its window's output.
    #[test]
    fn nan_through_the_tape() {
        let mut store = ParamStore::new();
        let x =
            store.add(Tensor::from_vec(vec![1.0f32, f32::NAN, 3.0, f32::NAN], &[1, 4]).unwrap());
        let geom = Pool2dGeom::new(1, (1, 4), (1, 4)).unwrap();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(x);
        let m = tape.max_pool2d(v, geom);
        assert!(tape.value(m).data()[0].is_nan());
        let a = tape.avg_pool2d(v, geom, true, None);
        assert!(tape.value(a).data()[0].is_nan());
        let loss = tape.sum(m);
        tape.backward(loss, &mut store.grads);
        drop(tape);
        assert_eq!(store.grad(x).data(), &[0.0, 0.0, 0.0, 1.0]);
    }
}
