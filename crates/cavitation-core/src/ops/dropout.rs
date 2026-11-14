//! Dropout: `dropout`, which draws its mask on a training tape, and `dropout_with_mask`,
//! which takes the mask.

use crate::graph::Var;
use crate::ops::Op;
use crate::{Real, Rng, Tape};

/// The operation's name, used in panic messages.
const NAME: &str = "dropout";

/// Panics, naming the operation and `p`, unless `p` is in `[0, 1)`.
fn check_p(p: f64) {
    assert!(
        (0.0..1.0).contains(&p),
        "{NAME}: the drop probability must be in [0, 1), got {p}"
    );
}

impl<T: Real> Tape<'_, T> {
    /// Records inverted dropout of `x` with the mask `keep`: each value whose `keep` entry
    /// is `true` is multiplied by `1 / (1 - p)`, each other value by 0, as PyTorch's dropout
    /// does with the same mask. The gradient goes back through the same multipliers.
    ///
    /// This is the form a test uses to reproduce a mask drawn elsewhere (PyTorch's, for
    /// example); a layer draws its mask with [`dropout`](Self::dropout).
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in `[0, 1)` or `keep` does not hold one entry per value of `x`,
    /// naming the operation.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// let x = tape.input(Tensor::full(&[4], 3.0f32).view());
    /// let y = tape.dropout_with_mask(x, &[true, false, true, true], 0.25);
    /// assert_eq!(tape.value(y).data(), &[4.0, 0.0, 4.0, 4.0]);
    /// ```
    pub fn dropout_with_mask(&mut self, x: Var, keep: &[bool], p: f64) -> Var {
        check_p(p);
        let len = self.shape(x).len();
        assert_eq!(
            keep.len(),
            len,
            "{NAME}: the mask has {} entries for an input of shape {:?}",
            keep.len(),
            self.shape(x).dims()
        );
        self.dropout_by(x, p, |i| keep[i])
    }

    /// Records inverted dropout of `x` with drop probability `p`. On a training tape each
    /// value is kept with probability `1 - p` and multiplied by `1 / (1 - p)`, the others set
    /// to 0, with the draws seeded by [`Tape::stream_seed`]`(stream)`: a layer passes a
    /// stream id of its own, so the same seed, step and stream give the same mask on every
    /// machine and for any thread count. On an evaluation tape, and when `p` is 0, it
    /// returns `x` itself and records nothing.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in `[0, 1)`, naming the operation.
    ///
    /// ```
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.training_tape(&store.values, 7, 1);
    /// let x = tape.input(Tensor::full(&[1000], 1.0f32).view());
    /// let y = tape.dropout(x, 0.5, 0);
    /// let kept = tape.value(y).data().iter().filter(|&&v| v == 2.0).count();
    /// let dropped = tape.value(y).data().iter().filter(|&&v| v == 0.0).count();
    /// assert_eq!(kept + dropped, 1000);
    /// assert!((400..600).contains(&kept));
    /// drop(tape);
    /// let mut tape = graph.tape(&store.values);
    /// let x = tape.input(Tensor::full(&[3], 1.0f32).view());
    /// assert_eq!(tape.dropout(x, 0.5, 0), x);
    /// ```
    pub fn dropout(&mut self, x: Var, p: f64, stream: u64) -> Var {
        check_p(p);
        let Some(seed) = self.stream_seed(stream) else {
            return x;
        };
        if p == 0.0 {
            return x;
        }
        let mut rng = Rng::seed(seed);
        let keep = 1.0 - p;
        self.dropout_by(x, p, |_| rng.uniform_f64() < keep)
    }

    /// Records `x` times the multipliers `kept(i)` chooses: `1 / (1 - p)` for a kept value,
    /// 0 for a dropped one. The multipliers are an input node of their own, so the product's
    /// backward pass sends the gradient through them to `x` alone.
    fn dropout_by(&mut self, x: Var, p: f64, mut kept: impl FnMut(usize) -> bool) -> Var {
        let shape = self.shape(x);
        let scale = T::from_f64(1.0 / (1.0 - p));
        let mut buf = self.take_buffer(shape.len());
        for (i, m) in buf.iter_mut().enumerate() {
            *m = if kept(i) { scale } else { T::ZERO };
        }
        let multipliers = self.push(Op::Input, shape, buf);
        self.mul(x, multipliers)
    }
}

/// Tests for dropout: by hand, evaluation, probabilities, the share kept, streams and the
/// gradient.
#[cfg(test)]
mod tests {
    use crate::{Graph, ParamStore, Tensor, gradcheck_params};

    /// Kept values are scaled by 1/(1 − p), dropped ones are 0, and the gradient is the
    /// same multipliers.
    #[test]
    fn dropout_by_hand() {
        let mut store = ParamStore::<f64>::new_generic();
        let w = store.add(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap());
        let mut graph = Graph::new_generic();
        let mut tape = graph.tape(&store.values);
        let v = tape.param(w);
        let y = tape.dropout_with_mask(v, &[true, false, true, true], 0.25);
        let expect = [4.0 / 3.0, 0.0, 4.0, 16.0 / 3.0];
        for (got, want) in tape.value(y).data().iter().zip(expect) {
            assert!((got - want).abs() < 1e-15, "{got} against {want}");
        }
        let loss = tape.sum(y);
        tape.backward(loss, &mut store.grads);
        drop(tape);
        let g = store.grad(w).data().to_vec();
        assert_eq!(g, [4.0 / 3.0, 0.0, 4.0 / 3.0, 4.0 / 3.0]);
    }

    /// On an evaluation tape, and at p = 0 on a training tape, dropout returns its input and
    /// records nothing.
    #[test]
    fn dropout_in_evaluation_and_at_p_zero_is_the_identity() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let x = Tensor::full(&[5], 2.0f32);
        let mut tape = graph.tape(&store.values);
        let v = tape.input(x.view());
        let nodes = tape.node_count();
        assert_eq!(tape.dropout(v, 0.9, 0), v);
        assert_eq!(tape.node_count(), nodes);
        drop(tape);
        let mut tape = graph.training_tape(&store.values, 0, 1);
        let v = tape.input(x.view());
        assert_eq!(tape.dropout(v, 0.0, 0), v);
        assert_eq!(tape.node_count(), nodes);
    }

    /// Probabilities outside [0, 1), NaN included, panic naming the operation.
    #[test]
    fn dropout_rejects_p_out_of_range() {
        for p in [-0.1, 1.0, 1.5, f64::NAN] {
            let r = std::panic::catch_unwind(|| {
                let store = ParamStore::new();
                let mut graph = Graph::new();
                let mut tape = graph.training_tape(&store.values, 0, 1);
                let v = tape.input(Tensor::full(&[2], 1.0f32).view());
                tape.dropout(v, p, 0);
            });
            let message = r.unwrap_err();
            let message = message.downcast_ref::<String>().unwrap();
            assert!(
                message.starts_with("dropout: the drop probability"),
                "{message}"
            );
        }
    }

    /// A mask of the wrong length panics, naming the input's shape.
    #[test]
    #[should_panic(expected = "dropout: the mask has 3 entries for an input of shape [2, 2]")]
    fn dropout_with_mask_checks_the_length() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let v = tape.input(Tensor::full(&[2, 2], 1.0f32).view());
        tape.dropout_with_mask(v, &[true; 3], 0.5);
    }

    /// Of a million values at p = 0.3, the share kept is 0.7 within 0.002 (about four
    /// standard deviations), and every value is either 0 or exactly 1/0.7 in f32.
    #[test]
    fn dropout_keeps_the_right_share() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.training_tape(&store.values, 3, 9);
        let v = tape.input(Tensor::full(&[1_000_000], 1.0f32).view());
        let y = tape.dropout(v, 0.3, 4);
        let scale = (1.0 / 0.7f64) as f32;
        let data = tape.value(y).data();
        assert!(data.iter().all(|&d| d == 0.0 || d == scale));
        let kept = data.iter().filter(|&&d| d != 0.0).count() as f64 / 1e6;
        assert!((kept - 0.7).abs() < 0.002, "kept {kept}");
    }

    /// The same seed, step and stream give the same mask; another stream or step gives
    /// another.
    #[test]
    fn dropout_masks_follow_seed_step_and_stream() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let x = Tensor::full(&[256], 1.0f32);
        let mut mask = |seed, step, stream| {
            let mut tape = graph.training_tape(&store.values, seed, step);
            let v = tape.input(x.view());
            let y = tape.dropout(v, 0.5, stream);
            tape.value(y).data().to_vec()
        };
        let base = mask(1, 1, 0);
        assert_eq!(base, mask(1, 1, 0));
        assert_ne!(base, mask(1, 1, 1));
        assert_ne!(base, mask(1, 2, 0));
        assert_ne!(base, mask(2, 1, 0));
    }

    /// The gradient check passes through a fixed mask in f64.
    #[test]
    fn dropout_grad_check() {
        let mut store = ParamStore::<f64>::new_generic();
        let x = Tensor::from_vec(vec![0.3, -1.2, 0.7, 2.0, -0.4, 1.1], &[2, 3]).unwrap();
        let keep = [true, true, false, true, false, true];
        gradcheck_params(&mut store, &[x], |tape, v| {
            let y = tape.dropout_with_mask(v[0], &keep, 0.4);
            let s = tape.tanh(y);
            tape.sum(s)
        })
        .assert_passed();
    }
}
