//! Dropout: `dropout`, which draws its mask on a training tape, `dropout_with_mask`, which
//! takes the mask, and `dropout_mask`, the draw itself without a tape.

use crate::graph::{Var, stream_seed};
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

/// Draws the dropout mask that [`Tape::dropout`] draws on a training tape seeded `seed` at
/// step `step`, for random stream `stream` and drop probability `p`, over `len` values: it
/// calls `write(i, kept)` once for each `i` in `0..len`, in order, with `kept` true for a
/// value the tape keeps (and scales by `1 / (1 - p)`) and false for one it sets to 0.
///
/// It is the mask without a tape, for a device that applies dropout itself and has to drop
/// the same values the tape would. At `p = 0` every value is kept, the tape's identity.
///
/// # Panics
///
/// Panics if `p` is not in `[0, 1)`, naming the operation.
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Tensor, dropout_mask};
/// let store = ParamStore::new();
/// let mut graph = Graph::new();
/// let mut tape = graph.training_tape(&store.values, 7, 3);
/// let x = tape.input(Tensor::full(&[64], 1.0f32).view());
/// let y = tape.dropout(x, 0.5, 2);
/// let mut mask = vec![false; 64];
/// dropout_mask(7, 3, 2, 0.5, 64, |i, kept| mask[i] = kept);
/// let kept: Vec<bool> = tape.value(y).data().iter().map(|&v| v != 0.0).collect();
/// assert_eq!(mask, kept);
/// ```
pub fn dropout_mask(
    seed: u64,
    step: u64,
    stream: u64,
    p: f64,
    len: usize,
    mut write: impl FnMut(usize, bool),
) {
    check_p(p);
    let mut kept = draws(stream_seed(seed, step, stream), p);
    for i in 0..len {
        write(i, kept());
    }
}

/// The draws behind [`dropout_mask`] and [`Tape::dropout`], from a stream's seed: each call
/// draws one uniform `f64` and keeps the next value when the draw is below `1 - p`.
fn draws(seed: u64, p: f64) -> impl FnMut() -> bool {
    let mut rng = Rng::seed(seed);
    let keep = 1.0 - p;
    move || rng.uniform_f64() < keep
}

impl<T: Real> Tape<'_, T> {
    /// Records inverted dropout of `x` with the mask `keep`: each value whose `keep` entry
    /// is `true` is multiplied by `1 / (1 - p)` (with `1 - p` rounded to the element type
    /// first, as PyTorch divides), each other value by 0, as PyTorch's dropout does with the
    /// same mask. The gradient goes back through the same multipliers.
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
    /// returns `x` itself and records nothing. [`dropout_mask`] draws the same mask without
    /// a tape.
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
        let mut kept = draws(seed, p);
        self.dropout_by(x, p, |_| kept())
    }

    /// Records `x` times the multipliers `kept(i)` chooses: `1 / (1 - p)` for a kept value,
    /// 0 for a dropped one. The multipliers are an input node of their own, so the product's
    /// backward pass sends the gradient through them to `x` alone.
    fn dropout_by(&mut self, x: Var, p: f64, mut kept: impl FnMut(usize) -> bool) -> Var {
        let shape = self.shape(x);
        // PyTorch divides in the element type: `bernoulli_(1 - p).div_(1 - p)`, so the scale
        // is 1 over `1 - p` rounded to `T`, not `1 / (1 - p)` rounded once.
        let scale = T::ONE / T::from_f64(1.0 - p);
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

    /// `dropout_mask` draws v0.3's masks, bit for bit: a GPU run and a v0.3 checkpoint's run
    /// drop the same values as the tape. Each row is (seed, step, stream, p) and the 128
    /// keep bits that v0.3's `Tape::dropout` drew, value `i` at bit `i % 64` of word `i / 64`;
    /// the rows cover the seed's extremes, an early and a late step and a range of `p`.
    #[test]
    fn dropout_mask_draws_the_v0_3_masks() {
        const V0_3: [(u64, u64, u64, f64, [u64; 2]); 5] = [
            (0, 1, 0, 0.5, [0x167a_f6d1_878d_2565, 0x09cf_a3ed_334e_f4f9]),
            (7, 3, 2, 0.1, [0xb9fe_bdfe_fdef_ffff, 0xfff9_dffb_b7fd_ffff]),
            (
                12345,
                99,
                5,
                0.9,
                [0x0042_0012_9080_8100, 0x2030_0120_0404_0000],
            ),
            (
                u64::MAX,
                0,
                1,
                0.25,
                [0xfdfd_7f7f_ffe6_fbed, 0x3ffe_5b1d_f9f5_5fbf],
            ),
            (
                42,
                1000,
                3,
                0.025,
                [0xbfff_7fff_ffff_ffff, 0xffff_fbbf_ffff_f7fb],
            ),
        ];
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let x = Tensor::full(&[128], 1.0f32);
        for (seed, step, stream, p, want) in V0_3 {
            let mut drawn = [0u64; 2];
            let mut next = 0;
            super::dropout_mask(seed, step, stream, p, 128, |i, kept| {
                assert_eq!(i, next, "the mask is written in order");
                next += 1;
                drawn[i / 64] |= u64::from(kept) << (i % 64);
            });
            assert_eq!(next, 128);
            assert_eq!(drawn, want, "dropout_mask at {seed}, {step}, {stream}, {p}");
            let mut tape = graph.training_tape(&store.values, seed, step);
            let v = tape.input(x.view());
            let y = tape.dropout(v, p, stream);
            let mut taped = [0u64; 2];
            for (i, &d) in tape.value(y).data().iter().enumerate() {
                taped[i / 64] |= u64::from(d != 0.0) << (i % 64);
            }
            assert_eq!(
                taped, want,
                "Tape::dropout at {seed}, {step}, {stream}, {p}"
            );
        }
    }

    /// At p = 0 `dropout_mask` keeps every value, as the tape's identity does.
    #[test]
    fn dropout_mask_at_p_zero_keeps_everything() {
        let mut kept = 0;
        super::dropout_mask(3, 4, 5, 0.0, 1000, |_, k| kept += usize::from(k));
        assert_eq!(kept, 1000);
    }

    /// A mask of length 0 writes nothing.
    #[test]
    fn dropout_mask_of_length_zero_writes_nothing() {
        super::dropout_mask(3, 4, 5, 0.5, 0, |_, _| panic!("nothing to write"));
    }

    /// `dropout_mask` refuses a probability outside [0, 1), as the tape does.
    #[test]
    #[should_panic(expected = "dropout: the drop probability must be in [0, 1), got 1")]
    fn dropout_mask_rejects_p_out_of_range() {
        super::dropout_mask(0, 0, 0, 1.0, 4, |_, _| {});
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

    /// The f32 scale is `1 / f32(1 - p)`, PyTorch's, not `f32(1 / (1 - p))`: at p = 0.025
    /// the two differ by one ulp.
    #[test]
    fn dropout_scale_divides_in_the_element_type() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let v = tape.input(Tensor::full(&[1], 1.0f32).view());
        let y = tape.dropout_with_mask(v, &[true], 0.025);
        let torch = 1.0f32 / (1.0 - 0.025f64) as f32;
        assert_ne!(
            torch,
            (1.0 / (1.0 - 0.025f64)) as f32,
            "p = 0.025 tells them apart"
        );
        assert_eq!(tape.value(y).data(), &[torch]);
    }

    /// A mask gives the same values on a training tape as on an evaluation tape: only
    /// `dropout`, not `dropout_with_mask`, depends on the mode.
    #[test]
    fn dropout_with_mask_ignores_the_mode() {
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let x = Tensor::from_vec(vec![1.0f32, -2.0, 3.0], &[3]).unwrap();
        let mut run = |training: bool| {
            let mut tape = if training {
                graph.training_tape(&store.values, 1, 1)
            } else {
                graph.tape(&store.values)
            };
            let v = tape.input(x.view());
            let y = tape.dropout_with_mask(v, &[true, false, true], 0.5);
            tape.value(y).data().to_vec()
        };
        assert_eq!(run(true), run(false));
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
