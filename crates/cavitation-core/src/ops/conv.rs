//! 2-D convolution on the tape: `conv2d`, with its backward pass.
//!
//! The batch is cut into groups of images ([`groups`]) run in parallel; each group copies
//! its images' windows into column matrices and multiplies them by the weight with the GEMM
//! of the current kernel level (`Real::gemm`), by [`conv2d_forward`] and
//! [`conv2d_backward`]. Nothing is kept between the forward and the backward pass, which
//! builds the columns again. A 1 × 1 kernel with stride 1 and no padding multiplies each
//! image itself.

use crate::graph::{BackwardCtx, Var};
use crate::kernels::conv::{Conv2dGeom, conv2d_backward, conv2d_forward, conv2d_scratch, groups};
use crate::kernels::{self, MatMut, MatRef};
use crate::ops::Op;
use crate::{Real, Shape, Tape};

/// The operation's name, used in panic messages.
const NAME: &str = "conv2d";

/// The batch size of a `conv2d` input of shape `sx`, after checking the input, weight and
/// bias shapes against `geom`.
///
/// # Panics
///
/// Panics, naming the operation, the geometry and the shapes, if the input is not
/// `[batch, in_width]`, the weight not `[out, in, kh, kw]` or the bias not `[out]`.
fn check_shapes(geom: &Conv2dGeom, sx: &Shape, sw: &Shape, sb: Option<&Shape>) -> usize {
    let (kh, kw) = geom.kernel();
    let weight = [geom.out_channels(), geom.in_channels(), kh, kw];
    let rows_ok = sx.rank() == 2 && sx.dims()[1] == geom.in_width();
    let weight_ok = sw.dims() == weight;
    let bias_ok = sb.is_none_or(|s| s.dims() == [geom.out_channels()]);
    assert!(
        rows_ok && weight_ok && bias_ok,
        "{NAME}: for {geom:?} the input must be [batch, {}], the weight {weight:?} and the \
         bias [{}], but they are {:?}, {:?} and {:?}",
        geom.in_width(),
        geom.out_channels(),
        sx.dims(),
        sw.dims(),
        sb.map(Shape::dims),
    );
    sx.dims()[0]
}

impl<T: Real> Tape<'_, T> {
    /// Records the 2-D convolution of the images in `x` by the kernels in `weight`, plus
    /// `bias`, as PyTorch's `nn.Conv2d` computes it (`groups = 1`, zero padding).
    ///
    /// `x` is `[batch, in_channels · height · width]`: each row one image in NCHW order
    /// (channel after channel, each row by row). `weight` is `[out, in, kh, kw]` and `bias`
    /// `[out]`. The result is `[batch, out_channels · out_height · out_width]`, each row one
    /// output image in the same order.
    ///
    /// The batch is cut into groups of images that run in parallel on the thread pool; each
    /// group copies its images' windows into column matrices and multiplies them by the
    /// weight with the GEMM of the current kernel level. The gradients are GEMMs too, group
    /// by group, and the weight gradient adds one partial per group in group order (the
    /// groups depend on the batch alone), so every result is the same for any thread
    /// count. An input that takes no
    /// gradient costs no input-gradient work.
    ///
    /// # Panics
    ///
    /// Panics if the shapes do not match `geom`, naming the operation, the geometry and the
    /// shapes.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// use cavitation_core::{Graph, ParamStore, Tensor};
    /// let store = ParamStore::new();
    /// let mut graph = Graph::new();
    /// let mut tape = graph.tape(&store.values);
    /// // One 3 x 3 image, one 2 x 2 kernel of ones and a bias of 0.5.
    /// let geom = Conv2dGeom::new(1, 1, (3, 3), (2, 2)).unwrap();
    /// let x: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    /// let x = tape.input(Tensor::from_vec(x, &[1, 9]).unwrap().view());
    /// let w = tape.input(Tensor::full(&[1, 1, 2, 2], 1.0f32).view());
    /// let b = tape.input(Tensor::full(&[1], 0.5f32).view());
    /// let y = tape.conv2d(x, w, Some(b), geom);
    /// assert_eq!(tape.value(y).data(), &[12.5, 16.5, 24.5, 28.5]);
    /// ```
    pub fn conv2d(&mut self, x: Var, weight: Var, bias: Option<Var>, geom: Conv2dGeom) -> Var {
        let sb = bias.map(|b| self.shape(b));
        let batch = check_shapes(&geom, &self.shape(x), &self.shape(weight), sb.as_ref());
        let threads = kernels::threads();
        let mut out = self.take_buffer(batch * geom.out_width());
        let mut scratch = self.take_buffer(conv2d_scratch(&geom, batch, threads));
        let bv = bias.map(|b| self.data(b));
        conv2d_forward(
            &geom,
            batch,
            self.data(x),
            self.data(weight),
            bv,
            &mut out,
            &mut scratch,
            threads,
            &gemm::<T>,
        );
        self.give_buffer(scratch);
        let start = self.push_labels(&geom.to_words());
        let so = Shape::new(&[batch, geom.out_width()]);
        self.push(Op::Conv2d(x, weight, bias, start), so, out)
    }
}

/// `c = a · b + beta · c` with the current level's GEMM.
fn gemm<T: Real>(a: MatRef<'_, T>, b: MatRef<'_, T>, beta: T, c: MatMut<'_, T>) {
    T::gemm(T::ONE, a, b, beta, c);
}

/// Backward of `conv2d`: `db` sums the output gradient; the input gradient and one partial
/// weight gradient per group come from [`conv2d_backward`], and the partials are added up in
/// group order, so every gradient is the same for any thread count.
pub(super) fn backward<T: Real>(
    x: Var,
    weight: Var,
    bias: Option<Var>,
    start: usize,
    out_grad: &[T],
    ctx: &mut BackwardCtx<'_, T>,
) {
    let geom = Conv2dGeom::from_words(&ctx.labels()[start..start + 12]);
    let batch = ctx.shape(x).dims()[0];
    let (oh, ow) = geom.out_size();
    let (windows, kl, cout) = (oh * ow, geom.kernel_len(), geom.out_channels());
    let out_w = geom.out_width();
    let (xv, wv) = (ctx.value(x), ctx.value(weight));
    if let Some(b) = bias
        && let Some(slot) = ctx.slot(b)
    {
        slot.update_with(|buf, beta| {
            for (o, d) in buf.iter_mut().enumerate() {
                let mut acc = T::ZERO;
                for n in 0..batch {
                    for &g in &out_grad[n * out_w + o * windows..][..windows] {
                        acc += g;
                    }
                }
                *d = if beta == T::ZERO { acc } else { *d + acc };
            }
        });
    }
    let threads = kernels::threads();
    let mut scratch = ctx.scratch(conv2d_scratch(&geom, batch, threads));
    if ctx.takes_grad(weight) {
        let count = groups(batch);
        let mut parts = ctx.scratch(count * cout * kl);
        if let Some(slot) = ctx.slot(weight) {
            conv2d_backward(
                &geom,
                batch,
                xv,
                wv,
                out_grad,
                None,
                Some(&mut parts),
                &mut scratch,
                threads,
                &gemm::<T>,
            );
            slot.update_with(|buf, beta| {
                for (i, d) in buf.iter_mut().enumerate() {
                    let mut acc = parts[i];
                    for g in 1..count {
                        acc += parts[g * cout * kl + i];
                    }
                    *d = if beta == T::ZERO { acc } else { *d + acc };
                }
            });
        }
        ctx.release(parts);
    }
    if ctx.takes_grad(x)
        && let Some(slot) = ctx.slot(x)
    {
        slot.update_with(|buf, beta| {
            let add = beta != T::ZERO;
            conv2d_backward(
                &geom,
                batch,
                xv,
                wv,
                out_grad,
                Some((buf, add)),
                None,
                &mut scratch,
                threads,
                &gemm::<T>,
            );
        });
    }
    ctx.release(scratch);
}

/// Tests for the convolution operation: against the direct loops, the 1 × 1 path, the
/// gradient check, skipped input gradients and reused buffers.
#[cfg(test)]
mod tests {
    use crate::kernels::conv::{
        Conv2dGeom, conv2d_backward_reference, conv2d_reference, images_per_chunk,
    };
    use crate::real::GEMM_CALLS;
    use crate::{Graph, ParamStore, Rng, Tensor, gradcheck_params};

    /// Geometries covering stride, padding, dilation, non-square images and kernels, 1 × 1
    /// kernels with and without stride, and a kernel the size of the image.
    fn geometries() -> Vec<Conv2dGeom> {
        let cases = [
            (2, 3, (5, 5), (3, 3), (1, 1), (0, 0), (1, 1)),
            (1, 2, (6, 7), (3, 2), (2, 1), (1, 0), (1, 1)),
            (3, 2, (7, 7), (3, 3), (1, 1), (2, 2), (2, 2)),
            (2, 2, (4, 6), (1, 1), (1, 1), (0, 0), (1, 1)),
            (2, 2, (5, 6), (1, 1), (2, 2), (0, 0), (1, 1)),
            (2, 1, (5, 4), (2, 3), (2, 2), (1, 1), (1, 1)),
            (1, 1, (3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
            (2, 3, (3, 3), (3, 3), (1, 1), (0, 0), (1, 1)),
            // Padding past the kernel: the first and last output rows and columns are pure
            // bias.
            (1, 2, (3, 3), (2, 2), (1, 1), (3, 2), (1, 1)),
            // Stride and dilation above 1 on the same axis.
            (2, 2, (9, 8), (3, 2), (2, 2), (1, 1), (2, 3)),
            // 288 · 1,024 column values per image: 7 images per chunk, so the 9 images of a
            // `conv2d_matches_direct_loops` batch span two chunks, the second short.
            (32, 2, (32, 32), (3, 3), (1, 1), (1, 1), (1, 1)),
        ];
        cases
            .into_iter()
            .map(|(cin, cout, input, kernel, stride, padding, dilation)| {
                Conv2dGeom::with_options(cin, cout, input, kernel, stride, padding, dilation)
                    .unwrap()
            })
            .collect()
    }

    /// `n` values uniform in [-1, 1) from `Rng::seed(seed)`.
    fn values(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::seed(seed);
        (0..n).map(|_| rng.range(-1.0, 1.0)).collect()
    }

    /// In f64 the tape's output and gradients match the direct loops within rounding, on
    /// every geometry, with and without a bias.
    #[test]
    fn conv2d_matches_direct_loops() {
        for (k, g) in geometries().iter().enumerate() {
            for with_bias in [true, false] {
                let batch = if images_per_chunk(g, 9) < 9 { 9 } else { 3 };
                let x = values(batch * g.in_width(), k as u64);
                let wv = values(g.weight_len(), 10 + k as u64);
                let bv = values(g.out_channels(), 20 + k as u64);
                let up = values(batch * g.out_width(), 30 + k as u64);
                let (kh, kw) = g.kernel();
                let mut store = ParamStore::<f64>::new_generic();
                let w = store.add(
                    Tensor::from_vec(wv.clone(), &[g.out_channels(), g.in_channels(), kh, kw])
                        .unwrap(),
                );
                let b = store.add(Tensor::from_vec(bv.clone(), &[g.out_channels()]).unwrap());
                let xi = store.add(Tensor::from_vec(x.clone(), &[batch, g.in_width()]).unwrap());
                let mut graph = Graph::new_generic();
                let mut tape = graph.tape(&store.values);
                let (xv, wvar, bvar) = (tape.param(xi), tape.param(w), tape.param(b));
                let y = tape.conv2d(xv, wvar, with_bias.then_some(bvar), *g);
                let got = tape.value(y).data().to_vec();
                let u = tape.input(
                    Tensor::from_vec(up.clone(), &[batch, g.out_width()])
                        .unwrap()
                        .view(),
                );
                let prod = tape.mul(y, u);
                let loss = tape.sum(prod);
                tape.backward(loss, &mut store.grads);
                drop(tape);
                let mut want = vec![0.0; got.len()];
                let bias = with_bias.then_some(bv.as_slice());
                conv2d_reference(g, batch, &x, &wv, bias, &mut want);
                let (mut dx, mut dw, mut db) =
                    (vec![0.0; x.len()], vec![0.0; wv.len()], vec![0.0; bv.len()]);
                conv2d_backward_reference(g, batch, &x, &wv, &up, &mut dx, &mut dw, &mut db);
                let close = |a: &[f64], b: &[f64], what: &str| {
                    for (i, (p, q)) in a.iter().zip(b).enumerate() {
                        assert!(
                            (p - q).abs() < 1e-12,
                            "geometry {k} {what}[{i}]: {p} against {q}"
                        );
                    }
                };
                close(&got, &want, "output");
                close(store.grad(xi).data(), &dx, "input gradient");
                close(store.grad(w).data(), &dw, "weight gradient");
                if with_bias {
                    close(store.grad(b).data(), &db, "bias gradient");
                }
            }
        }
    }

    /// The gradient check passes on three geometries, for the input, the weight and the
    /// bias.
    #[test]
    fn conv2d_grad_check() {
        for g in geometries().iter().step_by(3) {
            let batch = 2;
            let (kh, kw) = g.kernel();
            let mut store = ParamStore::<f64>::new_generic();
            let w = store.add(
                Tensor::from_vec(
                    values(g.weight_len(), 1),
                    &[g.out_channels(), g.in_channels(), kh, kw],
                )
                .unwrap(),
            );
            let b = store
                .add(Tensor::from_vec(values(g.out_channels(), 2), &[g.out_channels()]).unwrap());
            let x =
                Tensor::from_vec(values(batch * g.in_width(), 3), &[batch, g.in_width()]).unwrap();
            let geom = *g;
            gradcheck_params(&mut store, &[x], |tape, v| {
                let (wv, bv) = (tape.param(w), tape.param(b));
                let y = tape.conv2d(v[0], wv, Some(bv), geom);
                let t = tape.tanh(y);
                tape.sum(t)
            })
            .assert_passed();
        }
    }

    /// An input that takes no gradient costs no input-gradient GEMM: the backward pass of a
    /// 1 × 1 convolution of a tape input runs 1 GEMM (the weight gradient's), against 2 when
    /// the input is a parameter. One image makes one group, which runs on the calling
    /// thread, where the GEMM count is kept.
    #[test]
    fn conv2d_skips_the_input_gradient_of_an_input() {
        let g = Conv2dGeom::new(2, 3, (4, 4), (1, 1)).unwrap();
        let calls = |input_is_param: bool| {
            let mut store = ParamStore::new();
            let w = store.add(Tensor::full(&[3, 2, 1, 1], 0.5f32));
            let x = Tensor::full(&[1, g.in_width()], 1.0f32);
            let xp = store.add(x.clone());
            let mut graph = Graph::new();
            let mut tape = graph.tape(&store.values);
            let xv = if input_is_param {
                tape.param(xp)
            } else {
                tape.input(x.view())
            };
            let wv = tape.param(w);
            let y = tape.conv2d(xv, wv, None, g);
            let loss = tape.sum(y);
            GEMM_CALLS.with(|n| n.set(0));
            tape.backward(loss, &mut store.grads);
            GEMM_CALLS.with(|n| n.get())
        };
        assert_eq!(calls(false), 1);
        assert_eq!(calls(true), 2);
    }

    /// Pooled buffers holding old values do not leak into the results: three steps on one
    /// graph give the same output and gradients, in f32.
    #[test]
    fn conv2d_reused_buffers_give_same_results() {
        let g = Conv2dGeom::new(2, 3, (6, 5), (3, 3))
            .unwrap()
            .with_padding((1, 1))
            .unwrap();
        let mut rng = Rng::seed(8);
        let mut store = ParamStore::new();
        let w = store.add(
            Tensor::from_vec(
                (0..g.weight_len())
                    .map(|_| rng.range(-1.0, 1.0) as f32)
                    .collect(),
                &[3, 2, 3, 3],
            )
            .unwrap(),
        );
        let b = store.add(Tensor::full(&[3], 0.1f32));
        let x = Tensor::from_vec(
            (0..4 * g.in_width())
                .map(|_| rng.range(-1.0, 1.0) as f32)
                .collect(),
            &[4, g.in_width()],
        )
        .unwrap();
        let mut graph = Graph::new();
        let mut seen = Vec::new();
        for _ in 0..3 {
            let mut tape = graph.tape(&store.values);
            let xv = tape.input(x.view());
            let (wv, bv) = (tape.param(w), tape.param(b));
            let y = tape.conv2d(xv, wv, Some(bv), g);
            let out = tape.value(y).data().to_vec();
            let s = tape.tanh(y);
            let loss = tape.sum(s);
            tape.backward(loss, &mut store.grads);
            drop(tape);
            seen.push((
                out,
                store.grad(w).data().to_vec(),
                store.grad(b).data().to_vec(),
            ));
            store.zero_grad();
        }
        assert_eq!(seen[0], seen[1]);
        assert_eq!(seen[1], seen[2]);
    }

    /// In f32 at the current kernel level, the output and all three gradients stay within
    /// the GEMM bound of the f64 direct loops: `2 · (k + 2) · 2⁻²³ · Σ|terms|` with `k` the
    /// length of each sum (`in · kh · kw` for the output and the input gradient's GEMM,
    /// the image's windows for the weight gradient), plus a share for the bias and the
    /// per-image sums, on shapes large enough that the fast kernels run.
    #[test]
    fn conv2d_f32_within_the_gemm_bound() {
        let cases = [
            Conv2dGeom::with_options(8, 16, (12, 12), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap(),
            Conv2dGeom::with_options(5, 7, (11, 9), (3, 2), (2, 1), (1, 0), (1, 2)).unwrap(),
            Conv2dGeom::with_options(16, 24, (6, 6), (1, 1), (1, 1), (0, 0), (1, 1)).unwrap(),
        ];
        for (k, g) in cases.iter().enumerate() {
            let batch = 3;
            let x = values(batch * g.in_width(), 40 + k as u64);
            let wv = values(g.weight_len(), 50 + k as u64);
            let bv = values(g.out_channels(), 60 + k as u64);
            let up = values(batch * g.out_width(), 70 + k as u64);
            let f32s = |v: &[f64]| v.iter().map(|&a| a as f32).collect::<Vec<f32>>();
            // The f64 reference runs on the f32-rounded values, so both sides solve one problem.
            let round = |v: &[f64]| v.iter().map(|&a| f64::from(a as f32)).collect::<Vec<f64>>();
            let (x64, w64, b64, u64s) = (round(&x), round(&wv), round(&bv), round(&up));
            let (kh, kw) = g.kernel();
            let mut store = ParamStore::new();
            let xi = store.add(Tensor::from_vec(f32s(&x), &[batch, g.in_width()]).unwrap());
            let w = store.add(
                Tensor::from_vec(f32s(&wv), &[g.out_channels(), g.in_channels(), kh, kw]).unwrap(),
            );
            let b = store.add(Tensor::from_vec(f32s(&bv), &[g.out_channels()]).unwrap());
            let mut graph = Graph::new();
            let mut tape = graph.tape(&store.values);
            let (xv, wvar, bvar) = (tape.param(xi), tape.param(w), tape.param(b));
            let y = tape.conv2d(xv, wvar, Some(bvar), *g);
            let got: Vec<f64> = tape.value(y).data().iter().map(|&v| f64::from(v)).collect();
            let u = tape.input(
                Tensor::from_vec(f32s(&up), &[batch, g.out_width()])
                    .unwrap()
                    .view(),
            );
            let prod = tape.mul(y, u);
            let loss = tape.sum(prod);
            tape.backward(loss, &mut store.grads);
            drop(tape);
            let abs = |v: &[f64]| v.iter().map(|a| a.abs()).collect::<Vec<f64>>();
            let mut want = vec![0.0; got.len()];
            conv2d_reference(g, batch, &x64, &w64, Some(&b64), &mut want);
            let mut mag = vec![0.0; got.len()];
            conv2d_reference(g, batch, &abs(&x64), &abs(&w64), Some(&abs(&b64)), &mut mag);
            let (mut dx, mut dw, mut db) =
                (vec![0.0; x.len()], vec![0.0; wv.len()], vec![0.0; bv.len()]);
            conv2d_backward_reference(g, batch, &x64, &w64, &u64s, &mut dx, &mut dw, &mut db);
            let (mut mx, mut mw, mut mb) =
                (vec![0.0; x.len()], vec![0.0; wv.len()], vec![0.0; bv.len()]);
            conv2d_backward_reference(
                g,
                batch,
                &abs(&x64),
                &abs(&w64),
                &abs(&u64s),
                &mut mx,
                &mut mw,
                &mut mb,
            );
            let (oh, ow) = g.out_size();
            let eps = f64::from(f32::EPSILON) / 2.0;
            let check = |got: &[f64], want: &[f64], mag: &[f64], terms: usize, what: &str| {
                for (i, ((p, q), m)) in got.iter().zip(want).zip(mag).enumerate() {
                    let bound = 2.0 * (terms + 2) as f64 * eps * m + 1e-30;
                    assert!(
                        (p - q).abs() <= bound,
                        "geometry {k} {what}[{i}]: {p} against {q}, bound {bound}"
                    );
                }
            };
            check(&got, &want, &mag, g.kernel_len() + 1, "output");
            let grad = |id| -> Vec<f64> {
                store
                    .grad(id)
                    .data()
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect()
            };
            check(
                &grad(xi),
                &dx,
                &mx,
                g.out_channels() * kh * kw,
                "input gradient",
            );
            check(&grad(w), &dw, &mw, batch * oh * ow, "weight gradient");
            check(&grad(b), &db, &mb, batch * oh * ow, "bias gradient");
        }
    }

    /// A convolution's gradients add to what other uses already put there: with the input
    /// also summed and the weight used by two convolutions, the gradients are the sums of
    /// each use's, on a chunked geometry and a 1 × 1 one.
    #[test]
    fn conv2d_gradients_accumulate() {
        for g in [
            Conv2dGeom::with_options(2, 3, (5, 4), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap(),
            Conv2dGeom::new(2, 3, (5, 4), (1, 1)).unwrap(),
        ] {
            let batch = 2;
            let (kh, kw) = g.kernel();
            let mut store = ParamStore::<f64>::new_generic();
            let xi = store.add(
                Tensor::from_vec(values(batch * g.in_width(), 1), &[batch, g.in_width()]).unwrap(),
            );
            let w =
                store.add(Tensor::from_vec(values(g.weight_len(), 2), &[3, 2, kh, kw]).unwrap());
            let b = store.add(Tensor::from_vec(values(3, 3), &[3]).unwrap());
            let grads = |store: &mut ParamStore<f64>, twice: bool| {
                let mut graph = Graph::new_generic();
                let mut tape = graph.tape(&store.values);
                let (xv, wv, bv) = (tape.param(xi), tape.param(w), tape.param(b));
                let y = tape.conv2d(xv, wv, Some(bv), g);
                let mut loss = tape.sum(y);
                if twice {
                    let y2 = tape.conv2d(xv, wv, Some(bv), g);
                    let s2 = tape.sum(y2);
                    let sx = tape.sum(xv);
                    let t = tape.add(s2, sx);
                    loss = tape.add(loss, t);
                }
                tape.backward(loss, &mut store.grads);
                drop(tape);
                let out = [xi, w, b].map(|id| store.grad(id).data().to_vec());
                store.zero_grad();
                out
            };
            let once = grads(&mut store, false);
            let both = grads(&mut store, true);
            for (k, (one, two)) in once.iter().zip(&both).enumerate() {
                for (i, (a, c)) in one.iter().zip(two).enumerate() {
                    let want = 2.0 * a + if k == 0 { 1.0 } else { 0.0 };
                    assert!(
                        (c - want).abs() < 1e-12,
                        "{g:?} gradient {k}[{i}]: {c} against {want}"
                    );
                }
            }
        }
    }

    /// An empty batch gives an empty output and zero gradients for the weight and bias,
    /// even when the pooled buffers hold old values from a step with images.
    #[test]
    fn conv2d_empty_batch_gives_zero_gradients() {
        let g = Conv2dGeom::with_options(2, 3, (4, 4), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
        let mut store = ParamStore::<f64>::new_generic();
        let w = store.add(Tensor::from_vec(values(g.weight_len(), 1), &[3, 2, 3, 3]).unwrap());
        let b = store.add(Tensor::from_vec(values(3, 2), &[3]).unwrap());
        let mut graph = Graph::new_generic();
        for batch in [4, 0] {
            let x =
                Tensor::from_vec(values(batch * g.in_width(), 3), &[batch, g.in_width()]).unwrap();
            let mut tape = graph.tape(&store.values);
            let (xv, wv, bv) = (tape.input(x.view()), tape.param(w), tape.param(b));
            let y = tape.conv2d(xv, wv, Some(bv), g);
            assert_eq!(tape.shape(y).dims(), &[batch, g.out_width()]);
            let loss = tape.sum(y);
            tape.backward(loss, &mut store.grads);
            drop(tape);
            if batch == 0 {
                assert!(store.grad(w).data().iter().all(|&v| v == 0.0));
                assert!(store.grad(b).data().iter().all(|&v| v == 0.0));
            }
            store.zero_grad();
        }
    }

    /// Shapes that do not match the geometry panic, naming the operation and the shapes.
    #[test]
    #[should_panic(expected = "conv2d: for Conv2dGeom")]
    fn conv2d_checks_shapes() {
        let g = Conv2dGeom::new(1, 1, (3, 3), (2, 2)).unwrap();
        let store = ParamStore::new();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&store.values);
        let x = tape.input(Tensor::full(&[1, 8], 1.0f32).view());
        let w = tape.input(Tensor::full(&[1, 1, 2, 2], 1.0f32).view());
        tape.conv2d(x, w, None, g);
    }
}
