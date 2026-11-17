//! The `BatchNorm2d` layer: PyTorch's `nn.BatchNorm2d` on images stored as flat rows.

use cavitation_core::{
    Error, ParamId, ParamStore, ParamValues, Real, RunningStats, Tape, Tensor, Var,
};

use crate::{InferenceLayer, Layer};

/// 2-D batch normalisation, PyTorch's `nn.BatchNorm2d`: each channel of the images is
/// normalised by its mean and variance, then scaled by `weight` (γ) and shifted by `bias`
/// (β).
///
/// While training (on a [training tape](cavitation_core::Graph::training_tape)) it uses the
/// batch's statistics and moves its running mean and variance towards them after each
/// step ([`Graph::apply_buffer_updates`](cavitation_core::Graph::apply_buffer_updates),
/// which `fit` and `Trainer` call); in evaluation and at inference it uses the running
/// statistics. Its parameters are `weight` and `bias` and its buffers `running_mean`,
/// `running_var` and `num_batches_tracked`, under PyTorch's names, so a PyTorch state dict
/// loads into it by key. The batch count is kept as an `f32` in the store, so it stops
/// growing at 2²⁴ (16,777,216) batches; with momentum `None` (the cumulative average) the
/// factor then stays at 1 / 2²⁴. It is saved as PyTorch's `I64`.
///
/// A convolution or `Linear` right before it needs no bias: subtracting each channel's
/// mean cancels a per-channel bias, so its gradient is 0 up to rounding (and the health
/// checks report it as vanishing). Build that layer without one, as with PyTorch's
/// `bias=False`, for example [`Conv2d::with_init`](crate::Conv2d::with_init) with `None`.
///
/// ```
/// use cavitation_core::{Graph, ParamStore, Tensor};
/// use cavitation_nn::{BatchNorm2d, Layer};
/// let mut params = ParamStore::new();
/// let bn = BatchNorm2d::new(&mut params, 2, (2, 2));
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&params.values);
/// // Evaluation: the running statistics start at mean 0 and variance 1.
/// let x = tape.input(Tensor::full(&[1, 8], 3.0f32).view());
/// let y = bn.forward(&mut tape, x);
/// let want = 3.0 / (1.0f32 + 1e-5).sqrt();
/// assert!(tape.value(y).data().iter().all(|&v| (v - want).abs() < 1e-6));
/// ```
#[derive(Clone, Debug)]
pub struct BatchNorm2d {
    /// Channels of each image.
    channels: usize,
    /// Values per channel of one image: height times width.
    spatial: usize,
    /// γ, `[channels]`, if the layer is affine.
    weight: Option<ParamId>,
    /// β, `[channels]`, if the layer is affine.
    bias: Option<ParamId>,
    /// The running statistics, if the layer keeps them.
    running: Option<RunningStats>,
    /// Added to the variance before its square root.
    eps: f64,
}

impl BatchNorm2d {
    /// A layer for `channels` channels of `size` (height, width) images with PyTorch's
    /// defaults: `eps` 1e-5, momentum 0.1, affine, running statistics kept. γ starts at 1,
    /// β at 0, the running mean at 0, the running variance at 1 and the batch count at 0.
    ///
    /// # Panics
    ///
    /// Panics if `channels` or a size is 0.
    ///
    /// ```
    /// use cavitation_core::ParamStore;
    /// use cavitation_nn::BatchNorm2d;
    /// let mut params = ParamStore::new();
    /// let bn = BatchNorm2d::new(&mut params, 16, (8, 8));
    /// assert_eq!(params.len(), 5); // weight, bias and the three buffers
    /// assert_eq!(bn.eps(), 1e-5);
    /// ```
    pub fn new<T: Real>(
        params: &mut ParamStore<T>,
        channels: usize,
        size: (usize, usize),
    ) -> BatchNorm2d {
        BatchNorm2d::with_options(params, channels, size, 1e-5, Some(0.1), true, true)
            .expect("PyTorch's defaults are valid")
    }

    /// A layer with every setting given: `eps`, `momentum` (`None` for the cumulative
    /// average), whether it is `affine` (has γ and β) and whether it keeps running
    /// statistics (`track_running_stats`; without them it uses the batch's statistics in
    /// evaluation too).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if `channels` or a size is 0, `eps` is negative or not finite,
    /// or `momentum` is outside `[0, 1]`.
    ///
    /// ```
    /// use cavitation_core::ParamStore;
    /// use cavitation_nn::BatchNorm2d;
    /// let mut params = ParamStore::new();
    /// let bn = BatchNorm2d::with_options(&mut params, 4, (2, 2), 1e-3, None, false, true).unwrap();
    /// assert_eq!(params.len(), 3); // the buffers only
    /// assert!(BatchNorm2d::with_options(&mut params, 4, (2, 2), -1.0, None, true, true).is_err());
    /// # let _ = bn;
    /// ```
    pub fn with_options<T: Real>(
        params: &mut ParamStore<T>,
        channels: usize,
        size: (usize, usize),
        eps: f64,
        momentum: Option<f64>,
        affine: bool,
        track_running_stats: bool,
    ) -> Result<BatchNorm2d, Error> {
        let fail = |what: String| Err(Error::InvalidData(format!("batch_norm2d: {what}")));
        if channels == 0 || size.0 == 0 || size.1 == 0 {
            return fail(format!(
                "the channels and sizes must be at least 1, got {channels} and {size:?}"
            ));
        }
        if !(eps.is_finite() && eps >= 0.0) {
            return fail(format!("eps must be finite and not negative, got {eps}"));
        }
        if momentum.is_some_and(|m| !(0.0..=1.0).contains(&m)) {
            return fail(format!("the momentum must be in [0, 1], got {momentum:?}"));
        }
        let (weight, bias) = if affine {
            (
                Some(params.add(Tensor::full(&[channels], T::ONE))),
                Some(params.add(Tensor::zeros_generic(&[channels]))),
            )
        } else {
            (None, None)
        };
        let running = track_running_stats.then(|| RunningStats {
            mean: params.add(Tensor::zeros_generic(&[channels])),
            var: params.add(Tensor::full(&[channels], T::ONE)),
            count: Some(params.add(Tensor::zeros_generic(&[1]))),
            momentum,
        });
        Ok(BatchNorm2d {
            channels,
            spatial: size.0 * size.1,
            weight,
            bias,
            running,
            eps,
        })
    }

    /// Values per channel of each image.
    pub(crate) fn spatial(&self) -> usize {
        self.spatial
    }

    /// The number of channels.
    ///
    /// ```
    /// use cavitation_core::ParamStore;
    /// use cavitation_nn::BatchNorm2d;
    /// assert_eq!(BatchNorm2d::new(&mut ParamStore::new(), 3, (4, 4)).channels(), 3);
    /// ```
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// The value added to the variance before its square root.
    ///
    /// ```
    /// use cavitation_core::ParamStore;
    /// use cavitation_nn::BatchNorm2d;
    /// assert_eq!(BatchNorm2d::new(&mut ParamStore::new(), 3, (4, 4)).eps(), 1e-5);
    /// ```
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// The running statistics, if the layer keeps them.
    ///
    /// ```
    /// use cavitation_core::ParamStore;
    /// use cavitation_nn::BatchNorm2d;
    /// let bn = BatchNorm2d::new(&mut ParamStore::new(), 3, (4, 4));
    /// assert_eq!(bn.running().unwrap().momentum, Some(0.1));
    /// ```
    pub fn running(&self) -> Option<RunningStats> {
        self.running
    }

    /// γ's id, if the layer is affine.
    ///
    /// ```
    /// use cavitation_core::ParamStore;
    /// use cavitation_nn::BatchNorm2d;
    /// let mut params = ParamStore::new();
    /// let bn = BatchNorm2d::new(&mut params, 3, (4, 4));
    /// assert_eq!(params.value(bn.weight().unwrap()).data(), &[1.0; 3]);
    /// ```
    pub fn weight(&self) -> Option<ParamId> {
        self.weight
    }

    /// β's id, if the layer is affine.
    ///
    /// ```
    /// use cavitation_core::ParamStore;
    /// use cavitation_nn::BatchNorm2d;
    /// let mut params = ParamStore::new();
    /// let bn = BatchNorm2d::new(&mut params, 3, (4, 4));
    /// assert_eq!(params.value(bn.bias().unwrap()).data(), &[0.0; 3]);
    /// ```
    pub fn bias(&self) -> Option<ParamId> {
        self.bias
    }

    /// Whether `form` is this layer's evaluation form for `params` in shape: a
    /// [`InferenceLayer::BatchNorm2d`] with one value per channel, γ and β present exactly
    /// when the layer is affine, the layer's spatial size, and running statistics kept, so
    /// [`refresh_inference`](Self::refresh_inference) can overwrite its values.
    pub(crate) fn fits_inference(&self, params: &ParamValues<f32>, form: &InferenceLayer) -> bool {
        let InferenceLayer::BatchNorm2d {
            mean,
            inv,
            weight,
            bias,
            spatial,
        } = form
        else {
            return false;
        };
        let n = self.channels;
        let width = |id: ParamId| params.get(id).data().len() == n;
        self.running.is_some_and(|r| width(r.mean) && width(r.var))
            && mean.len() == n
            && inv.len() == n
            && *spatial == self.spatial
            && match (weight, self.weight) {
                (Some(w), Some(id)) => w.len() == n && width(id),
                (None, None) => true,
                _ => false,
            }
            && match (bias, self.bias) {
                (Some(b), Some(id)) => b.len() == n && width(id),
                (None, None) => true,
                _ => false,
            }
    }

    /// Overwrites the values of `form`, this layer's evaluation form, from `params`: each
    /// channel's running mean, `1 / √(var + eps)` computed as the tape computes it, and γ
    /// and β. Allocates nothing. Does nothing to a form
    /// [`fits_inference`](Self::fits_inference) rejects.
    pub(crate) fn refresh_inference(&self, params: &ParamValues<f32>, form: &mut InferenceLayer) {
        if !self.fits_inference(params, form) {
            return;
        }
        let (
            Some(r),
            InferenceLayer::BatchNorm2d {
                mean,
                inv,
                weight,
                bias,
                ..
            },
        ) = (self.running, form)
        else {
            return;
        };
        mean.copy_from_slice(params.get(r.mean).data());
        let eps = self.eps as f32;
        for (i, &v) in inv.iter_mut().zip(params.get(r.var).data()) {
            *i = 1.0 / (v + eps).sqrt();
        }
        if let (Some(w), Some(id)) = (weight.as_mut(), self.weight) {
            w.copy_from_slice(params.get(id).data());
        }
        if let (Some(b), Some(id)) = (bias.as_mut(), self.bias) {
            b.copy_from_slice(params.get(id).data());
        }
    }
}

impl BatchNorm2d {
    /// Records this layer followed by ReLU as one node ([`Tape::batch_norm2d_relu`]), for
    /// [`Sequential`](crate::Sequential)'s fused pairs.
    ///
    /// # Panics
    ///
    /// As [`Layer::forward`].
    pub(crate) fn forward_relu<T: Real>(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        let w = self.weight.map(|id| tape.param(id));
        let b = self.bias.map(|id| tape.param(id));
        tape.batch_norm2d_relu(x, w, b, self.running, self.channels, self.spatial, self.eps)
    }
}

impl<T: Real> Layer<T> for BatchNorm2d {
    /// Returns `"batch_norm2d"`.
    fn name(&self) -> &str {
        "batch_norm2d"
    }

    /// The width is unchanged.
    fn output_width(&self, input_width: usize) -> usize {
        input_width
    }

    /// Records batch normalisation of `x`: with the batch's statistics on a training tape
    /// (recording the running statistics' update) or when the layer keeps none, with the
    /// running statistics otherwise.
    ///
    /// # Panics
    ///
    /// Panics if `x` is not `[batch, channels · height · width]`, naming the layer.
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        let w = self.weight.map(|id| tape.param(id));
        let b = self.bias.map(|id| tape.param(id));
        tape.batch_norm2d(x, w, b, self.running, self.channels, self.spatial, self.eps)
    }

    /// `weight` and `bias`, if the layer is affine.
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        match (self.weight, self.bias) {
            (Some(w), Some(b)) => vec![("weight", w), ("bias", b)],
            _ => Vec::new(),
        }
    }

    /// `running_mean`, `running_var` and `num_batches_tracked`, if the layer keeps them.
    fn buffers(&self) -> Vec<(&'static str, ParamId)> {
        match self.running {
            Some(r) => {
                let mut b = vec![("running_mean", r.mean), ("running_var", r.var)];
                b.extend(r.count.map(|c| ("num_batches_tracked", c)));
                b
            }
            None => Vec::new(),
        }
    }

    /// The width of one input image.
    fn input_width(&self) -> Option<usize> {
        Some(self.channels * self.spatial)
    }

    /// Each output value depends on the whole batch in training, so a neuron mask on the
    /// input does not switch off one output alone.
    fn is_elementwise(&self) -> bool {
        false
    }

    /// [`InferenceLayer::BatchNorm2d`]: the evaluation form, with each channel's running
    /// mean and `1 / √(var + eps)` computed as the tape computes them, and γ and β, so a
    /// compiled model gives the tape's bits; `None` for a layer without running statistics,
    /// which normalises by each batch and has no fixed form.
    fn inference(&self, params: &ParamValues<f32>) -> Option<InferenceLayer> {
        self.running?;
        let n = self.channels;
        let mut form = InferenceLayer::BatchNorm2d {
            mean: vec![0.0; n],
            inv: vec![0.0; n],
            weight: self.weight.map(|_| vec![0.0; n]),
            bias: self.bias.map(|_| vec![0.0; n]),
            spatial: self.spatial,
        };
        self.refresh_inference(params, &mut form);
        Some(form)
    }

    /// The layer itself.
    fn batch_norm2d_form(&self, _: crate::layer::private::Token) -> Option<&BatchNorm2d> {
        Some(self)
    }
}

/// Tests for the `BatchNorm2d` layer: training updates through `Trainer`, evaluation,
/// saving and loading the buffers, inference, and the layer without running statistics.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Conv2d, Dataset, InferenceModel, Linear, Loss, Relu, Sequential, Sgd, Targets, Trainer,
        load, save,
    };
    use cavitation_core::{Conv2dGeom, Graph, Rng};

    /// A 1 → 2 channel convolution on 4 × 4 images, batch norm, ReLU and a `Linear` head,
    /// with 6 seeded rows of classes.
    fn net_and_data(params: &mut ParamStore) -> (Sequential, BatchNorm2d, Dataset) {
        let mut rng = Rng::seed(3);
        let g = Conv2dGeom::with_options(1, 2, (4, 4), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
        let conv = Conv2d::new(params, g, &mut rng);
        let bn = BatchNorm2d::new(params, 2, (4, 4));
        let net = Sequential::new()
            .push(conv)
            .push(bn.clone())
            .push(Relu)
            .push(Linear::new(params, 32, 3, &mut rng));
        let x: Vec<f32> = (0..6 * 16).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        let data = Dataset {
            x: Tensor::from_vec(x, &[6, 16]).unwrap(),
            targets: Targets::Classes(vec![0, 1, 2, 0, 1, 2]),
        };
        (net, bn, data)
    }

    /// Each kept `Trainer` step moves the running statistics by PyTorch's rule from the
    /// batch's statistics of the batch norm's input (computed here from the convolution's
    /// output on the tape), and counts the batch.
    #[test]
    fn trainer_updates_running_statistics() {
        let mut params = ParamStore::new();
        let (net, bn, data) = net_and_data(&mut params);
        let r = bn.running().unwrap();
        let mut trainer = Trainer::new(&net, Sgd::new(0.0), Loss::CrossEntropy, 6);
        let mut want_mean = [0.0f64; 2];
        let mut want_var = [1.0f64; 2];
        for _ in 0..3 {
            // The convolution's output for the whole batch, before the step.
            let mut graph = Graph::new();
            let mut tape = graph.tape(&params.values);
            let x = tape.input(data.x.view());
            let y = net.layer(0).forward(&mut tape, x);
            let out: Vec<f64> = tape.value(y).data().iter().map(|&v| f64::from(v)).collect();
            drop(tape);
            for c in 0..2 {
                let vals: Vec<f64> = (0..6)
                    .flat_map(|n| out[n * 32 + c * 16..][..16].to_vec())
                    .collect();
                let m = vals.iter().sum::<f64>() / 96.0;
                let v = vals.iter().map(|a| (a - m).powi(2)).sum::<f64>() / 95.0;
                want_mean[c] = 0.1 * m + 0.9 * want_mean[c];
                want_var[c] = 0.1 * v + 0.9 * want_var[c];
            }
            let view = crate::BatchView {
                x: data.x.view(),
                targets: crate::TargetsView::Classes(match &data.targets {
                    Targets::Classes(c) => c,
                    _ => unreachable!(),
                }),
            };
            trainer.step(&net, &mut params, view).unwrap();
        }
        for c in 0..2 {
            let (m, v) = (
                params.value(r.mean).data()[c],
                params.value(r.var).data()[c],
            );
            assert!(
                (f64::from(m) - want_mean[c]).abs() < 1e-5,
                "mean {m} {}",
                want_mean[c]
            );
            assert!(
                (f64::from(v) - want_var[c]).abs() < 1e-5,
                "var {v} {}",
                want_var[c]
            );
        }
        assert_eq!(params.value(r.count.unwrap()).data(), &[3.0]);
    }

    /// The compiled model gives the tape's evaluation bits (within `1e-5 · max(1, |tape|)`
    /// at the Accelerate level, where the network's convolution may round differently), and
    /// a saved model with its buffers loads back exactly; a file without `num_batches_tracked` loads with a count of
    /// 0.
    #[test]
    fn inference_and_save_load_carry_the_buffers() {
        let mut params = ParamStore::new();
        let (net, bn, data) = net_and_data(&mut params);
        let r = bn.running().unwrap();
        params
            .value_mut(r.mean)
            .data_mut()
            .copy_from_slice(&[0.3, -0.2]);
        params
            .value_mut(r.var)
            .data_mut()
            .copy_from_slice(&[1.7, 0.4]);
        params.value_mut(r.count.unwrap()).data_mut()[0] = 12.0;
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let x = tape.input(data.x.view());
        let y = net.forward(&mut tape, x);
        let want = tape.value(y).data().to_vec();
        drop(tape);
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        // The network's convolution gives the tape's bits at the library's own kernel
        // levels; Accelerate's sgemm may block the compiled model's products, of other
        // shapes than the tape's, differently, so there it is within 1e-5 · max(1, |tape|).
        let got = model.infer_batch(data.x.view()).data().to_vec();
        if cavitation_core::kernels::level() == cavitation_core::kernels::Level::Accelerate {
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() <= 1e-5 * w.abs().max(1.0), "{g} against {w}");
            }
        } else {
            assert_eq!(got, want);
        }
        let path =
            std::env::temp_dir().join(format!("cavitation-bn-{}.safetensors", std::process::id()));
        save(&path, &net, &params).unwrap();
        let mut fresh = ParamStore::new();
        let (net2, bn2, _) = net_and_data(&mut fresh);
        load(&path, &net2, &mut fresh).unwrap();
        let r2 = bn2.running().unwrap();
        assert_eq!(fresh.value(r2.mean).data(), &[0.3, -0.2]);
        assert_eq!(fresh.value(r2.var).data(), &[1.7, 0.4]);
        assert_eq!(fresh.value(r2.count.unwrap()).data(), &[12.0]);
        let bytes = std::fs::read(&path).unwrap();
        let file = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let count = file.tensor("1.num_batches_tracked").unwrap();
        assert_eq!(
            (count.dtype(), count.shape()),
            (safetensors::Dtype::I64, &[][..])
        );
        std::fs::remove_file(&path).unwrap();
    }

    /// A compiled model refreshes a `BatchNorm2d` in place: new running statistics, γ and β
    /// give bit for bit the outputs of compiling afresh. A model whose batch norm has a
    /// different number of channels, or is not affine where the compiled one is, does not
    /// fit and is left unchanged.
    #[test]
    fn refresh_in_place_matches_compile() {
        let mut params = ParamStore::new();
        let (net, bn, data) = net_and_data(&mut params);
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        let before = model.infer_batch(data.x.view()).data().to_vec();
        let r = bn.running().unwrap();
        params
            .value_mut(r.mean)
            .data_mut()
            .copy_from_slice(&[0.25, -0.5]);
        params
            .value_mut(r.var)
            .data_mut()
            .copy_from_slice(&[2.5, 0.3]);
        params
            .value_mut(bn.weight().unwrap())
            .data_mut()
            .copy_from_slice(&[1.5, -0.75]);
        params
            .value_mut(bn.bias().unwrap())
            .data_mut()
            .copy_from_slice(&[0.1, 0.2]);
        assert!(model.refresh_in_place(&net, &params));
        let got = model.infer_batch(data.x.view()).data().to_vec();
        let mut want = InferenceModel::compile(&net, &params).unwrap();
        assert_eq!(got, want.infer_batch(data.x.view()).data());
        assert_ne!(got, before);

        let mut rng = Rng::seed(5);
        let mut other = ParamStore::new();
        let g = Conv2dGeom::with_options(1, 2, (4, 4), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
        let plain =
            BatchNorm2d::with_options(&mut other, 2, (4, 4), 1e-5, Some(0.1), false, true).unwrap();
        let not_affine = Sequential::new()
            .push(Conv2d::new(&mut other, g, &mut rng))
            .push(plain)
            .push(Relu)
            .push(Linear::new(&mut other, 32, 3, &mut rng));
        assert!(!model.refresh_in_place(&not_affine, &other));
        assert_eq!(model.infer_batch(data.x.view()).data(), got.as_slice());
    }

    /// Without running statistics the layer normalises by each batch in evaluation too, and
    /// has no inference form.
    #[test]
    fn without_running_statistics_it_uses_the_batch() {
        let mut params = ParamStore::new();
        let bn = BatchNorm2d::with_options(&mut params, 1, (1, 2), 0.0, Some(0.1), false, false)
            .unwrap();
        assert!(Layer::<f32>::buffers(&bn).is_empty());
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let x = tape.input(Tensor::from_vec(vec![1.0f32, 3.0], &[1, 2]).unwrap().view());
        let y = bn.forward(&mut tape, x);
        assert_eq!(tape.value(y).data(), &[-1.0, 1.0]);
        drop(tape);
        let net = Sequential::new()
            .push(Linear::new(&mut params, 2, 2, &mut Rng::seed(0)))
            .push(bn);
        assert!(InferenceModel::compile(&net, &params).is_err());
    }
}
