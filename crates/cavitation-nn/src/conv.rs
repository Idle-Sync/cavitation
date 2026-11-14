//! The `Conv2d` layer: PyTorch's `nn.Conv2d` on images stored as flat rows.

use cavitation_core::{Conv2dGeom, ParamId, ParamStore, ParamValues, Real, Rng, Tape, Tensor, Var};

use crate::{InferenceLayer, Init, Layer};

/// A 2-D convolution layer, PyTorch's `nn.Conv2d` with `groups = 1` and zero padding.
///
/// Its input rows are images in NCHW order (channel after channel, each row by row, as
/// `torch.flatten(x, 1)` lays them out), `in_channels · height · width` values each, and its
/// output rows are the output images in the same order. The weight is `[out, in, kh, kw]` and
/// the bias `[out]`, under PyTorch's names `weight` and `bias`, so a PyTorch state dict loads
/// into it by key.
///
/// ```
/// use cavitation_core::{Conv2dGeom, Graph, ParamStore, Rng, Tensor};
/// use cavitation_nn::{Conv2d, Layer};
/// let mut params = ParamStore::new();
/// // 3 x 32 x 32 images to 16 channels of 32 x 32 (a 3 x 3 kernel, padding 1).
/// let geom = Conv2dGeom::with_options(3, 16, (32, 32), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
/// let conv = Conv2d::new(&mut params, geom, &mut Rng::seed(0));
/// assert_eq!(Layer::<f32>::output_width(&conv, 3 * 32 * 32), 16 * 32 * 32);
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&params.values);
/// let x = tape.input(Tensor::full(&[2, 3 * 32 * 32], 0.5f32).view());
/// let y = conv.forward(&mut tape, x);
/// assert_eq!(tape.shape(y).dims(), &[2, 16 * 32 * 32]);
/// ```
#[derive(Clone, Debug)]
pub struct Conv2d {
    /// The convolution's shape.
    geom: Conv2dGeom,
    /// The weight, `[out, in, kh, kw]`.
    weight: ParamId,
    /// The bias, `[out]`, or `None` for a layer without one.
    bias: Option<ParamId>,
}

impl Conv2d {
    /// A layer with PyTorch's default initialisation ([`Init::TorchDefault`] on the weight
    /// and the bias), adding both to `params`.
    ///
    /// ```
    /// use cavitation_core::{Conv2dGeom, ParamStore, Rng};
    /// use cavitation_nn::Conv2d;
    /// let mut params = ParamStore::new();
    /// let geom = Conv2dGeom::new(1, 6, (28, 28), (5, 5)).unwrap();
    /// let conv = Conv2d::new(&mut params, geom, &mut Rng::seed(0));
    /// assert_eq!(params.value(conv.weight()).shape().dims(), &[6, 1, 5, 5]);
    /// let bound = 1.0 / 25f32.sqrt();
    /// assert!(params.value(conv.weight()).data().iter().all(|w| w.abs() <= bound));
    /// ```
    pub fn new<T: Real>(params: &mut ParamStore<T>, geom: Conv2dGeom, rng: &mut Rng) -> Conv2d {
        Conv2d::with_init(
            params,
            geom,
            Init::TorchDefault,
            Some(Init::TorchDefault),
            rng,
        )
    }

    /// A layer whose weight is filled by `weight` and whose bias, if `bias` is `Some`, by
    /// that initialiser (the weight first, then the bias, drawing from `rng` in that order);
    /// `None` makes a layer without a bias, PyTorch's `bias=False`. The fans are
    /// `in · kh · kw` and `out · kh · kw`, as PyTorch computes them for a convolution.
    ///
    /// ```
    /// use cavitation_core::{Conv2dGeom, ParamStore, Rng};
    /// use cavitation_nn::{Conv2d, Init};
    /// let mut params = ParamStore::new();
    /// let geom = Conv2dGeom::new(3, 8, (8, 8), (3, 3)).unwrap();
    /// let conv = Conv2d::with_init(&mut params, geom, Init::HeNormal, None, &mut Rng::seed(0));
    /// assert_eq!(conv.bias(), None);
    /// assert_eq!(params.len(), 1);
    /// ```
    pub fn with_init<T: Real>(
        params: &mut ParamStore<T>,
        geom: Conv2dGeom,
        weight: Init,
        bias: Option<Init>,
        rng: &mut Rng,
    ) -> Conv2d {
        let (kh, kw) = geom.kernel();
        let (fan_in, fan_out) = (geom.kernel_len(), geom.out_channels() * kh * kw);
        let mut w = Tensor::<T>::zeros_generic(&[geom.out_channels(), geom.in_channels(), kh, kw]);
        weight.fill(&mut w, fan_in, fan_out, rng);
        let weight = params.add(w);
        let bias = bias.map(|init| {
            let mut b = Tensor::<T>::zeros_generic(&[geom.out_channels()]);
            init.fill(&mut b, fan_in, fan_out, rng);
            params.add(b)
        });
        Conv2d { geom, weight, bias }
    }

    /// The convolution's shape.
    ///
    /// ```
    /// use cavitation_core::{Conv2dGeom, ParamStore, Rng};
    /// use cavitation_nn::Conv2d;
    /// let geom = Conv2dGeom::new(1, 2, (5, 5), (3, 3)).unwrap();
    /// let conv = Conv2d::new(&mut ParamStore::new(), geom, &mut Rng::seed(0));
    /// assert_eq!(conv.geom(), geom);
    /// ```
    pub fn geom(&self) -> Conv2dGeom {
        self.geom
    }

    /// The weight's id, `[out, in, kh, kw]`.
    ///
    /// ```
    /// use cavitation_core::{Conv2dGeom, ParamStore, Rng};
    /// use cavitation_nn::Conv2d;
    /// let mut params = ParamStore::new();
    /// let geom = Conv2dGeom::new(2, 4, (5, 5), (3, 3)).unwrap();
    /// let conv = Conv2d::new(&mut params, geom, &mut Rng::seed(0));
    /// assert_eq!(params.value(conv.weight()).data().len(), 4 * 2 * 3 * 3);
    /// ```
    pub fn weight(&self) -> ParamId {
        self.weight
    }

    /// The bias's id, `[out]`, or `None` for a layer without one.
    ///
    /// ```
    /// use cavitation_core::{Conv2dGeom, ParamStore, Rng};
    /// use cavitation_nn::Conv2d;
    /// let mut params = ParamStore::new();
    /// let geom = Conv2dGeom::new(2, 4, (5, 5), (3, 3)).unwrap();
    /// let conv = Conv2d::new(&mut params, geom, &mut Rng::seed(0));
    /// assert_eq!(params.value(conv.bias().unwrap()).data().len(), 4);
    /// ```
    pub fn bias(&self) -> Option<ParamId> {
        self.bias
    }
}

impl<T: Real> Layer<T> for Conv2d {
    /// Returns `"conv2d"`.
    fn name(&self) -> &str {
        "conv2d"
    }

    /// The width of one output image, whatever `input_width` is: a mismatch is reported by
    /// `forward` and by the network's width checks, which read
    /// [`input_width`](Layer::input_width).
    fn output_width(&self, _input_width: usize) -> usize {
        self.geom.out_width()
    }

    /// Records the convolution of `x`, `[batch, in · height · width]`.
    ///
    /// # Panics
    ///
    /// Panics if `x` is not `[batch, in · height · width]`, naming the layer, the geometry
    /// and the shape.
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        let shape = tape.shape(x);
        assert!(
            shape.rank() == 2 && shape.dims()[1] == self.geom.in_width(),
            "conv2d: expected inputs of width {} ({} channels of {:?}), got shape {:?}",
            self.geom.in_width(),
            self.geom.in_channels(),
            self.geom.input(),
            shape.dims()
        );
        let w = tape.param(self.weight);
        let b = self.bias.map(|id| tape.param(id));
        tape.conv2d(x, w, b, self.geom)
    }

    /// Returns `weight` and `bias`, or only `weight` for a layer without a bias.
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        match self.bias {
            Some(bias) => vec![("weight", self.weight), ("bias", bias)],
            None => vec![("weight", self.weight)],
        }
    }

    /// `weight` or, if the layer has one, `bias`, without allocating.
    fn parameter(&self, local: &str) -> Option<ParamId> {
        match local {
            "weight" => Some(self.weight),
            "bias" => self.bias,
            _ => None,
        }
    }

    /// The width of one input image.
    fn input_width(&self) -> Option<usize> {
        Some(self.geom.in_width())
    }

    /// [`InferenceLayer::Conv2d`], with the weight and the bias copied as they are.
    ///
    /// # Panics
    ///
    /// Panics, naming the shapes, if the weight in `params` is not `[out, in, kh, kw]` or the
    /// bias not `[out]`.
    fn inference(&self, params: &ParamValues<f32>) -> Option<InferenceLayer> {
        let (kh, kw) = self.geom.kernel();
        let w = params.get(self.weight);
        let dims = [self.geom.out_channels(), self.geom.in_channels(), kh, kw];
        assert!(
            w.shape().dims() == dims,
            "conv2d: expected weight {dims:?}, got {:?}",
            w.shape().dims()
        );
        let bias = self.bias.map(|id| {
            let b = params.get(id);
            assert!(
                b.shape().dims() == [self.geom.out_channels()],
                "conv2d: expected bias [{}], got {:?}",
                self.geom.out_channels(),
                b.shape().dims()
            );
            b.data().to_vec()
        });
        Some(InferenceLayer::Conv2d {
            geom: self.geom,
            weight: w.data().to_vec(),
            bias,
        })
    }
}

/// Tests for the `Conv2d` layer: initialisation, the gradient check, width checks and
/// inference.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InferenceModel, Linear, Relu, Sequential};
    use cavitation_core::{Graph, gradcheck_params};

    /// A 2 → 3 channel 3 × 3 convolution with padding 1 on 6 × 5 images.
    fn geom() -> Conv2dGeom {
        Conv2dGeom::with_options(2, 3, (6, 5), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap()
    }

    /// PyTorch's default draws every weight and bias in ±1/√fan_in, with the uniform's
    /// variance (bound² / 3) to within 10%.
    #[test]
    fn default_init_matches_pytorch_bounds() {
        let mut params = ParamStore::new();
        let g = Conv2dGeom::new(16, 32, (8, 8), (3, 3)).unwrap();
        let conv = Conv2d::new(&mut params, g, &mut Rng::seed(3));
        let bound = 1.0 / (16.0f64 * 9.0).sqrt();
        let w = params.value(conv.weight()).data();
        assert!(w.iter().all(|&v| f64::from(v).abs() <= bound));
        let var = w.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / w.len() as f64;
        assert!(
            (var / (bound * bound / 3.0) - 1.0).abs() < 0.1,
            "variance {var}"
        );
        let b = params.value(conv.bias().unwrap()).data();
        assert!(b.iter().all(|&v| f64::from(v).abs() <= bound));
    }

    /// He initialisation reads the convolution's fan-in, `in · kh · kw`.
    #[test]
    fn he_init_uses_the_convolution_fans() {
        let mut params = ParamStore::new();
        let g = Conv2dGeom::new(4, 8, (6, 6), (3, 3)).unwrap();
        let conv = Conv2d::with_init(&mut params, g, Init::HeUniform, None, &mut Rng::seed(0));
        let bound = (6.0f64 / 36.0).sqrt();
        let w = params.value(conv.weight()).data();
        assert!(w.iter().all(|&v| f64::from(v).abs() <= bound));
        assert!(w.iter().any(|&v| f64::from(v).abs() > 0.9 * bound));
    }

    /// The gradient check passes through the layer and a following `Linear`, in f64.
    #[test]
    fn conv_layer_grad_check() {
        let mut params = ParamStore::<f64>::new_generic();
        let mut rng = Rng::seed(1);
        let conv = Conv2d::new(&mut params, geom(), &mut rng);
        let head = Linear::new(&mut params, geom().out_width(), 2, &mut rng);
        let net = Sequential::new_generic().push(conv).push(head);
        let x: Vec<f64> = (0..2 * geom().in_width())
            .map(|_| rng.range(-1.0, 1.0))
            .collect();
        let x = Tensor::from_vec(x, &[2, geom().in_width()]).unwrap();
        gradcheck_params(&mut params, &[x], |tape, v| {
            let y = net.forward(tape, v[0]);
            let t = tape.tanh(y);
            tape.sum(t)
        })
        .assert_passed();
    }

    /// An input of the wrong width panics, naming the layer, the geometry and the shape.
    #[test]
    #[should_panic(expected = "conv2d: expected inputs of width 60 (2 channels of (6, 5))")]
    fn wrong_width_panics() {
        let mut params = ParamStore::new();
        let conv = Conv2d::new(&mut params, geom(), &mut Rng::seed(0));
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let x = tape.input(Tensor::full(&[1, 59], 1.0f32).view());
        Layer::<f32>::forward(&conv, &mut tape, x);
    }

    /// Inference gives the tape's bits for one image and for a batch, through two
    /// convolutions (one 1 × 1) and their ReLUs, with and without a bias: both run the same
    /// GEMMs on the same column matrices.
    #[test]
    fn conv_inference_matches_tape() {
        for with_bias in [true, false] {
            let mut params = ParamStore::new();
            let mut rng = Rng::seed(9);
            let bias = with_bias.then_some(Init::TorchDefault);
            let c1 = Conv2d::with_init(&mut params, geom(), Init::TorchDefault, bias, &mut rng);
            let g2 = Conv2dGeom::new(3, 4, (6, 5), (1, 1)).unwrap();
            let c2 = Conv2d::with_init(&mut params, g2, Init::TorchDefault, bias, &mut rng);
            let net = Sequential::new().push(c1).push(Relu).push(c2).push(Relu);
            let rows = 5;
            let x: Vec<f32> = (0..rows * geom().in_width())
                .map(|_| rng.range(-1.0, 1.0) as f32)
                .collect();
            let x = Tensor::from_vec(x, &[rows, geom().in_width()]).unwrap();
            let mut graph = Graph::new();
            let mut tape = graph.tape(&params.values);
            let xv = tape.input(x.view());
            let y = net.forward(&mut tape, xv);
            let want = tape.value(y).data().to_vec();
            drop(tape);
            let mut model = InferenceModel::compile(&net, &params).unwrap();
            let width = g2.out_width();
            assert_eq!(model.infer_batch(x.view()).data(), want.as_slice());
            for (r, row) in x.data().chunks(geom().in_width()).enumerate() {
                assert_eq!(model.infer(row), &want[r * width..][..width], "row {r}");
            }
        }
    }
}
