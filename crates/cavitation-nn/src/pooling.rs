//! The pooling layers: `MaxPool2d`, `AvgPool2d` and `AdaptiveAvgPool2d`, PyTorch's layers of
//! the same names on images stored as flat rows.

use cavitation_core::kernels::pooling::Pool2dGeom;
use cavitation_core::{Error, ParamId, ParamValues, Real, Tape, Var};

use crate::{InferenceLayer, Layer};

/// Panics, naming `layer` and the shape, unless `x` is `[batch, width]`.
fn check_input<T: Real>(layer: &str, tape: &Tape<'_, T>, x: Var, width: usize) {
    let shape = tape.shape(x);
    assert!(
        shape.rank() == 2 && shape.dims()[1] == width,
        "{layer}: expected inputs of width {width}, got shape {:?}",
        shape.dims()
    );
}

/// 2-D max pooling, PyTorch's `nn.MaxPool2d`: each output is the largest value of its
/// window, ties going to the first in row-major order and a NaN winning, as PyTorch's CPU
/// kernel does; the gradient goes to each window's winner.
///
/// ```
/// use cavitation_core::kernels::pooling::Pool2dGeom;
/// use cavitation_core::{Graph, ParamStore, Tensor};
/// use cavitation_nn::{Layer, MaxPool2d};
/// let pool = MaxPool2d::new(Pool2dGeom::new(8, (32, 32), (2, 2)).unwrap());
/// assert_eq!(Layer::<f32>::output_width(&pool, 8 * 32 * 32), 8 * 16 * 16);
/// let store = ParamStore::new();
/// let mut graph = Graph::new();
/// let mut tape = graph.tape(&store.values);
/// let x = tape.input(Tensor::full(&[2, 8 * 32 * 32], 1.0f32).view());
/// let y = pool.forward(&mut tape, x);
/// assert_eq!(tape.shape(y).dims(), &[2, 8 * 16 * 16]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaxPool2d {
    /// The pooling's shape.
    geom: Pool2dGeom,
}

impl MaxPool2d {
    /// A max-pooling layer of the given shape.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// use cavitation_nn::MaxPool2d;
    /// let geom = Pool2dGeom::new(6, (28, 28), (2, 2)).unwrap();
    /// assert_eq!(MaxPool2d::new(geom).geom(), geom);
    /// ```
    pub fn new(geom: Pool2dGeom) -> MaxPool2d {
        MaxPool2d { geom }
    }

    /// The pooling's shape.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// use cavitation_nn::MaxPool2d;
    /// let geom = Pool2dGeom::new(1, (4, 4), (2, 2)).unwrap();
    /// assert_eq!(MaxPool2d::new(geom).geom().out_size(), (2, 2));
    /// ```
    pub fn geom(&self) -> Pool2dGeom {
        self.geom
    }
}

impl<T: Real> Layer<T> for MaxPool2d {
    /// Returns `"max_pool2d"`.
    fn name(&self) -> &str {
        "max_pool2d"
    }

    /// The width of one output image.
    fn output_width(&self, _input_width: usize) -> usize {
        self.geom.out_width()
    }

    /// Records max pooling of `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x` is not `[batch, channels · height · width]`, naming the layer.
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        check_input("max_pool2d", tape, x, self.geom.in_width());
        tape.max_pool2d(x, self.geom)
    }

    /// No parameters.
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        Vec::new()
    }

    /// The width of one input image.
    fn input_width(&self) -> Option<usize> {
        Some(self.geom.in_width())
    }

    /// [`InferenceLayer::MaxPool2d`].
    fn inference(&self, _params: &ParamValues<f32>) -> Option<InferenceLayer> {
        Some(InferenceLayer::MaxPool2d(self.geom))
    }
}

/// 2-D average pooling, PyTorch's `nn.AvgPool2d`: each output is the sum of its window's
/// pixels divided by the window's size counting the padding (`count_include_pad`, PyTorch's
/// default), by the pixels summed, or by a divisor given in its place.
///
/// ```
/// use cavitation_core::kernels::pooling::Pool2dGeom;
/// use cavitation_nn::{AvgPool2d, Layer};
/// let pool = AvgPool2d::new(Pool2dGeom::new(16, (8, 8), (2, 2)).unwrap());
/// assert_eq!(Layer::<f32>::output_width(&pool, 16 * 64), 16 * 16);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AvgPool2d {
    /// The pooling's shape; never dilated.
    geom: Pool2dGeom,
    /// Whether the padding counts towards the divisor.
    count_include_pad: bool,
    /// The divisor used in place of the window's size, if any.
    divisor_override: Option<usize>,
}

impl AvgPool2d {
    /// An average-pooling layer of the given shape with PyTorch's defaults: the padding
    /// counts, no divisor override.
    ///
    /// # Panics
    ///
    /// Panics if `geom` is dilated: average pooling has no dilation.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// use cavitation_nn::AvgPool2d;
    /// let pool = AvgPool2d::new(Pool2dGeom::new(1, (4, 4), (2, 2)).unwrap());
    /// assert!(pool.count_include_pad());
    /// ```
    pub fn new(geom: Pool2dGeom) -> AvgPool2d {
        AvgPool2d::with_options(geom, true, None).expect("an undilated geometry")
    }

    /// An average-pooling layer with every setting given.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if `geom` is dilated or `divisor_override` is `Some(0)`.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// use cavitation_nn::AvgPool2d;
    /// let geom = Pool2dGeom::new(1, (4, 4), (2, 2)).unwrap();
    /// let pool = AvgPool2d::with_options(geom, false, Some(3)).unwrap();
    /// assert_eq!(pool.divisor_override(), Some(3));
    /// assert!(AvgPool2d::with_options(geom, true, Some(0)).is_err());
    /// ```
    pub fn with_options(
        geom: Pool2dGeom,
        count_include_pad: bool,
        divisor_override: Option<usize>,
    ) -> Result<AvgPool2d, Error> {
        if geom.dilation() != (1, 1) {
            return Err(Error::InvalidData(format!(
                "avg_pool2d: average pooling has no dilation, got {:?}",
                geom.dilation()
            )));
        }
        if divisor_override == Some(0) {
            return Err(Error::InvalidData(
                "avg_pool2d: the divisor override must not be 0".to_string(),
            ));
        }
        Ok(AvgPool2d {
            geom,
            count_include_pad,
            divisor_override,
        })
    }

    /// The pooling's shape.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// use cavitation_nn::AvgPool2d;
    /// let geom = Pool2dGeom::new(1, (4, 4), (2, 2)).unwrap();
    /// assert_eq!(AvgPool2d::new(geom).geom(), geom);
    /// ```
    pub fn geom(&self) -> Pool2dGeom {
        self.geom
    }

    /// Whether the padding counts towards the divisor.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// use cavitation_nn::AvgPool2d;
    /// let geom = Pool2dGeom::new(1, (4, 4), (2, 2)).unwrap();
    /// assert!(!AvgPool2d::with_options(geom, false, None).unwrap().count_include_pad());
    /// ```
    pub fn count_include_pad(&self) -> bool {
        self.count_include_pad
    }

    /// The divisor used in place of the window's size, if any.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// use cavitation_nn::AvgPool2d;
    /// let geom = Pool2dGeom::new(1, (4, 4), (2, 2)).unwrap();
    /// assert_eq!(AvgPool2d::new(geom).divisor_override(), None);
    /// ```
    pub fn divisor_override(&self) -> Option<usize> {
        self.divisor_override
    }
}

impl<T: Real> Layer<T> for AvgPool2d {
    /// Returns `"avg_pool2d"`.
    fn name(&self) -> &str {
        "avg_pool2d"
    }

    /// The width of one output image.
    fn output_width(&self, _input_width: usize) -> usize {
        self.geom.out_width()
    }

    /// Records average pooling of `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x` is not `[batch, channels · height · width]`, naming the layer.
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        check_input("avg_pool2d", tape, x, self.geom.in_width());
        tape.avg_pool2d(x, self.geom, self.count_include_pad, self.divisor_override)
    }

    /// No parameters.
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        Vec::new()
    }

    /// The width of one input image.
    fn input_width(&self) -> Option<usize> {
        Some(self.geom.in_width())
    }

    /// [`InferenceLayer::AvgPool2d`].
    fn inference(&self, _params: &ParamValues<f32>) -> Option<InferenceLayer> {
        Some(InferenceLayer::AvgPool2d {
            geom: self.geom,
            count_include_pad: self.count_include_pad,
            divisor_override: self.divisor_override,
        })
    }
}

/// 2-D adaptive average pooling, PyTorch's `nn.AdaptiveAvgPool2d`: pools `channels` planes
/// of `input` to `output` (height, width) with PyTorch's windows; output `(1, 1)` is global
/// average pooling, the usual step before a CNN's `Linear` head.
///
/// ```
/// use cavitation_nn::{AdaptiveAvgPool2d, Layer};
/// let pool = AdaptiveAvgPool2d::new(64, (8, 8), (1, 1)).unwrap();
/// assert_eq!(Layer::<f32>::output_width(&pool, 64 * 64), 64);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveAvgPool2d {
    /// Channels of each image.
    channels: usize,
    /// Height and width of each input image.
    input: (usize, usize),
    /// Height and width of each output image.
    output: (usize, usize),
}

impl AdaptiveAvgPool2d {
    /// A layer pooling `channels` planes of `input` to `output`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if any count or size is 0, or an image's values
    /// (`channels · height · width`, in or out) overflow `usize`, naming the sizes.
    ///
    /// ```
    /// use cavitation_nn::AdaptiveAvgPool2d;
    /// assert!(AdaptiveAvgPool2d::new(3, (7, 7), (3, 3)).is_ok());
    /// assert!(AdaptiveAvgPool2d::new(3, (7, 7), (0, 3)).is_err());
    /// assert!(AdaptiveAvgPool2d::new(usize::MAX, (2, 2), (1, 1)).is_err());
    /// ```
    pub fn new(
        channels: usize,
        input: (usize, usize),
        output: (usize, usize),
    ) -> Result<AdaptiveAvgPool2d, Error> {
        if channels == 0 || input.0 == 0 || input.1 == 0 || output.0 == 0 || output.1 == 0 {
            return Err(Error::InvalidData(format!(
                "adaptive_avg_pool2d: the sizes must be at least 1, got {channels} channels, \
                 input {input:?} and output {output:?}"
            )));
        }
        let values = |(h, w): (usize, usize)| channels.checked_mul(h)?.checked_mul(w);
        if values(input).is_none() || values(output).is_none() {
            return Err(Error::InvalidData(format!(
                "adaptive_avg_pool2d: {channels} channels of {input:?} pooled to {output:?} \
                 is more values per image than memory can hold"
            )));
        }
        Ok(AdaptiveAvgPool2d {
            channels,
            input,
            output,
        })
    }

    /// Channels, input size and output size.
    ///
    /// ```
    /// use cavitation_nn::AdaptiveAvgPool2d;
    /// let pool = AdaptiveAvgPool2d::new(3, (7, 7), (1, 1)).unwrap();
    /// assert_eq!(pool.sizes(), (3, (7, 7), (1, 1)));
    /// ```
    pub fn sizes(&self) -> (usize, (usize, usize), (usize, usize)) {
        (self.channels, self.input, self.output)
    }

    /// Values in one input image.
    fn in_width(&self) -> usize {
        self.channels * self.input.0 * self.input.1
    }
}

impl<T: Real> Layer<T> for AdaptiveAvgPool2d {
    /// Returns `"adaptive_avg_pool2d"`.
    fn name(&self) -> &str {
        "adaptive_avg_pool2d"
    }

    /// The width of one output image: `channels · out_height · out_width`.
    fn output_width(&self, _input_width: usize) -> usize {
        self.channels * self.output.0 * self.output.1
    }

    /// Records adaptive average pooling of `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x` is not `[batch, channels · height · width]`, naming the layer.
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        check_input("adaptive_avg_pool2d", tape, x, self.in_width());
        tape.adaptive_avg_pool2d(x, self.channels, self.input, self.output)
    }

    /// No parameters.
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        Vec::new()
    }

    /// The width of one input image.
    fn input_width(&self) -> Option<usize> {
        Some(self.in_width())
    }

    /// [`InferenceLayer::AdaptiveAvgPool2d`].
    fn inference(&self, _params: &ParamValues<f32>) -> Option<InferenceLayer> {
        Some(InferenceLayer::AdaptiveAvgPool2d {
            channels: self.channels,
            input: self.input,
            output: self.output,
        })
    }
}

/// Tests for the pooling layers: a small CNN through the tape and through inference.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Conv2d, InferenceModel, Linear, Relu, Sequential};
    use cavitation_core::{Conv2dGeom, Graph, ParamStore, Rng, Tensor};

    /// A CNN with every pooling layer gives the tape's bits through `infer_batch` and
    /// `infer`: the inference steps run the tape's own loops.
    #[test]
    fn pooling_inference_matches_tape() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(4);
        let c = Conv2dGeom::with_options(1, 4, (9, 9), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
        let m = Pool2dGeom::with_options(4, (9, 9), (3, 3), (2, 2), (1, 1), (1, 1), true).unwrap();
        let (mh, mw) = m.out_size();
        let a =
            Pool2dGeom::with_options(4, (mh, mw), (2, 2), (1, 1), (1, 1), (1, 1), false).unwrap();
        let (ah, aw) = a.out_size();
        let net = Sequential::new()
            .push(Conv2d::new(&mut params, c, &mut rng))
            .push(Relu)
            .push(MaxPool2d::new(m))
            .push(AvgPool2d::with_options(a, false, None).unwrap())
            .push(AdaptiveAvgPool2d::new(4, (ah, aw), (2, 2)).unwrap())
            .push(Linear::new(&mut params, 16, 3, &mut rng));
        let rows = 4;
        let x: Vec<f32> = (0..rows * 81)
            .map(|_| rng.range(-1.0, 1.0) as f32)
            .collect();
        let x = Tensor::from_vec(x, &[rows, 81]).unwrap();
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let xv = tape.input(x.view());
        let mut v = xv;
        for i in 0..net.len() - 1 {
            v = net.layer(i).forward(&mut tape, v);
        }
        let want = tape.value(v).data().to_vec();
        drop(tape);
        let mut model = InferenceModel::compile(&net, &params).unwrap();
        let mut seen = Vec::new();
        model.infer_batch_observed(x.view(), &mut |layer, out| {
            if layer == 4 {
                seen = out.data().to_vec();
            }
        });
        assert_eq!(seen, want);
        assert_eq!(model.infer_batch(x.view()).data().len(), rows * 3);
    }

    /// A user-made adaptive pooling form whose input size overflows is refused when the
    /// model compiles, naming the layer and the sizes, instead of wrapping.
    #[test]
    #[should_panic(expected = "every size must be at least 1 and the input and output must fit")]
    fn compile_refuses_an_overflowing_adaptive_form() {
        /// An adaptive pooling of `usize::MAX` channels, built by hand.
        struct Huge;
        impl Layer<f32> for Huge {
            /// Returns `"huge"`.
            fn name(&self) -> &str {
                "huge"
            }
            /// One value per image.
            fn output_width(&self, _: usize) -> usize {
                1
            }
            /// Passes the input on; only the inference form is under test.
            fn forward(&self, _: &mut Tape<'_, f32>, x: Var) -> Var {
                x
            }
            /// No parameters.
            fn parameters(&self) -> Vec<(&'static str, ParamId)> {
                Vec::new()
            }
            /// The overflowing adaptive pooling form.
            fn inference(&self, _: &ParamValues<f32>) -> Option<InferenceLayer> {
                Some(InferenceLayer::AdaptiveAvgPool2d {
                    channels: usize::MAX,
                    input: (2, 2),
                    output: (1, 1),
                })
            }
        }
        let mut params = ParamStore::new();
        let net = Sequential::new()
            .push(Linear::new(&mut params, 2, 4, &mut Rng::seed(0)))
            .push(Huge);
        let _ = InferenceModel::compile(&net, &params);
    }
}
