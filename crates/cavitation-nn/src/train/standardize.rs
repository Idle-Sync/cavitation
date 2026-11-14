//! [`Standardizer`]: per-column mean and standard deviation, applied to inputs or value
//! targets, and folded into the first or last [`Linear`](crate::Linear) of a network so
//! that the network carries its own standardisation.

use cavitation_core::{Error, ParamStore, Tensor, TensorView};

use super::{Dataset, Targets};
use crate::Sequential;

/// A per-column standardisation `(x − mean) / scale`: the statistics of a training set, which
/// [`transform`](Self::transform)s inputs (or value targets) before training and
/// [`inverse_transform`](Self::inverse_transform)s them back.
///
/// [`fold_into_input`](Self::fold_into_input) writes the standardisation into the first
/// `Linear`'s weight and bias, so the trained network takes raw inputs and the statistics need
/// not travel beside the weights; [`fold_into_output`](Self::fold_into_output) does the same
/// for a network trained on standardised targets, so that it answers in the targets' own units.
///
/// The statistics are `f64`; every tensor is `f32`. `fit` uses the population standard
/// deviation (divisor `n`), as scikit-learn's `StandardScaler` does, and a column whose values
/// are all equal gets scale 1 (so it becomes all zeros instead of dividing by zero).
///
/// # Examples
///
/// ```
/// use cavitation_core::Tensor;
/// use cavitation_nn::Standardizer;
/// let x = Tensor::from_vec(vec![1.0f32, 10.0, 3.0, 10.0, 5.0, 40.0], &[3, 2]).unwrap();
/// let s = Standardizer::fit(x.view());
/// assert_eq!(s.mean(), &[3.0, 20.0]);
/// let mut z = x.clone();
/// s.transform(&mut z);
/// assert!((z.data()[0] + 1.224_744_9).abs() < 1e-6);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Standardizer {
    /// The mean of each column.
    mean: Vec<f64>,
    /// The scale of each column: its standard deviation, or 1 for a constant column.
    scale: Vec<f64>,
}

impl Standardizer {
    /// Computes each column's mean and population standard deviation (divisor `n`, in `f64`)
    /// from the rows of `x`. A column whose values are all equal gets scale 1.
    ///
    /// A NaN or infinite value in a column makes that column's mean or scale non-finite (NaN
    /// or infinite; the statistics are not checked), so [`transform`](Self::transform) then
    /// gives NaN or infinite values for the column and a fold refuses with
    /// `Error::InvalidData`; check the data first.
    ///
    /// # Panics
    ///
    /// Panics if `x` is not a matrix, or has no rows or no columns: there is nothing to
    /// compute statistics from.
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_nn::Standardizer;
    /// let x = Tensor::from_vec(vec![2.0f32, 7.0, 4.0, 7.0], &[2, 2]).unwrap();
    /// let s = Standardizer::fit(x.view());
    /// assert_eq!((s.mean(), s.scale()), (&[3.0, 7.0][..], &[1.0, 1.0][..]));
    /// ```
    pub fn fit(x: TensorView<'_, f32>) -> Standardizer {
        let (rows, width) = match x.shape().dims() {
            &[rows, width] => (rows, width),
            dims => panic!("Standardizer::fit: x must have shape [rows, columns], found {dims:?}"),
        };
        assert!(
            rows > 0 && width > 0,
            "Standardizer::fit: x has no rows or no columns ({rows} rows, {width} columns)"
        );
        let data = x.data();
        let n = rows as f64;
        let mut mean = Vec::with_capacity(width);
        let mut scale = Vec::with_capacity(width);
        for j in 0..width {
            let column = || data[j..].iter().step_by(width).map(|&v| f64::from(v));
            let m = column().sum::<f64>() / n;
            let variance = column().map(|v| (v - m) * (v - m)).sum::<f64>() / n;
            mean.push(m);
            scale.push(if variance == 0.0 {
                1.0
            } else {
                variance.sqrt()
            });
        }
        Standardizer { mean, scale }
    }

    /// A standardiser from statistics you already have, for example those of another
    /// library or another run.
    ///
    /// # Panics
    ///
    /// Panics if `mean` and `scale` have different lengths or are empty, or if a scale is
    /// not finite and positive (it is a divisor).
    ///
    /// ```
    /// use cavitation_nn::Standardizer;
    /// let s = Standardizer::from_parts(vec![1.0, 2.0], vec![0.5, 4.0]);
    /// assert_eq!(s.scale(), &[0.5, 4.0]);
    /// ```
    pub fn from_parts(mean: Vec<f64>, scale: Vec<f64>) -> Standardizer {
        assert!(
            mean.len() == scale.len() && !mean.is_empty(),
            "Standardizer::from_parts: {} means and {} scales (need the same number, at least 1)",
            mean.len(),
            scale.len()
        );
        if let Some(j) = scale.iter().position(|s| !(s.is_finite() && *s > 0.0)) {
            panic!(
                "Standardizer::from_parts: scale[{j}] is {}, but a scale must be finite and positive",
                scale[j]
            );
        }
        Standardizer { mean, scale }
    }

    /// The mean of each column.
    ///
    /// ```
    /// use cavitation_nn::Standardizer;
    /// let s = Standardizer::from_parts(vec![1.0], vec![2.0]);
    /// assert_eq!(s.mean(), &[1.0]);
    /// ```
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// The scale of each column: its standard deviation, or 1 for a constant column.
    ///
    /// ```
    /// use cavitation_nn::Standardizer;
    /// let s = Standardizer::from_parts(vec![1.0], vec![2.0]);
    /// assert_eq!(s.scale(), &[2.0]);
    /// ```
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }

    /// Replaces each value of `x` by `(value − mean) / scale` for its column. Each value is
    /// computed in `f64` and rounded once to `f32`.
    ///
    /// A NaN stays NaN and an infinite value stays infinite (or becomes NaN where the
    /// subtraction is infinity minus infinity); nothing is checked.
    ///
    /// # Panics
    ///
    /// Panics, naming both widths, if `x` is not a matrix with as many columns as the
    /// standardiser.
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_nn::Standardizer;
    /// let s = Standardizer::from_parts(vec![10.0], vec![2.0]);
    /// let mut x = Tensor::from_vec(vec![8.0f32, 12.0], &[2, 1]).unwrap();
    /// s.transform(&mut x);
    /// assert_eq!(x.data(), &[-1.0, 1.0]);
    /// ```
    pub fn transform(&self, x: &mut Tensor<f32>) {
        self.apply(x, "transform", |v, m, s| (v - m) / s);
    }

    /// Undoes [`transform`](Self::transform): replaces each value by `value · scale + mean`,
    /// computed in `f64` and rounded once to `f32`.
    ///
    /// The round trip is not exact: the standardised value is rounded to `f32` and the error
    /// that rounding leaves in the result is about `|x − mean|` times 6e-8, plus the final
    /// rounding. That is within 2 ulp of the original `x` when `|x − mean|` is no larger than
    /// `|x|`; for a value much smaller than its column's mean, the error is relative to the
    /// mean, not to the value (0.001 with mean 1000 comes back as 0.0009765625).
    ///
    /// # Panics
    ///
    /// Panics, naming both widths, if `x` is not a matrix with as many columns as the
    /// standardiser.
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_nn::Standardizer;
    /// let s = Standardizer::from_parts(vec![10.0], vec![2.0]);
    /// let mut z = Tensor::from_vec(vec![-1.0f32, 1.0], &[2, 1]).unwrap();
    /// s.inverse_transform(&mut z);
    /// assert_eq!(z.data(), &[8.0, 12.0]);
    /// ```
    pub fn inverse_transform(&self, x: &mut Tensor<f32>) {
        self.apply(x, "inverse_transform", |v, m, s| v * s + m);
    }

    /// Standardises `data.x` in place, as [`transform`](Self::transform).
    ///
    /// # Panics
    ///
    /// As [`transform`](Self::transform).
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_nn::{Dataset, Standardizer, Targets};
    /// let s = Standardizer::from_parts(vec![1.0], vec![2.0]);
    /// let mut data = Dataset {
    ///     x: Tensor::from_vec(vec![3.0f32, 5.0], &[2, 1]).unwrap(),
    ///     targets: Targets::Classes(vec![0, 1]),
    /// };
    /// s.transform_inputs(&mut data);
    /// assert_eq!(data.x.data(), &[1.0, 2.0]);
    /// ```
    pub fn transform_inputs(&self, data: &mut Dataset) {
        self.transform(&mut data.x);
    }

    /// Standardises the value targets of `data` in place, as [`transform`](Self::transform),
    /// for a network trained on standardised targets; undo it on the network's answers with
    /// [`inverse_transform`](Self::inverse_transform), or fold it into the network with
    /// [`fold_into_output`](Self::fold_into_output).
    ///
    /// # Errors
    ///
    /// `Error::InvalidData` if the targets are class labels, which have no scale to remove;
    /// `data` is unchanged then.
    ///
    /// # Panics
    ///
    /// As [`transform`](Self::transform), on the targets' width.
    ///
    /// ```
    /// use cavitation_core::Tensor;
    /// use cavitation_nn::{Dataset, Standardizer, Targets};
    /// let s = Standardizer::from_parts(vec![10.0], vec![5.0]);
    /// let mut data = Dataset {
    ///     x: Tensor::zeros(&[2, 1]),
    ///     targets: Targets::Values(Tensor::from_vec(vec![5.0f32, 15.0], &[2, 1]).unwrap()),
    /// };
    /// s.transform_targets(&mut data).unwrap();
    /// let standardised = Tensor::from_vec(vec![-1.0f32, 1.0], &[2, 1]).unwrap();
    /// assert_eq!(data.targets, Targets::Values(standardised));
    ///
    /// let mut classes = Dataset { x: Tensor::zeros(&[1, 1]), targets: Targets::Classes(vec![0]) };
    /// assert!(s.transform_targets(&mut classes).is_err());
    /// ```
    pub fn transform_targets(&self, data: &mut Dataset) -> Result<(), Error> {
        match &mut data.targets {
            Targets::Values(t) => {
                self.transform(t);
                Ok(())
            }
            Targets::Classes(_) => Err(Error::InvalidData(
                "Standardizer::transform_targets: the targets are class labels, which cannot be \
                 standardised"
                    .to_string(),
            )),
        }
    }

    /// Folds the standardisation into the first layer, which must be a `Linear` with as many
    /// inputs as the standardiser has columns, so that the network trained on standardised
    /// inputs gives the same answers on raw ones: with `z = (x − μ) / σ`, the weight becomes
    /// `W' = W / σ` (each column divided by its scale) and the bias `b' = b − W' μ`, because
    /// `W z + b = W' x + b'`.
    ///
    /// Everything is computed in `f64` from the `f32` weights and rounded once to `f32` when
    /// written, so the folded network agrees with standardising first to float rounding
    /// (about `(inputs + 1) · 1.2e-7` of the sum of the absolute terms), not bit for bit.
    /// Fold once: folding the same parameters again divides by the scale again.
    ///
    /// The change is made in `params`, so every clone of `model` used with that store and every
    /// layer that shares these parameters (a `Linear` pushed twice) sees it, and an
    /// [`InferenceModel`](crate::InferenceModel) or [`Evaluator`](crate::Evaluator) built
    /// before the fold keeps the old weights until it is compiled again or refreshed. Fold
    /// the store you will ship, after training, then compile.
    ///
    /// A mask on layer 0 acts after the affine map, which the fold preserves, so it does not
    /// matter here.
    ///
    /// # Errors
    ///
    /// `Error::InvalidData`, and `params` unchanged, if the model has no layers, its layer 0
    /// is not a `Linear` (the message names its kind), the `Linear` has no bias (the fold
    /// writes `b − W' μ` into one; build the layer with a bias), the `Linear`'s input width
    /// differs from the standardiser's, its parameters in `params` do not have the shapes
    /// the `Linear` declares, or a folded value is not finite (for example after a fit on
    /// non-finite data).
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{Linear, Sequential, Standardizer};
    /// let mut params = ParamStore::new();
    /// let weight = Tensor::from_vec(vec![2.0f32], &[1, 1]).unwrap();
    /// let bias = Tensor::from_vec(vec![1.0f32], &[1]).unwrap();
    /// let layer = Linear::from_values(&mut params, weight, bias).unwrap();
    /// let (w, b) = (layer.weight(), layer.bias().unwrap());
    /// let net = Sequential::new().push(layer);
    /// // z = (x - 10) / 4, so 2z + 1 = 0.5x - 4.
    /// Standardizer::from_parts(vec![10.0], vec![4.0]).fold_into_input(&net, &mut params).unwrap();
    /// assert_eq!(params.value(w).data(), &[0.5]);
    /// assert_eq!(params.value(b).data(), &[-4.0]);
    /// ```
    pub fn fold_into_input(
        &self,
        model: &Sequential,
        params: &mut ParamStore,
    ) -> Result<(), Error> {
        let first = Self::linear_at(model, 0, "fold_into_input")?;
        Self::needs_bias(first, 0, "fold_into_input")?;
        let (outputs, width) = (first.outputs(), self.width());
        if first.inputs() != width {
            return Err(Error::InvalidData(format!(
                "Standardizer::fold_into_input: layer 0 takes {} inputs but the standardiser \
                 has {width} columns",
                first.inputs()
            )));
        }
        let (weight, bias) = Self::parameters(first, params, "fold_into_input")?;
        let mut new_weight = Vec::with_capacity(outputs * width);
        let mut new_bias = Vec::with_capacity(outputs);
        for (row, &b) in weight.chunks(width).zip(&bias) {
            let mut shift = 0.0f64;
            for ((&w, &m), &s) in row.iter().zip(&self.mean).zip(&self.scale) {
                let scaled = f64::from(w) / s;
                shift += scaled * m;
                new_weight.push(scaled as f32);
            }
            new_bias.push((f64::from(b) - shift) as f32);
        }
        Self::write(first, params, new_weight, new_bias, "fold_into_input")
    }

    /// Folds the inverse standardisation into the last layer, which must be a `Linear` with
    /// as many outputs as the standardiser has columns, so that a network trained on
    /// standardised targets answers in the targets' own units: the weight becomes `W' = σ W`
    /// (each row multiplied by its scale) and the bias `b' = σ b + μ`, because
    /// `σ (W h + b) + μ = W' h + b'`.
    ///
    /// The rounding is as for [`fold_into_input`](Self::fold_into_input), and so are the
    /// notes on fold once and on shared parameters and compiled models. The last layer must
    /// be the `Linear` itself: an activation after it (a softmax, say) cannot be folded
    /// through. Nor can a mask with a neuron off: the mask sets that output to 0 after the
    /// bias, which is `μ` in the targets' units before the fold and 0 after it, so the fold
    /// refuses. Clear the mask, or fold before masking.
    ///
    /// # Errors
    ///
    /// `Error::InvalidData`, and `params` unchanged, as for
    /// [`fold_into_input`](Self::fold_into_input), with the last layer in place of layer 0 and
    /// its output width in place of its input width (and `σ b + μ` the bias it needs); also
    /// if the last layer has a mask with any neuron off.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{Linear, Sequential, Standardizer};
    /// let mut params = ParamStore::new();
    /// let weight = Tensor::from_vec(vec![2.0f32], &[1, 1]).unwrap();
    /// let bias = Tensor::from_vec(vec![1.0f32], &[1]).unwrap();
    /// let layer = Linear::from_values(&mut params, weight, bias).unwrap();
    /// let (w, b) = (layer.weight(), layer.bias().unwrap());
    /// let net = Sequential::new().push(layer);
    /// // y = 4 (2h + 1) + 10 = 8h + 14.
    /// let s = Standardizer::from_parts(vec![10.0], vec![4.0]);
    /// s.fold_into_output(&net, &mut params).unwrap();
    /// assert_eq!(params.value(w).data(), &[8.0]);
    /// assert_eq!(params.value(b).data(), &[14.0]);
    /// ```
    pub fn fold_into_output(
        &self,
        model: &Sequential,
        params: &mut ParamStore,
    ) -> Result<(), Error> {
        let last = Self::linear_at(model, model.len().wrapping_sub(1), "fold_into_output")?;
        let (outputs, inputs) = (last.outputs(), last.inputs());
        if outputs != self.width() {
            return Err(Error::InvalidData(format!(
                "Standardizer::fold_into_output: the last layer gives {outputs} outputs but the \
                 standardiser has {} columns",
                self.width()
            )));
        }
        let at = model.len() - 1;
        Self::needs_bias(last, at, "fold_into_output")?;
        if let Some(mask) = model.mask(at).filter(|m| m.count_on() < m.width()) {
            return Err(Error::InvalidData(format!(
                "Standardizer::fold_into_output: layer {at} has {} of {} neurons switched off; \
                 a masked output is 0 after the fold instead of the mean, so clear the mask or \
                 fold before masking",
                mask.width() - mask.count_on(),
                mask.width()
            )));
        }
        let (weight, bias) = Self::parameters(last, params, "fold_into_output")?;
        let mut new_weight = Vec::with_capacity(outputs * inputs);
        for (row, &s) in weight.chunks(inputs).zip(&self.scale) {
            new_weight.extend(row.iter().map(|&w| (s * f64::from(w)) as f32));
        }
        let new_bias = bias
            .iter()
            .zip(self.scale.iter().zip(&self.mean))
            .map(|(&b, (&s, &m))| (s * f64::from(b) + m) as f32)
            .collect();
        Self::write(last, params, new_weight, new_bias, "fold_into_output")
    }

    /// The number of columns.
    fn width(&self) -> usize {
        self.mean.len()
    }

    /// Replaces each value of `x` by `f(value, mean, scale)` for its column, computed in `f64`
    /// and rounded once; `what` names the caller in the panic message.
    ///
    /// # Panics
    ///
    /// Panics, naming both widths, if `x` is not a matrix with `width()` columns.
    fn apply(&self, x: &mut Tensor<f32>, what: &str, f: impl Fn(f64, f64, f64) -> f64) {
        let width = self.width();
        match x.shape().dims() {
            &[_, columns] if columns == width => {}
            &[_, columns] => panic!(
                "Standardizer::{what}: x has {columns} columns but the standardiser has {width}"
            ),
            dims => {
                panic!("Standardizer::{what}: x must have shape [rows, columns], found {dims:?}")
            }
        }
        for row in x.data_mut().chunks_mut(width) {
            for ((v, &m), &s) in row.iter_mut().zip(&self.mean).zip(&self.scale) {
                *v = f(f64::from(*v), m, s) as f32;
            }
        }
    }

    /// The `Linear` at layer `i` of `model`, or an `InvalidData` error naming the layer's kind
    /// (or that there are no layers); `what` names the caller.
    fn linear_at<'a>(
        model: &'a Sequential,
        i: usize,
        what: &str,
    ) -> Result<&'a crate::Linear, Error> {
        if model.is_empty() {
            return Err(Error::InvalidData(format!(
                "Standardizer::{what}: the model has no layers"
            )));
        }
        model.linear(i).ok_or_else(|| {
            Error::InvalidData(format!(
                "Standardizer::{what}: layer {i} is a `{}` layer, not a `linear`",
                model.layer(i).name()
            ))
        })
    }

    /// `InvalidData` naming layer `at` and the caller `what` if `linear` has no bias, which
    /// both folds must write.
    fn needs_bias(linear: &crate::Linear, at: usize, what: &str) -> Result<(), Error> {
        match linear.bias() {
            Some(_) => Ok(()),
            None => Err(Error::InvalidData(format!(
                "Standardizer::{what}: the linear layer {at} has no bias, and the fold must \
                 write the shifted bias into one; build that layer with a bias (Linear::new or \
                 Linear::with_init) to fold the standardiser into it"
            ))),
        }
    }

    /// A copy of `linear`'s weight and bias from `params`, or `InvalidData` if their shapes
    /// are not `[outputs, inputs]` and `[outputs]`. The caller has checked with
    /// [`needs_bias`](Self::needs_bias) that the layer has a bias.
    fn parameters(
        linear: &crate::Linear,
        params: &ParamStore,
        what: &str,
    ) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let bias = linear.bias().expect("the fold checked the bias");
        let (w, b) = (params.value(linear.weight()), params.value(bias));
        if w.shape().dims() != [linear.outputs(), linear.inputs()]
            || b.shape().dims() != [linear.outputs()]
        {
            return Err(Error::InvalidData(format!(
                "Standardizer::{what}: the parameters are {:?} and {:?}, not [{}, {}] and [{}]",
                w.shape().dims(),
                b.shape().dims(),
                linear.outputs(),
                linear.inputs(),
                linear.outputs()
            )));
        }
        Ok((w.data().to_vec(), b.data().to_vec()))
    }

    /// Writes the folded `weight` and `bias` of `linear` into `params`, or returns
    /// `InvalidData` without writing if any value is not finite.
    fn write(
        linear: &crate::Linear,
        params: &mut ParamStore,
        weight: Vec<f32>,
        bias: Vec<f32>,
        what: &str,
    ) -> Result<(), Error> {
        if !weight.iter().chain(&bias).all(|v| v.is_finite()) {
            return Err(Error::InvalidData(format!(
                "Standardizer::{what}: the folded weights are not finite (a scale or mean is NaN \
                 or too extreme for f32)"
            )));
        }
        params
            .value_mut(linear.weight())
            .data_mut()
            .copy_from_slice(&weight);
        params
            .value_mut(linear.bias().expect("parameters checked the bias"))
            .data_mut()
            .copy_from_slice(&bias);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! Tests for [`Standardizer`]: statistics, the transforms, and both folds.

    use super::*;
    use crate::{InferenceModel, Linear, Relu};
    use cavitation_core::{Graph, Rng};

    /// The bound a float computation of `k` accumulated terms with absolute sum `sum` must
    /// meet: the GEMM bound `2 · (k + 2) · 1.19e-7 · sum + 1e-30`.
    fn gemm_bound(k: usize, sum: f64) -> f64 {
        2.0 * (k as f64 + 2.0) * 1.19e-7 * sum + 1e-30
    }

    /// `rows × width` values where column `j` has mean about `1000 · (j + 1)` and spread
    /// about `5 · (j + 1)`.
    fn raw_inputs(rows: usize, width: usize, seed: u64) -> Tensor<f32> {
        let mut rng = Rng::seed(seed);
        let data = (0..rows * width)
            .map(|i| {
                let j = (i % width + 1) as f64;
                (1000.0 * j + 5.0 * j * rng.normal()) as f32
            })
            .collect();
        Tensor::from_vec(data, &[rows, width]).unwrap()
    }

    /// Runs `net` on `x`.
    fn run(net: &Sequential, params: &ParamStore, x: &Tensor<f32>) -> Vec<f32> {
        let mut graph = Graph::new();
        let mut tape = graph.tape(&params.values);
        let input = tape.input(x.view());
        let y = net.forward(&mut tape, input);
        tape.value(y).data().to_vec()
    }

    /// `fit` divides by `n`, not `n - 1`.
    #[test]
    fn fit_uses_population_deviation() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4, 1]).unwrap();
        let s = Standardizer::fit(x.view());
        assert_eq!(s.mean(), &[2.5]);
        assert!(
            (s.scale()[0] - 1.25f64.sqrt()).abs() < 1e-15,
            "{:?}",
            s.scale()
        );
    }

    /// A column of equal values, including one that is not exact in binary, has scale 1 and
    /// standardises to zeros.
    #[test]
    fn constant_column_has_scale_one() {
        let x = Tensor::from_vec(vec![0.1f32, 1.0, 0.1, 2.0, 0.1, 6.0], &[3, 2]).unwrap();
        let s = Standardizer::fit(x.view());
        assert_eq!(s.scale()[0], 1.0);
        assert_eq!(s.mean()[0], f64::from(0.1f32));
        let mut z = x.clone();
        s.transform(&mut z);
        assert_eq!([z.data()[0], z.data()[2], z.data()[4]], [0.0; 3]);
    }

    /// Fitting nothing is a shape error.
    #[test]
    #[should_panic(expected = "no rows")]
    fn fit_rejects_no_rows() {
        let _ = Standardizer::fit(Tensor::<f32>::zeros(&[0, 3]).view());
    }

    /// A width mismatch panics naming both widths.
    #[test]
    #[should_panic(expected = "3 columns")]
    fn transform_names_both_widths() {
        let s = Standardizer::from_parts(vec![0.0; 2], vec![1.0; 2]);
        s.transform(&mut Tensor::zeros(&[4, 3]));
    }

    /// A zero or negative scale is refused at construction.
    #[test]
    #[should_panic(expected = "scale")]
    fn from_parts_rejects_zero_scale() {
        let _ = Standardizer::from_parts(vec![0.0], vec![0.0]);
    }

    /// Class targets cannot be standardised: an error, and the data is untouched.
    #[test]
    fn transform_targets_refuses_classes() {
        let s = Standardizer::from_parts(vec![0.0], vec![1.0]);
        let mut data = Dataset {
            x: Tensor::zeros(&[2, 1]),
            targets: Targets::Classes(vec![0, 1]),
        };
        let before = data.clone();
        assert!(matches!(
            s.transform_targets(&mut data),
            Err(Error::InvalidData(_))
        ));
        assert_eq!(data, before);
    }

    /// `transform` then `inverse_transform` returns every value within 2 ulp.
    #[test]
    fn transform_round_trip() {
        let x = raw_inputs(500, 4, 1);
        let s = Standardizer::fit(x.view());
        let mut z = x.clone();
        s.transform(&mut z);
        s.inverse_transform(&mut z);
        for (i, (&back, &orig)) in z.data().iter().zip(x.data()).enumerate() {
            let ulp = f32::from_bits(orig.abs().to_bits() + 1) - orig.abs();
            assert!(
                (back - orig).abs() <= 2.0 * ulp,
                "element {i}: {orig} came back {back}"
            );
        }
    }

    /// The folded network on raw inputs agrees with the original on standardised inputs
    /// within the GEMM bound, with `k` = the inputs of the dot product plus those of the
    /// bias shift, and the terms `|W_j x_j| / σ_j`, `|W_j μ_j| / σ_j` and `|b|`.
    #[test]
    fn fold_input_matches_standardised_inputs() {
        let (rows, width, hidden) = (64, 6, 5);
        let x = raw_inputs(rows, width, 2);
        let s = Standardizer::fit(x.view());
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(3);
        let net = Sequential::new()
            .push(Linear::new(&mut params, width, hidden, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, hidden, 1, &mut rng));
        let first = net.linear(0).unwrap().clone();
        // Give the bias a value so that it takes part.
        params.value_mut(first.bias().unwrap()).data_mut().fill(0.3);
        let weight = params.value(first.weight()).data().to_vec();
        let bias = params.value(first.bias().unwrap()).data().to_vec();

        let mut z = x.clone();
        s.transform(&mut z);
        let expected = run(&net, &params, &z);

        s.fold_into_input(&net, &mut params).unwrap();
        let got = run(&net, &params, &x);

        // Bound the first layer's outputs, then check the network's: the second layer is
        // unchanged, so check the first layer alone with a one-layer twin.
        let twin = Sequential::new().push(first.clone());
        let folded = run(&twin, &params, &x);
        let mut original = params.clone();
        original
            .value_mut(first.weight())
            .data_mut()
            .copy_from_slice(&weight);
        original
            .value_mut(first.bias().unwrap())
            .data_mut()
            .copy_from_slice(&bias);
        let reference = run(&twin, &original, &z);
        for r in 0..rows {
            for o in 0..hidden {
                let mut sum = f64::from(bias[o]).abs();
                for j in 0..width {
                    let w = f64::from(weight[o * width + j]) / s.scale()[j];
                    sum += w.abs() * (f64::from(x.data()[r * width + j]).abs() + s.mean()[j].abs());
                }
                let (a, b) = (folded[r * hidden + o], reference[r * hidden + o]);
                let limit = gemm_bound(2 * width + 1, sum);
                assert!(
                    f64::from(a - b).abs() <= limit,
                    "row {r}, output {o}: {a} against {b}, limit {limit}"
                );
            }
        }
        // And the network's final answer follows within a loose multiple of the same.
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() <= 1e-3 * e.abs().max(1.0), "{g} against {e}");
        }
    }

    /// The network folded on its output answers what `inverse_transform` makes of the
    /// original's answers, within the GEMM bound with the terms `|σ W_j h_j|`, `|σ b|`
    /// and `|μ|`.
    #[test]
    fn fold_output_matches_inverse_transform() {
        let (rows, hidden, outputs) = (64, 7, 3);
        let mut rng = Rng::seed(4);
        let mut params = ParamStore::new();
        let net = Sequential::new()
            .push(Linear::new(&mut params, 2, hidden, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, hidden, outputs, &mut rng));
        let last = net.linear(2).unwrap().clone();
        params.value_mut(last.bias().unwrap()).data_mut().fill(-0.2);
        let s = Standardizer::from_parts(vec![500.0, -40.0, 3.0], vec![25.0, 0.5, 9.0]);
        let x = Tensor::from_vec(
            (0..rows * 2).map(|_| rng.normal() as f32).collect(),
            &[rows, 2],
        )
        .unwrap();

        let prefix = Sequential::new()
            .push(net.linear(0).unwrap().clone())
            .push(Relu);
        let hidden_out = run(&prefix, &params, &x);
        let weight = params.value(last.weight()).data().to_vec();
        let bias = params.value(last.bias().unwrap()).data().to_vec();
        let mut expected = Tensor::from_vec(run(&net, &params, &x), &[rows, outputs]).unwrap();
        s.inverse_transform(&mut expected);

        s.fold_into_output(&net, &mut params).unwrap();
        let got = run(&net, &params, &x);
        for r in 0..rows {
            for o in 0..outputs {
                let mut sum = (s.scale()[o] * f64::from(bias[o])).abs() + s.mean()[o].abs();
                for j in 0..hidden {
                    sum += (s.scale()[o]
                        * f64::from(weight[o * hidden + j])
                        * f64::from(hidden_out[r * hidden + j]))
                    .abs();
                }
                let (a, b) = (got[r * outputs + o], expected.data()[r * outputs + o]);
                let limit = gemm_bound(hidden + 1, sum);
                assert!(
                    f64::from(a - b).abs() <= limit,
                    "row {r}, output {o}: {a} against {b}, limit {limit}"
                );
            }
        }
    }

    /// Runs `model` on every row of `x` through `infer` and through `infer_batch`, and
    /// returns both answers, row after row.
    fn compiled(model: &mut InferenceModel, x: &Tensor<f32>) -> (Vec<f32>, Vec<f32>) {
        let width = x.shape().dims()[1];
        let one = x
            .data()
            .chunks(width)
            .flat_map(|row| model.infer(row).to_vec())
            .collect();
        let batch = model.infer_batch(x.view()).data().to_vec();
        (one, batch)
    }

    /// The largest `|a − b| / max(|b|, 1)` over two answers.
    fn worst(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
            .fold(0.0, f32::max)
    }

    /// Compiled for inference, a network folded on its input answers raw inputs as the
    /// unfolded network answers standardised ones, and a network folded on its output
    /// answers as `inverse_transform` makes of the unfolded one's answers, through both
    /// `infer` and `infer_batch`, within float rounding. The input fold's bound is looser
    /// because the folded first layer cancels raw inputs of about 1000 against the folded
    /// mean (measured worst relative difference 3.0e-5 against a bound of 3e-4); the output
    /// fold adds none (5.0e-7 against 1e-5); the two together measured 1.2e-4 against 1e-3.
    #[test]
    fn folds_agree_after_compiling() {
        let (rows, width, hidden, outputs) = (48, 5, 9, 3);
        let x = raw_inputs(rows, width, 6);
        let s_in = Standardizer::fit(x.view());
        let s_out = Standardizer::from_parts(vec![500.0, -40.0, 3.0], vec![25.0, 0.5, 9.0]);
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(7);
        let net = Sequential::new()
            .push(Linear::new(&mut params, width, hidden, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, hidden, outputs, &mut rng));
        let mut z = x.clone();
        s_in.transform(&mut z);

        let mut unfolded = InferenceModel::compile(&net, &params).unwrap();
        let (one, batch) = compiled(&mut unfolded, &z);
        let mut expected = Tensor::from_vec(batch.clone(), &[rows, outputs]).unwrap();
        s_out.inverse_transform(&mut expected);

        s_in.fold_into_input(&net, &mut params).unwrap();
        let mut folded = InferenceModel::compile(&net, &params).unwrap();
        let (one_in, batch_in) = compiled(&mut folded, &x);
        assert!(worst(&one_in, &one) <= 3e-4, "{}", worst(&one_in, &one));
        assert!(
            worst(&batch_in, &batch) <= 3e-4,
            "{}",
            worst(&batch_in, &batch)
        );

        s_out.fold_into_output(&net, &mut params).unwrap();
        let mut both = InferenceModel::compile(&net, &params).unwrap();
        let (one_out, batch_out) = compiled(&mut both, &x);
        let mut input_folded = Tensor::from_vec(batch_in, &[rows, outputs]).unwrap();
        s_out.inverse_transform(&mut input_folded);
        let tight = 1e-5;
        assert!(
            worst(&batch_out, input_folded.data()) <= tight,
            "{}",
            worst(&batch_out, input_folded.data())
        );
        assert!(worst(&one_out, input_folded.data()) <= tight);
        assert!(worst(&batch_out, expected.data()) <= 1e-3);
    }

    /// A first layer that is not a `Linear` is an `InvalidData` error naming its kind, and
    /// nothing is written; the same for the last layer and for an empty model.
    #[test]
    fn fold_needs_linear_first() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(5);
        let net = Sequential::new()
            .push(Relu)
            .push(Linear::new(&mut params, 2, 2, &mut rng))
            .push(Relu);
        let before = params.clone();
        let s = Standardizer::from_parts(vec![0.0; 2], vec![1.0; 2]);
        let Err(Error::InvalidData(msg)) = s.fold_into_input(&net, &mut params) else {
            panic!("expected InvalidData");
        };
        assert!(msg.contains("layer 0") && msg.contains("relu"), "{msg}");
        let Err(Error::InvalidData(msg)) = s.fold_into_output(&net, &mut params) else {
            panic!("expected InvalidData");
        };
        assert!(msg.contains("layer 2") && msg.contains("relu"), "{msg}");
        assert_eq!(
            params.value(net.linear(1).unwrap().weight()),
            before.value(net.linear(1).unwrap().weight())
        );
        assert!(matches!(
            s.fold_into_input(&Sequential::new(), &mut params),
            Err(Error::InvalidData(_))
        ));
    }

    /// A mask with a neuron off on the output layer makes `fold_into_output` refuse and write
    /// nothing; an all-on mask, or a mask on another layer, does not.
    #[test]
    fn fold_output_refuses_masked_output() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(8);
        let mut net = Sequential::new()
            .push(Linear::new(&mut params, 2, 3, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, 3, 2, &mut rng));
        let s = Standardizer::from_parts(vec![500.0, -40.0], vec![25.0, 0.5]);
        let w = net.linear(2).unwrap().weight();
        let before = params.value(w).clone();
        net.set_neuron(2, 1, false);
        let Err(Error::InvalidData(msg)) = s.fold_into_output(&net, &mut params) else {
            panic!("expected InvalidData");
        };
        assert!(
            msg.contains("layer 2") && msg.contains("clear the mask"),
            "{msg}"
        );
        assert_eq!(params.value(w), &before);
        net.set_neuron(2, 1, true);
        net.set_mask(1, crate::NeuronMask::all_on(3));
        assert!(s.fold_into_output(&net, &mut params).is_ok());
    }

    /// A width that differs from the `Linear`'s, or a fit on non-finite data, is refused
    /// without writing anything.
    #[test]
    fn fold_refuses_bad_widths_and_non_finite() {
        let mut params = ParamStore::new();
        let net = Sequential::new().push(Linear::new(&mut params, 2, 2, &mut Rng::seed(6)));
        let before = params.clone();
        let wide = Standardizer::from_parts(vec![0.0; 3], vec![1.0; 3]);
        let Err(Error::InvalidData(msg)) = wide.fold_into_input(&net, &mut params) else {
            panic!("expected InvalidData");
        };
        assert!(msg.contains('2') && msg.contains('3'), "{msg}");
        assert!(wide.fold_into_output(&net, &mut params).is_err());
        let x = Tensor::from_vec(vec![1.0f32, f32::NAN, 2.0, 3.0], &[2, 2]).unwrap();
        let bad = Standardizer::fit(x.view());
        assert!(bad.mean()[1].is_nan());
        assert!(bad.fold_into_input(&net, &mut params).is_err());
        let w = net.linear(0).unwrap().weight();
        assert_eq!(params.value(w), before.value(w));
    }

    /// Both folds refuse a `Linear` without a bias with `InvalidData` that says so and
    /// names the layer, and write nothing: each fold must write a bias.
    #[test]
    fn folds_refuse_a_linear_without_a_bias() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(9);
        let net = Sequential::new()
            .push(Linear::without_bias(
                &mut params,
                2,
                3,
                crate::Init::XavierUniform,
                &mut rng,
            ))
            .push(Relu)
            .push(Linear::without_bias(
                &mut params,
                3,
                2,
                crate::Init::XavierUniform,
                &mut rng,
            ));
        let before = params.clone();
        let s = Standardizer::from_parts(vec![10.0, -3.0], vec![4.0, 0.5]);
        let Err(Error::InvalidData(msg)) = s.fold_into_input(&net, &mut params) else {
            panic!("expected InvalidData");
        };
        assert!(
            msg.contains("fold_into_input") && msg.contains("layer 0") && msg.contains("no bias"),
            "{msg}"
        );
        let Err(Error::InvalidData(msg)) = s.fold_into_output(&net, &mut params) else {
            panic!("expected InvalidData");
        };
        assert!(
            msg.contains("fold_into_output") && msg.contains("layer 2") && msg.contains("no bias"),
            "{msg}"
        );
        for (_, id) in net.parameter_names() {
            assert_eq!(params.value(id), before.value(id));
        }
    }
}
