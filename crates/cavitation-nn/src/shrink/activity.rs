//! Per-neuron activity statistics over a data set: [`activity`] runs the data through the
//! compiled model in batches and returns a [`LayerActivity`] for each chosen `Linear`.

use cavitation_core::{Error, ParamStore, TensorView};

use crate::{Dataset, InferenceModel, Sequential};

/// How the neurons of one `Linear` layer behaved over a data set, from [`activity`]: one
/// value per neuron in each list, read where the neuron's output is final (after the
/// activation that follows the `Linear`, and after any mask there).
///
/// ```
/// use cavitation_nn::LayerActivity;
/// let a = LayerActivity {
///     linear: 0,
///     layer: 1,
///     mean: vec![0.5],
///     mean_abs: vec![0.5],
///     firing: vec![0.25],
///     max: vec![2.0],
/// };
/// assert_eq!(a.firing[0], 0.25);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct LayerActivity {
    /// The index of the `Linear` layer whose neurons these are.
    pub linear: usize,
    /// The index of the layer whose output was read:
    /// [`Sequential::neuron_layer`] of `linear`.
    pub layer: usize,
    /// The mean of each neuron's output over the rows.
    pub mean: Vec<f64>,
    /// The mean of the absolute value of each neuron's output over the rows.
    pub mean_abs: Vec<f64>,
    /// The share of rows on which each neuron's output is above 0 (for a ReLU, the share on
    /// which the neuron fires); a NaN output does not count as above 0.
    pub firing: Vec<f64>,
    /// The largest output of each neuron over the rows, ignoring NaN; negative infinity for
    /// a neuron whose every output is NaN.
    pub max: Vec<f64>,
}

/// Measures the neurons of each `Linear` listed in `linears` over every row of `data`, run
/// `batch` rows at a time through `model` compiled with `params` (the batched inference
/// path, [`InferenceModel::infer_batch_observed`]). Each `Linear`'s neurons are read at
/// [`Sequential::neuron_layer`]: after the element-wise activation that follows it, masked
/// as the next layer sees them. The sums are kept in `f64`; the targets of `data` are not
/// used. The result has one [`LayerActivity`] per entry of `linears`, in that order.
///
/// # Errors
///
/// Returns [`Error::InvalidData`] if `batch` is 0, if `data.x` is not a matrix with at
/// least one row, if its width is not the model's input width, or if an entry of
/// `linears` is not the index of a `Linear` layer (the message names it), and the errors
/// [`InferenceModel::compile`] returns.
///
/// # Panics
///
/// Panics as [`InferenceModel::compile`] does when the layers' widths do not fit one
/// another.
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::{Dataset, Linear, Relu, Sequential, Targets, activity};
/// let mut params = ParamStore::new();
/// let layer = Linear::new(&mut params, 1, 2, &mut Rng::seed(0));
/// params.value_mut(layer.weight()).data_mut().copy_from_slice(&[1.0, -1.0]);
/// params.value_mut(layer.bias()).data_mut().copy_from_slice(&[0.0, 0.0]);
/// let net = Sequential::new().push(layer).push(Relu);
/// let data = Dataset {
///     x: Tensor::from_vec(vec![-1.0f32, 1.0, 2.0, 3.0], &[4, 1]).unwrap(),
///     targets: Targets::Classes(vec![0; 4]),
/// };
/// let stats = activity(&net, &params, &data, &[0], 3).unwrap();
/// // Neuron 0 is relu(x): 0, 1, 2, 3. Neuron 1 is relu(-x): 1, 0, 0, 0.
/// assert_eq!(stats[0].layer, 1);
/// assert_eq!(stats[0].mean, [1.5, 0.25]);
/// assert_eq!(stats[0].firing, [0.75, 0.25]);
/// assert_eq!(stats[0].max, [3.0, 1.0]);
/// ```
pub fn activity(
    model: &Sequential,
    params: &ParamStore,
    data: &Dataset,
    linears: &[usize],
    batch: usize,
) -> Result<Vec<LayerActivity>, Error> {
    if batch == 0 {
        return Err(Error::InvalidData(
            "activity: the batch size is 0".to_string(),
        ));
    }
    let &[rows, features] = data.x.shape().dims() else {
        return Err(Error::InvalidData(format!(
            "activity: the inputs must have shape [rows, features], found {:?}",
            data.x.shape().dims()
        )));
    };
    if rows == 0 {
        return Err(Error::InvalidData(
            "activity: the data set has no rows".to_string(),
        ));
    }
    let mut stats = Vec::with_capacity(linears.len());
    for &linear in linears {
        let Some(layer) = model.linear(linear) else {
            return Err(Error::InvalidData(format!(
                "activity: layer {linear} is not a Linear layer"
            )));
        };
        let width = layer.outputs();
        stats.push(LayerActivity {
            linear,
            layer: model.neuron_layer(linear),
            mean: vec![0.0; width],
            mean_abs: vec![0.0; width],
            firing: vec![0.0; width],
            max: vec![f64::NEG_INFINITY; width],
        });
    }
    let mut compiled = InferenceModel::compile(model, params)?;
    if features != compiled.input_width() {
        return Err(Error::InvalidData(format!(
            "activity: the inputs are [{rows}, {features}], but the model takes width {}",
            compiled.input_width()
        )));
    }
    let x = data.x.data();
    for start in (0..rows).step_by(batch) {
        let end = (start + batch).min(rows);
        let view = TensorView::from_slice(
            &x[start * features..end * features],
            &[end - start, features],
        )
        .expect("the rows hold (end - start) × features values");
        compiled.infer_batch_observed(view, &mut |layer, out| {
            for s in stats.iter_mut().filter(|s| s.layer == layer) {
                accumulate(s, out);
            }
        });
    }
    let n = rows as f64;
    for s in &mut stats {
        for v in s
            .mean
            .iter_mut()
            .chain(&mut s.mean_abs)
            .chain(&mut s.firing)
        {
            *v /= n;
        }
    }
    Ok(stats)
}

/// Adds the rows of `out` (one column per neuron of `s`) to the running sums in `s`: the
/// sum, the sum of absolute values and the count above 0, and the maximum so far.
fn accumulate(s: &mut LayerActivity, out: TensorView<'_, f32>) {
    let width = s.mean.len();
    if width == 0 {
        return;
    }
    for row in out.data().chunks_exact(width) {
        for (j, &v) in row.iter().enumerate() {
            let v = f64::from(v);
            s.mean[j] += v;
            s.mean_abs[j] += v.abs();
            if v > 0.0 {
                s.firing[j] += 1.0;
            }
            s.max[j] = s.max[j].max(v);
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tests for `activity`: agreement with statistics collected through forward hooks on
    //! the tape, a dead neuron, and the errors.

    use super::*;
    use crate::inference::tests::{digits, layer_bounds};
    use crate::{Hooks, NeuronMask, Targets, Trace};
    use cavitation_core::{Graph, Rng, Tensor};

    /// `rows` rows of 64 values uniform in `[0, 1)` from `Rng::seed(seed)`, with labels 0.
    fn data(rows: usize, seed: u64) -> Dataset {
        let mut rng = Rng::seed(seed);
        let x: Vec<f32> = (0..rows * 64).map(|_| rng.uniform_f32()).collect();
        Dataset {
            x: Tensor::from_vec(x, &[rows, 64]).unwrap(),
            targets: Targets::Classes(vec![0; rows]),
        }
    }

    /// Statistics collected by hand: for each of `linears`, the rows of the output of its
    /// neuron layer, gathered through forward hooks on the tape over the same batches
    /// `activity` runs, and the pre-activations (the `Linear`'s own outputs) of those rows.
    fn hooked(
        net: &Sequential,
        params: &ParamStore,
        data: &Dataset,
        linears: &[usize],
        batch: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let (rows, features) = (data.len(), data.features());
        let mut outputs = vec![Vec::new(); linears.len()];
        let mut pre = vec![Vec::new(); linears.len()];
        let mut graph = Graph::new();
        let mut trace = Trace::new();
        for start in (0..rows).step_by(batch) {
            let end = (start + batch).min(rows);
            let x = &data.x.data()[start * features..end * features];
            let input = Tensor::from_vec(x.to_vec(), &[end - start, features]).unwrap();
            let mut tape = graph.tape(&params.values);
            let xv = tape.input(input.view());
            net.forward_traced(&mut tape, xv, &mut trace);
            let mut hooks = Hooks::new();
            for (&linear, (out, pre)) in linears.iter().zip(outputs.iter_mut().zip(&mut pre)) {
                hooks.on_forward(net.neuron_layer(linear), move |e| {
                    out.extend_from_slice(e.output.data());
                });
                hooks.on_forward(linear, move |e| pre.extend_from_slice(e.output.data()));
            }
            hooks.run_forward(&tape, &trace, net);
        }
        (outputs, pre)
    }

    /// On the digits-shaped network over 50 rows in batches of 16, `activity` for both
    /// `Linear` layers gives the means, mean absolute values and maxima of the hooked
    /// outputs within the GEMM bound, and exactly their firing shares: no pre-activation of
    /// the seeded set lies within 1e-4 of 0, nor within its own GEMM bound of 0, so the sign
    /// of each output is the tape's.
    #[test]
    fn activity_matches_hooks() {
        let mut params = ParamStore::new();
        let net = digits(&mut params, 21);
        let data = data(50, 28);
        let linears = [0, 2];
        let stats = activity(&net, &params, &data, &linears, 16).unwrap();
        let (outputs, pre) = hooked(&net, &params, &data, &linears, 16);
        let layers = layer_bounds(&net, &params, data.x.data(), 50);
        assert_eq!(stats.len(), 2);
        for (k, s) in stats.iter().enumerate() {
            assert_eq!((s.linear, s.layer), ([0, 2][k], [1, 2][k]));
            let width = s.mean.len();
            assert_eq!(width, [32, 10][k]);
            let near = pre[k].iter().filter(|v| v.abs() < 1e-4).count();
            assert_eq!(
                near, 0,
                "linear {}: pre-activations within 1e-4 of 0",
                s.linear
            );
            let pre_bound = &layers[s.linear].1;
            let sign_known = pre[k]
                .iter()
                .zip(pre_bound)
                .all(|(&v, &b)| f64::from(v).abs() > b);
            assert!(
                sign_known,
                "linear {}: a pre-activation within its bound",
                s.linear
            );
            let bound = &layers[s.layer].1;
            let fired = s.firing.iter().filter(|&&f| f > 0.0 && f < 1.0).count();
            assert!(
                fired > 0,
                "linear {}: no neuron fires on only some rows",
                s.linear
            );
            for j in 0..width {
                let col: Vec<f64> = outputs[k]
                    .iter()
                    .skip(j)
                    .step_by(width)
                    .map(|&v| f64::from(v))
                    .collect();
                let b: Vec<f64> = bound.iter().skip(j).step_by(width).copied().collect();
                assert_eq!(col.len(), 50);
                let mean_bound = b.iter().sum::<f64>() / 50.0;
                let max_bound = b.iter().copied().fold(0.0, f64::max);
                let mean = col.iter().sum::<f64>() / 50.0;
                let mean_abs = col.iter().map(|v| v.abs()).sum::<f64>() / 50.0;
                let firing = col.iter().filter(|&&v| v > 0.0).count() as f64 / 50.0;
                let max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let what = format!("linear {}, neuron {j}", s.linear);
                assert!(
                    (s.mean[j] - mean).abs() <= mean_bound + 1e-12,
                    "{what}: mean"
                );
                assert!(
                    (s.mean_abs[j] - mean_abs).abs() <= mean_bound + 1e-12,
                    "{what}: mean |a|"
                );
                assert!((s.max[j] - max).abs() <= max_bound, "{what}: max");
                assert_eq!(s.firing[j], firing, "{what}: firing");
            }
        }
    }

    /// A hidden neuron with bias −100 never fires: its firing share, mean, mean absolute
    /// value and maximum are all 0, while its neighbours fire; a masked neuron reads 0 too.
    #[test]
    fn dead_neuron_firing_is_zero() {
        let mut params = ParamStore::new();
        let mut net = digits(&mut params, 23);
        let bias = net.param("0.bias").unwrap();
        params.value_mut(bias).data_mut()[5] = -100.0;
        net.set_neuron_mask(0, NeuronMask::from_off(32, &[9]));
        let stats = activity(&net, &params, &data(40, 24), &[0], 7).unwrap();
        let s = &stats[0];
        for dead in [5, 9] {
            assert_eq!(
                (s.firing[dead], s.mean[dead], s.mean_abs[dead], s.max[dead]),
                (0.0, 0.0, 0.0, 0.0),
                "neuron {dead}"
            );
        }
        assert!(s.firing.iter().filter(|&&f| f > 0.0).count() > 20);
    }

    /// A zero batch, a data set with no rows or the wrong width, and an index that is not a
    /// `Linear` are `InvalidData`, each message saying which.
    #[test]
    fn activity_rejects_bad_requests() {
        let mut params = ParamStore::new();
        let net = digits(&mut params, 25);
        let good = data(4, 26);
        let empty = Dataset {
            x: Tensor::zeros(&[0, 64]),
            targets: Targets::Classes(vec![]),
        };
        let narrow = Dataset {
            x: Tensor::zeros(&[2, 63]),
            targets: Targets::Classes(vec![0; 2]),
        };
        let cases: [(&Dataset, &[usize], usize, &str); 4] = [
            (&good, &[0], 0, "batch size is 0"),
            (&empty, &[0], 4, "no rows"),
            (&narrow, &[0], 4, "takes width 64"),
            (&good, &[0, 1], 4, "layer 1 is not a Linear"),
        ];
        for (d, linears, batch, want) in cases {
            match activity(&net, &params, d, linears, batch) {
                Err(Error::InvalidData(msg)) => assert!(msg.contains(want), "{msg}"),
                other => panic!("{want}: expected InvalidData, got {other:?}"),
            }
        }
    }
}
