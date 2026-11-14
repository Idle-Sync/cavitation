//! Per-neuron activity statistics over a data set: [`activity`] runs the data through the
//! compiled model in batches (or on the tape, for a model with a layer that has no
//! inference form) and returns a [`LayerActivity`] for each chosen `Linear`.

use cavitation_core::{Error, Graph, ParamStore, TensorView};

use crate::train::check_widths;
use crate::{Dataset, InferenceModel, Sequential, Trace};

/// How the neurons of one `Linear` layer behaved over a data set, from [`activity`]: one
/// value per neuron in each list, read where the neuron's output is final, as the next
/// layer receives it (after the run of element-wise layers that follows the `Linear`, at
/// [`Sequential::neuron_layer`], and after any mask there).
///
/// It is `#[non_exhaustive]`, so a later release can add a statistic without breaking your
/// code: read its fields, and get one from [`activity`].
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::shrink::{LayerActivity, activity};
/// use cavitation_nn::{Dataset, Linear, Relu, Sequential, Targets};
/// let mut params = ParamStore::new();
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 3, &mut Rng::seed(0)))
///     .push(Relu);
/// let data = Dataset { x: Tensor::zeros(&[4, 2]), targets: Targets::Classes(vec![0; 4]) };
/// let stats: Vec<LayerActivity> = activity(&net, &params, &data, &[0], 4).unwrap();
/// assert_eq!((stats[0].linear, stats[0].layer, stats[0].mean.len()), (0, 1, 3));
/// ```
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
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
/// path, [`InferenceModel::infer_batch_observed`]). A model with a layer that has no
/// inference form (an activation of your own, say) runs on the tape instead, as an
/// [`Evaluator`](crate::Evaluator) does, with the same results within rounding. Each
/// `Linear`'s neurons are read at [`Sequential::neuron_layer`]: after the run of
/// element-wise layers that follows it, masked as the next layer sees them. The sums are
/// kept in `f64`; the targets of `data` are not used. The result has one [`LayerActivity`]
/// per entry of `linears`, in that order; with no entries it is empty, and the data is not
/// run.
///
/// # Errors
///
/// Returns [`Error::InvalidData`] if `batch` is 0, if `data.x` is not a matrix with at
/// least one row, if its width is not the model's input width, or if an entry of
/// `linears` is not the index of a `Linear` layer (the message names it).
///
/// # Panics
///
/// Panics as [`InferenceModel::compile`] (or [`Sequential::forward`]) does when the
/// layers' widths do not fit one another.
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::shrink::activity;
/// use cavitation_nn::{Dataset, Linear, Relu, Sequential, Targets};
/// let mut params = ParamStore::new();
/// let layer = Linear::new(&mut params, 1, 2, &mut Rng::seed(0));
/// params.value_mut(layer.weight()).data_mut().copy_from_slice(&[1.0, -1.0]);
/// params.value_mut(layer.bias().unwrap()).data_mut().copy_from_slice(&[0.0, 0.0]);
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
    if linears.is_empty() {
        return Ok(Vec::new());
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
    let x = data.x.data();
    let rows_of = |start: usize| {
        let end = (start + batch).min(rows);
        TensorView::from_slice(
            &x[start * features..end * features],
            &[end - start, features],
        )
        .expect("the rows hold (end - start) × features values")
    };
    match InferenceModel::compile(model, params) {
        Ok(mut compiled) => {
            if features != compiled.input_width() {
                return Err(Error::InvalidData(format!(
                    "activity: the inputs are [{rows}, {features}], but the model takes width {}",
                    compiled.input_width()
                )));
            }
            for start in (0..rows).step_by(batch) {
                compiled.infer_batch_observed(rows_of(start), &mut |layer, out| {
                    for s in stats.iter_mut().filter(|s| s.layer == layer) {
                        accumulate(s, out);
                    }
                });
            }
        }
        // A layer with no inference form, or an input width only the data can tell: run
        // on the tape, reading each layer's output from the trace.
        Err(Error::Unsupported(_)) => {
            check_widths(model, data).map_err(|e| match e {
                Error::InvalidData(msg) => Error::InvalidData(format!("activity: {msg}")),
                other => other,
            })?;
            let mut graph = Graph::new();
            let mut trace = Trace::with_capacity(model.len());
            for start in (0..rows).step_by(batch) {
                let mut tape = graph.tape(&params.values);
                let input = tape.input(rows_of(start));
                model.forward_traced(&mut tape, input, &mut trace);
                for s in &mut stats {
                    accumulate(s, tape.value(trace.output(s.layer)));
                }
            }
        }
        Err(e) => return Err(e),
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
    use crate::inference::tests::digits;
    use crate::{Activation, ActivationLayer, Hooks, NeuronMask, Targets, Trace};
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

    /// An integer uniform in `lo..hi` from `rng`.
    fn int(rng: &mut Rng, lo: i32, hi: i32) -> f32 {
        rng.range(f64::from(lo), f64::from(hi)).floor() as f32
    }

    /// The digits-shaped network with dyadic values from `Rng::seed(seed)`, and `rows` rows
    /// of inputs, chosen so that every sum is exact in `f32` on every kernel level (any
    /// order, fused or not) and no pre-activation is 0: inputs `k/8` with `k` in `0..8`,
    /// weights `m/16` with `m` in `-8..9`, first-layer biases odd multiples of `1/256` and
    /// last-layer biases odd multiples of `1/8192`. A first-layer pre-activation is then an
    /// odd multiple of `1/256` (at most 28 in size), and a last-layer one an odd multiple of
    /// `1/8192` (at most 448), so each lies at least `1/8192 ≈ 1.2e-4` from 0.
    fn dyadic(params: &mut ParamStore, rows: usize, seed: u64) -> (Sequential, Dataset) {
        let net = digits(params, seed);
        let mut rng = Rng::seed(seed);
        for (name, unit) in [("0.bias", 256.0f32), ("2.bias", 8192.0)] {
            let id = net.param(name).unwrap();
            for b in params.value_mut(id).data_mut() {
                *b = (2.0 * int(&mut rng, -8, 8) + 1.0) / unit;
            }
        }
        for name in ["0.weight", "2.weight"] {
            let id = net.param(name).unwrap();
            for w in params.value_mut(id).data_mut() {
                *w = int(&mut rng, -8, 9) / 16.0;
            }
        }
        let x: Vec<f32> = (0..rows * 64).map(|_| int(&mut rng, 0, 8) / 8.0).collect();
        let data = Dataset {
            x: Tensor::from_vec(x, &[rows, 64]).unwrap(),
            targets: Targets::Classes(vec![0; rows]),
        };
        (net, data)
    }

    /// On the digits-shaped network with dyadic values over 50 rows in batches of 16,
    /// `activity` for both `Linear` layers gives exactly the means, mean absolute values,
    /// firing shares and maxima of the outputs hooked on the tape: every sum is exact, so
    /// the GEMM bound is 0, and no pre-activation lies within 1e-4 of 0 (asserted).
    #[test]
    fn activity_matches_hooks() {
        let mut params = ParamStore::new();
        let (net, data) = dyadic(&mut params, 50, 21);
        let linears = [0, 2];
        let stats = activity(&net, &params, &data, &linears, 16).unwrap();
        let (outputs, pre) = hooked(&net, &params, &data, &linears, 16);
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
                assert_eq!(col.len(), 50);
                let mean = col.iter().sum::<f64>() / 50.0;
                let mean_abs = col.iter().map(|v| v.abs()).sum::<f64>() / 50.0;
                let firing = col.iter().filter(|&&v| v > 0.0).count() as f64 / 50.0;
                let max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let what = format!("linear {}, neuron {j}", s.linear);
                assert_eq!(s.mean[j], mean, "{what}: mean");
                assert_eq!(s.mean_abs[j], mean_abs, "{what}: mean |a|");
                assert_eq!(s.firing[j], firing, "{what}: firing");
                assert_eq!(s.max[j], max, "{what}: max");
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

    /// ReLU written as an activation of your own, so a network using it has no inference
    /// form.
    struct MyRelu;

    impl Activation<f32> for MyRelu {
        /// Returns `"my_relu"`.
        fn name(&self) -> &str {
            "my_relu"
        }
        /// `max(x, 0)`.
        fn f(&self, x: f32) -> f32 {
            x.max(0.0)
        }
        /// 1 above 0, else 0.
        fn df(&self, x: f32, _y: f32) -> f32 {
            if x > 0.0 { 1.0 } else { 0.0 }
        }
    }

    /// The dyadic digits-shaped network with its ReLU written as an activation of your own
    /// runs on the tape, and gives exactly the statistics of the built-in network compiled
    /// (every sum is exact on both paths), masked neuron included; a data set of the wrong
    /// width is `InvalidData` there too.
    #[test]
    fn activity_falls_back_to_the_tape() {
        let mut params = ParamStore::new();
        let (mut net, data) = dyadic(&mut params, 30, 27);
        net.set_neuron_mask(0, NeuronMask::from_off(32, &[4]));
        let mut own = Sequential::new()
            .push(net.linear(0).unwrap().clone())
            .push(ActivationLayer::new(MyRelu))
            .push(net.linear(2).unwrap().clone());
        own.set_neuron_mask(0, NeuronMask::from_off(32, &[4]));
        assert!(matches!(
            InferenceModel::compile(&own, &params),
            Err(Error::Unsupported(_))
        ));
        let want = activity(&net, &params, &data, &[2, 0], 8).unwrap();
        let got = activity(&own, &params, &data, &[2, 0], 8).unwrap();
        assert_eq!(got, want);
        assert_eq!(got[1].max[4], 0.0);
        let narrow = Dataset {
            x: Tensor::zeros(&[2, 63]),
            targets: Targets::Classes(vec![0; 2]),
        };
        match activity(&own, &params, &narrow, &[0], 4) {
            Err(Error::InvalidData(msg)) => assert!(msg.starts_with("activity: "), "{msg}"),
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }

    /// With no `Linear` listed the result is empty and the data is not run: a data set of
    /// the wrong width, which running would reject, gives an empty list.
    #[test]
    fn no_linears_reads_no_data() {
        let mut params = ParamStore::new();
        let net = digits(&mut params, 29);
        let narrow = Dataset {
            x: Tensor::zeros(&[2, 63]),
            targets: Targets::Classes(vec![0; 2]),
        };
        assert_eq!(activity(&net, &params, &narrow, &[], 4).unwrap(), []);
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
