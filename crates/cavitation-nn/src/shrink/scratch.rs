//! The from-scratch comparison: [`train_found_size`] trains a fresh network of the size a
//! shrunk network has, and measures both on the same rows, to show what pruning bought
//! beyond the size.

use std::fmt;

use cavitation_core::{Error, ParamStore, Rng};

use super::compact;
use crate::train::check_dataset;
use crate::{Dataset, Evaluation, FitConfig, History, Init, Linear, Prelu, Sequential, evaluate};

/// The rows run at a time when measuring the two networks.
const EVAL_BATCH: usize = 256;

/// How [`train_found_size`] builds and trains the fresh network: the starting values of
/// every `Linear`'s weight and bias, the seed they are drawn from, and the training recipe.
/// Taken by value, because the recipe can hold a callback.
///
/// The default is [`Linear::new`]'s initialisation (Xavier uniform weights, zero biases),
/// seed 0, and [`FitConfig::default`]. Give the recipe that trained the network you shrank,
/// for as many epochs in all as the pruned network had (its first training and every
/// step's retraining), so that the two are compared on equal training.
///
/// ```
/// use cavitation_nn::shrink::FromScratch;
/// use cavitation_nn::{FitConfig, Init};
/// let scratch = FromScratch {
///     weights: Init::HeUniform,
///     seed: 1,
///     fit: FitConfig { epochs: 44, ..FitConfig::default() },
///     ..FromScratch::default()
/// };
/// assert_eq!((scratch.biases, scratch.fit.epochs), (Init::Zeros, 44));
/// assert_eq!(FromScratch::default().weights, Init::XavierUniform);
/// ```
#[derive(Debug)]
pub struct FromScratch<'s> {
    /// How each `Linear`'s weight starts. Default [`Init::XavierUniform`].
    pub weights: Init,
    /// How each `Linear`'s bias starts. Default [`Init::Zeros`].
    pub biases: Init,
    /// The seed of the generator the starting values are drawn from, layer by layer in
    /// order, each weight before its bias. Default 0.
    pub seed: u64,
    /// The training: epochs, batch, optimiser, loss, schedule, shuffle seed, health checks.
    /// Its loss is also the one both networks are measured with.
    pub fit: FitConfig<'s>,
}

impl Default for FromScratch<'_> {
    /// Xavier uniform weights, zero biases, seed 0, [`FitConfig::default`].
    fn default() -> Self {
        FromScratch {
            weights: Init::XavierUniform,
            biases: Init::Zeros,
            seed: 0,
            fit: FitConfig::default(),
        }
    }
}

/// What [`train_found_size`] found: the pruned network's measures, the fresh network's
/// measures on the same rows, and the fresh network itself.
///
/// It is `#[non_exhaustive]`, so a later release can add a field without breaking your
/// code: read its fields, and get one from [`train_found_size`].
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::shrink::{FoundSize, FromScratch, train_found_size};
/// use cavitation_nn::{Dataset, FitConfig, Linear, NeuronMask, Relu, Sequential, Targets};
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let mut net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 6, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 6, 2, &mut rng));
/// net.set_neuron_mask(0, NeuronMask::from_off(6, &[0, 3]));
/// let data = Dataset { x: Tensor::zeros(&[8, 2]), targets: Targets::Classes(vec![0; 8]) };
/// let fit = FitConfig { epochs: 1, ..FitConfig::default() };
/// let scratch = FromScratch { fit, ..FromScratch::default() };
/// let found: FoundSize = train_found_size(&net, &params, &data, &data, scratch).unwrap();
/// assert_eq!(found.model.linear(0).unwrap().outputs(), 4);
/// assert_eq!(found.history.epochs.len(), 1);
/// println!("pruned {:?}, from scratch {:?}", found.pruned.accuracy, found.scratch.accuracy);
/// ```
#[derive(Clone)]
#[non_exhaustive]
pub struct FoundSize {
    /// The pruned network, as given (masked), measured on the evaluation rows.
    pub pruned: Evaluation,
    /// The fresh network, after its training, measured on the same rows.
    pub scratch: Evaluation,
    /// The fresh network: the compacted layout of the pruned one, with no masks.
    pub model: Sequential,
    /// The fresh network's trained parameters.
    pub params: ParamStore,
    /// The fresh network's training history, as [`fit`](crate::fit) returns it.
    pub history: History,
}

impl fmt::Debug for FoundSize {
    /// Writes the two measures, the fresh network's layer names and its history; the
    /// parameter values are left out.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let layers: Vec<&str> = (0..self.model.len())
            .map(|i| self.model.layer(i).name())
            .collect();
        f.debug_struct("FoundSize")
            .field("pruned", &self.pruned)
            .field("scratch", &self.scratch)
            .field("model", &layers)
            .field("history", &self.history)
            .finish_non_exhaustive()
    }
}

/// Trains a network of the size `model` has once its off neurons are removed, from new
/// starting values, and measures it beside `model` on `eval`: the comparison that tells
/// whether pruning found good weights or only a good size.
///
/// `model` and `params` are the pruned network, masked, as [`shrink`](super::shrink())
/// leaves them at the step it keeps; for another step, set that step's
/// [`checkpoint`](super::ShrinkStep::checkpoint) masks on the model and pass its
/// parameters. Any masked network [`compact`](super::compact()) accepts will do. The fresh
/// network is `compact`'s rebuild of `model` with every parameter started again: each
/// `Linear` gets new weights and biases as [`FromScratch::weights`] and
/// [`FromScratch::biases`] say (a `Linear` without a bias gets none), drawn in layer order
/// from `Rng::seed(scratch.seed)`, each
/// `Prelu` gets new slopes of 0.25 ([`Prelu::new`]), and the layers without parameters are
/// shared with `model`. The fresh network is then trained on `train` with
/// [`fit`](crate::fit) and `scratch.fit`, and both networks are measured on `eval` with
/// `scratch.fit.loss`, 256 rows at a time.
///
/// Measure on rows that neither the shrinking nor the training saw, a test set: the
/// pruned network's size was chosen on its validation rows, which flatter it. Give the
/// fresh network as many epochs as the pruned one had in all, or the comparison measures
/// the training, not the network.
///
/// With the same layout, seed, starting values and recipe, the result is the one you get
/// by building the network of the kept widths with those starting values yourself and
/// calling `fit`, bit for bit.
///
/// # Errors
///
/// Returns the errors of [`compact`](super::compact()) for a network it refuses (a layer
/// of your own with parameters, say), [`Error::InvalidData`] if `eval` is not a matrix of
/// inputs with at least one row and one target per row, and the errors of
/// [`fit`](crate::fit) and [`evaluate`](crate::evaluate).
///
/// # Panics
///
/// Panics as [`fit`](crate::fit) does.
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::shrink::{FromScratch, ShrinkConfig, Validation, shrink, train_found_size};
/// use cavitation_nn::{Dataset, FitConfig, Linear, Relu, Sequential, Targets, fit};
/// // Two classes split by the sign of the first input.
/// let mut rng = Rng::seed(1);
/// let x: Vec<f32> = (0..600).map(|_| rng.range(-1.0, 1.0) as f32).collect();
/// let labels = x.chunks(2).map(|r| usize::from(r[0] > 0.0)).collect();
/// let data = Dataset {
///     x: Tensor::from_vec(x, &[300, 2]).unwrap(),
///     targets: Targets::Classes(labels),
/// };
/// let (train, rest) = data.split_at(100);
/// let (held_out, test) = rest.split_at(100);
/// let mut params = ParamStore::new();
/// let mut net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 16, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 16, 2, &mut rng));
/// let recipe = |epochs| FitConfig { epochs, ..FitConfig::default() };
/// fit(&net, &mut params, &train, recipe(30)).unwrap();
/// let config = ShrinkConfig {
///     validation: Some(Validation::Given(held_out)),
///     max_removed_bp: 5000,
///     ..ShrinkConfig::default()
/// };
/// let report = shrink(&mut net, &mut params, &train, config).unwrap();
/// if let Some(chosen) = report.chosen {
///     // 30 epochs, then 3 after each step up to the chosen one.
///     let fit = recipe(30 + 3 * (chosen + 1));
///     let scratch = FromScratch { fit, ..FromScratch::default() };
///     let found = train_found_size(&net, &params, &train, &test, scratch).unwrap();
///     assert_eq!(found.model.linear(0).unwrap().outputs(), report.steps[chosen].kept[0]);
///     let (pruned, fresh) = (found.pruned.accuracy, found.scratch.accuracy);
///     println!("pruned {pruned:?}, from scratch {fresh:?}");
/// }
/// ```
pub fn train_found_size(
    model: &Sequential,
    params: &ParamStore,
    train: &Dataset,
    eval: &Dataset,
    scratch: FromScratch,
) -> Result<FoundSize, Error> {
    check_dataset(eval).map_err(|e| match e {
        Error::InvalidData(msg) => {
            Error::InvalidData(format!("train_found_size: the evaluation rows: {msg}"))
        }
        other => other,
    })?;
    let (layout, layout_params) = compact(model, params)?;
    let (fresh, mut fresh_params) = reinitialised(&layout, &layout_params, &scratch);
    let loss = scratch.fit.loss;
    let pruned = evaluate(model, params, eval, loss, EVAL_BATCH)?;
    let history = crate::fit(&fresh, &mut fresh_params, train, scratch.fit)?;
    let measured = evaluate(&fresh, &fresh_params, eval, loss, EVAL_BATCH)?;
    Ok(FoundSize {
        pruned,
        scratch: measured,
        model: fresh,
        params: fresh_params,
        history,
    })
}

/// `layout` (with its values in `values`) rebuilt in a new store with new starting values:
/// each `Linear` as `scratch` says (without a bias if it had none), drawn in layer order
/// from one generator, each `Prelu`
/// with as many slopes as before, all 0.25, and every other layer shared. `layout` comes
/// from `compact`, so it has no other layer with parameters.
fn reinitialised(
    layout: &Sequential,
    values: &ParamStore,
    scratch: &FromScratch,
) -> (Sequential, ParamStore) {
    let mut rng = Rng::seed(scratch.seed);
    let mut store = ParamStore::new();
    let mut out = Sequential::new();
    for i in 0..layout.len() {
        let layer = layout.layer(i);
        if let Some(linear) = layer.as_linear() {
            let (inputs, outputs) = (linear.inputs(), linear.outputs());
            out = out.push(match linear.bias() {
                Some(_) => Linear::with_init(
                    &mut store,
                    inputs,
                    outputs,
                    scratch.weights,
                    scratch.biases,
                    &mut rng,
                ),
                None => {
                    Linear::without_bias(&mut store, inputs, outputs, scratch.weights, &mut rng)
                }
            });
        } else if let Some(prelu) = layer.as_prelu() {
            let slopes = values.value(prelu.weight()).data().len();
            out = out.push(Prelu::new(&mut store, slopes));
        } else {
            out = out.push_shared(layout.shared_layer(i));
        }
    }
    (out, store)
}

#[cfg(test)]
mod tests {
    //! Unit tests of the rebuilt layout.

    use super::*;
    use crate::{NeuronMask, Relu};

    /// The fresh network has the compacted widths, a per-neuron `Prelu` with the kept
    /// number of slopes at 0.25, and weights drawn from the seed as `Linear::with_init`
    /// draws them in order.
    #[test]
    fn rebuilt_with_new_values() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(9);
        let mut net = Sequential::new()
            .push(Linear::new(&mut params, 3, 8, &mut rng))
            .push(Prelu::new(&mut params, 8))
            .push(Linear::new(&mut params, 8, 5, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, 5, 2, &mut rng));
        net.set_neuron_mask(0, NeuronMask::from_off(8, &[1, 2, 6]));
        net.set_neuron_mask(2, NeuronMask::from_off(5, &[4]));
        let (layout, values) = compact(&net, &params).unwrap();
        let scratch = FromScratch {
            weights: Init::HeUniform,
            seed: 4,
            ..FromScratch::default()
        };
        let (fresh, store) = reinitialised(&layout, &values, &scratch);
        assert!(fresh.mask(1).is_none() && fresh.neuron_mask(0).is_none());
        let mut want_store = ParamStore::new();
        let mut want_rng = Rng::seed(4);
        let widths = [(3, 5), (5, 4), (4, 2)];
        for (&(i, o), l) in widths.iter().zip([0, 2, 4]) {
            let want = Linear::with_init(
                &mut want_store,
                i,
                o,
                Init::HeUniform,
                Init::Zeros,
                &mut want_rng,
            );
            let got = fresh.linear(l).unwrap();
            assert_eq!((got.inputs(), got.outputs()), (i, o));
            assert_eq!(
                store.value(got.weight()).data(),
                want_store.value(want.weight()).data()
            );
            assert!(
                store
                    .value(got.bias().unwrap())
                    .data()
                    .iter()
                    .all(|&b| b == 0.0)
            );
        }
        let prelu = fresh.layer(1).as_prelu().unwrap();
        assert_eq!(store.value(prelu.weight()).data(), &[0.25; 5]);
    }
}
