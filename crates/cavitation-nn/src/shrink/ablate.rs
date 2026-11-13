//! The ablation sweep: [`ablate`] and [`ablate_with`] switch each neuron of chosen hidden
//! `Linear` layers off, one at a time, measure the network, and rank the neurons by how much
//! each one's absence costs.

use cavitation_core::{Error, ParamStore};

use super::neurons_on;
use crate::{Dataset, Evaluation, Evaluator, Loss, NeuronMask, Sequential, Targets};

/// What switching one neuron off costs, from [`ablate`]: the neuron, the network's
/// [`Evaluation`] without it (every other neuron as the model had it), and the drop.
///
/// It is `#[non_exhaustive]`, so a later release can add a field without breaking your
/// code: read its fields, and get one from [`ablate`] or [`ablate_with`].
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::shrink::{Ablation, ablate};
/// use cavitation_nn::{Dataset, Linear, Loss, Relu, Sequential, Targets};
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 3, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 3, 2, &mut rng));
/// let data = Dataset { x: Tensor::zeros(&[4, 2]), targets: Targets::Classes(vec![0; 4]) };
/// let all: Vec<Ablation> = ablate(&net, &params, &data, &[0], Loss::CrossEntropy, 4).unwrap();
/// assert_eq!(all.len(), 3);
/// assert!(all.iter().all(|a| a.linear == 0 && a.neuron < 3));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Ablation {
    /// The index of the `Linear` layer the neuron belongs to.
    pub linear: usize,
    /// The neuron's index in that layer's output.
    pub neuron: usize,
    /// The network's measures on the data set with this neuron off.
    pub evaluation: Evaluation,
    /// The cost of switching the neuron off. For class targets, the accuracy points lost:
    /// the accuracy without any neuron switched off minus the accuracy with this one off,
    /// times 100, worked out from whole rows (negative if the network does better without
    /// it; one row in 100 is exactly 1.0). For value targets, the relative increase of the
    /// loss, `(loss_off - loss) / |loss|`: dividing by the magnitude keeps a higher loss a
    /// positive drop when the loss is negative (as a KL divergence can be). When the loss is
    /// 0 the drop is 0 if the loss stays 0 and infinity otherwise. A NaN loss with the
    /// neuron off counts as an infinite drop.
    pub drop: f64,
}

/// Switches off each neuron of the `Linear` layers in `linears` in turn, measures `model`
/// on `data` without it, and returns the neurons ranked by what their absence costs, the
/// most costly first. See [`ablate_with`], which also reports each neuron as it finishes.
///
/// # Errors
///
/// As [`ablate_with`].
///
/// # Panics
///
/// Panics as [`ablate_with`] does.
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::shrink::ablate;
/// use cavitation_nn::{Dataset, Linear, Loss, Relu, Sequential, Targets};
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 3, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 3, 2, &mut rng));
/// let data = Dataset { x: Tensor::zeros(&[4, 2]), targets: Targets::Classes(vec![0; 4]) };
/// let ranked = ablate(&net, &params, &data, &[0], Loss::CrossEntropy, 4).unwrap();
/// assert!(ranked.windows(2).all(|w| w[0].drop >= w[1].drop));
/// // The output layer cannot be ablated: its neurons are the model's outputs.
/// assert!(ablate(&net, &params, &data, &[2], Loss::CrossEntropy, 4).is_err());
/// ```
pub fn ablate(
    model: &Sequential,
    params: &ParamStore,
    data: &Dataset,
    linears: &[usize],
    loss: Loss,
    batch: usize,
) -> Result<Vec<Ablation>, Error> {
    ablate_with(model, params, data, linears, loss, batch, &mut |_| {})
}

/// Like [`ablate`], and calls `on_neuron` with each neuron's [`Ablation`] as it finishes
/// (in sweep order: the `Linear`s in the order of `linears`, the neurons of each in
/// order), for a progress display.
///
/// One [`Evaluator`] runs the whole sweep: it is made once with masks on the listed
/// `Linear`s, and each neuron is switched off with
/// [`Evaluator::set_neuron_mask`], measured and switched back on, so nothing is compiled
/// or allocated per neuron. A model with a layer that has no inference form (an activation
/// of your own, say) is swept on the tape, as an [`Evaluator`] runs it. `model` is not
/// changed. A neuron that is off in any mask of its `Linear`'s run (the `Linear` itself and
/// the element-wise layers after it, up to [`Sequential::neuron_layer`]) is already off: it
/// has no entry and stays off in every measurement, so each entry is the model's own masked
/// network minus one more neuron. A `Linear` listed twice is swept once.
///
/// The sort is by [`Ablation::drop`], largest first, ties by `(linear, neuron)` ascending,
/// so the order is the same on every run. The measurements are deterministic, so a neuron
/// that changes nothing (one that never fires on `data`) has a drop of exactly 0.
///
/// # Errors
///
/// Returns [`Error::InvalidData`] if an entry of `linears` is not the index of a hidden
/// `Linear` layer: a layer that is not a `Linear`, or the last `Linear`, whose neurons are
/// the model's outputs (the message names the layer); if `batch` is 0; and as
/// [`Evaluator::evaluate`] does for the data and the loss. With no entry in `linears` it
/// measures the unablated network once (so the data is checked) and returns an empty list.
///
/// # Panics
///
/// Panics as [`Evaluator::new`] does when the layers' widths do not fit one another.
///
/// ```
/// use cavitation_core::{ParamStore, Rng, Tensor};
/// use cavitation_nn::shrink::ablate_with;
/// use cavitation_nn::{Dataset, Linear, Loss, Relu, Sequential, Targets};
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let net = Sequential::new()
///     .push(Linear::new(&mut params, 2, 3, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 3, 2, &mut rng));
/// let data = Dataset { x: Tensor::zeros(&[4, 2]), targets: Targets::Classes(vec![0; 4]) };
/// let mut done = 0;
/// let all = ablate_with(&net, &params, &data, &[0], Loss::CrossEntropy, 4, &mut |_| done += 1)
///     .unwrap();
/// assert_eq!((done, all.len()), (3, 3));
/// ```
pub fn ablate_with(
    model: &Sequential,
    params: &ParamStore,
    data: &Dataset,
    linears: &[usize],
    loss: Loss,
    batch: usize,
    on_neuron: &mut dyn FnMut(&Ablation),
) -> Result<Vec<Ablation>, Error> {
    let mut chosen: Vec<usize> = Vec::with_capacity(linears.len());
    for &linear in linears {
        check_hidden(model, linear, "ablate")?;
        if !chosen.contains(&linear) {
            chosen.push(linear);
        }
    }
    let mut evaluator = Evaluator::with_neuron_masks(model, params, loss, batch, &chosen)?;
    let base = evaluator.evaluate(data)?;
    let rows = data.len();
    let mut all = Vec::new();
    for &linear in &chosen {
        let width = model.linear(linear).expect("checked above").outputs();
        // The neurons that are off in any mask of the `Linear`'s run (the `Linear` and the
        // element-wise layers up to `neuron_layer`) are off already: no entry, and they
        // stay off, since the evaluator keeps every mask of the model in place.
        let on = neurons_on(model, linear);
        let start = model
            .neuron_mask(linear)
            .cloned()
            .unwrap_or_else(|| NeuronMask::all_on(width));
        let mut work = start.clone();
        for neuron in (0..width).filter(|&n| on[n]) {
            work.set(neuron, false);
            evaluator.set_neuron_mask(linear, &work)?;
            let evaluation = evaluator.evaluate(data)?;
            work.set(neuron, true);
            let ablation = Ablation {
                linear,
                neuron,
                evaluation,
                drop: drop_of(&base, &evaluation, &data.targets, rows),
            };
            on_neuron(&ablation);
            all.push(ablation);
        }
        evaluator.set_neuron_mask(linear, &start)?;
    }
    all.sort_by(|a, b| {
        b.drop
            .total_cmp(&a.drop)
            .then((a.linear, a.neuron).cmp(&(b.linear, b.neuron)))
    });
    Ok(all)
}

/// Returns [`Error::InvalidData`] naming `linear`, with the message prefixed `"{op}: "`,
/// unless it is a `Linear` layer of `model` with another `Linear` after it.
pub(super) fn check_hidden(model: &Sequential, linear: usize, op: &str) -> Result<(), Error> {
    if model.linear(linear).is_none() {
        return Err(Error::InvalidData(format!(
            "{op}: layer {linear} is not a Linear layer"
        )));
    }
    if (linear + 1..model.len()).all(|l| model.linear(l).is_none()) {
        return Err(Error::InvalidData(format!(
            "{op}: layer {linear} is the last Linear layer, so its neurons are the model's \
             output and not a hidden layer's; list the hidden Linear layers"
        )));
    }
    Ok(())
}

/// The cost of `ablated` against `base` over `rows` rows, as [`Ablation::drop`] defines it.
fn drop_of(base: &Evaluation, ablated: &Evaluation, targets: &Targets, rows: usize) -> f64 {
    match (targets, base.accuracy, ablated.accuracy) {
        (Targets::Classes(_), Some(before), Some(after)) => points_lost(before, after, rows),
        _ => relative_increase(base.loss, ablated.loss),
    }
}

/// The accuracy points lost going from the share `before` to the share `after` of `rows`
/// rows: whole rows lost times 100 over `rows`, so it is rounded once, and one row in 100
/// is exactly 1.0. Negative when `after` is the better.
pub(super) fn points_lost(before: f64, after: f64, rows: usize) -> f64 {
    let n = rows as f64;
    let lost = ((before - after) * n).round();
    lost * 100.0 / n
}

/// The relative increase from the loss `base` to `loss`, `(loss - base) / |base|`: 0 if
/// both are 0, infinity for a rise from 0 or a NaN `loss`.
fn relative_increase(base: f64, loss: f64) -> f64 {
    if loss.is_nan() {
        f64::INFINITY
    } else if base != 0.0 {
        (loss - base) / base.abs()
    } else if loss == base {
        0.0
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of the arms of `drop_of`.

    use super::*;

    /// An evaluation with `loss` and `accuracy` (R² unused).
    fn eval(loss: f64, accuracy: Option<f64>) -> Evaluation {
        Evaluation {
            loss,
            accuracy,
            r2: None,
        }
    }

    /// Value targets: a zero baseline gives 0 for an equal loss and infinity otherwise, NaN
    /// gives infinity, and a negative baseline is divided by its magnitude.
    #[test]
    fn value_drop_arms() {
        let values = Targets::Values(cavitation_core::Tensor::zeros(&[1, 1]));
        let drop =
            |base: f64, after: f64| drop_of(&eval(base, None), &eval(after, None), &values, 1);
        assert_eq!(drop(0.0, 0.0), 0.0);
        assert_eq!(drop(0.0, 0.5), f64::INFINITY);
        assert_eq!(drop(2.0, f64::NAN), f64::INFINITY);
        assert_eq!(drop(2.0, 3.0), 0.5);
        assert_eq!(drop(2.0, 1.0), -0.5);
        // A higher loss is a positive drop whatever the baseline's sign.
        assert_eq!(drop(-2.0, -1.0), 0.5);
        assert_eq!(drop(-2.0, -3.0), -0.5);
    }

    /// Class targets: equal accuracies drop 0, and whole rows give exact point counts.
    #[test]
    fn class_drop_is_exact() {
        let classes = Targets::Classes(vec![0; 100]);
        let drop = |before: f64, after: f64, rows: usize| {
            drop_of(
                &eval(1.0, Some(before)),
                &eval(1.0, Some(after)),
                &classes,
                rows,
            )
        };
        assert_eq!(drop(0.5, 0.5, 100), 0.0);
        assert_eq!(drop(0.97, 0.96, 100), 1.0);
        assert_eq!(drop(0.96, 0.97, 100), -1.0);
        assert_eq!(drop(1.0, 0.0, 100), 100.0);
    }
}
