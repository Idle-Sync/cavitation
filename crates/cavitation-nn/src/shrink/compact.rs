//! Compaction: [`compact`] rebuilds a masked network without its off neurons, as a smaller
//! dense network with its own parameter store.

use cavitation_core::{Error, Graph, ParamStore, Real, Tensor};

use crate::{Layer, Linear, Prelu, Sequential};

/// Rebuilds `model` with every switched-off hidden neuron removed: a smaller dense network
/// and a new [`ParamStore`] holding its values, with no masks and the same layer indices.
///
/// For each masked hidden `Linear` (its neurons masked at
/// [`Sequential::neuron_layer`], or at the `Linear` itself or a layer between, as long as
/// the off neurons still reach the next `Linear` as 0):
///
/// - the `Linear` is rebuilt from the rows of its weight and bias for the neurons that are
///   on;
/// - the next `Linear` is rebuilt from the matching columns of its weight;
/// - a `Prelu` with one slope per neuron between them is rebuilt with [`Prelu::new`] and its
///   kept slopes copied.
///
/// Every other `Linear` and `Prelu` is rebuilt in the new store with its values copied, and
/// layers without parameters are shared with `model`. Parameter names
/// ([`Sequential::parameter_names`]) keep their layer indices, so the result saves and
/// loads like any network.
///
/// Through [`InferenceModel::infer`](crate::InferenceModel::infer) the compacted network
/// gives the masked one's outputs bit for bit: each output of the matvec is a sum in input
/// order from +0, and a masked input adds an exact 0 to it, which leaves the sum as it is.
/// On the batched paths ([`InferenceModel::infer_batch`](crate::InferenceModel::infer_batch)
/// and the tape) the two agree within the GEMM tolerance for the masked layer's length,
/// and bit for bit while every sum fits in one block of the GEMM (256 terms on every
/// level). This holds for finite weights: a masked neuron whose outgoing weight is infinite
/// or NaN gives `0 · ∞ = NaN` in the masked network, while the compacted network drops that
/// weight and gives a number.
///
/// # Errors
///
/// [`Error::Unsupported`], naming the layer and what to do instead, for:
///
/// - a mask with no `Linear` after it, so that its neurons are the model's outputs;
/// - a mask whose off neurons pass an element-wise layer with `f(0) ≠ 0` on the way to the
///   next `Linear` (a mask on a `Linear` or a `Tanh` followed by a `Sigmoid`, say), so that
///   they do not send 0; mask the `Linear`'s neurons with
///   [`Sequential::set_neuron_mask`] instead, which puts the mask after every activation;
/// - a mask whose neurons do not reach the next `Linear` through element-wise layers only
///   (a mask before a `Softmax`, or on a layer that does not follow a `Linear`);
/// - a layer with parameters other than a `Linear` or a `Prelu`, whose parameters cannot be
///   moved to the new store;
/// - a `Linear` whose every neuron is off.
///
/// [`Error::InvalidData`], naming the layer, when a mask's width or a parameter's shape does
/// not fit its layer. `f(0)` is found by running each element-wise layer after a mask on a
/// row of zeros, so any layer, built-in or your own, is judged by what it computes.
///
/// # Panics
///
/// Panics as a layer's [`forward`](Layer::forward) does when the widths of the layers do
/// not fit one another.
///
/// ```
/// use cavitation_core::{ParamStore, Rng};
/// use cavitation_nn::shrink::compact;
/// use cavitation_nn::{InferenceModel, Linear, NeuronMask, Relu, Sequential};
/// let mut params = ParamStore::new();
/// let mut rng = Rng::seed(0);
/// let mut net = Sequential::new()
///     .push(Linear::new(&mut params, 4, 8, &mut rng))
///     .push(Relu)
///     .push(Linear::new(&mut params, 8, 2, &mut rng));
/// net.set_neuron_mask(0, NeuronMask::from_off(8, &[1, 4, 6]));
/// let (small, small_params) = compact(&net, &params).unwrap();
/// assert_eq!(small.linear(0).unwrap().outputs(), 5);
/// assert_eq!(small.linear(2).unwrap().inputs(), 5);
/// assert!(small.mask(1).is_none());
/// let x = [0.5, -1.0, 0.25, 2.0];
/// let mut masked = InferenceModel::compile(&net, &params).unwrap();
/// let mut compacted = InferenceModel::compile(&small, &small_params).unwrap();
/// assert_eq!(masked.infer(&x), compacted.infer(&x));
/// ```
pub fn compact<T: Real>(
    model: &Sequential<T>,
    params: &ParamStore<T>,
) -> Result<(Sequential<T>, ParamStore<T>), Error> {
    check_parameters(model)?;
    let owners = run_owners(model);
    let keep = kept_neurons(model, params, &owners)?;
    let mut store = ParamStore::<T>::new_generic();
    let mut out = Sequential::<T>::new_generic();
    for i in 0..model.len() {
        let layer = model.layer(i);
        if let Some(linear) = layer.as_linear() {
            // The columns kept: the on neurons of the `Linear` whose run ends just before.
            let cols = i
                .checked_sub(1)
                .and_then(|p| owners[p])
                .and_then(|owner| keep[owner].as_deref());
            let rows = keep[i].as_deref();
            let rebuilt = rebuild_linear(i, linear, params, &mut store, rows, cols)?;
            out = out.push(rebuilt);
        } else if let Some(prelu) = layer.as_prelu() {
            let kept = owners[i].and_then(|owner| keep[owner].as_deref());
            out = out.push(rebuild_prelu(i, prelu, params, &mut store, kept)?);
        } else {
            out = out.push_shared(model.shared_layer(i));
        }
    }
    Ok((out, store))
}

/// Refuses, naming it, a layer with parameters that is neither a `Linear` nor a `Prelu`.
fn check_parameters<T: Real>(model: &Sequential<T>) -> Result<(), Error> {
    for i in 0..model.len() {
        let layer = model.layer(i);
        if layer.as_linear().is_none()
            && layer.as_prelu().is_none()
            && !layer.parameters().is_empty()
        {
            return Err(Error::Unsupported(format!(
                "compact: layer {i} ({}) has parameters, and only Linear and Prelu layers can \
                 be rebuilt in the new store; build the network from Linear, Prelu and layers \
                 without parameters to compact it",
                layer.name()
            )));
        }
    }
    Ok(())
}

/// For each layer, the `Linear` whose neurons its output holds: the layer itself for a
/// `Linear`, the `Linear` before it for a layer in the run of element-wise layers after one
/// (up to [`Sequential::neuron_layer`]), and `None` otherwise.
fn run_owners<T: Real>(model: &Sequential<T>) -> Vec<Option<usize>> {
    let mut owners = vec![None; model.len()];
    for (linear, _) in model.linears() {
        for owner in &mut owners[linear..=model.neuron_layer(linear)] {
            *owner = Some(linear);
        }
    }
    owners
}

/// For each layer index, the on neurons of the `Linear` there if any mask switches some of
/// them off (`None` for an unmasked `Linear` and every other layer), after checking that
/// every masked neuron reaches the next `Linear` as 0.
fn kept_neurons<T: Real>(
    model: &Sequential<T>,
    params: &ParamStore<T>,
    owners: &[Option<usize>],
) -> Result<Vec<Option<Vec<usize>>>, Error> {
    let n = model.len();
    let mut on: Vec<Option<Vec<bool>>> = vec![None; n];
    for (m, &owner) in owners.iter().enumerate() {
        let Some(mask) = model.mask(m) else {
            continue;
        };
        let name = model.layer(m).name();
        let no_linear_after = (m + 1..n).all(|l| model.linear(l).is_none());
        if no_linear_after {
            return Err(Error::Unsupported(format!(
                "compact: layer {m} ({name}) carries a mask, but no Linear layer follows it, \
                 so its neurons are the model's outputs and cannot be removed; clear the mask \
                 with clear_mask({m}) before compacting"
            )));
        }
        let not_reaching = |why: String| {
            Error::Unsupported(format!(
                "compact: the neurons masked on layer {m} ({name}) {why}, so they cannot be \
                 removed; only the neurons of a Linear whose run of element-wise layers leads \
                 to the next Linear can be; clear the mask with clear_mask({m}) before \
                 compacting"
            ))
        };
        let Some(linear) = owner else {
            return Err(not_reaching(
                "are not the neurons of a Linear layer (the layer does not follow a Linear \
                 through element-wise layers only)"
                    .to_string(),
            ));
        };
        let end = model.neuron_layer(linear);
        if model.linear(end + 1).is_none() {
            let next = end + 1;
            return Err(not_reaching(format!(
                "reach layer {next} ({}), which is not a Linear",
                model.layer(next).name()
            )));
        }
        let width = model
            .linear(linear)
            .expect("the owner of a run is a Linear")
            .outputs();
        if mask.width() != width {
            return Err(Error::InvalidData(format!(
                "compact: the mask on layer {m} ({name}) has width {}, but the linear layer \
                 {linear} outputs width {width}",
                mask.width()
            )));
        }
        let off: Vec<usize> = mask.off().collect();
        for l in m + 1..=end {
            let zeros = zeros_through(model.layer(l), params, width);
            if zeros.len() != width {
                return Err(Error::InvalidData(format!(
                    "compact: layer {l} ({}) turns a row of width {width} into width {}",
                    model.layer(l).name(),
                    zeros.len()
                )));
            }
            if off.iter().any(|&j| zeros[j] != T::ZERO) {
                return Err(Error::Unsupported(format!(
                    "compact: the mask on layer {m} ({name}) switches neurons off before layer \
                     {l} ({}), which does not map 0 to 0, so an off neuron does not send 0 to \
                     the next Linear and cannot be removed; mask the neurons of the linear \
                     layer {linear} with set_neuron_mask({linear}, mask) instead, which puts \
                     the mask on layer {end}, after every activation",
                    model.layer(l).name()
                )));
            }
        }
        let entry = on[linear].get_or_insert_with(|| vec![true; width]);
        for j in off {
            entry[j] = false;
        }
    }
    on.into_iter()
        .enumerate()
        .map(|(linear, on)| {
            let Some(on) = on else {
                return Ok(None);
            };
            let kept: Vec<usize> = (0..on.len()).filter(|&j| on[j]).collect();
            if kept.is_empty() {
                return Err(Error::Unsupported(format!(
                    "compact: every neuron of the linear layer {linear} is off, which would \
                     leave it with no neuron; switch at least one on with \
                     set_neuron({linear}, neuron, true)"
                )));
            }
            Ok(Some(kept))
        })
        .collect()
}

/// The output of `layer` for one row of `width` zeros, run on a tape with `params`.
fn zeros_through<T: Real>(layer: &dyn Layer<T>, params: &ParamStore<T>, width: usize) -> Vec<T> {
    let mut graph = Graph::<T>::new_generic();
    let mut tape = graph.tape(&params.values);
    let zeros = Tensor::<T>::zeros_generic(&[1, width]);
    let x = tape.input(zeros.view());
    let y = layer.forward(&mut tape, x);
    tape.value(y).data().to_vec()
}

/// The `Linear` at layer `index` rebuilt in `store` from the rows `rows` (every row if
/// `None`) and the columns `cols` (every column if `None`) of its weight in `params`, and
/// the same rows of its bias.
fn rebuild_linear<T: Real>(
    index: usize,
    linear: &Linear,
    params: &ParamStore<T>,
    store: &mut ParamStore<T>,
    rows: Option<&[usize]>,
    cols: Option<&[usize]>,
) -> Result<Linear, Error> {
    let (w, b) = (params.value(linear.weight()), params.value(linear.bias()));
    let (outputs, inputs) = (linear.outputs(), linear.inputs());
    if w.shape().dims() != [outputs, inputs] || b.shape().dims() != [outputs] {
        return Err(Error::InvalidData(format!(
            "compact: the linear layer {index} expects weight [{outputs}, {inputs}] and bias \
             [{outputs}], the store holds {:?} and {:?}",
            w.shape().dims(),
            b.shape().dims()
        )));
    }
    if let Some(cols) = cols
        && cols.last().is_some_and(|&c| c >= inputs)
    {
        return Err(Error::InvalidData(format!(
            "compact: the linear layer {index} takes {inputs} inputs, fewer than the neurons \
             of the layer before it"
        )));
    }
    let all_rows: Vec<usize>;
    let rows = match rows {
        Some(rows) => rows,
        None => {
            all_rows = (0..outputs).collect();
            &all_rows
        }
    };
    let all_cols: Vec<usize>;
    let cols = match cols {
        Some(cols) => cols,
        None => {
            all_cols = (0..inputs).collect();
            &all_cols
        }
    };
    let (w, b) = (w.data(), b.data());
    let mut weight = Vec::with_capacity(rows.len() * cols.len());
    for &r in rows {
        weight.extend(cols.iter().map(|&c| w[r * inputs + c]));
    }
    let bias: Vec<T> = rows.iter().map(|&r| b[r]).collect();
    let weight = Tensor::from_vec(weight, &[rows.len(), cols.len()])?;
    let bias = Tensor::from_vec(bias, &[rows.len()])?;
    Linear::from_values(store, weight, bias)
}

/// The `Prelu` at layer `index` rebuilt in `store` with its slopes from `params`: only the
/// slopes of the neurons in `kept` when it has one slope per neuron and `kept` is given,
/// every slope otherwise.
fn rebuild_prelu<T: Real>(
    index: usize,
    prelu: &Prelu,
    params: &ParamStore<T>,
    store: &mut ParamStore<T>,
    kept: Option<&[usize]>,
) -> Result<Prelu, Error> {
    let slopes = params.value(prelu.weight()).data();
    let slopes: Vec<T> = match kept {
        Some(kept) if slopes.len() > 1 => {
            if kept.last().is_some_and(|&j| j >= slopes.len()) {
                return Err(Error::InvalidData(format!(
                    "compact: the prelu layer {index} has {} slopes, fewer than the neurons it \
                     receives",
                    slopes.len()
                )));
            }
            kept.iter().map(|&j| slopes[j]).collect()
        }
        _ => slopes.to_vec(),
    };
    let rebuilt = Prelu::new(store, slopes.len());
    store
        .value_mut(rebuilt.weight())
        .data_mut()
        .copy_from_slice(&slopes);
    Ok(rebuilt)
}
