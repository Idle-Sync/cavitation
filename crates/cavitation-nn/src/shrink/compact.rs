//! Compaction: [`compact`] rebuilds a masked network without its off neurons, as a smaller
//! dense network with its own parameter store.

use std::collections::BTreeMap;

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
/// On the kernel levels whose matvec computes every output with the same steps, a sum in
/// input order from +0 with a separate multiply and add (`reference` and `sse4.2`), the
/// compacted network gives the masked one's outputs bit for bit through
/// [`InferenceModel::infer`](crate::InferenceModel::infer): a masked input adds an exact 0
/// to the sum, which leaves it as it is. Those levels' GEMM also sums each block of 256
/// terms in order, so the batched paths
/// ([`InferenceModel::infer_batch`](crate::InferenceModel::infer_batch) and the tape) are
/// bit for bit there too while every summed length is at most 256. On the other levels
/// (`avx2`, `neon`, `accelerate`) an output's rounding depends on its position (the vector
/// part of the matvec fuses the multiply and add, its last few columns do not, and
/// Accelerate's GEMM keeps its own order), and removing neurons moves outputs between those
/// parts. There each `Linear` of each network is within the rounding bound every fast
/// kernel keeps against the reference kernels: for an output `j` of a sum of length `k` with
/// inputs `h`, weights `w` and bias `b`,
/// `2·(k + 2)·1.19e-7·(Σᵢ |hᵢ|·|wᵢⱼ| + |bⱼ|)`, the bound
/// [`matvec_f32`](cavitation_core::kernels::matvec::matvec_f32) states for the matvec and
/// that the GEMM keeps too. The bound of each layer is carried through the layers after it
/// (`Σᵢ boundᵢ·|wᵢⱼ|` into the next `Linear`, unchanged through an activation whose slope
/// is at most 1). The reference results of the two networks are identical, so their outputs
/// differ by at most twice the masked network's carried bound, with `k` its own summed
/// lengths.
///
/// All of this holds for finite weights: a masked neuron whose outgoing weight is infinite
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
/// - between a `Linear` that loses neurons and the next `Linear`, a layer that is neither an
///   [`activation`](Layer::activation) (a built-in or an
///   [`ActivationLayer`](crate::ActivationLayer)) nor a `Prelu`: compaction moves neurons to
///   lower positions, and only for those layers is it known that every position is treated
///   alike;
/// - a `Linear` whose every neuron is off.
///
/// A mask with no neuron off removes nothing and is dropped wherever it is. For each
/// removed neuron, the layers after the last mask in its run that switches it off must map
/// 0 to 0; `f(0)` is found by running each of them on a row of zeros with the model's own
/// values, so a `Prelu` with an infinite slope there fails too.
///
/// [`Error::InvalidData`], naming the layer, when a mask's width or a parameter's shape does
/// not fit its layer.
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
/// let want = masked.infer(&x).to_vec();
/// for (got, want) in compacted.infer(&x).iter().zip(&want) {
///     assert!((got - want).abs() <= 1e-5); // bit for bit on reference and sse4.2
/// }
/// ```
pub fn compact<T: Real>(
    model: &Sequential<T>,
    params: &ParamStore<T>,
) -> Result<(Sequential<T>, ParamStore<T>), Error> {
    check_parameters(model)?;
    let owners = run_owners(model);
    let masks = placed_masks(model, &owners)?;
    for (&linear, masked) in &masks {
        check_narrowed_run(model, params, linear, masked)?;
    }
    let keep = kept_neurons(model, params, &masks)?;
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
            out = out.push(rebuild_prelu(prelu, params, &mut store, kept));
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

/// The layers whose masks switch some neuron off, grouped by the `Linear` whose neurons
/// they mask, in layer order, after checking that each one sits in the run of element-wise
/// layers of a hidden `Linear` that leads to the next `Linear` and fits its width. Masks
/// with every neuron on are left out: they remove nothing.
fn placed_masks<T: Real>(
    model: &Sequential<T>,
    owners: &[Option<usize>],
) -> Result<BTreeMap<usize, Vec<usize>>, Error> {
    let n = model.len();
    let mut placed: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (m, &owner) in owners.iter().enumerate() {
        let Some(mask) = model.mask(m) else {
            continue;
        };
        if mask.off().next().is_none() {
            continue;
        }
        let name = model.layer(m).name();
        if (m + 1..n).all(|l| model.linear(l).is_none()) {
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
        let next = model.neuron_layer(linear) + 1;
        if model.linear(next).is_none() {
            return Err(not_reaching(format!(
                "reach layer {next} ({}), which is not a Linear",
                model.layer(next).name()
            )));
        }
        let width = linear_width(model, linear);
        if mask.width() != width {
            return Err(Error::InvalidData(format!(
                "compact: the mask on layer {m} ({name}) has width {}, but the linear layer \
                 {linear} outputs width {width}",
                mask.width()
            )));
        }
        placed.entry(linear).or_default().push(m);
    }
    Ok(placed)
}

/// The output width of the `Linear` at layer `linear`, which must be one.
fn linear_width<T: Real>(model: &Sequential<T>, linear: usize) -> usize {
    model
        .linear(linear)
        .expect("the owner of a run is a Linear")
        .outputs()
}

/// Checks the run of the `Linear` at layer `linear`, which loses neurons (its masks are on
/// the layers `masked`): every layer after it up to the next `Linear` must be an activation
/// or a `Prelu` with one slope or one per neuron, and the next `Linear` must take as many
/// inputs as the `Linear` has neurons.
fn check_narrowed_run<T: Real>(
    model: &Sequential<T>,
    params: &ParamStore<T>,
    linear: usize,
    masked: &[usize],
) -> Result<(), Error> {
    let (width, end) = (linear_width(model, linear), model.neuron_layer(linear));
    for l in linear + 1..=end {
        let layer = model.layer(l);
        if let Some(prelu) = layer.as_prelu() {
            let slopes = params.value(prelu.weight()).data().len();
            if slopes != 1 && slopes != width {
                return Err(Error::InvalidData(format!(
                    "compact: the prelu layer {l} has {slopes} slopes, but the linear layer \
                     {linear} before it outputs width {width}"
                )));
            }
        } else if layer.activation().is_none() {
            return Err(Error::Unsupported(format!(
                "compact: layer {l} ({}) is element-wise but compact cannot tell whether it \
                 treats every position alike, so it cannot follow the linear layer {linear} \
                 once neurons are removed from it; use an activation (a built-in or an \
                 ActivationLayer) or a Prelu there, or clear the masks on layers {masked:?}",
                layer.name()
            )));
        }
    }
    let next = model
        .linear(end + 1)
        .expect("placed_masks checked that a Linear follows the run");
    if next.inputs() != width {
        return Err(Error::InvalidData(format!(
            "compact: the linear layer {} takes {} inputs, but the linear layer {linear} before \
             it outputs width {width}",
            end + 1,
            next.inputs()
        )));
    }
    Ok(())
}

/// For each layer index, the on neurons of the `Linear` there if a mask switches some of
/// them off (`None` for an unmasked `Linear` and every other layer), after checking that
/// every removed neuron reaches the next `Linear` as 0: each layer after the last mask in
/// its run that switches it off must map 0 to 0 at its position.
fn kept_neurons<T: Real>(
    model: &Sequential<T>,
    params: &ParamStore<T>,
    masks: &BTreeMap<usize, Vec<usize>>,
) -> Result<Vec<Option<Vec<usize>>>, Error> {
    let mut keep = vec![None; model.len()];
    for (&linear, masked) in masks {
        let (width, end) = (linear_width(model, linear), model.neuron_layer(linear));
        let mut zeros: Vec<Option<Vec<T>>> = vec![None; end + 1];
        let mut kept = Vec::with_capacity(width);
        for j in 0..width {
            let last = masked.iter().rev().find(|&&m| {
                !model
                    .mask(m)
                    .expect("placed_masks lists masked layers")
                    .is_on(j)
            });
            let Some(&last) = last else {
                kept.push(j);
                continue;
            };
            for (l, cached) in zeros.iter_mut().enumerate().skip(last + 1) {
                let out =
                    cached.get_or_insert_with(|| zeros_through(model.layer(l), params, width));
                if out.len() != width {
                    return Err(Error::InvalidData(format!(
                        "compact: layer {l} ({}) turns a row of width {width} into width {}",
                        model.layer(l).name(),
                        out.len()
                    )));
                }
                if out[j] != T::ZERO {
                    return Err(Error::Unsupported(format!(
                        "compact: the mask on layer {last} ({}) switches neuron {j} off before \
                         layer {l} ({}), which does not map 0 to 0, so the neuron does not send \
                         0 to the next Linear and cannot be removed; clear_mask({last}) and \
                         mask the neurons of the linear layer {linear} with \
                         set_neuron_mask({linear}, mask) instead, which puts the mask on layer \
                         {end}, after every activation",
                        model.layer(last).name(),
                        model.layer(l).name()
                    )));
                }
            }
        }
        if kept.is_empty() {
            return Err(Error::Unsupported(format!(
                "compact: every neuron of the linear layer {linear} is off, which would leave \
                 it with no neuron; switch at least one on with \
                 set_neuron({linear}, neuron, true)"
            )));
        }
        keep[linear] = Some(kept);
    }
    Ok(keep)
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

/// The `Prelu` rebuilt in `store` with its slopes from `params`: only the slopes of the
/// neurons in `kept` when it has one slope per neuron and `kept` is given (the run was
/// checked to have one slope per neuron then), every slope otherwise.
fn rebuild_prelu<T: Real>(
    prelu: &Prelu,
    params: &ParamStore<T>,
    store: &mut ParamStore<T>,
    kept: Option<&[usize]>,
) -> Prelu {
    let slopes = params.value(prelu.weight()).data();
    let slopes: Vec<T> = match kept {
        Some(kept) if slopes.len() > 1 => kept.iter().map(|&j| slopes[j]).collect(),
        _ => slopes.to_vec(),
    };
    let rebuilt = Prelu::new(store, slopes.len());
    store
        .value_mut(rebuilt.weight())
        .data_mut()
        .copy_from_slice(&slopes);
    rebuilt
}
